from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import torch

from .complexity import ComplexityTracker
from .config import Config
from .data import DataModule
from .ewc import EWC
from .evaluator import Evaluator
from .growth import GrowthController
from .signals import repr_health_snapshot
from .trainer import Trainer


class Experiment:
    """
    Orchestrates the full continual learning experiment across multiple seeds.

    Runs:
      1. Dynamic LightGCN  — EWC + GrowthController
      2. EWC-only ablation — fixed capacity, EWC only
      3. Baseline A        — Cumulative, No-EWC
      4. Baseline B        — All-at-once static

    Aggregates results across seeds and exports CSV files.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.data = DataModule(cfg)
        self.trainer: Trainer
        self.evaluator: Evaluator

    def run(self):
        cfg = self.cfg
        self.data.load()
        self.trainer = Trainer(cfg, self.data)
        self.evaluator = Evaluator(cfg, self.data)

        print("\n── Pre-building cumulative graphs ──")
        shared_train, shared_test, shared_cum_adjs = self._build_shared_graphs()

        seeds_dyn, seeds_monitor, seeds_ewc = [], [], []
        seeds_seq, seeds_sta, seeds_seq_bwt = [], [], []
        growth_all = []
        cplx_dyn, cplx_ewc, cplx_seq, cplx_sta = [], [], [], []
        health_dyn, health_ewc, health_seq = [], [], []

        for seed in range(cfg.n_seeds):
            print(f"\n{'=' * 60}\nSEED {seed}\n{'=' * 60}")

            res = self.run_dynamic(seed)
            seeds_dyn.append(res["dyn"])
            seeds_monitor.append(res["monitor"])
            growth_all.extend([{**g, "seed": seed} for g in res["growth"]])
            cplx_dyn.extend(res["complexity"])
            health_dyn.extend([{**r, "seed": seed} for r in res["repr_health"]])

            ewc_final, ewc_cplx, ewc_health = self.run_ewc_only(seed)
            seeds_ewc.append(ewc_final)
            cplx_ewc.extend(ewc_cplx)
            health_ewc.extend([{**r, "seed": seed} for r in ewc_health])

            seq_res, sta_res, seq_cplx, sta_cplx, seq_health, seq_bwt = self.run_baselines(
                seed, shared_train, shared_test, shared_cum_adjs
            )
            seeds_seq.append(seq_res)
            seeds_sta.append(sta_res)
            seeds_seq_bwt.append(seq_bwt)
            cplx_seq.extend(seq_cplx)
            cplx_sta.extend(sta_cplx)
            health_seq.extend([{**r, "seed": seed} for r in seq_health])

            gc.collect()
            if cfg.device == "cuda":
                torch.cuda.empty_cache()

        self._report_and_save(
            seeds_dyn, seeds_monitor, seeds_ewc,
            seeds_seq, seeds_sta, seeds_seq_bwt,
            growth_all, cplx_dyn + cplx_ewc + cplx_seq + cplx_sta,
            health_dyn, health_ewc, health_seq,
        )

    def _build_shared_graphs(self):
        data = self.data
        period_train, period_test, cum_adjs = [], [], []
        cumtrain = data.pretrain_df.copy()
        for p in range(self.cfg.num_periods):
            tr, _, te = data.chrono_split(data.period_dfs[p])
            period_train.append(tr)
            period_test.append(te)
            cumtrain = pd.concat([cumtrain, tr], ignore_index=True)
            adj, _, _ = data.make_graph(cumtrain)
            cum_adjs.append(adj)
            print(f"  P{p}: cumtrain={len(cumtrain):,}  adj_nnz={adj.nnz():,}")
        return period_train, period_test, cum_adjs

    def run_dynamic(self, seed: int) -> dict:
        cfg, data, trainer, evaluator = self.cfg, self.data, self.trainer, self.evaluator
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"\n{'#' * 60}\n#  SEED {seed} — Dynamic\n{'#' * 60}")

        ct = ComplexityTracker("dynamic")
        model = trainer.build_dynamic()

        pt_train, pt_valid, _ = data.chrono_split(data.pretrain_df)
        pt_adj, pt_global_u, pt_lmap = data.make_graph(pt_train)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        pu, pi, N_, _ = data.prep_tensors(pt_train, pt_lmap)
        pt_gpu = pt_global_u.to(cfg.device)

        for epoch in range(1, cfg.pretrain_epochs + 1):
            ct.start_epoch()
            avg_bpr, _, _ = trainer.train_epoch_ewc(model, opt, pt_adj, pt_gpu, pu, pi, N_)
            ct.end_epoch()
            print(f"  epoch {epoch:02d}/{cfg.pretrain_epochs}  bpr={avg_bpr:.4f}")
        del pt_gpu
        if cfg.device == "cuda":
            torch.cuda.empty_cache()

        pt_vm = evaluator.run_eval(model, pt_valid, pt_adj, pt_train, seed=seed, label="pretrain-valid")
        print(f"  Pretrain valid : {evaluator.fmt(pt_vm)}")
        torch.save({"model": model.state_dict(), "K": model.K, "hidden": model.hidden},
                   cfg.out_dir / f"ckpt_pretrain_seed{seed}.pt")
        ct.log_period(-1, model, pt_adj.nnz(), seed)

        ewc = None
        cumtrain = data.pretrain_df.copy()
        controller = GrowthController(cfg)
        valid_records = []
        period_test_dfs, cum_adjs_list = [], []
        test_monitor_r20 = {}
        monitor_log, repr_health_log, growth_log, complexity_records = [], [], [], []

        for period in range(cfg.num_periods):
            print(f"\n── Period {period}  K={model.K}  H={model.hidden} ──")
            pdata = data.period_dfs[period]
            train_df, valid_df, test_df = data.chrono_split(pdata)
            print(f"  train={len(train_df):,}  valid={len(valid_df):,}  test={len(test_df):,}")

            cumtrain = pd.concat([cumtrain, train_df], ignore_index=True)
            cum_adj, cum_global_u, cum_lmap = data.make_graph(cumtrain)
            cum_adjs_list.append(cum_adj)
            period_test_dfs.append(test_df)

            valid_records.append({
                "period_idx": period, "valid_df": valid_df,
                "cum_adj": cum_adj, "ref_train_df": cumtrain.copy(), "best_r20": None,
            })

            ct.epoch_times = []
            opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            _, new_ewc, _, gate_hist, bpr_hist = trainer.run_period(
                model, opt, train_df, cum_adj, cum_global_u, cum_lmap,
                cfg.cl_epochs, ewc, valid_records, seed=seed,
                tag=f"P{period}", complexity_tracker=ct, evaluator=evaluator,
            )

            crec = ct.log_period(period, model, cum_adj.nnz(), seed)
            complexity_records.append(crec)

            if ewc is not None:
                new_ewc.merge_with(ewc)
            ewc = new_ewc

            print(f"\n  ── Test monitor P{period} ──")
            tm = evaluator.run_eval(model, test_df, cum_adj, cumtrain, seed=seed, label=f"TEST-MONITOR-P{period}")
            print(f"  Test P{period}: {evaluator.fmt(tm)}")
            test_monitor_r20[period] = tm["recall@20"]
            monitor_log.append({"period": period, "K": model.K, "hidden": model.hidden, **tm})

            cum_gu_health = cum_global_u.to(cfg.device)
            health = repr_health_snapshot(model, cum_adj, cum_gu_health)
            del cum_gu_health
            if cfg.device == "cuda":
                torch.cuda.empty_cache()
            repr_health_log.append({"period": period, "K": model.K, "hidden": model.hidden, **health})

            bwt_scores = []
            if period > 0:
                for prev in range(period):
                    hm = evaluator.run_eval(
                        model, period_test_dfs[prev], cum_adjs_list[prev],
                        valid_records[prev]["ref_train_df"], seed=seed, label=f"P{period}→P{prev}",
                    )
                    delta = hm["recall@20"] - test_monitor_r20[prev]
                    bwt_scores.append(delta)
                    print(f"    P{prev}: R@20={hm['recall@20']:.4f}  base={test_monitor_r20[prev]:.4f}  BWT={delta:+.4f}")
                    if cfg.device == "cuda":
                        torch.cuda.empty_cache()

            avg_bwt = float(np.mean(bwt_scores)) if bwt_scores else float("nan")
            if not np.isnan(avg_bwt):
                print(f"  ► Avg BWT: {avg_bwt:+.4f}")

            if period < cfg.num_periods - 1:
                cum_global_gpu = cum_global_u.to(cfg.device)
                old_hidden = model.hidden
                model, grew, reasons, grew_hidden = controller.update(
                    model, ewc, cum_adj, cum_global_gpu,
                    avg_bwt, bpr_hist, gate_hist,
                )
                del cum_global_gpu
                if cfg.device == "cuda":
                    torch.cuda.empty_cache()

                if grew:
                    growth_log.append({"period": period, "new_K": model.K,
                                       "new_hidden": model.hidden, "reasons": "; ".join(reasons)})
                    if ewc is not None:
                        ewc.pad_to_model(model)
                    if grew_hidden:
                        new_params = list(model.new_parameters(old_hidden))
                        if new_params:
                            wu_opt = torch.optim.Adam(new_params, lr=cfg.lr_warmup, weight_decay=cfg.weight_decay)
                            cum_gpu_wu = cum_global_u.to(cfg.device)
                            pu_wu, pi_wu, N_wu, _ = data.prep_tensors(train_df, cum_lmap)
                            for wu_ep in range(1, cfg.warmup_epochs + 1):
                                avg_wu, _, _ = trainer.train_epoch_ewc(
                                    model, wu_opt, cum_adj, cum_gpu_wu,
                                    pu_wu, pi_wu, N_wu, ewc=ewc, warmup=0.1)
                                print(f"    [warmup {wu_ep}/{cfg.warmup_epochs}] bpr={avg_wu:.4f}")
                            del cum_gpu_wu
                            if cfg.device == "cuda":
                                torch.cuda.empty_cache()
                    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

            torch.save({"period": period, "model": model.state_dict(), "K": model.K, "hidden": model.hidden},
                       cfg.out_dir / f"ckpt_period{period}_seed{seed}.pt")

        print("\n── Final test — Dynamic ──")
        final_dyn = []
        for period in range(cfg.num_periods):
            tm = evaluator.run_eval(
                model, period_test_dfs[period], cum_adjs_list[period],
                valid_records[period]["ref_train_df"], seed=seed, label=f"FINAL-DYN-P{period}",
            )
            bwt = tm["recall@20"] - test_monitor_r20[period]
            final_dyn.append({"period": period, "K": model.K, "hidden": model.hidden,
                               "bwt_test": round(bwt, 4),
                               **{f"test_{k}": v for k, v in tm.items()}})
            print(f"  P{period}: {evaluator.fmt(tm)}  BWT={bwt:+.4f}")
            if cfg.device == "cuda":
                torch.cuda.empty_cache()

        return {"dyn": final_dyn, "monitor": monitor_log, "growth": growth_log,
                "complexity": complexity_records, "repr_health": repr_health_log}

    def run_ewc_only(self, seed: int) -> tuple:
        cfg, data, trainer, evaluator = self.cfg, self.data, self.trainer, self.evaluator
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"\n── EWC-only seed {seed}  (fixed H={cfg.hidden_static} K={cfg.k_static}) ──")

        ct = ComplexityTracker("ewc_only")
        model = trainer.build_static()

        pt_train, _, _ = data.chrono_split(data.pretrain_df)
        pt_adj, pt_gu, pt_lmap = data.make_graph(pt_train)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        pu, pi, N_, _ = data.prep_tensors(pt_train, pt_lmap)
        gpu = pt_gu.to(cfg.device)
        for epoch in range(1, cfg.pretrain_epochs + 1):
            ct.start_epoch()
            trainer.train_epoch_ewc(model, opt, pt_adj, gpu, pu, pi, N_)
            ct.end_epoch()
        del gpu
        if cfg.device == "cuda":
            torch.cuda.empty_cache()
        ct.log_period(-1, model, pt_adj.nnz(), seed)

        ewc = None
        cumtrain = data.pretrain_df.copy()
        test_monitor_r20 = {}
        period_test_dfs, cum_adjs_list, cumtrain_snaps = [], [], {}
        complexity_records, repr_health_log = [], []

        for period in range(cfg.num_periods):
            pdata = data.period_dfs[period]
            train_df, _, test_df = data.chrono_split(pdata)
            period_test_dfs.append(test_df)

            cumtrain = pd.concat([cumtrain, train_df], ignore_index=True)
            cum_adj, cum_global_u, cum_lmap = data.make_graph(cumtrain)
            cum_adjs_list.append(cum_adj)

            pos_u, pos_i, N_, train_f = data.prep_tensors(train_df, cum_lmap)
            opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            cum_gpu = cum_global_u.to(cfg.device)
            ct.epoch_times = []

            for epoch in range(1, cfg.cl_epochs + 1):
                warmup = min(1.0, epoch / cfg.ewc_warmup_epochs) if ewc else 1.0
                ct.start_epoch()
                trainer.train_epoch_ewc(model, opt, cum_adj, cum_gpu, pos_u, pos_i, N_, ewc=ewc, warmup=warmup)
                ct.end_epoch()

            snap, F_ = trainer.accumulate_fisher(model, cum_adj, cum_gpu, train_f, cum_lmap)
            new_ewc = EWC(snap, F_, cfg)
            if ewc is not None:
                new_ewc.merge_with(ewc)
            ewc = new_ewc

            crec = ct.log_period(period, model, cum_adj.nnz(), seed)
            complexity_records.append(crec)

            health = repr_health_snapshot(model, cum_adj, cum_gpu)
            repr_health_log.append({"period": period, "K": model.K, "hidden": model.hidden, **health})
            print(f"  EWC-only Repr-health P{period}: repr_collapse_cos={health['repr_collapse_cos']:.4f}  "
                  f"oversmooth_cos={health['oversmooth_cos']:.4f}")

            del cum_gpu
            if cfg.device == "cuda":
                torch.cuda.empty_cache()

            tm = evaluator.run_eval(model, test_df, cum_adj, cumtrain, seed=seed, label=f"EWC-only-P{period}")
            test_monitor_r20[period] = tm["recall@20"]
            cumtrain_snaps[period] = cumtrain.copy()
            print(f"  EWC-only P{period}: {evaluator.fmt(tm)}")
            if cfg.device == "cuda":
                torch.cuda.empty_cache()

        final = []
        for period in range(cfg.num_periods):
            tm = evaluator.run_eval(model, period_test_dfs[period], cum_adjs_list[period],
                                    cumtrain_snaps[period], seed=seed, label=f"EWC-final-P{period}")
            bwt = tm["recall@20"] - test_monitor_r20[period]
            final.append({"period": period, "bwt_test": round(bwt, 4),
                          **{f"test_{k}": v for k, v in tm.items()}})
            print(f"  EWC-final P{period}: {evaluator.fmt(tm)}  BWT={bwt:+.4f}")
            if cfg.device == "cuda":
                torch.cuda.empty_cache()

        del model
        if cfg.device == "cuda":
            torch.cuda.empty_cache()
        return final, complexity_records, repr_health_log

    def run_baselines(self, seed: int, period_train_dfs: list, period_test_dfs: list, cum_adjs: list) -> tuple:
        cfg, data, trainer, evaluator = self.cfg, self.data, self.trainer, self.evaluator
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Baseline A: Cumulative, No-EWC
        print(f"\n── Baseline A: Cumulative, No-EWC  (seed {seed}, H={cfg.hidden_static} K={cfg.k_static}) ──")
        ct_seq = ComplexityTracker("cumulative_no_ewc")
        base_seq = trainer.build_static()
        cumtrain = data.pretrain_df.copy()
        pt_train, _, _ = data.chrono_split(data.pretrain_df)
        pt_adj, pt_gu, pt_lmap = data.make_graph(pt_train)
        pu, pi, N_, _ = data.prep_tensors(pt_train, pt_lmap)
        opt = torch.optim.Adam(base_seq.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        gpu = pt_gu.to(cfg.device)
        for epoch in range(1, cfg.pretrain_epochs + 1):
            ct_seq.start_epoch()
            trainer.train_epoch_simple(base_seq, opt, pt_adj, gpu, pu, pi, N_)
            ct_seq.end_epoch()
        del gpu
        if cfg.device == "cuda":
            torch.cuda.empty_cache()
        ct_seq.log_period(-1, base_seq, pt_adj.nnz(), seed)

        seq_metrics, seq_complexity, repr_health_log_seq = [], [], []
        test_monitor_seq, cumtrain_snapshots = {}, {}

        for period in range(cfg.num_periods):
            tr, te = period_train_dfs[period], period_test_dfs[period]
            cumtrain = pd.concat([cumtrain, tr], ignore_index=True)
            cum_adj_p = cum_adjs[period]
            _, bs_gu, bs_lmap = data.make_graph(cumtrain)
            bs_pu, bs_pi, bs_N, _ = data.prep_tensors(tr, bs_lmap)
            opt = torch.optim.Adam(base_seq.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            bs_gpu = bs_gu.to(cfg.device)
            ct_seq.epoch_times = []
            for epoch in range(1, cfg.cl_epochs + 1):
                ct_seq.start_epoch()
                trainer.train_epoch_simple(base_seq, opt, cum_adj_p, bs_gpu, bs_pu, bs_pi, bs_N)
                ct_seq.end_epoch()

            crec = ct_seq.log_period(period, base_seq, cum_adj_p.nnz(), seed)
            seq_complexity.append(crec)

            health = repr_health_snapshot(base_seq, cum_adj_p, bs_gpu)
            repr_health_log_seq.append({"period": period, "K": base_seq.K, "hidden": base_seq.hidden, **health})
            print(f"  CumNoEWC Repr-health P{period}: repr_collapse_cos={health['repr_collapse_cos']:.4f}  "
                  f"oversmooth_cos={health['oversmooth_cos']:.4f}")
            del bs_gpu
            if cfg.device == "cuda":
                torch.cuda.empty_cache()

            m = evaluator.run_eval(base_seq, te, cum_adj_p, cumtrain, seed=seed, label=f"CumNoEWC-P{period}")
            test_monitor_seq[period] = m["recall@20"]
            seq_metrics.append({"period": period, **m})
            cumtrain_snapshots[period] = cumtrain.copy()
            print(f"  CumNoEWC P{period}: {evaluator.fmt(m)}")

        print("\n  ── CumNoEWC forgetting (BWT) ──")
        seq_final = []
        for period in range(cfg.num_periods):
            tm = evaluator.run_eval(base_seq, period_test_dfs[period], cum_adjs[period],
                                    cumtrain_snapshots[period], seed=seed, label=f"CumNoEWC-FINAL-P{period}")
            bwt = tm["recall@20"] - test_monitor_seq[period]
            seq_final.append({"period": period, "bwt_test": round(bwt, 4),
                              **{f"test_{k}": v for k, v in tm.items()}})
            print(f"  CumNoEWC-final P{period}: {evaluator.fmt(tm)}  BWT={bwt:+.4f}")
            if cfg.device == "cuda":
                torch.cuda.empty_cache()

        del base_seq
        if cfg.device == "cuda":
            torch.cuda.empty_cache()

        # Baseline B: All-at-once static
        print(f"\n── Baseline B: All-at-once static  (seed {seed}, H={cfg.hidden_static} K={cfg.k_static}) ──")
        ct_sta = ComplexityTracker("all_at_once")
        base_sta = trainer.build_static()
        all_df = pd.concat([data.pretrain_df] + period_train_dfs, ignore_index=True)
        final_cum_adj = cum_adjs[-1]
        _, st_gu, st_lmap = data.make_graph(all_df)
        st_pu, st_pi, st_N, _ = data.prep_tensors(all_df, st_lmap)
        opt = torch.optim.Adam(base_sta.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        st_gpu = st_gu.to(cfg.device)
        st_ep = cfg.pretrain_epochs + cfg.num_periods * cfg.cl_epochs
        for epoch in range(1, st_ep + 1):
            ct_sta.start_epoch()
            trainer.train_epoch_simple(base_sta, opt, final_cum_adj, st_gpu, st_pu, st_pi, st_N)
            ct_sta.end_epoch()
        del st_gpu
        if cfg.device == "cuda":
            torch.cuda.empty_cache()

        sta_metrics, sta_complexity = [], []
        for period in range(cfg.num_periods):
            ref = pd.concat([data.pretrain_df] + period_train_dfs[: period + 1], ignore_index=True)
            m = evaluator.run_eval(base_sta, period_test_dfs[period], cum_adjs[period], ref,
                                   seed=seed, label=f"Static-P{period}")
            sta_metrics.append({"period": period, **m})
            print(f"  Static P{period}: {evaluator.fmt(m)}")
            if cfg.device == "cuda":
                torch.cuda.empty_cache()
            sta_complexity.append({
                "model": "all_at_once", "seed": seed, "period": period,
                "K": base_sta.K, "hidden": base_sta.hidden,
                "n_params": sum(p.numel() for p in base_sta.parameters() if p.requires_grad),
                "epoch_time_s": round(float(np.mean(ct_sta.epoch_times)) if ct_sta.epoch_times else 0.0, 4),
                "flops_per_epoch": 2 * final_cum_adj.nnz() * base_sta.hidden * base_sta.K * 2,
            })

        del base_sta
        if cfg.device == "cuda":
            torch.cuda.empty_cache()

        return seq_metrics, sta_metrics, seq_complexity, sta_complexity, repr_health_log_seq, seq_final

    @staticmethod
    def _aggregate(all_seeds_list: list, metrics: list) -> pd.DataFrame:
        rows = []
        for period in range(len(all_seeds_list[0])):
            row = {"period": period}
            for m in metrics:
                vals = [s[period][m] for s in all_seeds_list if len(s) > period]
                row[f"{m}_mean"] = round(float(np.mean(vals)), 4)
                row[f"{m}_std"] = round(float(np.std(vals)), 4)
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _aggregate_complexity(records: list) -> pd.DataFrame:
        df = pd.DataFrame(records)
        df = df[df["period"] >= 0]
        rows = []
        for (model_name, period), g in df.groupby(["model", "period"]):
            rows.append({
                "model": model_name, "period": period,
                "K_mean": round(g["K"].mean(), 2),
                "hidden_mean": round(g["hidden"].mean(), 2),
                "n_params_mean": int(g["n_params"].mean()),
                "epoch_time_s_mean": round(g["epoch_time_s"].mean(), 4),
                "epoch_time_s_std": round(g["epoch_time_s"].std(), 4),
                "flops_per_epoch_mean": int(g["flops_per_epoch"].mean()),
            })
        return pd.DataFrame(rows).sort_values(["model", "period"])

    @staticmethod
    def _aggregate_repr_health(records: list, model_name: str) -> pd.DataFrame:
        df = pd.DataFrame(records)
        if df.empty:
            return pd.DataFrame(columns=["model", "period",
                "repr_collapse_cos_mean", "repr_collapse_cos_std",
                "oversmooth_cos_mean", "oversmooth_cos_std"])
        rows = []
        for period, g in df.groupby("period"):
            rows.append({
                "model": model_name, "period": period,
                "repr_collapse_cos_mean": round(g["repr_collapse_cos"].mean(), 4),
                "repr_collapse_cos_std": round(g["repr_collapse_cos"].std(), 4),
                "oversmooth_cos_mean": round(g["oversmooth_cos"].mean(), 4),
                "oversmooth_cos_std": round(g["oversmooth_cos"].std(), 4),
            })
        return pd.DataFrame(rows).sort_values("period")

    def _report_and_save(
        self,
        seeds_dyn, seeds_monitor, seeds_ewc,
        seeds_seq, seeds_sta, seeds_seq_bwt,
        growth_all, cplx_all,
        health_dyn, health_ewc, health_seq,
    ):
        cfg = self.cfg
        agg = self._aggregate
        dyn_metrics = ["test_recall@10", "test_recall@20", "test_ndcg@10",
                       "test_mrr", "test_auc", "bwt_test",
                       "test_recall@20_warm", "test_recall@20_cold"]
        base_metrics = ["recall@10", "recall@20", "ndcg@10", "mrr", "auc",
                        "recall@20_warm", "recall@20_cold"]
        mon_metrics = ["recall@10", "recall@20", "ndcg@10", "mrr", "auc",
                       "recall@20_warm", "recall@20_cold"]

        dyn_agg     = agg(seeds_dyn,     dyn_metrics)
        monitor_agg = agg(seeds_monitor, mon_metrics)
        ewc_agg     = agg(seeds_ewc,     dyn_metrics)
        seq_agg     = agg(seeds_seq,     base_metrics)
        sta_agg     = agg(seeds_sta,     base_metrics)
        seq_bwt_agg = agg(seeds_seq_bwt, dyn_metrics)
        cplx_agg    = self._aggregate_complexity(cplx_all)
        repr_dyn    = self._aggregate_repr_health(health_dyn, "dynamic")
        repr_ewc    = self._aggregate_repr_health(health_ewc, "ewc_only")
        repr_seq    = self._aggregate_repr_health(health_seq, "cumulative_no_ewc")
        repr_agg    = pd.concat([repr_dyn, repr_ewc, repr_seq], ignore_index=True)

        P = cfg.num_periods
        sep = "=" * 70

        print(f"\n{sep}\nFINAL SUMMARY  ({cfg.n_seeds} seeds)  —  Dataset: {cfg.dataset.upper()}\n{sep}")

        def _table(df, title, cols):
            print(f"\n── {title} ──")
            for _, row in df.iterrows():
                parts = [f"P{int(row['period'])}"]
                for c in cols:
                    parts.append(f"{c}={row[f'{c}_mean']:.4f}±{row[f'{c}_std']:.4f}")
                print("  ".join(parts))

        _table(dyn_agg,     f"Dynamic LightGCN (EWC + Growth)  [N={cfg.n_seeds}]",
               ["test_recall@20", "test_recall@20_warm", "test_recall@20_cold", "test_ndcg@10", "test_auc", "bwt_test"])
        _table(ewc_agg,     f"EWC-only, fixed H={cfg.hidden_static} K={cfg.k_static}  [N={cfg.n_seeds}]",
               ["test_recall@20", "test_recall@20_warm", "test_recall@20_cold", "test_ndcg@10", "test_auc", "bwt_test"])
        _table(seq_agg,     f"Cumulative, No-EWC, fixed H={cfg.hidden_static} K={cfg.k_static}",
               ["recall@20", "recall@20_warm", "recall@20_cold", "ndcg@10", "auc"])
        _table(seq_bwt_agg, f"Cumulative, No-EWC — forgetting  [N={cfg.n_seeds}]",
               ["test_recall@20", "test_recall@20_warm", "test_recall@20_cold", "test_ndcg@10", "test_auc", "bwt_test"])
        _table(sta_agg,     f"All-at-once static, fixed H={cfg.hidden_static} K={cfg.k_static}",
               ["recall@20", "recall@20_warm", "recall@20_cold", "ndcg@10", "auc"])

        print("\n── R@20 four-way comparison ──")
        print(f"{'P':>3}  {'Dynamic':>8}  {'EWC-only':>8}  {'CumNoEWC':>8}  {'Static':>8}  {'dyn-ewc':>8}  {'dyn-seq':>8}  {'dyn-sta':>8}")
        deltas = {"ewc": [], "seq": [], "sta": []}
        for p in range(P):
            d  = dyn_agg.loc[dyn_agg["period"] == p, "test_recall@20_mean"].values[0]
            e  = ewc_agg.loc[ewc_agg["period"] == p, "test_recall@20_mean"].values[0]
            s  = seq_agg.loc[seq_agg["period"] == p, "recall@20_mean"].values[0]
            st = sta_agg.loc[sta_agg["period"] == p, "recall@20_mean"].values[0]
            deltas["ewc"].append(d - e); deltas["seq"].append(d - s); deltas["sta"].append(d - st)
            print(f"{p:>3}  {d:>8.4f}  {e:>8.4f}  {s:>8.4f}  {st:>8.4f}  {d-e:>+8.4f}  {d-s:>+8.4f}  {d-st:>+8.4f}")
        print(f"{'avg':>3}  {'':>8}  {'':>8}  {'':>8}  {'':>8}  "
              f"{np.mean(deltas['ewc']):>+8.4f}  {np.mean(deltas['seq']):>+8.4f}  {np.mean(deltas['sta']):>+8.4f}")

        print("\n── BWT comparison ──")
        print(f"{'P':>3}  {'Dyn BWT':>14}  {'EWC-only BWT':>14}  {'CumNoEWC BWT':>14}")
        for p in range(P):
            db = dyn_agg.loc[dyn_agg["period"] == p, "bwt_test_mean"].values[0]
            ds = dyn_agg.loc[dyn_agg["period"] == p, "bwt_test_std"].values[0]
            eb = ewc_agg.loc[ewc_agg["period"] == p, "bwt_test_mean"].values[0]
            es = ewc_agg.loc[ewc_agg["period"] == p, "bwt_test_std"].values[0]
            cb = seq_bwt_agg.loc[seq_bwt_agg["period"] == p, "bwt_test_mean"].values[0]
            cs = seq_bwt_agg.loc[seq_bwt_agg["period"] == p, "bwt_test_std"].values[0]
            print(f"{p:>3}  {db:>+8.4f}±{ds:.4f}  {eb:>+8.4f}±{es:.4f}  {cb:>+8.4f}±{cs:.4f}")
        print(f"{'avg':>3}  {dyn_agg['bwt_test_mean'].mean():>+8.4f}           "
              f"{ewc_agg['bwt_test_mean'].mean():>+8.4f}           "
              f"{seq_bwt_agg['bwt_test_mean'].mean():>+8.4f}")

        print("\n── Warm vs Cold R@20 — Dynamic ──")
        for p in range(P):
            c = dyn_agg.loc[dyn_agg["period"] == p, "test_recall@20_mean"].values[0]
            w = dyn_agg.loc[dyn_agg["period"] == p, "test_recall@20_warm_mean"].values[0]
            k = dyn_agg.loc[dyn_agg["period"] == p, "test_recall@20_cold_mean"].values[0]
            print(f"  P{p}  combined={c:.4f}  warm={w:.4f}  cold={k:.4f}")

        print("\n── Growth events ──")
        if growth_all:
            print(pd.DataFrame(growth_all).to_string(index=False))
        else:
            print("  None triggered.")

        print("\n── Complexity (mean per model per period) ──")
        for _, row in cplx_agg.iterrows():
            print(f"  {row['model']:<20}  P{int(row['period'])}  K={row['K_mean']:.1f}  "
                  f"hidden={row['hidden_mean']:.0f}  params={row['n_params_mean']:,}  "
                  f"t/ep={row['epoch_time_s_mean']:.4f}s  FLOPs={row['flops_per_epoch_mean']:,}")

        print("\n── Representation health (mean ± std) ──")
        for _, row in repr_agg.iterrows():
            print(f"  {row['model']:<20}  P{int(row['period'])}  "
                  f"repr_collapse={row['repr_collapse_cos_mean']:.4f}±{row['repr_collapse_cos_std']:.4f}  "
                  f"oversmooth={row['oversmooth_cos_mean']:.4f}±{row['oversmooth_cos_std']:.4f}")

        # Save
        dyn_agg.to_csv(     cfg.out_dir / f"results_dynamic_{cfg.dataset}.csv",     index=False)
        monitor_agg.to_csv( cfg.out_dir / f"results_monitor_{cfg.dataset}.csv",     index=False)
        ewc_agg.to_csv(     cfg.out_dir / f"results_ewconly_{cfg.dataset}.csv",     index=False)
        seq_agg.to_csv(     cfg.out_dir / f"results_cumnewc_{cfg.dataset}.csv",     index=False)
        seq_bwt_agg.to_csv( cfg.out_dir / f"results_cumnewc_bwt_{cfg.dataset}.csv", index=False)
        sta_agg.to_csv(     cfg.out_dir / f"results_static_{cfg.dataset}.csv",      index=False)
        cplx_agg.to_csv(    cfg.out_dir / f"complexity_{cfg.dataset}.csv",           index=False)
        repr_agg.to_csv(    cfg.out_dir / f"repr_health_{cfg.dataset}.csv",          index=False)
        if growth_all:
            pd.DataFrame(growth_all).to_csv(cfg.out_dir / f"growth_log_{cfg.dataset}.csv", index=False)
        for i, (dyn, ewc, bwt) in enumerate(zip(seeds_dyn, seeds_ewc, seeds_seq_bwt)):
            pd.DataFrame(dyn).to_csv( cfg.out_dir / f"raw_dyn_seed{i}_{cfg.dataset}.csv",        index=False)
            pd.DataFrame(ewc).to_csv( cfg.out_dir / f"raw_ewc_seed{i}_{cfg.dataset}.csv",        index=False)
            pd.DataFrame(bwt).to_csv( cfg.out_dir / f"raw_cumnewc_bwt_seed{i}_{cfg.dataset}.csv", index=False)

        print(f"\nAll results saved to {cfg.out_dir}/")
        print("Done.")
