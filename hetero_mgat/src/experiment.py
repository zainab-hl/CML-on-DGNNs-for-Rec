"""
experiment.py
=============
Top-level orchestration: one full continual-learning run of the
dynamic (EWC + severity-growth) model, the EWC-only ablation, the two
fixed-capacity baselines (Cumulative-No-EWC, All-at-once), the
multi-seed loop, and result aggregation / saving.

Usage
-----
    from hetero_mgat.config import Config
    from hetero_mgat.data import DataModule
    from hetero_mgat.experiment import Experiment

    cfg = Config()
    dm = DataModule(cfg)
    exp = Experiment(cfg, dm)
    exp.run()
"""
from __future__ import annotations

import gc
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from .complexity import ComplexityTracker
from .config import Config
from .data import DataModule
from .evaluator import Evaluator, fmt
from .ewc import EWC, accumulate_fisher
from .growth import GrowthController
from .model import build_dynamic, build_static
from .signals import repr_health_snapshot
from .trainer import run_period_training, simple_train_epoch, train_one_epoch


class Experiment:
    def __init__(self, cfg: Config, dm: DataModule):
        self.cfg = cfg
        self.dm = dm
        self.evaluator = Evaluator(cfg, dm)

        # accumulators across seeds
        self.all_seeds_dyn: List[list] = []
        self.all_seeds_monitor: List[list] = []
        self.all_seeds_ewc: List[list] = []
        self.all_seeds_sta: List[list] = []
        self.all_seeds_seq_bwt: List[list] = []
        self.all_growth: List[dict] = []
        self.all_complexity_dyn: List[dict] = []
        self.all_complexity_ewc: List[dict] = []
        self.all_complexity_seq: List[dict] = []
        self.all_complexity_sta: List[dict] = []
        self.all_repr_health_dyn: List[dict] = []
        self.all_repr_health_ewc: List[dict] = []
        self.all_repr_health_seq: List[dict] = []

    # ═════════════════════════════════════════════════════════════════
    # 1. Dynamic model (EWC + severity-growth)
    # ═════════════════════════════════════════════════════════════════
    def run_one_seed(self, seed: int) -> dict:
        cfg, dm, ev = self.cfg, self.dm, self.evaluator
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f'\n{"#" * 60}')
        print(f'#  SEED {seed}  (Dynamic: EWC + Severity-Growth)')
        print(f'{"#" * 60}')

        complexity_tracker = ComplexityTracker('dynamic')

        # ── Pretrain ─────────────────────────────────────────────────
        print('\n── PRETRAINING ──')
        model = build_dynamic(cfg, dm.NUM_BRANDS, dm.NUM_MAIN)
        pt_train, pt_valid, pt_test = dm.chrono_split(dm.pretrain_df)
        pt_train_ei = dm.make_edge_index(pt_train)
        ei_b, gu_b, nu_b, lmap_b, ne_b = pt_train_ei

        mask_pt = pt_train['u_idx'].apply(lambda u: int(u) in lmap_b)
        ptf = pt_train[mask_pt].reset_index(drop=True)
        pul_pt = torch.tensor([lmap_b[int(v)] for v in ptf['u_idx'].values], dtype=torch.long)
        pi_pt = torch.tensor(ptf['i_idx'].values, dtype=torch.long)
        N_pt = len(pul_pt)
        ht_pt = dm.build_hist_tensor(nu_b, lmap_b)
        ei_pt_g, gu_pt_g, ht_pt_g = dm.to_gpu(ei_b, gu_b, ht_pt)

        opt = torch.optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WD)
        for epoch in range(1, cfg.PRETRAIN_EPOCHS + 1):
            complexity_tracker.start_epoch()
            avg_bpr, _, _ = train_one_epoch(
                cfg, model, opt, ei_pt_g, gu_pt_g, ht_pt_g, dm.item_features,
                pul_pt, pi_pt, N_pt)
            complexity_tracker.end_epoch()
            print(f'  epoch {epoch:02d}/{cfg.PRETRAIN_EPOCHS}  bpr={avg_bpr:.4f}')

        pt_vm = ev.run_eval(model, pt_valid, pt_train_ei, pt_train,
                            label='pretrain-valid', seed=seed)
        print(f'  Pretrain valid : {fmt(pt_vm)}')
        torch.save({'model': model.state_dict(),
                    'heads': model.num_heads, 'hidden': model.hidden},
                   cfg.out / f'ckpt_pretrain_seed{seed}.pt')

        complexity_tracker.log_period(-1, model, ne_b, seed)

        # ── Shared CL period structures ─────────────────────────────
        period_train_dfs, period_test_dfs, cum_eis = [], [], []
        cumtrain_base = dm.pretrain_df.copy()
        for period in range(cfg.NUM_CL_PERIODS):
            pdata = dm.cl_df[dm.cl_df['cl_period'] == period].reset_index(drop=True)
            tr, va, te = dm.chrono_split(pdata)
            cumtrain_base = pd.concat([cumtrain_base, tr], ignore_index=True)
            period_train_dfs.append(tr)
            period_test_dfs.append(te)
            cum_eis.append(dm.make_edge_index(cumtrain_base))

        # ── Continual Learning ───────────────────────────────────────
        print('\n── Continual Learning (Dynamic) ──')
        ewc = None
        cumtrain = dm.pretrain_df.copy()
        controller = GrowthController(cfg, dm.NUM_BRANDS, dm.NUM_MAIN)
        growth_log: List[dict] = []
        valid_records: List[dict] = []
        period_best_r20: List[float] = []
        test_monitor_r20: Dict[int, float] = {}
        test_monitor_log: List[dict] = []
        complexity_records: List[dict] = []
        repr_health_log: List[dict] = []

        for period in range(cfg.NUM_CL_PERIODS):
            pdata = dm.cl_df[dm.cl_df['cl_period'] == period].reset_index(drop=True)
            train_df_p, valid_df_p, test_df_p = dm.chrono_split(pdata)
            print(f'\n── Period {period}  heads={model.num_heads}  hidden={model.hidden} ──')
            print(f'  train={len(train_df_p):,}  valid={len(valid_df_p):,}  '
                  f'test={len(test_df_p):,} [SEALED]')

            cumtrain = pd.concat([cumtrain, train_df_p], ignore_index=True)
            cum_ei = cum_eis[period]

            valid_records.append({
                'period_idx': period, 'valid_df': valid_df_p, 'cum_ei': cum_ei,
                'ref_train_df': cumtrain.copy(), 'best_r20': None,
            })

            complexity_tracker.epoch_times = []

            opt = torch.optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WD)
            epoch_log, new_ewc, period_bpr, gate_hist, bpr_hist = run_period_training(
                cfg, dm, ev, model, opt, train_df_p, cum_ei, cfg.CL_EPOCHS,
                ewc, valid_records, seed=seed, tag=f'P{period}',
                complexity_tracker=complexity_tracker,
            )

            crec = complexity_tracker.log_period(period, model, cum_ei[-1], seed)
            complexity_records.append(crec)

            if new_ewc is not None:
                new_ewc.merge_from(ewc)
                ewc = new_ewc

            print(f'\n  ── Test monitor P{period} (BWT base) ──')
            tm = ev.run_eval(model, test_df_p, cum_ei, cumtrain,
                             label=f'TEST-MONITOR-P{period}', seed=seed)
            print(f'  Test P{period}: {fmt(tm)}')
            test_monitor_r20[period] = tm['recall@20']
            test_monitor_log.append({
                'period': period, 'heads': model.num_heads,
                'hidden': model.hidden, **tm,
            })
            if cfg.device == 'cuda':
                torch.cuda.empty_cache()

            # Repr-health snapshot for the dynamic model.
            ei_h, gu_h, nu_h, lmap_h, _ = cum_ei
            ht_h = dm.build_hist_tensor(nu_h, lmap_h)
            ei_h_gpu, gu_h_gpu, ht_h_gpu = dm.to_gpu(ei_h, gu_h, ht_h)
            health = repr_health_snapshot(cfg, model, ei_h_gpu, gu_h_gpu, ht_h_gpu, dm.item_features)
            del ei_h_gpu, gu_h_gpu, ht_h_gpu
            if cfg.device == 'cuda':
                torch.cuda.empty_cache()
            repr_health_log.append({'period': period, 'heads': model.num_heads,
                                    'hidden': model.hidden, **health})
            print(f'  Repr-health P{period}: repr_collapse_cos={health["repr_collapse_cos"]:.4f}  '
                  f'oversmooth_cos={health["oversmooth_cos"]:.4f}')

            bwt_scores = []
            if period > 0:
                print(f'\n  ── BWT P0..P{period - 1} (test-split base) ──')
                for prev in range(period):
                    hm = ev.run_eval(model, period_test_dfs[prev], cum_eis[prev],
                                     valid_records[prev]['ref_train_df'],
                                     label=f'P{period}→P{prev}', seed=seed)
                    base = test_monitor_r20[prev]
                    delta = hm['recall@20'] - base
                    bwt_scores.append(delta)
                    print(f'    P{prev}: R@20={hm["recall@20"]:.4f}  '
                          f'base(monitor)={base:.4f}  BWT={delta:+.4f}')
                    if cfg.device == 'cuda':
                        torch.cuda.empty_cache()

            avg_bwt = float(np.mean(bwt_scores)) if bwt_scores else float('nan')
            if not np.isnan(avg_bwt):
                print(f'  ► Avg BWT: {avg_bwt:+.4f}')
            period_best_r20.append(test_monitor_r20[period])

            if period < cfg.NUM_CL_PERIODS - 1:
                model = self._maybe_grow(
                    cfg, dm, controller, model, ewc, cum_ei, avg_bwt,
                    bpr_hist, gate_hist, growth_log, period, train_df_p)
                opt = torch.optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WD)

            torch.save({'period': period, 'model': model.state_dict(),
                        'heads': model.num_heads, 'hidden': model.hidden},
                       cfg.out / f'ckpt_period{period}_seed{seed}.pt')
            print(f'  Checkpoint saved.')

        # ── Final test ───────────────────────────────────────────────
        print('\n── Final test (Dynamic) ──')
        final_dyn = []
        for period in range(cfg.NUM_CL_PERIODS):
            tm = ev.run_eval(model, period_test_dfs[period], cum_eis[period],
                             valid_records[period]['ref_train_df'],
                             label=f'FINAL-DYN-P{period}', seed=seed)
            bwt = tm['recall@20'] - period_best_r20[period]
            final_dyn.append({
                'period': period, 'heads': model.num_heads, 'hidden': model.hidden,
                'bwt_test': round(bwt, 4),
                **{f'test_{k}': v for k, v in tm.items()},
            })
            print(f'  P{period}: {fmt(tm)}  BWT={bwt:+.4f}')
            if cfg.device == 'cuda':
                torch.cuda.empty_cache()

        del ei_pt_g, gu_pt_g, ht_pt_g
        if cfg.device == 'cuda':
            torch.cuda.empty_cache()

        return {
            'dyn': final_dyn,
            'monitor': test_monitor_log,
            'growth': growth_log,
            'complexity': complexity_records,
            'repr_health': repr_health_log,
            '_shared': {
                'period_train_dfs': period_train_dfs,
                'period_test_dfs': period_test_dfs,
                'cum_eis': cum_eis,
            },
        }

    def _maybe_grow(self, cfg, dm, controller, model, ewc, cum_ei, avg_bwt,
                    bpr_hist, gate_hist, growth_log, period, train_df_p):
        ei_gc, gu_gc, nu_gc, lmap_gc, _ = cum_ei
        ht_gc = dm.build_hist_tensor(nu_gc, lmap_gc)
        ei_gc_gpu, gu_gc_gpu, ht_gc_gpu = dm.to_gpu(ei_gc, gu_gc, ht_gc)
        old_hidden = model.hidden

        model, grew, reasons, grew_hidden = controller.update(
            model, ewc, ei_gc_gpu, gu_gc_gpu, ht_gc_gpu, dm.item_features,
            avg_bwt, bpr_hist, gate_hist,
        )
        del ei_gc_gpu, gu_gc_gpu, ht_gc_gpu
        if cfg.device == 'cuda':
            torch.cuda.empty_cache()

        if not grew:
            return model

        growth_log.append({
            'period': period, 'new_heads': model.num_heads,
            'new_hidden': model.hidden, 'reasons': '; '.join(reasons),
        })
        if ewc is not None:
            if grew_hidden:
                ewc.pad_to_model(model)
            else:
                ewc.prune_to_model(model)

        if grew_hidden:
            print(f'  [WARMUP] Hidden grew {old_hidden}→{model.hidden}: '
                  f'{cfg.WARMUP_EPOCHS}-epoch warm-up on new params only.')
            new_params = list(model.new_parameters(old_hidden))
            if new_params:
                wu_opt = torch.optim.Adam(new_params, lr=cfg.LR_WARMUP, weight_decay=cfg.WD)
                ei_wu, gu_wu, nu_wu, lmap_wu, _ = cum_ei
                ht_wu = dm.build_hist_tensor(nu_wu, lmap_wu)
                ei_wu_gpu, gu_wu_gpu, ht_wu_gpu = dm.to_gpu(ei_wu, gu_wu, ht_wu)
                mask_wu = train_df_p['u_idx'].apply(lambda u: int(u) in lmap_wu)
                tdf_wu = train_df_p[mask_wu].reset_index(drop=True)
                pul_wu = torch.tensor([lmap_wu[int(v)] for v in tdf_wu['u_idx'].values],
                                      dtype=torch.long)
                pi_wu = torch.tensor(tdf_wu['i_idx'].values, dtype=torch.long)
                N_wu = len(pul_wu)
                for wu_ep in range(1, cfg.WARMUP_EPOCHS + 1):
                    avg_wu, _, _ = train_one_epoch(
                        cfg, model, wu_opt, ei_wu_gpu, gu_wu_gpu, ht_wu_gpu,
                        dm.item_features, pul_wu, pi_wu, N_wu,
                        ewc=ewc, warmup=0.1, skip_phase2=True,
                    )
                    print(f'    [warmup {wu_ep}/{cfg.WARMUP_EPOCHS}] bpr={avg_wu:.4f}')
                del ei_wu_gpu, gu_wu_gpu, ht_wu_gpu
                if cfg.device == 'cuda':
                    torch.cuda.empty_cache()

        return model

    # ═════════════════════════════════════════════════════════════════
    # 2. EWC-only ablation (fixed HIDDEN_STATIC/HEADS_STATIC)
    # ═════════════════════════════════════════════════════════════════
    def run_ewc_only_seed(self, seed: int, period_train_dfs, period_test_dfs, cum_eis):
        cfg, dm, ev = self.cfg, self.dm, self.evaluator
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f'\n── EWC-only seed {seed}  '
              f'(fixed HIDDEN={cfg.HIDDEN_STATIC} HEADS={cfg.HEADS_STATIC}) ──')

        ct = ComplexityTracker('ewc_only')
        model = build_static(cfg, dm.NUM_BRANDS, dm.NUM_MAIN)

        pt_train, pt_valid, pt_test = dm.chrono_split(dm.pretrain_df)
        pt_train_ei = dm.make_edge_index(pt_train)
        ei_b, gu_b, nu_b, lmap_b, ne_b = pt_train_ei

        mask_pt = pt_train['u_idx'].apply(lambda u: int(u) in lmap_b)
        ptf = pt_train[mask_pt].reset_index(drop=True)
        pul_pt = torch.tensor([lmap_b[int(v)] for v in ptf['u_idx'].values], dtype=torch.long)
        pi_pt = torch.tensor(ptf['i_idx'].values, dtype=torch.long)
        N_pt = len(pul_pt)
        ht_pt = dm.build_hist_tensor(nu_b, lmap_b)
        ei_pt_g, gu_pt_g, ht_pt_g = dm.to_gpu(ei_b, gu_b, ht_pt)

        opt = torch.optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WD)
        for epoch in range(1, cfg.PRETRAIN_EPOCHS + 1):
            ct.start_epoch()
            train_one_epoch(cfg, model, opt, ei_pt_g, gu_pt_g, ht_pt_g,
                            dm.item_features, pul_pt, pi_pt, N_pt)
            ct.end_epoch()
            print(f'  epoch {epoch:02d}/{cfg.PRETRAIN_EPOCHS}')
        del ei_pt_g, gu_pt_g, ht_pt_g
        if cfg.device == 'cuda':
            torch.cuda.empty_cache()
        ct.log_period(-1, model, ne_b, seed)

        pt_vm = ev.run_eval(model, pt_valid, pt_train_ei, pt_train,
                            label='pretrain-valid-EWC', seed=seed)
        print(f'  Pretrain valid (EWC-only): {fmt(pt_vm)}')

        ewc = None
        cumtrain = dm.pretrain_df.copy()
        valid_records: List[dict] = []
        test_monitor_r20: Dict[int, float] = {}
        complexity_records: List[dict] = []
        repr_health_log: List[dict] = []

        for period in range(cfg.NUM_CL_PERIODS):
            train_df_p = period_train_dfs[period]
            pdata = dm.cl_df[dm.cl_df['cl_period'] == period].reset_index(drop=True)
            _, valid_df_p, test_df_p = dm.chrono_split(pdata)

            cumtrain = pd.concat([cumtrain, train_df_p], ignore_index=True)
            cum_ei = cum_eis[period]
            ei, global_u, n_u_local, local_u_map, n_edges = cum_ei

            valid_records.append({
                'period_idx': period, 'valid_df': valid_df_p, 'cum_ei': cum_ei,
                'ref_train_df': cumtrain.copy(), 'best_r20': None,
            })

            mask = train_df_p['u_idx'].apply(lambda u: int(u) in local_u_map)
            train_f = train_df_p[mask].reset_index(drop=True)
            pos_u = torch.tensor([local_u_map[int(v)] for v in train_f['u_idx'].values],
                                 dtype=torch.long)
            pos_i = torch.tensor(train_f['i_idx'].values, dtype=torch.long)
            N_ = len(pos_u)
            hist_ten = dm.build_hist_tensor(n_u_local, local_u_map)
            ei_gpu, global_gpu, hist_gpu = dm.to_gpu(ei, global_u, hist_ten)

            opt = torch.optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WD)
            ct.epoch_times = []
            for epoch in range(1, cfg.CL_EPOCHS + 1):
                warmup = min(1.0, epoch / cfg.EWC_WARMUP_EPOCHS) if ewc else 1.0
                ct.start_epoch()
                avg_bpr, ewc_val, _ = train_one_epoch(
                    cfg, model, opt, ei_gpu, global_gpu, hist_gpu, dm.item_features,
                    pos_u, pos_i, N_, ewc=ewc, warmup=warmup,
                )
                ct.end_epoch()
                print(f'    [EWC P{period}] epoch {epoch:02d}/{cfg.CL_EPOCHS}  bpr={avg_bpr:.4f}')

            snap, F_ = accumulate_fisher(cfg, model, ei_gpu, global_gpu, hist_gpu,
                                         dm.item_features, train_f, local_u_map)
            new_ewc = EWC(cfg, snap, F_)
            new_ewc.merge_from(ewc)
            ewc = new_ewc

            crec = ct.log_period(period, model, n_edges, seed)
            complexity_records.append(crec)

            health = repr_health_snapshot(cfg, model, ei_gpu, global_gpu, hist_gpu, dm.item_features)
            repr_health_log.append({'period': period, 'heads': model.num_heads,
                                    'hidden': model.hidden, **health})
            print(f'  EWC-only Repr-health P{period}: '
                  f'repr_collapse_cos={health["repr_collapse_cos"]:.4f}  '
                  f'oversmooth_cos={health["oversmooth_cos"]:.4f}')

            del ei_gpu, global_gpu, hist_gpu
            if cfg.device == 'cuda':
                torch.cuda.empty_cache()

            tm = ev.run_eval(model, test_df_p, cum_ei, cumtrain,
                             label=f'EWC-only-P{period}', seed=seed)
            test_monitor_r20[period] = tm['recall@20']
            print(f'  EWC-only P{period}: {fmt(tm)}')
            if cfg.device == 'cuda':
                torch.cuda.empty_cache()

        final = []
        for period in range(cfg.NUM_CL_PERIODS):
            tm = ev.run_eval(model, period_test_dfs[period], cum_eis[period],
                             valid_records[period]['ref_train_df'],
                             label=f'EWC-final-P{period}', seed=seed)
            bwt = tm['recall@20'] - test_monitor_r20[period]
            final.append({'period': period, 'bwt_test': round(bwt, 4),
                          **{f'test_{k}': v for k, v in tm.items()}})
            print(f'  EWC-final P{period}: {fmt(tm)}  BWT={bwt:+.4f}')
            if cfg.device == 'cuda':
                torch.cuda.empty_cache()

        del model
        if cfg.device == 'cuda':
            torch.cuda.empty_cache()
        return final, complexity_records, repr_health_log

    # ═════════════════════════════════════════════════════════════════
    # 3. Baselines: Cumulative-No-EWC + All-at-once static
    # ═════════════════════════════════════════════════════════════════
    def run_baselines_seed(self, seed: int, period_train_dfs_ref, period_test_dfs_ref, cum_eis_ref):
        cfg, dm, ev = self.cfg, self.dm, self.evaluator
        torch.manual_seed(seed)
        np.random.seed(seed)

        # ── Baseline A: Cumulative, No-EWC ──────────────────────────
        print(f'\n── Baseline A: Cumulative, No-EWC  (seed {seed}, '
              f'fixed HIDDEN={cfg.HIDDEN_STATIC} HEADS={cfg.HEADS_STATIC}) ──')
        ct_seq = ComplexityTracker('cumulative_no_ewc')
        base_seq = build_static(cfg, dm.NUM_BRANDS, dm.NUM_MAIN)

        pt_train, _, _ = dm.chrono_split(dm.pretrain_df)
        pt_train_ei = dm.make_edge_index(pt_train)
        ei_b, gu_b, nu_b, lmap_b, ne_b = pt_train_ei
        mask_pt = pt_train['u_idx'].apply(lambda u: int(u) in lmap_b)
        ptf = pt_train[mask_pt].reset_index(drop=True)
        pul_pt = torch.tensor([lmap_b[int(v)] for v in ptf['u_idx'].values], dtype=torch.long)
        pi_pt = torch.tensor(ptf['i_idx'].values, dtype=torch.long)
        N_pt = len(pul_pt)
        ht_pt = dm.build_hist_tensor(nu_b, lmap_b)
        ei_pt_g, gu_pt_g, ht_pt_g = dm.to_gpu(ei_b, gu_b, ht_pt)

        opt = torch.optim.Adam(base_seq.parameters(), lr=cfg.LR, weight_decay=cfg.WD)
        for epoch in range(1, cfg.PRETRAIN_EPOCHS + 1):
            ct_seq.start_epoch()
            simple_train_epoch(cfg, base_seq, opt, ei_pt_g, gu_pt_g, ht_pt_g,
                               dm.item_features, pul_pt, pi_pt, N_pt)
            ct_seq.end_epoch()
        del ei_pt_g, gu_pt_g, ht_pt_g
        if cfg.device == 'cuda':
            torch.cuda.empty_cache()
        ct_seq.log_period(-1, base_seq, ne_b, seed)

        cumtrain = dm.pretrain_df.copy()
        test_monitor_seq: Dict[int, float] = {}
        seq_complexity: List[dict] = []
        repr_health_log_seq: List[dict] = []
        cumtrain_snapshots: Dict[int, pd.DataFrame] = {}

        for period in range(cfg.NUM_CL_PERIODS):
            tr = period_train_dfs_ref[period]
            pdata = dm.cl_df[dm.cl_df['cl_period'] == period].reset_index(drop=True)
            _, _, te = dm.chrono_split(pdata)

            cumtrain = pd.concat([cumtrain, tr], ignore_index=True)
            cum_ei_p = cum_eis_ref[period]
            ei, gu, nu, lmap, n_edges = cum_ei_p

            mask = tr['u_idx'].apply(lambda u: int(u) in lmap)
            tr_f = tr[mask].reset_index(drop=True)
            pu = torch.tensor([lmap[int(v)] for v in tr_f['u_idx'].values], dtype=torch.long)
            pi = torch.tensor(tr_f['i_idx'].values, dtype=torch.long)
            N_ = len(pu)
            ht = dm.build_hist_tensor(nu, lmap)
            ei_gpu, gu_gpu, ht_gpu = dm.to_gpu(ei, gu, ht)

            opt = torch.optim.Adam(base_seq.parameters(), lr=cfg.LR, weight_decay=cfg.WD)
            ct_seq.epoch_times = []
            for epoch in range(1, cfg.CL_EPOCHS + 1):
                ct_seq.start_epoch()
                simple_train_epoch(cfg, base_seq, opt, ei_gpu, gu_gpu, ht_gpu,
                                   dm.item_features, pu, pi, N_)
                ct_seq.end_epoch()

            crec = ct_seq.log_period(period, base_seq, n_edges, seed)
            seq_complexity.append(crec)

            health = repr_health_snapshot(cfg, base_seq, ei_gpu, gu_gpu, ht_gpu, dm.item_features)
            repr_health_log_seq.append({'period': period, 'heads': base_seq.num_heads,
                                        'hidden': base_seq.hidden, **health})
            print(f'  CumNoEWC Repr-health P{period}: '
                  f'repr_collapse_cos={health["repr_collapse_cos"]:.4f}  '
                  f'oversmooth_cos={health["oversmooth_cos"]:.4f}')

            del ei_gpu, gu_gpu, ht_gpu
            if cfg.device == 'cuda':
                torch.cuda.empty_cache()

            m = ev.run_eval(base_seq, te, cum_ei_p, cumtrain,
                            label=f'CumNoEWC-P{period}', seed=seed)
            test_monitor_seq[period] = m['recall@20']
            print(f'  CumNoEWC P{period}: {fmt(m)}')
            cumtrain_snapshots[period] = cumtrain.copy()
            if cfg.device == 'cuda':
                torch.cuda.empty_cache()

        # ── Forgetting (BWT) for Cumulative-No-EWC ──────────────────
        print(f'\n  ── CumNoEWC forgetting (BWT, test-split base) ──')
        seq_final = []
        for period in range(cfg.NUM_CL_PERIODS):
            ref = cumtrain_snapshots[period]
            tm = ev.run_eval(base_seq, period_test_dfs_ref[period], cum_eis_ref[period], ref,
                             label=f'CumNoEWC-FINAL-P{period}', seed=seed)
            bwt = tm['recall@20'] - test_monitor_seq[period]
            seq_final.append({'period': period, 'bwt_test': round(bwt, 4),
                              **{f'test_{k}': v for k, v in tm.items()}})
            print(f'  CumNoEWC-final P{period}: {fmt(tm)}  BWT={bwt:+.4f}')
            if cfg.device == 'cuda':
                torch.cuda.empty_cache()
        del base_seq
        if cfg.device == 'cuda':
            torch.cuda.empty_cache()

        # ── Baseline B: All-at-once static ──────────────────────────
        print(f'\n── Baseline B: All-at-once static  (seed {seed}, '
              f'fixed HIDDEN={cfg.HIDDEN_STATIC} HEADS={cfg.HEADS_STATIC}) ──')
        ct_sta = ComplexityTracker('all_at_once')
        base_sta = build_static(cfg, dm.NUM_BRANDS, dm.NUM_MAIN)

        all_df = pd.concat([dm.pretrain_df] + period_train_dfs_ref, ignore_index=True)
        all_ei = dm.make_edge_index(all_df)
        ei_c, gu_c, nu_c, lmap_c, ne_c = all_ei
        mask_c = all_df['u_idx'].apply(lambda u: int(u) in lmap_c)
        all_df_f = all_df[mask_c].reset_index(drop=True)
        pul_c = torch.tensor([lmap_c[int(v)] for v in all_df_f['u_idx'].values], dtype=torch.long)
        pi_c = torch.tensor(all_df_f['i_idx'].values, dtype=torch.long)
        N_c = len(pul_c)
        ht_c = dm.build_hist_tensor(nu_c, lmap_c)
        opt = torch.optim.Adam(base_sta.parameters(), lr=cfg.LR, weight_decay=cfg.WD)
        ei_c_gpu, gu_c_gpu, ht_c_gpu = dm.to_gpu(ei_c, gu_c, ht_c)

        st_ep = cfg.PRETRAIN_EPOCHS + cfg.NUM_CL_PERIODS * cfg.CL_EPOCHS
        for epoch in range(1, st_ep + 1):
            ct_sta.start_epoch()
            simple_train_epoch(cfg, base_sta, opt, ei_c_gpu, gu_c_gpu, ht_c_gpu,
                               dm.item_features, pul_c, pi_c, N_c)
            ct_sta.end_epoch()
            if epoch % 10 == 0 or epoch == st_ep:
                print(f'  [Static] epoch {epoch:02d}/{st_ep}')
        del ei_c_gpu, gu_c_gpu, ht_c_gpu
        if cfg.device == 'cuda':
            torch.cuda.empty_cache()

        sta_metrics = []
        for period in range(cfg.NUM_CL_PERIODS):
            ref = pd.concat([dm.pretrain_df] + period_train_dfs_ref[:period + 1], ignore_index=True)
            m = ev.run_eval(base_sta, period_test_dfs_ref[period], cum_eis_ref[period], ref,
                            label=f'Static-P{period}', seed=seed)
            sta_metrics.append({'period': period, **m})
            print(f'  Static P{period}: {fmt(m)}')
            if cfg.device == 'cuda':
                torch.cuda.empty_cache()

        sta_complexity = []
        for period in range(cfg.NUM_CL_PERIODS):
            crec = {
                'model': 'all_at_once', 'seed': seed, 'period': period,
                'heads': base_sta.num_heads, 'hidden': base_sta.hidden,
                'n_params': sum(p.numel() for p in base_sta.parameters() if p.requires_grad),
                'epoch_time_s': round(float(np.mean(ct_sta.epoch_times))
                                      if ct_sta.epoch_times else 0.0, 4),
                'flops_per_epoch': 2 * ne_c * base_sta.hidden * base_sta.num_heads * 2 * 2,
            }
            sta_complexity.append(crec)
        del base_sta
        if cfg.device == 'cuda':
            torch.cuda.empty_cache()

        return sta_metrics, seq_complexity, sta_complexity, repr_health_log_seq, seq_final

    # ═════════════════════════════════════════════════════════════════
    # 4. Multi-seed loop
    # ═════════════════════════════════════════════════════════════════
    def run(self) -> None:
        cfg = self.cfg
        for seed in range(cfg.N_SEEDS):
            print(f'\n{"=" * 60}')
            print(f'SEED {seed}')
            print(f'{"=" * 60}')

            results = self.run_one_seed(seed)
            self.all_seeds_dyn.append(results['dyn'])
            self.all_seeds_monitor.append(results['monitor'])
            self.all_growth.extend([{**g, 'seed': seed} for g in results['growth']])
            self.all_complexity_dyn.extend(results['complexity'])
            self.all_repr_health_dyn.extend([{**r, 'seed': seed} for r in results['repr_health']])

            shared = results['_shared']

            ewc_res, ewc_cplx, ewc_health = self.run_ewc_only_seed(
                seed, shared['period_train_dfs'], shared['period_test_dfs'], shared['cum_eis'])
            self.all_seeds_ewc.append(ewc_res)
            self.all_complexity_ewc.extend(ewc_cplx)
            self.all_repr_health_ewc.extend([{**r, 'seed': seed} for r in ewc_health])

            sta_res, seq_cplx, sta_cplx, seq_health, seq_bwt = self.run_baselines_seed(
                seed, shared['period_train_dfs'], shared['period_test_dfs'], shared['cum_eis'])
            self.all_seeds_sta.append(sta_res)
            self.all_complexity_seq.extend(seq_cplx)
            self.all_complexity_sta.extend(sta_cplx)
            self.all_repr_health_seq.extend([{**r, 'seed': seed} for r in seq_health])
            self.all_seeds_seq_bwt.append(seq_bwt)

            gc.collect()
            if cfg.device == 'cuda':
                torch.cuda.empty_cache()

        self.report()

    # ═════════════════════════════════════════════════════════════════
    # 5. Aggregation + final report + save
    # ═════════════════════════════════════════════════════════════════
    def _aggregate(self, all_seeds_list: List[list], metrics: List[str]) -> pd.DataFrame:
        rows = []
        for period in range(self.cfg.NUM_CL_PERIODS):
            row = {'period': period}
            for m in metrics:
                vals = [seed_data[period][m] for seed_data in all_seeds_list
                        if len(seed_data) > period]
                row[f'{m}_mean'] = round(float(np.mean(vals)), 4)
                row[f'{m}_std'] = round(float(np.std(vals)), 4)
            rows.append(row)
        return pd.DataFrame(rows)

    def _aggregate_complexity(self, records_list: List[dict]) -> pd.DataFrame:
        df = pd.DataFrame(records_list)
        df = df[df['period'] >= 0]
        grp = df.groupby(['model', 'period'])
        rows = []
        for (model_name, period), g in grp:
            rows.append({
                'model': model_name, 'period': period,
                'heads_mean': round(g['heads'].mean(), 2),
                'hidden_mean': round(g['hidden'].mean(), 2),
                'n_params_mean': int(g['n_params'].mean()),
                'epoch_time_s_mean': round(g['epoch_time_s'].mean(), 4),
                'epoch_time_s_std': round(g['epoch_time_s'].std(), 4),
                'flops_per_epoch_mean': int(g['flops_per_epoch'].mean()),
            })
        return pd.DataFrame(rows).sort_values(['model', 'period'])

    def _aggregate_repr_health(self, records_list: List[dict], model_name: str) -> pd.DataFrame:
        df = pd.DataFrame(records_list)
        if df.empty:
            return pd.DataFrame(columns=[
                'model', 'period', 'repr_collapse_cos_mean', 'repr_collapse_cos_std',
                'oversmooth_cos_mean', 'oversmooth_cos_std'])
        grp = df.groupby('period')
        rows = []
        for period, g in grp:
            rows.append({
                'model': model_name, 'period': period,
                'repr_collapse_cos_mean': round(g['repr_collapse_cos'].mean(), 4),
                'repr_collapse_cos_std': round(g['repr_collapse_cos'].std(), 4),
                'oversmooth_cos_mean': round(g['oversmooth_cos'].mean(), 4),
                'oversmooth_cos_std': round(g['oversmooth_cos'].std(), 4),
            })
        return pd.DataFrame(rows).sort_values('period')

    @staticmethod
    def _print_table(df: pd.DataFrame, title: str, mean_cols: List[str]) -> None:
        print(f'\n── {title} ──')
        header = 'period  ' + '  '.join(f'{c:>22}' for c in mean_cols)
        print(header)
        for _, row in df.iterrows():
            line = f'{int(row["period"]):>6}  '
            for c in mean_cols:
                mc, sc = f'{c}_mean', f'{c}_std'
                if sc in row.index:
                    line += f'{row[mc]:.4f}±{row[sc]:.4f}       '
                else:
                    line += f'{row[mc]:.4f}                  '
            print(line)

    def report(self) -> None:
        cfg = self.cfg
        print(f'\n{"=" * 70}')
        print(f'FINAL SUMMARY  ({cfg.N_SEEDS} seeds)  —  Dataset: {cfg.dataset.upper()}  '
              f'(Heterogeneous MGAT, 4-modality)')
        print(f'{"=" * 70}')

        dyn_metrics = ['test_recall@10', 'test_recall@20', 'test_ndcg@10',
                       'test_mrr', 'test_auc', 'bwt_test',
                       'test_recall@20_warm', 'test_recall@20_cold']
        base_metrics = ['recall@10', 'recall@20', 'ndcg@10', 'mrr', 'auc',
                        'recall@20_warm', 'recall@20_cold']

        dyn_agg = self._aggregate(self.all_seeds_dyn, dyn_metrics)
        ewc_agg = self._aggregate(self.all_seeds_ewc, dyn_metrics)
        sta_agg = self._aggregate(self.all_seeds_sta, base_metrics)
        seq_bwt_agg = self._aggregate(self.all_seeds_seq_bwt, dyn_metrics)

        cplx_records = (self.all_complexity_dyn + self.all_complexity_ewc +
                        self.all_complexity_seq + self.all_complexity_sta)
        cplx_agg = self._aggregate_complexity(cplx_records)

        repr_dyn_agg = self._aggregate_repr_health(self.all_repr_health_dyn, 'dynamic')
        repr_ewc_agg = self._aggregate_repr_health(self.all_repr_health_ewc, 'ewc_only')
        repr_seq_agg = self._aggregate_repr_health(self.all_repr_health_seq, 'cumulative_no_ewc')
        repr_health_agg = pd.concat([repr_dyn_agg, repr_ewc_agg, repr_seq_agg], ignore_index=True)

        self._print_table(dyn_agg,
                          f'Dynamic MGAT (EWC + Severity-Growth)  [N={cfg.N_SEEDS} seeds]',
                          ['test_recall@20', 'test_recall@20_warm', 'test_recall@20_cold',
                           'test_ndcg@10', 'test_auc', 'bwt_test'])
        self._print_table(ewc_agg,
                          f'Ablation: EWC-only, fixed H={cfg.HIDDEN_STATIC} heads={cfg.HEADS_STATIC} '
                          f'(no growth)  [N={cfg.N_SEEDS} seeds]',
                          ['test_recall@20', 'test_recall@20_warm', 'test_recall@20_cold',
                           'test_ndcg@10', 'test_auc', 'bwt_test'])
        self._print_table(seq_bwt_agg,
                          f'Baseline A forgetting: Cumulative, No-EWC, fixed H={cfg.HIDDEN_STATIC} '
                          f'heads={cfg.HEADS_STATIC}  [N={cfg.N_SEEDS} seeds]',
                          ['test_recall@20', 'test_recall@20_warm', 'test_recall@20_cold',
                           'test_ndcg@10', 'test_auc', 'bwt_test'])
        self._print_table(sta_agg,
                          f'Baseline B: All-at-once static, fixed H={cfg.HIDDEN_STATIC} '
                          f'heads={cfg.HEADS_STATIC}',
                          ['recall@20', 'recall@20_warm', 'recall@20_cold', 'ndcg@10', 'auc'])

        self._print_four_way(dyn_agg, ewc_agg, seq_bwt_agg, sta_agg)
        self._print_bwt(dyn_agg, ewc_agg, seq_bwt_agg)
        self._print_warm_cold(dyn_agg)
        self._print_growth_events()
        self._print_complexity(cplx_agg)
        self._print_repr_health(repr_health_agg, repr_dyn_agg, repr_ewc_agg, repr_seq_agg)

        self._save(dyn_agg, ewc_agg, seq_bwt_agg, sta_agg, cplx_agg, repr_health_agg)

    def _print_four_way(self, dyn_agg, ewc_agg, seq_bwt_agg, sta_agg) -> None:
        cfg = self.cfg
        print('\n── R@20 four-way comparison (mean across seeds) ──')
        print(f'{"P":>3}  {"Dynamic":>8}  {"EWC-only":>8}  {"CumNoEWC":>8}  '
              f'{"Static":>8}  {"dyn-ewc":>8}  {"dyn-cum":>8}  {"dyn-sta":>8}')
        avg_dyn_ewc, avg_dyn_cum, avg_dyn_sta = [], [], []
        for p in range(cfg.NUM_CL_PERIODS):
            d = dyn_agg.loc[dyn_agg['period'] == p, 'test_recall@20_mean'].values[0]
            e = ewc_agg.loc[ewc_agg['period'] == p, 'test_recall@20_mean'].values[0]
            c = seq_bwt_agg.loc[seq_bwt_agg['period'] == p, 'test_recall@20_mean'].values[0]
            s = sta_agg.loc[sta_agg['period'] == p, 'recall@20_mean'].values[0]
            avg_dyn_ewc.append(d - e)
            avg_dyn_cum.append(d - c)
            avg_dyn_sta.append(d - s)
            print(f'{p:>3}  {d:>8.4f}  {e:>8.4f}  {c:>8.4f}  {s:>8.4f}  '
                  f'{d - e:>+8.4f}  {d - c:>+8.4f}  {d - s:>+8.4f}')
        print(f'{"avg":>3}  {"":>8}  {"":>8}  {"":>8}  {"":>8}  '
              f'{np.mean(avg_dyn_ewc):>+8.4f}  {np.mean(avg_dyn_cum):>+8.4f}  '
              f'{np.mean(avg_dyn_sta):>+8.4f}')

    def _print_bwt(self, dyn_agg, ewc_agg, seq_bwt_agg) -> None:
        cfg = self.cfg
        print('\n── BWT comparison (mean ± std across seeds, same-split base) ──')
        print(f'{"P":>3}  {"Dyn BWT":>14}  {"EWC-only BWT":>14}  {"CumNoEWC BWT":>14}')
        for p in range(cfg.NUM_CL_PERIODS):
            db = dyn_agg.loc[dyn_agg['period'] == p, 'bwt_test_mean'].values[0]
            ds = dyn_agg.loc[dyn_agg['period'] == p, 'bwt_test_std'].values[0]
            eb = ewc_agg.loc[ewc_agg['period'] == p, 'bwt_test_mean'].values[0]
            es = ewc_agg.loc[ewc_agg['period'] == p, 'bwt_test_std'].values[0]
            cb = seq_bwt_agg.loc[seq_bwt_agg['period'] == p, 'bwt_test_mean'].values[0]
            cs = seq_bwt_agg.loc[seq_bwt_agg['period'] == p, 'bwt_test_std'].values[0]
            print(f'{p:>3}  {db:>+8.4f}±{ds:.4f}  {eb:>+8.4f}±{es:.4f}  {cb:>+8.4f}±{cs:.4f}')
        print(f'{"avg":>3}  '
              f'{dyn_agg["bwt_test_mean"].mean():>+8.4f}           '
              f'{ewc_agg["bwt_test_mean"].mean():>+8.4f}           '
              f'{seq_bwt_agg["bwt_test_mean"].mean():>+8.4f}')

    def _print_warm_cold(self, dyn_agg) -> None:
        cfg = self.cfg
        print('\n── Warm vs Cold R@20 — Dynamic model ──')
        print(f'{"P":>3}  {"Combined":>10}  {"Warm":>10}  {"Cold":>10}')
        for p in range(cfg.NUM_CL_PERIODS):
            c = dyn_agg.loc[dyn_agg['period'] == p, 'test_recall@20_mean'].values[0]
            w = dyn_agg.loc[dyn_agg['period'] == p, 'test_recall@20_warm_mean'].values[0]
            k = dyn_agg.loc[dyn_agg['period'] == p, 'test_recall@20_cold_mean'].values[0]
            print(f'{p:>3}  {c:>10.4f}  {w:>10.4f}  {k:>10.4f}')

    def _print_growth_events(self) -> None:
        print('\n── Growth events (all seeds) ──')
        if self.all_growth:
            print(pd.DataFrame(self.all_growth).to_string(index=False))
        else:
            print('  None triggered.')

    def _print_complexity(self, cplx_agg: pd.DataFrame) -> None:
        print('\n── Complexity comparison (mean across seeds per model per period) ──')
        print(f'{"model":<20}  {"P":>2}  {"heads":>5}  {"hidden":>7}  '
              f'{"#params":>10}  {"t/epoch(s)":>12}  {"FLOPs/epoch":>14}')
        for _, row in cplx_agg.iterrows():
            print(f'{row["model"]:<20}  {int(row["period"]):>2}  '
                  f'{row["heads_mean"]:>5.1f}  {row["hidden_mean"]:>7.1f}  '
                  f'{row["n_params_mean"]:>10,}  '
                  f'{row["epoch_time_s_mean"]:>10.4f}±{row["epoch_time_s_std"]:.4f}  '
                  f'{row["flops_per_epoch_mean"]:>14,}')

        print('\n── Complexity summary (avg across periods) ──')
        cplx_sum = (cplx_agg.groupby('model')
                    .agg(n_params_mean=('n_params_mean', 'mean'),
                         epoch_time_mean=('epoch_time_s_mean', 'mean'),
                         flops_mean=('flops_per_epoch_mean', 'mean'))
                    .reset_index())
        print(f'{"model":<20}  {"avg #params":>12}  {"avg t/ep (s)":>14}  {"avg FLOPs/ep":>16}')
        for _, row in cplx_sum.iterrows():
            print(f'{row["model"]:<20}  {row["n_params_mean"]:>12,.0f}  '
                  f'{row["epoch_time_mean"]:>14.4f}  '
                  f'{row["flops_mean"]:>16,.0f}')

    def _print_repr_health(self, repr_health_agg, repr_dyn_agg, repr_ewc_agg, repr_seq_agg) -> None:
        cfg = self.cfg
        print('\n── Representation health: repr-collapse & oversmoothing '
              f'(mean ± std across seeds)  [N={cfg.N_SEEDS} seeds] ──')
        print(f'{"model":<20}  {"P":>2}  {"repr_collapse_cos":>20}  {"oversmooth_cos":>18}')
        for _, row in repr_health_agg.iterrows():
            print(f'{row["model"]:<20}  {int(row["period"]):>2}  '
                  f'{row["repr_collapse_cos_mean"]:>10.4f}±{row["repr_collapse_cos_std"]:.4f}  '
                  f'{row["oversmooth_cos_mean"]:>10.4f}±{row["oversmooth_cos_std"]:.4f}')

        print('\n── Representation health: side-by-side by period '
              '(dynamic vs ewc_only vs cumulative_no_ewc) ──')
        print(f'{"P":>3}  '
              f'{"repr_dyn":>10}  {"repr_ewc":>10}  {"repr_seq":>10}  '
              f'{"smooth_dyn":>11}  {"smooth_ewc":>11}  {"smooth_seq":>11}')

        def _get(df, col, p):
            v = df.loc[df['period'] == p, col].values
            return float(v[0]) if len(v) else float('nan')

        for p in range(cfg.NUM_CL_PERIODS):
            rd = _get(repr_dyn_agg, 'repr_collapse_cos_mean', p)
            re = _get(repr_ewc_agg, 'repr_collapse_cos_mean', p)
            rs = _get(repr_seq_agg, 'repr_collapse_cos_mean', p)
            sd = _get(repr_dyn_agg, 'oversmooth_cos_mean', p)
            se = _get(repr_ewc_agg, 'oversmooth_cos_mean', p)
            ss = _get(repr_seq_agg, 'oversmooth_cos_mean', p)
            print(f'{p:>3}  {rd:>10.4f}  {re:>10.4f}  {rs:>10.4f}  '
                  f'{sd:>11.4f}  {se:>11.4f}  {ss:>11.4f}')

    def _save(self, dyn_agg, ewc_agg, seq_bwt_agg, sta_agg, cplx_agg, repr_health_agg) -> None:
        cfg = self.cfg
        out, ds = cfg.out, cfg.dataset
        dyn_agg.to_csv(out / f'results_dynamic_{ds}.csv', index=False)
        ewc_agg.to_csv(out / f'results_ewconly_{ds}.csv', index=False)
        seq_bwt_agg.to_csv(out / f'results_cumnewc_bwt_{ds}.csv', index=False)
        sta_agg.to_csv(out / f'results_static_{ds}.csv', index=False)
        cplx_agg.to_csv(out / f'complexity_{ds}.csv', index=False)
        repr_health_agg.to_csv(out / f'repr_health_{ds}.csv', index=False)
        if self.all_growth:
            pd.DataFrame(self.all_growth).to_csv(out / f'growth_log_{ds}.csv', index=False)
        for seed in range(cfg.N_SEEDS):
            pd.DataFrame(self.all_seeds_dyn[seed]).to_csv(
                out / f'raw_dyn_seed{seed}_{ds}.csv', index=False)
            pd.DataFrame(self.all_seeds_ewc[seed]).to_csv(
                out / f'raw_ewc_seed{seed}_{ds}.csv', index=False)
            pd.DataFrame(self.all_seeds_seq_bwt[seed]).to_csv(
                out / f'raw_cumnewc_bwt_seed{seed}_{ds}.csv', index=False)
            pd.DataFrame(self.all_seeds_sta[seed]).to_csv(
                out / f'raw_static_seed{seed}_{ds}.csv', index=False)

        print(f'\nAll results saved to {out}/')
        print('Done.')
