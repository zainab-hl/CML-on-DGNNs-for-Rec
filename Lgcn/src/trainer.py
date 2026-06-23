from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from .complexity import ComplexityTracker
from .config import Config
from .data import DataModule
from .ewc import EWC
from .model import LightGCN

if TYPE_CHECKING:
    from .evaluator import Evaluator


class Trainer:
    """BPR training, Fisher accumulation, and per-period orchestration."""

    def __init__(self, cfg: Config, data: DataModule):
        self.cfg = cfg
        self.data = data

    def _build_model(self, hidden: int, K: int) -> LightGCN:
        return LightGCN(
            self.data.num_users, self.data.num_items,
            self.data.T_text, self.data.T_image,
            self.cfg, hidden=hidden, K=K,
        ).to(self.cfg.device)

    def build_dynamic(self) -> LightGCN:
        return self._build_model(self.cfg.hidden_init, self.cfg.k_init)

    def build_static(self) -> LightGCN:
        return self._build_model(self.cfg.hidden_static, self.cfg.k_static)

    def train_epoch_ewc(
        self,
        model: LightGCN,
        optimizer,
        adj,
        global_u_gpu: torch.Tensor,
        pos_u_local: torch.Tensor,
        pos_i: torch.Tensor,
        N_: int,
        ewc: Optional[EWC] = None,
        warmup: float = 1.0,
    ):
        cfg = self.cfg
        model.train()
        perm = torch.randperm(N_)
        tot_bpr, steps, gates = 0.0, 0, None

        for s in range(0, N_, cfg.batch):
            idx = perm[s: s + cfg.batch]
            pu, pi = pos_u_local[idx], pos_i[idx]
            ni = torch.randint(0, self.data.num_items, (len(idx),), device=cfg.device)

            optimizer.zero_grad()
            x_u, x_i, gates_b, _, _ = model.encode(adj, global_u_gpu)
            ps = model.decode(x_u, x_i, pu.to(cfg.device), pi.to(cfg.device))
            ns = model.decode(x_u, x_i, pu.to(cfg.device), ni)
            bpr = -F.logsigmoid(ps - ns).mean()

            ewc_val = torch.tensor(0.0, device=cfg.device)
            if ewc is not None:
                ewc_val = ewc.penalty(model, warmup)

            (bpr + ewc_val).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tot_bpr += bpr.item()
            steps += 1
            if gates is None:
                gates = gates_b.detach()
            del x_u, x_i, ps, ns, bpr, gates_b

        if cfg.device == "cuda":
            torch.cuda.empty_cache()
        return tot_bpr / max(steps, 1), (ewc_val.item() if ewc else 0.0), gates

    def train_epoch_simple(
        self, model, optimizer, adj, global_u_gpu, pos_u_local, pos_i, N_
    ):
        cfg = self.cfg
        model.train()
        perm = torch.randperm(N_)
        tot_bpr, steps = 0.0, 0

        for s in range(0, N_, cfg.batch):
            idx = perm[s: s + cfg.batch]
            pu, pi = pos_u_local[idx], pos_i[idx]
            ni = torch.randint(0, self.data.num_items, (len(idx),), device=cfg.device)

            optimizer.zero_grad()
            x_u, x_i, _, _, _ = model.encode(adj, global_u_gpu)
            ps = model.decode(x_u, x_i, pu.to(cfg.device), pi.to(cfg.device))
            ns = model.decode(x_u, x_i, pu.to(cfg.device), ni)
            bpr = -F.logsigmoid(ps - ns).mean()
            bpr.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot_bpr += bpr.item()
            steps += 1
            del x_u, x_i, ps, ns, bpr

        if cfg.device == "cuda":
            torch.cuda.empty_cache()
        return tot_bpr / max(steps, 1)

    def accumulate_fisher(self, model, adj, global_u_gpu, sample_df, local_u_map) -> tuple[dict, dict]:
        cfg = self.cfg
        F_ = {n: torch.zeros_like(p) for n, p in model.lgcn_parameters()}
        model.train()
        sample = sample_df.sample(min(cfg.ewc_samples, len(sample_df)), random_state=42)
        n_batches = 0

        for start in range(0, len(sample), cfg.ewc_batch):
            batch = sample.iloc[start: start + cfg.ewc_batch]
            mask = batch["u_idx"].apply(lambda u: int(u) in local_u_map)
            batch = batch[mask]
            if batch.empty:
                continue

            pu = torch.tensor([local_u_map[int(v)] for v in batch["u_idx"].values], dtype=torch.long)
            pi = torch.tensor(batch["i_idx"].values, dtype=torch.long)
            ni = torch.randint(0, self.data.num_items, (len(batch),))

            model.zero_grad()
            x_u, x_i, _, _, _ = model.encode(adj, global_u_gpu)
            ps = model.decode(x_u, x_i, pu.to(cfg.device), pi.to(cfg.device))
            ns = model.decode(x_u, x_i, pu.to(cfg.device), ni.to(cfg.device))
            loss = F.binary_cross_entropy_with_logits(
                torch.cat([ps, ns]),
                torch.cat([torch.ones_like(ps), torch.zeros_like(ns)]),
            )
            loss.backward()
            for n, p in model.lgcn_parameters():
                if p.grad is not None:
                    F_[n] += p.grad.detach().pow(2)
            n_batches += 1
            del x_u, x_i, ps, ns, loss

        if n_batches > 0:
            for k in F_:
                F_[k] /= n_batches
        model.zero_grad()
        snap = {n: p.clone().detach() for n, p in model.lgcn_parameters()}
        if cfg.device == "cuda":
            torch.cuda.empty_cache()
        return snap, F_

    def run_period(
        self,
        model: LightGCN,
        optimizer,
        train_df,
        cum_adj,
        global_u: torch.Tensor,
        local_u_map: dict,
        epochs: int,
        ewc: Optional[EWC],
        past_valid_records: list,
        seed: int,
        tag: str = "",
        complexity_tracker: Optional[ComplexityTracker] = None,
        evaluator: "Evaluator" = None,
    ):
        cfg = self.cfg
        pos_u_local, pos_i, N_, train_f = self.data.prep_tensors(train_df, local_u_map)
        print(f"  Training [{tag}]  edges={N_:,}  users={len(local_u_map):,}  K={model.K}  H={model.hidden}")

        global_u_gpu = global_u.to(cfg.device)
        gate_history, bpr_history = [], []
        avg_bpr = 0.0

        for epoch in range(1, epochs + 1):
            warmup = min(1.0, epoch / cfg.ewc_warmup_epochs) if ewc else 1.0
            if complexity_tracker:
                complexity_tracker.start_epoch()
            avg_bpr, ewc_val, gates = self.train_epoch_ewc(
                model, optimizer, cum_adj, global_u_gpu,
                pos_u_local, pos_i, N_, ewc=ewc, warmup=warmup,
            )
            if complexity_tracker:
                complexity_tracker.end_epoch()
            gate_history.append(gates)
            bpr_history.append(avg_bpr)
            ep_str = f"  ewc={ewc_val:.4f}(×{warmup:.2f})" if ewc else ""
            print(f"    epoch {epoch:02d}/{epochs}  bpr={avg_bpr:.4f}{ep_str}")

        print("  → Capturing Fisher...")
        snap, F_ = self.accumulate_fisher(model, cum_adj, global_u_gpu, train_f, local_u_map)
        new_ewc = EWC(snap, F_, cfg)

        print(f"  → Validating on {len(past_valid_records)} period(s)...")
        valid_log = {"tag": tag}
        forgetting_detected = False

        for rec in past_valid_records:
            pidx = rec["period_idx"]
            vm = evaluator.run_eval(model, rec["valid_df"], rec["cum_adj"], rec["ref_train_df"], seed=seed)
            valid_log[f"P{pidx}_r20"] = vm["recall@20"]
            valid_log[f"P{pidx}_auc"] = vm["auc"]
            if cfg.device == "cuda":
                torch.cuda.empty_cache()

            if pidx != past_valid_records[-1]["period_idx"]:
                base = rec.get("best_r20") or 0.0
                if base > 0:
                    drop = (base - vm["recall@20"]) / base
                    if drop > cfg.forget_thr:
                        forgetting_detected = True
                        print(f"    ⚠ Forgetting P{pidx}: {base:.4f}→{vm['recall@20']:.4f} drop={drop:.1%}")
            if vm["recall@20"] > (rec.get("best_r20") or 0.0):
                rec["best_r20"] = vm["recall@20"]

        parts = [
            "P{} R20={:.4f}".format(r["period_idx"], valid_log.get(f"P{r['period_idx']}_r20", 0))
            for r in past_valid_records
        ]
        print(f"  Valid: {'  '.join(parts)}")

        if forgetting_detected and ewc is not None:
            ewc.boost()

        del global_u_gpu
        if cfg.device == "cuda":
            torch.cuda.empty_cache()
        return valid_log, new_ewc, avg_bpr, gate_history, bpr_history
