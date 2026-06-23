"""
ewc.py
======
Elastic Weight Consolidation: penalty term, Fisher-information
accumulation, and shape-aware padding/pruning across growth events.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import Config
from .model import HeteroMGAT, ItemFeatures


class EWC:
    def __init__(self, cfg: Config, params_snap: Dict[str, torch.Tensor],
                fisher_snap: Dict[str, torch.Tensor]):
        self.cfg = cfg
        self.params = params_snap
        self.fisher = fisher_snap
        self.boost_count = 0
        self.boost_mult = 1.0

        f_mean = float(np.mean([v.mean().item() for v in self.fisher.values()]))
        flag = '  ⚠ near-zero' if f_mean < 1e-6 else '  ✓'
        self.lambda_ = min(cfg.EWC_LAMBDA_BASE / (f_mean + 1e-8), 10_000.0)
        print(f'    Fisher mean={f_mean:.6f}{flag}  lambda={self.lambda_:.2f}')

    def boost(self) -> None:
        if self.boost_count < self.cfg.EWC_BOOST_MAX:
            self.boost_mult *= self.cfg.EWC_BOOST
            self.boost_count += 1
            print(f'    EWC boosted → ×{self.boost_mult:.1f}  '
                  f'({self.boost_count}/{self.cfg.EWC_BOOST_MAX})')

    def penalty(self, model: HeteroMGAT, warmup_factor: float = 1.0) -> torch.Tensor:
        lam = self.lambda_ * self.boost_mult * warmup_factor
        pen = torch.tensor(0., device=self.cfg.device)
        for name, p in model.gnn_parameters():
            if name in self.fisher and self.params[name].shape == p.shape:
                pen = pen + (
                    self.fisher[name] * (p - self.params[name]).pow(2)
                ).sum()
        return lam * pen

    def pad_to_model(self, model: HeteroMGAT) -> None:
        """Zero-pad Fisher/params to new shapes after hidden growth."""
        current = {n: p.shape for n, p in model.gnn_parameters()}
        padded = 0
        for n in list(self.fisher.keys()):
            if n not in current:
                del self.fisher[n]
                del self.params[n]
                continue
            old_s = self.fisher[n].shape
            new_s = current[n]
            if old_s == new_s:
                continue
            nf = torch.zeros(new_s, device=self.fisher[n].device)
            np_ = torch.zeros(new_s, device=self.params[n].device)
            sl = tuple(slice(0, s) for s in old_s)
            nf[sl] = self.fisher[n]
            np_[sl] = self.params[n]
            self.fisher[n] = nf
            self.params[n] = np_
            padded += 1
        for n, p in model.gnn_parameters():
            if n not in self.fisher:
                self.fisher[n] = torch.zeros_like(p.detach())
                self.params[n] = p.clone().detach()
                padded += 1
        if padded:
            print(f'    EWC zero-padded {padded} entries.')

    def prune_to_model(self, model: HeteroMGAT) -> None:
        current = {n: p.shape for n, p in model.gnn_parameters()}
        stale = [n for n in list(self.fisher)
                 if n not in current or current[n] != self.fisher[n].shape]
        for n in stale:
            del self.fisher[n]
            del self.params[n]
        if stale:
            print(f'    EWC pruned {len(stale)} stale entries.')

    def merge_from(self, old: 'EWC | None', decay: float = 0.7) -> None:
        """In-place: blend `old` Fisher values into self (used between CL
        periods), and carry over the boost state. `self` is the
        newly-computed EWC; `old` is the previous period's EWC."""
        if old is None:
            return
        for n in self.fisher:
            if n in old.fisher and old.fisher[n].shape == self.fisher[n].shape:
                self.fisher[n] = decay * old.fisher[n] + (1 - decay) * self.fisher[n]
        self.boost_count = old.boost_count
        self.boost_mult = old.boost_mult


def accumulate_fisher(cfg: Config, model: HeteroMGAT, ei_gpu: Dict,
                      global_u_gpu: torch.Tensor, hist_gpu: torch.Tensor,
                      feats: ItemFeatures, sample_df: pd.DataFrame,
                      local_u_map: Dict[int, int]):
    """Returns (params_snapshot, fisher_dict)."""
    F_ = {n: torch.zeros_like(p) for n, p in model.gnn_parameters()}
    model.train()
    sample = sample_df.sample(min(cfg.EWC_SAMPLES, len(sample_df)), random_state=42)
    n_batches = 0

    for start in range(0, len(sample), cfg.EWC_BATCH):
        batch = sample.iloc[start:start + cfg.EWC_BATCH]
        mask = batch['u_idx'].apply(lambda u: int(u) in local_u_map)
        batch = batch[mask]
        if batch.empty:
            continue

        pu = torch.tensor(
            [local_u_map[int(v)] for v in batch['u_idx'].values], dtype=torch.long)
        pi = torch.tensor(batch['i_idx'].values, dtype=torch.long)
        ni = torch.randint(0, feats.num_items, (len(batch),))

        model.zero_grad()
        x, _, _, _ = model.encode(ei_gpu, global_u_gpu, hist_gpu, feats)
        ps = model.decode(x, pu, pi, cfg.device)
        ns = model.decode(x, pu, ni, cfg.device)
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([ps, ns]),
            torch.cat([torch.ones_like(ps), torch.zeros_like(ns)]),
        )
        loss.backward()
        for n, p in model.gnn_parameters():
            if p.grad is not None:
                F_[n] += p.grad.detach().pow(2)
        n_batches += 1
        del x, ps, ns, loss

    if n_batches > 0:
        for k in F_:
            F_[k] /= n_batches
    model.zero_grad()
    snap = {n: p.clone().detach() for n, p in model.gnn_parameters()}
    if cfg.device == 'cuda':
        torch.cuda.empty_cache()
    return snap, F_
