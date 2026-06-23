from __future__ import annotations

import numpy as np
import torch

from .config import Config
from .model import LightGCN


class EWC:
    """
    Elastic Weight Consolidation regulariser.

    Computes a Fisher-scaled quadratic penalty to protect important
    parameters from large updates during continual learning.
    """

    def __init__(self, params_snap: dict, fisher_snap: dict, cfg: Config):
        self.cfg = cfg
        self.params = params_snap
        self.fisher = fisher_snap
        self.boost_count = 0
        self.boost_mult = 1.0

        f_mean = float(np.mean([v.mean().item() for v in self.fisher.values()]))
        flag = "  ⚠ near-zero" if f_mean < 1e-6 else "  ✓"
        self.lambda_ = min(cfg.ewc_lambda_base / (f_mean + 1e-8), 10_000.0)
        print(f"    Fisher mean={f_mean:.6f}{flag}  lambda={self.lambda_:.2f}")

    def penalty(self, model: LightGCN, warmup: float = 1.0) -> torch.Tensor:
        lam = self.lambda_ * self.boost_mult * warmup
        pen = torch.tensor(0.0, device=self.cfg.device)
        for n, p in model.lgcn_parameters():
            if n in self.fisher and self.params[n].shape == p.shape:
                pen = pen + (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
        return lam * pen

    def boost(self):
        if self.boost_count < self.cfg.ewc_boost_max:
            self.boost_mult *= self.cfg.ewc_boost
            self.boost_count += 1
            print(f"    EWC boosted → ×{self.boost_mult:.1f} ({self.boost_count}/{self.cfg.ewc_boost_max})")

    def merge_with(self, prev: EWC, decay: float = 0.7):
        """Exponential moving average of Fisher matrices across periods."""
        for n in self.fisher:
            if n in prev.fisher and prev.fisher[n].shape == self.fisher[n].shape:
                self.fisher[n] = decay * prev.fisher[n] + (1 - decay) * self.fisher[n]
        self.boost_count = prev.boost_count
        self.boost_mult = prev.boost_mult

    def pad_to_model(self, model: LightGCN):
        """Zero-pad Fisher and param snapshots to match a grown model's shapes."""
        current_shapes = {n: p.shape for n, p in model.lgcn_parameters()}
        padded = 0
        for n in list(self.fisher.keys()):
            if n not in current_shapes:
                del self.fisher[n]
                del self.params[n]
                continue
            old_s, new_s = self.fisher[n].shape, current_shapes[n]
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
        for n, p in model.lgcn_parameters():
            if n not in self.fisher:
                self.fisher[n] = torch.zeros_like(p.detach())
                self.params[n] = p.clone().detach()
                padded += 1
        if padded:
            print(f"    EWC zero-padded {padded} entries.")
