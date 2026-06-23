"""
complexity.py
=============
Records per-epoch wall-clock time and estimated FLOPs for every model
variant, so adaptive growth can be weighed against fixed-capacity
baselines on a cost basis, not just accuracy.
"""
from __future__ import annotations

import time
from typing import List

import numpy as np
import pandas as pd

from .model import HeteroMGAT


class ComplexityTracker:
    """
    FLOPs estimate (forward pass, message-passing dominant term):
        2 × nnz(adj) × hidden × heads × 2 layers   (sparse-ish attn matmul,
                                                      ×2 for fwd+bwd)
    Call .start_epoch() before each train epoch, .end_epoch() after.
    Call .log_period() to store a snapshot for the current period.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.epoch_times: List[float] = []
        self._ep_start = None
        self.records: List[dict] = []

    def start_epoch(self) -> None:
        self._ep_start = time.perf_counter()

    def end_epoch(self) -> None:
        if self._ep_start is not None:
            self.epoch_times.append(time.perf_counter() - self._ep_start)
            self._ep_start = None

    def log_period(self, period: int, model: HeteroMGAT, n_edges: int, seed: int) -> dict:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mean_t = float(np.mean(self.epoch_times)) if self.epoch_times else 0.0
        # FLOPs: 2 × edges × hidden × heads × 2 GAT layers (forward), ×2 for backward
        flops = 2 * n_edges * model.hidden * model.num_heads * 2 * 2
        self.records.append({
            'model': self.model_name,
            'seed': seed,
            'period': period,
            'heads': model.num_heads,
            'hidden': model.hidden,
            'n_params': n_params,
            'epoch_time_s': round(mean_t, 4),
            'flops_per_epoch': flops,
        })
        self.epoch_times = []
        return self.records[-1]

    def summary_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)
