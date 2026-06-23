from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd

from .model import LightGCN


class ComplexityTracker:
    """
    Records per-epoch wall-clock time and estimated FLOPs for one model variant.

    FLOPs estimate (forward + backward, message-passing dominant term):
        2 × nnz(adj) × hidden × K × 2
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.epoch_times: list[float] = []
        self._ep_start: Optional[float] = None
        self.records: list[dict] = []

    def start_epoch(self):
        self._ep_start = time.perf_counter()

    def end_epoch(self):
        if self._ep_start is not None:
            self.epoch_times.append(time.perf_counter() - self._ep_start)
            self._ep_start = None

    def log_period(self, period: int, model: LightGCN, adj_nnz: int, seed: int) -> dict:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mean_t = float(np.mean(self.epoch_times)) if self.epoch_times else 0.0
        flops = 2 * adj_nnz * model.hidden * model.K * 2
        rec = {
            "model": self.model_name,
            "seed": seed,
            "period": period,
            "K": model.K,
            "hidden": model.hidden,
            "n_params": n_params,
            "epoch_time_s": round(mean_t, 4),
            "flops_per_epoch": flops,
        }
        self.records.append(rec)
        self.epoch_times = []
        return rec

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)
