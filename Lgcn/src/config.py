from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch


@dataclass
class Config:
    # ── Dataset ────────────────────────────────────────────────────────────
    dataset: str = "sports"
    data_dir: Path = Path(
        "/kaggle/input/datasets/zainabhalhoul/amazon-baby-dataset"
        "/sports-20260405T095454Z-3-001/sports"
    )
    out_dir: Path = Path("/kaggle/working")
    n_seeds: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Embedding dimensions ───────────────────────────────────────────────
    text_dim_out: int = 256
    image_dim_out: int = 256

    # ── Dynamic model init (grows during CL) ──────────────────────────────
    hidden_init: int = 128
    k_init: int = 3
    user_emb_dim: int = 64

    # ── Fixed capacity for all static baselines ────────────────────────────
    hidden_static: int = 256
    k_static: int = 4

    # ── Growth limits ──────────────────────────────────────────────────────
    hidden_max: int = 512
    k_max: int = 6

    # ── Training ───────────────────────────────────────────────────────────
    pretrain_epochs: int = 30
    cl_epochs: int = 20
    batch: int = 2048
    lr: float = 1e-3
    lr_warmup: float = 5e-5
    warmup_epochs: int = 3
    weight_decay: float = 1e-5
    dropout: float = 0.2
    train_ratio: float = 0.70
    valid_ratio: float = 0.15

    # ── EWC ────────────────────────────────────────────────────────────────
    ewc_lambda_base: float = 15.0
    ewc_warmup_epochs: int = 3
    ewc_samples: int = 300
    ewc_batch: int = 64
    ewc_boost: float = 3.0
    ewc_boost_max: int = 5

    # ── Growth signals ─────────────────────────────────────────────────────
    forget_thr: float = 0.08
    modal_dom_consec: int = 2
    plateau_window: int = 8
    plateau_rel_drop: float = 0.04
    repr_collapse_thr: float = 0.4
    repr_collapse_sample: int = 512
    oversmooth_thr: float = 0.60
    oversmooth_sample: int = 512

    # ── Severity thresholds ────────────────────────────────────────────────
    sev_threshold: float = 0.4
    n_modalities: int = 2

    # ── Evaluation ─────────────────────────────────────────────────────────
    eval_users: int = 500
    neg_samples: int = 49
    num_periods: int = 4

    # ── Temporal boundaries ────────────────────────────────────────────────
    pretrain_cut: pd.Timestamp = field(
        default_factory=lambda: pd.Timestamp("2012-07-01", tz="UTC")
    )
    period_cuts: list = field(
        default_factory=lambda: [
            pd.Timestamp("2013-01-01", tz="UTC"),
            pd.Timestamp("2013-07-01", tz="UTC"),
            pd.Timestamp("2014-01-01", tz="UTC"),
        ]
    )

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        self.data_dir = Path(self.data_dir)
        self.out_dir.mkdir(exist_ok=True)
        self.s2_spread_thr: float = 1.0 / (self.n_modalities + 1)
