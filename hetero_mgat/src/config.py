"""
config.py
=========
Single source of truth for all hyperparameters. Import `Config` and
construct one instance (`cfg = Config()`), or override fields for
experiments, e.g. `cfg = Config(N_SEEDS=1, HIDDEN_INIT=64)`.
"""
from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass
class Config:
    # ── Paths ───────────────────────────────────────────────────────────
    root: Path = Path('/kaggle/input/datasets/zainabhalhoul/amazon-appliances/amazon_appliances_3core')
    out: Path = Path('/kaggle/working')
    dataset: str = 'appliances'

    # ── Device ──────────────────────────────────────────────────────────
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Multi-seed ──────────────────────────────────────────────────────
    N_SEEDS: int = 3

    # ── Dynamic model (grows from this init) ───────────────────────────
    HIDDEN_INIT: int = 128
    HEADS_INIT: int = 3
    DROPOUT: float = 0.1

    # ── Static / EWC-only / Cumulative-No-EWC / All-at-once baselines ──
    HIDDEN_STATIC: int = 256
    HEADS_STATIC: int = 3

    # ── Growth limits (dynamic model only) ─────────────────────────────
    HIDDEN_MAX: int = 256
    HEADS_MAX: int = 6

    # ── Feature dims ────────────────────────────────────────────────────
    CLIP_DIM: int = 128
    HASH_SIZE: int = 30_000
    CAT_EMB_DIM: int = 32
    TOP_K_SUBS: int = 30
    META_DIM: int = 5
    META_BN_DIM: int = 16

    # ── Training ────────────────────────────────────────────────────────
    PRETRAIN_EPOCHS: int = 30
    CL_EPOCHS: int = 15
    BATCH: int = 1024
    PHASE2_BATCH: int = 512
    LR: float = 1e-4
    LR_WARMUP: float = 1e-4
    WARMUP_EPOCHS: int = 5
    WD: float = 1e-5
    TRAIN_RATIO: float = 0.70
    VALID_RATIO: float = 0.15

    # ── EWC ─────────────────────────────────────────────────────────────
    EWC_LAMBDA_BASE: float = 3
    EWC_WARMUP_EPOCHS: int = 3
    EWC_SAMPLES: int = 512
    EWC_BATCH: int = 64
    EWC_BOOST: float = 3.0
    EWC_BOOST_MAX: int = 5

    # ── Growth signals ──────────────────────────────────────────────────
    FORGET_THR: float = 0.08
    MODAL_DOM_CONSEC: int = 1
    PLATEAU_WINDOW: int = 8
    PLATEAU_REL_DROP: float = 0.05
    REPR_COLLAPSE_THR: float = 0.5
    OVERSMOOTH_THR: float = 0.80
    SIGNAL_SAMPLE: int = 256

    # ── Growth severity ───────────────────────────────────────────────
    SEV_THRESHOLD: float = 0.45

    # ── S2 gate-spread threshold ──────────────────────────────────────
    N_MODALITIES: int = 4

    # ── Evaluation ──────────────────────────────────────────────────────
    EVAL_USERS: int = 300
    NEG_SAMPLES: int = 99
    NUM_CL_PERIODS: int = 5
    PRETRAIN_CUT: str = '2017-01-01'

    @property
    def DATA(self) -> Path:
        return self.root / 'data'

    @property
    def S2_SPREAD_THR(self) -> float:
        """fires when max(gate) - min(gate) > 1/(n_modalities+1)"""
        return 1.0 / (self.N_MODALITIES + 1)

    def __post_init__(self):
        self.out.mkdir(exist_ok=True, parents=True)

    def summary(self) -> str:
        lines = [
            f'Dataset     : {self.dataset}',
            f'Device      : {self.device}',
            f'Seeds       : {self.N_SEEDS}',
            f'Dynamic init: HIDDEN={self.HIDDEN_INIT}  HEADS={self.HEADS_INIT}  '
            f'(grows up to H={self.HIDDEN_MAX}, heads={self.HEADS_MAX})',
            f'Static fixed: HIDDEN={self.HIDDEN_STATIC}  HEADS={self.HEADS_STATIC}  '
            f'(EWC-only, Cumulative-No-EWC, All-at-once)',
            f'CLIP dim    : {self.CLIP_DIM} (PCA-128)',
            f'EWC lambda_base={self.EWC_LAMBDA_BASE}',
            f'S2 spread threshold = 1/({self.N_MODALITIES}+1) = {self.S2_SPREAD_THR:.4f}',
            f'Growth SEV_THRESHOLD = {self.SEV_THRESHOLD}  '
            f'(sev_width→hidden, sev_depth→heads; GAT depth fixed at 2 layers)',
            f'Eval   : cold users → mean-item fallback  |  graph: cumulative',
            f'Gate fixes : uniform-init [GATE-1]  '
            f'meta-bottleneck 5→{self.META_BN_DIM}→H [GATE-2]  '
            f'post-growth reset [GATE-3]',
        ]
        return '\n'.join(lines)
