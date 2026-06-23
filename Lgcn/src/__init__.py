from .config import Config
from .data import DataModule
from .model import LightGCN, ModalFusion
from .ewc import EWC
from .signals import signal_repr_collapse, signal_oversmooth, signal_bpr_plateau, repr_health_snapshot
from .growth import GrowthController
from .trainer import Trainer
from .evaluator import Evaluator
from .experiment import Experiment
from .complexity import ComplexityTracker

__all__ = [
    "Config",
    "DataModule",
    "LightGCN",
    "ModalFusion",
    "EWC",
    "GrowthController",
    "ComplexityTracker",
    "Trainer",
    "Evaluator",
    "Experiment",
    "signal_repr_collapse",
    "signal_oversmooth",
    "signal_bpr_plateau",
    "repr_health_snapshot",
]
