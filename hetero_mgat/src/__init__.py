"""
hetero_mgat
===========
Heterogeneous MGAT + EWC + Severity-Based Growth Controller, for
continual recommendation on Amazon Appliances (3-core subset, PCA-128
embeddings).

Quickstart
----------
    from hetero_mgat.config import Config
    from hetero_mgat.data import DataModule
    from hetero_mgat.experiment import Experiment

    cfg = Config()              # override fields as needed
    dm = DataModule(cfg)        # loads parquet, builds features/splits
    Experiment(cfg, dm).run()   # trains all 4 model variants, all seeds
"""
from .config import Config
from .data import DataModule
from .model import HeteroMGAT, ItemFeatures, build_dynamic, build_static
from .ewc import EWC, accumulate_fisher
from .growth import GrowthController
from .complexity import ComplexityTracker
from .evaluator import Evaluator
from .experiment import Experiment

__all__ = [
    'Config', 'DataModule',
    'HeteroMGAT', 'ItemFeatures', 'build_dynamic', 'build_static',
    'EWC', 'accumulate_fisher',
    'GrowthController',
    'ComplexityTracker',
    'Evaluator',
    'Experiment',
]
