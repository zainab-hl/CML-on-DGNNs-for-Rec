
import argparse
from pathlib import Path

from hetero_mgat.config import Config
from hetero_mgat.data import DataModule
from hetero_mgat.experiment import Experiment


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=None,
                   help='Dataset root directory (overrides Config default)')
    p.add_argument('--out', type=Path, default=None,
                   help='Output directory for checkpoints/results')
    p.add_argument('--seeds', type=int, default=None, help='Number of seeds')
    p.add_argument('--pretrain-epochs', type=int, default=None)
    p.add_argument('--cl-epochs', type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    overrides = {}
    if args.root is not None:
        overrides['root'] = args.root
    if args.out is not None:
        overrides['out'] = args.out
    if args.seeds is not None:
        overrides['N_SEEDS'] = args.seeds
    if args.pretrain_epochs is not None:
        overrides['PRETRAIN_EPOCHS'] = args.pretrain_epochs
    if args.cl_epochs is not None:
        overrides['CL_EPOCHS'] = args.cl_epochs

    cfg = Config(**overrides)
    print(cfg.summary())

    dm = DataModule(cfg)
    Experiment(cfg, dm).run()


if __name__ == '__main__':
    main()
