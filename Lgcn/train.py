"""

Usage
-----
    python train.py
    python train.py --dataset baby --n-seeds 5
    python train.py --dataset clothing --hidden-init 64 --k-init 2
"""

import argparse
from pathlib import Path

from lgcn_cl import Config, Experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Continual Learning LightGCN")
    parser.add_argument("--dataset",       type=str,   default="sports")
    parser.add_argument("--data-dir",      type=Path,  default=None)
    parser.add_argument("--out-dir",       type=Path,  default=Path("outputs"))
    parser.add_argument("--n-seeds",       type=int,   default=3)
    parser.add_argument("--hidden-init",   type=int,   default=128)
    parser.add_argument("--k-init",        type=int,   default=3)
    parser.add_argument("--hidden-static", type=int,   default=256)
    parser.add_argument("--k-static",      type=int,   default=4)
    parser.add_argument("--pretrain-epochs", type=int, default=30)
    parser.add_argument("--cl-epochs",     type=int,   default=20)
    parser.add_argument("--ewc-lambda",    type=float, default=15.0)
    parser.add_argument("--sev-threshold", type=float, default=0.4)
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config(
        dataset=args.dataset,
        out_dir=args.out_dir,
        n_seeds=args.n_seeds,
        hidden_init=args.hidden_init,
        k_init=args.k_init,
        hidden_static=args.hidden_static,
        k_static=args.k_static,
        pretrain_epochs=args.pretrain_epochs,
        cl_epochs=args.cl_epochs,
        ewc_lambda_base=args.ewc_lambda,
        sev_threshold=args.sev_threshold,
    )
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir

    print(f"Dataset      : {cfg.dataset}")
    print(f"Device       : {cfg.device}")
    print(f"Seeds        : {cfg.n_seeds}")
    print(f"Dynamic init : H={cfg.hidden_init}  K={cfg.k_init}  (max H={cfg.hidden_max}, K={cfg.k_max})")
    print(f"Static fixed : H={cfg.hidden_static}  K={cfg.k_static}")
    print(f"EWC lambda   : {cfg.ewc_lambda_base}")
    print(f"Sev threshold: {cfg.sev_threshold}")

    Experiment(cfg).run()


if __name__ == "__main__":
    main()
