"""
Top-level runner for all SparseMixer gradient estimator experiments.

Usage
-----
# Run everything (may take 30-60 min)
python -m sparse_mixer.run_all

# Fast smoke test (~2 min)
python -m sparse_mixer.run_all --fast

# Run individual experiment groups
python -m sparse_mixer.run_all --group bias_variance
python -m sparse_mixer.run_all --group ablation
python -m sparse_mixer.run_all --group specialization

# Combine flags
python -m sparse_mixer.run_all --group ablation --fast

Experiment groups
-----------------
bias_variance   : Exp 1-4  (K=1/K=2 bias & variance, convergence, e2e)
ablation        : Exp 5-11 (scale N, temperature, repeats, η, K, optim-var, grid)
specialization  : Exp 12   (expert specialization / switching linear regression)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _section(title: str):
    bar = '=' * 70
    print(f"\n{bar}\n{title}\n{bar}")


def _elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{s/60:.1f} min" if s >= 60 else f"{s:.1f}s"


# ─────────────────────────────────────────────────────────────────────────────
# Group runners
# ─────────────────────────────────────────────────────────────────────────────

def run_bias_variance(fast: bool = False):
    _section("GROUP: Bias & Variance (Exp 1-4)")
    t0 = time.time()
    from sparse_mixer.exp_bias_variance import main as _main
    _main(n_samples=100 if fast else 500, fast=fast)
    print(f"\n[bias_variance] done in {_elapsed(t0)}")


def run_ablation(fast: bool = False):
    _section("GROUP: Ablation Studies (Exp 5-11)")
    t0 = time.time()
    from sparse_mixer.exp_ablation import main as _main
    _main(n_samples=80 if fast else 300, fast=fast)
    print(f"\n[ablation] done in {_elapsed(t0)}")


def run_specialization(fast: bool = False):
    _section("GROUP: Expert Specialization (Exp 12)")
    t0 = time.time()
    from sparse_mixer.exp_specialization import main as _main
    _main(fast=fast)
    print(f"\n[specialization] done in {_elapsed(t0)}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

GROUP_MAP = {
    'bias_variance'  : run_bias_variance,
    'ablation'       : run_ablation,
    'specialization' : run_specialization,
}

ALL_GROUPS = list(GROUP_MAP.keys())


def main():
    p = argparse.ArgumentParser(
        description='SparseMixer gradient estimator experiments runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        '--group', choices=ALL_GROUPS + ['all'], default='all',
        help='Experiment group to run (default: all)',
    )
    p.add_argument(
        '--fast', action='store_true',
        help='Use reduced settings for a quick smoke test (~2 min total)',
    )
    args = p.parse_args()

    groups = ALL_GROUPS if args.group == 'all' else [args.group]

    t_total = time.time()
    print("SparseMixer Experiments")
    print(f"  Groups   : {groups}")
    print(f"  Fast mode: {args.fast}")
    print(f"  Results  : {RESULTS_DIR}")

    for g in groups:
        GROUP_MAP[g](fast=args.fast)

    print(f"\n{'='*70}")
    print(f"ALL DONE in {_elapsed(t_total)}")
    print(f"Results saved to: {RESULTS_DIR}/")
    print('='*70)


if __name__ == '__main__':
    main()
