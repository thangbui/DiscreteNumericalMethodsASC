"""
SparseMixer gradient estimator experiments — top-level runner.

Experiment groups
-----------------
bias_variance   Exp 1-4   K=1/K=2 bias & variance, convergence, e2e classification
ablation        Exp 5-11  Scale N, temperature, Rao-Gumbel repeats, η, K, optim-var, grid
specialization  Exp 12    Expert specialization / switching linear regression
sparse_routing  Exp 13    Very sparse routing: K=8 selected from N=256 experts
cv_diagnosis    Exp 14    Why does ReinMax-CV underperform? Distribution-mismatch analysis

Usage
-----
# Run every group (may take 30-60 min total)
python -m sparse_mixer.run_all

# Fast smoke test of everything (~5 min)
python -m sparse_mixer.run_all --fast

# Run a single group
python -m sparse_mixer.run_all --group sparse_routing
python -m sparse_mixer.run_all --group bias_variance --fast
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner(title: str):
    bar = '=' * 70
    print(f"\n{bar}\n{title}\n{bar}")


def _elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{s/60:.1f} min" if s >= 60 else f"{s:.1f}s"


# ─────────────────────────────────────────────────────────────────────────────
# Group runners (lazy imports to keep startup fast)
# ─────────────────────────────────────────────────────────────────────────────

def run_bias_variance(fast: bool = False):
    _banner("GROUP 1 — Bias & Variance  (Exp 1–4)")
    t0 = time.time()
    from sparse_mixer.exp_bias_variance import main as _main
    _main(n_samples=100 if fast else 500, fast=fast)
    print(f"\n[bias_variance] done in {_elapsed(t0)}")


def run_ablation(fast: bool = False):
    _banner("GROUP 2 — Ablation Studies  (Exp 5–11)")
    t0 = time.time()
    from sparse_mixer.exp_ablation import main as _main
    _main(n_samples=80 if fast else 300, fast=fast)
    print(f"\n[ablation] done in {_elapsed(t0)}")


def run_specialization(fast: bool = False):
    _banner("GROUP 3 — Expert Specialization  (Exp 12)")
    t0 = time.time()
    from sparse_mixer.exp_specialization import main as _main
    _main(fast=fast)
    print(f"\n[specialization] done in {_elapsed(t0)}")


def run_sparse_routing(fast: bool = False):
    _banner("GROUP 4 — Very Sparse Routing: K=8, N=256  (Exp 13)")
    t0 = time.time()
    from sparse_mixer.exp_sparse_routing import main as _main
    _main(fast=fast)
    print(f"\n[sparse_routing] done in {_elapsed(t0)}")


def run_cv_diagnosis(fast: bool = False):
    _banner("GROUP 5 — ReinMax-CV Diagnosis  (Exp 14)")
    t0 = time.time()
    from sparse_mixer.exp_cv_diagnosis import main as _main
    _main(fast=fast)
    print(f"\n[cv_diagnosis] done in {_elapsed(t0)}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

GROUP_MAP = {
    'bias_variance'  : run_bias_variance,
    'ablation'       : run_ablation,
    'specialization' : run_specialization,
    'sparse_routing' : run_sparse_routing,
    'cv_diagnosis'   : run_cv_diagnosis,
}

ALL_GROUPS = list(GROUP_MAP.keys())


def main():
    p = argparse.ArgumentParser(
        description='SparseMixer gradient estimator experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--group', choices=ALL_GROUPS + ['all'], default='all',
                   help='Experiment group to run  (default: all)')
    p.add_argument('--fast', action='store_true',
                   help='Use reduced settings for a quick smoke test')
    args = p.parse_args()

    groups = ALL_GROUPS if args.group == 'all' else [args.group]

    t_total = time.time()
    print("SparseMixer Gradient Estimator Experiments")
    print(f"  Groups   : {groups}")
    print(f"  Fast mode: {args.fast}")
    print(f"  Results  : {RESULTS_DIR}/")

    for g in groups:
        GROUP_MAP[g](fast=args.fast)

    print(f"\n{'='*70}")
    print(f"ALL DONE  ({_elapsed(t_total)})")
    print(f"Results → {RESULTS_DIR}/")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
