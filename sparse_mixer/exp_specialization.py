"""
Experiment 12 — Expert Specialization via Switching Linear Regression
======================================================================

Motivation
----------
The bias/variance experiments on QuadraticObjective show that ReinMax-v3
reduces gradient variance by ~40-56%.  But does this actually matter for a
*real* learning problem?

This experiment uses a much harder task where gradient variance directly
determines whether the model succeeds:

    Task: N experts, C classes, each class c has a **fixed target linear map**
          W_c ∈ ℝ^{d_out × d_in}.  The model receives (x, class_label) and
          must route x to the "correct" expert and minimise ||expert(x) − W_c x||².

Why this is hard
----------------
1. The router must learn *which* expert handles which class — this is a
   combinatorial assignment problem discovered from gradient signal alone.
2. The experts must simultaneously learn the corresponding W_c.
3. The routing gradient is the *only* signal connecting input class to expert
   selection — it flows through the discrete routing decision.
4. High gradient variance ⟹ noisy routing updates ⟹ **dead expert collapse**:
   one or two experts dominate, others are never updated, and loss plateaus.
5. Low variance ⟹ smoother routing gradient ⟹ experts **specialise** earlier
   and the model reaches lower reconstruction loss.

Metrics
-------
• Training MSE loss (lower = better fit)
• Routing Specialization Score: fraction of tokens routed to their class's
  "plurality expert" (0 = random routing, 1 = perfect specialization)
• Both are measured across multiple random seeds for statistical reliability.

Experiments
-----------
Exp 12a — Single-run training curves (loss + specialization) per method
Exp 12b — Multi-seed (n_seeds=5) final loss and specialization bars
Exp 12c — Sensitivity to N (experts > classes): harder as N grows
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sparse_mixer.estimators import (
    st_single, reinmax_single, reinmax_v3_single, reinmax_cv_single,
)
from sparse_mixer.objectives import SwitchingLinearTask
from sparse_mixer.metrics    import train_specialization
from sparse_mixer.plotting   import (
    COLOURS, save_fig, specialization_curves, specialization_multi_seed,
    bar_panel, line_panel,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
SPEC_DIR    = os.path.join(RESULTS_DIR, 'specialization')
os.makedirs(SPEC_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Method registry
# ─────────────────────────────────────────────────────────────────────────────

def _make_methods(tau: float = 1.0, repeats: int = 30) -> Dict[str, callable]:
    return {
        'st'         : lambda l: st_single(l, tau),
        'reinmax'    : lambda l: reinmax_single(l, tau),
        'reinmax_v3' : lambda l: reinmax_v3_single(l, tau, repeats),
        'reinmax_cv' : lambda l: reinmax_cv_single(l, tau, 0.5, repeats),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Exp 12a — Single-seed training curves
# ─────────────────────────────────────────────────────────────────────────────

def run_single_seed(
    n_experts: int = 4,
    n_classes: int = 4,
    d_in: int = 8,
    d_out: int = 8,
    n_steps: int = 2000,
    batch_size: int = 256,
    lr: float = 3e-3,
    tau: float = 1.0,
    repeats: int = 30,
    seed: int = 42,
    log_every: int = 200,
):
    print(f"\n{'='*60}")
    print(f"Exp 12a: Expert Specialization (single seed)")
    print(f"  N={n_experts}  C={n_classes}  d_in={d_in}  d_out={d_out}")
    print(f"  n_steps={n_steps}  lr={lr}  τ={tau}  R={repeats}")
    print('='*60)

    task    = SwitchingLinearTask(n_experts, n_classes, d_in, d_out, seed=seed)
    methods = _make_methods(tau, repeats)

    all_results = {}
    for name, fn in methods.items():
        t0 = time.time()
        print(f"  [{name}] training ...", end='', flush=True)
        r = train_specialization(fn, task, n_steps=n_steps, batch_size=batch_size,
                                  lr=lr, seed=seed, log_every=log_every)
        elapsed = time.time() - t0
        print(f"  loss={r['final_loss']:.4f}  spec={r['final_spec']:.3f}  ({elapsed:.1f}s)")
        all_results[name] = r

    return all_results


def plot_single_seed(all_results, save_dir=SPEC_DIR):
    specialization_curves(
        all_results,
        path=os.path.join(save_dir, 'spec_single_seed.png'),
        suptitle='Expert Specialization Training (single seed, K=1)',
    )


# ─────────────────────────────────────────────────────────────────────────────
# Exp 12b — Multi-seed statistical comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_multi_seed(
    n_experts: int = 4,
    n_classes: int = 4,
    d_in: int = 8,
    d_out: int = 8,
    n_steps: int = 2000,
    batch_size: int = 256,
    lr: float = 3e-3,
    tau: float = 1.0,
    repeats: int = 30,
    seeds: List[int] = (0, 1, 2, 3, 4),
    log_every: int = 500,
):
    """Run each method for multiple seeds; return mean ± std of final metrics."""
    print(f"\n{'='*60}")
    print(f"Exp 12b: Expert Specialization (multi-seed, n_seeds={len(seeds)})")
    print('='*60)

    methods = _make_methods(tau, repeats)
    agg = {name: dict(final_loss=[], final_spec=[]) for name in methods}

    for seed in seeds:
        task = SwitchingLinearTask(n_experts, n_classes, d_in, d_out, seed=seed)
        print(f"\n  -- seed={seed} --")
        for name, fn in methods.items():
            r = train_specialization(fn, task, n_steps=n_steps,
                                      batch_size=batch_size, lr=lr,
                                      seed=seed, log_every=log_every)
            agg[name]['final_loss'].append(r['final_loss'])
            agg[name]['final_spec'].append(r['final_spec'])
            print(f"    [{name:12s}] loss={r['final_loss']:.4f}  spec={r['final_spec']:.3f}")

    # Print summary
    print(f"\n{'  Method':16s}  {'Mean Loss':>10}  {'Std Loss':>10}  "
          f"{'Mean Spec':>10}  {'Std Spec':>10}")
    for name in methods:
        ml = np.mean(agg[name]['final_loss'])
        sl = np.std( agg[name]['final_loss'])
        ms = np.mean(agg[name]['final_spec'])
        ss = np.std( agg[name]['final_spec'])
        print(f"  {name:16s}  {ml:10.4f}  {sl:10.4f}  {ms:10.4f}  {ss:10.4f}")

    return agg


def plot_multi_seed(agg, save_dir=SPEC_DIR):
    specialization_multi_seed(
        agg,
        path=os.path.join(save_dir, 'spec_multi_seed.png'),
        suptitle='Expert Specialization — Multi-Seed (mean ± std)',
    )


# ─────────────────────────────────────────────────────────────────────────────
# Exp 12c — Sensitivity to N (harder as N >> C)
# ─────────────────────────────────────────────────────────────────────────────

def run_n_sensitivity(
    n_classes: int = 4,
    n_experts_list: List[int] = (4, 6, 8, 12),
    d_in: int = 8,
    d_out: int = 8,
    n_steps: int = 1500,
    batch_size: int = 256,
    lr: float = 3e-3,
    tau: float = 1.0,
    repeats: int = 30,
    seed: int = 42,
    log_every: int = 500,
):
    """
    Fix C=n_classes, vary N.  As N grows beyond C, the routing problem has
    more "decoy" experts and becomes harder — this is where low-variance
    estimators help most.
    """
    print(f"\n{'='*60}")
    print(f"Exp 12c: N sensitivity  (C={n_classes}, N={list(n_experts_list)})")
    print('='*60)

    methods = _make_methods(tau, repeats)
    # results[method][metric][N_idx]
    results = {name: dict(final_loss=[], final_spec=[]) for name in methods}

    for N in n_experts_list:
        print(f"\n  N={N}:")
        task = SwitchingLinearTask(N, n_classes, d_in, d_out, seed=seed)
        for name, fn in methods.items():
            r = train_specialization(fn, task, n_steps=n_steps,
                                      batch_size=batch_size, lr=lr,
                                      seed=seed, log_every=log_every)
            results[name]['final_loss'].append(r['final_loss'])
            results[name]['final_spec'].append(r['final_spec'])
            print(f"    [{name:12s}] loss={r['final_loss']:.4f}  spec={r['final_spec']:.3f}")

    return results, list(n_experts_list)


def plot_n_sensitivity(results, n_experts_list, save_dir=SPEC_DIR):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    line_panel(axes[0], n_experts_list,
               {n: results[n]['final_loss'] for n in results},
               'N (experts)', 'Final MSE Loss', 'Final Loss vs. N (C=4)',
               yscale='log')
    line_panel(axes[1], n_experts_list,
               {n: results[n]['final_spec'] for n in results},
               'N (experts)', 'Specialization Score',
               'Specialization Score vs. N (C=4)')
    plt.suptitle('Effect of N on Expert Specialization  (C=4 classes fixed)', fontsize=12)
    save_fig(os.path.join(save_dir, 'spec_vs_N.png'))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(fast: bool = False):
    n_steps   = 500   if fast else 2000
    n_seeds   = 2     if fast else 5
    n_steps_c = 300   if fast else 1500
    log_every = 100   if fast else 200
    repeats   = 10    if fast else 30

    summary = {}

    # Exp 12a — single seed
    r12a = run_single_seed(
        n_experts=4, n_classes=4, d_in=8, d_out=8,
        n_steps=n_steps, batch_size=256, lr=3e-3,
        tau=1.0, repeats=repeats, seed=42, log_every=log_every,
    )
    plot_single_seed(r12a)

    summary['single_seed'] = {
        n: dict(final_loss=r['final_loss'], final_spec=r['final_spec'])
        for n, r in r12a.items()
    }

    # Exp 12b — multi-seed
    r12b = run_multi_seed(
        n_experts=4, n_classes=4, d_in=8, d_out=8,
        n_steps=n_steps, batch_size=256, lr=3e-3,
        tau=1.0, repeats=repeats,
        seeds=list(range(n_seeds)), log_every=log_every * 3,
    )
    plot_multi_seed(r12b)

    summary['multi_seed'] = {
        name: dict(
            mean_loss=float(np.mean(vals['final_loss'])),
            std_loss =float(np.std( vals['final_loss'])),
            mean_spec=float(np.mean(vals['final_spec'])),
            std_spec =float(np.std( vals['final_spec'])),
        )
        for name, vals in r12b.items()
    }

    # Exp 12c — N sensitivity
    r12c, ne_list = run_n_sensitivity(
        n_classes=4, n_experts_list=[4, 6, 8, 12],
        d_in=8, d_out=8, n_steps=n_steps_c,
        batch_size=256, lr=3e-3, tau=1.0, repeats=repeats,
        seed=42, log_every=log_every * 2,
    )
    plot_n_sensitivity(r12c, ne_list)

    summary['n_sensitivity'] = {
        name: dict(final_spec=vals['final_spec'],
                   final_loss=vals['final_loss'])
        for name, vals in r12c.items()
    }

    path = os.path.join(SPEC_DIR, 'specialization_summary.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSpecialization summary → {path}")

    # Print final table
    print(f"\n{'='*60}")
    print("SPECIALIZATION EXPERIMENT SUMMARY")
    print('='*60)
    print(f"\nSingle-seed results (N=4, C=4, K=1):")
    for name, r in r12a.items():
        print(f"  {name:14s}  loss={r['final_loss']:.4f}  "
              f"spec={r['final_spec']:.3f}")

    if r12b:
        print(f"\nMulti-seed results ({n_seeds} seeds):")
        for name, vals in summary['multi_seed'].items():
            print(f"  {name:14s}  "
                  f"loss={vals['mean_loss']:.4f}±{vals['std_loss']:.4f}  "
                  f"spec={vals['mean_spec']:.3f}±{vals['std_spec']:.3f}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--fast', action='store_true',
                   help='Run with smaller settings for quick smoke test')
    args = p.parse_args()
    main(fast=args.fast)
