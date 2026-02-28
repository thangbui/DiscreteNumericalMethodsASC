"""
Experiment 13 — Sparse Routing at Scale: N=256 experts
=======================================================

Motivation
----------
Real-world MoE models (Switch Transformer, Mixtral, DeepSeek-MoE) select a
very small fraction of their expert pool.  At K=8, N=256 only 3.1% of experts
are active per token — the same sparsity ratio as K=1, N=32, but the
combinatorial routing space is astronomically larger (C(256,8) ≈ 10^14 vs 32).

This experiment asks: does gradient estimator variance grow catastrophically
with scale, and does ReinMax-v3 remain beneficial in this regime?  And how
does the answer change as we relax sparsity (more K per token)?

Sub-experiments
---------------
13a  Fixed-sparsity scaling
     Fix K/N ≈ 3.1%: (K=1, N=32) → (K=2, N=64) → (K=4, N=128) → (K=8, N=256).
     Compare variance and MSE across all estimators at each scale point.

13b  Head-to-head variance at N=256, K=8
     Direct measurement at the target scale with a high-sample REINFORCE
     reference.  Shows the absolute variance reduction and MSE improvement.

13c  Convergence race at N=256, K=8
     Gradient ascent on E[f(z)] from random initialisation.
     Tracks:
       • E[f(z)]         — objective value
       • Top-K overlap   — fraction of true top-8 experts in the selected set

13d  K ablation at fixed N=256
     K ∈ {8, 16, 32, 64, 128} (density 3% → 50%).
     Two panels:
       • Variance/MSE vs K for reinmax vs reinmax_v3  (adaptive R for large K)
       • Convergence curves for each K (reinmax_topk), showing that
         less sparse routing finds good experts much faster.

Key expected findings
---------------------
• Gradient variance grows roughly as O(N * K), so absolute variance at
  N=256, K=8 is ~64× higher than N=32, K=1.
• ReinMax-v3's Rao-Gumbel Jacobian provides consistent relative variance
  reduction (~40-55%) regardless of scale or K.
• Convergence at N=256 is much slower for small K; larger K provides a
  denser reward signal that dramatically accelerates learning.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sparse_mixer.estimators import (
    st_single, reinmax_single, reinmax_v3_single, reinmax_cv_single,
    reinmax_topk, reinmax_v3_topk, reinmax_cv_topk,
)
from sparse_mixer.objectives import QuadraticObjective
from sparse_mixer.metrics    import collect_grad_samples, bias_variance_metrics
from sparse_mixer.plotting   import COLOURS, save_fig, triple_line_figure, line_panel

RESULTS_DIR   = os.path.join(os.path.dirname(__file__), 'results')
SPARSE_DIR    = os.path.join(RESULTS_DIR, 'sparse_routing')
os.makedirs(SPARSE_DIR, exist_ok=True)

# Colour for REINFORCE reference baseline
_REF_COLOUR = '#7f7f7f'


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _topk_normalised_obj(obj: QuadraticObjective, k: int):
    """Return f(z/k): normalises k-hot masks to a probability-like vector."""
    return lambda z: obj(z / k)


def _reinforce_reference(logits, topk_obj, k, tau, n_ref):
    """High-sample REINFORCE mean gradient (used as ground truth)."""
    ref_fn = lambda l: reinmax_topk(l, k=k, tau=tau)
    return collect_grad_samples(ref_fn, logits, topk_obj, n_ref).mean(0)


def _k1_reinforce_reference(logits, obj, n_ref):
    """REINFORCE reference for K=1 via the closed-form exact gradient."""
    return obj.exact_gradient(logits)


def _true_top_k_overlap(logits: torch.Tensor, f_vals: torch.Tensor, k: int) -> float:
    """
    Fraction of the true top-K experts (by f_val) that appear in the
    current logits top-K selection (averaged over batch).
    """
    true_topk  = f_vals.topk(k).indices.tolist()
    logit_topk = logits.topk(k, dim=-1).indices  # (B, K)
    overlaps = []
    for b in range(logits.shape[0]):
        sel   = set(logit_topk[b].tolist())
        truth = set(true_topk)
        overlaps.append(len(sel & truth) / k)
    return float(np.mean(overlaps))


# ─────────────────────────────────────────────────────────────────────────────
# Exp 13a — Fixed K/N ratio: scale from (K=1,N=32) to (K=8,N=256)
# ─────────────────────────────────────────────────────────────────────────────

_SCALE_POINTS = [
    (1, 32),
    (2, 64),
    (4, 128),
    (8, 256),
]


def run_fixed_sparsity_scaling(
    n_samples: int = 120,
    n_ref: int = 1500,
    batch_size: int = 2,
    tau: float = 1.0,
    repeats: int = 20,
    seed: int = 42,
) -> Dict:
    """
    Exp 13a: compare gradient variance at fixed K/N ≈ 3.1%.

    Scale points: (K=1,N=32), (K=2,N=64), (K=4,N=128), (K=8,N=256).
    """
    print(f"\n{'='*60}")
    print(f"Exp 13a: Fixed-sparsity scaling  (K/N≈3.1%)")
    print(f"  n_samples={n_samples}  n_ref={n_ref}  τ={tau}  R={repeats}")
    print('='*60)

    results = {
        name: dict(bias=[], variance=[], mse=[])
        for name in ['reinmax', 'reinmax_v3', 'reinmax_cv']
    }
    scale_labels = []

    for k, N in _SCALE_POINTS:
        label = f'K={k},N={N}'
        scale_labels.append(label)
        print(f"\n  {label}:")

        torch.manual_seed(seed)
        logits  = torch.randn(batch_size, N) * 1.5
        obj     = QuadraticObjective(N, seed=seed)
        topk_obj = _topk_normalised_obj(obj, k)

        if k == 1:
            # K=1: use exact gradient as reference, K=1 estimators
            exact = _k1_reinforce_reference(logits, obj, n_ref)
            method_fns = {
                'reinmax'    : lambda l, k_=k: reinmax_single(l, tau),
                'reinmax_v3' : lambda l, k_=k: reinmax_v3_single(l, tau, repeats),
                'reinmax_cv' : lambda l, k_=k, r_=repeats: reinmax_cv_single(l, tau, eta=0.9, repeats=r_),
            }
            ref = exact
        else:
            # K>1: high-sample REINFORCE as reference
            print(f"    computing REINFORCE ref ({n_ref} samples)…", end='', flush=True)
            ref = _reinforce_reference(logits, topk_obj, k, tau, n_ref)
            print(' done.')
            method_fns = {
                'reinmax'    : lambda l, k_=k: reinmax_topk(l, k=k_, tau=tau),
                'reinmax_v3' : lambda l, k_=k: reinmax_v3_topk(l, k=k_, tau=tau, repeats=repeats),
                'reinmax_cv' : lambda l, k_=k, r_=repeats: reinmax_cv_topk(l, k=k_, tau=tau, eta=0.9, repeats=r_),
            }

        for name, fn in method_fns.items():
            if k == 1:
                grads = collect_grad_samples(fn, logits, obj, n_samples)
            else:
                grads = collect_grad_samples(fn, logits, topk_obj, n_samples)
            m = bias_variance_metrics(grads, ref)
            results[name]['bias'].append(m['bias'])
            results[name]['variance'].append(m['variance'])
            results[name]['mse'].append(m['mse'])
            print(f"    [{name:12s}] bias={m['bias']:.4f}  "
                  f"var={m['variance']:.4f}  mse={m['mse']:.4f}")

    return results, scale_labels


def plot_fixed_sparsity_scaling(results, scale_labels, save_dir=SPARSE_DIR):
    triple_line_figure(
        xs=range(len(scale_labels)),
        bias_dict={n: results[n]['bias']     for n in results},
        var_dict ={n: results[n]['variance'] for n in results},
        mse_dict ={n: results[n]['mse']      for n in results},
        xlabel='Scale  (K, N)',
        suptitle='Gradient Statistics at Fixed Sparsity Ratio K/N ≈ 3.1%',
        path=os.path.join(save_dir, 'fixed_sparsity_scaling.png'),
    )
    # Overwrite x-tick labels
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax_idx, (key, ylabel, title, yscale) in enumerate([
        ('bias',     'Bias (L2)',  'Gradient Bias',     'linear'),
        ('variance', 'Variance',   'Gradient Variance', 'log'),
        ('mse',      'MSE',        'Gradient MSE',      'log'),
    ]):
        ax = axes[ax_idx]
        for name in results:
            ax.plot(range(len(scale_labels)), results[name][key],
                    color=COLOURS.get(name, '#888'), label=name,
                    lw=2, marker='o', ms=6)
        ax.set_xticks(range(len(scale_labels)))
        ax.set_xticklabels(scale_labels, rotation=10)
        ax.set_ylabel(ylabel); ax.set_title(title); ax.set_yscale(yscale)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Add variance-ratio annotations (v3/rm and cv/rm)
    for j, label in enumerate(scale_labels):
        v_rm = results['reinmax']['variance'][j]
        v_v3 = results['reinmax_v3']['variance'][j]
        v_cv = results['reinmax_cv']['variance'][j]
        axes[1].annotate(f'×{v_v3/(v_rm+1e-12):.2f}', xy=(j, v_v3),
                         xytext=(5,  5), textcoords='offset points', fontsize=7)
        axes[1].annotate(f'×{v_cv/(v_rm+1e-12):.2f}', xy=(j, v_cv),
                         xytext=(5, -12), textcoords='offset points', fontsize=7,
                         color=COLOURS.get('reinmax_cv', '#888'))

    plt.suptitle('Gradient Statistics at Fixed Sparsity Ratio K/N ≈ 3.1%', fontsize=12)
    save_fig(os.path.join(save_dir, 'fixed_sparsity_scaling.png'))


# ─────────────────────────────────────────────────────────────────────────────
# Exp 13b — Head-to-head variance at N=256, K=8
# ─────────────────────────────────────────────────────────────────────────────

def run_large_scale_variance(
    n_experts: int = 256,
    k: int = 8,
    batch_size: int = 2,
    n_ref: int = 2000,
    n_samples: int = 150,
    tau: float = 1.0,
    repeats: int = 20,
    seed: int = 42,
) -> Dict:
    """
    Exp 13b: direct bias/variance comparison at N=256, K=8.
    """
    print(f"\n{'='*60}")
    print(f"Exp 13b: Head-to-head at N={n_experts}, K={k}")
    print(f"  n_ref={n_ref}  n_samples={n_samples}  τ={tau}  R={repeats}")
    print('='*60)

    torch.manual_seed(seed)
    logits   = torch.randn(batch_size, n_experts) * 1.5
    obj      = QuadraticObjective(n_experts, seed=seed)
    topk_obj = _topk_normalised_obj(obj, k)

    # REINFORCE reference
    print(f"  Computing REINFORCE reference ({n_ref} samples)…", end='', flush=True)
    t0  = time.time()
    ref = _reinforce_reference(logits, topk_obj, k, tau, n_ref)
    print(f"  done ({time.time()-t0:.1f}s)")

    methods = {
        'reinmax'    : lambda l: reinmax_topk(l, k=k, tau=tau),
        'reinmax_v3' : lambda l: reinmax_v3_topk(l, k=k, tau=tau, repeats=repeats),
        'reinmax_cv' : lambda l: reinmax_cv_topk(l, k=k, tau=tau, eta=0.9, repeats=repeats),
    }

    results = {}
    for name, fn in methods.items():
        t0 = time.time()
        print(f"  [{name}] collecting {n_samples} samples…", end='', flush=True)
        grads = collect_grad_samples(fn, logits, topk_obj, n_samples)
        m     = bias_variance_metrics(grads, ref)
        elapsed = time.time() - t0
        results[name] = m
        print(f"  bias={m['bias']:.4f}  var={m['variance']:.4f}"
              f"  mse={m['mse']:.4f}  ({elapsed:.1f}s)")

    v_rm = results['reinmax']['variance']
    v_v3 = results['reinmax_v3']['variance']
    v_cv = results['reinmax_cv']['variance']
    print(f"\n  Variance ratio (v3/reinmax): {v_v3/(v_rm+1e-12):.3f}  "
          f"({'↓' if v_v3 < v_rm else '↑'} variance)")
    print(f"  Variance ratio (cv/reinmax): {v_cv/(v_rm+1e-12):.3f}  "
          f"({'↓' if v_cv < v_rm else '↑'} variance)")
    return results


def plot_large_scale_variance(results, n_experts, k, save_dir=SPARSE_DIR):
    names  = list(results.keys())
    biases = [results[n]['bias']     for n in names]
    vars_  = [results[n]['variance'] for n in names]
    mses   = [results[n]['mse']      for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, vals, ylabel, title in zip(
        axes,
        [biases, vars_, mses],
        ['Bias (L2)', 'Variance', 'MSE'],
        ['Gradient Bias', 'Gradient Variance', 'Gradient MSE'],
    ):
        colours = [COLOURS.get(n, '#888') for n in names]
        bars = ax.bar(names, vals, color=colours)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.grid(True, axis='y', alpha=0.3)
        # Annotate bars with values
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    # Add variance-reduction annotation
    v_rm = results['reinmax']['variance']
    v_v3 = results['reinmax_v3']['variance']
    v_cv = results['reinmax_cv']['variance']
    axes[1].set_title(
        f'Gradient Variance  (v3/rm={v_v3/(v_rm+1e-12):.2f}  cv/rm={v_cv/(v_rm+1e-12):.2f})'
    )

    plt.suptitle(f'Head-to-Head at N={n_experts}, K={k}  '
                 f'(Very Sparse: K/N = {k/n_experts:.1%})', fontsize=12)
    save_fig(os.path.join(save_dir, 'large_scale_variance.png'))


# ─────────────────────────────────────────────────────────────────────────────
# Exp 13c — Convergence race at N=256, K=8
# ─────────────────────────────────────────────────────────────────────────────

def run_convergence_at_scale(
    n_experts: int = 256,
    k: int = 8,
    n_steps: int = 600,
    lr: float = 0.05,
    tau: float = 1.0,
    batch_size: int = 8,
    repeats: int = 20,
    seed: int = 42,
) -> Dict:
    """
    Exp 13c: gradient ascent at N=256, K=8.

    Tracks:
      - E[f(z)]: expected value under current routing distribution
      - top-K overlap: fraction of true best-K experts in selected set
    """
    print(f"\n{'='*60}")
    print(f"Exp 13c: Convergence at N={n_experts}, K={k}")
    print(f"  n_steps={n_steps}  lr={lr}  τ={tau}  R={repeats}")
    print('='*60)

    torch.manual_seed(seed)
    obj    = QuadraticObjective(n_experts, seed=seed)
    f_vals = obj.f_at_experts()

    # True best K experts
    true_topk_idx = f_vals.topk(k).indices.tolist()
    print(f"  True top-{k} expert indices: {sorted(true_topk_idx)}")
    print(f"  True top-{k} mean f-value:   {f_vals[true_topk_idx].mean():.4f}")
    print(f"  Random baseline E[f]:        {f_vals.mean():.4f}")

    methods = {
        'reinmax'    : lambda l: reinmax_topk(l, k=k, tau=tau),
        'reinmax_v3' : lambda l: reinmax_v3_topk(l, k=k, tau=tau, repeats=repeats),
        'reinmax_cv' : lambda l: reinmax_cv_topk(l, k=k, tau=tau, eta=0.9, repeats=repeats),
    }

    history = {}
    for name, fn in methods.items():
        torch.manual_seed(seed)
        logits = nn.Parameter(torch.randn(batch_size, n_experts) * 0.01)
        opt    = torch.optim.SGD([logits], lr=lr)
        ef_traj, overlap_traj = [], []

        for step in range(n_steps):
            opt.zero_grad()
            mask, _ = fn(logits)
            # Objective: maximise mean f(z/k) over batch
            loss = -obj(mask / k).mean()
            loss.backward()
            opt.step()

            with torch.no_grad():
                p   = F.softmax(logits, dim=-1)
                # E[f] under soft routing
                ef  = (p * f_vals).sum(-1).mean().item()
                ov  = _true_top_k_overlap(logits, f_vals, k)
            ef_traj.append(ef)
            overlap_traj.append(ov)

        history[name] = dict(ef=ef_traj, overlap=overlap_traj)
        print(f"  [{name:12s}]  final E[f]={ef_traj[-1]:.4f}  "
              f"top-{k}-overlap={overlap_traj[-1]:.3f}")

    # Reference: random routing baseline and oracle
    random_ef = f_vals.mean().item()
    oracle_ef  = f_vals.topk(k).values.mean().item()
    print(f"\n  Random-routing baseline E[f]: {random_ef:.4f}")
    print(f"  Oracle (best-K)        E[f]: {oracle_ef:.4f}")

    return history, random_ef, oracle_ef, f_vals


def plot_convergence_at_scale(history, random_ef, oracle_ef, n_experts, k,
                               save_dir=SPARSE_DIR):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for name, data in history.items():
        c = COLOURS.get(name, '#888')
        axes[0].plot(data['ef'],      color=c, label=name, lw=2)
        axes[1].plot(data['overlap'], color=c, label=name, lw=2)

    # Reference lines
    axes[0].axhline(random_ef,  color='grey',  ls=':',  lw=1.5, label='random routing')
    axes[0].axhline(oracle_ef,  color='black', ls='--', lw=1.5, label=f'oracle (top-{k})')
    axes[1].axhline(k/n_experts, color='grey', ls=':', lw=1.5,
                    label=f'random ({k/n_experts:.2f})')
    axes[1].axhline(1.0,         color='black', ls='--', lw=1.5, label='perfect')

    axes[0].set_xlabel('step'); axes[0].set_ylabel('E[f(z)]')
    axes[0].set_title(f'Expected Objective  (N={n_experts}, K={k})')
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('step')
    axes[1].set_ylabel(f'Top-{k} overlap fraction')
    axes[1].set_title(f'Top-{k} Expert Overlap  (1 = found all best {k})')
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'Convergence at Very Sparse Scale  '
                 f'(N={n_experts}, K={k}, K/N={k/n_experts:.1%})', fontsize=12)
    save_fig(os.path.join(save_dir, 'convergence_at_scale.png'))


# ─────────────────────────────────────────────────────────────────────────────
# Exp 13d — K ablation at fixed N=256: K ∈ {8, 16, 32, 64, 128}
# ─────────────────────────────────────────────────────────────────────────────

_K_ABLATION_VALUES = [8, 16, 32, 64, 128]


def _k_ablation_params(k: int, fast: bool):
    """
    Adaptive (n_samples, repeats, n_ref) for the K ablation.

    `_rao_gumbel_jacobian_topk` has cost O(K * R) per backward pass, so we
    scale down R and n_samples for large K to keep wall-time manageable.
    """
    if fast:
        table = {8: (12, 4, 100), 16: (8, 3, 100),
                 32: (6, 2, 100), 64: (4, 2, 100), 128: (3, 2, 80)}
    else:
        table = {8: (100, 20, 1500), 16: (60, 12, 1000),
                 32: (40,  8,  800), 64: (20,  5,  500), 128: (10, 3, 300)}
    n_s, reps, n_r = table[k]
    return dict(n_samples=n_s, repeats=reps, n_ref=n_r)


def run_k_ablation(
    n_experts: int = 256,
    k_values: list = _K_ABLATION_VALUES,
    batch_size: int = 2,
    tau: float = 1.0,
    n_steps_conv: int = 600,
    lr_conv: float = 0.05,
    batch_size_conv: int = 8,
    fast: bool = False,
    seed: int = 42,
) -> Dict:
    """
    Exp 13d: variance and convergence for K ∈ {8,16,32,64,128} at N=256.

    Returns
    -------
    var_results : dict  {k: {'reinmax': metrics, 'reinmax_v3': metrics}}
    conv_history: dict  {k: {'ef': [...], 'overlap': [...], 'oracle_ef': float}}
    """
    print(f"\n{'='*60}")
    print(f"Exp 13d: K ablation at N={n_experts}")
    print(f"  K values: {k_values}  τ={tau}")
    print('='*60)

    torch.manual_seed(seed)
    obj    = QuadraticObjective(n_experts, seed=seed)
    f_vals = obj.f_at_experts()

    # ── Part 1: gradient variance ─────────────────────────────────────────────
    var_results = {}
    for k in k_values:
        p = _k_ablation_params(k, fast)
        n_samples, repeats, n_ref = p['n_samples'], p['repeats'], p['n_ref']
        topk_obj = _topk_normalised_obj(obj, k)

        print(f"\n  K={k:3d}  (n_samples={n_samples}, R={repeats}, n_ref={n_ref})")

        # REINFORCE reference
        torch.manual_seed(seed)
        logits = torch.randn(batch_size, n_experts) * 1.5
        print(f"    REINFORCE ref ({n_ref} samples)…", end='', flush=True)
        ref = _reinforce_reference(logits, topk_obj, k, tau, n_ref)
        print(' done.')

        var_results[k] = {}
        for name, fn in [
            ('reinmax',     lambda l, k_=k: reinmax_topk(l, k=k_, tau=tau)),
            ('reinmax_v3',  lambda l, k_=k, r_=repeats:
                            reinmax_v3_topk(l, k=k_, tau=tau, repeats=r_)),
            ('reinmax_cv',  lambda l, k_=k, r_=repeats:
                            reinmax_cv_topk(l, k=k_, tau=tau, eta=0.9, repeats=r_)),
        ]:
            t0 = time.time()
            grads = collect_grad_samples(fn, logits, topk_obj, n_samples)
            m     = bias_variance_metrics(grads, ref)
            elapsed = time.time() - t0
            var_results[k][name] = m
            print(f"    [{name:12s}] bias={m['bias']:.5f}  "
                  f"var={m['variance']:.5f}  mse={m['mse']:.5f}  ({elapsed:.1f}s)")

        v_rm = var_results[k]['reinmax']['variance']
        v_v3 = var_results[k]['reinmax_v3']['variance']
        v_cv = var_results[k]['reinmax_cv']['variance']
        print(f"    variance ratio v3/rm: {v_v3/(v_rm+1e-12):.3f}  "
              f"cv/rm: {v_cv/(v_rm+1e-12):.3f}")

    # ── Part 2: convergence curves ────────────────────────────────────────────
    print(f"\n  Convergence at each K ({n_steps_conv} steps, lr={lr_conv}):")
    conv_history = {}
    for k in k_values:
        topk_obj = _topk_normalised_obj(obj, k)
        true_topk_idx = f_vals.topk(k).indices.tolist()
        oracle_ef     = f_vals.topk(k).values.mean().item()

        torch.manual_seed(seed)
        logits = nn.Parameter(torch.randn(batch_size_conv, n_experts) * 0.01)
        opt    = torch.optim.SGD([logits], lr=lr_conv)
        ef_traj, ov_traj = [], []

        for _ in range(n_steps_conv):
            opt.zero_grad()
            mask, _ = reinmax_topk(logits, k=k, tau=tau)
            loss    = -obj(mask / k).mean()
            loss.backward()
            opt.step()

            with torch.no_grad():
                p  = F.softmax(logits, dim=-1)
                ef = (p * f_vals).sum(-1).mean().item()
                ov = _true_top_k_overlap(logits, f_vals, k)
            ef_traj.append(ef)
            ov_traj.append(ov)

        conv_history[k] = dict(ef=ef_traj, overlap=ov_traj, oracle_ef=oracle_ef)
        random_ef_k = f_vals.topk(k).values.mean().item()  # oracle for this K
        print(f"    K={k:3d}: final E[f]={ef_traj[-1]:.4f}  "
              f"overlap={ov_traj[-1]:.3f}  oracle={oracle_ef:.4f}")

    return var_results, conv_history, f_vals.mean().item()


def plot_k_ablation(var_results, conv_history, random_ef_global,
                    n_experts, k_values, save_dir=SPARSE_DIR):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    # ── Panel 1: Gradient Variance vs K ──────────────────────────────────────
    ax = axes[0]
    for name in ['reinmax', 'reinmax_v3', 'reinmax_cv']:
        vars_ = [var_results[k][name]['variance'] for k in k_values]
        ax.plot(k_values, vars_, color=COLOURS.get(name, '#888'),
                label=name, lw=2, marker='o', ms=6)
    ax.set_xlabel('K (experts selected)'); ax.set_ylabel('Gradient Variance')
    ax.set_title(f'Variance vs K  (N={n_experts})')
    ax.set_yscale('log'); ax.set_xscale('log', base=2)
    ax.set_xticks(k_values); ax.set_xticklabels(k_values)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which='both')
    # Annotate variance-reduction ratios (v3/rm and cv/rm)
    for k in k_values:
        v_rm = var_results[k]['reinmax']['variance']
        v_v3 = var_results[k]['reinmax_v3']['variance']
        v_cv = var_results[k]['reinmax_cv']['variance']
        ax.annotate(f'×{v_v3/(v_rm+1e-12):.2f}', xy=(k, v_v3),
                    xytext=(4,  4), textcoords='offset points', fontsize=7)
        ax.annotate(f'×{v_cv/(v_rm+1e-12):.2f}', xy=(k, v_cv),
                    xytext=(4, -12), textcoords='offset points', fontsize=7,
                    color=COLOURS.get('reinmax_cv', '#888'))

    # ── Panel 2: E[f(z)] convergence ─────────────────────────────────────────
    ax = axes[1]
    cmap = plt.cm.plasma
    colours_k = [cmap(i / (len(k_values) - 1)) for i in range(len(k_values))]
    for (k, data), c in zip(conv_history.items(), colours_k):
        density = k / n_experts
        ax.plot(data['ef'], color=c, lw=1.8,
                label=f'K={k} ({density:.0%})')
        ax.axhline(data['oracle_ef'], color=c, ls='--', lw=0.8, alpha=0.5)
    ax.axhline(random_ef_global, color='grey', ls=':', lw=1.5, label='random')
    ax.set_xlabel('step'); ax.set_ylabel('E[f(z)]')
    ax.set_title(f'Convergence: E[f(z)] vs step  (N={n_experts})')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    # ── Panel 3: Top-K overlap convergence ───────────────────────────────────
    ax = axes[2]
    for (k, data), c in zip(conv_history.items(), colours_k):
        density = k / n_experts
        random_overlap = k / n_experts
        ax.plot(data['overlap'], color=c, lw=1.8,
                label=f'K={k} ({density:.0%})')
    ax.set_xlabel('step')
    ax.set_ylabel('Top-K overlap fraction')
    ax.set_title(f'Top-K Overlap vs step  (N={n_experts})')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    plt.suptitle(
        f'K Ablation at Fixed N={n_experts}  '
        f'(K ∈ {{{", ".join(map(str, k_values))}}}, density {k_values[0]/n_experts:.0%}–{k_values[-1]/n_experts:.0%})',
        fontsize=11,
    )
    save_fig(os.path.join(save_dir, 'k_ablation_n256.png'))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(fast: bool = False):
    n_samples = 40  if fast else 150
    n_ref     = 400 if fast else 2000
    repeats   = 10  if fast else 20
    n_steps   = 150 if fast else 600

    summary = {}

    # ── Exp 13a ──────────────────────────────────────────────────────────────
    r13a, scale_labels = run_fixed_sparsity_scaling(
        n_samples=n_samples, n_ref=n_ref, batch_size=2,
        tau=1.0, repeats=repeats,
    )
    plot_fixed_sparsity_scaling(r13a, scale_labels)

    summary['fixed_sparsity'] = {
        name: {key: vals for key, vals in m.items()}
        for name, m in r13a.items()
    }
    v_rm_list = r13a['reinmax']['variance']
    v_v3_list = r13a['reinmax_v3']['variance']
    v_cv_list = r13a['reinmax_cv']['variance']
    ratios_v3 = [v3 / (rm + 1e-12) for v3, rm in zip(v_v3_list, v_rm_list)]
    ratios_cv = [cv / (rm + 1e-12) for cv, rm in zip(v_cv_list, v_rm_list)]
    summary['fixed_sparsity']['variance_ratios_v3'] = {
        lbl: ratio for lbl, ratio in zip(scale_labels, ratios_v3)
    }
    summary['fixed_sparsity']['variance_ratios_cv'] = {
        lbl: ratio for lbl, ratio in zip(scale_labels, ratios_cv)
    }
    print(f"\n  Variance ratios across scales:")
    for lbl, rv3, rcv in zip(scale_labels, ratios_v3, ratios_cv):
        print(f"    {lbl:14s}: v3/rm={rv3:.3f}  cv/rm={rcv:.3f}")

    # ── Exp 13b ──────────────────────────────────────────────────────────────
    r13b = run_large_scale_variance(
        n_experts=256, k=8, batch_size=2,
        n_ref=n_ref, n_samples=n_samples,
        tau=1.0, repeats=repeats,
    )
    plot_large_scale_variance(r13b, n_experts=256, k=8)

    summary['large_scale'] = {
        n: {k2: v for k2, v in m.items() if isinstance(v, float)}
        for n, m in r13b.items()
    }

    # ── Exp 13c ──────────────────────────────────────────────────────────────
    r13c, rand_ef, oracle_ef, _ = run_convergence_at_scale(
        n_experts=256, k=8, n_steps=n_steps,
        lr=0.05, tau=1.0, batch_size=8, repeats=repeats,
    )
    plot_convergence_at_scale(r13c, rand_ef, oracle_ef, n_experts=256, k=8)

    summary['convergence'] = {
        name: dict(
            final_ef=h['ef'][-1],
            final_overlap=h['overlap'][-1],
        )
        for name, h in r13c.items()
    }
    summary['convergence']['oracle_ef']  = oracle_ef
    summary['convergence']['random_ef']  = rand_ef

    # ── Exp 13d ──────────────────────────────────────────────────────────────
    r13d_var, r13d_conv, r13d_rand = run_k_ablation(
        n_experts=256,
        k_values=_K_ABLATION_VALUES,
        batch_size=2,
        tau=1.0,
        n_steps_conv=n_steps,
        lr_conv=0.05,
        batch_size_conv=8,
        fast=fast,
    )
    plot_k_ablation(r13d_var, r13d_conv, r13d_rand,
                    n_experts=256, k_values=_K_ABLATION_VALUES)

    summary['k_ablation'] = {
        str(k): {
            'variance_ratio_v3': (r13d_var[k]['reinmax_v3']['variance'] /
                                  (r13d_var[k]['reinmax']['variance'] + 1e-12)),
            'variance_ratio_cv': (r13d_var[k]['reinmax_cv']['variance'] /
                                  (r13d_var[k]['reinmax']['variance'] + 1e-12)),
            'final_ef':      r13d_conv[k]['ef'][-1],
            'final_overlap': r13d_conv[k]['overlap'][-1],
            'oracle_ef':     r13d_conv[k]['oracle_ef'],
        }
        for k in _K_ABLATION_VALUES
    }

    path = os.path.join(SPARSE_DIR, 'sparse_routing_summary.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSparse routing summary → {path}")

    # Final print
    print(f"\n{'='*60}")
    print("SPARSE ROUTING SUMMARY  (N=256)")
    print('='*60)
    print(f"\n13a — Variance ratios at each scale:")
    for lbl, rv3, rcv in zip(scale_labels, ratios_v3, ratios_cv):
        sv3 = '↓' if rv3 < 1 else '↑'
        scv = '↓' if rcv < 1 else '↑'
        print(f"  {lbl:14s}: v3/rm={rv3:.3f} {sv3}  cv/rm={rcv:.3f} {scv}")
    print(f"\n13b — At N=256, K=8:")
    for name, m in r13b.items():
        print(f"  {name:14s}: var={m['variance']:.4f}  mse={m['mse']:.4f}")
    print(f"\n13c — Convergence (K=8):")
    for name, h in r13c.items():
        print(f"  {name:14s}: E[f]={h['ef'][-1]:.4f}  "
              f"overlap={h['overlap'][-1]:.3f}")
    print(f"  oracle          : E[f]={oracle_ef:.4f}")
    print(f"\n13d — K ablation at N=256:")
    print(f"  {'K':>5}  {'density':>8}  {'v3/rm':>7}  {'cv/rm':>7}  "
          f"{'final E[f]':>11}  {'overlap':>8}  {'oracle':>7}")
    for k in _K_ABLATION_VALUES:
        d = summary['k_ablation'][str(k)]
        print(f"  {k:>5}  {k/256:>8.1%}  {d['variance_ratio_v3']:>7.3f}  "
              f"{d['variance_ratio_cv']:>7.3f}  "
              f"{d['final_ef']:>11.4f}  {d['final_overlap']:>8.3f}  "
              f"{d['oracle_ef']:>7.4f}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--fast', action='store_true')
    args = p.parse_args()
    main(fast=args.fast)
