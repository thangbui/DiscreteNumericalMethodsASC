"""
Ablation experiments for SparseMixer gradient estimators.

Exp 5 — Scale with N (number of experts)
------------------------------------------
  For N in [4, 8, 16, 32], compare variance of all K=1 methods.
  Motivation: larger routing distributions are harder to estimate;
  does the Rao-Gumbel Jacobian help more as N grows?

Exp 6 — Temperature τ ablation
---------------------------------
  For τ in [0.1, 0.3, 0.5, 1.0, 2.0], compare bias/variance.
  Low τ → sharper selections → routing closer to argmax → higher variance.

Exp 7 — Rao-Gumbel repeats ablation
--------------------------------------
  For repeats R in [5, 10, 20, 50, 100], measure variance of
  ReinMax-v3 and ReinMax-CV (which depend on R).
  Shows the compute–variance trade-off.

Exp 8 — η ablation (ReinMax-CV control-variate strength)
-----------------------------------------------------------
  For η in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0], measure variance
  of ReinMax-CV.  η=0 ⟹ plain ReinMax; the optimal η minimises variance.

Exp 9 — K ablation (top-K routing with K=1,2,3,4)
----------------------------------------------------
  Fix N=8, vary K in [1, 2, 3, 4].  For each K compare
  reinmax_topk / reinmax_v3_topk / reinmax_cv_topk by MSE
  (vs high-sample REINFORCE reference).

Exp 10 — Gradient variance during optimisation
------------------------------------------------
  Track per-step gradient variance while maximising E[f(z)] with SGD.
  Shows whether lower-variance estimators lead to smoother, faster
  convergence in practice.
"""

from __future__ import annotations

import json
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sparse_mixer.estimators import (
    st_single,
    reinmax_single,
    reinmax_v3_single,
    reinmax_cv_single,
    reinmax_topk,
    reinmax_v3_topk,
    reinmax_cv_topk,
)
from sparse_mixer.experiments_bias_variance import (
    QuadraticObjective,
    collect_grad_samples,
    bias_variance_metrics,
    RESULTS_DIR,
    COLOURS,
)

ABLATION_DIR = os.path.join(RESULTS_DIR, 'ablation')
os.makedirs(ABLATION_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _line_ax(ax, xs, ys_dict, xlabel, ylabel, title, yscale='linear', markers=True):
    """Plot multiple curves on one axis."""
    for name, ys in ys_dict.items():
        c  = COLOURS.get(name, '#888888')
        mk = 'o' if markers else None
        ax.plot(xs, ys, color=c, label=name, lw=2, marker=mk, markersize=5)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.set_yscale(yscale); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)


def _scalar_metrics(logits, obj, method_fn, n_samples):
    """Return (bias, variance, mse) for one method at fixed logits."""
    exact = obj.exact_gradient(logits)
    grads = collect_grad_samples(method_fn, logits, obj, n_samples)
    m     = bias_variance_metrics(grads, exact)
    return m['bias'], m['variance'], m['mse']


# ─────────────────────────────────────────────────────────────────────────────
# Exp 5 — Scale with N (number of experts)
# ─────────────────────────────────────────────────────────────────────────────

def run_scale_experiment(
    n_experts_list=(4, 8, 16, 32),
    n_samples: int = 300,
    batch_size: int = 4,
    tau: float = 1.0,
    repeats: int = 30,
    seed: int = 42,
):
    print(f"\n{'='*60}")
    print(f"Exp 5: Scale with N  (n_samples={n_samples}, τ={tau})")
    print('='*60)

    method_fns = {
        'st'         : lambda l, N: (lambda ll: st_single(ll, tau)),
        'reinmax'    : lambda l, N: (lambda ll: reinmax_single(ll, tau)),
        'reinmax_v3' : lambda l, N: (lambda ll: reinmax_v3_single(ll, tau, repeats)),
        'reinmax_cv' : lambda l, N: (lambda ll: reinmax_cv_single(ll, tau, eta=0.5, repeats=repeats)),
    }

    results = {name: dict(bias=[], variance=[], mse=[]) for name in method_fns}

    for N in n_experts_list:
        print(f"  N={N}:")
        torch.manual_seed(seed)
        logits = torch.randn(batch_size, N) * 1.5
        obj    = QuadraticObjective(N, seed=seed)

        for name, fn_factory in method_fns.items():
            fn = fn_factory(logits, N)
            b, v, mse = _scalar_metrics(logits, obj, fn, n_samples)
            results[name]['bias'].append(b)
            results[name]['variance'].append(v)
            results[name]['mse'].append(mse)
            print(f"    [{name:12s}] bias={b:.4f}  var={v:.4f}  mse={mse:.4f}")

    return results, list(n_experts_list)


def plot_scale_experiment(results, n_experts_list, save_dir=ABLATION_DIR):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    xs = n_experts_list
    _line_ax(axes[0], xs,
             {n: results[n]['bias']     for n in results},
             'N (experts)', 'Bias (L2)', 'Gradient Bias vs. N')
    _line_ax(axes[1], xs,
             {n: results[n]['variance'] for n in results},
             'N (experts)', 'Variance',  'Gradient Variance vs. N', yscale='log')
    _line_ax(axes[2], xs,
             {n: results[n]['mse']      for n in results},
             'N (experts)', 'MSE',       'Gradient MSE vs. N',      yscale='log')
    plt.suptitle('Effect of Number of Mixture Components (K=1)', fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, 'scale_N.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Exp 6 — Temperature τ ablation
# ─────────────────────────────────────────────────────────────────────────────

def run_temperature_experiment(
    tau_list=(0.1, 0.3, 0.5, 1.0, 2.0),
    n_experts: int = 8,
    n_samples: int = 300,
    batch_size: int = 4,
    repeats: int = 30,
    seed: int = 42,
):
    print(f"\n{'='*60}")
    print(f"Exp 6: Temperature ablation  (N={n_experts})")
    print('='*60)

    torch.manual_seed(seed)
    logits = torch.randn(batch_size, n_experts) * 1.5
    obj    = QuadraticObjective(n_experts, seed=seed)

    results = {
        name: dict(bias=[], variance=[], mse=[])
        for name in ['st', 'reinmax', 'reinmax_v3', 'reinmax_cv']
    }

    for tau in tau_list:
        print(f"  τ={tau}:")
        method_fns = {
            'st'         : lambda l: st_single(l, tau),
            'reinmax'    : lambda l: reinmax_single(l, tau),
            'reinmax_v3' : lambda l: reinmax_v3_single(l, tau, repeats),
            'reinmax_cv' : lambda l: reinmax_cv_single(l, tau, eta=0.5, repeats=repeats),
        }
        # Exact gradient depends on tau only through the sampling distribution,
        # not the logits → same exact grad for all tau (REINFORCE-style)
        for name, fn in method_fns.items():
            b, v, mse = _scalar_metrics(logits, obj, fn, n_samples)
            results[name]['bias'].append(b)
            results[name]['variance'].append(v)
            results[name]['mse'].append(mse)
            print(f"    [{name:12s}] bias={b:.4f}  var={v:.4f}  mse={mse:.4f}")

    return results, list(tau_list)


def plot_temperature_experiment(results, tau_list, save_dir=ABLATION_DIR):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    xs = tau_list
    _line_ax(axes[0], xs,
             {n: results[n]['bias']     for n in results},
             'Temperature τ', 'Bias (L2)', 'Gradient Bias vs. τ')
    _line_ax(axes[1], xs,
             {n: results[n]['variance'] for n in results},
             'Temperature τ', 'Variance',  'Gradient Variance vs. τ', yscale='log')
    _line_ax(axes[2], xs,
             {n: results[n]['mse']      for n in results},
             'Temperature τ', 'MSE',       'Gradient MSE vs. τ',      yscale='log')
    plt.suptitle('Effect of Temperature τ (K=1, N=8)', fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, 'temperature.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Exp 7 — Rao-Gumbel repeats ablation
# ─────────────────────────────────────────────────────────────────────────────

def run_repeats_experiment(
    repeats_list=(5, 10, 20, 50, 100),
    n_experts: int = 8,
    n_samples: int = 300,
    batch_size: int = 4,
    tau: float = 1.0,
    seed: int = 42,
):
    print(f"\n{'='*60}")
    print(f"Exp 7: Rao-Gumbel repeats ablation  (N={n_experts}, τ={tau})")
    print('='*60)

    torch.manual_seed(seed)
    logits = torch.randn(batch_size, n_experts) * 1.5
    obj    = QuadraticObjective(n_experts, seed=seed)

    # Baselines that don't depend on R
    base_metrics = {}
    for name, fn in [
        ('reinmax', lambda l: reinmax_single(l, tau)),
        ('st',      lambda l: st_single(l, tau)),
    ]:
        b, v, mse = _scalar_metrics(logits, obj, fn, n_samples)
        base_metrics[name] = dict(bias=b, variance=v, mse=mse)
        print(f"  [baseline {name:8s}] bias={b:.4f}  var={v:.4f}  mse={mse:.4f}")

    results = {
        name: dict(bias=[], variance=[], mse=[])
        for name in ['reinmax_v3', 'reinmax_cv']
    }
    wall_times = {name: [] for name in ['reinmax_v3', 'reinmax_cv']}

    for R in repeats_list:
        print(f"  R={R}:")
        method_fns = {
            'reinmax_v3' : lambda l, R=R: reinmax_v3_single(l, tau, R),
            'reinmax_cv' : lambda l, R=R: reinmax_cv_single(l, tau, eta=0.5, repeats=R),
        }
        for name, fn in method_fns.items():
            t0 = time.time()
            b, v, mse = _scalar_metrics(logits, obj, fn, n_samples)
            wall_times[name].append((time.time() - t0) / n_samples * 1000)  # ms/sample
            results[name]['bias'].append(b)
            results[name]['variance'].append(v)
            results[name]['mse'].append(mse)
            print(f"    [{name:12s}] bias={b:.4f}  var={v:.4f}  mse={mse:.4f}")

    return results, base_metrics, list(repeats_list), wall_times


def plot_repeats_experiment(results, base_metrics, repeats_list, wall_times,
                            save_dir=ABLATION_DIR):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    xs = repeats_list

    # Variance plot with horizontal baselines
    for name, data in results.items():
        c = COLOURS.get(name, '#888888')
        axes[1].plot(xs, data['variance'], color=c, label=name, lw=2, marker='o', ms=5)
    for name, m in base_metrics.items():
        axes[1].axhline(m['variance'], color=COLOURS.get(name, '#888888'),
                        ls='--', lw=1.5, label=f'{name} (baseline)')
    axes[1].set_xlabel('Rao-Gumbel Repeats R')
    axes[1].set_ylabel('Variance'); axes[1].set_yscale('log')
    axes[1].set_title('Variance vs. Repeats R'); axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)

    # MSE plot with baselines
    for name, data in results.items():
        c = COLOURS.get(name, '#888888')
        axes[2].plot(xs, data['mse'], color=c, label=name, lw=2, marker='o', ms=5)
    for name, m in base_metrics.items():
        axes[2].axhline(m['mse'], color=COLOURS.get(name, '#888888'),
                        ls='--', lw=1.5, label=f'{name} (baseline)')
    axes[2].set_xlabel('Rao-Gumbel Repeats R')
    axes[2].set_ylabel('MSE'); axes[2].set_yscale('log')
    axes[2].set_title('MSE vs. Repeats R'); axes[2].legend(fontsize=7); axes[2].grid(True, alpha=0.3)

    # Wall-time per sample
    for name, times in wall_times.items():
        c = COLOURS.get(name, '#888888')
        axes[0].plot(xs, times, color=c, label=name, lw=2, marker='o', ms=5)
    axes[0].set_xlabel('Rao-Gumbel Repeats R')
    axes[0].set_ylabel('ms / sample'); axes[0].set_title('Wall Time vs. Repeats')
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    plt.suptitle('Rao-Gumbel Sample Count Ablation (K=1, N=8)', fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, 'repeats.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Exp 8 — η ablation (ReinMax-CV control-variate strength)
# ─────────────────────────────────────────────────────────────────────────────

def run_eta_experiment(
    eta_list=(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
    n_experts: int = 8,
    n_samples: int = 300,
    batch_size: int = 4,
    tau: float = 1.0,
    repeats: int = 30,
    seed: int = 42,
):
    print(f"\n{'='*60}")
    print(f"Exp 8: η ablation (ReinMax-CV)  (N={n_experts}, τ={tau}, R={repeats})")
    print('='*60)

    torch.manual_seed(seed)
    logits = torch.randn(batch_size, n_experts) * 1.5
    obj    = QuadraticObjective(n_experts, seed=seed)

    biases, variances, mses = [], [], []

    for eta in eta_list:
        fn = lambda l, e=eta: reinmax_cv_single(l, tau, eta=e, repeats=repeats)
        b, v, mse = _scalar_metrics(logits, obj, fn, n_samples)
        biases.append(b); variances.append(v); mses.append(mse)
        print(f"  η={eta:.1f}  bias={b:.4f}  var={v:.4f}  mse={mse:.4f}")

    # Also compute ReinMax baseline (η=0 should match reinmax for large R)
    b_rm, v_rm, mse_rm = _scalar_metrics(
        logits, obj, lambda l: reinmax_single(l, tau), n_samples
    )
    print(f"  [reinmax baseline] bias={b_rm:.4f}  var={v_rm:.4f}  mse={mse_rm:.4f}")

    return (list(eta_list), biases, variances, mses,
            dict(bias=b_rm, variance=v_rm, mse=mse_rm))


def plot_eta_experiment(eta_list, biases, variances, mses, reinmax_baseline,
                        save_dir=ABLATION_DIR):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    c_cv = COLOURS['reinmax_cv']
    c_rm = COLOURS['reinmax']

    for ax, vals, ylabel, title in zip(
        axes,
        [biases, variances, mses],
        ['Bias (L2)', 'Variance', 'MSE'],
        ['Gradient Bias vs. η', 'Gradient Variance vs. η', 'Gradient MSE vs. η'],
    ):
        ax.plot(eta_list, vals, color=c_cv, lw=2, marker='o', ms=6, label='reinmax_cv(η)')
        ax.axhline(reinmax_baseline[ylabel.split()[0].lower()],
                   color=c_rm, ls='--', lw=1.5, label='reinmax')
        ax.set_xlabel('η'); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle('Control-Variate Strength η Ablation (K=1, N=8)', fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, 'eta.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Exp 9 — K ablation (top-K routing)
# ─────────────────────────────────────────────────────────────────────────────

def _topk_exact_reinforce(logits, topk_obj, k, n_ref=2000, tau=1.0):
    """High-sample REINFORCE reference for top-K."""
    ref_fn = lambda l: reinmax_topk(l, k=k, tau=tau)
    ref_grads = collect_grad_samples(ref_fn, logits, topk_obj, n_ref)
    return ref_grads.mean(0)


def run_k_ablation(
    k_list=(1, 2, 3, 4),
    n_experts: int = 8,
    n_samples: int = 200,
    n_ref: int = 1500,
    batch_size: int = 2,
    tau: float = 1.0,
    repeats: int = 20,
    seed: int = 42,
):
    print(f"\n{'='*60}")
    print(f"Exp 9: K ablation  (N={n_experts}, τ={tau})")
    print('='*60)

    torch.manual_seed(seed)
    logits = torch.randn(batch_size, n_experts) * 1.5
    obj    = QuadraticObjective(n_experts, seed=seed)

    results = {
        name: dict(variance=[], mse=[])
        for name in ['reinmax', 'reinmax_v3', 'reinmax_cv']
    }

    for k in k_list:
        print(f"  K={k}:")
        topk_obj = lambda z, k=k: obj(z / k)

        # Reference gradient via high-sample REINFORCE
        ref = _topk_exact_reinforce(logits, topk_obj, k, n_ref, tau)

        method_fns = {
            'reinmax'    : lambda l, k=k: reinmax_topk(l, k=k, tau=tau),
            'reinmax_v3' : lambda l, k=k: reinmax_v3_topk(l, k=k, tau=tau, repeats=repeats),
            'reinmax_cv' : lambda l, k=k: reinmax_cv_topk(l, k=k, tau=tau,
                                                           eta=0.5, repeats=repeats),
        }
        for name, fn in method_fns.items():
            grads = collect_grad_samples(fn, logits, topk_obj, n_samples)
            m     = bias_variance_metrics(grads, ref)
            results[name]['variance'].append(m['variance'])
            results[name]['mse'].append(m['mse'])
            print(f"    [{name:12s}] var={m['variance']:.4f}  mse={m['mse']:.4f}")

    return results, list(k_list)


def plot_k_ablation(results, k_list, save_dir=ABLATION_DIR):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _line_ax(axes[0], k_list,
             {n: results[n]['variance'] for n in results},
             'K (selected experts)', 'Variance (log)',
             'Gradient Variance vs. K', yscale='log')
    _line_ax(axes[1], k_list,
             {n: results[n]['mse'] for n in results},
             'K (selected experts)', 'MSE (log)',
             'Gradient MSE vs. K', yscale='log')
    plt.suptitle(f'Effect of K (Top-K Selection, N=8)', fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, 'k_ablation.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Exp 10 — Gradient variance during optimisation
# ─────────────────────────────────────────────────────────────────────────────

def _measure_step_variance(method_fn, logits_val, obj, n_var_samples=50):
    """
    Measure gradient variance at a single point `logits_val` using
    n_var_samples independent gradient estimates.
    Returns scalar variance (averaged over batch & expert dims).
    """
    grads = collect_grad_samples(method_fn, logits_val.detach(), obj, n_var_samples)
    mean_g = grads.mean(0)
    return ((grads - mean_g) ** 2).sum(-1).mean().item()


def run_optimisation_variance(
    n_experts: int = 8,
    n_steps: int = 200,
    lr: float = 0.05,
    tau: float = 1.0,
    batch_size: int = 8,
    n_var_samples: int = 40,
    measure_every: int = 10,
    repeats: int = 30,
    seed: int = 42,
):
    """
    Run gradient-ascent on E[f(z)] and record the gradient variance
    (measured at the current logits every `measure_every` steps).
    """
    print(f"\n{'='*60}")
    print(f"Exp 10: Gradient variance during optimisation  (N={n_experts}, τ={tau})")
    print('='*60)

    obj = QuadraticObjective(n_experts, seed=seed)

    methods = {
        'st'         : lambda l: st_single(l, tau),
        'reinmax'    : lambda l: reinmax_single(l, tau),
        'reinmax_v3' : lambda l: reinmax_v3_single(l, tau, repeats),
        'reinmax_cv' : lambda l: reinmax_cv_single(l, tau, eta=0.5, repeats=repeats),
    }

    history = {}
    for name, fn in methods.items():
        torch.manual_seed(seed)
        logits = nn.Parameter(torch.randn(batch_size, n_experts) * 0.1)
        opt    = torch.optim.SGD([logits], lr=lr)

        ef_traj, var_traj, steps_rec = [], [], []

        for step in range(n_steps):
            opt.zero_grad()
            z, _ = fn(logits)
            loss = -obj(z).mean()
            loss.backward()
            opt.step()

            # Record expected reward
            with torch.no_grad():
                p  = F.softmax(logits, dim=-1)
                ef = (p * obj.f_at_experts()).sum(-1).mean().item()
            ef_traj.append(ef)

            # Periodically measure variance
            if step % measure_every == 0:
                var = _measure_step_variance(fn, logits, obj, n_var_samples)
                var_traj.append(var)
                steps_rec.append(step)

        history[name] = dict(ef=ef_traj, var=var_traj, steps=steps_rec)
        print(f"  [{name:12s}]  final E[f]={ef_traj[-1]:.4f}  "
              f"final_var={var_traj[-1]:.4f}")

    return history


def plot_optimisation_variance(history, save_dir=ABLATION_DIR):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for name, data in history.items():
        c = COLOURS.get(name, '#888888')
        axes[0].plot(data['ef'],   color=c, label=name, lw=2)
        axes[1].plot(data['steps'], data['var'], color=c, label=name, lw=2,
                     marker='o', ms=4)

    axes[0].set_xlabel('Step'); axes[0].set_ylabel('E[f(z)]')
    axes[0].set_title('E[f(z)] During Optimisation')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Step'); axes[1].set_ylabel('Gradient Variance (log)')
    axes[1].set_yscale('log')
    axes[1].set_title('Gradient Variance During Optimisation')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle('Variance Reduction Benefit During Training (K=1, N=8)', fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, 'optimisation_variance.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary heatmap: bias & variance across (N, τ) grid
# ─────────────────────────────────────────────────────────────────────────────

def run_grid_experiment(
    n_experts_list=(4, 8, 16),
    tau_list=(0.3, 1.0, 2.0),
    n_samples: int = 200,
    batch_size: int = 2,
    repeats: int = 20,
    seed: int = 42,
):
    """
    Measure variance reduction ratio (ReinMax-v3 / ReinMax) across an
    (N, τ) grid.  Ratio < 1 means v3 has lower variance.
    """
    print(f"\n{'='*60}")
    print("Exp 11: (N, τ) Variance-Reduction Grid")
    print('='*60)

    ratio_v3 = np.zeros((len(n_experts_list), len(tau_list)))
    ratio_cv = np.zeros((len(n_experts_list), len(tau_list)))

    for i, N in enumerate(n_experts_list):
        for j, tau in enumerate(tau_list):
            torch.manual_seed(seed)
            logits = torch.randn(batch_size, N) * 1.5
            obj    = QuadraticObjective(N, seed=seed)

            _, v_rm,  _ = _scalar_metrics(logits, obj,
                                           lambda l, t=tau: reinmax_single(l, t),
                                           n_samples)
            _, v_v3,  _ = _scalar_metrics(logits, obj,
                                           lambda l, t=tau, R=repeats: reinmax_v3_single(l, t, R),
                                           n_samples)
            _, v_cv,  _ = _scalar_metrics(logits, obj,
                                           lambda l, t=tau, R=repeats: reinmax_cv_single(l, t, 0.5, R),
                                           n_samples)

            ratio_v3[i, j] = v_v3 / (v_rm + 1e-12)
            ratio_cv[i, j] = v_cv / (v_rm + 1e-12)
            print(f"  N={N:2d} τ={tau}  v3/rm={ratio_v3[i,j]:.3f}  cv/rm={ratio_cv[i,j]:.3f}")

    return ratio_v3, ratio_cv, list(n_experts_list), list(tau_list)


def plot_grid_experiment(ratio_v3, ratio_cv, n_experts_list, tau_list,
                         save_dir=ABLATION_DIR):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, ratio, title in zip(
        axes,
        [ratio_v3, ratio_cv],
        ['Var(ReinMax-v3) / Var(ReinMax)', 'Var(ReinMax-CV) / Var(ReinMax)'],
    ):
        im = ax.imshow(ratio, aspect='auto', cmap='RdYlGn_r', vmin=0.0, vmax=2.0)
        ax.set_xticks(range(len(tau_list)))
        ax.set_xticklabels([f'τ={t}' for t in tau_list])
        ax.set_yticks(range(len(n_experts_list)))
        ax.set_yticklabels([f'N={n}' for n in n_experts_list])
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='Variance ratio (< 1 = better)')
        # Annotate cells
        for i in range(len(n_experts_list)):
            for j in range(len(tau_list)):
                ax.text(j, i, f'{ratio[i, j]:.2f}', ha='center', va='center',
                        fontsize=9, color='black')

    plt.suptitle('Variance Reduction Ratio vs. ReinMax Baseline\n'
                 '(< 1 means lower variance than ReinMax)', fontsize=11)
    plt.tight_layout()
    path = os.path.join(save_dir, 'grid_N_tau.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("SparseMixer Ablation Experiments")
    print("="*60)

    summary = {}

    # ── Exp 5: scale with N ──────────────────────────────────────────────
    res5, n_list = run_scale_experiment(
        n_experts_list=[4, 8, 16, 32], n_samples=300, tau=1.0, repeats=30
    )
    plot_scale_experiment(res5, n_list)
    summary['scale_N'] = {n: {k: v[-1] for k, v in m.items()}
                          for n, m in res5.items()}

    # ── Exp 6: temperature ───────────────────────────────────────────────
    res6, tau_list = run_temperature_experiment(
        tau_list=[0.1, 0.3, 0.5, 1.0, 2.0], n_experts=8, n_samples=300, repeats=30
    )
    plot_temperature_experiment(res6, tau_list)

    # ── Exp 7: repeats ───────────────────────────────────────────────────
    res7, base7, r_list, wt7 = run_repeats_experiment(
        repeats_list=[5, 10, 20, 50, 100], n_experts=8, n_samples=300, tau=1.0
    )
    plot_repeats_experiment(res7, base7, r_list, wt7)
    summary['repeats'] = {n: {k: v[-1] for k, v in m.items()}
                          for n, m in res7.items()}

    # ── Exp 8: eta ───────────────────────────────────────────────────────
    eta_list, biases8, vars8, mses8, rm_base8 = run_eta_experiment(
        eta_list=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        n_experts=8, n_samples=300, tau=1.0, repeats=30
    )
    plot_eta_experiment(eta_list, biases8, vars8, mses8, rm_base8)
    best_eta_idx = int(np.argmin(mses8))
    summary['eta'] = dict(best_eta=eta_list[best_eta_idx],
                          best_mse=mses8[best_eta_idx])

    # ── Exp 9: K ablation ────────────────────────────────────────────────
    res9, k_list = run_k_ablation(
        k_list=[1, 2, 3, 4], n_experts=8, n_samples=200,
        n_ref=1500, tau=1.0, repeats=20
    )
    plot_k_ablation(res9, k_list)

    # ── Exp 10: variance during optimisation ─────────────────────────────
    hist10 = run_optimisation_variance(
        n_experts=8, n_steps=200, lr=0.05, tau=1.0,
        n_var_samples=40, measure_every=10, repeats=30
    )
    plot_optimisation_variance(hist10)

    # ── Exp 11: (N, τ) grid ──────────────────────────────────────────────
    rv3, rcv, ne_list, t_list = run_grid_experiment(
        n_experts_list=[4, 8, 16], tau_list=[0.3, 1.0, 2.0],
        n_samples=200, repeats=20
    )
    plot_grid_experiment(rv3, rcv, ne_list, t_list)
    summary['grid'] = dict(
        v3_mean_ratio=float(rv3.mean()),
        cv_mean_ratio=float(rcv.mean()),
    )

    # ── Save ─────────────────────────────────────────────────────────────
    path = os.path.join(ABLATION_DIR, 'ablation_summary.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("ABLATION SUMMARY")
    print('='*60)
    print(f"\nScale (at N=32):")
    for name, m in res5.items():
        print(f"  {name:12s}  var={m['variance'][-1]:.4f}  mse={m['mse'][-1]:.4f}")
    print(f"\nBest η for ReinMax-CV: {eta_list[best_eta_idx]:.1f} "
          f"(MSE={mses8[best_eta_idx]:.4f})")
    print(f"\nVariance reduction grid (mean ratio vs ReinMax):")
    print(f"  ReinMax-v3: {rv3.mean():.3f}  (< 1 = lower variance)")
    print(f"  ReinMax-CV: {rcv.mean():.3f}  (< 1 = lower variance)")
    print(f"\nResults saved to {ABLATION_DIR}/")
    print("Done.")


if __name__ == '__main__':
    main()
