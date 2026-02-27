"""
Experiments 5-11: Ablation studies for SparseMixer gradient estimators.

Exp 5  — Scale with N (number of experts): 4, 8, 16, 32
Exp 6  — Temperature τ ablation: 0.1 → 2.0
Exp 7  — Rao-Gumbel sample-count ablation: R = 5 → 100
Exp 8  — η ablation (ReinMax-CV control-variate strength)
Exp 9  — K ablation (top-K routing with K=1,2,3,4)
Exp 10 — Gradient variance during optimisation
Exp 11 — (N, τ) variance-reduction ratio grid heatmap
"""

from __future__ import annotations

import json
import os
import sys
import time

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
from sparse_mixer.metrics    import collect_grad_samples, bias_variance_metrics, scalar_bvm
from sparse_mixer.plotting   import (
    COLOURS, save_fig, line_panel, triple_line_figure, heatmap,
)

RESULTS_DIR  = os.path.join(os.path.dirname(__file__), 'results')
ABLATION_DIR = os.path.join(RESULTS_DIR, 'ablation')
os.makedirs(ABLATION_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Exp 5 — Scale with N
# ─────────────────────────────────────────────────────────────────────────────

def run_scale_experiment(
    n_experts_list=(4, 8, 16, 32),
    n_samples: int = 300,
    batch_size: int = 4,
    tau: float = 1.0,
    repeats: int = 30,
    seed: int = 42,
):
    print(f"\n{'='*60}\nExp 5: Scale with N  (n_samples={n_samples}, τ={tau})\n{'='*60}")

    method_fns = {
        'st'         : lambda l, t=tau: st_single(l, t),
        'reinmax'    : lambda l, t=tau: reinmax_single(l, t),
        'reinmax_v3' : lambda l, t=tau, R=repeats: reinmax_v3_single(l, t, R),
        'reinmax_cv' : lambda l, t=tau, R=repeats: reinmax_cv_single(l, t, 0.5, R),
    }

    results = {name: dict(bias=[], variance=[], mse=[]) for name in method_fns}

    for N in n_experts_list:
        print(f"  N={N}:")
        torch.manual_seed(seed)
        logits = torch.randn(batch_size, N) * 1.5
        obj    = QuadraticObjective(N, seed=seed)
        for name, fn in method_fns.items():
            b, v, mse = scalar_bvm(logits, obj, fn, n_samples)
            results[name]['bias'].append(b)
            results[name]['variance'].append(v)
            results[name]['mse'].append(mse)
            print(f"    [{name:12s}] bias={b:.4f}  var={v:.4f}  mse={mse:.4f}")

    return results, list(n_experts_list)


def plot_scale_experiment(results, n_experts_list, save_dir=ABLATION_DIR):
    triple_line_figure(
        xs=n_experts_list,
        bias_dict={n: results[n]['bias']     for n in results},
        var_dict ={n: results[n]['variance'] for n in results},
        mse_dict ={n: results[n]['mse']      for n in results},
        xlabel='N (experts)',
        suptitle='Effect of Number of Mixture Components (K=1)',
        path=os.path.join(save_dir, 'scale_N.png'),
    )


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
    print(f"\n{'='*60}\nExp 6: Temperature ablation  (N={n_experts})\n{'='*60}")

    torch.manual_seed(seed)
    logits = torch.randn(batch_size, n_experts) * 1.5
    obj    = QuadraticObjective(n_experts, seed=seed)

    results = {n: dict(bias=[], variance=[], mse=[])
               for n in ['st', 'reinmax', 'reinmax_v3', 'reinmax_cv']}

    for tau in tau_list:
        print(f"  τ={tau}:")
        method_fns = {
            'st'         : lambda l, t=tau: st_single(l, t),
            'reinmax'    : lambda l, t=tau: reinmax_single(l, t),
            'reinmax_v3' : lambda l, t=tau, R=repeats: reinmax_v3_single(l, t, R),
            'reinmax_cv' : lambda l, t=tau, R=repeats: reinmax_cv_single(l, t, 0.5, R),
        }
        for name, fn in method_fns.items():
            b, v, mse = scalar_bvm(logits, obj, fn, n_samples)
            results[name]['bias'].append(b)
            results[name]['variance'].append(v)
            results[name]['mse'].append(mse)
            print(f"    [{name:12s}] bias={b:.4f}  var={v:.4f}  mse={mse:.4f}")

    return results, list(tau_list)


def plot_temperature_experiment(results, tau_list, save_dir=ABLATION_DIR):
    triple_line_figure(
        xs=tau_list,
        bias_dict={n: results[n]['bias']     for n in results},
        var_dict ={n: results[n]['variance'] for n in results},
        mse_dict ={n: results[n]['mse']      for n in results},
        xlabel='Temperature τ',
        suptitle='Effect of Temperature τ (K=1, N=8)',
        path=os.path.join(save_dir, 'temperature.png'),
    )


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
    print(f"\n{'='*60}\nExp 7: Rao-Gumbel repeats  (N={n_experts}, τ={tau})\n{'='*60}")

    torch.manual_seed(seed)
    logits = torch.randn(batch_size, n_experts) * 1.5
    obj    = QuadraticObjective(n_experts, seed=seed)

    # Baselines that don't depend on R
    base = {}
    for name, fn in [('reinmax', lambda l: reinmax_single(l, tau)),
                     ('st',      lambda l: st_single(l, tau))]:
        b, v, mse = scalar_bvm(logits, obj, fn, n_samples)
        base[name] = dict(bias=b, variance=v, mse=mse)
        print(f"  [baseline {name}] bias={b:.4f}  var={v:.4f}  mse={mse:.4f}")

    results    = {n: dict(bias=[], variance=[], mse=[]) for n in ['reinmax_v3', 'reinmax_cv']}
    wall_times = {n: [] for n in ['reinmax_v3', 'reinmax_cv']}

    for R in repeats_list:
        print(f"  R={R}:")
        method_fns = {
            'reinmax_v3': lambda l, R=R: reinmax_v3_single(l, tau, R),
            'reinmax_cv': lambda l, R=R: reinmax_cv_single(l, tau, 0.5, R),
        }
        for name, fn in method_fns.items():
            t0 = time.time()
            b, v, mse = scalar_bvm(logits, obj, fn, n_samples)
            wall_times[name].append((time.time() - t0) / n_samples * 1000)
            results[name]['bias'].append(b)
            results[name]['variance'].append(v)
            results[name]['mse'].append(mse)
            print(f"    [{name}] bias={b:.4f}  var={v:.4f}  mse={mse:.4f}")

    return results, base, list(repeats_list), wall_times


def plot_repeats_experiment(results, base_metrics, repeats_list, wall_times,
                            save_dir=ABLATION_DIR):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    xs = repeats_list

    # Wall time
    for name, times in wall_times.items():
        axes[0].plot(xs, times, color=COLOURS.get(name, '#888'), label=name,
                     lw=2, marker='o', ms=5)
    axes[0].set_xlabel('R'); axes[0].set_ylabel('ms / sample')
    axes[0].set_title('Wall Time vs. R'); axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Variance with baselines
    for name, data in results.items():
        axes[1].plot(xs, data['variance'], color=COLOURS.get(name, '#888'),
                     label=name, lw=2, marker='o', ms=5)
    for name, m in base_metrics.items():
        axes[1].axhline(m['variance'], color=COLOURS.get(name, '#888'),
                        ls='--', lw=1.5, label=f'{name} (base)')
    axes[1].set_xlabel('R'); axes[1].set_ylabel('Variance'); axes[1].set_yscale('log')
    axes[1].set_title('Variance vs. R'); axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)

    # MSE with baselines
    for name, data in results.items():
        axes[2].plot(xs, data['mse'], color=COLOURS.get(name, '#888'),
                     label=name, lw=2, marker='o', ms=5)
    for name, m in base_metrics.items():
        axes[2].axhline(m['mse'], color=COLOURS.get(name, '#888'),
                        ls='--', lw=1.5, label=f'{name} (base)')
    axes[2].set_xlabel('R'); axes[2].set_ylabel('MSE'); axes[2].set_yscale('log')
    axes[2].set_title('MSE vs. R'); axes[2].legend(fontsize=7); axes[2].grid(True, alpha=0.3)

    plt.suptitle('Rao-Gumbel Sample Count Ablation (K=1, N=8)', fontsize=12)
    save_fig(os.path.join(save_dir, 'repeats.png'))


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
    print(f"\n{'='*60}\nExp 8: η ablation (ReinMax-CV)  (N={n_experts}, τ={tau}, R={repeats})\n{'='*60}")

    torch.manual_seed(seed)
    logits = torch.randn(batch_size, n_experts) * 1.5
    obj    = QuadraticObjective(n_experts, seed=seed)

    biases, variances, mses = [], [], []
    for eta in eta_list:
        fn = lambda l, e=eta: reinmax_cv_single(l, tau, eta=e, repeats=repeats)
        b, v, mse = scalar_bvm(logits, obj, fn, n_samples)
        biases.append(b); variances.append(v); mses.append(mse)
        print(f"  η={eta:.1f}  bias={b:.4f}  var={v:.4f}  mse={mse:.4f}")

    b_rm, v_rm, mse_rm = scalar_bvm(logits, obj, lambda l: reinmax_single(l, tau), n_samples)
    print(f"  [reinmax baseline] bias={b_rm:.4f}  var={v_rm:.4f}  mse={mse_rm:.4f}")

    return (list(eta_list), biases, variances, mses,
            dict(bias=b_rm, variance=v_rm, mse=mse_rm))


def plot_eta_experiment(eta_list, biases, variances, mses, reinmax_baseline,
                        save_dir=ABLATION_DIR):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    c_cv = COLOURS['reinmax_cv']
    c_rm = COLOURS['reinmax']
    for ax, vals, key, ylabel, title in zip(
        axes,
        [biases, variances, mses],
        ['bias', 'variance', 'mse'],
        ['Bias (L2)', 'Variance', 'MSE'],
        ['Gradient Bias vs. η', 'Gradient Variance vs. η', 'Gradient MSE vs. η'],
    ):
        ax.plot(eta_list, vals, color=c_cv, lw=2, marker='o', ms=6, label='reinmax_cv(η)')
        ax.axhline(reinmax_baseline[key], color=c_rm, ls='--', lw=1.5, label='reinmax')
        ax.set_xlabel('η'); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.suptitle('Control-Variate Strength η Ablation (K=1, N=8)', fontsize=12)
    save_fig(os.path.join(save_dir, 'eta.png'))


# ─────────────────────────────────────────────────────────────────────────────
# Exp 9 — K ablation (top-K routing)
# ─────────────────────────────────────────────────────────────────────────────

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
    print(f"\n{'='*60}\nExp 9: K ablation  (N={n_experts}, τ={tau})\n{'='*60}")

    torch.manual_seed(seed)
    logits = torch.randn(batch_size, n_experts) * 1.5
    obj    = QuadraticObjective(n_experts, seed=seed)

    results = {n: dict(variance=[], mse=[]) for n in ['reinmax', 'reinmax_v3', 'reinmax_cv']}

    for k in k_list:
        print(f"  K={k}:")
        topk_obj = lambda z, k=k: obj(z / k)
        # High-sample REINFORCE reference
        ref_fn    = lambda l, k=k: reinmax_topk(l, k=k, tau=tau)
        ref_grads = collect_grad_samples(ref_fn, logits, topk_obj, n_ref)
        ref_mean  = ref_grads.mean(0)

        method_fns = {
            'reinmax'    : lambda l, k=k: reinmax_topk(l, k=k, tau=tau),
            'reinmax_v3' : lambda l, k=k: reinmax_v3_topk(l, k=k, tau=tau, repeats=repeats),
            'reinmax_cv' : lambda l, k=k: reinmax_cv_topk(l, k=k, tau=tau, eta=0.5, repeats=repeats),
        }
        for name, fn in method_fns.items():
            grads = collect_grad_samples(fn, logits, topk_obj, n_samples)
            m     = bias_variance_metrics(grads, ref_mean)
            results[name]['variance'].append(m['variance'])
            results[name]['mse'].append(m['mse'])
            print(f"    [{name:12s}] var={m['variance']:.4f}  mse={m['mse']:.4f}")

    return results, list(k_list)


def plot_k_ablation(results, k_list, save_dir=ABLATION_DIR):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    line_panel(axes[0], k_list, {n: results[n]['variance'] for n in results},
               'K (selected experts)', 'Variance', 'Gradient Variance vs. K', yscale='log')
    line_panel(axes[1], k_list, {n: results[n]['mse']      for n in results},
               'K (selected experts)', 'MSE',      'Gradient MSE vs. K',      yscale='log')
    plt.suptitle('Effect of K (Top-K Selection, N=8)', fontsize=12)
    save_fig(os.path.join(save_dir, 'k_ablation.png'))


# ─────────────────────────────────────────────────────────────────────────────
# Exp 10 — Gradient variance during optimisation
# ─────────────────────────────────────────────────────────────────────────────

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
    print(f"\n{'='*60}\nExp 10: Gradient variance during optimisation  (N={n_experts})\n{'='*60}")

    obj = QuadraticObjective(n_experts, seed=seed)

    methods = {
        'st'         : lambda l: st_single(l, tau),
        'reinmax'    : lambda l: reinmax_single(l, tau),
        'reinmax_v3' : lambda l: reinmax_v3_single(l, tau, repeats),
        'reinmax_cv' : lambda l: reinmax_cv_single(l, tau, 0.5, repeats),
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
            loss  = -obj(z).mean()
            loss.backward()
            opt.step()

            with torch.no_grad():
                p  = F.softmax(logits, dim=-1)
                ef = (p * obj.f_at_experts()).sum(-1).mean().item()
            ef_traj.append(ef)

            if step % measure_every == 0:
                grads = collect_grad_samples(fn, logits.detach(), obj, n_var_samples)
                g_mean = grads.mean(0)
                var = ((grads - g_mean) ** 2).sum(-1).mean().item()
                var_traj.append(var)
                steps_rec.append(step)

        history[name] = dict(ef=ef_traj, var=var_traj, steps=steps_rec)
        print(f"  [{name}]  final E[f]={ef_traj[-1]:.4f}  final_var={var_traj[-1]:.4f}")

    return history


def plot_optimisation_variance(history, save_dir=ABLATION_DIR):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for name, data in history.items():
        c = COLOURS.get(name, '#888')
        axes[0].plot(data['ef'],              color=c, label=name, lw=2)
        axes[1].plot(data['steps'], data['var'], color=c, label=name, lw=2,
                     marker='o', ms=4)
    axes[0].set_xlabel('Step'); axes[0].set_ylabel('E[f(z)]')
    axes[0].set_title('Expected Objective'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel('Step'); axes[1].set_ylabel('Gradient Variance (log)')
    axes[1].set_yscale('log')
    axes[1].set_title('Gradient Variance During Optimisation')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.suptitle('Variance Reduction Benefit During Training (K=1, N=8)', fontsize=12)
    save_fig(os.path.join(save_dir, 'optimisation_variance.png'))


# ─────────────────────────────────────────────────────────────────────────────
# Exp 11 — (N, τ) variance-reduction grid heatmap
# ─────────────────────────────────────────────────────────────────────────────

def run_grid_experiment(
    n_experts_list=(4, 8, 16),
    tau_list=(0.3, 1.0, 2.0),
    n_samples: int = 200,
    batch_size: int = 2,
    repeats: int = 20,
    seed: int = 42,
):
    """Variance ratio (ReinMax-v3 / ReinMax) and (ReinMax-CV / ReinMax) over (N, τ)."""
    print(f"\n{'='*60}\nExp 11: (N, τ) Variance-Reduction Grid\n{'='*60}")

    ratio_v3 = np.zeros((len(n_experts_list), len(tau_list)))
    ratio_cv = np.zeros((len(n_experts_list), len(tau_list)))

    for i, N in enumerate(n_experts_list):
        for j, tau in enumerate(tau_list):
            torch.manual_seed(seed)
            logits = torch.randn(batch_size, N) * 1.5
            obj    = QuadraticObjective(N, seed=seed)

            _, v_rm, _ = scalar_bvm(logits, obj, lambda l, t=tau: reinmax_single(l, t),              n_samples)
            _, v_v3, _ = scalar_bvm(logits, obj, lambda l, t=tau, R=repeats: reinmax_v3_single(l, t, R), n_samples)
            _, v_cv, _ = scalar_bvm(logits, obj, lambda l, t=tau, R=repeats: reinmax_cv_single(l, t, 0.5, R), n_samples)

            ratio_v3[i, j] = v_v3 / (v_rm + 1e-12)
            ratio_cv[i, j] = v_cv / (v_rm + 1e-12)
            print(f"  N={N:2d} τ={tau}  v3/rm={ratio_v3[i,j]:.3f}  cv/rm={ratio_cv[i,j]:.3f}")

    return ratio_v3, ratio_cv, list(n_experts_list), list(tau_list)


def plot_grid_experiment(ratio_v3, ratio_cv, n_experts_list, tau_list,
                         save_dir=ABLATION_DIR):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    col_labels = [f'τ={t}' for t in tau_list]
    row_labels  = [f'N={n}' for n in n_experts_list]

    for ax, ratio, title in zip(
        axes,
        [ratio_v3, ratio_cv],
        ['Var(ReinMax-v3) / Var(ReinMax)', 'Var(ReinMax-CV) / Var(ReinMax)'],
    ):
        heatmap(ax, ratio, row_labels, col_labels, title,
                xlabel='Temperature', ylabel='N experts',
                fmt='.2f', cmap='RdYlGn_r')

    plt.suptitle('Variance Reduction Ratio vs. ReinMax  (< 1 = lower variance)', fontsize=11)
    save_fig(os.path.join(save_dir, 'grid_N_tau.png'))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(n_samples: int = 300, fast: bool = False):
    if fast:
        n_samples = 80

    summary = {}

    # Exp 5
    r5, n_list = run_scale_experiment(n_experts_list=[4, 8, 16, 32],
                                      n_samples=n_samples, tau=1.0, repeats=30)
    plot_scale_experiment(r5, n_list)
    summary['scale_N'] = {n: {k: v[-1] for k, v in m.items()} for n, m in r5.items()}

    # Exp 6
    r6, t_list = run_temperature_experiment(
        tau_list=[0.1, 0.3, 0.5, 1.0, 2.0], n_experts=8, n_samples=n_samples, repeats=30)
    plot_temperature_experiment(r6, t_list)

    # Exp 7
    r7, base7, rlist, wt7 = run_repeats_experiment(
        repeats_list=[5, 10, 20, 50, 100], n_experts=8, n_samples=n_samples, tau=1.0)
    plot_repeats_experiment(r7, base7, rlist, wt7)
    summary['repeats'] = {n: {k: v[-1] for k, v in m.items()} for n, m in r7.items()}

    # Exp 8
    eta_list, biases8, vars8, mses8, rm_base8 = run_eta_experiment(
        eta_list=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        n_experts=8, n_samples=n_samples, tau=1.0, repeats=30)
    plot_eta_experiment(eta_list, biases8, vars8, mses8, rm_base8)
    best_eta_idx = int(np.argmin(mses8))
    summary['eta'] = dict(best_eta=eta_list[best_eta_idx], best_mse=mses8[best_eta_idx])

    # Exp 9
    r9, k_list = run_k_ablation(
        k_list=[1, 2, 3, 4], n_experts=8, n_samples=max(n_samples, 200),
        n_ref=1500 if not fast else 300, tau=1.0, repeats=20)
    plot_k_ablation(r9, k_list)

    # Exp 10
    hist10 = run_optimisation_variance(
        n_experts=8, n_steps=200, lr=0.05, tau=1.0,
        n_var_samples=40 if not fast else 10, measure_every=10, repeats=30)
    plot_optimisation_variance(hist10)

    # Exp 11
    rv3, rcv, ne_list, tl = run_grid_experiment(
        n_experts_list=[4, 8, 16], tau_list=[0.3, 1.0, 2.0],
        n_samples=n_samples, repeats=20)
    plot_grid_experiment(rv3, rcv, ne_list, tl)
    summary['grid'] = dict(v3_mean_ratio=float(rv3.mean()),
                            cv_mean_ratio=float(rcv.mean()))

    path = os.path.join(ABLATION_DIR, 'ablation_summary.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nAblation summary → {path}")
    print(f"  Mean v3/rm variance ratio: {rv3.mean():.3f}  "
          f"(< 1 = ReinMax-v3 lower variance)")
    print(f"  Mean cv/rm variance ratio: {rcv.mean():.3f}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--n-samples', type=int, default=300)
    p.add_argument('--fast', action='store_true')
    args = p.parse_args()
    main(n_samples=args.n_samples, fast=args.fast)
