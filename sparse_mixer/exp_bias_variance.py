"""
Experiments 1-4: Gradient bias & variance for SparseMixer estimators.

Exp 1 — K=1 bias & variance
  Closed-form REINFORCE gradient as ground truth; compare ST, ReinMax,
  ReinMax-v3, ReinMax-CV on bias / variance / MSE.

Exp 2 — K=1 convergence
  Gradient ascent on E[f(z)]; track E[f] and P(best expert) vs. step.

Exp 3 — K=2 bias & variance
  Top-K routing; high-sample REINFORCE as reference gradient.

Exp 4 — End-to-end SparseMixerModel classification
  Two-layer SparseMixerModel on a synthetic Gaussian-mixture task.
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sparse_mixer.estimators import (
    st_single, reinmax_single, reinmax_v3_single, reinmax_cv_single,
    reinmax_topk, reinmax_v3_topk, reinmax_cv_topk,
)
from sparse_mixer.sparse_mixer import SparseMixerModel
from sparse_mixer.objectives  import QuadraticObjective
from sparse_mixer.metrics     import collect_grad_samples, bias_variance_metrics
from sparse_mixer.plotting    import (
    COLOURS, convergence_figure, grad_distribution_grid,
    bar_panel, save_fig, line_panel,
)

import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1  —  K=1 bias & variance
# ─────────────────────────────────────────────────────────────────────────────

def run_k1_bias_variance(
    n_experts: int = 8,
    batch_size: int = 4,
    n_samples: int = 500,
    tau: float = 1.0,
    seed: int = 42,
):
    torch.manual_seed(seed)
    print(f"\n{'='*60}")
    print(f"Exp 1: K=1 Bias & Variance  (N={n_experts}, τ={tau})")
    print('='*60)

    logits = torch.randn(batch_size, n_experts) * 1.5
    obj    = QuadraticObjective(n_experts, seed=seed)
    exact  = obj.exact_gradient(logits)

    methods = {
        'st'         : lambda l: st_single(l, tau),
        'reinmax'    : lambda l: reinmax_single(l, tau),
        'reinmax_v3' : lambda l: reinmax_v3_single(l, tau, repeats=50),
        'reinmax_cv' : lambda l: reinmax_cv_single(l, tau, eta=0.5, repeats=50),
    }

    results = {}
    for name, fn in methods.items():
        t0 = time.time()
        print(f"  [{name}] collecting {n_samples} samples ...", end='', flush=True)
        grads = collect_grad_samples(fn, logits, obj, n_samples)
        m     = bias_variance_metrics(grads, exact)
        elapsed = time.time() - t0
        print(f"  bias={m['bias']:.4f}  var={m['variance']:.4f}"
              f"  mse={m['mse']:.4f}  std={m['std']:.4f}  ({elapsed:.1f}s)")
        results[name] = m

    return results, exact


def plot_k1_bias_variance(results, exact, save_dir=RESULTS_DIR):
    names  = list(results.keys())
    biases = [results[n]['bias']     for n in names]
    vars_  = [results[n]['variance'] for n in names]
    mses   = [results[n]['mse']      for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    bar_panel(axes[0], names, biases,  'Bias (L2)',  'Gradient Bias')
    bar_panel(axes[1], names, vars_,   'Variance',   'Gradient Variance')
    bar_panel(axes[2], names, mses,    'MSE',        'Gradient MSE')
    plt.suptitle('K=1 Gradient Bias & Variance Comparison', fontsize=12)
    save_fig(os.path.join(save_dir, 'k1_bias_variance.png'))

    grad_distribution_grid(results, b=0, n=0,
                           path=os.path.join(save_dir, 'k1_gradient_distributions.png'))


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2  —  Convergence (K=1)
# ─────────────────────────────────────────────────────────────────────────────

def run_k1_convergence(
    n_experts: int = 8,
    n_steps: int = 300,
    lr: float = 0.05,
    tau: float = 1.0,
    batch_size: int = 16,
    seed: int = 42,
):
    torch.manual_seed(seed)
    print(f"\n{'='*60}")
    print(f"Exp 2: K=1 Convergence  (N={n_experts}, τ={tau}, lr={lr})")
    print('='*60)

    obj      = QuadraticObjective(n_experts, seed=seed)
    f_vals   = obj.f_at_experts()
    best_exp = f_vals.argmax().item()
    print(f"  Best expert: {best_exp}  f-values: {f_vals.numpy().round(2)}")

    methods = {
        'st'         : lambda l: st_single(l, tau),
        'reinmax'    : lambda l: reinmax_single(l, tau),
        'reinmax_v3' : lambda l: reinmax_v3_single(l, tau, repeats=50),
        'reinmax_cv' : lambda l: reinmax_cv_single(l, tau, eta=0.5, repeats=50),
    }

    history = {}
    for name, fn in methods.items():
        torch.manual_seed(seed)
        logits = nn.Parameter(torch.randn(batch_size, n_experts) * 0.1)
        opt    = torch.optim.SGD([logits], lr=lr)
        ef_traj, pb_traj = [], []

        for _ in range(n_steps):
            opt.zero_grad()
            z, _ = fn(logits)
            loss = -obj(z).mean()
            loss.backward()
            opt.step()
            with torch.no_grad():
                p   = F.softmax(logits, dim=-1)
                ef  = (p * f_vals).sum(-1).mean().item()
                pb  = p[:, best_exp].mean().item()
            ef_traj.append(ef)
            pb_traj.append(pb)

        history[name] = dict(ef=ef_traj, pb=pb_traj)
        print(f"  [{name}]  final E[f]={ef_traj[-1]:.4f}  P(best)={pb_traj[-1]:.3f}")

    return history, best_exp


def plot_k1_convergence(history, best_exp, save_dir=RESULTS_DIR):
    convergence_figure(history, best_exp,
                       path=os.path.join(save_dir, 'k1_convergence.png'))


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3  —  K=2 bias & variance
# ─────────────────────────────────────────────────────────────────────────────

def run_k2_bias_variance(
    n_experts: int = 8,
    batch_size: int = 4,
    n_samples: int = 400,
    tau: float = 1.0,
    seed: int = 42,
    reinforce_ref: int = 5000,
):
    torch.manual_seed(seed)
    print(f"\n{'='*60}")
    print(f"Exp 3: K=2 Bias & Variance  (N={n_experts}, τ={tau})")
    print('='*60)

    logits  = torch.randn(batch_size, n_experts) * 1.5
    obj     = QuadraticObjective(n_experts, seed=seed)
    topk_obj = lambda z: obj(z / 2)

    print(f"  Computing REINFORCE reference ({reinforce_ref} samples) ...",
          end='', flush=True)
    ref_fn    = lambda l: reinmax_topk(l, k=2, tau=tau)
    ref_grads = collect_grad_samples(ref_fn, logits, topk_obj, reinforce_ref)
    ref_mean  = ref_grads.mean(0)
    print(" done.")

    methods = {
        'reinmax_topk'    : lambda l: reinmax_topk(l, k=2, tau=tau),
        'reinmax_v3_topk' : lambda l: reinmax_v3_topk(l, k=2, tau=tau, repeats=30),
        'reinmax_cv_topk' : lambda l: reinmax_cv_topk(l, k=2, tau=tau, eta=0.5, repeats=30),
    }

    results = {}
    for name, fn in methods.items():
        t0 = time.time()
        print(f"  [{name}] collecting {n_samples} samples ...", end='', flush=True)
        grads = collect_grad_samples(fn, logits, topk_obj, n_samples)
        m     = bias_variance_metrics(grads, ref_mean)
        elapsed = time.time() - t0
        print(f"  bias={m['bias']:.4f}  var={m['variance']:.4f}"
              f"  mse={m['mse']:.4f}  ({elapsed:.1f}s)")
        results[name] = m

    return results, ref_mean


def plot_k2_bias_variance(results, ref_mean, save_dir=RESULTS_DIR):
    names  = list(results.keys())
    biases = [results[n]['bias']     for n in names]
    vars_  = [results[n]['variance'] for n in names]
    mses   = [results[n]['mse']      for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    bar_panel(axes[0], names, biases,  'Bias (L2)', 'Gradient Bias (K=2)')
    bar_panel(axes[1], names, vars_,   'Variance',  'Gradient Variance (K=2)')
    bar_panel(axes[2], names, mses,    'MSE',       'Gradient MSE (K=2)')
    plt.suptitle('K=2 Gradient Bias & Variance Comparison (ref=5k REINFORCE)', fontsize=12)
    save_fig(os.path.join(save_dir, 'k2_bias_variance.png'))


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 4  —  End-to-end classification
# ─────────────────────────────────────────────────────────────────────────────

def _make_classification_data(n_classes=4, n_train=2000, n_test=500,
                               d_model=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    means = torch.randn(n_classes, d_model, generator=g) * 2.0
    def _sample(n):
        labels = torch.randint(0, n_classes, (n,))
        x = means[labels] + torch.randn(n, d_model, generator=g) * 0.5
        return x, labels
    return _sample(n_train), _sample(n_test)


def run_e2e_classification(
    d_model: int = 16,
    n_experts: int = 4,
    n_classes: int = 4,
    n_layers: int = 2,
    n_steps: int = 500,
    batch_size: int = 64,
    lr: float = 3e-3,
    tau: float = 1.0,
    seed: int = 42,
):
    torch.manual_seed(seed)
    print(f"\n{'='*60}")
    print(f"Exp 4: E2E Classification  (N={n_experts}, n_layers={n_layers})")
    print('='*60)

    (x_tr, y_tr), (x_te, y_te) = _make_classification_data(
        n_classes, seed=seed, d_model=d_model)

    methods = ['st', 'reinmax', 'reinmax_v3', 'reinmax_cv', 'gumbel_softmax']
    results = {}

    for method in methods:
        torch.manual_seed(seed)
        model = SparseMixerModel(
            input_dim=d_model, d_model=d_model, n_layers=n_layers,
            n_experts=n_experts, k=1, n_classes=n_classes,
            method=method, tau=tau,
        )
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_traj, acc_traj = [], []

        for step in range(n_steps):
            idx  = torch.randperm(len(x_tr))[:batch_size]
            xb, yb = x_tr[idx].unsqueeze(1), y_tr[idx]   # (B, 1, d)
            opt.zero_grad()
            logits, aux = model(xb)
            loss = F.cross_entropy(logits, yb) + aux
            loss.backward()
            opt.step()
            loss_traj.append(loss.item())

            if (step + 1) % 50 == 0:
                with torch.no_grad():
                    tl, _ = model(x_te.unsqueeze(1))
                    acc   = (tl.argmax(-1) == y_te).float().mean().item()
                acc_traj.append(acc)

        with torch.no_grad():
            tl, _ = model(x_te.unsqueeze(1))
            final_acc = (tl.argmax(-1) == y_te).float().mean().item()

        results[method] = dict(loss_traj=loss_traj, acc_traj=acc_traj,
                                final_acc=final_acc)
        print(f"  [{method:14s}]  final_acc={final_acc:.3f}")

    return results


def plot_e2e_classification(results, save_dir=RESULTS_DIR):
    names = list(results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    for name in names:
        ys = results[name]['loss_traj']
        ax.plot(ys, color=COLOURS.get(name, '#888'), label=name, lw=1.5, alpha=0.8)
    ax.set_xlabel('step'); ax.set_ylabel('loss')
    ax.set_title('Training Loss'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    final_accs = [results[n]['final_acc'] for n in names]
    bar_panel(ax, names, final_accs, 'Test Accuracy', 'Final Test Accuracy')

    plt.suptitle('End-to-End SparseMixer Classification', fontsize=12)
    save_fig(os.path.join(save_dir, 'e2e_classification.png'))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(n_samples: int = 500, fast: bool = False):
    if fast:
        n_samples = 100

    # Exp 1
    r1, exact1 = run_k1_bias_variance(n_samples=n_samples)
    plot_k1_bias_variance(r1, exact1)

    # Exp 2
    r2, best_exp = run_k1_convergence()
    plot_k1_convergence(r2, best_exp)

    # Exp 3
    r3, ref3 = run_k2_bias_variance(n_samples=max(n_samples, 400),
                                     reinforce_ref=5000 if not fast else 500)
    plot_k2_bias_variance(r3, ref3)

    # Exp 4
    r4 = run_e2e_classification()
    plot_e2e_classification(r4)

    # Summary
    summary = {
        'k1': {n: {k: v for k, v in m.items() if isinstance(v, float)}
               for n, m in r1.items()},
        'k2': {n: {k: v for k, v in m.items() if isinstance(v, float)}
               for n, m in r3.items()},
        'e2e': {n: {'final_acc': r['final_acc']} for n, r in r4.items()},
    }
    path = os.path.join(RESULTS_DIR, 'bias_variance_summary.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved → {path}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--n-samples', type=int, default=500)
    p.add_argument('--fast', action='store_true')
    args = p.parse_args()
    main(n_samples=args.n_samples, fast=args.fast)
