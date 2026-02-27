"""
Experiments validating SparseMixer gradient estimators.

Experiment 1 — Gradient bias & variance (K=1)
----------------------------------------------
  For a simple quadratic objective f(z) = b^T z + z^T A z we can compute
  the *exact* gradient  d/d_theta E[f(z)]  analytically:

      exact = p * (f_vals - E_p[f])

  where p = softmax(theta) and f_vals[i] = f(e_i).

  Each estimator is run for n_samples one-sample gradient estimates.
  We then measure:
    bias     = || mean_estimate - exact ||_2
    variance = E[ || estimate - mean_estimate ||^2 ]
    mse      = E[ || estimate - exact ||^2 ] = bias^2 + variance

Experiment 2 — Convergence (K=1)
---------------------------------
  Maximise E[f(z)] by gradient ascent on logits.
  Tracks E[f(z)] and P(best expert) vs. step.

Experiment 3 — Top-K bias & variance (K=2)
-------------------------------------------
  Same setup but with K=2; exact gradient via high-sample REINFORCE.

Experiment 4 — SparseMixer end-to-end (classification)
--------------------------------------------------------
  Train a 2-layer SparseMixerModel on a synthetic Gaussian-mixture
  classification task and compare final test accuracy across methods.
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

# Allow running from the repo root
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
from sparse_mixer.sparse_mixer import SparseMixerModel

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Objective function
# ─────────────────────────────────────────────────────────────────────────────

class QuadraticObjective:
    """f(z) = b^T z + z^T A z   (A, b fixed)."""

    def __init__(self, n: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.A = torch.randn(n, n, generator=g) * 0.3
        self.b = torch.randn(n, generator=g) * 0.5
        self.n = n

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, N) → (B,)"""
        return (z @ self.A * z).sum(-1) + (z * self.b).sum(-1)

    def f_at_experts(self) -> torch.Tensor:
        """Returns f(e_i) for each expert i.  Shape: (N,)"""
        I = torch.eye(self.n)
        return self(I)   # (N,)

    def exact_gradient(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Analytic gradient d/d_theta E[f(z)]  where p = softmax(theta).
        exact = p * (f_vals - E_p[f])
        Returns (B, N).
        """
        p      = F.softmax(logits, dim=-1)                    # (B, N)
        f_vals = self.f_at_experts().to(logits.device)        # (N,)
        E_f    = (p * f_vals).sum(-1, keepdim=True)           # (B, 1)
        return p * (f_vals - E_f)                             # (B, N)


# ─────────────────────────────────────────────────────────────────────────────
# Core measurement helpers
# ─────────────────────────────────────────────────────────────────────────────

def collect_grad_samples(
    method_fn,
    logits: torch.Tensor,
    obj: QuadraticObjective,
    n_samples: int,
) -> torch.Tensor:
    """
    Draw n_samples one-sample gradient estimates for a given method.

    method_fn: logits (requires_grad) → (z, p)

    Returns: (n_samples, B, N)
    """
    B, N = logits.shape
    grads = []
    for _ in range(n_samples):
        l = logits.detach().clone().requires_grad_(True)
        z, p = method_fn(l)
        loss = obj(z).sum()
        loss.backward()
        grads.append(l.grad.detach().clone())
    return torch.stack(grads, dim=0)   # (S, B, N)


def bias_variance_metrics(
    grads: torch.Tensor,
    exact: torch.Tensor,
) -> dict:
    """
    Compute bias, variance, MSE from collected gradient samples.

    grads : (S, B, N)
    exact : (B, N)
    """
    mean_g = grads.mean(dim=0)                              # (B, N)
    bias   = (mean_g - exact).norm().item()
    var    = ((grads - mean_g) ** 2).sum(dim=-1).mean().item()
    mse    = ((grads - exact ) ** 2).sum(dim=-1).mean().item()
    std    = grads.std(dim=0).mean().item()
    return dict(bias=bias, variance=var, mse=mse, std=std,
                mean_grad=mean_g, all_grads=grads)


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
    print(f"  Best expert: {best_exp}  "
          f"f-values: {f_vals.numpy().round(2)}")

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
        print(f"  [{name}]  final E[f]={ef_traj[-1]:.4f}  "
              f"P(best)={pb_traj[-1]:.3f}")

    return history, best_exp


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
    """
    For K=2 there is no closed-form exact gradient, so we use high-sample
    REINFORCE as the reference.
    """
    torch.manual_seed(seed)
    print(f"\n{'='*60}")
    print(f"Exp 3: K=2 Bias & Variance  (N={n_experts}, τ={tau})")
    print('='*60)

    logits = torch.randn(batch_size, n_experts) * 1.5
    obj    = QuadraticObjective(n_experts, seed=seed)

    # normalise k-hot: f(z/k)
    def topk_obj(z): return obj(z / 2)

    # Reference: high-sample REINFORCE
    print(f"  Computing REINFORCE reference ({reinforce_ref} samples) ...",
          end='', flush=True)
    ref_fn = lambda l: reinmax_topk(l, k=2, tau=tau)
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


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 4  —  End-to-end classification
# ─────────────────────────────────────────────────────────────────────────────

def make_classification_data(
    n_classes: int = 4,
    n_per_class: int = 200,
    input_dim: int = 8,
    seed: int = 0,
):
    """Gaussian mixture classification dataset."""
    g = torch.Generator().manual_seed(seed)
    means  = torch.randn(n_classes, input_dim, generator=g) * 2
    X_list, Y_list = [], []
    for c in range(n_classes):
        X = torch.randn(n_per_class, input_dim, generator=g) + means[c]
        Y = torch.full((n_per_class,), c, dtype=torch.long)
        X_list.append(X); Y_list.append(Y)
    X = torch.cat(X_list); Y = torch.cat(Y_list)
    # shuffle
    idx = torch.randperm(X.size(0), generator=g)
    return X[idx], Y[idx]


def run_e2e_classification(
    n_experts: int = 8,
    n_epochs: int = 30,
    lr: float = 3e-3,
    batch_size: int = 64,
    tau: float = 1.0,
    seed: int = 42,
):
    torch.manual_seed(seed)
    print(f"\n{'='*60}")
    print(f"Exp 4: End-to-end classification  (N={n_experts}, τ={tau})")
    print('='*60)

    n_classes, input_dim = 4, 8
    X, Y = make_classification_data(n_classes=n_classes, input_dim=input_dim)
    n_train = int(0.8 * len(X))
    X_tr, Y_tr = X[:n_train], Y[:n_train]
    X_te, Y_te = X[n_train:], Y[n_train:]

    methods = ['st', 'reinmax', 'reinmax_v3', 'reinmax_cv']
    history = {}

    for method in methods:
        torch.manual_seed(seed)
        model = SparseMixerModel(
            input_dim=input_dim,
            d_model=32,
            n_layers=2,
            n_experts=n_experts,
            k=1,
            n_classes=n_classes,
            d_ff=64,
            method=method,
            tau=tau,
            eta=0.5,
            repeats=20,
        )
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        tr_acc_traj, te_acc_traj = [], []

        for epoch in range(n_epochs):
            model.train()
            # mini-batches
            perm = torch.randperm(n_train)
            total_loss, n_correct, n_total = 0.0, 0, 0
            for i in range(0, n_train, batch_size):
                idx   = perm[i: i + batch_size]
                xb    = X_tr[idx].unsqueeze(1)    # (B, 1, input_dim)
                yb    = Y_tr[idx]
                opt.zero_grad()
                logits, aux = model(xb)
                loss = F.cross_entropy(logits, yb) + aux
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(yb)
                n_correct  += (logits.argmax(-1) == yb).sum().item()
                n_total    += len(yb)
            tr_acc = n_correct / n_total

            model.eval()
            with torch.no_grad():
                logits_te, _ = model(X_te.unsqueeze(1))
                te_acc = (logits_te.argmax(-1) == Y_te).float().mean().item()

            tr_acc_traj.append(tr_acc)
            te_acc_traj.append(te_acc)

        history[method] = dict(tr_acc=tr_acc_traj, te_acc=te_acc_traj)
        print(f"  [{method}]  train_acc={tr_acc_traj[-1]:.3f}"
              f"  test_acc={te_acc_traj[-1]:.3f}")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

COLOURS = {
    'st'              : '#2196F3',
    'reinmax'         : '#4CAF50',
    'reinmax_v3'      : '#FF9800',
    'reinmax_cv'      : '#E91E63',
    'reinmax_topk'    : '#4CAF50',
    'reinmax_v3_topk' : '#FF9800',
    'reinmax_cv_topk' : '#E91E63',
}


def _bar_ax(ax, names, vals, ylabel, title, log=False):
    colours = [COLOURS.get(n, '#888888') for n in names]
    ax.bar(range(len(names)), vals, color=colours)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log and min(v for v in vals if v > 0) > 0:
        ax.set_yscale('log')
    ax.grid(True, alpha=0.3)


def plot_k1_bias_variance(results, exact_grad, save_dir=RESULTS_DIR):
    names = list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    _bar_ax(axes[0], names, [results[n]['bias']     for n in names], 'L2 norm', 'Gradient Bias')
    _bar_ax(axes[1], names, [results[n]['variance'] for n in names], 'Variance', 'Gradient Variance', log=True)
    _bar_ax(axes[2], names, [results[n]['mse']      for n in names], 'MSE', 'Gradient MSE', log=True)
    plt.suptitle('K=1 Gradient Quality', fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, 'k1_bias_variance.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → {path}")


def plot_k1_gradient_distributions(results, exact_grad, save_dir=RESULTS_DIR):
    names = list(results.keys())
    b, n  = 0, 0    # visualise batch=0, feature=0
    exact_val = exact_grad[b, n].item()

    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4), sharey=False)
    if len(names) == 1:
        axes = [axes]

    for i, name in enumerate(names):
        g_t = results[name]['all_grads'][:, b, n].detach().float()
        g   = g_t.tolist()
        axes[i].hist(g, bins=50, density=True, alpha=0.75,
                     color=COLOURS.get(name, '#888888'))
        axes[i].axvline(exact_val,          color='red',   lw=2, ls='--', label='exact')
        axes[i].axvline(g_t.mean().item(),  color='black', lw=2, ls='--', label='mean est.')
        axes[i].set_title(f'{name}\nstd={g_t.std().item():.3f}')
        axes[i].legend(fontsize=7)
        axes[i].grid(True, alpha=0.3)

    plt.suptitle('Gradient Distributions (batch=0, dim=0)', fontsize=11)
    plt.tight_layout()
    path = os.path.join(save_dir, 'k1_gradient_distributions.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → {path}")


def plot_convergence(history, best_exp, save_dir=RESULTS_DIR):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, data in history.items():
        c = COLOURS.get(name, '#888888')
        axes[0].plot(data['ef'], label=name, color=c, lw=2)
        axes[1].plot(data['pb'], label=name, color=c, lw=2)
    for ax, ylabel, title in zip(
        axes,
        ['E[f(z)]', f'P(expert={best_exp})'],
        ['Convergence of E[f(z)]', 'Best Expert Probability'],
    ):
        ax.set_xlabel('Step'); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.suptitle('K=1 Convergence', fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, 'k1_convergence.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → {path}")


def plot_k2_bias_variance(results, save_dir=RESULTS_DIR):
    names = list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    _bar_ax(axes[0], names, [results[n]['bias']     for n in names], 'L2 norm', 'Gradient Bias')
    _bar_ax(axes[1], names, [results[n]['variance'] for n in names], 'Variance', 'Gradient Variance', log=True)
    _bar_ax(axes[2], names, [results[n]['mse']      for n in names], 'MSE', 'Gradient MSE', log=True)
    plt.suptitle('K=2 Gradient Quality (vs. REINFORCE reference)', fontsize=11)
    plt.tight_layout()
    path = os.path.join(save_dir, 'k2_bias_variance.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → {path}")


def plot_e2e_classification(history, save_dir=RESULTS_DIR):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, data in history.items():
        c = COLOURS.get(name, '#888888')
        axes[0].plot(data['tr_acc'], label=name, color=c, lw=2)
        axes[1].plot(data['te_acc'], label=name, color=c, lw=2)
    for ax, title in zip(axes, ['Train Accuracy', 'Test Accuracy']):
        ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
        ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    plt.suptitle('End-to-end Classification', fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, 'e2e_classification.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("SparseMixer + ReinMax Gradient Estimator Experiments")
    print("="*60)

    summary = {}

    # ── Exp 1: K=1 bias & variance ───────────────────────────────────────
    res1, exact1 = run_k1_bias_variance(
        n_experts=8, batch_size=4, n_samples=500, tau=1.0
    )
    summary['k1'] = {n: {k: v for k, v in m.items()
                         if k not in ('mean_grad', 'all_grads')}
                     for n, m in res1.items()}

    print("\nPlotting Exp 1 ...")
    plot_k1_bias_variance(res1, exact1)
    plot_k1_gradient_distributions(res1, exact1)

    # ── Exp 2: K=1 convergence ───────────────────────────────────────────
    hist2, best2 = run_k1_convergence(
        n_experts=8, n_steps=300, lr=0.05, tau=1.0, batch_size=16
    )
    print("\nPlotting Exp 2 ...")
    plot_convergence(hist2, best2)

    # ── Exp 3: K=2 bias & variance ───────────────────────────────────────
    res3, ref3 = run_k2_bias_variance(
        n_experts=8, batch_size=4, n_samples=400, tau=1.0, reinforce_ref=3000
    )
    summary['k2'] = {n: {k: v for k, v in m.items()
                         if k not in ('mean_grad', 'all_grads')}
                     for n, m in res3.items()}
    print("\nPlotting Exp 3 ...")
    plot_k2_bias_variance(res3)

    # ── Exp 4: end-to-end ────────────────────────────────────────────────
    hist4 = run_e2e_classification(
        n_experts=8, n_epochs=30, lr=3e-3, tau=1.0
    )
    summary['e2e'] = {n: dict(final_tr=d['tr_acc'][-1], final_te=d['te_acc'][-1])
                      for n, d in hist4.items()}
    print("\nPlotting Exp 4 ...")
    plot_e2e_classification(hist4)

    # ── Save summary ─────────────────────────────────────────────────────
    summary_path = os.path.join(RESULTS_DIR, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)

    print("\nK=1 Bias / Variance / MSE:")
    for name, m in summary['k1'].items():
        print(f"  {name:12s}  bias={m['bias']:.4f}  "
              f"var={m['variance']:.4f}  mse={m['mse']:.4f}")

    print("\nK=2 (vs. REINFORCE reference):")
    for name, m in summary['k2'].items():
        print(f"  {name:18s}  bias={m['bias']:.4f}  "
              f"var={m['variance']:.4f}  mse={m['mse']:.4f}")

    print("\nEnd-to-end classification:")
    for name, m in summary['e2e'].items():
        print(f"  {name:12s}  test_acc={m['final_te']:.3f}")

    print(f"\nAll results saved to {RESULTS_DIR}/")
    print("Done.")


if __name__ == '__main__':
    main()
