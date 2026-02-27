"""
Shared plotting utilities for SparseMixer gradient estimator experiments.

All public functions save a figure to disk and close it (no display).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette (consistent across all figures)
# ─────────────────────────────────────────────────────────────────────────────

COLOURS: Dict[str, str] = {
    'st'               : '#1f77b4',
    'reinmax'          : '#ff7f0e',
    'reinmax_v3'       : '#2ca02c',
    'reinmax_cv'       : '#d62728',
    'gumbel_softmax'   : '#9467bd',
    'reinmax_topk'     : '#ff7f0e',
    'reinmax_v3_topk'  : '#2ca02c',
    'reinmax_cv_topk'  : '#d62728',
}

_DEFAULT_COLOUR = '#888888'


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _colour(name: str) -> str:
    return COLOURS.get(name, _DEFAULT_COLOUR)


def save_fig(path: str, dpi: int = 150):
    """Tight-layout, save, and close."""
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()
    print(f"  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Line panels
# ─────────────────────────────────────────────────────────────────────────────

def line_panel(
    ax,
    xs: Sequence,
    ys_dict: Dict[str, Sequence],
    xlabel: str,
    ylabel: str,
    title: str,
    yscale: str = 'linear',
    markers: bool = True,
):
    """Plot multiple curves on one axis."""
    for name, ys in ys_dict.items():
        c  = _colour(name)
        mk = 'o' if markers else None
        ax.plot(xs, ys, color=c, label=name, lw=2, marker=mk, markersize=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_yscale(yscale)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def triple_line_figure(
    xs: Sequence,
    bias_dict: Dict[str, Sequence],
    var_dict: Dict[str, Sequence],
    mse_dict: Dict[str, Sequence],
    xlabel: str,
    suptitle: str,
    path: str,
):
    """Bias / Variance / MSE as three sub-panels."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    line_panel(axes[0], xs, bias_dict, xlabel, 'Bias (L2)', 'Gradient Bias')
    line_panel(axes[1], xs, var_dict,  xlabel, 'Variance',  'Gradient Variance', yscale='log')
    line_panel(axes[2], xs, mse_dict,  xlabel, 'MSE',       'Gradient MSE',      yscale='log')
    plt.suptitle(suptitle, fontsize=12)
    save_fig(path)


# ─────────────────────────────────────────────────────────────────────────────
# Bar panels
# ─────────────────────────────────────────────────────────────────────────────

def bar_panel(
    ax,
    names: List[str],
    values: List[float],
    ylabel: str,
    title: str,
    colour_by_name: bool = True,
    hline: Optional[float] = None,
):
    """Vertical bar chart."""
    colours = [_colour(n) for n in names] if colour_by_name else None
    bars = ax.bar(names, values, color=colours)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=20)
    if hline is not None:
        ax.axhline(hline, color='k', ls='--', lw=1, label=f'ref={hline:.3f}')
        ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def heatmap(
    ax,
    data: np.ndarray,
    row_labels: List,
    col_labels: List,
    title: str,
    xlabel: str,
    ylabel: str,
    fmt: str = '.3f',
    cmap: str = 'RdYlGn_r',
):
    """Annotated heatmap."""
    im = ax.imshow(data, cmap=cmap, aspect='auto')
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, format(data[i, j], fmt),
                    ha='center', va='center', fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.04)


# ─────────────────────────────────────────────────────────────────────────────
# Gradient distribution histograms
# ─────────────────────────────────────────────────────────────────────────────

def grad_distribution_grid(
    results: Dict[str, dict],
    b: int = 0,
    n: int = 0,
    path: str = 'grad_dist.png',
):
    """
    Plot gradient sample distributions for one (batch, expert) pair.

    Parameters
    ----------
    results : dict[name → dict] where dict has 'all_grads' (S, B, N)
              and 'mean_grad' (B, N)
    b, n    : which batch/expert index to plot
    path    : output path
    """
    names = list(results.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3))
    if len(names) == 1:
        axes = [axes]

    for i, name in enumerate(names):
        g_t = results[name]['all_grads'][:, b, n].detach().float()
        g   = g_t.tolist()
        axes[i].hist(g, bins=40, color=_colour(name), alpha=0.75, density=True)
        axes[i].axvline(g_t.mean().item(), color='k', ls='--', lw=1.5, label='mean')
        axes[i].set_title(f'{name}\nstd={g_t.std().item():.3f}')
        axes[i].set_xlabel('gradient')
        axes[i].legend(fontsize=7)

    plt.suptitle(f'Gradient Distributions  (batch={b}, expert={n})', fontsize=11)
    save_fig(path)


# ─────────────────────────────────────────────────────────────────────────────
# Convergence / training curves
# ─────────────────────────────────────────────────────────────────────────────

def convergence_figure(
    history: Dict[str, dict],
    best_exp: int,
    path: str,
):
    """Two-panel: E[f(z)] and P(best expert) over optimisation steps."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for name, h in history.items():
        c = _colour(name)
        axes[0].plot(h['ef'], color=c, label=name, lw=2)
        axes[1].plot(h['pb'], color=c, label=name, lw=2)

    axes[0].set_xlabel('step'); axes[0].set_ylabel('E[f(z)]')
    axes[0].set_title('Expected Objective'); axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('step'); axes[1].set_ylabel(f'P(expert {best_exp})')
    axes[1].set_title('Probability of Best Expert'); axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Convergence Comparison (K=1)', fontsize=12)
    save_fig(path)


# ─────────────────────────────────────────────────────────────────────────────
# Specialization experiment figures
# ─────────────────────────────────────────────────────────────────────────────

def specialization_curves(
    all_results: Dict[str, dict],
    path: str,
    suptitle: str = 'Expert Specialization Training',
):
    """
    Three panels for the specialization experiment:
      1. Training loss curves
      2. Specialization score curves
      3. Final loss / specialization bar charts
    """
    names = list(all_results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Panel 1: training loss
    ax = axes[0]
    for name in names:
        ys = all_results[name]['loss_traj']
        ax.plot(ys, color=_colour(name), label=name, lw=1.5, alpha=0.85)
    # smooth with simple running average for readability
    ax.set_xlabel('step'); ax.set_ylabel('MSE loss')
    ax.set_title('Training Loss'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Panel 2: specialization score
    ax = axes[1]
    for name in names:
        xs = all_results[name]['spec_steps']
        ys = all_results[name]['spec_traj']
        ax.plot(xs, ys, color=_colour(name), label=name, lw=2, marker='o', markersize=4)
    ax.set_xlabel('step'); ax.set_ylabel('Specialization score')
    ax.set_title('Expert Specialization'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Panel 3: final bars
    ax = axes[2]
    final_specs = [all_results[n]['final_spec'] for n in names]
    bar_panel(ax, names, final_specs, 'Final specialization score',
              'Final Specialization Score')

    plt.suptitle(suptitle, fontsize=12)
    save_fig(path)


def specialization_multi_seed(
    all_results: Dict[str, Dict[str, list]],
    path: str,
    suptitle: str = 'Expert Specialization (multi-seed)',
):
    """
    Summary across multiple seeds: mean ± std bars for loss and spec score.

    all_results : {name → {'final_loss': [s1, s2, ...], 'final_spec': [s1, ...]}}
    """
    names = list(all_results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax_idx, metric, ylabel, title in [
        (0, 'final_loss', 'Final MSE Loss',        'Final Loss by Method'),
        (1, 'final_spec', 'Final Spec. Score',     'Final Specialization Score'),
    ]:
        ax = axes[ax_idx]
        means = [float(np.mean(all_results[n][metric])) for n in names]
        stds  = [float(np.std( all_results[n][metric])) for n in names]
        colours = [_colour(n) for n in names]
        ax.bar(names, means, yerr=stds, color=colours, capsize=5, alpha=0.85)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.tick_params(axis='x', rotation=20); ax.grid(True, axis='y', alpha=0.3)

    plt.suptitle(suptitle, fontsize=12)
    save_fig(path)
