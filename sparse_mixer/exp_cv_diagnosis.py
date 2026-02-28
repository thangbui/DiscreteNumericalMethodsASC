"""
Experiment 14 — Control Variate Diagnosis: Why does ReinMax-CV Underperform?
=============================================================================

Background
----------
ReinMax-CV (reinmax_cv_single / reinmax_cv_topk) adds a zero-mean correction
to the base ReinMax gradient:

    grad_cv = grad_reinmax + η * (gv_GR − gv_GS)

where
    gv_GS = J_GS · grad_z    (Gumbel-Softmax Jacobian at the specific noise G)
    gv_GR = J_GR · grad_z    (Rao-Gumbel expected Jacobian, R samples)

The idea: J_GR is a lower-variance estimate of the same conditional expectation
as J_GS, so replacing one with the other should reduce variance.

Observed behaviour (from ablation experiments)
    • At every (N, τ, K) setting, reinmax_cv INCREASES variance vs reinmax
    • The η ablation shows variance rising monotonically with η
    • Bias also increases with η (unexpected for a zero-mean CV)

Root cause
----------
J_GS and J_GR are NOT estimating the same quantity.

J_GS is computed as:
    p_gs = softmax((new_logits + G) / τ)     G ~ Gumbel for ORIGINAL logits
    J_GS = J_softmax(p_gs) / τ

J_GR is computed via Rao-Gumbel sampling conditioned on z:
    G' ~ Rao-Gumbel conditional for NEW_LOGITS = log((p + z)/2)
    J_GR = E[J_softmax(softmax((new_logits + G') / τ))]

G was drawn for the original distribution softmax(logits).
G' (used by J_GR) is drawn for the shifted distribution softmax(new_logits).

Since new_logits ≠ logits, the Rao-Gumbel conditional distributions differ:
    G | {argmax(logits + G) = z}        ≠   G' | {argmax(new_logits + G') = z}

This mismatch means E[gv_GS] ≠ E[gv_GR], so the correction η*(gv_GR − gv_GS)
has a non-zero mean → the CV introduces bias, not just variance.

Sub-experiments
---------------
14a  Correction mean
     Fix logits and z; run many Rao-Gumbel samples to estimate
     E[gv_GR | z]  and  E[gv_GS | z]  separately.
     Shows they are NOT equal (bias of correction ≠ 0).

14b  Variance decomposition across (z, G) samples
     Collect paired (grad_rm, correction) samples and decompose:
         Var(grad_cv) = Var(grad_rm) + Var(correction) + 2·Cov(grad_rm, correction)
     Shows Var(correction) >> |2·Cov|, so η > 0 always increases variance.

14c  Correlation between correction and gradient error
     corr(error_rm, correction) per expert — shows it is near-zero.
     The correction is adding uncorrelated noise, not reducing the main error.

14d  Optimal η sweep
     Computes η* = −Cov(grad_rm, correction) / Var(correction) empirically.
     Shows η* ≈ 0 and that even optimal η gives minimal improvement.

14e  Verdict: what actually works?
     Side-by-side comparison (reinmax, reinmax_v3, reinmax_cv at η=1, η=η*).
     Shows reinmax_v3 is the correct alternative: it directly replaces the
     analytical Jacobian in term1 with the Rao-Gumbel estimate, avoiding the
     distribution mismatch entirely.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sparse_mixer.estimators import (
    reinmax_single, reinmax_v3_single, reinmax_cv_single,
    _sample_gumbel, _softmax_jacobian, _rao_gumbel_jacobian,
)
from sparse_mixer.objectives import QuadraticObjective
from sparse_mixer.metrics import collect_grad_samples, bias_variance_metrics
from sparse_mixer.plotting import COLOURS, save_fig

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
CV_DIR      = os.path.join(RESULTS_DIR, 'cv_diagnosis')
os.makedirs(CV_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Core paired sampler — same (z, G) for both rm and cv components
# ─────────────────────────────────────────────────────────────────────────────

def _grad_z_for_obj(z: torch.Tensor, obj: QuadraticObjective) -> torch.Tensor:
    """
    Compute dL/dz for L = obj(z).sum(), without autograd, for QuadraticObjective.
    L = sum_b [z_b @ A * z_b + z_b · b]
    dL/dz_bi = ((A + A^T) z_b)_i + b_i
    Returns (B, N).
    """
    AT = obj.A + obj.A.t()   # (N, N)
    return z @ AT + obj.b    # (B, N)


@torch.no_grad()
def paired_rm_and_correction(
    logits: torch.Tensor,
    obj: QuadraticObjective,
    tau: float,
    repeats: int,
    n_samples: int,
    seed: int = 0,
):
    """
    Collect n_samples of (grad_rm, correction) sharing the SAME (z, G) per sample.

    Parameters
    ----------
    logits  : (B, N)  fixed router logits
    obj     : QuadraticObjective
    tau     : temperature for CV Jacobian
    repeats : R for Rao-Gumbel Jacobian (J_GR)
    n_samples : number of Monte-Carlo draws

    Returns
    -------
    grads_rm     : (n_samples, B, N)
    corrections  : (n_samples, B, N)  η=1 correction = gv_GR − gv_GS
    gv_gs_all    : (n_samples, B, N)  single-sample GS Jacobian component
    gv_gr_all    : (n_samples, B, N)  Rao-Gumbel Jacobian component
    z_all        : (n_samples, B, N)  selected masks
    """
    B, N = logits.shape
    p    = F.softmax(logits, dim=-1)

    grads_rm, corrections = [], []
    gv_gs_all, gv_gr_all, z_all = [], [], []

    torch.manual_seed(seed)
    for _ in range(n_samples):
        # ── Sample z from Gumbel-argmax ───────────────────────────────────
        G     = _sample_gumbel(logits.shape)
        idx   = (logits + G).argmax(dim=-1, keepdim=True)
        z     = torch.zeros_like(logits).scatter_(-1, idx, 1.0)   # (B, N)

        # ── Upstream gradient for the objective ───────────────────────────
        gz = _grad_z_for_obj(z, obj)   # (B, N)

        # ── ReinMax term0 + term1 ─────────────────────────────────────────
        shifted = 0.5 * (p + z)
        g1 = 2.0 * gz * shifted
        g1 = g1 - shifted * g1.sum(-1, keepdim=True)
        g0 = -0.5 * gz * p          # grad_p ≈ 0 for this objective
        g0 = g0 - p * g0.sum(-1, keepdim=True)
        grad_rm = g0 + g1
        grad_rm = grad_rm - grad_rm.mean(-1, keepdim=True)

        # ── CV Jacobian components ─────────────────────────────────────────
        new_pi     = 0.5 * (p + z)
        new_logits = new_pi.log()

        # J_GS: uses the SAVED G (for original logits, not new_logits)
        p_gs  = F.softmax((new_logits + G) / tau, dim=-1)
        J_gs  = _softmax_jacobian(p_gs) / tau          # (B, N, N)
        gv_gs = torch.bmm(J_gs, gz.unsqueeze(-1)).squeeze(-1)   # (B, N)

        # J_GR: Rao-Gumbel samples conditioned on z, for new_logits
        J_gr  = _rao_gumbel_jacobian(new_logits, z, tau, repeats)  # (B, N, N)
        gv_gr = torch.bmm(J_gr, gz.unsqueeze(-1)).squeeze(-1)       # (B, N)

        correction = gv_gr - gv_gs   # η=1 contribution

        grads_rm.append(grad_rm)
        corrections.append(correction)
        gv_gs_all.append(gv_gs)
        gv_gr_all.append(gv_gr)
        z_all.append(z)

    return (
        torch.stack(grads_rm,    0),   # (S, B, N)
        torch.stack(corrections, 0),   # (S, B, N)
        torch.stack(gv_gs_all,   0),   # (S, B, N)
        torch.stack(gv_gr_all,   0),   # (S, B, N)
        torch.stack(z_all,       0),   # (S, B, N)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Exp 14a — Correction mean: bias check
# ─────────────────────────────────────────────────────────────────────────────

def run_correction_mean(
    n_experts: int = 8,
    batch_size: int = 4,
    tau: float = 1.0,
    repeats_gr: int = 200,   # large R to get accurate E[gv_GR | z]
    n_z_samples: int = 1000,
    n_fix_z: int = 10,       # number of fixed z values to examine
    seed: int = 42,
):
    """
    Exp 14a: for a fixed z, estimate E[gv_GR | z] and E[gv_GS | z] separately.

    If the correction is truly zero-mean, these should be equal.
    """
    print(f"\n{'='*60}")
    print(f"Exp 14a: Correction mean (is E[gv_GR | z] = E[gv_GS | z]?)")
    print(f"  N={n_experts}  τ={tau}  R_gr={repeats_gr}  n_z={n_z_samples}")
    print('='*60)

    torch.manual_seed(seed)
    logits = torch.randn(1, n_experts)       # batch_size=1 for clarity
    obj    = QuadraticObjective(n_experts, seed=seed)
    p      = F.softmax(logits, dim=-1)

    mismatches = []

    # Draw n_fix_z different z values; for each, average over many G samples
    torch.manual_seed(seed + 1)
    for z_trial in range(n_fix_z):
        # Pick a random z
        idx = torch.multinomial(p, 1)
        z   = torch.zeros_like(logits).scatter_(-1, idx, 1.0)
        gz  = _grad_z_for_obj(z, obj)    # (1, N)

        new_pi     = 0.5 * (p + z)
        new_logits = new_pi.log()

        # E[gv_GR | z]: average R=repeats_gr samples from Rao-Gumbel for new_logits
        gv_gr_list = []
        for _ in range(100):
            J_gr  = _rao_gumbel_jacobian(new_logits, z, tau, repeats_gr)
            gv_gr = torch.bmm(J_gr, gz.unsqueeze(-1)).squeeze(-1)
            gv_gr_list.append(gv_gr)
        E_gv_gr = torch.stack(gv_gr_list).mean(0)    # (1, N)

        # E[gv_GS | z]: average over n_z_samples draws of G conditioned on z
        # G | {argmax(logits + G) = z} follows the Rao-Gumbel conditional for logits
        # We approximate by rejection sampling: draw G, keep if it selects z
        gv_gs_list = []
        kept = 0
        torch.manual_seed(seed + 100 * z_trial)
        while kept < 200:
            G_try = _sample_gumbel(logits.shape)
            if (logits + G_try).argmax(-1).item() == idx.item():
                p_gs  = F.softmax((new_logits + G_try) / tau, dim=-1)
                J_gs  = _softmax_jacobian(p_gs) / tau
                gv_gs = torch.bmm(J_gs, gz.unsqueeze(-1)).squeeze(-1)
                gv_gs_list.append(gv_gs)
                kept += 1
        E_gv_gs = torch.stack(gv_gs_list).mean(0)   # (1, N)

        # Mismatch: L2 distance
        diff = (E_gv_gr - E_gv_gs).norm().item()
        rel  = diff / (E_gv_gs.norm().item() + 1e-8)
        mismatches.append(dict(diff=diff, rel=rel,
                               E_gv_gr=E_gv_gr.squeeze(),
                               E_gv_gs=E_gv_gs.squeeze()))
        print(f"  z={z.argmax().item():2d}  |E[gv_GR]-E[gv_GS]|={diff:.5f}  "
              f"relative={rel:.3f}")

    mean_diff = np.mean([m['diff'] for m in mismatches])
    mean_rel  = np.mean([m['rel']  for m in mismatches])
    print(f"\n  Mean |E[gv_GR]-E[gv_GS]|: {mean_diff:.5f}  ({mean_rel:.1%} relative)")
    print(f"  → {'BIASED: correction has non-zero mean' if mean_rel > 0.01 else 'unbiased'}")
    return mismatches


# ─────────────────────────────────────────────────────────────────────────────
# Exp 14b — Variance decomposition
# ─────────────────────────────────────────────────────────────────────────────

def run_variance_decomposition(
    n_experts: int = 8,
    batch_size: int = 4,
    tau: float = 1.0,
    repeats: int = 50,
    n_samples: int = 2000,
    seed: int = 42,
):
    """
    Exp 14b: Var(grad_cv) = Var(grad_rm) + Var(correction) + 2·Cov.
    Shows Var(correction) >> |2·Cov|.
    """
    print(f"\n{'='*60}")
    print(f"Exp 14b: Variance decomposition  (N={n_experts}, τ={tau})")
    print('='*60)

    torch.manual_seed(seed)
    logits = torch.randn(batch_size, n_experts) * 1.5
    obj    = QuadraticObjective(n_experts, seed=seed)

    grads_rm, corrections, _, _, _ = paired_rm_and_correction(
        logits, obj, tau, repeats, n_samples, seed=seed,
    )
    # (S, B, N) → flatten B*N into a single "dimension"
    G_rm  = grads_rm.reshape(n_samples, -1)     # (S, D)
    G_corr = corrections.reshape(n_samples, -1)  # (S, D)
    G_cv  = G_rm + G_corr                        # (S, D)

    var_rm   = G_rm.var(0).mean().item()
    var_corr = G_corr.var(0).mean().item()
    # Cov(rm, correction) per element, then average
    cov = ((G_rm - G_rm.mean(0)) * (G_corr - G_corr.mean(0))).mean(0).mean().item()
    var_cv   = G_cv.var(0).mean().item()
    var_cv_pred = var_rm + var_corr + 2 * cov

    print(f"\n  Var(grad_rm)   = {var_rm:.5f}  (baseline)")
    print(f"  Var(correction)= {var_corr:.5f}  (noise added by η=1 CV)")
    print(f"  Cov            = {cov:.5f}")
    print(f"  2·Cov          = {2*cov:.5f}")
    print(f"  Var(grad_cv)   = {var_cv:.5f}  (measured)")
    print(f"  Var(grad_cv)   = {var_cv_pred:.5f}  (predicted = rm + corr + 2·cov)")
    print(f"\n  Net change Var(cv) - Var(rm) = {var_cv - var_rm:+.5f}")
    if var_cv > var_rm:
        print(f"  → CV INCREASES variance by {100*(var_cv/var_rm-1):.1f}%")
    else:
        print(f"  → CV reduces variance by {100*(1-var_cv/var_rm):.1f}%")
    print(f"\n  Var(correction) / |2·Cov| = {var_corr / (abs(2*cov)+1e-12):.1f}  "
          f"(>1 means correction noise dominates)")

    return dict(var_rm=var_rm, var_corr=var_corr, cov=cov, var_cv=var_cv,
                G_rm=G_rm, G_corr=G_corr)


# ─────────────────────────────────────────────────────────────────────────────
# Exp 14c — Correlation between correction and gradient error
# ─────────────────────────────────────────────────────────────────────────────

def run_correlation_analysis(
    n_experts: int = 8,
    batch_size: int = 4,
    tau: float = 1.0,
    repeats: int = 50,
    n_samples: int = 2000,
    seed: int = 42,
):
    """
    Exp 14c: corr(error_rm, correction) — should be negative for a useful CV.
    """
    print(f"\n{'='*60}")
    print(f"Exp 14c: Correlation (error_rm, correction)  (N={n_experts})")
    print('='*60)

    torch.manual_seed(seed)
    logits = torch.randn(batch_size, n_experts) * 1.5
    obj    = QuadraticObjective(n_experts, seed=seed)
    true_grad = obj.exact_gradient(logits)     # (B, N)

    grads_rm, corrections, _, _, _ = paired_rm_and_correction(
        logits, obj, tau, repeats, n_samples, seed=seed,
    )
    # (S, B, N)
    errors = grads_rm - true_grad.unsqueeze(0)  # (S, B, N)

    # Per-element Pearson correlation across samples
    E = errors.reshape(n_samples, -1)       # (S, D)
    C = corrections.reshape(n_samples, -1)  # (S, D)
    E_c = E - E.mean(0, keepdim=True)
    C_c = C - C.mean(0, keepdim=True)
    corr_per_dim = (E_c * C_c).mean(0) / (E_c.std(0) * C_c.std(0) + 1e-8)  # (D,)
    mean_corr = corr_per_dim.mean().item()

    print(f"\n  Per-element corr(error, correction):")
    print(f"    Mean   = {mean_corr:.4f}")
    print(f"    Std    = {corr_per_dim.std().item():.4f}")
    print(f"    Min    = {corr_per_dim.min().item():.4f}")
    print(f"    Max    = {corr_per_dim.max().item():.4f}")

    if mean_corr < -0.1:
        verdict = "USEFUL: negative correlation means CV reduces error"
    elif mean_corr < 0.05:
        verdict = "USELESS: near-zero correlation, correction adds noise without benefit"
    else:
        verdict = "HARMFUL: positive correlation means CV adds correlated noise"
    print(f"\n  Verdict: {verdict}")
    return corr_per_dim.numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Exp 14d — Optimal η sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_optimal_eta(
    n_experts: int = 8,
    batch_size: int = 4,
    tau: float = 1.0,
    repeats: int = 50,
    n_samples: int = 2000,
    eta_values: list = None,
    seed: int = 42,
):
    """
    Exp 14d: variance vs η, and the empirically optimal η*.
    """
    print(f"\n{'='*60}")
    print(f"Exp 14d: Optimal η  (N={n_experts}, τ={tau})")
    print('='*60)

    if eta_values is None:
        eta_values = [-0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]

    torch.manual_seed(seed)
    logits = torch.randn(batch_size, n_experts) * 1.5
    obj    = QuadraticObjective(n_experts, seed=seed)

    grads_rm, corrections, _, _, _ = paired_rm_and_correction(
        logits, obj, tau, repeats, n_samples, seed=seed,
    )
    G_rm  = grads_rm.reshape(n_samples, -1)
    G_corr = corrections.reshape(n_samples, -1)

    var_rm   = G_rm.var(0).mean().item()
    var_corr = G_corr.var(0).mean().item()
    cov      = ((G_rm - G_rm.mean(0)) * (G_corr - G_corr.mean(0))).mean(0).mean().item()

    # Optimal η: minimise Var(rm + η·correction) = var_rm + η²·var_corr + 2η·cov
    # d/dη = 0 → η* = −cov / var_corr
    eta_star = -cov / (var_corr + 1e-12)
    var_star = var_rm + eta_star**2 * var_corr + 2 * eta_star * cov
    print(f"\n  Optimal η*  = {eta_star:.4f}")
    print(f"  Var at η=0  = {var_rm:.5f}  (plain reinmax)")
    print(f"  Var at η*   = {var_star:.5f}  (best achievable with this CV)")
    print(f"  Var at η=1  = {var_rm + var_corr + 2*cov:.5f}  (current default)")

    # Var for each η value
    var_per_eta = {}
    print(f"\n  Variance vs η:")
    for eta in eta_values:
        G_cv  = G_rm + eta * G_corr
        v     = G_cv.var(0).mean().item()
        var_per_eta[eta] = v
        marker = " ← optimal" if abs(eta - eta_star) < 0.05 else ""
        print(f"    η={eta:+.2f}  var={v:.5f}{marker}")

    return dict(eta_star=eta_star, var_rm=var_rm, var_star=var_star,
                var_per_eta=var_per_eta, cov=cov, var_corr=var_corr)


# ─────────────────────────────────────────────────────────────────────────────
# Exp 14e — Verdict: methods head-to-head
# ─────────────────────────────────────────────────────────────────────────────

def run_verdict(
    n_experts: int = 8,
    batch_size: int = 4,
    tau: float = 1.0,
    repeats: int = 50,
    n_samples: int = 1000,
    eta_star: float = 0.0,
    seed: int = 42,
):
    """
    Exp 14e: side-by-side bias/var/mse for reinmax, reinmax_v3, reinmax_cv(η=1),
    reinmax_cv(η=η*).
    """
    print(f"\n{'='*60}")
    print(f"Exp 14e: Verdict — bias/var/mse comparison  (N={n_experts})")
    print('='*60)

    torch.manual_seed(seed)
    logits   = torch.randn(batch_size, n_experts) * 1.5
    obj      = QuadraticObjective(n_experts, seed=seed)
    exact    = obj.exact_gradient(logits)

    methods = {
        'reinmax'        : lambda l: reinmax_single(l, tau),
        'reinmax_v3'     : lambda l: reinmax_v3_single(l, tau, repeats),
        'reinmax_cv(η=1)': lambda l: reinmax_cv_single(l, tau, eta=1.0, repeats=repeats),
        f'reinmax_cv(η={eta_star:.2f})': lambda l: reinmax_cv_single(l, tau, eta=float(eta_star), repeats=repeats),
    }

    results = {}
    print(f"\n  {'Method':25s}  {'Bias':>8}  {'Var':>8}  {'MSE':>8}")
    print(f"  {'-'*55}")
    for name, fn in methods.items():
        grads = collect_grad_samples(fn, logits, obj, n_samples)
        m     = bias_variance_metrics(grads, exact)
        results[name] = m
        print(f"  {name:25s}  {m['bias']:>8.4f}  {m['variance']:>8.4f}  {m['mse']:>8.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_cv_diagnosis(decomp, corr_vals, eta_results, verdict,
                      save_dir=CV_DIR):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    # ── Panel 1: Variance decomposition ──────────────────────────────────────
    ax = axes[0]
    labels = ['Var(rm)', 'Var(correction)', '2·Cov', 'Net Var(cv)']
    var_rm   = decomp['var_rm']
    var_corr = decomp['var_corr']
    cov2     = 2 * decomp['cov']
    var_cv   = decomp['var_cv']
    values   = [var_rm, var_corr, cov2, var_cv]
    bar_colours = ['#4878D0', '#EE854A', '#6ACC65', '#D65F5F']
    bars = ax.bar(labels, values, color=bar_colours)
    ax.axhline(var_rm, color='#4878D0', ls='--', lw=1.5, alpha=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, max(val, 0),
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Variance'); ax.set_title('Variance Decomposition\nη=1 correction')
    ax.tick_params(axis='x', labelrotation=20)
    ax.grid(True, axis='y', alpha=0.3)

    # ── Panel 2: Correlation histogram ───────────────────────────────────────
    ax = axes[1]
    ax.hist(corr_vals, bins=20, color='#956CB4', edgecolor='white', lw=0.5)
    ax.axvline(0,                       color='black', lw=1.5, ls='--')
    ax.axvline(float(corr_vals.mean()), color='red',   lw=2,   ls='-',
               label=f'mean={float(corr_vals.mean()):.3f}')
    ax.set_xlabel('corr(error, correction)')
    ax.set_ylabel('Count (over experts × batch)')
    ax.set_title('Correlation:\nerror_rm vs correction')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ── Panel 3: Variance vs η ───────────────────────────────────────────────
    ax = axes[2]
    etas = sorted(eta_results['var_per_eta'].keys())
    vars_ = [eta_results['var_per_eta'][e] for e in etas]
    ax.plot(etas, vars_, color='#D65F5F', lw=2, marker='o', ms=5)
    ax.axhline(eta_results['var_rm'],  color='#4878D0', ls='--', lw=1.5,
               label='reinmax (η=0)')
    ax.axvline(eta_results['eta_star'], color='green', ls=':', lw=1.5,
               label=f"η*={eta_results['eta_star']:.3f}")
    ax.set_xlabel('η'); ax.set_ylabel('Var(grad_cv)')
    ax.set_title('Variance vs η\n(parabola in η)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ── Panel 4: Verdict bar chart ────────────────────────────────────────────
    ax = axes[3]
    names = list(verdict.keys())
    vars_v = [verdict[n]['variance'] for n in names]
    short  = ['rm', 'v3', 'cv(1)', f"cv({eta_results['eta_star']:.2f})"]
    bar_c  = [COLOURS.get('reinmax', '#4878D0'),
               COLOURS.get('reinmax_v3', '#EE854A'),
               '#D65F5F', '#6ACC65']
    bars = ax.bar(short, vars_v, color=bar_c)
    for bar, val in zip(bars, vars_v):
        ax.text(bar.get_x() + bar.get_width()/2, val,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Variance'); ax.set_title('Verdict: Variance\n(lower = better)')
    ax.grid(True, axis='y', alpha=0.3)

    plt.suptitle('ReinMax-CV Diagnosis: Why the Control Variate Underperforms',
                 fontsize=12)
    save_fig(os.path.join(save_dir, 'cv_diagnosis.png'))


def plot_cv_diagnosis_gv(gv_gs_samples, gv_gr_samples, save_dir=CV_DIR):
    """Scatter E[gv_GS] vs E[gv_GR] per expert — should be equal if no mismatch."""
    mean_gs = gv_gs_samples.reshape(-1, gv_gs_samples.shape[-1]).mean(0).numpy()
    mean_gr = gv_gr_samples.reshape(-1, gv_gr_samples.shape[-1]).mean(0).numpy()
    N = len(mean_gs)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.scatter(mean_gs, mean_gr, c=range(N), cmap='tab10', s=80, zorder=3)
    lim = max(abs(mean_gs).max(), abs(mean_gr).max()) * 1.2
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=1.5, label='y=x (no mismatch)')
    for i in range(N):
        ax.annotate(f'e{i}', (mean_gs[i], mean_gr[i]),
                    xytext=(3, 3), textcoords='offset points', fontsize=7)
    ax.set_xlabel('E[gv_GS | z]  (using G from original logits)')
    ax.set_ylabel('E[gv_GR | z]  (Rao-Gumbel for new_logits)')
    ax.set_title('Distribution mismatch:\nE[gv_GS] vs E[gv_GR] per expert')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1]
    diff = mean_gr - mean_gs
    ax.bar(range(N), diff, color=['#D65F5F' if d > 0 else '#4878D0' for d in diff])
    ax.axhline(0, color='black', lw=1)
    ax.set_xlabel('expert index'); ax.set_ylabel('E[gv_GR] - E[gv_GS]')
    ax.set_title('Correction mean per expert\n(zero-mean CV would show all zeros)')
    ax.set_xticks(range(N))
    ax.grid(True, axis='y', alpha=0.3)

    plt.suptitle('The Distribution Mismatch: G is from the Wrong Conditional',
                 fontsize=11)
    save_fig(os.path.join(save_dir, 'cv_mismatch.png'))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(fast: bool = False):
    n_samples = 500  if fast else 2000
    repeats   = 20   if fast else 50
    n_fix_z   = 4    if fast else 10
    n_experts = 8
    batch     = 4
    tau       = 1.0
    seed      = 42

    summary = {}

    # ── Exp 14a ──────────────────────────────────────────────────────────────
    mismatches = run_correction_mean(
        n_experts=n_experts, batch_size=batch, tau=tau,
        repeats_gr=repeats*4, n_z_samples=n_samples,
        n_fix_z=n_fix_z, seed=seed,
    )
    summary['mean_mismatch'] = {
        'mean_diff': float(np.mean([m['diff'] for m in mismatches])),
        'mean_rel':  float(np.mean([m['rel']  for m in mismatches])),
    }

    # ── Exp 14b ──────────────────────────────────────────────────────────────
    decomp = run_variance_decomposition(
        n_experts=n_experts, batch_size=batch, tau=tau,
        repeats=repeats, n_samples=n_samples, seed=seed,
    )
    summary['variance_decomp'] = {
        k: v for k, v in decomp.items()
        if not isinstance(v, torch.Tensor)
    }

    # ── Exp 14c ──────────────────────────────────────────────────────────────
    corr_vals = run_correlation_analysis(
        n_experts=n_experts, batch_size=batch, tau=tau,
        repeats=repeats, n_samples=n_samples, seed=seed,
    )
    summary['correlation'] = {
        'mean': float(corr_vals.mean()),
        'std':  float(corr_vals.std()),
    }

    # ── Exp 14d ──────────────────────────────────────────────────────────────
    eta_results = run_optimal_eta(
        n_experts=n_experts, batch_size=batch, tau=tau,
        repeats=repeats, n_samples=n_samples, seed=seed,
    )
    summary['optimal_eta'] = {
        'eta_star': eta_results['eta_star'],
        'var_at_eta_star': eta_results['var_star'],
        'var_at_eta_0':    eta_results['var_rm'],
        'var_at_eta_1':    eta_results['var_per_eta'].get(1.0),
    }

    # ── Exp 14e ──────────────────────────────────────────────────────────────
    verdict = run_verdict(
        n_experts=n_experts, batch_size=batch, tau=tau,
        repeats=repeats, n_samples=n_samples,
        eta_star=eta_results['eta_star'], seed=seed,
    )
    summary['verdict'] = {
        n: {k: v for k, v in m.items() if isinstance(v, float)}
        for n, m in verdict.items()
    }

    # ── Collect gv_gs vs gv_gr for mismatch plot ──────────────────────────────
    torch.manual_seed(seed)
    logits = torch.randn(batch, n_experts) * 1.5
    obj    = QuadraticObjective(n_experts, seed=seed)
    _, _, gv_gs_s, gv_gr_s, _ = paired_rm_and_correction(
        logits, obj, tau, repeats, n_samples, seed=seed,
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_cv_diagnosis(decomp, corr_vals, eta_results, verdict)
    plot_cv_diagnosis_gv(gv_gs_s, gv_gr_s)

    path = os.path.join(CV_DIR, 'cv_diagnosis_summary.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nCV diagnosis summary → {path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("CV DIAGNOSIS SUMMARY")
    print('='*60)
    print(f"\n14a — Correction mean (relative mismatch):")
    print(f"  E[gv_GR] ≠ E[gv_GS] by {summary['mean_mismatch']['mean_rel']:.1%} on average")
    print(f"  → CV introduces bias proportional to η × {summary['mean_mismatch']['mean_diff']:.5f}")
    print(f"\n14b — Variance decomposition:")
    vd = summary['variance_decomp']
    print(f"  Var(rm)        = {vd['var_rm']:.5f}")
    print(f"  Var(correction)= {vd['var_corr']:.5f}  ({vd['var_corr']/vd['var_rm']:.1f}× rm)")
    print(f"  2·Cov          = {2*vd['cov']:.5f}")
    print(f"  Var(cv at η=1) = {vd['var_cv']:.5f}  ({vd['var_cv']/vd['var_rm']:.1f}× rm)")
    print(f"\n14c — Correlation(error, correction):")
    print(f"  Mean = {summary['correlation']['mean']:.4f}  (0=useless, <0=helpful, >0=harmful)")
    print(f"\n14d — Optimal η:")
    oe = summary['optimal_eta']
    print(f"  η* = {oe['eta_star']:.4f}")
    print(f"  Var(rm at η=0) = {oe['var_at_eta_0']:.5f}")
    print(f"  Var(cv at η* ) = {oe['var_at_eta_star']:.5f}  ({100*(oe['var_at_eta_star']/oe['var_at_eta_0']-1):+.1f}%)")
    print(f"\n14e — Verdict (bias / var / mse):")
    print(f"  {'Method':25s}  {'Bias':>8}  {'Var':>8}  {'MSE':>8}")
    for name, m in verdict.items():
        print(f"  {name:25s}  {m['bias']:>8.4f}  {m['variance']:>8.4f}  {m['mse']:>8.4f}")
    print(f"\nRoot cause: J_GS uses G ~ Rao-Gumbel(logits) as if it were")
    print(f"            G ~ Rao-Gumbel(new_logits).  Because logits ≠ new_logits,")
    print(f"            E[gv_GS] ≠ E[gv_GR] and the 'zero-mean' assumption fails.")
    print(f"\nFix: ReinMax-v3 avoids this entirely by replacing the analytical")
    print(f"     Jacobian in term1 directly with J_GR (no J_GS involved).")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--fast', action='store_true')
    args = p.parse_args()
    main(fast=args.fast)
