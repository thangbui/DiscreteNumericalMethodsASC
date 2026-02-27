"""
Shared measurement utilities for SparseMixer gradient estimator experiments.

Functions
---------
collect_grad_samples      : Monte-Carlo gradient samples for QuadraticObjective
bias_variance_metrics     : Bias / variance / MSE decomposition
measure_step_variance     : Per-step gradient variance during optimisation
routing_specialization    : Specialization score after training a linear MoE
"""

from __future__ import annotations

import time
from typing import Callable, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from sparse_mixer.objectives import QuadraticObjective, SwitchingLinearTask, ExpertLinearModel


# ─────────────────────────────────────────────────────────────────────────────
# Quadratic objective helpers
# ─────────────────────────────────────────────────────────────────────────────

def collect_grad_samples(
    method_fn: Callable,
    logits: torch.Tensor,
    obj: QuadraticObjective,
    n_samples: int,
) -> torch.Tensor:
    """
    Draw n_samples one-sample gradient estimates for a given method.

    Parameters
    ----------
    method_fn : callable(logits) → (z, p)
    logits    : (B, N)  fixed logits (detached; will be cloned per sample)
    obj       : QuadraticObjective
    n_samples : number of independent gradient estimates

    Returns
    -------
    grads : (n_samples, B, N)
    """
    grads = []
    for _ in range(n_samples):
        l = logits.detach().clone().requires_grad_(True)
        z, p = method_fn(l)
        loss = obj(z).sum()
        loss.backward()
        grads.append(l.grad.detach().clone())
    return torch.stack(grads, dim=0)


def bias_variance_metrics(
    grads: torch.Tensor,
    exact: torch.Tensor,
) -> Dict[str, Any]:
    """
    Bias / variance / MSE decomposition from Monte-Carlo gradient samples.

    Parameters
    ----------
    grads : (S, B, N)  — S one-sample gradient estimates
    exact : (B, N)     — ground-truth gradient

    Returns
    -------
    dict with keys:
        bias      : L2 norm of (mean_estimate − exact), scalar
        variance  : E[||estimate − mean_estimate||²], scalar
        mse       : E[||estimate − exact||²], scalar
        std       : mean per-element std, scalar
        mean_grad : (B, N)
        all_grads : (S, B, N)
    """
    mean_g = grads.mean(dim=0)
    bias   = (mean_g - exact).norm().item()
    var    = ((grads - mean_g) ** 2).sum(dim=-1).mean().item()
    mse    = ((grads - exact ) ** 2).sum(dim=-1).mean().item()
    std    = grads.std(dim=0).mean().item()
    return dict(bias=bias, variance=var, mse=mse, std=std,
                mean_grad=mean_g, all_grads=grads)


def scalar_bvm(
    logits: torch.Tensor,
    obj: QuadraticObjective,
    method_fn: Callable,
    n_samples: int,
) -> tuple:
    """Convenience wrapper → (bias, variance, mse)."""
    exact = obj.exact_gradient(logits)
    grads = collect_grad_samples(method_fn, logits, obj, n_samples)
    m     = bias_variance_metrics(grads, exact)
    return m['bias'], m['variance'], m['mse']


# ─────────────────────────────────────────────────────────────────────────────
# Gradient variance during optimisation
# ─────────────────────────────────────────────────────────────────────────────

def measure_step_variance(
    method_fn: Callable,
    obj: QuadraticObjective,
    n_experts: int,
    batch_size: int,
    n_steps: int,
    lr: float,
    samples_per_step: int,
    seed: int = 0,
) -> Dict[str, list]:
    """
    Gradient ascent on E[f(z)] tracking per-step gradient variance.

    Parameters
    ----------
    method_fn        : estimator callable
    obj              : QuadraticObjective
    n_experts        : N
    batch_size       : B
    n_steps          : training steps
    lr               : SGD learning rate
    samples_per_step : how many gradient samples to draw per step for variance est.

    Returns
    -------
    dict with:
        ef_traj  : list[float]  expected f value per step
        var_traj : list[float]  gradient variance per step
    """
    torch.manual_seed(seed)
    logits = nn.Parameter(torch.randn(batch_size, n_experts) * 0.1)
    opt    = torch.optim.SGD([logits], lr=lr)
    f_vals = obj.f_at_experts()

    ef_traj, var_traj = [], []

    for _ in range(n_steps):
        # Gradient variance estimate (samples_per_step draws)
        with torch.no_grad():
            l_fixed = logits.detach().clone()
        grads_list = []
        for _ in range(samples_per_step):
            l = l_fixed.clone().requires_grad_(True)
            z, _ = method_fn(l)
            loss = -obj(z).sum()
            loss.backward()
            grads_list.append(l.grad.detach())
        g_stack = torch.stack(grads_list, 0)   # (S, B, N)
        var = ((g_stack - g_stack.mean(0)) ** 2).sum(-1).mean().item()
        var_traj.append(var)

        # One optimisation step
        opt.zero_grad()
        z, _ = method_fn(logits)
        loss  = -obj(z).mean()
        loss.backward()
        opt.step()

        with torch.no_grad():
            p  = F.softmax(logits, dim=-1)
            ef = (p * f_vals.to(p.device)).sum(-1).mean().item()
        ef_traj.append(ef)

    return dict(ef=ef_traj, var=var_traj)


# ─────────────────────────────────────────────────────────────────────────────
# Expert specialization training
# ─────────────────────────────────────────────────────────────────────────────

def train_specialization(
    method_fn: Callable,
    task: SwitchingLinearTask,
    n_steps: int = 2000,
    batch_size: int = 256,
    lr: float = 3e-3,
    seed: int = 0,
    log_every: int = 200,
    device: str = 'cpu',
) -> Dict[str, list]:
    """
    Train an ExpertLinearModel on SwitchingLinearTask and record metrics.

    Parameters
    ----------
    method_fn  : estimator callable(logits) → (z, p)
    task       : SwitchingLinearTask
    n_steps    : optimisation steps
    batch_size : training batch size
    lr         : Adam learning rate
    seed       : RNG seed
    log_every  : print interval

    Returns
    -------
    dict with:
        loss_traj          : list[float]  training MSE per step
        spec_traj          : list[float]  specialization score per log_every steps
        spec_steps         : list[int]    step indices for spec_traj
        final_loss         : float
        final_spec         : float
    """
    torch.manual_seed(seed)
    model = ExpertLinearModel(
        d_in=task.d_in,
        d_out=task.d_out,
        n_experts=task.n_experts,
        method_fn=method_fn,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    loss_traj, spec_traj, spec_steps = [], [], []

    for step in range(n_steps):
        x, label, target = task.sample_batch(batch_size, device=device,
                                             seed=seed + step)
        opt.zero_grad()
        pred, z, logits = model(x)
        loss = task.reconstruction_loss(pred, target)
        loss.backward()
        opt.step()

        loss_traj.append(loss.item())

        if (step + 1) % log_every == 0 or step == n_steps - 1:
            with torch.no_grad():
                # Eval on a fresh batch
                x_e, label_e, target_e = task.sample_batch(
                    batch_size * 4, device=device, seed=seed + 10000 + step)
                pred_e, _, logits_e = model(x_e)
                eval_loss = task.reconstruction_loss(pred_e, target_e).item()
                spec = task.specialization_score(logits_e, label_e)
            spec_traj.append(spec)
            spec_steps.append(step + 1)

    final_loss = loss_traj[-1]
    final_spec = spec_traj[-1]
    return dict(
        loss_traj=loss_traj,
        spec_traj=spec_traj,
        spec_steps=spec_steps,
        final_loss=final_loss,
        final_spec=final_spec,
    )
