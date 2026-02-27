"""
Objective functions used across SparseMixer experiments.

QuadraticObjective
  Simple f(z) = b^T z + z^T A z with closed-form REINFORCE gradient.
  Used for bias/variance experiments (K=1 and K>1).

SwitchingLinearTask
  Harder experiment: N experts, C classes, each class has a fixed target
  linear map W_c.  The model must learn *which* expert handles which class
  (router specialization) while each expert learns its assigned linear map.

  This is much harder than QuadraticObjective because:
    • Routing signal only emerges once experts start to diverge.
    • High gradient variance → noisy routing updates → dead-expert collapse.
    • Low-variance estimators (ReinMax-v3) allow routing to stabilise earlier
      and reach lower reconstruction loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# QuadraticObjective
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
        return self(I)

    def exact_gradient(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Analytic gradient d/d_theta E[f(z)]  where p = softmax(theta).
        exact = p * (f_vals - E_p[f])
        Returns (B, N).
        """
        p      = F.softmax(logits, dim=-1)
        f_vals = self.f_at_experts().to(logits.device)
        E_f    = (p * f_vals).sum(-1, keepdim=True)
        return p * (f_vals - E_f)


# ─────────────────────────────────────────────────────────────────────────────
# SwitchingLinearTask
# ─────────────────────────────────────────────────────────────────────────────

class SwitchingLinearTask:
    """
    Expert-specialization task: N experts, C classes.

    Each class c has a fixed target linear map W_c ∈ R^{d_out × d_in}.
    The task is:
        Given input x ∈ R^{d_in} with label y ∈ {0,...,C-1},
        produce ŷ = expert_z(x) ≈ W_y x.

    The router z is discrete: one-hot over N experts.
    Experts are shared learned linear maps {M_0,...,M_{N-1}}.

    Gradient challenge:
        ∂L/∂router_logits  requires flowing gradients through the discrete z.
        High estimator variance → noisy routing signal → dead-expert collapse.
        Low variance → smooth gradient → experts specialise and routing stabilises.

    Parameters
    ----------
    n_experts : number of experts (N ≥ C)
    n_classes : number of ground-truth classes (C)
    d_in      : input dimension
    d_out     : output dimension
    seed      : RNG seed for target maps W_c
    """

    def __init__(
        self,
        n_experts: int = 4,
        n_classes: int = 4,
        d_in: int = 8,
        d_out: int = 8,
        seed: int = 0,
    ):
        assert n_experts >= n_classes, "Need at least as many experts as classes"
        self.n_experts = n_experts
        self.n_classes = n_classes
        self.d_in  = d_in
        self.d_out = d_out

        g = torch.Generator().manual_seed(seed)
        # Fixed target maps — one per class (not trainable)
        self.W = torch.randn(n_classes, d_out, d_in, generator=g) / d_in ** 0.5

    def sample_batch(
        self,
        batch_size: int,
        device: torch.device | str = 'cpu',
        seed: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample (x, label, target).

        Returns
        -------
        x      : (B, d_in)   standard Gaussian inputs
        label  : (B,) int64  class labels
        target : (B, d_out)  W_label @ x
        """
        if seed is not None:
            torch.manual_seed(seed)
        W      = self.W.to(device)
        label  = torch.randint(0, self.n_classes, (batch_size,), device=device)
        x      = torch.randn(batch_size, self.d_in, device=device)
        target = torch.einsum('boi,bi->bo', W[label], x)   # (B, d_out)
        return x, label, target

    def reconstruction_loss(
        self,
        pred: torch.Tensor,   # (B, d_out)
        target: torch.Tensor, # (B, d_out)
    ) -> torch.Tensor:
        """Per-sample MSE, averaged over the batch."""
        return F.mse_loss(pred, target)

    def specialization_score(
        self,
        router_logits: torch.Tensor,  # (B, n_experts)
        labels: torch.Tensor,         # (B,) int64
    ) -> float:
        """
        Fraction of tokens routed to the 'correct' expert.

        'Correct' expert for class c = the expert that class-c tokens are
        most frequently routed to (plurality assignment, so this is
        permutation-invariant).

        Returns a float in [0, 1].
        """
        assigned = router_logits.argmax(dim=-1).cpu()   # (B,)
        labels   = labels.cpu()
        # Find plurality assignment for each class
        correct = 0
        total   = labels.numel()
        for c in range(self.n_classes):
            mask = (labels == c)
            if mask.sum() == 0:
                continue
            chosen = assigned[mask]
            # Expert that class c sends the most tokens to
            counts  = torch.bincount(chosen, minlength=self.n_experts)
            best_e  = counts.argmax().item()
            correct += (chosen == best_e).sum().item()
        return correct / total


# ─────────────────────────────────────────────────────────────────────────────
# ExpertLinearModel
# ─────────────────────────────────────────────────────────────────────────────

class ExpertLinearModel(nn.Module):
    """
    Lightweight MoE model for the SwitchingLinearTask:
      1. Router : Linear(d_in → n_experts) → discrete routing z
      2. Experts: n_experts independent Linear(d_in → d_out) maps
      3. Output : z-selected expert output  (K=1)

    The gradient estimator for the discrete routing is passed in as a
    callable `method_fn(logits) → (z, p)`.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_experts: int,
        method_fn,           # estimator callable
    ):
        super().__init__()
        self.router  = nn.Linear(d_in, n_experts, bias=False)
        self.experts = nn.ModuleList([nn.Linear(d_in, d_out, bias=False)
                                      for _ in range(n_experts)])
        self.method_fn = method_fn
        self.n_experts = n_experts

    def forward(self, x: torch.Tensor):
        """
        x : (B, d_in)
        Returns: pred (B, d_out), z (B, n_experts), logits (B, n_experts)
        """
        logits = self.router(x)                 # (B, n_experts)
        z, _   = self.method_fn(logits)         # (B, n_experts)

        # Stack expert outputs: (B, n_experts, d_out)
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)

        # Select via z (one-hot / soft): weighted sum → (B, d_out)
        pred = (z.unsqueeze(-1) * expert_outs).sum(dim=1)
        return pred, z, logits
