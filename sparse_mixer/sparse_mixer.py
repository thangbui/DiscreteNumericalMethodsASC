"""
SparseMixer: Sparse Mixture-of-Experts layer with discrete top-K routing.

Reference: Liu, Gao, Chen (2023) — SparseMixer / ReinMax

Architecture overview
---------------------
1. Router  : Linear(d_model → n_experts) → routing probabilities p = softmax(s)
2. Routing : Sample a discrete k-hot mask via one of:
               st / reinmax / reinmax_v3 / reinmax_cv / gumbel_softmax
3. Experts : n independent FFN experts
4. Output  : Weighted sum of expert outputs, weights = mask * p (renormalised)

Load-balancing auxiliary loss (Switch Transformer style):
   L_aux = n_experts * Σ_i  f_i * P_i
where f_i = fraction of tokens routed to expert i  (depends on discrete mask)
      P_i = mean router probability for expert i   (differentiable)

Shapes used throughout
----------------------
B = batch size
T = sequence length  (tokens per sample)
d = d_model
N = n_experts
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sparse_mixer.estimators import (
    st_single,
    reinmax_single,
    reinmax_v3_single,
    reinmax_cv_single,
    reinmax_topk,
    reinmax_v3_topk,
    reinmax_cv_topk,
    _gumbel_top_k,
)


# ─────────────────────────────────────────────────────────────────────────────
# Expert
# ─────────────────────────────────────────────────────────────────────────────

class Expert(nn.Module):
    """Two-layer MLP expert with GELU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


# ─────────────────────────────────────────────────────────────────────────────
# SparseMixer layer
# ─────────────────────────────────────────────────────────────────────────────

class SparseMixerLayer(nn.Module):
    """
    Sparse Mixture-of-Experts routing layer.

    Parameters
    ----------
    d_model   : token dimension
    n_experts : total number of experts  (N)
    k         : number of experts selected per token  (K)
    d_ff      : expert hidden dimension  (default 4 * d_model)
    dropout   : dropout inside each expert
    method    : gradient estimator — one of
                  'st', 'reinmax', 'reinmax_v3', 'reinmax_cv', 'gumbel_softmax'
    tau       : temperature for the estimator
    eta       : CV mixing coefficient (only used by reinmax_cv)
    alpha     : ReinMax α parameter
    repeats   : Monte-Carlo samples for Rao-Gumbel Jacobian
    """

    METHODS = ('st', 'reinmax', 'reinmax_v3', 'reinmax_cv', 'gumbel_softmax')

    def __init__(
        self,
        d_model: int,
        n_experts: int = 8,
        k: int = 1,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        method: str = 'reinmax',
        tau: float = 1.0,
        eta: float = 0.5,
        alpha: float = 1.0,
        repeats: int = 50,
    ):
        super().__init__()
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}, got '{method}'")

        self.d_model   = d_model
        self.n_experts = n_experts
        self.k         = k
        self.method    = method
        self.tau       = tau
        self.eta       = eta
        self.alpha     = alpha
        self.repeats   = repeats

        d_ff = d_ff or 4 * d_model

        self.router  = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(d_model, d_ff, dropout) for _ in range(n_experts)])
        self.register_buffer('expert_usage', torch.zeros(n_experts))

    # ── routing helpers ────────────────────────────────────────────────────

    def _route_k1(self, logits: torch.Tensor):
        """Single-expert selection (K=1)."""
        if self.method == 'st':
            return st_single(logits, self.tau)
        if self.method == 'reinmax':
            return reinmax_single(logits, self.tau, self.alpha)
        if self.method == 'reinmax_v3':
            return reinmax_v3_single(logits, self.tau, self.repeats)
        if self.method == 'reinmax_cv':
            return reinmax_cv_single(logits, self.tau, self.eta, self.repeats)
        if self.method == 'gumbel_softmax':
            z = F.gumbel_softmax(logits, tau=self.tau, hard=True)
            return z, F.softmax(logits, dim=-1)
        raise ValueError(self.method)

    def _route_topk(self, logits: torch.Tensor):
        """Top-K selection (K>1)."""
        if self.method == 'reinmax':
            return reinmax_topk(logits, self.k, self.tau)
        if self.method == 'reinmax_v3':
            return reinmax_v3_topk(logits, self.k, self.tau, self.repeats)
        if self.method == 'reinmax_cv':
            return reinmax_cv_topk(logits, self.k, self.tau, self.eta, self.repeats)
        if self.method in ('st', 'gumbel_softmax'):
            mask, _, _ = _gumbel_top_k(logits, self.k)
            p_soft = F.softmax(logits / self.tau, dim=-1)
            mask_st = mask - p_soft.detach() + p_soft       # ST trick
            return mask_st, F.softmax(logits, dim=-1)
        raise ValueError(self.method)

    # ── forward ───────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (B, T, d_model)

        Returns
        -------
        output      : (B, T, d_model)
        aux_loss    : scalar load-balancing loss
        router_probs: (B*T, N)  soft routing probabilities
        """
        B, T, d = x.shape
        x_flat = x.reshape(B * T, d)                         # (BT, d)

        router_logits = self.router(x_flat)                  # (BT, N)

        if self.k == 1:
            mask, router_probs = self._route_k1(router_logits)
        else:
            mask, router_probs = self._route_topk(router_logits)

        output   = self._expert_mix(x_flat, mask, router_probs)  # (BT, d)
        aux_loss = self._load_balance_loss(router_probs, mask)

        with torch.no_grad():
            self.expert_usage += mask.detach().float().sum(dim=0)

        return output.reshape(B, T, d), aux_loss, router_probs

    def _expert_mix(
        self,
        x: torch.Tensor,          # (BT, d)
        mask: torch.Tensor,       # (BT, N)  hard {0,1} or soft ST
        probs: torch.Tensor,      # (BT, N)  softmax probabilities
    ) -> torch.Tensor:
        """Compute the weighted mixture of expert outputs."""
        # Use mask * probs as importance weights, then renormalise
        scores = mask * probs                                        # (BT, N)
        denom  = scores.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        weights = scores / denom                                     # (BT, N)

        # Stack expert outputs: (BT, N, d)
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)

        # Weighted sum: (BT, d)
        return (weights.unsqueeze(-1) * expert_outs).sum(dim=1)

    def _load_balance_loss(self, probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Switch-Transformer load-balancing loss.
        L = N * Σ_i  f_i * P_i
        f_i = fraction of tokens routed to expert i  (detached)
        P_i = mean router probability for expert i   (differentiable)
        """
        f = mask.detach().float().mean(dim=0)   # (N,)
        P = probs.mean(dim=0)                    # (N,)
        return self.n_experts * (f * P).sum()


# ─────────────────────────────────────────────────────────────────────────────
# SparseMixer block (residual wrapper)
# ─────────────────────────────────────────────────────────────────────────────

class SparseMixerBlock(nn.Module):
    """
    Residual block: x → LayerNorm → SparseMixerLayer → x + out
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int = 8,
        k: int = 1,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        method: str = 'reinmax',
        tau: float = 1.0,
        eta: float = 0.5,
        alpha: float = 1.0,
        repeats: int = 50,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.moe  = SparseMixerLayer(
            d_model=d_model,
            n_experts=n_experts,
            k=k,
            d_ff=d_ff,
            dropout=dropout,
            method=method,
            tau=tau,
            eta=eta,
            alpha=alpha,
            repeats=repeats,
        )

    def forward(self, x: torch.Tensor):
        h, aux, probs = self.moe(self.norm(x))
        return x + h, aux


# ─────────────────────────────────────────────────────────────────────────────
# Full SparseMixer model
# ─────────────────────────────────────────────────────────────────────────────

class SparseMixerModel(nn.Module):
    """
    Stacked SparseMixerBlocks with input projection and classification head.

    This is a lightweight model for experiments. In production settings
    the MoE blocks would be interleaved with attention layers.

    Parameters
    ----------
    input_dim : dimension of each input token
    d_model   : internal model dimension
    n_layers  : number of SparseMixerBlocks
    n_experts : experts per layer
    k         : top-K per token
    n_classes : number of output classes
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_layers: int,
        n_experts: int = 8,
        k: int = 1,
        n_classes: int = 10,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        method: str = 'reinmax',
        tau: float = 1.0,
        eta: float = 0.5,
        alpha: float = 1.0,
        repeats: int = 50,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([
            SparseMixerBlock(
                d_model=d_model,
                n_experts=n_experts,
                k=k,
                d_ff=d_ff,
                dropout=dropout,
                method=method,
                tau=tau,
                eta=eta,
                alpha=alpha,
                repeats=repeats,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor, aux_weight: float = 0.01):
        """
        Parameters
        ----------
        x          : (B, T, input_dim)
        aux_weight : coefficient for the auxiliary load-balancing loss

        Returns
        -------
        logits    : (B, n_classes)
        aux_loss  : scalar
        """
        h       = self.input_proj(x)
        aux_sum = x.new_zeros(1)
        for block in self.blocks:
            h, aux = block(h)
            aux_sum = aux_sum + aux
        h      = self.norm(h).mean(dim=1)   # mean pool over T
        logits = self.head(h)
        return logits, aux_sum * aux_weight

    def routing_entropy(self) -> dict:
        """Return per-layer expert usage entropy (diagnostic)."""
        info = {}
        for i, block in enumerate(self.blocks):
            usage = block.moe.expert_usage.float()
            usage = usage / (usage.sum() + 1e-8)
            H = -(usage * (usage + 1e-8).log()).sum().item()
            info[f'layer_{i}_entropy'] = H
        return info
