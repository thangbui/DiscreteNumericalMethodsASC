"""
Gradient estimators for discrete top-K routing in Mixture-of-Experts (SparseMixer).

All estimators operate on 2-D tensors of shape (B, N):
  B = batch size (or batch * sequence length)
  N = number of experts

For K=1 (single-expert selection):
  - st_single          : Straight-Through (ST)
  - reinmax_single     : ReinMax  (Liu et al. 2023)
  - reinmax_v3_single  : ReinMax-v3 — uses Rao-Gumbel conditional Jacobian for (pi+D)/2
  - reinmax_cv_single  : ReinMax-CV — adds a Gumbel-Softmax control variate

For K>1 (top-K selection):
  - reinmax_topk     : ReinMax extended to k-hot masks
  - reinmax_v3_topk  : ReinMax-v3 for top-K
  - reinmax_cv_topk  : ReinMax-CV for top-K

Theory note
-----------
All methods estimate  d/d_logits E_{z~Cat(softmax(logits))}[f(z)].

The exact gradient is:  p * (f_vals - E[f])   (REINFORCE / score function)

ReinMax decomposes it into two terms:
  term0: gradient of the "first-step" softmax      (-1/(2α) * grad + grad_p) * p
  term1: gradient through the shifted mixture       2 * J_{(pi+D)/2} * grad

ReinMax-v3 replaces the analytical Jacobian in term1 with a Monte-Carlo estimate
via the Rao-Gumbel conditional distribution, which gives a lower-variance Jacobian.

ReinMax-CV adds a control variate to further reduce variance:
  grad_cv = grad_reinmax - η*(J_GS * grad) + η*(J_GR * grad)
where J_GS is the Gumbel-Softmax Jacobian (high-variance, unbiased per-sample)
and J_GR is the Rao-Gumbel expected Jacobian (lower-variance).
"""

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def _sample_gumbel(shape, device=None, dtype=None):
    """Sample i.i.d. Gumbel(0,1) noise."""
    U = torch.zeros(shape, device=device, dtype=dtype).uniform_().clamp_(1e-20, 1.0)
    return -(-U.log()).log()


def _gumbel_top_k(logits: torch.Tensor, k: int):
    """
    Draw a k-hot mask from the Gumbel-top-K distribution.

    Returns
    -------
    mask      : (B, N) k-hot float tensor
    perturbed : (B, N) logits + Gumbel noise
    gumbels   : (B, N) raw Gumbel samples
    """
    gumbels = _sample_gumbel(logits.shape, device=logits.device, dtype=logits.dtype)
    perturbed = logits + gumbels
    # top-K threshold
    topk_vals = perturbed.topk(k, dim=-1).values          # (B, K)
    threshold = topk_vals[..., -1:]                        # (B, 1)
    mask = (perturbed >= threshold).float()
    return mask, perturbed, gumbels


def _softmax_jacobian(p: torch.Tensor) -> torch.Tensor:
    """
    Jacobian of softmax.  p: (B, N) -> (B, N, N)
    J_ij = p_i*(delta_ij - p_j)
    """
    return torch.diag_embed(p) - p.unsqueeze(-1) * p.unsqueeze(-2)


def _rao_gumbel_jacobian(
    logits: torch.Tensor,
    z: torch.Tensor,
    tau: float,
    repeats: int,
) -> torch.Tensor:
    """
    Monte-Carlo estimate of  E_{G | argmax(logits+G) = D} [J_softmax_tau(logits+G)]
    via the Rao-Gumbel conditional sampling trick.

    Parameters
    ----------
    logits  : (B, N)  base log-weights (e.g. log((pi+D)/2) or raw logits)
    z       : (B, N)  one-hot indicating the selected expert D
    tau     : float   temperature
    repeats : int     number of Monte-Carlo samples R

    Returns
    -------
    J_avg : (B, N, N)  average Jacobian
    """
    B, N = logits.shape
    R = repeats
    logits_d = logits.detach()
    action_bool = z.bool()          # (B, N)

    # Exponential samples: shape (B, N, R)
    E = logits_d.new_empty(B, N, R).exponential_()

    # Ei: exponential for the *selected* category — shape (B, R)
    Ei = E[action_bool].reshape(B, R)

    # Unnormalised weights
    wei = logits_d.exp().clamp(min=1e-20)               # (B, N)
    Z   = wei.sum(dim=-1, keepdim=True)                  # (B, 1)

    # Conditional Gumbel samples given argmax = D
    EiZ = (Ei / Z).unsqueeze(1)                          # (B, 1, R)
    ratio = E / wei.unsqueeze(-1)                        # (B, N, R)
    ratio[action_bool] = 0.0                             # zero out selected slot
    new_logits = -(ratio + EiZ + 1e-20).log()            # (B, N, R)

    # Softmax of conditional perturbations
    new_pi = (new_logits / tau).softmax(dim=1)           # (B, N, R)

    # Jacobian per sample, then average
    new_pi_t = new_pi.permute(0, 2, 1)                  # (B, R, N)
    J = torch.diag_embed(new_pi_t) \
        - new_pi_t.unsqueeze(-1) * new_pi_t.unsqueeze(-2)  # (B, R, N, N)
    J_avg = J.mean(dim=1) / tau                          # (B, N, N)
    return J_avg


# ─────────────────────────────────────────────────────────────────────────────
# K = 1  (single-expert selection)
# ─────────────────────────────────────────────────────────────────────────────

def st_single(logits: torch.Tensor, tau: float = 1.0):
    """
    Straight-Through estimator for single-expert selection.

    Forward : hard one-hot from argmax(logits + Gumbel)
    Backward: gradient flows through softmax_tau(logits)
    """
    p_soft = F.softmax(logits / tau, dim=-1)
    p_base = F.softmax(logits, dim=-1)
    idx    = torch.multinomial(p_base, num_samples=1)    # (B, 1)
    z_hard = torch.zeros_like(logits).scatter_(-1, idx, 1.0)
    z_st   = z_hard - p_soft.detach() + p_soft           # ST
    return z_st, p_base


class _ReinMaxSingle(torch.autograd.Function):
    """ReinMax for single-expert selection (Liu et al. 2023)."""

    @staticmethod
    def forward(ctx, logits, tau, alpha):
        p = F.softmax(logits, dim=-1)
        idx = torch.multinomial(p, num_samples=1)
        z   = torch.zeros_like(logits).scatter_(-1, idx, 1.0)
        ctx.save_for_backward(
            logits, z, p,
            logits.new_tensor(tau),
            logits.new_tensor(alpha),
        )
        return z, p

    @staticmethod
    def backward(ctx, grad_z, grad_p):
        logits, z, p, tau_t, alpha_t = ctx.saved_tensors
        tau_v, alpha_v = tau_t.item(), alpha_t.item()

        p_tau   = F.softmax(logits / tau_v, dim=-1)
        pi_alpha = (1.0 - 1.0 / (2.0 * alpha_v)) * p_tau \
                 + (1.0 / (2.0 * alpha_v)) * z

        # term1: gradient through (p_tau + z)/2
        shifted = 0.5 * (p_tau + z)
        g1 = 2.0 * grad_z * shifted
        g1 = g1 - shifted * (2.0 * grad_z * pi_alpha).sum(dim=-1, keepdim=True)

        # term0: gradient through base softmax
        g0 = (-1.0 / (2.0 * alpha_v) * grad_z + grad_p) * p
        g0 = g0 - p * g0.sum(dim=-1, keepdim=True)

        g = g0 + g1
        return g - g.mean(dim=-1, keepdim=True), None, None


class _ReinMaxV3Single(torch.autograd.Function):
    """
    ReinMax-v3 for single-expert selection.

    Replaces the analytical Jacobian in term1 with the Rao-Gumbel
    conditional Jacobian evaluated at log((pi+D)/2).
    """

    @staticmethod
    def forward(ctx, logits, tau, repeats):
        p   = F.softmax(logits, dim=-1)
        idx = torch.multinomial(p, num_samples=1)
        z   = torch.zeros_like(logits).scatter_(-1, idx, 1.0)
        ctx.save_for_backward(
            logits, z, p,
            logits.new_tensor(tau),
            logits.new_tensor(float(repeats)),
        )
        return z, p

    @staticmethod
    def backward(ctx, grad_z, grad_p):
        logits, z, p, tau_t, rep_t = ctx.saved_tensors
        tau_v = tau_t.item()
        R     = int(rep_t.item())

        # Shifted log-probs: log((pi + D)/2)
        shifted_logits = (0.5 * (p + z)).log()

        # Rao-Gumbel Jacobian at the shifted distribution
        J = _rao_gumbel_jacobian(shifted_logits, z, tau_v, R)   # (B, N, N)

        # term1: 2 * J * grad_z
        g1 = 2.0 * torch.matmul(J, grad_z.unsqueeze(-1)).squeeze(-1)

        # term0: same as ReinMax (alpha=1)
        g0 = -0.5 * grad_z * p + grad_p * p
        g0 = g0 - p * g0.sum(dim=-1, keepdim=True)

        g = g0 + g1
        return g - g.mean(dim=-1, keepdim=True), None, None


class _ReinMaxCVSingle(torch.autograd.Function):
    """
    ReinMax-CV for single-expert selection.

    Control variate:  grad_cv = grad_reinmax - η*(J_GS - J_GR)*grad_z
      J_GS: Gumbel-Softmax Jacobian at the specific Gumbel realisation
      J_GR: Expected Rao-Gumbel Jacobian (lower-variance estimate)

    Because E[J_GS * grad_z] ≈ E[J_GR * grad_z], the CV is zero-mean
    in expectation and reduces variance.
    """

    @staticmethod
    def forward(ctx, logits, tau, eta, repeats):
        p       = F.softmax(logits, dim=-1)
        gumbels = _sample_gumbel(logits.shape, device=logits.device, dtype=logits.dtype)
        idx     = (logits + gumbels).argmax(dim=-1, keepdim=True)
        z       = torch.zeros_like(logits).scatter_(-1, idx, 1.0)
        ctx.save_for_backward(
            logits, z, p, gumbels,
            logits.new_tensor(tau),
            logits.new_tensor(eta),
            logits.new_tensor(float(repeats)),
        )
        return z, p

    @staticmethod
    def backward(ctx, grad_z, grad_p):
        logits, z, p, G, tau_t, eta_t, rep_t = ctx.saved_tensors
        tau_v = tau_t.item()
        eta_v = eta_t.item()
        R     = int(rep_t.item())

        # ── Base ReinMax gradient (alpha=1, tau=1 for term1) ──────────────
        p_tau1 = F.softmax(logits, dim=-1)              # tau=1 for term1
        shifted = 0.5 * (p_tau1 + z)
        g1 = 2.0 * grad_z * shifted
        g1 = g1 - shifted * g1.sum(dim=-1, keepdim=True)
        g0 = -0.5 * grad_z * p + grad_p * p
        g0 = g0 - p * g0.sum(dim=-1, keepdim=True)
        grad_reinmax = g0 + g1

        # ── Control variate ───────────────────────────────────────────────
        new_pi     = 0.5 * (p + z)
        new_logits = new_pi.log()

        # J_GS: Gumbel-Softmax Jacobian at (new_logits + G) / tau
        p_gs  = F.softmax((new_logits + G) / tau_v, dim=-1)
        J_gs  = _softmax_jacobian(p_gs) / tau_v                   # (B, N, N)
        gv_gs = torch.matmul(J_gs, grad_z.unsqueeze(-1)).squeeze(-1)

        # J_GR: Rao-Gumbel expected Jacobian at new_logits
        J_gr  = _rao_gumbel_jacobian(new_logits, z, tau_v, R)     # (B, N, N)
        gv_gr = torch.matmul(J_gr, grad_z.unsqueeze(-1)).squeeze(-1)

        # CV correction
        g = grad_reinmax - eta_v * gv_gs + eta_v * gv_gr
        return g - g.mean(dim=-1, keepdim=True), None, None, None


def reinmax_single(logits: torch.Tensor, tau: float = 1.0, alpha: float = 1.0):
    """ReinMax gradient estimator for single-expert selection."""
    return _ReinMaxSingle.apply(logits, tau, alpha)


def reinmax_v3_single(logits: torch.Tensor, tau: float = 1.0, repeats: int = 50):
    """ReinMax-v3 (Rao-Gumbel Jacobian) for single-expert selection."""
    return _ReinMaxV3Single.apply(logits, tau, repeats)


def reinmax_cv_single(logits: torch.Tensor, tau: float = 1.0, eta: float = 0.5, repeats: int = 50):
    """ReinMax-CV (control-variate) for single-expert selection."""
    return _ReinMaxCVSingle.apply(logits, tau, eta, repeats)


# ─────────────────────────────────────────────────────────────────────────────
# K > 1  (top-K selection)
# ─────────────────────────────────────────────────────────────────────────────

def _rao_gumbel_jacobian_topk(
    logits: torch.Tensor,
    mask: torch.Tensor,
    k: int,
    tau: float,
    repeats: int,
) -> torch.Tensor:
    """
    Rao-Gumbel Jacobian for a k-hot mask.

    Decomposes the k-hot selection into k independent single-expert Jacobians
    (one per selected expert) and averages them.

    Returns (B, N, N).
    """
    B, N = logits.shape
    J_total = torch.zeros(B, N, N, dtype=logits.dtype, device=logits.device)
    # shifted log-weights: log((softmax(logits) + mask/k) / 2)
    p = F.softmax(logits, dim=-1)
    shifted_logits = (0.5 * (p + mask / k)).log()

    for b in range(B):
        selected = mask[b].bool().nonzero(as_tuple=True)[0]  # (k,)
        for s_idx in selected:
            z_b = torch.zeros(1, N, dtype=logits.dtype, device=logits.device)
            z_b[0, s_idx] = 1.0
            J_b = _rao_gumbel_jacobian(shifted_logits[b:b+1], z_b, tau, repeats)
            J_total[b] += J_b[0]
    return J_total / k


class _ReinMaxTopK(torch.autograd.Function):
    """ReinMax extended to top-K selection."""

    @staticmethod
    def forward(ctx, logits, k, tau):
        p = F.softmax(logits, dim=-1)
        mask, _, _ = _gumbel_top_k(logits, k)
        ctx.save_for_backward(
            logits, mask, p,
            logits.new_tensor(float(k)),
            logits.new_tensor(tau),
        )
        return mask, p

    @staticmethod
    def backward(ctx, grad_mask, grad_p):
        logits, mask, p, k_t, tau_t = ctx.saved_tensors
        k     = int(k_t.item())
        tau_v = tau_t.item()

        p_tau   = F.softmax(logits / tau_v, dim=-1)
        z_norm  = mask / k                           # treat k-hot as distribution
        shifted = 0.5 * (p_tau + z_norm)

        g1 = 2.0 * grad_mask * shifted
        g1 = g1 - shifted * g1.sum(dim=-1, keepdim=True)

        g0 = (-0.5 * grad_mask + grad_p) * p
        g0 = g0 - p * g0.sum(dim=-1, keepdim=True)

        g = g0 + g1
        return g - g.mean(dim=-1, keepdim=True), None, None


class _ReinMaxV3TopK(torch.autograd.Function):
    """ReinMax-v3 for top-K selection (Rao-Gumbel Jacobian)."""

    @staticmethod
    def forward(ctx, logits, k, tau, repeats):
        p = F.softmax(logits, dim=-1)
        mask, _, _ = _gumbel_top_k(logits, k)
        ctx.save_for_backward(
            logits, mask, p,
            logits.new_tensor(float(k)),
            logits.new_tensor(tau),
            logits.new_tensor(float(repeats)),
        )
        return mask, p

    @staticmethod
    def backward(ctx, grad_mask, grad_p):
        logits, mask, p, k_t, tau_t, rep_t = ctx.saved_tensors
        k     = int(k_t.item())
        tau_v = tau_t.item()
        R     = int(rep_t.item())

        J  = _rao_gumbel_jacobian_topk(logits, mask, k, tau_v, R)   # (B, N, N)
        g1 = 2.0 * torch.matmul(J, grad_mask.unsqueeze(-1)).squeeze(-1)

        g0 = (-0.5 * grad_mask + grad_p) * p
        g0 = g0 - p * g0.sum(dim=-1, keepdim=True)

        g = g0 + g1
        return g - g.mean(dim=-1, keepdim=True), None, None, None


class _ReinMaxCVTopK(torch.autograd.Function):
    """ReinMax-CV for top-K selection."""

    @staticmethod
    def forward(ctx, logits, k, tau, eta, repeats):
        p = F.softmax(logits, dim=-1)
        mask, _, gumbels = _gumbel_top_k(logits, k)
        ctx.save_for_backward(
            logits, mask, p, gumbels,
            logits.new_tensor(float(k)),
            logits.new_tensor(tau),
            logits.new_tensor(eta),
            logits.new_tensor(float(repeats)),
        )
        return mask, p

    @staticmethod
    def backward(ctx, grad_mask, grad_p):
        logits, mask, p, G, k_t, tau_t, eta_t, rep_t = ctx.saved_tensors
        k     = int(k_t.item())
        tau_v = tau_t.item()
        eta_v = eta_t.item()
        R     = int(rep_t.item())

        # ── Base ReinMax ──────────────────────────────────────────────────
        z_norm  = mask / k
        shifted = 0.5 * (p + z_norm)
        g1 = 2.0 * grad_mask * shifted
        g1 = g1 - shifted * g1.sum(dim=-1, keepdim=True)
        g0 = (-0.5 * grad_mask + grad_p) * p
        g0 = g0 - p * g0.sum(dim=-1, keepdim=True)
        grad_reinmax = g0 + g1

        # ── Control variate ───────────────────────────────────────────────
        new_pi     = 0.5 * (p + z_norm)
        new_logits = new_pi.log()

        # J_GS: Gumbel-Softmax Jacobian
        p_gs  = F.softmax((new_logits + G) / tau_v, dim=-1)
        J_gs  = _softmax_jacobian(p_gs) / tau_v
        gv_gs = torch.matmul(J_gs, grad_mask.unsqueeze(-1)).squeeze(-1)

        # J_GR: Rao-Gumbel Jacobian (decomposed over selected experts)
        J_gr  = _rao_gumbel_jacobian_topk(new_logits, mask, k, tau_v, R)
        gv_gr = torch.matmul(J_gr, grad_mask.unsqueeze(-1)).squeeze(-1)

        g = grad_reinmax - eta_v * gv_gs + eta_v * gv_gr
        return g - g.mean(dim=-1, keepdim=True), None, None, None, None


def reinmax_topk(logits: torch.Tensor, k: int, tau: float = 1.0):
    """ReinMax for top-K selection."""
    return _ReinMaxTopK.apply(logits, k, tau)


def reinmax_v3_topk(logits: torch.Tensor, k: int, tau: float = 1.0, repeats: int = 50):
    """ReinMax-v3 for top-K selection."""
    return _ReinMaxV3TopK.apply(logits, k, tau, repeats)


def reinmax_cv_topk(logits: torch.Tensor, k: int, tau: float = 1.0, eta: float = 0.5, repeats: int = 50):
    """ReinMax-CV for top-K selection."""
    return _ReinMaxCVTopK.apply(logits, k, tau, eta, repeats)
