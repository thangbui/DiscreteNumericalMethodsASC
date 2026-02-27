"""
SparseMixer: Sparse Mixture-of-Experts with discrete routing via ReinMax estimators.

Reference:
  Liu, Gao, Chen (2023) - SparseMixer / ReinMax
  "ReinMax: Bridging Discrete and Continuous Optimization"

Gradient estimators:
  - ST           : Straight-Through
  - reinmax      : ReinMax (baseline)
  - reinmax_v3   : ReinMax-v3 (Rao-Gumbel Jacobian for the shifted distribution)
  - reinmax_cv   : ReinMax-CV (+ Gumbel-Softmax control variate)

Both K=1 (single expert) and K>1 (top-K) selection are supported.
"""

from sparse_mixer.estimators import (
    st_single,
    reinmax_single,
    reinmax_v3_single,
    reinmax_cv_single,
    reinmax_topk,
    reinmax_v3_topk,
    reinmax_cv_topk,
)
from sparse_mixer.sparse_mixer import Expert, SparseMixerLayer, SparseMixerBlock, SparseMixerModel
