from __future__ import annotations

import math

import torch
from torch import Tensor

from .weights import normalize_weights


def effective_sample_size(weights: Tensor, *, normalized: bool = False) -> Tensor:
    """Compute ESS = 1 / sum_i w_i^2 along the last dimension."""
    w = weights if normalized else normalize_weights(weights)
    return 1.0 / torch.sum(w.square(), dim=-1)


def normalized_entropy(weights: Tensor, *, normalized: bool = False, eps: float = 1e-12) -> Tensor:
    """Shannon entropy divided by log(M), yielding a value in [0,1] for M>1."""
    w = weights if normalized else normalize_weights(weights, eps=eps)
    m = w.shape[-1]
    if m <= 1:
        return torch.ones_like(w.sum(dim=-1))
    entropy = -torch.sum(w * torch.log(w.clamp_min(eps)), dim=-1)
    return entropy / math.log(m)


def retained_fraction(hard_weights: Tensor) -> Tensor:
    """Fraction of nonzero particles/examples under a hard filter."""
    return (hard_weights > 0).to(torch.float32).mean(dim=-1)
