from __future__ import annotations

import torch
from torch import Tensor

from .weights import normalize_weights


def sequential_importance_update(
    weights: Tensor,
    potentials: Tensor,
    *,
    eps: float = 1e-12,
) -> Tensor:
    """Apply one static-state Feynman--Kac/SIS update.

    The state itself is unchanged; only the particle weights are updated:
        w_t^i ∝ w_{t-1}^i G_t^i.
    """
    if weights.shape != potentials.shape:
        raise ValueError("weights and potentials must have the same shape")
    if torch.any(weights < 0):
        raise ValueError("weights must be nonnegative")
    if torch.any((potentials < 0) | (potentials > 1)):
        raise ValueError("potentials must lie in [0, 1]")
    return normalize_weights(weights * potentials, eps=eps)


def posterior_mass(mask: Tensor, weights: Tensor) -> Tensor:
    """Return normalized posterior mass assigned to a Boolean subset."""
    if mask.shape != weights.shape:
        raise ValueError("mask and weights must have the same shape")
    normalized = normalize_weights(weights)
    return normalized[mask.to(torch.bool)].sum()


def top_particle_index(weights: Tensor) -> int:
    """Index of the maximum-weight particle."""
    if weights.numel() == 0:
        raise ValueError("weights cannot be empty")
    return int(torch.argmax(weights).item())
