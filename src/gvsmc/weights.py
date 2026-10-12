from __future__ import annotations

import torch
from torch import Tensor


def hard_randomized_gv_weight(target: Tensor, prediction: Tensor) -> Tensor:
    """Return 1 when the validator prediction matches randomized target, else 0.

    Parameters
    ----------
    target:
        Integer-coded randomized target labels, shape (...,).
    prediction:
        Integer-coded realized validator labels, same shape as ``target``.
    """
    if target.shape != prediction.shape:
        raise ValueError("target and prediction must have identical shapes")
    return (target == prediction).to(dtype=torch.float32)


def soft_randomized_gv_weight_from_label_logprobs(
    label_logprobs: Tensor,
    target_index: Tensor,
    *,
    temperature: float = 1.0,
) -> Tensor:
    """Probability assigned to the randomized target among admissible labels.

    ``label_logprobs[..., k]`` is a sequence-level log-probability score for
    admissible validator label ``k`` (for example A/B or Yes/No). We renormalize
    over these labels before selecting the randomized target.

    This produces the soft potential

        G = p(target) / sum_k p(k)

    and is therefore robust to probability mass assigned to unrelated tokens.
    """
    if label_logprobs.ndim < 2:
        raise ValueError("label_logprobs must have shape (..., num_labels)")
    if label_logprobs.shape[:-1] != target_index.shape:
        raise ValueError("target_index shape must equal label_logprobs.shape[:-1]")
    if label_logprobs.shape[-1] < 2:
        raise ValueError("at least two admissible validator labels are required")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    scaled = label_logprobs / temperature
    probs = torch.softmax(scaled, dim=-1)
    return probs.gather(-1, target_index.unsqueeze(-1)).squeeze(-1)


def soft_randomized_gv_weight_from_signed_logodds(
    signed_logodds: Tensor,
    *,
    temperature: float = 1.0,
) -> Tensor:
    """Convert target-signed binary validator log-odds to a soft GV potential.

    ``signed_logodds`` should be positive when the validator favors the
    randomized target and negative when it favors the opposite label.

        G = sigmoid(signed_logodds / temperature)
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return torch.sigmoid(signed_logodds / temperature)


def normalize_weights(weights: Tensor, *, eps: float = 1e-12) -> Tensor:
    """Normalize nonnegative weights along the last dimension."""
    if torch.any(weights < 0):
        raise ValueError("weights must be nonnegative")
    total = weights.sum(dim=-1, keepdim=True)
    if torch.any(total <= eps):
        raise ValueError("cannot normalize a set of weights with near-zero total mass")
    return weights / total


def sequential_weight_update(
    previous_normalized_weights: Tensor,
    potentials: Tensor,
    *,
    eps: float = 1e-12,
) -> Tensor:
    """One Feynman-Kac / SMC weighting update without resampling."""
    if previous_normalized_weights.shape != potentials.shape:
        raise ValueError("previous weights and potentials must have the same shape")
    if torch.any(potentials < 0):
        raise ValueError("potentials must be nonnegative")
    return normalize_weights(previous_normalized_weights * potentials, eps=eps)
