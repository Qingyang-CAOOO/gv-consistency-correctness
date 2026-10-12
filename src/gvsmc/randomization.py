from __future__ import annotations

from dataclasses import dataclass
from random import Random


@dataclass(frozen=True)
class RandomizedBinaryChoice:
    option_a: str
    option_b: str
    target_index: int  # 0 for A, 1 for B


def binary_choice_with_target_index(
    target: str,
    distractor: str,
    *,
    target_index: int,
) -> RandomizedBinaryChoice:
    """Place the target deterministically at A (0) or B (1)."""
    if target_index == 0:
        return RandomizedBinaryChoice(target, distractor, 0)
    if target_index == 1:
        return RandomizedBinaryChoice(distractor, target, 1)
    raise ValueError("target_index must be 0 (A) or 1 (B)")


def randomize_binary_order(
    target: str,
    distractor: str,
    *,
    rng: Random,
) -> RandomizedBinaryChoice:
    """Randomly put the target in A or B with probability 1/2."""
    target_index = 0 if rng.random() < 0.5 else 1
    return binary_choice_with_target_index(
        target,
        distractor,
        target_index=target_index,
    )


def balanced_targets(n: int, *, rng: Random) -> list[int]:
    """Construct a nearly perfectly balanced randomized target sequence."""
    if n < 0:
        raise ValueError("n must be nonnegative")
    labels = [0] * (n // 2) + [1] * (n - n // 2)
    rng.shuffle(labels)
    return labels
