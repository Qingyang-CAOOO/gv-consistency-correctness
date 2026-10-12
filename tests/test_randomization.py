from random import Random

from gvsmc.randomization import (
    balanced_targets,
    binary_choice_with_target_index,
)


def test_binary_choice_target_index():
    a = binary_choice_with_target_index("correct", "wrong", target_index=0)
    b = binary_choice_with_target_index("correct", "wrong", target_index=1)

    assert (a.option_a, a.option_b, a.target_index) == ("correct", "wrong", 0)
    assert (b.option_a, b.option_b, b.target_index) == ("wrong", "correct", 1)


def test_balanced_targets():
    labels = balanced_targets(9, rng=Random(42))
    assert len(labels) == 9
    assert abs(labels.count(0) - labels.count(1)) == 1
