import math

import pytest

from gvsmc.cft import build_cft_examples


def record():
    return {
        "index": 7,
        "parse_success": True,
        "generator_prompt": "Generate two answers",
        "raw_generator_response": "Paris || Berlin",
        "validator_prompt": "one sampled placement",
        "target_label": "A",
        "soft_weight": .9,
        "hard_weight": 1,
        "sampled_hard_weight": 1,
        "p_correct_when_a": .9,
        "p_correct_when_b": .3,
        "validator_prompt_when_a": "A: Paris; B: Berlin",
        "validator_prompt_when_b": "A: Berlin; B: Paris",
    }


def test_marginalized_objective_weights_both_validator_placements():
    examples = build_cft_examples([record()], mode="soft-marginalized")
    assert [ex.component for ex in examples] == ["generator", "validator", "validator"]
    assert [ex.completion for ex in examples] == ["Paris || Berlin", "A", "B"]
    assert [ex.weight for ex in examples] == pytest.approx([.6, .45, .15])
    assert [ex.prompt for ex in examples] == [
        "Generate two answers",
        "A: Paris; B: Berlin",
        "A: Berlin; B: Paris",
    ]
    # Generator and validator each have total weight .6; normalized objective
    # is exactly the sum of the two weighted likelihood components divided
    # by 2 times the marginal potential.
    losses = [2., 3., 5.]
    empirical = sum(ex.weight * l for ex, l in zip(examples, losses)) / sum(ex.weight for ex in examples)
    expected = (.6 * 2 + .5 * .9 * 3 + .5 * .3 * 5) / (2 * .6)
    assert empirical == pytest.approx(expected)


def test_marginalized_rejects_missing_both_order_scores():
    r = record()
    del r["p_correct_when_b"]
    with pytest.raises(ValueError, match="both-order"):
        build_cft_examples([r], mode="soft-marginalized")


def test_marginalized_rejects_invalid_probabilities():
    r = record()
    r["p_correct_when_a"] = 1.2
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        build_cft_examples([r], mode="soft-marginalized")


def test_single_order_modes_unchanged():
    r = record()
    assert [ex.weight for ex in build_cft_examples([r], mode="hard")] == [1., 1.]
    assert [ex.weight for ex in build_cft_examples([r], mode="soft")] == [.9, .9]
