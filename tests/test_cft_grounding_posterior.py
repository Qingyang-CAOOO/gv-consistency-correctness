import copy

import numpy as np
import pytest

from scripts.fit_cft_grounding_posterior import (
    feature_vector, label, paired_bootstrap, predictive_metrics,
    validated_data,
)


def paired_record(i, truth, *, parsed=True):
    data = {
        "question": f"Question {i}?", "parse_success": parsed,
        "gold_answers": ["ANSWER"],  # not an inference feature
        "intended_correct_em": bool(truth),
        "intended_misleading_em": False,
    }
    if parsed:
        data.update({
            "generator_candidate_positive_score": {
                "per_token_logprob": -1.25 + i / 30,
                "logprob": -3.75, "num_tokens": 3,
            },
            "generator_candidate_negative_score": {
                "per_token_logprob": -2.2, "logprob": -4.4, "num_tokens": 2,
            },
            "p_correct_when_a": .8 if truth else .4,
            "p_correct_when_b": .7 if truth else .3,
        })
    return data


def test_features_are_gold_blind():
    row = paired_record(0, True)
    ga = feature_vector(row, with_validator=False)
    gva = feature_vector(row, with_validator=True)
    assert len(ga) == 3 and len(gva) == 5
    changed = copy.deepcopy(row)
    changed["gold_answers"] = ["not the original gold"]
    changed["intended_correct_em"] = False
    changed["intended_misleading_em"] = True
    assert feature_vector(changed, with_validator=False) == ga
    assert feature_vector(changed, with_validator=True) == gva
    assert label(row) == 1
    assert label(changed) == 0


def test_parsing_and_four_state_binary_label():
    rows = [
        paired_record(0, True), paired_record(1, False),
        paired_record(2, False, parsed=False),
    ]
    parsed, x, y = validated_data(rows, with_validator=True)
    assert len(parsed) == 2
    assert x.shape == (2, 5)
    assert y.tolist() == [1, 0]
    with pytest.raises(ValueError, match="unparsed"):
        label(rows[2])


def test_predictive_metrics_and_paired_bootstrap():
    y = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
    baseline = np.full(len(y), .5)
    informative = np.where(y == 1, .75, .25)
    assert predictive_metrics(y, informative)["nll"] < predictive_metrics(y, baseline)["nll"]
    result = paired_bootstrap(y, baseline, informative, 40, 2026)
    assert result["nll"]["gv_minus_g"] < 0
    assert result["brier"]["gv_minus_g"] < 0
    assert result["weighted_grounded_precision"]["gv_minus_g"] > 0


def test_invalid_both_order_scores_rejected():
    row = paired_record(1, False)
    row["p_correct_when_b"] = 1.2
    with pytest.raises(ValueError, match="probabilities"):
        feature_vector(row, with_validator=True)
