import json

import numpy as np
import pytest

from gvsmc.cft import build_cft_examples
from scripts.apply_frozen_cft_posteriors import (
    add_posterior_weights, verify_exact_question_set,
)


class FakePosterior:
    def __init__(self, n_features, values):
        self.n_features_in_ = n_features
        self.values = np.asarray(values, dtype=float)

    def predict_proba(self, features):
        assert features.shape == (len(self.values), self.n_features_in_)
        return np.column_stack((1-self.values, self.values))


def example(index):
    return {
        "question": f"Q {index}?",
        "index": index,
        "gold_answers": ["secret gold"],
        "intended_correct_em": True,
        "intended_misleading_em": False,
        "grounded_pair": True,
        "parse_success": True,
        "generator_prompt": "Generate two answers to Q",
        "raw_generator_response": "gold || wrong",
        "validator_prompt": "A: gold\nB: wrong",
        "target_label": "A",
        "intended_correct": "gold",
        "intended_misleading": "wrong",
        "soft_weight": .8,
        "hard_weight": 1.0,
        "generator_candidate_positive_score": {
            "per_token_logprob": -1.1, "num_tokens": 2
        },
        "generator_candidate_negative_score": {
            "per_token_logprob": -2.3, "num_tokens": 1
        },
        "p_correct_when_a": .9,
        "p_correct_when_b": .7,
    }


def test_frozen_posterior_modes_keep_same_training_targets():
    rows = [example(0), example(1)]
    output, summary = add_posterior_weights(
        rows,
        FakePosterior(3, [.4, .6]),
        FakePosterior(5, [.3, .85]),
    )
    assert summary["num_label_fields_removed"] == 8
    assert all("gold_answers" not in r for r in output)
    assert all("intended_correct_em" not in r for r in output)
    assert all("grounded_pair" not in r for r in output)
    for mode, field in [
        ("posterior-g", "posterior_q_g"),
        ("posterior-gv", "posterior_q_gv"),
        ("posterior-g-ess", "posterior_q_g_essmatched"),
    ]:
        examples = build_cft_examples(output, mode=mode)
        assert len(examples) == 4
        assert [x.completion for x in examples] == [
            "gold || wrong", "A", "gold || wrong", "A"
        ]
        assert [x.weight for x in examples] == pytest.approx([
            output[0][field],output[0][field],
            output[1][field],output[1][field],
        ])
    assert summary["q_g_essmatched_ess_fraction"] == pytest.approx(
        summary["q_gv_ess_fraction"], abs=1e-7
    )


def test_posterior_features_do_not_depend_on_gold_fields():
    rows = [example(0), example(1)]
    left, _ = add_posterior_weights(
        rows, FakePosterior(3, [.4, .6]), FakePosterior(5, [.3, .85])
    )
    for r in rows:
        r["gold_answers"] = ["changed secret"]
        r["intended_correct_em"] = False
        r["intended_misleading_em"] = True
        r["grounded_pair"] = False
    right, _ = add_posterior_weights(
        rows, FakePosterior(3, [.4, .6]), FakePosterior(5, [.3, .85])
    )
    for a, b in zip(left, right):
        assert a["posterior_q_g"] == b["posterior_q_g"]
        assert a["posterior_q_gv"] == b["posterior_q_gv"]
        assert a["posterior_q_g_essmatched"] == b["posterior_q_g_essmatched"]


def test_missing_posterior_fields_rejected_before_training():
    with pytest.raises(ValueError, match="requires posterior_q_gv"):
        build_cft_examples([example(0)], mode="posterior-gv")


def test_cft_partition_verification(tmp_path):
    source = [example(0), example(1)]
    expected = tmp_path / "cft.jsonl"
    excluded = tmp_path / "val.jsonl"
    expected.write_text(
        "\n".join(json.dumps({"question": x["question"]}) for x in source)+"\n"
    )
    excluded.write_text(json.dumps({"question": "unseen question"})+"\n")
    verify_exact_question_set(source, expected, [excluded])
    excluded.write_text(json.dumps({"question": source[0]["question"]})+"\n")
    with pytest.raises(ValueError, match="overlap"):
        verify_exact_question_set(source, expected, [excluded])
