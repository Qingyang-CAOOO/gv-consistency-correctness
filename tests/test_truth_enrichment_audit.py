import pytest

from scripts.audit_iterative_truth_enrichment import summarize


def row(idx, plus, minus, hard, soft, *, parsed=True, a=None, b=None):
    r = {
        "index": idx,
        "question": f"Question {idx}?",
        "parse_success": parsed,
        "parse_mode": "delimiter_line" if parsed else "unrecognized_format",
    }
    if parsed:
        r.update({
            "intended_correct_em": plus,
            "intended_misleading_em": minus,
            "hard_weight": hard,
            "soft_weight": soft,
            "sampled_hard_weight": hard,
        })
        if a is not None and b is not None:
            r.update({"p_correct_when_a": a, "p_correct_when_b": b})
    return r


def test_four_states_parse_and_truth_enrichment():
    rows = [
        row(0, True, False, 1, .8, a=.9, b=.7),
        row(1, False, False, 1, .6, a=.3, b=.9),
        row(2, False, True, 0, .2, a=.1, b=.3),
        row(3, False, False, 0, 0., parsed=False),
    ]
    s = summarize(rows)
    assert s["num_questions"] == 4
    assert s["num_parsed"] == 3
    assert s["parse_rate"] == pytest.approx(.75)
    assert s["factual_state_counts_given_parsed"] == {
        "10": 1, "01": 1, "11": 0, "00": 1
    }
    assert s["valid_and_grounded_pair_rate_all_questions"] == pytest.approx(.25)
    assert s["hard_weight"]["selected_grounded_precision"] == pytest.approx(.5)
    assert s["soft_weight"]["selected_grounded_precision"] == pytest.approx(.5)
    assert s["hard_weight"]["precision_lift_over_prevalence"] == pytest.approx(1 / 6)
    assert s["soft_weight"]["precision_lift_over_prevalence"] == pytest.approx(1 / 6)
    assert s["order_marginalized"]["precision_lift_over_prevalence"] == pytest.approx(1 / 6)
    assert s["hard_weight"]["truth_enrichment_ratio"] == pytest.approx(2.)
    assert s["soft_weight"]["truth_enrichment_ratio"] == pytest.approx(2.)
    assert s["order_marginalized"]["mean_abs_order_gap"] == pytest.approx(1 / 3)


def test_unparsed_not_counted_as_false_factual_label():
    rows = [row(0, True, False, 1, .8),
            row(1, False, False, 0, .2, parsed=False)]
    s = summarize(rows)
    assert s["grounded_rate_given_parsed"] == 1
    assert s["valid_and_grounded_pair_rate_all_questions"] == .5
    assert s["soft_weight"]["truth_enrichment_ratio"] is None


def test_all_unparsed_records_have_no_conditional_truth_stats():
    s = summarize([row(0, False, False, 0, 0., parsed=False)])
    assert s["num_parsed"] == 0
    assert s["grounded_rate_given_parsed"] is None
