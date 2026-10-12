import pytest

from scripts.audit_single_round_cft import outcome_metrics, paired_comparison, paired_grounding_transitions
from scripts.audit_iterative_truth_enrichment import summarize


def row(i, parsed, pos, neg, hard=1., soft=.7):
    out = {"question": f"Question {i}", "parse_success": parsed}
    if parsed:
        out.update({
            "intended_correct_em": bool(pos),
            "intended_misleading_em": bool(neg),
            "hard_weight": hard,
            "sampled_hard_weight": hard,
            "soft_weight": soft,
        })
    return out


def test_paired_downstream_metrics_and_question_alignment():
    baseline = [
        row(0, True, True, False),
        row(1, False, False, False),
        row(2, True, False, False),
        row(3, True, True, False),
    ]
    better = [
        row(0, True, True, False),
        row(1, True, True, False),
        row(2, True, False, False),
        row(3, True, True, False),
    ]
    metric = "grounded_pair_em_all_questions"
    assert outcome_metrics(summarize(baseline))[metric] == pytest.approx(.5)
    results = paired_comparison(baseline, better, bootstrap=30, seed=42)
    assert results[metric]["second_minus_first"] == pytest.approx(.25)
    wrong_order = better[1:] + better[:1]
    with pytest.raises(ValueError, match="ordering differ"):
        paired_comparison(baseline, wrong_order, bootstrap=0, seed=42)


def test_question_level_grounding_transition_counts():
    first = [
        row(0, True, True, False),
        row(1, True, True, False),
        row(2, True, False, False),
        row(3, False, False, False),
    ]
    second = [
        row(0, True, True, False),
        row(1, False, False, False),
        row(2, True, True, False),
        row(3, True, True, False),
    ]
    transitions = paired_grounding_transitions(first, second)
    assert transitions == {
        "both_grounded": 1,
        "lost_grounded": 1,
        "gained_grounded": 2,
        "neither_grounded": 0,
        "net_gain": 1,
        "number_of_questions": 4,
    }
