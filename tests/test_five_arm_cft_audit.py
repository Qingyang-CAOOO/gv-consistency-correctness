import pytest

from scripts.audit_five_arm_cft import five_arm_audit, exact_paired_mcnemar


def row(i, *, parsed=True, correct=False, wrong=False, accepted=False):
    r = {"question": f"Question {i}", "parse_success": parsed}
    if parsed:
        r.update({
            "intended_correct_em": correct,
            "intended_misleading_em": wrong,
            "hard_weight": 1.0 if accepted else 0.0,
        })
    return r


def test_five_arm_comparison_and_transitions():
    baseline = [
        row(0, correct=True, accepted=True),
        row(1, correct=False, accepted=True),
        row(2, parsed=False),
        row(3, correct=True, wrong=True),
    ]
    gv = [
        row(0, correct=True, accepted=True),
        row(1, correct=True, accepted=True),
        row(2, parsed=False),
        row(3, correct=True, wrong=True),
    ]
    hard = [
        row(0, correct=True, accepted=True),
        row(1, correct=True, accepted=False),
        row(2, parsed=False),
        row(3, correct=False, accepted=True),
    ]
    soft = [dict(r) for r in baseline]
    soft[0]["intended_correct_em"] = False
    all_data = {
        "posterior-g": baseline,
        "posterior-gv": gv,
        "posterior-g-ess": baseline,
        "hard": hard,
        "soft": soft,
    }
    out = five_arm_audit(all_data, bootstrap=100, seed=42)
    assert out["n_questions"] == 4
    assert out["arm_metrics"]["posterior-g"]["grounded_pair_em_all_questions"] == .25
    assert out["arm_metrics"]["posterior-gv"]["grounded_pair_em_all_questions"] == .5
    assert out["comparisons"]["posterior-g"]["target_minus_baseline"][
        "grounded_pair_em_all_questions"
    ] == .25
    assert out["comparisons"]["posterior-g"]["paired_grounding_transitions"][
        "gained_grounded"
    ] == 1
    assert out["comparisons"]["posterior-g-ess"]["paired_grounding_transitions"][
        "lost_grounded"
    ] == 0
    assert out["arm_metrics"]["posterior-gv"]["hard_gv_consistency_given_parsed"] == 2/3
    assert len(out["comparisons"]["hard"]["bootstrap_95_ci"][
        "grounded_pair_em_all_questions"
    ]) == 2
    assert out == five_arm_audit(all_data, bootstrap=100, seed=42)


def test_five_arm_question_alignment_is_enforced():
    a = [row(0, correct=True), row(1, correct=False)]
    b = [row(1, correct=False), row(0, correct=True)]
    with pytest.raises(ValueError, match="question order"):
        five_arm_audit({"posterior-g": a, "posterior-gv": b}, bootstrap=0)


def test_five_arm_duplicate_questions_rejected():
    a = [row(0), row(0)]
    with pytest.raises(ValueError, match="duplicate"):
        five_arm_audit({"posterior-g": a, "posterior-gv": a}, bootstrap=0)


def test_five_arm_zero_bootstrap_does_not_crash():
    a = [row(0, correct=True), row(1, correct=False)]
    out = five_arm_audit({"posterior-g": a, "posterior-gv": a}, bootstrap=0)
    assert out["comparisons"]["posterior-g"]["bootstrap_95_ci"][
        "grounded_pair_em_all_questions"
    ] is None


def test_exact_mcnemar_for_reported_gv_soft_discordance():
    # 14 improved, 5 worsened; the percentile bootstrap CI excludes zero,
    # but two-sided exact paired McNemar does not at alpha=.05.
    assert exact_paired_mcnemar(14, 5) == pytest.approx(0.063568115234375)
    assert exact_paired_mcnemar(5, 14) == pytest.approx(0.063568115234375)
    assert exact_paired_mcnemar(0, 0) == pytest.approx(1.0)
    assert exact_paired_mcnemar(1, 0) == pytest.approx(1.0)
    assert exact_paired_mcnemar(10, 0) == pytest.approx(2/1024)
    with pytest.raises(ValueError):
        exact_paired_mcnemar(-1, 5)


def test_exact_mcnemar_attached_to_five_arm_audit():
    a = [row(0, correct=False), row(1, correct=True)]
    b = [row(0, correct=True), row(1, correct=True)]
    result = five_arm_audit(
        {"posterior-g": a, "soft": a, "posterior-gv": b},
        bootstrap=12, seed=44
    )
    assert result["comparisons"]["soft"]["exact_mcnemar_two_sided_p"] == 1.0
    assert "paired_test" in result
