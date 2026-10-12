import pytest

from scripts.audit_multiseed_cft import (
    collect_runs, parse_run_spec, summarize_across_seeds
)


def make_rows(answers):
    rows = []
    for i, truth in enumerate(answers):
        rows.append({
            "question": f"Question {i}",
            "parse_success": True,
            "intended_correct_em": bool(truth),
            "intended_misleading_em": False,
            "hard_weight": float(bool(truth)),
        })
    return rows


def test_cross_seed_summary_tracks_training_seed_variation():
    rows = {
        (2026, "posterior-g"): make_rows([0, 1, 0, 0]),
        (2026, "posterior-g-ess"): make_rows([0, 1, 0, 1]),
        (2026, "posterior-gv"): make_rows([1, 1, 0, 1]),
        (2027, "posterior-g"): make_rows([1, 0, 1, 0]),
        (2027, "posterior-g-ess"): make_rows([1, 0, 1, 1]),
        (2027, "posterior-gv"): make_rows([0, 0, 1, 1]),
        (2028, "posterior-g"): make_rows([0, 0, 1, 0]),
        (2028, "posterior-g-ess"): make_rows([0, 0, 1, 1]),
        (2028, "posterior-gv"): make_rows([0, 1, 1, 1]),
    }
    result = summarize_across_seeds(rows)
    a = result["across_training_seeds"]["posterior-g"]
    assert a["per_seed_differences"] == pytest.approx([.5, 0, .5])
    assert a["positive_seed_count"] == 2
    assert a["zero_seed_count"] == 1
    assert a["num_training_seeds"] == 3
    assert result["per_seed"]["2026"]["contrasts"]["posterior-g"]["gained_grounded"] == 2
    assert result["per_seed"]["2026"]["contrasts"]["posterior-g"]["lost_grounded"] == 0


def test_multi_seed_refuses_incomplete_or_unaligned_arm_sets():
    rows = {
        (2026, "posterior-g"): make_rows([1, 0]),
        (2026, "posterior-gv"): make_rows([1, 1]),
        (2027, "posterior-g"): make_rows([1, 0]),
        (2027, "posterior-gv"): make_rows([0, 1]),
    }
    with pytest.raises(ValueError, match="expected exactly"):
        summarize_across_seeds(rows)
    result = summarize_across_seeds(rows, baselines=("posterior-g",))
    assert result["num_development_questions"] == 2
    rows[2027, "posterior-g"][0]["question"] = "DIFFERENT!"
    with pytest.raises(ValueError, match="question"):
        summarize_across_seeds(rows, baselines=("posterior-g",))


def test_multi_seed_requires_replicated_training_seeds():
    single = {
        (2026, "posterior-g"): make_rows([0, 1]),
        (2026, "posterior-gv"): make_rows([0, 1]),
    }
    with pytest.raises(ValueError, match="two independent"):
        summarize_across_seeds(single, baselines=("posterior-g",))


def test_multi_seed_run_parser_and_duplicate_guard():
    assert parse_run_spec("2027:posterior-gv=/foo/bar.jsonl") == (
        2027, "posterior-gv", "/foo/bar.jsonl"
    )
    with pytest.raises(ValueError, match="expected"):
        parse_run_spec("posterior-gv=/foo")
    with pytest.raises(ValueError, match="duplicate"):
        collect_runs(["2027:posterior-gv=/tmp/a", "2027:posterior-gv=/tmp/b"])
