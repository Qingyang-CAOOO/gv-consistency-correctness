import numpy as np
import pytest

from scripts.audit_posterior_ess_matched import (
    calculate, ess_fraction, sharpen_to_ess,
)
from scripts.make_official_triviaqa_evaluation_manifest import (
    frozen_partitions, normalized,
)


def test_ess_matched_power_recovers_known_target():
    q = np.asarray([.12, .31, .48, .83])
    target = ess_fraction(q ** 2)
    exponent, weighted = sharpen_to_ess(q, target)
    assert exponent == pytest.approx(2.0, abs=1e-5)
    assert ess_fraction(weighted) == pytest.approx(target, abs=1e-8)


def test_ess_match_is_label_blind_and_tracks_precision():
    q = np.asarray([.1, .2, .8, .9])
    validator = np.asarray([.05, .25, .75, .95])
    y = np.asarray([0, 0, 1, 1])
    r = calculate(y, q, validator)
    assert r["q_g_matched_ess"] == pytest.approx(r["q_gv_ess"], abs=1e-8)
    assert 0 <= r["q_gv_precision"] <= 1


def test_ess_match_rejects_unattainable_uniform_weights():
    with pytest.raises(ValueError, match="cannot achieve"):
        sharpen_to_ess([.5, .5, .5, .5], .75)


def test_official_fresh_partition_excludes_rankalign_and_is_deterministic():
    rows = [
        {"question": f"Question {i}", "gold_answers": [f"a{i}"]}
        for i in range(15)
    ] + [{"question": "  QUESTION 0 ", "gold_answers": ["extra"]}]
    a, stats = frozen_partitions(rows, {normalized("Question 3")},
                                 confirm_size=5, final_size=5, seed=2026)
    b, _ = frozen_partitions(rows, {normalized("Question 3")},
                            confirm_size=5, final_size=5, seed=2026)
    assert a == b
    assert stats["excluded_rankalign_overlap_rows"] == 1
    assert stats["excluded_duplicate_source_rows"] == 1
    selected = [normalized(r["question"]) for group in a.values() for r in group]
    assert normalized("Question 3") not in selected
    assert len(selected) == len(set(selected))


def test_official_partition_refuses_insufficient_unseen_questions():
    rows = [{"question": "Q1", "gold_answers": ["A"]}]
    with pytest.raises(ValueError, match="insufficient"):
        frozen_partitions(rows, set(), confirm_size=1, final_size=1, seed=42)


def test_official_duplicate_aliases_are_unioned_without_label_conflict():
    rows = [
        {"question": "Q1", "gold_answers": ["New York"],
         "official_question_id": "qid1", "official_answer_value": "New York"},
        {"question": " Q1 ", "gold_answers": ["NYC", "new york"],
         "official_question_id": "qid1", "official_answer_value": "New York"},
        {"question": "Q2", "gold_answers": ["Paris"],
         "official_question_id": "qid2", "official_answer_value": "Paris"},
    ]
    parts, stats = frozen_partitions(
        rows, set(), confirm_size=1, final_size=1, seed=42
    )
    all_rows = parts["confirmation"] + parts["final_holdout"]
    merged_q1 = next(row for row in all_rows if normalized(row["question"]) == "q1")
    assert merged_q1["gold_answers"] == ["New York", "NYC"]
    assert stats["excluded_duplicate_source_rows"] == 1
    assert stats["aliases_added_from_duplicate_rows"] == 1
    assert stats["excluded_conflicting_gold_question_keys"] == 0


def test_official_conflicting_gold_labels_drop_entire_question():
    rows = [
        {"question": "Q1", "gold_answers": ["Rome"],
         "official_question_id": "qid1", "official_answer_value": "Rome"},
        {"question": "Q1", "gold_answers": ["Paris"],
         "official_question_id": "qid1", "official_answer_value": "Paris"},
        {"question": "Q2", "gold_answers": ["Berlin"],
         "official_question_id": "qid2", "official_answer_value": "Berlin"},
        {"question": "Q3", "gold_answers": ["Vienna"],
         "official_question_id": "qid3", "official_answer_value": "Vienna"},
    ]
    parts, stats = frozen_partitions(
        rows, set(), confirm_size=1, final_size=1, seed=42
    )
    keys = [normalized(r["question"]) for group in parts.values() for r in group]
    assert "q1" not in keys
    assert stats["excluded_conflicting_gold_question_keys"] == 1
