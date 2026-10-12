import json

import pytest

from scripts.merge_gv_calibration_features import merge_sources


def write_rows(path, values):
    with path.open("w", encoding="utf-8") as f:
        for item in values:
            f.write(json.dumps(item) + "\n")


def completed(question):
    return {
        "question": question,
        "parse_success": True,
        "generator_candidate_positive_score": {"per_token_logprob": -1.0, "num_tokens": 2},
        "generator_candidate_negative_score": {"per_token_logprob": -2.0, "num_tokens": 2},
        "p_correct_when_a": 0.7,
        "p_correct_when_b": 0.6,
        "intended_correct_em": True,
        "intended_misleading_em": False,
    }


def test_merge_historical_and_new_does_not_overlap_protected(tmp_path):
    old, new, cft, val = (tmp_path / n for n in ("old.jsonl", "new.jsonl", "cft.jsonl", "val.jsonl"))
    write_rows(old, [completed("Old question?")])
    write_rows(new, [completed("New question?")])
    write_rows(cft, [{"question": "Unseen CFT?"}])
    write_rows(val, [{"question": "Unseen calibration val?"}])
    merged = merge_sources([old, new], [cft, val])
    assert len(merged) == 2


def test_merge_rejects_leakage_into_new_cft_or_validation(tmp_path):
    old, new, cft = (tmp_path / n for n in ("old.jsonl", "new.jsonl", "cft.jsonl"))
    write_rows(old, [completed("Old question?")])
    write_rows(new, [completed("New question?")])
    write_rows(cft, [{"question": "  old   QUESTION?  "}])
    with pytest.raises(ValueError, match="overlaps"):
        merge_sources([old, new], [cft])


def test_merge_rejects_duplicates_between_sources(tmp_path):
    old, new, cft = (tmp_path / n for n in ("old.jsonl", "new.jsonl", "cft.jsonl"))
    write_rows(old, [completed("Old question?")])
    write_rows(new, [completed("old question?")])
    write_rows(cft, [{"question": "Unseen?"}])
    with pytest.raises(ValueError, match="duplicate across"):
        merge_sources([old, new], [cft])


def test_merge_rejects_missing_generator_feature(tmp_path):
    old, new, cft = (tmp_path / n for n in ("old.jsonl", "new.jsonl", "cft.jsonl"))
    r = completed("Old question?")
    del r["generator_candidate_positive_score"]
    write_rows(old, [r])
    write_rows(new, [completed("New question?")])
    write_rows(cft, [{"question": "Unseen?"}])
    with pytest.raises(ValueError, match="missing calibration fields"):
        merge_sources([old, new], [cft])
