"""Regression tests: exact false-retention subgroups and gold-blind score join."""
import json

import numpy as np
import pytest

from scripts.audit_false_retention import (
    build_population, bootstrap_contrasts, run, selection_precision, summary,
)
from scripts.audit_stage5_dynamics import ARMS, source_paths


def raw(i, correct=False, *, parse=True, hard=True):
    r = {"question": f"Fact question {i}?", "parse_success": parse}
    if parse:
        r.update({
            "intended_correct": f"c{i}", "intended_misleading": f"m{i}",
            "intended_correct_em": bool(correct),
            "intended_misleading_em": False,
            "hard_weight": float(hard),
            "soft_weight": .9 if hard else .4,
            "p_correct_when_a": .8 if hard else .3,
            "p_correct_when_b": .9 if hard else .4,
        })
    return r


def predictions(raws, *, training=False):
    out = []
    for r in raws:
        if not r["parse_success"]:
            if training:
                out.append({k: v for k, v in r.items()
                            if k not in ("intended_correct_em", "intended_misleading_em")})
            continue
        t = bool(r["intended_correct_em"])
        if training:
            x = {k: v for k, v in r.items()
                 if k not in ("intended_correct_em", "intended_misleading_em")}
            x.update({"posterior_q_g": .6 if t else .3,
                      "posterior_q_gv": .8 if t else .2,
                      "posterior_q_g_essmatched": .7 if t else .25})
        else:
            x = {"question": r["question"], "grounded_pair_em_proxy": int(t),
                 "q_g": .6 if t else .3, "q_gv": .8 if t else .2,
                 "order_marginalized_consistency": .85 if r["hard_weight"] else .35}
        out.append(x)
    return out


def test_false_retention_and_selection_mass():
    src = [raw(0, True), raw(1, False), raw(2, False, hard=False),
           raw(3, True)]
    y, w, power = build_population(src, predictions(src))
    assert len(y) == 4
    assert power is not None
    stats = summary(y, w)
    assert stats["hard_retained_count"] == 3
    assert stats["hard_false_retained_count"] == 1
    assert stats["hard_false_retained_fraction"] == pytest.approx(1 / 3)
    assert stats["methods"]["hard_gv"]["weighted_grounded_precision"] == pytest.approx(2 / 3)
    assert stats["methods"]["q_gv"]["mean_weight_hard_retained_grounded"] == pytest.approx(.8)
    assert stats["methods"]["q_gv"]["mean_weight_hard_retained_ungrounded"] == pytest.approx(.2)
    assert stats["methods"]["q_gv"]["within_hard_retained_truth_gap"] == pytest.approx(.6)
    assert stats["methods"]["q_gv"]["hard_retained_reweighted_grounded_precision"] == pytest.approx(8 / 9)
    assert stats["methods"]["q_gv"]["weighted_grounded_precision"] > stats["methods"]["hard_gv"]["weighted_grounded_precision"]


def test_no_leak_and_exact_matching_enforced():
    src = [raw(0, True), raw(1, False)]
    train = predictions(src, training=True)
    y, w, _ = build_population(src, train, gold_blind=True)
    assert len(y) == 2 and selection_precision(y, w["q_gv"]) == pytest.approx(.8)
    with pytest.raises(ValueError, match="leaked"):
        build_population(src, [{**train[0], "intended_correct_em": True}, train[1]], gold_blind=True)
    with pytest.raises(ValueError, match="candidate mismatch"):
        build_population(src, [{**train[0], "intended_correct": "wrong"}, train[1]], gold_blind=True)
    with pytest.raises(ValueError, match="truth-label mismatch"):
        build_population(src, [
            {**predictions(src)[0], "grounded_pair_em_proxy": 0}, predictions(src)[1]
        ])


def test_bootstrap_conditional_retained_gap():
    src = [raw(i, correct=i % 3 == 0, hard=i % 4 != 0) for i in range(40)]
    y, w, _ = build_population(src, predictions(src))
    contrasts = bootstrap_contrasts(y, w, 60, 42)
    gap = contrasts["q_gv_minus_q_g_within_hard_truth_gap"]
    assert gap["estimate"] == pytest.approx(.3)
    assert gap["valid_draws"] > 45
    assert len(gap["paired_question_bootstrap_95_ci"]) == 2


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_end_to_end_15_arms_and_confirmation(tmp_path):
    official = [raw(i, correct=i < 177, parse=i < 394, hard=i % 4 != 0) for i in range(512)]
    root = tmp_path
    write_jsonl(root / "official_confirmation_features_n512.jsonl", official)
    pp = root / "posterior_base_v2_confirmation"
    write_jsonl(pp / "frozen_confirmation_predictions.jsonl", predictions(official))
    (pp / "summary.json").write_text(json.dumps({
        "no_refitting": True, "num_questions": 512, "num_parsed": 394,
        "data": str(root / "official_confirmation_features_n512.jsonl"),
    }))
    for arm in ARMS:
        for t in (1, 2, 3):
            p = source_paths(root, arm, t, 2026)
            rr = [raw(i, correct=i % 3 == 0, hard=i % 4 != 0) for i in range(320)]
            write_jsonl(p["train_raw"], rr)
            write_jsonl(p["train_weights"], predictions(rr, training=True))
            p["train_weights"].with_suffix(".summary.json").write_text(json.dumps({
                "gold_blind": True, "no_refit": True,
                "calibrator_q_g_sha256": "sha1",
                "calibrator_q_gv_sha256": "sha2",
            }))
    result = run(root, bootstrap=0)
    assert result["official_confirmation"]["n_parsed"] == 394
    assert len(result["cft_train_by_arm_round"]) == 5
    assert result["cft_train_by_arm_round"]["hard"]["3"]["hard_false_retained_count"] > 0
    assert result["sources"]["same_frozen_posterior_sha256_train_rounds"] == ["sha1", "sha2"]
    assert result["official_confirmation"]["paired_question_bootstrap"] is None
    with pytest.raises(ValueError, match="only has training seed"):
        run(root, bootstrap=0, seed=2027)
