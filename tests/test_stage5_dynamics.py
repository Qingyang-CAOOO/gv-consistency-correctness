"""CPU-only regression tests for the Stage-5 post-hoc feedback audit."""
import json

import numpy as np
import pytest

from scripts.audit_stage5_dynamics import (
    ARMS, analyze, auc, paired_round_transition, source_paths,
    summarize_train, summarize_weight,
)


def row(i, *, parse=True, good=False):
    r = {
        "question": f"What is {i}?", "parse_success": parse,
        "hard_weight": int(good), "soft_weight": 0.85 if good else 0.6,
    }
    if parse:
        r.update({
            "intended_correct": f"yes-{i}", "intended_misleading": f"no-{i}",
            "intended_correct_em": good, "intended_misleading_em": False,
        })
    return r


def weighted_row(r):
    x = {
        k: v for k, v in r.items()
        if k not in ("gold_answers", "intended_correct_em",
                     "intended_misleading_em", "grounded_pair")
    }
    if r["parse_success"]:
        x.update({
            "posterior_q_g": .7 if r["intended_correct_em"] else .25,
            "posterior_q_gv": .8 if r["intended_correct_em"] else .2,
            "posterior_q_g_essmatched": .68 if r["intended_correct_em"] else .23,
        })
    return x


def test_paired_transition_decomposition():
    old = [row(0, parse=False), row(1, good=True),
           row(2), row(3, good=True)]
    new = [row(0, good=True), row(1, parse=False),
           row(2, good=True), row(3)]
    t = paired_round_transition(old, new, bootstrap=30, seed=2026)
    assert t["gained_grounded"] == 2
    assert t["lost_grounded"] == 2
    assert t["grounding_gained_on_newly_parsed"] == 1
    assert t["grounding_lost_to_parse_failure"] == 1
    assert t["grounding_gained_among_parsed_both_rounds"] == 1
    assert t["grounding_lost_among_parsed_both_rounds"] == 1
    assert t["net_grounded"] == 0
    assert len(t["paired_question_bootstrap_95_ci_pp"]) == 2


def test_summarize_weight_posterior_proper_scoring():
    x = summarize_weight(np.array([1, 0]), [.75, .25], posterior=True)
    assert x["weighted_grounded_precision"] == pytest.approx(.75)
    assert x["weight_ess_fraction"] == pytest.approx(.8)
    assert x["brier_em_proxy"] == pytest.approx(.0625)
    assert x["auroc_em_proxy"] == 1
    assert auc(np.array([1, 0]), np.array([.5, .5])) == .5


def test_train_gold_blind_enforced():
    raw = [row(0, good=True), row(1)]
    weighted = [weighted_row(r) for r in raw]
    out = summarize_train(raw, weighted)
    assert out["n_parsed"] == 2
    assert out["frozen_posterior_gv_minus_g"]["nll"] < 0
    assert out["frozen_posterior_gv_minus_g"]["brier"] < 0
    with pytest.raises(ValueError, match="leaked"):
        summarize_train(
            raw, [{**weighted[0], "intended_correct_em": True}, weighted[1]]
        )
    with pytest.raises(ValueError, match="candidate mismatch"):
        summarize_train(
            raw, [{**weighted[0], "intended_correct": "different"}, weighted[1]]
        )


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def test_all_15_paths_and_posterior_provenance(tmp_path):
    for arm in ARMS:
        for t in (1, 2, 3):
            paths = source_paths(tmp_path, arm, t, 2026)
            raw = [row(i, good=(i % 3 == 0)) for i in range(320)]
            ev = [
                row(i, good=(i % (3 + (t == 3)) == 0))
                for i in range(512)
            ]
            write_jsonl(paths["eval"], ev)
            write_jsonl(paths["train_raw"], raw)
            write_jsonl(paths["train_weights"], [weighted_row(r) for r in raw])
            paths["train_weights"].with_suffix(".summary.json").write_text(
                json.dumps({
                    "no_refit": True, "gold_blind": True,
                    "calibrator_q_g_sha256": "fake1",
                    "calibrator_q_gv_sha256": "fake2",
                })
            )
    report = analyze(tmp_path, bootstrap=12)
    assert len(report["eval_metrics"]) == 5
    assert report["eval_metrics"]["soft"]["2"]["grounded"] == 171
    assert report["paired_round_transitions"]["posterior-gv"]["2->3"]["net_grounded"] < 0
    assert report["train_selection_diagnostics"]["soft"]["3"]["n_parsed"] == 320
    assert report["posterior_hashes"]["calibrator_q_g_sha256"] == "fake1"
    with pytest.raises(ValueError, match="seed2026"):
        analyze(tmp_path, seed=2027, bootstrap=0)
