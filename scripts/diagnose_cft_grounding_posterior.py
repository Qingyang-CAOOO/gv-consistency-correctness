#!/usr/bin/env python3
"""Detailed held-out correctness-calibration diagnostics on saved predictions.

Descriptive only. Do not use this repeatedly inspected validation set to
select among many models or to claim out-of-sample confirmation.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def key(text):
    return " ".join(text.casefold().split())


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    keys = [key(str(r["question"])) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate question identity in {path}")
    return {key(str(r["question"])): r for r in rows}


def label_state(r):
    return f"{int(bool(r['intended_correct_em']))}{int(bool(r['intended_misleading_em']))}"


def reliability_bins(rows, method, n_bins=5):
    ranked = sorted(rows, key=lambda r: float(r[method]))
    buckets = np.array_split(np.arange(len(ranked)), n_bins)
    result = []
    for bucket in buckets:
        if len(bucket) == 0:
            continue
        group = [ranked[int(i)] for i in bucket]
        pred = float(np.mean([float(r[method]) for r in group]))
        obs = float(np.mean([float(r["grounded"]) for r in group]))
        result.append({"n": len(group), "predicted_mean": pred, "observed_fraction": obs,
                       "absolute_gap": abs(pred - obs)})
    return {
        "bins": result,
        "expected_calibration_error_five_quantile_bins": sum(
            r["n"] * r["absolute_gap"] for r in result
        ) / len(rows),
    }


def group_metrics(rows):
    result = {"num_parsed": len(rows)}
    for state in ("10", "01", "11", "00"):
        group = [r for r in rows if r["state"] == state]
        if not group:
            result[state] = {"count": 0}
            continue
        delta = []
        for r in group:
            y = int(r["grounded"])
            p_g = min(1-1e-9, max(1e-9, float(r["q_g"])))
            p_v = min(1-1e-9, max(1e-9, float(r["q_gv"])))
            loss_g = -math.log(p_g if y else (1-p_g))
            loss_v = -math.log(p_v if y else (1-p_v))
            delta.append(loss_v-loss_g)
        result[state] = {
            "count": len(group),
            "q_g_mean": float(np.mean([r["q_g"] for r in group])),
            "q_gv_mean": float(np.mean([r["q_gv"] for r in group])),
            "mean_gv_minus_g_nll": float(np.mean(delta)),
            "gv_better_fraction": float(np.mean(np.asarray(delta)<0)),
        }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validation-features", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--show-examples", type=int, default=6)
    a = p.parse_args()
    if a.show_examples < 0:
        raise ValueError("show-examples must be nonnegative")
    validation = read_jsonl(a.validation_features)
    preds = read_jsonl(a.predictions)
    if not set(preds).issubset(set(validation)):
        raise ValueError("some prediction questions are absent from validation features")
    expected_keys = {
        k for k,r in validation.items() if bool(r.get("parse_success",False))
    }
    if set(preds) != expected_keys:
        raise ValueError("prediction questions do not match all parsed validation rows")
    rows = []
    for k,r in preds.items():
        origin = validation[k]
        state = label_state(origin)
        truth = int(state == "10")
        if truth != int(r["grounded_pair_em_proxy"]):
            raise ValueError("saved prediction truth label disagrees with validation source")
        rows.append({
            "question": str(origin["question"]),
            "intended_correct": origin["intended_correct"],
            "intended_misleading": origin["intended_misleading"],
            "state": state,
            "grounded": truth,
            "q_g": float(r["q_g"]),
            "q_gv": float(r["q_gv"]),
            "soft_gv": float(r["order_marginalized_consistency"]),
        })
    report = {
        "n_validation_questions":len(validation),
        "n_parsed":len(rows),
        "four_state_calibration": group_metrics(rows),
        "reliability": {
            name: reliability_bins(rows, name)
            for name in ("q_g","q_gv","soft_gv")
        },
        "interpretation": (
            "Descriptive on previously examined 94-question holdout; per-state "
            "subgroups are small and not sufficient for reliable inference."
        ),
    }
    def example_score(r):
        y = r["grounded"]
        p_g = min(1-1e-9, max(1e-9, r["q_g"]))
        p_gv = min(1-1e-9, max(1e-9, r["q_gv"]))
        return (
            -math.log(p_gv if y else (1-p_gv))
            +math.log(p_g if y else (1-p_g))
        )
    sorted_rows = sorted(rows,key=example_score)
    report["examples_gv_helped_most"] = [
        {**r,"gv_minus_g_nll":example_score(r)} for r in sorted_rows[:a.show_examples]
    ]
    report["examples_gv_hurt_most"] = [
        {**r,"gv_minus_g_nll":example_score(r)} for r in sorted_rows[-a.show_examples:][::-1]
    ]
    dst = Path(a.output)
    if dst.exists():
        raise FileExistsError(f"refusing to overwrite existing diagnostic: {dst}")
    dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    print("Validation parsed:",len(rows), "of",len(validation))
    print("Per-state means and GV minus G NLL:")
    print(json.dumps(report["four_state_calibration"],indent=2))
    print("5-bin expected calibration errors:")
    for method, data in report["reliability"].items():
        print(method, data["expected_calibration_error_five_quantile_bins"])
    print("Wrote:",dst)


if __name__=="__main__":
    main()
