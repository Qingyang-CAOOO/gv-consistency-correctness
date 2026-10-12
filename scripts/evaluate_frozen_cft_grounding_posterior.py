#!/usr/bin/env python3
"""Prospectively evaluate *frozen* q_G/q_GV calibrators on new GV trajectories.

Never fits or tunes classifiers. Only parses previously generated/scored
pairs, loads model artifacts from --model-dir and computes held-out metrics.
Use official_confirmation only; do not inspect official_final_holdout now.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from fit_cft_grounding_posterior import (
    GENERATOR_FEATURE_NAMES, VALIDATOR_FEATURE_NAMES,
    paired_bootstrap, predictive_metrics, read_dataset, validated_data,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args()
    if a.bootstrap < 0:
        raise ValueError("--bootstrap must be nonnegative")
    model_dir = Path(a.model_dir)
    train_summary = json.loads((model_dir / "summary.json").read_text(encoding="utf-8"))
    if tuple(train_summary["generator_feature_names"]) != GENERATOR_FEATURE_NAMES:
        raise ValueError("generator model feature schema mismatch")
    if tuple(train_summary["validator_feature_names"]) != VALIDATOR_FEATURE_NAMES:
        raise ValueError("GV model feature schema mismatch")

    rows = read_dataset(a.data)
    parsed_g, xg, yg = validated_data(rows, with_validator=False)
    parsed_gv, xgv, ygv = validated_data(rows, with_validator=True)
    if not np.array_equal(yg, ygv):
        raise AssertionError("target mismatch between measurement branches")
    if [r["question"] for r in parsed_g] != [r["question"] for r in parsed_gv]:
        raise AssertionError("validation row mismatch")
    # joblib uses pickle: load only model artifacts created by our own
    # trusted training process, never external downloaded model files.
    g = joblib.load(model_dir / "q_g_model.joblib")
    gv = joblib.load(model_dir / "q_gv_model.joblib")
    if g.n_features_in_ != len(GENERATOR_FEATURE_NAMES):
        raise ValueError("frozen generator feature count mismatch")
    if gv.n_features_in_ != len(GENERATOR_FEATURE_NAMES) + len(VALIDATOR_FEATURE_NAMES):
        raise ValueError("frozen GV feature count mismatch")
    pg = g.predict_proba(xg)[:, 1]
    pgv = gv.predict_proba(xgv)[:, 1]
    soft = np.asarray([
        .5 * (float(r["p_correct_when_a"]) + float(r["p_correct_when_b"]))
        for r in parsed_gv], dtype=float)
    out = Path(a.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite frozen test evaluation: {out}")
    out.mkdir(parents=True, exist_ok=True)
    predictions_path = out / "frozen_confirmation_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as f:
        for row, truth, prior, posterior, gval in zip(parsed_g, yg, pg, pgv, soft):
            f.write(json.dumps({
                "question": row["question"],
                "grounded_pair_em_proxy": int(truth),
                "q_g": float(prior),
                "q_gv": float(posterior),
                "order_marginalized_consistency": float(gval),
            }, ensure_ascii=False) + "\n")
    report = {
        "frozen_model_dir": str(model_dir),
        "data": str(a.data),
        "num_questions": len(rows),
        "num_parsed": len(parsed_g),
        "num_grounded": int(yg.sum()),
        "no_refitting": True,
        "generator_features": GENERATOR_FEATURE_NAMES,
        "validator_features": VALIDATOR_FEATURE_NAMES,
        "metrics": {
            "generator_only": predictive_metrics(yg, pg),
            "generator_validator": predictive_metrics(yg, pgv),
            "order_marginalized_consistency": predictive_metrics(yg, soft),
            "constant_train_prior": predictive_metrics(
                yg, np.full_like(pg, float(train_summary["train_prevalence"]))
            ),
        },
        "paired_bootstrap_gv_minus_g": paired_bootstrap(yg, pg, pgv, a.bootstrap, a.seed),
        "restrictions": (
            "Frozen, no hyperparameter changes; use only prospective confirmation "
            "split now. Preserve final question split for post-training evaluation."
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
