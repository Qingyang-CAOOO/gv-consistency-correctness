#!/usr/bin/env python3
"""Apply FROZEN calibrated q_G/q_GV models to NEW original-CFT trajectories.

Inference-time features use only generator answer likelihoods and paired-order
validator responses. Gold answers / EM truth labels do not enter features and
are *removed* from the output training JSONL. Matched ESS generator weights
are diagnostic selection weights, not calibrated probabilities.

An exact frozen expected-question file is required. Output is protected from
overwrite. This script never trains the posterior or the language model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

if __package__:
    from .audit_posterior_ess_matched import ess_fraction, sharpen_to_ess
    from .fit_cft_grounding_posterior import (
        GENERATOR_FEATURE_NAMES, VALIDATOR_FEATURE_NAMES,
        feature_vector, question_key, read_dataset,
    )
else:
    from audit_posterior_ess_matched import ess_fraction, sharpen_to_ess
    from fit_cft_grounding_posterior import (
        GENERATOR_FEATURE_NAMES, VALIDATOR_FEATURE_NAMES,
        feature_vector, question_key, read_dataset,
    )
from tqdm import tqdm


SENSITIVE_LABEL_FIELDS = {
    "gold_answers", "intended_correct_em", "intended_misleading_em",
    "grounded_pair", "candidate_gold_mask", "is_grounded",
}


def verify_exact_question_set(rows, expected_path, forbidden_paths):
    actual = [question_key(row) for row in rows]
    expected = [question_key(row) for row in read_dataset(expected_path)]
    if len(set(actual)) != len(actual):
        raise ValueError("duplicate input question identities")
    if len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError("input is not exactly the frozen CFT-train partition")
    for path in forbidden_paths:
        forbidden = {question_key(row) for row in read_dataset(path)}
        overlap = forbidden.intersection(actual)
        if overlap:
            raise ValueError(f"{len(overlap)} CFT questions overlap forbidden source {path}")


def add_posterior_weights(rows, g_model, gv_model):
    parsed = [r for r in rows if bool(r.get("parse_success", False))]
    if len(parsed) == 0:
        raise ValueError("no successfully parsed CFT pairs to weight")
    x_g = np.asarray([feature_vector(r, with_validator=False) for r in parsed])
    x_gv = np.asarray([feature_vector(r, with_validator=True) for r in parsed])
    if g_model.n_features_in_ != len(GENERATOR_FEATURE_NAMES):
        raise ValueError("frozen q_G model's feature dimension disagrees")
    if gv_model.n_features_in_ != len(GENERATOR_FEATURE_NAMES) + len(VALIDATOR_FEATURE_NAMES):
        raise ValueError("frozen q_GV model's feature dimension disagrees")
    qg = np.asarray(g_model.predict_proba(x_g)[:, 1], dtype=float)
    qgv = np.asarray(gv_model.predict_proba(x_gv)[:, 1], dtype=float)
    if not np.isfinite(qg).all() or not np.isfinite(qgv).all():
        raise ValueError("nonfinite posterior predictions")
    if (qg <= 0).any() or (qg > 1).any() or (qgv <= 0).any() or (qgv > 1).any():
        raise ValueError("posterior predictions outside (0,1]")

    gamma, qg_ess = sharpen_to_ess(qg, ess_fraction(qgv))
    output = []
    cursor = 0
    for row in tqdm(rows, desc="Apply frozen posteriors", unit="question"):
        # Hard exclusion of supervision columns before training; the models
        # above have already been scored from an explicit feature allowlist.
        rec = {k: v for k, v in row.items() if k not in SENSITIVE_LABEL_FIELDS}
        if rec.get("parse_success", False):
            rec["posterior_q_g"] = float(qg[cursor])
            rec["posterior_q_gv"] = float(qgv[cursor])
            rec["posterior_q_g_essmatched"] = float(qg_ess[cursor])
            cursor += 1
        output.append(rec)
    assert cursor == len(parsed)
    summary = {
        "num_questions": len(rows),
        "num_parsed": len(parsed),
        "num_label_fields_removed": sum(
            sum(field in r for field in SENSITIVE_LABEL_FIELDS) for r in rows
        ),
        "posterior_g_mean": float(np.mean(qg)),
        "posterior_gv_mean": float(np.mean(qgv)),
        "q_g_ess_fraction": ess_fraction(qg),
        "q_gv_ess_fraction": ess_fraction(qgv),
        "q_g_essmatched_ess_fraction": ess_fraction(qg_ess),
        "q_g_essmatching_power": float(gamma),
        "weights_definition": (
            "q_G and q_GV are frozen supervised EM-proxy correctness predictors. "
            "q_G ESS-matched weights are q_G**gamma rescaled uniformly to have "
            "max 1; they are not correctness probabilities."
        ),
    }
    return output, summary


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--expected-question-file", required=True)
    p.add_argument("--forbid-question-file", action="append", default=[])
    p.add_argument("--output", required=True)
    a = p.parse_args()
    inp, out = Path(a.data), Path(a.output)
    if inp.resolve() == out.resolve():
        raise ValueError("source data and output must differ")
    if out.exists():
        raise FileExistsError(f"refusing to overwrite frozen predictions: {out}")
    rows = read_dataset(inp)
    verify_exact_question_set(rows, a.expected_question_file, a.forbid_question_file)
    model_dir = Path(a.model_dir)
    model_summary = json.loads((model_dir / "summary.json").read_text(encoding="utf-8"))
    if tuple(model_summary["generator_feature_names"]) != GENERATOR_FEATURE_NAMES:
        raise ValueError("frozen q_G feature schema mismatch")
    if tuple(model_summary["validator_feature_names"]) != VALIDATOR_FEATURE_NAMES:
        raise ValueError("frozen q_GV feature schema mismatch")
    # Joblib files must be produced and trusted by this project's own pipeline.
    g_model = joblib.load(model_dir / "q_g_model.joblib")
    gv_model = joblib.load(model_dir / "q_gv_model.joblib")
    output_rows, summary = add_posterior_weights(rows, g_model, gv_model)
    summary.update({
        "input": str(inp),
        "input_sha256": sha256_file(inp),
        "expected_question_file": a.expected_question_file,
        "forbidden_question_files": a.forbid_question_file,
        "calibrator_dir": str(model_dir),
        "calibrator_q_g_sha256": sha256_file(model_dir / "q_g_model.joblib"),
        "calibrator_q_gv_sha256": sha256_file(model_dir / "q_gv_model.joblib"),
        "gold_blind": True,
        "no_refit": True,
        "training_eligibility": "only the frozen NEW CFT-train question partition",
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False)+"\n")
    out.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
