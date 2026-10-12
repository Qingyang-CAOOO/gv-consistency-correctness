#!/usr/bin/env python3
"""Fit q_G(T) vs q_GV(T) for parsed original CFT answer pairs.

T=1 iff the intended-correct answer matches a gold alias while intended-
misleading does not (an exact-match *proxy* for factual validity).
Fitting uses only the frozen calibration-train split, and evaluation only
the separate calibration-val split. Features never include gold or EM fields.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss, roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

GENERATOR_FEATURE_NAMES = (
    "positive_per_token_logprob",
    "negative_per_token_logprob",
    "log_positive_to_negative_token_count",
)
VALIDATOR_FEATURE_NAMES = (
    "mean_intended_correct_probability_both_orders",
    "absolute_A_B_order_gap",
)


def question_key(record):
    return " ".join(str(record["question"]).casefold().split())


def feature_vector(record, *, with_validator):
    """Explicit allowlist of gold-blind features; unchanged by label mutations."""
    pos = record["generator_candidate_positive_score"]
    neg = record["generator_candidate_negative_score"]
    np_ = int(pos["num_tokens"])
    nn = int(neg["num_tokens"])
    if min(np_, nn) <= 0:
        raise ValueError("zero candidate token length")
    features = [
        float(pos["per_token_logprob"]),
        float(neg["per_token_logprob"]),
        math.log(np_ / nn),
    ]
    if with_validator:
        a = float(record["p_correct_when_a"])
        b = float(record["p_correct_when_b"])
        if not (0 <= a <= 1 and 0 <= b <= 1):
            raise ValueError("invalid A/B probabilities")
        features.extend((0.5 * (a + b), abs(a - b)))
    if not all(math.isfinite(value) for value in features):
        raise ValueError("nonfinite feature")
    return features


def label(record):
    if not record.get("parse_success", False):
        raise ValueError("label requested for an unparsed trajectory")
    return int(bool(record["intended_correct_em"]) and not bool(record["intended_misleading_em"]))


def read_dataset(path):
    with Path(path).open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    keys = [question_key(r) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate question identities: {path}")
    return rows


def validated_data(rows, *, with_validator):
    parsed = [r for r in rows if r.get("parse_success", False)]
    x = np.asarray([feature_vector(r, with_validator=with_validator) for r in parsed],
                   dtype=np.float64)
    y = np.asarray([label(r) for r in parsed], dtype=np.int64)
    if len(parsed) == 0 or len(np.unique(y)) < 2:
        raise ValueError("calibration requires parsed rows in both binary classes")
    return parsed, x, y


def predictive_metrics(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, dtype=np.int64)
    result = {
        "nll": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "mean_probability": float(np.mean(p)),
        "base_rate": float(np.mean(y)),
        "weighted_grounded_precision": float(np.dot(y, p) / p.sum()),
        "ess_fraction": float(p.sum() ** 2 / (len(p) * np.square(p).sum())),
    }
    result["auroc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    result["average_precision"] = float(average_precision_score(y, p)) if y.sum() > 0 else None
    return result


def paired_bootstrap(y, p_g, p_gv, n_draws, seed):
    rng = np.random.default_rng(seed)
    keys = ("nll", "brier", "weighted_grounded_precision", "auroc")
    observations = {
        k: predictive_metrics(y, p_gv)[k] - predictive_metrics(y, p_g)[k]
        for k in keys
    }
    draws = {k: [] for k in keys}
    with tqdm(total=n_draws, unit="draw", desc="Posterior paired bootstrap",
              dynamic_ncols=True, mininterval=1.0) as progress:
        for _ in range(n_draws):
            ids = rng.integers(0, len(y), len(y))
            yg, pg, pgv = y[ids], p_g[ids], p_gv[ids]
            mg, mgv = predictive_metrics(yg, pg), predictive_metrics(yg, pgv)
            for k in keys:
                if mg[k] is not None and mgv[k] is not None:
                    draws[k].append(mgv[k] - mg[k])
            progress.update(1)
    return {
        k: {"gv_minus_g": observations[k],
            "bootstrap_95_ci": (
                [float(x) for x in np.quantile(v, [0.025, 0.975])]
                if len(v) >= 10 else None
            )}
        for k, v in draws.items()
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--regularization-c", type=float, default=1.0)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    if a.regularization_c <= 0 or a.bootstrap < 0:
        raise ValueError("regularization-C must be positive and bootstrap nonnegative")
    train = read_dataset(a.train)
    val = read_dataset(a.validation)
    overlap = {question_key(r) for r in train} & {question_key(r) for r in val}
    if overlap:
        raise ValueError(f"calibration train and validation overlap on {len(overlap)} questions")
    train_p, x_gt, y_t = validated_data(train, with_validator=False)
    val_p, x_gv, y_v = validated_data(val, with_validator=False)
    train_v, x_gvt, y_vt = validated_data(train, with_validator=True)
    val_v, x_gvv, y_vv = validated_data(val, with_validator=True)
    if [question_key(r) for r in train_p] != [question_key(r) for r in train_v]:
        raise AssertionError("training generator-only/GV feature rows differ")
    if [question_key(r) for r in val_p] != [question_key(r) for r in val_v]:
        raise AssertionError("validation generator-only/GV feature rows differ")
    if not np.array_equal(y_t, y_vt) or not np.array_equal(y_v, y_vv):
        raise AssertionError("calibration labels disagree")

    g_model = make_pipeline(
        StandardScaler(), LogisticRegression(C=a.regularization_c, max_iter=2000)
    )
    gv_model = make_pipeline(
        StandardScaler(), LogisticRegression(C=a.regularization_c, max_iter=2000)
    )
    g_model.fit(x_gt, y_t)
    gv_model.fit(x_gvt, y_t)
    p_g = g_model.predict_proba(x_gv)[:, 1]
    p_gv = gv_model.predict_proba(x_gvv)[:, 1]
    p_soft = np.asarray(
        [(float(r["p_correct_when_a"]) + float(r["p_correct_when_b"])) / 2
         for r in val_p], dtype=float
    )

    out = Path(a.output_dir)
    if out.exists() and any(out.iterdir()) and not a.overwrite:
        raise FileExistsError(f"{out} has files; choose a new directory")
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(g_model, out / "q_g_model.joblib")
    joblib.dump(gv_model, out / "q_gv_model.joblib")
    predictions = []
    for row, truth, pg, pgv, soft in zip(val_p, y_v, p_g, p_gv, p_soft):
        predictions.append({
            "question": row["question"], "grounded_pair_em_proxy": int(truth),
            "q_g": float(pg), "q_gv": float(pgv),
            "order_marginalized_consistency": float(soft),
        })
    with (out / "calibration_validation_predictions.jsonl").open("w", encoding="utf-8") as f:
        for rec in predictions:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summary = {
        "train_source": str(a.train), "validation_source": str(a.validation),
        "train_questions": len(train), "train_parsed": len(train_p),
        "val_questions": len(val), "val_parsed": len(val_p),
        "train_prevalence": float(y_t.mean()), "val_prevalence": float(y_v.mean()),
        "regularization_C": a.regularization_c,
        "generator_feature_names": GENERATOR_FEATURE_NAMES,
        "validator_feature_names": VALIDATOR_FEATURE_NAMES,
        "metrics": {
            "generator_only": predictive_metrics(y_v, p_g),
            "generator_validator": predictive_metrics(y_v, p_gv),
            "uncalibrated_soft_marginal": predictive_metrics(y_v, p_soft),
            "constant_train_prevalence": predictive_metrics(
                y_v, np.full_like(p_g, y_t.mean())
            ),
        },
        "paired_bootstrap_gv_minus_g": paired_bootstrap(
            y_v, p_g, p_gv, a.bootstrap, a.seed
        ),
        "interpretation": (
            "supervised-calibrated binary prediction of the grounded-pair "
            "exact-match proxy, conditional on successful parsing; no CFT run yet"
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
