#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

from gvsmc.fusion import clipped_logit


def parse_args():
    p = argparse.ArgumentParser(
        description="Fit nested G,V1,...,Vt pairwise calibration models for sequential fusion."
    )
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--c-grid", default="0.01,0.1,1,10,100")
    p.add_argument("--eps", type=float, default=1e-6)
    return p.parse_args()


def load_examples(path, eps):
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError("no records")
    n_templates = len(rows[0]["pairwise_win_probabilities_by_template"])
    X, y, w, groups = [], [], [], []
    n_questions = 0
    n_pairs = 0

    for q_group, r in enumerate(rows):
        gold = np.asarray(r["candidate_gold_mask"], dtype=bool)
        if not gold.any() or not (~gold).any():
            continue
        prior = np.asarray(r["initial_weights"], dtype=float)
        matrices = r["pairwise_win_probabilities_by_template"]
        pairs = [(int(i), int(j)) for i in np.where(gold)[0] for j in np.where(~gold)[0]]
        if not pairs:
            continue
        n_questions += 1
        n_pairs += len(pairs)
        weight = 1.0 / (2.0 * len(pairs))
        for i, j in pairs:
            feat = [math.log(float(prior[i])) - math.log(float(prior[j]))]
            feat.extend(
                clipped_logit(float(matrices[t][i][j]), eps)
                for t in range(n_templates)
            )
            X.append(feat)
            y.append(1)
            w.append(weight)
            groups.append(q_group)
            X.append([-x for x in feat])
            y.append(0)
            w.append(weight)
            groups.append(q_group)

    return (
        np.asarray(X, dtype=float), np.asarray(y, dtype=int),
        np.asarray(w, dtype=float), np.asarray(groups, dtype=int),
        n_templates, n_questions, n_pairs
    )


def fit_lr(X, y, w, C):
    m = LogisticRegression(C=float(C), fit_intercept=False, solver="lbfgs", max_iter=2000)
    m.fit(X, y, sample_weight=w)
    return m


def select_c(X, y, w, groups, c_grid, folds):
    n_splits = min(int(folds), len(np.unique(groups)))
    splitter = GroupKFold(n_splits=n_splits)
    scores = {}
    for C in c_grid:
        losses = []
        for tr, va in splitter.split(X, y, groups):
            m = fit_lr(X[tr], y[tr], w[tr], C)
            p = m.predict_proba(X[va])[:, 1]
            losses.append(float(log_loss(y[va], p, sample_weight=w[va], labels=[0, 1])))
        scores[float(C)] = float(np.mean(losses))
    best = min(scores, key=scores.get)
    return best, scores


def main():
    args = parse_args()
    c_grid = [float(x) for x in args.c_grid.split(",") if x.strip()]
    X, y, w, groups, n_templates, n_questions, n_pairs = load_examples(args.data, args.eps)
    stages = []

    for stage in range(n_templates + 1):
        width = 1 + stage
        Xs = X[:, :width]
        best_c, cv = select_c(Xs, y, w, groups, c_grid, args.cv_folds)
        model = fit_lr(Xs, y, w, best_c)
        stages.append({
            "stage": stage,
            "num_validator_measurements": stage,
            "coef": [float(v) for v in model.coef_[0].tolist()],
            "c": float(best_c),
            "best_cv_logloss": float(cv[best_c]),
            "cv_logloss_by_c": {str(k): float(v) for k, v in cv.items()},
        })

    for stage in range(1, len(stages)):
        stages[stage]["incremental_cv_logloss_improvement"] = (
            stages[stage - 1]["best_cv_logloss"] - stages[stage]["best_cv_logloss"]
        )

    payload = {
        "eps": float(args.eps),
        "num_templates": int(n_templates),
        "stages": stages,
        "metadata": {
            "source_data": args.data,
            "pairwise_evaluable_questions": int(n_questions),
            "gold_nongold_pairs": int(n_pairs),
            "oriented_examples": int(len(y)),
            "question_weighting": "each pairwise-evaluable question has total weight 1",
            "symmetric_orientation": True,
            "cv_folds": min(args.cv_folds, n_questions),
            "c_grid": c_grid,
            "interpretation": (
                "Nested symmetric logistic calibration. Stage t conditions the truth comparison "
                "on generator prior odds and validator measurements V1,...,Vt. Sequential evidence "
                "is defined by the difference between adjacent stage logits."
            ),
        },
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
