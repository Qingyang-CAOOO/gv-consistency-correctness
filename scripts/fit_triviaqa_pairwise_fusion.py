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

from gvsmc.fusion import PairwiseFusionModel, clipped_logit


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Fit generator-only and generator+validator pairwise calibration models on "
            "gold/non-gold TriviaQA candidate pairs. Questions are equally weighted and "
            "kept intact during grouped cross-validation."
        )
    )
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--c-grid", default="0.01,0.1,1,10,100")
    p.add_argument("--eps", type=float, default=1e-6)
    return p.parse_args()


def load_examples(path: str, eps: float):
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    xs = []
    ys = []
    ws = []
    groups = []
    covered_questions = 0
    cross_pairs = 0

    for q_group, r in enumerate(rows):
        if not r.get("gold_coverage"):
            continue
        gold = np.asarray(r["candidate_gold_mask"], dtype=bool)
        prior = np.asarray(r["initial_weights"], dtype=float)
        M = r["pairwise_win_probabilities"]
        pairs = [(int(i), int(j)) for i in np.where(gold)[0] for j in np.where(~gold)[0]]
        if not pairs:
            continue
        covered_questions += 1
        cross_pairs += len(pairs)
        oriented_weight = 1.0 / (2.0 * len(pairs))

        for i, j in pairs:
            s_g = math.log(float(prior[i])) - math.log(float(prior[j]))
            p_gold = float(M[i][j])
            s_v = clipped_logit(p_gold, eps)
            xs.append([s_g, s_v])
            ys.append(1)
            ws.append(oriented_weight)
            groups.append(q_group)
            xs.append([-s_g, -s_v])
            ys.append(0)
            ws.append(oriented_weight)
            groups.append(q_group)

    if covered_questions < 2:
        raise RuntimeError("need at least two gold-covered questions")
    return (
        np.asarray(xs, dtype=float),
        np.asarray(ys, dtype=int),
        np.asarray(ws, dtype=float),
        np.asarray(groups, dtype=int),
        covered_questions,
        cross_pairs,
    )


def fit_lr(X, y, w, C):
    model = LogisticRegression(
        C=float(C),
        fit_intercept=False,
        solver="lbfgs",
        max_iter=2000,
    )
    model.fit(X, y, sample_weight=w)
    return model


def select_c(X, y, w, groups, c_grid, folds):
    unique_groups = np.unique(groups)
    n_splits = min(int(folds), len(unique_groups))
    if n_splits < 2:
        raise RuntimeError("need at least two groups for grouped cross-validation")
    splitter = GroupKFold(n_splits=n_splits)
    scores = {}
    for C in c_grid:
        fold_losses = []
        for train_idx, val_idx in splitter.split(X, y, groups):
            model = fit_lr(X[train_idx], y[train_idx], w[train_idx], C)
            p = model.predict_proba(X[val_idx])[:, 1]
            fold_losses.append(
                float(log_loss(y[val_idx], p, sample_weight=w[val_idx], labels=[0, 1]))
            )
        scores[float(C)] = float(np.mean(fold_losses))
    best_c = min(scores, key=scores.get)
    return best_c, scores


def main():
    args = parse_args()
    c_grid = [float(x) for x in args.c_grid.split(",") if x.strip()]
    X, y, w, groups, n_questions, n_pairs = load_examples(args.data, args.eps)

    base_c, base_cv = select_c(X[:, :1], y, w, groups, c_grid, args.cv_folds)
    full_c, full_cv = select_c(X, y, w, groups, c_grid, args.cv_folds)
    baseline = fit_lr(X[:, :1], y, w, base_c)
    full = fit_lr(X, y, w, full_c)

    fusion = PairwiseFusionModel(
        baseline_generator_coef=float(baseline.coef_[0, 0]),
        full_generator_coef=float(full.coef_[0, 0]),
        validator_coef=float(full.coef_[0, 1]),
        baseline_c=float(base_c),
        full_c=float(full_c),
        eps=float(args.eps),
    )

    metadata = {
        "source_data": args.data,
        "pairwise_evaluable_questions": int(n_questions),
        "gold_nongold_pairs": int(n_pairs),
        "oriented_examples": int(len(y)),
        "question_weighting": "each covered question has total weight 1",
        "symmetric_orientation": True,
        "cv_folds": min(args.cv_folds, n_questions),
        "c_grid": c_grid,
        "baseline_cv_logloss_by_c": {str(k): v for k, v in base_cv.items()},
        "full_cv_logloss_by_c": {str(k): v for k, v in full_cv.items()},
        "baseline_best_cv_logloss": float(base_cv[base_c]),
        "full_best_cv_logloss": float(full_cv[full_c]),
        "cv_logloss_improvement": float(base_cv[base_c] - full_cv[full_c]),
        "interpretation": (
            "Pairwise conditional calibration: baseline uses generator prior log-odds; full "
            "model adds order-marginalized validator log-odds. The difference in fitted "
            "logits defines incremental validator evidence conditional on generation."
        ),
    }
    fusion.to_json(args.output, metadata=metadata)

    print(json.dumps({"model": {
        "baseline_generator_coef": fusion.baseline_generator_coef,
        "full_generator_coef": fusion.full_generator_coef,
        "validator_coef": fusion.validator_coef,
        "baseline_c": fusion.baseline_c,
        "full_c": fusion.full_c,
    }, "metadata": metadata}, indent=2))
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
