#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from gvsmc.fusion import clipped_logit


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate nested sequential pairwise calibration on held-out TriviaQA pools."
    )
    p.add_argument("--data", required=True)
    p.add_argument("--sequence-model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def mean(xs):
    xs = [float(x) for x in xs if np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def normalize(w):
    w = np.asarray(w, dtype=float)
    s = float(w.sum())
    if s <= 0.0:
        raise ValueError("weights must have positive mass")
    return w / s


def ess_fraction(w):
    w = normalize(w)
    return float((1.0 / np.square(w).sum()) / len(w))


def entropy_fraction(w):
    w = normalize(w)
    if len(w) <= 1:
        return 0.0
    return float(-(w * np.log(np.clip(w, 1e-15, 1.0))).sum() / math.log(len(w)))


def stage_eta(coef, features):
    return float(np.dot(np.asarray(coef, dtype=float), np.asarray(features[:len(coef)], dtype=float)))


def evaluate_row(r, model):
    eps = float(model["eps"])
    stages = model["stages"]
    n_templates = int(model["num_templates"])
    matrices = r["pairwise_win_probabilities_by_template"]
    if len(matrices) != n_templates:
        raise ValueError("template count mismatch between data and sequence model")

    prior = normalize(r["initial_weights"])
    gold = np.asarray(r["candidate_gold_mask"], dtype=bool)
    k = len(prior)
    pair_features = [[None for _ in range(k)] for _ in range(k)]
    for i in range(k):
        for j in range(i + 1, k):
            feat = [math.log(float(prior[i])) - math.log(float(prior[j]))]
            feat.extend(clipped_logit(float(matrices[t][i][j]), eps) for t in range(n_templates))
            pair_features[i][j] = feat
            pair_features[j][i] = [-x for x in feat]

    weights_by_step = [prior.copy()]
    for t in range(1, n_templates + 1):
        prev_coef = stages[t - 1]["coef"]
        curr_coef = stages[t]["coef"]
        delta = np.zeros((k, k), dtype=float)
        for i in range(k):
            for j in range(i + 1, k):
                feat = pair_features[i][j]
                d = stage_eta(curr_coef, feat) - stage_eta(prev_coef, feat)
                delta[i, j] = d
                delta[j, i] = -d
        candidate_increment = delta.sum(axis=1) / k
        logw = np.log(weights_by_step[-1]) + candidate_increment
        logw -= np.max(logw)
        weights_by_step.append(normalize(np.exp(logw)))

    pair_nll = [float("nan")] * (n_templates + 1)
    pair_brier = [float("nan")] * (n_templates + 1)
    pair_acc = [float("nan")] * (n_templates + 1)
    pairwise_evaluable = bool(gold.any()) and bool((~gold).any())
    if pairwise_evaluable:
        nlls = [[] for _ in range(n_templates + 1)]
        briers = [[] for _ in range(n_templates + 1)]
        accs = [[] for _ in range(n_templates + 1)]
        for i in np.where(gold)[0]:
            for j in np.where(~gold)[0]:
                feat = pair_features[int(i)][int(j)]
                for t in range(n_templates + 1):
                    q = sigmoid(stage_eta(stages[t]["coef"], feat))
                    nlls[t].append(-math.log(max(q, 1e-15)))
                    briers[t].append((1.0 - q) ** 2)
                    accs[t].append(float(q > 0.5))
        pair_nll = [mean(x) for x in nlls]
        pair_brier = [mean(x) for x in briers]
        pair_acc = [mean(x) for x in accs]

    gold_mass = [float(w[gold].sum()) for w in weights_by_step]
    top_em = [bool(gold[int(np.argmax(w))]) for w in weights_by_step]
    return {
        "index": r["index"],
        "question": r["question"],
        "gold_coverage": bool(gold.any()),
        "pairwise_evaluable": pairwise_evaluable,
        "candidate_gold_mask": r["candidate_gold_mask"],
        "weights_by_step": [w.tolist() for w in weights_by_step],
        "gold_mass_by_step": gold_mass,
        "top_em_by_step": top_em,
        "ess_fraction_by_step": [ess_fraction(w) for w in weights_by_step],
        "normalized_entropy_by_step": [entropy_fraction(w) for w in weights_by_step],
        "pair_nll_by_stage": pair_nll,
        "pair_brier_by_stage": pair_brier,
        "pair_accuracy_by_stage": pair_acc,
    }


def aggregate(rows, n_templates):
    covered = [r for r in rows if r["gold_coverage"]]
    pairwise = [r for r in rows if r["pairwise_evaluable"]]
    out = {
        "num_questions": len(rows),
        "num_covered": len(covered),
        "num_pairwise_evaluable": len(pairwise),
        "top1_em_by_step": [mean([float(r["top_em_by_step"][t]) for r in rows]) for t in range(n_templates + 1)],
        "gold_mass_covered_by_step": [mean([r["gold_mass_by_step"][t] for r in covered]) for t in range(n_templates + 1)],
        "ess_fraction_by_step": [mean([r["ess_fraction_by_step"][t] for r in rows]) for t in range(n_templates + 1)],
        "normalized_entropy_by_step": [mean([r["normalized_entropy_by_step"][t] for r in rows]) for t in range(n_templates + 1)],
        "pair_nll_by_stage": [mean([r["pair_nll_by_stage"][t] for r in pairwise]) for t in range(n_templates + 1)],
        "pair_brier_by_stage": [mean([r["pair_brier_by_stage"][t] for r in pairwise]) for t in range(n_templates + 1)],
        "pair_accuracy_by_stage": [mean([r["pair_accuracy_by_stage"][t] for r in pairwise]) for t in range(n_templates + 1)],
    }
    out["top1_increment_by_step"] = [
        out["top1_em_by_step"][t] - out["top1_em_by_step"][t - 1]
        for t in range(1, n_templates + 1)
    ]
    out["gold_mass_increment_by_step"] = [
        out["gold_mass_covered_by_step"][t] - out["gold_mass_covered_by_step"][t - 1]
        for t in range(1, n_templates + 1)
    ]
    out["pair_nll_improvement_by_step"] = [
        out["pair_nll_by_stage"][t - 1] - out["pair_nll_by_stage"][t]
        for t in range(1, n_templates + 1)
    ]
    out["pair_brier_improvement_by_step"] = [
        out["pair_brier_by_stage"][t - 1] - out["pair_brier_by_stage"][t]
        for t in range(1, n_templates + 1)
    ]
    out["final_top1_delta"] = out["top1_em_by_step"][-1] - out["top1_em_by_step"][0]
    out["final_gold_mass_delta"] = out["gold_mass_covered_by_step"][-1] - out["gold_mass_covered_by_step"][0]
    out["final_pair_nll_improvement"] = out["pair_nll_by_stage"][0] - out["pair_nll_by_stage"][-1]
    out["final_pair_brier_improvement"] = out["pair_brier_by_stage"][0] - out["pair_brier_by_stage"][-1]
    return out


def main():
    args = parse_args()
    model = json.loads(Path(args.sequence_model).read_text(encoding="utf-8"))
    n_templates = int(model["num_templates"])
    source = [
        json.loads(line)
        for line in Path(args.data).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [evaluate_row(r, model) for r in source]
    summary = aggregate(rows, n_templates)

    scalar_keys = ["final_top1_delta", "final_gold_mass_delta", "final_pair_nll_improvement", "final_pair_brier_improvement"]
    step_keys = ["gold_mass_increment_by_step", "pair_nll_improvement_by_step", "pair_brier_improvement_by_step"]
    rng = np.random.default_rng(args.seed)
    scalar_draws = {k: [] for k in scalar_keys}
    step_draws = {k: [[] for _ in range(n_templates)] for k in step_keys}
    n = len(rows)
    for _ in range(args.bootstrap):
        sample = [rows[int(i)] for i in rng.integers(0, n, size=n)]
        a = aggregate(sample, n_templates)
        for k in scalar_keys:
            if np.isfinite(a[k]):
                scalar_draws[k].append(a[k])
        for k in step_keys:
            for t, value in enumerate(a[k]):
                if np.isfinite(value):
                    step_draws[k][t].append(value)

    ci = {}
    for k, vals in scalar_draws.items():
        lo, hi = np.quantile(np.asarray(vals, dtype=float), [0.025, 0.975])
        ci[k] = [float(lo), float(hi)]
    for k, per_step in step_draws.items():
        ci[k] = []
        for vals in per_step:
            lo, hi = np.quantile(np.asarray(vals, dtype=float), [0.025, 0.975])
            ci[k].append([float(lo), float(hi)])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for r in rows:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    payload = {
        "sequence_model": args.sequence_model,
        "summary": summary,
        "bootstrap_95_ci": ci,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "interpretation": (
            "Sequential conditional fusion. At step t, only the difference between nested "
            "calibrated logits q_t(G,V1,...,Vt) and q_{t-1}(G,V1,...,V_{t-1}) is projected "
            "to candidate evidence and applied. This conditions each new validator measurement "
            "on all earlier measurements rather than assuming conditional independence."
        ),
    }
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote: {out}")
    print(f"wrote: {summary_path}")


if __name__ == "__main__":
    main()
