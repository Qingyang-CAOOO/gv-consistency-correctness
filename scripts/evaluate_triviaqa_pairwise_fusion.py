#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from gvsmc.fusion import PairwiseFusionModel


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a train-fitted pairwise fusion model on held-out TriviaQA pools."
    )
    p.add_argument("--data", required=True)
    p.add_argument("--fusion-model", required=True)
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
    return float(np.mean(xs)) if xs else float("nan")


def entropy_fraction(w):
    w = np.asarray(w, dtype=float)
    if len(w) <= 1:
        return 0.0
    return float(-(w * np.log(np.clip(w, 1e-15, 1.0))).sum() / math.log(len(w)))


def ess_fraction(w):
    w = np.asarray(w, dtype=float)
    return float((1.0 / np.square(w).sum()) / len(w))


def evaluate_row(r, model):
    prior = np.asarray(r["initial_weights"], dtype=float)
    gold = np.asarray(r["candidate_gold_mask"], dtype=bool)
    M = r["pairwise_win_probabilities"]
    fused = model.fused_weights(prior, M)
    top0 = int(np.argmax(prior))
    top1 = int(np.argmax(fused))

    pair_base_nll = []
    pair_full_nll = []
    pair_base_brier = []
    pair_full_brier = []
    pair_base_acc = []
    pair_full_acc = []
    if bool(gold.any()):
        for i in np.where(gold)[0]:
            for j in np.where(~gold)[0]:
                s_g, s_v = model.pair_features(prior[int(i)], prior[int(j)], float(M[int(i)][int(j)]))
                q0 = sigmoid(model.baseline_logit(s_g))
                q1 = sigmoid(model.full_logit(s_g, s_v))
                pair_base_nll.append(-math.log(max(q0, 1e-15)))
                pair_full_nll.append(-math.log(max(q1, 1e-15)))
                pair_base_brier.append((1.0 - q0) ** 2)
                pair_full_brier.append((1.0 - q1) ** 2)
                pair_base_acc.append(float(q0 > 0.5))
                pair_full_acc.append(float(q1 > 0.5))

    initial_gold_mass = float(prior[gold].sum())
    fused_gold_mass = float(fused[gold].sum())
    return {
        "index": r["index"],
        "question": r["question"],
        "candidates": r["candidates"],
        "candidate_gold_mask": r["candidate_gold_mask"],
        "gold_coverage": bool(gold.any()),
        "pairwise_evaluable": bool(gold.any()) and bool((~gold).any()),
        "initial_weights": prior.tolist(),
        "fused_weights": fused.tolist(),
        "initial_top_index": top0,
        "fused_top_index": top1,
        "initial_top_em": bool(gold[top0]),
        "fused_top_em": bool(gold[top1]),
        "initial_gold_mass": initial_gold_mass,
        "fused_gold_mass": fused_gold_mass,
        "delta_gold_mass": fused_gold_mass - initial_gold_mass,
        "initial_ess_fraction": ess_fraction(prior),
        "fused_ess_fraction": ess_fraction(fused),
        "initial_normalized_entropy": entropy_fraction(prior),
        "fused_normalized_entropy": entropy_fraction(fused),
        "pair_base_nll": mean(pair_base_nll),
        "pair_full_nll": mean(pair_full_nll),
        "pair_base_brier": mean(pair_base_brier),
        "pair_full_brier": mean(pair_full_brier),
        "pair_base_accuracy": mean(pair_base_acc),
        "pair_full_accuracy": mean(pair_full_acc),
    }


def aggregate(rows):
    covered = [r for r in rows if r["gold_coverage"]]
    pairwise = [r for r in rows if r["pairwise_evaluable"]]
    top_correct = [r for r in covered if r["initial_top_em"]]
    top_wrong = [r for r in covered if not r["initial_top_em"]]
    return {
        "num_questions": len(rows),
        "num_covered": len(covered),
        "num_pairwise_evaluable": len(pairwise),
        "initial_top1_em": mean([float(r["initial_top_em"]) for r in rows]),
        "fused_top1_em": mean([float(r["fused_top_em"]) for r in rows]),
        "top1_em_delta": mean([float(r["fused_top_em"]) - float(r["initial_top_em"]) for r in rows]),
        "initial_gold_mass_covered": mean([r["initial_gold_mass"] for r in covered]),
        "fused_gold_mass_covered": mean([r["fused_gold_mass"] for r in covered]),
        "gold_mass_delta": mean([r["delta_gold_mass"] for r in covered]),
        "gold_mass_delta_prior_top_correct": mean([r["delta_gold_mass"] for r in top_correct]),
        "gold_mass_delta_prior_top_wrong": mean([r["delta_gold_mass"] for r in top_wrong]),
        "pair_base_nll": mean([r["pair_base_nll"] for r in pairwise]),
        "pair_full_nll": mean([r["pair_full_nll"] for r in pairwise]),
        "pair_nll_improvement": mean([r["pair_base_nll"] - r["pair_full_nll"] for r in pairwise]),
        "pair_base_brier": mean([r["pair_base_brier"] for r in pairwise]),
        "pair_full_brier": mean([r["pair_full_brier"] for r in pairwise]),
        "pair_brier_improvement": mean([r["pair_base_brier"] - r["pair_full_brier"] for r in pairwise]),
        "pair_base_accuracy": mean([r["pair_base_accuracy"] for r in pairwise]),
        "pair_full_accuracy": mean([r["pair_full_accuracy"] for r in pairwise]),
        "initial_ess_fraction": mean([r["initial_ess_fraction"] for r in rows]),
        "fused_ess_fraction": mean([r["fused_ess_fraction"] for r in rows]),
        "initial_normalized_entropy": mean([r["initial_normalized_entropy"] for r in rows]),
        "fused_normalized_entropy": mean([r["fused_normalized_entropy"] for r in rows]),
    }


def main():
    args = parse_args()
    model, metadata = PairwiseFusionModel.from_json(args.fusion_model)
    source_rows = [
        json.loads(line)
        for line in Path(args.data).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [evaluate_row(r, model) for r in source_rows]
    summary = aggregate(rows)

    keys = [
        "top1_em_delta",
        "gold_mass_delta",
        "gold_mass_delta_prior_top_correct",
        "gold_mass_delta_prior_top_wrong",
        "pair_nll_improvement",
        "pair_brier_improvement",
    ]
    rng = np.random.default_rng(args.seed)
    draws = {k: [] for k in keys}
    n = len(rows)
    for _ in range(args.bootstrap):
        sample = [rows[int(i)] for i in rng.integers(0, n, size=n)]
        a = aggregate(sample)
        for k in keys:
            if np.isfinite(a[k]):
                draws[k].append(a[k])

    ci = {}
    for k in keys:
        arr = np.asarray(draws[k], dtype=float)
        lo, hi = np.quantile(arr, [0.025, 0.975])
        ci[k] = [float(lo), float(hi)]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for r in rows:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")

    payload = {
        "fusion_model": args.fusion_model,
        "fusion_training_metadata": metadata,
        "summary": summary,
        "bootstrap_95_ci": ci,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "interpretation": (
            "Held-out conditional evidence fusion. Positive pairwise NLL/Brier improvement means "
            "the validator adds predictive information beyond the generator prior. Candidate fused "
            "weights are obtained by projecting pairwise incremental log-odds onto candidate scores "
            "and multiplying only that incremental evidence into the original generator prior."
        ),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote: {output}")
    print(f"wrote: {summary_path}")


if __name__ == "__main__":
    main()
