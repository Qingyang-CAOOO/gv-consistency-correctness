#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description="Question-cluster bootstrap for the pairwise TriviaQA pilot."
    )
    p.add_argument("--data", required=True)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def _mean(xs):
    return float(np.mean(xs)) if xs else float("nan")


def _question_stats(r):
    M = r["pairwise_win_probabilities"]
    gold = np.asarray(r["candidate_gold_mask"], dtype=bool)
    w0 = np.asarray(r["initial_weights"], dtype=float)

    cross = []
    prior_favors_gold = []
    prior_favors_wrong = []
    prior_favored_all = []

    for i in range(len(w0)):
        for j in range(i + 1, len(w0)):
            d = float(w0[i] - w0[j])
            if abs(d) < 1e-12:
                continue
            p_i = float(M[i][j])
            prior_favored_all.append(p_i if d > 0 else 1.0 - p_i)

    if bool(gold.any()):
        for i in np.where(gold)[0]:
            for j in np.where(~gold)[0]:
                p_gold = float(M[int(i)][int(j)])
                cross.append(p_gold)
                if w0[int(i)] > w0[int(j)] + 1e-12:
                    prior_favors_gold.append(p_gold)
                elif w0[int(i)] + 1e-12 < w0[int(j)]:
                    prior_favors_wrong.append(p_gold)

    return {
        "covered": bool(r.get("gold_coverage")),
        "cross": cross,
        "prior_favors_gold": prior_favors_gold,
        "prior_favors_wrong": prior_favors_wrong,
        "prior_favored_all": prior_favored_all,
        "delta_gold": float(r["delta_gold_mass"]),
        "prior_top_correct": bool(r["initial_top_em"]),
        "top1_delta": float(r["posterior_top_em"]) - float(r["initial_top_em"]),
    }


def _flatten(stats, key):
    out = []
    for s in stats:
        out.extend(s[key])
    return out


def metrics(stats):
    covered = [s for s in stats if s["covered"]]
    top_correct = [s for s in covered if s["prior_top_correct"]]
    top_wrong = [s for s in covered if not s["prior_top_correct"]]

    cross = _flatten(covered, "cross")
    pf_gold = _flatten(covered, "prior_favors_gold")
    pf_wrong = _flatten(covered, "prior_favors_wrong")
    prior_all = _flatten(stats, "prior_favored_all")

    gold_given_g_gold = _mean(pf_gold)
    gold_given_g_wrong = _mean(pf_wrong)
    follows_given_g_wrong = _mean([1.0 - x for x in pf_wrong])

    return {
        "pair_truth_prob": _mean(cross),
        "pair_truth_minus_half": _mean(cross) - 0.5 if cross else float("nan"),
        "question_truth_prob": _mean([_mean(s["cross"]) for s in covered if s["cross"]]),
        "gold_mass_delta": _mean([s["delta_gold"] for s in covered]),
        "top1_em_delta": _mean([s["top1_delta"] for s in stats]),
        "delta_gold_prior_top_correct": _mean([s["delta_gold"] for s in top_correct]),
        "delta_gold_prior_top_wrong": _mean([s["delta_gold"] for s in top_wrong]),
        "delta_wrong_minus_correct": (
            _mean([s["delta_gold"] for s in top_wrong])
            - _mean([s["delta_gold"] for s in top_correct])
        ),
        "gv_concordance": _mean(prior_all),
        "gold_given_g_favors_gold": gold_given_g_gold,
        "gold_given_g_favors_wrong": gold_given_g_wrong,
        "gold_given_g_wrong_minus_half": (
            gold_given_g_wrong - 0.5 if pf_wrong else float("nan")
        ),
        "v_follows_g_given_g_gold": gold_given_g_gold,
        "v_follows_g_given_g_wrong": follows_given_g_wrong,
        "follow_gap_gold_minus_wrong": (
            gold_given_g_gold - follows_given_g_wrong
            if pf_gold and pf_wrong else float("nan")
        ),
        "n_questions": len(stats),
        "n_covered": len(covered),
        "n_top_correct_covered": len(top_correct),
        "n_top_wrong_covered": len(top_wrong),
        "n_cross_pairs": len(cross),
        "n_prior_favors_gold_pairs": len(pf_gold),
        "n_prior_favors_wrong_pairs": len(pf_wrong),
    }


def main():
    args = parse_args()
    rows = [
        json.loads(line)
        for line in Path(args.data).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError("no records")
    stats = [_question_stats(r) for r in rows]
    obs = metrics(stats)

    keys = [
        "pair_truth_prob",
        "pair_truth_minus_half",
        "question_truth_prob",
        "gold_mass_delta",
        "top1_em_delta",
        "delta_gold_prior_top_correct",
        "delta_gold_prior_top_wrong",
        "delta_wrong_minus_correct",
        "gv_concordance",
        "gold_given_g_favors_gold",
        "gold_given_g_favors_wrong",
        "gold_given_g_wrong_minus_half",
        "v_follows_g_given_g_gold",
        "v_follows_g_given_g_wrong",
        "follow_gap_gold_minus_wrong",
    ]

    rng = np.random.default_rng(args.seed)
    draws = {k: [] for k in keys}
    n = len(stats)
    for _ in range(args.bootstrap):
        sample = [stats[int(i)] for i in rng.integers(0, n, size=n)]
        m = metrics(sample)
        for k in keys:
            if np.isfinite(m[k]):
                draws[k].append(m[k])

    print("Question-cluster bootstrap (resample questions, preserve within-question pairs)")
    print(
        f"questions={obs['n_questions']} covered={obs['n_covered']} "
        f"covered-top-correct={obs['n_top_correct_covered']} "
        f"covered-top-wrong={obs['n_top_wrong_covered']}"
    )
    print(
        f"cross-pairs={obs['n_cross_pairs']} "
        f"G-favors-gold={obs['n_prior_favors_gold_pairs']} "
        f"G-favors-wrong={obs['n_prior_favors_wrong_pairs']}"
    )
    print(f"bootstrap={args.bootstrap} seed={args.seed}\n")

    for k in keys:
        arr = np.asarray(draws[k], dtype=float)
        lo, hi = np.quantile(arr, [0.025, 0.975])
        print(f"{k:34s} {obs[k]: .6f}  95% CI [{lo: .6f}, {hi: .6f}]")


if __name__ == "__main__":
    main()
