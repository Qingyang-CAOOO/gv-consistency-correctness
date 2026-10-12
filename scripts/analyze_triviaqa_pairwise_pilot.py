#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Analyze paper-style pairwise TriviaQA pilot.")
    p.add_argument("--data", required=True)
    return p.parse_args()


def mean(xs):
    return float(np.mean(xs)) if xs else float("nan")


def median(xs):
    return float(np.median(xs)) if xs else float("nan")


def main():
    args = parse_args()
    rows = [
        json.loads(line)
        for line in Path(args.data).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError("no records")

    covered = [r for r in rows if r.get("gold_coverage")]

    gold_vs_nongold = []
    per_question_cross = []
    generator_top_pairwise = []
    other_pairwise = []
    delta_initial_top_mass = []
    same_top = []
    delta_gold_if_initial_top_correct = []
    delta_gold_if_initial_top_wrong = []
    prior_favored_probs = []
    prior_validator_concordance = []
    prior_diffs = []
    validator_diffs = []
    gold_pref_when_prior_favors_gold = []
    gold_pref_when_prior_favors_wrong = []
    cross_validator_follows_prior = []

    for r in rows:
        M = r["pairwise_win_probabilities"]
        gold = np.asarray(r["candidate_gold_mask"], dtype=bool)
        potentials = np.asarray(r["candidate_pairwise_potentials"], dtype=float)
        w0 = np.asarray(r["initial_weights"], dtype=float)
        w1 = np.asarray(r["posterior_weights"], dtype=float)
        top0 = int(r["initial_top_index"])
        top1 = int(r["posterior_top_index"])

        generator_top_pairwise.append(float(potentials[top0]))
        other_pairwise.extend([float(potentials[j]) for j in range(len(potentials)) if j != top0])
        delta_initial_top_mass.append(float(w1[top0] - w0[top0]))
        same_top.append(float(top0 == top1))

        # Pairwise agreement between the generator empirical prior and validator.
        for i in range(len(w0)):
            for j in range(i + 1, len(w0)):
                prior_diff = float(w0[i] - w0[j])
                if abs(prior_diff) < 1e-12:
                    continue
                p_i = float(M[i][j])
                validator_diff = p_i - 0.5
                prior_diffs.append(prior_diff)
                validator_diffs.append(validator_diff)
                p_prior_favored = p_i if prior_diff > 0 else 1.0 - p_i
                prior_favored_probs.append(p_prior_favored)
                prior_validator_concordance.append(float(p_prior_favored > 0.5))

        if r.get("gold_coverage"):
            # For gold/non-gold pairs, ask whether validation follows the generator
            # prior when the generator prior favors truth vs when it favors the wrong answer.
            for i in np.where(gold)[0]:
                for j in np.where(~gold)[0]:
                    p_gold = float(M[int(i)][int(j)])
                    if w0[int(i)] > w0[int(j)] + 1e-12:
                        gold_pref_when_prior_favors_gold.append(p_gold)
                        cross_validator_follows_prior.append(p_gold)
                    elif w0[int(i)] + 1e-12 < w0[int(j)]:
                        gold_pref_when_prior_favors_wrong.append(p_gold)
                        cross_validator_follows_prior.append(1.0 - p_gold)

            cross = []
            for i in np.where(gold)[0]:
                for j in np.where(~gold)[0]:
                    p = float(M[int(i)][int(j)])
                    cross.append(p)
                    gold_vs_nongold.append(p)
            if cross:
                per_question_cross.append(mean(cross))

            delta = float(r["delta_gold_mass"])
            if bool(r["initial_top_em"]):
                delta_gold_if_initial_top_correct.append(delta)
            else:
                delta_gold_if_initial_top_wrong.append(delta)

    print(f"questions={len(rows)} covered={len(covered)}")
    print()
    print("Direct semantic discrimination")
    print(f"  mean P(gold beats non-gold):       {mean(gold_vs_nongold): .6f}")
    print(f"  median P(gold beats non-gold):     {median(gold_vs_nongold): .6f}")
    print(f"  fraction cross-pairs > 0.5:        {mean([x > 0.5 for x in gold_vs_nongold]): .3f}")
    print(f"  mean per-question cross-pair prob: {mean(per_question_cross): .6f}")
    print(f"  fraction questions cross > 0.5:    {mean([x > 0.5 for x in per_question_cross]): .3f}")
    print()
    print("Generator-validator alignment")
    print(f"  mean pairwise potential of generator top: {mean(generator_top_pairwise): .6f}")
    print(f"  mean pairwise potential of other answers: {mean(other_pairwise): .6f}")
    print(f"  mean change in generator-top mass:        {mean(delta_initial_top_mass): .6f}")
    print(f"  fraction posterior top unchanged:         {mean(same_top): .3f}")
    print()
    print("Truth-conditioned effect")
    print(f"  mean delta gold mass | prior top correct: {mean(delta_gold_if_initial_top_correct): .6f}")
    print(f"  mean delta gold mass | prior top wrong:   {mean(delta_gold_if_initial_top_wrong): .6f}")
    print()
    print("Pairwise generator-validator concordance")
    print(f"  mean P(validator picks prior-favored):    {mean(prior_favored_probs): .6f}")
    print(f"  fraction prior-favored prob > 0.5:        {mean(prior_validator_concordance): .3f}")
    if len(prior_diffs) >= 2:
        corr = float(np.corrcoef(np.asarray(prior_diffs), np.asarray(validator_diffs))[0, 1])
    else:
        corr = float("nan")
    print(f"  corr(prior weight diff, validator diff):  {corr: .6f}")
    print(f"  P(gold wins) | prior favors gold:         {mean(gold_pref_when_prior_favors_gold): .6f}  (n={len(gold_pref_when_prior_favors_gold)})")
    print(f"  P(gold wins) | prior favors wrong:        {mean(gold_pref_when_prior_favors_wrong): .6f}  (n={len(gold_pref_when_prior_favors_wrong)})")
    print(f"  P(V follows G) on non-tied gold/non pairs:{mean(cross_validator_follows_prior): .6f}  (n={len(cross_validator_follows_prior)})")
    print(f"  P(V follows G) | G favors gold:           {mean(gold_pref_when_prior_favors_gold): .6f}")
    print(f"  P(V follows G) | G favors wrong:          {mean([1.0 - x for x in gold_pref_when_prior_favors_wrong]): .6f}")


if __name__ == "__main__":
    main()
