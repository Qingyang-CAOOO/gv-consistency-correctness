#!/usr/bin/env python3
"""Truth-enrichment audit of saved generator-coupled TriviaQA GV trajectories.

No inference or retraining is performed. Parsing is an observed outcome, not a
negative factual label. The "truth" labels here are conservative TriviaQA EM
matches for the two designated answers, not independently adjudicated facts.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-root", required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--test-limit", type=int, default=512)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output", default=None)
    return p.parse_args()


def question_key(row):
    return " ".join(str(row["question"]).casefold().split())


def read_rows(path):
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    keys = [question_key(r) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate question keys in {path}")
    return sorted(rows, key=question_key)


def valid_rows(rows):
    return [r for r in rows if bool(r.get("parse_success", False))]


def ratio(num, den):
    return float(num / den) if den > 0 else None


def selected_truth(rows, field):
    weights = np.array([float(r[field]) for r in rows], dtype=float)
    truth = np.array([bool(r["intended_correct_em"]) and not bool(r["intended_misleading_em"]) for r in rows], dtype=float)
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError(f"invalid {field} weights")
    return ratio(float(np.dot(weights, truth)), float(weights.sum()))


def enrichment(rows, field):
    correct = [r for r in rows if r["intended_correct_em"] and not r["intended_misleading_em"]]
    not_correct = [r for r in rows if not (r["intended_correct_em"] and not r["intended_misleading_em"])]
    mean_good = ratio(sum(float(r[field]) for r in correct), len(correct))
    mean_bad = ratio(sum(float(r[field]) for r in not_correct), len(not_correct))
    if mean_good is None or mean_bad is None:
        lr = None
    elif mean_bad == 0:
        lr = None  # denominator zero: record the conditional means and leave LR undefined
    else:
        lr = mean_good / mean_bad
    return {"mean_potential_grounded": mean_good,
            "mean_potential_not_grounded": mean_bad,
            "truth_enrichment_ratio": lr}


def ess_fraction(rows, field):
    if not rows:
        return None
    w = np.array([float(r[field]) for r in rows], dtype=float)
    denom = float(np.square(w).sum())
    return ratio(float(w.sum() ** 2), len(w) * denom)


def summarize(rows):
    n = len(rows)
    if n == 0:
        raise ValueError("no questions")
    valid = valid_rows(rows)
    v = len(valid)
    truth = [bool(r["intended_correct_em"]) and not bool(r["intended_misleading_em"]) for r in valid]
    states = Counter(
        f"{int(bool(r['intended_correct_em']))}{int(bool(r['intended_misleading_em']))}"
        for r in valid
    )
    base = {
        "num_questions": n,
        "num_parsed": v,
        "parse_rate": v / n,
        "parse_failure_modes": dict(Counter(str(r.get("parse_mode", "unknown")) for r in rows if not r.get("parse_success", False))),
        "factual_state_counts_given_parsed": {x: states.get(x, 0) for x in ("10", "01", "11", "00")},
        "grounded_rate_given_parsed": ratio(sum(truth), v),
        "intended_correct_em_given_parsed": ratio(sum(bool(r["intended_correct_em"]) for r in valid), v),
        "valid_and_intended_correct_rate_all_questions": sum(bool(r["intended_correct_em"]) for r in valid) / n,
        "valid_and_grounded_pair_rate_all_questions": sum(truth) / n,
    }
    if not valid:
        return base
    for key in ("hard_weight", "sampled_hard_weight", "soft_weight"):
        if any(key not in r for r in valid):
            continue
        base[key] = {
            "selection_mass_per_parsed": float(np.mean([float(r[key]) for r in valid])),
            "selected_grounded_precision": selected_truth(valid, key),
            "selection_recall_grounded": ratio(
                sum(float(r[key]) for r, good in zip(valid, truth) if good),
                sum(truth),
            ),
            "ess_fraction_parsed": ess_fraction(valid, key),
            **enrichment(valid, key),
        }
        selected_precision = base[key]["selected_grounded_precision"]
        prevalence = base["grounded_rate_given_parsed"]
        base[key]["precision_lift_over_prevalence"] = (
            selected_precision - prevalence
            if selected_precision is not None and prevalence is not None else None
        )
    if all("p_correct_when_a" in r and "p_correct_when_b" in r for r in valid):
        enriched = [
            {**r, "_marginalized": .5 * (float(r["p_correct_when_a"]) + float(r["p_correct_when_b"]))}
            for r in valid
        ]
        base["order_marginalized"] = {
            "selection_mass_per_parsed": float(np.mean([r["_marginalized"] for r in enriched])),
            "selected_grounded_precision": selected_truth(enriched, "_marginalized"),
            "selection_recall_grounded": ratio(
                sum(float(r["_marginalized"]) for r, good in zip(enriched, truth) if good),
                sum(truth),
            ),
            "ess_fraction_parsed": ess_fraction(enriched, "_marginalized"),
            "mean_abs_order_gap": float(np.mean([
                abs(float(r["p_correct_when_a"]) - float(r["p_correct_when_b"]))
                for r in enriched
            ])),
            **enrichment(enriched, "_marginalized"),
        }
        base["order_marginalized"]["precision_lift_over_prevalence"] = (
            base["order_marginalized"]["selected_grounded_precision"]
            - base["grounded_rate_given_parsed"]
            if base["order_marginalized"]["selected_grounded_precision"] is not None
            and base["grounded_rate_given_parsed"] is not None else None
        )
    return base


def finite_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def tracked_metrics(summary):
    result = {
        "parse_rate": summary["parse_rate"],
        "grounded_rate_given_parsed": summary["grounded_rate_given_parsed"],
        "valid_and_intended_correct_rate_all_questions": summary["valid_and_intended_correct_rate_all_questions"],
        "valid_and_grounded_pair_rate_all_questions": summary["valid_and_grounded_pair_rate_all_questions"],
    }
    for method in ("hard_weight", "sampled_hard_weight", "soft_weight", "order_marginalized"):
        if method in summary:
            for field in ("selected_grounded_precision", "precision_lift_over_prevalence",
                          "truth_enrichment_ratio", "ess_fraction_parsed",
                          "selection_mass_per_parsed"):
                result[f"{method}.{field}"] = summary[method][field]
    return result


def bootstrap_ci(draws):
    good = np.asarray([x for x in draws if finite_number(x)], dtype=float)
    if len(good) < 10:
        return None
    return [float(x) for x in np.quantile(good, [.025, .975])]


def main():
    args = parse_args()
    if args.bootstrap < 0 or args.rounds < 1:
        raise ValueError("bootstrap must be nonnegative and rounds positive")
    root = Path(args.artifact_root)
    targets = [("base", 0, root / "eval" / f"base_test_n{args.test_limit}.jsonl")]
    targets += [
        (mode, t, root / "eval" / mode / f"iter{t}_test_n{args.test_limit}.jsonl")
        for mode in ("hard", "soft") for t in range(1, args.rounds + 1)
    ]
    data = {(mode, t): read_rows(path) for mode, t, path in targets}
    keys_by_arm = {arm: [question_key(r) for r in rows] for arm, rows in data.items()}
    shared_questions = len(set(tuple(x) for x in keys_by_arm.values())) == 1
    result = {
        "source": {f"{mode}:{t}": str(path) for mode, t, path in targets},
        "truth_definition": "T=1 iff parsed and intended-correct EM=1 and intended-misleading EM=0; truth rates given parsed never count unparsed as wrong",
        "parse_handling": "unparsed trajectories excluded from conditional selection statistics, but included in unconditional valid-and-correct rates",
        "hard_definition": "greedy A/B validator agreement, distinct from sampled-hard Bernoulli selection",
        "question_sets_match": shared_questions,
        "bootstrap_unit": "question",
        "arms": {},
        "paired_differences": {},
        "within_checkpoint_order_differences": {},
    }
    rng = np.random.default_rng(args.seed)
    # Every bootstrap draw increments this one overall progress bar. The
    # displayed remaining time is estimated from actual completed draws.
    n_paired = (2 * args.rounds + args.rounds) if shared_questions else 0
    total_draws = args.bootstrap * (len(data) + n_paired)
    progress = tqdm(
        total=total_draws, desc="Truth audit bootstrap", unit="draw",
        dynamic_ncols=True, mininterval=1.0,
    )
    for arm, rows in data.items():
        key = f"{arm[0]}:{arm[1]}"
        progress.set_postfix_str(key)
        summary = summarize(rows)
        values = tracked_metrics(summary)
        draws = {metric: [] for metric in values}
        order_metrics = (
            "selected_grounded_precision",
            "precision_lift_over_prevalence",
            "truth_enrichment_ratio",
            "ess_fraction_parsed",
        )
        order_comparisons = {
            metric: [] for metric in order_metrics
        } if "order_marginalized" in summary else {}
        for _ in range(args.bootstrap):
            ids = rng.integers(0, len(rows), len(rows))
            sampled = summarize([rows[int(i)] for i in ids])
            sampled_values = tracked_metrics(sampled)
            for metric in draws:
                if metric in sampled_values:
                    draws[metric].append(sampled_values[metric])
            for metric in order_comparisons:
                marginal = sampled_values.get(f"order_marginalized.{metric}")
                sampled_order = sampled_values.get(f"soft_weight.{metric}")
                if finite_number(marginal) and finite_number(sampled_order):
                    order_comparisons[metric].append(float(marginal - sampled_order))
            progress.update(1)
        if order_comparisons:
            result["within_checkpoint_order_differences"][key] = {
                metric: {
                    "marginalized_minus_sampled_order": (
                        float(values[f"order_marginalized.{metric}"] -
                              values[f"soft_weight.{metric}"])
                        if finite_number(values.get(f"order_marginalized.{metric}"))
                        and finite_number(values.get(f"soft_weight.{metric}")) else None
                    ),
                    "bootstrap_95_ci": bootstrap_ci(draws_),
                }
                for metric, draws_ in order_comparisons.items()
            }
        result["arms"][key] = {
            "summary": summary,
            "bootstrap_95_ci": {metric: bootstrap_ci(samples) for metric, samples in draws.items()},
        }
    if shared_questions:
        # Paired resampling preserves the relationship between model outputs
        # on the same question across different checkpoints.
        paired_targets = [
            (("base", 0), (mode, t))
            for mode in ("hard", "soft") for t in range(1, args.rounds + 1)
        ] + [
            (("hard", t), ("soft", t))
            for t in range(1, args.rounds + 1)
        ]
        for left, right in paired_targets:
            progress.set_postfix_str(f"{left[0]}:{left[1]} vs {right[0]}:{right[1]}")
            h, s = data[left], data[right]
            if len(h) != len(s):
                raise AssertionError("aligned question lengths must match")
            baseline = tracked_metrics(summarize(h))
            alternative = tracked_metrics(summarize(s))
            metrics = sorted(set(baseline) & set(alternative))
            draws = {metric: [] for metric in metrics}
            for _ in range(args.bootstrap):
                ids = rng.integers(0, len(h), len(h))
                hs = tracked_metrics(summarize([h[int(i)] for i in ids]))
                ss = tracked_metrics(summarize([s[int(i)] for i in ids]))
                for metric in metrics:
                    if finite_number(hs.get(metric)) and finite_number(ss.get(metric)):
                        draws[metric].append(float(hs[metric] - ss[metric]))
                progress.update(1)
            comparison_key = f"{left[0]}:{left[1]}_minus_{right[0]}:{right[1]}"
            result["paired_differences"][comparison_key] = {
                metric: {
                    "difference": float(baseline[metric] - alternative[metric])
                    if finite_number(baseline[metric]) and finite_number(alternative[metric]) else None,
                    "bootstrap_95_ci": bootstrap_ci(draws[metric]),
                } for metric in metrics
            }
    progress.close()
    output = Path(args.output) if args.output else root / "truth_enrichment_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("arm\tparse\tgrounded|parsed\thard_precision\tsoft_precision\tLR_hard\tLR_soft\tESS_soft")
    for arm, item in result["arms"].items():
        s = item["summary"]
        hard, soft = s.get("hard_weight", {}), s.get("soft_weight", {})
        def fmt(x):
            return f"{x:.4f}" if finite_number(x) else "NA"
        print("\t".join([
            arm, fmt(s["parse_rate"]), fmt(s["grounded_rate_given_parsed"]),
            fmt(hard.get("selected_grounded_precision")),
            fmt(soft.get("selected_grounded_precision")),
            fmt(hard.get("truth_enrichment_ratio")),
            fmt(soft.get("truth_enrichment_ratio")),
            fmt(soft.get("ess_fraction_parsed")),
        ]))
    print("Question-aligned paired bootstrap:", shared_questions)
    print("Order-marginalized minus sampled-order soft potential:")
    for arm, metrics in result["within_checkpoint_order_differences"].items():
        prec = metrics["selected_grounded_precision"]
        lr = metrics["truth_enrichment_ratio"]
        print(
            arm,
            "delta_precision=", prec["marginalized_minus_sampled_order"],
            "95%CI=", prec["bootstrap_95_ci"],
            "delta_enrichment_ratio=", lr["marginalized_minus_sampled_order"],
            "95%CI=", lr["bootstrap_95_ci"],
        )
    print("Wrote:", output)


if __name__ == "__main__":
    main()
