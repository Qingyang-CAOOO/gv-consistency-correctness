#!/usr/bin/env python3
"""Summarize matched CFT results across independent LoRA training seeds.

Seed is the experimental replication unit for training variation. The
same 320 training questions and 512 development questions may be used
across seeds, but they must not be counted as independent extra questions.
This produces per-seed and across-seed DESCRIPTIVE statistics, not an
anti-conservative pooled question-level significance test. With only
three training seeds, between-seed uncertainty remains substantial.

Example: --run '2026:posterior-g=PATH' --run '2026:posterior-gv=PATH'
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

if __package__:
    from .audit_five_arm_cft import row_arrays, metrics, exact_paired_mcnemar
    from .audit_single_round_cft import paired_grounding_transitions
    from .audit_iterative_truth_enrichment import question_key, read_rows
else:
    from audit_five_arm_cft import row_arrays, metrics, exact_paired_mcnemar
    from audit_single_round_cft import paired_grounding_transitions
    from audit_iterative_truth_enrichment import question_key, read_rows


def parse_run_spec(spec):
    if ":" not in spec or "=" not in spec:
        raise ValueError(f"expected SEED:ARM=PATH, got {spec!r}")
    seed, rest = spec.split(":", 1)
    arm, path = rest.split("=", 1)
    if not seed.isdecimal() or not arm or not path:
        raise ValueError(f"expected SEED:ARM=PATH, got {spec!r}")
    return int(seed), arm, path


def collect_runs(specs):
    paths = {}
    for spec in specs:
        seed, arm, path = parse_run_spec(spec)
        if (seed, arm) in paths:
            raise ValueError(f"duplicate seed/arm pair {seed}:{arm}")
        paths[seed, arm] = path
    return paths


def summarize_across_seeds(data, *, target="posterior-gv", baselines=("posterior-g", "posterior-g-ess")):
    if target in baselines or len(set(baselines)) != len(baselines):
        raise ValueError("target and distinct baselines required")
    if not data:
        raise ValueError("no model evaluations provided")
    seeds = sorted({seed for seed, _ in data})
    required = set(baselines) | {target}
    if len(seeds) < 2:
        raise ValueError("at least two independent training seeds are required")
    for seed in seeds:
        arms = {arm for s, arm in data if s == seed}
        if arms != required:
            raise ValueError(f"seed {seed}: expected exactly {sorted(required)}, got {sorted(arms)}")

    questions = None
    for (seed, arm), rows in sorted(data.items()):
        keys = [question_key(r) for r in rows]
        if not keys or len(set(keys)) != len(keys):
            raise ValueError(f"empty or duplicated question identities: {seed}:{arm}")
        if questions is None:
            questions = keys
        elif keys != questions:
            raise ValueError("evaluation questions/order differ across arms or training seeds")

    by_seed = {}
    differences = {b: [] for b in baselines}
    for seed in seeds:
        arm_metrics = {}
        for arm in sorted(required):
            values = metrics(row_arrays(data[seed, arm]))
            arm_metrics[arm] = {name: float(arr[0]) for name, arr in values.items()}
        by_seed[str(seed)] = {"arms": arm_metrics, "contrasts": {}}
        for b in baselines:
            delta = (
                arm_metrics[target]["grounded_pair_em_all_questions"]
                - arm_metrics[b]["grounded_pair_em_all_questions"]
            )
            change = paired_grounding_transitions(data[seed, b], data[seed, target])
            p = exact_paired_mcnemar(change["gained_grounded"], change["lost_grounded"])
            by_seed[str(seed)]["contrasts"][b] = {
                "gv_minus_baseline_grounded_yield": float(delta),
                "gained_grounded": change["gained_grounded"],
                "lost_grounded": change["lost_grounded"],
                "exact_mcnemar_p_descriptive": p,
            }
            differences[b].append(delta)

    across = {}
    for b, vals in differences.items():
        vals = np.asarray(vals, dtype=float)
        across[b] = {
            "per_seed_differences": [float(v) for v in vals],
            "mean_difference": float(vals.mean()),
            "sample_std_across_training_seeds": float(vals.std(ddof=1)),
            "min_difference": float(vals.min()),
            "max_difference": float(vals.max()),
            "positive_seed_count": int(np.sum(vals > 0)),
            "zero_seed_count": int(np.sum(vals == 0)),
            "negative_seed_count": int(np.sum(vals < 0)),
            "num_training_seeds": len(seeds),
        }
    return {
        "training_seeds": seeds,
        "num_development_questions": len(questions),
        "target": target,
        "baselines": list(baselines),
        "per_seed": by_seed,
        "across_training_seeds": across,
        "statistics_policy": (
            "Descriptive over independent LoRA training seeds; DO NOT treat "
            "seed x question as independent observations. For only 2-3 seeds, "
            "mean/std are unstable and not confirmatory significance tests."
        ),
        "evaluation_policy": (
            "Same previously inspected development questions and fixed "
            "decoding seeds; official final_holdout remains untouched."
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, action="append", help="SEED:ARM=FILE")
    p.add_argument("--target", default="posterior-gv")
    p.add_argument("--baseline", action="append", default=None)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    paths = collect_runs(args.run)
    data = {(seed, arm): read_rows(Path(path)) for (seed, arm), path in paths.items()}
    baselines = tuple(args.baseline or ["posterior-g", "posterior-g-ess"])
    result = summarize_across_seeds(data, target=args.target, baselines=baselines)
    result["files"] = {f"{seed}:{arm}": path for (seed, arm), path in paths.items()}
    dest = Path(args.output)
    if dest.exists():
        raise FileExistsError(f"refusing to overwrite existing analysis: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"Evaluated {len(result['training_seeds'])} training seeds on "
          f"{result['num_development_questions']} aligned development questions.")
    for name, summary in result["across_training_seeds"].items():
        print(f"{args.target} minus {name}: "
              f"mean={100*summary['mean_difference']:+.2f} pp, "
              f"seed-SD={100*summary['sample_std_across_training_seeds']:.2f} pp, "
              f"positive seeds={summary['positive_seed_count']}/{len(result['training_seeds'])}")
        for seed in result["training_seeds"]:
            contrast = result["per_seed"][str(seed)]["contrasts"][name]
            print(f"  seed={seed}: {100*contrast['gv_minus_baseline_grounded_yield']:+.2f} pp "
                  f"(gained={contrast['gained_grounded']}, "
                  f"lost={contrast['lost_grounded']})")
    print("Wrote:", dest)


if __name__ == "__main__":
    main()
