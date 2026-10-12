#!/usr/bin/env python3
"""Fast five-arm, question-aligned CFT-development audit.

Primary: all-question parsed-and-grounded EM yield (GV posterior minus
generator posterior). Others are exploratory, without multiple-comparison
adjustment. Question-bootstrap CIs are CONDITIONAL ON ONE TRAINING SEED;
they do not estimate between-training-seed variability.

No GPUs, re-generation, posterior fitting or final-holdout access.
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np
from tqdm import tqdm

if __package__:
    from .audit_iterative_truth_enrichment import question_key, read_rows
    from .audit_single_round_cft import paired_grounding_transitions
else:
    from audit_iterative_truth_enrichment import question_key, read_rows
    from audit_single_round_cft import paired_grounding_transitions


DEFAULT_TARGET = "posterior-gv"
PRIMARY_BASELINE = "posterior-g"
ENDPOINTS = (
    "grounded_pair_em_all_questions",
    "intended_correct_em_all_questions",
    "parse_rate",
    "grounded_pair_em_given_parsed",
    "hard_gv_consistency_given_parsed",
)


def row_arrays(rows):
    parse = np.array([bool(r.get("parse_success", False)) for r in rows], dtype=float)
    correct = np.array(
        [bool(r.get("parse_success", False)) and bool(r.get("intended_correct_em", False))
         for r in rows], dtype=float
    )
    grounded = np.array([
        bool(r.get("parse_success", False))
        and bool(r.get("intended_correct_em", False))
        and not bool(r.get("intended_misleading_em", False))
        for r in rows
    ], dtype=float)
    accepted = np.array([
        bool(r.get("parse_success", False)) and float(r.get("hard_weight", 0)) >= 0.5
        for r in rows
    ], dtype=float)
    return {"parsed": parse, "correct": correct, "grounded": grounded,
            "hard_consistent": accepted}


def metrics(arrays, ids=None):
    """Return arrays of metric values, one for each bootstrap resample."""
    if ids is None:
        ids = np.arange(len(arrays["parsed"]), dtype=int)[None, :]
    n = ids.shape[1]
    parsed = arrays["parsed"][ids].sum(axis=1)
    correct = arrays["correct"][ids].sum(axis=1)
    grounded = arrays["grounded"][ids].sum(axis=1)
    hard = arrays["hard_consistent"][ids].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        given = np.where(parsed > 0, grounded / parsed, np.nan)
        consistency = np.where(parsed > 0, hard / parsed, np.nan)
    return {
        "grounded_pair_em_all_questions": grounded / n,
        "intended_correct_em_all_questions": correct / n,
        "parse_rate": parsed / n,
        "grounded_pair_em_given_parsed": given,
        "hard_gv_consistency_given_parsed": consistency,
    }


def exact_paired_mcnemar(gained, lost):
    """Exact two-sided conditional McNemar p-value (binomial, n discordant).

    Discordant grounding transitions are conditionally Binomial(n, 1/2)
    under the paired equality null. Unlike asymptotic chi-square tests
    this remains valid with very small discordant counts.
    """
    if not (isinstance(gained, int) and isinstance(lost, int)):
        raise TypeError("gained and lost counts must be integers")
    if gained < 0 or lost < 0:
        raise ValueError("discordant counts must be nonnegative")
    n = gained + lost
    if n == 0:
        return 1.0
    k = min(gained, lost)
    return min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n)


def five_arm_audit(data, *, target=DEFAULT_TARGET, bootstrap=5000, seed=2026):
    if bootstrap < 0:
        raise ValueError("bootstrap must be nonnegative")
    if target not in data or PRIMARY_BASELINE not in data or len(data) < 2:
        raise ValueError("target and generator baseline must both be included")
    qlists = {name: [question_key(r) for r in rows] for name, rows in data.items()}
    expected = next(iter(qlists.values()))
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("empty dataset or duplicate question identity")
    if any(keys != expected for keys in qlists.values()):
        raise ValueError("all arms must have identical question order and identities")

    matrices = {name: row_arrays(rows) for name, rows in data.items()}
    point = {name: {k: float(v[0]) for k, v in metrics(m).items()}
             for name, m in matrices.items()}
    baselines = [name for name in data if name != target]
    comparisons = {
        name: {
            "target_minus_baseline": {
                metric: float(point[target][metric] - point[name][metric])
                for metric in ENDPOINTS
            },
            "paired_grounding_transitions": paired_grounding_transitions(data[name], data[target]),
            "bootstrap_95_ci": {},
        } for name in baselines
    }

    for name in baselines:
        transitions = comparisons[name]["paired_grounding_transitions"]
        comparisons[name]["exact_mcnemar_two_sided_p"] = exact_paired_mcnemar(
            transitions["gained_grounded"], transitions["lost_grounded"]
        )

    draws = {
        name: {metric: [] for metric in ENDPOINTS}
        for name in baselines
    }
    rng = np.random.default_rng(seed)
    n = len(expected)
    with tqdm(total=bootstrap, desc="Five-arm paired bootstrap",
              unit="draw", dynamic_ncols=True, mininterval=1.0) as bar:
        for first in range(0, bootstrap, 128):
            size = min(128, bootstrap - first)
            indices = rng.integers(0, n, size=(size, n))
            sampled = {name: metrics(m, indices) for name, m in matrices.items()}
            for name in baselines:
                for metric in ENDPOINTS:
                    diff = sampled[target][metric] - sampled[name][metric]
                    draws[name][metric].extend(diff[np.isfinite(diff)].tolist())
            bar.update(size)
    for name in baselines:
        for metric in ENDPOINTS:
            values = draws[name][metric]
            comparisons[name]["bootstrap_95_ci"][metric] = (
                [float(x) for x in np.quantile(values, [0.025, 0.975])]
                if values else None
            )
    return {
        "n_questions": n,
        "target": target,
        "preregistered_primary": f"{target} minus {PRIMARY_BASELINE} all-question grounded yield",
        "arm_metrics": point,
        "comparisons": comparisons,
        "bootstrap": bootstrap,
        "seed": seed,
        "uncertainty_scope": (
            "paired question bootstrap conditional on a single fixed training seed; "
            "development questions already examined, not fresh confirmation"
        ),
        "multiplicity": (
            "GV versus G all-question grounded yield is the predeclared primary "
            "downstream contrast; other contrasts and endpoints are exploratory, "
            "with unadjusted 95% intervals and unadjusted exact McNemar p-values"
        ),
        "paired_test": (
            "Two-sided exact McNemar conditional binomial test on discordant "
            "binary all-question parse-and-grounded outcomes"
        ),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", action="append", required=True, help="NAME=/path/file.jsonl")
    p.add_argument("--target", default=DEFAULT_TARGET)
    p.add_argument("--output", required=True)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def main():
    args = parse_args()
    paths = {}
    for spec in args.arm:
        if "=" not in spec:
            raise ValueError(f"expected NAME=PATH: {spec}")
        name, path = spec.split("=", 1)
        if not name or not path or name in paths:
            raise ValueError(f"empty or duplicate arm: {spec}")
        paths[name] = path
    data = {name: read_rows(Path(path)) for name, path in paths.items()}
    audit = five_arm_audit(
        data, target=args.target, bootstrap=args.bootstrap, seed=args.seed
    )
    audit["source_files"] = paths
    dest = Path(args.output)
    if dest.exists():
        raise FileExistsError(f"refusing to overwrite {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("arm                 parsed%  grounded / all%  grounded / parsed%  hard GV%")
    for name, m in audit["arm_metrics"].items():
        print(f"{name:20s}{100*m['parse_rate']:7.2f}"
              f"{100*m['grounded_pair_em_all_questions']:17.2f}"
              f"{100*m['grounded_pair_em_given_parsed']:20.2f}"
              f"{100*m['hard_gv_consistency_given_parsed']:10.2f}")
    print("\n" + args.target + " minus baseline (grounded-pair yield):")
    for name, d in audit["comparisons"].items():
        delta = d["target_minus_baseline"]["grounded_pair_em_all_questions"]
        ci = d["bootstrap_95_ci"]["grounded_pair_em_all_questions"]
        t = d["paired_grounding_transitions"]
        pval = d["exact_mcnemar_two_sided_p"]
        print(f"  {name}: {100*delta:+.2f} pp, CI "
              f"[{100*ci[0]:+.2f}, {100*ci[1]:+.2f}] pp, "
              f"gained={t['gained_grounded']}, lost={t['lost_grounded']}, "
              f"exact McNemar p={pval:.5f}"
              if ci else f"  {name}: {100*delta:+.2f} pp, "
              f"exact McNemar p={pval:.5f}; no bootstrap CI")
    print("Wrote:", dest)


if __name__ == "__main__":
    main()
