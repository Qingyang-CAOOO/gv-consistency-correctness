#!/usr/bin/env python3
"""Paired-question development evaluation of single-round frozen-posterior CFT.

All arms must evaluate exactly the same set of questions. Question-level
bootstrap measures uncertainty *conditional on one training seed*, not
training-seed variance. Do not pass a previously untouched final test file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

if __package__:
    from .audit_iterative_truth_enrichment import (
        bootstrap_ci, finite_number, read_rows, question_key, summarize,
    )
else:
    from audit_iterative_truth_enrichment import (
        bootstrap_ci, finite_number, read_rows, question_key, summarize,
    )


def outcome_metrics(summary):
    values = {
        "parse_rate": summary["parse_rate"],
        "grounded_pair_em_given_parsed": summary["grounded_rate_given_parsed"],
        "intended_correct_em_given_parsed": summary["intended_correct_em_given_parsed"],
        "grounded_pair_em_all_questions": summary["valid_and_grounded_pair_rate_all_questions"],
        "intended_correct_em_all_questions": summary["valid_and_intended_correct_rate_all_questions"],
    }
    for name in ("hard_weight", "soft_weight"):
        if name in summary:
            values[f"{name}.acceptance_mean"] = summary[name]["selection_mass_per_parsed"]
            values[f"{name}.truth_enrichment_ratio"] = summary[name]["truth_enrichment_ratio"]
    return values


def paired_grounding_transitions(left, right):
    """Count exact question-level changes in parse-and-grounded outcome."""
    if [question_key(r) for r in left] != [question_key(r) for r in right]:
        raise ValueError("evaluation question identities or ordering differ")
    def grounded(r):
        return bool(
            r.get("parse_success", False)
            and r.get("intended_correct_em", False)
            and not r.get("intended_misleading_em", False)
        )
    old = [grounded(r) for r in left]
    new = [grounded(r) for r in right]
    return {
        "both_grounded": sum(a and b for a, b in zip(old, new)),
        "lost_grounded": sum(a and not b for a, b in zip(old, new)),
        "gained_grounded": sum(not a and b for a, b in zip(old, new)),
        "neither_grounded": sum(not a and not b for a, b in zip(old, new)),
        "net_gain": sum(new) - sum(old),
        "number_of_questions": len(left),
    }


def paired_comparison(left, right, *, bootstrap, seed):
    if [question_key(r) for r in left] != [question_key(r) for r in right]:
        raise ValueError("evaluation question identities or ordering differ")
    a = outcome_metrics(summarize(left))
    b = outcome_metrics(summarize(right))
    common = sorted(set(a) & set(b))
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in common}
    with tqdm(total=bootstrap, unit="draw", desc="Paired CFT outcomes",
              dynamic_ncols=True, mininterval=1.0) as bar:
        for _ in range(bootstrap):
            ids = rng.integers(0, len(left), len(left))
            aa = outcome_metrics(summarize([left[int(i)] for i in ids]))
            bb = outcome_metrics(summarize([right[int(i)] for i in ids]))
            for name in common:
                if finite_number(aa.get(name)) and finite_number(bb.get(name)):
                    draws[name].append(bb[name] - aa[name])
            bar.update(1)
    return {
        metric: {
            "second_minus_first": float(b[metric]-a[metric])
            if finite_number(a[metric]) and finite_number(b[metric]) else None,
            "bootstrap_95_ci": bootstrap_ci(draws[metric]),
        } for metric in common
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--arm", action="append", required=True,
        help="Repeat NAME=path/to/evaluation.jsonl (e.g. posterior-g=...).",
    )
    p.add_argument("--first", default="posterior-g")
    p.add_argument("--second", default="posterior-gv")
    p.add_argument("--output", required=True)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args()
    if a.bootstrap < 0:
        raise ValueError("bootstrap must be nonnegative")
    sources = {}
    for item in a.arm:
        if "=" not in item:
            raise ValueError(f"--arm must be NAME=PATH: {item}")
        name, path = item.split("=", 1)
        if not name or not path or name in sources:
            raise ValueError(f"invalid or duplicate arm: {item}")
        sources[name] = path
    if a.first not in sources or a.second not in sources:
        raise ValueError("--first and --second must both identify supplied arms")
    data = {name: read_rows(Path(path)) for name, path in sources.items()}
    qlists = {name: [question_key(r) for r in rows] for name, rows in data.items()}
    if len(set(tuple(keys) for keys in qlists.values())) != 1:
        raise ValueError("all training arms must be compared on exactly the same questions")
    summaries = {name: summarize(rows) for name, rows in data.items()}
    differences = paired_comparison(
        data[a.first], data[a.second], bootstrap=a.bootstrap, seed=a.seed,
    )
    transitions = paired_grounding_transitions(data[a.first], data[a.second])
    output = {
        "source": sources,
        "arms": {name: {"summary": outcome_metrics(summary),
                        "factual_states": summary["factual_state_counts_given_parsed"]}
                 for name, summary in summaries.items()},
        "comparison": f"{a.second} minus {a.first}",
        "paired_differences": differences,
        "paired_grounding_transitions": transitions,
        "seed": a.seed,
        "bootstrap_unit": "question",
        "interpretation": (
            "Development-only single-seed CFT comparison; repeat training seeds "
            "and reserve external official_final_holdout for final evaluation."
        ),
    }
    path = Path(a.output)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
