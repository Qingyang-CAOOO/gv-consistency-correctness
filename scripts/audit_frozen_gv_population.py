#!/usr/bin/env python3
"""Audit factual selection on one frozen, possibly both-order-scored GV population.

Run on train or evaluation JSONL; this script never loads a language model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from audit_iterative_truth_enrichment import (
    bootstrap_ci, finite_number, read_rows, summarize, tracked_metrics,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def compare_weights(rows, *, n_bootstrap, rng, progress=None):
    """Paired order-marginalized minus randomized-order selection diagnostics."""
    base = summarize(rows)
    if "order_marginalized" not in base or "soft_weight" not in base:
        return None
    fields = ("selected_grounded_precision", "precision_lift_over_prevalence",
              "truth_enrichment_ratio", "ess_fraction_parsed")
    observed = tracked_metrics(base)
    draws = {f: [] for f in fields}
    for _ in range(n_bootstrap):
        ids = rng.integers(0, len(rows), len(rows))
        resampled = tracked_metrics(summarize([rows[int(i)] for i in ids]))
        for metric in fields:
            left = resampled.get(f"order_marginalized.{metric}")
            right = resampled.get(f"soft_weight.{metric}")
            if finite_number(left) and finite_number(right):
                draws[metric].append(float(left - right))
        if progress is not None:
            progress.update(1)
    return {
        metric: {
            "marginalized_minus_sampled_order": (
                float(observed[f"order_marginalized.{metric}"] -
                      observed[f"soft_weight.{metric}"])
                if finite_number(observed.get(f"order_marginalized.{metric}"))
                and finite_number(observed.get(f"soft_weight.{metric}"))
                else None
            ),
            "bootstrap_95_ci": bootstrap_ci(draws[metric]),
        } for metric in fields
    }


def main():
    args = parse_args()
    if args.bootstrap < 0:
        raise ValueError("--bootstrap must be nonnegative")
    rows = read_rows(Path(args.data))
    observed = summarize(rows)
    metrics = tracked_metrics(observed)
    rng = np.random.default_rng(args.seed)
    draws = {f: [] for f in metrics}
    has_paired = "order_marginalized" in observed and "soft_weight" in observed
    with tqdm(total=args.bootstrap * (1 + int(has_paired)),
              desc="Frozen population audit", unit="draw",
              dynamic_ncols=True, mininterval=1.0) as progress:
        for _ in range(args.bootstrap):
            ids = rng.integers(0, len(rows), len(rows))
            sample = tracked_metrics(summarize([rows[int(i)] for i in ids]))
            for f in draws:
                if f in sample:
                    draws[f].append(sample[f])
            progress.update(1)
        paired = compare_weights(
            rows, n_bootstrap=args.bootstrap, rng=rng, progress=progress,
        ) if has_paired else None
    result = {
        "source": str(args.data),
        "summary": observed,
        "bootstrap_95_ci": {field: bootstrap_ci(xs) for field, xs in draws.items()},
        "order_marginalized_minus_sampled_order": paired,
        "seed": args.seed,
        "bootstrap": args.bootstrap,
        "truth_definition": "Parsed intended-correct EM=1, intended-misleading EM=0",
        "scope": "Within-population descriptive and question-bootstrap evidence; no model retraining",
    }
    output = Path(args.output) if args.output else Path(args.data).with_suffix(".truth_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("\nSource:", args.data)
    print("Questions:", observed["num_questions"], "Parsed:", observed["num_parsed"])
    print("Grounded given parsed:", observed["grounded_rate_given_parsed"])
    print("Factual states:", observed["factual_state_counts_given_parsed"])
    for k in ("hard_weight", "soft_weight", "order_marginalized"):
        if k in observed:
            d = observed[k]
            print(k,
                  "precision:", d["selected_grounded_precision"],
                  "enrichment:", d["truth_enrichment_ratio"],
                  "ESS:", d["ess_fraction_parsed"])
    if paired is not None:
        print("Marginalized minus sampled-order soft selection:")
        for k in ("selected_grounded_precision", "truth_enrichment_ratio", "ess_fraction_parsed"):
            print(" ", k, paired[k])
    print("Wrote:", output)


if __name__ == "__main__":
    main()
