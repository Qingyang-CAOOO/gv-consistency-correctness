#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Summarize iterative hard-vs-soft consistency FT.")
    p.add_argument("--artifact-root", required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--test-limit", type=int, default=512)
    return p.parse_args()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(x):
    if x is None:
        return "NA"
    if isinstance(x, (int, float)):
        return f"{float(x):.4f}"
    return str(x)


def eval_path(root, mode, iteration, n):
    if mode == "base":
        return root / "eval" / f"base_test_n{n}.summary.json"
    return root / "eval" / mode / f"iter{iteration}_test_n{n}.summary.json"


def train_path(root, mode, iteration, iter1_adapter):
    if iteration == 1:
        return iter1_adapter / "gvsmc_training_summary.json"
    return root / "checkpoints" / mode / f"iter{iteration}" / "gvsmc_training_summary.json"


def main():
    args = parse_args()
    root = Path(args.artifact_root)
    manifest_path = root / "iterative_cft_manifest.json"
    manifest = load(manifest_path)

    print("Iterative CFT comparison")
    print("training semantics:", manifest["training_semantics"])
    print()
    header = [
        "mode", "iter", "parse", "G_EM", "grounded", "hard_C", "soft_G",
        "V_acc", "soft_ESS", "hard_keep", "order_gap", "train_sel", "train_w"
    ]
    print("\t".join(header))

    base = load(eval_path(root, "base", 0, args.test_limit))
    print("\t".join([
        "base", "0",
        fmt(base.get("parse_success_rate")),
        fmt(base.get("generator_intended_correct_em")),
        fmt(base.get("generator_grounded_pair_fraction")),
        fmt(base.get("hard_consistency")),
        fmt(base.get("mean_soft_consistency")),
        fmt(base.get("validator_accuracy_on_grounded_pairs")),
        fmt(base.get("soft_ess_fraction")),
        fmt(base.get("hard_retained_fraction")),
        fmt(base.get("paired_order_mean_abs_position_gap")),
        "NA", "NA"
    ]))

    # Training summaries are already embedded in the manifest, so no path recovery is needed.
    embedded = {
        (r.get("mode"), r.get("iteration")): r
        for r in manifest["records"]
        if r.get("mode") in {"hard", "soft"}
    }

    for mode in ("hard", "soft"):
        for iteration in range(1, args.rounds + 1):
            e = load(eval_path(root, mode, iteration, args.test_limit))
            rec = embedded.get((mode, iteration), {})
            t = rec.get("training", {})
            print("\t".join([
                mode, str(iteration),
                fmt(e.get("parse_success_rate")),
                fmt(e.get("generator_intended_correct_em")),
                fmt(e.get("generator_grounded_pair_fraction")),
                fmt(e.get("hard_consistency")),
                fmt(e.get("mean_soft_consistency")),
                fmt(e.get("validator_accuracy_on_grounded_pairs")),
                fmt(e.get("soft_ess_fraction")),
                fmt(e.get("hard_retained_fraction")),
                fmt(e.get("paired_order_mean_abs_position_gap")),
                fmt(t.get("selected_trajectories")),
                fmt(t.get("mean_sequence_weight")),
            ]))

    print()
    print("Primary interpretation targets:")
    print("  1. Does hard CFT drive hard_C upward faster but collapse hard_keep / soft_ESS?")
    print("  2. Does soft CFT preserve a larger effective training population?")
    print("  3. Do generator EM and grounded-pair quality improve, plateau, or regress across rounds?")
    print("  4. Does V accuracy track consistency, or does consistency improve without factual quality?")


if __name__ == "__main__":
    main()
