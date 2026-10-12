#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create reproducible Bernoulli sampled-hard masks from stored soft GV weights."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.data)
    rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    parsed = [row for row in rows if bool(row.get("parse_success", False))]
    if not parsed:
        raise ValueError("no parsed rows")

    soft = [float(row["soft_weight"]) for row in parsed]
    expected = sum(soft)
    variance = sum(g * (1.0 - g) for g in soft)
    sd = math.sqrt(variance)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"parsed trajectories: {len(parsed)}")
    print(f"expected retained mass: {expected:.6f}")
    print(f"Bernoulli retained-count sd: {sd:.6f}")

    for seed in args.seeds:
        rng = random.Random(seed)
        retained = 0
        out_rows = []
        for row in rows:
            row = dict(row)
            if bool(row.get("parse_success", False)):
                g = float(row["soft_weight"])
                sampled = float(rng.random() < g)
                row["sampled_hard_weight"] = sampled
                row["sampled_hard_mask_seed"] = seed
                retained += int(sampled)
            out_rows.append(row)

        out = out_dir / f"{src.stem}.sampled_hard_seed{seed}.jsonl"
        with out.open("w", encoding="utf-8") as handle:
            for row in out_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        summary = {
            "source": str(src),
            "seed": seed,
            "num_rows": len(rows),
            "num_parsed": len(parsed),
            "expected_retained_mass": expected,
            "retained_count_variance": variance,
            "retained_count_sd": sd,
            "sampled_retained_count": retained,
        }
        out.with_suffix(".summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(f"seed {seed}: retained {retained}; wrote {out}")


if __name__ == "__main__":
    main()
