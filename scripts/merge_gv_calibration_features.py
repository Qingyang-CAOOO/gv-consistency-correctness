#!/usr/bin/env python3
"""Merge existing GV calibration feature populations with strict question disjointness.

Intended use: reuse the historical base-model CFT training population as
*supervised calibrator training data* for protocol-v2 CFT-III. These legacy
questions must NEVER be passed into protocol-v2 CFT training; the script
enforces exclusions against the frozen new CFT and validation manifests.

This is an exploratory enlargement, not a new independent validation set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm


def question_key(row):
    return " ".join(str(row["question"]).casefold().split())


def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    keys = [question_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate question identity within source: {path}")
    return rows


def merge_sources(sources, excludes):
    forbidden = set()
    for path in excludes:
        forbidden.update(question_key(r) for r in load_jsonl(path))
    results = []
    seen = set()
    for path in sources:
        rows = load_jsonl(path)
        for row in tqdm(rows, desc=f"Merge {Path(path).name}", unit="question"):
            key = question_key(row)
            if key in forbidden:
                raise ValueError(f"calibrator source overlaps protected CFT/validation questions: {key}")
            if key in seen:
                raise ValueError(f"duplicate across calibrator sources: {key}")
            seen.add(key)
            if row.get("parse_success"):
                needed = (
                    "generator_candidate_positive_score", "generator_candidate_negative_score",
                    "p_correct_when_a", "p_correct_when_b", "intended_correct_em",
                    "intended_misleading_em",
                )
                missing = [field for field in needed if field not in row]
                if missing:
                    raise ValueError(f"{key} missing calibration fields: {missing}")
            results.append(dict(row))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True,
                        help="Repeat for each original or historical calibrated features JSONL")
    parser.add_argument("--exclude", action="append", required=True,
                        help="Repeat for protected new-CFT and calibration validation question files")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.source) < 2:
        raise ValueError("provide at least two source feature files")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite merged data: {output}")
    if output.resolve() in {Path(x).resolve() for x in args.source}:
        raise ValueError("output must not overwrite a source")
    merged = merge_sources(args.source, args.exclude)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for row in merged:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "source": args.source, "excluded_question_sources": args.exclude,
        "num_questions": len(merged),
        "num_parsed": sum(bool(r.get("parse_success")) for r in merged),
        "purpose": (
            "Supervised calibration training only; the historical CFT questions "
            "are disjoint from *new* CFT-training but are not a new/independent cohort."
        ),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
