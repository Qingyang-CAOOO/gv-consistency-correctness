#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Callable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired comparison of Base/Hard-CFT/Soft-CFT generator-coupled "
            "TriviaQA GV JSONL runs."
        )
    )
    parser.add_argument("--base", required=True)
    parser.add_argument("--hard", required=True)
    parser.add_argument("--soft", required=True)
    parser.add_argument("--sampled-hard", default=None)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            index = int(row["index"])
            if index in rows:
                raise ValueError(f"duplicate index {index} in {path}")
            rows[index] = row
    return rows


def verify_alignment(runs: dict[str, dict[int, dict]]) -> list[int]:
    index_sets = {name: set(rows) for name, rows in runs.items()}
    first_name, first_set = next(iter(index_sets.items()))
    for name, index_set in index_sets.items():
        if index_set != first_set:
            raise ValueError(
                f"index mismatch: {first_name} has {len(first_set)}, "
                f"{name} has {len(index_set)}"
            )

    indices = sorted(first_set)
    for index in indices:
        questions = {runs[name][index]["question"] for name in runs}
        if len(questions) != 1:
            raise ValueError(f"question mismatch at index {index}")
    return indices


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def parsed(row: dict) -> bool:
    return bool(row.get("parse_success", False))


def parsed_metric(row: dict, key: str) -> float:
    if not parsed(row):
        raise ValueError(f"metric {key} requested for unparsed row")
    return float(row[key])


def end_to_end_metrics(rows: dict[int, dict], indices: list[int]) -> dict[str, float]:
    n = len(indices)
    parse_values = [float(parsed(rows[i])) for i in indices]
    correct = [
        float(parsed(rows[i]) and bool(rows[i].get("intended_correct_em", False)))
        for i in indices
    ]
    grounded = [
        float(parsed(rows[i]) and bool(rows[i].get("grounded_pair", False)))
        for i in indices
    ]
    consistent = [
        float(parsed(rows[i]) and float(rows[i].get("hard_weight", 0.0)) >= 0.5)
        for i in indices
    ]
    return {
        "num_questions": n,
        "parse_success_rate": mean(parse_values),
        "parsed_and_intended_correct_em_fraction": mean(correct),
        "parsed_and_grounded_pair_fraction": mean(grounded),
        "parsed_and_hard_consistent_fraction": mean(consistent),
    }


COMMON_METRICS: dict[str, Callable[[dict], float]] = {
    "intended_correct_em": lambda row: float(bool(row["intended_correct_em"])),
    "intended_misleading_em": lambda row: float(bool(row["intended_misleading_em"])),
    "grounded_pair": lambda row: float(bool(row["grounded_pair"])),
    "hard_consistency": lambda row: float(row["hard_weight"]),
    "soft_consistency": lambda row: float(row["soft_weight"]),
    "paired_order_correct_prob": lambda row: float(row["paired_order_avg_correct_prob"]),
    "abs_position_gap": lambda row: abs(float(row["paired_position_gap_a_minus_b"])),
}


def common_parsed_metrics(
    rows: dict[int, dict],
    common_indices: list[int],
) -> dict[str, float]:
    result: dict[str, float] = {"num_common_parsed": len(common_indices)}
    for metric, getter in COMMON_METRICS.items():
        result[metric] = mean([getter(rows[i]) for i in common_indices])
    return result


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return math.nan
    position = (len(sorted_values) - 1) * p
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return sorted_values[lo]
    frac = position - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def paired_bootstrap_diff(
    diffs: list[float],
    *,
    draws: int,
    rng: random.Random,
) -> dict[str, float]:
    if not diffs:
        return {"mean_diff": math.nan, "ci95_low": math.nan, "ci95_high": math.nan}
    observed = mean(diffs)
    n = len(diffs)
    samples: list[float] = []
    for _ in range(draws):
        samples.append(mean([diffs[rng.randrange(n)] for _ in range(n)]))
    samples.sort()
    return {
        "mean_diff": observed,
        "ci95_low": percentile(samples, 0.025),
        "ci95_high": percentile(samples, 0.975),
    }


def pairwise_comparison(
    name_a: str,
    rows_a: dict[int, dict],
    name_b: str,
    rows_b: dict[int, dict],
    all_indices: list[int],
    *,
    draws: int,
    rng: random.Random,
) -> dict:
    # Differences are B - A.
    all_getters: dict[str, Callable[[dict], float]] = {
        "parse_success": lambda row: float(parsed(row)),
        "parsed_and_intended_correct_em": lambda row: float(
            parsed(row) and bool(row.get("intended_correct_em", False))
        ),
        "parsed_and_grounded_pair": lambda row: float(
            parsed(row) and bool(row.get("grounded_pair", False))
        ),
        "parsed_and_hard_consistent": lambda row: float(
            parsed(row) and float(row.get("hard_weight", 0.0)) >= 0.5
        ),
    }
    end_to_end = {}
    for metric, getter in all_getters.items():
        diffs = [getter(rows_b[i]) - getter(rows_a[i]) for i in all_indices]
        end_to_end[metric] = paired_bootstrap_diff(diffs, draws=draws, rng=rng)

    common = [
        i for i in all_indices
        if parsed(rows_a[i]) and parsed(rows_b[i])
    ]
    parsed_only = {"num_common_parsed": len(common)}
    for metric, getter in COMMON_METRICS.items():
        diffs = [getter(rows_b[i]) - getter(rows_a[i]) for i in common]
        parsed_only[metric] = paired_bootstrap_diff(diffs, draws=draws, rng=rng)

    return {
        "comparison": f"{name_b} - {name_a}",
        "num_questions": len(all_indices),
        "end_to_end": end_to_end,
        "common_parsed": parsed_only,
    }


def fmt(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.4f}"


def main() -> None:
    args = parse_args()
    if args.bootstrap_draws <= 0:
        raise ValueError("--bootstrap-draws must be positive")

    runs = {
        "Base": read_jsonl(args.base),
        "Hard CFT": read_jsonl(args.hard),
        "Soft CFT": read_jsonl(args.soft),
    }
    if args.sampled_hard is not None:
        runs["Sampled-Hard CFT"] = read_jsonl(args.sampled_hard)
    indices = verify_alignment(runs)
    all_common = [
        i for i in indices
        if all(parsed(runs[name][i]) for name in runs)
    ]

    summaries = {
        name: {
            "end_to_end": end_to_end_metrics(rows, indices),
            "all_run_common_parsed": common_parsed_metrics(rows, all_common),
        }
        for name, rows in runs.items()
    }

    rng = random.Random(args.seed)
    comparison_pairs = [
        ("Base", "Hard CFT"),
        ("Base", "Soft CFT"),
        ("Hard CFT", "Soft CFT"),
    ]
    if "Sampled-Hard CFT" in runs:
        comparison_pairs.extend(
            [
                ("Base", "Sampled-Hard CFT"),
                ("Hard CFT", "Sampled-Hard CFT"),
                ("Soft CFT", "Sampled-Hard CFT"),
            ]
        )

    comparisons = [
        pairwise_comparison(
            name_a,
            runs[name_a],
            name_b,
            runs[name_b],
            indices,
            draws=args.bootstrap_draws,
            rng=rng,
        )
        for name_a, name_b in comparison_pairs
    ]

    result = {
        "num_questions": len(indices),
        "num_all_run_common_parsed": len(all_common),
        "bootstrap_draws": args.bootstrap_draws,
        "seed": args.seed,
        "runs": summaries,
        "comparisons": comparisons,
    }

    print("\nEnd-to-end metrics over the same questions")
    print(
        f"{'Run':<10} {'Parse':>8} {'Parse+EM':>10} "
        f"{'Parse+Ground':>13} {'Parse+Cons':>11}"
    )
    for name, summary in summaries.items():
        e = summary["end_to_end"]
        print(
            f"{name:<10} {fmt(e['parse_success_rate']):>8} "
            f"{fmt(e['parsed_and_intended_correct_em_fraction']):>10} "
            f"{fmt(e['parsed_and_grounded_pair_fraction']):>13} "
            f"{fmt(e['parsed_and_hard_consistent_fraction']):>11}"
        )

    print(
        f"\nAll-run common parsed intersection: "
        f"{len(all_common)}/{len(indices)} questions"
    )
    print(
        f"{'Run':<10} {'Correct':>8} {'Ground':>8} {'Hard C':>8} "
        f"{'Soft G':>8} {'PairProb':>9} {'|Gap|':>8}"
    )
    for name, summary in summaries.items():
        m = summary["all_run_common_parsed"]
        print(
            f"{name:<10} {fmt(m['intended_correct_em']):>8} "
            f"{fmt(m['grounded_pair']):>8} "
            f"{fmt(m['hard_consistency']):>8} "
            f"{fmt(m['soft_consistency']):>8} "
            f"{fmt(m['paired_order_correct_prob']):>9} "
            f"{fmt(m['abs_position_gap']):>8}"
        )

    print("\nPaired bootstrap 95% CIs (difference = second run - first run)")
    for comparison in comparisons:
        print(f"\n{comparison['comparison']}")
        for section in ("end_to_end", "common_parsed"):
            print(f"  {section}:")
            values = comparison[section]
            for metric, stats in values.items():
                if metric == "num_common_parsed":
                    print(f"    {metric}: {stats}")
                    continue
                print(
                    f"    {metric}: {fmt(stats['mean_diff'])} "
                    f"[{fmt(stats['ci95_low'])}, {fmt(stats['ci95_high'])}]"
                )

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote: {output}")


if __name__ == "__main__":
    main()
