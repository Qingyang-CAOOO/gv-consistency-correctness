#!/usr/bin/env python3
"""Freeze question-disjoint TriviaQA partitions, preserving historical subsets."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from gvsmc.data.triviaqa import load_rankalign_triviaqa_questions, rankalign_triviaqa_path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rankalign-root", default="external/rankalign")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--legacy-train-size", type=int, default=512)
    p.add_argument("--legacy-dev-size", type=int, default=512)
    p.add_argument("--cft-train-size", type=int, default=320)
    p.add_argument("--calibration-train-size", type=int, default=256)
    p.add_argument("--calibration-val-size", type=int, default=128)
    p.add_argument("--final-test-size", type=int, default=0)
    p.add_argument(
        "--historical-train-seeds",
        type=int,
        nargs="+",
        default=[142, 42],
        help="Reserve the union of all historical N=512 train question subsets before allocating new questions.",
    )
    p.add_argument("--seed", type=int, default=311)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--inspect-only",
        action="store_true",
        help="Print question counts and partition budgets, without writing any files.",
    )
    return p.parse_args()


def key(question):
    return " ".join(question.casefold().split())


def save_partition(path, rows, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; refuse to replace frozen partition without --overwrite")
    with path.open("w", encoding="utf-8") as f:
        for q in rows:
            f.write(json.dumps({"question": q.question, "gold_answers": list(q.gold_answers)}, ensure_ascii=False) + "\n")


def main():
    a = parse_args()
    for value in (a.legacy_train_size, a.legacy_dev_size, a.cft_train_size,
                  a.calibration_train_size, a.calibration_val_size):
        if value <= 0:
            raise ValueError("historical, CFT, and calibration sizes must be strictly positive")
    if a.final_test_size < 0:
        raise ValueError("final test size cannot be negative")
    if len(set(a.historical_train_seeds)) != len(a.historical_train_seeds):
        raise ValueError("duplicate historical train seeds")
    root = Path(a.output_dir)
    train_path = rankalign_triviaqa_path(a.rankalign_root, "train")
    test_path = rankalign_triviaqa_path(a.rankalign_root, "test")
    historical_train_order = load_rankalign_triviaqa_questions(train_path, seed=142)
    historical_test_order = load_rankalign_triviaqa_questions(test_path, seed=42)
    historical_by_seed = {
        str(seed): load_rankalign_triviaqa_questions(
            train_path, seed=seed, max_examples=a.legacy_train_size
        )
        for seed in a.historical_train_seeds
    }
    historical_keys = {
        key(q.question) for rows in historical_by_seed.values() for q in rows
    }

    train_keys = [key(q.question) for q in historical_train_order]
    test_keys = [key(q.question) for q in historical_test_order]
    if len(set(train_keys)) != len(train_keys) or len(set(test_keys)) != len(test_keys):
        raise ValueError("duplicate normalized question keys within a RankAlign split")
    overlap = set(train_keys) & set(test_keys)
    if overlap:
        raise ValueError(f"{len(overlap)} train/test questions overlap by normalized identity")
    requested_new_train = a.cft_train_size + a.calibration_train_size + a.calibration_val_size
    requested_train = len(historical_keys) + requested_new_train
    requested_test = a.legacy_dev_size + a.final_test_size
    budget = {
        "source_train_questions": len(historical_train_order),
        "requested_train_questions": requested_train,
        "historical_train_seeds": a.historical_train_seeds,
        "historical_union_train_questions": len(historical_keys),
        "remaining_train_after_all_historical_runs": len(historical_train_order) - len(historical_keys),
        "requested_new_train_questions": requested_new_train,
        "source_test_questions": len(historical_test_order),
        "requested_test_questions": requested_test,
        "remaining_test_after_development": len(historical_test_order) - a.legacy_dev_size,
        "requested_final_holdout_questions": a.final_test_size,
        "external_final_evaluation_required": a.final_test_size == 0,
    }
    print("Partition budget:", json.dumps(budget, indent=2), flush=True)
    if a.inspect_only:
        return
    if len(historical_train_order) < requested_train:
        raise ValueError(
            f"insufficient train questions ({len(historical_train_order)}), "
            f"requested {requested_train}; adjust --cft-train-size, "
            "--calibration-train-size or --calibration-val-size without overlapping questions"
        )
    if len(historical_test_order) < requested_test:
        raise ValueError(
            f"insufficient test questions ({len(historical_test_order)}), "
            f"requested {requested_test}; reduce --final-test-size only if a useful "
            "untouched holdout remains, or reserve a separate external evaluation dataset"
        )

    legacy = historical_train_order[:a.legacy_train_size]
    unused_train = [
        q for q in historical_train_order
        if key(q.question) not in historical_keys
    ]
    rng = random.Random(a.seed)
    rng.shuffle(unused_train)

    offset = 0
    def allocate(count):
        nonlocal offset
        selected = unused_train[offset:offset+count]
        offset += count
        return selected

    parts = {
        "legacy_train": legacy,
        "cft_train": allocate(a.cft_train_size),
        "calibration_train": allocate(a.calibration_train_size),
        "calibration_val": allocate(a.calibration_val_size),
        "development": historical_test_order[:a.legacy_dev_size],
    }
    if a.final_test_size > 0:
        parts["final_holdout"] = historical_test_order[
            a.legacy_dev_size:a.legacy_dev_size + a.final_test_size
        ]
    all_keys = [key(q.question) for rows in parts.values() for q in rows]
    if len(all_keys) != len(set(all_keys)):
        raise AssertionError("question leakage between new partitions")
    for name in ("cft_train", "calibration_train", "calibration_val"):
        if any(key(q.question) in historical_keys for q in parts[name]):
            raise AssertionError(f"{name} overlaps historical train questions")
    if not a.overwrite:
        existing = [root / f"{name}.jsonl" for name in parts]
        existing.append(root / "split_manifest.json")
        collisions = [str(p) for p in existing if p.exists()]
        if collisions:
            raise FileExistsError("refusing partial overwrite: " + ", ".join(collisions))
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, items in parts.items():
        path = root / f"{name}.jsonl"
        save_partition(path, items, a.overwrite)
        encoded = "\n".join(key(q.question) for q in items).encode("utf-8")
        files[name] = {
            "path": str(path),
            "num_questions": len(items),
            "question_sequence_sha256": hashlib.sha256(encoded).hexdigest(),
        }
    manifest = {
        "version": 1,
        "train_source": str(train_path),
        "test_source": str(test_path),
        "historical_train_seed": 142,
        "historical_train_seeds": a.historical_train_seeds,
        "historical_train_union_size": len(historical_keys),
        "historical_dev_seed": 42,
        "new_train_partition_seed": a.seed,
        "normalized_identity": "casefold + collapse whitespace",
        "source_train_questions": len(historical_train_order),
        "source_test_questions": len(historical_test_order),
        "partitions": files,
        "final_holdout_policy": (
            "External untouched evaluation is required; RankAlign test contains only 531 questions, "
            "512 of which were already used for development."
            if a.final_test_size == 0 else
            "Do not inspect final_holdout until all model choices and hyperparameters are frozen."
        ),
        "interpretation": (
            "Reserve union of historical train question subsets (seed 142 CFT and seed 42 "
            "Phase 3 calibration); allocate new CFT and calibration questions from the remainder. "
            "Previously explored RankAlign test is development only."
        ),
    }
    manifest_path = root / "split_manifest.json"
    if manifest_path.exists() and not a.overwrite:
        raise FileExistsError(f"{manifest_path} already exists")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Wrote: {manifest_path}")


if __name__ == "__main__":
    main()
