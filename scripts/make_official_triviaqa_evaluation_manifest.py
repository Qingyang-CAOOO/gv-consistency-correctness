#!/usr/bin/env python3
"""Reserve fresh question-disjoint official TriviaQA validation populations.

Downloads only rc.nocontext validation Parquet (~7.34MB), excludes ALL
RankAlign train/test question identities, then writes confirmation and final
holdout JSONL for run_triviaqa_gv.py --question-file. No model is invoked.

The confirmation split is for evaluating already frozen q_G/q_GV predictors,
not hyperparameter tuning. The final split must remain untouched until the
CFT-III design is frozen. Same TriviaQA benchmark, not cross-domain evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from gvsmc.data.triviaqa import load_rankalign_triviaqa_questions, rankalign_triviaqa_path

DATASET_ID = "mandarjoshi/trivia_qa"
REVISION = "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f"
PARQUET_FILE = "rc.nocontext/validation-00000-of-00001.parquet"


def normalized(question):
    return " ".join(str(question).casefold().split())


def source_rows(local_file):
    import pyarrow.parquet as pq

    table = pq.read_table(local_file, columns=["question", "question_id", "answer"])
    for row in table.to_pylist():
        question = str(row["question"]).strip()
        answer = row.get("answer")
        aliases = answer.get("aliases", []) if isinstance(answer, dict) else []
        if not aliases and isinstance(answer, dict) and answer.get("value"):
            aliases = [answer["value"]]
        gold = list(dict.fromkeys(str(a).strip() for a in aliases if str(a).strip()))
        if question and gold:
            yield {
                "question": question,
                "gold_answers": gold,
                "official_question_id": str(row.get("question_id", "")),
                "official_answer_value": str(
                    answer.get("value") or "" if isinstance(answer, dict) else ""
                ),
            }


def frozen_partitions(candidate_rows, forbidden, confirm_size, final_size, seed):
    if confirm_size <= 0 or final_size <= 0:
        raise ValueError("confirm and final sizes must be positive")
    available = {}
    duplicate_source = 0
    overlap_count = 0
    invalid = 0
    aliases_added_from_duplicates = 0
    conflicting_question_keys = set()
    for row in candidate_rows:
        question = normalized(row.get("question", ""))
        answers = row.get("gold_answers", [])
        if not question or not answers:
            invalid += 1
            continue
        if question in forbidden:
            overlap_count += 1
            continue
        if question in available:
            duplicate_source += 1
            previous = available[question]
            old_id = str(previous.get("official_question_id", ""))
            new_id = str(row.get("official_question_id", ""))
            old_value = normalized(previous.get("official_answer_value", ""))
            new_value = normalized(row.get("official_answer_value", ""))
            if (old_id and new_id and old_id != new_id) or (
                old_value and new_value and old_value != new_value
            ):
                conflicting_question_keys.add(question)
                continue
            # Multiple evidence documents often repeat the same question.
            # Preserve all documented answer aliases rather than selecting
            # only the first document's gold-answer array.
            used = {normalized(a) for a in previous["gold_answers"]}
            for answer in answers:
                alias = str(answer).strip()
                if alias and normalized(alias) not in used:
                    previous["gold_answers"].append(alias)
                    used.add(normalized(alias))
                    aliases_added_from_duplicates += 1
            continue
        available[question] = dict(row)
        available[question]["gold_answers"] = list(answers)
    # When two official rows disagree on canonical answer or question ID,
    # the factual event is ambiguous. Drop the entire question.
    for question in conflicting_question_keys:
        available.pop(question, None)
    if len(available) < confirm_size + final_size:
        raise ValueError(
            f"insufficient fresh official questions: {len(available)} after exclusion; "
            f"need {confirm_size+final_size}"
        )
    entries = [available[k] for k in sorted(available)]
    random.Random(seed).shuffle(entries)
    result = {
        "confirmation": entries[:confirm_size],
        "final_holdout": entries[confirm_size:confirm_size + final_size],
    }
    stats = {
        "eligible_questions": len(available),
        "excluded_rankalign_overlap_rows": overlap_count,
        "excluded_duplicate_source_rows": duplicate_source,
        "aliases_added_from_duplicate_rows": aliases_added_from_duplicates,
        "excluded_conflicting_gold_question_keys": len(conflicting_question_keys),
        "excluded_invalid_rows": invalid,
    }
    assert len(set(normalized(r["question"]) for group in result.values() for r in group)) == confirm_size + final_size
    return result, stats


def sha256_question_keys(rows):
    return hashlib.sha256("\n".join(
        normalized(row["question"]) for row in rows
    ).encode("utf-8")).hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rankalign-root", default="external/rankalign")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--confirm-size", type=int, default=512)
    p.add_argument("--final-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=312)
    p.add_argument("--parquet-file", default=None,
                   help="Optional downloaded Parquet path for reproducible/offline execution.")
    p.add_argument("--inspect-only", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    train = load_rankalign_triviaqa_questions(
        rankalign_triviaqa_path(a.rankalign_root, "train"), seed=142,
    )
    test = load_rankalign_triviaqa_questions(
        rankalign_triviaqa_path(a.rankalign_root, "test"), seed=42,
    )
    forbidden = {normalized(q.question) for q in train + test}
    if a.parquet_file:
        parquet = Path(a.parquet_file)
    else:
        from huggingface_hub import hf_hub_download
        parquet = Path(hf_hub_download(
            repo_id=DATASET_ID, repo_type="dataset",
            filename=PARQUET_FILE, revision=REVISION,
        ))
    raw_sha256 = hashlib.sha256(parquet.read_bytes()).hexdigest()
    rows, stats = frozen_partitions(
        source_rows(parquet), forbidden,
        a.confirm_size, a.final_size, a.seed,
    )
    manifest = {
        "source": DATASET_ID,
        "revision": REVISION if not a.parquet_file else "provided_local_parquet",
        "parquet": PARQUET_FILE if not a.parquet_file else str(parquet),
        "parquet_sha256": raw_sha256,
        "rankalign_all_train_plus_test_excluded": True,
        "rankalign_unique_question_identities": len(forbidden),
        "normalized_identity": "casefold, collapse whitespace",
        "seed": a.seed,
        "selection_stats": stats,
        "splits": {
            name: {"size": len(items), "question_sequence_sha256": sha256_question_keys(items)}
            for name, items in rows.items()
        },
        "holdout_policy": {
            "confirmation": (
                "For one prospective evaluation of previously frozen q_G/q_GV "
                "models; not for fitting, hyperparameters or repeated ablations."
            ),
            "final_holdout": (
                "Do not evaluate or inspect final_holdout until matched CFT-III "
                "training protocol and method choices are fixed."
            ),
        },
        "caveats": (
            "Same TriviaQA source benchmark: independent question identities, "
            "not independent domain; paraphrase/semantic duplicate filtering "
            "not established by normalized exact identity. Duplicated official "
            "question records are merged only when canonical answers and IDs "
            "do not conflict; otherwise their question identity is excluded."
        ),
    }
    print(json.dumps(manifest, indent=2), flush=True)
    if a.inspect_only:
        print("Inspection only: no split files created.")
        return
    dst = Path(a.output_dir)
    filepaths = {name: dst / f"official_{name}.jsonl" for name in rows}
    filepaths["manifest"] = dst / "official_eval_manifest.json"
    if any(path.exists() for path in filepaths.values()):
        raise FileExistsError("refusing to overwrite existing official evaluation files")
    dst.mkdir(parents=True, exist_ok=True)
    for name, entries in rows.items():
        with filepaths[name].open("w", encoding="utf-8") as f:
            for row in entries:
                f.write(json.dumps({
                    "question": row["question"],
                    "gold_answers": row["gold_answers"],
                }, ensure_ascii=False) + "\n")
    filepaths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n",encoding="utf-8")
    print("Wrote confirmation/final manifests; do not evaluate final_holdout yet.")


if __name__ == "__main__":
    main()
