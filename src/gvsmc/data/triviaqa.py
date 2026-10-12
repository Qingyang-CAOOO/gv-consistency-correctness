from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class TriviaQAPair:
    """A question paired with one correct and one misleading candidate answer."""

    question: str
    correct_answer: str
    misleading_answer: str


@dataclass(frozen=True)
class TriviaQAQuestion:
    """A TriviaQA question with answer aliases recovered from positive PN rows."""

    question: str
    gold_answers: tuple[str, ...]


def _normalize_correct_label(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"yes", "true", "correct", "1", "positive", "pos"}:
            return True
        if normalized in {"no", "false", "incorrect", "0", "negative", "neg"}:
            return False
    raise ValueError(f"cannot interpret correctness label: {value!r}")


def _candidate_answer(row: Mapping[str, Any]) -> str:
    """Extract the single candidate answer represented by a RankAlign PN row."""
    for key in ("candidate_answer", "candidate", "answer", "response"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    answers = row.get("answers")
    if isinstance(answers, str) and answers.strip():
        return answers.strip()
    if isinstance(answers, (list, tuple)):
        for value in answers:
            if isinstance(value, str) and value.strip():
                return value.strip()

    raise ValueError(f"row has no usable candidate answer; keys={sorted(row.keys())}")


def _candidate_answers(row: Mapping[str, Any]) -> list[str]:
    """Extract all answer strings represented by one RankAlign row."""
    answers: list[str] = []

    for key in ("candidate_answer", "candidate", "answer", "response"):
        value = row.get(key)
        if isinstance(value, str) and value.strip() and value.strip() not in answers:
            answers.append(value.strip())

    value = row.get("answers")
    if isinstance(value, str) and value.strip() and value.strip() not in answers:
        answers.append(value.strip())
    elif isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, str) and item.strip() and item.strip() not in answers:
                answers.append(item.strip())

    return answers


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected an object on line {line_number} of {path}")
            rows.append(value)
    return rows


def pairs_from_rankalign_rows(rows: Iterable[Mapping[str, Any]]) -> list[TriviaQAPair]:
    """Pair positive/negative RankAlign TriviaQA rows by exact question text."""
    grouped: dict[str, dict[bool, list[str]]] = defaultdict(
        lambda: {True: [], False: []}
    )

    for row in rows:
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("every TriviaQA row must contain a non-empty 'question'")
        if "correct" not in row:
            raise ValueError(
                "RankAlign PN rows must contain a 'correct' label; "
                "use data/dataset/triviaqa_pn_{train,test}.jsonl"
            )

        answer = _candidate_answer(row)
        is_correct = _normalize_correct_label(row["correct"])
        bucket = grouped[question.strip()][is_correct]
        if answer not in bucket:
            bucket.append(answer)

    pairs: list[TriviaQAPair] = []
    for question, candidates in grouped.items():
        positives = candidates[True]
        negatives = candidates[False]
        if not positives or not negatives:
            continue

        correct = positives[0]
        misleading = next((x for x in negatives if x != correct), None)
        if misleading is None:
            continue

        pairs.append(
            TriviaQAPair(
                question=question,
                correct_answer=correct,
                misleading_answer=misleading,
            )
        )

    return pairs


def questions_from_rankalign_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[TriviaQAQuestion]:
    """Recover unique questions and gold aliases from positive RankAlign PN rows.

    The PN files contain both positive and negative candidate rows. Only answers
    attached to rows labeled correct are treated as gold aliases.
    """
    grouped: dict[str, list[str]] = defaultdict(list)

    for row in rows:
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("every TriviaQA row must contain a non-empty 'question'")
        if "correct" not in row:
            raise ValueError(
                "RankAlign PN rows must contain a 'correct' label; "
                "use data/dataset/triviaqa_pn_{train,test}.jsonl"
            )
        if not _normalize_correct_label(row["correct"]):
            continue

        bucket = grouped[question.strip()]
        for answer in _candidate_answers(row):
            if answer not in bucket:
                bucket.append(answer)

    return [
        TriviaQAQuestion(question=question, gold_answers=tuple(answers))
        for question, answers in grouped.items()
        if answers
    ]


def load_rankalign_triviaqa_questions(
    path: str | Path,
    *,
    seed: int = 0,
    max_examples: int | None = None,
) -> list[TriviaQAQuestion]:
    """Load unique TriviaQA questions for a generator-coupled GV experiment."""
    questions = questions_from_rankalign_rows(read_jsonl(path))
    rng = random.Random(seed)
    rng.shuffle(questions)

    if max_examples is not None:
        if max_examples < 0:
            raise ValueError("max_examples must be nonnegative")
        questions = questions[:max_examples]

    if not questions:
        raise ValueError(f"no TriviaQA questions with gold answers could be built from {path}")
    return questions


def load_rankalign_triviaqa_pairs(
    path: str | Path,
    *,
    seed: int = 0,
    max_examples: int | None = None,
) -> list[TriviaQAPair]:
    """Load, pair, deterministically shuffle, and optionally truncate PN data."""
    pairs = pairs_from_rankalign_rows(read_jsonl(path))
    rng = random.Random(seed)
    rng.shuffle(pairs)

    if max_examples is not None:
        if max_examples < 0:
            raise ValueError("max_examples must be nonnegative")
        pairs = pairs[:max_examples]

    if not pairs:
        raise ValueError(f"no positive/negative TriviaQA pairs could be built from {path}")
    return pairs


def rankalign_triviaqa_path(rankalign_root: str | Path, split: str) -> Path:
    """Resolve RankAlign's positive/negative TriviaQA split."""
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    path = (
        Path(rankalign_root)
        / "data"
        / "dataset"
        / f"triviaqa_pn_{split}.jsonl"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"RankAlign TriviaQA file not found: {path}. "
            "Run scripts/bootstrap_rankalign.sh first."
        )
    return path
