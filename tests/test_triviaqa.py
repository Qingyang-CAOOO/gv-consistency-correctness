import json
from pathlib import Path

from gvsmc.data.triviaqa import (
    load_rankalign_triviaqa_pairs,
    pairs_from_rankalign_rows,
    questions_from_rankalign_rows,
)


def test_pairs_from_rankalign_rows():
    rows = [
        {"question": "Capital of France?", "answers": ["Paris"], "correct": "yes"},
        {"question": "Capital of France?", "answers": ["Rome"], "correct": "no"},
        {"question": "2+2?", "answers": ["4"], "correct": "true"},
        {"question": "2+2?", "answers": ["5"], "correct": "false"},
    ]
    pairs = pairs_from_rankalign_rows(rows)
    by_question = {p.question: p for p in pairs}

    assert by_question["Capital of France?"].correct_answer == "Paris"
    assert by_question["Capital of France?"].misleading_answer == "Rome"
    assert by_question["2+2?"].correct_answer == "4"
    assert by_question["2+2?"].misleading_answer == "5"


def test_loader_skips_unpaired_and_truncates(tmp_path: Path):
    path = tmp_path / "toy.jsonl"
    rows = [
        {"question": "q1", "answers": ["right1"], "correct": "yes"},
        {"question": "q1", "answers": ["wrong1"], "correct": "no"},
        {"question": "q2", "answers": ["right2"], "correct": "yes"},
        {"question": "q2", "answers": ["wrong2"], "correct": "no"},
        {"question": "q3", "answers": ["right3"], "correct": "yes"},
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    pairs = load_rankalign_triviaqa_pairs(path, seed=1, max_examples=1)
    assert len(pairs) == 1
    assert pairs[0].question in {"q1", "q2"}


def test_questions_use_only_positive_rows_as_gold():
    rows = [
        {
            "question": "Who played Neo?",
            "answers": ["Keanu Reeves", "Keanu Charles Reeves"],
            "correct": "yes",
        },
        {
            "question": "Who played Neo?",
            "answers": ["Tom Cruise"],
            "correct": "no",
        },
    ]

    questions = questions_from_rankalign_rows(rows)

    assert len(questions) == 1
    assert questions[0].question == "Who played Neo?"
    assert questions[0].gold_answers == (
        "Keanu Reeves",
        "Keanu Charles Reeves",
    )
