from __future__ import annotations

import re
import string
from collections.abc import Iterable


def normalize_qa_answer(text: str) -> str:
    """SQuAD-style normalization for conservative TriviaQA exact matching."""
    lowered = text.lower()
    no_punctuation = "".join(ch for ch in lowered if ch not in string.punctuation)
    no_articles = re.sub(r"\b(a|an|the)\b", " ", no_punctuation)
    return " ".join(no_articles.split())


def matches_any_gold(candidate: str, gold_answers: Iterable[str]) -> bool:
    normalized_candidate = normalize_qa_answer(candidate)
    if not normalized_candidate:
        return False
    return any(
        normalized_candidate == normalize_qa_answer(gold)
        for gold in gold_answers
    )
