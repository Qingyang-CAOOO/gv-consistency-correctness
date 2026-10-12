from __future__ import annotations

from .randomization import RandomizedBinaryChoice


def triviaqa_generator_prompt(question: str) -> str:
    """QA generator prompt following the original GV paper's paired-answer design."""
    return (
        "Generate one correct answer and one misleading answer (delimited by ||) "
        "to the following question: "
        f"{question}\n"
        "Answer:"
    )


def triviaqa_validator_prompt(question: str, choice: RandomizedBinaryChoice) -> str:
    """Minimal randomized A/B validator prompt for closed-book QA."""
    return (
        "Answer the following multiple choice question.\n"
        f"Question: {question}\n"
        f"A: {choice.option_a}\n"
        f"B: {choice.option_b}\n"
        "Answer (A or B):"
    )


def triviaqa_paper_validator_prompt(
    question: str,
    choice: RandomizedBinaryChoice,
) -> str:
    """Validator wording matching the QA template shown in the GV paper."""
    return (
        "Answer the following multiple choice question:"
        f"{question}\n"
        f"A: {choice.option_a}\n"
        f"B: {choice.option_b}\n"
        "Answer (A or B):"
    )



_PAIRWISE_VALIDATOR_TEMPLATES = (
    (
        "Answer the following multiple choice question.\n"
        "Question: {question}\n"
        "A: {option_a}\n"
        "B: {option_b}\n"
        "Answer (A or B):"
    ),
    (
        "Choose which candidate correctly answers the question.\n"
        "Question: {question}\n"
        "A: {option_a}\n"
        "B: {option_b}\n"
        "Choose A or B:"
    ),
    (
        "Fact-check the two candidate answers and select the correct one.\n"
        "Question: {question}\n"
        "A: {option_a}\n"
        "B: {option_b}\n"
        "Answer with A or B:"
    ),
    (
        "Which answer is factually correct for the question?\n"
        "Question: {question}\n"
        "A: {option_a}\n"
        "B: {option_b}\n"
        "Select A or B:"
    ),
)


def triviaqa_pairwise_validator_prompt(
    question: str,
    choice: RandomizedBinaryChoice,
    *,
    template_index: int,
) -> str:
    """Semantically equivalent pairwise QA measurements for sequential fusion."""
    if not 0 <= template_index < len(_PAIRWISE_VALIDATOR_TEMPLATES):
        raise ValueError("invalid pairwise validator template index")
    return _PAIRWISE_VALIDATOR_TEMPLATES[template_index].format(
        question=question,
        option_a=choice.option_a,
        option_b=choice.option_b,
    )


def num_triviaqa_pairwise_validator_templates() -> int:
    return len(_PAIRWISE_VALIDATOR_TEMPLATES)

def triviaqa_candidate_prompt(question: str) -> str:
    """Short-answer generator prompt for latent-hypothesis particles."""
    return (
        "Answer the following question with only a short answer and no explanation.\n"
        f"Question: {question}\n"
        "Answer:"
    )


_CORRECTNESS_TEMPLATES = (
    (
        "Question: {question}\n"
        "Candidate answer: {candidate}\n"
        "Which statement is correct?\n"
        "{options}\n"
        "Answer (A or B):"
    ),
    (
        "Fact-check the candidate answer to the question.\n"
        "Question: {question}\n"
        "Candidate: {candidate}\n"
        "{options}\n"
        "Choose A or B:"
    ),
    (
        "Verify whether the candidate correctly answers the question.\n"
        "Question: {question}\n"
        "Candidate answer: {candidate}\n"
        "{options}\n"
        "Answer with A or B:"
    ),
    (
        "Judge the factual correctness of the candidate answer.\n"
        "Question: {question}\n"
        "Candidate answer: {candidate}\n"
        "{options}\n"
        "Select A or B:"
    ),
)


def triviaqa_correctness_validator_prompt(
    question: str,
    candidate: str,
    *,
    target_index: int,
    template_index: int,
) -> str:
    """Randomized-label correctness validator used as a sequential potential."""
    if target_index not in (0, 1):
        raise ValueError("target_index must be 0 or 1")
    if not 0 <= template_index < len(_CORRECTNESS_TEMPLATES):
        raise ValueError("invalid correctness template index")

    correct = "The candidate answer is correct."
    incorrect = "The candidate answer is incorrect."
    if target_index == 0:
        option_a, option_b = correct, incorrect
    else:
        option_a, option_b = incorrect, correct
    options = f"A: {option_a}\nB: {option_b}"
    return _CORRECTNESS_TEMPLATES[template_index].format(
        question=question,
        candidate=candidate,
        options=options,
    )


def num_triviaqa_correctness_templates() -> int:
    return len(_CORRECTNESS_TEMPLATES)
