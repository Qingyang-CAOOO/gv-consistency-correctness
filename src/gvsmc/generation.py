from __future__ import annotations

from dataclasses import dataclass
import re

import torch

from .scoring import DEFAULT_SYSTEM_PROMPT, _infer_input_device, prompt_token_ids


@dataclass(frozen=True)
class GeneratedAnswerPair:
    intended_correct: str
    intended_misleading: str


def _clean_generated_candidate(text: str) -> str:
    """Remove only presentation artifacts, not semantic answer content."""
    value = text.strip()
    value = re.sub(r"^[-*•]+\s*", "", value).strip()
    value = re.sub(
        r"^\*{0,2}(?:correct(?: answer)?|misleading(?: answer)?|answer)\*{0,2}\s*:\s*",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()
    return value


def parse_generated_answer_pair_with_mode(
    text: str,
) -> tuple[GeneratedAnswerPair | None, str]:
    """Parse common semantically equivalent renderings of a two-answer pair.

    Accepted forms are deliberately limited:
    1. one line ``correct || misleading``;
    2. an empty ``Answer:`` line followed by that canonical line;
    3. two explicitly labeled Correct/Misleading lines, with or without a
       trailing ``||`` on the Correct line.

    Missing candidates and outputs with more than one delimiter remain failures.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None, "empty"

    # Ignore a standalone echoed prompt marker such as "Answer:".
    if re.fullmatch(r"\*{0,2}answer\*{0,2}\s*:", lines[0], re.IGNORECASE):
        lines = lines[1:]
        if not lines:
            return None, "answer_marker_only"

    # Canonical form: exactly one delimiter on one line with both sides nonempty.
    for line in lines:
        if line.count("||") == 1:
            left_raw, right_raw = line.split("||", maxsplit=1)
            left = _clean_generated_candidate(left_raw)
            right = _clean_generated_candidate(right_raw)
            if left and right:
                return GeneratedAnswerPair(left, right), "delimiter_line"

    # Labeled two-line form, e.g. "**Correct Answer:** X ||" followed by
    # "**Misleading Answer:** Y". This is a formatting variation, not inference.
    correct_value: str | None = None
    misleading_value: str | None = None
    for line in lines:
        normalized = line.strip()
        if re.match(r"^\*{0,2}correct(?: answer)?\*{0,2}\s*:", normalized, re.IGNORECASE):
            value = re.sub(
                r"^\*{0,2}correct(?: answer)?\*{0,2}\s*:\s*",
                "",
                normalized,
                flags=re.IGNORECASE,
            )
            value = value.removesuffix("||").strip()
            correct_value = _clean_generated_candidate(value)
        elif re.match(r"^\*{0,2}misleading(?: answer)?\*{0,2}\s*:", normalized, re.IGNORECASE):
            value = re.sub(
                r"^\*{0,2}misleading(?: answer)?\*{0,2}\s*:\s*",
                "",
                normalized,
                flags=re.IGNORECASE,
            )
            misleading_value = _clean_generated_candidate(value)

    if correct_value and misleading_value:
        return (
            GeneratedAnswerPair(correct_value, misleading_value),
            "labeled_lines",
        )

    delimiter_count = text.count("||")
    if delimiter_count > 1:
        return None, "ambiguous_multiple_delimiters"
    if delimiter_count == 1:
        return None, "incomplete_delimiter_pair"
    return None, "unrecognized_format"


def parse_generated_answer_pair(text: str) -> GeneratedAnswerPair | None:
    pair, _ = parse_generated_answer_pair_with_mode(text)
    return pair

@torch.inference_mode()
def generate_text(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    *,
    is_chat: bool = False,
    device: torch.device | str | None = None,
    max_new_tokens: int = 32,
    do_sample: bool = False,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: int | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Generate one continuation from the LM with reproducible per-example seeding."""
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if do_sample and temperature <= 0:
        raise ValueError("temperature must be positive when sampling")
    if do_sample and not 0 < top_p <= 1:
        raise ValueError("top_p must lie in (0, 1]")

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    dev = torch.device(device) if device is not None else _infer_input_device(model)
    input_ids = prompt_token_ids(
        tokenizer,
        prompt,
        is_chat=is_chat,
        system_prompt=system_prompt,
    )
    encoded = torch.tensor([input_ids], dtype=torch.long, device=dev)
    attention_mask = torch.ones_like(encoded)

    kwargs = {
        "input_ids": encoded,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id,
    }
    if do_sample:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p

    output = model.generate(**kwargs)
    continuation = output[0, encoded.shape[1] :]
    return tokenizer.decode(continuation, skip_special_tokens=True).strip()
