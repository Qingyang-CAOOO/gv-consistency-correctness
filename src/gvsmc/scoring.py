from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


DEFAULT_SYSTEM_PROMPT = "Answer directly without explanation."


def _infer_input_device(model: torch.nn.Module) -> torch.device:
    """Best-effort device for inputs, including Accelerate-dispatched models."""
    if hasattr(model, "device"):
        device = getattr(model, "device")
        if isinstance(device, torch.device) and device.type != "meta":
            return device
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("cannot infer device from a model with no parameters") from exc


def prompt_token_ids(
    tokenizer,
    prompt: str,
    *,
    is_chat: bool = False,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[int]:
    """Tokenize the fixed prompt prefix up to the validator answer."""
    if is_chat:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors=None,
            )
        except Exception as exc:
            # Some chat templates (notably Gemma) do not accept a system role.
            # Retry with the instruction folded into the user turn while keeping
            # the same semantic content.
            if not system_prompt:
                raise
            fallback_messages = [
                {
                    "role": "user",
                    "content": f"{system_prompt}\n\n{prompt}",
                }
            ]
            try:
                ids = tokenizer.apply_chat_template(
                    fallback_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors=None,
                )
            except Exception:
                raise exc
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)

    encoded = tokenizer(prompt, add_special_tokens=True)
    return list(encoded["input_ids"])


def completion_token_ids(tokenizer, completion: str) -> list[int]:
    ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"completion {completion!r} tokenized to zero tokens")
    return list(ids)


def completion_token_ids_for_prompt(
    tokenizer,
    prompt: str,
    completion: str,
    *,
    is_chat: bool = False,
) -> list[int]:
    """Tokenize a completion and verify the base-LM prompt boundary is stable.

    For a plain completion model we score the continuation represented by
    prompt + completion. Separate tokenization is valid only when tokenization
    is compositional across that string boundary.
    """
    suffix = completion_token_ids(tokenizer, completion)
    if is_chat:
        return suffix

    prefix = prompt_token_ids(tokenizer, prompt, is_chat=False)
    joint = list(
        tokenizer(prompt + completion, add_special_tokens=True)["input_ids"]
    )
    if prefix + suffix != joint:
        raise ValueError(
            "prompt/completion tokenization is not compositional at the boundary; "
            f"cannot safely score completion {completion!r} with separate tokens"
        )
    return suffix


@torch.inference_mode()
def completion_logprob(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    completion: str,
    *,
    is_chat: bool = False,
    device: torch.device | str | None = None,
) -> float:
    """Sequence log-probability of completion conditional on prompt."""
    prefix = prompt_token_ids(tokenizer, prompt, is_chat=is_chat)
    suffix = completion_token_ids_for_prompt(
        tokenizer,
        prompt,
        completion,
        is_chat=is_chat,
    )
    if not prefix:
        raise ValueError("prompt tokenized to zero tokens")

    dev = torch.device(device) if device is not None else _infer_input_device(model)
    input_ids = torch.tensor([prefix + suffix], dtype=torch.long, device=dev)
    attention_mask = torch.ones_like(input_ids)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    log_probs = outputs.logits.log_softmax(dim=-1)

    start = len(prefix)
    predictive = log_probs[:, start - 1 : input_ids.shape[1] - 1, :]
    targets = input_ids[:, start:]
    token_log_probs = predictive.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float(token_log_probs.sum().item())


@torch.inference_mode()
def validator_label_logprobs(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    labels: Sequence[str] = ("A", "B"),
    *,
    is_chat: bool = False,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return sequence-level conditional log-probabilities for validator labels."""
    if len(labels) < 2:
        raise ValueError("at least two validator labels are required")

    prefix = prompt_token_ids(tokenizer, prompt, is_chat=is_chat)
    if not prefix:
        raise ValueError("prompt tokenized to zero tokens")

    label_ids = [
        completion_token_ids_for_prompt(
            tokenizer,
            prompt,
            label,
            is_chat=is_chat,
        )
        for label in labels
    ]
    dev = torch.device(device) if device is not None else _infer_input_device(model)

    if all(len(ids) == 1 for ids in label_ids):
        input_ids = torch.tensor([prefix], dtype=torch.long, device=dev)
        attention_mask = torch.ones_like(input_ids)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        last_log_probs = outputs.logits[:, -1, :].log_softmax(dim=-1).squeeze(0)
        indices = torch.tensor([ids[0] for ids in label_ids], device=dev)
        return last_log_probs.index_select(0, indices).detach().cpu().float()

    scores = [
        completion_logprob(
            model,
            tokenizer,
            prompt,
            label,
            is_chat=is_chat,
            device=dev,
        )
        for label in labels
    ]
    return torch.tensor(scores, dtype=torch.float32)
