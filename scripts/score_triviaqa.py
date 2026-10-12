#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from random import Random

import torch
from tqdm import tqdm

from gvsmc.data.triviaqa import (
    load_rankalign_triviaqa_pairs,
    rankalign_triviaqa_path,
)
from gvsmc.metrics import (
    effective_sample_size,
    normalized_entropy,
    retained_fraction,
)
from gvsmc.prompts import triviaqa_validator_prompt
from gvsmc.randomization import (
    balanced_targets,
    binary_choice_with_target_index,
)
from gvsmc.scoring import validator_label_logprobs
from gvsmc.weights import (
    hard_randomized_gv_weight,
    normalize_weights,
    soft_randomized_gv_weight_from_label_logprobs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score RankAlign TriviaQA positive/negative answer pairs with "
            "Li-style randomized A/B validation."
        )
    )
    parser.add_argument("--rankalign-root", default="external/rankalign")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="outputs/triviaqa_soft_gv/scores.jsonl",
    )
    parser.add_argument(
        "--chat",
        choices=["auto", "true", "false"],
        default="auto",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cuda, cpu, mps, or explicit torch device.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--sampled-hard-draws",
        type=int,
        default=1024,
        help=(
            "Monte Carlo draws per example used to verify "
            "E[C_sampled | H] = G. No extra model forward passes are required."
        ),
    )
    parser.add_argument(
        "--paired-order-diagnostic",
        action="store_true",
        help=(
            "Score each question in both A/B orders to diagnose option-position bias. "
            "This adds a second validator pass per example."
        ),
    )
    return parser.parse_args()


def _resolve_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_chat(arg: str, model_name: str) -> bool:
    if arg == "true":
        return True
    if arg == "false":
        return False
    lowered = model_name.lower()
    return "instruct" in lowered or "chat" in lowered


def _rankalign_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _load_model(model_name: str, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    kwargs = {}
    if device.type == "cuda":
        kwargs["torch_dtype"] = torch.bfloat16
    if "gemma-2" in model_name.lower():
        kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.to(device)
    model.eval()
    return model, tokenizer


def _soft_target_probability(label_logprobs: torch.Tensor, target_index: int, temperature: float) -> float:
    target = torch.tensor([target_index], dtype=torch.long)
    value = soft_randomized_gv_weight_from_label_logprobs(
        label_logprobs.unsqueeze(0),
        target,
        temperature=temperature,
    )[0]
    return float(value.item())


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.sampled_hard_draws <= 0:
        raise ValueError("--sampled-hard-draws must be positive")

    rankalign_root = Path(args.rankalign_root)
    data_path = rankalign_triviaqa_path(rankalign_root, args.split)
    pairs = load_rankalign_triviaqa_pairs(
        data_path,
        seed=args.seed,
        max_examples=args.limit,
    )

    device = _resolve_device(args.device)
    is_chat = _resolve_chat(args.chat, args.model)
    print(f"model: {args.model}")
    print(f"device: {device}")
    print(f"chat template: {is_chat}")
    print(f"examples: {len(pairs)}")

    model, tokenizer = _load_model(args.model, device)

    rng = Random(args.seed)
    targets = balanced_targets(len(pairs), rng=rng)
    sampled_rng = Random(args.seed + 1_000_003)

    # Base completion models naturally continue the trailing colon with a
    # leading space (RankAlign follows the same convention). Chat templates
    # already insert the assistant turn boundary, so use bare A/B there.
    validator_labels = ("A", "B") if is_chat else (" A", " B")
    print(f"validator labels: {validator_labels!r}")

    records = []
    hard_values = []
    soft_values = []
    sampled_hard_values = []
    sampled_hard_mc_values = []
    paired_correct_probs = []
    paired_position_gaps = []

    iterator = zip(pairs, targets)
    for index, (item, target_index) in enumerate(
        tqdm(iterator, total=len(pairs), desc="validator scoring")
    ):
        choice = binary_choice_with_target_index(
            item.correct_answer,
            item.misleading_answer,
            target_index=target_index,
        )
        prompt = triviaqa_validator_prompt(item.question, choice)
        label_logprobs = validator_label_logprobs(
            model,
            tokenizer,
            prompt,
            labels=validator_labels,
            is_chat=is_chat,
            device=device,
        )

        target = torch.tensor([target_index], dtype=torch.long)
        prediction_index = int(torch.argmax(label_logprobs).item())
        prediction = torch.tensor([prediction_index], dtype=torch.long)

        hard = hard_randomized_gv_weight(target, prediction)[0]
        soft = soft_randomized_gv_weight_from_label_logprobs(
            label_logprobs.unsqueeze(0),
            target,
            temperature=args.temperature,
        )[0]

        hard_values.append(hard)
        soft_values.append(soft)

        normalized_label_probs = torch.softmax(
            label_logprobs / args.temperature,
            dim=-1,
        )
        p_a = float(normalized_label_probs[0].item())
        sampled_prediction_index = 0 if sampled_rng.random() < p_a else 1
        sampled_hard = float(sampled_prediction_index == target_index)
        sampled_hard_mc = sum(
            sampled_rng.random() < float(soft.item())
            for _ in range(args.sampled_hard_draws)
        ) / args.sampled_hard_draws
        sampled_hard_values.append(sampled_hard)
        sampled_hard_mc_values.append(sampled_hard_mc)

        record = {
            "index": index,
            "question": item.question,
            "correct_answer": item.correct_answer,
            "misleading_answer": item.misleading_answer,
            "option_a": choice.option_a,
            "option_b": choice.option_b,
            "target_index": target_index,
            "target_label": "A" if target_index == 0 else "B",
            "prediction_index": prediction_index,
            "prediction_label": "A" if prediction_index == 0 else "B",
            "logprob_a": float(label_logprobs[0].item()),
            "logprob_b": float(label_logprobs[1].item()),
            "hard_weight": float(hard.item()),
            "soft_weight": float(soft.item()),
            "sampled_prediction_index": sampled_prediction_index,
            "sampled_prediction_label": (
                "A" if sampled_prediction_index == 0 else "B"
            ),
            "sampled_hard_weight": sampled_hard,
            "sampled_hard_mc_mean": sampled_hard_mc,
            "validator_prompt": prompt,
        }

        if args.paired_order_diagnostic:
            choice_a = binary_choice_with_target_index(
                item.correct_answer,
                item.misleading_answer,
                target_index=0,
            )
            choice_b = binary_choice_with_target_index(
                item.correct_answer,
                item.misleading_answer,
                target_index=1,
            )
            prompt_a = triviaqa_validator_prompt(item.question, choice_a)
            prompt_b = triviaqa_validator_prompt(item.question, choice_b)
            logprobs_a = validator_label_logprobs(
                model,
                tokenizer,
                prompt_a,
                labels=validator_labels,
                is_chat=is_chat,
                device=device,
            )
            logprobs_b = validator_label_logprobs(
                model,
                tokenizer,
                prompt_b,
                labels=validator_labels,
                is_chat=is_chat,
                device=device,
            )
            p_correct_when_a = _soft_target_probability(
                logprobs_a, 0, args.temperature
            )
            p_correct_when_b = _soft_target_probability(
                logprobs_b, 1, args.temperature
            )
            order_avg = 0.5 * (p_correct_when_a + p_correct_when_b)
            position_gap = p_correct_when_a - p_correct_when_b
            paired_correct_probs.append(order_avg)
            paired_position_gaps.append(position_gap)
            record.update(
                {
                    "p_correct_when_a": p_correct_when_a,
                    "p_correct_when_b": p_correct_when_b,
                    "paired_order_avg_correct_prob": order_avg,
                    "paired_position_gap_a_minus_b": position_gap,
                }
            )

        records.append(record)

    hard_tensor = torch.stack(hard_values)
    soft_tensor = torch.stack(soft_values)
    normalized = normalize_weights(soft_tensor)
    ess = effective_sample_size(soft_tensor)
    sampled_hard_tensor = torch.tensor(
        sampled_hard_values,
        dtype=torch.float32,
    )
    sampled_hard_mc_tensor = torch.tensor(
        sampled_hard_mc_values,
        dtype=torch.float32,
    )

    target_tensor = torch.tensor(targets, dtype=torch.long)
    prediction_tensor = torch.tensor(
        [record["prediction_index"] for record in records],
        dtype=torch.long,
    )
    target_a = target_tensor == 0
    target_b = target_tensor == 1

    summary = {
        "model": args.model,
        "split": args.split,
        "seed": args.seed,
        "temperature": args.temperature,
        "num_examples": len(records),
        "rankalign_commit": _rankalign_commit(rankalign_root),
        "validator_labels": list(validator_labels),
        "target_a_fraction": float(target_a.to(torch.float32).mean().item()),
        "prediction_a_fraction": float(
            (prediction_tensor == 0).to(torch.float32).mean().item()
        ),
        "hard_consistency": float(hard_tensor.mean().item()),
        "hard_consistency_target_a": float(hard_tensor[target_a].mean().item()),
        "hard_consistency_target_b": float(hard_tensor[target_b].mean().item()),
        "mean_soft_consistency": float(soft_tensor.mean().item()),
        "sampled_hard_consistency": float(sampled_hard_tensor.mean().item()),
        "sampled_hard_mc_mean": float(sampled_hard_mc_tensor.mean().item()),
        "sampled_hard_mc_minus_soft_mean": float(
            (sampled_hard_mc_tensor.mean() - soft_tensor.mean()).item()
        ),
        "sampled_hard_draws_per_example": args.sampled_hard_draws,
        "mean_soft_consistency_target_a": float(soft_tensor[target_a].mean().item()),
        "mean_soft_consistency_target_b": float(soft_tensor[target_b].mean().item()),
        "hard_retained_fraction": float(retained_fraction(hard_tensor).item()),
        "soft_ess": float(ess.item()),
        "soft_ess_fraction": float((ess / len(soft_tensor)).item()),
        "soft_normalized_entropy": float(normalized_entropy(soft_tensor).item()),
        "soft_max_normalized_weight": float(normalized.max().item()),
    }

    if args.paired_order_diagnostic:
        paired_probs_tensor = torch.tensor(paired_correct_probs, dtype=torch.float32)
        paired_gaps_tensor = torch.tensor(paired_position_gaps, dtype=torch.float32)
        summary.update(
            {
                "paired_order_mean_correct_prob": float(
                    paired_probs_tensor.mean().item()
                ),
                "paired_order_mean_position_gap_a_minus_b": float(
                    paired_gaps_tensor.mean().item()
                ),
                "paired_order_mean_abs_position_gap": float(
                    paired_gaps_tensor.abs().mean().item()
                ),
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"wrote: {output_path}")
    print(f"wrote: {summary_path}")


if __name__ == "__main__":
    main()
