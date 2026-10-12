#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from tqdm import tqdm

from gvsmc.prompts import triviaqa_paper_validator_prompt
from gvsmc.randomization import binary_choice_with_target_index
from gvsmc.scoring import validator_label_logprobs
from gvsmc.weights import soft_randomized_gv_weight_from_label_logprobs


RUN_NAMES = ("Base", "Hard CFT", "Soft CFT")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-evaluate Base/Hard/Soft generators with Base/Hard/Soft "
            "validators on the same question intersection. This produces a "
            "3x3 generator-validator coupling matrix."
        )
    )
    parser.add_argument("--base-data", required=True)
    parser.add_argument("--hard-data", required=True)
    parser.add_argument("--soft-data", required=True)
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    parser.add_argument("--hard-adapter", required=True)
    parser.add_argument("--soft-adapter", required=True)
    parser.add_argument("--validator-temperature", type=float, default=1.0)
    parser.add_argument("--chat", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            index = int(row["index"])
            if index in rows:
                raise ValueError(f"duplicate index {index} in {path}")
            rows[index] = row
    return rows


def common_parsed_indices(runs: dict[str, dict[int, dict]]) -> list[int]:
    index_sets = [set(rows) for rows in runs.values()]
    common = set.intersection(*index_sets)
    result = []
    for index in sorted(common):
        questions = {rows[index]["question"] for rows in runs.values()}
        if len(questions) != 1:
            raise ValueError(f"question mismatch at index {index}")
        if all(bool(rows[index].get("parse_success", False)) for rows in runs.values()):
            result.append(index)
    if not result:
        raise ValueError("no common parsed trajectories")
    return result


def resolve_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_chat(arg: str, model_name: str) -> bool:
    if arg == "true":
        return True
    if arg == "false":
        return False
    lowered = model_name.lower()
    return "instruct" in lowered or "chat" in lowered or lowered.endswith("-it")


def load_model(
    model_name: str,
    adapter: str | None,
    device: torch.device,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {}
    if device.type == "cuda":
        kwargs["torch_dtype"] = torch.bfloat16
    if "gemma-2" in model_name.lower():
        kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.to(device)
    model.eval()
    return model, tokenizer


def soft_target_probability(
    label_logprobs: torch.Tensor,
    target_index: int,
    temperature: float,
) -> float:
    target = torch.tensor([target_index], dtype=torch.long)
    return float(
        soft_randomized_gv_weight_from_label_logprobs(
            label_logprobs.unsqueeze(0),
            target,
            temperature=temperature,
        )[0].item()
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def evaluate_cell(
    *,
    model,
    tokenizer,
    rows: dict[int, dict],
    indices: list[int],
    is_chat: bool,
    device: torch.device,
    temperature: float,
) -> dict:
    validator_labels = ("A", "B") if is_chat else (" A", " B")

    hard_values: list[float] = []
    soft_values: list[float] = []
    pair_probs: list[float] = []
    position_gaps: list[float] = []
    prediction_a: list[float] = []
    grounded_hard: list[float] = []
    grounded_soft: list[float] = []

    for index in tqdm(indices, leave=False):
        row = rows[index]
        correct = row["intended_correct"]
        misleading = row["intended_misleading"]
        question = row["question"]
        original_target = int(row["target_index"])

        choice_a = binary_choice_with_target_index(
            correct, misleading, target_index=0
        )
        choice_b = binary_choice_with_target_index(
            correct, misleading, target_index=1
        )
        prompt_a = triviaqa_paper_validator_prompt(question, choice_a)
        prompt_b = triviaqa_paper_validator_prompt(question, choice_b)

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

        p_correct_a = soft_target_probability(logprobs_a, 0, temperature)
        p_correct_b = soft_target_probability(logprobs_b, 1, temperature)
        pair_prob = 0.5 * (p_correct_a + p_correct_b)
        gap = p_correct_a - p_correct_b

        original_logprobs = logprobs_a if original_target == 0 else logprobs_b
        original_soft = p_correct_a if original_target == 0 else p_correct_b
        prediction = int(torch.argmax(original_logprobs).item())
        original_hard = float(prediction == original_target)

        hard_values.append(original_hard)
        soft_values.append(original_soft)
        pair_probs.append(pair_prob)
        position_gaps.append(gap)
        prediction_a.append(float(prediction == 0))

        if bool(row.get("grounded_pair", False)):
            grounded_hard.append(original_hard)
            grounded_soft.append(original_soft)

    return {
        "n": len(indices),
        "hard_consistency": mean(hard_values),
        "mean_soft_consistency": mean(soft_values),
        "paired_order_mean_correct_prob": mean(pair_probs),
        "paired_order_mean_position_gap_a_minus_b": mean(position_gaps),
        "paired_order_mean_abs_position_gap": mean([abs(x) for x in position_gaps]),
        "prediction_a_fraction": mean(prediction_a),
        "grounded_pair_count": len(grounded_hard),
        "validator_accuracy_on_grounded_pairs": (
            mean(grounded_hard) if grounded_hard else None
        ),
        "mean_soft_on_grounded_pairs": (
            mean(grounded_soft) if grounded_soft else None
        ),
    }


def generator_metrics(rows: dict[int, dict], indices: list[int]) -> dict:
    correct = [float(bool(rows[i]["intended_correct_em"])) for i in indices]
    misleading = [float(bool(rows[i]["intended_misleading_em"])) for i in indices]
    grounded = [float(bool(rows[i]["grounded_pair"])) for i in indices]
    return {
        "n": len(indices),
        "intended_correct_em": mean(correct),
        "intended_misleading_em": mean(misleading),
        "grounded_pair_fraction": mean(grounded),
    }


def main() -> None:
    args = parse_args()
    if args.validator_temperature <= 0:
        raise ValueError("--validator-temperature must be positive")

    generators = {
        "Base": read_jsonl(args.base_data),
        "Hard CFT": read_jsonl(args.hard_data),
        "Soft CFT": read_jsonl(args.soft_data),
    }
    indices = common_parsed_indices(generators)
    print(f"common parsed intersection: {len(indices)} questions")

    result = {
        "model": args.model,
        "num_common_parsed": len(indices),
        "indices": indices,
        "generator_metrics": {
            name: generator_metrics(rows, indices)
            for name, rows in generators.items()
        },
        "validators": {},
    }

    device = resolve_device(args.device)
    is_chat = resolve_chat(args.chat, args.model)
    validators = {
        "Base": None,
        "Hard CFT": args.hard_adapter,
        "Soft CFT": args.soft_adapter,
    }

    for validator_name, adapter in validators.items():
        print(f"\nloading validator: {validator_name}")
        model, tokenizer = load_model(args.model, adapter, device)
        row_result = {}
        for generator_name, rows in generators.items():
            print(f"  scoring generator: {generator_name}")
            row_result[generator_name] = evaluate_cell(
                model=model,
                tokenizer=tokenizer,
                rows=rows,
                indices=indices,
                is_chat=is_chat,
                device=device,
                temperature=args.validator_temperature,
            )
        result["validators"][validator_name] = row_result

        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nHard GV consistency matrix")
    print("rows = generator, columns = validator")
    print(f"{'Generator':<10} {'Base V':>10} {'Hard V':>10} {'Soft V':>10}")
    for generator_name in RUN_NAMES:
        values = [
            result["validators"][validator][generator_name]["hard_consistency"]
            for validator in RUN_NAMES
        ]
        print(
            f"{generator_name:<10} "
            + " ".join(f"{value:>10.4f}" for value in values)
        )

    print("\nMean soft GV consistency matrix")
    print("rows = generator, columns = validator")
    print(f"{'Generator':<10} {'Base V':>10} {'Hard V':>10} {'Soft V':>10}")
    for generator_name in RUN_NAMES:
        values = [
            result["validators"][validator][generator_name]["mean_soft_consistency"]
            for validator in RUN_NAMES
        ]
        print(
            f"{generator_name:<10} "
            + " ".join(f"{value:>10.4f}" for value in values)
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote: {output}")


if __name__ == "__main__":
    main()
