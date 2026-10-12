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
    load_rankalign_triviaqa_questions,
    rankalign_triviaqa_path,
)
from gvsmc.evaluation import matches_any_gold
from gvsmc.generation import (
    generate_text,
    parse_generated_answer_pair_with_mode,
)
from gvsmc.metrics import effective_sample_size, normalized_entropy, retained_fraction
from gvsmc.prompts import triviaqa_generator_prompt, triviaqa_paper_validator_prompt
from gvsmc.randomization import balanced_targets, binary_choice_with_target_index
from gvsmc.scoring import validator_label_logprobs
from gvsmc.weights import (
    hard_randomized_gv_weight,
    normalize_weights,
    soft_randomized_gv_weight_from_label_logprobs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a generator-coupled TriviaQA GV pilot: the LM first generates "
            "one intended-correct and one intended-misleading answer, then the "
            "same LM validates their randomized A/B order."
        )
    )
    parser.add_argument("--rankalign-root", default="external/rankalign")
    parser.add_argument(
        "--question-file",
        default=None,
        help="Optional frozen JSONL of {question, gold_answers}; overrides split shuffling.",
    )
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument(
        "--adapter",
        default=None,
        help=(
            "Optional PEFT/LoRA adapter directory. When provided, load the "
            "adapter on top of --model for both generation and validation."
        ),
    )
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--generation-seed", type=int, default=43)
    parser.add_argument("--order-seed", type=int, default=44)
    parser.add_argument("--validator-sample-seed", type=int, default=45)
    parser.add_argument(
        "--generation-mode",
        choices=["greedy", "sample"],
        default="greedy",
    )
    parser.add_argument("--generation-temperature", type=float, default=0.8)
    parser.add_argument("--generation-top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--validator-temperature", type=float, default=1.0)
    parser.add_argument("--sampled-hard-draws", type=int, default=1024)
    parser.add_argument(
        "--paired-order-diagnostic",
        action="store_true",
        help="Also score each parsed generator pair in both A/B orders.",
    )
    parser.add_argument(
        "--output",
        default="outputs/triviaqa_generated_gv/scores.jsonl",
    )
    parser.add_argument(
        "--chat",
        choices=["auto", "true", "false"],
        default="auto",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cuda, cpu, mps, or an explicit torch device.",
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
    return "instruct" in lowered or "chat" in lowered or lowered.endswith("-it")


def _rankalign_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _load_model(model_name: str, device: torch.device, adapter: str | None = None):
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


def _soft_target_probability(
    label_logprobs: torch.Tensor,
    target_index: int,
    temperature: float,
) -> float:
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
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.validator_temperature <= 0:
        raise ValueError("--validator-temperature must be positive")
    if args.sampled_hard_draws <= 0:
        raise ValueError("--sampled-hard-draws must be positive")

    rankalign_root = Path(args.rankalign_root)
    if args.question_file:
        from gvsmc.data.triviaqa import TriviaQAQuestion
        question_file = Path(args.question_file)
        questions = []
        with question_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                if not item.get("question") or not item.get("gold_answers"):
                    raise ValueError("question-file rows must include question and gold_answers")
                questions.append(TriviaQAQuestion(
                    question=str(item["question"]),
                    gold_answers=tuple(str(x) for x in item["gold_answers"]),
                ))
        keys = [" ".join(q.question.casefold().split()) for q in questions]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate question identities in question-file")
        questions = questions[:args.limit]
        if len(questions) < args.limit:
            raise ValueError(f"question-file has only {len(questions)} rows; requested {args.limit}")
    else:
        data_path = rankalign_triviaqa_path(rankalign_root, args.split)
        questions = load_rankalign_triviaqa_questions(
            data_path,
            seed=args.data_seed,
            max_examples=args.limit,
        )

    device = _resolve_device(args.device)
    is_chat = _resolve_chat(args.chat, args.model)
    print(f"model: {args.model}")
    print(f"device: {device}")
    print(f"chat template: {is_chat}")
    print(f"questions: {len(questions)}")
    print(f"generation mode: {args.generation_mode}")
    print(f"adapter: {args.adapter}")

    model, tokenizer = _load_model(args.model, device, adapter=args.adapter)
    validator_labels = ("A", "B") if is_chat else (" A", " B")
    print(f"validator labels: {validator_labels!r}")

    order_rng = Random(args.order_seed)
    targets = balanced_targets(len(questions), rng=order_rng)
    validator_rng = Random(args.validator_sample_seed)

    records: list[dict] = []
    hard_values: list[torch.Tensor] = []
    soft_values: list[torch.Tensor] = []
    sampled_hard_values: list[float] = []
    sampled_hard_mc_values: list[float] = []
    valid_targets: list[int] = []
    paired_correct_probs: list[float] = []
    paired_position_gaps: list[float] = []

    iterator = zip(questions, targets)
    for index, (item, target_index) in enumerate(
        tqdm(iterator, total=len(questions), desc="generator -> validator")
    ):
        generator_prompt = triviaqa_generator_prompt(item.question)
        raw_generation = generate_text(
            model,
            tokenizer,
            generator_prompt,
            is_chat=is_chat,
            device=device,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.generation_mode == "sample",
            temperature=args.generation_temperature,
            top_p=args.generation_top_p,
            seed=args.generation_seed + index,
        )
        pair, parse_mode = parse_generated_answer_pair_with_mode(raw_generation)

        record = {
            "index": index,
            "question": item.question,
            "gold_answers": list(item.gold_answers),
            "generator_prompt": generator_prompt,
            "raw_generator_response": raw_generation,
            "parse_success": pair is not None,
            "parse_mode": parse_mode,
            "generation_seed": args.generation_seed + index,
        }

        if pair is None:
            records.append(record)
            continue

        intended_correct_em = matches_any_gold(
            pair.intended_correct, item.gold_answers
        )
        intended_misleading_em = matches_any_gold(
            pair.intended_misleading, item.gold_answers
        )
        grounded_pair = intended_correct_em and not intended_misleading_em

        choice = binary_choice_with_target_index(
            pair.intended_correct,
            pair.intended_misleading,
            target_index=target_index,
        )
        validator_prompt = triviaqa_paper_validator_prompt(item.question, choice)
        label_logprobs = validator_label_logprobs(
            model,
            tokenizer,
            validator_prompt,
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
            temperature=args.validator_temperature,
        )[0]

        normalized_label_probs = torch.softmax(
            label_logprobs / args.validator_temperature,
            dim=-1,
        )
        p_a = float(normalized_label_probs[0].item())
        sampled_prediction_index = 0 if validator_rng.random() < p_a else 1
        sampled_hard = float(sampled_prediction_index == target_index)
        sampled_hard_mc = sum(
            validator_rng.random() < float(soft.item())
            for _ in range(args.sampled_hard_draws)
        ) / args.sampled_hard_draws

        hard_values.append(hard)
        soft_values.append(soft)
        sampled_hard_values.append(sampled_hard)
        sampled_hard_mc_values.append(sampled_hard_mc)
        valid_targets.append(target_index)

        record.update(
            {
                "intended_correct": pair.intended_correct,
                "intended_misleading": pair.intended_misleading,
                "intended_correct_em": intended_correct_em,
                "intended_misleading_em": intended_misleading_em,
                "grounded_pair": grounded_pair,
                "option_a": choice.option_a,
                "option_b": choice.option_b,
                "target_index": target_index,
                "target_label": "A" if target_index == 0 else "B",
                "validator_prompt": validator_prompt,
                "logprob_a": float(label_logprobs[0].item()),
                "logprob_b": float(label_logprobs[1].item()),
                "prediction_index": prediction_index,
                "prediction_label": "A" if prediction_index == 0 else "B",
                "hard_weight": float(hard.item()),
                "soft_weight": float(soft.item()),
                "sampled_prediction_index": sampled_prediction_index,
                "sampled_prediction_label": (
                    "A" if sampled_prediction_index == 0 else "B"
                ),
                "sampled_hard_weight": sampled_hard,
                "sampled_hard_mc_mean": sampled_hard_mc,
            }
        )

        if args.paired_order_diagnostic:
            choice_a = binary_choice_with_target_index(
                pair.intended_correct, pair.intended_misleading, target_index=0
            )
            choice_b = binary_choice_with_target_index(
                pair.intended_correct, pair.intended_misleading, target_index=1
            )
            prompt_a = triviaqa_paper_validator_prompt(item.question, choice_a)
            prompt_b = triviaqa_paper_validator_prompt(item.question, choice_b)
            logprobs_a = validator_label_logprobs(
                model, tokenizer, prompt_a, labels=validator_labels,
                is_chat=is_chat, device=device
            )
            logprobs_b = validator_label_logprobs(
                model, tokenizer, prompt_b, labels=validator_labels,
                is_chat=is_chat, device=device
            )
            p_correct_when_a = _soft_target_probability(
                logprobs_a, 0, args.validator_temperature
            )
            p_correct_when_b = _soft_target_probability(
                logprobs_b, 1, args.validator_temperature
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

    parsed_records = [record for record in records if record["parse_success"]]
    if not parsed_records:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        summary = {
            "model": args.model,
            "adapter": args.adapter,
            "split": "custom_question_file" if args.question_file else args.split,
            "rankalign_commit": _rankalign_commit(rankalign_root),
            "num_questions": len(records),
            "num_parsed": 0,
            "parse_success_rate": 0.0,
            "data_seed": args.data_seed,
            "question_file": args.question_file,
            "generation_seed": args.generation_seed,
            "order_seed": args.order_seed,
            "validator_sample_seed": args.validator_sample_seed,
            "generation_mode": args.generation_mode,
            "generation_temperature": args.generation_temperature,
            "generation_top_p": args.generation_top_p,
            "max_new_tokens": args.max_new_tokens,
            "status": "no_parseable_generator_pairs",
        }
        summary_path = output_path.with_suffix(".summary.json")
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        print("\nNo generator responses matched the required 'correct || misleading' format.")
        print("First raw generations:")
        for record in records[:5]:
            print(f"- {record['question']}")
            print(f"  {record['raw_generator_response']!r}")
        print(json.dumps(summary, indent=2))
        print(f"wrote: {output_path}")
        print(f"wrote: {summary_path}")
        return

    hard_tensor = torch.stack(hard_values)
    soft_tensor = torch.stack(soft_values)
    sampled_hard_tensor = torch.tensor(sampled_hard_values, dtype=torch.float32)
    sampled_hard_mc_tensor = torch.tensor(
        sampled_hard_mc_values, dtype=torch.float32
    )
    normalized = normalize_weights(soft_tensor)
    ess = effective_sample_size(soft_tensor)
    target_tensor = torch.tensor(valid_targets, dtype=torch.long)
    prediction_tensor = torch.tensor(
        [record["prediction_index"] for record in parsed_records],
        dtype=torch.long,
    )
    target_a = target_tensor == 0
    target_b = target_tensor == 1

    parse_failures = [
        record for record in records if not record["parse_success"]
    ]
    parse_mode_counts: dict[str, int] = {}
    for record in records:
        mode = record["parse_mode"]
        parse_mode_counts[mode] = parse_mode_counts.get(mode, 0) + 1
    parse_failure_with_delimiter = sum(
        "||" in record["raw_generator_response"] for record in parse_failures
    )
    parse_failure_multiline = sum(
        len(
            [
                line
                for line in record["raw_generator_response"].splitlines()
                if line.strip()
            ]
        ) > 1
        for record in parse_failures
    )

    intended_correct_em = torch.tensor(
        [float(record["intended_correct_em"]) for record in parsed_records]
    )
    intended_misleading_em = torch.tensor(
        [float(record["intended_misleading_em"]) for record in parsed_records]
    )
    grounded_mask = torch.tensor(
        [bool(record["grounded_pair"]) for record in parsed_records]
    )

    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "split": "custom_question_file" if args.question_file else args.split,
        "rankalign_commit": _rankalign_commit(rankalign_root),
        "num_questions": len(records),
        "num_parsed": len(parsed_records),
        "parse_success_rate": len(parsed_records) / len(records),
        "question_file": args.question_file,
        "parse_mode_counts": parse_mode_counts,
        "num_parse_failures": len(parse_failures),
        "parse_failures_with_delimiter_anywhere": parse_failure_with_delimiter,
        "parse_failures_multiline": parse_failure_multiline,
        "data_seed": args.data_seed,
        "generation_seed": args.generation_seed,
        "order_seed": args.order_seed,
        "validator_sample_seed": args.validator_sample_seed,
        "generation_mode": args.generation_mode,
        "generation_temperature": args.generation_temperature,
        "generation_top_p": args.generation_top_p,
        "max_new_tokens": args.max_new_tokens,
        "validator_temperature": args.validator_temperature,
        "validator_labels": list(validator_labels),
        "generator_intended_correct_em": float(intended_correct_em.mean().item()),
        "generator_intended_misleading_em": float(
            intended_misleading_em.mean().item()
        ),
        "generator_grounded_pair_fraction": float(
            grounded_mask.to(torch.float32).mean().item()
        ),
        "target_a_fraction_all": float(
            torch.tensor(targets, dtype=torch.float32).eq(0).to(torch.float32).mean().item()
        ),
        "target_a_fraction_parsed": float(
            target_a.to(torch.float32).mean().item()
        ),
        "target_a_fraction": float(target_a.to(torch.float32).mean().item()),
        "prediction_a_fraction": float(
            (prediction_tensor == 0).to(torch.float32).mean().item()
        ),
        "hard_consistency": float(hard_tensor.mean().item()),
        "mean_soft_consistency": float(soft_tensor.mean().item()),
        "sampled_hard_consistency": float(sampled_hard_tensor.mean().item()),
        "sampled_hard_mc_mean": float(sampled_hard_mc_tensor.mean().item()),
        "sampled_hard_mc_minus_soft_mean": float(
            (sampled_hard_mc_tensor.mean() - soft_tensor.mean()).item()
        ),
        "sampled_hard_draws_per_example": args.sampled_hard_draws,
        "hard_retained_fraction": float(retained_fraction(hard_tensor).item()),
        "soft_ess": float(ess.item()),
        "soft_ess_fraction": float((ess / len(soft_tensor)).item()),
        "soft_normalized_entropy": float(normalized_entropy(soft_tensor).item()),
        "soft_max_normalized_weight": float(normalized.max().item()),
    }

    if target_a.any():
        summary["hard_consistency_target_a"] = float(
            hard_tensor[target_a].mean().item()
        )
        summary["mean_soft_consistency_target_a"] = float(
            soft_tensor[target_a].mean().item()
        )
    if target_b.any():
        summary["hard_consistency_target_b"] = float(
            hard_tensor[target_b].mean().item()
        )
        summary["mean_soft_consistency_target_b"] = float(
            soft_tensor[target_b].mean().item()
        )

    if grounded_mask.any():
        summary["grounded_pair_count"] = int(grounded_mask.sum().item())
        summary["validator_accuracy_on_grounded_pairs"] = float(
            hard_tensor[grounded_mask].mean().item()
        )
        summary["mean_soft_on_grounded_pairs"] = float(
            soft_tensor[grounded_mask].mean().item()
        )
    else:
        summary["grounded_pair_count"] = 0
        summary["validator_accuracy_on_grounded_pairs"] = None
        summary["mean_soft_on_grounded_pairs"] = None

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
