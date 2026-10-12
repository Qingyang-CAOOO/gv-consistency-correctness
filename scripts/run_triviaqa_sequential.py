#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path
from random import Random

import torch
from tqdm import tqdm

from gvsmc.data.triviaqa import load_rankalign_triviaqa_questions, rankalign_triviaqa_path
from gvsmc.evaluation import matches_any_gold, normalize_qa_answer
from gvsmc.generation import generate_text
from gvsmc.metrics import effective_sample_size, normalized_entropy
from gvsmc.prompts import (
    num_triviaqa_correctness_templates,
    triviaqa_candidate_prompt,
    triviaqa_correctness_validator_prompt,
)
from gvsmc.randomization import balanced_targets
from gvsmc.scoring import validator_label_logprobs
from gvsmc.sequential import posterior_mass, sequential_importance_update, top_particle_index
from gvsmc.weights import normalize_weights, soft_randomized_gv_weight_from_label_logprobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sequential Feynman--Kac inference over TriviaQA answer hypotheses. "
            "Generator samples define a finite latent-state pool; randomized "
            "correctness validators provide sequential soft potentials."
        )
    )
    parser.add_argument("--rankalign-root", default="external/rankalign")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--generator-samples", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--generation-seed", type=int, default=7000)
    parser.add_argument("--order-seed", type=int, default=8000)
    parser.add_argument(
        "--order-mode",
        choices=["sampled", "marginalized"],
        default="sampled",
        help=(
            "sampled: one randomized A/B target position per candidate; "
            "marginalized: score both target positions and average over the randomization nuisance variable"
        ),
    )
    parser.add_argument("--generation-temperature", type=float, default=0.8)
    parser.add_argument("--generation-top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--validator-temperature", type=float, default=1.0)
    parser.add_argument("--chat", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        default="outputs/triviaqa_sequential/sequential.jsonl",
    )
    return parser.parse_args()


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


def load_model(model_name: str, adapter: str | None, device: torch.device):
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


def clean_short_answer(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    value = lines[0]
    value = re.sub(r"^[-*•]+\s*", "", value).strip()
    value = re.sub(r"^(?:answer|final answer)\s*:\s*", "", value, flags=re.IGNORECASE).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'\"', "'"}:
        value = value[1:-1].strip()
    return value


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
    return sum(values) / len(values) if values else float("nan")


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.generator_samples <= 0:
        raise ValueError("--generator-samples must be positive")
    if args.rounds < 0:
        raise ValueError("--rounds must be nonnegative")
    if args.rounds > num_triviaqa_correctness_templates():
        raise ValueError(
            f"--rounds={args.rounds} exceeds the number of distinct validator templates "
            f"({num_triviaqa_correctness_templates()})"
        )

    rankalign_root = Path(args.rankalign_root)
    data_path = rankalign_triviaqa_path(rankalign_root, args.split)
    questions = load_rankalign_triviaqa_questions(
        data_path,
        seed=args.data_seed,
        max_examples=args.limit,
    )

    device = resolve_device(args.device)
    is_chat = resolve_chat(args.chat, args.model)
    print(f"model: {args.model}")
    print(f"adapter: {args.adapter}")
    print(f"device: {device}")
    print(f"questions: {len(questions)}")
    print(f"generator samples/question: {args.generator_samples}")
    print(f"sequential rounds: {args.rounds}")
    print(f"order mode: {args.order_mode}")

    model, tokenizer = load_model(args.model, args.adapter, device)
    validator_labels = ("A", "B") if is_chat else (" A", " B")

    records: list[dict] = []
    usable_records: list[dict] = []

    for q_index, item in enumerate(tqdm(questions, desc="questions")):
        generator_prompt = triviaqa_candidate_prompt(item.question)
        pool: OrderedDict[str, dict] = OrderedDict()
        raw_generations: list[str] = []

        for sample_index in range(args.generator_samples):
            raw = generate_text(
                model,
                tokenizer,
                generator_prompt,
                is_chat=is_chat,
                device=device,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.generation_temperature,
                top_p=args.generation_top_p,
                seed=args.generation_seed + q_index * 10000 + sample_index,
            )
            raw_generations.append(raw)
            candidate = clean_short_answer(raw)
            key = normalize_qa_answer(candidate)
            if not key:
                continue
            if key not in pool:
                pool[key] = {"answer": candidate, "count": 0}
            pool[key]["count"] += 1

        candidates = [entry["answer"] for entry in pool.values()]
        counts = [int(entry["count"]) for entry in pool.values()]
        record = {
            "index": q_index,
            "question": item.question,
            "gold_answers": list(item.gold_answers),
            "generator_prompt": generator_prompt,
            "raw_generations": raw_generations,
            "num_unique_candidates": len(candidates),
            "candidates": candidates,
            "candidate_counts": counts,
        }

        if len(candidates) < 2:
            record["status"] = "insufficient_unique_candidates"
            records.append(record)
            continue

        initial = normalize_weights(torch.tensor(counts, dtype=torch.float32))
        gold_mask = torch.tensor(
            [matches_any_gold(candidate, item.gold_answers) for candidate in candidates],
            dtype=torch.bool,
        )
        weights = initial.clone()
        initial_top = top_particle_index(weights)
        initial_gold_mass = float(posterior_mass(gold_mask, weights).item())
        initial_ess = float(effective_sample_size(weights, normalized=True).item())
        initial_entropy = float(normalized_entropy(weights, normalized=True).item())

        record.update(
            {
                "status": "ok",
                "candidate_gold_mask": [bool(x) for x in gold_mask.tolist()],
                "gold_coverage": bool(gold_mask.any().item()),
                "initial_weights": [float(x) for x in weights.tolist()],
                "initial_top_index": initial_top,
                "initial_top_answer": candidates[initial_top],
                "initial_top_em": bool(gold_mask[initial_top].item()),
                "initial_gold_mass": initial_gold_mass,
                "initial_ess": initial_ess,
                "initial_ess_fraction": initial_ess / len(candidates),
                "initial_normalized_entropy": initial_entropy,
                "rounds": [],
            }
        )

        for round_index in range(args.rounds):
            potentials: list[float] = []
            potentials_target_a: list[float] = []
            potentials_target_b: list[float] = []

            if args.order_mode == "sampled":
                rng = Random(args.order_seed + q_index * 100 + round_index)
                targets = balanced_targets(len(candidates), rng=rng)

                for candidate_index, candidate in enumerate(candidates):
                    target_index = int(targets[candidate_index])
                    validator_prompt = triviaqa_correctness_validator_prompt(
                        item.question,
                        candidate,
                        target_index=target_index,
                        template_index=round_index,
                    )
                    label_logprobs = validator_label_logprobs(
                        model,
                        tokenizer,
                        validator_prompt,
                        labels=validator_labels,
                        is_chat=is_chat,
                        device=device,
                    )
                    potentials.append(
                        soft_target_probability(
                            label_logprobs,
                            target_index,
                            args.validator_temperature,
                        )
                    )
            else:
                targets = [None] * len(candidates)
                for candidate in candidates:
                    per_order: list[float] = []
                    for target_index in (0, 1):
                        validator_prompt = triviaqa_correctness_validator_prompt(
                            item.question,
                            candidate,
                            target_index=target_index,
                            template_index=round_index,
                        )
                        label_logprobs = validator_label_logprobs(
                            model,
                            tokenizer,
                            validator_prompt,
                            labels=validator_labels,
                            is_chat=is_chat,
                            device=device,
                        )
                        per_order.append(
                            soft_target_probability(
                                label_logprobs,
                                target_index,
                                args.validator_temperature,
                            )
                        )
                    g_a, g_b = per_order
                    potentials_target_a.append(g_a)
                    potentials_target_b.append(g_b)
                    potentials.append(0.5 * (g_a + g_b))

            potential_tensor = torch.tensor(potentials, dtype=torch.float32)
            weights = sequential_importance_update(weights, potential_tensor)
            top_index = top_particle_index(weights)
            ess = float(effective_sample_size(weights, normalized=True).item())
            entropy = float(normalized_entropy(weights, normalized=True).item())
            gold_mass = float(posterior_mass(gold_mask, weights).item())

            record["rounds"].append(
                {
                    "round": round_index + 1,
                    "template_index": round_index,
                    "order_mode": args.order_mode,
                    "target_indices": (
                        [int(x) for x in targets]
                        if args.order_mode == "sampled"
                        else None
                    ),
                    "potentials": potentials,
                    "potentials_target_a": (
                        potentials_target_a
                        if args.order_mode == "marginalized"
                        else None
                    ),
                    "potentials_target_b": (
                        potentials_target_b
                        if args.order_mode == "marginalized"
                        else None
                    ),
                    "weights": [float(x) for x in weights.tolist()],
                    "top_index": top_index,
                    "top_answer": candidates[top_index],
                    "top_em": bool(gold_mask[top_index].item()),
                    "gold_mass": gold_mass,
                    "ess": ess,
                    "ess_fraction": ess / len(candidates),
                    "normalized_entropy": entropy,
                    "max_weight": float(weights.max().item()),
                }
            )

        records.append(record)
        usable_records.append(record)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if not usable_records:
        raise RuntimeError("no questions produced at least two unique candidate hypotheses")

    top_em_by_step = [
        mean([float(record["initial_top_em"]) for record in usable_records])
    ]
    gold_mass_by_step = [
        mean([float(record["initial_gold_mass"]) for record in usable_records])
    ]
    ess_fraction_by_step = [
        mean([float(record["initial_ess_fraction"]) for record in usable_records])
    ]
    entropy_by_step = [
        mean([float(record["initial_normalized_entropy"]) for record in usable_records])
    ]

    covered = [record for record in usable_records if record["gold_coverage"]]
    gold_mass_conditional_by_step = [
        mean([float(record["initial_gold_mass"]) for record in covered])
    ]

    for round_index in range(args.rounds):
        top_em_by_step.append(
            mean([float(record["rounds"][round_index]["top_em"]) for record in usable_records])
        )
        gold_mass_by_step.append(
            mean([float(record["rounds"][round_index]["gold_mass"]) for record in usable_records])
        )
        ess_fraction_by_step.append(
            mean([float(record["rounds"][round_index]["ess_fraction"]) for record in usable_records])
        )
        entropy_by_step.append(
            mean([float(record["rounds"][round_index]["normalized_entropy"]) for record in usable_records])
        )
        gold_mass_conditional_by_step.append(
            mean([float(record["rounds"][round_index]["gold_mass"]) for record in covered])
        )

    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "split": args.split,
        "num_questions": len(records),
        "num_usable_questions": len(usable_records),
        "usable_fraction": len(usable_records) / len(records),
        "mean_unique_candidates": mean(
            [float(record["num_unique_candidates"]) for record in usable_records]
        ),
        "generator_samples": args.generator_samples,
        "rounds": args.rounds,
        "gold_candidate_coverage": mean(
            [float(record["gold_coverage"]) for record in usable_records]
        ),
        "top1_em_by_step": top_em_by_step,
        "gold_mass_by_step": gold_mass_by_step,
        "gold_mass_conditional_on_coverage_by_step": gold_mass_conditional_by_step,
        "mean_ess_fraction_by_step": ess_fraction_by_step,
        "mean_normalized_entropy_by_step": entropy_by_step,
        "data_seed": args.data_seed,
        "generation_seed": args.generation_seed,
        "order_seed": args.order_seed,
        "order_mode": args.order_mode,
        "generation_temperature": args.generation_temperature,
        "generation_top_p": args.generation_top_p,
        "validator_temperature": args.validator_temperature,
        "interpretation": (
            "Generator-only empirical candidate measure; no validator update was applied."
            if args.rounds == 0
            else (
                "Sequential Feynman-Kac consistency measure over a static finite answer pool; "
                + (
                    "order_mode=marginalized integrates out randomized A/B label placement before weighting. "
                    if args.order_mode == "marginalized"
                    else "order_mode=sampled uses one randomized A/B placement per candidate. "
                )
                + "This is not a calibrated factual posterior unless the validator potentials satisfy "
                "a separate likelihood/calibration model."
            )
        ),
    }

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
