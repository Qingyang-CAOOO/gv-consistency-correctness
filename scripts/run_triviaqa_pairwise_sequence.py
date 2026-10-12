#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from gvsmc.prompts import (
    num_triviaqa_pairwise_validator_templates,
    triviaqa_pairwise_validator_prompt,
)
from gvsmc.randomization import binary_choice_with_target_index
from gvsmc.scoring import validator_label_logprobs
from gvsmc.weights import soft_randomized_gv_weight_from_label_logprobs


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Score several semantically equivalent pairwise TriviaQA validators on a frozen "
            "candidate pool. Each template is evaluated in both A/B orders and marginalized."
        )
    )
    p.add_argument("--data", required=True)
    p.add_argument("--template0-data", default=None)
    p.add_argument("--model", default="google/gemma-2-2b-it")
    p.add_argument("--adapter", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--chat", choices=["auto", "true", "false"], default="auto")
    p.add_argument("--validator-temperature", type=float, default=1.0)
    p.add_argument("--output", required=True)
    return p.parse_args()


def resolve_device(arg):
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_chat(arg, model_name):
    if arg == "true":
        return True
    if arg == "false":
        return False
    name = model_name.lower()
    return "instruct" in name or "chat" in name or name.endswith("-it")


def load_model(model_name, adapter, device):
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


def target_prob(logprobs, target_index, temperature):
    target = torch.tensor([target_index], dtype=torch.long)
    return float(
        soft_randomized_gv_weight_from_label_logprobs(
            logprobs.unsqueeze(0), target, temperature=temperature
        )[0].item()
    )


def load_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main():
    args = parse_args()
    source = [r for r in load_jsonl(args.data) if r.get("status") == "ok"]
    if not source:
        raise RuntimeError("no usable candidate pools")
    template0 = {}
    if args.template0_data:
        template0 = {int(r["index"]): r for r in load_jsonl(args.template0_data)}

    device = resolve_device(args.device)
    is_chat = resolve_chat(args.chat, args.model)
    labels = ("A", "B") if is_chat else (" A", " B")
    n_templates = num_triviaqa_pairwise_validator_templates()
    print(f"model: {args.model}")
    print(f"device: {device}")
    print(f"usable candidate pools: {len(source)}")
    print(f"pairwise templates: {n_templates}")
    print(f"reuse template 0: {bool(template0)}")

    model, tokenizer = load_model(args.model, args.adapter, device)
    results = []

    for row in tqdm(source, desc="candidate pools"):
        candidates = list(row["candidates"])
        k = len(candidates)
        matrices = []
        order_gaps = []

        for template_index in range(n_templates):
            if template_index == 0 and int(row["index"]) in template0:
                cached = template0[int(row["index"])]
                matrices.append(cached["pairwise_win_probabilities"])
                order_gaps.append(float(cached["mean_pairwise_order_gap"]))
                continue

            M = [[None for _ in range(k)] for _ in range(k)]
            gaps = []
            for i in range(k):
                for j in range(i + 1, k):
                    per_order = []
                    for target_index in (0, 1):
                        choice = binary_choice_with_target_index(
                            candidates[i], candidates[j], target_index=target_index
                        )
                        prompt = triviaqa_pairwise_validator_prompt(
                            row["question"], choice, template_index=template_index
                        )
                        logprobs = validator_label_logprobs(
                            model, tokenizer, prompt, labels=labels,
                            is_chat=is_chat, device=device
                        )
                        per_order.append(
                            target_prob(logprobs, target_index, args.validator_temperature)
                        )
                    p_i = 0.5 * (per_order[0] + per_order[1])
                    M[i][j] = p_i
                    M[j][i] = 1.0 - p_i
                    gaps.append(abs(per_order[0] - per_order[1]))
            matrices.append(M)
            order_gaps.append(sum(gaps) / len(gaps) if gaps else float("nan"))

        results.append({
            "index": row["index"],
            "question": row["question"],
            "candidates": candidates,
            "candidate_gold_mask": row["candidate_gold_mask"],
            "gold_coverage": row["gold_coverage"],
            "initial_weights": row["initial_weights"],
            "initial_top_index": row["initial_top_index"],
            "initial_top_em": row["initial_top_em"],
            "pairwise_win_probabilities_by_template": matrices,
            "mean_pairwise_order_gap_by_template": order_gaps,
        })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for r in results:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "source_data": args.data,
        "template0_data": args.template0_data,
        "num_questions": len(results),
        "num_templates": n_templates,
        "mean_order_gap_by_template": [
            float(sum(r["mean_pairwise_order_gap_by_template"][t] for r in results) / len(results))
            for t in range(n_templates)
        ],
        "interpretation": (
            "Multiple order-marginalized pairwise validator measurements on the same frozen "
            "candidate pools. Template 0 matches the Phase-3A pairwise validator."
        ),
    }
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote: {out}")
    print(f"wrote: {summary_path}")


if __name__ == "__main__":
    main()
