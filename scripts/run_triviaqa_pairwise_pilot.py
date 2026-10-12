#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from gvsmc.metrics import effective_sample_size, normalized_entropy
from gvsmc.prompts import triviaqa_validator_prompt
from gvsmc.randomization import binary_choice_with_target_index
from gvsmc.scoring import validator_label_logprobs
from gvsmc.sequential import posterior_mass, sequential_importance_update, top_particle_index
from gvsmc.weights import soft_randomized_gv_weight_from_label_logprobs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Paper-style pairwise QA validator pilot on a frozen candidate pool. "
            "Each pair is evaluated in both A/B orders and the nuisance order is marginalized."
        )
    )
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="google/gemma-2-2b-it")
    p.add_argument("--adapter", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--chat", choices=["auto", "true", "false"], default="auto")
    p.add_argument("--validator-temperature", type=float, default=1.0)
    p.add_argument("--output", required=True)
    return p.parse_args()


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
    name = model_name.lower()
    return "instruct" in name or "chat" in name or name.endswith("-it")


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


def target_prob(logprobs: torch.Tensor, target_index: int, temperature: float) -> float:
    target = torch.tensor([target_index], dtype=torch.long)
    return float(
        soft_randomized_gv_weight_from_label_logprobs(
            logprobs.unsqueeze(0), target, temperature=temperature
        )[0].item()
    )


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in Path(args.data).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    usable = [r for r in rows if r.get("status") == "ok"]
    if not usable:
        raise RuntimeError("no usable candidate pools in input data")

    device = resolve_device(args.device)
    is_chat = resolve_chat(args.chat, args.model)
    labels = ("A", "B") if is_chat else (" A", " B")
    print(f"model: {args.model}")
    print(f"adapter: {args.adapter}")
    print(f"device: {device}")
    print(f"usable candidate pools: {len(usable)}")
    print("validator: paper-style pairwise QA; A/B order marginalized")

    model, tokenizer = load_model(args.model, args.adapter, device)
    results = []

    for row in tqdm(usable, desc="candidate pools"):
        candidates = list(row["candidates"])
        k = len(candidates)
        if k < 2:
            continue
        prior = torch.tensor(row["initial_weights"], dtype=torch.float32)
        gold = torch.tensor(row["candidate_gold_mask"], dtype=torch.bool)

        pairwise = [[None for _ in range(k)] for _ in range(k)]
        order_gaps = []

        for i in range(k):
            for j in range(i + 1, k):
                p_i_by_order = []
                for target_index in (0, 1):
                    choice = binary_choice_with_target_index(
                        candidates[i],
                        candidates[j],
                        target_index=target_index,
                    )
                    prompt = triviaqa_validator_prompt(row["question"], choice)
                    logprobs = validator_label_logprobs(
                        model,
                        tokenizer,
                        prompt,
                        labels=labels,
                        is_chat=is_chat,
                        device=device,
                    )
                    p_i_by_order.append(
                        target_prob(logprobs, target_index, args.validator_temperature)
                    )
                p_i = 0.5 * (p_i_by_order[0] + p_i_by_order[1])
                p_j = 1.0 - p_i
                pairwise[i][j] = p_i
                pairwise[j][i] = p_j
                order_gaps.append(abs(p_i_by_order[0] - p_i_by_order[1]))

        potentials = []
        for i in range(k):
            wins = [float(pairwise[i][j]) for j in range(k) if j != i]
            potentials.append(mean(wins))

        potential_tensor = torch.tensor(potentials, dtype=torch.float32)
        posterior = sequential_importance_update(prior, potential_tensor)
        before_top = top_particle_index(prior)
        after_top = top_particle_index(posterior)
        before_mass = float(posterior_mass(gold, prior).item())
        after_mass = float(posterior_mass(gold, posterior).item())
        before_ess = float(effective_sample_size(prior, normalized=True).item())
        after_ess = float(effective_sample_size(posterior, normalized=True).item())
        before_entropy = float(normalized_entropy(prior, normalized=True).item())
        after_entropy = float(normalized_entropy(posterior, normalized=True).item())

        results.append(
            {
                "index": row["index"],
                "question": row["question"],
                "candidates": candidates,
                "candidate_gold_mask": [bool(x) for x in gold.tolist()],
                "gold_coverage": bool(gold.any().item()),
                "initial_weights": [float(x) for x in prior.tolist()],
                "pairwise_win_probabilities": pairwise,
                "candidate_pairwise_potentials": potentials,
                "posterior_weights": [float(x) for x in posterior.tolist()],
                "initial_top_index": before_top,
                "initial_top_em": bool(gold[before_top].item()),
                "posterior_top_index": after_top,
                "posterior_top_em": bool(gold[after_top].item()),
                "initial_gold_mass": before_mass,
                "posterior_gold_mass": after_mass,
                "delta_gold_mass": after_mass - before_mass,
                "initial_ess_fraction": before_ess / k,
                "posterior_ess_fraction": after_ess / k,
                "initial_normalized_entropy": before_entropy,
                "posterior_normalized_entropy": after_entropy,
                "mean_pairwise_order_gap": mean(order_gaps),
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for r in results:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")

    covered = [r for r in results if r["gold_coverage"]]
    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "source_data": args.data,
        "num_usable_questions": len(results),
        "num_gold_covered": len(covered),
        "gold_candidate_coverage": len(covered) / len(results) if results else float("nan"),
        "initial_top1_em": mean([float(r["initial_top_em"]) for r in results]),
        "pairwise_top1_em": mean([float(r["posterior_top_em"]) for r in results]),
        "initial_gold_mass": mean([r["initial_gold_mass"] for r in results]),
        "pairwise_gold_mass": mean([r["posterior_gold_mass"] for r in results]),
        "initial_gold_mass_covered": mean([r["initial_gold_mass"] for r in covered]),
        "pairwise_gold_mass_covered": mean([r["posterior_gold_mass"] for r in covered]),
        "mean_delta_gold_mass_covered": mean([r["delta_gold_mass"] for r in covered]),
        "fraction_delta_gold_mass_positive_covered": mean(
            [float(r["delta_gold_mass"] > 0) for r in covered]
        ),
        "initial_ess_fraction": mean([r["initial_ess_fraction"] for r in results]),
        "pairwise_ess_fraction": mean([r["posterior_ess_fraction"] for r in results]),
        "initial_normalized_entropy": mean([r["initial_normalized_entropy"] for r in results]),
        "pairwise_normalized_entropy": mean([r["posterior_normalized_entropy"] for r in results]),
        "mean_pairwise_order_gap": mean([r["mean_pairwise_order_gap"] for r in results]),
        "interpretation": (
            "One paper-style pairwise QA validation measurement over the same frozen candidate pools. "
            "A/B order is marginalized before candidate-level reweighting."
        ),
    }
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote: {out}")
    print(f"wrote: {summary_path}")


if __name__ == "__main__":
    main()
