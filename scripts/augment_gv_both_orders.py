#!/usr/bin/env python3
"""Score both A/B placements on frozen generated GV pairs without regenerating answers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from gvsmc.prompts import triviaqa_paper_validator_prompt
from gvsmc.randomization import binary_choice_with_target_index
from gvsmc.scoring import validator_label_logprobs
from run_triviaqa_gv import _load_model, _resolve_chat, _resolve_device, _soft_target_probability


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="google/gemma-2-2b-it")
    p.add_argument("--adapter", default=None,
                   help="Must be the checkpoint that generated/scored these frozen trajectories.")
    p.add_argument("--device", default="auto")
    p.add_argument("--chat", choices=["auto", "true", "false"], default="auto")
    p.add_argument("--validator-temperature", type=float, default=1.0)
    p.add_argument("--reuse-existing-scores", action="store_true",
                   help="Reuse both-order scores already in the records; only valid if model/adapter/temperature match.")
    return p.parse_args()


def main():
    args = parse_args()
    if Path(args.data).resolve() == Path(args.output).resolve():
        raise ValueError("output must differ from input so source data remain immutable")
    if args.validator_temperature <= 0:
        raise ValueError("validator temperature must be positive")
    with Path(args.data).open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    need = [
        r for r in rows if r.get("parse_success", False)
        and not (args.reuse_existing_scores
                 and "p_correct_when_a" in r and "p_correct_when_b" in r)
    ]
    device = _resolve_device(args.device)
    print(f"PyTorch device: {device}", flush=True)
    is_chat = _resolve_chat(args.chat, args.model)
    labels = ("A", "B") if is_chat else (" A", " B")
    model, tokenizer = (None, None)
    if need:
        model, tokenizer = _load_model(args.model, device, adapter=args.adapter)
    need_ids = {id(item) for item in need}
    out_rows = []
    gaps = []
    # ETA reflects the number of actual model-scored pairs, not the faster
    # parse-failure and already-cached rows.
    progress = tqdm(
        total=len(need), desc="A/B validator scores", unit="pair",
        dynamic_ncols=True, mininterval=1.0,
    )
    for row in rows:
        r = dict(row)
        if not r.get("parse_success", False):
            out_rows.append(r)
            continue
        choices = [
            binary_choice_with_target_index(
                r["intended_correct"], r["intended_misleading"], target_index=index)
            for index in (0, 1)
        ]
        prompts = [
            triviaqa_paper_validator_prompt(r["question"], choice)
            for choice in choices
        ]
        r["validator_prompt_when_a"], r["validator_prompt_when_b"] = prompts
        if id(row) in need_ids:
            probs = []
            for index, prompt in enumerate(prompts):
                scores = validator_label_logprobs(
                    model, tokenizer, prompt, labels=labels,
                    is_chat=is_chat, device=device,
                )
                probs.append(_soft_target_probability(scores, index, args.validator_temperature))
            r["p_correct_when_a"], r["p_correct_when_b"] = probs
            progress.update(1)
        a = float(r["p_correct_when_a"])
        b = float(r["p_correct_when_b"])
        if not (0 <= a <= 1 and 0 <= b <= 1):
            raise ValueError("invalid A/B probabilities")
        r["order_marginalized_soft_weight"] = .5 * (a + b)
        r["paired_order_mean_abs_gap"] = abs(a - b)
        gaps.append(abs(a - b))
        out_rows.append(r)
    progress.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "source": str(args.data),
        "output": str(output),
        "model": args.model,
        "adapter": args.adapter,
        "validator_temperature": args.validator_temperature,
        "num_records": len(rows),
        "num_parsed": len(gaps),
        "num_pairs_recomputed": len(need),
        "mean_abs_order_gap": sum(gaps) / len(gaps) if gaps else None,
        "training_potential": "order_marginalized_soft_weight=0.5*(p_correct_when_a+p_correct_when_b)",
        "caution": "Reusing scores is valid only if the model, adapter, label tokenization, and temperature match the original scores.",
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Wrote:", output)


if __name__ == "__main__":
    main()
