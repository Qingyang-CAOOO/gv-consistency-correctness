#!/usr/bin/env python3
"""Add generator-only conditional answer likelihoods to frozen paired-answer GV data.

Never uses TriviaQA gold answers for scoring; gold fields are copied unchanged
only so a separate calibration script can construct supervised labels.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from tqdm import tqdm

from gvsmc.prompts import triviaqa_candidate_prompt
from gvsmc.scoring import completion_logprob, completion_token_ids_for_prompt
from run_triviaqa_gv import _load_model, _resolve_chat, _resolve_device


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="google/gemma-2-2b-it")
    p.add_argument("--adapter", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--chat", choices=("auto", "true", "false"), default="auto")
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def candidate_features(model, tokenizer, question, candidate, *, is_chat, device):
    prompt = triviaqa_candidate_prompt(question)
    token_ids = completion_token_ids_for_prompt(
        tokenizer, prompt, candidate, is_chat=is_chat
    )
    logp = completion_logprob(
        model, tokenizer, prompt, candidate,
        is_chat=is_chat, device=device,
    )
    n = len(token_ids)
    if n < 1 or not math.isfinite(logp):
        raise ValueError("empty candidate or nonfinite candidate log probability")
    return {"logprob": float(logp), "per_token_logprob": float(logp / n),
            "num_tokens": int(n)}


def main():
    a = parse_args()
    src, dst = Path(a.data), Path(a.output)
    if src.resolve() == dst.resolve():
        raise ValueError("source and destination must differ")
    if dst.exists() and not a.overwrite:
        raise FileExistsError(f"output exists: {dst}; use a distinct output or --overwrite")
    with src.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    device = _resolve_device(a.device)
    if a.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA required; select an available GPU runtime first")
    print(f"PyTorch device: {device}; adapter: {a.adapter}", flush=True)
    chat = _resolve_chat(a.chat, a.model)
    valid = sum(bool(r.get("parse_success")) for r in rows)
    model = tokenizer = None
    if valid:
        model, tokenizer = _load_model(a.model, device, adapter=a.adapter)
    results = []
    with tqdm(total=valid, desc="Generator preferences", unit="pair",
              dynamic_ncols=True, mininterval=1.0) as bar:
        for row in rows:
            rec = dict(row)
            if rec.get("parse_success"):
                pos = candidate_features(
                    model, tokenizer, rec["question"], rec["intended_correct"],
                    is_chat=chat, device=device,
                )
                neg = candidate_features(
                    model, tokenizer, rec["question"], rec["intended_misleading"],
                    is_chat=chat, device=device,
                )
                rec["generator_candidate_positive_score"] = pos
                rec["generator_candidate_negative_score"] = neg
                rec["generator_candidate_score_margin"] = (
                    pos["per_token_logprob"] - neg["per_token_logprob"]
                )
                bar.update(1)
            results.append(rec)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "source": str(src), "output": str(dst), "model": a.model,
        "adapter": a.adapter, "device": str(device), "num_records": len(rows),
        "num_parsed": valid,
        "score_definition": (
            "length-normalized log P(candidate answer | direct-answer question prompt), "
            "computed independently for both candidates"
        ),
        "leakage_policy": (
            "The scoring algorithm never reads gold answers or EM labels; "
            "those source fields are copied for later supervised calibration."
        ),
    }
    dst.with_suffix(".summary.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
