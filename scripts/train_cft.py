#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from gvsmc.cft import CFTExample, build_cft_examples, read_gv_jsonl
from gvsmc.scoring import (
    DEFAULT_SYSTEM_PROMPT,
    completion_token_ids_for_prompt,
    prompt_token_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA consistency fine-tuning on frozen generator-validator JSONL."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--mode",
        choices=[
            "hard", "sampled-hard", "soft", "soft-marginalized",
            "posterior-g", "posterior-gv", "posterior-g-ess",
        ],
        required=True,
    )
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument(
        "--updates-per-epoch",
        type=int,
        default=None,
        help=(
            "If set, partition each epoch into exactly this many optimizer "
            "updates. This is useful for matching hard and soft CFT training "
            "budgets despite different dataset sizes."
        ),
    )
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=0.3)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Debug-only cap applied before hard/soft selection.",
    )
    parser.add_argument(
        "--components",
        choices=["both", "generator", "validator"],
        default="both",
    )
    return parser.parse_args()


def _is_chat_model(model_name: str) -> bool:
    lowered = model_name.lower()
    return (
        "instruct" in lowered
        or "chat" in lowered
        or lowered.endswith("-it")
    )


def _chat_prefix_and_full_ids(tokenizer, prompt: str, completion: str):
    variants = [
        [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        [
            {
                "role": "user",
                "content": f"{DEFAULT_SYSTEM_PROMPT}\n\n{prompt}",
            }
        ],
    ]

    last_error = None
    for prompt_messages in variants:
        try:
            prefix = tokenizer.apply_chat_template(
                prompt_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors=None,
            )
            full = tokenizer.apply_chat_template(
                prompt_messages
                + [{"role": "assistant", "content": completion}],
                add_generation_prompt=False,
                tokenize=True,
                return_tensors=None,
            )
        except Exception as exc:
            last_error = exc
            continue

        if prefix and isinstance(prefix[0], list):
            prefix = prefix[0]
        if full and isinstance(full[0], list):
            full = full[0]
        prefix = list(prefix)
        full = list(full)

        if full[: len(prefix)] != prefix:
            raise ValueError(
                "chat template full sequence does not preserve the generation prefix"
            )
        return prefix, full

    if last_error is not None:
        raise last_error
    raise RuntimeError("could not construct chat-formatted training sequence")


def _encode_example(
    tokenizer,
    example: CFTExample,
    *,
    is_chat: bool,
    max_seq_length: int,
) -> dict:
    if is_chat:
        prefix, full = _chat_prefix_and_full_ids(
            tokenizer,
            example.prompt,
            example.completion,
        )
    else:
        prefix = prompt_token_ids(tokenizer, example.prompt, is_chat=False)
        suffix = completion_token_ids_for_prompt(
            tokenizer,
            example.prompt,
            example.completion,
            is_chat=False,
        )
        full = prefix + suffix

    if len(full) > max_seq_length:
        raise ValueError(
            f"trajectory {example.trajectory_index} {example.component} "
            f"has {len(full)} tokens > max_seq_length={max_seq_length}"
        )
    if len(full) <= len(prefix):
        raise ValueError(
            f"trajectory {example.trajectory_index} has no completion tokens"
        )

    labels = [-100] * len(prefix) + full[len(prefix) :]
    return {
        "input_ids": full,
        "labels": labels,
        "weight": float(example.weight),
        "trajectory_index": example.trajectory_index,
        "component": example.component,
    }


class EncodedDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


class CompletionCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, rows: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(row["input_ids"]) for row in rows)
        batch_size = len(rows)

        input_ids = torch.full(
            (batch_size, max_len),
            self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.long,
        )
        labels = torch.full(
            (batch_size, max_len),
            -100,
            dtype=torch.long,
        )
        weights = torch.empty(batch_size, dtype=torch.float32)

        for i, row in enumerate(rows):
            length = len(row["input_ids"])
            input_ids[i, :length] = torch.tensor(row["input_ids"])
            attention_mask[i, :length] = 1
            labels[i, :length] = torch.tensor(row["labels"])
            weights[i] = float(row["weight"])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "weights": weights,
        }


def _sequence_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    token_loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view_as(shift_labels)
    mask = shift_labels.ne(-100)
    if not torch.all(mask.sum(dim=1) > 0):
        raise ValueError("every training sequence must contain completion tokens")
    return (token_loss * mask).sum(dim=1)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive")
    if args.updates_per_epoch is not None and args.updates_per_epoch <= 0:
        raise ValueError("--updates-per-epoch must be positive")
    if args.max_seq_length <= 0:
        raise ValueError("--max-seq-length must be positive")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite an existing CFT checkpoint: {output_dir}; "
            "use a new --output-dir for a new independent run"
        )

    _set_seed(args.seed)

    records = read_gv_jsonl(args.data)
    if args.max_trajectories is not None:
        if args.max_trajectories <= 0:
            raise ValueError("--max-trajectories must be positive")
        records = records[: args.max_trajectories]

    components = (
        ("generator", "validator")
        if args.components == "both"
        else (args.components,)
    )
    examples = build_cft_examples(
        records,
        mode=args.mode,
        components=components,
    )

    trajectory_ids = sorted({example.trajectory_index for example in examples})
    weight_mean = sum(example.weight for example in examples) / len(examples)
    print(f"mode: {args.mode}")
    print(f"source records: {len(records)}")
    print(f"selected trajectories: {len(trajectory_ids)}")
    print(f"training sequences: {len(examples)}")
    print(f"mean sequence weight: {weight_mean:.6f}")
    print(f"components: {components}")

    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        get_constant_schedule_with_warmup,
    )

    is_chat = _is_chat_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    encoded_rows = [
        _encode_example(
            tokenizer,
            example,
            is_chat=is_chat,
            max_seq_length=args.max_seq_length,
        )
        for example in examples
    ]
    lengths = [len(row["input_ids"]) for row in encoded_rows]
    completion_lengths = [
        sum(label != -100 for label in row["labels"])
        for row in encoded_rows
    ]
    print(
        "token lengths: "
        f"max={max(lengths)}, mean={sum(lengths)/len(lengths):.1f}; "
        "completion max="
        f"{max(completion_lengths)}, "
        f"mean={sum(completion_lengths)/len(completion_lengths):.1f}"
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CFT training currently requires a CUDA GPU")

    device = torch.device("cuda")
    use_bf16 = bool(torch.cuda.is_bf16_supported())
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"training dtype: {dtype}")

    model_kwargs = {"torch_dtype": dtype}
    if "gemma-2" in args.model.lower():
        model_kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.config.use_cache = False
    model.to(device)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    _set_seed(args.seed)
    model = get_peft_model(model, lora_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total_params = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.3f}%)"
    )

    dataset = EncodedDataset(encoded_rows)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        generator=generator,
        collate_fn=CompletionCollator(tokenizer.pad_token_id),
    )

    trainable = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=0.0,
    )

    if args.updates_per_epoch is None:
        updates_per_epoch = math.ceil(
            len(loader) / args.gradient_accumulation_steps
        )
    else:
        if args.updates_per_epoch > len(loader):
            raise ValueError(
                "--updates-per-epoch cannot exceed the number of training batches"
            )
        updates_per_epoch = args.updates_per_epoch
    total_updates = updates_per_epoch * args.epochs
    warmup_steps = math.ceil(total_updates * args.warmup_ratio)
    scheduler = get_constant_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
    )
    print(
        f"optimizer updates: {total_updates} "
        f"({updates_per_epoch}/epoch), warmup={warmup_steps}"
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_update = 0
    loss_history: list[float] = []
    progress = tqdm(
        total=total_updates, desc=f"CFT {args.mode}", unit="update",
        dynamic_ncols=True, mininterval=1.0,
    )

    for epoch in range(args.epochs):
        epoch_weighted_loss = 0.0
        epoch_examples = 0
        num_batches = len(loader)

        for batch_index, batch in enumerate(loader):
            if args.updates_per_epoch is None:
                group_start = (
                    batch_index // args.gradient_accumulation_steps
                ) * args.gradient_accumulation_steps
                group_end = min(
                    group_start + args.gradient_accumulation_steps,
                    num_batches,
                )
            else:
                # Partition all batches into exactly K contiguous groups whose
                # sizes differ by at most one. This lets hard and soft CFT use
                # the same number of optimizer updates per epoch while still
                # making one complete pass through their respective datasets.
                k = args.updates_per_epoch
                group_index = min(
                    (batch_index * k) // num_batches,
                    k - 1,
                )
                group_start = math.ceil(group_index * num_batches / k)
                group_end = math.ceil((group_index + 1) * num_batches / k)

            group_size = group_end - group_start

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            weights = batch["weights"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            seq_nll = _sequence_nll(outputs.logits, labels)

            # Global normalization gives a minibatch estimator of
            # sum_i w_i L_i / sum_i w_i, rather than renormalizing each
            # minibatch independently.
            loss = (weights * seq_nll).mean() / weight_mean
            (loss / group_size).backward()

            epoch_weighted_loss += float(loss.detach().item())
            epoch_examples += 1

            end_of_group = batch_index + 1 == group_end
            if end_of_group:
                torch.nn.utils.clip_grad_norm_(
                    trainable,
                    args.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1
                progress.set_postfix(
                    epoch=f"{epoch + 1}/{args.epochs}",
                    seq_nll=f"{epoch_weighted_loss / epoch_examples:.3f}",
                )
                progress.update(1)

        epoch_mean = epoch_weighted_loss / max(epoch_examples, 1)
        loss_history.append(epoch_mean)
        print(
            f"epoch {epoch + 1}/{args.epochs} complete: "
            f"mean_weighted_seq_nll={epoch_mean:.4f}"
        )

    progress.close()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    summary = {
        "mode": args.mode,
        "model": args.model,
        "data": str(args.data),
        "source_records": len(records),
        "selected_trajectories": len(trajectory_ids),
        "training_sequences": len(examples),
        "components": list(components),
        "mean_sequence_weight": weight_mean,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "requested_updates_per_epoch": args.updates_per_epoch,
        "actual_updates_per_epoch": updates_per_epoch,
        "max_seq_length": args.max_seq_length,
        "seed": args.seed,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": "all-linear",
        },
        "dtype": str(dtype),
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "optimizer_updates": total_updates,
        "warmup_steps": warmup_steps,
        "epoch_mean_weighted_sequence_nll": loss_history,
    }
    (output_dir / "gvsmc_training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "gvsmc_args.json").write_text(
        json.dumps(vars(args), indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"saved adapter: {output_dir}")


if __name__ == "__main__":
    main()
