from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class CFTExample:
    trajectory_index: int
    component: str
    prompt: str
    completion: str
    weight: float


def read_gv_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object on line {line_number}")
            rows.append(value)
    return rows


def build_cft_examples(
    records: Iterable[Mapping[str, Any]],
    *,
    mode: str,
    components: tuple[str, ...] = ("generator", "validator"),
) -> list[CFTExample]:
    """Build greedy-hard, sampled-hard, or soft CFT examples.

    hard uses the greedy A/B validator prediction. sampled-hard uses one
    sampled A/B validator prediction. soft keeps every parsed trajectory and
    weights it by G=P(Y_V=R | X,Y_G,R). sampled-hard is the stochastic hard
    comparator for the Rao--Blackwell relation E[C | X,Y_G,R] = G.
    soft-marginalized integrates over both randomized A/B positions in both
    the consistency potential and validator-training objective.
    posterior-g, posterior-gv and posterior-g-ess use frozen supervised
    predictions of grounded-pair correctness (never EM labels as weights).
    They train the *same* original sampled-order generator and validator
    completions as soft CFT, changing only the trajectory weight.
    """
    posterior_fields = {
        "posterior-g": "posterior_q_g",
        "posterior-gv": "posterior_q_gv",
        "posterior-g-ess": "posterior_q_g_essmatched",
    }
    allowed_modes = {"hard", "sampled-hard", "soft", "soft-marginalized"} | set(posterior_fields)
    if mode not in allowed_modes:
        raise ValueError(f"unsupported CFT mode {mode!r}; choose from {sorted(allowed_modes)}")

    allowed = {"generator", "validator"}
    if not components or any(component not in allowed for component in components):
        raise ValueError("components must be a non-empty subset of generator/validator")

    examples: list[CFTExample] = []
    for record in records:
        if not bool(record.get("parse_success", False)):
            continue

        if mode in {"hard", "sampled-hard"}:
            key = "hard_weight" if mode == "hard" else "sampled_hard_weight"
            hard_weight = float(record[key])
            if hard_weight < 0.5:
                continue
            weight = 1.0
        elif mode == "soft":
            weight = float(record["soft_weight"])
            if not 0.0 <= weight <= 1.0:
                raise ValueError(f"invalid soft weight {weight}")
        elif mode in posterior_fields:
            field = posterior_fields[mode]
            if field not in record:
                raise ValueError(f"{mode} requires {field} on all parsed trajectories")
            weight = float(record[field])
            if not math.isfinite(weight) or not 0.0 < weight <= 1.0:
                raise ValueError(f"invalid {field} value: {weight}")
        else:
            required = ("p_correct_when_a", "p_correct_when_b",
                        "validator_prompt_when_a", "validator_prompt_when_b")
            if not all(name in record for name in required):
                raise ValueError("soft-marginalized needs both-order augmented GV trajectories")
            p_a = float(record["p_correct_when_a"])
            p_b = float(record["p_correct_when_b"])
            if not (0.0 <= p_a <= 1.0 and 0.0 <= p_b <= 1.0):
                raise ValueError("both-order probabilities must lie in [0,1]")
            weight = .5 * (p_a + p_b)

        trajectory_index = int(record["index"])

        if "generator" in components:
            prompt = str(record["generator_prompt"])
            completion = str(record["raw_generator_response"]).strip()
            if not prompt or not completion:
                raise ValueError(
                    f"trajectory {trajectory_index} has empty generator text"
                )
            examples.append(
                CFTExample(
                    trajectory_index=trajectory_index,
                    component="generator",
                    prompt=prompt,
                    completion=completion,
                    weight=weight,
                )
            )

        if "validator" in components:
            if mode == "soft-marginalized":
                # Full Rao-Blackwellized weighted objective over randomized
                # label placement: generator weight (pA+pB)/2, validator
                # weights pA/2 and pB/2, not (pA+pB)/2 on one prompt.
                for target, prompt_key, prob in (
                    ("A", "validator_prompt_when_a", p_a),
                    ("B", "validator_prompt_when_b", p_b),
                ):
                    examples.append(
                        CFTExample(
                            trajectory_index=trajectory_index,
                            component="validator",
                            prompt=str(record[prompt_key]),
                            completion=target,
                            weight=0.5 * prob,
                        )
                    )
            else:
                prompt = str(record["validator_prompt"])
                target = str(record["target_label"]).strip()
                if target not in {"A", "B"}:
                    raise ValueError(
                        f"trajectory {trajectory_index} has invalid target {target!r}"
                    )
                examples.append(
                    CFTExample(
                        trajectory_index=trajectory_index,
                        component="validator",
                        prompt=prompt,
                        completion=target,
                        weight=weight,
                    )
                )

    if not examples:
        raise ValueError("no CFT examples were constructed")
    return examples
