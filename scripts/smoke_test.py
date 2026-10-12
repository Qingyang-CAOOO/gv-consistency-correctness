from __future__ import annotations

import torch

from gvsmc.metrics import effective_sample_size, normalized_entropy
from gvsmc.weights import (
    hard_randomized_gv_weight,
    normalize_weights,
    soft_randomized_gv_weight_from_label_logprobs,
)


def main() -> None:
    target = torch.tensor([0, 1, 0, 1])
    realized = torch.tensor([0, 0, 0, 1])
    hard = hard_randomized_gv_weight(target, realized)

    # Columns correspond to A/B sequence log-probabilities.
    label_logprobs = torch.tensor(
        [
            [-0.10, -2.30],
            [-0.20, -1.40],
            [-0.70, -0.80],
            [-1.50, -0.10],
        ]
    )
    soft = soft_randomized_gv_weight_from_label_logprobs(label_logprobs, target)
    normalized = normalize_weights(soft)

    print("hard weights:", hard.tolist())
    print("soft weights:", [round(x, 4) for x in soft.tolist()])
    print("normalized:", [round(x, 4) for x in normalized.tolist()])
    print("ESS:", round(effective_sample_size(soft).item(), 4))
    print("normalized entropy:", round(normalized_entropy(soft).item(), 4))


if __name__ == "__main__":
    main()
