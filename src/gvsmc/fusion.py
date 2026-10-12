from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np


def clipped_logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p) - math.log1p(-p)


@dataclass(frozen=True)
class PairwiseFusionModel:
    baseline_generator_coef: float
    full_generator_coef: float
    validator_coef: float
    baseline_c: float
    full_c: float
    eps: float = 1e-6

    def to_json(self, path: str | Path, *, metadata: dict | None = None) -> None:
        payload = {"model": asdict(self), "metadata": metadata or {}}
        Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> tuple["PairwiseFusionModel", dict]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload["model"]), dict(payload.get("metadata", {}))

    def pair_features(
        self, prior_i: float, prior_j: float, validator_p_i: float
    ) -> tuple[float, float]:
        if prior_i <= 0.0 or prior_j <= 0.0:
            raise ValueError("generator prior weights must be positive")
        s_g = math.log(float(prior_i)) - math.log(float(prior_j))
        s_v = clipped_logit(float(validator_p_i), self.eps)
        return s_g, s_v

    def baseline_logit(self, s_g: float) -> float:
        return self.baseline_generator_coef * float(s_g)

    def full_logit(self, s_g: float, s_v: float) -> float:
        return self.full_generator_coef * float(s_g) + self.validator_coef * float(s_v)

    def incremental_logit(self, s_g: float, s_v: float) -> float:
        return self.full_logit(s_g, s_v) - self.baseline_logit(s_g)

    def fused_weights(
        self,
        prior: Sequence[float],
        pairwise_win_probabilities: Sequence[Sequence[float | None]],
    ) -> np.ndarray:
        prior_arr = np.asarray(prior, dtype=float)
        if prior_arr.ndim != 1 or len(prior_arr) < 2:
            raise ValueError("prior must be a one-dimensional vector with at least two entries")
        if np.any(prior_arr <= 0.0):
            raise ValueError("all prior weights must be positive")
        prior_arr = prior_arr / prior_arr.sum()
        k = len(prior_arr)
        increments = np.zeros((k, k), dtype=float)
        for i in range(k):
            for j in range(i + 1, k):
                p_i = pairwise_win_probabilities[i][j]
                if p_i is None:
                    raise ValueError(f"missing pairwise probability for ({i}, {j})")
                s_g, s_v = self.pair_features(prior_arr[i], prior_arr[j], float(p_i))
                d = self.incremental_logit(s_g, s_v)
                increments[i, j] = d
                increments[j, i] = -d

        # Complete-graph least-squares projection of pairwise incremental
        # log-odds d_ij onto candidate potentials r_i-r_j.
        candidate_increment = increments.sum(axis=1) / k
        logw = np.log(prior_arr) + candidate_increment
        logw -= np.max(logw)
        fused = np.exp(logw)
        return fused / fused.sum()
