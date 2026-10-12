#!/usr/bin/env python3
"""Check whether q_GV's factual-selection gain survives ESS-matched q_G weights.

ESS matching is a descriptive *selection* diagnostic, not probability
calibration: q_G ** gamma must NOT be used as an estimate of P(T=1).
The exponent is chosen using unlabeled validation predictions only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


def normalize_question(question):
    return " ".join(str(question).casefold().split())


def read_rows(path):
    with Path(path).open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    keys = [normalize_question(r["question"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate question identities")
    if not rows:
        raise ValueError("empty predictions")
    return rows


def ess_fraction(weights):
    w = np.asarray(weights, dtype=np.float64)
    if len(w) == 0 or np.any(w < 0) or not np.isfinite(w).all():
        raise ValueError("invalid weights")
    total = float(np.sum(w))
    if total <= 0:
        raise ValueError("zero total weight")
    return float((total * total) / (len(w) * np.dot(w, w)))


def precision(y, weights):
    w = np.asarray(weights, dtype=np.float64)
    return float(np.dot(y, w) / np.sum(w))


def sharpen_to_ess(probs, target_ess, tol=1e-8):
    """Find exponent >=0 with ESS(probs**exponent) near target ESS."""
    p = np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0)
    if not np.isfinite(p).all() or len(p) == 0:
        raise ValueError("invalid probabilities")
    if not (0 < target_ess <= 1):
        raise ValueError("target ESS fraction outside (0,1]")
    # Stable normalization avoids underflow at large exponents.
    logp = np.log(p)
    def weights(exp):
        logweights = exp * logp
        return np.exp(logweights - np.max(logweights))
    initial = ess_fraction(weights(1.0))
    lo, hi = 0.0, 1.0
    if target_ess < initial:
        lo, hi = 1.0, 2.0
        while ess_fraction(weights(hi)) > target_ess and hi < 2048:
            hi *= 2
        if ess_fraction(weights(hi)) > target_ess:
            raise ValueError("cannot achieve target ESS with powered probabilities")
    else:
        lo, hi = 0.0, 1.0
    for _ in range(90):
        mid = .5 * (lo + hi)
        ess = ess_fraction(weights(mid))
        if abs(ess - target_ess) < tol:
            lo = hi = mid
            break
        if ess > target_ess:
            lo = mid
        else:
            hi = mid
    gamma = .5 * (lo + hi)
    return gamma, weights(gamma)


def calculate(y, q_g, q_gv):
    target = ess_fraction(q_gv)
    gamma, matched = sharpen_to_ess(q_g, target)
    return {
        "q_g_precision": precision(y, q_g),
        "q_gv_precision": precision(y, q_gv),
        "q_g_ess_matched_precision": precision(y, matched),
        "q_g_ess": ess_fraction(q_g),
        "q_gv_ess": target,
        "q_g_matched_ess": ess_fraction(matched),
        "q_g_weight_power": float(gamma),
        "q_gv_minus_q_g_essmatched": precision(y, q_gv) - precision(y, matched),
        "q_gv_minus_q_g_original": precision(y, q_gv) - precision(y, q_g),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args()
    if a.bootstrap < 0:
        raise ValueError("bootstrap must be nonnegative")
    rows = read_rows(a.predictions)
    y = np.array([int(r["grounded_pair_em_proxy"]) for r in rows])
    qg = np.array([float(r["q_g"]) for r in rows])
    qgv = np.array([float(r["q_gv"]) for r in rows])
    if not np.isin(y, [0, 1]).all():
        raise ValueError("truth labels must be binary")
    if np.any(qg < 0) or np.any(qg > 1) or np.any(qgv < 0) or np.any(qgv > 1):
        raise ValueError("posterior predictions outside [0,1]")
    observed = calculate(y, qg, qgv)
    draws = {"q_gv_minus_q_g_essmatched": [], "q_gv_minus_q_g_original": []}
    rng = np.random.default_rng(a.seed)
    with tqdm(total=a.bootstrap, unit="draw", desc="ESS-matched posterior audit",
              dynamic_ncols=True, mininterval=1.0) as bar:
        for _ in range(a.bootstrap):
            ids = rng.integers(0, len(y), len(y))
            outcome = calculate(y[ids], qg[ids], qgv[ids])
            for metric in draws:
                draws[metric].append(outcome[metric])
            bar.update(1)
    result = {
        "source": a.predictions,
        "num_parsed": len(rows),
        "observed": observed,
        "paired_bootstrap_95_ci": {
            name: [float(z) for z in np.quantile(v, [.025, .975])]
            if len(v) >= 10 else None for name, v in draws.items()
        },
        "interpretation": (
            "ESS-matched power weights are diagnostic selection scores, NOT calibrated "
            "correctness probabilities. Observational within-development-set result."
        ),
    }
    path = Path(a.output)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
