#!/usr/bin/env python3
"""Post-hoc, CPU-only false-retention audit for GV consistency vs correctness.

PRIMARY analysis cohort: the ALREADY INSPECTED official 512-question
confirmation split (394 parsed). Frozen predictions are read from disk;
this is NOT a new prospective confirmation of the post-hoc subgroup tests.

SECONDARY: 320-question CFT TRAIN cohorts, five arms x three rounds.
No model training, fitting, generation, or final-holdout access.
Labels are conservative exact-match grounded-pair proxies, not semantic truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

if __package__:
    from .audit_stage5_dynamics import (
        ARMS, ensure_question_alignment, load_jsonl, parsed, grounded,
        question_key, source_paths,
    )
    from .audit_posterior_ess_matched import sharpen_to_ess, ess_fraction
else:
    from audit_stage5_dynamics import (
        ARMS, ensure_question_alignment, load_jsonl, parsed, grounded,
        question_key, source_paths,
    )
    from audit_posterior_ess_matched import sharpen_to_ess, ess_fraction


SENSITIVE_LABEL_FIELDS = {
    "gold_answers", "intended_correct_em", "intended_misleading_em",
    "grounded_pair", "candidate_gold_mask", "is_grounded",
}
METHODS = (
    "hard_gv", "soft_gv_sampled", "soft_gv_marginalized",
    "q_g", "q_g_essmatched", "q_gv",
)
CONTRASTS = ("hard_gv", "soft_gv_sampled", "soft_gv_marginalized",
             "q_g", "q_g_essmatched")


def sha256_file(path):
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1 << 20), b""):
            hasher.update(data)
    return hasher.hexdigest()


def check_weight(array, name, *, binary=False, probability=False):
    w = np.asarray(array, dtype=float)
    if not w.size or not np.isfinite(w).all() or (w < 0).any() or (w > 1 + 1e-10).any():
        raise ValueError(f"{name}: nonfinite or invalid weight")
    if binary and not np.isin(w, (0., 1.)).all():
        raise ValueError(f"{name}: hard consistency must be binary")
    if probability and (w <= 0).any():
        raise ValueError(f"{name}: frozen posterior must be strictly positive")
    if float(w.sum()) <= 0:
        raise ValueError(f"{name}: zero total selection weight")
    return w


def build_population(raw, predictions=None, *, gold_blind=False):
    """Match parsed raw records to immutable predicted scores by question ID."""
    if predictions is None:
        if not gold_blind:
            raise ValueError("train populations must supply gold-blind weighted records")
        raise ValueError("missing matched weighted train records")
    if gold_blind:
        ensure_question_alignment(raw, predictions, "train weighted-vs-raw")
        for a, b in zip(raw, predictions):
            if parsed(a) != parsed(b):
                raise ValueError("train parse-success mismatch")
            if any(x in b for x in SENSITIVE_LABEL_FIELDS):
                raise ValueError("gold label leaked into training weight file")
            if parsed(a) and any(a.get(k) != b.get(k) for k in
                                 ("intended_correct", "intended_misleading")):
                raise ValueError("train candidate mismatch")
        pairs = [(a, b) for a, b in zip(raw, predictions) if parsed(a)]
    else:
        # Official predictions contain only PARSED questions and omit raw scores.
        raw_parsed = [a for a in raw if parsed(a)]
        if len(raw_parsed) != len(predictions):
            raise ValueError("official parsed vs predictions count mismatch")
        ensure_question_alignment(raw_parsed, predictions, "official confirmation")
        for a, b in zip(raw_parsed, predictions):
            if int(grounded(a)) != int(b["grounded_pair_em_proxy"]):
                raise ValueError("official predictions truth-label mismatch")
        pairs = list(zip(raw_parsed, predictions))

    if not pairs:
        raise ValueError("no parsed trajectories")
    y = np.asarray([int(grounded(a)) for a, _ in pairs], dtype=np.int8)
    hard = check_weight([a["hard_weight"] for a, _ in pairs], "hard_gv", binary=True)
    soft = check_weight([a["soft_weight"] for a, _ in pairs], "soft_gv_sampled")
    if gold_blind:
        soft_marg = check_weight([
            0.5 * (float(a["p_correct_when_a"]) + float(a["p_correct_when_b"]))
            for a, _ in pairs
        ], "soft_gv_marginalized")
        qg = check_weight([b["posterior_q_g"] for _, b in pairs],
                          "q_g", probability=True)
        qgv = check_weight([b["posterior_q_gv"] for _, b in pairs],
                           "q_gv", probability=True)
        qgess = check_weight([b["posterior_q_g_essmatched"] for _, b in pairs],
                             "q_g_essmatched")
        power = None  # Matched ESS already determined per train population.
    else:
        soft_marg = check_weight(
            [b["order_marginalized_consistency"] for _, b in pairs],
            "soft_gv_marginalized")
        qg = check_weight([b["q_g"] for _, b in pairs],
                          "q_g", probability=True)
        qgv = check_weight([b["q_gv"] for _, b in pairs],
                           "q_gv", probability=True)
        power, qgess = sharpen_to_ess(qg, ess_fraction(qgv))
        qgess = check_weight(qgess, "q_g_essmatched")
    weights = {
        "hard_gv": hard, "soft_gv_sampled": soft,
        "soft_gv_marginalized": soft_marg,
        "q_g": qg, "q_g_essmatched": qgess, "q_gv": qgv,
    }
    return y, weights, power


def selection_precision(y, w):
    return float(np.dot(y, w) / w.sum())


def summary(y, weights):
    """Within-population false-retention and wrong-example weighting."""
    if len(y) < 2 or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("grounding labels must be binary")
    h = weights["hard_gv"] == 1
    true_h, false_h = h & (y == 1), h & (y == 0)
    good, bad = y == 1, y == 0
    hard_true = int(true_h.sum())
    hard_false = int(false_h.sum())
    if not hard_true or not hard_false:
        raise ValueError("hard-retained group needs both true and false cases")
    # Top quartile of actual randomized-order soft weight is a diagnostic
    # subgroup, not a purported soft-CFT retention rule.
    threshold = float(np.quantile(weights["soft_gv_sampled"], 0.75))
    high_soft = weights["soft_gv_sampled"] >= threshold
    out = {
        "n_parsed": int(len(y)),
        "grounded_count": int(good.sum()),
        "grounded_prevalence": float(good.mean()),
        "hard_retained_count": int(h.sum()),
        "hard_retained_grounded": hard_true,
        "hard_false_retained_count": hard_false,
        "hard_false_retained_fraction": float(hard_false / h.sum()),
        "hard_false_retained_fraction_among_all_ungrounded": float(hard_false / bad.sum()) if bad.sum() else None,
        "high_soft_quartile_cutoff": threshold,
        "high_soft_quartile_n_including_ties": int(high_soft.sum()),
        "high_soft_quartile_false_count": int((high_soft & bad).sum()),
        "high_soft_quartile_false_fraction": float((high_soft & bad).sum() / high_soft.sum()),
        "methods": {},
    }
    for name, w in weights.items():
        out["methods"][name] = {
            "weighted_grounded_precision": selection_precision(y, w),
            "false_selection_mass_fraction": float(np.dot(bad, w) / w.sum()),
            "false_hard_retained_mass_fraction": float(w[false_h].sum() / w.sum()),
            "ess_fraction_parsed": ess_fraction(w),
            "mean_weight_hard_retained_grounded": float(w[true_h].mean()),
            "mean_weight_hard_retained_ungrounded": float(w[false_h].mean()),
            "within_hard_retained_truth_gap": float(w[true_h].mean() - w[false_h].mean()),
            "hard_retained_reweighted_grounded_precision": selection_precision(y[h], w[h]),
            "mean_weight_high_soft_grounded": float(w[high_soft & good].mean()) if (high_soft & good).any() else None,
            "mean_weight_high_soft_ungrounded": float(w[high_soft & bad].mean()) if (high_soft & bad).any() else None,
        }
    return out


def contrasts(y, weights):
    p_gv = selection_precision(y, weights["q_gv"])
    groups = summary(y, weights)["methods"]
    return {
        **{
            "q_gv_minus_" + base: p_gv - selection_precision(y, weights[base])
            for base in CONTRASTS
        },
        "q_gv_minus_q_g_within_hard_truth_gap":
            groups["q_gv"]["within_hard_retained_truth_gap"]
            - groups["q_g"]["within_hard_retained_truth_gap"],
        "q_gv_minus_soft_within_hard_truth_gap":
            groups["q_gv"]["within_hard_retained_truth_gap"]
            - groups["soft_gv_sampled"]["within_hard_retained_truth_gap"],
        "q_gv_reweight_minus_hard_retained_precision":
            groups["q_gv"]["hard_retained_reweighted_grounded_precision"]
            - selection_precision(y, weights["hard_gv"]),
    }


def bootstrap_contrasts(y, weights, n_draws, seed):
    if n_draws == 0:
        return None
    observed = contrasts(y, weights)
    sampled = {k: [] for k in observed}
    rng = np.random.default_rng(seed)
    for _ in range(n_draws):
        ix = rng.integers(0, len(y), len(y))
        yy = y[ix]
        ww = {k: v[ix] for k, v in weights.items()}
        # A bootstrap draw can omit the already minority hard-retained
        # positive/negative group on a small synthetic sample. Do not
        # manufacture a gap or coerce missing groups to zero.
        h = ww["hard_gv"] == 1
        if not h.any() or not (yy[h] == 0).any() or not (yy[h] == 1).any():
            continue
        delta = contrasts(yy, ww)
        for k, value in delta.items():
            sampled[k].append(value)
    return {
        k: {
            "estimate": value,
            "paired_question_bootstrap_95_ci": (
                [float(x) for x in np.quantile(sampled[k], (0.025, 0.975))]
                if len(sampled[k]) >= 10 else None
            ),
            "valid_draws": len(sampled[k]),
        } for k, value in observed.items()
    }


def run(root, *, bootstrap=3000, seed=2026):
    if bootstrap < 0:
        raise ValueError("bootstrap must be nonnegative")
    if seed != 2026:
        raise ValueError("this Stage-5 feedback pilot only has training seed 2026")
    feature_path = root / "official_confirmation_features_n512.jsonl"
    preds_path = root / "posterior_base_v2_confirmation" / "frozen_confirmation_predictions.jsonl"
    info_path = root / "posterior_base_v2_confirmation" / "summary.json"
    raw = load_jsonl(feature_path, n=512)
    # Confirmation predictions only include parsed questions (394 observed).
    if not preds_path.is_file():
        raise FileNotFoundError(preds_path)
    with preds_path.open(encoding="utf-8") as f:
        pred = [json.loads(line) for line in f if line.strip()]
    pred.sort(key=question_key)
    if len(set(map(question_key, pred))) != len(pred):
        raise ValueError("duplicate official confirmation predicted question IDs")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("no_refitting") is not True:
        raise ValueError("confirmation evaluator did not record frozen no-refit provenance")
    if info.get("num_questions") != 512 or info.get("num_parsed") != len(pred):
        raise ValueError("official confirmation source count/provenance mismatch")
    if Path(info["data"]).name != feature_path.name:
        raise ValueError("official confirmation summary input does not match saved features")
    y, w, gamma = build_population(raw, pred)
    if len(y) != 394 or int(y.sum()) != 177:
        raise ValueError("official frozen confirmation count/grounding differs from the recorded 394 parsed, 177 grounded")
    official = summary(y, w)
    official["q_g_essmatching_power_for_confirmation_population"] = float(gamma)
    official["paired_question_bootstrap"] = bootstrap_contrasts(y, w, bootstrap, seed)

    train = {}
    frozen_hashes = None
    for arm in ARMS:
        train[arm] = {}
        for t in (1, 2, 3):
            src = source_paths(root, arm, t, seed)
            rr = load_jsonl(src["train_raw"], n=320)
            ww = load_jsonl(src["train_weights"], n=320)
            meta = json.loads(src["train_weights"].with_suffix(".summary.json").read_text(encoding="utf-8"))
            if meta.get("gold_blind") is not True or meta.get("no_refit") is not True:
                raise ValueError(f"{arm} round {t}: missing gold-blind/no-refit provenance")
            hashes = tuple(meta.get(k) for k in ("calibrator_q_g_sha256", "calibrator_q_gv_sha256"))
            if not all(hashes):
                raise ValueError(f"{arm} round {t}: missing frozen posterior hashes")
            if frozen_hashes is None:
                frozen_hashes = hashes
            if hashes != frozen_hashes:
                raise ValueError("different posterior files used across CFT rounds")
            yy, weights, _ = build_population(rr, ww, gold_blind=True)
            train[arm][str(t)] = summary(yy, weights)
    return {
        "scope": {
            "question": "Do correctness weights distinguish hard/soft GV false retention?",
            "official": "Post-hoc subgroup analysis on previously inspected prospective confirmation predictions; NOT new confirmatory testing",
            "train": "CFT training questions in-sample relative to adapted LM; exploratory feedback dynamics only",
            "truth": "T=1 iff parsed, intended-correct exact-match=1 and intended-misleading exact-match=0",
            "hard": "single randomized-order greedy A/B consistency C=1; not truth",
            "soft": "continuous original sampled-order validator probability; top-quartile subgroup only diagnostic, NOT binary CFT retention",
            "q_g_ess": "qG power-changed to match qGV ESS, not a calibrated correctness probability; confirmation exponent fixed for bootstrap",
            "bootstrap": "paired resampling by question, conditional on single evaluation and fixed model, retrospective exploratory intervals",
            "do_not_read": "official_final_holdout",
            "no_new_gpu_or_calibration": True,
        },
        "sources": {
            "confirmation_features": str(feature_path),
            "confirmation_predictions": str(preds_path),
            "confirmation_summary": str(info_path),
            "confirmation_features_sha256": sha256_file(feature_path),
            "confirmation_predictions_sha256": sha256_file(preds_path),
            "same_frozen_posterior_sha256_train_rounds": list(frozen_hashes),
        },
        "official_confirmation": official,
        "cft_train_by_arm_round": train,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--bootstrap", type=int, default=3000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    dest = Path(args.output)
    if dest.exists():
        raise FileExistsError(f"refusing to overwrite {dest}")
    report = run(Path(args.root), bootstrap=args.bootstrap, seed=args.seed)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    o = report["official_confirmation"]
    print("OFFICIAL CONFIRMATION: post-hoc false-retention subgroup (not new holdout)")
    print(f"  parsed={o['n_parsed']} grounded={o['grounded_count']}; "
          f"hard retained={o['hard_retained_count']}, falsely retained={o['hard_false_retained_count']} "
          f"({100*o['hard_false_retained_fraction']:.2f}% of hard retained)")
    print(f"  high-soft subset (top quartile incl ties): {o['high_soft_quartile_false_count']}/{o['high_soft_quartile_n_including_ties']} EM-ungrounded")
    print("  Weight method                  precision    false mass     ESS   hard-retained weight gap")
    for key in METHODS:
        x = o["methods"][key]
        print(f"  {key:25} {x['weighted_grounded_precision']:.4f} "
              f"{x['false_selection_mass_fraction']:.4f} "
              f"{x['ess_fraction_parsed']:.4f} "
              f"{x['within_hard_retained_truth_gap']:+.4f}")
    print("  Paired conditional-on-confirmation contrasts (qGV minus baseline):")
    for k, v in o["paired_question_bootstrap"].items():
        print(f"  {k}: {v['estimate']:+.4f}, CI={v['paired_question_bootstrap_95_ci']}")
    print("TRAIN trajectory false-retention dynamics (in-sample; descriptive):")
    for arm in ARMS:
        for t in (1, 2, 3):
            s = report["cft_train_by_arm_round"][arm][str(t)]
            gv = s["methods"]["q_gv"]
            print(f"  {arm:16} r{t}: hard falsely retained {s['hard_false_retained_count']}/{s['hard_retained_count']}; "
                  f"qGV gap {gv['within_hard_retained_truth_gap']:+.3f}; "
                  f"qGV precision {gv['weighted_grounded_precision']:.3f}")
    print("No official final holdout accessed. No GPU inference or training.")
    print("Wrote:", dest)


if __name__ == "__main__":
    main()
