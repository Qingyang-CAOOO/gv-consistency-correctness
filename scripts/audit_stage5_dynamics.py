#!/usr/bin/env python3
"""CPU-only post-hoc audit of protocol-v2 iterative CFT (5 arms x 3 rounds).

Uses saved data ONLY; never fits a model or reads the official final holdout.
EM truth is a conservative exact-match proxy. Transferred, fixed posterior
calibration on the repeatedly seen 320 CFT training questions is descriptive,
not independent validation of posterior transport to updated checkpoints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ARMS = ("hard", "soft", "posterior-g", "posterior-g-ess", "posterior-gv")
R1_NAMES = {
    "hard": "hard-matched-n320-seed2026_dev_n512.jsonl",
    "soft": "soft-matched-n320-seed2026_dev_n512.jsonl",
    "posterior-g": "posterior-g_dev_n512.jsonl",
    "posterior-g-ess": "posterior-g-ess_dev_n512.jsonl",
    "posterior-gv": "posterior-gv_dev_n512.jsonl",
}
WEIGHTS = {
    "q_g": "posterior_q_g",
    "q_gv": "posterior_q_gv",
    "q_g_ess": "posterior_q_g_essmatched",
    "hard": "hard_weight",
    "soft": "soft_weight",
}
POSTERIORS = ("q_g", "q_gv")


def question_key(record):
    return " ".join(str(record["question"]).casefold().split())


def load_jsonl(path: Path, *, n: int):
    if "official_final_holdout" in str(path):
        raise ValueError("final holdout may not be audited")
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if len(records) != n:
        raise ValueError(f"{path}: expected exactly {n} records, got {len(records)}")
    records.sort(key=question_key)
    keys = [question_key(r) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path}: duplicate questions")
    return records


def ensure_question_alignment(left, right, context):
    if [question_key(r) for r in left] != [question_key(r) for r in right]:
        raise ValueError(f"{context}: question identities mismatch")


def parsed(record):
    return bool(record.get("parse_success", False))


def grounded(record):
    return parsed(record) and bool(record.get("intended_correct_em", False)) and not bool(record.get("intended_misleading_em", False))


def binary_eval(rows):
    return {
        "parsed": np.asarray([parsed(r) for r in rows], dtype=bool),
        "grounded": np.asarray([grounded(r) for r in rows], dtype=bool),
        "correct": np.asarray([parsed(r) and bool(r.get("intended_correct_em", False)) for r in rows], dtype=bool),
        "hard_consistent": np.asarray([parsed(r) and float(r.get("hard_weight", 0)) >= 0.5 for r in rows], dtype=bool),
    }


def summarize_eval(rows):
    d = binary_eval(rows)
    n = len(rows)
    p = int(d["parsed"].sum())
    g = int(d["grounded"].sum())
    return {
        "n_questions": n,
        "parsed": p,
        "grounded": g,
        "parse_rate": p / n,
        "grounded_yield_all_questions": g / n,
        "grounded_rate_given_parsed": g / p if p else None,
        "hard_gv_consistency_given_parsed": float(d["hard_consistent"].sum() / p) if p else None,
        "intended_correct_em_all_questions": float(d["correct"].mean()),
    }


def paired_round_transition(old, new, *, bootstrap=5000, seed=2026):
    ensure_question_alignment(old, new, "paired round transition")
    a, b = binary_eval(old), binary_eval(new)
    op, np_ = a["parsed"], b["parsed"]
    og, ng = a["grounded"], b["grounded"]
    gained = int((~og & ng).sum())
    lost = int((og & ~ng).sum())
    both_parsed = op & np_
    r = {
        "n_questions": len(old),
        "gained_grounded": gained,
        "lost_grounded": lost,
        "net_grounded": gained - lost,
        "net_grounding_pp": 100 * (gained - lost) / len(old),
        "parse_gained": int((~op & np_).sum()),
        "parse_lost": int((op & ~np_).sum()),
        "grounding_gained_on_newly_parsed": int((~op & ng).sum()),
        "grounding_lost_to_parse_failure": int((og & ~np_).sum()),
        "grounding_gained_among_parsed_both_rounds": int((both_parsed & ~og & ng).sum()),
        "grounding_lost_among_parsed_both_rounds": int((both_parsed & og & ~ng).sum()),
        "n_parsed_in_both_rounds": int(both_parsed.sum()),
    }
    if gained != r["grounding_gained_on_newly_parsed"] + r["grounding_gained_among_parsed_both_rounds"]:
        raise AssertionError("grounding-gain partition mismatch")
    if lost != r["grounding_lost_to_parse_failure"] + r["grounding_lost_among_parsed_both_rounds"]:
        raise AssertionError("grounding-loss partition mismatch")
    if bootstrap:
        diff = ng.astype(float) - og.astype(float)
        rng = np.random.default_rng(seed)
        draws = np.empty(bootstrap, dtype=float)
        for start in range(0, bootstrap, 256):
            end = min(start + 256, bootstrap)
            ids = rng.integers(0, len(old), (end - start, len(old)))
            draws[start:end] = 100 * diff[ids].mean(axis=1)
        r["paired_question_bootstrap_95_ci_pp"] = [
            float(x) for x in np.quantile(draws, [0.025, 0.975])
        ]
    else:
        r["paired_question_bootstrap_95_ci_pp"] = None
    return r


def auc(labels, probs):
    yes = probs[labels == 1]
    no = probs[labels == 0]
    if not len(yes) or not len(no):
        return None
    comp = yes[:, None] - no[None, :]
    return float((np.count_nonzero(comp > 0) + 0.5 * np.count_nonzero(comp == 0)) / comp.size)


def summarize_weight(label, weights, *, posterior=False):
    w = np.asarray(weights, dtype=float)
    if not np.all(np.isfinite(w)) or np.any(w < 0) or not np.any(w > 0):
        raise ValueError("selection weights must be finite nonnegative with positive mass")
    if posterior and (np.any(w <= 0) or np.any(w >= 1)):
        raise ValueError("predicted qG/qGV probabilities must lie strictly inside (0,1)")
    t = np.asarray(label, dtype=float)
    precision = float(np.dot(w, t) / w.sum())
    prevalence = float(t.mean())
    good, bad = w[t == 1], w[t == 0]
    mean_good = float(good.mean()) if len(good) else None
    mean_bad = float(bad.mean()) if len(bad) else None
    details = {
        "n_parsed": len(w),
        "mean_weight": float(w.mean()),
        "weight_ess_fraction": float(w.sum() ** 2 / (len(w) * np.dot(w, w))),
        "grounded_pair_base_rate": prevalence,
        "weighted_grounded_precision": precision,
        "precision_lift": precision - prevalence,
        "mean_weight_grounded": mean_good,
        "mean_weight_ungrounded": mean_bad,
        "truth_enrichment_ratio": mean_good / mean_bad if mean_good is not None and mean_bad not in (None, 0) else None,
    }
    if posterior:
        details.update({
            "nll_em_proxy": float(-(t * np.log(w) + (1 - t) * np.log1p(-w)).mean()),
            "brier_em_proxy": float(np.square(w - t).mean()),
            "auroc_em_proxy": auc(t, w),
            "mean_predicted_validity_minus_base_rate": float(w.mean()) - prevalence,
        })
    return details


def summarize_train(raw, weighted):
    ensure_question_alignment(raw, weighted, "raw-vs-gold-blind training question population")
    labels, values = [], {k: [] for k in WEIGHTS}
    for x, y in zip(raw, weighted):
        if parsed(x) != parsed(y):
            raise ValueError("raw-vs-weighted parse-success mismatch")
        for fld in ("intended_correct", "intended_misleading"):
            if parsed(x) and x.get(fld) != y.get(fld):
                raise ValueError(f"raw-vs-weighted generated candidate mismatch: {fld}")
        if any(k in y for k in ("gold_answers", "intended_correct_em", "intended_misleading_em", "grounded_pair", "candidate_gold_mask", "is_grounded")):
            raise ValueError("gold/EM label leaked into supposedly gold-blind training file")
        if parsed(x):
            labels.append(float(grounded(x)))
            for name, field in WEIGHTS.items():
                z = y.get(field, x.get(field))
                if z is None:
                    raise ValueError(f"parsed row lacks {field}")
                values[name].append(z)
    if not labels:
        raise ValueError("no parsed training trajectories")
    truth = np.asarray(labels, dtype=float)
    methods = {
        name: summarize_weight(truth, value, posterior=name in POSTERIORS)
        for name, value in values.items()
    }
    q_g, q_gv = methods["q_g"], methods["q_gv"]
    return {
        "n_questions": len(raw),
        "n_parsed": len(labels),
        "grounded_pair_count": int(truth.sum()),
        "grounded_rate_given_parsed": float(truth.mean()),
        "selection_methods": methods,
        "frozen_posterior_gv_minus_g": {
            "nll": q_gv["nll_em_proxy"] - q_g["nll_em_proxy"],
            "brier": q_gv["brier_em_proxy"] - q_g["brier_em_proxy"],
            "auroc": (
                q_gv["auroc_em_proxy"] - q_g["auroc_em_proxy"]
                if q_gv["auroc_em_proxy"] is not None and q_g["auroc_em_proxy"] is not None else None
            ),
        },
    }


def source_paths(root, arm, round_idx, seed):
    it = root / "iterative-posterior-v1" / f"seed{seed}"
    if round_idx == 1:
        return {
            "eval": root / "single-round" / "eval" / R1_NAMES[arm],
            "train_raw": root / "single-round" / "cft_train_gv_n320.jsonl",
            "train_weights": root / "single-round" / "cft_train_posterior_weights_n320.jsonl",
        }
    prefix = it / arm / f"round{round_idx}"
    return {
        "eval": prefix / "development_gv_n512.jsonl",
        "train_raw": prefix / "train_gv_n320.jsonl",
        "train_weights": prefix / "train_frozen_weights_n320.jsonl",
    }


def analyze(root, *, seed=2026, bootstrap=5000):
    if seed != 2026:
        raise ValueError("Stage-5 pilot paths and initial checkpoints are fixed to seed2026")
    result = {
        "policy": {
            "study": "retrospective development-only iterative-CFT dynamics; 5 arms, 3 rounds, training seed2026",
            "endpoint": "all-question parse-and-grounded exact-match validity proxy",
            "prediction_drift_scope": "CFT TRAIN question cohort (in-sample w.r.t adapters); posterior base-trained and frozen; not independent validation/calibration",
            "uncertainty_scope": "question-paired bootstrap conditional on one training seed; no multiple-comparison correction",
            "prohibited_data": "official_final_holdout.jsonl; never accessed",
            "no_model_inference_or_training": True,
        },
        "paths": {}, "eval_metrics": {}, "paired_round_transitions": {},
        "train_selection_diagnostics": {}, "posterior_hashes": None,
    }
    basekeys = None
    training_keys = None
    for arm in ARMS:
        result["eval_metrics"][arm] = {}
        result["paired_round_transitions"][arm] = {}
        result["train_selection_diagnostics"][arm] = {}
        data = {}
        for t in (1, 2, 3):
            ps = source_paths(root, arm, t, seed)
            ev, raw, weighted = (
                load_jsonl(ps[key], n=size) for key, size in
                (("eval", 512), ("train_raw", 320), ("train_weights", 320))
            )
            eval_keys = [question_key(r) for r in ev]
            train_keys = [question_key(r) for r in raw]
            if basekeys is None:
                basekeys = eval_keys
            elif eval_keys != basekeys:
                raise ValueError(f"development question alignment failed: {arm}, round {t}")
            if training_keys is None:
                training_keys = train_keys
            elif train_keys != training_keys:
                raise ValueError(f"training question alignment failed: {arm}, round {t}")
            meta_path = ps["train_weights"].with_suffix(".summary.json")
            if not meta_path.exists():
                raise FileNotFoundError(meta_path)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("no_refit") is not True or meta.get("gold_blind") is not True:
                raise ValueError(f"frozen posterior provenance missing: {arm}, round {t}")
            hashes = {k: meta.get(k) for k in ("calibrator_q_g_sha256", "calibrator_q_gv_sha256")}
            if any(not v for v in hashes.values()):
                raise ValueError("frozen posterior calibrator hashes missing")
            if result["posterior_hashes"] is None:
                result["posterior_hashes"] = hashes
            elif hashes != result["posterior_hashes"]:
                raise ValueError(f"posterior calibrator SHA256 drift detected: {arm}, round {t}")
            data[t] = ev
            result["paths"][f"{arm}:round{t}"] = {k: str(v) for k, v in ps.items()}
            result["eval_metrics"][arm][str(t)] = summarize_eval(ev)
            result["train_selection_diagnostics"][arm][str(t)] = summarize_train(raw, weighted)
        for first, second in ((1, 2), (2, 3), (1, 3)):
            result["paired_round_transitions"][arm][f"{first}->{second}"] = paired_round_transition(
                data[first], data[second], bootstrap=bootstrap, seed=seed + first * 10 + second
            )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, help="Drive protocol-v2 artifact root")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if args.bootstrap < 0:
        raise ValueError("--bootstrap must be nonnegative")
    root, dest = Path(args.root), Path(args.output)
    if dest.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {dest}")
    report = analyze(root, seed=args.seed, bootstrap=args.bootstrap)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("DEVELOPMENT trajectory: arm/round parsed, grounded, hard-GV%")
    for arm in ARMS:
        for t in (1, 2, 3):
            d = report["eval_metrics"][arm][str(t)]
            print(f"  {arm:16} r{t}: {d['parsed']:3}/512 parsed, {d['grounded']:3}/512 grounded, "
                  f"hard GV {100*d['hard_gv_consistency_given_parsed']:.2f}%")
    print("\nPAIRED grounding transitions (question-bootstrap CIs in pp, descriptive only)")
    for arm in ARMS:
        for name, d in report["paired_round_transitions"][arm].items():
            print(f"  {arm:16} {name}: gained {d['gained_grounded']}, lost {d['lost_grounded']}; "
                  f"delta {d['net_grounding_pp']:+.2f}pp, CI={d['paired_question_bootstrap_95_ci_pp']}; "
                  f"new parse contribution +{d['grounding_gained_on_newly_parsed']}/-{d['grounding_lost_to_parse_failure']}")
    print("\nTRAIN cohort post-hoc frozen-posterior diagnostics (in-sample adapters; NOT validation)")
    for arm in ARMS:
        for t in (1, 2, 3):
            s = report["train_selection_diagnostics"][arm][str(t)]
            c = s["frozen_posterior_gv_minus_g"]
            gv = s["selection_methods"]["q_gv"]
            print(f"  {arm:16} r{t}: {s['n_parsed']}/320 parsed; "
                  f"qGV-G NLL {c['nll']:+.4f}, Brier {c['brier']:+.4f}; "
                  f"qGV precision {gv['weighted_grounded_precision']:.3f}, ESS {gv['weight_ess_fraction']:.3f}")
    print("Verified same frozen posterior SHA256 across all rounds; no official holdout accessed.")
    print("Wrote:", dest)


if __name__ == "__main__":
    main()
