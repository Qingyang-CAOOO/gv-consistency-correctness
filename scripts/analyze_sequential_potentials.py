#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Audit sequential GV potentials independently of accumulation.")
    p.add_argument("--data", required=True)
    return p.parse_args()


def safe_mean(xs):
    return float(np.mean(xs)) if xs else float("nan")


def safe_median(xs):
    return float(np.median(xs)) if xs else float("nan")


def main():
    args = parse_args()
    rows = [
        json.loads(line)
        for line in Path(args.data).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    usable = [r for r in rows if r.get("status") == "ok"]
    covered = [r for r in usable if r.get("gold_coverage")]
    if not usable:
        raise RuntimeError("no usable records")

    num_rounds = len(usable[0]["rounds"])
    print(f"usable={len(usable)} covered={len(covered)} rounds={num_rounds}")
    print()
    print("Each template below is applied ONCE to the original generator prior.")
    print("This removes contamination from earlier sequential updates.\n")

    for t in range(num_rounds):
        deltas = []
        log_evidence = []
        gold_g = []
        nongold_g = []
        by_group = {
            (True, 0): [], (True, 1): [],
            (False, 0): [], (False, 1): [],
        }

        for r in usable:
            gold = np.asarray(r["candidate_gold_mask"], dtype=bool)
            g = np.asarray(r["rounds"][t]["potentials"], dtype=float)
            targets_raw = r["rounds"][t].get("target_indices")
            if targets_raw is not None:
                targets = np.asarray(targets_raw, dtype=int)
                for yi, gi, ti in zip(gold, g, targets):
                    by_group[(bool(yi), int(ti))].append(float(gi))

            if not gold.any():
                continue

            w0 = np.asarray(r["initial_weights"], dtype=float)
            w0 = w0 / w0.sum()
            p0 = float(w0[gold].sum())
            if not (0.0 < p0 < 1.0):
                continue

            gg = float(np.sum(w0[gold] * g[gold]) / p0)
            gn = float(np.sum(w0[~gold] * g[~gold]) / (1.0 - p0))
            z = float(np.sum(w0 * g))
            p1 = float(np.sum(w0[gold] * g[gold]) / z)

            deltas.append(p1 - p0)
            gold_g.append(gg)
            nongold_g.append(gn)
            if gg > 0.0 and gn > 0.0:
                log_evidence.append(math.log(gg / gn))

        print(f"Template {t + 1}")
        print(f"  covered questions:             {len(deltas)}")
        print(f"  one-step mean Δ gold mass:     {safe_mean(deltas): .6f}")
        print(f"  one-step median Δ gold mass:   {safe_median(deltas): .6f}")
        print(f"  fraction Δ gold mass > 0:      {safe_mean([d > 0 for d in deltas]): .3f}")
        print(f"  weighted mean G(gold):         {safe_mean(gold_g): .6f}")
        print(f"  weighted mean G(non-gold):     {safe_mean(nongold_g): .6f}")
        print(f"  mean log evidence gold/non:    {safe_mean(log_evidence): .6f}")
        print(f"  median log evidence gold/non:  {safe_median(log_evidence): .6f}")
        print(f"  fraction log evidence > 0:     {safe_mean([x > 0 for x in log_evidence]): .3f}")
        if any(by_group.values()):
            print(f"  G(gold,target=A):              {safe_mean(by_group[(True,0)]): .6f}")
            print(f"  G(gold,target=B):              {safe_mean(by_group[(True,1)]): .6f}")
            print(f"  G(non-gold,target=A):          {safe_mean(by_group[(False,0)]): .6f}")
            print(f"  G(non-gold,target=B):          {safe_mean(by_group[(False,1)]): .6f}")
        print()


if __name__ == "__main__":
    main()
