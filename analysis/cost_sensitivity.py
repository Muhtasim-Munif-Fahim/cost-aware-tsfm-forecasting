#!/usr/bin/env python3
"""S12 -- sensitivity of the cost-adjusted decision rule to electricity price and PUE.

The decision objective is MASE + wtp * usd_per_1k, where usd_per_1k is the TDP-proxy
cost computed at harness defaults (price=$0.15/kWh, PUE=1.4). Both price and PUE scale
usd_per_1k LINEARLY, so varying them is equivalent to rescaling wtp:
    winner(wtp, price, PUE) == winner(wtp * (price/0.15) * (PUE/1.4), 0.15, 1.4).
This script proves that mechanically for referees: it recomputes every winner map over
a price x PUE x wtp grid from the saved *_regime.csv files and reports how many cells
flip relative to the baseline map (flips occur only via the effective-wtp rescale).

Usage:
  python analysis/cost_sensitivity.py --regime-glob "results/v1/regime/canonical_*_regime.csv" \
      --out-csv results/v1/regime/s12_cost_sensitivity.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import pandas as pd

LEDGER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "paper", "RESULTS_LEDGER.md")
BASE_PRICE, BASE_PUE = 0.15, 1.4
WTPS = [0, 500, 1500, 5000, 20000]
PRICES = [0.05, 0.15, 0.30]
PUES = [1.0, 1.4, 2.0]


def next_ledger_id() -> int:
    if not os.path.exists(LEDGER_PATH):
        return 1
    text = open(LEDGER_PATH, encoding="utf-8").read()
    ids = [int(m) for m in re.findall(r"L-(\d+)", text)]
    return max(ids, default=0) + 1


def append_ledger_stub(claim: str, value: str, artifact: str, command: str, code_tag: str):
    n = next_ledger_id()
    row = (f"| L-{n:03d} | {claim} | {value} | | {artifact} | `{command}` | | {code_tag} | "
           f"n/a | | Phase 3 script |\n")
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(row)


def winner(rows: pd.DataFrame, wtp: float, scale: float) -> str:
    agg = rows.groupby("model", as_index=False)[["MASE", "usd_per_1k"]].mean()
    obj = agg["MASE"] + wtp * agg["usd_per_1k"] * scale
    return agg.loc[obj.idxmin(), "model"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime-glob", default="results/v1/regime/canonical_*_regime.csv")
    ap.add_argument("--out-csv", default="results/v1/regime/s12_cost_sensitivity.csv")
    ap.add_argument("--code-tag", default="")
    args = ap.parse_args()

    rows = []
    for fp in sorted(glob.glob(args.regime_glob)):
        run = os.path.basename(fp).replace("_regime.csv", "")
        df = pd.read_csv(fp)
        for price in PRICES:
            for pue in PUES:
                scale = (price / BASE_PRICE) * (pue / BASE_PUE)
                flips = 0
                total = 0
                for W in sorted(df.train_weeks.unique()):
                    sub = df[df.train_weeks == W]
                    for wtp in WTPS:
                        base = winner(sub, wtp, 1.0)
                        alt = winner(sub, wtp, scale)
                        total += 1
                        flips += int(alt != base)
                rows.append({"run": run, "price_kwh": price, "pue": pue,
                             "effective_wtp_multiplier": round(scale, 3),
                             "cells": total, "cells_flipped": flips,
                             "flip_rate": round(flips / total, 3)})

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print(out.to_string(index=False))
    worst = out.flip_rate.max()
    print(f"\nmax flip rate across grid: {worst:.1%}")
    print(f"saved -> {args.out_csv}")

    append_ledger_stub(
        claim="S12 sensitivity: decision-rule winner-map stability under electricity price "
              f"{PRICES} x PUE {PUES} (price/PUE act as linear wtp rescales)",
        value=f"max cell-flip rate {worst:.1%} across all 6 regime runs and grid points; see CSV",
        artifact=args.out_csv,
        command=f"python analysis/cost_sensitivity.py --regime-glob \"{args.regime_glob}\" "
                f"--out-csv {args.out_csv}",
        code_tag=args.code_tag or "n/a",
    )


if __name__ == "__main__":
    main()
