#!/usr/bin/env python3
"""Phase 3 -- panel-level split-conformal intervals from saved per-city predictions.

Per ANALYSIS_PLAN.md S6: calibrate on the first half of each series' backtest
predictions, report coverage/width on the second half. Primary: pooled per
tier x model. Supplementary: per-city.

nas_gru seeds are averaged into one series first (same convention as dm_panel.py).

Usage:
  python analysis/conformal_panel.py --preds-dir results/v1/pm25_panel/pilot_2city_preds \
      --manifest cities_manifest.csv --out-prefix results/v1/pm25_panel/pilot_2city --domain pm25
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# stats_rigor/run_forecast live in src/ after the 2026-07-17 restructure; adding only
# the repo root left this script unrunnable from a clean checkout.
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)
from stats_rigor import split_conformal  # noqa: E402

LEDGER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "paper", "RESULTS_LEDGER.md")


def collapse_seeds(preds: dict) -> dict:
    out, seed_cols = {}, []
    for k, v in preds.items():
        if k.startswith("nas_gru_s"):
            seed_cols.append(v)
        elif k != "y_true":
            out[k] = v
    if seed_cols:
        out["nas_gru"] = np.mean(np.stack(seed_cols, axis=0), axis=0)
    return out


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-dir", required=True)
    ap.add_argument("--manifest", default="cities_manifest.csv",
                     help="for tier lookup (rich/scarce), joined by lowercased/underscored city name")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--code-tag", default="")
    args = ap.parse_args()

    man = pd.read_csv(args.manifest, encoding="utf-8")
    tier = dict(zip(man.city.str.lower().str.replace(" ", "_"), man.tier))

    files = sorted(glob.glob(os.path.join(args.preds_dir, "*.npz")))
    if not files:
        raise SystemExit(f"no .npz files in {args.preds_dir}")

    rows = []
    for fp in files:
        city = os.path.splitext(os.path.basename(fp))[0]
        d = np.load(fp)
        y_true = d["y_true"]
        preds = collapse_seeds({k: d[k] for k in d.files})
        half = len(y_true) // 2
        for model, p in preds.items():
            r = split_conformal(y_true[:half], p[:half], y_true[half:], p[half:], alpha=args.alpha)
            rows.append({"city": city, "tier": tier.get(city, "?"), "model": model, **r})

    df = pd.DataFrame(rows)
    per_city_path = f"{args.out_prefix}_conformal_percity.csv"
    df.to_csv(per_city_path, index=False)

    pooled = (df.groupby(["tier", "model"])
              .agg(mean_coverage=("coverage", "mean"), mean_width=("width", "mean"),
                   n_cities=("city", "count"))
              .reset_index())
    pooled_path = f"{args.out_prefix}_conformal_pooled.csv"
    pooled.to_csv(pooled_path, index=False)

    print(pooled.to_string(index=False))
    print(f"\nsaved -> {per_city_path}, {pooled_path}")

    target = 1 - args.alpha
    append_ledger_stub(
        claim=f"Split-conformal ({target:.0%}) coverage/width pooled per tier x model, "
              f"{args.domain} domain ({df.city.nunique()} cities)",
        value=f"see {os.path.basename(pooled_path)}",
        artifact=pooled_path,
        command=f"python analysis/conformal_panel.py --preds-dir {args.preds_dir} "
                f"--manifest {args.manifest} --out-prefix {args.out_prefix} --domain {args.domain}",
        code_tag=args.code_tag or "n/a",
    )


if __name__ == "__main__":
    main()
