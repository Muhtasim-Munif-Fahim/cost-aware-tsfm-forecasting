#!/usr/bin/env python3
"""Phase 3 -- panel-level Diebold-Mariano tests from saved per-city predictions.

Reads the <out_prefix>_preds/<city>.npz files written by `run_forecast.py cities`
(--preds-dir), runs the all-pairs DM test (stats_rigor.paired_summary, h=24,
absolute-error loss, HLN small-sample correction) per city, and rolls up a
panel-level significant-win-count summary per model pair.

nas_gru has 5 seed columns (nas_gru_s42..s46) in the npz; they are averaged into
a single `nas_gru` prediction series before the DM test, per ANALYSIS_PLAN.md
S4 ("report mean +/- sd across seeds, no cherry-picking a best seed") -- the DM
test compares against that ensemble-mean forecast, not any individual seed.

Two robustness layers (added at pre-writing review):
  * The seed-ensemble mean is a *stronger* forecaster than a deployed single-seed
    model (variance reduction), so every nas_gru pair is ALSO tested per seed;
    `<out>_dm_panel_perseed.csv` reports, per city x pair, how many of the 5 seeds
    individually reach the same significant verdict. Paper must label the primary
    nas_gru DM rows "5-seed ensemble mean".
  * Per-city p-values within each model pair are Benjamini-Hochberg adjusted across
    the panel (29 tests per pair); the summary reports both raw and FDR-adjusted
    significant-win counts.

Usage:
  python analysis/dm_panel.py --preds-dir results/v1/pm25_panel/pilot_2city_preds \
      --out-prefix results/v1/pm25_panel/pilot_2city --domain pm25 --h 24
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stats_rigor import paired_summary  # noqa: E402

LEDGER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "paper", "RESULTS_LEDGER.md")


def collapse_seeds(preds: dict) -> dict:
    """Average nas_gru_s42..s46 into one 'nas_gru' series; pass other models through."""
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
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--domain", required=True, help="label for the ledger claim, e.g. pm25/weather")
    ap.add_argument("--h", type=int, default=24)
    ap.add_argument("--code-tag", default="")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.preds_dir, "*.npz")))
    if not files:
        raise SystemExit(f"no .npz files in {args.preds_dir}")

    from stats_rigor import diebold_mariano

    per_city_rows, per_seed_rows = [], []
    for fp in files:
        city = os.path.splitext(os.path.basename(fp))[0]
        d = np.load(fp)
        y_true = d["y_true"]
        raw = {k: d[k] for k in d.files}
        preds = collapse_seeds(raw)
        for r in paired_summary(y_true, preds, h=args.h):
            per_city_rows.append({"city": city, **r})
        # per-seed robustness: rerun every nas_gru pair once per individual seed
        seed_keys = sorted(k for k in raw if k.startswith("nas_gru_s"))
        others = [k for k in raw if k not in seed_keys and k != "y_true"]
        for sk in seed_keys:
            for other in others:
                r = diebold_mariano(y_true, raw[sk], raw[other], h=args.h)
                winner = sk if r["mean_loss_diff"] < 0 else other
                per_seed_rows.append({"city": city, "seed": sk.replace("nas_gru_s", ""),
                                      "model_a": "nas_gru", "model_b": other, **r,
                                      "better": ("nas_gru" if winner == sk else other),
                                      "significant_5pct": (r["p_value"] or 1) < 0.05})

    df = pd.DataFrame(per_city_rows)
    # Benjamini-Hochberg within each model pair, across the panel's cities
    df["p_value_fdr"] = np.nan
    for (_, _), idx in df.groupby(["model_a", "model_b"]).groups.items():
        p = df.loc[idx, "p_value"].values
        order = np.argsort(p)
        m = len(p)
        adj = np.empty(m)
        prev = 1.0
        for rank in range(m - 1, -1, -1):
            i = order[rank]
            prev = min(prev, p[i] * m / (rank + 1))
            adj[i] = prev
        df.loc[idx, "p_value_fdr"] = adj
    df["significant_5pct_fdr"] = df.p_value_fdr < 0.05
    per_city_path = f"{args.out_prefix}_dm_panel.csv"
    df.to_csv(per_city_path, index=False)

    if per_seed_rows:
        ps = pd.DataFrame(per_seed_rows)
        ps.to_csv(f"{args.out_prefix}_dm_panel_perseed.csv", index=False)

    # panel-level rollup: for each model pair, how many cities is the win significant in
    rows = []
    for (a, b), g in df.groupby(["model_a", "model_b"]):
        sig = g[g.significant_5pct]
        sig_fdr = g[g.significant_5pct_fdr]
        rows.append({"model_a": a, "model_b": b, "n_cities": len(g),
                     "n_significant": len(sig),
                     "a_sig_wins": (sig.better == a).sum(),
                     "b_sig_wins": (sig.better == b).sum(),
                     "a_sig_wins_fdr": (sig_fdr.better == a).sum(),
                     "b_sig_wins_fdr": (sig_fdr.better == b).sum(),
                     "median_p_value": g.p_value.median()})
    summary = pd.DataFrame(rows)
    summary_path = f"{args.out_prefix}_dm_panel_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(summary.to_string(index=False))
    print(f"\nsaved -> {per_city_path}, {summary_path}" +
          (f", {args.out_prefix}_dm_panel_perseed.csv" if per_seed_rows else ""))

    append_ledger_stub(
        claim=f"Panel-level DM-significant win counts, {args.domain} domain ({len(files)} cities)",
        value=f"see {os.path.basename(summary_path)}",
        artifact=summary_path,
        command=f"python analysis/dm_panel.py --preds-dir {args.preds_dir} "
                f"--out-prefix {args.out_prefix} --domain {args.domain} --h {args.h}",
        code_tag=args.code_tag or "n/a",
    )


if __name__ == "__main__":
    main()
