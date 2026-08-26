#!/usr/bin/env python3
"""Phase 3 -- FM-advantage vs data-volume correlation.

Per ANALYSIS_PLAN.md S6: Pearson correlation between (usable hours) and
(lgbm MASE - chronos MASE) across the panel, with a bootstrap (10,000 resample)
95% CI on the correlation. Positive correlation => FM advantage grows with less
data (i.e. the correlation of usable_hours with (lgbm - chronos) should be
negative if chronos's edge shrinks as data grows -- sign is reported as-is,
not pre-interpreted).

Usage:
  python analysis/fm_advantage_correlation.py \
      --cities-csv results/v1/pm25_panel/pilot_2city_cities.csv \
      --quality-csv cities_quality.csv \
      --out-prefix results/v1/pm25_panel/pilot_2city --domain pm25
"""
from __future__ import annotations

import argparse
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

LEDGER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "paper", "RESULTS_LEDGER.md")


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


def bootstrap_corr_ci(x, y, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.array([pearsonr(x[i], y[i])[0] for i in idx])
    boots = boots[np.isfinite(boots)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities-csv", required=True)
    ap.add_argument("--quality-csv", default="cities_quality.csv")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--specialist", default="lgbm_direct")
    ap.add_argument("--fm", default="chronos")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--code-tag", default="")
    args = ap.parse_args()

    df = pd.read_csv(args.cities_csv)
    per_city = df[df.model.isin([args.specialist, args.fm])].pivot_table(
        index="city", columns="model", values="MASE", aggfunc="mean")
    per_city["fm_advantage"] = per_city[args.specialist] - per_city[args.fm]

    qual = pd.read_csv(args.quality_csv)
    qual = qual.set_index(qual.city.str.lower().str.replace(" ", "_"))["usable_hours"]

    merged = per_city.join(qual.rename("usable_hours"), how="inner").dropna(
        subset=["usable_hours", "fm_advantage"])
    if len(merged) < 3:
        raise SystemExit(f"only {len(merged)} cities with both MASE + usable_hours; need >= 3")

    x, y = merged["usable_hours"].values, merged["fm_advantage"].values
    r, p = pearsonr(x, y)
    lo, hi = bootstrap_corr_ci(x, y, n_boot=args.n_boot)

    out_path = f"{args.out_prefix}_fm_advantage_corr.csv"
    merged.reset_index().rename(columns={"index": "city"}).to_csv(out_path, index=False)

    summary = pd.DataFrame([{"domain": args.domain, "n_cities": len(merged),
                              "pearson_r": r, "p_value": p, "boot_ci_lo": lo, "boot_ci_hi": hi}])
    summary_path = f"{args.out_prefix}_fm_advantage_corr_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(summary.to_string(index=False))
    print(f"\nsaved -> {out_path}, {summary_path}")

    append_ledger_stub(
        claim=f"Correlation(usable_hours, {args.specialist}-{args.fm} MASE advantage), "
              f"{args.domain} domain ({len(merged)} cities)",
        value=f"r={r:.3f}, p={p:.4f}, 95% bootstrap CI [{lo:.3f}, {hi:.3f}]",
        artifact=summary_path,
        command=f"python analysis/fm_advantage_correlation.py --cities-csv {args.cities_csv} "
                f"--quality-csv {args.quality_csv} --out-prefix {args.out_prefix} --domain {args.domain}",
        code_tag=args.code_tag or "n/a",
    )


if __name__ == "__main__":
    main()
