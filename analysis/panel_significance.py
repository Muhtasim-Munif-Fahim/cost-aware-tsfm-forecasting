#!/usr/bin/env python3
"""Phase 3 -- panel-level significance tests from a <out_prefix>_cities.csv table.

Per ANALYSIS_PLAN.md S6:
  - Binomial sign test on per-city win/loss, specialist (lgbm_direct) vs FM (chronos).
  - Wilcoxon signed-rank test on paired per-city MASE (same pair).
  - Friedman test + Nemenyi post-hoc across all tiers present, as a supplementary
    robustness check (critical-difference diagram data).

nas_gru seeds are averaged into one per-city MASE first (mean across seeds), matching
the "report mean +/- sd across seeds" convention used in dm_panel.py / conformal_panel.py.

Usage:
  python analysis/panel_significance.py --cities-csv results/v1/pm25_panel/pilot_2city_cities.csv \
      --out-prefix results/v1/pm25_panel/pilot_2city --domain pm25
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, friedmanchisquare, studentized_range, wilcoxon

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


def nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    """Critical difference for the Nemenyi post-hoc test (avg-rank scale)."""
    q = studentized_range.isf(alpha, k, np.inf) / np.sqrt(2)
    return float(q * np.sqrt(k * (k + 1) / (6 * n)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities-csv", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--specialist", default="lgbm_direct")
    ap.add_argument("--fm", default="chronos")
    ap.add_argument("--code-tag", default="")
    args = ap.parse_args()

    df = pd.read_csv(args.cities_csv)
    # collapse nas_gru seeds to one mean-MASE-per-city row
    df["model_g"] = np.where(df.model == "nas_gru", "nas_gru", df.model)
    per_city = (df.groupby(["city", "model_g"])["MASE"].mean().reset_index()
                .rename(columns={"model_g": "model"}))
    wide = per_city.pivot(index="city", columns="model", values="MASE").dropna()

    if args.specialist not in wide.columns or args.fm not in wide.columns:
        raise SystemExit(f"need both {args.specialist!r} and {args.fm!r} columns; "
                          f"have {list(wide.columns)}")

    spec, fm = wide[args.specialist].values, wide[args.fm].values
    wins_spec = int((spec < fm).sum())
    n = len(spec)
    sign_test = binomtest(wins_spec, n, p=0.5)
    wil = wilcoxon(spec, fm) if n >= 6 else None

    fried_cols = [c for c in wide.columns]
    fried = friedmanchisquare(*[wide[c].values for c in fried_cols]) if len(fried_cols) >= 3 else None
    avg_ranks = wide.rank(axis=1, ascending=True).mean().sort_values()
    cd = nemenyi_cd(len(fried_cols), n) if fried is not None else np.nan

    summary_rows = [
        {"test": "binomial_sign", "comparison": f"{args.specialist}_vs_{args.fm}",
         "n_cities": n, "spec_wins": wins_spec, "fm_wins": n - wins_spec,
         "statistic": np.nan, "p_value": sign_test.pvalue},
    ]
    if wil is not None:
        summary_rows.append({"test": "wilcoxon_signed_rank",
                              "comparison": f"{args.specialist}_vs_{args.fm}",
                              "n_cities": n, "spec_wins": np.nan, "fm_wins": np.nan,
                              "statistic": wil.statistic, "p_value": wil.pvalue})
    if fried is not None:
        summary_rows.append({"test": "friedman", "comparison": "all_tiers", "n_cities": n,
                              "spec_wins": np.nan, "fm_wins": np.nan,
                              "statistic": fried.statistic, "p_value": fried.pvalue})

    summary = pd.DataFrame(summary_rows)
    summary_path = f"{args.out_prefix}_panel_significance.csv"
    summary.to_csv(summary_path, index=False)

    ranks_path = f"{args.out_prefix}_nemenyi_ranks.csv"
    ranks_df = avg_ranks.rename("avg_rank").reset_index().rename(columns={"index": "model"})
    ranks_df["nemenyi_cd"] = cd
    ranks_df.to_csv(ranks_path, index=False)

    print(summary.to_string(index=False))
    print(f"\navg ranks (lower=better), Nemenyi CD={cd:.3f} (alpha=0.05, k={len(fried_cols)}, n={n}):")
    print(ranks_df.to_string(index=False))
    print(f"\nsaved -> {summary_path}, {ranks_path}")

    append_ledger_stub(
        claim=f"Panel-level sign/Wilcoxon test, {args.specialist} vs {args.fm}, "
              f"{args.domain} domain ({n} cities); Friedman+Nemenyi across {len(fried_cols)} tiers",
        value=f"sign p={sign_test.pvalue:.4f}, wilcoxon p={(wil.pvalue if wil else float('nan')):.4f}, "
              f"friedman p={(fried.pvalue if fried else float('nan')):.4f}",
        artifact=summary_path,
        command=f"python analysis/panel_significance.py --cities-csv {args.cities_csv} "
                f"--out-prefix {args.out_prefix} --domain {args.domain}",
        code_tag=args.code_tag or "n/a",
    )


if __name__ == "__main__":
    main()
