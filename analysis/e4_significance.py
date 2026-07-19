#!/usr/bin/env python3
"""Phase 3 -- statistical backing for the E4 interpretation-matrix classification.

ANALYSIS_PLAN.md S8's interpretation matrix separates bucket 3 ("zero-shot FM beats
transfer even at 100% fine-tune") from bucket 4 ("all three statistically
indistinguishable") -- so the bucket assignment needs significance tests, not just
mean-MASE comparisons. e4_transfer.py stores per-(city, strategy, fraction, seed)
MASE but not raw predictions, so the tests operate at the CITY level (n=15 paired
observations per fraction), which is also the level the claim generalizes over:

  - paired Wilcoxon signed-rank on per-city MASE: chronos_zeroshot vs nas_transfer
    (seed-mean), and chronos_zeroshot vs lgbm_refit, at each fraction;
  - binomial sign test on per-city win/loss for the same pairs;
  - Holm correction across the fraction-level Wilcoxon family (one family per
    comparator), since 4 fractions x 2 comparators = 8 tests invite cherry-picking.

Bucket call: bucket 3 requires chronos to be significantly better at EVERY fraction
(Holm-adjusted Wilcoxon p < .05 with chronos ahead); if no fraction shows a
significant difference in either direction, that is bucket 4; mixed outcomes are
reported as such (with per-fraction detail) rather than force-fitted.

Usage:
  python analysis/e4_significance.py --results-csv results/v1/e4_transfer/canonical_pm25_results.csv \
      --out-prefix results/v1/e4_transfer/canonical_pm25 --domain pm25
"""
from __future__ import annotations

import argparse
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

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


def holm(pvals):
    """Holm step-down adjusted p-values (order-preserving)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-csv", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--code-tag", default="")
    args = ap.parse_args()

    df = pd.read_csv(args.results_csv)
    chronos = df[df.strategy == "chronos_zeroshot"].groupby("city").MASE.mean()
    nas = df[df.strategy == "nas_transfer"].groupby(["city", "fraction"]).MASE.mean().unstack()
    lgbm = df[df.strategy == "lgbm_refit"].groupby(["city", "fraction"]).MASE.mean().unstack()

    rows = []
    for comp_name, comp in (("nas_transfer", nas), ("lgbm_refit", lgbm)):
        for frac in sorted(c for c in comp.columns):
            pair = pd.DataFrame({"chronos": chronos, "comp": comp[frac]}).dropna()
            n = len(pair)
            if n < 6:
                continue
            diff = pair.chronos - pair.comp          # negative => chronos better
            w = wilcoxon(pair.chronos, pair.comp)
            chronos_wins = int((diff < 0).sum())
            sign = binomtest(chronos_wins, n, p=0.5)
            rows.append({"comparator": comp_name, "fraction": frac, "n_cities": n,
                         "chronos_wins": chronos_wins, "comp_wins": n - chronos_wins,
                         "mean_mase_chronos": pair.chronos.mean(),
                         "mean_mase_comp": pair.comp.mean(),
                         "median_diff": float(diff.median()),
                         "wilcoxon_stat": w.statistic, "wilcoxon_p": w.pvalue,
                         "sign_p": sign.pvalue})

    out = pd.DataFrame(rows)
    # Holm within each comparator family (across fractions)
    out["wilcoxon_p_holm"] = np.nan
    for comp_name in out.comparator.unique():
        mask = out.comparator == comp_name
        out.loc[mask, "wilcoxon_p_holm"] = holm(out.loc[mask, "wilcoxon_p"].values)
    out["chronos_sig_better_5pct"] = (out.wilcoxon_p_holm < 0.05) & (out.median_diff < 0)
    out["comp_sig_better_5pct"] = (out.wilcoxon_p_holm < 0.05) & (out.median_diff > 0)

    out_path = f"{args.out_prefix}_significance.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))

    # bucket logic (vs nas_transfer, the transfer strategy the matrix is about)
    nas_rows = out[out.comparator == "nas_transfer"]
    if nas_rows.chronos_sig_better_5pct.all():
        bucket = "3 (FM significantly better at every fraction)"
    elif not (nas_rows.chronos_sig_better_5pct.any() or nas_rows.comp_sig_better_5pct.any()):
        bucket = "4 (statistically indistinguishable at every fraction)"
    else:
        sig_f = nas_rows[nas_rows.chronos_sig_better_5pct].fraction.tolist()
        sig_c = nas_rows[nas_rows.comp_sig_better_5pct].fraction.tolist()
        bucket = f"mixed (FM sig better at fractions {sig_f}; transfer sig better at {sig_c})"
    print(f"\nE4 interpretation-matrix bucket (city-level, Holm-corrected Wilcoxon): {bucket}")
    print(f"saved -> {out_path}")

    append_ledger_stub(
        claim=f"E4 statistical classification, {args.domain} domain: city-level paired Wilcoxon "
              f"(Holm-corrected) + sign tests, chronos_zeroshot vs nas_transfer/lgbm_refit per fraction",
        value=f"bucket = {bucket}",
        artifact=out_path,
        command=f"python analysis/e4_significance.py --results-csv {args.results_csv} "
                f"--out-prefix {args.out_prefix} --domain {args.domain}",
        code_tag=args.code_tag or "n/a",
    )


if __name__ == "__main__":
    main()
