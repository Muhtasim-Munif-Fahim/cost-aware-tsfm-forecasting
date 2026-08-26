#!/usr/bin/env python3
"""R2.1 -- does the headline specialist-vs-foundation-model conclusion survive when the
panel is stratified by pandemic exposure and by observation epoch?

Reviewer 2 noted that the 29-city panel is non-contemporaneous (usable windows span
2016-2026) and that ten cities overlap the COVID-19 period, and asked for a stratified or
sensitivity analysis showing the aggregate results are not driven by that heterogeneity.

`analysis/pandemic_exposure.py` already MEASURES exposure. This script re-analyses the
already-computed per-city results under that split -- no models are re-run and no data are
excluded or reweighted, so nothing here can change a published number; it only asks
whether the same conclusion holds inside each stratum.

Strata:
  * pandemic exposure: overlap_frac > 0 (from results/v1/pandemic_exposure.csv) vs not.
    A city is called "heavily exposed" at overlap_frac >= 0.5.
  * epoch: whether the city's usable window ends before/after 2021-12-31, which separates
    the pre/early-pandemic records from the recent ones.

For each stratum we report the paired specialist-vs-FM contrast the paper leads with,
tested with a paired Wilcoxon signed-rank on per-city MASE (the same test the panel
analysis uses). Small strata are reported with their n and an explicit
underpowered flag rather than being silently interpreted -- with 10 exposed cities the
test can fail to reject purely for lack of power, which is NOT evidence of equivalence.

Usage:  python analysis/pandemic_stratified.py [--domain pm25|weather|both]
Output: results/v1/pandemic_stratified.csv
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPOSURE = os.path.join(ROOT, "results", "v1", "pandemic_exposure.csv")
QUALITY = os.path.join(ROOT, "cities_quality.csv")
OUT = os.path.join(ROOT, "results", "v1", "pandemic_stratified.csv")

# (label, path) per domain -- the causal-covariate panel is the manuscript's primary
# deployable setting, so it is the one stratified here.
# `causal_primary` -- NOT `causal_full` -- is the manuscript's cited artifact: RESULTS_LEDGER
# L-037 sources Table 1 and the Abstract from causal_primary_cities.csv (pm25 chronos 0.662 /
# lgbm 0.692). causal_full_cities.csv differs in the pm25 domain (lgbm 0.7068) and would
# stratify a panel the paper does not report.
PANELS = {
    "pm25": os.path.join(ROOT, "results", "v1", "pm25_panel", "causal_primary_cities.csv"),
    "weather": os.path.join(ROOT, "results", "v1", "weather_panel", "causal_primary_cities.csv"),
}
# contrasts the paper leads with: (specialist, foundation model)
CONTRASTS = [("lgbm_direct", "chronos")]
MIN_N = 8   # below this a paired Wilcoxon is too underpowered to interpret either way


def mase_by_city(path, model):
    df = pd.read_csv(path)
    d = df[df.model == model]
    if d.empty:
        return pd.Series(dtype=float)
    # average any seed replicates first (matches the panel analysis convention)
    return d.groupby("city").MASE.mean()


def contrast(a: pd.Series, b: pd.Series, label_a, label_b, stratum, domain, n_total):
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    n = len(common)
    diff = (a - b).values                      # >0 => FM better (lower MASE)
    row = {"domain": domain, "stratum": stratum, "n_cities": n,
           "model_a": label_a, "model_b": label_b,
           "mean_mase_a": float(a.mean()) if n else np.nan,
           "mean_mase_b": float(b.mean()) if n else np.nan,
           "mean_diff_a_minus_b": float(np.mean(diff)) if n else np.nan,
           "median_diff": float(np.median(diff)) if n else np.nan,
           "n_cities_fm_better": int((diff > 0).sum()) if n else 0,
           "underpowered": n < MIN_N,
           "frac_of_panel": round(n / n_total, 3) if n_total else np.nan}
    if n >= 3 and np.any(diff != 0):
        try:
            stat, p = wilcoxon(a.values, b.values)
            row["wilcoxon_stat"], row["p_value"] = float(stat), float(p)
        except ValueError as e:                # all-zero differences etc.
            row["wilcoxon_stat"], row["p_value"] = np.nan, np.nan
            row["note"] = str(e)
    else:
        row["wilcoxon_stat"], row["p_value"] = np.nan, np.nan
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="both", choices=["pm25", "weather", "both"])
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    exp = pd.read_csv(EXPOSURE).set_index("city")
    qual = pd.read_csv(QUALITY)
    qual = qual[qual.PASS == True].set_index("city")  # noqa: E712
    qual["end"] = pd.to_datetime(qual.usable_to)

    domains = ["pm25", "weather"] if args.domain == "both" else [args.domain]
    rows = []
    for dom in domains:
        path = PANELS[dom]
        if not os.path.exists(path):
            print(f"skip {dom}: {path} not found")
            continue
        cities = pd.read_csv(path).city.unique()
        n_total = len(cities)

        frac = pd.Series(0.0, index=cities)
        frac.update(exp.overlap_frac.reindex(cities).fillna(0.0))
        ends = qual.end.reindex(cities)

        strata = {
            "all": pd.Index(cities),
            "pandemic_exposed": frac[frac > 0].index,
            "pandemic_unexposed": frac[frac == 0].index,
            "pandemic_heavy(>=0.5)": frac[frac >= 0.5].index,
            "window_ends_le_2021": ends[ends <= "2021-12-31"].index,
            "window_ends_gt_2021": ends[ends > "2021-12-31"].index,
        }

        for spec, fm in CONTRASTS:
            a_all, b_all = mase_by_city(path, spec), mase_by_city(path, fm)
            if a_all.empty or b_all.empty:
                print(f"skip {dom} {spec} vs {fm}: model missing from panel")
                continue
            for name, idx in strata.items():
                rows.append(contrast(a_all.reindex(idx).dropna(),
                                     b_all.reindex(idx).dropna(),
                                     spec, fm, name, dom, n_total))

    out = pd.DataFrame(rows)

    # These are exploratory SUBGROUP tests: 5 strata x 2 domains = 10 of them (plus the
    # two whole-panel rows, excluded from correction because they nest the strata).
    # Without multiplicity control a nominal p < 0.05 in one subgroup is close to expected
    # by chance, so BH-adjust across the subgroup family and report both columns -- the
    # paper already uses BH for the per-city DM panel, so this matches house convention.
    out["p_value_bh"] = np.nan
    sub = out[(out.stratum != "all") & out.p_value.notna()]
    if len(sub):
        pv = sub.p_value.values
        order = np.argsort(pv)
        m = len(pv)
        adj = np.empty(m)
        prev = 1.0
        for rank in range(m - 1, -1, -1):
            i = order[rank]
            prev = min(prev, pv[i] * m / (rank + 1))
            adj[i] = min(prev, 1.0)
        out.loc[sub.index, "p_value_bh"] = adj

    out.to_csv(args.out, index=False)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    show = ["domain", "stratum", "n_cities", "mean_mase_a", "mean_mase_b",
            "mean_diff_a_minus_b", "n_cities_fm_better", "p_value", "p_value_bh",
            "underpowered"]
    print(out[show].round(4).to_string(index=False))
    print(f"\nsaved -> {args.out}")
    print("\nmean_diff_a_minus_b > 0 means the foundation model has the lower MASE.")
    print("`underpowered` strata must not be read as evidence of equivalence -- a paired")
    print(f"Wilcoxon on n < {MIN_N} cities can fail to reject for lack of power alone.")


if __name__ == "__main__":
    main()
