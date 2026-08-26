"""Build causal-primary decision winner maps for Figure 6.

The causal regime runs were launched without --with-nas (nas_gru is causal-invariant --
it consumes only the past context window, no future covariates). This script grafts the
nas_gru rows from the canonical regime CSVs onto the causal regime CSVs (causal
lgbm/chronos/chronos_cov/naive), then recomputes the winning tier per
(training-history regime x cost-penalty lambda) cell, writing a decision CSV in the same
format Figure 6 consumes.

Winner objective (matches analysis/cost_sensitivity.py): argmin_model MASE + lambda * usd_per_1k
at the harness base price/PUE (scale = 1).

Usage: python analysis/build_causal_decision.py
Output: results/v1/regime/causal_<city>_<dom>_decision.csv  (x6)
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG = os.path.join(ROOT, "results", "v1", "regime")
CITIES = ["beijing", "seoul", "nairobi"]
DOMAINS = ["pm25", "weather"]
WTPS = [0, 500, 1500, 5000, 20000]


def winner(sub, wtp):
    agg = sub.groupby("model", as_index=False)[["MASE", "usd_per_1k"]].mean()
    obj = agg["MASE"] + wtp * agg["usd_per_1k"]
    return agg.loc[obj.idxmin(), "model"]


def main():
    made = 0
    for city in CITIES:
        for dom in DOMAINS:
            causal_fp = os.path.join(REG, f"causal_{city}_{dom}_regime.csv")
            canon_fp = os.path.join(REG, f"canonical_{city}_{dom}_regime.csv")
            if not os.path.exists(causal_fp):
                print(f"  [skip {city}_{dom}] no causal regime CSV yet")
                continue
            causal = pd.read_csv(causal_fp)
            # graft nas_gru rows from canonical (causal-invariant tier)
            if os.path.exists(canon_fp):
                canon = pd.read_csv(canon_fp)
                nas = canon[canon.model == "nas_gru"]
                merged = pd.concat([causal[causal.model != "nas_gru"], nas], ignore_index=True)
            else:
                merged = causal
                print(f"  [warn {city}_{dom}] no canonical regime -> nas_gru absent from map")
            rows = []
            for W in sorted(merged.train_weeks.unique()):
                sub = merged[merged.train_weeks == W]
                row = {"train_weeks": W}
                for wtp in WTPS:
                    row[f"wtp={wtp}"] = winner(sub, wtp)
                rows.append(row)
            out = pd.DataFrame(rows)
            out_fp = os.path.join(REG, f"causal_{city}_{dom}_decision.csv")
            out.to_csv(out_fp, index=False)
            made += 1
            print(f"  {city}_{dom}: {os.path.relpath(out_fp, ROOT)}")
    print(f"built {made} causal decision maps")


if __name__ == "__main__":
    main()
