"""Assemble the causal-primary panel (reviewer fix: make the deployable, causal-covariate
configuration the main-text result rather than the perfect-foresight one).

Tiers and their causal source:
  seasonal_naive, chronos (univariate), nas_gru_s* (past-only context => already causal)
      <- canonical_preds / canonical_cities.csv   (unchanged; causal-invariant)
  lgbm_direct   <- causal_ablation_preds / causal_ablation_cities.csv  (R020/R021, complete)
  chronos_cov   <- causal_full_preds / causal_full_cities.csv          (new causal run)

Produces, per domain:
  results/v1/<dom>_panel/causal_primary_preds/<city>.npz   (merged, for DM + conformal)
  results/v1/<dom>_panel/causal_primary_cities.csv         (merged tier rows)

Then run dm_panel.py / conformal_panel.py against causal_primary_preds, and make_tables
with the causal DOM/DM files, to produce the causal Table 1.

Usage: python analysis/build_causal_primary.py
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOMAINS = ["pm25", "weather"]
TOL = 1e-6


def merge_domain(dom):
    base = os.path.join(ROOT, "results", "v1", f"{dom}_panel")
    canon_dir = os.path.join(base, "canonical_preds")
    lgbm_dir = os.path.join(base, "causal_ablation_preds")
    cov_dir = os.path.join(base, "causal_full_preds")
    out_dir = os.path.join(base, "causal_primary_preds")
    os.makedirs(out_dir, exist_ok=True)

    merged_cities = []
    for fp in sorted(glob.glob(os.path.join(canon_dir, "*.npz"))):
        city = os.path.splitext(os.path.basename(fp))[0]
        canon = dict(np.load(fp))
        lg_fp = os.path.join(lgbm_dir, f"{city}.npz")
        cov_fp = os.path.join(cov_dir, f"{city}.npz")
        if not (os.path.exists(lg_fp) and os.path.exists(cov_fp)):
            print(f"  [skip {dom}/{city}] missing causal source "
                  f"(lgbm={os.path.exists(lg_fp)}, cov={os.path.exists(cov_fp)})")
            continue
        lg = dict(np.load(lg_fp))
        cov = dict(np.load(cov_fp))
        yt = canon["y_true"]
        # sanity: same test window across sources
        for src, name in [(lg, "causal_ablation"), (cov, "causal_full")]:
            if src["y_true"].shape != yt.shape or np.max(np.abs(src["y_true"] - yt)) > TOL:
                raise SystemExit(f"{dom}/{city}: y_true mismatch vs {name} -- windows differ")
        out = {"y_true": yt, "seasonal_naive": canon["seasonal_naive"],
               "chronos": canon["chronos"], "lgbm_direct": lg["lgbm_direct"],
               "chronos_cov": cov["chronos_cov"]}
        for k in canon:
            if k.startswith("nas_gru_s"):
                out[k] = canon[k]
        np.savez_compressed(os.path.join(out_dir, f"{city}.npz"), **out)
        merged_cities.append(city)
    print(f"  {dom}: merged {len(merged_cities)} cities -> {os.path.relpath(out_dir, ROOT)}")

    # merged cities.csv: naive/chronos/nas from canonical, lgbm from causal_ablation,
    # chronos_cov from causal_full -- MASE is on the same fixed per-series scale, so rows
    # are directly comparable across sources.
    cc = pd.read_csv(os.path.join(base, "canonical_cities.csv"))
    ab = pd.read_csv(os.path.join(base, "causal_ablation_cities.csv"))
    cf = pd.read_csv(os.path.join(base, "causal_full_cities.csv"))
    keep = pd.concat([
        cc[cc.model.isin(["seasonal_naive", "chronos", "nas_gru"])],
        ab[ab.model == "lgbm_direct"],
        cf[cf.model == "chronos_cov"],
    ], ignore_index=True)
    keep = keep[keep.city.isin(merged_cities)]
    out_csv = os.path.join(base, "causal_primary_cities.csv")
    keep.to_csv(out_csv, index=False)
    print(f"  {dom}: causal_primary_cities.csv tiers={sorted(keep.model.unique())} "
          f"cities={keep.city.nunique()}")


def main():
    for dom in DOMAINS:
        merge_domain(dom)


if __name__ == "__main__":
    main()
