#!/usr/bin/env python3
"""R1.3 -- do the Diebold-Mariano verdicts survive the dependence structure of the
pooled loss series?

Reviewer 1: "The Diebold-Mariano analysis pools errors across six folds and all forecast
lead times, producing observations that are not obviously independent because errors
belonging to the same forecast trajectory, neighboring time points, and different horizons
can be strongly correlated."

The primary test already handles this parametrically: `src/stats_rigor.diebold_mariano`
estimates the variance of the mean loss differential with a Newey-West HAC estimator
(Bartlett kernel, lags to h-1 = 23) rather than the i.i.d. variance, then applies the
Harvey-Leybourne-Newbold correction against t(n-1). What the manuscript lacked was
evidence that the verdicts do not depend on that parametric choice. This script supplies
it with three checks that make progressively weaker assumptions:

  1. Moving-block bootstrap (MBB). Resamples contiguous blocks of the loss-differential
     series, so serial dependence *within* a block is preserved by construction and never
     modelled. Block length is swept over {12, 24, 48} -- one half, exactly one, and two
     forecast trajectories -- because a block bootstrap's validity depends on the block
     capturing the dependence range, and sweeping it is how that assumption is tested
     rather than assumed.

  2. Trajectory-level test. Collapses each 24-step trajectory to its mean loss
     differential, giving one value per fold origin (n = 6). Fold origins are
     non-overlapping, so these are the closest thing in the design to independent draws.
     This discards all within-trajectory and cross-lead dependence instead of correcting
     for it -- maximally conservative, and badly underpowered at n = 6, which is the point:
     it can only ever weaken a verdict, never manufacture one.

  3. Per-lead panel consistency. For each lead 1..24 separately, tests the per-city mean
     loss differential across cities. If a verdict were an artifact of pooling across
     leads, it would not reproduce lead by lead.

Agreement between the primary DM verdict and these three is what answers the reviewer.
Disagreement is reported rather than smoothed over.

Usage:
  python analysis/dm_dependence_robustness.py --preds-dir results/v1/pm25_panel/causal_primary_preds --domain pm25
Output: results/v1/<domain>_dm_dependence_robustness.csv  (+ per-lead CSV)
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from stats_rigor import diebold_mariano  # noqa: E402

HORIZON = 24
N_FOLDS = 6
BLOCK_LENGTHS = (12, 24, 48)
N_BOOT = 2000
# The manuscript's confirmatory contrast (specialist vs zero-shot foundation model).
PAIRS = [("lgbm_direct", "chronos")]


def collapse_seeds(raw: dict) -> dict:
    """Average nas_gru_s42..s46 into one series, matching analysis/dm_panel.py."""
    out, seeds = {}, []
    for k, v in raw.items():
        if k.startswith("nas_gru_s"):
            seeds.append(v)
        elif k != "y_true":
            out[k] = v
    if seeds:
        out["nas_gru"] = np.mean(np.stack(seeds, 0), 0)
    return out


def loss_diff(y, pa, pb):
    """Absolute-error loss differential; negative mean => model A better."""
    return np.abs(y - pa) - np.abs(y - pb)


def moving_block_bootstrap(d, block, n_boot=N_BOOT, seed=0):
    """Percentile CI for mean(d) under a moving-block bootstrap.

    Blocks are contiguous, so within-block serial correlation is carried through the
    resample untouched -- no variance model is imposed.
    """
    rng = np.random.default_rng(seed)
    n = len(d)
    if n < block + 1:
        return np.nan, np.nan
    n_blocks = int(np.ceil(n / block))
    starts_max = n - block + 1
    idx = (rng.integers(0, starts_max, size=(n_boot, n_blocks))[:, :, None]
           + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-dir", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results", "v1"))
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.preds_dir, "*.npz")))
    if not files:
        raise SystemExit(f"no .npz in {args.preds_dir}")

    rows, lead_rows = [], []
    for fp in files:
        city = os.path.splitext(os.path.basename(fp))[0]
        raw = dict(np.load(fp))
        y = raw["y_true"]
        preds = collapse_seeds(raw)
        for a, b in PAIRS:
            if a not in preds or b not in preds:
                continue
            d = loss_diff(y, preds[a], preds[b])
            n = len(d)

            # (0) primary parametric verdict, for comparison
            dm = diebold_mariano(y, preds[a], preds[b], h=HORIZON)
            dm_sig = bool((dm["p_value"] or 1) < 0.05)

            row = {"domain": args.domain, "city": city, "model_a": a, "model_b": b,
                   "n_obs": n, "mean_loss_diff": float(d.mean()),
                   "dm_p": dm["p_value"], "dm_significant": dm_sig}

            # (1) moving-block bootstrap across block lengths
            for L in BLOCK_LENGTHS:
                lo, hi = moving_block_bootstrap(d, L, seed=abs(hash((city, a, b, L))) % 2**32)
                row[f"mbb{L}_lo"], row[f"mbb{L}_hi"] = lo, hi
                row[f"mbb{L}_significant"] = bool(np.isfinite(lo) and (lo > 0 or hi < 0))

            # (2) trajectory-level test (n = 6 fold origins)
            if n == N_FOLDS * HORIZON:
                per_fold = d.reshape(N_FOLDS, HORIZON).mean(axis=1)
                row["fold_mean_diff"] = float(per_fold.mean())
                if np.any(per_fold != 0):
                    row["fold_t_p"] = float(stats.ttest_1samp(per_fold, 0.0).pvalue)
                    n_neg = int((per_fold < 0).sum())
                    row["fold_sign_p"] = float(stats.binomtest(n_neg, N_FOLDS, 0.5).pvalue)
                    row["fold_n_favouring_a"] = n_neg
                else:
                    row["fold_t_p"] = row["fold_sign_p"] = np.nan
                row["fold_significant"] = bool((row.get("fold_t_p") or 1) < 0.05)

                # (3) per-lead means, aggregated across cities afterwards
                per_lead = d.reshape(N_FOLDS, HORIZON).mean(axis=0)
                for l_i, v in enumerate(per_lead, start=1):
                    lead_rows.append({"domain": args.domain, "city": city, "model_a": a,
                                      "model_b": b, "lead": l_i, "mean_loss_diff": float(v)})
            rows.append(row)

    out = pd.DataFrame(rows)
    p_out = os.path.join(args.out_dir, f"{args.domain}_dm_dependence_robustness.csv")
    out.to_csv(p_out, index=False)

    leads = pd.DataFrame(lead_rows)
    p_lead = os.path.join(args.out_dir, f"{args.domain}_dm_perlead.csv")
    lead_summ = []
    if not leads.empty:
        for (a, b, l_i), g in leads.groupby(["model_a", "model_b", "lead"]):
            v = g.mean_loss_diff.values
            try:
                w = float(stats.wilcoxon(v).pvalue)
            except ValueError:
                w = np.nan
            lead_summ.append({"domain": args.domain, "model_a": a, "model_b": b, "lead": l_i,
                              "n_cities": len(v), "mean_diff": float(v.mean()),
                              "n_favouring_a": int((v < 0).sum()), "wilcoxon_p": w})
        pd.DataFrame(lead_summ).to_csv(p_lead, index=False)

    pd.set_option("display.width", 220, "display.max_columns", 40)
    print(f"=== {args.domain}: DM dependence robustness ({len(out)} city x pair rows) ===\n")
    for (a, b), g in out.groupby(["model_a", "model_b"]):
        n = len(g)
        print(f"{a} vs {b}   ({n} cities)")
        print(f"  primary DM significant           : {int(g.dm_significant.sum())}/{n}")
        for L in BLOCK_LENGTHS:
            col = f"mbb{L}_significant"
            agree = int((g[col] == g.dm_significant).sum())
            print(f"  moving-block bootstrap L={L:<3}      : {int(g[col].sum()):>2}/{n} significant"
                  f"   agrees with DM on {agree}/{n} cities")
        if "fold_significant" in g:
            agree = int((g.fold_significant == g.dm_significant).sum())
            print(f"  trajectory-level t-test (n=6)    : {int(g.fold_significant.sum()):>2}/{n} significant"
                  f"   agrees with DM on {agree}/{n} cities  [underpowered by design]")
        print(f"  cities favouring {a:<12}: {int((g.mean_loss_diff < 0).sum())}/{n}")
        print()

    if lead_summ:
        ls = pd.DataFrame(lead_summ)
        print("--- per-lead panel consistency (is the verdict an artifact of pooling leads?) ---")
        for (a, b), g in ls.groupby(["model_a", "model_b"]):
            same = int((np.sign(g.mean_diff) == np.sign(g.mean_diff.iloc[0])).sum())
            print(f"  {a} vs {b}: sign of mean diff identical at {same}/{len(g)} of 24 leads; "
                  f"Wilcoxon p<0.05 at {int((g.wilcoxon_p < 0.05).sum())}/{len(g)} leads")
            print(f"    mean diff range across leads: [{g.mean_diff.min():+.4f}, {g.mean_diff.max():+.4f}]")
    print(f"\nsaved -> {p_out}")
    if lead_summ:
        print(f"saved -> {p_lead}")


if __name__ == "__main__":
    main()
