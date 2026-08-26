"""Reviewer-requested pretraining-contamination check (added 2026-07-15, logged in
deviations).

Chronos-Bolt is pretrained on large public time-series collections whose exact contents
we cannot audit; "zero-shot" guarantees no gradient updates on the test series but not
that the series (or a public copy) was absent from pretraining. Two lines of defence:

  (1) The OpenAQ panel was fetched fresh and most windows fall in 2024-2026; a strictly
      POST-cutoff slice cannot have entered a corpus assembled earlier. We re-evaluate the
      specialist-vs-foundation-model comparison on each city's data from CUTOFF onward
      (>= enough hours for 6 folds) and check the PM2.5 tie still holds.
  (2) The UCI Beijing set (2013-2017) is a common public benchmark and IS contamination-
      vulnerable; we therefore treat the 12-station result as corroborating, not primary,
      and rely on (1) for the clean check.

CUTOFF = 2024-10-01 (after the Chronos / Chronos-Bolt release corpus). Cities without
that much recent coverage are reported as excluded, not silently dropped.

Usage: python analysis/contamination_check.py
Output: results/v1/contamination_postcutoff.csv (+ printed summary)
"""
import glob
import os
import sys
import types

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))   # run_forecast lives in src/
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results", "v1", "contamination_postcutoff.csv")
CUTOFF = pd.Timestamp("2024-10-01")
MIN_POST_HOURS = 24 * 90    # 90 days, enough for the 6-fold protocol on the tail
H, FOLDS = 24, 6


def main():
    import run_forecast as rf
    man = pd.read_csv(os.path.join(ROOT, "cities_manifest.csv"), encoding="utf-8")
    slug_tier = dict(zip(man.city.str.lower().str.replace(" ", "_"), man.tier))
    files = sorted(glob.glob(os.path.join(ROOT, "data/cities/*.csv")))
    rows, excluded = [], []
    for f in files:
        slug = os.path.splitext(os.path.basename(f))[0]
        if slug not in slug_tier:
            continue
        a = types.SimpleNamespace(source="csv", data_path=f, column=None, min_hours=2160,
                                  pm25_window_dir="data/cities_final")
        try:
            y, exog = rf.load_single(a)
        except Exception as e:
            excluded.append((slug, f"load: {e}"))
            continue
        post = y[y.index >= CUTOFF]
        if len(post) < MIN_POST_HOURS:
            excluded.append((slug, f"post-cutoff hours {len(post)} < {MIN_POST_HOURS}"))
            continue
        ex_post = exog.loc[post.index] if exog is not None and len(exog) else exog
        folds = rf.make_folds(post, H, FOLDS)
        scale = rf.mase_scale(post.values[:folds[0].train_end])
        yt = np.concatenate([fold.y_test for fold in folds])
        ytr = post.values[:folds[0].train_end]
        preds = {}
        preds["chronos"] = rf.run_chronos(post, ex_post, folds, H)[0]
        preds["lgbm_direct"] = rf.run_lgbm_direct(post, ex_post, folds, H, True, False)[0]
        preds["seasonal_naive"] = rf.run_seasonal_naive(post, ex_post, folds, H)[0]
        r = {"city": slug, "tier": slug_tier[slug], "post_hours": len(post)}
        for m, p in preds.items():
            r[f"MASE_{m}"] = round(rf.score(yt, p, ytr, scale=scale)["MASE"], 4)
        r["chronos_beats_lgbm"] = bool(r["MASE_chronos"] < r["MASE_lgbm_direct"])
        rows.append(r)
        print(f"  {slug:16} post={len(post):6}h  chronos={r['MASE_chronos']:.3f} "
              f"lgbm={r['MASE_lgbm_direct']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    if len(df):
        d = (df.MASE_lgbm_direct - df.MASE_chronos)  # >0 FM better
        print(f"\nPOST-CUTOFF (>= {CUTOFF.date()}), n={len(df)} cities with coverage:")
        print(f"  chronos mean MASE {df.MASE_chronos.mean():.3f} vs lgbm {df.MASE_lgbm_direct.mean():.3f}")
        print(f"  chronos beats lgbm in {int(df.chronos_beats_lgbm.sum())}/{len(df)} cities")
        print(f"  mean paired diff (lgbm-chronos) {d.mean():+.3f}")
    print(f"  excluded (insufficient post-cutoff coverage): {len(excluded)}")
    print(f"written -> {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
