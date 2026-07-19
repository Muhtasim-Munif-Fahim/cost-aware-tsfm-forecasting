#!/usr/bin/env python3
"""Data-quality audit for the city panel. Reports raw coverage and the usable contiguous
window per city (via the same extract_usable_window the harness uses), and which cities
pass the min-hours gate. Run before experiments to avoid analyzing gappy series.

Usage: python data_audit.py [--min-hours 8760] [--data-dir data/cities]
"""
import argparse, glob, os
import pandas as pd
from run_forecast import extract_usable_window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/cities")
    ap.add_argument("--manifest", default="cities_manifest.csv")
    ap.add_argument("--min-hours", type=int, default=8760, help="min usable contiguous hours (1 yr)")
    args = ap.parse_args()

    tier = {}
    if os.path.exists(args.manifest):
        m = pd.read_csv(args.manifest, encoding="utf-8")
        tier = dict(zip(m.city.str.lower().str.replace(" ", "_"), m.tier))

    rows = []
    for f in sorted(glob.glob(os.path.join(args.data_dir, "*.csv"))):
        slug = os.path.splitext(os.path.basename(f))[0]
        d = pd.read_csv(f)
        ts = pd.to_datetime(d["timestamp"])
        raw = pd.Series(d["PM2.5"].astype(float).values, index=ts)
        span_h = (ts.max() - ts.min()).total_seconds() / 3600 + 1
        usable = extract_usable_window(raw, min_hours=args.min_hours)
        rows.append({
            "city": slug, "tier": tier.get(slug, "?"), "raw_rows": len(d),
            "raw_cov_%": round(100 * len(d) / span_h, 1),
            "usable_hours": len(usable) if usable is not None else 0,
            "usable_from": str(usable.index[0].date()) if usable is not None else "-",
            "usable_to": str(usable.index[-1].date()) if usable is not None else "-",
            "PASS": usable is not None,
        })
    df = pd.DataFrame(rows).sort_values(["PASS", "tier", "usable_hours"], ascending=[False, True, False])
    df.to_csv("cities_quality.csv", index=False)
    print(df.to_string(index=False))
    npass = df.PASS.sum()
    prich = df[(df.PASS) & (df.tier == "rich")].shape[0]
    pscarce = df[(df.PASS) & (df.tier == "scarce")].shape[0]
    print(f"\nPASS: {npass}/{len(df)}  (rich={prich}, scarce={pscarce})  min_hours={args.min_hours}")
    print("saved -> cities_quality.csv")


if __name__ == "__main__":
    main()
