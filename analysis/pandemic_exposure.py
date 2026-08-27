"""Quantify how much of the 29-city panel falls in the 2020-21 pandemic period.

Added 2026-07-28 in response to the pre-submission review: the panel is
non-contemporaneous (each city contributes whichever window its source record
supports), so some cities sit wholly inside the pandemic period and others not at
all. No data are excluded or reweighted here -- this script only measures the
exposure so the Supplementary Information can report it.

Window overlap is computed against 2020-01-01 -- 2021-12-31 inclusive, which is a
deliberately generous bracket: it covers the first lockdowns through the period of
substantially altered urban activity, and errs toward over-reporting exposure.

Usage: python analysis/pandemic_exposure.py
Output: results/v1/pandemic_exposure.csv
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "v1", "pandemic_exposure.csv")

LO = pd.Timestamp("2020-01-01")
HI = pd.Timestamp("2021-12-31")


def main():
    df = pd.read_csv(os.path.join(ROOT, "cities_quality.csv"))
    df = df[df.PASS == True].copy()  # noqa: E712
    df["start"] = pd.to_datetime(df.usable_from)
    df["end"] = pd.to_datetime(df.usable_to)

    overlap_start = df.start.clip(lower=LO)
    overlap_end = df.end.clip(upper=HI)
    df["overlap_days"] = (overlap_end - overlap_start).dt.days.clip(lower=0)
    df["span_days"] = (df.end - df.start).dt.days
    df["overlap_frac"] = (df.overlap_days / df.span_days).round(3)

    hit = df[df.overlap_days > 0].sort_values("overlap_frac", ascending=False)
    out = hit[["city", "tier", "usable_from", "usable_to", "usable_hours",
               "overlap_days", "span_days", "overlap_frac"]]
    out.to_csv(OUT, index=False)

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(out.to_string(index=False))
    print()
    print(f"cities overlapping {LO.date()}--{HI.date()}: {len(hit)} of {len(df)}")
    print(f"  of which scarce-tier: {(hit.tier == 'scarce').sum()}, "
          f"rich-tier: {(hit.tier == 'rich').sum()}")
    print(f"usable hours in affected cities: {hit.usable_hours.sum():,} of "
          f"{df.usable_hours.sum():,} "
          f"({hit.usable_hours.sum() / df.usable_hours.sum():.1%})")
    print(f"wholly inside the window: "
          f"{', '.join(sorted(hit[hit.overlap_frac >= 1.0].city))}")
    print(f"panel window span: {df.start.min().date()} to {df.end.max().date()}")
    print(f"\nwritten -> {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
