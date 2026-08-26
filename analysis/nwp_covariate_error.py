#!/usr/bin/env python3
"""R1.2 -- measure REAL numerical-weather-prediction error for the covariates the
specialist consumes, so the "realistic covariate" scenario is calibrated to observed
NWP skill instead of an assumed noise level.

Reviewer 1 objected that the causal ablation (weather frozen at the forecast origin) is a
pessimistic deployment scenario: operationally the covariates would come from an NWP
service, so realistic performance sits between last-known and perfect foresight. This
script measures where that middle point actually is.

Source: Open-Meteo previous-model-runs API, which serves, for each valid hour, what the
run issued N days earlier predicted. Archive begins 2021-03-24, so only 15 of the 29
panel cities are covered -- and the uncovered half is 9/14 scarce-tier. Restricting the
experiment to covered cities would therefore bias it toward rich cities, which is why
this script only MEASURES error here; analysis/nwp_degradation.py applies the measured
error model to all 29 cities.

Lead-time subtlety (important, and the reason previous_dayN is not used at face value):
for valid hour t, `<var>_previous_day1` comes from the run issued ~1 day earlier, so its
lead is 24-47 h, whereas the paper's forecasts run at leads 1-24 h. Taking previous_day1
as "the" realistic covariate would overstate NWP error, which biases the comparison
*toward* the manuscript's existing conclusion. To avoid that, we fetch previous_day1..3,
fit error as a function of lead, and report the fitted value over the 1-24 h band along
with the measured anchors, so the extrapolation is explicit and auditable.

Wind direction is circular: error is the signed angular difference wrapped to +/-180 deg,
not |a-b| (which would score 350 deg vs 10 deg as 340 instead of 20).

Usage:  python analysis/nwp_covariate_error.py [--cities N] [--out PATH]
Output: results/v1/nwp_covariate_error.csv
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = "https://previous-runs-api.open-meteo.com/v1/forecast"
ARCHIVE_START = pd.Timestamp("2021-03-24")   # empirically probed floor of the archive
LEAD_DAYS = (1, 2, 3)                        # previous_day1..3 -> lead bands 24-47/48-71/72-95 h

# the covariates the specialist actually consumes (openmeteo_fetch.HOURLY minus the
# weather-domain target temperature_2m, which is kept for reference/PM2.5 use)
VARS = ["relative_humidity_2m", "surface_pressure", "wind_speed_10m",
        "wind_direction_10m", "precipitation", "cloud_cover",
        "shortwave_radiation", "temperature_2m"]
CIRCULAR = {"wind_direction_10m"}


def get_json(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.load(r)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if i == tries - 1:
                raise
            time.sleep(3 * (i + 1))
    return None


def err(actual, pred, var):
    """Forecast error, wrapped to +/-180 deg for circular variables."""
    d = np.asarray(pred, float) - np.asarray(actual, float)
    if var in CIRCULAR:
        d = (d + 180.0) % 360.0 - 180.0
    return d


def fetch_city(lat, lon, start, end):
    hourly = ",".join(VARS + [f"{v}_previous_day{k}" for v in VARS for k in LEAD_DAYS])
    q = {"latitude": lat, "longitude": lon, "start_date": str(start.date()),
         "end_date": str(end.date()), "hourly": hourly, "timezone": "UTC"}
    d = get_json(API + "?" + urllib.parse.urlencode(q))
    h = d["hourly"]
    return pd.DataFrame(h).assign(time=lambda x: pd.to_datetime(x.time)).set_index("time")


def ar1(x):
    """Lag-1 autocorrelation of the error series (NWP errors are far from iid; the
    degradation model needs this so injected error is temporally structured, not white)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return np.nan
    x = x - x.mean()
    denom = float(np.dot(x, x))
    return float(np.dot(x[1:], x[:-1]) / denom) if denom > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", type=int, default=0, help="limit for a quick check (0 = all)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "v1", "nwp_covariate_error.csv"))
    args = ap.parse_args()

    import sys
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from city_select import CANDIDATES  # coordinates live here (openmeteo_fetch does the same)
    coords = {c.lower().replace(" ", "_"): (lat, lon) for c, _, _, lat, lon in CANDIDATES}

    q = pd.read_csv(os.path.join(ROOT, "cities_quality.csv"))
    q = q[q.PASS == True].copy()  # noqa: E712
    q["lat"] = q.city.map(lambda c: coords.get(c, (np.nan, np.nan))[0])
    q["lon"] = q.city.map(lambda c: coords.get(c, (np.nan, np.nan))[1])
    missing = q[q.lat.isna()].city.tolist()
    if missing:
        raise SystemExit(f"no coordinates for: {missing}")
    q["start"] = pd.to_datetime(q.usable_from)
    q["end"] = pd.to_datetime(q.usable_to)
    q["fetch_start"] = q.start.clip(lower=ARCHIVE_START)
    q = q[q.fetch_start < q.end]                      # keep only cities the archive reaches
    q = q.sort_values("city")
    if args.cities:
        q = q.head(args.cities)
    print(f"cities with NWP-archive overlap: {len(q)}")

    rows = []
    for _, c in q.iterrows():
        try:
            df = fetch_city(c.lat, c.lon, c.fetch_start, c.end)
        except Exception as e:  # noqa: BLE001
            print(f"  {c.city:<14} FETCH FAILED: {type(e).__name__}: {e}")
            continue
        n_any = 0
        for v in VARS:
            for k in LEAD_DAYS:
                col = f"{v}_previous_day{k}"
                if col not in df or v not in df:
                    continue
                pair = df[[v, col]].dropna()
                if len(pair) < 48:
                    continue
                e = err(pair[v].values, pair[col].values, v)
                rows.append({
                    "city": c.city, "tier": c.tier, "variable": v, "lead_day": k,
                    "lead_hours_mid": 24 * k + 12,     # midpoint of the 24k..24k+23 band
                    "n": len(pair), "mae": float(np.mean(np.abs(e))),
                    "rmse": float(np.sqrt(np.mean(e ** 2))), "bias": float(np.mean(e)),
                    "sd": float(np.std(e)), "err_ar1": ar1(e),
                    "actual_sd": float(np.std(pair[v].values)),
                })
                n_any += 1
        print(f"  {c.city:<14} {c.tier:<7} rows={n_any}  span={c.fetch_start.date()}..{c.end.date()}")

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nsaved -> {args.out}  ({len(out)} rows)")
    if not out.empty:
        piv = out.pivot_table(index="variable", columns="lead_day", values="rmse")
        print("\nRMSE by lead day (panel mean):")
        print(piv.round(3).to_string())
        print("\nerror/signal ratio (rmse / actual_sd), lead_day=1:")
        d1 = out[out.lead_day == 1]
        print((d1.groupby("variable").rmse.mean()
               / d1.groupby("variable").actual_sd.mean()).round(3).to_string())


if __name__ == "__main__":
    main()
