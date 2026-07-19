#!/usr/bin/env python3
"""Fetch hourly weather covariates per city from the Open-Meteo historical archive
(free, no API key) -> data/weather/<city>.csv. Same weather source Green-NAS used,
so the merged paper's two domains share weather inputs.

Covariates: temperature_2m, relative_humidity_2m, surface_pressure, wind_speed_10m,
wind_direction_10m, precipitation, cloud_cover, shortwave_radiation.

Usage: python openmeteo_fetch.py [--start 2016-01-01] [--end 2026-06-30] [--max-new N]
Reads city coordinates from city_select.CANDIDATES.
"""
import argparse, csv, json, os, time, urllib.request, urllib.parse
from city_select import CANDIDATES

HOURLY = ("temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,"
          "wind_direction_10m,precipitation,cloud_cover,shortwave_radiation")
BASE = "https://archive-api.open-meteo.com/v1/archive"


def slug(city):
    return city.lower().replace(" ", "_")


def get(url):
    with urllib.request.urlopen(url, timeout=90) as r:
        return json.load(r)


def fetch_city(lat, lon, start, end):
    q = urllib.parse.urlencode({"latitude": lat, "longitude": lon, "start_date": start,
                                "end_date": end, "hourly": HOURLY, "timezone": "UTC"})
    for attempt in range(4):
        try:
            d = get(f"{BASE}?{q}")
            return d.get("hourly", {})
        except Exception as e:  # noqa: BLE001
            print(f"    attempt {attempt+1} error {e}; wait 20s", flush=True)
            time.sleep(20)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2026-06-30")
    ap.add_argument("--max-new", type=int, default=0)
    args = ap.parse_args()
    os.makedirs("data/weather", exist_ok=True)
    new = 0
    for city, country, tier, lat, lon in CANDIDATES:
        out = f"data/weather/{slug(city)}.csv"
        if os.path.exists(out):
            continue
        if args.max_new and new >= args.max_new:
            print(f"reached max-new={args.max_new}, stopping", flush=True)
            break
        new += 1
        print(f"{city} ({lat},{lon}) ...", flush=True)
        h = fetch_city(lat, lon, args.start, args.end)
        if not h or "time" not in h:
            print(f"  {city}: no data, skip", flush=True)
            continue
        cols = [c for c in h if c != "time"]
        with open(out, "w", newline="", encoding="utf-8") as fp:
            w = csv.writer(fp)
            w.writerow(["timestamp"] + cols)
            for i, t in enumerate(h["time"]):
                w.writerow([t] + [h[c][i] for c in cols])
        print(f"  saved {len(h['time'])} rows -> {out}", flush=True)
        time.sleep(1)
    print("WEATHER DONE", flush=True)


if __name__ == "__main__":
    main()
