#!/usr/bin/env python3
"""Fetch all sensors in cities_manifest.csv -> data/cities/<city>.csv (timestamp,PM2.5).

Resumable: each city's progress checkpoints to data/cities/<city>.csv.partial.json after
every month, so a kill mid-fetch (this environment reaps long-running background processes)
only loses the current month, not the whole city. Partially-fetched cities are always
finished before starting brand-new ones, so MAX_NEW/timeouts never abandon near-complete work.
OPENAQ_KEY from env.
"""
import csv, os, time
from openaq_fetch import fetch_resumable, _partial_path

os.makedirs("data/cities", exist_ok=True)
KEY = os.environ["OPENAQ_KEY"]
MAX_NEW = int(os.environ.get("MAX_NEW", "0")) or 10**9   # fetch at most N *new* (not-yet-started) cities


def slug(city):
    return city.lower().replace(" ", "_")


with open("cities_manifest.csv", encoding="utf-8") as fp:
    rows = list(csv.DictReader(fp))

# order: cities with an in-progress partial file first (finish what's started), then fresh ones
def sort_key(r):
    out = f"data/cities/{slug(r['city'])}.csv"
    return 0 if os.path.exists(_partial_path(out)) else 1

rows.sort(key=sort_key)

new_count = 0
for r in rows:
    out = f"data/cities/{slug(r['city'])}.csv"
    if os.path.exists(out):
        continue
    is_fresh = not os.path.exists(_partial_path(out))
    if is_fresh:
        if new_count >= MAX_NEW:
            continue
        new_count += 1
    y0, y1 = int(r["first"][:4]), int(r["last"][:4])
    print(f"{r['city']} sensor={r['sensor_id']} {y0}-{y1} ...", flush=True)
    try:
        status = fetch_resumable(int(r["sensor_id"]), KEY, y0, y1, out)
    except Exception as e:  # noqa: BLE001
        print(f"    ERROR {e}; will resume next run", flush=True)
        continue
    print(f"  -> {status}", flush=True)
    time.sleep(1)

remaining = sum(1 for r in rows if not os.path.exists(f"data/cities/{slug(r['city'])}.csv"))
print(f"BATCH CHUNK DONE ({remaining} cities still incomplete)", flush=True)
