#!/usr/bin/env python3
"""Pull full hourly PM2.5 history for one OpenAQ v3 sensor -> timestamp,PM2.5 CSV.

Key read from env OPENAQ_KEY (never hard-coded). Year-windowed + paginated to dodge
deep-pagination caps; polite sleep between calls.

Usage: OPENAQ_KEY=... python openaq_fetch.py --sensor 24434 --out data/dhaka_pm25.csv \
                                              --start 2016 --end 2025
"""
import argparse
import os
import time
import urllib.request
import urllib.parse
import json

BASE = "https://api.openaq.org/v3/sensors/{sid}/hours"


def get(url, key):
    req = urllib.request.Request(url, headers={"X-API-Key": key})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _month_windows(start_year, end_year):
    for yr in range(start_year, end_year + 1):
        for mo in range(1, 13):
            nyr, nmo = (yr, mo + 1) if mo < 12 else (yr + 1, 1)
            yield f"{yr}-{mo:02d}-01T00:00:00Z", f"{nyr}-{nmo:02d}-01T00:00:00Z", f"{yr}-{mo:02d}"


def _partial_path(out_csv):
    return out_csv + ".partial.json"


def _load_partial(out_csv):
    p = _partial_path(out_csv)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {"done_months": [], "rows": {}}


def _save_partial(out_csv, state):
    p = _partial_path(out_csv)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp, p)


def fetch_resumable(sensor, key, start, end, out_csv, month_deadline_s=180):
    """Month-windowed, checkpointed fetch: a month has <=744 hours < limit=1000, so every
    window is a single page and completeness is VERIFIABLE (meta.found == len(results)).
    Progress is saved to `<out_csv>.partial.json` after EVERY month, so a kill (this
    environment reaps long-running background processes) only loses the current month, not
    the whole city. Returns 'done' if all months finished, 'partial' otherwise.
    """
    state = _load_partial(out_csv)
    done = set(state["done_months"])
    rows = state["rows"]
    t_start = time.time()
    for w_from, w_to, label in _month_windows(start, end):
        if time.time() - t_start > month_deadline_s * 200:  # generous overall safety valve
            break
        if label in done:
            continue
        q = urllib.parse.urlencode({"datetime_from": w_from, "datetime_to": w_to,
                                    "limit": 1000, "page": 1})
        url = f"{BASE.format(sid=sensor)}?{q}"
        ok = False
        for attempt in range(5):
            try:
                d = get(url, key)
            except Exception as e:  # noqa: BLE001
                wait = 10 * (attempt + 1)
                print(f"  {label} error {e}; retry {attempt+1}/4 in {wait}s", flush=True)
                time.sleep(wait)
                continue
            res = d.get("results", [])
            found = d.get("meta", {}).get("found", len(res))
            found_n = int(str(found).lstrip(">")) if found is not None else len(res)
            if found_n > len(res):
                print(f"  {label} short page ({len(res)}/{found_n}); retry", flush=True)
                time.sleep(10)
                continue
            for r in res:
                if r["value"] is not None:
                    rows[r["period"]["datetimeFrom"]["utc"]] = r["value"]
            if res:
                print(f"  {label}: {len(res)} rows (total {len(rows)})", flush=True)
            ok = True
            break
        if ok:
            done.add(label)
            state["done_months"] = sorted(done)
            _save_partial(out_csv, state)
        time.sleep(0.35)
    all_months = [lbl for _, _, lbl in _month_windows(start, end)]
    finished = done.issuperset(all_months)
    if finished:
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("timestamp,PM2.5\n")
            for ts in sorted(rows):
                f.write(f"{ts},{rows[ts]}\n")
        pp = _partial_path(out_csv)
        if os.path.exists(pp):
            os.remove(pp)
        print(f"saved {len(rows)} hourly rows -> {out_csv}", flush=True)
        return "done"
    print(f"  partial: {len(done)}/{len(all_months)} months done, checkpoint saved", flush=True)
    return "partial"


def fetch(sensor, key, start, end):
    """Non-resumable convenience wrapper (kept for one-off scripts / tests)."""
    rows = {}
    for w_from, w_to, label in _month_windows(start, end):
        q = urllib.parse.urlencode({"datetime_from": w_from, "datetime_to": w_to,
                                    "limit": 1000, "page": 1})
        url = f"{BASE.format(sid=sensor)}?{q}"
        for attempt in range(5):
            try:
                d = get(url, key)
            except Exception as e:  # noqa: BLE001
                wait = 10 * (attempt + 1)
                print(f"  {label} error {e}; retry {attempt+1}/4 in {wait}s", flush=True)
                time.sleep(wait)
                continue
            res = d.get("results", [])
            found = d.get("meta", {}).get("found", len(res))
            found_n = int(str(found).lstrip(">")) if found is not None else len(res)
            if found_n > len(res):
                time.sleep(10)
                continue
            for r in res:
                if r["value"] is not None:
                    rows[r["period"]["datetimeFrom"]["utc"]] = r["value"]
            break
        time.sleep(0.35)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=2016)
    ap.add_argument("--end", type=int, default=2025)
    args = ap.parse_args()
    key = os.environ.get("OPENAQ_KEY")
    if not key:
        raise SystemExit("set OPENAQ_KEY env var")
    rows = fetch(args.sensor, key, args.start, args.end)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("timestamp,PM2.5\n")
        for ts in sorted(rows):
            f.write(f"{ts},{rows[ts]}\n")
    print(f"saved {len(rows)} hourly rows -> {args.out}")


if __name__ == "__main__":
    main()
