#!/usr/bin/env python3
"""Pick one long-record PM2.5 sensor per target city from OpenAQ v3.

Balanced panel: data-rich (Global North + China) vs data-scarce (Global South).
For each city coord, find the pm25 sensor with the longest history within radius,
require >= MIN_YEARS span. Emit cities_manifest.csv (city,tier,sensor_id,first,last,years).

OPENAQ_KEY from env. Usage: OPENAQ_KEY=... python city_select.py
"""
import urllib.request, urllib.parse, json, time, os, csv

KEY = os.environ.get("OPENAQ_KEY")   # required only when actually querying (main)
MIN_YEARS = 1.5
RADIUS_M = 25000

# (city, country, tier, lat, lon) — tier: rich | scarce
CANDIDATES = [
    # data-rich
    ("Los Angeles", "US", "rich", 34.05, -118.24),
    ("New York", "US", "rich", 40.71, -74.01),
    ("London", "GB", "rich", 51.51, -0.13),
    ("Paris", "FR", "rich", 48.85, 2.35),
    ("Berlin", "DE", "rich", 52.52, 13.40),
    ("Madrid", "ES", "rich", 40.42, -3.70),
    ("Toronto", "CA", "rich", 43.65, -79.38),
    ("Amsterdam", "NL", "rich", 52.37, 4.90),
    ("Vienna", "AT", "rich", 48.21, 16.37),
    ("Sydney", "AU", "rich", -33.87, 151.21),
    ("Santiago", "CL", "rich", -33.45, -70.67),
    ("Mexico City", "MX", "rich", 19.43, -99.13),
    ("Bangkok", "TH", "rich", 13.76, 100.50),
    ("Seoul", "KR", "rich", 37.57, 126.98),
    ("Krakow", "PL", "rich", 50.06, 19.94),
    # data-scarce (Global South)
    ("Dhaka", "BD", "scarce", 23.78, 90.41),
    ("Delhi", "IN", "scarce", 28.61, 77.21),
    ("Mumbai", "IN", "scarce", 19.08, 72.88),
    ("Kolkata", "IN", "scarce", 22.57, 88.36),
    ("Kathmandu", "NP", "scarce", 27.72, 85.32),
    ("Lahore", "PK", "scarce", 31.55, 74.34),
    ("Karachi", "PK", "scarce", 24.86, 67.01),
    ("Hanoi", "VN", "scarce", 21.03, 105.85),
    ("Jakarta", "ID", "scarce", -6.21, 106.85),
    ("Manila", "PH", "scarce", 14.60, 120.98),
    ("Lima", "PE", "scarce", -12.05, -77.04),
    ("Bogota", "CO", "scarce", 4.71, -74.07),
    ("Lagos", "NG", "scarce", 6.52, 3.38),
    ("Nairobi", "KE", "scarce", -1.29, 36.82),
    ("Kampala", "UG", "scarce", 0.35, 32.58),
    ("Accra", "GH", "scarce", 5.60, -0.19),
    ("Addis Ababa", "ET", "scarce", 9.03, 38.74),
]


def get(url):
    req = urllib.request.Request(url, headers={"X-API-Key": KEY})
    return json.load(urllib.request.urlopen(req, timeout=30))


def span_years(f, l):
    return (int(l[:4]) * 12 + int(l[5:7]) - (int(f[:4]) * 12 + int(f[5:7]))) / 12


def best_sensor(lat, lon):
    q = urllib.parse.urlencode({"coordinates": f"{lat},{lon}", "radius": RADIUS_M,
                                "parameters_id": 2, "limit": 100})
    d = get(f"https://api.openaq.org/v3/locations?{q}")
    best = None
    for r in d.get("results", []):
        df, dl = r.get("datetimeFirst"), r.get("datetimeLast")
        f = df.get("utc") if isinstance(df, dict) else df
        l = dl.get("utc") if isinstance(dl, dict) else dl
        if not f or not l:
            continue
        yrs = span_years(f, l)
        sid = next((s["id"] for s in r.get("sensors", []) if s["parameter"]["name"] == "pm25"), None)
        if sid and yrs >= MIN_YEARS and (best is None or yrs > best[0]):
            best = (yrs, sid, f[:7], l[:7], r["name"])
    return best


def main():
    rows = []
    for city, country, tier, lat, lon in CANDIDATES:
        try:
            b = best_sensor(lat, lon)
        except Exception as e:  # noqa: BLE001
            print(f"  {city:14s} ERROR {e}", flush=True)
            time.sleep(1)
            continue
        if b:
            yrs, sid, f, l, name = b
            rows.append({"city": city, "country": country, "tier": tier, "sensor_id": sid,
                         "first": f, "last": l, "years": round(yrs, 1), "station": name})
            print(f"  {city:14s} {tier:6s} sensor={sid} {f}->{l} ({yrs:.1f}yr)", flush=True)
        else:
            print(f"  {city:14s} {tier:6s} -- no >= {MIN_YEARS}yr pm25 sensor", flush=True)
        time.sleep(0.4)
    with open("cities_manifest.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["city", "country", "tier", "sensor_id",
                                           "first", "last", "years", "station"])
        w.writeheader()
        w.writerows(rows)
    rich = sum(1 for r in rows if r["tier"] == "rich")
    scarce = sum(1 for r in rows if r["tier"] == "scarce")
    print(f"\nMANIFEST: {len(rows)} cities ({rich} rich + {scarce} scarce) -> cities_manifest.csv")


if __name__ == "__main__":
    main()
