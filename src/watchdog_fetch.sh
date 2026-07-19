#!/bin/bash
# Self-healing fetch loop: keeps relaunching batch_fetch.py / openmeteo_fetch.py in small
# chunks until every city has both a PM2.5 and a weather file. Survives individual chunk
# deaths (this environment reaps long-running background python after ~5-8 min).
# scripts live in src/ but read/write data cwd-relative to the repo root
cd "$(dirname "$0")/.."
export OPENAQ_KEY="${OPENAQ_KEY:?set OPENAQ_KEY}"

target=$(tail -n +2 cities_manifest.csv | wc -l)

while true; do
  pm25=$(ls data/cities/*.csv 2>/dev/null | wc -l)
  wx=$(ls data/weather/*.csv 2>/dev/null | wc -l)
  echo "[watchdog] pm25=$pm25/$target weather=$wx/$target"
  if [ "$pm25" -ge "$target" ] && [ "$wx" -ge "$target" ]; then
    echo "[watchdog] ALL DONE"
    break
  fi
  if [ "$wx" -lt "$target" ]; then
    timeout 240 python -u src/openmeteo_fetch.py --start 2016-01-01 --end 2026-06-30 --max-new 4 2>&1 | tail -20
  fi
  if [ "$pm25" -lt "$target" ]; then
    # resumable now (per-month checkpoints) — a kill only loses the current month
    timeout 280 env MAX_NEW=1 python -u src/batch_fetch.py 2>&1 | tail -30
  fi
  sleep 2
done
