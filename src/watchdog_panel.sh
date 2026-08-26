#!/bin/bash
# Self-healing driver for the cross-city panel run. mode_cities is now checkpointed
# (appends each city's results immediately, skips completed cities on restart), so a kill
# only loses the current in-flight city, not the whole run.
# scripts live in src/ but read/write data cwd-relative to the repo root
cd "$(dirname "$0")/.."
target=29
out="pm25_final_panel_cities.csv"

while true; do
  done_n=0
  if [ -f "$out" ]; then
    done_n=$(python -c "import pandas as pd; print(pd.read_csv('$out').city.nunique())" 2>/dev/null || echo 0)
  fi
  echo "[panel-watchdog] cities_done=$done_n/$target"
  if [ "$done_n" -ge "$target" ]; then
    echo "[panel-watchdog] ALL DONE"
    break
  fi
  timeout 280 python -u src/run_forecast.py cities --data-dir data/cities_final \
    --manifest cities_manifest.csv --min-hours 2160 --folds 6 --horizon 24 \
    --with-chronos --with-nas --retrain-per-fold --out-prefix pm25_final_panel 2>&1 | tail -30
  sleep 2
done
