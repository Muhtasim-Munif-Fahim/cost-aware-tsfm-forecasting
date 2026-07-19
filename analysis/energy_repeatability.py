#!/usr/bin/env python3
"""S5 repeatability check -- codecarbon measured energy, fixed workload rerun 5x on 3 cities.

ANALYSIS_PLAN.md S5: "Repeatability: fixed workload rerun 5x on 3 cities -> report mean +/- sd;
if sd/mean > 20% for a tier, report as a range in the main table rather than a point estimate."

Fixed workload = the exact same `single` invocation (6 folds, h=24, all 5 tiers, nas_gru pinned
to seed 42 so the training workload is identical across reps). Cities: Beijing-Aotizhongxin
(primary depth), Seoul (rich), Nairobi (scarce).

Aggregates the 5 reps' `measured_j_per_1k` per (city, tier) into mean, sd, sd/mean and writes
results/v1/energy/repeatability_summary.csv, flagging tiers above the 20% pre-registered gate.

Usage (driver -- runs 15 subprocesses sequentially, ~1-1.5h):
  python analysis/energy_repeatability.py --run          # execute the 15 runs
  python analysis/energy_repeatability.py                # aggregate only (runs already done)
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER_PATH = os.path.join(ROOT, "paper", "RESULTS_LEDGER.md")
OUT_DIR = os.path.join(ROOT, "results", "v1", "energy")
REPS = 5
CITIES = {
    "beijing": ["--source", "pm25", "--data-path", "data/beijing_pm25/PRSA_Data_20130301-20170228",
                 "--column", "Aotizhongxin"],
    "seoul": ["--source", "csv", "--data-path", "data/cities/seoul.csv"],
    "nairobi": ["--source", "csv", "--data-path", "data/cities/nairobi.csv"],
}


def next_ledger_id() -> int:
    if not os.path.exists(LEDGER_PATH):
        return 1
    text = open(LEDGER_PATH, encoding="utf-8").read()
    ids = [int(m) for m in re.findall(r"L-(\d+)", text)]
    return max(ids, default=0) + 1


def append_ledger_stub(claim: str, value: str, artifact: str, command: str, code_tag: str):
    n = next_ledger_id()
    row = (f"| L-{n:03d} | {claim} | {value} | | {artifact} | `{command}` | | {code_tag} | "
           f"n/a | | Phase 3 script |\n")
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(row)


def run_all():
    os.makedirs(OUT_DIR, exist_ok=True)
    for city, src_args in CITIES.items():
        for rep in range(1, REPS + 1):
            prefix = os.path.join(OUT_DIR, f"rep_{city}_r{rep}")
            if os.path.exists(f"{prefix}_results.csv"):
                print(f"[skip] {city} rep{rep} (exists)", flush=True)
                continue
            cmd = [sys.executable, os.path.join("src", "run_forecast.py"),
                   "single", *src_args,
                   "--with-chronos", "--with-nas", "--seeds", "42",
                   "--folds", "6", "--measure-energy", "--out-prefix", prefix]
            print(f"[run] {city} rep{rep}", flush=True)
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"[FAIL] {city} rep{rep}: {r.stderr[-500:]}", flush=True)


def aggregate(code_tag=""):
    rows = []
    for city in CITIES:
        for rep in range(1, REPS + 1):
            f = os.path.join(OUT_DIR, f"rep_{city}_r{rep}_results.csv")
            if not os.path.exists(f):
                continue
            df = pd.read_csv(f)
            df["city"], df["rep"] = city, rep
            rows.append(df[["city", "rep", "model", "measured_j_per_1k", "measured_usd_per_1k"]])
    if not rows:
        raise SystemExit("no rep results found -- run with --run first")
    allr = pd.concat(rows)
    summ = (allr.groupby(["city", "model"]).measured_j_per_1k
            .agg(n_reps="count", mean_j_per_1k="mean", sd_j_per_1k="std")
            .reset_index())
    summ["sd_over_mean"] = summ.sd_j_per_1k / summ.mean_j_per_1k
    summ["exceeds_20pct_gate"] = summ.sd_over_mean > 0.20
    out = os.path.join(OUT_DIR, "repeatability_summary.csv")
    summ.to_csv(out, index=False)
    print(summ.to_string(index=False))
    n_flag = int(summ.exceeds_20pct_gate.sum())
    print(f"\ntiers exceeding the 20% sd/mean gate: {n_flag}/{len(summ)}")
    print(f"saved -> {out}")
    append_ledger_stub(
        claim="S5 energy repeatability: codecarbon measured_j_per_1k, fixed workload x5 reps "
              "on Beijing/Seoul/Nairobi, per tier mean +/- sd and 20% sd/mean gate",
        value=f"{n_flag}/{len(summ)} (city,tier) cells exceed the 20% gate; see CSV",
        artifact=out,
        command="python analysis/energy_repeatability.py --run && python analysis/energy_repeatability.py",
        code_tag=code_tag,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="execute the 15 measurement runs first")
    ap.add_argument("--code-tag", default="")
    a = ap.parse_args()
    if a.run:
        run_all()
    else:
        aggregate(a.code_tag)
