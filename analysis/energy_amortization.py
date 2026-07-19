"""Reviewer-requested energy decomposition (added 2026-07-15, logged in deviations).

The headline "~10x less energy" compares the specialist's train+infer energy (it retrains
per fold) against the foundation model's inference-only energy. That is honest for a
frequent-retraining deployment but not for a train-once one. Here we separate one-time
training energy from per-forecast inference energy using only whole-call codecarbon
measurements (clean; no unit mixing), and compute the amortization crossover.

Method (per city, GPU and CPU-only):
  E_retrain = energy( lgbm, retrain_per_fold=True  )  = 6*train + 6*infer
  E_once    = energy( lgbm, retrain_per_fold=False )  = 1*train + 6*infer
  => train_per_fit = (E_retrain - E_once) / 5
     infer6_lgbm   = E_once - train_per_fit
     infer_fc_lgbm = infer6_lgbm / (6*24)
  E_chronos = energy( chronos )                       = 6*infer   (no training)
     infer_fc_chr = E_chronos / (6*24)
  crossover N (forecasts served by ONE trained specialist before its amortized energy
  per forecast falls to the FM's inference energy):
     N = train_per_fit / (infer_fc_chr - infer_fc_lgbm)   if infer_fc_chr > infer_fc_lgbm
     else "no crossover" (FM cheaper even at inference; specialist never catches up)

Usage:
  python analysis/energy_amortization.py --run   # measures (slow; run alone, no other load)
  python analysis/energy_amortization.py         # summarise existing measurements
Output: results/v1/energy/amortization.csv  (+ _summary.csv)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUTDIR = os.path.join(ROOT, "results", "v1", "energy")
RAW = os.path.join(OUTDIR, "amortization.csv")
SUMMARY = os.path.join(OUTDIR, "amortization_summary.csv")

CITIES = {  # city -> (source, data_path)
    "beijing": ("pm25", "data/beijing_pm25/PRSA_Data_20130301-20170228"),
    "seoul": ("csv", "data/cities/seoul.csv"),
    "nairobi": ("csv", "data/cities/nairobi.csv"),
}
REPS = 3
H = 24
FOLDS = 6


def _measure(fn):
    import run_forecast as rf
    (_res), measured = rf._EnergyMeter.measure(fn)
    return measured["kwh"] * 3.6e6  # joules


def run(force_cpu):
    import types
    import run_forecast as rf
    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    rows = []
    for city, (source, path) in CITIES.items():
        a = types.SimpleNamespace(source=source, data_path=path, column=None,
                                  min_hours=2160, pm25_window_dir="data/cities_final")
        if source == "pm25":
            a.column = "Aotizhongxin"
        y, exog = rf.load_single(a)
        folds = rf.make_folds(y, H, FOLDS)
        for rep in range(REPS):
            e_retrain = _measure(lambda: rf.run_lgbm_direct(y, exog, folds, H, True, False))
            e_once = _measure(lambda: rf.run_lgbm_direct(y, exog, folds, H, False, False))
            e_chr = _measure(lambda: rf.run_chronos(y, exog, folds, H))
            rows.append(dict(city=city, rep=rep, device="cpu" if force_cpu else "gpu",
                             e_retrain_j=e_retrain, e_once_j=e_once, e_chronos_j=e_chr))
            print(f"  {city} rep{rep} {'cpu' if force_cpu else 'gpu'}: "
                  f"retrain={e_retrain:.1f} once={e_once:.1f} chr={e_chr:.1f} J", flush=True)
    return rows


def summarise():
    df = pd.read_csv(RAW)
    out = []
    for (city, device), g in df.groupby(["city", "device"]):
        e_retrain, e_once, e_chr = g.e_retrain_j.mean(), g.e_once_j.mean(), g.e_chronos_j.mean()
        train_per_fit = (e_retrain - e_once) / 5.0
        infer6_lgbm = e_once - train_per_fit
        infer_fc_lgbm = infer6_lgbm / (FOLDS * H)
        infer_fc_chr = e_chr / (FOLDS * H)
        gap = infer_fc_chr - infer_fc_lgbm
        crossover = (train_per_fit / gap) if gap > 0 else np.inf
        out.append(dict(city=city, device=device,
                        train_per_fit_j=round(train_per_fit, 1),
                        infer_fc_lgbm_j=round(infer_fc_lgbm, 4),
                        infer_fc_chronos_j=round(infer_fc_chr, 4),
                        lgbm_infer_cheaper=bool(gap > 0),
                        crossover_forecasts=(round(crossover) if np.isfinite(crossover) else "none")))
    s = pd.DataFrame(out)
    s.to_csv(SUMMARY, index=False)
    print(s.to_string(index=False))
    print(f"\nwritten -> {os.path.relpath(SUMMARY, ROOT)}")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="take measurements (else summarise)")
    ap.add_argument("--cpu", action="store_true", help="force CPU-only")
    ap.add_argument("--both", action="store_true", help="measure GPU then CPU")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    if args.run:
        rows = []
        if args.both:
            rows += run(force_cpu=False)
            rows += run(force_cpu=True)
        else:
            rows += run(force_cpu=args.cpu)
        pd.DataFrame(rows).to_csv(RAW, index=False)
        print(f"raw -> {os.path.relpath(RAW, ROOT)}")
    summarise()


if __name__ == "__main__":
    main()
