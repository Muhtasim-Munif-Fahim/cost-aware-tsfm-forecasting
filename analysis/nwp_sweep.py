#!/usr/bin/env python3
"""R1.2 -- covariate-quality sweep: how much of the specialist's perfect-foresight
advantage survives when its future weather comes from a real forecast?

Reviewer 1: "the use of the last observed meteorological value throughout the forecasting
horizon represents a rather pessimistic deployment scenario ... realistic performance is
likely to lie between the last-known-covariate setting and the perfect-foresight upper
bound ... the authors should consider adding an intermediate experiment based on
realistically forecast meteorological covariates."

Rather than add a single intermediate point, this sweeps the whole covariate-quality axis
and marks where the two existing scenarios and real NWP fall on it, so the reviewer can
see the entire interpolation rather than trusting one calibration choice:

    alpha = 0            perfect foresight        (the manuscript's canonical setting)
    alpha = 1            measured real-NWP error  (the requested intermediate point)
    alpha = alpha_persist  last-known / persistence (the manuscript's causal ablation)

alpha scales the AR(1) forecast error injected into the weather covariates by
`src/covariate_degradation.py`, calibrated per variable against the Open-Meteo
previous-model-runs archive by `analysis/nwp_covariate_error.py`.

Because the univariate FM tiers (`chronos`, `timesfm`) never see covariates, their MASE
must be identical at every alpha. The sweep runs them anyway and asserts that invariance
as a built-in leakage check: if a covariate-free tier moves with alpha, degradation has
leaked somewhere it should not have.

Stages:
  --stage model   build + report the error model (no runs; fast)
  --stage run     execute the panel at each alpha (long; checkpointed by the runner)
  --stage report  aggregate the sweep into results/v1/nwp_sweep_summary.csv

Usage:
  python analysis/nwp_sweep.py --stage model
  python analysis/nwp_sweep.py --stage run --domain pm25
  python analysis/nwp_sweep.py --stage report
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

ERR_CSV = os.path.join(ROOT, "results", "v1", "nwp_covariate_error.csv")
OUT_DIR = os.path.join(ROOT, "results", "v1")
ALPHAS = [0.0, 0.5, 1.0, 2.0, 4.0]
SEEDS = [42, 43, 44]          # replicate noise draws so a result is not one lucky sample

DOMAIN_ARGS = {
    "pm25": ["--source", "pm25", "--data-dir", "data/cities",
             "--manifest", "cities_manifest.csv"],
    "weather": ["--source", "weather_csv", "--data-dir", "data/weather",
                "--manifest", "cities_manifest.csv",
                "--pm25-window-dir", "data/cities_final"],
}


# --------------------------------------------------------------------------- #
# stage: model
# --------------------------------------------------------------------------- #
def build_model(err_csv=ERR_CSV, verbose=True):
    from covariate_degradation import load_params
    params = load_params(err_csv)
    raw = pd.read_csv(err_csv)
    raw = raw[raw.actual_sd > 0].copy()
    raw["err_frac"] = raw.rmse / raw.actual_sd

    rows = []
    for var, p in sorted(params.items()):
        g = raw[(raw.variable == var) & (raw.lead_day == 1)]
        rows.append({
            "variable": var,
            "sigma_frac_fitted_1_24h": round(p["sigma_frac"], 4),
            "measured_frac_lead1": round(p["measured_lead1_frac"], 4),
            "bias_frac": round(p["bias_frac"], 4),
            "err_ar1": round(p["ar1"], 4),
            "n_cities": int(g.city.nunique()),
            # between-city spread: if the error/signal ratio is stable across cities, the
            # transfer of these parameters to the 14 archive-uncovered cities is defensible
            "frac_sd_across_cities": round(float(g.err_frac.std()), 4) if len(g) > 1 else np.nan,
            "frac_min": round(float(g.err_frac.min()), 4) if len(g) else np.nan,
            "frac_max": round(float(g.err_frac.max()), 4) if len(g) else np.nan,
        })
    model = pd.DataFrame(rows)
    if verbose:
        pd.set_option("display.width", 200, "display.max_columns", 20)
        print("--- per-variable NWP error model ---")
        print(model.to_string(index=False))
        print("\nsigma_frac_fitted_1_24h is error/signal at OUR lead band (1-24 h), obtained by")
        print("regressing error on lead across previous_day1..3 and evaluating at 12 h.")
        print("measured_frac_lead1 is the raw lead-24-47 h measurement, which OVERSTATES")
        print("error for our horizon -- reported so the extrapolation stays auditable.")
        print("\nCoverage caveat: the archive serves different variables by region. Lagos,")
        print("Accra and Lima return only temperature_2m, so non-temperature variables are")
        print("calibrated on the remaining cities (mostly Europe/N.America/Asia).")
    return params, model


def persistence_alpha(params, domain="weather", verbose=True):
    """Locate the manuscript's last-known ablation on the same alpha axis.

    Computes, per variable, the error/signal ratio incurred by holding the covariate at the
    forecast origin for 24 h, then expresses it as a multiple of the fitted NWP error. That
    multiple is the alpha at which injected error matches persistence error, i.e. where the
    causal ablation sits relative to real NWP.
    """
    from covariate_degradation import persistence_error_frac
    wdir = os.path.join(ROOT, "data", "weather")
    files = sorted(f for f in os.listdir(wdir) if f.endswith(".csv"))
    per_city = []
    for fn in files:
        df = pd.read_csv(os.path.join(wdir, fn))
        tcol = df.columns[0]
        df[tcol] = pd.to_datetime(df[tcol])
        df = df.set_index(tcol)
        per_city.append(persistence_error_frac(df, horizon=24))
    if not per_city:
        return np.nan, pd.DataFrame()
    pf = pd.DataFrame(per_city).mean()

    rows = []
    for var, frac in pf.items():
        s = params.get(var, {}).get("sigma_frac", np.nan)
        rows.append({"variable": var, "persistence_frac": round(float(frac), 4),
                     "nwp_frac": round(float(s), 4) if np.isfinite(s) else np.nan,
                     "alpha_equiv": round(float(frac / s), 3) if (np.isfinite(s) and s > 0) else np.nan})
    tab = pd.DataFrame(rows).sort_values("variable")
    a = float(np.nanmedian(tab.alpha_equiv.values))
    if verbose:
        print("\n--- where does the manuscript's last-known ablation sit on this axis? ---")
        print(tab.to_string(index=False))
        print(f"\nmedian alpha_equiv = {a:.2f}  -> holding covariates at the forecast origin")
        print(f"costs roughly {a:.1f}x the error of a real 24 h NWP forecast.")
    return a, tab


# --------------------------------------------------------------------------- #
# stage: run
# --------------------------------------------------------------------------- #
def run_sweep(domain, alphas, seeds, extra_models=True, dry=False):
    base = [sys.executable, os.path.join(ROOT, "src", "run_forecast.py"), "cities"]
    base += DOMAIN_ARGS[domain]
    base += ["--retrain-per-fold", "--folds", "6", "--with-chronos"]
    if extra_models:
        base += ["--with-timesfm"]
    for a in alphas:
        # alpha = 0 is deterministic (exact no-op), so one seed suffices there
        for sd in ([seeds[0]] if a == 0 else seeds):
            tag = f"nwp_a{a:g}_s{sd}"
            out = os.path.join(OUT_DIR, f"{domain}_panel", f"sweep_{tag}")
            cmd = base + ["--covariate-noise", str(a), "--covariate-noise-seed", str(sd),
                          "--nwp-error-csv", ERR_CSV, "--out-prefix", out]
            print(f"[{domain}] alpha={a} seed={sd} -> {out}")
            if dry:
                print("   " + " ".join(cmd))
                continue
            r = subprocess.run(cmd, cwd=ROOT)
            if r.returncode != 0:
                print(f"   FAILED (exit {r.returncode}) -- continuing to next cell")


# --------------------------------------------------------------------------- #
# stage: report
# --------------------------------------------------------------------------- #
def report(domains=("pm25", "weather")):
    rows = []
    for dom in domains:
        d = os.path.join(OUT_DIR, f"{dom}_panel")
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not (fn.startswith("sweep_nwp_a") and fn.endswith("_cities.csv")):
                continue
            stem = fn[len("sweep_nwp_a"):-len("_cities.csv")]
            a_str, s_str = stem.split("_s")
            df = pd.read_csv(os.path.join(d, fn))
            n_cities_cell = df.city.nunique()
            for model, g in df.groupby("model"):
                rows.append({"domain": dom, "alpha": float(a_str), "seed": int(s_str),
                             "model": model, "n_cities": g.city.nunique(),
                             "mean_mase": float(g.groupby("city").MASE.mean().mean())})
    out = pd.DataFrame(rows)
    if out.empty:
        print("no sweep outputs found yet -- run --stage run first")
        return out

    # A run that dies part-way (the harness checkpoints per city, and a native crash in a
    # model tier will abort a cell mid-panel) leaves a SHORT cell behind. Averaging cells
    # built from different city sets silently compares different panels, so refuse to
    # report unless every cell covers the same cities.
    per_dom_max = out.groupby("domain").n_cities.transform("max")
    short = out[out.n_cities < per_dom_max]
    if not short.empty:
        bad = (short.groupby(["domain", "alpha", "seed"]).n_cities.first()
                    .reset_index().to_string(index=False))
        print("REFUSING TO REPORT -- these cells are incomplete relative to the "
              "fullest cell in their domain:")
        print(bad)
        print("Re-run the affected cells (the runner resumes from its checkpoint) or "
              "delete them. Comparing alphas across different city sets is invalid.")
        return out
    raw_path = os.path.join(OUT_DIR, "nwp_sweep_raw.csv")
    if os.path.exists(raw_path):
        prev_raw = pd.read_csv(raw_path)
        keep_raw = prev_raw[~prev_raw.domain.isin(out.domain.unique())]
        if len(keep_raw):
            out = pd.concat([out, keep_raw], ignore_index=True)
    out.to_csv(raw_path, index=False)

    summ = (out.groupby(["domain", "model", "alpha"])
               .agg(mean_mase=("mean_mase", "mean"), sd_across_seeds=("mean_mase", "std"),
                    n_seeds=("seed", "nunique"), n_cities=("n_cities", "max"))
               .reset_index())
    # Merge with any rows for domains this run did not cover. Writing the frame straight
    # out meant `--stage report --domain pm25` silently discarded the weather rows written
    # by the previous per-domain call, leaving a summary that looked complete but held one
    # domain.
    summ_path = os.path.join(OUT_DIR, "nwp_sweep_summary.csv")
    if os.path.exists(summ_path):
        prev = pd.read_csv(summ_path)
        keep = prev[~prev.domain.isin(summ.domain.unique())]
        if len(keep):
            summ = pd.concat([summ, keep], ignore_index=True)
    summ.to_csv(summ_path, index=False)

    pd.set_option("display.width", 200, "display.max_columns", 20)
    for dom in summ.domain.unique():
        print(f"\n--- {dom}: mean MASE vs covariate-error level ---")
        piv = summ[summ.domain == dom].pivot_table(index="model", columns="alpha",
                                                   values="mean_mase")
        print(piv.round(4).to_string())

    # leakage check: covariate-free tiers must not move with alpha
    print("\n--- invariance check (covariate-free tiers must be flat in alpha) ---")
    bad = []
    for (dom, model), g in summ.groupby(["domain", "model"]):
        if model not in ("chronos", "timesfm", "seasonal_naive"):
            continue
        spread = float(g.mean_mase.max() - g.mean_mase.min())
        flag = "OK" if spread < 1e-6 else "*** MOVED ***"
        if spread >= 1e-6:
            bad.append((dom, model, spread))
        print(f"  {dom:<8} {model:<15} range across alpha = {spread:.2e}  {flag}")
    if bad:
        print("\nWARNING: a covariate-free tier changed with alpha; degradation has leaked")
        print("into a path it should not touch. Investigate before using these numbers.")
    print(f"\nsaved -> {os.path.join(OUT_DIR, 'nwp_sweep_summary.csv')}")
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["model", "run", "report"])
    ap.add_argument("--domain", default="both", choices=["pm25", "weather", "both"])
    ap.add_argument("--alphas", default=None, help="comma-separated override, e.g. 0,1,2")
    ap.add_argument("--seeds", default=None, help="comma-separated override")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    doms = ["pm25", "weather"] if args.domain == "both" else [args.domain]
    alphas = [float(x) for x in args.alphas.split(",")] if args.alphas else ALPHAS
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else SEEDS

    if args.stage == "model":
        params, _ = build_model()
        persistence_alpha(params)
    elif args.stage == "run":
        for dom in doms:
            run_sweep(dom, alphas, seeds, dry=args.dry_run)
    else:
        report(tuple(doms))


if __name__ == "__main__":
    main()
