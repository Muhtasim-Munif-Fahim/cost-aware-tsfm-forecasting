"""Manuscript tables — supplementary family FIRST, then main T1/T2 derived from it.

Outputs LaTeX bodies (booktabs) + markdown previews under tables/out/.
Every number is read from canonical CSVs; nothing hand-typed.

Supplementary:
  S1  per-city panel (29 rows: tier, sensor window, usable hours)
  S2  model hyperparameters per tier (mirrors run_forecast.py configuration)
  S3a/S3b pooled split-conformal coverage/width per domain (causal + perfect)
  S4a/S4b full DM pair matrices per domain (raw + FDR win counts)
  S5/S6 per-city split-conformal coverage/width (causal-primary), per domain
  S7  energy repeatability full (15 city x tier cells)
  S9a/S9b horizon-48 panels
  S10 E4 full grid (strategy x fraction means with actual hours)
  S12 cost sensitivity grid (causal decision maps)
Main:
  T1  panel MASE mean+/-sd per tier x domain + FDR sig wins vs chronos
  T2  measured energy/cost per tier (3 cities), ranges where sd/mean>20%
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "figures"))

from formatters import latex_escape  # noqa: E402
from sanity_checks import check_min_rows  # noqa: E402
from naming import TIER_ORDER, DISPLAY  # noqa: E402

OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

DOM_FILES = {
    "pm25": "results/v1/pm25_panel/canonical_cities.csv",
    "weather": "results/v1/weather_panel/canonical_cities.csv",
}
DM_FILES = {
    "pm25": "results/v1/pm25_panel/canonical_dm_panel_summary.csv",
    "weather": "results/v1/weather_panel/canonical_dm_panel_summary.csv",
}
DOM_LABEL = {"pm25": "PM$_{2.5}$", "weather": "Temperature"}


def write(name: str, latex_body: str, md: str):
    with open(os.path.join(OUT, f"{name}.tex"), "w", encoding="utf-8") as f:
        f.write(latex_body)
    with open(os.path.join(OUT, f"{name}.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(f"wrote tables/out/{name}.tex + .md")


def df_to_latex_md(df: pd.DataFrame, name: str, escape_cols=None):
    esc = df.copy()
    if escape_cols:
        for c in escape_cols:
            esc[c] = esc[c].map(latex_escape)
    body = esc.to_latex(index=False, escape=False, na_rep="—",
                        column_format="l" + "r" * (len(df.columns) - 1))
    write(name, body, df.to_markdown(index=False))


# --------------------------------------------------------------------------
# S1 — per-city panel (FIRST, per skill rule 3)
# --------------------------------------------------------------------------
def s1_city_panel():
    qual = pd.read_csv(os.path.join(ROOT, "cities_quality.csv"))
    qual = qual[qual.PASS == True].copy()  # noqa: E712
    man = pd.read_csv(os.path.join(ROOT, "cities_manifest.csv"))
    man["slug"] = man.city.str.lower().str.replace(" ", "_")
    qual = qual.merge(man[["slug", "country", "sensor_id"]],
                      left_on="city", right_on="slug", how="left")
    s1 = qual[["city", "country", "tier", "sensor_id", "usable_from",
               "usable_to", "usable_hours"]].sort_values(["tier", "city"])
    s1["city"] = s1.city.str.replace("_", " ").str.title()
    check_min_rows(len(s1), 29, "S1 per-city panel")
    df_to_latex_md(s1.rename(columns={
        "city": "City", "country": "Country", "tier": "Tier",
        "sensor_id": "OpenAQ sensor", "usable_from": "Window start",
        "usable_to": "Window end", "usable_hours": "Usable hours"}),
        "S1_city_panel", escape_cols=["City"])
    return s1


# --------------------------------------------------------------------------
# S2 — model hyperparameters (mirrors run_forecast.py; keep in sync with code)
# --------------------------------------------------------------------------
def s2_hyperparams():
    rows = [
        ("Seasonal-naïve", "persistence lag", "168 h (same hour last week)"),
        ("LightGBM (specialist)", "models", "one direct model per lead time (24), retrained per fold"),
        ("LightGBM (specialist)", "n\\_estimators / learning\\_rate / num\\_leaves", "250 / 0.05 / 31"),
        ("LightGBM (specialist)", "subsample / subsample\\_freq", "0.8 / 1 (freq added after pre-campaign audit; bagging inert without it)"),
        ("LightGBM (specialist)", "random\\_state", "42"),
        ("LightGBM (specialist)", "features", "lagged target, calendar, meteorological covariates"),
        ("NAS-GRU", "architecture", "Green-NAS-A: 2 stacked GRU layers, 128 units, direct multi-horizon head"),
        ("NAS-GRU", "optimizer / loss", "Adam, learning rate $10^{-3}$, MSE"),
        ("NAS-GRU", "early stopping", "patience 10, max 50 epochs, validation fraction 0.1"),
        ("NAS-GRU", "lookback / batch size", "24 h / 256"),
        ("NAS-GRU", "seeds", "\\{42, 43, 44, 45, 46\\}; per-city seed-mean before any test"),
        ("Chronos-Bolt (zero-shot)", "checkpoint", "amazon/chronos-bolt-small (47{,}718{,}016 parameters), no training or fine-tuning"),
        ("Chronos-Bolt (zero-shot)", "context / point forecast", "672 h (four weeks) / predicted mean"),
        ("Chronos-Bolt + covariates", "covariate model", "ridge regression ($\\alpha = 1.0$) on standardized calendar + weather features"),
        ("Chronos-Bolt + covariates", "scheme", "FM forecasts the residual of the ridge fit; ridge horizon prediction added back"),
    ]
    out = pd.DataFrame(rows, columns=["Tier", "Setting", "Value"])
    body = out.to_latex(index=False, escape=False, na_rep="—",
                        column_format="llp{0.55\\textwidth}")
    write("S2_hyperparams", body, out.to_markdown(index=False))


# --------------------------------------------------------------------------
# S3 — pooled split-conformal coverage/width, causal + perfect, per domain
# --------------------------------------------------------------------------
CONF_POOLED = {
    "pm25": ("results/v1/pm25_panel/causal_primary_conformal_pooled.csv",
             "results/v1/pm25_panel/canonical_conformal_pooled.csv"),
    "weather": ("results/v1/weather_panel/causal_primary_conformal_pooled.csv",
                "results/v1/weather_panel/canonical_conformal_pooled.csv"),
}


def s3_conformal_pooled():
    for dom, (causal_p, perfect_p) in CONF_POOLED.items():
        ca = pd.read_csv(os.path.join(ROOT, causal_p))
        pf = pd.read_csv(os.path.join(ROOT, perfect_p))
        m = ca.merge(pf, on=["tier", "model"], suffixes=("_causal", "_perfect"))
        m = m.set_index("model").loc[[t for t in TIER_ORDER if t in set(m.model)]].reset_index()
        m = m.sort_values(["tier"], kind="stable")
        out = pd.DataFrame({
            "Data tier": m.tier,
            "Model": m.model.map(lambda x: DISPLAY.get(x, x)),
            "Coverage (causal)": m.mean_coverage_causal.map(lambda v: f"{v:.3f}"),
            "Width (causal)": m.mean_width_causal.map(lambda v: f"{v:.2f}"),
            "Coverage (perfect)": m.mean_coverage_perfect.map(lambda v: f"{v:.3f}"),
            "Width (perfect)": m.mean_width_perfect.map(lambda v: f"{v:.2f}"),
        })
        df_to_latex_md(out, f"S3_conformal_{dom}", escape_cols=["Model"])


# --------------------------------------------------------------------------
# S5/S6 — per-city split-conformal (causal-primary), one table per domain
# --------------------------------------------------------------------------
CONF_PERCITY = {
    "pm25": ("results/v1/pm25_panel/causal_primary_conformal_percity.csv", "S5_conformal_percity_pm25"),
    "weather": ("results/v1/weather_panel/causal_primary_conformal_percity.csv", "S6_conformal_percity_weather"),
}


SHORT = {"chronos": "Chronos", "chronos_cov": "Chronos+cov", "lgbm_direct": "LightGBM",
         "nas_gru": "NAS-GRU", "seasonal_naive": "Naïve"}


def s5s6_conformal_percity():
    for dom, (path, name) in CONF_PERCITY.items():
        df = pd.read_csv(os.path.join(ROOT, path))
        check_min_rows(len(df), 145, f"{name} per-city conformal")
        cov = df.pivot(index=["city", "tier"], columns="model", values="coverage")
        wid = df.pivot(index=["city", "tier"], columns="model", values="width")
        models = [m for m in TIER_ORDER if m in cov.columns]
        out = pd.DataFrame(index=cov.index)
        for m in models:
            out[f"{SHORT[m]} cov."] = cov[m].map(lambda v: f"{v:.2f}")
            out[f"{SHORT[m]} width"] = wid[m].map(lambda v: f"{v:.1f}")
        out = out.reset_index()
        out["city"] = out.city.str.replace("_", " ").str.title()
        out = out.rename(columns={"city": "City", "tier": "Tier"})
        df_to_latex_md(out, name, escape_cols=["City"])


# --------------------------------------------------------------------------
# S4 — full DM matrices (both domains)
# --------------------------------------------------------------------------
def s4_dm():
    for dom, path in DM_FILES.items():
        dm = pd.read_csv(os.path.join(ROOT, path))
        dm["pair"] = (dm.model_a.map(lambda m: DISPLAY.get(m, m)) + " vs " +
                      dm.model_b.map(lambda m: DISPLAY.get(m, m)))
        out = dm[["pair", "n_cities", "n_significant", "a_sig_wins", "b_sig_wins",
                  "a_sig_wins_fdr", "b_sig_wins_fdr", "median_p_value"]].copy()
        out["median_p_value"] = out.median_p_value.map(lambda p: f"{p:.3f}")
        df_to_latex_md(out.rename(columns={
            "pair": "Model pair", "n_cities": "Cities",
            "n_significant": "Sig. (raw)", "a_sig_wins": "A wins (raw)",
            "b_sig_wins": "B wins (raw)", "a_sig_wins_fdr": "A wins (FDR)",
            "b_sig_wins_fdr": "B wins (FDR)", "median_p_value": "Median $P$"}),
            f"S4_dm_{dom}", escape_cols=["Model pair"])


# --------------------------------------------------------------------------
# S7 — energy repeatability full
# --------------------------------------------------------------------------
def s7_energy():
    rep = pd.read_csv(os.path.join(ROOT, "results/v1/energy/repeatability_summary.csv"))
    check_min_rows(len(rep), 15, "S7 energy repeatability")
    out = rep.copy()
    out["model"] = out.model.map(lambda m: DISPLAY.get(m, m))
    out["mean_j_per_1k"] = out.mean_j_per_1k.map(lambda v: f"{v:,.0f}")
    out["sd_j_per_1k"] = out.sd_j_per_1k.map(lambda v: f"{v:,.1f}")
    out["sd_over_mean"] = out.sd_over_mean.map(lambda v: f"{v:.3f}")
    out["city"] = out.city.str.title()
    df_to_latex_md(out.rename(columns={
        "city": "City", "model": "Tier", "n_reps": "Reps",
        "mean_j_per_1k": "Mean (J/1k)", "sd_j_per_1k": "SD (J/1k)",
        "sd_over_mean": "SD/mean", "exceeds_20pct_gate": "$>$20\\% gate"}),
        "S7_energy_repeatability", escape_cols=["Tier"])
    return rep


# --------------------------------------------------------------------------
# S9 — horizon-48 panels
# --------------------------------------------------------------------------
def s9_h48():
    for dom, path in (("pm25", "results/v1/horizon48/pm25_h48_cities.csv"),
                      ("weather", "results/v1/horizon48/weather_h48_cities.csv")):
        df = pd.read_csv(os.path.join(ROOT, path))
        piv = (df.groupby(["tier", "model"]).MASE.agg(["mean", "std"]).reset_index())
        piv["MASE (mean ± sd)"] = piv.apply(
            lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1)
        wide = piv.pivot(index="model", columns="tier",
                         values="MASE (mean ± sd)").reindex(TIER_ORDER)
        wide.insert(0, "Tier", [DISPLAY[m] for m in wide.index])
        df_to_latex_md(wide.reset_index(drop=True), f"S9_h48_{dom}",
                       escape_cols=["Tier"])


# --------------------------------------------------------------------------
# S10 — E4 full grid
# --------------------------------------------------------------------------
def s10_e4():
    df = pd.read_csv(os.path.join(ROOT, "results/v1/e4_transfer/canonical_pm25_results.csv"))
    g = (df.groupby(["strategy", "fraction"])
         .agg(mean_MASE=("MASE", "mean"), sd_MASE=("MASE", "std"),
              mean_hours=("n_train_hours", "mean"), n_cities=("city", "nunique"))
         .reset_index())
    ch = df[df.strategy == "chronos_zeroshot"]
    g0 = pd.DataFrame([{"strategy": "chronos_zeroshot", "fraction": np.nan,
                        "mean_MASE": ch.MASE.mean(), "sd_MASE": ch.MASE.std(),
                        "mean_hours": 0, "n_cities": ch.city.nunique()}])
    g = pd.concat([g0, g[g.strategy != "chronos_zeroshot"]], ignore_index=True)
    g["MASE (mean ± sd)"] = g.apply(lambda r: f"{r.mean_MASE:.3f} ± {r.sd_MASE:.3f}", axis=1)
    g["mean_hours"] = g.mean_hours.map(lambda v: f"{v:,.0f}")
    E4_DISPLAY = {"chronos_zeroshot": "Chronos-Bolt (zero-shot)",
                  "nas_transfer": "NAS-GRU (transfer + fine-tune)",
                  "lgbm_refit": "LightGBM (refit on budget)"}
    g["strategy"] = g.strategy.map(lambda s: E4_DISPLAY.get(s, s))
    g["fraction"] = g.fraction.map(lambda f: "—" if pd.isna(f) else f"{f:g}")
    out = g[["strategy", "fraction", "MASE (mean ± sd)", "mean_hours", "n_cities"]]
    df_to_latex_md(out.rename(columns={
        "strategy": "Strategy", "fraction": "Nominal fraction (\\%)",
        "mean_hours": "Actual training hours (mean)", "n_cities": "Cities"}),
        "S10_e4_grid", escape_cols=["Strategy"])


# --------------------------------------------------------------------------
# S12 — cost sensitivity
# --------------------------------------------------------------------------
def s12_sensitivity():
    # Causal decision maps are the main Fig. 6; the sensitivity grid must be
    # computed on the same configuration (supervisor review B5).
    df = pd.read_csv(os.path.join(ROOT, "results/v1/regime/s12_cost_sensitivity_causal.csv"))
    df = df.copy()
    df["run"] = (df.run.str.replace("causal_", "", regex=False)
                 .str.replace("_", " ").str.title()
                 .str.replace("Pm25", "PM2.5").str.replace("Weather", "Temperature"))
    df["price_kwh"] = df.price_kwh.map(lambda v: f"{v:.2f}")
    df["pue"] = df.pue.map(lambda v: f"{v:.1f}")
    df["effective_wtp_multiplier"] = df.effective_wtp_multiplier.map(lambda v: f"{v:.3f}")
    df["flip_rate"] = df.flip_rate.map(lambda v: f"{v:.2f}")
    df_to_latex_md(df.rename(columns={
        "run": "Regime run", "price_kwh": "Price (\\$/kWh)", "pue": "PUE",
        "effective_wtp_multiplier": "Effective $\\lambda$ multiplier", "cells": "Cells",
        "cells_flipped": "Flipped", "flip_rate": "Flip rate"}),
        "S12_cost_sensitivity", escape_cols=["Regime run"])


# --------------------------------------------------------------------------
# T1 — main: panel MASE + FDR DM wins vs chronos
# --------------------------------------------------------------------------
def s13_amortization():
    """S13 — energy train/inference decomposition + amortization crossover."""
    df = pd.read_csv(os.path.join(ROOT, "results/v1/energy/amortization_summary.csv"))
    out = pd.DataFrame({
        "City": df.city.str.capitalize(),
        "Device": df.device.str.upper(),
        "Train energy / fit (J)": df.train_per_fit_j.map(lambda v: f"{v:,.0f}"),
        "LightGBM infer (J/forecast)": df.infer_fc_lgbm_j.map(lambda v: f"{v:.3f}"),
        "Chronos infer (J/forecast)": df.infer_fc_chronos_j.map(lambda v: f"{v:.3f}"),
        "Crossover (forecasts)": df.crossover_forecasts.map(
            lambda v: f"{int(v):,}" if str(v).replace('.', '').isdigit() else str(v)),
    })
    df_to_latex_md(out, "S13_amortization", escape_cols=["City", "Device"])


def s16_equivalence_margins():
    """S16 — TOST equivalence verdict across a margin sweep (reviewer #11)."""
    df = pd.read_csv(os.path.join(ROOT, "results/v1/equivalence_tests.csv"))
    df["min_margin"] = df[["tost90_lo", "tost90_hi"]].abs().max(axis=1)
    LAB = {
        "pm25-perfect: lgbm_direct vs chronos (per city)": "PM2.5, perfect-foresight (specialist vs FM)",
        "pm25-causal: lgbm_direct vs chronos (per city)": "PM2.5, causal (specialist vs FM)",
        "weather-causal: lgbm_direct vs chronos (per city)": "Temperature, causal (specialist vs FM)",
        "E4 pm25: nas_transfer@0% vs chronos_zeroshot": "E4 transfer 0\\% vs zero-shot",
        "E4 pm25: nas_transfer@1% vs chronos_zeroshot": "E4 transfer 1\\% vs zero-shot",
        "E4 pm25: nas_transfer@10% vs chronos_zeroshot": "E4 transfer 10\\% vs zero-shot",
        "E4 pm25: nas_transfer@100% vs chronos_zeroshot": "E4 transfer 100\\% vs zero-shot",
    }
    df = df.set_index("comparison").loc[list(LAB)].reset_index()
    mark = lambda ok: "\\checkmark" if ok else "--"
    out = pd.DataFrame({
        "Comparison": [LAB[c] for c in df.comparison],
        "Point diff": df.mean_diff.map(lambda v: f"{v:+.3f}"),
        "TOST 90\\% CI": df.apply(lambda r: f"[{r.tost90_lo:+.3f}, {r.tost90_hi:+.3f}]", axis=1),
        "Min. margin": df.min_margin.map(lambda v: f"{v:.3f}"),
        "$\\delta$=0.02": [mark(m <= 0.02) for m in df.min_margin],
        "$\\delta$=0.05": [mark(m <= 0.05) for m in df.min_margin],
        "$\\delta$=0.10": [mark(m <= 0.10) for m in df.min_margin],
    })
    df_to_latex_md(out, "S16_equivalence_margins", escape_cols=["Comparison"])


def s15_contamination():
    """S15 — post-cutoff (unseen-data) check on the PM2.5 specialist-vs-FM comparison."""
    df = pd.read_csv(os.path.join(ROOT, "results/v1/contamination_postcutoff.csv"))
    out = pd.DataFrame({
        "City": df.city.str.replace("_", " ").str.title(),
        "Post-cutoff hours": df.post_hours.map(lambda v: f"{int(v):,}"),
        "Chronos MASE": df.MASE_chronos.map(lambda v: f"{v:.3f}"),
        "LightGBM MASE": df.MASE_lgbm_direct.map(lambda v: f"{v:.3f}"),
        "Naive MASE": df.MASE_seasonal_naive.map(lambda v: f"{v:.3f}"),
    })
    df_to_latex_md(out, "S15_contamination", escape_cols=["City"])


def t1_main(dom_files=DOM_FILES, dm_files=DM_FILES, nem_files=None, name="T1_panel"):
    """T1 full-metrics: rows = tier x domain, columns = MASE/MAE/RMSE (mean +/- sd
    across cities, per-city seed-means first), Friedman-Nemenyi average rank, and
    FDR-corrected DM significant-win counts vs zero-shot chronos."""
    DOM_SHORT = {"pm25": "PM2.5", "weather": "Temp."}
    rows = []
    for dom, path in dom_files.items():
        df = pd.read_csv(os.path.join(ROOT, path))
        dm = pd.read_csv(os.path.join(ROOT, dm_files[dom]))
        nem = (pd.read_csv(os.path.join(ROOT, nem_files[dom])).set_index("model")
               if nem_files else None)
        for m in TIER_ORDER:
            sub = df[df.model == m]
            per_city = sub.groupby("city")[["MASE", "MAE", "RMSE"]].mean()
            def ms(col):
                return f"{per_city[col].mean():.3f} ± {per_city[col].std(ddof=1):.3f}"
            rank = (f"{nem.loc[m, 'avg_rank']:.2f}"
                    if nem is not None and m in nem.index else "—")
            if m == "chronos":
                wins = "—"
            else:
                pair = dm[((dm.model_a == m) & (dm.model_b == "chronos")) |
                          ((dm.model_a == "chronos") & (dm.model_b == m))]
                if len(pair) == 1:
                    p = pair.iloc[0]
                    m_wins = p.a_sig_wins_fdr if p.model_a == m else p.b_sig_wins_fdr
                    c_wins = p.b_sig_wins_fdr if p.model_a == m else p.a_sig_wins_fdr
                    wins = f"{int(m_wins)} / {int(c_wins)}"
                else:
                    wins = "—"
            rows.append({"Tier": DISPLAY[m], "Domain": DOM_SHORT[dom],
                         "MASE (mean ± sd)": ms("MASE"),
                         "MAE (mean ± sd)": ms("MAE"),
                         "RMSE (mean ± sd)": ms("RMSE"),
                         "Avg rank": rank,
                         "FDR wins (tier/FM)": wins})
    # domain-major blocks (PM2.5 tiers, then temperature tiers), tier order within
    out = pd.DataFrame(rows).sort_values(
        ["Domain", "Tier"],
        key=lambda s: (s.map({"PM2.5": 0, "Temp.": 1}) if s.name == "Domain"
                       else s.map({DISPLAY[m]: i for i, m in enumerate(TIER_ORDER)})),
        kind="stable").reset_index(drop=True)
    check_min_rows(len(out), 2 * len(TIER_ORDER), f"{name} tier x domain")
    df_to_latex_md(out, name, escape_cols=["Tier"])


# Causal-primary (deployable) and perfect-foresight (upper bound) panel file sets.
CAUSAL_DOM = {
    "pm25": "results/v1/pm25_panel/causal_primary_cities.csv",
    "weather": "results/v1/weather_panel/causal_primary_cities.csv",
}
CAUSAL_DM = {
    "pm25": "results/v1/pm25_panel/causal_primary_dm_panel_summary.csv",
    "weather": "results/v1/weather_panel/causal_primary_dm_panel_summary.csv",
}
CAUSAL_NEM = {
    "pm25": "results/v1/pm25_panel/causal_primary_nemenyi_ranks.csv",
    "weather": "results/v1/weather_panel/causal_primary_nemenyi_ranks.csv",
}
CANON_NEM = {
    "pm25": "results/v1/pm25_panel/canonical_nemenyi_ranks.csv",
    "weather": "results/v1/weather_panel/canonical_nemenyi_ranks.csv",
}


def t1_combined(name="T1_panel"):
    """Main T1 — combined accuracy table: 2 domains x 2 covariate configs x 5
    tiers (20 rows). Same per-cell logic as t1_main; blocks assembled with
    \\multirow so a single table carries the deployable result and its
    perfect-foresight upper bound."""
    DOM_SHORT = {"pm25": "PM$_{2.5}$", "weather": "Temperature"}
    CONFIGS = [
        ("Causal (deployable)", CAUSAL_DOM, CAUSAL_DM, CAUSAL_NEM),
        ("Perfect foresight", DOM_FILES, DM_FILES, CANON_NEM),
    ]
    lines = [
        "\\begin{tabular}{lllrrrrr}",
        "\\toprule",
        "Domain & Covariates & Tier & MASE & MAE & RMSE & "
        "Avg.\\ rank & FDR wins \\\\",
        " & & & (mean $\\pm$ sd) & (mean $\\pm$ sd) & (mean $\\pm$ sd) & "
        " & (tier / FM) \\\\",
        "\\midrule",
    ]
    md_rows = []
    for d_i, dom in enumerate(("pm25", "weather")):
        if d_i:
            lines.append("\\midrule")
        for c_i, (cfg_label, dom_files, dm_files, nem_files) in enumerate(CONFIGS):
            if c_i:
                lines.append("\\cmidrule(lr){2-8}")
            df = pd.read_csv(os.path.join(ROOT, dom_files[dom]))
            dm = pd.read_csv(os.path.join(ROOT, dm_files[dom]))
            nem = pd.read_csv(os.path.join(ROOT, nem_files[dom])).set_index("model")
            for t_i, m in enumerate(TIER_ORDER):
                sub = df[df.model == m]
                per_city = sub.groupby("city")[["MASE", "MAE", "RMSE"]].mean()
                def ms(col):
                    return (f"{per_city[col].mean():.3f} $\\pm$ "
                            f"{per_city[col].std(ddof=1):.3f}")
                rank = f"{nem.loc[m, 'avg_rank']:.2f}" if m in nem.index else "—"
                if m == "chronos":
                    wins = "—"
                else:
                    pair = dm[((dm.model_a == m) & (dm.model_b == "chronos")) |
                              ((dm.model_a == "chronos") & (dm.model_b == m))]
                    if len(pair) == 1:
                        p = pair.iloc[0]
                        m_wins = p.a_sig_wins_fdr if p.model_a == m else p.b_sig_wins_fdr
                        c_wins = p.b_sig_wins_fdr if p.model_a == m else p.a_sig_wins_fdr
                        wins = f"{int(m_wins)} / {int(c_wins)}"
                    else:
                        wins = "—"
                dom_cell = (f"\\multirow{{10}}{{*}}{{{DOM_SHORT[dom]}}}"
                            if (c_i == 0 and t_i == 0) else "")
                cfg_cell = (f"\\multirow{{5}}{{*}}{{{cfg_label}}}"
                            if t_i == 0 else "")
                lines.append(
                    f"{dom_cell} & {cfg_cell} & {DISPLAY[m]} & {ms('MASE')} & "
                    f"{ms('MAE')} & {ms('RMSE')} & {rank} & {wins} \\\\")
                md_rows.append({
                    "Domain": DOM_SHORT[dom].replace("$_{2.5}$", "2.5"),
                    "Covariates": cfg_label, "Tier": DISPLAY[m],
                    "MASE": ms("MASE").replace(" $\\pm$ ", " ± "),
                    "MAE": ms("MAE").replace(" $\\pm$ ", " ± "),
                    "RMSE": ms("RMSE").replace(" $\\pm$ ", " ± "),
                    "Avg rank": rank, "FDR wins (tier/FM)": wins})
    lines += ["\\bottomrule", "\\end{tabular}"]
    check_min_rows(len(md_rows), 20, f"{name} combined")
    write(name, "\n".join(lines) + "\n",
          pd.DataFrame(md_rows).to_markdown(index=False))


# --------------------------------------------------------------------------
# T2 (main, displayed as Table 2) — E4 transfer-vs-zero-shot long table
# --------------------------------------------------------------------------
def t_e4(name="T3_e4"):
    """E4 grid as a main-text table (replaces the old Fig. 5 line chart):
    zero-shot reference row + every comparator x budget row with per-city
    wins, Holm-corrected Wilcoxon P, and the TOST 90% CI where computed."""
    res = pd.read_csv(os.path.join(ROOT, "results/v1/e4_transfer/canonical_pm25_results.csv"))
    sig = pd.read_csv(os.path.join(ROOT, "results/v1/e4_transfer/canonical_pm25_significance.csv"))
    eq = pd.read_csv(os.path.join(ROOT, "results/v1/equivalence_tests.csv"))
    eq_by_frac = {}
    for f in (0, 1, 10, 100):
        row = eq[eq.comparison == f"E4 pm25: nas_transfer@{f}% vs chronos_zeroshot"]
        if len(row) == 1:
            r = row.iloc[0]
            eq_by_frac[float(f)] = f"[{r.tost90_lo:+.3f}, {r.tost90_hi:+.3f}]"

    def city_ms(strategy, fraction=None):
        sub = res[res.strategy == strategy]
        if fraction is not None:
            sub = sub[sub.fraction == fraction]
        per_city = sub.groupby("city").MASE.mean()
        return (f"{per_city.mean():.3f} $\\pm$ {per_city.std(ddof=1):.3f}",
                sub.n_train_hours.mean())

    E4_DISPLAY = {"nas_transfer": "NAS-GRU (transfer + fine-tune)",
                  "lgbm_refit": "LightGBM (refit on budget)"}
    lines = [
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Strategy & Budget & Hours & MASE & $\\Delta$ vs.\\ zero-shot & "
        "Wins & Holm $P$ & TOST 90\\% CI \\\\",
        " & (\\%) & (mean) & (mean $\\pm$ sd) & (panel mean) & "
        "(strat.\\ / FM) & (Wilcoxon) & ($\\delta = 0.05$) \\\\",
        "\\midrule",
    ]
    md_rows = []
    ms0, _ = city_ms("chronos_zeroshot")
    lines.append(f"Chronos-Bolt (zero-shot) & — & — & {ms0} & — & — & — & — \\\\")
    md_rows.append({"Strategy": "Chronos-Bolt (zero-shot)", "Budget %": "—",
                    "Hours": "—", "MASE": ms0.replace(" $\\pm$ ", " ± "),
                    "Δ": "—", "Wins": "—", "Holm P": "—", "TOST 90% CI": "—"})
    for strat in ("nas_transfer", "lgbm_refit"):
        lines.append("\\midrule")
        fracs = sorted(sig[sig.comparator == strat].fraction.unique())
        for f_i, f in enumerate(fracs):
            s = sig[(sig.comparator == strat) & (sig.fraction == f)].iloc[0]
            ms, hours = city_ms(strat, f)
            delta = s.mean_mase_comp - s.mean_mase_chronos
            wins = f"{int(s.comp_wins)} / {int(s.chronos_wins)}"
            holm = f"{s.wilcoxon_p_holm:.3f}"
            tost = eq_by_frac.get(float(f), "—") if strat == "nas_transfer" else "—"
            strat_cell = (f"\\multirow{{{len(fracs)}}}{{*}}{{{E4_DISPLAY[strat]}}}"
                          if f_i == 0 else "")
            lines.append(
                f"{strat_cell} & {f:g} & {hours:,.0f} & {ms} & {delta:+.3f} & "
                f"{wins} & {holm} & {tost} \\\\")
            md_rows.append({"Strategy": E4_DISPLAY[strat], "Budget %": f"{f:g}",
                            "Hours": f"{hours:,.0f}",
                            "MASE": ms.replace(" $\\pm$ ", " ± "),
                            "Δ": f"{delta:+.3f}", "Wins": wins, "Holm P": holm,
                            "TOST 90% CI": tost})
    lines += ["\\bottomrule", "\\end{tabular}"]
    check_min_rows(len(md_rows), 8, f"{name} E4 rows")
    write(name, "\n".join(lines) + "\n",
          pd.DataFrame(md_rows).to_markdown(index=False))


# --------------------------------------------------------------------------
# T3 (main, displayed as Table 3) — measured energy, per-city rows
# --------------------------------------------------------------------------
def t2_energy(rep: pd.DataFrame):
    """Per-city energy rows: 5 tiers x 3 replication cities (15 rows), each
    with measured J/1k (range where the repeatability gate fired), per-city
    cost at the decision rule's central assumptions, the per-city CPU
    train/inference decomposition where measured, and parameter count on the
    tier's first row."""
    PRICE = 0.15  # $/kWh, harness default
    PUE = 1.4     # decision-rule central assumption; Methods states cost = J * price * PUE
    CITIES = ["beijing", "seoul", "nairobi"]

    def fmt(row):
        if row.sd_over_mean > 0.20:
            reps = []
            for r in range(1, 6):
                f = os.path.join(ROOT, f"results/v1/energy/rep_{row.city}_r{r}_results.csv")
                d = pd.read_csv(f)
                v = d.loc[d.model == row.model, "measured_j_per_1k"]
                if len(v):
                    reps.append(float(v.iloc[0]))
            return f"{min(reps):,.0f}–{max(reps):,.0f}"
        return f"{row.mean_j_per_1k:,.0f} $\\pm$ {row.sd_j_per_1k:,.0f}"

    def fmt_sci(v):
        m, e = f"{v:.1e}".split("e")
        return f"${m} \\times 10^{{{int(e)}}}$"

    # Model size: parameter count as a hardware-independent size proxy.
    # Per-forecast latency/RAM are not persisted per canonical run; the GPU
    # decomposition stays in Table S13.
    pm = pd.read_csv(os.path.join(ROOT, DOM_FILES["pm25"]))
    nparams = pm.groupby("model").n_params.max()

    def fmt_params(v):
        if pd.isna(v) or v == 0:
            return "—"
        return f"{v/1e6:,.2f}M" if v >= 1e6 else f"{v:,.0f}"

    # Per-city CPU train/inference decomposition (specialist + zero-shot FM only).
    am = pd.read_csv(os.path.join(ROOT, "results/v1/energy/amortization_summary.csv"))
    cpu = am[am.device == "cpu"].set_index(am[am.device == "cpu"].city.str.lower())

    def decomp(m, city):
        if city not in cpu.index:
            return "—", "—", "—"
        r = cpu.loc[city]
        if m == "lgbm_direct":
            cross = float(r.crossover_forecasts)
            return (f"{r.train_per_fit_j:,.0f}", f"{r.infer_fc_lgbm_j:.3f}",
                    f"{cross:,.0f}")
        if m == "chronos":
            return "—", f"{r.infer_fc_chronos_j:.3f}", "—"
        return "—", "—", "—"

    rep = rep.copy()
    rep["cell"] = rep.apply(fmt, axis=1)
    rep["usd"] = rep.mean_j_per_1k / 3.6e6 * PRICE * PUE
    cells = rep.set_index(["model", "city"])

    lines = [
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Tier & City & Measured & Cost & Train & Inference & Crossover & "
        "Params \\\\",
        " & & (J/1k) & (USD/1k) & (J/fit, CPU) & (J/forecast, CPU) & "
        "(forecasts, CPU) & \\\\",
        "\\midrule",
    ]
    md_rows = []
    for m_i, m in enumerate(TIER_ORDER):
        if m_i:
            lines.append("\\cmidrule(lr){2-8}")
        for c_i, city in enumerate(CITIES):
            r = cells.loc[(m, city)]
            tr, inf, cross = decomp(m, city)
            params = fmt_params(nparams.get(m)) if c_i == 0 else ""
            tier_cell = f"\\multirow{{3}}{{*}}{{{DISPLAY[m]}}}" if c_i == 0 else ""
            lines.append(
                f"{tier_cell} & {city.title()} & {r.cell} & "
                f"{fmt_sci(r.usd)} & {tr} & {inf} & {cross} & {params} \\\\")
            md_rows.append({"Tier": DISPLAY[m], "City": city.title(),
                            "Measured (J/1k)": r.cell.replace(" $\\pm$ ", " ± "),
                            "Cost (USD/1k)": f"{r.usd:.1e}",
                            "Train (J/fit)": tr, "Infer (J/fc)": inf,
                            "Crossover": cross,
                            "Params": params or "''"})
    lines += ["\\bottomrule", "\\end{tabular}"]
    check_min_rows(len(md_rows), 15, "T2 energy per-city")
    write("T2_energy", "\n".join(lines) + "\n",
          pd.DataFrame(md_rows).to_markdown(index=False))


if __name__ == "__main__":
    s1_city_panel()
    s2_hyperparams()
    s3_conformal_pooled()
    s4_dm()
    s5s6_conformal_percity()
    rep = s7_energy()
    s9_h48()
    s10_e4()
    s12_sensitivity()
    if os.path.exists(os.path.join(ROOT, "results/v1/energy/amortization_summary.csv")):
        s13_amortization()
    if os.path.exists(os.path.join(ROOT, "results/v1/contamination_postcutoff.csv")):
        s15_contamination()
    if os.path.exists(os.path.join(ROOT, "results/v1/equivalence_tests.csv")):
        s16_equivalence_margins()
    # Main T1 = combined causal + perfect-foresight blocks (both configs in one
    # table); S14 keeps the per-config perfect-foresight layout for the SI.
    if all(os.path.exists(os.path.join(ROOT, p)) for p in CAUSAL_DOM.values()) and \
       all(os.path.exists(os.path.join(ROOT, p)) for p in CAUSAL_DM.values()):
        t1_combined("T1_panel")                                       # main: both configs
        t1_main(DOM_FILES, DM_FILES, CANON_NEM, "S14_panel_perfect")  # supp: perfect upper bound
    else:
        t1_main()  # fallback: perfect-foresight only
    t_e4("T3_e4")   # main-text E4 table (displayed as Table 2)
    t2_energy(rep)  # main-text energy table (displayed as Table 3)
    print("ALL TABLES DONE")
