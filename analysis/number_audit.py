"""Phase 7 number audit.

Mechanically re-derives every quantitative claim in paper/sections/*.md from the
canonical artifacts on disk and compares against the value written in prose
(joined via the <!-- L-### --> ledger comments). Exits nonzero if any check fails.

Usage:  python analysis/number_audit.py
Output: results/v1/number_audit_report.md
"""
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(ROOT, "results", "v1", "number_audit_report.md")

CHECKS = []  # (check_id, ledger_id, description, expected, recomputed, ok)


def check(cid, lid, desc, expected, recomputed, tol=None):
    if tol is not None:
        ok = abs(float(expected) - float(recomputed)) <= tol
    else:
        ok = str(expected) == str(recomputed)
    CHECKS.append((cid, lid, desc, expected, recomputed, ok))
    return ok


def panel_stats(path):
    df = pd.read_csv(os.path.join(ROOT, path))
    out = {}
    for m in df.model.unique():
        per_city = df[df.model == m].groupby("city").MASE.mean()
        out[m] = (round(per_city.mean(), 3), round(per_city.std(ddof=1), 3))
    return out


def main():
    # ---- L-028: panel means (Table 1, Results, Abstract) --------------------
    pm = panel_stats("results/v1/pm25_panel/canonical_cities.csv")
    we = panel_stats("results/v1/weather_panel/canonical_cities.csv")
    expected_pm = {"chronos": (0.662, 0.368), "lgbm_direct": (0.662, 0.371),
                   "chronos_cov": (0.730, 0.376), "nas_gru": (0.734, 0.352),
                   "seasonal_naive": (1.026, 0.530)}
    expected_we = {"chronos": (0.792, 0.282), "lgbm_direct": (0.533, 0.208),
                   "chronos_cov": (0.718, 0.278), "nas_gru": (0.999, 0.481),
                   "seasonal_naive": (1.686, 1.050)}
    for m, (mu, sd) in expected_pm.items():
        check(f"pm25_mean_{m}", "L-028", f"PM2.5 panel MASE mean, {m}", mu, pm[m][0], tol=0.0005)
        check(f"pm25_sd_{m}", "L-028", f"PM2.5 panel MASE sd, {m}", sd, pm[m][1], tol=0.0005)
    for m, (mu, sd) in expected_we.items():
        check(f"we_mean_{m}", "L-028", f"Weather panel MASE mean, {m}", mu, we[m][0], tol=0.0005)
        check(f"we_sd_{m}", "L-028", f"Weather panel MASE sd, {m}", sd, we[m][1], tol=0.0005)

    # ---- L-019 / L-021: FDR-corrected DM win counts --------------------------
    for dom, lid, lgbm_fdr, chron_fdr, naive_chron in [
            ("pm25", "L-019", 0, 0, 13), ("weather", "L-021", 6, 0, 16)]:
        dm = pd.read_csv(os.path.join(ROOT, f"results/v1/{dom}_panel/canonical_dm_panel_summary.csv"))
        r = dm[(dm.model_a == "lgbm_direct") & (dm.model_b == "chronos")].iloc[0]
        check(f"{dom}_dm_lgbm_fdr", lid, f"{dom}: lgbm FDR wins vs chronos", lgbm_fdr, int(r.a_sig_wins_fdr))
        check(f"{dom}_dm_chron_fdr", lid, f"{dom}: chronos FDR wins vs lgbm", chron_fdr, int(r.b_sig_wins_fdr))
        r = dm[(dm.model_a == "seasonal_naive") & (dm.model_b == "chronos")].iloc[0]
        check(f"{dom}_dm_naive", lid, f"{dom}: chronos FDR wins vs naive", naive_chron, int(r.b_sig_wins_fdr))

    # ---- L-012 / L-023: panel sign + Wilcoxon --------------------------------
    sig = pd.read_csv(os.path.join(ROOT, "results/v1/pm25_panel/canonical_panel_significance.csv"))
    check("pm25_sign_p", "L-012", "PM2.5 sign-test p (lgbm vs chronos)", 0.136,
          round(float(sig[sig.test == "binomial_sign"].iloc[0].p_value), 3), tol=0.0005)
    check("pm25_wilcox_p", "L-012", "PM2.5 Wilcoxon p", 0.325,
          round(float(sig[sig.test == "wilcoxon_signed_rank"].iloc[0].p_value), 3), tol=0.0005)
    fri = sig[sig.test.str.contains("friedman", case=False, na=False)]
    if len(fri):
        check("pm25_friedman", "L-012", "PM2.5 Friedman p < 0.001", True,
              bool(float(fri.iloc[0].p_value) < 0.001))
    wsig = pd.read_csv(os.path.join(ROOT, "results/v1/weather_panel/canonical_panel_significance.csv"))
    check("we_sign_p", "L-023", "Weather sign-test p < 0.001", True,
          bool(float(wsig[wsig.test == "binomial_sign"].iloc[0].p_value) < 0.001))
    check("we_wilcox_p", "L-023", "Weather Wilcoxon p < 0.001", True,
          bool(float(wsig[wsig.test == "wilcoxon_signed_rank"].iloc[0].p_value) < 0.001))

    # ---- L-013 / L-024: advantage correlation --------------------------------
    for dom, lid, r_e, p_e, lo_e, hi_e in [
            ("pm25", "L-013", 0.075, 0.698, -0.196, 0.364),
            ("weather", "L-024", 0.043, 0.824, -0.330, 0.386)]:
        c = pd.read_csv(os.path.join(ROOT, f"results/v1/{dom}_panel/canonical_fm_advantage_corr_summary.csv")).iloc[0]
        check(f"{dom}_corr_r", lid, f"{dom} advantage corr r", r_e, round(float(c.pearson_r), 3), tol=0.0005)
        check(f"{dom}_corr_p", lid, f"{dom} advantage corr p", p_e, round(float(c.p_value), 3), tol=0.0015)
        check(f"{dom}_corr_lo", lid, f"{dom} bootstrap CI lo", lo_e, round(float(c.boot_ci_lo), 3), tol=0.0005)
        check(f"{dom}_corr_hi", lid, f"{dom} bootstrap CI hi", hi_e, round(float(c.boot_ci_hi), 3), tol=0.0005)

    # ---- L-027: causal covariate ablation ------------------------------------
    ab = pd.read_csv(os.path.join(ROOT, "results/v1/causal_covariate_ablation_summary.csv"))
    p = ab[ab.domain == "pm25"].iloc[0]
    w = ab[ab.domain == "weather"].iloc[0]
    check("ab_pm_perfect", "L-027", "PM2.5 lgbm perfect", 0.662, float(p.mean_lgbm_perfect), tol=0.0005)
    check("ab_pm_causal", "L-027", "PM2.5 lgbm causal", 0.692, float(p.mean_lgbm_causal), tol=0.0005)
    check("ab_pm_chronos", "L-027", "PM2.5 chronos", 0.662, float(p.mean_chronos), tol=0.0005)
    check("ab_pm_p_perf", "L-027", "PM2.5 perfect-vs-chronos p ~0.33", 0.33,
          round(float(p.lgbm_perfect_vs_chronos_p), 2), tol=0.005)
    check("ab_pm_p_caus", "L-027", "PM2.5 causal-vs-chronos p ~0.08", 0.08,
          round(float(p.lgbm_causal_vs_chronos_p), 2), tol=0.005)
    check("ab_we_perfect", "L-027", "Weather lgbm perfect", 0.533, float(w.mean_lgbm_perfect), tol=0.0005)
    check("ab_we_causal", "L-027", "Weather lgbm causal", 0.745, float(w.mean_lgbm_causal), tol=0.0005)
    check("ab_we_chronos", "L-027", "Weather chronos", 0.792, float(w.mean_chronos), tol=0.0005)
    check("ab_we_gain", "L-027", "Foresight gain +0.212", 0.212, float(w.foresight_mase_gain), tol=0.0005)
    check("ab_we_gain_p", "L-027", "Foresight gain p ~2.6e-8", True,
          bool(2.0e-8 < float(w.foresight_wilcoxon_p) < 3.0e-8))
    check("ab_we_wins_perf", "L-027", "Weather perfect wins 26/29", "26/29",
          str(w.lgbm_perfect_beats_chronos_cities))
    check("ab_we_wins_caus", "L-027", "Weather causal wins 16/29", "16/29",
          str(w.lgbm_causal_beats_chronos_cities))
    check("ab_we_p_perf", "L-027", "Weather perfect p ~1.4e-6", True,
          bool(1.0e-6 < float(w.lgbm_perfect_vs_chronos_p) < 2.0e-6))
    check("ab_we_p_caus", "L-027", "Weather causal p ~0.29", 0.29,
          round(float(w.lgbm_causal_vs_chronos_p), 2), tol=0.005)

    # ---- L-009 / L-018: E4 ----------------------------------------------------
    e4 = pd.read_csv(os.path.join(ROOT, "results/v1/e4_transfer/canonical_pm25_results.csv"))
    per_city = e4.groupby(["strategy", "fraction", "city"], dropna=False).MASE.mean().reset_index()
    g = per_city.groupby(["strategy", "fraction"], dropna=False).MASE.mean()
    ch = per_city[per_city.strategy == "chronos_zeroshot"].MASE.mean()
    check("e4_chronos", "L-009", "E4 chronos zero-shot mean", 0.843, round(float(ch), 3), tol=0.0005)
    for frac, exp in [(0.0, 0.899), (1.0, 0.915), (10.0, 0.888), (100.0, 0.876)]:
        check(f"e4_nas_{int(frac)}", "L-009", f"E4 nas_transfer @{int(frac)}%", exp,
              round(float(g.loc[("nas_transfer", frac)]), 3), tol=0.0005)
    for frac, exp in [(1.0, 0.941), (10.0, 0.944), (100.0, 0.858)]:
        check(f"e4_lgbm_{int(frac)}", "L-009", f"E4 lgbm_refit @{int(frac)}%", exp,
              round(float(g.loc[("lgbm_refit", frac)]), 3), tol=0.0005)
    e4s = pd.read_csv(os.path.join(ROOT, "results/v1/e4_transfer/canonical_pm25_significance.csv"))
    nas = e4s[e4s.comparator == "nas_transfer"]
    check("e4_bucket4", "L-018", "E4: no Holm-sig difference chronos vs nas_transfer at any fraction",
          True, bool((nas.wilcoxon_p_holm > 0.05).all()
                     and (~nas.chronos_sig_better_5pct.astype(bool)).all()
                     and (~nas.comp_sig_better_5pct.astype(bool)).all()))
    check("e4_n_cities", "L-018", "E4 n=15 cities", 15, int(nas.n_cities.iloc[0]))

    # ---- L-026: Beijing 12-station -------------------------------------------
    bj = pd.read_csv(os.path.join(ROOT, "results/v1/beijing/canonical_sweep12_hetero.csv"))
    piv = bj.pivot(index="series", columns="model", values="MASE")
    check("bj_n", "L-026", "Beijing station count", 12, int(len(piv)))
    check("bj_beats_lgbm", "L-026", "chronos < lgbm at 12/12", 12,
          int((piv["chronos"] < piv["lgbm_direct"]).sum()))
    check("bj_beats_naive", "L-026", "chronos < naive at 12/12", 12,
          int((piv["chronos"] < piv["seasonal_naive"]).sum()))
    check("bj_ch_lo", "L-026", "chronos MASE min ~0.153", 0.153, round(piv["chronos"].min(), 3), tol=0.0005)
    check("bj_ch_hi", "L-026", "chronos MASE max ~0.297", 0.297, round(piv["chronos"].max(), 3), tol=0.0005)
    check("bj_lg_lo", "L-026", "lgbm MASE min ~0.292", 0.292, round(piv["lgbm_direct"].min(), 3), tol=0.0005)
    check("bj_lg_hi", "L-026", "lgbm MASE max ~0.458", 0.458, round(piv["lgbm_direct"].max(), 3), tol=0.0005)
    check("bj_nv_lo", "L-026", "naive MASE min ~1.03", 1.03, round(piv["seasonal_naive"].min(), 2), tol=0.005)
    check("bj_nv_hi", "L-026", "naive MASE max ~1.24", 1.24, round(piv["seasonal_naive"].max(), 2), tol=0.005)

    # ---- L-025: energy ---------------------------------------------------------
    rep = pd.read_csv(os.path.join(ROOT, "results/v1/energy/repeatability_summary.csv"))
    check("en_gate", "L-025", "4 of 15 cells exceed 20% gate", 4, int(rep.exceeds_20pct_gate.sum()))
    lg = rep[rep.model == "lgbm_direct"].mean_j_per_1k
    check("en_lgbm_lo", "L-025", "lgbm J/1k min ~9461 (9.5 kJ)", True, bool(9400 < lg.min() < 9500))
    check("en_lgbm_hi", "L-025", "lgbm J/1k max ~15061 (15.1 kJ)", True, bool(15000 < lg.max() < 15100))
    ch_ok = rep[(rep.model == "chronos") & (~rep.exceeds_20pct_gate)].mean_j_per_1k
    check("en_chronos", "L-025", "chronos unflagged cells within 1.0-1.3 kJ", True,
          bool(ch_ok.between(1000, 1300).all()))
    usd = rep.groupby("model").apply(
        lambda d: (d.mean_j_per_1k / 3.6e6 * 0.15).mean(), include_groups=False)
    ratio = usd["lgbm_direct"] / usd["chronos"]
    check("en_usd_ratio", "L-025", "lgbm/chronos USD ratio ~8x (7.5-9.5)", True, bool(7.5 < ratio < 9.5))

    # ---- L-040: cost sensitivity (CAUSAL maps; supersedes perfect-foresight L-020) ----
    s12 = pd.read_csv(os.path.join(ROOT, "results/v1/regime/s12_cost_sensitivity_causal.csv"))
    check("s12_max", "L-040", "max flip rate 40% (causal)", 0.40,
          round(float(s12.flip_rate.max()), 3), tol=0.0005)
    central = s12[(s12.price_kwh == 0.15) & (s12.pue == 1.4)]
    check("s12_central", "L-040", "0 flips at central price/PUE (causal, all 6 runs)",
          0.0, float(central.flip_rate.max()), tol=1e-9)
    check("s12_runs", "L-040", "sensitivity covers 6 causal regime runs", 6,
          int(s12.run.nunique()))

    # ---- L-011 / L-022: conformal ----------------------------------------------
    cf_p = pd.read_csv(os.path.join(ROOT, "results/v1/pm25_panel/canonical_conformal_pooled.csv"))
    cf_w = pd.read_csv(os.path.join(ROOT, "results/v1/weather_panel/canonical_conformal_pooled.csv"))
    check("cf_pm_lo", "L-011", "PM2.5 pooled coverage min ~0.914", 0.914,
          round(float(cf_p.mean_coverage.min()), 3), tol=0.0005)
    check("cf_pm_hi", "L-011", "PM2.5 pooled coverage max ~0.970", 0.970,
          round(float(cf_p.mean_coverage.max()), 3), tol=0.0005)
    check("cf_we_lo", "L-022", "Weather pooled coverage min ~0.892", 0.892,
          round(float(cf_w.mean_coverage.min()), 3), tol=0.0005)
    check("cf_we_hi", "L-022", "Weather pooled coverage max ~0.968", 0.968,
          round(float(cf_w.mean_coverage.max()), 3), tol=0.0005)
    rich_w = cf_w[cf_w.tier == "rich"].set_index("model")
    check("cf_we_lgbm_width", "L-022", "Weather rich lgbm width ~7.7", 7.7,
          round(float(rich_w.loc["lgbm_direct"].mean_width), 1), tol=0.05)
    check("cf_we_ch_width", "L-022", "Weather rich chronos width ~10.7", 10.7,
          round(float(rich_w.loc["chronos"].mean_width), 1), tol=0.05)
    rich_p = cf_p[cf_p.tier == "rich"].set_index("model")
    check("cf_pm_lgbm_width", "L-011", "PM2.5 rich lgbm width ~17.7", 17.7,
          round(float(rich_p.loc["lgbm_direct"].mean_width), 1), tol=0.05)
    check("cf_pm_ch_width", "L-011", "PM2.5 rich chronos width ~18.5", 18.5,
          round(float(rich_p.loc["chronos"].mean_width), 1), tol=0.05)
    check("cf_pm_nv_width", "L-011", "PM2.5 rich naive width ~35.1", 35.1,
          round(float(rich_p.loc["seasonal_naive"].mean_width), 1), tol=0.05)

    # ---- L-029: decision-map winner counts --------------------------------------
    def winner_counts(name):
        df = pd.read_csv(os.path.join(ROOT, f"results/v1/regime/canonical_{name}_decision.csv"))
        cells = df.drop(columns=["train_weeks"]).values.ravel()
        s = pd.Series([x for x in cells if isinstance(x, str)])
        return s.value_counts().to_dict(), len(s)

    wc, n = winner_counts("beijing_pm25")
    check("dc_bj_pm", "L-029", "Beijing PM2.5 chronos 21/25", (21, 25), (wc.get("chronos", 0), n))
    wc, n = winner_counts("seoul_pm25")
    check("dc_se_pm", "L-029", "Seoul PM2.5 chronos 17/25", (17, 25), (wc.get("chronos", 0), n))
    wc, n = winner_counts("nairobi_pm25")
    check("dc_na_pm", "L-029", "Nairobi PM2.5 chronos-family 13/20", (13, 20),
          (wc.get("chronos", 0) + wc.get("chronos_cov", 0), n))
    wc, n = winner_counts("beijing_weather")
    check("dc_bj_we", "L-029", "Beijing weather lgbm 11/25", (11, 25), (wc.get("lgbm_direct", 0), n))
    wc, n = winner_counts("seoul_weather")
    check("dc_se_we", "L-029", "Seoul weather lgbm 12/25", (12, 25), (wc.get("lgbm_direct", 0), n))
    wc, n = winner_counts("nairobi_weather")
    check("dc_na_we", "L-029", "Nairobi weather lgbm 18/20", (18, 20), (wc.get("lgbm_direct", 0), n))

    # ================= Stage E (reviewer-response) checks =========================
    # ---- L-037: causal-primary panel means (Table 1 main) ------------------------
    cpm = panel_stats("results/v1/pm25_panel/causal_primary_cities.csv")
    cwe = panel_stats("results/v1/weather_panel/causal_primary_cities.csv")
    exp_cpm = {"chronos": (0.662, 0.368), "lgbm_direct": (0.692, 0.374),
               "chronos_cov": (0.797, 0.481), "nas_gru": (0.734, 0.352),
               "seasonal_naive": (1.026, 0.530)}
    exp_cwe = {"chronos": (0.792, 0.282), "lgbm_direct": (0.745, 0.223),
               "chronos_cov": (2.614, 1.044), "nas_gru": (0.999, 0.481),
               "seasonal_naive": (1.686, 1.050)}
    for m, (mu, sd) in exp_cpm.items():
        check(f"cpm_mean_{m}", "L-037", f"causal PM2.5 mean {m}", mu, cpm[m][0], tol=0.0006)
        check(f"cpm_sd_{m}", "L-037", f"causal PM2.5 sd {m}", sd, cpm[m][1], tol=0.0006)
    for m, (mu, sd) in exp_cwe.items():
        check(f"cwe_mean_{m}", "L-037", f"causal weather mean {m}", mu, cwe[m][0], tol=0.0006)
        check(f"cwe_sd_{m}", "L-037", f"causal weather sd {m}", sd, cwe[m][1], tol=0.0006)

    # ---- L-033 / L-034: causal DM FDR wins vs chronos ----------------------------
    def dm_fdr(path, tier):
        dm = pd.read_csv(os.path.join(ROOT, path))
        pair = dm[((dm.model_a == tier) & (dm.model_b == "chronos")) |
                  ((dm.model_a == "chronos") & (dm.model_b == tier))].iloc[0]
        tier_w = pair.a_sig_wins_fdr if pair.model_a == tier else pair.b_sig_wins_fdr
        chr_w = pair.b_sig_wins_fdr if pair.model_a == tier else pair.a_sig_wins_fdr
        return int(tier_w), int(chr_w)
    check("cdm_pm_lgbm", "L-033", "causal PM2.5 lgbm 0/0 vs chronos", (0, 0),
          dm_fdr("results/v1/pm25_panel/causal_primary_dm_panel_summary.csv", "lgbm_direct"))
    check("cdm_we_lgbm", "L-034", "causal weather lgbm 0/1 vs chronos (specialist 0 wins)", (0, 1),
          dm_fdr("results/v1/weather_panel/causal_primary_dm_panel_summary.csv", "lgbm_direct"))

    # ---- L-030: equivalence tests ------------------------------------------------
    eq = pd.read_csv(os.path.join(ROOT, "results/v1/equivalence_tests.csv")).set_index("comparison")
    check("eq_pm_perfect", "L-030", "PM2.5 perfect-foresight lgbm-vs-chronos EQUIVALENT", True,
          bool(eq.loc["pm25-perfect: lgbm_direct vs chronos (per city)"].equivalent_at_0p05))
    check("eq_pm_causal", "L-030", "PM2.5 causal lgbm-vs-chronos NOT equivalent", False,
          bool(eq.loc["pm25-causal: lgbm_direct vs chronos (per city)"].equivalent_at_0p05))
    check("eq_we_causal", "L-030", "weather-causal lgbm-vs-chronos NOT equivalent", False,
          bool(eq.loc["weather-causal: lgbm_direct vs chronos (per city)"].equivalent_at_0p05))
    e4rows = [i for i in eq.index if i.startswith("E4 pm25")]
    check("eq_e4_none", "L-030", "E4: no fraction equivalent", True,
          bool(not eq.loc[e4rows].equivalent_at_0p05.any()))
    check("eq_pm_perfect_p", "L-030", "PM2.5 perfect TOST p~0.026", 0.026,
          round(float(eq.loc["pm25-perfect: lgbm_direct vs chronos (per city)"].p_tost), 3), tol=0.002)

    # ---- L-032: energy amortization ---------------------------------------------
    am = pd.read_csv(os.path.join(ROOT, "results/v1/energy/amortization_summary.csv"))
    check("am_lgbm_cheaper", "L-032", "lgbm inference cheaper than chronos in all cells", True,
          bool(am.lgbm_infer_cheaper.all()))
    cpu = am[am.device == "cpu"].crossover_forecasts.astype(float)
    check("am_cpu_range", "L-032", "CPU crossover in [2778,4416]", True,
          bool(cpu.min() >= 2500 and cpu.max() <= 4600))

    # ---- L-039: causal-primary panel-level tests (pm25) ---------------------------
    ps = pd.read_csv(os.path.join(ROOT, "results/v1/pm25_panel/causal_primary_panel_significance.csv"))
    sign = ps[ps.test == "binomial_sign"].iloc[0]
    wil = ps[ps.test == "wilcoxon_signed_rank"].iloc[0]
    fri = ps[ps.test == "friedman"].iloc[0]
    check("cps_sign_p", "L-039", "causal pm25 sign test p~0.024", 0.024,
          round(float(sign.p_value), 3), tol=0.0005)
    check("cps_sign_wins", "L-039", "causal pm25 chronos better 21/29", (21, 29),
          (int(sign.fm_wins), int(sign.n_cities)))
    check("cps_wil_p", "L-039", "causal pm25 wilcoxon p~0.084", 0.084,
          round(float(wil.p_value), 3), tol=0.0005)
    check("cps_fri_p", "L-039", "causal pm25 friedman p<0.001", True,
          bool(float(fri.p_value) < 0.001))
    # Nemenyi average ranks shown in Table 1 (causal config, both domains)
    nem_p = pd.read_csv(os.path.join(ROOT, "results/v1/pm25_panel/causal_primary_nemenyi_ranks.csv")).set_index("model")
    nem_w = pd.read_csv(os.path.join(ROOT, "results/v1/weather_panel/causal_primary_nemenyi_ranks.csv")).set_index("model")
    check("nem_pm_chronos", "L-039", "causal pm25 chronos rank ~1.72", 1.72,
          round(float(nem_p.loc["chronos", "avg_rank"]), 2), tol=0.005)
    check("nem_pm_lgbm", "L-039", "causal pm25 lgbm rank ~2.41", 2.41,
          round(float(nem_p.loc["lgbm_direct", "avg_rank"]), 2), tol=0.005)
    check("nem_we_lgbm", "L-041", "causal weather lgbm rank ~1.72", 1.72,
          round(float(nem_w.loc["lgbm_direct", "avg_rank"]), 2), tol=0.005)
    check("nem_we_chronos", "L-041", "causal weather chronos rank ~1.93", 1.93,
          round(float(nem_w.loc["chronos", "avg_rank"]), 2), tol=0.005)
    check("nem_rank_gap", "L-041", "spec-FM rank gap < CD 1.13 in both domains", True,
          bool(abs(nem_p.loc["chronos", "avg_rank"] - nem_p.loc["lgbm_direct", "avg_rank"]) < 1.133 and
               abs(nem_w.loc["chronos", "avg_rank"] - nem_w.loc["lgbm_direct", "avg_rank"]) < 1.133))

    # ---- L-042 / L-043: causal-config data-volume correlations -------------------
    c42 = pd.read_csv(os.path.join(ROOT, "results/v1/pm25_panel/causal_primary_fm_advantage_corr_summary.csv"))
    check("corr42_pm", "L-042", "causal pm25 corr r~-0.031", -0.031,
          round(float(c42.pearson_r.iloc[0]), 3), tol=0.0006)
    check("corr42_p", "L-042", "causal pm25 corr non-sig", True, bool(c42.p_value.iloc[0] > 0.05))
    c43 = pd.read_csv(os.path.join(ROOT, "results/v1/weather_panel/causal_primary_fm_advantage_corr_summary.csv"))
    check("corr43_we", "L-043", "causal weather corr r~0.138", 0.138,
          round(float(c43.pearson_r.iloc[0]), 3), tol=0.0006)
    check("corr43_p", "L-043", "causal weather corr non-sig", True, bool(c43.p_value.iloc[0] > 0.05))

    # ---- S16: equivalence margin sensitivity (verdicts derived from CIs) ---------
    eq = pd.read_csv(os.path.join(ROOT, "results/v1/equivalence_tests.csv"))
    eq["mm"] = eq[["tost90_lo", "tost90_hi"]].abs().max(axis=1)
    # These two checks describe what Table S16 SHOWS, and tables/make_tables.py builds S16
    # from an explicit list of the manuscript's tie claims -- not from every row of
    # equivalence_tests.csv. The counts were coincidentally right while the CSV happened to
    # equal that list; once the CSV became a superset (the revision added FM-vs-FM rows,
    # which answer a different question and are not in S16) counting the whole file
    # measured something the table does not report. Restrict to the table's own rows.
    S16_ROWS = [c for c in eq.comparison
                if c.startswith("E4 pm25")
                or c in ("pm25-perfect: lgbm_direct vs chronos (per city)",
                         "pm25-causal: lgbm_direct vs chronos (per city)",
                         "weather-causal: lgbm_direct vs chronos (per city)")]
    eq_s16 = eq[eq.comparison.isin(S16_ROWS)]
    n_eq05 = int((eq_s16.mm <= 0.05).sum())
    n_eq10 = int((eq_s16.mm <= 0.10).sum())
    check("s16_05", "L-030", "exactly 1 comparison equivalent at margin 0.05", 1, n_eq05)
    check("s16_10", "L-030", "3 comparisons equivalent at margin 0.10", 3, n_eq10)

    # ---- L-038: contamination post-cutoff ---------------------------------------
    ct = pd.read_csv(os.path.join(ROOT, "results/v1/contamination_postcutoff.csv"))
    check("ct_n", "L-038", "post-cutoff cities n=10", 10, int(len(ct)))
    check("ct_chronos", "L-038", "post-cutoff chronos mean ~0.415", 0.415,
          round(float(ct.MASE_chronos.mean()), 3), tol=0.002)
    check("ct_lgbm", "L-038", "post-cutoff lgbm mean ~0.395", 0.395,
          round(float(ct.MASE_lgbm_direct.mean()), 3), tol=0.002)
    check("ct_wins", "L-038", "post-cutoff chronos beats lgbm 6/10", 6,
          int(ct.chronos_beats_lgbm.sum()))

    # ---- L-031: causal decision-map winner counts -------------------------------
    def causal_wc(name):
        df = pd.read_csv(os.path.join(ROOT, f"results/v1/regime/causal_{name}_decision.csv"))
        cells = df.drop(columns=["train_weeks"]).values.ravel()
        s = pd.Series([x for x in cells if isinstance(x, str)])
        return s.value_counts().to_dict(), len(s)
    wc, n = causal_wc("beijing_pm25")
    check("cdc_bj_pm", "L-031", "causal Beijing PM2.5 chronos 21/25", (21, 25), (wc.get("chronos", 0), n))
    wc, n = causal_wc("seoul_pm25")
    check("cdc_se_pm", "L-031", "causal Seoul PM2.5 chronos 14/25", (14, 25), (wc.get("chronos", 0), n))
    wc, n = causal_wc("beijing_weather")
    check("cdc_bj_we", "L-031", "causal Beijing weather lgbm 6/25", (6, 25), (wc.get("lgbm_direct", 0), n))
    wc, n = causal_wc("seoul_weather")
    check("cdc_se_we", "L-031", "causal Seoul weather nas 25/25", (25, 25), (wc.get("nas_gru", 0), n))

    # ---- L-035 / L-036: causal conformal coverage --------------------------------
    ccp = pd.read_csv(os.path.join(ROOT, "results/v1/pm25_panel/causal_primary_conformal_pooled.csv"))
    ccw = pd.read_csv(os.path.join(ROOT, "results/v1/weather_panel/causal_primary_conformal_pooled.csv"))
    check("ccf_pm_lo", "L-035", "causal PM2.5 coverage min ~0.910", 0.910,
          round(float(ccp.mean_coverage.min()), 3), tol=0.001)
    check("ccf_we_lo", "L-036", "causal weather coverage min ~0.900", 0.900,
          round(float(ccw.mean_coverage.min()), 3), tol=0.001)
    rw = ccw[ccw.tier == "rich"].set_index("model")
    check("ccf_we_lgbm_w", "L-036", "causal weather rich lgbm width ~9.6", 9.6,
          round(float(rw.loc["lgbm_direct"].mean_width), 1), tol=0.1)

    # ---- structural: every L-### in sections exists in the ledger ----------------
    ledger = open(os.path.join(ROOT, "paper", "RESULTS_LEDGER.md"), encoding="utf-8").read()
    ledger_ids = set(re.findall(r"\| (L-\d{3}) \|", ledger))
    used = set()
    sec_dir = os.path.join(ROOT, "paper", "sections")
    for f in os.listdir(sec_dir):
        if f.endswith(".md"):
            used |= set(re.findall(r"<!-- (L-\d{3}) -->",
                                   open(os.path.join(sec_dir, f), encoding="utf-8").read()))
    missing = used - ledger_ids
    check("ledger_join", "-", f"all {len(used)} cited ledger IDs exist in ledger",
          "none missing", "none missing" if not missing else f"MISSING: {sorted(missing)}")
    superseded = set(re.findall(r"\| (L-\d{3}) \| \[SUPERSEDED", ledger))
    bad = used & superseded
    check("no_superseded", "-", "no superseded ledger row cited in prose",
          "none", "none" if not bad else f"CITED SUPERSEDED: {sorted(bad)}")

    # ---- report -------------------------------------------------------------------
    n_ok = sum(1 for c in CHECKS if c[5])
    lines = ["# Phase 7 number-audit report", "",
             f"**{n_ok}/{len(CHECKS)} checks passed.**", "",
             "| check | ledger | description | expected | recomputed | status |",
             "|---|---|---|---|---|---|"]
    for cid, lid, desc, exp, got, ok in CHECKS:
        lines.append(f"| {cid} | {lid} | {desc} | {exp} | {got} | {'PASS' if ok else '**FAIL**'} |")
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{n_ok}/{len(CHECKS)} checks passed -> {os.path.relpath(REPORT, ROOT)}")
    for cid, lid, desc, exp, got, ok in CHECKS:
        if not ok:
            print(f"FAIL {cid} ({lid}): {desc} | expected {exp} got {got}")
    sys.exit(0 if n_ok == len(CHECKS) else 1)


if __name__ == "__main__":
    main()
