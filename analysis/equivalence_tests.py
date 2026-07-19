"""Reviewer-requested equivalence analysis (added 2026-07-15, post-hoc, logged in the
deviations record).

A non-significant sign/Wilcoxon test is not evidence of equivalence. For each "tie" claim
in the manuscript we therefore report the paired per-city (or per-fraction) MASE difference,
a bootstrap 95% CI, and a two-one-sided-test (TOST) equivalence decision against a
pre-specified interpretability margin.

Margin (pre-specified here, NOT tuned to the data): delta = 0.05 MASE, absolute. Rationale:
~7-8% of the ~0.66-0.79 panel-mean MASE, and below the cell-to-cell resolution at which the
cost-adjusted decision rule changes its winner; a paired difference whose CI lies entirely
within +/-0.05 MASE is not operationally meaningful. We report equivalence at this margin and
also give the raw CI so a reader can apply their own threshold.

Difference sign convention: d_i = MASE(specialist) - MASE(foundation model). d>0 favours the FM.

Usage: python analysis/equivalence_tests.py
Output: results/v1/equivalence_tests.csv
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "v1", "equivalence_tests.csv")
DELTA = 0.05           # equivalence margin, MASE (pre-specified)
NBOOT = 10000
SEED = 12345           # fixed so the bootstrap CI is reproducible


def _by_city(path, model):
    df = pd.read_csv(os.path.join(ROOT, path))
    return df[df.model == model].groupby("city").MASE.mean()


def tost(d, delta=DELTA, nboot=NBOOT):
    """Paired TOST on differences d. Returns dict with mean, bootstrap 95% CI, the TOST
    90% CI (the interval that must lie within +/-delta for equivalence), and paired-t
    one-sided p-values for the two nulls |mu|>=delta."""
    from scipy import stats
    d = np.asarray(d, float)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    # TOST via paired t: p_lower tests mu <= -delta, p_upper tests mu >= +delta
    t_lower = (mean + delta) / se
    t_upper = (mean - delta) / se
    p_lower = float(stats.t.sf(t_lower, df=n - 1))       # H0: mu <= -delta
    p_upper = float(stats.t.cdf(t_upper, df=n - 1))      # H0: mu >= +delta
    p_tost = max(p_lower, p_upper)
    tcrit = stats.t.ppf(0.95, df=n - 1)                  # 90% CI for TOST
    ci90 = (mean - tcrit * se, mean + tcrit * se)
    rng = np.random.default_rng(SEED)
    boots = d[rng.integers(0, n, size=(nboot, n))].mean(axis=1)
    ci95 = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    equivalent = (ci90[0] > -delta) and (ci90[1] < delta)
    return dict(n=n, mean_diff=mean, ci95_lo=ci95[0], ci95_hi=ci95[1],
                tost90_lo=ci90[0], tost90_hi=ci90[1], p_tost=p_tost,
                equivalent_at_0p05=bool(equivalent))


def main():
    rows = []

    # 1a. PM2.5: perfect-foresight lgbm_direct vs chronos, per city (upper bound / S14)
    sp = _by_city("results/v1/pm25_panel/canonical_cities.csv", "lgbm_direct")
    fm = _by_city("results/v1/pm25_panel/canonical_cities.csv", "chronos")
    d = (sp - fm).dropna()
    rows.append({"comparison": "pm25-perfect: lgbm_direct vs chronos (per city)", **tost(d)})

    # 1b. PM2.5: CAUSAL lgbm_direct vs chronos, per city (Table 1 main)
    sp = _by_city("results/v1/pm25_panel/causal_ablation_cities.csv", "lgbm_direct")
    d = (sp - fm).dropna()
    rows.append({"comparison": "pm25-causal: lgbm_direct vs chronos (per city)", **tost(d)})

    # 2. Weather: causal lgbm_direct vs chronos, per city (the headline causal tie)
    sp = _by_city("results/v1/weather_panel/causal_ablation_cities.csv", "lgbm_direct")
    fm = _by_city("results/v1/weather_panel/canonical_cities.csv", "chronos")
    d = (sp - fm).dropna()
    rows.append({"comparison": "weather-causal: lgbm_direct vs chronos (per city)", **tost(d)})

    # 3. E4: nas_transfer vs chronos_zeroshot, per fraction (per-city seed means, 15 cities)
    e4 = pd.read_csv(os.path.join(ROOT, "results/v1/e4_transfer/canonical_pm25_results.csv"))
    ch = e4[e4.strategy == "chronos_zeroshot"].groupby("city").MASE.mean()
    for frac in [0.0, 1.0, 10.0, 100.0]:
        nas = (e4[(e4.strategy == "nas_transfer") & (e4.fraction == frac)]
               .groupby("city").MASE.mean())
        common = nas.index.intersection(ch.index)
        d = (nas.loc[common] - ch.loc[common]).dropna()
        rows.append({"comparison": f"E4 pm25: nas_transfer@{int(frac)}% vs chronos_zeroshot",
                     **tost(d)})

    out = pd.DataFrame(rows)
    for c in ["mean_diff", "ci95_lo", "ci95_hi", "tost90_lo", "tost90_hi", "p_tost"]:
        out[c] = out[c].round(4)
    out.to_csv(OUT, index=False)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(out.to_string(index=False))
    print(f"\nmargin delta = {DELTA} MASE; equivalent = TOST 90% CI within +/-delta")
    print(f"written -> {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
