#!/usr/bin/env python3
"""Headline robustness: does the 'specialist wins the weather domain' result survive when the
specialist is denied a perfect weather forecast?

Compares, per domain and tier, the canonical (perfect-foresight covariate) lgbm_direct MASE
against the causal-covariate ablation lgbm_direct MASE (weather at forecast origin), and both
against the covariate-free chronos zero-shot baseline. Panel-level paired Wilcoxon on per-city
MASE quantifies (a) how much perfect foresight helps the specialist in each domain, and
(b) whether the specialist still beats the FM once foresight is removed.

Inputs (per domain):
  canonical  = <panel>/canonical_cities.csv        (lgbm_direct = perfect-foresight)
  causal     = <panel>/causal_ablation_cities.csv  (lgbm_direct = causal covariates)

Usage:
  python analysis/causal_covariate_ablation.py
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER_PATH = os.path.join(ROOT, "paper", "RESULTS_LEDGER.md")
DOMAINS = {
    "pm25": ("results/v1/pm25_panel/canonical_cities.csv",
             "results/v1/pm25_panel/causal_ablation_cities.csv"),
    "weather": ("results/v1/weather_panel/canonical_cities.csv",
                "results/v1/weather_panel/causal_ablation_cities.csv"),
}
OUT = "results/v1/causal_covariate_ablation_summary.csv"


def next_ledger_id():
    text = open(LEDGER_PATH, encoding="utf-8").read()
    ids = [int(m) for m in re.findall(r"L-(\d+)", text)]
    return max(ids, default=0) + 1


def append_ledger_stub(claim, value, artifact, command, code_tag):
    n = next_ledger_id()
    row = (f"| L-{n:03d} | {claim} | {value} | | {artifact} | `{command}` | | {code_tag} | "
           f"n/a | | Phase 3 script |\n")
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(row)


def lgbm_by_city(path):
    df = pd.read_csv(path)
    return df[df.model == "lgbm_direct"].set_index("city").MASE


def chronos_by_city(path):
    df = pd.read_csv(path)
    return df[df.model == "chronos"].set_index("city").MASE


def main():
    rows = []
    for dom, (canon_p, causal_p) in DOMAINS.items():
        if not (os.path.exists(canon_p) and os.path.exists(causal_p)):
            print(f"[skip {dom}] missing {canon_p if not os.path.exists(canon_p) else causal_p}")
            continue
        lg_pf = lgbm_by_city(canon_p)
        lg_ca = lgbm_by_city(causal_p)
        ch = chronos_by_city(canon_p)
        j = pd.DataFrame({"lgbm_pf": lg_pf, "lgbm_causal": lg_ca, "chronos": ch}).dropna()
        n = len(j)

        # (a) perfect-foresight benefit to the specialist
        w_fore = wilcoxon(j.lgbm_pf, j.lgbm_causal)
        pf_gain = float((j.lgbm_causal - j.lgbm_pf).mean())   # >0 => foresight helps

        # (b) specialist vs FM, with vs without foresight
        w_pf = wilcoxon(j.lgbm_pf, j.chronos)
        w_ca = wilcoxon(j.lgbm_causal, j.chronos)
        lg_pf_wins = int((j.lgbm_pf < j.chronos).sum())
        lg_ca_wins = int((j.lgbm_causal < j.chronos).sum())

        rows.append({
            "domain": dom, "n_cities": n,
            "mean_lgbm_perfect": round(j.lgbm_pf.mean(), 3),
            "mean_lgbm_causal": round(j.lgbm_causal.mean(), 3),
            "mean_chronos": round(j.chronos.mean(), 3),
            "foresight_mase_gain": round(pf_gain, 3),
            "foresight_wilcoxon_p": w_fore.pvalue,
            "lgbm_perfect_beats_chronos_cities": f"{lg_pf_wins}/{n}",
            "lgbm_perfect_vs_chronos_p": w_pf.pvalue,
            "lgbm_causal_beats_chronos_cities": f"{lg_ca_wins}/{n}",
            "lgbm_causal_vs_chronos_p": w_ca.pvalue,
        })

    if not rows:
        raise SystemExit("no domains ready -- run R020/R021 first")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(ROOT, OUT), index=False)
    print(out.to_string(index=False))

    # headline interpretation
    for r in rows:
        verdict = ("SPECIALIST LEAD IS PERFECT-FORESIGHT-DEPENDENT"
                   if r["lgbm_perfect_vs_chronos_p"] < 0.05 <= r["lgbm_causal_vs_chronos_p"]
                   else "specialist lead robust to causal covariates"
                   if r["lgbm_causal_vs_chronos_p"] < 0.05
                   else "no significant specialist lead either way")
        print(f"  [{r['domain']}] {verdict}")
    print(f"\nsaved -> {OUT}")

    append_ledger_stub(
        claim="Perfect-foresight covariate ablation: per-domain lgbm_direct MASE with perfect-"
              "forecast vs causal (origin) weather covariates, both vs chronos zero-shot; "
              "paired Wilcoxon on per-city MASE (headline robustness of the domain-flip claim)",
        value="see CSV; verdict per domain printed by the script",
        artifact=OUT,
        command="python analysis/causal_covariate_ablation.py",
        code_tag="e14e80a-dirty",
    )


if __name__ == "__main__":
    main()
