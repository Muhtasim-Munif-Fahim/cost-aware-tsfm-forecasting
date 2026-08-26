#!/usr/bin/env python3
"""Supplementary tables added for the revision (S17, S18).

Kept separate from `make_tables.py` deliberately: that script regenerates every table in
the manuscript, and re-running it during a revision would rewrite tables whose numbers are
already audited and cited, for no reason. This builds only the two new ones, reusing the
same formatting helper so the output is stylistically identical.

  S17 -- Diebold-Mariano dependence robustness (answers R1.3)
  S18 -- pandemic / observation-epoch stratified sensitivity (answers R2.1)

Usage: python tables/make_revision_tables.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tables"))
from make_tables import df_to_latex_md  # noqa: E402

DOM = {"pm25": "PM2.5", "weather": "Temperature"}


def s17_dm_robustness():
    rows = []
    for dom in ("pm25", "weather"):
        p = os.path.join(ROOT, "results", "v1", f"{dom}_dm_dependence_robustness.csv")
        if not os.path.exists(p):
            print(f"  skip S17/{dom}: {os.path.basename(p)} not found")
            continue
        d = pd.read_csv(p)
        n = len(d)
        rec = {"Domain": DOM[dom], "Cities": n,
               "HAC DM": f"{int(d.dm_significant.sum())}/{n}"}
        for L in (12, 24, 48):
            col = f"mbb{L}_significant"
            if col in d:
                agree = int((d[col] == d.dm_significant).sum())
                rec[f"MBB $L$={L}"] = f"{int(d[col].sum())}/{n} ({agree})"
        if "fold_significant" in d:
            agree = int((d.fold_significant == d.dm_significant).sum())
            rec["Trajectory ($n$=6)"] = f"{int(d.fold_significant.sum())}/{n} ({agree})"
        rows.append(rec)
    if not rows:
        return
    df_to_latex_md(pd.DataFrame(rows), "S17_dm_dependence")
    print("  wrote S17_dm_dependence")


def s18_stratified():
    p = os.path.join(ROOT, "results", "v1", "pandemic_stratified.csv")
    if not os.path.exists(p):
        print("  skip S18: pandemic_stratified.csv not found")
        return
    d = pd.read_csv(p)
    LAB = {"all": "Whole panel",
           "pandemic_exposed": "Pandemic-exposed",
           "pandemic_unexposed": "Not exposed",
           "pandemic_heavy(>=0.5)": "Heavily exposed",
           "window_ends_le_2021": "Window ends $\\le$ 2021",
           "window_ends_gt_2021": "Window ends $>$ 2021"}
    out = pd.DataFrame({
        "Domain": d.domain.map(DOM),
        "Stratum": d.stratum.map(lambda s: LAB.get(s, s)),
        "Cities": d.n_cities,
        "Specialist": d.mean_mase_a.map(lambda v: f"{v:.3f}"),
        "Zero-shot FM": d.mean_mase_b.map(lambda v: f"{v:.3f}"),
        "Difference": d.mean_diff_a_minus_b.map(lambda v: f"{v:+.3f}"),
        "FM better": d.apply(lambda r: f"{int(r.n_cities_fm_better)}/{int(r.n_cities)}", axis=1),
        "$P$": d.p_value.map(lambda v: "--" if pd.isna(v) else f"{v:.3f}"),
        "$P_{\\mathrm{BH}}$": d.p_value_bh.map(lambda v: "--" if pd.isna(v) else f"{v:.3f}"),
    })
    df_to_latex_md(out, "S18_pandemic_stratified")
    print("  wrote S18_pandemic_stratified")


if __name__ == "__main__":
    print("revision supplementary tables:")
    s17_dm_robustness()
    s18_stratified()
