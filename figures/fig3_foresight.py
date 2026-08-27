"""Figure 3 (money figure) — perfect-foresight covariate ablation.

Paired slopegraph per domain: LightGBM MASE with a perfect weather forecast (left)
vs causal last-known covariates (right); per-city thin slopes, bold mean slope,
Chronos zero-shot panel mean as a horizontal reference band.
Asserts the recomputed means equal ledger L-027 before rendering.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from house_style import apply_house_style, figsize, save_figure, panel_label
from naming import tier_colors, DOMAIN_DISPLAY

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOMAINS = {
    "pm25": ("results/v1/pm25_panel/canonical_cities.csv",
             "results/v1/pm25_panel/causal_ablation_cities.csv",
             {"pf": 0.662, "ca": 0.692, "ch": 0.662},          # L-027
             "$P$ = 0.33", "$P$ = 0.08"),
    "weather": ("results/v1/weather_panel/canonical_cities.csv",
                "results/v1/weather_panel/causal_ablation_cities.csv",
                {"pf": 0.533, "ca": 0.745, "ch": 0.792},       # L-027
                "$P$ < 0.001", "$P$ = 0.29"),
}


def load_domain(canon_p, causal_p):
    cn = pd.read_csv(os.path.join(ROOT, canon_p))
    ca = pd.read_csv(os.path.join(ROOT, causal_p))
    lg_pf = cn[cn.model == "lgbm_direct"].set_index("city").MASE
    lg_ca = ca[ca.model == "lgbm_direct"].set_index("city").MASE
    ch = cn[cn.model == "chronos"].set_index("city").MASE
    j = pd.DataFrame({"pf": lg_pf, "ca": lg_ca, "ch": ch}).dropna()
    assert len(j) == 29, f"expected 29 cities, got {len(j)}"
    return j


def main():
    apply_house_style(journal="nature")
    colors = tier_colors()
    sp, fm = colors["lgbm_direct"], colors["chronos"]

    fig, axes = plt.subplots(1, 2, figsize=figsize(width="double", aspect=0.55))

    for ax, (dom, (canon_p, causal_p, expect, p_pf, p_ca)), lab in zip(
            axes, DOMAINS.items(), "ab"):
        j = load_domain(canon_p, causal_p)
        # ledger guard: recomputed means must match L-027 to 3 dp
        for k, col in (("pf", "pf"), ("ca", "ca"), ("ch", "ch")):
            got = round(float(j[col].mean()), 3)
            assert abs(got - expect[k]) < 0.0011, f"{dom} {k}: {got} != L-027 {expect[k]}"

        x0, x1 = 0.0, 1.0
        for _, r in j.iterrows():
            ax.plot([x0, x1], [r.pf, r.ca], color=sp, alpha=0.22, linewidth=0.7,
                    zorder=2)
        ax.plot([x0, x1], [j.pf.mean(), j.ca.mean()], color=sp, linewidth=2.4,
                zorder=4, solid_capstyle="round")
        ax.scatter([x0, x1], [j.pf.mean(), j.ca.mean()], s=22, color=sp, zorder=5)

        # chronos reference band: mean +/- sem
        m, sem = j.ch.mean(), j.ch.std(ddof=1) / np.sqrt(len(j))
        ax.axhspan(m - sem, m + sem, color=fm, alpha=0.18, zorder=1)
        ax.axhline(m, color=fm, linewidth=1.2, zorder=3)
        ax.text(1.30, m, "Chronos-Bolt\n(zero-shot)", color=fm, fontsize=6.2,
                va="center", ha="left",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                          boxstyle="square,pad=0.12"))
        ax.text(0.02, 0.02, "$n$ = 29 cities", transform=ax.transAxes,
                fontsize=6.0, ha="left", va="bottom", color="#555555")

        # column p-values vs chronos (Wilcoxon, per L-027)
        ax.text(x0, ax.get_ylim()[0], "", fontsize=1)  # anchor
        ax.set_xticks([x0, x1])
        ax.set_xticklabels(["Perfect weather\nforecast", "Causal\n(last known)"],
                           fontsize=6.8)
        ymax = max(j.pf.max(), j.ca.max())
        # Reviewer fix: label explicitly as "vs Chronos" so a figure-only reader
        # cannot mistake this for a perfect-vs-causal comparison test.
        ax.text(x0, ymax * 1.03, f"{p_pf}\nvs Chronos", ha="center", fontsize=5.8,
                color="#222222", linespacing=1.15)
        ax.text(x1, ymax * 1.03, f"{p_ca}\nvs Chronos", ha="center", fontsize=5.8,
                color="#222222", linespacing=1.15)

        ax.set_xlim(-0.28, 1.62)
        ax.set_ylabel("MASE (fixed per-series scale)")
        ax.set_title(DOMAIN_DISPLAY[dom], fontsize=8, pad=10)
        panel_label(ax, lab, x=-0.22)

    fig.subplots_adjust(wspace=0.42)
    save_figure(fig, os.path.join(ROOT, "figures/main/F3_foresight_ablation"))
    print("saved -> figures/main/F3_foresight_ablation.{pdf,png}")


if __name__ == "__main__":
    main()
