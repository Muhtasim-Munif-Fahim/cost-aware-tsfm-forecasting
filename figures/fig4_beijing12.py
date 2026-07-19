"""Figure 4 — Beijing 12-station depth check (double column, two panels).

(a) Per-station paired dots: Chronos-Bolt vs LightGBM MASE, stations sorted by
    Chronos MASE; seasonal-naive as a light tick for scale.
(b) Specialist-to-foundation-model MASE ratio per station (same order), dashed
    parity line: shows the advantage is uniform across the network.
Asserts 12/12 before render.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from house_style import apply_house_style, figsize, save_figure
from naming import tier_colors
from sanity_checks import check_min_rows

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results/v1/beijing/canonical_sweep12_hetero.csv")


def main():
    apply_house_style(journal="nature")
    colors = tier_colors()
    fm, sp, nv = colors["chronos"], colors["lgbm_direct"], colors["seasonal_naive"]

    df = pd.read_csv(SRC)
    piv = df.pivot_table(index="series", columns="model", values="MASE")
    piv.index = piv.index.str.replace("pm25:", "", regex=False)
    piv = piv.sort_values("chronos", ascending=False)
    check_min_rows(len(piv), 12, "Beijing station table")
    wins = int((piv["chronos"] < piv["lgbm_direct"]).sum())
    assert wins == 12, f"expected 12/12 chronos wins, got {wins}"   # L-026

    fig, (ax, axr) = plt.subplots(
        1, 2, figsize=figsize(width="double", aspect=0.42), sharey=True,
        gridspec_kw={"width_ratios": [1.5, 1.0], "wspace": 0.06})
    y = np.arange(len(piv))

    # ------------------------------------------------------------- (a) dumbbell
    ax.hlines(y, piv["chronos"], piv["lgbm_direct"], color="#bbbbbb",
              linewidth=0.8, zorder=2)
    ax.scatter(piv["chronos"], y, s=17, color=fm, zorder=4,
               label="Chronos-Bolt (zero-shot)")
    ax.scatter(piv["lgbm_direct"], y, s=17, color=sp, zorder=3,
               label="LightGBM (specialist)")
    ax.scatter(piv["seasonal_naive"], y, marker="|", s=42, color=nv,
               linewidths=1.1, zorder=2, label="Seasonal-naïve floor")

    ax.set_yticks(y)
    ax.set_yticklabels(piv.index, fontsize=6.4)
    ax.set_xscale("log")
    ax.set_xticks([0.15, 0.3, 0.6, 1.2])
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.minorticks_off()
    ax.set_xlim(0.13, 1.45)
    ax.set_xlabel("MASE (fixed per-series scale; log axis)")
    ax.tick_params(axis="y", length=0)
    ax.margins(y=0.04)
    ax.set_title("a", loc="left", fontweight="bold", fontsize=8)
    handles, labels = ax.get_legend_handles_labels()

    # ---------------------------------------------- (b) specialist/FM MASE ratio
    ratio = piv["lgbm_direct"] / piv["chronos"]
    axr.barh(y, ratio, height=0.62, color=sp, alpha=0.85, zorder=3)
    axr.axvline(1.0, color="#555555", linewidth=0.9, linestyle="--", zorder=4)
    axr.text(1.0, len(piv) - 0.15, " parity", fontsize=5.8, color="#555555",
             ha="left", va="top")
    for yi, r in zip(y, ratio):
        axr.text(r + 0.04, yi, f"{r:.2f}", va="center", fontsize=5.6,
                 color="#333333")
    axr.set_xlim(0, ratio.max() * 1.22)
    axr.set_xlabel("Specialist ÷ zero-shot MASE ratio")
    axr.tick_params(axis="y", length=0)
    axr.margins(y=0.04)
    axr.set_title("b", loc="left", fontweight="bold", fontsize=8)

    # Single shared legend below both panels (keeps it clear of the dumbbell
    # dots and the naïve floor ticks in panel a).
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=6.4, handletextpad=0.4, columnspacing=1.4,
               bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.20)

    save_figure(fig, os.path.join(ROOT, "figures/main/F4_beijing12"))
    print("saved -> figures/main/F4_beijing12.{pdf,png}")


if __name__ == "__main__":
    main()
