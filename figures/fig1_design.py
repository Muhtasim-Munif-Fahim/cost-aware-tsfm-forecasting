"""Figure 1 — study design schematic (typographic flow, no wireframe boxes).

Three columns joined by hairline rules: data panel -> strategies -> evaluation.
All counts hardcoded from the locked design (no CSV numbers).
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt

from house_style import apply_house_style, figsize, save_figure
from naming import tier_colors

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INK = "#222222"
MUTE = "#666666"


def col_header(ax, x, text, rule_w=0.26):
    ax.text(x, 0.97, text.upper(), fontsize=7.2, fontweight="bold", color=INK,
            ha="left", va="top")
    ax.plot([x, x + rule_w], [0.935, 0.935], color=INK, linewidth=0.8)


def item(ax, x, y, title, sub, color=INK):
    ax.text(x, y, title, fontsize=6.8, fontweight="bold", color=color,
            ha="left", va="top")
    ax.text(x, y - 0.035, sub, fontsize=6.0, color=MUTE, ha="left", va="top",
            linespacing=1.25)


def main():
    apply_house_style(journal="nature")
    colors = tier_colors()

    fig, ax = plt.subplots(figsize=figsize(width="double", aspect=0.52))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---- column 1: data ------------------------------------------------
    # header rules span the full column width for a uniform grid
    x1 = 0.015
    col_header(ax, x1, "Data", rule_w=0.30)
    item(ax, x1, 0.86, "29-city OpenAQ panel",
         "hourly PM$_{2.5}$, 14 data-rich + 15 data-scarce\ncities; usable windows 2.5k–19.8k h,\nquality-gated (§Methods)")
    item(ax, x1, 0.64, "Same cities, second domain",
         "2-m temperature (Open-Meteo reanalysis),\nclipped to each city's PM$_{2.5}$ window\n→ isolates the domain effect")
    item(ax, x1, 0.42, "Beijing depth set",
         "UCI multi-site archive: 12 stations,\n2013–2017, meteorology included")
    item(ax, x1, 0.22, "Pre-registration",
         "analysis plan locked before the campaign;\nevery deviation logged; per-claim ledger\nlinks each number to its artifact")

    # ---- column 2: strategies ------------------------------------------
    x2 = 0.365
    col_header(ax, x2, "Five forecasting strategies", rule_w=0.31)
    item(ax, x2, 0.86, "Chronos-Bolt (zero-shot)",
         "48M-param time-series foundation model;\nno training, 4-week context", colors["chronos"])
    item(ax, x2, 0.70, "Chronos-Bolt + covariates",
         "FM forecasts the residual of a ridge\ncovariate model", colors["chronos_cov"])
    item(ax, x2, 0.54, "LightGBM (tuned specialist)",
         "direct multi-horizon, lag/calendar/weather\nfeatures; retrained per fold", colors["lgbm_direct"])
    item(ax, x2, 0.38, "NAS-GRU (searched specialist)",
         "2×GRU-128 from the published Green-NAS\nsearch; 5 seeds; + transfer variant (E4)", colors["nas_gru"])
    item(ax, x2, 0.22, "Seasonal-naïve floor",
         "168-h persistence", colors["seasonal_naive"])

    # ---- column 3: evaluation -------------------------------------------
    x3 = 0.72
    col_header(ax, x3, "Evaluation", rule_w=0.27)
    item(ax, x3, 0.86, "Rolling-origin backtest",
         "6 folds × 24 h (48 h supplement);\nMASE on a fixed per-series scale")
    item(ax, x3, 0.67, "Covariate realism ablation",
         "perfect-foresight vs causal (last-known)\nweather covariates — Fig. 3")
    item(ax, x3, 0.48, "Statistical rigor",
         "per-city Diebold–Mariano (HLN, FDR);\npanel sign/Wilcoxon; Friedman–Nemenyi;\nsplit-conformal intervals; TOST equivalence")
    item(ax, x3, 0.24, "Measured energy → decision rule",
         "codecarbon J per 1,000 forecasts;\nargmin MASE + λ · USD/1k over\nhistory × cost penalty λ — Fig. 6")

    # hairline connectors between columns, centred on the content band
    for xa, xb in ((0.318, 0.358), (0.678, 0.713)):
        ax.annotate("", xy=(xb, 0.52), xytext=(xa, 0.52),
                    arrowprops=dict(arrowstyle="-|>", color=MUTE, linewidth=0.9,
                                    shrinkA=0, shrinkB=0))

    save_figure(fig, os.path.join(ROOT, "figures/main/F1_design"))
    print("saved -> figures/main/F1_design.{pdf,png}")


if __name__ == "__main__":
    main()
