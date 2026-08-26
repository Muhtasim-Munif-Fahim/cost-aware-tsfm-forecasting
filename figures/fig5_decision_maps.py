"""Figure 6 — cost-adjusted decision winner maps (2 domains x 3 cities).

Categorical heatmaps: training-history regime (rows) x cost-penalty coefficient
lambda (cols); cell colour = tier minimising MASE + lambda * USD/1k. Cell letters
duplicate the colour coding for grayscale readers (supervisor review, colour-only
encoding). Skipped regimes (insufficient history) hatched grey. Shared legend below.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from house_style import apply_house_style, figsize, save_figure, panel_label
from naming import tier_colors, SHORT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CITIES = ["beijing", "seoul", "nairobi"]
DOMAINS = ["pm25", "weather"]
CITY_TITLE = {"beijing": "Beijing", "seoul": "Seoul (rich)", "nairobi": "Nairobi (scarce)"}
DOM_TITLE = {"pm25": "PM$_{2.5}$", "weather": "Temperature"}
WTPS = ["wtp=0", "wtp=500", "wtp=1500", "wtp=5000", "wtp=20000"]
WTP_LABELS = ["0", "500", "1.5k", "5k", "20k"]
ALL_WEEKS = [4, 12, 26, 52, 104]
# Grayscale-safe letter per tier, duplicated in the legend and the caption.
LETTER = {"chronos": "Z", "chronos_cov": "C", "lgbm_direct": "L",
          "nas_gru": "N", "seasonal_naive": "S"}


def _text_color(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#111111" if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else "white"


def main(prefix="causal", out_name="F5_decision_maps"):
    apply_house_style(journal="nature")
    colors = tier_colors()
    tiers_present = set()

    fig, axes = plt.subplots(2, 3, figsize=figsize(width="double", aspect=0.62),
                             sharex=True, sharey=True)

    panel_iter = iter("abcdef")
    for r, dom in enumerate(DOMAINS):
        for c, city in enumerate(CITIES):
            ax = axes[r, c]
            path = os.path.join(ROOT, f"results/v1/regime/{prefix}_{city}_{dom}_decision.csv")
            df = pd.read_csv(path).set_index("train_weeks")
            for i, W in enumerate(ALL_WEEKS):
                for k, wtp in enumerate(WTPS):
                    if W in df.index:
                        tier = df.loc[W, wtp]
                        tiers_present.add(tier)
                        ax.add_patch(plt.Rectangle((k, i), 1, 1,
                                                   facecolor=colors[tier],
                                                   edgecolor="white", linewidth=1.0))
                        ax.text(k + 0.5, i + 0.5, LETTER[tier], ha="center",
                                va="center", fontsize=5.2,
                                color=_text_color(colors[tier]), zorder=4)
                    else:
                        ax.add_patch(plt.Rectangle((k, i), 1, 1, facecolor="#eeeeee",
                                                   edgecolor="white", linewidth=1.0,
                                                   hatch="///"))
            ax.set_xlim(0, len(WTPS))
            ax.set_ylim(0, len(ALL_WEEKS))
            ax.set_xticks(np.arange(len(WTPS)) + 0.5)
            ax.set_xticklabels(WTP_LABELS)
            ax.set_yticks(np.arange(len(ALL_WEEKS)) + 0.5)
            ax.set_yticklabels([str(w) for w in ALL_WEEKS], fontsize=6.5)
            ax.tick_params(length=0)
            for s in ("top", "right", "left", "bottom"):
                ax.spines[s].set_visible(False)
            ax.set_aspect("auto")
            if r == 0:
                ax.set_title(CITY_TITLE[city], fontsize=7.5, pad=3)
            if c == 0:
                ax.set_ylabel(f"{DOM_TITLE[dom]}\ntraining history (weeks)", fontsize=7)
            panel_label(ax, next(panel_iter), x=-0.14, y=1.02)

    fig.supxlabel(r"Cost-penalty coefficient $\lambda$ (MASE per US\$ per 1,000 forecasts)",
                  fontsize=7.2, y=0.115)

    order = ["chronos", "chronos_cov", "lgbm_direct", "nas_gru", "seasonal_naive"]
    handles = [mpatches.Patch(facecolor=colors[t], label=f"{SHORT[t]} ({LETTER[t]})")
               for t in order if t in tiers_present]
    handles.append(mpatches.Patch(facecolor="#eeeeee", hatch="///",
                                  label="insufficient history"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, 0.005), fontsize=6.4, frameon=False,
               handlelength=1.2, columnspacing=1.0)
    fig.subplots_adjust(hspace=0.22, wspace=0.12, bottom=0.17)

    save_figure(fig, os.path.join(ROOT, f"figures/main/{out_name}"))
    print(f"saved -> figures/main/{out_name}.{{pdf,png}} (prefix={prefix})")


if __name__ == "__main__":
    import sys
    if "--perfect" in sys.argv:
        main(prefix="canonical", out_name="SF_decision_maps_perfect")  # supplementary upper bound
    else:
        main(prefix="causal", out_name="F5_decision_maps")             # main: causal-primary
