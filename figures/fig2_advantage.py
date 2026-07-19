"""Figure 2 — per-city FM advantage (specialist − foundation model MASE), both domains.

Panel a: PM2.5 under the CAUSAL-primary configuration (matches Table 1; supervisor
review B9 — the old panel used the perfect-foresight panel, contradicting the
main-result config). Sign test P = 0.024 favouring the FM (21/29), Wilcoxon n.s.
Panel b: temperature (face-value specialist shift; perfect-foresight caveat in caption).
Encoding: horizontal diverging stem-and-dot per city, sorted by advantage within panel;
rich tier = filled marker, scarce = open marker; dashed zero reference; panel-level
sign-test p from the ledger values annotated.
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

PANELS = [
    ("pm25", os.path.join(ROOT, "results/v1/pm25_panel/causal_primary_cities.csv"),
     "$P$ = 0.024 (sign test), 21/29"),   # L-040; causal-primary, matches Table 1
    ("weather", os.path.join(ROOT, "results/v1/weather_panel/canonical_cities.csv"),
     "$P$ < 0.001 (sign test)*"),         # L-023; * = perfect-foresight covariates
]
PANEL_SUBTITLE = {"pm25": "causal covariates", "weather": "perfect-foresight covariates"}


def advantage_frame(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    piv = df.pivot_table(index=["city", "tier"], columns="model", values="MASE",
                         aggfunc="mean").reset_index()
    piv["advantage"] = piv["lgbm_direct"] - piv["chronos"]   # >0 => FM better
    return piv.sort_values("advantage").reset_index(drop=True)


def main():
    apply_house_style(journal="nature")
    colors = tier_colors()
    fm_color, sp_color = colors["chronos"], colors["lgbm_direct"]

    # Reviewer fix: 29 city labels per panel were cramped at aspect 0.62 / 5.6pt;
    # taller figure + larger label font trades a little compactness for readability.
    fig, axes = plt.subplots(1, 2, figsize=figsize(width="double", aspect=0.85),
                             sharex=True)

    for ax, (dom, path, ptext), lab in zip(axes, PANELS, "ab"):
        d = advantage_frame(path)
        assert len(d) == 29, f"{dom}: expected 29 cities, got {len(d)}"
        y = np.arange(len(d))
        adv = d["advantage"].values
        rich = (d["tier"] == "rich").values

        stem_c = [fm_color if a > 0 else sp_color for a in adv]
        ax.hlines(y, 0, adv, color=stem_c, linewidth=0.9, alpha=0.75, zorder=2)
        for mask, fc_open in ((rich, False), (~rich, True)):
            mc = [fm_color if a > 0 else sp_color for a in adv[mask]]
            ax.scatter(adv[mask], y[mask],
                       s=13, c=("white" if fc_open else mc),
                       edgecolors=mc, linewidths=0.8, zorder=3)

        ax.axvline(0, color="#444444", linestyle="--", linewidth=0.7, zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels(d["city"].str.replace("_", " ").str.title(), fontsize=6.4)
        ax.set_title(f"{DOMAIN_DISPLAY[dom]} ({PANEL_SUBTITLE[dom]})",
                     fontsize=7.5, pad=4)
        # p annotation in the top-left empty corner. In panel b (mostly negative
        # advantages) the shared x-axis puts x=0 near the right edge, so a
        # bottom/right label would collide with the dashed zero line; the short
        # top stems leave the upper-left clear in both panels.
        ax.text(0.02, 0.985, ptext, transform=ax.transAxes, fontsize=6.5,
                ha="left", va="top", color="#222222")
        panel_label(ax, lab, x=-0.30 if lab == "a" else -0.28)
        ax.tick_params(axis="y", length=0)
        ax.margins(y=0.015)
        ax.set_xlabel("")

    fig.supxlabel("MASE difference (specialist − foundation model);  > 0 favours FM",
                  fontsize=7.5, y=0.075)

    # two-part legend: colour = winner, fill = data tier
    import matplotlib.lines as mlines
    h = [
        mlines.Line2D([], [], marker="s", ls="none", mfc=fm_color, mec=fm_color,
                      markersize=4, label="Foundation model better"),
        mlines.Line2D([], [], marker="s", ls="none", mfc=sp_color, mec=sp_color,
                      markersize=4, label="Specialist better"),
        mlines.Line2D([], [], marker="o", ls="none", mfc="#666666", mec="#666666",
                      markersize=4, label="Rich tier (filled)"),
        mlines.Line2D([], [], marker="o", ls="none", mfc="white", mec="#666666",
                      markersize=4, label="Scarce tier (open)"),
    ]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.012),
               fontsize=6.2, frameon=False, columnspacing=1.0, handletextpad=0.4)
    fig.text(0.99, 0.005, "*with perfect-foresight covariates (see Fig. 3)",
             fontsize=5.8, ha="right", va="bottom", color="#555555",
             style="italic")
    fig.subplots_adjust(wspace=0.52, bottom=0.155)

    save_figure(fig, os.path.join(ROOT, "figures/main/F2_panel_advantage"))
    print("saved -> figures/main/F2_panel_advantage.{pdf,png}")


if __name__ == "__main__":
    main()
