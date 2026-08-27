"""Supplementary Fig. S2 — TOST equivalence forest.

All seven pre-specified 'tie' comparisons as paired MASE differences with TOST 90% CIs
against the +/-0.05-MASE equivalence margin (shaded band). Data: the canonical
equivalence artifact (results/v1/equivalence_tests.csv, ledger L-030). Positive
difference = the first-named (locally trained) model is worse, i.e. favours the
zero-shot foundation model. Equivalence is established only where the whole 90% CI
lies inside the band (filled marker): the perfect-foresight PM2.5 comparison.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import pandas as pd

from house_style import apply_house_style, figsize, save_figure
from naming import tier_colors

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARGIN = 0.05

# artifact comparison key -> (display label, group)
LABELS = {
    "pm25-perfect: lgbm_direct vs chronos (per city)":
        ("PM$_{2.5}$, perfect-foresight covariates", "specialist vs zero-shot"),
    "pm25-causal: lgbm_direct vs chronos (per city)":
        ("PM$_{2.5}$, causal covariates", "specialist vs zero-shot"),
    "weather-causal: lgbm_direct vs chronos (per city)":
        ("Temperature, causal covariates", "specialist vs zero-shot"),
    "E4 pm25: nas_transfer@0% vs chronos_zeroshot":
        ("E4 transfer, 0% fine-tune", "E4 transfer vs zero-shot"),
    "E4 pm25: nas_transfer@1% vs chronos_zeroshot":
        ("E4 transfer, 1% fine-tune", "E4 transfer vs zero-shot"),
    "E4 pm25: nas_transfer@10% vs chronos_zeroshot":
        ("E4 transfer, 10% fine-tune", "E4 transfer vs zero-shot"),
    "E4 pm25: nas_transfer@100% vs chronos_zeroshot":
        ("E4 transfer, 100% fine-tune", "E4 transfer vs zero-shot"),
}


def main():
    apply_house_style(journal="nature")
    colors = tier_colors()
    c_eq, c_ne = colors["chronos"], "#555555"

    df = pd.read_csv(os.path.join(ROOT, "results/v1/equivalence_tests.csv"))
    # Select this figure's rows rather than asserting the file contains only them.
    # equivalence_tests.csv is a superset from the revision onward: it also carries
    # foundation-model-family comparisons, which answer a different question and do not
    # belong in a figure captioned "the seven pre-specified tie comparisons".
    missing = [c for c in LABELS if c not in set(df.comparison)]
    assert not missing, f"equivalence artifact is missing expected comparisons: {missing}"
    df = df.set_index("comparison").loc[list(LABELS)].reset_index()

    import matplotlib.transforms as mtransforms

    fig, ax = plt.subplots(figsize=figsize(width="single", aspect=0.9))

    n = len(df)
    # equivalence margin band (full height) + reference lines bounded to the row
    # region so they do not bisect the margin label sitting in the band below.
    line_lo, line_hi = -0.2, n - 0.55
    ax.axvspan(-MARGIN, MARGIN, color="#e8e8e8", zorder=0)
    ax.plot([0, 0], [line_lo, line_hi], color="#444444", linestyle="--",
            linewidth=0.7, zorder=1)
    for x in (-MARGIN, MARGIN):
        ax.plot([x, x], [line_lo, line_hi], color="#999999", linewidth=0.6,
                zorder=1)

    ys = list(range(n - 1, -1, -1))          # first row on top
    for y, (_, r) in zip(ys, df.iterrows()):
        eq = bool(r.equivalent_at_0p05)
        col = c_eq if eq else c_ne
        ax.plot([r.tost90_lo, r.tost90_hi], [y, y], color=col, linewidth=1.6,
                solid_capstyle="butt", zorder=3)
        ax.plot(r.mean_diff, y, marker="o", markersize=4.2, mfc=(col if eq else "white"),
                mec=col, mew=0.9, zorder=4)

    ax.set_yticks(ys)
    ax.set_yticklabels([LABELS[c][0] for c in df.comparison], fontsize=7)

    # x-range: data-driven, with a dedicated clear column at right for P values
    ci_hi = float(df.tost90_hi.max())        # 0.129
    x_hi = ci_hi + 0.068
    ax.set_xlim(-0.13, x_hi)
    for y, (_, r) in zip(ys, df.iterrows()):
        ax.text(x_hi - 0.004, y, f"$P$ = {r.p_tost:.3f}", fontsize=6,
                ha="right", va="center", color="#333333")

    # group separator between the 3 tie claims and the 4 E4 budgets
    ax.axhline(3.5, color="#cccccc", linewidth=0.6, zorder=1)

    # margin label in the empty band region below the last row
    ax.set_ylim(-0.85, n - 0.45)
    ax.text(0, -0.62, f"equivalence margin ±{MARGIN} MASE", fontsize=6.2,
            ha="center", va="center", color="#777777")

    ax.set_xlabel("Paired MASE difference, TOST 90% CI\n(positive favours the zero-shot foundation model)",
                  fontsize=7)

    import matplotlib.lines as mlines
    h = [mlines.Line2D([], [], marker="o", ls="-", color=c_eq, mfc=c_eq, markersize=4,
                       label="Equivalent at ±0.05 (CI inside band)"),
         mlines.Line2D([], [], marker="o", ls="-", color=c_ne, mfc="white", mec=c_ne,
                       markersize=4, label="Equivalence not established")]
    fig.legend(handles=h, loc="lower center", ncol=2, fontsize=6.2, frameon=False,
               bbox_to_anchor=(0.55, -0.005), handletextpad=0.5, columnspacing=1.2)

    fig.subplots_adjust(left=0.32, right=0.98, bottom=0.22, top=0.97)
    save_figure(fig, os.path.join(ROOT, "figures/main/SF2_equivalence"))
    print("saved -> figures/main/SF2_equivalence.{pdf,png}")


if __name__ == "__main__":
    main()
