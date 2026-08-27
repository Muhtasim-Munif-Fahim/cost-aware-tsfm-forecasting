"""Figure 5 — E4 crux: transfer learning vs zero-shot across fine-tune budgets.

x = nominal fine-tune fraction (categorical 0/1/10/100%), y = MASE (mean across
15 scarce cities). NAS-GRU transfer: mean with band = ±sd across cities of the
per-city seed-means. LightGBM refit: dashed. Chronos zero-shot: horizontal
reference. Verdict annotations from the Holm-corrected Wilcoxon file.
Asserts recomputed means equal ledger L-009.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from house_style import apply_house_style, figsize, save_figure
from naming import tier_colors

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results/v1/e4_transfer/canonical_pm25_results.csv")

L009 = {  # ledger guard values
    "chronos": 0.843,
    "nas": {0.0: 0.899, 1.0: 0.915, 10.0: 0.888, 100.0: 0.876},
    "lgbm": {1.0: 0.941, 10.0: 0.944, 100.0: 0.858},
}


def main():
    apply_house_style(journal="nature")
    colors = tier_colors()
    fm, sp, tr = colors["chronos"], colors["lgbm_direct"], colors["nas_gru"]

    df = pd.read_csv(SRC)
    ch = df[df.strategy == "chronos_zeroshot"].groupby("city").MASE.mean()
    nas_city = (df[df.strategy == "nas_transfer"]
                .groupby(["city", "fraction"]).MASE.mean().unstack())
    lgbm_city = (df[df.strategy == "lgbm_refit"]
                 .groupby(["city", "fraction"]).MASE.mean().unstack())

    fracs = [0.0, 1.0, 10.0, 100.0]
    nas_m = nas_city.mean()
    nas_sd = nas_city.std(ddof=1)
    lgbm_m = lgbm_city.mean()

    # ledger guards (3 dp)
    assert abs(round(ch.mean(), 3) - L009["chronos"]) < 0.0011
    for f in fracs:
        assert abs(round(nas_m[f], 3) - L009["nas"][f]) < 0.0011, f"nas frac {f}"
    for f in [1.0, 10.0, 100.0]:
        assert abs(round(lgbm_m[f], 3) - L009["lgbm"][f]) < 0.0011, f"lgbm frac {f}"

    # Paired per-city differences vs the zero-shot reference: this is the
    # exact quantity the Holm-Wilcoxon tests operate on, and pairing removes
    # the large between-city variance that would otherwise swamp the bands.
    nas_diff = nas_city.sub(ch, axis=0)          # >0 => worse than zero-shot
    lgbm_diff = lgbm_city.sub(ch, axis=0)
    n_city = nas_diff.shape[0]

    x = np.arange(len(fracs))
    fig, ax = plt.subplots(figsize=figsize(width="single", aspect=0.85))

    ax.axhline(0.0, color=fm, linewidth=1.3, zorder=2)
    ax.text(0.03, 0.005, "Chronos-Bolt zero-shot reference ($\\Delta$ = 0, no target data)",
            color=fm, fontsize=6.2, ha="left", va="bottom",
            transform=ax.get_yaxis_transform())

    def band(series_diff, cols, color, ls, marker, ms, label):
        mu = series_diff.mean()
        sem = series_diff.std(ddof=1) / np.sqrt(series_diff.notna().sum())
        xs = [x[fracs.index(f)] for f in cols]
        ax.fill_between(xs, [mu[f] - sem[f] for f in cols],
                        [mu[f] + sem[f] for f in cols],
                        color=color, alpha=0.18, zorder=2, linewidth=0)
        ax.plot(xs, [mu[f] for f in cols], color=color, marker=marker,
                markersize=ms, linestyle=ls, zorder=4, label=label)

    band(nas_diff, fracs, tr, "-", "o", 3.5, "NAS-GRU, transfer + fine-tune")
    band(lgbm_diff, [1.0, 10.0, 100.0], sp, "--", "s", 3.0,
         "LightGBM, refit on budget")

    ax.set_xticks(x)
    ax.set_xticklabels(["0", "1", "10", "100"])
    ax.set_xlabel("Fine-tune budget (% of target-city history, nominal)")
    ax.set_ylabel("$\\Delta$MASE vs zero-shot (paired, 15 cities)")
    ax.legend(loc="upper left", fontsize=6.2, bbox_to_anchor=(0.01, 1.0))
    ax.text(0.02, 0.03,
            "No fraction differs significantly from zero-shot\n"
            "(Holm-corrected Wilcoxon, all $P$ ≥ 0.19); band = ±s.e.m.",
            transform=ax.transAxes, fontsize=6.2, va="bottom")
    ax.margins(x=0.04)

    save_figure(fig, os.path.join(ROOT, "figures/main/F5_e4_transfer"))
    print("saved -> figures/main/F5_e4_transfer.{pdf,png}")


if __name__ == "__main__":
    main()
