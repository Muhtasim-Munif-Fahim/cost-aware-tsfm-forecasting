"""Figure 6 — how the specialist's advantage depends on covariate quality (R1.2).

Reviewer 1 asked for an intermediate experiment between the manuscript's two covariate
scenarios (last-known at the forecast origin, and perfect foresight). Rather than a single
intermediate point, this plots the whole covariate-quality axis and marks where each
scenario falls on it.

x-axis: alpha, the multiplier on measured 24 h numerical-weather-prediction error injected
into the specialist's future covariates (0 = perfect foresight; 1 = real-NWP error level,
calibrated per variable against the Open-Meteo previous-model-runs archive).
y-axis: panel-mean MASE.

The covariate-free foundation-model tiers are flat in alpha by construction; plotting them
doubles as a visual leakage check, since any slope would mean the degradation reached a
path it should not have.

Only cells covering the full panel are plotted -- a partially completed sweep cell would
otherwise be averaged over a different city set and silently shift the curve.
"""
from __future__ import annotations

import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd

from house_style import apply_house_style, figsize, save_figure, panel_label
from naming import SHORT, DOMAIN_DISPLAY, tier_colors
import palettes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOMAINS = ["pm25", "weather"]
LINES = ["lgbm_direct", "chronos", "timesfm"]
LABEL = {**SHORT, "timesfm": "TimesFM"}


def load_cells(domain):
    """{(alpha, seed): per-city MASE frame} for cells covering the whole panel."""
    out, partial = {}, {}
    pat = os.path.join(ROOT, "results", "v1", f"{domain}_panel", "sweep_nwp_a*_cities.csv")
    for f in sorted(glob.glob(pat)):
        m = re.search(r"sweep_nwp_a([\d.]+)_s(\d+)_cities", os.path.basename(f))
        if not m:
            continue
        df = pd.read_csv(f)
        piv = df.pivot_table(index="city", columns="model", values="MASE")
        key = (float(m.group(1)), int(m.group(2)))
        (out if len(piv) >= 29 else partial)[key] = piv
    return out, partial


def causal_reference(domain, model="lgbm_direct"):
    p = os.path.join(ROOT, "results", "v1", f"{domain}_panel", "causal_primary_cities.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    return d[d.model == model].groupby("city").MASE.mean()


def main():
    apply_house_style()
    # Take tier colours from the shared house map, NOT literal hex: F2-F5 draw chronos
    # in the primary colour and lgbm_direct in the secondary, and this figure used to
    # hardcode them the other way round, so the same tier changed colour between figures.
    colors = dict(tier_colors())
    colors["timesfm"] = palettes.NATURE["green"]

    fig, axes = plt.subplots(1, 2, figsize=figsize(width="double", aspect=0.42))
    any_data = False

    for ax, dom in zip(axes, DOMAINS):
        cells, partial = load_cells(dom)
        if partial:
            print(f"[{dom}] EXCLUDED incomplete cells: "
                  + ", ".join(f"alpha={a} seed={s} ({len(v)}/29)" for (a, s), v in partial.items()))
        if not cells:
            ax.text(0.5, 0.5, "sweep not yet available", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(DOMAIN_DISPLAY[dom])
            continue
        any_data = True

        common = None
        for v in cells.values():
            common = v.index if common is None else common.intersection(v.index)
        cau = causal_reference(dom)
        if cau is not None:
            common = common.intersection(cau.index)

        alphas = sorted({a for a, _ in cells})
        for model in LINES:
            ys, es = [], []
            for a in alphas:
                vals = [v[model].reindex(common).mean() for (aa, _), v in cells.items()
                        if aa == a and model in v]
                ys.append(np.mean(vals) if vals else np.nan)
                es.append(np.std(vals) if len(vals) > 1 else 0.0)
            ax.errorbar(alphas, ys, yerr=es, marker="o", ms=3.5, lw=1.6, capsize=2,
                        color=colors.get(model), label=LABEL.get(model, model), zorder=3)

        if cau is not None:
            ax.axhline(cau.reindex(common).mean(), ls=":", lw=1.2, color=colors["lgbm_direct"],
                       alpha=0.9, zorder=2)
            ax.text(0.995, cau.reindex(common).mean(), "last-known", va="bottom",
                    ha="right", fontsize=6, color=colors["lgbm_direct"],
                    transform=mtransforms.blended_transform_factory(ax.transAxes, ax.transData),
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.8), zorder=4)

        # Labels sit INSIDE the axes on a blended transform: at ylim top they landed in the
        # title's space and collided with it.
        # Reserve a clear band at the top: without it the rotated labels below run
        # straight through the flat foundation-model lines in the temperature panel.
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.16 * (hi - lo))
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        for xv, lab in [(0.0, "perfect foresight"), (1.0, "real NWP")]:
            if xv <= max(alphas):
                ax.axvline(xv, ls="--", lw=0.8, color="0.55", zorder=1)
                ax.text(xv, 0.97, lab, transform=trans, rotation=90, fontsize=6,
                        ha="right", va="top", color="0.35",
                        bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.8))

        ax.set_xlabel(r"covariate error level $\alpha$  (0 = perfect, 1 = real NWP)")
        ax.set_ylabel("panel-mean MASE")
        ax.set_title(f"{DOMAIN_DISPLAY[dom]}  ($n$ = {len(common)} cities)", pad=8)
        if len(alphas) > 1:
            ax.set_xlim(-0.08 * max(alphas), 1.08 * max(alphas))
        ax.grid(alpha=0.25, lw=0.5)

    if any_data:
        # Figure-level legend in a reserved strip, as in F2 and F5. The per-axes
        # loc="best" it replaced floated over the PM2.5 data and moved between renders.
        import matplotlib.lines as mlines
        handles = [mlines.Line2D([], [], color=colors[m], marker="o", ms=3.5, lw=1.6,
                                 label=LABEL.get(m, m)) for m in LINES]
        handles.append(mlines.Line2D([], [], color=colors["lgbm_direct"], ls=":", lw=1.2,
                                     label="LightGBM, last-known covariates"))
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
                   fontsize=6.2, bbox_to_anchor=(0.5, -0.012),
                   columnspacing=1.0, handletextpad=0.4)
        panel_label(axes[0], "a")
        panel_label(axes[1], "b")
        # Reserve the strip the legend sits in, as F2 does; without it the legend
        # is drawn straight over both panels' x-axis labels.
        fig.subplots_adjust(wspace=0.30, bottom=0.24)
        save_figure(fig, os.path.join(ROOT, "figures", "main", "F6_covariate_quality"))
        print("saved -> figures/main/F6_covariate_quality.{pdf,png}")
    else:
        print("no complete sweep cells yet; figure not written")
    plt.close(fig)


if __name__ == "__main__":
    main()
