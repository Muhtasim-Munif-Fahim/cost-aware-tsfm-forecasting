"""
House style for journal-quality matplotlib figures.

    from house_style import apply_house_style, figsize, save_figure
    from palettes import ACTIVE_ROLES
    apply_house_style(journal="lancet")

apply_house_style does three things:
  1. Sets fonts, sizes, spines, ticks, legend, savefig DPI.
  2. Sets the color cycler (axes.prop_cycle) to the journal's palette.
     This is the line that makes plain plt.plot() produce journal colors.
  3. Rebinds palettes.ACTIVE_ROLES so role names map to journal hex codes.
"""

from __future__ import annotations
import matplotlib as mpl
from cycler import cycler
from typing import Literal, Tuple

import palettes as _palettes

Journal = Literal[
    "lancet", "lancet_global_health", "lancet_digital_health",
    "lancet_infectious_diseases",
    "bmj", "bmj_global_health", "bmj_open",
    "nature", "nature_medicine", "nature_communications",
    "nature_digital_medicine",
    "nejm", "jama", "annals",
    "plos_medicine", "plos_one", "plos_computational_biology",
    "science",
    "icml", "neurips", "iclr",
    "generic",
]


# Column widths in inches (mm x 0.0393701). Sources: 2024-2025 author guides.
COLUMN_WIDTHS_INCH = {
    "lancet":                    {"single": 3.346, "1.5col": 5.512, "double": 7.087},
    "lancet_global_health":      {"single": 3.346, "1.5col": 5.512, "double": 7.087},
    "lancet_digital_health":     {"single": 3.346, "1.5col": 5.512, "double": 7.087},
    "lancet_infectious_diseases":{"single": 3.346, "1.5col": 5.512, "double": 7.087},
    "bmj":                       {"single": 3.346, "double": 6.929},
    "bmj_global_health":         {"single": 3.346, "double": 6.929},
    "bmj_open":                  {"single": 3.346, "double": 6.929},
    "nature":                    {"single": 3.504, "double": 7.205},
    "nature_medicine":           {"single": 3.504, "double": 7.205},
    "nature_communications":     {"single": 3.504, "double": 7.205},
    "nature_digital_medicine":   {"single": 3.504, "double": 7.205},
    "nejm":                      {"single": 3.346, "double": 6.929},
    "jama":                      {"single": 3.346, "double": 6.929},
    "annals":                    {"single": 3.346, "double": 6.929},
    "plos_medicine":             {"single": 3.27,  "double": 6.83},
    "plos_one":                  {"single": 3.27,  "double": 6.83},
    "plos_computational_biology":{"single": 3.27,  "double": 6.83},
    "science":                   {"single": 2.244, "1.5col": 4.567, "double": 7.283},
    "icml":                      {"single": 3.25,  "double": 6.75},
    "neurips":                   {"single": 3.25,  "double": 6.75},
    "iclr":                      {"single": 3.25,  "double": 6.75},
    "generic":                   {"single": 3.346, "double": 7.087},
}


_ACTIVE_JOURNAL: str = "generic"


def apply_house_style(
    journal: Journal = "generic",
    font_family: str = "sans-serif",
) -> None:
    """Apply rcParams + color cycler + ACTIVE_ROLES for the journal.

    Conference venues (icml/neurips/iclr) default to serif if not overridden.
    """
    global _ACTIVE_JOURNAL
    _ACTIVE_JOURNAL = journal

    # Conferences default to Times serif unless caller overrides.
    if journal in {"icml", "neurips", "iclr"} and font_family == "sans-serif":
        font_family = "serif"

    if font_family == "sans-serif":
        font_list = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
    else:
        font_list = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]

    palette_cycle = _palettes.palette_for(journal)

    rc = {
        # Font
        "font.family":      font_family,
        "font.size":        8,
        "axes.titlesize":   9,
        "axes.labelsize":   8,
        "xtick.labelsize":  7,
        "ytick.labelsize":  7,
        "legend.fontsize":  7,
        "figure.titlesize": 9,
        "mathtext.fontset": "dejavusans",

        # Lines / markers
        "lines.linewidth":      1.2,
        "lines.markersize":     4,
        "lines.markeredgewidth":0.8,
        "axes.linewidth":       0.6,
        "patch.linewidth":      0.6,

        # Ticks
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.major.width":0.6,
        "ytick.major.width":0.6,
        "xtick.direction":  "out",
        "ytick.direction":  "out",
        "xtick.major.pad":  2.0,
        "ytick.major.pad":  2.0,

        # Spines: top + right off by default
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.spines.left":  True,
        "axes.spines.bottom":True,
        "axes.edgecolor":    "#222222",
        "axes.labelcolor":   "#222222",
        "xtick.color":       "#222222",
        "ytick.color":       "#222222",
        "text.color":        "#222222",

        # Grid off; on only for explicit reference axes
        "axes.grid":      False,
        "grid.linewidth": 0.4,
        "grid.color":     "#dddddd",
        "grid.alpha":     0.6,

        # Legend
        "legend.frameon":         False,
        "legend.handlelength":    1.5,
        "legend.handletextpad":   0.5,
        "legend.borderpad":       0.3,
        "legend.borderaxespad":   0.4,
        "legend.columnspacing":   1.2,
        "legend.labelspacing":    0.3,

        # Figure
        "figure.dpi":          300,
        "savefig.dpi":         600,
        "savefig.bbox":        "tight",
        "savefig.pad_inches":  0.05,
        "savefig.transparent": False,
        "figure.facecolor":    "white",
        "axes.facecolor":      "white",

        # PDF embedding -- TrueType so production can edit
        "pdf.fonttype": 42,
        "ps.fonttype":  42,
        "svg.fonttype": "none",

        # Hatching
        "hatch.linewidth": 0.5,

        # *** THE FIX ***: bind the color cycler to the journal palette.
        "axes.prop_cycle": cycler(color=palette_cycle),
    }
    if font_family == "sans-serif":
        rc["font.sans-serif"] = font_list
    else:
        rc["font.serif"] = font_list

    mpl.rcParams.update(rc)

    # Rebind roles in the palettes module so every importer sees the new map.
    _palettes._set_active(journal)


def figsize(journal: Journal = None, width: str = "single", aspect: float = 0.75) -> Tuple[float, float]:
    """Return (w, h) in inches for a journal column width and aspect ratio."""
    journal = journal or _ACTIVE_JOURNAL
    if journal not in COLUMN_WIDTHS_INCH:
        journal = "generic"
    options = COLUMN_WIDTHS_INCH[journal]
    if width not in options:
        width = "single" if "single" in options else next(iter(options))
    w = options[width]
    return (w, w * aspect)


def panel_label(ax, label: str, x: float = -0.18, y: float = 1.05, **kwargs) -> None:
    """Place a bold panel label (a, b, c, ... for Nature) at upper-left of an axes."""
    kwargs.setdefault("fontsize", 9)
    kwargs.setdefault("fontweight", "bold")
    kwargs.setdefault("ha", "left")
    kwargs.setdefault("va", "bottom")
    ax.text(x, y, label, transform=ax.transAxes, **kwargs)


def save_figure(fig, path_stem: str, dpi_png: int = 600) -> None:
    """Save PDF (vector, for journal) + PNG (for review)."""
    from pathlib import Path
    Path(path_stem).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{path_stem}.pdf", format="pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(f"{path_stem}.png", format="png", dpi=dpi_png,
                bbox_inches="tight", pad_inches=0.05)


def clean_axis(ax, left: bool = True, bottom: bool = True) -> None:
    """Remove top/right spines; optionally hide left or bottom."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if not left:
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False, labelleft=False)
    if not bottom:
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(bottom=False, labelbottom=False)


def active_journal() -> str:
    return _ACTIVE_JOURNAL
