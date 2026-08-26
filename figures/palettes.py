"""
Palettes for journal-quality figures.

Two ways to use:

1. As named roles (preferred). Always import ACTIVE_ROLES, never ROLES directly.
   ACTIVE_ROLES is rebound by apply_house_style() to the journal's palette.

       from house_style import apply_house_style
       from palettes import ACTIVE_ROLES
       apply_house_style(journal="lancet")
       color = ACTIVE_ROLES["primary"]

2. As cycler input. apply_house_style() also sets matplotlib's color cycler
   for the journal, so a bare `plt.plot()` already produces journal colors.

All palettes are colorblind-checked (Okabe-Ito is the safest categorical).
"""

from __future__ import annotations
from typing import Dict, List

# ---------------------------------------------------------------------------
# Okabe-Ito -- safest categorical palette for color-vision deficiencies.
# ---------------------------------------------------------------------------
OKABE_ITO = {
    "black":          "#000000",
    "orange":         "#E69F00",
    "sky_blue":       "#56B4E9",
    "bluish_green":   "#009E73",
    "yellow":         "#F0E442",
    "blue":           "#0072B2",
    "vermillion":     "#D55E00",
    "reddish_purple": "#CC79A7",
    "grey":           "#999999",
}
OKABE_ITO_ORDER: List[str] = [
    OKABE_ITO["blue"],
    OKABE_ITO["vermillion"],
    OKABE_ITO["bluish_green"],
    OKABE_ITO["orange"],
    OKABE_ITO["sky_blue"],
    OKABE_ITO["reddish_purple"],
    OKABE_ITO["yellow"],
    OKABE_ITO["grey"],
]

# ---------------------------------------------------------------------------
# Lancet -- extracted from published Lancet figures. Slightly desaturated.
# ---------------------------------------------------------------------------
LANCET = {
    "red":       "#AD002A",
    "blue":      "#00468B",
    "teal":      "#42B540",
    "orange":    "#ED0000",
    "purple":    "#925E9F",
    "yellow":    "#FDAF91",
    "grey":      "#1B1919",
    "lightblue": "#0099B4",
}
LANCET_ORDER: List[str] = [
    LANCET["blue"], LANCET["red"], LANCET["teal"], LANCET["purple"],
    LANCET["orange"], LANCET["lightblue"], LANCET["yellow"], LANCET["grey"],
]

# ---------------------------------------------------------------------------
# Nature -- characteristic muted tones.
# ---------------------------------------------------------------------------
NATURE = {
    "blue":   "#3B4992",
    "red":    "#EE0000",
    "green":  "#008B45",
    "purple": "#631879",
    "teal":   "#008280",
    "rose":   "#BB0021",
    "olive":  "#5F559B",
    "brown":  "#A20056",
    "grey":   "#808180",
}
NATURE_ORDER: List[str] = [
    NATURE["blue"], NATURE["red"], NATURE["green"], NATURE["purple"],
    NATURE["teal"], NATURE["rose"], NATURE["olive"], NATURE["brown"],
]

# ---------------------------------------------------------------------------
# NEJM -- high contrast.
# ---------------------------------------------------------------------------
NEJM = {
    "red":    "#BC3C29",
    "blue":   "#0072B5",
    "yellow": "#E18727",
    "teal":   "#20854E",
    "purple": "#7876B1",
    "olive":  "#6F99AD",
    "rose":   "#FFDC91",
    "pink":   "#EE4C97",
}
NEJM_ORDER: List[str] = [
    NEJM["blue"], NEJM["red"], NEJM["teal"], NEJM["yellow"],
    NEJM["purple"], NEJM["olive"], NEJM["pink"], NEJM["rose"],
]

# ---------------------------------------------------------------------------
# JAMA
# ---------------------------------------------------------------------------
JAMA = {
    "navy":   "#374E55",
    "amber":  "#DF8F44",
    "teal":   "#00A1D5",
    "red":    "#B24745",
    "green":  "#79AF97",
    "purple": "#6A6599",
    "olive":  "#80796B",
}
JAMA_ORDER: List[str] = [
    JAMA["navy"], JAMA["amber"], JAMA["teal"], JAMA["red"],
    JAMA["green"], JAMA["purple"], JAMA["olive"],
]

PALETTE_ORDERS: Dict[str, List[str]] = {
    "okabe_ito": OKABE_ITO_ORDER,
    "lancet":    LANCET_ORDER,
    "nature":    NATURE_ORDER,
    "nejm":      NEJM_ORDER,
    "jama":      JAMA_ORDER,
}

# ---------------------------------------------------------------------------
# Sequential / diverging colormap names (pass to cmap=).
# ---------------------------------------------------------------------------
SEQUENTIAL = {
    "default":     "viridis",
    "colorblind":  "cividis",
    "mortality":   "YlOrRd",
    "blue_serial": "Blues",
    "grey_serial": "Greys",
}
DIVERGING = {
    "default":     "RdBu_r",
    "colorblind":  "PiYG",
    "neutral_div": "PuOr",
}


# ---------------------------------------------------------------------------
# Journal routing
# ---------------------------------------------------------------------------
def _journal_family(journal: str) -> str:
    j = journal.lower()
    if j.startswith("lancet"):
        return "lancet"
    if j.startswith("nature") or j == "science":
        return "nature"
    if j.startswith("nejm"):
        return "nejm"
    if j.startswith("jama"):
        return "jama"
    if j.startswith("bmj") or j.startswith("plos") or j == "annals":
        return "okabe_ito"
    if j in {"icml", "neurips", "iclr"}:
        return "okabe_ito"
    return "okabe_ito"


def palette_for(journal: str) -> List[str]:
    """Hex list to feed matplotlib's axes.prop_cycle for this journal."""
    return PALETTE_ORDERS[_journal_family(journal)]


def roles_for(journal: str) -> Dict[str, str]:
    """Return a journal-specific role -> hex map.

    Roles are the stable names manuscript code references (primary,
    secondary, highlight, negative, positive, neutral, reference, ci_fill).
    Importing palette_for + roles_for is enough to produce a fully
    journal-styled manuscript with one source of truth.
    """
    family = _journal_family(journal)
    order = PALETTE_ORDERS[family]

    if family == "lancet":
        return {
            "primary":   LANCET["blue"],
            "secondary": LANCET["red"],
            "tertiary":  LANCET["teal"],
            "quaternary":LANCET["purple"],
            "highlight": LANCET["orange"],
            "neutral":   "#7F7F7F",
            "negative":  LANCET["red"],
            "positive":  LANCET["blue"],
            "reference": "#444444",
            "ci_fill":   "#CCCCCC",
        }
    if family == "nature":
        return {
            "primary":   NATURE["blue"],
            "secondary": NATURE["red"],
            "tertiary":  NATURE["green"],
            "quaternary":NATURE["purple"],
            "highlight": NATURE["rose"],
            "neutral":   "#7F7F7F",
            "negative":  NATURE["red"],
            "positive":  NATURE["blue"],
            "reference": "#444444",
            "ci_fill":   "#CCCCCC",
        }
    if family == "nejm":
        return {
            "primary":   NEJM["blue"],
            "secondary": NEJM["red"],
            "tertiary":  NEJM["teal"],
            "quaternary":NEJM["purple"],
            "highlight": NEJM["yellow"],
            "neutral":   "#7F7F7F",
            "negative":  NEJM["red"],
            "positive":  NEJM["blue"],
            "reference": "#444444",
            "ci_fill":   "#CCCCCC",
        }
    if family == "jama":
        return {
            "primary":   JAMA["navy"],
            "secondary": JAMA["red"],
            "tertiary":  JAMA["teal"],
            "quaternary":JAMA["green"],
            "highlight": JAMA["amber"],
            "neutral":   "#7F7F7F",
            "negative":  JAMA["red"],
            "positive":  JAMA["navy"],
            "reference": "#444444",
            "ci_fill":   "#CCCCCC",
        }
    # Default Okabe-Ito
    return {
        "primary":   OKABE_ITO["blue"],
        "secondary": OKABE_ITO["vermillion"],
        "tertiary":  OKABE_ITO["bluish_green"],
        "quaternary":OKABE_ITO["orange"],
        "highlight": OKABE_ITO["reddish_purple"],
        "neutral":   "#7F7F7F",
        "negative":  "#B2182B",
        "positive":  "#2166AC",
        "reference": "#444444",
        "ci_fill":   "#CCCCCC",
    }


# ACTIVE_ROLES is rebound by house_style.apply_house_style().
# Default to Okabe-Ito so importing palettes alone still works.
ACTIVE_ROLES: Dict[str, str] = roles_for("generic")

# Back-compat alias for old code that imported ROLES.
ROLES: Dict[str, str] = ACTIVE_ROLES


def _set_active(journal: str) -> None:
    """Called by house_style.apply_house_style. Do not call directly."""
    global ACTIVE_ROLES, ROLES
    ACTIVE_ROLES = roles_for(journal)
    ROLES = ACTIVE_ROLES


def categorical_palette(name: str = "okabe_ito", n: int = 8) -> List[str]:
    base = PALETTE_ORDERS.get(name, OKABE_ITO_ORDER)
    if n > len(base):
        raise ValueError(
            f"Palette '{name}' has {len(base)} colors; you asked for {n}. "
            "Reduce categories or add a second encoding (shape, hatch)."
        )
    return base[:n]
