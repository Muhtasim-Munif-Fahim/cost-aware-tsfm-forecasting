"""Single source of truth for model display names, ordering, and role colors.

Every figure and table imports from here so a renamed tier or recolored series
changes everywhere at once (no palette/name drift across display items).
"""
from __future__ import annotations

import palettes

# Canonical tier order for tables/legends (floor last).
TIER_ORDER = ["chronos", "chronos_cov", "lgbm_direct", "nas_gru", "seasonal_naive"]

DISPLAY = {
    "chronos":        "Chronos-Bolt (zero-shot)",
    "chronos_cov":    "Chronos-Bolt + covariates",
    "lgbm_direct":    "LightGBM (specialist)",
    "nas_gru":        "NAS-GRU",
    "seasonal_naive": "Seasonal-naïve",
}

SHORT = {
    "chronos":        "Chronos",
    "chronos_cov":    "Chronos+cov",
    "lgbm_direct":    "LightGBM",
    "nas_gru":        "NAS-GRU",
    "seasonal_naive": "Naïve",
}

DOMAIN_DISPLAY = {"pm25": "PM$_{2.5}$", "weather": "Temperature"}


def tier_colors() -> dict:
    """Role->hex per FIGURE_BRIEF.md. Call AFTER apply_house_style()."""
    R = palettes.ACTIVE_ROLES
    return {
        "chronos":        R["primary"],          # Nature blue
        "lgbm_direct":    R["secondary"],        # Nature red
        "nas_gru":        R["quaternary"],       # Nature purple
        "chronos_cov":    palettes.NATURE["teal"],
        "seasonal_naive": R["neutral"],          # grey
    }
