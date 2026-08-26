"""
Number, p-value, OR, and n formatters.

Import once and use everywhere -- never inline `f"{x:.2f}"` in figure or table
code. Inline formatting is how a manuscript ends up with 2-decimal ORs in
Table 2 and 3-decimal ORs in the forest plot.

Conventions (per critique_checklist.md):

    Odds / hazard / risk ratios:  2 dp in body tables and forest plot bodies,
                                  3 dp in supplementary effect tables.
    p-values:                     "<0.001" if p<0.001, else 3 dp.
    Percentages:                  1 dp.
    Counts / ns:                  thousand-separated; "10.2M" in tight columns.
    Continuous summaries:         match the precision of the underlying scale
                                  (1 dp for years, 0 dp for births).
"""

from __future__ import annotations
from typing import Optional, Union
import math


Number = Union[int, float]


def _dec(s: str, dec_separator: str) -> str:
    """Substitute the decimal point in a formatted numeric string.
    Lancet uses U+00B7 middle-dot; most journals use period (.)."""
    if dec_separator == ".":
        return s
    return s.replace(".", dec_separator)


def format_p(p: Optional[Number], dec_separator: str = ".") -> str:
    """3 dp, or '<0.001' for very small, or em-dash for missing."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "—"
    if p < 0.001:
        return _dec("<0.001", dec_separator)
    if p >= 0.999:
        return _dec(">0.999", dec_separator)
    return _dec(f"{p:.3f}", dec_separator)


def format_or_ci(est: Number, lo: Number, hi: Number, dp: int = 2,
                 dash: str = "–", dec_separator: str = ".") -> str:
    """e.g. '1.13 (1.08-1.18)'. dp=3 for supplementary."""
    if any(math.isnan(x) for x in (est, lo, hi) if isinstance(x, float)):
        return "—"
    fmt = f"{{:.{dp}f}}"
    s = f"{fmt.format(est)} ({fmt.format(lo)}{dash}{fmt.format(hi)})"
    return _dec(s, dec_separator)


def format_rate_ci(est: Number, lo: Number, hi: Number, dp: int = 2,
                    dash: str = "–", dec_separator: str = ".",
                    bracket: str = "[]") -> str:
    """Rate-style CI with square brackets."""
    if any(math.isnan(x) for x in (est, lo, hi) if isinstance(x, float)):
        return "—"
    fmt = f"{{:.{dp}f}}"
    lb, rb = bracket[0], bracket[1]
    s = f"{fmt.format(est)} {lb}{fmt.format(lo)}{dash}{fmt.format(hi)}{rb}"
    return _dec(s, dec_separator)


def format_count(n: Optional[int], dec_separator: str = ".") -> str:
    """Thousand-separated integer; no decimal. Em-dash for missing."""
    if n is None or (isinstance(n, float) and math.isnan(n)):
        return "—"
    return f"{int(n):,}"


def format_pct(x: Number, dp: int = 1, with_sign: bool = False) -> str:
    """1 dp percentage. Pass x as a percentage already (e.g. 24.4), not 0.244."""
    if isinstance(x, float) and math.isnan(x):
        return "—"
    fmt = f"{{:+.{dp}f}}" if with_sign else f"{{:.{dp}f}}"
    return fmt.format(x)


def format_n(n: Optional[int], abbrev_at: int = 1_000_000) -> str:
    """Thousand-separated. Compact 'XM' / 'XK' if n >= abbrev_at."""
    if n is None or (isinstance(n, float) and math.isnan(n)):
        return "—"
    n = int(n)
    if n >= abbrev_at:
        if n >= 1_000_000_000:
            return f"{n/1e9:.1f}B"
        if n >= 1_000_000:
            return f"{n/1e6:.1f}M"
        return f"{n/1e3:.0f}K"
    return f"{n:,}"


def format_n_pct(n: int, total: int, dp: int = 1) -> str:
    """e.g. '2,494,048 (24.4)' -- for n (%) categorical cells."""
    if total == 0:
        return f"{format_n(n)} (—)"
    pct = 100.0 * n / total
    return f"{format_n(n)} ({format_pct(pct, dp)})"


def format_mean_sd(values, dp: int = 1) -> str:
    """For continuous-normal cells."""
    import numpy as np
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return "—"
    return f"{arr.mean():.{dp}f} ({arr.std(ddof=1):.{dp}f})"


def format_median_iqr(values, dp: int = 0,
                       dash: str = "–") -> str:
    """For continuous-skewed cells."""
    import numpy as np
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return "—"
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    return f"{med:.{dp}f} ({q1:.{dp}f}{dash}{q3:.{dp}f})"


# ---------------------------------------------------------------------------
# LaTeX escaping
# ---------------------------------------------------------------------------
_LATEX_REPLACE = {
    "\\": r"\textbackslash{}",
    "&":  r"\&",
    "%":  r"\%",
    "$":  r"\$",
    "#":  r"\#",
    "_":  r"\_",
    "{":  r"\{",
    "}":  r"\}",
    "~":  r"\textasciitilde{}",
    "^":  r"\textasciicircum{}",
}


def latex_escape(s: object) -> str:
    """Escape a cell value for safe inclusion in a LaTeX tabular row."""
    if s is None:
        return ""
    out = str(s)
    # Backslash first to avoid double-escaping.
    out = out.replace("\\", _LATEX_REPLACE["\\"])
    for ch, rep in _LATEX_REPLACE.items():
        if ch == "\\":
            continue
        out = out.replace(ch, rep)
    return out
