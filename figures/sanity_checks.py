"""
Statistical sanity checks. Run these BEFORE writing a metric into a
manuscript table. Many "mediocre" tables are mediocre because they
quietly report an implausible value.

Each check raises a clear error rather than returning False.
"""

from __future__ import annotations
from typing import Optional
import math


def brier_max(p_bar: float) -> float:
    """Maximum-uncertainty Brier for a calibrated model that always
    predicts the marginal rate: p_bar * (1 - p_bar)."""
    if not 0.0 <= p_bar <= 1.0:
        raise ValueError(f"p_bar must be in [0,1], got {p_bar}")
    return p_bar * (1.0 - p_bar)


def check_brier(brier: float, p_bar: float, tol: float = 0.005) -> None:
    """Raise if Brier exceeds the marginal-prediction baseline (+ tol)."""
    bmax = brier_max(p_bar)
    if brier > bmax + tol:
        raise ValueError(
            f"Implausible Brier {brier:.4f} for prevalence p_bar={p_bar:.4f} "
            f"(marginal Brier {bmax:.4f}). Likely a computation error."
        )


def check_ci_p_consistency(p: float, ci_lo: float, ci_hi: float,
                            null_value: float = 1.0,
                            alpha: float = 0.05) -> None:
    """Raise if a reported p contradicts its 95% CI."""
    if any(math.isnan(x) for x in (p, ci_lo, ci_hi)):
        return
    crosses_null = (ci_lo <= null_value <= ci_hi)
    if p < alpha and crosses_null:
        raise ValueError(
            f"Inconsistent: p={p:.4f} < {alpha} but {ci_lo:.4f}-{ci_hi:.4f} "
            f"contains null {null_value}."
        )
    if p >= alpha and not crosses_null:
        raise ValueError(
            f"Inconsistent: p={p:.4f} >= {alpha} but {ci_lo:.4f}-{ci_hi:.4f} "
            f"excludes null {null_value}."
        )


def check_percentages_sum(values, axis_label: str = "row",
                          tol: float = 1.5) -> None:
    """Raise if a row/column of percentages doesn't sum to ~100."""
    s = float(sum(v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))))
    if abs(s - 100.0) > tol:
        raise ValueError(
            f"{axis_label} percentages sum to {s:.2f}, not 100 (tol={tol})."
        )


def check_effect_direction(estimate: float, expected_sign: str,
                            null_value: float = 1.0,
                            label: str = "") -> None:
    """Raise if a covariate's effect goes the wrong way."""
    if expected_sign == "either":
        return
    if expected_sign == "protective" and estimate >= null_value:
        raise ValueError(
            f"'{label}' coded as protective but estimate {estimate} >= {null_value}."
        )
    if expected_sign == "harmful" and estimate <= null_value:
        raise ValueError(
            f"'{label}' coded as harmful but estimate {estimate} <= {null_value}."
        )


def check_min_rows(n_rows: int, expected_min: int, what: str) -> None:
    """Hard guard against truncated supplementary tables."""
    if n_rows < expected_min:
        raise ValueError(
            f"{what} has only {n_rows} rows; expected at least {expected_min}. "
            "Looks truncated -- supplementary tables ship complete."
        )


def check_calibration_bins(observed, expected_bins: int = 10) -> None:
    """Raise if a calibration plot uses too few bins."""
    n = len(observed)
    if n < expected_bins:
        raise ValueError(
            f"Calibration uses {n} bins; expected >= {expected_bins}."
        )
