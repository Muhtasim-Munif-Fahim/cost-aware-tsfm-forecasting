#!/usr/bin/env python3
"""R1.2 -- turn perfect-foresight covariates into realistic forecast covariates.

Reviewer 1 objected that the causal ablation (weather frozen at the forecast origin) is a
pessimistic deployment scenario, because operationally the future covariates would come
from an NWP service. This module builds the intermediate scenario: covariates that carry
the error an actual 24 h-lead forecast carries, calibrated to error measured by
`analysis/nwp_covariate_error.py` against the Open-Meteo previous-model-runs archive.

Why degrade the covariate series rather than patch the model
------------------------------------------------------------
In `run_forecast.design()` the weather covariates enter ONLY as the future block
(`fut_` = value at origin+h, or `cov0_` = value at the origin); `origin_lag_features`
lags the TARGET, not the exogenous frame. So replacing the exog frame with a degraded copy
and leaving the existing perfect-foresight code path untouched yields exactly "every
future-covariate feature is an NWP forecast instead of a perfect one" -- and it does so
identically for lgbm_direct, chronos_cov and timesfm_cov, which is what makes the
comparison across tiers fair.

The degradation is applied to the whole series, so the model is trained AND evaluated on
forecast-quality covariates. That deliberately parallels how `causal_cov` already works
(it redefines the feature globally rather than only over the test block) and keeps the
estimate stable: degrading only the 6 fold-test blocks would leave 144 h per city to
estimate the effect from. The consequence -- errors-in-variables attenuation of the
covariate relationship at fit time -- is real, but it applies equally to every
covariate-consuming tier, so it cannot manufacture a difference between them.

Error model
-----------
Per variable, the injected error is a stationary AR(1) process, because NWP error is
strongly autocorrelated in time and white noise would be far too easy for a model to
average away:

    s_0 ~ N(0, sigma)
    s_l = rho * s_{l-1} + sqrt(1 - rho^2) * sigma * eps_l ,  eps ~ N(0, 1)

which has stationary sd exactly `sigma` and lag-1 autocorrelation exactly `rho`. A
constant `bias` term is added on top. `sigma` is expressed as a FRACTION of each city's
own signal sd, so parameters measured on the 15 archive-covered cities transfer to the
14 uncovered ones (the archive starts 2021-03-24 and the uncovered half is 9/14
scarce-tier, so a covered-cities-only experiment would be rich-city biased).

`alpha` scales the error magnitude and is the sweep knob:
    alpha = 0    -> perfect foresight (identical to the current default path)
    alpha = 1    -> the measured real-NWP error level
    alpha > 1    -> progressively worse than NWP, toward the persistence baseline

Physical admissibility is enforced after injection: bounded variables are clipped,
non-negative variables floored at 0, and wind direction wrapped to [0, 360).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Physical constraints per covariate. (lo, hi); None = unbounded on that side.
BOUNDS = {
    "relative_humidity_2m": (0.0, 100.0),
    "cloud_cover": (0.0, 100.0),
    "precipitation": (0.0, None),
    "shortwave_radiation": (0.0, None),
    "wind_speed_10m": (0.0, None),
    "surface_pressure": (None, None),
    "temperature_2m": (None, None),
    "wind_direction_10m": (None, None),   # handled circularly, not clipped
}
CIRCULAR = {"wind_direction_10m": 360.0}


def ar1_series(n, rho, sigma, rng):
    """Stationary AR(1) with lag-1 autocorrelation `rho` and stationary sd `sigma`."""
    if not np.isfinite(rho):
        rho = 0.0
    rho = float(np.clip(rho, -0.99, 0.99))
    if sigma <= 0 or n <= 0:
        return np.zeros(max(n, 0))
    out = np.empty(n)
    out[0] = rng.normal(0.0, sigma)
    innov = np.sqrt(1.0 - rho ** 2) * sigma
    for i in range(1, n):
        out[i] = rho * out[i - 1] + rng.normal(0.0, innov)
    return out


def degrade_exog(exog: pd.DataFrame, params: dict, alpha: float = 1.0,
                 seed: int = 42) -> pd.DataFrame:
    """Return a copy of `exog` with realistic forecast error injected.

    params: {variable: {"sigma_frac": float, "bias_frac": float, "ar1": float}}
            sigma_frac and bias_frac are expressed relative to that column's own sd, so
            error levels measured on archive-covered cities transfer to uncovered ones.
    alpha:  scales error magnitude. 0 reproduces the input exactly (perfect foresight).
    seed:   per-call RNG seed; the sweep varies it across replicates so results are not
            an artifact of one noise draw.
    """
    if alpha == 0:
        return exog.copy()
    rng = np.random.default_rng(seed)
    out = exog.copy()
    for col in out.columns:
        p = params.get(col)
        if p is None:                      # no measurement for this variable -> leave clean
            continue
        vals = out[col].to_numpy(dtype=float)
        finite = np.isfinite(vals)
        if finite.sum() < 2:
            continue
        signal_sd = float(np.nanstd(vals[finite]))
        if signal_sd <= 0:
            continue
        sigma = alpha * float(p.get("sigma_frac", 0.0)) * signal_sd
        bias = alpha * float(p.get("bias_frac", 0.0)) * signal_sd
        err = ar1_series(len(vals), p.get("ar1", 0.0), sigma, rng) + bias
        noisy = vals + err

        if col in CIRCULAR:                                  # wrap, never clip
            noisy = np.mod(noisy, CIRCULAR[col])
        else:
            lo, hi = BOUNDS.get(col, (None, None))
            if lo is not None:
                noisy = np.maximum(noisy, lo)
            if hi is not None:
                noisy = np.minimum(noisy, hi)
        noisy[~finite] = np.nan                              # never invent data over gaps
        out[col] = noisy
    return out


def persistence_error_frac(exog: pd.DataFrame, horizon: int = 24) -> dict:
    """Error/signal ratio of the LAST-KNOWN (causal) covariate, per variable.

    This places the manuscript's existing causal ablation on the same x-axis as the NWP
    sweep: it is the error you incur by holding the covariate at the forecast origin for
    `horizon` steps, measured in the same "fraction of signal sd" units as `sigma_frac`.
    Lets the sweep figure mark where "last-known" sits relative to real NWP skill.
    """
    out = {}
    for col in exog.columns:
        v = exog[col].to_numpy(dtype=float)
        if len(v) <= horizon:
            continue
        a, b = v[horizon:], v[:-horizon]          # value at t vs value held from t-horizon
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 48:
            continue
        d = a[m] - b[m]
        if col in CIRCULAR:
            period = CIRCULAR[col]
            d = (d + period / 2) % period - period / 2
        sd = float(np.nanstd(a[m]))
        if sd > 0:
            out[col] = float(np.sqrt(np.mean(d ** 2)) / sd)
    return out


def load_params(csv_path, lead_hours_target: float = 12.0) -> dict:
    """Build the per-variable error model from analysis/nwp_covariate_error.py output.

    The archive serves `<var>_previous_dayN`, whose lead band is 24N..24N+23 h, while the
    paper forecasts at leads 1-24 h. Taking lead_day=1 at face value would OVERSTATE NWP
    error -- which biases toward the manuscript's existing conclusion, the wrong direction
    for credibility. So error/signal is regressed on lead across the measured lead days and
    evaluated at `lead_hours_target` (default 12 h, the midpoint of our 1-24 h block).
    The fit is reported by analysis/nwp_sweep.py alongside the measured anchors so the
    extrapolation below the measured range stays explicit and auditable.
    """
    df = pd.read_csv(csv_path)
    df = df[df.actual_sd > 0].copy()
    df["err_frac"] = df.rmse / df.actual_sd
    df["bias_frac"] = df.bias / df.actual_sd

    params = {}
    for var, g in df.groupby("variable"):
        by_lead = g.groupby("lead_hours_mid").agg(
            err_frac=("err_frac", "mean"), bias_frac=("bias_frac", "mean"),
            ar1=("err_ar1", "mean"), n=("n", "sum"))
        if len(by_lead) >= 2:
            x = by_lead.index.to_numpy(dtype=float)
            slope, intercept = np.polyfit(x, by_lead.err_frac.to_numpy(), 1)
            fitted = float(intercept + slope * lead_hours_target)
            lo = float(by_lead.err_frac.min())
            # never extrapolate to a negative or absurdly small error, and never above the
            # shortest measured lead -- error grows with lead, so the 1-24 h value is a
            # lower bound on the lead-24-47 h measurement
            fitted = float(np.clip(fitted, 0.02, lo))
        else:
            fitted = float(by_lead.err_frac.iloc[0])
        params[var] = {
            "sigma_frac": fitted,
            "bias_frac": float(by_lead.bias_frac.mean()),
            "ar1": float(np.nan_to_num(by_lead.ar1.mean(), nan=0.0)),
            "measured_lead1_frac": float(by_lead.err_frac.iloc[0]),
            "n_obs": int(by_lead.n.sum()),
        }
    return params
