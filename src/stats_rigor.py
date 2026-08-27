#!/usr/bin/env python3
"""Rigor layer for the merged paper: split-conformal intervals + Diebold-Mariano test.

Mirrors the Green-NAS paper's validation style (split conformal, significance testing)
so all tiers/domains in the journal version share one statistical framework.
"""
from __future__ import annotations

import numpy as np


def split_conformal(y_cal, pred_cal, y_test, pred_test, alpha=0.05):
    """Split conformal prediction with absolute-residual scores.

    Calibrate the quantile on (y_cal, pred_cal); report empirical coverage and mean
    interval width on (y_test, pred_test). Returns dict(q, coverage, width).
    """
    y_cal, pred_cal = np.asarray(y_cal, float), np.asarray(pred_cal, float)
    y_test, pred_test = np.asarray(y_test, float), np.asarray(pred_test, float)
    scores = np.abs(y_cal - pred_cal)
    n = len(scores)
    if n == 0:
        return {"q": np.nan, "coverage": np.nan, "width": np.nan}
    k = int(np.ceil((n + 1) * (1 - alpha)))
    q = float(np.sort(scores)[min(k, n) - 1])
    covered = np.abs(y_test - pred_test) <= q
    return {"q": q, "coverage": float(covered.mean()), "width": 2 * q}


def diebold_mariano(y_true, pred_a, pred_b, h=1, loss="ae"):
    """Diebold-Mariano test: is model A's forecast loss significantly different from B's?

    loss='ae' (absolute error) or 'se' (squared error). h = forecast horizon, used both for
    the HAC variance's lag window (Bartlett-kernel-weighted autocovariances up to lag h-1,
    a Newey-West-style variant of the original DM(1995) unweighted sum -- ensures a
    positive-semi-definite variance estimate, which matters at h=24) and for the
    Harvey-Leybourne-Newbold (1997) small-sample correction. Returns dict(dm_stat, p_value,
    mean_loss_diff); negative dm_stat => A better.

    p-value uses the Student-t(n-1) reference distribution, NOT the standard normal: HLN's
    correction is specifically derived to be paired with t(n-1) for small-sample validity
    (using normal instead is a common but slightly anti-conservative shortcut -- p-values
    come out a touch too small, mattering most when n is not large).
    """
    y = np.asarray(y_true, float)
    la = np.abs(y - np.asarray(pred_a, float))
    lb = np.abs(y - np.asarray(pred_b, float))
    if loss == "se":
        la, lb = la ** 2, lb ** 2
    d = la - lb
    n = len(d)
    if n < 10:
        return {"dm_stat": np.nan, "p_value": np.nan, "mean_loss_diff": float(np.mean(d))}
    dbar = d.mean()
    # HAC (Newey-West, Bartlett kernel) variance of dbar with lag window h-1
    gamma0 = np.mean((d - dbar) ** 2)
    var = gamma0
    for k in range(1, max(h, 1)):
        cov = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        var += 2 * (1 - k / max(h, 1)) * cov
    var /= n
    if var <= 0:
        return {"dm_stat": np.nan, "p_value": np.nan, "mean_loss_diff": float(dbar)}
    dm = dbar / np.sqrt(var)
    # Harvey-Leybourne-Newbold small-sample correction
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_c = dm * hln
    from scipy import stats
    p = float(2 * stats.t.sf(abs(dm_c), df=n - 1))
    return {"dm_stat": float(dm_c), "p_value": p, "mean_loss_diff": float(dbar)}


def paired_summary(y_true, preds_by_model, h=24):
    """All-pairs DM matrix. preds_by_model: dict name -> prediction array."""
    names = list(preds_by_model)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            r = diebold_mariano(y_true, preds_by_model[a], preds_by_model[b], h=h)
            winner = a if r["mean_loss_diff"] < 0 else b
            rows.append({"model_a": a, "model_b": b, **r,
                         "better": winner, "significant_5pct": (r["p_value"] or 1) < 0.05})
    return rows
