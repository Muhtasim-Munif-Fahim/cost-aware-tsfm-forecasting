#!/usr/bin/env python3
"""
Forecast harness for the SciRep energy paper (supersedes smoke_test.py).

Adds over the smoke test:
  (a) covariates  : calendar + exogenous weather, merged on timestamp; SHAP on the specialist.
  (b) direct LGBM : one model per horizon step (no recursive error accumulation).
  (c) sweep mode  : many buildings -> per-building metric distribution = heterogeneity engine
                    + cold-start signal (series length vs error).

Models: seasonal_naive (floor) | lgbm_direct (specialist) | chronos (FM, optional).

Modes
  single : one series, full metrics + break-even PNG + SHAP.
  sweep  : N series, heterogeneity CSV + MASE histogram + length-vs-error scatter.

Examples
  python run_forecast.py single --source synthetic --shap
  python run_forecast.py single --source bdg2 --data-path electricity_cleaned.csv \
         --weather-path weather.csv --weather-col airTemperature --with-chronos --shap
  python run_forecast.py sweep  --source synthetic --n 40
  python run_forecast.py sweep  --source bdg2 --data-path electricity_cleaned.csv --n 200

Deps: numpy pandas lightgbm matplotlib  (optional: shap, chronos-forecasting torch)

Design note: direct-LGBM models are trained ONCE on the pre-backtest region (fast, fine
for heterogeneity screening). Pass --retrain-per-fold for the publication runs.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import time
import types
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DAY, WEEK = 24, 168
YLAGS = [0, 1, 2, 3, DAY, DAY + 1, WEEK]          # relative to origin (0 = current obs)
YROLLS = [DAY, WEEK]
CAL_COLS = ["hour", "dow", "month", "is_weekend"]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _synth_one(seed: int, n_hours: int = 24 * 150) -> tuple[pd.Series, pd.Series]:
    """Return (load, temperature) — temperature genuinely drives load (for SHAP)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-01-01", periods=n_hours, freq="h")
    t = np.arange(n_hours)
    temp = (15 + 10 * np.sin(2 * np.pi * (t % (DAY * 365)) / (DAY * 365))
            + 5 * np.sin(2 * np.pi * (t % DAY) / DAY - 2.0) + rng.normal(0, 1.5, n_hours))
    base = 40 + 8 * rng.random()
    daily = 12 * np.sin(2 * np.pi * (t % DAY) / DAY - 1.0)
    weekly = 6 * np.sin(2 * np.pi * (t % WEEK) / WEEK)
    weekend = np.where(idx.dayofweek >= 5, -5.0, 0.0)
    cooling = 0.6 * np.clip(temp - 22, 0, None)      # temp -> load coupling
    heating = 0.5 * np.clip(12 - temp, 0, None)
    noise = rng.normal(0, 2.0, n_hours)
    load = np.clip(base + daily + weekly + weekend + cooling + heating + 0.002 * t + noise, 1, None)
    load = pd.Series(load, index=idx, name=f"synth_{seed}")
    return load, pd.Series(temp, index=idx, name="temp")


PM25_EXOG = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]


def _read_pm25_station(path) -> tuple[pd.Series, pd.DataFrame]:
    """One UCI Beijing multi-site PRSA csv -> (PM2.5 series, meteorology exog)."""
    df = pd.read_csv(path)
    idx = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.set_index(idx)
    y = sanitize_pm25(df["PM2.5"].astype(float)).interpolate(limit=12).dropna()
    y.name = f"pm25:{df['station'].iloc[0]}"
    exog = df[PM25_EXOG].astype(float).interpolate(limit=12).reindex(y.index)
    return y, exog


def _read_pm25_station_weather(path) -> tuple[pd.Series, pd.DataFrame]:
    """Same UCI Beijing PRSA csv, WEATHER domain: TEMP (2m air temperature) is the target,
    the remaining meteorology (PRES, DEWP, RAIN, WSPM) is the exog. Mirrors the OpenAQ
    weather-domain structure (temperature target, other weather vars as covariates) so the
    Beijing second-domain regime run is comparable to the panel weather runs. PM2.5 is
    deliberately NOT used as an exog here, to keep the exog set weather-only and symmetric
    with the OpenAQ `weather_csv` path (which never sees air quality)."""
    df = pd.read_csv(path)
    idx = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.set_index(idx)
    y = df["TEMP"].astype(float).interpolate(limit=12).dropna()
    y.name = f"weather:{df['station'].iloc[0]}"
    wx_cols = [c for c in PM25_EXOG if c != "TEMP"]
    exog = df[wx_cols].astype(float).interpolate(limit=12).reindex(y.index)
    return y, exog


def _pm25_files(data_path: str) -> list:
    import glob as _g
    import os
    if os.path.isdir(data_path):
        return sorted(_g.glob(os.path.join(data_path, "PRSA_Data_*.csv")))
    return [data_path]


PM25_SENTINEL_HIGH = 985.0   # OpenAQ/sensor cap value (appears as an exact repeated spike)


def sanitize_pm25(s: pd.Series) -> pd.Series:
    """Drop physically-impossible PM2.5 values (sensor errors / sentinels) to NaN so they
    become gaps rather than corrupting lags and MASE: <= 0 (negatives, zero-coded-missing)
    and >= 985 (a repeated cap sentinel; genuine extreme pollution reads just below it, e.g.
    Delhi 976). Found contaminating 12/29 OpenAQ cities (Seoul had 10000, Mumbai -15)."""
    bad = (s <= 0) | (s >= PM25_SENTINEL_HIGH)
    return s.mask(bad)


def extract_usable_window(s: pd.Series, segment_break: int = 48, interp_limit: int = 6,
                          min_hours: int = 2160, min_cov: float = 0.6, sanitize: bool = True):
    """Return the longest usable contiguous window of an hourly series, or None.

    Gappy air-quality records break lag features and make 'N weeks of history' meaningless.
    Strategy: (optionally) sanitize impossible values -> NaN; split the series where the gap
    between consecutive observations exceeds `segment_break` hours; keep the segment with the
    most real observations; interpolate only small (<= interp_limit h) gaps inside it; require
    the result to span >= min_hours at >= min_cov local coverage. `min_hours` doubles as the
    data-richness gate (90 days default).
    """
    if sanitize:
        s = sanitize_pm25(s).dropna()
    s = s[~s.index.duplicated()].sort_index()
    if len(s) < 2:
        return None
    dt = s.index.to_series().diff().dt.total_seconds().div(3600)
    seg_id = (dt > segment_break).cumsum()
    best, best_obs = None, -1
    for _, idx in s.groupby(seg_id).groups.items():
        seg = s.loc[idx]
        if len(seg) > best_obs:
            best, best_obs = seg, len(seg)
    if best is None:
        return None
    full = best.asfreq("h").interpolate(limit=interp_limit).dropna()
    span_h = (full.index[-1] - full.index[0]).total_seconds() / 3600 + 1
    if len(full) < min_hours or len(full) / span_h < min_cov:
        return None
    return full


def resample_subhourly(s: pd.Series) -> pd.Series:
    """Detect sensors reporting faster than hourly (e.g. every 30 min) and resample to hourly
    mean before gating. Without this, sub-hourly rows silently inflate raw coverage % beyond
    100 and confuse `extract_usable_window`'s segment logic (found on Mumbai: median inter-
    observation gap of 30 min, ~42% of rows on the half-hour). A leave-as-is series (true
    hourly cadence) is returned unchanged."""
    s = s[~s.index.duplicated()].sort_index()
    if len(s) < 3:
        return s
    median_gap_min = s.index.to_series().diff().dt.total_seconds().median() / 60
    if median_gap_min < 45:   # meaningfully sub-hourly (allows minor jitter around 60)
        s = s.resample("h").mean()
    return s


def _load_single_raw(args) -> tuple[pd.Series, pd.DataFrame]:
    """Return (target series, exog DataFrame aligned to it — weather only; calendar added later)."""
    if args.source == "synthetic":
        y, temp = _synth_one(seed=0)
        return y, temp.to_frame()
    if args.source == "pm25":
        files = _pm25_files(args.data_path)
        if args.column:
            files = [f for f in files if args.column.lower() in f.lower()] or files
        return _read_pm25_station(files[0])
    if args.source == "weather_pm25":
        # Beijing second domain: forecast TEMP (2m temperature) from the same PRSA station,
        # closing the §7 gap (Beijing-weather was previously infeasible -- the PRSA
        # year/month/day/hour + embedded TEMP format is incompatible with the Open-Meteo
        # `weather_csv` loader). See ANALYSIS_PLAN.md Deviations log 2026-07-14.
        files = _pm25_files(args.data_path)
        if args.column:
            files = [f for f in files if args.column.lower() in f.lower()] or files
        return _read_pm25_station_weather(files[0])
    if args.source == "csv":
        # generic timestamp,value CSV (e.g. OpenAQ pull) — no local weather exog.
        # Quality gate: keep only the longest usable contiguous window (see extract_usable_window).
        df = pd.read_csv(args.data_path)
        tcol = next((c for c in ("timestamp", "utc_timestamp", "time", "datetime") if c in df.columns), df.columns[0])
        vcol = args.column or next((c for c in df.columns if c != tcol), df.columns[1])
        raw = pd.Series(df[vcol].astype(float).values,
                        index=pd.to_datetime(df[tcol], utc=True).dt.tz_localize(None),
                        name=f"csv:{os.path.basename(args.data_path)}")
        raw = resample_subhourly(raw)
        y = extract_usable_window(raw, min_hours=getattr(args, "min_hours", 8760))
        if y is None:
            raise ValueError(f"no usable window (>= {getattr(args,'min_hours',8760)}h contiguous)")
        y.name = raw.name
        # attach matching weather covariates if available (data/weather/<same-name>.csv)
        exog = _attach_city_weather(args, y.index)
        return y, exog
    if args.source == "weather_csv":
        # Second domain: temperature_2m forecasting from Open-Meteo, same 29-city panel.
        # ANALYSIS_PLAN.md D3: clip to the SAME usable window as that city's PM2.5 series
        # (not full Open-Meteo history) so the cross-domain comparison isolates the domain
        # effect rather than confounding it with "weather data happens to be longer."
        pm25_dir = getattr(args, "pm25_window_dir", None) or "data/cities_final"
        pm25_path = os.path.join(pm25_dir, os.path.basename(args.data_path))
        pm25_args = types.SimpleNamespace(source="csv", data_path=pm25_path, column=None,
                                                 min_hours=getattr(args, "min_hours", 2160))
        pm25_y, _ = load_single(pm25_args)   # reuses the exact sanitize/resample/gate pipeline
        df = pd.read_csv(args.data_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated()]
        # Clip the PM2.5-defined window to the weather file's REAL coverage before reindexing.
        # The previous blanket .ffill().bfill() FABRICATED a constant tail wherever the PM2.5
        # window extended past the weather pull's end (found 2026-07-15: 4 cities with
        # 193-289h fabricated tails -- longer than the whole 144h test window -- giving
        # context-copying models absurd scores, e.g. chronos MASE 0.028 on Vienna).
        idx = pm25_y.index
        idx = idx[(idx >= df.index.min()) & (idx <= df.index.max())]
        min_h = getattr(args, "min_hours", 2160)
        if len(idx) < min_h:
            raise ValueError(f"weather coverage clips PM2.5 window to {len(idx)}h (< {min_h}h)")
        df = df.reindex(idx).interpolate(limit=6)
        if df.isna().any().any():
            bad = df.columns[df.isna().any()].tolist()
            raise ValueError(f"weather series has gaps > 6h inside coverage (cols: {bad}) -- "
                             "refusing to fabricate values")
        y = df["temperature_2m"].rename(f"weather:{os.path.basename(args.data_path)}")
        exog = df.drop(columns="temperature_2m")
        return y, exog
    if args.source == "bdg2":
        df = pd.read_csv(args.data_path, parse_dates=["timestamp"]).set_index("timestamp")
        col = args.column or df.notna().mean().pipe(lambda c: c[c >= 0.90].index[0])
        y = df[col].astype(float).interpolate(limit=6).dropna()
        y.name = f"bdg2:{col}"
    elif args.source == "opsd":
        df = pd.read_csv(args.data_path, parse_dates=["utc_timestamp"]).set_index("utc_timestamp")
        cands = [c for c in df.columns if c.endswith("_load_actual_entsoe_transparency")] \
            or [c for c in df.columns if "load_actual" in c]
        col = args.column or df[cands].notna().mean().idxmax()
        y = df[col].astype(float).interpolate(limit=6).dropna()
        y.name = f"opsd:{col}"
    else:
        raise ValueError(args.source)
    exog = _load_weather(args, y.index)
    return y, exog


def _attach_city_weather(args, index) -> pd.DataFrame:
    """Merge Open-Meteo weather for a city CSV: data/weather/<same-basename>.
    Weather is gap-free reanalysis; reindex to the (gated) PM2.5 index. Empty frame if absent."""
    wdir = getattr(args, "weather_dir", None) or os.path.join(os.path.dirname(os.path.dirname(args.data_path)), "weather")
    wpath = os.path.join(wdir, os.path.basename(args.data_path))
    if not os.path.exists(wpath):
        return pd.DataFrame(index=index)
    w = pd.read_csv(wpath)
    tcol = w.columns[0]
    w.index = pd.to_datetime(w[tcol], utc=True).dt.tz_localize(None)
    w = w.drop(columns=[tcol])
    w = w[~w.index.duplicated()]
    return w.reindex(index).interpolate(limit=6).ffill().bfill()


_NWP_PARAMS_CACHE = {}


def _nwp_params(csv_path):
    if csv_path not in _NWP_PARAMS_CACHE:
        from covariate_degradation import load_params
        _NWP_PARAMS_CACHE[csv_path] = load_params(csv_path)
    return _NWP_PARAMS_CACHE[csv_path]


def load_single(args) -> tuple[pd.Series, pd.DataFrame]:
    """`_load_single_raw` plus the R1.2 realistic-covariate degradation.

    Applied here, centrally, so every caller and every `--source` inherits it and so the
    existing perfect-foresight code path in `design()` is left untouched: degrading the
    exog frame IS the realistic-NWP scenario, because weather enters the design matrix only
    as the future-covariate block. --covariate-noise 0 (default) is an exact no-op, so this
    cannot perturb any previously published run.
    """
    y, exog = _load_single_raw(args)
    alpha = float(getattr(args, "covariate_noise", 0.0) or 0.0)
    if alpha and exog is not None and not exog.columns.empty:
        from covariate_degradation import degrade_exog
        exog = degrade_exog(exog, _nwp_params(args.nwp_error_csv), alpha=alpha,
                            seed=int(getattr(args, "covariate_noise_seed", 42)))
    return y, exog


def _load_weather(args, index) -> pd.DataFrame:
    if not args.weather_path:
        return pd.DataFrame(index=index)
    w = pd.read_csv(args.weather_path)
    tcol = next((c for c in ("timestamp", "utc_timestamp", "time") if c in w.columns), w.columns[0])
    w[tcol] = pd.to_datetime(w[tcol])
    w = w.set_index(tcol)
    cols = [args.weather_col] if args.weather_col else \
        [c for c in w.columns if w[c].dtype.kind in "fi"][:3]
    return w[cols].reindex(index).interpolate(limit=12)


def sweep_series(args):
    """Yield (name, y, exog) for many series."""
    if args.source == "synthetic":
        for k in range(args.n):
            y, temp = _synth_one(seed=k + 1)
            yield y.name, y, temp.to_frame()
    elif args.source == "pm25":
        for f in _pm25_files(args.data_path)[: args.n]:
            y, exog = _read_pm25_station(f)
            if len(y) > WEEK * 4:
                yield y.name, y, exog
    elif args.source == "bdg2":
        df = pd.read_csv(args.data_path, parse_dates=["timestamp"]).set_index("timestamp")
        cov = df.notna().mean()
        cols = cov[cov >= 0.85].index[: args.n]
        exog = _load_weather(args, df.index)
        for c in cols:
            y = df[c].astype(float).interpolate(limit=6).dropna()
            if len(y) > WEEK * 4:
                yield f"bdg2:{c}", y, exog.reindex(y.index)
    else:
        raise ValueError(f"sweep unsupported for source {args.source}")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def mase_scale(y_train, m=DAY) -> float:
    """In-sample MAE of the m-step naive forecast = MASE denominator."""
    y_train = np.asarray(y_train, float)
    return max(np.mean(np.abs(y_train[m:] - y_train[:-m])), 1e-9)


def _mase(y_true, y_pred, y_train, m=DAY, scale=None):
    denom = scale if scale is not None else mase_scale(y_train, m)
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / denom)


def score(y_true, y_pred, y_train, scale=None):
    """scale = fixed MASE denominator; pass one shared value to compare MASE across
    regimes/series fairly (else each regime normalizes by its own training data)."""
    yt = np.asarray(y_true, float)
    r = float(np.sqrt(np.mean((yt - np.asarray(y_pred)) ** 2)))
    mean = np.mean(yt) or 1e-9
    return {"MAE": float(np.mean(np.abs(yt - np.asarray(y_pred)))), "RMSE": r,
            "MASE": _mase(yt, y_pred, y_train, scale=scale),
            "nRMSE": r / np.std(yt) if np.std(yt) > 0 else np.nan,
            "RMSE_pct_mean": 100 * r / mean}


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def calendar(index) -> pd.DataFrame:
    return pd.DataFrame({"hour": index.hour, "dow": index.dayofweek,
                         "month": index.month,
                         "is_weekend": (index.dayofweek >= 5).astype(int)}, index=index)


def origin_lag_features(y: pd.Series) -> pd.DataFrame:
    """Features known at origin t (no leakage): lags + past rolling means."""
    f = pd.DataFrame(index=y.index)
    for k in YLAGS:
        f[f"ylag_{k}"] = y.shift(k)
    for w in YROLLS:
        f[f"rollmean_{w}"] = y.shift(0).rolling(w).mean()   # up to and incl. t
    return f


def future_covariates(y_index, exog: pd.DataFrame) -> pd.DataFrame:
    """Covariates known for the future horizon: calendar (always) + weather forecast (assumed)."""
    cov = calendar(y_index)
    if exog is not None and not exog.columns.empty:
        cov = cov.join(exog.reindex(y_index).add_prefix("wx_"))
    return cov


# --------------------------------------------------------------------------- #
# Backtest folds
# --------------------------------------------------------------------------- #
@dataclass
class Fold:
    train_end: int
    y_test: np.ndarray


def make_folds(y, horizon, n_folds):
    folds, n = [], len(y)
    for k in range(n_folds, 0, -1):
        end = n - k * horizon
        if end <= WEEK * 3:
            continue
        folds.append(Fold(end, y.values[end:end + horizon]))
    return folds


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def run_seasonal_naive(y, exog, folds, horizon):
    preds, t0 = [], time.perf_counter()
    for f in folds:
        hist = y.values[:f.train_end]
        preds.append(hist[-WEEK:][:horizon] if len(hist) >= WEEK else np.repeat(hist[-1], horizon))
    return np.concatenate(preds), time.perf_counter() - t0, 0, None


def run_lgbm_direct(y, exog, folds, horizon, retrain_per_fold=False, want_shap=False,
                    random_state=42, causal_cov=False):
    """One model per horizon step. Trained once on pre-backtest region (or per fold).

    random_state fixes LightGBM's RNG (bagging row sample, feature-sample ties) so this
    tier's numbers are reproducible. Phase-1 audit finding: `subsample=0.8` was previously
    set without `subsample_freq`, which LightGBM treats as bagging DISABLED (subsample only
    takes effect when subsample_freq > 0) -- silently running as a plain, undocumented
    deterministic GBM instead of the intended stochastic/regularized one. Fixed here by
    setting subsample_freq=1 so the bagging fraction is actually applied every iteration.

    causal_cov (ablation, added 2026-07-15): if True, WEATHER covariates enter at the
    forecast ORIGIN (last-known value) instead of at the target time origin+h. The default
    (False) uses the covariate at origin+h -- i.e. a PERFECT weather forecast, an idealization
    that is especially generous in the weather-forecasting domain where the covariates
    (humidity/pressure/radiation) are near-deterministic physical drivers of the target
    (temperature). The causal variant is the deployable-without-an-NWP-forecast baseline and
    isolates how much of the specialist's advantage is perfect-foresight covariate access
    rather than forecasting skill. Calendar features are future-known either way (never
    "perfect foresight"), so they always enter at origin+h."""
    import lightgbm as lgb
    LF = origin_lag_features(y)
    CAL = calendar(y.index)
    WX = (exog.reindex(y.index).add_prefix("wx_")
          if exog is not None and not exog.columns.empty else None)
    n = len(y)
    models_cache, shap_payload = None, None
    latency, n_params, preds = 0.0, 0, []

    def design(h):
        parts = [LF, CAL.shift(-h).add_prefix("futcal_")]   # calendar@origin+h is legit
        if WX is not None:
            if causal_cov:
                parts.append(WX.add_prefix("cov0_"))         # weather@origin (last known)
            else:
                parts.append(WX.shift(-h).add_prefix("fut_"))  # weather@origin+h (perfect forecast)
        X = parts[0].join(parts[1:])
        yt = y.shift(-h)
        return X, yt

    def fit_all(limit):
        mods = []
        for h in range(1, horizon + 1):
            X, yt = design(h)
            mask = X.index[: limit - h]                  # targets land before `limit`
            d = X.loc[mask].join(yt.rename("t")).dropna()
            m = lgb.LGBMRegressor(n_estimators=250, learning_rate=0.05, num_leaves=31,
                                  subsample=0.8, subsample_freq=1, random_state=random_state,
                                  verbose=-1)
            m.fit(d.drop(columns="t"), d["t"])
            mods.append((m, X.columns))
        return mods

    if not retrain_per_fold:
        models_cache = fit_all(folds[0].train_end)
        n_params = sum(m.n_estimators * m.num_leaves for m, _ in models_cache)

    for f in folds:
        mods = fit_all(f.train_end) if retrain_per_fold else models_cache
        origin = f.train_end - 1
        row = []
        t0 = time.perf_counter()
        for h in range(1, horizon + 1):
            m, cols = mods[h - 1]
            X, _ = design(h)
            row.append(float(m.predict(X.iloc[[origin]][cols])[0]))
        latency += time.perf_counter() - t0
        preds.append(np.array(row))

    if want_shap and models_cache:
        shap_payload = _shap_specialist(models_cache, design, folds[0].train_end)
    return np.concatenate(preds), latency, n_params, shap_payload


def _shap_specialist(models_cache, design, limit):
    """SHAP for the h=1 model (drivers of next-hour load). Rebuilds the exact h=1 design
    matrix via the runner's own `design` closure so it tracks the causal_cov setting."""
    try:
        import shap
    except Exception as e:  # noqa: BLE001
        return {"error": f"shap unavailable: {e}"}
    m, cols = models_cache[0]
    X1, _ = design(1)
    X = X1[cols].iloc[:limit].dropna()
    expl = shap.TreeExplainer(m)
    sv = expl.shap_values(X)
    imp = pd.Series(np.abs(sv).mean(0), index=cols).sort_values(ascending=False)
    return {"mean_abs_shap": imp, "X": X, "shap_values": sv}


class _NASGru(object):
    """Lazy holder so torch is only imported when the tier is used."""
    _cls = None

    @classmethod
    def get(cls):
        if cls._cls is None:
            import torch.nn as nn

            class NASGru(nn.Module):
                """Green-NAS-A architecture (2xGRU-128, NSGA-II Pareto winner from the
                published conference paper), adapted with a direct multi-horizon head."""

                def __init__(self, input_dim, horizon, hidden=128):
                    super().__init__()
                    self.gru = nn.GRU(input_dim, hidden, num_layers=2, batch_first=True)
                    self.head = nn.Linear(hidden, horizon)

                def forward(self, x):
                    out, _ = self.gru(x)
                    return self.head(out[:, -1, :])

            cls._cls = NASGru
        return cls._cls


def zscore_city(y, exog, train_end, horizon=None):
    """Fit z-score stats on ONE city's own pre-test training region (indices [0, train_end)),
    return (z matrix incl. target as column 0, y_mu, y_sd, feature columns). Shared by
    run_nas_model and e4_transfer.py's pretrain/fine-tune corpus construction -- each city
    is normalized to its OWN scale so a pooled multi-city corpus doesn't leak absolute-scale
    differences (Bangkok PM2.5 ~30 vs. Nairobi ~10, say) into the shared feature space."""
    cov = future_covariates(y.index, exog)
    feats = pd.concat([y.rename("y"), cov], axis=1).astype(np.float32)
    tr0 = feats.iloc[:train_end]
    mu, sd = tr0.mean(), tr0.std().replace(0, 1)
    z = ((feats - mu) / sd).values
    return z, float(mu["y"]), float(sd["y"]), list(feats.columns)


def build_windows(z, lo, hi, lookback, horizon):
    """(X,Y) tensor pairs from a z-scored feature matrix, origins in [lo+lookback, hi-horizon)."""
    import torch
    xs, ys = [], []
    for t in range(lo + lookback, hi - horizon):
        xs.append(z[t - lookback:t])
        ys.append(z[t:t + horizon, 0])
    if not xs:
        return None, None
    return torch.tensor(np.array(xs)), torch.tensor(np.array(ys))


def train_nas_gru(model, Xtr, Ytr, Xval, Yval, dev, max_epochs=50, patience=10, batch=256, lr=1e-3):
    """Adam + early-stopping training loop shared by run_nas_model (from-scratch, per-city)
    and e4_transfer.py (pooled pretrain AND per-city fine-tune -- fine-tune just calls this
    again on a pretrained model with a smaller max_epochs/lr and the target city's own,
    possibly fraction-truncated, windows)."""
    import torch
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.MSELoss()
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xtr, Ytr),
                                     batch_size=min(batch, len(Xtr)), shuffle=True)
    best_val, best_state, bad = float("inf"), None, 0
    for _ in range(max_epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss = lossf(model(xb.to(dev)), yb.to(dev))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = lossf(model(Xval.to(dev)), Yval.to(dev)).item()
        if vloss < best_val - 1e-5:
            best_val, best_state, bad = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def run_nas_model(y, exog, folds, horizon, lookback=DAY, max_epochs=50, patience=10,
                  batch=256, val_frac=0.1, seed=42):
    """Green-NAS tier: train the published NAS-discovered GRU (Green-NAS-A, 2xGRU-128) per
    fold on (target + weather + calendar) windows, direct multi-horizon head. Matches the
    conference paper's training protocol: Adam(lr=1e-3), early stopping (patience=10),
    max 50 epochs, seed 42 (config.py RANDOM_SEED). CPU/GPU per availability."""
    import torch
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    NASGru = _NASGru.get()
    z, y_mu, y_sd, _cols = zscore_city(y, exog, folds[0].train_end)

    n_train = folds[0].train_end
    n_val = max(int(n_train * val_frac), lookback + horizon + 1)
    Xtr, Ytr = build_windows(z, 0, n_train - n_val, lookback, horizon)
    Xval, Yval = build_windows(z, n_train - n_val - lookback, n_train, lookback, horizon)

    model = NASGru(z.shape[1], horizon).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    model = train_nas_gru(model, Xtr, Ytr, Xval, Yval, dev, max_epochs, patience, batch)

    latency, preds = 0.0, []
    model.eval()
    with torch.no_grad():
        for f in folds:
            ctx = torch.tensor(z[f.train_end - lookback:f.train_end][None]).to(dev)
            t0 = time.perf_counter()
            out = model(ctx)[0].cpu().numpy()
            latency += time.perf_counter() - t0
            preds.append(out * y_sd + y_mu)
    return np.concatenate(preds), latency, n_params, None


def _load_chronos(model_name):
    import torch
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(
        model_name, device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float32)
    return torch, pipe


def run_chronos(y, exog, folds, horizon, model_name="amazon/chronos-bolt-small"):
    """Plain univariate zero-shot FM (no covariates) — ablation baseline."""
    torch, pipe = _load_chronos(model_name)
    n_params = sum(p.numel() for p in pipe.model.parameters())
    latency, preds = 0.0, []
    for f in folds:
        ctx = torch.tensor(y.values[:f.train_end][-WEEK * 4:], dtype=torch.float32)
        t0 = time.perf_counter()
        _, mean = pipe.predict_quantiles(ctx, prediction_length=horizon,
                                         quantile_levels=[0.1, 0.5, 0.9])
        latency += time.perf_counter() - t0
        preds.append(mean[0].cpu().numpy())
    return np.concatenate(preds), latency, n_params, None


def run_chronos_cov(y, exog, folds, horizon, model_name="amazon/chronos-bolt-small",
                    causal_cov=False):
    """Covariate-informed FM: a light covariate model captures calendar+weather effect,
    the FM forecasts the residual (its temporal strength), then the covariate effect for
    the future horizon is added back. Fair FM-vs-specialist comparison + clean ablation.

    causal_cov (added 2026-07-15, reviewer-fix): mirrors the lgbm `design()` ablation.
    When True, the WEATHER covariates over the forecast horizon block
    [train_end, train_end+h) are frozen at their last-known value at the origin
    (train_end-1) before the covariate model is evaluated for that block, so the
    added-back future effect `fut` uses no perfect weather forecast. History rows are
    untouched (the residual context stays on actual past weather) and calendar features
    are future-known either way, so only the future weather is made causal."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    torch, pipe = _load_chronos(model_name)
    n_params = sum(p.numel() for p in pipe.model.parameters())
    cov = future_covariates(y.index, exog)
    wx_cols = [c for c in cov.columns if c.startswith("wx_")]
    latency, preds = 0.0, []
    for f in folds:
        t0 = time.perf_counter()
        d = cov.iloc[:f.train_end].join(y.rename("t")).dropna()
        cov_in = cov.ffill().bfill()
        if causal_cov and wx_cols and f.train_end >= 1:
            cov_in = cov_in.copy()
            last = cov_in[wx_cols].iloc[f.train_end - 1].values
            idx = [cov_in.columns.get_loc(c) for c in wx_cols]
            cov_in.iloc[f.train_end:f.train_end + horizon, idx] = last
        if len(d) >= 48 and d.drop(columns="t").std().sum() > 0:
            sc = StandardScaler().fit(d.drop(columns="t"))
            cm = Ridge(alpha=1.0).fit(sc.transform(d.drop(columns="t")), d["t"])
            cov_hat = cm.predict(sc.transform(cov_in))
        else:
            cov_hat = np.zeros(len(y))   # covariate model unfittable -> plain FM on the series
        resid = y.values - cov_hat
        ctx = torch.tensor(resid[:f.train_end][-WEEK * 4:], dtype=torch.float32)
        _, mean = pipe.predict_quantiles(ctx, prediction_length=horizon,
                                         quantile_levels=[0.1, 0.5, 0.9])
        fut = cov_hat[f.train_end:f.train_end + horizon]
        latency += time.perf_counter() - t0
        preds.append(mean[0].cpu().numpy() + fut)
    return np.concatenate(preds), latency, n_params, None


def _load_timesfm(model_name="google/timesfm-2.5-200m-pytorch", horizon=DAY, context=WEEK * 4):
    """Load TimesFM 2.5 (200M) -- the second FM family added for the R1.1 generality check.

    Deliberately NOT cached across runner calls. `_load_chronos` reloads on every runner
    call and the energy meter wraps the entire runner, so a cached TimesFM would have its
    checkpoint-load energy excluded while Chronos paid it, making the two families'
    measured joules incomparable. Cold-process load is 7.5 s vs Chronos's 29.6 s, so the
    symmetric choice does not penalise TimesFM.
    """
    import torch
    import timesfm
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_name)
    m.compile(timesfm.ForecastConfig(max_context=context, max_horizon=horizon,
                                     normalize_inputs=True,
                                     use_continuous_quantile_head=True))
    return torch, m


def _timesfm_point(model, ctx, horizon):
    """Point forecast for one context window -> np.ndarray(horizon,)."""
    pt, _ = model.forecast(horizon=horizon, inputs=[np.asarray(ctx, dtype=np.float32)])
    return np.asarray(pt, dtype=float)[0][:horizon]


def run_timesfm(y, exog, folds, horizon, model_name="google/timesfm-2.5-200m-pytorch"):
    """Univariate zero-shot FM, second family (no covariates).

    Mirrors run_chronos line for line -- same fold loop, same 4-week context window -- so
    that any TimesFM-vs-Chronos difference is attributable to the model rather than to the
    evaluation harness.
    """
    torch, model = _load_timesfm(model_name, horizon=horizon)
    n_params = sum(p.numel() for p in model.model.parameters())
    latency, preds = 0.0, []
    for f in folds:
        ctx = y.values[:f.train_end][-WEEK * 4:]
        t0 = time.perf_counter()
        out = _timesfm_point(model, ctx, horizon)
        latency += time.perf_counter() - t0
        preds.append(out)
    return np.concatenate(preds), latency, n_params, None


def run_timesfm_cov(y, exog, folds, horizon, model_name="google/timesfm-2.5-200m-pytorch",
                    causal_cov=False):
    """Covariate-informed TimesFM, using the SAME residual-Ridge scheme as run_chronos_cov.

    TimesFM 2.5 ships a native `forecast_with_covariates`, which is deliberately NOT used
    here: holding the covariate pathway identical across both FM families keeps the
    comparison about the foundation model itself rather than about two different covariate
    implementations. `causal_cov` freezes future weather at the forecast origin exactly as
    in run_chronos_cov / the lgbm design() ablation.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    torch, model = _load_timesfm(model_name, horizon=horizon)
    n_params = sum(p.numel() for p in model.model.parameters())
    cov = future_covariates(y.index, exog)
    wx_cols = [c for c in cov.columns if c.startswith("wx_")]
    latency, preds = 0.0, []
    for f in folds:
        t0 = time.perf_counter()
        d = cov.iloc[:f.train_end].join(y.rename("t")).dropna()
        cov_in = cov.ffill().bfill()
        if causal_cov and wx_cols and f.train_end >= 1:
            cov_in = cov_in.copy()
            last = cov_in[wx_cols].iloc[f.train_end - 1].values
            idx = [cov_in.columns.get_loc(c) for c in wx_cols]
            cov_in.iloc[f.train_end:f.train_end + horizon, idx] = last
        if len(d) >= 48 and d.drop(columns="t").std().sum() > 0:
            sc = StandardScaler().fit(d.drop(columns="t"))
            cm = Ridge(alpha=1.0).fit(sc.transform(d.drop(columns="t")), d["t"])
            cov_hat = cm.predict(sc.transform(cov_in))
        else:
            cov_hat = np.zeros(len(y))   # covariate model unfittable -> plain FM on the series
        resid = y.values - cov_hat
        ctx = resid[:f.train_end][-WEEK * 4:]
        out = _timesfm_point(model, ctx, horizon)
        fut = cov_hat[f.train_end:f.train_end + horizon]
        latency += time.perf_counter() - t0
        preds.append(out + fut)
    return np.concatenate(preds), latency, n_params, None


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    model: str
    metrics: dict = field(default_factory=dict)
    latency_s: float = 0.0
    per_forecast_ms: float = 0.0
    n_params: int = 0
    energy_j_per_1k: float = 0.0          # TDP-proxy estimate (supplementary cross-check)
    usd_per_1k: float = 0.0               # TDP-proxy estimate
    measured_j_per_1k: float = None       # codecarbon-measured (primary, when available)
    measured_usd_per_1k: float = None
    measured_cpu_j_per_1k: float = None
    measured_gpu_j_per_1k: float = None
    energy_source: str = "tdp_proxy"      # "codecarbon" | "tdp_proxy" (fallback)
    seed: int = None                      # None for deterministic tiers; int for nas_gru/E4 runs
    shap: object = None


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def _release_gpu() -> None:
    """Free the GPU between runner calls.

    Every tier reloads its checkpoint per runner call (see `_load_timesfm` for why that
    symmetry is required for energy comparability), but the loaded models are not released
    when the runner returns: measured growth is ~882 MB per city, with reserved memory
    saturating near 5.9 GB and free memory falling to ~430 MB on an 8 GB card. The PM2.5
    panel survives that because its series are short; the weather panel, whose series run to
    ~92k hours and need larger activation buffers, died at city 17 with a native access
    violation (0xC0000005) rather than a catchable CUDA OOM.

    Called from the runner loop's `finally`, so it executes AFTER `_EnergyMeter.measure`
    has stopped. That placement is deliberate and load-bearing: putting this inside a runner
    would fold cleanup cost into that tier's measured joules and change already-published
    energy numbers. Outside the meter it cannot affect any measured quantity, and MASE is
    deterministic, so no previously reported result moves.
    """
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass          # cleanup must never be able to fail a run


def cost_of(model: str, latency_s: float, n_fc: int, args) -> tuple[float, float]:
    """Energy (J) and USD per 1,000 forecasts -- TDP-proxy estimate (supplementary
    cross-check; see `_EnergyMeter` / `--measure-energy` for the primary, codecarbon-
    measured signal).

    Proxy = latency * device_power * PUE. GPU power for FM models when CUDA present,
    else CPU TDP.

    IMPORTANT ASYMMETRY (documented, not a bug): `latency_s` here is INFERENCE-ONLY
    (the fold-prediction loop) for every tier. The codecarbon-measured energy
    (`Result.measured_*`), however, wraps the ENTIRE runner call -- for trainable tiers
    (nas_gru; and indirectly chronos_cov's per-fold Ridge fit) that includes TRAINING
    time too. So `energy_j_per_1k` (proxy) and `measured_j_per_1k` are NOT directly
    comparable for nas_gru: the proxy is inference-only, the measurement is train+infer.
    Both are reported; do not treat them as two estimates of the same quantity for a
    trainable tier. Analysis/Methods must state this explicitly.
    """
    is_torch_tier = (model.startswith("chronos") or model.startswith("nas_")
                     or model.startswith("timesfm"))
    power_w = args.gpu_tdp if (is_torch_tier and _cuda_available()) else args.cpu_tdp
    energy_j_total = latency_s * power_w * args.pue
    per_1k = 1000.0 / max(n_fc, 1)
    energy_j_per_1k = energy_j_total * per_1k
    usd_per_1k = (energy_j_per_1k / 3.6e6) * args.price_kwh   # J -> kWh -> $
    return energy_j_per_1k, usd_per_1k


class _EnergyMeter:
    """codecarbon-measured energy for one runner call. A FRESH EmissionsTracker per
    measurement (not a shared start_task/stop_task tracker): reusing one tracker across
    many start_task/stop_task cycles was observed to occasionally corrupt internal task-
    lifecycle state ("_active_task_emissions_at_start was None") during back-to-back calls,
    which is unacceptable for an unattended multi-hour campaign. Fresh-tracker overhead is
    ~5-7s/call (NVML + CPU-model detection) -- accepted for reliability. Falls back to None
    (TDP-proxy-only) if codecarbon is missing or fails to initialize on this machine."""
    _unavailable = False

    @classmethod
    def measure(cls, fn):
        """Run fn(); return (fn_result, measured_dict_or_None)."""
        if cls._unavailable:
            return fn(), None
        try:
            from codecarbon import EmissionsTracker
            tracker = EmissionsTracker(measure_power_secs=1, save_to_file=False,
                                       log_level="error", tracking_mode="process",
                                       allow_multiple_runs=True)
        except Exception:  # noqa: BLE001
            cls._unavailable = True
            return fn(), None
        tracker.start()
        try:
            result = fn()
        finally:
            tracker.stop()
        data = tracker.final_emissions_data
        measured = {"kwh": data.energy_consumed, "cpu_kwh": data.cpu_energy,
                   "gpu_kwh": data.gpu_energy} if data is not None else None
        return result, measured


def parse_seeds(seeds_arg):
    """'--seeds 42,43,44' -> [42,43,44]. None/empty -> [42] (single-seed default; pass
    --seeds explicitly for the 5-seed campaign runs, so quick dev/smoke tests stay cheap)."""
    if not seeds_arg:
        return [42]
    return [int(s.strip()) for s in str(seeds_arg).split(",") if s.strip()]


def evaluate(y, exog, args, mase_scale_fixed=None, capture=None):
    """capture: optional dict; filled with {'y_true':..., 'preds': {model: array}} for
    downstream rigor stats (conformal, Diebold-Mariano)."""
    folds = make_folds(y, args.horizon, args.folds)
    if not folds:
        return None, None
    y_true = np.concatenate([f.y_test for f in folds])
    y_train = y.values[:folds[0].train_end]
    n_fc = len(folds) * args.horizon

    seeds = parse_seeds(getattr(args, "seeds", None))   # nas_gru (+ E4) multi-seed; other tiers ignore

    # (name, seed_or_None, fn) -- seed is attached to the Result for per-seed aggregation;
    # a tier with seed=None is deterministic (or internally uses a single fixed seed).
    runners = [
        ("seasonal_naive", None, lambda: run_seasonal_naive(y, exog, folds, args.horizon)),
        ("lgbm_direct", None, lambda: run_lgbm_direct(y, exog, folds, args.horizon,
                                                       args.retrain_per_fold, args.shap,
                                                       causal_cov=getattr(args, "causal_cov", False))),
    ]
    if getattr(args, "with_nas", False):
        for sd in seeds:
            runners.append(("nas_gru", sd,
                            lambda sd=sd: run_nas_model(y, exog, folds, args.horizon, seed=sd)))
    if args.with_chronos:
        runners.append(("chronos", None, lambda: run_chronos(y, exog, folds, args.horizon,
                                                              args.chronos_model)))
        runners.append(("chronos_cov", None, lambda: run_chronos_cov(y, exog, folds, args.horizon,
                                                                      args.chronos_model,
                                                                      causal_cov=getattr(args, "causal_cov", False))))
    if getattr(args, "with_chronos_base", False):
        # Same family and same code path as the chronos tier; only the checkpoint differs.
        # Registered as its own tier so that a single run carries small AND base, which lets
        # the scale effect within a family be separated from the family effect across
        # architectures (base at 205M sits close to TimesFM's 231M).
        runners.append(("chronos_base", None,
                        lambda: run_chronos(y, exog, folds, args.horizon,
                                            args.chronos_base_model)))
        runners.append(("chronos_base_cov", None,
                        lambda: run_chronos_cov(y, exog, folds, args.horizon,
                                                args.chronos_base_model,
                                                causal_cov=getattr(args, "causal_cov", False))))
    if getattr(args, "with_timesfm", False):
        runners.append(("timesfm", None, lambda: run_timesfm(y, exog, folds, args.horizon,
                                                              args.timesfm_model)))
        runners.append(("timesfm_cov", None, lambda: run_timesfm_cov(y, exog, folds, args.horizon,
                                                                      args.timesfm_model,
                                                                      causal_cov=getattr(args, "causal_cov", False))))
    measure_energy = getattr(args, "measure_energy", False)
    results = []
    for name, seed, fn in runners:
        capture_key = name if seed is None else f"{name}_s{seed}"
        try:
            if measure_energy:
                (preds, latency, n_params, shap), measured = _EnergyMeter.measure(fn)
            else:
                (preds, latency, n_params, shap), measured = fn(), None
            e1k, usd1k = cost_of(name, latency, n_fc, args)
            r = Result(name, score(y_true, preds, y_train, scale=mase_scale_fixed),
                      latency, 1000 * latency / max(n_fc, 1), n_params, e1k, usd1k)
            r.seed = seed
            if measured is not None:
                per_1k = 1000.0 / max(n_fc, 1)
                r.measured_j_per_1k = measured["kwh"] * 3.6e6 * per_1k
                r.measured_cpu_j_per_1k = measured["cpu_kwh"] * 3.6e6 * per_1k
                r.measured_gpu_j_per_1k = measured["gpu_kwh"] * 3.6e6 * per_1k
                r.measured_usd_per_1k = measured["kwh"] * per_1k * args.price_kwh
                r.energy_source = "codecarbon"
            r.shap = shap
            results.append(r)
            if capture is not None:
                capture.setdefault("preds", {})[capture_key] = preds
                capture["y_true"] = y_true
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {name} (seed={seed}): {e}")
        finally:
            _release_gpu()
    return results, folds


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_breakeven(results, title, path, xkey="per_forecast_ms"):
    labels = {"per_forecast_ms": "inference cost (ms / forecast, log)",
              "usd_per_1k": "energy cost (USD / 1k forecasts, log)"}
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for r in results:
        x = max(getattr(r, xkey), 1e-9)
        ax.scatter(x, r.metrics["MASE"], s=90)
        ax.annotate(r.model, (x, r.metrics["MASE"]), xytext=(6, 4),
                    textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel(labels[xkey])
    ax.set_ylabel("MASE (lower = better)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_shap(shap, path, top=12):
    if not shap or "mean_abs_shap" not in shap:
        return False
    plt = _mpl()
    imp = shap["mean_abs_shap"].head(top)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(imp.index, imp.values)
    ax.set_xlabel("mean |SHAP|  (next-hour load driver)")
    ax.set_title("Specialist feature attribution")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return True


def plot_heterogeneity(df, prefix):
    plt = _mpl()
    # MASE distribution per model
    fig, ax = plt.subplots(figsize=(6, 4))
    for model in df["model"].unique():
        ax.hist(df[df.model == model]["MASE"].dropna(), bins=20, alpha=0.5, label=model)
    ax.set_xlabel("MASE"); ax.set_ylabel("# series"); ax.legend()
    ax.set_title("Per-series MASE distribution (heterogeneity)")
    fig.tight_layout(); fig.savefig(f"{prefix}_hetero_mase.png", dpi=150)
    # cold-start: series length vs specialist MASE
    spec = df[df.model == "lgbm_direct"]
    if not spec.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(spec["n_hours"], spec["MASE"], s=25, alpha=0.6)
        ax.set_xlabel("series length (hours)"); ax.set_ylabel("lgbm_direct MASE")
        ax.set_title("Cold-start signal: data volume vs error")
        fig.tight_layout(); fig.savefig(f"{prefix}_coldstart.png", dpi=150)


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def mode_single(args):
    y, exog = load_single(args)
    print(f"series: {y.name}  n={len(y)}  exog={list(exog.columns)}")
    cap = {} if getattr(args, "rigor", False) else None
    results, _ = evaluate(y, exog, args, capture=cap)
    if not results:
        raise SystemExit("series too short")
    if cap and len(cap.get("preds", {})) >= 2:
        from stats_rigor import paired_summary, split_conformal
        yt = cap["y_true"]
        half = len(yt) // 2   # first half of backtest = conformal calibration, second = report
        print("\n-- Rigor: split-conformal (95%) --")
        for m, p in cap["preds"].items():
            c = split_conformal(yt[:half], p[:half], yt[half:], p[half:])
            print(f"  {m:16s} coverage={c['coverage']:.3f} width={c['width']:.1f}")
        print("-- Rigor: Diebold-Mariano (|dm|>1.96 => significant at 5%) --")
        for r in paired_summary(yt, cap["preds"], h=args.horizon):
            sig = "SIG" if r["significant_5pct"] else "ns"
            print(f"  {r['model_a']:14s} vs {r['model_b']:14s} dm={r['dm_stat']:+.2f} "
                  f"p={r['p_value']:.4f} better={r['better']} [{sig}]")
        pd.DataFrame(paired_summary(yt, cap["preds"], h=args.horizon)).to_csv(
            f"{args.out_prefix}_dm_tests.csv", index=False)
        print(f"saved -> {args.out_prefix}_dm_tests.csv")
    def _r(v, nd=6):
        return round(v, nd) if v is not None else None
    rows = [{"model": r.model, "seed": r.seed, **{k: round(v, 4) for k, v in r.metrics.items()},
             "latency_s": round(r.latency_s, 3), "per_forecast_ms": round(r.per_forecast_ms, 3),
             "energy_j_per_1k": round(r.energy_j_per_1k, 2), "usd_per_1k": round(r.usd_per_1k, 6),
             "measured_j_per_1k": _r(r.measured_j_per_1k, 2),
             "measured_usd_per_1k": _r(r.measured_usd_per_1k, 8),
             "measured_cpu_j_per_1k": _r(r.measured_cpu_j_per_1k, 2),
             "measured_gpu_j_per_1k": _r(r.measured_gpu_j_per_1k, 2),
             "energy_source": r.energy_source,
             "n_params": r.n_params} for r in results]
    table = pd.DataFrame(rows)
    table.to_csv(f"{args.out_prefix}_results.csv", index=False)
    print("\n" + table.to_string(index=False))
    plot_breakeven(results, f"Accuracy vs latency — {y.name}", f"{args.out_prefix}_breakeven.png")
    plot_breakeven(results, f"Accuracy vs energy $ — {y.name}",
                   f"{args.out_prefix}_breakeven_cost.png", xkey="usd_per_1k")
    print(f"\nsaved -> {args.out_prefix}_results.csv, {args.out_prefix}_breakeven.png, "
          f"{args.out_prefix}_breakeven_cost.png")
    spec = next((r for r in results if r.model == "lgbm_direct"), None)
    if args.shap and spec and plot_shap(spec.shap, f"{args.out_prefix}_shap.png"):
        print(f"saved -> {args.out_prefix}_shap.png")
        print("\ntop SHAP drivers:\n" + spec.shap["mean_abs_shap"].head(8).round(3).to_string())
    elif args.shap and spec and spec.shap:
        print(f"[shap] {spec.shap.get('error')}")


def penalized_winner(rows: pd.DataFrame, wtp: float) -> str:
    """Cost-adjusted recommendation: argmin over MODELS (mean over any seed replicates first)
    of  MASE + wtp * usd_per_1k.
    wtp = willingness-to-pay = MASE units you'd trade for +$1 per 1k forecasts.
    wtp=0 -> pure accuracy winner; large wtp -> cheapest model wins.

    `rows` may contain multiple rows per model (e.g. 5 seeds of nas_gru) -- these are
    averaged to one row per model BEFORE the argmin. Without this, a stochastic tier with
    N seed rows effectively gets N draws at the argmin, unfairly inflating its chance of
    "winning" a cell versus deterministic tiers that contribute only one row."""
    agg = rows.groupby("model", as_index=False)[["MASE", "usd_per_1k"]].mean()
    obj = agg["MASE"] + wtp * agg["usd_per_1k"]
    return agg.loc[obj.idxmin(), "model"]


def mode_regime(args):
    """Vary training history length (cold-start -> data-rich) and report, per regime,
    the accuracy winner and the cost-adjusted winner across willingness-to-pay levels.
    This is the operator decision rule the paper delivers."""
    y, exog = load_single(args)
    weeks = [int(w) for w in args.train_weeks.split(",")]
    wtps = [float(w) for w in args.wtp.split(",")]
    # Fixed MASE denominator shared across all regimes: naive error on the full training
    # history preceding the common test window. Without this, each regime normalizes by its
    # own (different-length) training data and MASE is not comparable across regimes.
    test_start = len(y) - args.folds * args.horizon
    scale_fixed = mase_scale(y.values[:test_start])
    print(f"series: {y.name}  n={len(y)}  regimes(weeks)={weeks}  wtp={wtps}  "
          f"fixed_MASE_scale={scale_fixed:.3f}")

    recs = []
    for W in weeks:
        need = W * WEEK + args.folds * args.horizon
        if need > len(y):
            print(f"  [skip] {W}w: needs {need}h, have {len(y)}h")
            continue
        sub = y.iloc[-need:]
        sub_exog = exog.reindex(sub.index) if exog is not None and not exog.columns.empty else exog
        results, _ = evaluate(sub, sub_exog, args, mase_scale_fixed=scale_fixed)
        if not results:
            continue
        for r in results:
            recs.append({"train_weeks": W, "model": r.model, "seed": r.seed,
                         "MASE": r.metrics["MASE"],
                         "usd_per_1k": r.usd_per_1k, "per_forecast_ms": r.per_forecast_ms,
                         "measured_usd_per_1k": r.measured_usd_per_1k,
                         "energy_source": r.energy_source})
        print(f"  {W:>3}w  " + "  ".join(f"{r.model}:{r.metrics['MASE']:.3f}" for r in results))
    df = pd.DataFrame(recs)
    if df.empty:
        raise SystemExit("no regime evaluated (series too short)")
    df.to_csv(f"{args.out_prefix}_regime.csv", index=False)

    # decision table: regime x wtp -> recommended model
    table = {}
    for W in sorted(df.train_weeks.unique()):
        sub = df[df.train_weeks == W]
        table[W] = {f"wtp={w:g}": penalized_winner(sub, w) for w in wtps}
    dec = pd.DataFrame(table).T
    dec.index.name = "train_weeks"
    dec.to_csv(f"{args.out_prefix}_decision.csv")
    print("\nDecision rule (recommended model by regime x willingness-to-pay):\n"
          + dec.to_string())
    _plot_regime(df, dec, wtps, args.out_prefix)
    print(f"\nsaved -> {args.out_prefix}_regime.csv, {args.out_prefix}_decision.csv, "
          f"{args.out_prefix}_crossover.png, {args.out_prefix}_winnermap.png")


def _plot_regime(df, dec, wtps, prefix):
    plt = _mpl()
    # crossover: MASE vs training weeks, one line per model. Aggregate seed replicates
    # (e.g. 5x nas_gru) to mean +/- std BEFORE plotting -- otherwise a stochastic tier
    # draws a jagged multi-valued "line" (several y per x) instead of one clean trend.
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for m in df.model.unique():
        g = df[df.model == m].groupby("train_weeks")["MASE"].agg(["mean", "std"]).sort_index()
        ax.plot(g.index, g["mean"], marker="o", label=m)
        if g["std"].notna().any():
            ax.fill_between(g.index, g["mean"] - g["std"].fillna(0), g["mean"] + g["std"].fillna(0),
                            alpha=0.15)
    ax.set_xlabel("training history (weeks)"); ax.set_ylabel("MASE (lower = better)")
    ax.set_title("Cold-start -> data-rich crossover"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{prefix}_crossover.png", dpi=150)
    # winner map: regime rows x wtp cols, integer-coded model, annotated
    models = sorted(df.model.unique())
    code = {m: i for i, m in enumerate(models)}
    grid = dec.apply(lambda col: col.map(lambda m: code.get(m, -1))).values
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(wtps), 0.8 + 0.5 * len(dec)))
    ax.imshow(grid, aspect="auto", cmap="Set2", vmin=0, vmax=max(len(models) - 1, 1))
    ax.set_xticks(range(len(dec.columns))); ax.set_xticklabels(dec.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(dec.index))); ax.set_yticklabels([f"{w}w" for w in dec.index])
    for i in range(len(dec.index)):
        for j in range(len(dec.columns)):
            ax.text(j, i, dec.values[i, j], ha="center", va="center", fontsize=8)
    ax.set_title("Recommended model (regime x willingness-to-pay)")
    fig.tight_layout(); fig.savefig(f"{prefix}_winnermap.png", dpi=150)


def mode_sweep(args):
    recs = []
    for i, (name, y, exog) in enumerate(sweep_series(args)):
        results, _ = evaluate(y, exog, args)
        if not results:
            continue
        for r in results:
            recs.append({"series": name, "model": r.model, "n_hours": len(y),
                         "mean_load": float(y.mean()), **r.metrics,
                         "per_forecast_ms": r.per_forecast_ms, "usd_per_1k": r.usd_per_1k,
                         "measured_usd_per_1k": r.measured_usd_per_1k,
                         "energy_source": r.energy_source})
        if (i + 1) % 10 == 0:
            print(f"  ...{i + 1} series done")
    df = pd.DataFrame(recs)
    if df.empty:
        raise SystemExit("no series evaluated")
    df.to_csv(f"{args.out_prefix}_hetero.csv", index=False)
    print(f"\nseries evaluated: {df.series.nunique()}")
    summ = df.groupby("model")["MASE"].describe()[["mean", "50%", "std", "min", "max"]]
    print("\nMASE by model:\n" + summ.round(3).to_string())
    # win-rate: fraction of series where specialist beats naive
    piv = df.pivot_table(index="series", columns="model", values="MASE")
    if {"lgbm_direct", "seasonal_naive"}.issubset(piv.columns):
        wr = (piv["lgbm_direct"] < piv["seasonal_naive"]).mean()
        print(f"\nlgbm_direct beats seasonal_naive on {wr:.0%} of series")
    if "chronos" in piv.columns and "lgbm_direct" in piv.columns:
        wr = (piv["chronos"] < piv["lgbm_direct"]).mean()
        print(f"chronos beats lgbm_direct on {wr:.0%} of series (accuracy only, ignoring cost)")
    plot_heterogeneity(df, args.out_prefix)
    print(f"\nsaved -> {args.out_prefix}_hetero.csv, {args.out_prefix}_hetero_mase.png, "
          f"{args.out_prefix}_coldstart.png")


def mode_cities(args):
    """Cross-city panel: run the model set per city, compare FM vs specialist by tier
    (data-rich vs data-scarce). The paper's generality figure.

    Checkpointed: each city's rows are appended to `<out_prefix>_cities.csv` immediately
    after that city finishes, and completed cities are skipped on restart. A multi-hour,
    multi-tier run (LightGBM retrain-per-fold + NAS-GRU training + Chronos, x29 cities) can
    otherwise lose everything to a single kill — the OpenAQ fetcher had the same bug.
    """
    import types
    man = pd.read_csv(args.manifest, encoding="utf-8")
    tier = dict(zip(man.city.str.lower().str.replace(" ", "_"), man.tier))
    import glob
    files = sorted(glob.glob(os.path.join(args.data_dir, "*.csv")))
    out_path = f"{args.out_prefix}_cities.csv"
    done_cities = set()
    if os.path.exists(out_path):
        prior = pd.read_csv(out_path)
        done_cities = set(prior.city.unique())
        print(f"resuming: {len(done_cities)} cities already in {out_path}, skipping", flush=True)
    header_written = os.path.exists(out_path)
    for i, fp in enumerate(files, 1):
        slug = os.path.splitext(os.path.basename(fp))[0]
        if slug in done_cities:
            continue
        a2 = types.SimpleNamespace(**vars(args))
        # weather_csv panel runs need the real source (clips to the matching PM2.5 window);
        # everything else defaults to the generic "csv" loader, as before.
        a2.source = "weather_csv" if args.source == "weather_csv" else "csv"
        a2.data_path, a2.column = fp, None
        try:
            y, exog = load_single(a2)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}/{len(files)}] {slug}: load error {e}", flush=True)
            continue
        if args.max_years:
            y = y.iloc[-int(args.max_years * 365 * 24):]
        cap = {} if getattr(args, "save_preds", True) else None
        results, _ = evaluate(y, exog, args, capture=cap)
        if not results:
            print(f"  [{i}/{len(files)}] {slug}: too short", flush=True)
            continue
        if cap and cap.get("preds"):
            preds_dir = getattr(args, "preds_dir", None) or f"{args.out_prefix}_preds"
            os.makedirs(preds_dir, exist_ok=True)
            np.savez_compressed(os.path.join(preds_dir, f"{slug}.npz"),
                                y_true=cap["y_true"], **cap["preds"])
        city_recs = [{"city": slug, "tier": tier.get(slug, "?"), "model": r.model, "seed": r.seed,
                      "n_hours": len(y), **r.metrics, "usd_per_1k": r.usd_per_1k,
                      "measured_j_per_1k": r.measured_j_per_1k,
                      "measured_usd_per_1k": r.measured_usd_per_1k,
                      "measured_cpu_j_per_1k": r.measured_cpu_j_per_1k,
                      "measured_gpu_j_per_1k": r.measured_gpu_j_per_1k,
                      "energy_source": r.energy_source, "n_params": r.n_params}
                     for r in results]
        pd.DataFrame(city_recs).to_csv(out_path, mode="a", header=not header_written, index=False)
        header_written = True
        best = min((r for r in results), key=lambda r: r.metrics["MASE"])
        print(f"  [{i}/{len(files)}] {slug:16s} {tier.get(slug,'?'):6s} "
              f"best={best.model}({best.metrics['MASE']:.3f}) n={len(y)}  [checkpointed]", flush=True)
    df = pd.read_csv(out_path)
    if df.empty:
        raise SystemExit("no cities evaluated")

    # --- analysis ---
    print(f"\ncities evaluated: {df.city.nunique()}  "
          f"(rich={df[df.tier=='rich'].city.nunique()}, scarce={df[df.tier=='scarce'].city.nunique()})")
    print("\nmean MASE by model x tier:")
    print(df.pivot_table(index="model", columns="tier", values="MASE", aggfunc="mean").round(3).to_string())
    piv = df.pivot_table(index="city", columns="model", values="MASE")
    tser = df.drop_duplicates("city").set_index("city")["tier"]
    if {"chronos", "lgbm_direct"}.issubset(piv.columns):
        for t in ["rich", "scarce"]:
            cities_t = tser[tser == t].index
            sub = piv.loc[piv.index.isin(cities_t)]
            wr = (sub["chronos"] < sub["lgbm_direct"]).mean()
            print(f"  chronos beats specialist on {wr:.0%} of {t} cities (accuracy)")
    _plot_cities(df, piv, tser, args.out_prefix)
    print(f"\nsaved -> {args.out_prefix}_cities.csv, {args.out_prefix}_cities_bytier.png, "
          f"{args.out_prefix}_cities_advantage.png")


def _plot_cities(df, piv, tser, prefix):
    plt = _mpl()
    models = ["seasonal_naive", "lgbm_direct", "chronos", "chronos_cov"]
    models = [m for m in models if m in df.model.unique()]
    # 1) grouped bar: mean MASE by model x tier
    fig, ax = plt.subplots(figsize=(7, 4.2))
    tiers = ["rich", "scarce"]
    x = np.arange(len(models)); w = 0.38
    for k, t in enumerate(tiers):
        means = [df[(df.model == m) & (df.tier == t)]["MASE"].mean() for m in models]
        ax.bar(x + (k - 0.5) * w, means, w, label=t)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("mean MASE"); ax.set_title("Model accuracy by city tier"); ax.legend()
    ax.grid(True, axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(f"{prefix}_cities_bytier.png", dpi=150)
    # 2) per-city FM advantage = specialist MASE - chronos MASE (positive = FM better)
    if {"chronos", "lgbm_direct"}.issubset(piv.columns):
        adv = (piv["lgbm_direct"] - piv["chronos"]).dropna().sort_values()
        colors = ["#d62728" if tser.get(c) == "scarce" else "#1f77b4" for c in adv.index]
        fig, ax = plt.subplots(figsize=(7, max(4, 0.28 * len(adv))))
        ax.barh(range(len(adv)), adv.values, color=colors)
        ax.set_yticks(range(len(adv))); ax.set_yticklabels(adv.index, fontsize=7)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("specialist MASE - FM MASE  (>0: FM wins)")
        ax.set_title("Per-city FM advantage (red=data-scarce, blue=data-rich)")
        fig.tight_layout(); fig.savefig(f"{prefix}_cities_advantage.png", dpi=150)


# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    for name in ("single", "sweep", "regime", "cities"):
        sp = sub.add_parser(name)
        sp.add_argument("--source",
                        choices=["synthetic", "bdg2", "opsd", "pm25", "csv", "weather_csv", "weather_pm25"],
                        default="synthetic")
        sp.add_argument("--pm25-window-dir", default="data/cities_final",
                        help="(weather_csv only) PM2.5 city dir defining the usable window "
                             "each weather series is clipped to (ANALYSIS_PLAN.md D3)")
        sp.add_argument("--data-path", default=None)
        sp.add_argument("--column", default=None)
        sp.add_argument("--weather-path", default=None)
        sp.add_argument("--weather-col", default=None)
        sp.add_argument("--horizon", type=int, default=DAY)
        sp.add_argument("--folds", type=int, default=14)
        sp.add_argument("--retrain-per-fold", action="store_true")
        sp.add_argument("--covariate-noise", type=float, default=0.0,
                        help="R1.2 realistic-covariate scenario: scale factor on "
                             "measured NWP forecast error injected into the weather "
                             "covariates. 0 = perfect foresight (default, exact "
                             "no-op); 1 = measured real-NWP error level; >1 = worse, "
                             "toward the last-known/persistence baseline")
        sp.add_argument("--covariate-noise-seed", type=int, default=42,
                        help="RNG seed for the injected covariate error; vary across "
                             "replicates so results are not one noise draw")
        sp.add_argument("--nwp-error-csv",
                        default="results/v1/nwp_covariate_error.csv",
                        help="per-variable NWP error measured by "
                             "analysis/nwp_covariate_error.py")
        sp.add_argument("--causal-covariates", dest="causal_cov", action="store_true",
                        help="lgbm_direct ablation: weather covariates enter at forecast "
                             "origin (last-known) instead of origin+h (perfect forecast); "
                             "isolates perfect-foresight covariate advantage")
        sp.add_argument("--with-chronos", action="store_true")
        sp.add_argument("--with-nas", action="store_true",
                        help="add Green-NAS GRU tier (published NAS-discovered architecture)")
        sp.add_argument("--seeds", default=None,
                        help="comma-separated seeds for stochastic tiers (nas_gru); "
                             "e.g. 42,43,44,45,46. Default: single seed 42.")
        sp.add_argument("--chronos-model", default="amazon/chronos-bolt-small")
        sp.add_argument("--with-chronos-base", action="store_true",
                        help="add Chronos-Bolt-base as a checkpoint-scale control "
                             "within the Chronos family (205M vs the 48M small "
                             "checkpoint); separates scale from architecture family")
        sp.add_argument("--chronos-base-model", default="amazon/chronos-bolt-base")
        sp.add_argument("--with-timesfm", action="store_true",
                        help="add the TimesFM 2.5 tier (second FM family; R1.1 "
                             "generality check that results are not specific to "
                             "one architecture/checkpoint)")
        sp.add_argument("--timesfm-model", default="google/timesfm-2.5-200m-pytorch")
        sp.add_argument("--price-kwh", type=float, default=0.15, help="electricity price $/kWh")
        sp.add_argument("--cpu-tdp", type=float, default=65.0, help="CPU power draw (W)")
        sp.add_argument("--gpu-tdp", type=float, default=300.0, help="GPU power draw (W), FM on CUDA")
        sp.add_argument("--pue", type=float, default=1.4, help="datacenter PUE overhead")
        sp.add_argument("--measure-energy", action="store_true",
                        help="codecarbon-measured energy per runner (primary cost signal; "
                             "adds ~5-7s/runner for NVML+CPU-model detection). TDP proxy "
                             "(--cpu-tdp/--gpu-tdp/--pue) is always computed as a cross-check.")
        sp.add_argument("--min-hours", type=int, default=2160,
                        help="min usable contiguous hours for csv/cities quality gate (~90 days)")
        sp.add_argument("--out-prefix", default=name)
        if name == "single":
            sp.add_argument("--shap", action="store_true")
            sp.add_argument("--rigor", action="store_true",
                            help="conformal intervals + Diebold-Mariano significance tests")
        elif name == "sweep":
            sp.add_argument("--n", type=int, default=40, help="# series to sweep")
            sp.add_argument("--shap", action="store_true", help=argparse.SUPPRESS)
        elif name == "regime":
            sp.add_argument("--train-weeks", default="4,12,26,52",
                            help="history-length regimes (weeks), comma-separated")
            sp.add_argument("--wtp", default="0,0.5,2,10",
                            help="willingness-to-pay levels (MASE per $/1k), comma-separated")
            sp.add_argument("--shap", action="store_true", help=argparse.SUPPRESS)
        else:  # cities
            sp.add_argument("--manifest", default="cities_manifest.csv")
            sp.add_argument("--data-dir", default="data/cities")
            sp.add_argument("--max-years", type=float, default=0,
                            help="truncate each city to last N years (0 = full)")
            sp.add_argument("--preds-dir", default=None,
                            help="dir for per-city prediction .npz (default: <out_prefix>_preds/); "
                                 "enables post-hoc DM tests / conformal intervals without rerunning")
            sp.add_argument("--no-save-preds", dest="save_preds", action="store_false",
                            help="skip saving per-city predictions (saves disk, disables post-hoc rigor)")
            sp.set_defaults(save_preds=True)
            sp.add_argument("--shap", action="store_true", help=argparse.SUPPRESS)
    return p


def _sha256_of(path):
    import hashlib
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        return None


def _git_commit():
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__) or ".",
                                        stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def _pkg_versions():
    versions = {}
    # codecarbon governs every measured energy number and timesfm is a model tier, so both
    # belong in the provenance record; their absence was a reproducibility gap.
    for name in ("numpy", "pandas", "lightgbm", "sklearn", "torch", "chronos", "timesfm",
                 "codecarbon", "matplotlib", "shap"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[name] = None
    return versions


def _gpu_info():
    try:
        import torch
        if torch.cuda.is_available():
            return {"name": torch.cuda.get_device_name(0), "cuda": torch.version.cuda}
    except Exception:  # noqa: BLE001
        pass
    return None


def write_runconfig(args, out_prefix):
    """Audit-trail backbone: argv, resolved args, git commit, input file hashes, package
    versions, GPU, timestamps -> <out_prefix>_runconfig.json. Written once per invocation
    (not per-city/per-fold) so it always reflects exactly what was asked for and what code/
    data produced it. Referenced by paper/RESULTS_LEDGER.md rows."""
    import sys as _sys
    import datetime as _dt
    inputs = {}
    for key in ("data_path", "data_dir", "manifest", "weather_path", "weather_dir"):
        val = getattr(args, key, None)
        if val and os.path.isfile(val):
            inputs[key] = {"path": val, "sha256": _sha256_of(val)}
        elif val and os.path.isdir(val):
            files = sorted(glob.glob(os.path.join(val, "*.csv")))
            inputs[key] = {"path": val, "n_files": len(files),
                           "sha256_of_files": {os.path.basename(f): _sha256_of(f) for f in files}}
    cfg = {
        "argv": _sys.argv,
        "resolved_args": {k: v for k, v in vars(args).items()},
        "git_commit": _git_commit(),
        "inputs": inputs,
        "package_versions": _pkg_versions(),
        "gpu": _gpu_info(),
        "hostname": platform.node(),
        "python_version": _sys.version,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    path = f"{out_prefix}_runconfig.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, default=str)
    return path


def main():
    args = build_parser().parse_args()
    if args.mode not in ("cities",) and args.source != "synthetic" and not args.data_path:
        raise SystemExit(f"--data-path required for --source {args.source}")
    cfg_path = write_runconfig(args, args.out_prefix)
    print(f"[runconfig] {cfg_path}", flush=True)
    {"single": mode_single, "sweep": mode_sweep, "regime": mode_regime,
     "cities": mode_cities}[args.mode](args)


if __name__ == "__main__":
    main()
