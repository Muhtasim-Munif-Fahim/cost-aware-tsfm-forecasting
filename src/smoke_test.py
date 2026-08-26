#!/usr/bin/env python3
"""
Smoke test for the SciRep energy-forecasting paper.

Goal: on ONE series (one BDG2 building OR one OPSD grid load series), run a
rolling-origin backtest comparing three model tiers and render the accuracy-vs-cost
break-even picture BEFORE scaling to 1,600+ buildings.

Tiers
  - seasonal_naive : the floor (weekly, 168h)
  - lgbm           : the efficient specialist (calendar + lag/rolling features)
  - chronos        : the time-series foundation model, zero-shot (optional)

Metrics : MASE, nRMSE, RMSE-as-%-of-mean, MAE
Cost    : total inference wall time, per-forecast latency (ms), rough param count
Output  : results.csv + breakeven.png (x = latency/forecast, y = MASE) + console table

Runs out-of-the-box on --source synthetic (no download) to validate the pipeline.
Switch to real data with --source bdg2 --data-path <electricity_cleaned.csv>
or --source opsd --data-path <time_series_60min_singleindex.csv>.

Deps: numpy pandas lightgbm matplotlib   (optional: chronos-forecasting torch)
"""
from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HOUR = 1
DAY = 24
WEEK = 168


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_synthetic(n_hours: int = 24 * 120, seed: int = 0) -> pd.Series:
    """Deterministic hourly load with daily+weekly seasonality, trend, noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-01-01", periods=n_hours, freq="h")
    t = np.arange(n_hours)
    daily = 12 * np.sin(2 * np.pi * (t % DAY) / DAY - 1.0)
    weekly = 6 * np.sin(2 * np.pi * (t % WEEK) / WEEK)
    weekend = np.where((idx.dayofweek >= 5), -5.0, 0.0)
    trend = 0.002 * t
    noise = rng.normal(0, 2.0, n_hours)
    load = 60 + daily + weekly + weekend + trend + noise
    return pd.Series(np.clip(load, 1, None), index=idx, name="synthetic_building")


def load_bdg2(path: str, building: str | None = None) -> pd.Series:
    """BDG2 electricity_cleaned.csv: 'timestamp' col + one col per building."""
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
    if building is None:
        # first column with >=90% coverage
        cov = df.notna().mean()
        building = cov[cov >= 0.90].index[0]
    s = df[building].astype(float)
    s = s.interpolate(limit=6).dropna()
    s.name = f"bdg2:{building}"
    return s


def load_opsd(path: str, column: str | None = None) -> pd.Series:
    """OPSD time_series singleindex CSV. Pick a *_load_actual_entsoe_transparency col."""
    df = pd.read_csv(path, parse_dates=["utc_timestamp"]).set_index("utc_timestamp")
    if column is None:
        cands = [c for c in df.columns if c.endswith("_load_actual_entsoe_transparency")]
        if not cands:
            cands = [c for c in df.columns if "load_actual" in c]
        column = df[cands].notna().mean().idxmax()
    s = df[column].astype(float).interpolate(limit=6).dropna()
    s.name = f"opsd:{column}"
    return s


def load_series(args) -> pd.Series:
    if args.source == "synthetic":
        return load_synthetic()
    if args.source == "bdg2":
        return load_bdg2(args.data_path, args.column)
    if args.source == "opsd":
        return load_opsd(args.data_path, args.column)
    raise ValueError(args.source)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def mase(y_true, y_pred, y_train, m: int = DAY) -> float:
    """Mean Absolute Scaled Error; scale = in-sample MAE of m-step naive."""
    y_train = np.asarray(y_train, float)
    denom = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    denom = denom if denom > 1e-9 else 1e-9
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / denom)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def score(y_true, y_pred, y_train) -> dict:
    yt = np.asarray(y_true, float)
    r = rmse(yt, y_pred)
    mean = np.mean(yt) if np.mean(yt) != 0 else 1e-9
    return {
        "MAE": float(np.mean(np.abs(yt - np.asarray(y_pred)))),
        "RMSE": r,
        "MASE": mase(yt, y_pred, y_train),
        "nRMSE": r / np.std(yt) if np.std(yt) > 0 else np.nan,
        "RMSE_pct_mean": 100 * r / mean,
    }


# --------------------------------------------------------------------------- #
# Feature engineering (specialist)
# --------------------------------------------------------------------------- #
LAGS = [1, 2, 3, DAY, DAY + 1, WEEK]
ROLLS = [DAY, WEEK]


def make_features(s: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"y": s})
    idx = s.index
    df["hour"] = idx.hour
    df["dow"] = idx.dayofweek
    df["month"] = idx.month
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    for lag in LAGS:
        df[f"lag_{lag}"] = s.shift(lag)
    for w in ROLLS:
        df[f"rollmean_{w}"] = s.shift(1).rolling(w).mean()
    return df


# --------------------------------------------------------------------------- #
# Models — each returns (preds_over_all_folds, latency_seconds, n_params)
# --------------------------------------------------------------------------- #
@dataclass
class Fold:
    train_end: int
    y_test: np.ndarray
    test_idx: pd.DatetimeIndex


def make_folds(s: pd.Series, horizon: int, n_folds: int) -> list[Fold]:
    folds = []
    n = len(s)
    for k in range(n_folds, 0, -1):
        end = n - k * horizon
        if end <= WEEK * 2:
            continue
        folds.append(Fold(end, s.values[end:end + horizon], s.index[end:end + horizon]))
    return folds


def run_seasonal_naive(s, folds, horizon):
    preds, t0 = [], time.perf_counter()
    for f in folds:
        hist = s.values[:f.train_end]
        preds.append(hist[-WEEK:][:horizon] if len(hist) >= WEEK
                     else np.repeat(hist[-1], horizon))
    return np.concatenate(preds), time.perf_counter() - t0, 0


def run_lgbm(s, folds, horizon):
    import lightgbm as lgb
    feats = make_features(s)
    preds, latency, n_params = [], 0.0, 0
    for f in folds:
        tr = feats.iloc[:f.train_end].dropna()
        X, y = tr.drop(columns="y"), tr["y"]
        model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                  num_leaves=31, subsample=0.8, verbose=-1)
        model.fit(X, y)
        n_params = model.n_estimators * model.num_leaves  # rough proxy
        # recursive multi-step
        s_ext = s.iloc[:f.train_end].copy()
        step_preds = []
        t0 = time.perf_counter()
        for h in range(horizon):
            fr = make_features(s_ext).iloc[[-1]].drop(columns="y")
            yhat = float(model.predict(fr)[0])
            step_preds.append(yhat)
            nxt = s.index[f.train_end + h]
            s_ext.loc[nxt] = yhat
        latency += time.perf_counter() - t0
        preds.append(np.array(step_preds))
    return np.concatenate(preds), latency, n_params


def run_chronos(s, folds, horizon, model_name="amazon/chronos-bolt-small"):
    try:
        import torch
        from chronos import BaseChronosPipeline
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"chronos unavailable: {e}")
    pipe = BaseChronosPipeline.from_pretrained(
        model_name, device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float32)
    n_params = sum(p.numel() for p in pipe.model.parameters())
    preds, latency = [], 0.0
    for f in folds:
        ctx = torch.tensor(s.values[:f.train_end][-WEEK * 4:], dtype=torch.float32)
        t0 = time.perf_counter()
        q, mean = pipe.predict_quantiles(context=ctx, prediction_length=horizon,
                                         quantile_levels=[0.1, 0.5, 0.9])
        latency += time.perf_counter() - t0
        preds.append(mean[0].cpu().numpy())
    return np.concatenate(preds), latency, n_params


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    model: str
    metrics: dict = field(default_factory=dict)
    latency_s: float = 0.0
    per_forecast_ms: float = 0.0
    n_params: int = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "bdg2", "opsd"], default="synthetic")
    ap.add_argument("--data-path", default=None)
    ap.add_argument("--column", default=None, help="building/series column; auto if omitted")
    ap.add_argument("--horizon", type=int, default=DAY)
    ap.add_argument("--folds", type=int, default=14, help="rolling-origin folds")
    ap.add_argument("--with-chronos", action="store_true")
    ap.add_argument("--chronos-model", default="amazon/chronos-bolt-small")
    ap.add_argument("--out-prefix", default="smoke")
    args = ap.parse_args()

    if args.source != "synthetic" and not args.data_path:
        ap.error(f"--data-path required for --source {args.source}")

    s = load_series(args)
    print(f"series: {s.name}  n={len(s)}  span={s.index[0]} -> {s.index[-1]}")
    folds = make_folds(s, args.horizon, args.folds)
    if not folds:
        raise SystemExit("series too short for requested folds/horizon")
    y_true = np.concatenate([f.y_test for f in folds])
    y_train = s.values[:folds[0].train_end]
    n_forecasts = len(folds) * args.horizon
    print(f"folds={len(folds)} horizon={args.horizon} -> {n_forecasts} step-forecasts")

    runners = [("seasonal_naive", run_seasonal_naive), ("lgbm", run_lgbm)]
    if args.with_chronos:
        runners.append(("chronos", lambda a, b, c: run_chronos(a, b, c, args.chronos_model)))

    results = []
    for name, fn in runners:
        try:
            preds, latency, n_params = fn(s, folds, args.horizon)
            r = Result(name, score(y_true, preds, y_train), latency,
                       1000 * latency / max(n_forecasts, 1), n_params)
            results.append(r)
            print(f"  [ok] {name:14s} MASE={r.metrics['MASE']:.3f} "
                  f"latency={latency:.2f}s ({r.per_forecast_ms:.2f} ms/fc)")
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {name}: {e}")

    # ---- table + csv ----
    rows = []
    for r in results:
        rows.append({"model": r.model, **{k: round(v, 4) for k, v in r.metrics.items()},
                     "latency_s": round(r.latency_s, 3),
                     "per_forecast_ms": round(r.per_forecast_ms, 3),
                     "n_params": r.n_params})
    table = pd.DataFrame(rows)
    csv_path = f"{args.out_prefix}_results.csv"
    table.to_csv(csv_path, index=False)
    print("\n" + table.to_string(index=False))
    print(f"\nsaved -> {csv_path}")

    # ---- break-even plot: accuracy vs cost ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for r in results:
            ax.scatter(max(r.per_forecast_ms, 1e-3), r.metrics["MASE"], s=90)
            ax.annotate(r.model, (max(r.per_forecast_ms, 1e-3), r.metrics["MASE"]),
                        xytext=(6, 4), textcoords="offset points", fontsize=9)
        ax.set_xscale("log")
        ax.set_xlabel("inference cost  (ms / forecast, log)")
        ax.set_ylabel("MASE  (lower = better)")
        ax.set_title(f"Accuracy vs cost — {s.name}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        png_path = f"{args.out_prefix}_breakeven.png"
        fig.savefig(png_path, dpi=150)
        print(f"saved -> {png_path}")
    except Exception as e:  # noqa: BLE001
        print(f"[plot skipped] {e}")


if __name__ == "__main__":
    main()
