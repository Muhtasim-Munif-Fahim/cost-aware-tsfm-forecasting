#!/usr/bin/env python3
"""E4 crux experiment: transfer-learned NAS-GRU vs. zero-shot Chronos vs. lgbm-refit-on-the-
same-data-budget, in the data-scarce regime. Tests whether Green-NAS's published finding
("1% fine-tune data ~ full-data accuracy via transfer") still beats "0% data via a zero-shot
foundation model," now that small TSFMs exist. See paper/ANALYSIS_PLAN.md Sec.8 for the full
protocol and the pre-registered interpretation matrix (locked before this was run).

Protocol
  1. Pretrain NAS-GRU (Green-NAS-A, 2xGRU-128) on the POOLED rich-tier cities. Each city is
     z-scored on its OWN pre-test training region before pooling (no target-city leakage: a
     scarce city is only ever fine-tuned on, never in the pretrain corpus).
  2. For each scarce-tier city x each fine-tune fraction in {0,1,10,100}% x each seed:
       fraction=0  -> zero-shot: evaluate the pretrained model AS-IS (no fine-tuning) on this
                      city's own held-out test folds, using the city's own z-score transform.
       fraction>0  -> fine-tune: continue training from the pretrained weights on the most
                      recent `fraction`% of this city's own pre-test window (recency, not an
                      arbitrary early slice, since it's closest to the test period), then
                      evaluate on the same test folds.
  3. Comparators, same city/test-folds/fraction budget:
       - chronos zero-shot: run once per city (fraction-independent -- it never sees any
         target-city training data, by construction).
       - lgbm_direct refit on the SAME fraction of this city's history (truncate from the
         start, keep the test window fixed) -- "transfer vs. every strategy at that same
         data budget," not "transfer vs. nothing." Undefined at fraction=0 (LightGBM cannot
         train on zero rows) -- reported as NaN for that cell, chronos is the 0%-data point.

Checkpointed per (city, fraction, seed): a kill only loses the current in-flight combo, not
the whole run (same lesson as the OpenAQ fetcher / cities-mode panic of earlier phases).

Usage:
  python e4_transfer.py --manifest cities_manifest.csv --data-dir data/cities_final \
      --weather-dir data/weather --source csv --fractions 0,1,10,100 --seeds 42,43,44,45,46 \
      --out-prefix results/v1/e4_transfer/pm25_e4
"""
from __future__ import annotations

import argparse
import glob
import os
import types

import numpy as np
import pandas as pd

import run_forecast as rf

LOOKBACK = rf.DAY
MIN_WINDOW_HOURS = rf.DAY + 24 + 1   # lookback + horizon(24) + 1, the floor for any windowed split
# lgbm_direct's feature set includes a WEEK=168h lag/rolling feature (run_forecast.YLAGS/YROLLS);
# any truncated window shorter than that has every row NaN'd out by dropna(), leaving an empty
# training frame ("Input data must be 2 dimensional and non empty"). See its cutoff floor below
# (rf.WEEK + args.horizon + 10h margin for a few surviving training rows past the WEEK-lag warmup).


def _city_args(source, data_path, min_hours, weather_dir=None):
    a = types.SimpleNamespace(source=source, data_path=data_path, column=None, min_hours=min_hours)
    if weather_dir:
        a.weather_dir = weather_dir
    return a


def load_city_series(city_csv, args):
    """(y, exog) for one city, gated exactly like the main panel."""
    a = _city_args(args.source, city_csv, args.min_hours,
                   getattr(args, "weather_dir", None))
    if args.source == "weather_csv":
        a.pm25_window_dir = args.pm25_window_dir
    return rf.load_single(a)


def build_pretrain_corpus(rich_files, args):
    """Pool z-scored windows across all rich-tier cities (each on its own scale). Returns
    (Xtr, Ytr, Xval, Yval, input_dim) as torch tensors, or raises if nothing usable."""
    import torch
    Xtr_l, Ytr_l, Xval_l, Yval_l = [], [], [], []
    input_dim = None
    for fp in rich_files:
        slug = os.path.splitext(os.path.basename(fp))[0]
        try:
            y, exog = load_city_series(fp, args)
        except Exception as e:  # noqa: BLE001
            print(f"  [pretrain] skip {slug}: {e}", flush=True)
            continue
        folds = rf.make_folds(y, args.horizon, args.folds)
        if not folds:
            continue
        z, _y_mu, _y_sd, _cols = rf.zscore_city(y, exog, folds[0].train_end)
        input_dim = z.shape[1]
        n_train = folds[0].train_end
        n_val = max(int(n_train * args.val_frac), MIN_WINDOW_HOURS)
        if n_train - n_val <= LOOKBACK:
            print(f"  [pretrain] skip {slug}: too short for train/val split", flush=True)
            continue
        Xtr, Ytr = rf.build_windows(z, 0, n_train - n_val, LOOKBACK, args.horizon)
        Xval, Yval = rf.build_windows(z, n_train - n_val - LOOKBACK, n_train, LOOKBACK, args.horizon)
        if Xtr is None or Xval is None or len(Xtr) == 0 or len(Xval) == 0:
            print(f"  [pretrain] skip {slug}: no windows produced", flush=True)
            continue
        Xtr_l.append(Xtr); Ytr_l.append(Ytr); Xval_l.append(Xval); Yval_l.append(Yval)
        print(f"  [pretrain] {slug}: +{len(Xtr)} train / +{len(Xval)} val windows", flush=True)
    if not Xtr_l:
        raise SystemExit("no rich cities produced usable pretrain windows")
    return (torch.cat(Xtr_l), torch.cat(Ytr_l), torch.cat(Xval_l), torch.cat(Yval_l), input_dim)


def pretrain_nas_gru(rich_files, args):
    """Returns (state_dict, input_dim) for the pooled-rich-city-pretrained NAS-GRU."""
    import torch
    torch.manual_seed(args.pretrain_seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, Ytr, Xval, Yval, input_dim = build_pretrain_corpus(rich_files, args)
    NASGru = rf._NASGru.get()
    model = NASGru(input_dim, args.horizon).to(dev)
    model = rf.train_nas_gru(model, Xtr, Ytr, Xval, Yval, dev,
                             max_epochs=args.pretrain_epochs, patience=10, batch=256, lr=1e-3)
    print(f"  [pretrain] done: {len(Xtr)} pooled train windows, input_dim={input_dim}", flush=True)
    return {k: v.clone().cpu() for k, v in model.state_dict().items()}, input_dim


def _finetune_slice(z, n_train, fraction, horizon):
    """Most-recent `fraction`% of the pre-test window [0, n_train), split into (train, val)
    sub-windows for fine-tuning. Falls back to no-validation (fixed-epoch) fine-tuning when
    the slice is too small to support both a train and a val window -- common at fraction=1%
    on an already-short scarce city.

    NOTE (document in Methods, not a bug): `cutoff` is floored at MIN_WINDOW_HOURS, since no
    windowed model can train on less than lookback+horizon+1 hours. For a short scarce city
    this means the ACTUAL data used at a nominal fraction can exceed the requested percentage
    (e.g. "1%" of a ~2200h city floors to 49h = ~2.2%, not 1%). The reported `fraction` column
    is the REQUESTED/nominal value throughout; Phase 3 analysis should report actual hours
    used (n_train_hours in the output) alongside it rather than assume fraction is exact."""
    if fraction <= 0:
        return None
    cutoff = max(int(n_train * fraction / 100), MIN_WINDOW_HOURS)
    lo = max(n_train - cutoff, 0)
    n_val = max(int((n_train - lo) * 0.15), 0)
    if (n_train - n_val) - lo <= LOOKBACK + 1 or n_val < horizon + 1:
        # too small for a val split -- use it all for training, no early stopping
        Xtr, Ytr = rf.build_windows(z, lo, n_train, LOOKBACK, horizon)
        return (Xtr, Ytr, None, None) if Xtr is not None and len(Xtr) > 0 else None
    Xtr, Ytr = rf.build_windows(z, lo, n_train - n_val, LOOKBACK, horizon)
    Xval, Yval = rf.build_windows(z, n_train - n_val - LOOKBACK, n_train, LOOKBACK, horizon)
    if Xtr is None or len(Xtr) == 0:
        return None
    return (Xtr, Ytr, Xval, Yval)


def run_city_e4(city_csv, pretrained_state, input_dim, args, done_combos):
    """Yield result dicts for one scarce city across all (fraction, seed) combos not already
    in `done_combos` (resume support), plus the fraction-independent chronos comparator."""
    import torch
    slug = os.path.splitext(os.path.basename(city_csv))[0]
    y, exog = load_city_series(city_csv, args)
    folds = rf.make_folds(y, args.horizon, args.folds)
    if not folds:
        print(f"  [{slug}] too short for {args.folds} folds, skip", flush=True)
        return
    y_true = np.concatenate([f.y_test for f in folds])
    y_train_full = y.values[:folds[0].train_end]
    scale = rf.mase_scale(y_train_full)
    n_train = folds[0].train_end
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    NASGru = rf._NASGru.get()

    if (slug, "chronos_zeroshot", None, None) not in done_combos:
        try:
            preds, latency, n_params, _ = rf.run_chronos(y, exog, folds, args.horizon)
            mase = rf.score(y_true, preds, y_train_full, scale)["MASE"]
            yield {"city": slug, "strategy": "chronos_zeroshot", "fraction": None, "seed": None,
                  "MASE": mase, "n_train_hours": n_train}
        except Exception as e:  # noqa: BLE001
            print(f"  [{slug}] chronos error: {e}", flush=True)

    z, y_mu, y_sd, _cols = rf.zscore_city(y, exog, n_train)

    for frac in args.fractions:
        # lgbm-refit-on-fraction: deterministic given data -> compute once per (city, fraction),
        # not per seed. Undefined at fraction=0 (no data to train on).
        if frac > 0 and (slug, "lgbm_refit", frac, None) not in done_combos:
            cutoff = max(int(n_train * frac / 100), rf.WEEK + args.horizon + 10)
            lo = max(n_train - cutoff, 0)
            y_trunc = y.iloc[lo:]
            exog_trunc = exog.reindex(y_trunc.index) if exog is not None and not exog.empty else exog
            try:
                # rebuild folds relative to the truncated series so train_end still lands
                # at the same absolute point as the untruncated folds (same test window)
                trunc_folds = [rf.Fold(f.train_end - lo, f.y_test) for f in folds]
                preds, latency, n_params, _ = rf.run_lgbm_direct(
                    y_trunc, exog_trunc, trunc_folds, args.horizon, retrain_per_fold=True)
                mase = rf.score(y_true, preds, y_train_full, scale)["MASE"]
                yield {"city": slug, "strategy": "lgbm_refit", "fraction": frac, "seed": None,
                      "MASE": mase, "n_train_hours": cutoff}
            except Exception as e:  # noqa: BLE001
                print(f"  [{slug}] lgbm_refit frac={frac}: {e}", flush=True)

        # fraction=0 (zero-shot) is DETERMINISTIC: the freshly-constructed model's random
        # init is immediately overwritten by load_state_dict(pretrained_state), so no
        # fine-tuning happens and `seed` cannot affect the prediction at all. Looping over
        # every seed here would just recompute the identical number N times (confirmed
        # empirically: two seeds gave bit-identical MASE) -- wasted compute in the real
        # campaign and a confusing ledger row ("5 different seeds" that are actually one
        # number repeated). Compute once with seed=None, matching chronos_zeroshot's
        # convention, exactly like lgbm_refit already does for its seed-independent case.
        seed_iter = [None] if frac == 0 else args.seeds
        for seed in seed_iter:
            if (slug, "nas_transfer", frac, seed) in done_combos:
                continue
            torch.manual_seed(seed if seed is not None else 0)
            model = NASGru(input_dim, args.horizon).to(dev)
            model.load_state_dict({k: v.to(dev) for k, v in pretrained_state.items()})
            if frac > 0:
                sl = _finetune_slice(z, n_train, frac, args.horizon)
                if sl is not None:
                    Xtr, Ytr, Xval, Yval = sl
                    if Xval is not None:
                        model = rf.train_nas_gru(model, Xtr, Ytr, Xval, Yval, dev,
                                                 max_epochs=args.finetune_epochs, patience=5,
                                                 batch=min(64, len(Xtr)), lr=5e-4)
                    else:
                        # too little data for a val split -- fixed-epoch fine-tune, no early stop
                        import torch as _t
                        opt = _t.optim.Adam(model.parameters(), lr=5e-4)
                        lossf = _t.nn.MSELoss()
                        dl = _t.utils.data.DataLoader(_t.utils.data.TensorDataset(Xtr, Ytr),
                                                      batch_size=min(64, len(Xtr)), shuffle=True)
                        model.train()
                        for _ in range(min(args.finetune_epochs, 10)):
                            for xb, yb in dl:
                                opt.zero_grad()
                                loss = lossf(model(xb.to(dev)), yb.to(dev))
                                loss.backward()
                                opt.step()
            model.eval()
            preds = []
            with torch.no_grad():
                for f in folds:
                    ctx = torch.tensor(z[f.train_end - LOOKBACK:f.train_end][None]).to(dev)
                    out = model(ctx)[0].cpu().numpy()
                    preds.append(out * y_sd + y_mu)
            preds = np.concatenate(preds)
            mase = rf.score(y_true, preds, y_train_full, scale)["MASE"]
            # actual fine-tune data used, not the city's full window (0 for zero-shot; must
            # match _finetune_slice's own cutoff formula so this is truthful, not just n_train)
            actual_hours = 0 if frac == 0 else max(int(n_train * frac / 100), MIN_WINDOW_HOURS)
            yield {"city": slug, "strategy": "nas_transfer", "fraction": frac, "seed": seed,
                  "MASE": mase, "n_train_hours": actual_hours}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="cities_manifest.csv")
    ap.add_argument("--data-dir", default="data/cities_final")
    ap.add_argument("--weather-dir", default="data/weather")
    ap.add_argument("--pm25-window-dir", default="data/cities_final")
    ap.add_argument("--source", choices=["csv", "weather_csv"], default="csv")
    ap.add_argument("--min-hours", type=int, default=2160)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--fractions", default="0,1,10,100")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--pretrain-seed", type=int, default=42)
    ap.add_argument("--pretrain-epochs", type=int, default=50)
    ap.add_argument("--finetune-epochs", type=int, default=15)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--out-prefix", default="e4")
    ap.add_argument("--lookback", type=int, default=None,
                    help="GRU context length in hours (default 24; set 168/672 for the "
                         "context-matched sensitivity so the transfer model can see the "
                         "weekly cycle the 4-week Chronos context already carries)")
    args = ap.parse_args()
    args.fractions = [float(f) for f in args.fractions.split(",")]
    args.seeds = [int(s) for s in args.seeds.split(",")]
    if args.lookback is not None:
        global LOOKBACK, MIN_WINDOW_HOURS
        LOOKBACK = args.lookback
        MIN_WINDOW_HOURS = LOOKBACK + args.horizon + 1
        print(f"[context-matched] LOOKBACK={LOOKBACK}h", flush=True)

    man = pd.read_csv(args.manifest, encoding="utf-8")
    slug_tier = dict(zip(man.city.str.lower().str.replace(" ", "_"), man.tier))
    files = sorted(glob.glob(os.path.join(args.data_dir, "*.csv")))
    rich_files = [f for f in files if slug_tier.get(os.path.splitext(os.path.basename(f))[0]) == "rich"]
    scarce_files = [f for f in files if slug_tier.get(os.path.splitext(os.path.basename(f))[0]) == "scarce"]
    print(f"rich (pretrain corpus): {len(rich_files)} cities | scarce (fine-tune targets): "
          f"{len(scarce_files)} cities", flush=True)

    out_path = f"{args.out_prefix}_results.csv"
    ckpt_path = f"{args.out_prefix}_pretrained.pt"
    done_combos = set()
    if os.path.exists(out_path):
        prior = pd.read_csv(out_path)
        done_combos = set(
            (r.city, r.strategy, None if pd.isna(r.fraction) else r.fraction,
             None if pd.isna(r.seed) else int(r.seed))
            for r in prior.itertuples())
        print(f"resuming: {len(done_combos)} combos already in {out_path}", flush=True)
    header_written = os.path.exists(out_path)

    import torch
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        pretrained_state, input_dim = ckpt["state_dict"], ckpt["input_dim"]
        print(f"loaded cached pretrained model from {ckpt_path}", flush=True)
    else:
        pretrained_state, input_dim = pretrain_nas_gru(rich_files, args)
        torch.save({"state_dict": pretrained_state, "input_dim": input_dim}, ckpt_path)
        print(f"saved pretrained model -> {ckpt_path}", flush=True)

    for i, fp in enumerate(scarce_files, 1):
        slug = os.path.splitext(os.path.basename(fp))[0]
        print(f"[{i}/{len(scarce_files)}] {slug} ...", flush=True)
        try:
            for rec in run_city_e4(fp, pretrained_state, input_dim, args, done_combos):
                pd.DataFrame([rec]).to_csv(out_path, mode="a", header=not header_written, index=False)
                header_written = True
                done_combos.add((rec["city"], rec["strategy"], rec["fraction"], rec["seed"]))
                print(f"    {rec['strategy']:16s} frac={rec['fraction']} seed={rec['seed']} "
                      f"MASE={rec['MASE']:.3f}  [checkpointed]", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [{slug}] city-level error: {e}", flush=True)

    print("E4 DONE", flush=True)


if __name__ == "__main__":
    main()
