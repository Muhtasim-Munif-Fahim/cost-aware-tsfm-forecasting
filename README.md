# Cost-Aware Evaluation of Time-Series Foundation Models for Urban Air-Quality and Temperature Forecasting

Research code, data pipeline, and results for a paper targeting **Scientific Reports**.
The study asks a practical, cost-aware question about time-series foundation models (TSFMs)
across **two forecasting domains — hourly PM2.5 and 2 m temperature — over 29 cities**.

## Research question

> **When** does a small zero-shot time-series foundation model match or beat a tuned,
> efficient local specialist once you price in deployment (compute/energy) cost — and how do
> the three deployment strategies (a NAS-discovered tiny specialist, transfer learning from
> data-rich cities, and a zero-shot foundation model) compare across data-availability
> regimes and across data-rich vs. data-scarce cities?

**Deliverable:** an operator decision rule — *which model class to deploy per regime,
cost-adjusted* — with particular relevance to data-scarce, low-resource cities. Every
quantitative claim in the manuscript is traced to an artifact and command in
[`paper/RESULTS_LEDGER.md`](paper/RESULTS_LEDGER.md) and verified by the audit gate
(`analysis/number_audit.py`).

## Why it fits Scientific Reports

- SR publishes PM2.5 / load forecasting continuously; review is **soundness-only**, rewarding
  rigor (strong baselines, multiple seeds/folds, interpretability, reproducibility) over novelty.
- The gap is real (verified via Consensus): existing work optimizes accuracy only; none price
  deployment cost or give a cross-city decision rule.
- Interdisciplinary (ML × environmental health × sustainability/equity) = SR's sweet spot.

## Models compared

| Tier | Model | Notes |
|------|-------|-------|
| Floor | `seasonal_naive` | weekly (168 h) persistence |
| Specialist (GBM) | `lgbm_direct` | LightGBM, one model per horizon step (direct, no recursion); calendar + weather + lag/rolling features |
| Specialist (NAS) | `nas_gru` | Green-NAS-discovered tiny GRU (from the IEEE QPAIN conference study); trained locally per city, and the transfer-learning source when fine-tuned across cities |
| Foundation model | `chronos` | Amazon Chronos-Bolt, zero-shot, univariate |
| FM + covariates | `chronos_cov` | covariate model (Ridge on calendar+weather) → FM forecasts residual → covariate effect added back (ablation) |

The three **deployment strategies** the paper contrasts map onto these tiers: the tiny
local specialist (`nas_gru` / `lgbm_direct`), transfer learning (`nas_gru` fine-tuned from
data-rich cities, see `src/e4_transfer.py`), and the zero-shot foundation model (`chronos`).

**Metrics:** MASE, nRMSE, RMSE-as-%-of-mean, MAE.
**Cost:** inference latency, energy (J) and USD per 1k forecasts (TDP × latency × PUE proxy — see caveats).
**Interpretability:** SHAP on the specialist.

## Data (all open)

| Dataset | Role | Access |
|---------|------|--------|
| **UCI Beijing Multi-Site PM2.5** | 12 stations, 2013–17 hourly, incl. meteorology — primary/data-rich | `data/beijing_pm25/` (downloaded from UCI) |
| **OpenAQ city panel** | data-rich vs. data-scarce cities, PM2.5 hourly | `data/cities/*.csv` via `city_select.py` + `batch_fetch.py`; see `cities_manifest.csv` |
| **Open-Meteo** | 2 m temperature + meteorological covariates per city | `data/weather/*.csv` via `openmeteo_fetch.py` |

Two forecasting **domains** are evaluated across **29 cities**: hourly **PM2.5** and 2 m
**temperature**. The temperature domain is where the "specialist wins" result is stress-tested
against a perfect-foresight-covariate artifact (see the manuscript's foresight ablation).

## Repository layout

```
MANUSCRIPT/           Built manuscript + supplement PDFs (the readable deliverables).
src/                  All executable code (run from the repo root, e.g. `python src/run_forecast.py`):
  run_forecast.py       Main harness. Modes: single | sweep | regime | cities.
  e4_transfer.py        E4 crux experiment (transfer vs zero-shot across fine-tune budgets).
  city_select.py        Pick longest-record PM2.5 sensor per target city -> cities_manifest.csv
  openaq_fetch.py       Fetch one OpenAQ sensor's full hourly history -> timestamp,PM2.5 CSV
  batch_fetch.py        Fetch every sensor in the manifest -> data/cities/<city>.csv
  openmeteo_fetch.py    Fetch weather covariates -> data/weather/<city>.csv
  data_audit.py         Data-integrity checks over the fetched panel.
  stats_rigor.py        Standalone statistical-rigor helpers.
  smoke_test.py         Minimal 3-tier validator (quick sanity check).
  watchdog_*.sh         Self-healing fetch/panel loops (run from repo root).
paper/                Manuscript prose: sections/*.md (canonical), latex/ (build scripts only),
                      RESULTS_LEDGER.md (claim -> artifact ledger), ANALYSIS_PLAN.md (pre-specified plan).
analysis/             Post-hoc analyses (conformal, equivalence, energy, contamination, audit).
figures/              Figure scripts -> figures/main/F*.{pdf,png}.
tables/               Table generator -> tables/out/*.{tex,md}.
results/              Canonical result artifacts (results/v1/...); the reproducibility record.
data/                 beijing_pm25/ (UCI), cities/ + weather/ (OpenAQ + Open-Meteo pulls).
cities_manifest.csv   The city panel (city, tier, sensor_id, span) -- repo-root data manifest.
cities_quality.csv    Per-city gate outcomes (usable window, hours, PASS flag).
```

> The manuscript LaTeX source and journal template are **not included** in this public
> repository; the built `MANUSCRIPT/*.pdf` are provided as the readable deliverables. The
> `paper/latex/` build scripts (`md2tex.py`, `make_submission.py`, `build.sh`) document the
> `sections/*.md` → LaTeX → PDF pipeline; the `.tex`/`.cls`/`.bib` inputs are regenerated
> locally (LaTeX source available from the authors on request).

## Reproduce

Dependencies:
```bash
pip install numpy pandas lightgbm matplotlib shap
pip install torch chronos-forecasting          # foundation-model tier (GPU recommended)
```

### 1. Get the data
Beijing (UCI) is already under `data/beijing_pm25/`. To rebuild the city panel:
```bash
export OPENAQ_KEY=<your-openaq-api-key>          # free key from explore.openaq.org
python src/city_select.py                             # -> cities_manifest.csv
python src/batch_fetch.py                             # -> data/cities/*.csv  (~1 hr)
```

### 2. Single station (metrics + break-even + SHAP)
```bash
python src/run_forecast.py single --source pm25 \
  --data-path data/beijing_pm25/PRSA_Data_20130301-20170228 \
  --column Aotizhongxin --with-chronos --shap --out-prefix pm25_fm
```

### 3. Regime study — the decision-rule figure (fair settings)
```bash
python src/run_forecast.py regime --source pm25 \
  --data-path data/beijing_pm25/PRSA_Data_20130301-20170228 \
  --column Aotizhongxin --with-chronos --retrain-per-fold \
  --train-weeks 4,12,26,52,104 --wtp 0,500,1500,5000,20000 --out-prefix pm25_fair
```
Outputs: `*_decision.csv` (regime × willingness-to-pay → recommended model),
`*_winnermap.png`, `*_crossover.png`.

### 4. Cross-city panel — the generality figure
```bash
python src/run_forecast.py cities --data-dir data/cities --manifest cities_manifest.csv \
  --with-chronos --retrain-per-fold --out-prefix pm25_panel
```
Outputs: `*_cities.csv`, `*_cities_bytier.png` (accuracy by tier),
`*_cities_advantage.png` (per-city FM advantage, colored rich/scarce).

Key flags: `--folds` (rolling-origin folds), `--horizon` (default 24 h),
`--retrain-per-fold` (fair, for publication runs; default trains once for speed),
cost knobs `--price-kwh --cpu-tdp --gpu-tdp --pue`.

## Methodology notes (important)

- **`--retrain-per-fold`** gives the specialist a fair shot; the default (train-once) is a
  speed shortcut for screening only. Use the flag for all reported numbers.
- **Fixed MASE scale (regime mode):** MASE is normalized by one shared denominator computed
  from pre-test history, so MASE is comparable across regimes (otherwise each regime normalizes
  by its own training data and the crossover plot shows a normalization artifact, not accuracy).
- **Regime test window is fixed:** all regimes evaluate the same final window; only training
  history length before it varies.

## Headline findings

Numbers below are the manuscript's canonical, ledger-backed values (see
[`paper/RESULTS_LEDGER.md`](paper/RESULTS_LEDGER.md); read the full paper in
[`MANUSCRIPT/manuscript.pdf`](MANUSCRIPT/manuscript.pdf)):

- **PM2.5:** the zero-shot foundation model and the local specialist are statistically
  indistinguishable under deployable causal covariates (MASE **0.662 vs. 0.692**).
- **Temperature:** the specialist's apparent edge (0.533 vs. 0.792) **shrinks to a small,
  non-significant difference (0.745)** once its covariates are restricted to values knowable
  at forecast time — a perfect-foresight artifact that can manufacture spurious cross-domain
  conclusions.
- **Transfer learning** never significantly beats zero-shot at any fine-tune budget.
- **Energy:** under frequent retraining the untrained (zero-shot) model uses ~**10× less**
  measured energy; the gap is training-driven, so a *once-trained* specialist eventually
  becomes the more energy-efficient option (amortization crossover).

The statistical layer backing these claims — Diebold–Mariano, split-conformal coverage, TOST
equivalence, sign/Wilcoxon/Friedman, FDR control — lives in `analysis/` and is re-derived from
`results/v1/` by `analysis/number_audit.py`.

## Known limitations

1. Two forecasting domains (PM2.5, 2 m temperature) and one primary horizon; broader horizons
   are reported in the supplement.
2. Foundation-model tier is Chronos-Bolt; other TSFMs (TimesFM/Moirai) are not exhaustively swept.
3. Energy is **measured** (codecarbon) on the run hardware; absolute figures are hardware- and
   region-dependent, and on Windows hosts a constant-power estimate is used where NVML/RAPL is
   unavailable — treat cross-study absolute joules with care, relative comparisons are the claim.
4. Per-city records vary in quality; the data-quality gate (`src/data_audit.py`,
   `cities_quality.csv`) documents the usable-window and min-hours filtering.

## Notes

- The OpenAQ API key must be provided via the `OPENAQ_KEY` environment variable; it is never
  written to disk or committed.
- Watchdog/build scripts are bash (`*.sh`); on a Windows host run them under Git Bash / WSL.
- Early scaffolding used a synthetic generator and briefly an energy-forecasting framing
  (BDG2/OPSD); the harness still supports `--source synthetic|bdg2|opsd` for those baselines.

## Licensing

- **Code** — MIT ([`LICENSE`](LICENSE)).
- **Figures, tables, derived data, and result artifacts** — CC BY 4.0
  ([`LICENSE-DATA-FIGURES.md`](LICENSE-DATA-FIGURES.md)).
- **Raw input data** are not redistributed; they remain under the terms of UCI, OpenAQ, and
  Open-Meteo (regenerate via the pipeline above).

## Citation

If you use this code, data, or results, please cite the paper — see
[`CITATION.cff`](CITATION.cff). A Zenodo archive DOI will be added on release.
