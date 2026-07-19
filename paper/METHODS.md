# Methods (living document — becomes the manuscript's Methods section)

Status: **skeleton**. Each subsection is filled in during the phase noted, and finalized against
actual `_runconfig.json` records in Phase 5 — nothing here should describe intended behavior that
wasn't actually run; if code changes, this document changes with it.

## 1. Study design & research question
*(Phase 6 draft, informed by ANALYSIS_PLAN.md §1)*
Three deployment strategies × two domains, cost-aware framing. Relationship to the IEEE QPAIN
conference paper (Green-NAS): what is inherited (NAS-discovered architecture, weather domain,
edge/Global-South framing) vs. new (foundation-model tier, cost-adjusted decision rule, air-quality
domain, cross-domain generality test, transfer-vs-zero-shot crux).

## 2. Data sources & acquisition
*(Phase 1/2, filled as fetches are finalized)*
- OpenAQ v3 PM2.5: sensor IDs in `cities_manifest.csv`; month-windowed, resumable, checkpointed
  fetch (`openaq_fetch.py`) — rationale: OpenAQ's deep pagination is unstable (observed losing
  most of a year's data on a single large page request); month windows keep every request under
  the API's page limit and verifiable (`meta.found == len(results)`).
- Open-Meteo historical archive (temperature_2m + 7 covariates, same set as Green-NAS's
  `STANDARD_FEATURES`): free, no key, gap-free reanalysis (`openmeteo_fetch.py`).
- UCI Beijing Multi-Site Air Quality (12 stations, 2013–2017, hourly, meteorology included).
- Access dates, licenses: TODO fill from fetch logs.

## 3. Data quality gating & preprocessing
*(Phase 1.1, 2.2 — see ANALYSIS_PLAN.md §2 for locked parameters)*
`sanitize_pm25` rule and prevalence found; `extract_usable_window` procedure; sub-hourly
resampling (Mumbai); pre-specified exclusions with pointer to Table S5; note that all Beijing
numbers in this paper are post-sanitization regenerations (changelog vs. any earlier reporting
in Table S8).

## 4. Panel construction & regime definitions
29-city panel (rich/scarce tier), per-city usable windows (Table S1/S2); weather-domain window
matching rationale; training-history regime grid (weeks).

## 5. Forecasting task & evaluation protocol
Hourly cadence, 24h direct multi-horizon (48h sensitivity, supplement), 6-fold rolling-origin
backtest, fixed common test window across regimes, fixed MASE denominator rationale.

## 6. Model tiers & training
Per tier, filled from actual configs: `seasonal_naive`; `lgbm_direct` (feature set, per-horizon
direct models, retrain-per-fold, hyperparameters incl. corrected bagging config, random_state);
`nas_gru` (Green-NAS-A 2×GRU-128 provenance, Adam 1e-3, early stopping patience 10, ≤50 epochs,
5 seeds); `chronos` / `chronos_cov` (Bolt-small, 4-week context window, residual-covariate
construction); parameter counts per tier.

## 7. E4 transfer-learning protocol
Pretraining corpus construction, per-city z-scoring, no-leakage guarantee, fine-tune fractions,
optimizer/epochs, comparators, identical test folds. (ANALYSIS_PLAN.md §8)

## 8. Metrics
MASE (m=24) definition and fixed-scale rationale, nRMSE, MAE, RMSE-%-of-mean; win-count definition.

## 9. Cost & energy measurement
codecarbon protocol, NVML vs. CPU-estimate distinction (explicit Windows limitation statement),
hardware spec, PUE/$-per-kWh assumptions, per-1k-forecast normalization, repeatability protocol,
TDP-proxy cross-check, cost-adjusted objective definition.

## 10. Statistical framework
Diebold–Mariano (HLN correction, h, loss function), per-city + panel-level sign test + Wilcoxon,
Friedman–Nemenyi supplement, split-conformal protocol (calibration/report split, pooled vs.
per-city), multi-seed aggregation rule, bootstrap CI method for the advantage-correlation.

## 11. Interpretability
DROPPED (2026-07-16 supervisor review, B7): SHAP was never part of the locked analysis
plan and was not run in the canonical campaign (the harness supports `--shap` in
`single` mode only; no canonical artifact exists). The manuscript subsection promising
SHAP "in the supplement" was removed rather than backfilled. If SHAP is wanted for a
revision, run it as a logged post-hoc addition.

## 12. Reproducibility statement
Code tag (`v1.0-campaign`), seeds, run-config/ledger system description, data availability
statement, environment (Python version, GPU), OpenAQ key handling (env var only, never
committed/logged).
