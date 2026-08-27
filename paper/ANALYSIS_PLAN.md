# Analysis Plan (pre-specified, version-control-locked) — v1.0-campaign

Locked before the canonical compute campaign (Phase 2) runs. Any deviation from this document
during or after the campaign must be logged in the "Deviations" section at the bottom with a
reason — it cannot be silently edited to match results after the fact.

## 1. Study design

Two deployment strategies compared against a foundation model, across two domains:
- **Search** (Green-NAS-A: 2×GRU-128, NSGA-II-discovered in the published IEEE QPAIN conference
  paper) — the "make it tiny" strategy.
- **Specialist** (LightGBM + weather/calendar covariates, direct multi-horizon) — the "make it
  efficient" baseline strategy.
- **Foundation model** (Chronos-Bolt-small, zero-shot; + a covariate-residual variant) — the
  "don't train anything" strategy.
- **Floor**: seasonal-naive (168h persistence).

Domains: (a) urban PM2.5 air quality (primary), (b) 2m air temperature (secondary, same cities,
same harness, Green-NAS linkage).

## 2. Panel membership (locked)

**29 cities** = all rows in `cities_quality.csv` with `PASS == True` under the gate below, PLUS
Mumbai once its sub-hourly resampling fix (Phase 1.1) is applied and re-audited.

**Gate parameters (from `extract_usable_window` in `run_forecast.py`, fixed before this campaign,
not tuned post-hoc):**
- `segment_break = 48h` — split the series wherever the gap since the previous observation exceeds this.
- `interp_limit = 6h` — interpolate only gaps this short or shorter inside the kept segment.
- `min_hours = 2160` (90 days) — minimum length of the final usable window.
- `min_cov = 0.6` — minimum fraction of that window that must be real (pre-interpolation) observations.
- Value sanitization (`sanitize_pm25`): PM2.5 readings `<= 0` or `>= 985` are treated as missing
  before gating (985 is a repeated sensor/API sentinel found contaminating 12/29 raw city files;
  see `archive/pilot_2026-07-13/README.md`).

**Pre-registered exclusions** (documented, not silently dropped): London (sensor 206 has only 247
real hourly readings across its entire nominal 2016–2026 span — a hardware/uptime limitation, not
a pipeline bug), Kathmandu, Kolkata (both fail `min_hours` after gap-interpolation despite
reasonable raw row counts — internal gap structure, confirmed by direct inspection).

**Weather-domain windows**: for each city, the temperature-forecasting window is clipped to
that same city's PM2.5 usable window (same start/end timestamps), not to the full Open-Meteo
history. Rationale: this preserves the data-scarcity structure city-for-city, so a cross-domain
comparison isolates the *domain* effect rather than confounding it with "weather data happens to
be longer."

## 3. Task & evaluation protocol

- Cadence: hourly. Primary horizon: **24h** direct multi-horizon (one model per lead time for
  the specialist tier). Secondary/supplement: **48h**.
- Backtest: rolling-origin, **6 folds**, evaluated on the final common window per series (all
  regimes/history-lengths tested against the *same* test window — only training history varies).
- **MASE denominator**: one fixed scale per series, computed from the naive (m=24) in-sample
  error over the pre-test training history — shared across all regimes for that series so MASE
  is comparable across different training-history lengths (this was a bug in an early pilot run;
  see archive README).

## 4. Model tiers & seeds

- `seasonal_naive`, `lgbm_direct`, `chronos`, `chronos_cov`: deterministic given data (LightGBM
  `random_state=42` fixed in Phase 1.4; the Phase-1 audit also found `subsample=0.8` was set
  without `subsample_freq`, making bagging inert — resolved in Phase 1.4, either by setting
  `subsample_freq=1` or removing the parameter, and documented in the hyperparameter table either way).
- `nas_gru`: stochastic (init + minibatch order) — run at **5 seeds: {42, 43, 44, 45, 46}**.
  Report mean ± sd across seeds; no cherry-picking a "best seed."
- E4 transfer experiment: same 5 seeds, crossed with fine-tune fraction.

## 5. Cost & energy measurement

- Primary: **codecarbon**-measured energy (kWh → J → USD/1k forecasts), GPU via NVML (real
  measurement for chronos/chronos_cov/nas_gru on the RTX 3060 Ti), CPU via codecarbon's constant-
  power estimate on Windows (weaker signal — stated as a limitation, not hidden).
  - Repeatability: fixed workload rerun 5× on 3 cities → report mean ± sd; if sd/mean > 20% for
    a tier, report as a range in the main table rather than a point estimate.
- Secondary (supplementary, cross-check only): TDP × latency × PUE proxy already implemented
  (`cost_of()` in `run_forecast.py`).
- Assumptions: PUE and $/kWh held at the harness defaults; a sensitivity table (S12) varies both.

## 6. Statistical framework

- **Primary comparison**: per-city Diebold–Mariano test (absolute-error loss, Harvey-Leybourne-
  Newbold small-sample correction, h=24) between each pair of tiers, on the held-out backtest
  predictions.
- **Panel-level**: binomial sign test on per-city win/loss (specialist vs. FM), plus a Wilcoxon
  signed-rank test on paired per-city MASE. Friedman + Nemenyi post-hoc across all 5 tiers as a
  supplementary robustness check.
- **Uncertainty**: split-conformal prediction intervals (95%), calibrated on the first half of
  each series' backtest predictions, reported on the second half. Primary: pooled per tier×domain.
  Supplementary: per-city.
- **FM-advantage vs. data-volume**: Pearson correlation between (usable hours) and (lgbm MASE −
  chronos MASE) across the panel, with a bootstrap (10,000 resample) 95% CI on the correlation.

## 7. Cost-adjusted decision rule

Objective: `MASE + wtp * usd_per_1k`, minimized over tiers, where `wtp` ("willingness to pay") is
swept over a fixed grid (`{0, 500, 1500, 5000, 20000}` MASE-units per $/1k forecasts, matching the
harness's existing `regime` mode default). Winner maps are produced per (training-history regime ×
wtp) cell, per domain, for: Beijing (primary depth city), plus one rich (Seoul — longest usable
window in the rich tier) and one scarce (Nairobi — longest usable window in the scarce tier) panel
city, in both domains.

## 8. E4 — transfer-learned NAS-GRU vs. zero-shot foundation model (the crux experiment)

- **Question**: does Green-NAS's published finding ("1% fine-tune data ≈ full-data accuracy via
  transfer") still beat "0% data via a zero-shot foundation model," now that small TSFMs exist?
- **Protocol**: pretrain NAS-GRU on the pooled **rich-tier** cities (per-city z-scoring so scale
  differences don't leak; the target scarce city is never in the pretrain corpus). Fine-tune on
  each **scarce**-tier city (all 14 that pass the panel gate) at **{0%, 1%, 10%, 100%}** of that
  city's training window, 5 seeds each. Compare against: zero-shot Chronos on the same city/test
  window, and an lgbm_direct model refit on the same fraction of data (so the comparison isn't
  "transfer vs. nothing" but "transfer vs. every strategy at that same data budget").
- **Pre-registered interpretation matrix** (locked before running, so no outcome requires ad hoc
  spin):
  1. Transfer beats zero-shot FM at all fractions → Green-NAS's thesis holds; report transfer as
     the recommended data-scarce strategy.
  2. Transfer beats FM only at 10%+ → refined rule: "transfer needs a minimum seed of local data;
     below that, use the FM."
  3. Zero-shot FM beats transfer even at 100% fine-tune → the small-TSFM calculus has fully
     inverted the conference paper's premise; report as the headline reversal.
  4. All three (transfer, FM, lgbm-on-fraction) are statistically indistinguishable (DM tests) →
     report as "strategy choice doesn't matter at this scale; cost/simplicity should decide,"
     and let the cost table make the recommendation.

## 9. Primary endpoints (what the abstract will report)

1. Panel mean MASE per tier per domain (± sd across cities for the fixed tiers, ± sd across
   seeds×cities for nas_gru).
2. Per-city win counts + DM-significant win counts, specialist vs. FM.
3. Correlation (usable hours, FM advantage) + bootstrap CI.
4. Cost-adjusted decision-rule winner maps (which cells flip to which tier).
5. E4 outcome classified into one of the four interpretation-matrix buckets above.

## 10. Deviations log

*(Append here, dated, with reason, if anything in this document changes after Phase 2 begins.)*

- **2026-07-13 (Phase 1.3 implementation finding, not a plan change):** clarifying, not
  deviating — the TDP-proxy cost (`energy_j_per_1k`) measures inference latency only for
  every tier; the codecarbon-measured cost (`measured_j_per_1k`) wraps the entire runner
  call, which for `nas_gru` includes model training. The two are therefore NOT directly
  comparable for that tier (proxy = inference-only, measurement = train+infer). Both are
  reported in every output row; §5/§9 analysis and Methods §9 must state this explicitly
  rather than silently averaging or ratio-ing across the two energy columns.

- **2026-07-14 (Phase 2 pilots, R002/R003 in `results/v1/RUNS.csv`):** the two pilot cities-mode
  runs (29-city PM2.5 panel, 29-city weather panel) were launched without `--folds 6`, so they
  ran on the harness default of **14 folds** instead of the pre-specified 6 (§3). Horizon (24h)
  and everything else matched the plan. These are pilot/pipeline-validation runs, not the
  canonical numbers that will appear in the manuscript — flagging here per this document's own
  rule rather than silently re-running. The canonical panel regeneration must pass `--folds 6`
  explicitly; do not reuse R002/R003's saved predictions as final results.

- **2026-07-15 (CRITICAL — added analysis, perfect-foresight covariates, §6/§9):** all
  covariate-using tiers (`lgbm_direct`, `chronos_cov`, `nas_gru`) consume the weather covariate
  at the forecast TARGET time origin+h — i.e. they assume a *perfect* weather forecast over the
  horizon (`future_covariates`, `COV.shift(-h)`). Plain `chronos` (univariate) cannot use
  covariates at all. This asymmetry is benign in the PM2.5 domain (weather is a weak 24h predictor
  of pollution) but SEVERE in the weather-forecasting domain, where the covariates
  (humidity/pressure/radiation/dewpoint) are near-deterministic physical drivers of the target
  (temperature) — so a perfect forecast of them nearly hands the model the answer. One-city probe
  (Nairobi): lgbm MASE weather 0.557 (perfect-foresight) vs 0.707 (causal, covariate at origin) =
  +27%; PM2.5 0.664 vs 0.663 = 0%. This means the headline "specialist wins the weather domain"
  is at risk of being a perfect-foresight artifact, not a forecasting-skill result.
  **Resolution:** added a `--causal-covariates` ablation to `run_forecast.py` (weather covariate
  enters at the origin = last-known value; calendar stays future-known) and a full-panel causal
  run for both domains (R020 PM2.5, R021 weather). The manuscript must (i) state the
  perfect-foresight assumption explicitly in Methods, (ii) report the causal-covariate MASE
  alongside the main numbers, and (iii) reframe the domain claim as conditional on covariate
  availability if the causal ablation collapses the specialist's weather-domain lead. This is a
  pre-specified *addition* prompted by review, logged here rather than silently folded in.
  **VERDICT (R020/R021, L-027):** it DID collapse. Weather panel — lgbm_direct with perfect-
  foresight covariates beats chronos on 26/29 cities (mean MASE 0.533 vs 0.792, Wilcoxon
  p=1.4e-6); with causal covariates only 16/29 (0.745 vs 0.792, p=0.29 = statistical TIE).
  Perfect foresight is worth +0.212 MASE in weather (p=2.6e-8) vs ~0 in PM2.5. **The paper's
  central claim is therefore NOT "chronos wins PM2.5, specialists win weather" (a domain flip).
  The corrected claim is: under realistic (last-known) covariate availability the small zero-shot
  TSFM is statistically competitive with the tuned specialist in BOTH domains; the specialist's
  apparent weather-forecasting edge is an artifact of assuming a perfect NWP forecast.** Abstract,
  intro, and results must be written to this corrected finding; the perfect-foresight numbers may
  still be shown as an upper bound on specialist performance, clearly labelled as such.

- **2026-07-15 (E4, §8 — two disclosures from review):** (a) *Bucket classification is
  test-based, not mean-based:* the interpretation-matrix outcome is assigned by
  `analysis/e4_significance.py` (city-level paired Wilcoxon, Holm-corrected across fractions,
  n=15 cities) — result is **bucket 4** (chronos leads nas_transfer on mean MASE at every
  fraction but never significantly), not bucket 3 as a means-only reading suggested.
  (b) *Data-budget floors are asymmetric at small fractions:* nas_transfer's fine-tune slice
  floors at 49h (lookback+horizon+1) while lgbm_refit's refit window floors at 202h
  (WEEK-lag warmup + horizon + margin) — at frac=1% on scarce cities both floors bind, so
  lgbm_refit actually receives up to ~4× more data than nas_transfer at the same nominal
  fraction. Direction is conservative (it advantages the baseline that still loses), and
  actual hours used are recorded per row (`n_train_hours`); Methods must report actual hours
  alongside nominal fractions. (c) *Perfect-foresight covariates do not threaten E4:* E4 is
  PM2.5-only, where the causal ablation shows perfect-foresight weather is worth ~0 MASE
  (panel: 0.662 perfect vs 0.692 causal, and lgbm≈chronos either way). nas_transfer uses those
  covariates and STILL fails to significantly beat univariate chronos — so its parity with the
  FM is not propped up by covariate access; if anything the small foresight advantage makes the
  no-significant-difference verdict conservative.

- **2026-07-15 (weather domain, §2/D3 — CRITICAL data-integrity fix):** the `weather_csv`
  loader reindexed each city's Open-Meteo series to the full PM2.5 usable window and then
  applied blanket `ffill().bfill()`. For 4 cities (Seoul +289h, Vienna +265h, Berlin +241h,
  Amsterdam +193h) the PM2.5 window extends past the weather pull's end (2026-06-30), so the
  loader FABRICATED a constant tail longer than the entire 144h test window — context-copying
  models scored absurdly (e.g. chronos MASE 0.028 on Vienna). Fix: clip the window to real
  weather coverage, interpolate only ≤6h gaps, refuse (raise) rather than fabricate; re-gate
  `min_hours` after clipping (all 4 cities still pass). The 4 cities' weather-panel rows,
  all weather-domain rigor artifacts, and the Seoul-weather regime run were regenerated under
  the fixed loader (R013/R014 supersede those slices of R005/R010). PM2.5 domain unaffected.
  Detected in pre-writing review by flagging a physically implausible MASE.

- **2026-07-14 (Phase 2 canonical campaign, cost-decision-rule regime runs, §7):** Beijing was
  initially not run in the weather domain. Reason: Beijing's UCI PRSA source uses year/month/day/hour
  columns and its own embedded TEMP field, not the `timestamp,temperature_2m` format the harness's
  `weather_csv` loader (`--pm25-window-dir`-clipped Open-Meteo pull) expects. **RESOLVED 2026-07-15:**
  added a `weather_pm25` source to `run_forecast.py` (`_read_pm25_station_weather`) that forecasts
  TEMP from the same PRSA station with the remaining meteorology (PRES/DEWP/RAIN/WSPM) as exog —
  weather-only exog, symmetric with the OpenAQ `weather_csv` path (no air-quality leakage). Beijing
  weather regime = R012. §7's "Beijing, plus one rich, one scarce, in both domains" is now fully
  6/6 covered. (The pre-specified protocol did not change; this closed an implementation gap, so
  it is logged as a resolution rather than a deviation from the plan.)

- **2026-07-16 (Stage E — reviewer-response additions, post-hoc, disclosed here):** a
  peer-style review prompted five additions, none of which change the pre-specified
  Phase-2 protocol; they are analyses layered on the existing canonical runs.
  (a) *Causal-covariate chronos_cov + causal-primary main results.* The perfect-foresight
  ablation (2026-07-15 entry) was extended from lgbm-only to chronos_cov (run_chronos_cov
  gains a `causal_cov` flag that freezes weather covariates at the forecast origin over the
  horizon; nas_gru was verified to be already causal — it consumes only the past context
  window). Table 1 and Fig 6 now report the deployable **causal** configuration as primary;
  the perfect-foresight numbers move to Table S14 / Fig SF as a labelled upper bound.
  (b) *Equivalence tests.* A non-significant test is not equivalence; added paired TOST +
  bootstrap CI at a pre-specified 0.05-MASE margin for the three tie claims
  (`analysis/equivalence_tests.py`). Verdicts: PM2.5 perfect-foresight lgbm is equivalent
  to chronos (TOST p=0.026); PM2.5 causal, weather-causal, and E4 are not — the prose was
  corrected to claim equivalence only where established.
  (c) *Energy amortization.* Split one-time training from per-forecast inference energy
  (`analysis/energy_amortization.py`); the ~10x measured gap is training-driven and a
  once-trained specialist crosses over to lower energy after ~2,800-4,400 forecasts (CPU).
  The headline energy claim is now stated as retrain-frequency-specific (Table S13).
  (d) *Contamination check.* Re-evaluated on strictly post-2024-10 OpenAQ windows (outside
  the Chronos-Bolt corpus); the PM2.5 competitiveness holds on unseen data
  (`analysis/contamination_check.py`, Table S15).
  (e) *Cost-penalty coefficient.* The decision-rule multiplier previously mislabelled
  "willingness-to-pay" is renamed lambda (cost-penalty coefficient); MASE + lambda*USD is
  unchanged, only the label and axis. A context-length-matched E4 (168h/672h GRU lookback)
  was attempted but exceeded the 8 GB study GPU and is reported as a hardware limitation.

- **2026-07-16 (supervisor pre-submission review — fixes, disclosed here):** an internal
  full-manuscript review (paper/SUPERVISOR_REVIEW_FINDINGS.md) found four blocking
  inconsistencies, all fixed without changing any pre-specified protocol:
  (f) *S12 configuration mismatch.* The cost-sensitivity grid (Table S12) had been computed
  on the perfect-foresight regime maps while Fig. 6/Table S12 present the causal maps.
  Recomputed on `causal_*_regime.csv` (ledger L-040, superseding L-020): flip rate 0% at the
  central 0.15 USD/kWh x PUE 1.4 assumption for all six runs, max 40% at the most extreme
  price x PUE corners (was 35% under the perfect config). Prose updated accordingly.
  (g) *Causal panel-level tests (pm25).* The panel sign/Wilcoxon tests had only been run on
  the perfect-foresight panel (L-012). Run on the causal-primary panel (L-039): sign test
  P = 0.024 with chronos better in 21/29 cities, Wilcoxon P = 0.084, Friedman P < 0.001.
  Note the sign test favours the foundation model; the manuscript reports this transparently
  and keeps the conservative "no significant difference" framing (Wilcoxon and FDR-DM
  non-significant, TOST equivalence not established).
  (h) *Interpretability subsection removed.* Methods promised SHAP "reported in the
  supplement" but SHAP was never in this locked plan and was never run in the canonical
  campaign (harness supports it only in `single` mode). Removed rather than backfilled.
  (i) *Abstract pipeline.* main.tex carried a hand-copied abstract that had drifted from the
  canonical paper/sections source (claiming causal-PM2.5 equivalence, 0.662 vs 0.662, and a
  "collapses to a tie" temperature claim). md2tex.py now emits generated/abstract.tex from
  00_title_abstract.md; the abstract was rewritten to 197 words with equivalence correctly
  scoped to the perfect-foresight configuration. Fig. 2a likewise switched from the
  perfect-foresight to the causal panel to match Table 1.
  (j) *Scarce-city count in this plan.* Section 8 above says E4 runs on "all 14" scarce
  cities that pass the panel gate; the gate in fact passed 15 scarce cities and E4 ran all
  15 (L-009 artifacts; 14 was a drafting miscount fixed per the no-silent-edit rule by this
  note, not by editing §8). Tier definition (a-priori infrastructure classification fixed in
  city_select.py before any data retrieval) is now stated in Methods.
