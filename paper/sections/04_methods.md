# Methods

<!-- Stage C draft. Finalize against _runconfig.json records in Phase 5/7; nothing here
     may describe intended behavior that was not actually run. -->

## Study design

We compare three deployment strategies for hyper-local hourly forecasting in
data-scarce cities: *search* (a NAS-discovered tiny recurrent model), *specialize*
(a gradient-boosting model with covariates), and *zero-shot foundation model*,
each evaluated against a seasonal-naïve floor. Both domains, urban PM2.5 (primary)
and 2 m air temperature (secondary), use the same 29-city panel and the same
evaluation harness.
The study extends our IEEE QPAIN conference paper \cite{fahim2026greennas}, from which
it inherits the NAS-discovered architecture, the weather domain, and the edge/energy
framing; the foundation-model tier, the cost-adjusted decision rule, the air-quality
domain, the cross-domain generality test, and the transfer-versus-zero-shot crux
experiment are new.

## Data sources

**PM2.5.** Hourly PM2.5 from the OpenAQ v3 API \cite{openaq}, one sensor per city
(sensor IDs in `cities_manifest.csv`). For each candidate city we selected the single
OpenAQ sensor with the longest gap-gated usable window (the deterministic selection
rule; ties broken by higher pre-interpolation coverage), so "reference sensor" here
means the best-covered instrument available for that city rather than a certified
reference-grade monitor; OpenAQ aggregates heterogeneous instruments, and we do not
assume uniform calibration across cities. Because OpenAQ's deep pagination is unstable
for large requests, fetches were month-windowed, resumable, and checkpointed, with
per-request verification that the reported record count matched the records received. **Temperature and meteorological covariates.** The Open-Meteo historical
archive \cite{zippenfenig2023openmeteo}, a gap-free reanalysis product built on ERA5
\cite{hersbach2020era5}: 2 m temperature as the target (weather domain) plus the seven
surface covariates used in the conference study: relative humidity, surface pressure,
10 m wind speed and direction, precipitation, cloud cover, and shortwave radiation. In
the temperature domain the covariate set excludes the target and contains no variable
mathematically derived from it. Because these are reanalysis fields rather than a single
physical station, the temperature target is a location-specific gridded series; we
describe it as location-level rather than station-level, and note that its "data
scarcity" is imposed by clipping to each city's PM2.5 window (Section on panel
construction) rather than being intrinsic to the gap-free weather record. **Beijing
depth check.** The UCI Beijing Multi-Site Air Quality dataset (12 stations, hourly,
March 2013–February 2017, meteorology included) \cite{zhang2017cautionary}.

## Data quality gating and preprocessing

PM2.5 readings ≤ 0 or ≥ 985 µg/m³ were treated as missing before any gating. Non-positive
concentrations are physically invalid for PM2.5 mass and are a standard low-cost-sensor
artifact. The 985 rule targets a repeated sensor/API *sentinel* value, not high pollution
per se: the value 985 recurs as an exact, isolated spike (identical to the digit
in the raw feed, with no neighbouring elevated readings) across 12 of 29 raw city files,
the signature of a saturation/error code rather than a genuine extreme episode. This is
why the argument rests on the sentinel pattern rather than on 985 being implausibly high.
Both rules were fixed before the campaign; the panel-membership outcome is
insensitive to the exact 985 ceiling. From each
sanitized series we extracted a usable window by splitting at gaps longer than 48 h,
interpolating only gaps ≤ 6 h inside the kept segment, and requiring a final window of
≥ 2,160 h (90 days) with ≥ 60% real (pre-interpolation) observations. All gate
parameters were fixed before the campaign, not tuned post hoc. Mumbai's sub-hourly
records were resampled to the hourly grid before gating. Three cities were excluded
under pre-specified criteria: London (its sensor has only 247 real hourly readings
over the nominal 2016–2026 span; the sparsity is a property of the source record,
not of our retrieval) and
Kathmandu and Kolkata (both fail the minimum-hours gate due to internal gap structure,
confirmed by direct inspection). The final panel is 29 cities: 14 data-rich and 15
data-scarce (Supplementary Table S1). The rich/scarce tier was assigned a priori, when
the candidate-city list was fixed and before any data were retrieved, by the maturity
of each city's regulatory air-quality monitoring infrastructure: data-rich cities are
served by long-established government monitoring programmes (North America, Europe,
developed Asia-Pacific, and the established networks of Santiago, Mexico City, and
Bangkok), while data-scarce cities are those whose OpenAQ coverage comes mainly from
recent, sparse, or low-cost/donor-funded deployments (South and Southeast Asia,
Africa, and the Andean cities). The label is deliberately an
infrastructure-context classification, not a function of realized record length
(several scarce-tier cities end up with long usable windows, e.g. Nairobi;
Supplementary Table S1), so tier membership was never revised after the data were
seen.

For the temperature domain, each city's forecasting window was clipped to that same
city's PM2.5 usable window, so cross-domain comparisons isolate the domain effect
rather than confounding it with longer weather histories. The temperature loader does
not extrapolate beyond observed coverage: windows are clipped to real weather coverage
and only gaps ≤ 6 h are interpolated; a window that would require extrapolation is
excluded rather than filled. An earlier loader version forward-filled beyond coverage
for four cities and produced an implausible MASE; the error was identified during
internal review, and all affected artifacts were regenerated under the corrected
loader (deviations log).

## Forecasting task and evaluation protocol

Hourly cadence; primary horizon h = 24 with direct multi-horizon forecasting (one
specialist model per lead time); h = 48 as a supplementary sensitivity. Rolling-origin
backtesting \cite{bergmeir2012use} with six folds, all evaluated on the final common
test window per series, so that different training-history regimes are compared on
identical test data. Accuracy is reported as MASE \cite{hyndman2006another} with one
fixed scale per series, namely the in-sample error of the seasonal-naïve (m = 24)
forecast over the pre-test training history; the scale is shared across all regimes
for that series so MASE remains comparable when training history varies. MAE, RMSE and normalized
variants are recorded in every artifact.

## Model tiers

**Seasonal-naïve.** 168-h persistence (same hour last week), the explicit floor tier.
Note this weekly-persistence floor is a distinct model from the daily (m = 24) naïve
whose in-sample error sets the fixed MASE denominator; the two serve different roles
(a competitor tier versus a scale) and are not required to coincide.
**LightGBM specialist.** One direct model per lead time \cite{ke2017lightgbm} with
lagged target, calendar features, and meteorological covariates; retrained per fold;
`random_state = 42`. (A pre-campaign audit found `subsample = 0.8` set without
`subsample_freq`, making bagging inert; the configuration was corrected and is
documented in the hyperparameter table, Supplementary Table S2.) **NAS-GRU.** The Green-NAS-A architecture
(two stacked GRU layers, 128 units) discovered by NSGA-II multi-objective search in
the conference study \cite{fahim2026greennas,deb2002nsga2}; trained with Adam
(learning rate 10⁻³), early stopping (patience 10, ≤ 50 epochs), 24-h lookback.
Because training is stochastic, it was run at five seeds {42–46} and reported as
mean ± sd; no best-seed selection. **Chronos-Bolt (zero-shot).**
`amazon/chronos-bolt-small` \cite{ansari2024chronos,autogluon2024chronosbolt} with a
four-week context window and no training or fine-tuning of any kind; the point forecast
is the model's predicted mean. **Chronos-Bolt + covariates.** A residual scheme: a
ridge regression on calendar and weather covariates is fit over the training history,
the foundation model forecasts the residual (its temporal strength), and the covariate
model's prediction for the horizon is added back. This tier consumes future covariates
and is therefore subject to the same perfect-foresight/causal distinction as the
specialist (below); in the causal configuration the weather covariates over the horizon
are frozen at their last-known value at the forecast origin before the covariate
model is evaluated.

**Covariate timing (perfect foresight vs causal).** We distinguish two covariate-timing
regimes and fix their roles explicitly. **Primary (deployable): causal covariates.**
Each meteorological covariate enters at its last value known at the forecast origin
(calendar features remain future-known, since a calendar is deterministic); this is the
configuration a real operator can run, and it is the one reported in Table 1, Fig. 2a,
Fig. 3, and Fig. 5. **Ceiling (upper bound): perfect-foresight covariates.** Every
covariate-using tier receives the meteorological covariate at the forecast *target* time
(the standard `future_covariates` benchmark convention), equivalent to assuming a perfect
weather forecast over the horizon; these numbers are reported only as a clearly-labelled
upper bound on covariate-using-tier performance (Supplementary Table S14, Supplementary
Fig. S1). The causal ablation was logged prospectively in the deviations record before it
was run, and the full 29-city panel was re-run in both domains for both
covariate-consuming tiers: the LightGBM specialist (the strongest covariate user and
the tier carrying the domain claim) and the covariate-residual Chronos variant. NAS-GRU
consumes only the past context window and is causal by construction. The univariate
zero-shot tier uses no covariates and is identical in both regimes. This primary/ceiling
hierarchy is stated identically in every table and figure caption that involves a
covariate-using tier.

## E4: transfer learning versus zero-shot (crux experiment)

NAS-GRU was pretrained on the pooled 14 rich-tier cities with per-city z-scoring (the
target scarce city is never in the pretraining corpus), then fine-tuned on each of the
15 scarce-tier cities at nominal budgets of {0, 1, 10, 100}% of that city's training
window, five seeds each. One caveat applies to the strict interpretation of the 0%
budget: the target city's per-city z-scoring statistics are computed from its
pre-test training history, so the 0% condition uses no target-city *gradient updates*
but does use target-city data for input/output normalization. This advantages the
transferred model at the smallest budgets, which is conservative here because it still
does not overtake the zero-shot foundation model. Comparators on identical test folds: zero-shot Chronos-Bolt,
and a LightGBM specialist refit on the same nominal budget, so the comparison is
"transfer versus every strategy at the same data budget," not "transfer versus
nothing." Two floors are asymmetric at the smallest budgets: the fine-tune slice
floors at 49 h (lookback + horizon + 1) while the LightGBM refit window floors at
202 h (weekly-lag warm-up + horizon + margin), so at the 1% budget the baseline
receives up to ~4× more data than the transfer model. The direction is conservative
(it advantages the baseline, which still loses on mean MASE), and actual hours used
are recorded per row and reported alongside nominal fractions (Supplementary
Table S10). The outcome was classified against a pre-specified four-way
interpretation matrix by city-level paired Wilcoxon tests, Holm-corrected across
fractions, not by comparing means.

## Energy and cost measurement

Primary measurement wraps each tier's entire runner call with codecarbon
\cite{codecarbon}: GPU energy is measured with NVML on the study machine's RTX 3060 Ti
(the Chronos and NAS-GRU tiers), while CPU energy is codecarbon's constant-power
estimate, a limitation of power instrumentation on Windows that we report as such. Energy is normalized to joules per
1,000 forecasts and converted to USD at 0.15 USD/kWh with PUE 1.4 (defaults varied in
a sensitivity analysis, Supplementary Table S12). Note that for trained tiers the
measured figure includes per-fold training as well as inference, whereas a
supplementary TDP × latency proxy measures inference only; the two are therefore not
directly comparable for trained tiers and are never averaged or ratioed against each
other. Repeatability: a fixed workload was rerun five times on three cities
(Beijing, Seoul, Nairobi); any city×tier cell with sd/mean > 20% is reported as a
range rather than a point estimate (4 of 15 cells; Supplementary Table S7)
\cite{garciamartin2019estimation,schwartz2020green}.

**Cost-adjusted decision rule.** For each training-history regime (4–104 weeks,
city-dependent) we select the tier minimizing MASE + λ · (USD per 1,000 forecasts),
where the cost-penalty coefficient λ is swept over {0, 500, 1500, 5000, 20000} MASE
units per
USD per 1,000 forecasts, producing winner maps per domain for Beijing (depth city),
Seoul (rich tier), and Nairobi (scarce tier). Electricity price and PUE act on this
objective as linear rescalings of λ; the sensitivity sweep quantifies map stability.

## Time standardization

All timestamps are handled in Coordinated Universal Time (UTC) end to end: OpenAQ
values are taken from each record's UTC period start, the Open-Meteo archive is
requested with `timezone=UTC`, and every series is parsed as UTC and made
timezone-naive on a common UTC hourly grid before alignment. Calendar features
(hour-of-day, day-of-week) are therefore computed on the UTC clock rather than each
city's local time. Because the same UTC calendar is used identically for every tier and
both domains, and the covariate-timing contrast (Fig. 3) compares a tier against itself
under two covariate regimes, a fixed UTC offset cannot bias the covariate-timing message;
it only means the learned "diurnal" phase is expressed in UTC. No daylight-saving
adjustment is applied (UTC has none).

## Statistical framework

**Hypothesis families.** The confirmatory comparisons are the two Table 1 causal-covariate
domain comparisons (specialist vs zero-shot foundation model on PM2.5 and on temperature)
and the four E4 fine-tune budgets (transfer vs zero-shot); city-level DM significance is
FDR-controlled within each domain, and E4 Wilcoxon tests are Holm-corrected across the four
budgets. All other reported P values are secondary or sensitivity analyses and are read
descriptively: the perfect-foresight ceilings (Table S14), the 48-hour horizon (Table S9),
the panel-level sign/Wilcoxon and Friedman–Nemenyi robustness checks, the equivalence
(TOST) analyses, and the data-volume correlations. We flag each as such where reported and
do not apply cross-family multiplicity correction across these distinct question families.

Pairwise per-city comparisons use the Diebold–Mariano test
\cite{diebold1995comparing} on absolute-error loss with the
Harvey–Leybourne–Newbold small-sample correction \cite{harvey1997testing}. The loss
series for each city is the per-step absolute error pooled over all six folds and all
24 lead times (≈ 144 error values per city, one 24-step trajectory per fold origin),
so the loss series covers the same horizons that MASE averages; h = 24 enters only as
the forecast-horizon order in the HLN correction, not as a lead-24-only filter. Significant-win counts are
additionally controlled with Benjamini–Hochberg FDR across the 29 cities. Because a
non-significant test is not evidence of equivalence, the three "tie" comparisons (PM2.5
specialist vs foundation model, causal-covariate temperature specialist vs foundation
model, and E4 transfer vs zero-shot) are additionally assessed with a paired
two-one-sided-test (TOST) against a 0.05-MASE equivalence margin and a
10,000-resample bootstrap 95% CI on the paired difference. To be transparent about
provenance: the equivalence *analysis* was added after peer review (logged in the
deviations record), and the 0.05-MASE margin was fixed before the TOST was run and
before its outcome was seen; it is not a pre-campaign registered quantity, and we do
not describe it as one. The margin is a deliberately strict fraction of the panel MASE
spread; on PM2.5 a 0.05-MASE difference corresponds to roughly 0.8 µg/m³ of MAE at the
rich-tier scale, a difference an operator would consider negligible. Because the verdict
does depend on the margin, we report the full margin sensitivity (Supplementary
Table S16): at a strict 0.02 margin no comparison is equivalent; at 0.05 only the
perfect-foresight PM2.5 comparison is; at a lenient 0.10 margin the causal PM2.5
comparison and the 100%-budget E4 comparison also clear it, while causal temperature
and the smaller E4 budgets never do. Main-text equivalence claims use the 0.05 margin
throughout and we state its dependence explicitly rather than presenting equivalence as
margin-free. Panel-level comparisons use a binomial sign test on per-city wins and
a Wilcoxon signed-rank test on paired per-city MASE; a Friedman test with Nemenyi
post-hoc across all five tiers \cite{demsar2006statistical} is reported as a
supplementary robustness check. Uncertainty is quantified with split-conformal 95%
prediction intervals \cite{vovk2005algorithmic,shafer2008conformal,angelopoulos2023conformal}
calibrated on the first half of each series' backtest predictions and evaluated on
the second half, pooled per tier × domain (Supplementary Table S3; per-city in
Supplementary Tables S5 and S6); we note that
conformal exchangeability is only approximate under temporal dependence
\cite{xu2021conformal,stankeviciute2021conformal}. The relationship between data
volume and the foundation-model advantage is a Pearson correlation between usable
hours and the per-city specialist-minus-FM MASE gap, with a 10,000-resample bootstrap
95% CI. Multi-seed tiers aggregate to a per-city seed-mean before any test.

## Reproducibility

All canonical runs were executed under a tagged code state with per-run configuration
records (`_runconfig.json`), a run registry, and SHA-256 manifests of all input data.
Every quantitative claim in this manuscript is backed by a row in a results ledger
mapping the claim to the artifact and exact command that produced it, and was
re-verified mechanically before submission. Seeds: LightGBM 42; NAS-GRU {42–46}.
The OpenAQ API key is supplied through an environment variable and never stored in code,
logs, or history. Code, the analysis plan, deviations log, and fetch scripts are
released with the paper.
