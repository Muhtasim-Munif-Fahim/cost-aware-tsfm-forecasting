# Methods

<!-- Stage C draft. Finalize against _runconfig.json records in Phase 5/7; nothing here
     may describe intended behavior that was not actually run. -->

## Study design

We compare three deployment strategies for hyper-local hourly forecasting in
data-scarce cities: *search* (a NAS-discovered tiny recurrent model), *specialize*
(a gradient-boosting model with covariates), and *zero-shot foundation model*. With
the covariate-residual foundation-model variant and a seasonal-naïve floor, these
three strategies are realized as the five model tiers reported throughout. Both
domains, urban PM2.5 (primary) and 2 m air temperature (secondary), use the same
29-city panel and the same evaluation harness.
The study extends our earlier conference paper \cite{fahim2026greennas}, from which
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
assume uniform calibration across cities. Retrieval was month-windowed, resumable,
and checkpointed, with per-request record-count verification (Supplementary
Information). **Temperature and meteorological covariates.** The Open-Meteo historical
archive \cite{zippenfenig2023openmeteo}, a gap-free reanalysis product built on ERA5
\cite{hersbach2020era5}: 2 m temperature as the target (weather domain) plus the seven
surface covariates used in our conference study: relative humidity, surface pressure,
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

PM2.5 readings ≤ 0 or ≥ 985 µg/m³ were treated as missing before any gating.
Non-positive concentrations are physically invalid for PM2.5 mass; 985 is a recurring
sensor/API sentinel value rather than a genuine extreme episode. Both rules were fixed
before the campaign, and panel membership is insensitive to the exact ceiling
(Supplementary Information). From each
sanitized series we extracted a usable window by splitting at gaps longer than 48 h,
interpolating only gaps ≤ 6 h inside the kept segment, and requiring a final window of
≥ 2,160 h (90 days) with ≥ 60% real (pre-interpolation) observations. All gate
parameters were fixed before the campaign, not tuned post hoc. Mumbai's sub-hourly
records were resampled to the hourly grid before gating. Three cities, London,
Kathmandu, and Kolkata, failed those gates and were excluded (Supplementary
Information). The final panel is 29 cities: 14 data-rich and 15
data-scarce (Supplementary Table S2).

Because each city contributes whichever window its source record supports, the panel
is not contemporaneous: usable windows span 2016 to 2026, and ten cities overlap the
2020–21 pandemic period. We did not exclude that period, and we report the exposure
and its consequences for the panel comparisons in the Supplementary Information.

The rich/scarce tier was assigned a priori, when
the candidate-city list was fixed and before any data were retrieved, by the maturity
of each city's regulatory air-quality monitoring infrastructure: data-rich cities are
served by long-established government monitoring programmes (North America, Europe,
developed Asia-Pacific, and the established networks of Santiago, Mexico City, and
Bangkok), while data-scarce cities are those whose OpenAQ coverage comes mainly from
recent, sparse, or low-cost/donor-funded deployments (South and Southeast Asia,
Africa, and the Andean cities). The label is deliberately an
infrastructure-context classification, not a function of realized record length
(several scarce-tier cities end up with long usable windows, e.g. Nairobi;
Supplementary Table S2), so tier membership was never revised after the data were
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
documented in the hyperparameter table, Supplementary Table S3.) **NAS-GRU.** The Green-NAS-A architecture
(two stacked GRU layers, 128 units) discovered by NSGA-II multi-objective search in
our conference study \cite{fahim2026greennas,deb2002nsga2}; trained with Adam
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

**Additional foundation models (generality check).** To separate properties of zero-shot
foundation models from properties of one architecture and one checkpoint, the whole panel
is additionally run with two further models that vary those two explanations one at a
time. `amazon/chronos-bolt-base` (205 M parameters) changes checkpoint scale by 4.3-fold
while holding family, developer and pretraining corpus fixed;
`google/timesfm-2.5-200m-pytorch` (231 M parameters, decoder-only) changes family,
developer and pretraining corpus at a scale matched to Chronos-Bolt-base. The comparison
therefore isolates a scale effect within a family from a family effect at matched scale.
Everything else is held constant: the same folds, four-week context window, metric, and
covariate-residual scheme for each model's covariate-consuming variant. TimesFM 2.5
provides a native covariate interface, which is not used, so that the covariate pathway is
identical across families and the comparison concerns the foundation model rather than two
covariate implementations. None of the checkpoints is trained or fine-tuned.
Chronos-Bolt-small remains the tier carried through Table 1, the energy measurements and
the cost-adjusted decision rule; the other two enter as accuracy generality checks.

**Covariate timing (perfect foresight vs causal).** We distinguish two covariate-timing
regimes and fix their roles explicitly. **Primary (deployable): causal covariates.**
Each meteorological covariate enters at its last value known at the forecast origin
(calendar features remain future-known, since a calendar is deterministic); this is the
configuration a real operator can run, and it is the one reported in Table 1, Fig. 2a,
Fig. 3, and Fig. 5. **Ceiling (upper bound): perfect-foresight covariates.** Every
covariate-using tier receives the meteorological covariate at the forecast *target* time
(the standard `future_covariates` benchmark convention), equivalent to assuming a perfect
weather forecast over the horizon; these numbers are reported only as a clearly-labelled
upper bound on covariate-using-tier performance (Supplementary Table S13, Supplementary
Fig. S1). The causal ablation was logged prospectively in the deviations record before it
was run, and the full 29-city panel was re-run in both domains for both
covariate-consuming tiers: the LightGBM specialist (the strongest covariate user and
the tier carrying the domain claim) and the covariate-residual Chronos variant. NAS-GRU
consumes only the past context window and is causal by construction. The univariate
zero-shot tier uses no covariates and is identical in both regimes. This primary/ceiling
hierarchy is stated identically in every table and figure caption that involves a
covariate-using tier.

**Realistic forecast covariates.** The two regimes above bracket deployment rather than
describe it: an operator would normally take future covariates from a numerical weather
prediction (NWP) service, which is neither perfect nor frozen at the origin. We therefore
added a third, calibrated regime. Using the Open-Meteo previous-model-runs archive we
measured the true error of archived forecasts for every covariate the specialist consumes,
expressed as a fraction of each variable's own standard deviation so that error levels
transfer across cities. Forecast error is then injected into the covariate series as a
stationary AR(1) process matched to the measured error magnitude, bias and lag-1
autocorrelation, with physical bounds enforced afterwards (non-negativity, percentage
ranges, and a circular wrap for wind direction). A scalar $\alpha$ scales the injected
error, giving a continuous covariate-quality axis on which $\alpha = 0$ reproduces the
perfect-foresight regime exactly and $\alpha = 1$ is the measured real-NWP error level.
Because the weather covariates enter the design matrix only as the future-covariate block,
degrading the covariate series is equivalent to replacing perfect foresight with a forecast
of the stated quality, and it does so identically for every covariate-consuming tier.

Two properties of the calibration archive constrain this construction and are detailed in
the Supplementary Information: its coverage is rich-tier-biased (15 of 29 cities), so the
error model is calibrated on covered cities and applied to all 29, and its lead times
(24--47 h) exceed our forecast horizon (1--24 h), so error is regressed on lead time rather
than taken at face value. A third property constrains interpretation directly: matching
the error *magnitude* of the last-known regime does not reproduce its effect. Equating
error variance predicts that regime at $\alpha = 1.54$, yet the panel does not reach it
even at $\alpha = 3$ (0.731 against 0.745), and linear extrapolation places the equivalent
near $\alpha \approx 3.4$. Holding a covariate flat for 24 h removes the diurnal cycle, a
coherent distortion, whereas AR(1) noise of the same variance blurs that cycle but
preserves it, so persistence costs appreciably more than its variance implies. We therefore
report the last-known result directly rather than reading it off the $\alpha$ axis,
treating the variance-matched equivalent as a lower bound.

## E4: transfer learning versus zero-shot (crux experiment)

NAS-GRU was pretrained on the pooled 14 rich-tier cities with per-city z-scoring (the
target scarce city is never in the pretraining corpus), then fine-tuned on each of the
15 scarce-tier cities at nominal budgets of {0, 1, 10, 100}% of that city's training
window, five seeds each. Comparators on identical test folds: zero-shot Chronos-Bolt,
and a LightGBM specialist refit on the same nominal budget, so the comparison is
"transfer versus every strategy at the same data budget," not "transfer versus
nothing." Two features of the smallest budgets both cut against the transfer model's
favour and are detailed in the Supplementary Information: the 0% condition uses
target-city data for normalization though not for gradient updates, and the two
window floors are asymmetric, so at the 1% budget the LightGBM baseline receives up
to ~4× more data. Actual hours used are recorded per row and reported alongside
nominal fractions (Supplementary Table S10). The outcome was classified against a pre-specified four-way
interpretation matrix by city-level paired Wilcoxon tests, Holm-corrected across
fractions, not by comparing means.

## Energy and cost measurement

Primary measurement wraps each tier's entire runner call with codecarbon
\cite{codecarbon}, using a fresh tracker per call in process tracking mode at a
one-second sampling interval. The metered region is the whole runner call: checkpoint
load, and for trained tiers the per-fold training, as well as inference.

The two hardware components are not on equal footing.
GPU energy is a hardware measurement, read from NVML counters on the study machine's
RTX 3060 Ti (the Chronos and NAS-GRU tiers). CPU energy is not: the study machine runs
Windows, where the RAPL interface that would provide a hardware CPU reading is
unavailable, so codecarbon derives CPU draw from a thermal-design-power model. It matches
the processor (an AMD Ryzen 5 3600, 6 cores / 12 threads) to a 65 W entry in its bundled
power table and scales that by the process's measured CPU utilisation. The CPU figures are
therefore modelled rather than metered.

We accordingly interpret the CPU-side numbers as comparative, directional quantities: they
support relative statements between tiers measured on this one machine under an identical
protocol, but do not license claims about absolute hardware energy draw or extrapolation
to other processors. The GPU-side numbers, which dominate the foundation-model and NAS
tiers and carry most of the headline energy contrast, are NVML-measured and stand on
firmer ground. Energy is normalized to joules per 1,000 forecasts and converted to USD at
0.15 USD/kWh with PUE 1.4 (defaults varied in a sensitivity analysis, Supplementary
Table S11). The supplementary TDP × latency proxy measures inference only, so it is never
averaged or ratioed against a trained tier's measured figure. Repeatability: a fixed
workload was rerun five times on three cities
(Beijing, Seoul, Nairobi); any city×tier cell with sd/mean > 20% is reported as a
range rather than a point estimate (4 of 15 cells; Supplementary Table S8)
\cite{garciamartin2019estimation,schwartz2020green}.

**Cost-adjusted decision rule.** For each training-history regime (4–104 weeks,
city-dependent) we select the tier minimizing MASE + λ · (USD per 1,000 forecasts),
where the cost-penalty coefficient λ is swept over {0, 500, 1500, 5000, 20000} MASE
units per
USD per 1,000 forecasts, producing winner maps per domain for Beijing (depth city),
Seoul (rich tier), and Nairobi (scarce tier). Electricity price and PUE act on this
objective as linear rescalings of λ; the sensitivity sweep quantifies map stability.

All timestamps are handled in Coordinated Universal Time (UTC) end to end, so calendar
features are computed on the UTC clock rather than each city's local time. The same
convention applies to every tier and both domains, so it cannot bias any of the
contrasts reported here; the full treatment is in the Supplementary Information.

## Statistical framework

**Hypothesis families.** The confirmatory comparisons are the two Table 1 causal-covariate
domain comparisons (specialist vs zero-shot foundation model on PM2.5 and on temperature)
and the four E4 fine-tune budgets (transfer vs zero-shot); city-level DM significance is
FDR-controlled within each domain, and E4 Wilcoxon tests are Holm-corrected across the four
budgets. All other reported P values are secondary or sensitivity analyses and are read
descriptively: the perfect-foresight ceilings (Table S13), the 48-hour horizon (Table S9),
the panel-level sign/Wilcoxon and Friedman–Nemenyi robustness checks, the equivalence
(TOST) analyses, and the data-volume correlations. We flag each as such where reported and
do not apply cross-family multiplicity correction across these distinct question families.

Pairwise per-city comparisons use the Diebold–Mariano test
\cite{diebold1995comparing} on absolute-error loss. The loss-differential series for each
city is the per-step difference in absolute error, pooled over all six folds and all
24 lead times (≈ 144 values per city, one 24-step trajectory per fold origin), so it
covers the same horizons that MASE averages.

We do not treat these values as independent. Errors belonging to the same
forecast trajectory, to neighbouring valid times, and to different lead times are all
serially correlated, so the variance of the mean loss differential is estimated with a
heteroskedasticity- and autocorrelation-consistent (HAC) Newey–West estimator, using a
Bartlett kernel over lags up to $h-1 = 23$, in place of the i.i.d. variance that the
original DM statistic assumes. The resulting statistic carries the
Harvey–Leybourne–Newbold small-sample correction \cite{harvey1997testing} and is referred
to a Student $t(n-1)$ distribution rather than a standard normal, the reference the HLN
correction is derived for. Fold origins are non-overlapping, so residual cross-fold
dependence reflects the series' own persistence, while within-trajectory and cross-horizon
dependence is absorbed by the HAC lag window. Significant-win counts are additionally
controlled with Benjamini–Hochberg FDR across the 29 cities.

Because that variance correction is nonetheless parametric, every per-city verdict was
re-derived under two weaker sets of assumptions. A moving-block bootstrap resamples
contiguous blocks of the loss-differential series, carrying within-block dependence
through the resample rather than modelling it, with block length swept over 12, 24 and 48
steps (half a forecast trajectory, one, and two) to test rather than assume that the block
spans the dependence range. A trajectory-level test then collapses each 24-step trajectory
to its mean, leaving one value per non-overlapping fold origin ($n = 6$) and discarding
within-trajectory and cross-lead dependence entirely. The bootstrap returns more
significant cities than the HAC test in both domains, so the parametric correction is
conservative rather than permissive on these data, while the deliberately underpowered
trajectory-level test returns fewer, as its $n = 6$ implies (exact counts in
Supplementary Table S17).
Verdicts also hold lead by lead: testing each of the 24 lead times separately across
cities, the sign of the mean loss differential is unchanged at 21 of 24 leads for
temperature and 14 of 24 for PM2.5, and no individual lead reaches significance for
temperature (3 of 24 for PM2.5), so neither domain's conclusion is an artifact of pooling
across horizons. Because a
non-significant test is not evidence of equivalence, the three "tie" comparisons (PM2.5
specialist vs foundation model, causal-covariate temperature specialist vs foundation
model, and E4 transfer vs zero-shot) are additionally assessed with a paired
two-one-sided test (TOST) against an equivalence margin of δ = 0.05 MASE, at
α = 0.05.

Two conventions apply throughout: δ and α share the value 0.05 despite measuring distinct
things, and TOST tests against a 90% interval rather than the conventional 95% one used
for the accompanying bootstrap CI (Supplementary Information).

Because MASE is scale-free, one numerical margin does not carry one physical meaning:
δ = 0.05 denotes a different quantity in each domain and tier, and we state those
quantities rather than leaving the margin abstract. Translating δ through each stratum's
mean seasonal-naive MAE gives, for PM2.5, 0.29 µg/m³ in the rich tier, 1.33 µg/m³ in the
scarce tier and 0.83 µg/m³ panel-wide; and for temperature 0.18 K, 0.10 K and 0.14 K
respectively. The E4 transfer comparisons are run on the 15 scarce-tier cities, so
1.33 µg/m³ is the figure that applies there.

Each of these is small relative to the operating thresholds of its own domain. The
narrowest PM2.5 decision band at the low-concentration end of the US AQI scale spans
roughly 9 µg/m³ and the WHO 2021 24-hour guideline value is 15 µg/m³, so even the
scarce-tier figure is well under a sixth of the narrowest band an operator acts on. For
temperature, 0.10–0.18 K is an order of magnitude below the whole-degree thresholds used
in heat-health warning systems. A single δ is
therefore defensible as a *strict* margin in both domains, but it is strict to differing
degrees, and the scarce-tier PM2.5 case is the least strict of the three.

The verdict does depend on the margin, so we report the full sensitivity across
δ = 0.02, 0.05 and 0.10 together with the minimum margin at which each comparison attains
equivalence (Supplementary Table S15), and state that dependence rather than presenting
equivalence as margin-free. The provenance of both the equivalence analysis and the margin
is documented in the Supplementary Information.

Panel-level comparisons use a binomial sign test on per-city wins and
a Wilcoxon signed-rank test on paired per-city MASE; a Friedman test with Nemenyi
post-hoc across all five tiers \cite{demsar2006statistical} is reported as a
supplementary robustness check. Uncertainty is quantified with split-conformal 95%
prediction intervals \cite{vovk2005algorithmic,shafer2008conformal,angelopoulos2023conformal}
calibrated on the first half of each series' backtest predictions and evaluated on
the second half, pooled per tier × domain (Supplementary Table S4; per-city in
Supplementary Tables S6 and S7); we note that
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
