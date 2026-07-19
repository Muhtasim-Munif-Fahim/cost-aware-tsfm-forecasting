# Results

<!-- Section drafted at Stage C against RESULTS_LEDGER rows; every quantitative claim
     carries an L-### comment that the Phase 7 number-audit subagent joins mechanically. -->

## Study design

We compared five forecasting strategies on hourly, city-level series in two domains,
urban PM2.5 (primary) and 2 m air temperature (secondary), over a 29-city panel
(14 data-rich, 15 data-scarce cities). The protocol fixes pre-specified quality
gates, six-fold rolling-origin backtesting, and a 24-hour direct multi-horizon task
(Fig. 1; 48-hour results in Supplementary Table S9). The strategies span the three
deployment philosophies available to a resource-constrained operator: *search*
(NAS-GRU, the two-layer, 128-unit gated-recurrent-unit (GRU) architecture discovered
by multi-objective neural architecture search in our earlier conference study
\cite{fahim2026greennas}), *specialize* (LightGBM with weather and calendar
covariates, one direct model per lead time \cite{ke2017lightgbm}), and *train
nothing* (Chronos-Bolt-small used zero-shot, with and without a covariate-residual
variant \cite{ansari2024chronos,autogluon2024chronosbolt}), against a seasonal-naïve
floor. Accuracy is reported as MASE on a fixed per-series scale, and end-to-end
energy was measured with codecarbon \cite{codecarbon} (GPU metered with NVML, CPU
estimated; Methods). Pairwise differences were tested with Diebold–Mariano tests
under the Harvey–Leybourne–Newbold correction
\cite{diebold1995comparing,harvey1997testing}, with panel-level sign and Wilcoxon
tests, and with Benjamini–Hochberg FDR control across cities.

\input{floats/F1_design}

## A zero-shot foundation model matches the tuned specialist on PM2.5

On the 29-city PM2.5 panel, zero-shot Chronos-Bolt and the tuned LightGBM specialist
were statistically indistinguishable. In the deployable causal-covariate configuration
reported in Table 1, panel-mean MASE is 0.662 ± 0.368 (Chronos-Bolt) versus
0.692 ± 0.374 (specialist) <!-- L-037 --> (Fig. 2a); with perfect-foresight covariates the
specialist matches Chronos-Bolt exactly (0.662, Supplementary Table S14) <!-- L-028 -->.
Neither model recorded a single FDR-corrected DM-significant win over the other
across all 29 cities, in either the causal panel (0/0) <!-- L-033 --> or the
perfect-foresight panel (0/0) <!-- L-019 --> (full perfect-foresight pairwise
matrices in Supplementary Table S4). In the perfect-foresight configuration the
panel-level tests were non-significant (sign test P = 0.136, Wilcoxon signed-rank
P = 0.325) <!-- L-012 -->. Under causal covariates the per-city balance in fact tilts
toward the foundation model, which is better in 21 of 29 cities (sign test
P = 0.024), but the magnitude-weighted test does not reach significance (Wilcoxon
P = 0.084) <!-- L-039 -->; we therefore describe the deployable comparison
conservatively, as no significant difference rather than a foundation-model win.
In the perfect-foresight case the agreement clears a stricter bar: a paired
two-one-sided-test at a 0.05-MASE margin declared the perfect-foresight specialist
*equivalent* to Chronos-Bolt (TOST P = 0.026; 90% CI on the per-city difference
[−0.042, 0.042], inside the margin) <!-- L-030 -->. All seven equivalence tests are
shown in Supplementary Fig. S2, and their sensitivity to the margin in Supplementary
Table S16. Under causal covariates the point difference is +0.030 MASE
and non-significant, but the confidence interval is a little too wide to certify
equivalence at that margin (90% CI upper bound 0.071) <!-- L-030 -->. We therefore
report the causal PM2.5 comparison as "no significant difference" and reserve the
equivalence claim for the perfect-foresight case (the margin is not a registered
pre-campaign quantity; Methods). Both models cleared the seasonal-naïve floor (naïve
MASE 1.026 ± 0.530; 13 FDR-significant Chronos-Bolt wins over naïve) <!-- L-028 --> <!-- L-019 -->.
The NAS-searched model and the covariate-residual Chronos variant trailed
(0.734 ± 0.352 and 0.797 ± 0.481) <!-- L-037 -->, and a Friedman test across all five
tiers rejected tier equivalence overall (P < 0.001 in both configurations)
<!-- L-012 --> <!-- L-039 -->. The tie is therefore specific to the
specialist-versus-foundation-model comparison; the leaderboard as a whole is not
flat.

\input{floats/F2_advantage}

\input{floats/T1_panel}

The foundation model's advantage showed no detectable correlation with how much
local history a city had: in the deployable causal configuration, the correlation
between usable hours and the specialist-minus-FM MASE gap was r = −0.031 (P = 0.873,
bootstrap 95% CI [−0.287, 0.251]) <!-- L-042 -->, and the perfect-foresight panel is
likewise flat (r = 0.075, P = 0.698) <!-- L-013 -->.
Zero-shot competitiveness is therefore not a scarce-data phenomenon that longer
records would erase. This is a cross-sectional check across heterogeneous cities and is
read descriptively; usable hours is a coarse proxy, since the zero-shot model reads a
fixed four-week context while trained tiers can exploit the full history.

## The specialist's temperature edge requires a perfect weather forecast

At face value the temperature domain looked like a specialist win. With
perfect-foresight covariates (reported as an upper bound in Supplementary Table S14),
LightGBM reached panel-mean MASE 0.533 ± 0.208 versus 0.792 ± 0.282 for zero-shot
Chronos-Bolt <!-- L-028 -->, with 6 FDR-significant specialist wins and none in the
other direction <!-- L-021 --> (sign and Wilcoxon P < 0.001) <!-- L-023 --> (Fig. 2b).
That comparison, however, grants every covariate-using model the meteorological
covariates at the forecast *target* time (the standard `future_covariates`
convention), equivalent to assuming a perfect weather forecast over the horizon. For
PM2.5 this assumption adds little information (weather is a weak 24-h predictor of
pollution), but for temperature the covariates (humidity, pressure, radiation, dew
point) are near-deterministic physical drivers of the target.

We therefore re-ran the specialist, the strongest covariate-using tier and the one
carrying the domain claim, with *causal* covariates: each weather covariate enters at
its last value known at the forecast origin, while calendar features remain
future-known. The restriction removed the temperature lead (Fig. 3). Panel-mean MASE
rose from 0.533 to 0.745, the per-city win rate against Chronos-Bolt fell from 26/29
(Wilcoxon P = 1.4 × 10⁻⁶) to 16/29 (P = 0.29, no longer significant), and the
foresight assumption alone was worth +0.212 MASE (P = 2.6 × 10⁻⁸) <!-- L-027 -->. In
the causal-covariate panel that Table 1 now reports as the main result, the
specialist records zero FDR-significant DM wins over the foundation model, and the
foundation model records one (0/1) <!-- L-034 -->, a reversal of the 6/0 count seen
under perfect foresight. The covariate-residual Chronos variant is hit far harder by
the same restriction in this domain: with the weather covariates frozen at the
origin, its ridge component extrapolates stale values across the horizon, and its
panel-mean MASE degrades to 2.614 ± 1.044, worse than the seasonal-naïve floor
(Table 1) <!-- L-037 -->. The same ablation left PM2.5 unmoved (0.662
perfect-foresight vs 0.692 causal, both statistically indistinguishable from
Chronos-Bolt at 0.662; P = 0.33 and P = 0.08) <!-- L-027 -->. Held to the stricter
equivalence standard, the two domains differ only in which direction the residual
uncertainty leans. On PM2.5, formal equivalence at the 0.05-MASE margin is
established in the perfect-foresight configuration (above); under causal covariates
it is not, and the interval leaves room only on the specialist's unfavourable side
(90% CI [−0.012, 0.071]). On temperature under causal covariates equivalence is
likewise not established, but the interval leans the specialist's way: the paired
difference favours it by a small, non-significant margin (0.745 vs 0.792; 90% CI
[−0.114, 0.018]) <!-- L-030 -->. The corrected cross-domain finding is therefore not
a domain flip but a graded one. Under realistic covariate availability, the small
zero-shot foundation model shows no statistically significant accuracy deficit to
the tuned specialist in either domain (formal equivalence is established on PM2.5
when the specialist is granted perfect-foresight covariates), and the specialist's
large apparent weather edge is an artifact of assuming a perfect forecast of its own
inputs. Perfect-foresight numbers are retained in all tables as a clearly-labelled
upper bound on specialist performance.
As in PM2.5, the temperature-domain FM advantage showed no detectable linear
association with data volume in the causal configuration (Pearson r = 0.138, P = 0.477,
95% CI [−0.220, 0.452]) <!-- L-043 -->. With only 29 cities this is an absence of
detectable association, not evidence of true independence.

\input{floats/F3_foresight}

## Beijing multi-station depth check

The panel design trades depth for breadth (one sensor per city), so we added a
12-station within-city check on the UCI Beijing multi-site dataset
\cite{zhang2017cautionary}. Zero-shot Chronos-Bolt beat the LightGBM specialist at
12 of 12 stations, and beat both the specialist and the seasonal-naïve floor at all
12 (Chronos-Bolt MASE range 0.153–0.297; LightGBM 0.292–0.458; naïve 1.03–1.24)
<!-- L-026 --> (Fig. 4). The panel-level tie is therefore not an averaging artifact
over heterogeneous stations.

\input{floats/F4_beijing}

## The evaluated transfer recipe does not outperform zero-shot in the data-scarce case

The crux question inherited from our conference study \cite{fahim2026greennas} is
whether a NAS-discovered model *pretrained on data-rich cities and fine-tuned on a
scarce city* still beats a foundation model that has seen none of the target city's
data. Across the 15 scarce-tier cities and fine-tune budgets of {0, 1, 10, 100}% of
local history (5 seeds each), zero-shot Chronos-Bolt held the best mean MASE at every
budget: 0.843 versus 0.899/0.915/0.888/0.876 for transferred NAS-GRU and
0.941/0.944/0.858 for a LightGBM refit on the same budget <!-- L-009 --> (Table 2).
The transfer recipe never overtakes zero-shot: after Holm correction across
fractions, the city-level paired Wilcoxon tests find no fraction at which transfer
significantly beats zero-shot <!-- L-018 -->. The paired difference in fact favours
the zero-shot model at every budget, nominally significantly at the 0% and 1%
budgets (bootstrap 95% CI on the difference excludes zero) and non-significantly at
10% and 100% <!-- L-030 -->. Equivalence is not established at any budget (all TOST
90% CIs exceed the 0.05-MASE margin; Supplementary Fig. S2) <!-- L-030 -->. We are
therefore not claiming that the two are interchangeable, only that the previously
published transfer recipe does not recover a data-scarce advantage over simply
running the foundation model. Strategy choice in the scarce regime therefore turns
on cost and operational simplicity, and there the zero-shot model wins (next
subsection).

\input{floats/T2_e4}

## Energy, cost, and the deployment decision

Under our backtesting-and-retraining protocol, the measured end-to-end energy reverses
the expectation that the foundation model is the expensive option (Table 3). Because
the specialist retrains one model per lead time per fold while the zero-shot model
only runs inference, LightGBM consumed 9.5–15.1 kJ per 1,000 forecasts across the
three replication cities, against 1.0–1.2 kJ for zero-shot Chronos-Bolt on a consumer
GPU (one repeatability-flagged Nairobi cell ranged up to 5.4 kJ). That is roughly an
order of magnitude in energy, and about 8× in cost (6.9 × 10⁻⁴ versus 8.2 × 10⁻⁵ USD
per 1,000 forecasts at 0.15 USD/kWh and a power-usage-effectiveness (PUE) factor of
1.4) <!-- L-025 -->. NAS-GRU, which must be trained locally, sits with the specialist
(7.3–16.8 kJ/1k) <!-- L-025 -->. A five-repetition repeatability protocol flagged 4 of 15
city×tier cells as exceeding a 20% sd/mean gate; those cells are reported as ranges
rather than point estimates <!-- L-025 -->.

This measured comparison charges the specialist for the training it repeats every
fold and the zero-shot model for inference only, so it reflects a frequent-retraining
deployment rather than a train-once one. That distinction matters. Separating
one-time training energy from per-forecast inference energy shows the measured gap is
almost entirely a training gap. At inference alone, the specialist's 24 small
per-horizon tree models are cheaper per forecast than a single Chronos-Bolt forward
pass, in all three replication cities on CPU and in the two cities with a valid GPU
estimate (the Beijing GPU inference measurement is a sub-kilojoule artifact;
Supplementary Table S13) <!-- L-032 -->. A specialist
that is trained once and then reused therefore crosses over to being the more
energy-efficient option after roughly 2,800–4,400 forecasts on CPU (≈ 120–180 days of
hourly 24-h-ahead forecasting for one series) and 250–2,100 on GPU <!-- L-032 -->. The
order-of-magnitude energy advantage of the zero-shot model is thus specific to
deployments that retrain frequently (more often than every few months); it is not a
universal property. The electricity
cost of either tier is below 10⁻³ USD per 1,000 forecasts, so in absolute monetary
terms the difference is small next to hardware, memory, and engineering costs. The
zero-shot model's durable advantages are avoiding the training step altogether and
its operational simplicity, not the electricity bill (Discussion).

\input{floats/T3_energy}

Folding cost into the objective produces deployment winner maps across
training-history regimes for one depth city (Beijing) plus one rich (Seoul) and one
scarce (Nairobi) panel city in both domains (Fig. 5). The objective is MASE +
λ · USD/1k, where the cost-penalty coefficient λ, the number of MASE units one is
willing to trade for one USD per 1,000 forecasts, is swept over
{0, 500, 1500, 5000, 20000}; λ = 0 selects on accuracy alone, and larger λ penalizes
cost more heavily. The main maps use the deployable
causal covariates; a perfect-foresight version is a labelled upper bound (Supplementary
Fig. S1). In PM2.5 the two configurations are nearly identical (weather is barely
informative there) and the foundation-model family wins most of the causal map: 21/25
cells in Beijing, 14/25 in Seoul, and 13/20 in Nairobi to the Chronos family (8 zero-shot
+ 5 covariate variant) <!-- L-031 -->. In temperature the configurations diverge sharply.
Under perfect foresight the specialist takes many cells (11/25 Beijing, 12/25 Seoul, 18/20
Nairobi) <!-- L-029 -->, but under causal covariates its share collapses: the Chronos
family and NAS-GRU take most cells (Beijing 15/25 Chronos vs 6/25 specialist; Seoul 25/25
NAS-GRU; Nairobi 11/20 Chronos plus 8/20 NAS-GRU) <!-- L-031 -->, consistent with the
ablation. Winner maps are robust to the cost assumptions at the central setting and
degrade gracefully away from it: sweeping electricity price over
0.05–0.30 USD/kWh and PUE over 1.0–2.0 (both act as linear rescalings of λ) flips no
cells at the central assumption and at most 40% of cells at the most extreme
price × PUE corners <!-- L-040 -->
(Supplementary Table S12).

\input{floats/F5_decision}

## Uncertainty calibration

Split-conformal 95% intervals were near nominal for all tiers in both domains
(Supplementary Table S3; per-city coverage and width in Supplementary Tables S5–S6).
In the causal-covariate panel, pooled empirical coverage ranged 0.910–0.970 on PM2.5
and 0.900–0.968 on temperature; the low end is the weather rich-tier zero-shot pool
at 0.90 against the 0.95 nominal <!-- L-035 --> <!-- L-036 -->. Under these realistic
covariates the specialist's temperature intervals (rich-tier mean width 9.6 °C-scale
units) were only marginally narrower than zero-shot Chronos-Bolt's (10.7) <!-- L-036 -->,
and most of the sharpness the perfect-foresight configuration gave it (7.7,
Supplementary Table S3) <!-- L-022 --> is lost. On PM2.5 the foundation model's
intervals were in fact tighter than the
specialist's (rich-tier 18.5 vs 21.0) <!-- L-035 -->. The naïve floor achieved coverage
only through much wider intervals (35.1 vs ~18 on rich-tier PM2.5) <!-- L-035 -->.
