# Results

<!-- Section drafted at Stage C against RESULTS_LEDGER rows; every quantitative claim
     carries an L-### comment that the Phase 7 number-audit subagent joins mechanically. -->

## Study design

Five model tiers in two domains, urban PM2.5 (primary) and 2 m air temperature
(secondary), on one 29-city panel under one pre-specified harness (Fig. 1;
Methods). The tiers span the three options open to a resource-constrained operator:
*search* (NAS-GRU, the architecture carried over from our earlier conference study
\cite{fahim2026greennas}), *specialize* (LightGBM with weather and calendar
covariates \cite{ke2017lightgbm}), and *train nothing* (Chronos-Bolt-small run
zero-shot, with and without a covariate-residual variant
\cite{ansari2024chronos,autogluon2024chronosbolt}), plus a seasonal-naïve floor.
Accuracy is MASE on a fixed per-series scale; energy is measured end to end
\cite{codecarbon}. Results below are at the 24-hour horizon; the 48-hour horizon
behaves the same way (Supplementary Table S9). Abbreviations used throughout are
collected in Supplementary Table S1.

\input{floats/F1_design}

## A zero-shot foundation model matches the tuned specialist on PM2.5

On the 29-city PM2.5 panel, zero-shot Chronos-Bolt and the tuned LightGBM specialist
were statistically indistinguishable. In the deployable causal-covariate configuration
reported in Table 1, panel-mean MASE is 0.662 ± 0.368 (Chronos-Bolt) versus
0.692 ± 0.374 (specialist) <!-- L-037 --> (Fig. 2a); with perfect-foresight covariates the
specialist matches Chronos-Bolt exactly (0.662, Supplementary Table S13) <!-- L-028 -->.
Neither model recorded a single FDR-corrected DM-significant win over the other
across all 29 cities, in either the causal panel (0/0) <!-- L-033 --> or the
perfect-foresight panel (0/0) <!-- L-019 --> (full perfect-foresight pairwise
matrices in Supplementary Table S5). In the perfect-foresight configuration the
panel-level tests were non-significant (sign test P = 0.136, Wilcoxon signed-rank
P = 0.325) <!-- L-012 -->. Under causal covariates the per-city balance in fact tilts
toward the foundation model, which is better in 21 of 29 cities (sign test
P = 0.024), but the magnitude-weighted test does not reach significance (Wilcoxon
P = 0.084) <!-- L-039 -->. We therefore describe the deployable comparison
conservatively, as no significant difference rather than a foundation-model win.

A non-significant test is not evidence of equivalence, so we tested equivalence
directly. Under perfect foresight the agreement clears that stricter bar: a paired
two-one-sided test against a margin of δ = 0.05 MASE declares the perfect-foresight
specialist *equivalent* to Chronos-Bolt (TOST P = 0.026; the 90% interval that TOST
requires at α = 0.05, [−0.042, 0.042], lies inside the margin) <!-- L-030 -->. Under causal
covariates the point difference is +0.030 MASE and non-significant, but the interval
is a little too wide to certify equivalence at the same margin (upper bound 0.071)
<!-- L-030 -->, so we report that comparison as "no significant difference" only. All
seven equivalence tests appear in Supplementary Fig. S2 and their sensitivity to δ in
Supplementary Table S15.

This behaviour is not specific to one architecture or checkpoint, and we separated those
two possibilities rather than conflating them. The panel was repeated with two further
zero-shot models on identical folds: Chronos-Bolt-base (205 M parameters), which changes
checkpoint scale by 4.3x while holding the family fixed, and TimesFM 2.5 (231 M
parameters, decoder-only, a different developer and pretraining corpus), which changes
family at a scale matched to Chronos-Bolt-base. The harness, folds, context window and
covariate pathway are identical throughout, so any difference is attributable to the model
and not to the evaluation.

Neither axis moves the result. All three models fall within 0.003 MASE of one another
(0.659 TimesFM, 0.660 Chronos-Bolt-base, 0.662 Chronos-Bolt-small, against 0.692 for the
specialist), and no pair of them records a single FDR-significant DM win in any of the 29
cities; the two 200 M-class models from different developers do not differ significantly in
even one city. Paired TOST certifies *equivalence* at δ = 0.05 along both axes separately:
scale within a family (P = 1.0 × 10⁻⁵) and family at matched scale (P = 2.5 × 10⁻⁵), as
well as between the original Chronos-Bolt-small and TimesFM (P = 3.6 × 10⁻⁵). Per-city MASE
correlates at r = 0.99. Each model stands in the same relation to the specialist (mean
differences +0.030 to +0.033 MASE, no FDR-significant win in either direction), each clears
the seasonal-naïve floor decisively (12-15 FDR-significant wins), and the covariate-residual
variant degrades for all three alike (0.780-0.797).

We therefore report the PM2.5 conclusion for compact zero-shot foundation models rather
than for Chronos-Bolt alone. The claim rests on three checkpoints spanning two families and
a 4.3-fold range of model size, which is a stronger basis than one architecture but is
still not the whole model class: every model evaluated here is compact by
foundation-model standards, and none exceeds 231 M parameters.

On temperature the three models do not collapse onto one another as they do on PM2.5, and
the same two-axis design localises why. Panel-mean MASE rises from 0.793 (Chronos-Bolt-small)
to 0.806 (Chronos-Bolt-base) to 0.829 (TimesFM). Changing checkpoint scale 4.3-fold within
the Chronos family is certified *equivalent* (paired difference −0.013, TOST P = 7.6 × 10⁻⁴),
whereas changing family at matched scale is not (−0.024, P = 0.105). The divergence in this
domain is therefore a property of the architecture family rather than of model size.

That distinction carries into the comparison the paper leads with. Against the tuned
specialist, the paired panel difference is −0.048 for Chronos-Bolt-small and −0.061 for
Chronos-Bolt-base, both with 95% intervals containing zero, but −0.085 for TimesFM, whose
interval excludes it ([−0.148, −0.022]). The per-city Diebold–Mariano tests detect none of
this: no FM pair records more than one FDR-significant city, and the specialist records no
FDR-significant win over TimesFM in any city. Since the per-city test is the less powerful
of the two, we read the panel-level result as the more informative and report both. On
temperature, then, our claim of parity between specialist and zero-shot model holds for the
Chronos family across a 4.3-fold size range but not for TimesFM, and we state the claim at
that scope rather than for zero-shot models generally. All three clear the seasonal-naïve
floor, and the covariate-residual variants degrade alike (2.61 ± 0.01 across the three),
confirming that collapse is a property of the residual scheme under causal covariates
rather than of any one model.
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

## The specialist's temperature edge depends heavily on covariate timing

At face value the temperature domain looked like a specialist win. With
perfect-foresight covariates (reported as an upper bound in Supplementary Table S13),
LightGBM reached panel-mean MASE 0.533 ± 0.208 versus 0.792 ± 0.282 for zero-shot
Chronos-Bolt <!-- L-028 -->, with 6 FDR-significant specialist wins and none in the
other direction <!-- L-021 --> (sign and Wilcoxon P < 0.001) <!-- L-023 --> (Fig. 2b).
That comparison, however, is a perfect-foresight one: it hands every covariate-using
tier the weather at the forecast *target* time (Methods). For PM2.5 the assumption
adds little, because weather is a weak 24-h predictor of pollution. For temperature
the covariates (humidity, pressure, radiation, dew point) are near-deterministic
physical drivers of the target, so the assumption does a great deal of work.

We therefore re-ran the specialist, the strongest covariate user and the tier
carrying the domain claim, under causal covariates. The restriction removed the
temperature lead (Fig. 3). Panel-mean MASE
rose from 0.533 to 0.745, the per-city win rate against Chronos-Bolt fell from 26/29
(Wilcoxon P = 1.4 × 10⁻⁶) to 16/29 (P = 0.29, no longer significant), and the
foresight assumption alone was worth +0.212 MASE (P = 2.6 × 10⁻⁸) <!-- L-027 -->. In
the causal-covariate panel that Table 1 now reports as the main result, the
specialist records zero FDR-significant DM wins over the foundation model, and the
foundation model records one (0/1) <!-- L-034 -->, a reversal of the 6/0 count seen
under perfect foresight. The same restriction hits the covariate-residual Chronos
variant far harder in this domain, because its ridge component extrapolates stale
covariates across the horizon. Its panel-mean MASE degrades to 2.614 ± 1.044, worse
than the seasonal-naïve floor (Table 1) <!-- L-037 -->. The same ablation left PM2.5 unmoved (0.662
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
when the specialist is granted perfect-foresight covariates), and much of the
specialist's large apparent weather edge is attributable to assuming a perfect forecast of
its own inputs. How much is quantified in the next section, which replaces that assumption
with a measured one rather than removing it. Perfect-foresight numbers are retained in all tables as a clearly-labelled
upper bound on specialist performance.
As in PM2.5, the temperature-domain FM advantage showed no detectable linear
association with data volume in the causal configuration (Pearson r = 0.138, P = 0.477,
95% CI [−0.220, 0.452]) <!-- L-043 -->. With only 29 cities this is an absence of
detectable association, not evidence of true independence.

\input{floats/F3_foresight}

## Realistic forecast covariates place the temperature edge between the two extremes

The causal and perfect-foresight regimes bracket deployment rather than describe it. To
locate the deployable point inside that bracket we measured the true error of archived
24-hour weather forecasts for every covariate the specialist consumes and re-ran the panel
with covariates degraded to that measured quality, sweeping a scale factor α over the
covariate-quality axis (Methods; Fig. 6).

On temperature the specialist's advantage over the zero-shot foundation model falls
monotonically as covariate quality degrades, and the realistic point sits well inside the
bracket rather than at either end:

with perfect foresight (α = 0) the specialist reaches 0.533 MASE, a lead of +0.260 over
the zero-shot model; with covariates degraded to the measured real-forecast error level
(α = 1) it reaches 0.638, a lead of +0.154, or 59% of the perfect-foresight lead; and with
last-known covariates it reaches 0.745, a lead of +0.048, or 18%.

A specialist supplied with covariates of genuine forecast quality therefore retains about
three-fifths of the advantage that perfect foresight confers, against under one-fifth for
the last-known floor. The result is not an artifact of a single noise draw: across
replicate injections the panel mean varies by 0.009–0.018 MASE, roughly a sixth of the
0.106 MASE shift from perfect foresight to realistic covariates. As a check that the degradation reached only the intended
path, the covariate-free tiers are bitwise identical at every α.

Two qualifications belong with this result. First, matching the *magnitude* of last-known
error does not reproduce its *effect*. Equating error variance predicts the last-known
regime at α = 1.54, where the panel gives 0.677 against the regime's own 0.745; the panel
does not reach that level even at α = 3 (0.731), and extrapolation places the equivalent
near α ≈ 3.4. Holding a covariate flat for 24 hours removes the diurnal cycle, a coherent
distortion, whereas noise of the same variance blurs that cycle but preserves it, so
persistence costs appreciably more than its variance implies. We therefore report the
last-known figure directly rather than reading it off the α axis.
covariates carry information about the target: on PM2.5 the perfect-foresight and
last-known specialists differ by ~0.03 MASE to begin with, so there is almost no bracket to
locate a point inside, which is the same domain asymmetry reported above.

The practical reading is that the future-covariates convention distorts a cross-domain
comparison substantially but does not invent the specialist's temperature advantage. A
deployment with a trustworthy weather feed should expect to keep most of that advantage;
one without should expect to lose it.

\input{floats/F6_covariate}

## Beijing multi-station depth check

One sensor per city trades depth for breadth, so we added a 12-station within-city
check \cite{zhang2017cautionary}. Zero-shot Chronos-Bolt beat the LightGBM specialist at
12 of 12 stations, and beat both the specialist and the seasonal-naïve floor at all
12 (Chronos-Bolt MASE range 0.153–0.297; LightGBM 0.292–0.458; naïve 1.03–1.24)
<!-- L-026 --> (Fig. 4). The panel-level tie is therefore not an averaging artifact
over heterogeneous stations.

\input{floats/F4_beijing}

## The evaluated transfer recipe does not outperform zero-shot in the data-scarce case

The crux question inherited from our earlier conference study \cite{fahim2026greennas}
is whether a NAS-discovered model *pretrained on data-rich cities and fine-tuned on a
scarce city* still beats a foundation model that has never seen the target city. It
does not. Across the 15 scarce-tier cities at fine-tune budgets of {0, 1, 10, 100}% of
local history (Methods), zero-shot Chronos-Bolt held the best mean MASE at every
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
(7.3–16.8 kJ/1k) <!-- L-025 -->. Cells that failed the repeatability gate are shown as
ranges rather than point estimates throughout (4 of 15; Methods) <!-- L-025 -->.

The direction survives a substantially larger foundation model. Measured on the Beijing
depth city under the same protocol, TimesFM consumed 1.15 kJ per 1,000 forecasts against
0.53 kJ for Chronos-Bolt, a ratio close to their 4.8× difference in parameter count, but
still 2.8× less than the 3.22 kJ the retrained specialist required on the same series.
Model size therefore moves the foundation-model tier's energy cost without moving the
comparison it is part of, because the specialist's cost is dominated by retraining rather
than by inference. We report this as a single-city check rather than a panel result, since
only Chronos-Bolt was measured across the three replication cities.

That comparison reflects a frequent-retraining deployment, not a train-once one, and
the distinction matters. Separating one-time training energy from per-forecast
inference energy shows the measured gap is almost entirely a training gap. At
inference alone, the specialist's 24 small
per-horizon tree models are cheaper per forecast than a single Chronos-Bolt forward
pass, in all three replication cities on CPU and in the two cities with a valid GPU
estimate (the Beijing GPU inference measurement is a sub-kilojoule artifact;
Supplementary Table S12) <!-- L-032 -->. A specialist
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
scarce (Nairobi) panel city in both domains (Fig. 5). Each cell selects the tier
minimizing MASE + λ · USD/1k over the sweep of the cost-penalty coefficient λ
(Methods): λ = 0 selects on accuracy alone, and larger λ penalizes cost more heavily.
The main maps use the deployable causal covariates; a perfect-foresight version is a
labelled upper bound (Supplementary Fig. S1). In PM2.5 the two configurations are nearly identical (weather is barely
informative there) and the foundation-model family wins most of the causal map: 21/25
cells in Beijing, 14/25 in Seoul, and 13/20 in Nairobi to the Chronos family (8 zero-shot
+ 5 covariate variant) <!-- L-031 -->. In temperature the configurations diverge sharply.
Under perfect foresight the specialist takes many cells (11/25 Beijing, 12/25 Seoul, 18/20
Nairobi) <!-- L-029 -->, but under causal covariates its share collapses: the Chronos
family and NAS-GRU take most cells (Beijing 15/25 Chronos vs 6/25 specialist; Seoul 25/25
NAS-GRU; Nairobi 11/20 Chronos plus 8/20 NAS-GRU) <!-- L-031 -->, consistent with the
ablation. The maps are robust to the cost assumptions at the central setting and
degrade gracefully away from it: across the full electricity-price × PUE sweep, no
cells flip at the central assumption and at most 40% flip at the most extreme corners
<!-- L-040 --> (Supplementary Table S11).

\input{floats/F5_decision}

## Uncertainty calibration

Split-conformal 95% intervals were near nominal for all tiers in both domains
(Supplementary Table S4; per-city coverage and width in Supplementary Tables S6–S7).
In the causal-covariate panel, pooled empirical coverage ranged 0.910–0.970 on PM2.5
and 0.900–0.968 on temperature; the low end is the weather rich-tier zero-shot pool
at 0.90 against the 0.95 nominal <!-- L-035 --> <!-- L-036 -->. Under these realistic
covariates the specialist's temperature intervals (rich-tier mean width 9.6 °C-scale
units) were only marginally narrower than zero-shot Chronos-Bolt's (10.7) <!-- L-036 -->,
and most of the sharpness the perfect-foresight configuration gave it (7.7,
Supplementary Table S4) <!-- L-022 --> is lost. On PM2.5 the foundation model's
intervals were in fact tighter than the
specialist's (rich-tier 18.5 vs 21.0) <!-- L-035 -->. The naïve floor achieved coverage
only through much wider intervals (35.1 vs ~18 on rich-tier PM2.5) <!-- L-035 -->.
