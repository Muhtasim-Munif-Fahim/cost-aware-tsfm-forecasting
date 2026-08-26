# Discussion

## The efficiency calculus has changed

The premise of our earlier conference study was that foundation models are too
computationally heavy for edge deployment, so data-scarce cities need searched
architectures and transfer learning \cite{fahim2026greennas}; the assumption was
widely shared at the time and, there as elsewhere, was carried as a design constraint
rather than measured. This study keeps the same evaluation philosophy and supplies the
missing comparison. With a compact foundation model in the panel, the constraint no longer
binds. Under realistic
covariate availability, zero-shot Chronos-Bolt is statistically indistinguishable
from the tuned specialist on PM2.5 (and equivalent by a formal TOST when the
specialist is given perfect covariates), no longer significantly worse on
temperature, and never significantly beaten by the transfer recipe at any fine-tune
budget, where it leads at every one. Under a frequent-retraining protocol it also
uses roughly an order of magnitude less energy, because it never trains.

Why should a model that has never seen a city forecast it about as well as a model
trained there? A plausible reading is that the quantities a local specialist must
estimate from a short, gappy record (the shape of the diurnal cycle, weekly
structure, the persistence of pollution episodes) are the same generic regularities
a foundation model has already absorbed from large, heterogeneous pretraining
corpora \cite{ansari2024chronos}, so a four-week context window suffices to localize
them. This reading is consistent with our observation that the zero-shot advantage
shows no detectable association with local record length. It also sharpens, rather
than contradicts, recent skepticism about foundation-model leaderboards
\cite{karaouli2025foundational,aksu2024gifteval}. In our panel the zero-shot model
does not beat a well-tuned gradient-boosting specialist, which extends the long
record of simple methods matching complex ones \cite{makridakis2018statistical};
but matching the specialist *without training* is precisely what changes the
deployment decision. The earlier work's comparisons remain internally sound, and
search remains the right tool when an application needs a bespoke model; what changed
is that small TSFMs \cite{ansari2024chronos,autogluon2024chronosbolt} opened a second
route to the same destination and moved the accuracy–cost frontier with it. "Make it tiny by search" and "make it cheap by not training" now land in
the same accuracy band, and the second is operationally simpler. For an operator in a data-scarce city, the practical default shifts from
training a small model to downloading one.

## A covariate-timing pitfall that can produce spurious domain conclusions

A cross-domain claim can be an artifact of covariate timing alone. Had we used the
standard `future_covariates` convention uncritically, this paper would have reported
a clean domain flip (foundation model wins PM2.5, specialist wins temperature), with
26/29 significant city wins backing the second half. The causal ablation shows that
result depends substantially on granting the specialist a perfect forecast of its own
near-deterministic drivers: with last-known covariates the specialist's lead is no
longer statistically significant (16/29 cities, P = 0.29), shrinking from +0.259 to
+0.047 MASE, though a paired equivalence test does not certify the two as equivalent
in this domain. The asymmetry is domain-dependent (worth +0.212 MASE in temperature
and ~0 in PM2.5) because covariate informativeness about the target differs by
domain, and that is exactly what makes the artifact hard to detect in cross-domain
comparisons: the same convention, applied uniformly, distorts one domain and not the
other. Forecast-evaluation guides catalogue leakage and backtest pitfalls
\cite{hewamalage2023forecast,bergmeir2012use}; we add covariate foresight to that
list and recommend that benchmarks using future covariates report a causal-covariate
ablation alongside, or at minimum bracket reality between the two configurations. An
operator with access to NWP-grade covariate forecasts sits between our causal floor and
perfect-foresight ceiling \cite{rasp2018neural}, and we locate that operating point rather
than leaving it bracketed. Calibrating the covariates against measured 24-hour forecast
error places the specialist at 0.638 MASE, a lead of +0.154 over the zero-shot model:
about three-fifths of the perfect-foresight advantage, against under a fifth for the
last-known floor. Two consequences follow. The convention's distortion is real and large
enough to reverse a cross-domain reading, so a causal ablation remains necessary; but the
distortion does not dissolve the specialist's temperature advantage, and a deployment with
a trustworthy weather feed should expect to keep most of it. Reporting only the two extremes
would have overstated the correction in one direction exactly as the uncritical convention
overstates it in the other.

## Cost-aware deployment guidance

The winner maps turn the statistics into a decision, and for an operator that
decision reduces to three questions our measurements can answer.

**How much local history is there?** It matters less than one might expect.
At the shortest regime we map, four weeks of training history, the zero-shot model
already takes most cells, and the size of its advantage shows no detectable
association with record length across the panel (r = −0.031, P = 0.873 on PM2.5
under causal covariates). Over the full causal maps the Chronos family takes 21 of
25 cells in Beijing, 14 of 25 in Seoul and 13 of 20 in Nairobi. An operator choosing
on accuracy alone, at realistic covariate availability, has little reason to train.

**Is there a trustworthy forecast of the covariates?** This is the condition under
which a specialist clearly earns its place, and it is the one an operator is most
likely to get wrong. Granted perfect-foresight weather, the specialist takes many
temperature cells (11/25 Beijing, 12/25 Seoul, 18/20 Nairobi); restricted to what is
knowable at the forecast origin, that share collapses. Most operators sit between
those two cases, and the width of the bracket, worth +0.212 MASE in temperature and
about zero in PM2.5, measures directly how much an NWP-grade covariate feed
\cite{rasp2018neural} is worth before it is worth training a specialist to consume
it.

**How often will the model be refreshed?** This, rather than the electricity price,
decides which option is cheaper. Under frequent retraining the zero-shot model uses
roughly an order of magnitude less energy and about eight times less money, because
it never trains. But training is a one-time cost that amortizes: a specialist trained
once and then served crosses over to being the more energy-efficient choice after
roughly 2,800–4,400 forecasts on CPU, about 120–180 days of hourly 24-hour-ahead
service for one series. Refresh more often than that and the zero-shot model wins on
cost; train once and serve for a year and the specialist does. Because price and PUE
act on the objective as linear rescalings of λ, a different local tariff slides an
operator along that same axis rather than reordering the options: no cell flips at
the central assumption, and at most 40% flip at the most extreme corners.

Two limits on how hard to push this. In absolute terms the electricity cost of either
tier is below 10⁻³ USD per 1,000 forecasts, so the energy argument is about scale and
operational simplicity rather than the utility bill; for a single city, hardware and
engineering time dominate. And the E4 result removes the middle option: because the
transfer recipe neither beats zero-shot on accuracy in the scarce regime nor is
demonstrably equivalent to it, there is no accuracy case for the extra pipeline it
requires. This is the deployment analogue of arguments that energy belongs in the
evaluation objective
\cite{schwartz2020green,strubell2019energy,garciamartin2019estimation}. Supplementary
Table S16 sets out the procedure for re-deriving these maps for a new city.

## Limitations

Seven limitations bound our claims. First, energy and hardware. CPU-side consumption is
not metered: the study machine runs Windows, where RAPL is unavailable, so codecarbon
models CPU draw from the processor's 65 W TDP entry scaled by process utilisation, whereas
GPU energy is read from NVML hardware counters. Every CPU-side energy figure here should
therefore be read as a **comparative, directional** quantity, valid for ranking tiers
measured on one machine under an identical protocol, not as an absolute measurement of
hardware energy and not transferable to other processors. We also benchmark a consumer GPU
and a CPU-only configuration, not a physical embedded device, so the "edge/CPU-class"
energy numbers are indicative rather than a Raspberry-Pi- or Jetson-class
measurement. We also did not log per-forecast latency or peak memory for the
canonical runs, so deployment-critical figures beyond energy (inference latency,
RAM/VRAM footprint, model storage) are reported only as the parameter count in
Table 3. The measured comparison includes per-fold training for trained tiers, so
the headline energy gap is specific to frequent-retraining deployment; our
amortization analysis shows a once-trained specialist eventually becomes more
energy-efficient per forecast, and both figures are reported. Second,
contamination. Chronos-Bolt is pretrained on large public time-series collections
whose exact composition we cannot fully audit, so we cannot rule out that the UCI
Beijing series or ERA5/OpenAQ-derived data entered its corpus. To guard against
this we re-evaluated on strictly post-2024-10 city windows (10 cities with
sufficient coverage, a subset that by construction skews toward well-instrumented
cities, since it requires a recent continuous record), which postdate the model's
pretraining corpus \cite{autogluon2024chronosbolt} and cannot have been seen: the
zero-shot model remained competitive with the specialist there (mean MASE 0.415 vs
0.395, Chronos-Bolt better in 6 of 10 cities, both far below the seasonal-naïve
floor; Supplementary Table S14)
<!-- L-038 -->, so the PM2.5 result is not an artifact of pretraining exposure. We
nonetheless treat the public Beijing 12-station set as corroborating rather than
primary evidence. Third, coverage. The panel uses one sensor per city, so
within-city spatial generalization rests on the Beijing 12-station check alone.
Fourth, model scope. The foundation-model tier is Chronos-Bolt-small; larger TSFMs
or future revisions may shift the accuracy–cost frontier, and our conclusions are
explicitly frontier-relative, not model-family-absolute
\cite{aksu2024gifteval,karaouli2025foundational}. Fifth, task and comparator scope.
We test hourly, 24–48 h direct multi-horizon point forecasting, with each tier at
its deployed context length (Chronos-Bolt four weeks, NAS-GRU 24 h). A
context-length-matched comparison (168 h and 672 h GRU lookbacks) is a natural
sensitivity check; our attempts to run it exceeded the memory of the study GPU
(8 GB), so we report the comparison at native contexts and flag context length as a
factor we did not isolate on this hardware. Longer horizons, probabilistic scoring
rules, and multivariate targets are likewise untested here. Sixth, statistics.
Split-conformal exchangeability holds only approximately under temporal dependence
\cite{xu2021conformal,stankeviciute2021conformal}, and with 15 scarce cities the E4
tests have limited power. Even where we establish equivalence (PM2.5 under
perfect-foresight covariates), it is at a 0.05-MASE margin, and where we do not
(causal PM2.5, temperature, E4) "no significant difference" remains a parity
statement, not proof of interchangeability. Seventh, period heterogeneity. Each city
contributes whichever window its source record supports, so the panel is not
contemporaneous, and ten of the 29 cities (34% of usable hours, seven of them
scarce-tier) overlap the 2020–21 pandemic period. Every comparison here is
within-city and between-tier against a per-series scale, so a local regime shift
rescales the problem for all tiers at once rather than favouring one; and the
post-cutoff check above, which lies entirely outside that period, reproduces the
result. A stratified re-analysis of the per-city results adds to this: the direction of
both domains' panel conclusions is preserved in the pandemic-exposed, unexposed and
heavily exposed strata alike, and no stratum differs significantly once the ten subgroup
tests are corrected for multiplicity. Two limits remain. The exposure is correlated with
the rich/scarce contrast rather than spread evenly across it, and the strata are smaller
than the panel and correspondingly less powerful, so a null within a stratum is not
positive evidence of equivalence. Together these checks bound the concern more tightly
than the post-cutoff test alone, without removing it (Supplementary Information).

## Outlook

Two extensions follow directly. Light fine-tuning of small TSFMs on pooled regional
data (the transfer recipe applied to the foundation model rather than the searched
model) would test whether the two strategies' strengths combine. The decision-rule
framework also generalizes: any deployment context can re-derive winner maps by
inserting local electricity prices, hardware, retraining frequency, and its own
cost-penalty coefficient, since all code, analysis plans, and per-claim
provenance ledgers are released. As monitoring networks in low- and middle-income
countries grow \cite{pinder2019opportunities,openaq}, our results indicate that the
marginal cost of a competent forecast for a new city is now close to the cost of
running a small pretrained model's inference loop, which bears directly on how
air-quality early-warning capacity can be scaled
\cite{cohen2017estimates,who2021guidelines}.
