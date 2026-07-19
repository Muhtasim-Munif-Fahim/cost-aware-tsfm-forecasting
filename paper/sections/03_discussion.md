# Discussion

## The efficiency calculus has changed

Our conference paper's premise was that foundation models are too computationally
heavy for edge deployment, so data-scarce cities need searched architectures and
transfer learning \cite{fahim2026greennas}; the assumption was widely shared at the
time. This study, run on the same evaluation philosophy but with a small foundation
model in the comparison, finds that the premise no longer holds. Under realistic
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
deployment decision. The earlier work's comparisons were internally sound; what
changed is that small TSFMs \cite{ansari2024chronos,autogluon2024chronosbolt} moved
the accuracy–cost frontier. "Make it tiny by search" and "make it cheap by not
training" now land in the same accuracy band, and the second is operationally
simpler. For an operator in a data-scarce city, the practical default shifts from
training a small model to downloading one.

## A covariate-timing pitfall that can produce spurious domain conclusions

A cross-domain claim can be an artifact of covariate timing alone. Had we used the
standard `future_covariates` convention uncritically, this paper would have reported
a clean domain flip (foundation model wins PM2.5, specialist wins temperature), with
26/29 significant city wins backing the second half. The causal ablation shows that
result depends entirely on granting the specialist a perfect forecast of its own
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
operator with access to NWP-grade covariate forecasts sits between our causal floor
and perfect-foresight ceiling \cite{rasp2018neural}; our data show how wide that
bracket can be.

## Cost-aware deployment guidance

The decision-rule maps translate the statistics into operational advice. Where
accuracy dominates (cost-penalty coefficient λ near zero) and covariate forecasts
are unavailable, the zero-shot model wins most cells in both domains; the specialist
earns its place mainly where high-quality covariate forecasts genuinely exist. At
large λ (strong cost aversion) the near-free naïve and, in the causal maps, the
amortizable locally-trained tiers become attractive, but the maps are robust to the
exact electricity price and PUE (no cell flips at the central assumptions, ≤ 40% at
the most extreme price × PUE corners). The E4 result complements this picture.
Because the transfer recipe does not beat zero-shot on accuracy in the scarce regime
and the two are not demonstrably equivalent, the decision turns on cost and
simplicity, and the zero-shot model needs no training step at all. This is the
deployment analogue of arguments that energy belongs in the evaluation objective
\cite{schwartz2020green,strubell2019energy,garciamartin2019estimation}, with one
caveat from our amortization analysis: which option is "cheaper" depends on
retraining frequency and model lifetime, not on the electricity price alone.

## Limitations

Six limitations bound our claims. First, energy and hardware. CPU-side consumption
on Windows is codecarbon's constant-power estimate rather than a hardware
measurement (GPU energy is NVML-measured), and we benchmark a consumer GPU and a
CPU-only configuration, not a physical embedded device, so the "edge/CPU-class"
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
floor; Supplementary Table S15)
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
statement, not proof of interchangeability.

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
