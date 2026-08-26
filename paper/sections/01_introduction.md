# Introduction

Ambient fine particulate matter is among the largest environmental risk factors for
human health, responsible for millions of premature deaths annually and concentrated
disproportionately in cities of low- and middle-income countries
\cite{cohen2017estimates,who2021guidelines}. Managing that burden (issuing health
advisories, timing traffic or industrial interventions, planning sensor maintenance)
requires hyper-local, short-horizon forecasts at the level of an individual monitor.
The cities that need such forecasts most are also those where monitoring is newest
and sparsest. Low-cost sensor networks and open platforms such as OpenAQ have widened
coverage \cite{snyder2013changing,openaq}, but the records they produce are short,
gappy, and heterogeneous, and the institutions operating them rarely command cloud
budgets or ML engineering teams \cite{pinder2019opportunities}. The same constraint
binds for local weather variables such as near-surface temperature, whose
station-level forecasts feed heat-warning and energy systems \cite{rasp2018neural}:
global deep-learning weather models now rival numerical weather prediction (NWP)
\cite{lam2023graphcast,bi2023pangu}, but their outputs and compute remain out of reach
of a single-station operator on an edge budget.

A resource-constrained operator today faces three deployment strategies. The first is
to train an efficient specialist locally. Gradient-boosted trees remain a strong
widely deployable baseline for tabularized forecasting
\cite{ke2017lightgbm,makridakis2022m5,elsayed2021really}, deep specialists from
recurrent probabilistic models to graph and transformer architectures are widely
represented in the air-quality and general forecasting literatures
\cite{zheng2015forecasting,qi2019hybrid,tao2019air,chang2020lstm,salinas2020deepar,oreshkin2020nbeats,zhou2021informer,kong2025deep},
and neural architecture search (NAS) refines the recipe by looking for an
architecture that meets an explicit efficiency budget
\cite{zoph2017neural,elsken2019nas,liu2018darts}. The second is transfer learning:
pretrain where data are plentiful, then fine-tune on the short local record
\cite{oreshkin2021meta}. In our prior conference study we combined the two routes: we
discovered a 153k-parameter recurrent forecaster by multi-objective search and showed
that transfer from data-rich cities recovers most of the accuracy lost to short local
histories \cite{fahim2026greennas}. The present work inherits that architecture and
its edge/energy framing, and adds the foundation-model tier, the cost-adjusted
decision rule, the air-quality domain, and the transfer-versus-zero-shot crux
experiment. The third strategy, available only
since 2023, is to train nothing at all: pretrained time-series foundation models
(TSFMs) such as Chronos, TimesFM, Moirai, Lag-Llama and TimeGPT forecast zero-shot
from the local context window alone
\cite{ansari2024chronos,das2024timesfm,woo2024moirai,rasul2023lagllama,garza2023timegpt,goswami2024moment,liang2024foundation}.

Which strategy should such an operator choose? The benchmark literature does not
settle the question. Foundation-model evaluations report aggregate leaderboards
across heterogeneous datasets \cite{aksu2024gifteval}, and recent critiques question
how robust the zero-shot advantage is \cite{karaouli2025foundational}; the doubt
echoes much older evidence that complex methods do not automatically beat simple ones
\cite{makridakis2018statistical}. Deployment cost is largely absent from these
comparisons, although the green-AI literature has long argued that energy belongs in
the objective
\cite{schwartz2020green,strubell2019energy,patterson2021carbon,lacoste2019quantifying,rolnick2022tackling}
and the edge-ML community treats the joule budget as the binding constraint
\cite{banbury2020benchmarking,abadade2023tinyml}. Underlying much of that work,
including our conference study above \cite{fahim2026greennas}, is the assumption that
foundation models are too heavy for the edge and that efficiency must therefore come
from search and transfer. The assumption is reasonable, and it has been adopted far more often than it
has been tested. Compact foundation models make the test feasible: Chronos-Bolt-small
runs on a consumer GPU \cite{autogluon2024chronosbolt}, so what was a design
premise becomes a question the data can answer.

Three gaps follow from that literature, and this study addresses each.
First, foundation models are assessed as aggregate leaderboards over heterogeneous
collections \cite{aksu2024gifteval,karaouli2025foundational}, which establishes whether a
model is good on average but not which of the three strategies a particular operator
should deploy under a particular budget. Second, where deployment cost appears at all it
is a parameter count or an inference latency rather than measured energy, so the
green-AI position that energy belongs in the objective
\cite{schwartz2020green,strubell2019energy,patterson2021carbon} has not been carried
through into a model-selection rule an operator could apply. Third, the
air-quality forecasting literature and the foundation-model literature have developed
largely apart: the former tunes specialists on one city or one country
\cite{zheng2015forecasting,chang2020lstm,tao2019air}, the latter evaluates
general-purpose models on curated archives, and neither speaks to the short, gappy record
of the data-scarce city that motivates this work. Abbreviations and symbols used
throughout are collected in Supplementary Table S1.

A methodological problem compounds the strategy question. Forecast evaluation has
well-documented pitfalls \cite{hewamalage2023forecast,bergmeir2012use}, and we
examine one that matters specifically for comparisons across domains. Standard
evaluation harnesses hand covariate-using models their exogenous inputs at the
forecast target time (the `future_covariates` convention), which quietly assumes a
perfect weather forecast over the horizon. The assumption is harmless where
covariates carry little information about the target and decisive where they are
near-deterministic drivers of it, and we show below that it can, by itself, produce
a spurious domain-level conclusion about which strategy wins.

Here we test the three strategies directly. We compare five tiers on a 29-city panel
(14 data-rich, 15 data-scarce) in two domains, urban PM2.5 and location-level
temperature: a seasonal-naïve floor, a LightGBM specialist, the NAS-discovered
Green-NAS-A model, zero-shot Chronos-Bolt, and a covariate-residual Chronos variant.
Evaluation uses pre-specified quality gates, six-fold rolling-origin backtesting,
and Diebold–Mariano, sign, and Wilcoxon tests under false-discovery-rate (FDR)
control, with paired equivalence tests wherever we claim a tie; uncertainty is
quantified with split-conformal intervals, and energy is measured directly on
edge/CPU-class hardware (a consumer GPU and a CPU-only configuration). A
pre-specified crux experiment (E4) pits the transfer-learning recipe against the
zero-shot model across fine-tune budgets on the scarce cities. A
cost-adjusted decision rule, minimizing the mean absolute scaled error (MASE) +
λ · USD per 1,000 forecasts with λ a cost-penalty coefficient, then converts the
results into deployment winner maps.

The paper makes three contributions.

The first is the test that the premise had not received. To our knowledge this is the
first head-to-head comparison of search, transfer, and zero-shot deployment for
hyper-local urban forecasting in which accuracy, measured energy, and cost enter a
single decision rule, on a panel large enough to separate the data-rich case from the
data-scarce one. The premise does not survive it. On PM2.5, zero-shot Chronos-Bolt
and the tuned specialist are statistically indistinguishable across the panel:
neither records an FDR-significant win over the other, a paired two-one-sided
equivalence test (TOST) declares the two equivalent at a 0.05-MASE margin when the
specialist is granted perfect-foresight covariates, and under deployable causal
covariates the foundation model is, if anything, ahead. This is not a property of one
checkpoint: two further zero-shot models reproduce it, one changing checkpoint scale
4.3-fold within the same family and one changing family at matched scale, and all three
are certified equivalent to one another at the same margin. Its advantage shows no
detectable association with local data volume. In the data-scarce regime, transfer
learning never significantly beats the zero-shot model at any fine-tune budget, and
the zero-shot model leads at every one.

The second is methodological and outlives this dataset: covariate timing alone can
distort a domain-level conclusion. On temperature, most of the specialist's large
apparent superiority is an artifact of the standard future-covariates convention. We
quantify how much by calibrating an intermediate regime against measured 24-hour weather
forecast error: supplied with covariates of that realistic quality the specialist keeps
roughly three-fifths of its perfect-foresight advantage, and restricted to values knowable
at the forecast origin it keeps under a fifth. The convention therefore inflates a
cross-domain contrast that a realistically equipped specialist only partly supports, and
the corrected picture is graded rather than domain-flipped. Because the distortion scales with how informative
the covariates are about the target, it is nearly invisible within one domain and
surfaces only in comparisons across domains, which is exactly where it does the most
damage.
Any benchmark that consumes future covariates and compares across domains needs a
causal-covariate ablation to rule it out.

The third is the deployment consequence, which is what an operator can act on. Under
a frequent-retraining protocol the zero-shot model uses roughly an order of magnitude
less measured energy than the tiers that train locally. Separating training from
inference shows that gap is training-driven rather than intrinsic: a once-trained
specialist eventually amortizes and becomes the more energy-efficient option, so the
defensible recommendation is conditional on refresh cadence rather than universal.
For a city that retrains often, lacks a trustworthy forecast of its own covariates,
or has no ML engineering capacity to spare, the practical default has moved from
training a small model to downloading one. We release the decision rule so that an
operator can re-derive that answer at their own electricity price, hardware, and
retraining frequency.
