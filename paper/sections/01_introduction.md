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
to train an efficient specialist locally. Gradient-boosted trees remain the strongest
widely deployable baseline for tabularized forecasting
\cite{ke2017lightgbm,makridakis2022m5,elsayed2021really}, deep specialists from
recurrent probabilistic models to graph and transformer architectures dominate the
air-quality and general forecasting literatures
\cite{zheng2015forecasting,qi2019hybrid,tao2019air,chang2020lstm,salinas2020deepar,oreshkin2020nbeats,zhou2021informer,kong2025deep},
and neural architecture search (NAS) refines the recipe by looking for an
architecture that meets an explicit efficiency budget
\cite{zoph2017neural,elsken2019nas,liu2018darts}. The second is transfer learning:
pretrain where data are plentiful, then fine-tune on the short local record
\cite{oreshkin2021meta}. Our IEEE QPAIN conference paper, Green-NAS, combined the two
routes; it discovered a 153k-parameter recurrent forecaster by multi-objective search
and showed that transfer from data-rich cities recovers most of the accuracy lost to
short local histories \cite{fahim2026greennas}. The third strategy, available only
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
\cite{banbury2020benchmarking,abadade2023tinyml}. Our own conference paper worked
from the assumption, standard in the field at the time, that foundation models are
too heavy for the edge and efficiency must therefore come from search and transfer
\cite{fahim2026greennas}. Small foundation models have since undermined that premise:
Chronos-Bolt-small runs on a consumer GPU \cite{autogluon2024chronosbolt}, so the
assumption is now a question the data can answer.

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
pre-specified crux experiment (E4) pits the conference paper's transfer-learning
recipe against the zero-shot model across fine-tune budgets on the scarce cities. A
cost-adjusted decision rule, minimizing the mean absolute scaled error (MASE) +
λ · USD per 1,000 forecasts with λ a cost-penalty coefficient, then converts the
results into deployment winner maps.

Four findings emerge. On PM2.5, zero-shot Chronos-Bolt and the tuned specialist are
statistically indistinguishable across the panel: neither records an FDR-significant
win over the other, a paired two-one-sided equivalence test (TOST) declares the two
equivalent at a 0.05-MASE margin when the specialist is granted perfect-foresight
covariates, and under deployable causal covariates the foundation model is, if
anything, ahead. The foundation model's advantage shows no detectable association
with local data volume. On temperature, the specialist's large apparent superiority
is an artifact of perfect-foresight covariates: once covariates are restricted to
values knowable at the forecast origin, its lead shrinks to a small, non-significant
margin. The corrected cross-domain picture is therefore graded rather than
domain-flipped: no significant difference in either domain, with formal equivalence
established only in the perfect-foresight PM2.5 case. In the data-scarce regime,
transfer learning, the strategy our conference paper recommended, never
significantly beats the zero-shot model at any fine-tune budget, and the zero-shot
model leads at every budget. Finally, under a frequent-retraining protocol the
zero-shot model uses roughly an order of magnitude less measured energy than the
tiers that train locally. Separating training from inference shows the gap is
training-driven and specific to frequent retraining: a once-trained specialist
eventually becomes the more energy-efficient option, and the zero-shot model's
durable advantages are that it avoids training altogether and is simpler to operate.
Together these results quantify how small time-series foundation models have changed
the efficiency calculus for resource-constrained forecasting in data-scarce cities,
and they identify a covariate-timing pitfall that cross-domain forecasting
comparisons need to control for.
