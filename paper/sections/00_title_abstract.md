# Title

Cost-Aware Evaluation of Time-Series Foundation Models for Urban Air-Quality and
Temperature Forecasting

<!-- Retitled 2026-07-17 (was the MERGED_PAPER_PLAN.md working title "Search,
     transfer, or foundation model? ..."). Target: Scientific Reports
     (abstract <= 200 words, unreferenced). This file is the canonical abstract
     source: paper/latex/md2tex.py emits generated/abstract.tex from the
     "# Abstract" block below; do NOT edit the abstract inline in main.tex. -->

# Abstract

Cities with the greatest need for hyper-local air-quality and weather forecasts often
have the shortest data records and the tightest compute budgets. The usual responses,
searching for a tiny architecture or transferring from data-rich cities, assume
foundation models are too heavy for such deployments, an assumption more often adopted
than tested. Across 29 cities and two domains (hourly PM2.5 and 2 m temperature), under
pre-specified protocols with measured energy and a cost-adjusted decision rule, we compare
all three. On PM2.5 three zero-shot foundation models spanning two families and a
4.3-fold size range are statistically indistinguishable from the tuned specialist
(MASE 0.659-0.662 versus 0.692 under deployable causal covariates) and formally equivalent
to one another. The specialist's
apparent temperature edge (0.533 versus 0.792) depends heavily on covariate timing: 0.638
with covariates carrying measured 24-hour forecast error, 0.745 with only values knowable
at the forecast origin. Perfect foresight therefore distorts the cross-domain comparison. Any
benchmark using future covariates carries the same risk. Transfer learning
never significantly beats zero-shot at any budget. Under frequent retraining the untrained
model uses roughly ten times less measured energy, though a once-trained specialist
eventually amortizes. Where models are refreshed often, downloading one now matches
training one.

<!-- Word target: <= 200 (Scientific Reports). Numbers: L-037 (causal panel means),
     L-028 (perfect-foresight means), L-027 (foresight ablation), L-030 (equivalence),
     L-025/L-032 (energy). -->
<!-- Note: title keeps "edge"; hardware scope is hedged in text (edge/CPU-class,
     consumer GPU + CPU-only; no physical embedded device). -->
