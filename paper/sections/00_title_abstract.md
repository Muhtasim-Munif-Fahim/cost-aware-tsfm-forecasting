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
have the shortest data records and the tightest compute budgets. Across 29 cities and
two domains (hourly PM2.5 and 2 m temperature), under pre-specified protocols with
measured energy and a cost-adjusted decision rule, we compare three deployment
strategies: a NAS-discovered tiny specialist, transfer learning from data-rich
cities, and a small zero-shot time-series foundation model. On PM2.5 the
zero-shot model and the specialist are statistically indistinguishable (MASE 0.662 vs
0.692 under deployable causal covariates; formally equivalent at a 0.05-MASE margin
only when the specialist is granted perfect-foresight covariates). The specialist's
apparent temperature edge (0.533 vs 0.792) shrinks to a small, non-significant
difference (0.745) once its covariates are restricted to values knowable at forecast
time: a perfect-foresight artifact that can generate spurious cross-domain
conclusions. Transfer learning never significantly beats zero-shot at any fine-tune
budget. Under frequent retraining the untrained model uses roughly ten times less
measured energy; the gap is training-driven, and a once-trained specialist eventually
becomes the more energy-efficient option. For data-scarce deployment with frequent
model refreshes, a downloaded foundation model matches local training on accuracy and
is simpler and cheaper to operate.

<!-- Word target: <= 200 (Scientific Reports). Numbers: L-037 (causal panel means),
     L-028 (perfect-foresight means), L-027 (foresight ablation), L-030 (equivalence),
     L-025/L-032 (energy). -->
<!-- Note: title keeps "edge"; hardware scope is hedged in text (edge/CPU-class,
     consumer GPU + CPU-only; no physical embedded device). -->
