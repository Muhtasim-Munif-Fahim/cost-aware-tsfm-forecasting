#!/usr/bin/env python3
"""Build the anonymized Supplementary Software bundle for peer review.

Scientific Reports sends Supplementary Information to referees, so shipping the code
here gives them the artifact with no repository URL that would name the authors.

This is a REFEREE-FACING artifact, not a copy of the repository. Two consequences
drive the whole design:

  * Scope. Only the code that produces reported results ships: the forecasting
    harness, the data-retrieval scripts, and the statistical analyses. Figure and
    table generation, the internal number audit, the LaTeX build, the analysis plan
    and the results ledger are all excluded. A referee reads the method and checks
    it against the reported numbers; they do not regenerate our tables.

  * Comments. The repository's source carries internal project history -- who asked
    for what, when it was added, which deviation record logs it. "Reviewer" in those
    notes means the internal pre-submission review, which a journal referee would
    read as a record of changes some other referee demanded. All of it is stripped
    on the way into the zip; the repository keeps its own copy untouched.

Sanitizing prose with regexes is fragile, so this does not try. Module docstrings are
replaced wholesale from an authored table, embedded phrases are replaced literally
from a reviewed list, and three gates hard-fail the build rather than let something
slip: a denylist scan, the author-name scan, and an ast.parse() of every file so a
scrub cannot silently break Python.

Usage:  python paper/latex/make_code_si.py
Output: SUBMISSION/Supplementary_Software.zip  (SUBMISSION/ is gitignored)
"""
from __future__ import annotations

import ast
import os
import re
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))            # paper/latex
ROOT = os.path.dirname(os.path.dirname(HERE))                # repo root
OUT_DIR = os.path.join(ROOT, "SUBMISSION")
OUT_ZIP = os.path.join(OUT_DIR, "Supplementary_Software.zip")

# --------------------------------------------------------------------------- #
# Shipping set: repo path -> path inside the bundle. The bundle layout is
# deliberately NOT the repo layout.
# --------------------------------------------------------------------------- #
SHIP = {
    # how the forecasts and the input data were produced
    "src/run_forecast.py":                    "methods/run_forecast.py",
    "src/e4_transfer.py":                     "methods/e4_transfer.py",
    "src/stats_rigor.py":                     "methods/stats_rigor.py",
    "src/city_select.py":                     "methods/city_select.py",
    "src/openaq_fetch.py":                    "methods/openaq_fetch.py",
    "src/openmeteo_fetch.py":                 "methods/openmeteo_fetch.py",
    "src/batch_fetch.py":                     "methods/batch_fetch.py",
    # what produced each reported result
    "analysis/build_causal_primary.py":       "analysis/build_causal_primary.py",
    "analysis/build_causal_decision.py":      "analysis/build_causal_decision.py",
    "analysis/causal_covariate_ablation.py":  "analysis/causal_covariate_ablation.py",
    "analysis/panel_significance.py":         "analysis/panel_significance.py",
    "analysis/dm_panel.py":                   "analysis/dm_panel.py",
    "analysis/equivalence_tests.py":          "analysis/equivalence_tests.py",
    "analysis/conformal_panel.py":            "analysis/conformal_panel.py",
    "analysis/e4_significance.py":            "analysis/e4_significance.py",
    "analysis/energy_repeatability.py":       "analysis/energy_repeatability.py",
    "analysis/energy_amortization.py":        "analysis/energy_amortization.py",
    "analysis/cost_sensitivity.py":           "analysis/cost_sensitivity.py",
    "analysis/contamination_check.py":        "analysis/contamination_check.py",
    "analysis/fm_advantage_correlation.py":   "analysis/fm_advantage_correlation.py",
    "analysis/pandemic_exposure.py":          "analysis/pandemic_exposure.py",
}

VERBATIM = {"requirements.txt": "requirements.txt"}   # no docstring to rewrite

# --------------------------------------------------------------------------- #
# Authored module docstrings. Each states what the script computes, what it reads,
# and what it writes. No dates, no attribution, no project history.
# --------------------------------------------------------------------------- #
DOCSTRINGS = {
"methods/run_forecast.py": '''"""Forecasting harness: fits every model tier on a series and scores it.

Implements the five tiers compared in the paper -- the seasonal-naive floor, the
LightGBM direct multi-horizon specialist, the NAS-GRU, zero-shot Chronos-Bolt, and
Chronos-Bolt with a covariate-residual correction -- together with the rolling-origin
backtest, the fixed per-series MASE scale, and the covariate-timing switch that
separates the perfect-foresight configuration from the deployable causal one.

Accuracy is reported as MASE against one fixed scale per series (the in-sample error
of the seasonal-naive m = 24 forecast over the pre-test history), so scores stay
comparable when the training history varies. Energy is measured around the whole call
with codecarbon: GPU via NVML, CPU by codecarbon's constant-power estimate.

Modes:
  single   one series, full metrics, optionally saving per-step predictions
  cities   the multi-city panel
  sweep    many series, for the heterogeneity and record-length views
  regime   the training-history sweep behind the cost-adjusted decision maps
"""''',

"methods/e4_transfer.py": '''"""Transfer learning versus zero-shot forecasting in the data-scarce regime.

Tests whether a NAS-GRU pretrained on data-rich cities and fine-tuned on a scarce city
outperforms a foundation model that has never seen the target city.

Protocol: pretrain the NAS-GRU (two stacked GRU layers, 128 units) on the pooled
rich-tier cities, each z-scored on its own pre-test training region before pooling, so
a scarce city is only ever fine-tuned on and never enters the pretraining corpus. Then,
for each scarce city, each fine-tune budget in {0, 1, 10, 100}% of local history and
each seed: at 0% evaluate the pretrained model as-is on that city's held-out folds; above
0% continue training from the pretrained weights on the most recent share of the city's
pre-test window. Comparators on identical test folds are zero-shot Chronos-Bolt and a
LightGBM specialist refit on the same nominal budget.

Writes per-(city, strategy, fraction, seed) MASE, with the actual training hours behind
each nominal budget recorded alongside.
"""''',

"methods/stats_rigor.py": '''"""Shared statistical primitives: split-conformal intervals and the Diebold-Mariano test.

One implementation used by every tier and both domains, so the statistical treatment is
identical throughout.
"""''',

"methods/city_select.py": '''"""Select one long-record PM2.5 sensor per candidate city from the OpenAQ v3 API.

For each city coordinate, finds the PM2.5 sensor with the longest history within the
search radius and requires a minimum span. Writes the city manifest
(city, tier, sensor_id, first, last, years).

The rich/scarce tier is a property of the candidate list below, fixed before any data
were retrieved: it reflects the maturity of each city's regulatory monitoring
infrastructure, not the record length that retrieval happens to yield.

The OpenAQ key is read from the OPENAQ_KEY environment variable and never stored.
"""''',

"methods/openaq_fetch.py": '''"""Retrieve the full hourly PM2.5 history for one OpenAQ v3 sensor as timestamp,PM2.5 CSV.

Requests are year-windowed and paginated, because deep pagination is unstable for large
ranges, with a polite delay between calls. The API key is read from the OPENAQ_KEY
environment variable and never stored.
"""''',

"methods/openmeteo_fetch.py": '''"""Retrieve hourly weather covariates per city from the Open-Meteo historical archive.

Covariates: 2 m temperature, relative humidity, surface pressure, 10 m wind speed and
direction, precipitation, cloud cover, and shortwave radiation. In the temperature domain
the target is 2 m temperature and the covariate set excludes it, containing no variable
mathematically derived from it.

City coordinates are read from the candidate list in city_select.py. No API key required.
"""''',

"methods/batch_fetch.py": '''"""Retrieve every sensor listed in the city manifest into one CSV per city.

Resumable: each city checkpoints after every month, so an interrupted run loses only the
current month rather than the whole city, and partially fetched cities are always finished
before new ones are started. The OpenAQ key is read from the OPENAQ_KEY environment
variable.
"""''',

"analysis/build_causal_primary.py": '''"""Assemble the causal-covariate panel that the main results report.

Each tier contributes from the run that is causal for it. The seasonal-naive floor,
zero-shot Chronos-Bolt and the NAS-GRU consume only past context, so they are already
causal and are taken unchanged. The LightGBM specialist and the covariate-residual
Chronos variant both consume future covariates, so each is taken from its
causal-covariate run, in which every meteorological covariate enters at its last value
known at the forecast origin.

Writes the merged per-city predictions and the merged per-tier table, which the
Diebold-Mariano and conformal analyses then read.
"""''',

"analysis/build_causal_decision.py": '''"""Build the cost-adjusted deployment winner maps under causal covariates.

The NAS-GRU consumes only its past context window, so it is causal by construction and
its rows are carried over unchanged; the remaining tiers come from the causal-covariate
regime runs. For each (training-history regime x cost-penalty coefficient) cell the
winning tier is the one minimizing

    MASE + lambda * (USD per 1,000 forecasts)

evaluated at the central price and PUE assumptions. Writes one decision table per city
and domain.
"""''',

"analysis/causal_covariate_ablation.py": '''"""Quantify how much of the specialist's accuracy depends on a perfect covariate forecast.

Compares, per domain, the LightGBM specialist scored with perfect-foresight covariates
(each covariate taken at the forecast target time) against the same specialist scored
with causal covariates (each covariate at its last value known at the forecast origin),
and both against the covariate-free zero-shot foundation model.

A paired Wilcoxon test on per-city MASE quantifies how much foresight is worth in each
domain, and whether the specialist still leads once it is removed.
"""''',

"analysis/panel_significance.py": '''"""Panel-level significance tests over the per-city MASE table.

  - binomial sign test on per-city wins, specialist versus foundation model;
  - Wilcoxon signed-rank test on the paired per-city MASE for the same pair;
  - Friedman test with Nemenyi post-hoc across all tiers present, reported as a
    supplementary robustness check with its critical-difference values.

Multi-seed tiers are averaged to one per-city MASE before any test, so no test is run
against a single hand-picked seed.
"""''',

"analysis/dm_panel.py": '''"""Panel-level Diebold-Mariano tests from the saved per-city predictions.

Runs the all-pairs Diebold-Mariano test per city on absolute-error loss with the
Harvey-Leybourne-Newbold small-sample correction, then rolls the results up into a
panel-level count of significant wins per model pair.

Two robustness layers. Multi-seed tiers are tested as their seed-ensemble mean, which is
a stronger forecaster than any deployed single-seed model, so every such pair is ALSO
tested seed by seed and the per-seed agreement is reported alongside. Per-city p-values
within each model pair are then Benjamini-Hochberg adjusted across the panel, and both
raw and adjusted win counts are reported.
"""''',

"analysis/equivalence_tests.py": '''"""Equivalence tests for every comparison the paper reports as a tie.

A non-significant test is not evidence of equivalence. For each tie claim this reports
the paired per-city (or per-budget) MASE difference, a bootstrap 95% confidence interval,
and a two-one-sided-test (TOST) decision against an interpretability margin.

The margin is delta = 0.05 MASE, absolute: roughly 7-8% of the panel-mean MASE, and below
the resolution at which the cost-adjusted decision rule changes its winner. A paired
difference whose interval lies entirely inside +/- delta is not operationally meaningful.
The raw interval is reported alongside so a reader can apply a different threshold.

Note that TOST tests against the 1 - 2*alpha = 90% interval at alpha = 0.05, not the 95%
interval; the 95% bootstrap interval reported next to it is the conventional one on the
paired difference. The two are different quantities and are labelled as such.

Sign convention: d = MASE(specialist) - MASE(foundation model), so d > 0 favours the
foundation model.
"""''',

"analysis/conformal_panel.py": '''"""Split-conformal prediction intervals from the saved per-city predictions.

Calibrates the quantile on the first half of each series' backtest predictions and
reports empirical coverage and mean interval width on the second half, pooled per
tier x domain and again per city. Multi-seed tiers are averaged to one series first.

Conformal exchangeability holds only approximately under temporal dependence, so these
are reported as calibration evidence rather than exact guarantees.
"""''',

"analysis/e4_significance.py": '''"""Significance tests for the transfer-versus-zero-shot comparison.

The per-(city, strategy, fraction, seed) MASE table records scores but not raw
predictions, so the tests operate at the city level (n = 15 paired scarce cities per
budget), which is also the level the claim generalizes over:

  - paired Wilcoxon signed-rank on per-city MASE, zero-shot versus transferred model
    (seed-mean) and zero-shot versus the refit specialist, at each budget;
  - binomial sign test on per-city wins for the same pairs;
  - Holm correction across the budget family for each comparator, since four budgets by
    two comparators is eight tests and invites selective reading.

The verdict is assigned against a four-way interpretation matrix fixed before the tests
were run, rather than by comparing means.
"""''',

"analysis/energy_repeatability.py": '''"""Repeatability of the measured-energy figures.

A fixed workload -- the same invocation, six folds, all tiers, the stochastic tier pinned
to one seed so the training work is identical -- is rerun five times on three cities
spanning the depth, rich and scarce cases.

Aggregates the five repetitions per (city, tier) into mean, standard deviation and
sd/mean, and flags any cell above the 20% gate. Flagged cells are reported in the paper
as ranges rather than point estimates.
"""''',

"analysis/energy_amortization.py": '''"""Separate one-time training energy from per-forecast inference energy.

The whole-call energy comparison charges the specialist for the training it repeats every
fold and the foundation model for inference only, which is faithful to a
frequent-retraining deployment but not to a train-once one. This decomposes the two using
only whole-call codecarbon measurements, so no units are mixed, and computes the
amortization crossover.

Per city, on GPU and on CPU:

    E_retrain = energy(specialist, retraining every fold) = 6*train + 6*infer
    E_once    = energy(specialist, trained once)          = 1*train + 6*infer
    train_per_fit = (E_retrain - E_once) / 5
    infer_per_forecast = (E_once - train_per_fit) / (6*24)
    E_foundation = energy(zero-shot) = 6*infer, no training

The crossover N is the number of forecasts one trained specialist serves before its
amortized energy per forecast falls to the foundation model's inference energy:

    N = train_per_fit / (infer_foundation - infer_specialist)

reported as no crossover where the foundation model is already cheaper at inference.
"""''',

"analysis/cost_sensitivity.py": '''"""Sensitivity of the cost-adjusted decision maps to electricity price and PUE.

The decision objective is MASE + lambda * (USD per 1,000 forecasts), where lambda is the
cost-penalty coefficient (named wtp in this code) and the cost is computed at the central
assumptions of 0.15 USD/kWh and PUE 1.4. Price and PUE both scale the cost term linearly,
so varying either is equivalent to rescaling lambda:

    winner(lambda, price, PUE) == winner(lambda * (price/0.15) * (PUE/1.4), 0.15, 1.4)

This recomputes every winner map over a price x PUE x lambda grid from the saved regime
tables and reports how many cells flip relative to the central map, which demonstrates
that relationship mechanically rather than asserting it.
"""''',

"analysis/contamination_check.py": '''"""Check the foundation-model result against pretraining exposure.

Chronos-Bolt is pretrained on large public time-series collections whose exact contents
cannot be audited. Zero-shot guarantees no gradient updates on the test series, but not
that the series or a public copy was absent from the pretraining corpus.

The check re-evaluates the specialist-versus-foundation-model comparison using only each
city's data from 2024-10-01 onward, which postdates the pretraining corpus and therefore
cannot have entered it, for cities with enough recent coverage to support six folds.
Cities without that coverage are reported as excluded rather than dropped silently.

The public multi-station benchmark used elsewhere in the paper spans 2013-2017 and is
contamination-vulnerable, so it is treated as corroborating rather than primary evidence.
"""''',

"analysis/fm_advantage_correlation.py": '''"""Relationship between local record length and the foundation model's advantage.

Pearson correlation between a city's usable hours and its per-city MASE gap
(specialist minus foundation model), with a 10,000-resample bootstrap 95% confidence
interval.

The sign is reported as-is rather than pre-interpreted. With a panel of this size the
result bounds the association rather than establishing independence.
"""''',

"analysis/pandemic_exposure.py": '''"""Measure how much of the city panel falls in the 2020-21 pandemic period.

Each city contributes whichever usable window its source record supports, so the panel is
not contemporaneous and some cities sit wholly inside the pandemic period while others
lie entirely outside it. Nothing is excluded or reweighted here; this only measures the
exposure.

Overlap is computed against 2020-01-01 to 2021-12-31 inclusive, a deliberately generous
bracket that errs toward over-reporting exposure.
"""''',
}

# --------------------------------------------------------------------------- #
# Literal replacements for internal language embedded inside otherwise useful
# comments and inner docstrings. Reviewed one by one; applied as exact strings.
# --------------------------------------------------------------------------- #
PHRASES = [
    # cross-references to internal planning documents
    ("Per ANALYSIS_PLAN.md S6: ", ""),
    ("Per ANALYSIS_PLAN.md S6:", ""),
    ("ANALYSIS_PLAN.md S8's interpretation matrix separates",
     "The pre-specified interpretation matrix separates"),
    ('ANALYSIS_PLAN.md S5: "Repeatability: fixed workload rerun 5x on 3 cities -> report mean +/- sd;\n'
     'if sd/mean > 20% for a tier, report as a range in the main table rather than a point estimate."',
     ""),
    ("See paper/ANALYSIS_PLAN.md Sec.8 for the full\nprotocol and the pre-registered interpretation matrix (locked before this was run).",
     "The interpretation matrix was fixed before this was run."),
    ("(ANALYSIS_PLAN.md D3)", ""),
    ("See ANALYSIS_PLAN.md Deviations log 2026-07-14.", ""),
    ("per ANALYSIS_PLAN.md\nS4 (\"report mean +/- sd across seeds, no cherry-picking a best seed\")",
     "reporting mean +/- sd across seeds rather than a best seed"),
    # internal review / project history
    ("Two robustness layers (added at pre-writing review):", "Two robustness layers:"),
    ("causal_cov (ablation, added 2026-07-15): if True,", "causal_cov: if True,"),
    ("causal_cov (added 2026-07-15, reviewer-fix): mirrors", "causal_cov: mirrors"),
    ("Phase-1 audit finding: ", "Note: "),
    ("Phase 3 -- ", ""),
    ("Audit-trail backbone:", "Provenance record:"),
    # terminology the manuscript fixed: never "pre-registered"
    ("20% pre-registered gate", "20% gate"),
    ("pre-registered interpretation matrix", "pre-specified interpretation matrix"),
    # internal project framing
    ("Forecast harness for the SciRep energy paper (supersedes smoke_test.py).",
     "Forecasting harness."),
    ("Rigor layer for the merged paper: ", ""),
    ("Mirrors the Green-NAS paper's validation style (split conformal, significance testing)\n"
     "so all tiers/domains in the journal version share one statistical framework.", ""),
    ("Same weather source Green-NAS used,\nso the merged paper's two domains share weather inputs.", ""),
    ("Tests whether Green-NAS's published finding\n"
     '("1% fine-tune data ~ full-data accuracy via transfer") still beats "0% data via a zero-shot\n'
     'foundation model," now that small TSFMs exist.', ""),
    ("this environment reaps long-running background processes", "long runs can be interrupted"),
    ("for the 5-seed campaign runs", "for the 5-seed runs"),
    ("an unattended multi-hour campaign", "an unattended multi-hour run"),
    ("Build causal-primary decision winner maps for Figure 6.",
     "Build the causal-covariate decision winner maps."),
    ("format Figure 6 consumes", "format the decision-map figure consumes"),
    ("headline interpretation", "interpretation"),
    ("(headline robustness of the domain-flip claim)", ""),
    ("the headline causal tie", "the causal tie"),
    ("Phase 3 analysis should report actual hours", "downstream analysis reports actual hours"),
    ("wasted compute in the real\n        # campaign and a confusing ledger row",
     "wasted compute and a\n        # misleading record"),
    ("Referenced by paper/RESULTS_LEDGER.md rows.", ""),
]

# Whole-line comments matching this are dropped entirely.
DROP_COMMENT = re.compile(
    r"^\s*#.*\b(reviewer|deviations?\s+(log|record)|ANALYSIS_PLAN|RESULTS_LEDGER|"
    r"ledger row|pre-writing review|pre-submission|supervisor|Phase[- ]\d)\b",
    re.I,
)

# Gate 1: internal-process language must not survive.
# Two deliberate carve-outs, both legitimate statistical English that must NOT trip:
#   "post-hoc"  -> "Nemenyi post-hoc test"
#   "deviation" -> "standard deviation"; only the plural project sense is banned.
DENY = re.compile(
    r"\breviewer\b|\breviewer-|(?<!standard )\bdeviations\b|deviations?\s+(log|record)|"
    r"\bsupervisor\b|\bpre-submission\b|\bpre-registered\b|\baudit finding\b|"
    r"\bheadline\b|\bcampaign\b|"
    r"ANALYSIS_PLAN|RESULTS_LEDGER|\bQPAIN\b|added 20\d\d-\d\d-\d\d|\bPhase[- ]\d",
    re.I,
)

# Gate 2: author identity and local filesystem paths.
IDENTIFYING = re.compile(
    r"fahim|karim|muhtasim|rajshahi|ru\.ac\.bd|s1911024120|"
    r"[A-Za-z]:\\(Users|Survey)|/Users/[a-z]",
    re.I,
)

LICENSE_TEXT = """MIT License

Copyright (c) 2026 The Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

README = """# Supplementary Software

Code for the manuscript "Cost-Aware Evaluation of Time-Series Foundation Models for
Urban Air-Quality and Temperature Forecasting". Author names, affiliations and the
repository URL are withheld for double-anonymous review.

This bundle contains the code that produced the reported results: the forecasting
harness, the data-retrieval scripts, and the statistical analyses. It is meant for
reading and checking the method against the numbers in the paper. Result artifacts are
not included, so the tables and figures cannot be regenerated from this bundle alone;
the complete archive is released publicly on publication.

Install with `pip install -r requirements.txt`.

## methods/

| Script | Purpose |
| --- | --- |
| `run_forecast.py` | Forecasting harness: all five model tiers, the rolling-origin backtest, the fixed per-series MASE scale, the perfect-foresight/causal covariate switch, and the energy measurement. |
| `e4_transfer.py` | Transfer learning versus zero-shot in the data-scarce regime, across fine-tune budgets. |
| `stats_rigor.py` | Split-conformal intervals and the Diebold-Mariano test, shared by every tier and domain. |
| `city_select.py` | Selects one PM2.5 sensor per candidate city and fixes the rich/scarce tier assignment. |
| `openaq_fetch.py` | Retrieves hourly PM2.5 for one sensor. |
| `openmeteo_fetch.py` | Retrieves the hourly weather covariates and the temperature target. |
| `batch_fetch.py` | Retrieves every sensor in the city manifest. |

## analysis/

| Script | Produces |
| --- | --- |
| `build_causal_primary.py` | The causal-covariate panel reported as the main result (Table 1). |
| `build_causal_decision.py` | The cost-adjusted deployment winner maps (Figure 5). |
| `causal_covariate_ablation.py` | The covariate-timing result: how much of the specialist's temperature lead requires a perfect weather forecast (Figure 3). |
| `panel_significance.py` | Sign, Wilcoxon and Friedman-Nemenyi panel tests (Table 1). |
| `dm_panel.py` | Per-city Diebold-Mariano tests with FDR-adjusted win counts (Table 1, Table S5). |
| `equivalence_tests.py` | TOST equivalence tests for every reported tie (Table S15, Figure S2). |
| `conformal_panel.py` | Split-conformal coverage and interval width (Tables S4, S6, S7). |
| `e4_significance.py` | Holm-corrected transfer-versus-zero-shot tests (Table 2, Table S10). |
| `energy_repeatability.py` | Five-repetition energy repeatability and the 20% gate (Table 3, Table S8). |
| `energy_amortization.py` | Training/inference energy decomposition and the amortization crossover (Table S12). |
| `cost_sensitivity.py` | Price and PUE sensitivity of the decision maps (Table S11). |
| `contamination_check.py` | The post-cutoff check against pretraining exposure (Table S14). |
| `fm_advantage_correlation.py` | Correlation between record length and the foundation model's advantage. |
| `pandemic_exposure.py` | Share of the panel falling in the 2020-21 pandemic period. |

## Notes

Accuracy is MASE against one fixed scale per series, so scores remain comparable when
the training history varies. Energy is measured around the whole call with codecarbon:
GPU via NVML, CPU by codecarbon's constant-power estimate.

`OPENAQ_KEY` must be set in the environment for the retrieval scripts. No key is stored
in this bundle.

One difference from the archived version: the analysis scripts in the full archive each
append a provenance row to a project record file when they finish. That bookkeeping has
been removed here, because the record file is not part of this bundle. It performs no
computation and affects no reported value.
"""


def replace_docstring(text: str, dest: str) -> str:
    """Swap the module docstring for the authored one."""
    new = DOCSTRINGS.get(dest)
    if new is None:
        return text
    tree = ast.parse(text)
    node = tree.body[0] if tree.body else None
    is_doc = (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
              and isinstance(node.value.value, str))
    lines = text.split("\n")
    if is_doc:
        lines[node.lineno - 1:node.end_lineno] = new.split("\n")
    else:                                   # no docstring: insert after any shebang
        at = 1 if lines and lines[0].startswith("#!") else 0
        lines[at:at] = new.split("\n")
    return "\n".join(lines)


LEDGER_FUNCS = {"next_ledger_id", "append_ledger_stub"}


def strip_ledger_bookkeeping(text: str) -> str:
    r"""Remove the internal results-ledger bookkeeping from a shipped script.

    Eight analysis scripts append a provenance row to the project's internal ledger
    markdown when they finish. It is pure bookkeeping I/O -- it computes nothing and
    touches no reported number -- but it exposes our record-keeping workflow, and the
    ledger it writes to is not part of this bundle, so leaving it in would ship code
    that writes to a file the referee does not have.

    Excision is done through the AST rather than by pattern-matching text, so
    multi-line calls and multi-line assignments are removed exactly. Removal is
    disclosed in the bundle README.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text
    drop: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in LEDGER_FUNCS:
            start = min([d.lineno for d in node.decorator_list] + [node.lineno])
            drop.append((start, node.end_lineno))
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "LEDGER_PATH" for t in node.targets):
            drop.append((node.lineno, node.end_lineno))
        elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
              and isinstance(node.value.func, ast.Name)
              and node.value.func.id in LEDGER_FUNCS):
            drop.append((node.lineno, node.end_lineno))
    if not drop:
        return text
    cut = {n for lo, hi in drop for n in range(lo, hi + 1)}
    kept = [ln for i, ln in enumerate(text.split("\n"), 1) if i not in cut]
    out = "\n".join(kept)
    # Removing a call that was the sole statement of a block would leave it empty;
    # the syntax gate catches that, but check here so the message names the cause.
    try:
        ast.parse(out)
    except SyntaxError as e:
        raise SystemExit(
            f"ledger excision left an empty block at line {e.lineno}: {e.msg}") from e
    return out


def sanitize(text: str, dest: str) -> str:
    for old, new in PHRASES:
        text = text.replace(old, new)
    text = strip_ledger_bookkeeping(text)
    text = replace_docstring(text, dest)
    kept = [ln for ln in text.split("\n") if not DROP_COMMENT.match(ln)]
    text = "\n".join(kept)
    if dest == "analysis/conformal_panel.py":
        # In the repo this inserts the repo ROOT, but stats_rigor.py lives in src/, so the
        # import cannot resolve. In this layout it is the sibling methods/ directory.
        text = text.replace(
            "sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))",
            "sys.path.insert(0, os.path.join(\n"
            "    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), \"methods\"))",
        )
    return text


def gate(items: dict[str, str]) -> None:
    """Denylist, identity and syntax checks. Any failure aborts the build."""
    problems: list[str] = []
    for dest, body in sorted(items.items()):
        if dest.endswith(".py"):
            try:
                ast.parse(body)
            except SyntaxError as e:
                problems.append(f"{dest}: SANITIZER BROKE PYTHON at line {e.lineno}: {e.msg}")
        for i, line in enumerate(body.split("\n"), 1):
            for label, pat in (("internal", DENY), ("identity", IDENTIFYING)):
                if pat.search(line):
                    problems.append(f"{dest}:{i}: [{label}] {line.strip()[:96]}")
    if problems:
        print(f"GATE FAILED -- {len(problems)} issue(s) would have shipped:")
        for p in problems[:60]:
            print("  " + p)
        raise SystemExit(1)
    print(f"gates passed: {len(items)} files (internal language, identity, syntax)")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    missing = [s for s in list(SHIP) + list(VERBATIM)
               if not os.path.exists(os.path.join(ROOT, s))]
    if missing:
        raise SystemExit("missing expected file(s): " + ", ".join(missing))

    out: dict[str, str] = {}
    for src, dest in SHIP.items():
        with open(os.path.join(ROOT, src), encoding="utf-8") as f:
            out[dest] = sanitize(f.read(), dest)
    for src, dest in VERBATIM.items():
        with open(os.path.join(ROOT, src), encoding="utf-8") as f:
            out[dest] = f.read()
    out["README.md"] = README
    out["LICENSE"] = LICENSE_TEXT

    gate(out)

    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for dest, body in sorted(out.items()):
            z.writestr(dest, body)
    size = os.path.getsize(OUT_ZIP)
    print(f"wrote {os.path.relpath(OUT_ZIP, ROOT).replace(os.sep, '/')} "
          f"({len(out)} entries, {size / 1024:.0f} kB)")
    if size > 50 * 1024 * 1024:
        raise SystemExit("bundle exceeds the 50 MB Supplementary Information limit")


if __name__ == "__main__":
    main()
