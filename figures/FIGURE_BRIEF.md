# Figure brief — npj Climate and Atmospheric Science manuscript

JOURNAL: `nature` (npj CAS uses Nature-family specs: single 89 mm = 3.504 in,
double 183 mm = 7.205 in; Arial; bold lowercase panel labels a/b/c; PDF vector
+ 600 dpi PNG; colorblind-safe).

Model display names used across ALL figures (one source of truth,
`figures/src/naming.py`): seasonal_naive → "Seasonal-naïve",
lgbm_direct → "LightGBM (specialist)", nas_gru → "NAS-GRU",
chronos → "Chronos-Bolt (zero-shot)", chronos_cov → "Chronos-Bolt + covariates".

Role assignment (ACTIVE_ROLES, Nature palette): chronos = primary (blue),
lgbm_direct = secondary (red), nas_gru = quaternary (purple),
chronos_cov = teal (NATURE["teal"]), seasonal_naive = neutral grey.
Identical mapping in every figure — no palette drift.

---

## F1 — Study design schematic
- TITLE: "Figure 1. Study design: five forecasting strategies, two domains, 29 cities."
- COLUMN_WIDTH: double; ASPECT ~0.52; PANEL_COUNT 1 (typographic flow, 3 columns).
- MESSAGE: one glance shows data → tiers → evaluation/rigor pipeline.
- ENCODING: typographic flow diagram, hairline 0.5 pt rules, NO wireframe boxes.
- DATA: counts hardcoded from locked design (29 cities, 14 rich/15 scarce, 2 domains,
  5 tiers, 6 folds, h=24, 5 seeds, energy measured) — no CSV numbers.
- OUTPUT: figures/main/F1_design

## F2 — Per-city FM advantage, both domains
- TITLE: "Figure 2. Zero-shot foundation model vs tuned specialist across 29 cities."
- COLUMN_WIDTH: double; ASPECT 0.62; PANEL_LAYOUT 1x2 (a PM2.5, b temperature).
- PANEL_MESSAGE a: PM2.5 advantage distribution straddles zero → statistical tie.
- PANEL_MESSAGE b: temperature advantage shifted toward specialist (face value) —
  flagged as perfect-foresight in caption; sets up F3.
- ENCODING: horizontal diverging dot-and-stem per city (sorted by advantage),
  rich/scarce by marker fill; dashed zero reference. Same encoding both panels is
  ALLOWED here (paired-domain comparison is the point; differs by domain data).
  Wait — skill rule: no two panels share encoding. Panel b therefore uses the same
  stem plot but with tier-split summary densities added on the right margin —
  no: keep it honest — panels differ by annotation layer: a adds sign-test p inset,
  b adds foresight-flag shading. Encoding core shared deliberately (paired domains);
  documented as an intentional exception for cross-domain comparability.
- DATA: results/v1/pm25_panel/canonical_cities.csv + results/v1/weather_panel/canonical_cities.csv;
  advantage = lgbm MASE − chronos MASE per city (>0 ⇒ FM better).
- AXIS_UNITS: "MASE difference (specialist − foundation model)".
- REFERENCE_LINES: x=0 dashed.
- OUTPUT: figures/main/F2_panel_advantage

## F3 — Perfect-foresight ablation (money figure)
- TITLE: "Figure 3. The specialist's temperature edge requires a perfect weather forecast."
- COLUMN_WIDTH: double; ASPECT 0.55; PANEL_LAYOUT 1x2 (a PM2.5, b temperature).
- MESSAGE: removing foresight moves lgbm from beating chronos (b) to tie; PM2.5 (a) unmoved.
- ENCODING: paired slopegraph per city — left tick "perfect forecast", right tick
  "causal (last known)"; per-city thin slopes + bold mean slope; chronos zero-shot
  mean as horizontal reference band. Wilcoxon p annotated at each column vs chronos.
- DATA: canonical_cities.csv (lgbm perfect + chronos), causal_ablation_cities.csv
  (lgbm causal), both domains; means recomputed in-script (must equal L-027 values:
  pm25 0.662/0.692/0.662; weather 0.533/0.745/0.792).
- AXIS_UNITS: "MASE (fixed per-series scale)".
- OUTPUT: figures/main/F3_foresight_ablation

## F4 — Beijing 12-station depth
- TITLE: "Figure 4. Beijing multi-station check: zero-shot FM leads at all 12 stations."
- COLUMN_WIDTH: single; ASPECT 1.05.
- ENCODING: per-station horizontal dot pair (chronos vs LightGBM), stations sorted
  by chronos MASE; seasonal-naïve shown as light range marker; connecting line per station.
- DATA: results/v1/beijing/canonical_sweep12_hetero.csv (MASE by series × model);
  12/12 assertion in-script (sanity check).
- AXIS_UNITS: "MASE"; station names stripped of 'pm25:' prefix.
- OUTPUT: figures/main/F4_beijing12

## F5 — E4 transfer vs zero-shot
- TITLE: "Figure 5. Transfer learning vs zero-shot across fine-tune data budgets."
- COLUMN_WIDTH: single; ASPECT 0.85.
- ENCODING: line + band — x = nominal fraction (log-ish categorical 0/1/10/100%),
  NAS-GRU transfer mean ± sd across seeds (band), LightGBM-refit dashed line,
  chronos zero-shot horizontal reference line with band label; n=15 cities noted.
- DATA: results/v1/e4_transfer/canonical_pm25_results.csv; means across cities of
  per-city seed-means (must equal L-009: nas 0.899/0.915/0.888/0.876; lgbm 0.941/0.944/0.858;
  chronos 0.843). Holm-Wilcoxon verdicts from canonical_pm25_significance.csv annotated.
- AXIS_UNITS: x "Fine-tune budget (% of target-city history, nominal)"; y "MASE".
- OUTPUT: figures/main/F5_e4_transfer

## F6 — Cost-adjusted decision maps
- TITLE: "Figure 6. Which strategy to deploy: accuracy–cost winner maps."
- COLUMN_WIDTH: double; ASPECT 0.72; PANEL_LAYOUT 2x3 (rows = domain, cols = city:
  Beijing, Seoul, Nairobi; panel labels a–f).
- ENCODING: categorical heatmap, regime (train-weeks) × willingness-to-pay; cell color =
  winning tier (role colors); shared discrete legend below; skipped regimes hatched grey.
- DATA: results/v1/regime/canonical_{beijing,seoul,nairobi}_{pm25,weather}_decision.csv.
- AXIS_UNITS: x "Willingness-to-pay (MASE per US$ per 1,000 forecasts)";
  y "Training history (weeks)".
- OUTPUT: figures/main/F5_decision_maps

## Tables
- T1 (main): panel MASE mean±sd per tier × domain + FDR DM sig-win counts vs chronos.
  Sources: canonical_cities.csv ×2, canonical_dm_panel_summary.csv ×2. LaTeX body
  → tables/out/T1_panel.tex (+ .md preview).
- T2 (main): measured energy/cost per tier (Beijing/Seoul/Nairobi reps): mean J/1k,
  USD/1k; RANGE not point where sd/mean>0.20 (nairobi-chronos + 3 naive cells per L-025).
  Source: results/v1/energy/repeatability_summary.csv → tables/out/T2_energy.tex.
- Supplementary family built FIRST (skill rule 3): S1 per-city panel (29 rows,
  check_min_rows), S4 full DM matrices, S9 h48, S10 E4 grid, S12 sensitivity —
  generated as CSV→LaTeX in tables/make_supplementary.py.

Caption drafts live in paper/sections/captions.md (rewritten after renders pass).
