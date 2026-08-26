# Pilot artifacts (pre-campaign) — archived 2026-07-13

Everything in this directory was produced **before** the Phase 0–1 audit rebuild
(see `../../paper/ANALYSIS_PLAN.md`). It validated the research direction and headline
story but is **not** citable in the manuscript — no number here has a `RESULTS_LEDGER.md`
entry. Kept for provenance only. The canonical, audited results live under `results/v1/`.

## Known defects contaminating these files

| File(s) | Defect | Status |
|---|---|---|
| `pm25_single_*`, `pm25_sweep_*`, `single_*`, `sweep_*`, `smoke_*`, `regime_*` (early, pre-`pm25_` prefix) | Beijing loader ran **before** `sanitize_pm25` existed — sentinel/impossible values (v<=0, v>=985) were not stripped | Fixed in Phase 1.1; Beijing fully regenerated in Phase 2.2 |
| `pm25_fm_*`, `pm25_fmregime_*`, `pm25_fmsweep_*` | First Chronos integration; per-regime MASE denominator drifted (not yet fixed to a shared scale) | Superseded by `pm25_fair_*` |
| `pm25_fair_*` | Fixed MASE scale + retrain-per-fold; single-station only, single-seed NAS n/a (no NAS tier yet) | Valid methodology, superseded by full campaign |
| `pm25_alltiers_*` | First 5-tier run (incl. nas_gru, chronos_cov) on Beijing; nas_gru single-seed, 15-epoch cap (later upgraded to early-stopping/50-epoch) | Superseded by Phase 2.2 regeneration |
| `contaminated_results/` | **First cross-city panel run** — 12/29 cities had sentinel/impossible PM2.5 values (985 cap, Seoul 10000, negative readings in Mumbai/Vienna/Madrid) that were not caught by the quality gate (gate checked gaps/coverage, not value plausibility) | `sanitize_pm25` added; this exact contamination class is what Phase 1.1 guards against project-wide |
| `pm25_final_panel_cities*`, `final_*` | Clean 28-city panel (Mumbai dropped: 30-min sensor broke lgbm/nas/cov tiers) — this is the **pilot headline result** that motivated the full campaign, but pre-dates: multi-seed NAS, codecarbon measured energy, run-config capture, per-city rigor (DM/conformal) capture, and the Mumbai fix | Regenerated in Phase 2.3 as the 29-city canonical panel |
| `pm25_clean_panel.csv` | Same panel as above with Mumbai row dropped, deduplicated | Same as above |

## Why archived, not deleted
Nothing here is wrong in a way that invalidates the *decision* to proceed — it's exactly what
motivated the Phase 0-7 audited rebuild. Diffing the final canonical results against these
pilot numbers (Risk R5 in the plan) is a sanity check, not busywork.
