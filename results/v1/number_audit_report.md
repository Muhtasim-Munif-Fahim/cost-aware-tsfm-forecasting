# Phase 7 number-audit report

**152/152 checks passed.**

| check | ledger | description | expected | recomputed | status |
|---|---|---|---|---|---|
| pm25_mean_chronos | L-028 | PM2.5 panel MASE mean, chronos | 0.662 | 0.662 | PASS |
| pm25_sd_chronos | L-028 | PM2.5 panel MASE sd, chronos | 0.368 | 0.368 | PASS |
| pm25_mean_lgbm_direct | L-028 | PM2.5 panel MASE mean, lgbm_direct | 0.662 | 0.662 | PASS |
| pm25_sd_lgbm_direct | L-028 | PM2.5 panel MASE sd, lgbm_direct | 0.371 | 0.371 | PASS |
| pm25_mean_chronos_cov | L-028 | PM2.5 panel MASE mean, chronos_cov | 0.73 | 0.73 | PASS |
| pm25_sd_chronos_cov | L-028 | PM2.5 panel MASE sd, chronos_cov | 0.376 | 0.376 | PASS |
| pm25_mean_nas_gru | L-028 | PM2.5 panel MASE mean, nas_gru | 0.734 | 0.734 | PASS |
| pm25_sd_nas_gru | L-028 | PM2.5 panel MASE sd, nas_gru | 0.352 | 0.352 | PASS |
| pm25_mean_seasonal_naive | L-028 | PM2.5 panel MASE mean, seasonal_naive | 1.026 | 1.026 | PASS |
| pm25_sd_seasonal_naive | L-028 | PM2.5 panel MASE sd, seasonal_naive | 0.53 | 0.53 | PASS |
| we_mean_chronos | L-028 | Weather panel MASE mean, chronos | 0.792 | 0.792 | PASS |
| we_sd_chronos | L-028 | Weather panel MASE sd, chronos | 0.282 | 0.282 | PASS |
| we_mean_lgbm_direct | L-028 | Weather panel MASE mean, lgbm_direct | 0.533 | 0.533 | PASS |
| we_sd_lgbm_direct | L-028 | Weather panel MASE sd, lgbm_direct | 0.208 | 0.208 | PASS |
| we_mean_chronos_cov | L-028 | Weather panel MASE mean, chronos_cov | 0.718 | 0.718 | PASS |
| we_sd_chronos_cov | L-028 | Weather panel MASE sd, chronos_cov | 0.278 | 0.278 | PASS |
| we_mean_nas_gru | L-028 | Weather panel MASE mean, nas_gru | 0.999 | 0.999 | PASS |
| we_sd_nas_gru | L-028 | Weather panel MASE sd, nas_gru | 0.481 | 0.481 | PASS |
| we_mean_seasonal_naive | L-028 | Weather panel MASE mean, seasonal_naive | 1.686 | 1.686 | PASS |
| we_sd_seasonal_naive | L-028 | Weather panel MASE sd, seasonal_naive | 1.05 | 1.05 | PASS |
| pm25_dm_lgbm_fdr | L-019 | pm25: lgbm FDR wins vs chronos | 0 | 0 | PASS |
| pm25_dm_chron_fdr | L-019 | pm25: chronos FDR wins vs lgbm | 0 | 0 | PASS |
| pm25_dm_naive | L-019 | pm25: chronos FDR wins vs naive | 13 | 13 | PASS |
| weather_dm_lgbm_fdr | L-021 | weather: lgbm FDR wins vs chronos | 6 | 6 | PASS |
| weather_dm_chron_fdr | L-021 | weather: chronos FDR wins vs lgbm | 0 | 0 | PASS |
| weather_dm_naive | L-021 | weather: chronos FDR wins vs naive | 16 | 16 | PASS |
| pm25_sign_p | L-012 | PM2.5 sign-test p (lgbm vs chronos) | 0.136 | 0.136 | PASS |
| pm25_wilcox_p | L-012 | PM2.5 Wilcoxon p | 0.325 | 0.325 | PASS |
| pm25_friedman | L-012 | PM2.5 Friedman p < 0.001 | True | True | PASS |
| we_sign_p | L-023 | Weather sign-test p < 0.001 | True | True | PASS |
| we_wilcox_p | L-023 | Weather Wilcoxon p < 0.001 | True | True | PASS |
| pm25_corr_r | L-013 | pm25 advantage corr r | 0.075 | 0.075 | PASS |
| pm25_corr_p | L-013 | pm25 advantage corr p | 0.698 | 0.698 | PASS |
| pm25_corr_lo | L-013 | pm25 bootstrap CI lo | -0.196 | -0.196 | PASS |
| pm25_corr_hi | L-013 | pm25 bootstrap CI hi | 0.364 | 0.364 | PASS |
| weather_corr_r | L-024 | weather advantage corr r | 0.043 | 0.043 | PASS |
| weather_corr_p | L-024 | weather advantage corr p | 0.824 | 0.824 | PASS |
| weather_corr_lo | L-024 | weather bootstrap CI lo | -0.33 | -0.33 | PASS |
| weather_corr_hi | L-024 | weather bootstrap CI hi | 0.386 | 0.386 | PASS |
| ab_pm_perfect | L-027 | PM2.5 lgbm perfect | 0.662 | 0.662 | PASS |
| ab_pm_causal | L-027 | PM2.5 lgbm causal | 0.692 | 0.692 | PASS |
| ab_pm_chronos | L-027 | PM2.5 chronos | 0.662 | 0.662 | PASS |
| ab_pm_p_perf | L-027 | PM2.5 perfect-vs-chronos p ~0.33 | 0.33 | 0.33 | PASS |
| ab_pm_p_caus | L-027 | PM2.5 causal-vs-chronos p ~0.08 | 0.08 | 0.08 | PASS |
| ab_we_perfect | L-027 | Weather lgbm perfect | 0.533 | 0.533 | PASS |
| ab_we_causal | L-027 | Weather lgbm causal | 0.745 | 0.745 | PASS |
| ab_we_chronos | L-027 | Weather chronos | 0.792 | 0.792 | PASS |
| ab_we_gain | L-027 | Foresight gain +0.212 | 0.212 | 0.212 | PASS |
| ab_we_gain_p | L-027 | Foresight gain p ~2.6e-8 | True | True | PASS |
| ab_we_wins_perf | L-027 | Weather perfect wins 26/29 | 26/29 | 26/29 | PASS |
| ab_we_wins_caus | L-027 | Weather causal wins 16/29 | 16/29 | 16/29 | PASS |
| ab_we_p_perf | L-027 | Weather perfect p ~1.4e-6 | True | True | PASS |
| ab_we_p_caus | L-027 | Weather causal p ~0.29 | 0.29 | 0.29 | PASS |
| e4_chronos | L-009 | E4 chronos zero-shot mean | 0.843 | 0.843 | PASS |
| e4_nas_0 | L-009 | E4 nas_transfer @0% | 0.899 | 0.899 | PASS |
| e4_nas_1 | L-009 | E4 nas_transfer @1% | 0.915 | 0.915 | PASS |
| e4_nas_10 | L-009 | E4 nas_transfer @10% | 0.888 | 0.888 | PASS |
| e4_nas_100 | L-009 | E4 nas_transfer @100% | 0.876 | 0.876 | PASS |
| e4_lgbm_1 | L-009 | E4 lgbm_refit @1% | 0.941 | 0.941 | PASS |
| e4_lgbm_10 | L-009 | E4 lgbm_refit @10% | 0.944 | 0.944 | PASS |
| e4_lgbm_100 | L-009 | E4 lgbm_refit @100% | 0.858 | 0.858 | PASS |
| e4_bucket4 | L-018 | E4: no Holm-sig difference chronos vs nas_transfer at any fraction | True | True | PASS |
| e4_n_cities | L-018 | E4 n=15 cities | 15 | 15 | PASS |
| bj_n | L-026 | Beijing station count | 12 | 12 | PASS |
| bj_beats_lgbm | L-026 | chronos < lgbm at 12/12 | 12 | 12 | PASS |
| bj_beats_naive | L-026 | chronos < naive at 12/12 | 12 | 12 | PASS |
| bj_ch_lo | L-026 | chronos MASE min ~0.153 | 0.153 | 0.153 | PASS |
| bj_ch_hi | L-026 | chronos MASE max ~0.297 | 0.297 | 0.297 | PASS |
| bj_lg_lo | L-026 | lgbm MASE min ~0.292 | 0.292 | 0.292 | PASS |
| bj_lg_hi | L-026 | lgbm MASE max ~0.458 | 0.458 | 0.458 | PASS |
| bj_nv_lo | L-026 | naive MASE min ~1.03 | 1.03 | 1.03 | PASS |
| bj_nv_hi | L-026 | naive MASE max ~1.24 | 1.24 | 1.24 | PASS |
| en_gate | L-025 | 4 of 15 cells exceed 20% gate | 4 | 4 | PASS |
| en_lgbm_lo | L-025 | lgbm J/1k min ~9461 (9.5 kJ) | True | True | PASS |
| en_lgbm_hi | L-025 | lgbm J/1k max ~15061 (15.1 kJ) | True | True | PASS |
| en_chronos | L-025 | chronos unflagged cells within 1.0-1.3 kJ | True | True | PASS |
| en_usd_ratio | L-025 | lgbm/chronos USD ratio ~8x (7.5-9.5) | True | True | PASS |
| s12_max | L-040 | max flip rate 40% (causal) | 0.4 | 0.4 | PASS |
| s12_central | L-040 | 0 flips at central price/PUE (causal, all 6 runs) | 0.0 | 0.0 | PASS |
| s12_runs | L-040 | sensitivity covers 6 causal regime runs | 6 | 6 | PASS |
| cf_pm_lo | L-011 | PM2.5 pooled coverage min ~0.914 | 0.914 | 0.914 | PASS |
| cf_pm_hi | L-011 | PM2.5 pooled coverage max ~0.970 | 0.97 | 0.97 | PASS |
| cf_we_lo | L-022 | Weather pooled coverage min ~0.892 | 0.892 | 0.892 | PASS |
| cf_we_hi | L-022 | Weather pooled coverage max ~0.968 | 0.968 | 0.968 | PASS |
| cf_we_lgbm_width | L-022 | Weather rich lgbm width ~7.7 | 7.7 | 7.7 | PASS |
| cf_we_ch_width | L-022 | Weather rich chronos width ~10.7 | 10.7 | 10.7 | PASS |
| cf_pm_lgbm_width | L-011 | PM2.5 rich lgbm width ~17.7 | 17.7 | 17.7 | PASS |
| cf_pm_ch_width | L-011 | PM2.5 rich chronos width ~18.5 | 18.5 | 18.5 | PASS |
| cf_pm_nv_width | L-011 | PM2.5 rich naive width ~35.1 | 35.1 | 35.1 | PASS |
| dc_bj_pm | L-029 | Beijing PM2.5 chronos 21/25 | (21, 25) | (21, 25) | PASS |
| dc_se_pm | L-029 | Seoul PM2.5 chronos 17/25 | (17, 25) | (17, 25) | PASS |
| dc_na_pm | L-029 | Nairobi PM2.5 chronos-family 13/20 | (13, 20) | (13, 20) | PASS |
| dc_bj_we | L-029 | Beijing weather lgbm 11/25 | (11, 25) | (11, 25) | PASS |
| dc_se_we | L-029 | Seoul weather lgbm 12/25 | (12, 25) | (12, 25) | PASS |
| dc_na_we | L-029 | Nairobi weather lgbm 18/20 | (18, 20) | (18, 20) | PASS |
| cpm_mean_chronos | L-037 | causal PM2.5 mean chronos | 0.662 | 0.662 | PASS |
| cpm_sd_chronos | L-037 | causal PM2.5 sd chronos | 0.368 | 0.368 | PASS |
| cpm_mean_lgbm_direct | L-037 | causal PM2.5 mean lgbm_direct | 0.692 | 0.692 | PASS |
| cpm_sd_lgbm_direct | L-037 | causal PM2.5 sd lgbm_direct | 0.374 | 0.374 | PASS |
| cpm_mean_chronos_cov | L-037 | causal PM2.5 mean chronos_cov | 0.797 | 0.797 | PASS |
| cpm_sd_chronos_cov | L-037 | causal PM2.5 sd chronos_cov | 0.481 | 0.481 | PASS |
| cpm_mean_nas_gru | L-037 | causal PM2.5 mean nas_gru | 0.734 | 0.734 | PASS |
| cpm_sd_nas_gru | L-037 | causal PM2.5 sd nas_gru | 0.352 | 0.352 | PASS |
| cpm_mean_seasonal_naive | L-037 | causal PM2.5 mean seasonal_naive | 1.026 | 1.026 | PASS |
| cpm_sd_seasonal_naive | L-037 | causal PM2.5 sd seasonal_naive | 0.53 | 0.53 | PASS |
| cwe_mean_chronos | L-037 | causal weather mean chronos | 0.792 | 0.792 | PASS |
| cwe_sd_chronos | L-037 | causal weather sd chronos | 0.282 | 0.282 | PASS |
| cwe_mean_lgbm_direct | L-037 | causal weather mean lgbm_direct | 0.745 | 0.745 | PASS |
| cwe_sd_lgbm_direct | L-037 | causal weather sd lgbm_direct | 0.223 | 0.223 | PASS |
| cwe_mean_chronos_cov | L-037 | causal weather mean chronos_cov | 2.614 | 2.614 | PASS |
| cwe_sd_chronos_cov | L-037 | causal weather sd chronos_cov | 1.044 | 1.044 | PASS |
| cwe_mean_nas_gru | L-037 | causal weather mean nas_gru | 0.999 | 0.999 | PASS |
| cwe_sd_nas_gru | L-037 | causal weather sd nas_gru | 0.481 | 0.481 | PASS |
| cwe_mean_seasonal_naive | L-037 | causal weather mean seasonal_naive | 1.686 | 1.686 | PASS |
| cwe_sd_seasonal_naive | L-037 | causal weather sd seasonal_naive | 1.05 | 1.05 | PASS |
| cdm_pm_lgbm | L-033 | causal PM2.5 lgbm 0/0 vs chronos | (0, 0) | (0, 0) | PASS |
| cdm_we_lgbm | L-034 | causal weather lgbm 0/1 vs chronos (specialist 0 wins) | (0, 1) | (0, 1) | PASS |
| eq_pm_perfect | L-030 | PM2.5 perfect-foresight lgbm-vs-chronos EQUIVALENT | True | True | PASS |
| eq_pm_causal | L-030 | PM2.5 causal lgbm-vs-chronos NOT equivalent | False | False | PASS |
| eq_we_causal | L-030 | weather-causal lgbm-vs-chronos NOT equivalent | False | False | PASS |
| eq_e4_none | L-030 | E4: no fraction equivalent | True | True | PASS |
| eq_pm_perfect_p | L-030 | PM2.5 perfect TOST p~0.026 | 0.026 | 0.026 | PASS |
| am_lgbm_cheaper | L-032 | lgbm inference cheaper than chronos in all cells | True | True | PASS |
| am_cpu_range | L-032 | CPU crossover in [2778,4416] | True | True | PASS |
| cps_sign_p | L-039 | causal pm25 sign test p~0.024 | 0.024 | 0.024 | PASS |
| cps_sign_wins | L-039 | causal pm25 chronos better 21/29 | (21, 29) | (21, 29) | PASS |
| cps_wil_p | L-039 | causal pm25 wilcoxon p~0.084 | 0.084 | 0.084 | PASS |
| cps_fri_p | L-039 | causal pm25 friedman p<0.001 | True | True | PASS |
| nem_pm_chronos | L-039 | causal pm25 chronos rank ~1.72 | 1.72 | 1.72 | PASS |
| nem_pm_lgbm | L-039 | causal pm25 lgbm rank ~2.41 | 2.41 | 2.41 | PASS |
| nem_we_lgbm | L-041 | causal weather lgbm rank ~1.72 | 1.72 | 1.72 | PASS |
| nem_we_chronos | L-041 | causal weather chronos rank ~1.93 | 1.93 | 1.93 | PASS |
| nem_rank_gap | L-041 | spec-FM rank gap < CD 1.13 in both domains | True | True | PASS |
| corr42_pm | L-042 | causal pm25 corr r~-0.031 | -0.031 | -0.031 | PASS |
| corr42_p | L-042 | causal pm25 corr non-sig | True | True | PASS |
| corr43_we | L-043 | causal weather corr r~0.138 | 0.138 | 0.138 | PASS |
| corr43_p | L-043 | causal weather corr non-sig | True | True | PASS |
| s16_05 | L-030 | exactly 1 comparison equivalent at margin 0.05 | 1 | 1 | PASS |
| s16_10 | L-030 | 3 comparisons equivalent at margin 0.10 | 3 | 3 | PASS |
| ct_n | L-038 | post-cutoff cities n=10 | 10 | 10 | PASS |
| ct_chronos | L-038 | post-cutoff chronos mean ~0.415 | 0.415 | 0.415 | PASS |
| ct_lgbm | L-038 | post-cutoff lgbm mean ~0.395 | 0.395 | 0.395 | PASS |
| ct_wins | L-038 | post-cutoff chronos beats lgbm 6/10 | 6 | 6 | PASS |
| cdc_bj_pm | L-031 | causal Beijing PM2.5 chronos 21/25 | (21, 25) | (21, 25) | PASS |
| cdc_se_pm | L-031 | causal Seoul PM2.5 chronos 14/25 | (14, 25) | (14, 25) | PASS |
| cdc_bj_we | L-031 | causal Beijing weather lgbm 6/25 | (6, 25) | (6, 25) | PASS |
| cdc_se_we | L-031 | causal Seoul weather nas 25/25 | (25, 25) | (25, 25) | PASS |
| ccf_pm_lo | L-035 | causal PM2.5 coverage min ~0.910 | 0.91 | 0.91 | PASS |
| ccf_we_lo | L-036 | causal weather coverage min ~0.900 | 0.9 | 0.9 | PASS |
| ccf_we_lgbm_w | L-036 | causal weather rich lgbm width ~9.6 | 9.6 | 9.6 | PASS |
| ledger_join | - | all 26 cited ledger IDs exist in ledger | none missing | none missing | PASS |
| no_superseded | - | no superseded ledger row cited in prose | none | none | PASS |
