| Domain      | Covariates          | Tier                      | MASE          | MAE             | RMSE            |   Avg rank | FDR wins (tier/FM)   |
|:------------|:--------------------|:--------------------------|:--------------|:----------------|:----------------|-----------:|:---------------------|
| PM2.5       | Causal (deployable) | Chronos-Bolt (zero-shot)  | 0.662 ± 0.368 | 10.553 ± 13.524 | 15.093 ± 19.273 |       1.72 | —                    |
| PM2.5       | Causal (deployable) | Chronos-Bolt + covariates | 0.797 ± 0.481 | 13.500 ± 19.643 | 18.425 ± 26.624 |       3.34 | 0 / 5                |
| PM2.5       | Causal (deployable) | LightGBM (specialist)     | 0.692 ± 0.374 | 10.902 ± 13.617 | 15.245 ± 19.246 |       2.41 | 0 / 0                |
| PM2.5       | Causal (deployable) | NAS-GRU                   | 0.734 ± 0.352 | 11.426 ± 14.931 | 15.592 ± 20.791 |       2.9  | 0 / 2                |
| PM2.5       | Causal (deployable) | Seasonal-naïve            | 1.026 ± 0.530 | 16.505 ± 21.235 | 22.274 ± 28.983 |       4.62 | 0 / 13               |
| PM2.5       | Perfect foresight   | Chronos-Bolt (zero-shot)  | 0.662 ± 0.368 | 10.553 ± 13.524 | 15.093 ± 19.273 |       1.83 | —                    |
| PM2.5       | Perfect foresight   | Chronos-Bolt + covariates | 0.730 ± 0.376 | 11.745 ± 15.266 | 15.959 ± 20.478 |       3.14 | 1 / 2                |
| PM2.5       | Perfect foresight   | LightGBM (specialist)     | 0.662 ± 0.371 | 10.845 ± 14.060 | 14.810 ± 18.895 |       2.17 | 0 / 0                |
| PM2.5       | Perfect foresight   | NAS-GRU                   | 0.734 ± 0.352 | 11.426 ± 14.931 | 15.592 ± 20.791 |       3.07 | 0 / 2                |
| PM2.5       | Perfect foresight   | Seasonal-naïve            | 1.026 ± 0.530 | 16.505 ± 21.235 | 22.274 ± 28.983 |       4.79 | 0 / 13               |
| Temperature | Causal (deployable) | Chronos-Bolt (zero-shot)  | 0.792 ± 0.282 | 1.303 ± 0.717   | 1.819 ± 1.040   |       1.93 | —                    |
| Temperature | Causal (deployable) | Chronos-Bolt + covariates | 2.614 ± 1.044 | 3.923 ± 1.821   | 4.919 ± 2.217   |       4.72 | 0 / 27               |
| Temperature | Causal (deployable) | LightGBM (specialist)     | 0.745 ± 0.223 | 1.242 ± 0.701   | 1.629 ± 0.903   |       1.72 | 0 / 1                |
| Temperature | Causal (deployable) | NAS-GRU                   | 0.999 ± 0.481 | 1.626 ± 1.093   | 2.070 ± 1.261   |       2.59 | 0 / 6                |
| Temperature | Causal (deployable) | Seasonal-naïve            | 1.686 ± 1.050 | 2.716 ± 1.780   | 3.300 ± 2.085   |       4.03 | 0 / 16               |
| Temperature | Perfect foresight   | Chronos-Bolt (zero-shot)  | 0.792 ± 0.282 | 1.303 ± 0.717   | 1.819 ± 1.040   |       2.9  | —                    |
| Temperature | Perfect foresight   | Chronos-Bolt + covariates | 0.718 ± 0.278 | 1.228 ± 0.795   | 1.559 ± 1.012   |       2.41 | 1 / 2                |
| Temperature | Perfect foresight   | LightGBM (specialist)     | 0.533 ± 0.208 | 0.888 ± 0.550   | 1.153 ± 0.730   |       1.28 | 6 / 0                |
| Temperature | Perfect foresight   | NAS-GRU                   | 0.999 ± 0.481 | 1.626 ± 1.093   | 2.070 ± 1.261   |       3.59 | 0 / 6                |
| Temperature | Perfect foresight   | Seasonal-naïve            | 1.686 ± 1.050 | 2.716 ± 1.780   | 3.300 ± 2.085   |       4.83 | 0 / 16               |