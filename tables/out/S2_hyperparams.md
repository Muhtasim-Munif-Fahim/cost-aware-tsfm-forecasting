| Tier                      | Setting                                      | Value                                                                             |
|:--------------------------|:---------------------------------------------|:----------------------------------------------------------------------------------|
| Seasonal-naïve            | persistence lag                              | 168 h (same hour last week)                                                       |
| LightGBM (specialist)     | models                                       | one direct model per lead time (24), retrained per fold                           |
| LightGBM (specialist)     | n\_estimators / learning\_rate / num\_leaves | 250 / 0.05 / 31                                                                   |
| LightGBM (specialist)     | subsample / subsample\_freq                  | 0.8 / 1 (freq added after pre-campaign audit; bagging inert without it)           |
| LightGBM (specialist)     | random\_state                                | 42                                                                                |
| LightGBM (specialist)     | features                                     | lagged target, calendar, meteorological covariates                                |
| NAS-GRU                   | architecture                                 | Green-NAS-A: 2 stacked GRU layers, 128 units, direct multi-horizon head           |
| NAS-GRU                   | optimizer / loss                             | Adam, learning rate $10^{-3}$, MSE                                                |
| NAS-GRU                   | early stopping                               | patience 10, max 50 epochs, validation fraction 0.1                               |
| NAS-GRU                   | lookback / batch size                        | 24 h / 256                                                                        |
| NAS-GRU                   | seeds                                        | \{42, 43, 44, 45, 46\}; per-city seed-mean before any test                        |
| Chronos-Bolt (zero-shot)  | checkpoint                                   | amazon/chronos-bolt-small (47{,}718{,}016 parameters), no training or fine-tuning |
| Chronos-Bolt (zero-shot)  | context / point forecast                     | 672 h (four weeks) / predicted mean                                               |
| Chronos-Bolt + covariates | covariate model                              | ridge regression ($\alpha = 1.0$) on standardized calendar + weather features     |
| Chronos-Bolt + covariates | scheme                                       | FM forecasts the residual of the ridge fit; ridge horizon prediction added back   |