| Data tier   | Model                     |   Coverage (causal) |   Width (causal) |   Coverage (perfect) |   Width (perfect) |
|:------------|:--------------------------|--------------------:|-----------------:|---------------------:|------------------:|
| rich        | Chronos-Bolt (zero-shot)  |               0.9   |            10.72 |                0.9   |             10.72 |
| rich        | Chronos-Bolt + covariates |               0.923 |            23.91 |                0.892 |              9.17 |
| rich        | LightGBM (specialist)     |               0.923 |             9.56 |                0.958 |              7.7  |
| rich        | NAS-GRU                   |               0.91  |            11.75 |                0.91  |             11.75 |
| rich        | Seasonal-naïve            |               0.934 |            18.04 |                0.934 |             18.04 |
| scarce      | Chronos-Bolt (zero-shot)  |               0.936 |             5.9  |                0.936 |              5.9  |
| scarce      | Chronos-Bolt + covariates |               0.939 |            13.79 |                0.943 |              3.79 |
| scarce      | LightGBM (specialist)     |               0.944 |             4.38 |                0.949 |              2.68 |
| scarce      | NAS-GRU                   |               0.95  |             6.02 |                0.95  |              6.02 |
| scarce      | Seasonal-naïve            |               0.968 |             8.96 |                0.968 |              8.96 |