| Data tier   | Model                     |   Coverage (causal) |   Width (causal) |   Coverage (perfect) |   Width (perfect) |
|:------------|:--------------------------|--------------------:|-----------------:|---------------------:|------------------:|
| rich        | Chronos-Bolt (zero-shot)  |               0.929 |            18.47 |                0.929 |             18.47 |
| rich        | Chronos-Bolt + covariates |               0.91  |            17.8  |                0.936 |             19.48 |
| rich        | LightGBM (specialist)     |               0.935 |            21.05 |                0.926 |             17.69 |
| rich        | NAS-GRU                   |               0.914 |            18.78 |                0.914 |             18.78 |
| rich        | Seasonal-naïve            |               0.97  |            35.07 |                0.97  |             35.07 |
| scarce      | Chronos-Bolt (zero-shot)  |               0.931 |           108.64 |                0.931 |            108.64 |
| scarce      | Chronos-Bolt + covariates |               0.928 |           123.94 |                0.931 |            112.69 |
| scarce      | LightGBM (specialist)     |               0.913 |           105.83 |                0.916 |            102.17 |
| scarce      | NAS-GRU                   |               0.931 |           107.96 |                0.931 |            107.96 |
| scarce      | Seasonal-naïve            |               0.949 |           163.85 |                0.949 |            163.85 |