| Model pair                                            |   Cities |   Sig. (raw) |   A wins (raw) |   B wins (raw) |   A wins (FDR) |   B wins (FDR) |   Median $P$ |
|:------------------------------------------------------|---------:|-------------:|---------------:|---------------:|---------------:|---------------:|-------------:|
| Chronos-Bolt (zero-shot) vs Chronos-Bolt + covariates |       29 |            5 |              2 |              3 |              2 |              1 |        0.37  |
| Chronos-Bolt (zero-shot) vs NAS-GRU                   |       29 |            9 |              8 |              1 |              6 |              0 |        0.107 |
| Chronos-Bolt + covariates vs NAS-GRU                  |       29 |           14 |             12 |              2 |             12 |              2 |        0.051 |
| LightGBM (specialist) vs Chronos-Bolt (zero-shot)     |       29 |           11 |             11 |              0 |              6 |              0 |        0.167 |
| LightGBM (specialist) vs Chronos-Bolt + covariates    |       29 |            9 |              9 |              0 |              6 |              0 |        0.153 |
| LightGBM (specialist) vs NAS-GRU                      |       29 |           20 |             20 |              0 |             20 |              0 |        0.003 |
| Seasonal-naïve vs Chronos-Bolt (zero-shot)            |       29 |           20 |              0 |             20 |              0 |             16 |        0.017 |
| Seasonal-naïve vs Chronos-Bolt + covariates           |       29 |           21 |              0 |             21 |              0 |             20 |        0.01  |
| Seasonal-naïve vs LightGBM (specialist)               |       29 |           26 |              0 |             26 |              0 |             26 |        0.001 |
| Seasonal-naïve vs NAS-GRU                             |       29 |           16 |              2 |             14 |              2 |             13 |        0.022 |