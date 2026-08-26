| Model pair                                            |   Cities |   Sig. (raw) |   A wins (raw) |   B wins (raw) |   A wins (FDR) |   B wins (FDR) |   Median $P$ |
|:------------------------------------------------------|---------:|-------------:|---------------:|---------------:|---------------:|---------------:|-------------:|
| Chronos-Bolt (zero-shot) vs Chronos-Bolt + covariates |       29 |            5 |              4 |              1 |              2 |              1 |        0.295 |
| Chronos-Bolt (zero-shot) vs NAS-GRU                   |       29 |            6 |              5 |              1 |              2 |              0 |        0.308 |
| Chronos-Bolt + covariates vs NAS-GRU                  |       29 |            6 |              3 |              3 |              2 |              1 |        0.27  |
| LightGBM (specialist) vs Chronos-Bolt (zero-shot)     |       29 |            3 |              2 |              1 |              0 |              0 |        0.429 |
| LightGBM (specialist) vs Chronos-Bolt + covariates    |       29 |            3 |              3 |              0 |              2 |              0 |        0.486 |
| LightGBM (specialist) vs NAS-GRU                      |       29 |            7 |              6 |              1 |              6 |              1 |        0.384 |
| Seasonal-naïve vs Chronos-Bolt (zero-shot)            |       29 |           18 |              0 |             18 |              0 |             13 |        0.029 |
| Seasonal-naïve vs Chronos-Bolt + covariates           |       29 |           12 |              0 |             12 |              0 |              9 |        0.08  |
| Seasonal-naïve vs LightGBM (specialist)               |       29 |           13 |              0 |             13 |              0 |              8 |        0.056 |
| Seasonal-naïve vs NAS-GRU                             |       29 |           15 |              0 |             15 |              0 |             11 |        0.047 |