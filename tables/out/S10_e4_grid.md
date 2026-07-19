| Strategy                       | Nominal fraction (\%)   | MASE (mean ± sd)   | Actual training hours (mean)   |   Cities |
|:-------------------------------|:------------------------|:-------------------|:-------------------------------|---------:|
| Chronos-Bolt (zero-shot)       | —                       | 0.843 ± 0.354      | 0                              |       15 |
| LightGBM (refit on budget)     | 1                       | 0.941 ± 0.311      | 202                            |       15 |
| LightGBM (refit on budget)     | 10                      | 0.944 ± 0.383      | 599                            |       15 |
| LightGBM (refit on budget)     | 100                     | 0.858 ± 0.382      | 5,995                          |       15 |
| NAS-GRU (transfer + fine-tune) | 0                       | 0.899 ± 0.340      | 0                              |       15 |
| NAS-GRU (transfer + fine-tune) | 1                       | 0.915 ± 0.331      | 66                             |       15 |
| NAS-GRU (transfer + fine-tune) | 10                      | 0.888 ± 0.340      | 599                            |       15 |
| NAS-GRU (transfer + fine-tune) | 100                     | 0.876 ± 0.374      | 5,995                          |       15 |