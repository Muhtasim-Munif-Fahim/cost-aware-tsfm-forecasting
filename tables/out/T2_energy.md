| Tier                      | City    | Measured (J/1k)   |   Cost (USD/1k) | Train (J/fit)   | Infer (J/fc)   | Crossover   | Params   |
|:--------------------------|:--------|:------------------|----------------:|:----------------|:---------------|:------------|:---------|
| Chronos-Bolt (zero-shot)  | Beijing | 1,229 ± 241       |         7.2e-05 | —               | 0.112          | —           | 47.72M   |
| Chronos-Bolt (zero-shot)  | Seoul   | 1,025 ± 141       |         6e-05   | —               | 0.110          | —           | ''       |
| Chronos-Bolt (zero-shot)  | Nairobi | 1,056–5,417       |         0.00011 | —               | 0.111          | —           | ''       |
| Chronos-Bolt + covariates | Beijing | 763 ± 102         |         4.5e-05 | —               | —              | —           | 47.72M   |
| Chronos-Bolt + covariates | Seoul   | 781 ± 94          |         4.6e-05 | —               | —              | —           | ''       |
| Chronos-Bolt + covariates | Nairobi | 842 ± 109         |         4.9e-05 | —               | —              | —           | ''       |
| LightGBM (specialist)     | Beijing | 15,061 ± 2,374    |         0.00088 | 317             | 0.033          | 4,041       | —        |
| LightGBM (specialist)     | Seoul   | 10,832 ± 406      |         0.00063 | 246             | 0.022          | 2,778       | ''       |
| LightGBM (specialist)     | Nairobi | 9,461 ± 836       |         0.00055 | 203             | 0.065          | 4,416       | ''       |
| NAS-GRU                   | Beijing | 9,367–16,788      |         0.00077 | —               | —              | —           | 157,080  |
| NAS-GRU                   | Seoul   | 7,496 ± 440       |         0.00044 | —               | —              | —           | ''       |
| NAS-GRU                   | Nairobi | 7,298 ± 105       |         0.00043 | —               | —              | —           | ''       |
| Seasonal-naïve            | Beijing | 2 ± 0             |         1.2e-07 | —               | —              | —           | —        |
| Seasonal-naïve            | Seoul   | 1–36              |         5.3e-07 | —               | —              | —           | ''       |
| Seasonal-naïve            | Nairobi | 1–40              |         9.3e-07 | —               | —              | —           | ''       |