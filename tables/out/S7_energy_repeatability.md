| City    | Tier                      |   Reps | Mean (J/1k)   | SD (J/1k)   |   SD/mean | $>$20\% gate   |
|:--------|:--------------------------|-------:|:--------------|:------------|----------:|:---------------|
| Beijing | Chronos-Bolt (zero-shot)  |      5 | 1,229         | 240.9       |     0.196 | False          |
| Beijing | Chronos-Bolt + covariates |      5 | 763           | 102.3       |     0.134 | False          |
| Beijing | LightGBM (specialist)     |      5 | 15,061        | 2,373.7     |     0.158 | False          |
| Beijing | NAS-GRU                   |      5 | 13,237        | 2,690.1     |     0.203 | True           |
| Beijing | Seasonal-naïve            |      5 | 2             | 0.4         |     0.2   | False          |
| Nairobi | Chronos-Bolt (zero-shot)  |      5 | 1,946         | 1,940.3     |     0.997 | True           |
| Nairobi | Chronos-Bolt + covariates |      5 | 842           | 108.9       |     0.129 | False          |
| Nairobi | LightGBM (specialist)     |      5 | 9,461         | 835.8       |     0.088 | False          |
| Nairobi | NAS-GRU                   |      5 | 7,298         | 104.6       |     0.014 | False          |
| Nairobi | Seasonal-naïve            |      5 | 16            | 19.3        |     1.203 | True           |
| Seoul   | Chronos-Bolt (zero-shot)  |      5 | 1,025         | 141.3       |     0.138 | False          |
| Seoul   | Chronos-Bolt + covariates |      5 | 781           | 94.2        |     0.12  | False          |
| Seoul   | LightGBM (specialist)     |      5 | 10,832        | 406.1       |     0.037 | False          |
| Seoul   | NAS-GRU                   |      5 | 7,496         | 440.4       |     0.059 | False          |
| Seoul   | Seasonal-naïve            |      5 | 9             | 15.2        |     1.684 | True           |