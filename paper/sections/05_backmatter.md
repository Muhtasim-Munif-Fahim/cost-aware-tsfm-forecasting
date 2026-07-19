# Back matter

## Data availability

Hourly PM2.5 observations were obtained from the OpenAQ v3 API (openaq.org; sensor
IDs for all 29 cities are listed in `cities_manifest.csv` and Supplementary Table S1).
Temperature and meteorological covariates were obtained from the Open-Meteo historical
archive (open-meteo.com). The Beijing multi-site dataset is available from the UCI
Machine Learning Repository (Beijing Multi-Site Air-Quality Data). SHA-256 manifests of all input files used in the
canonical campaign are included in the code release; fetch scripts reproduce the exact
extracts. <!-- TODO: archive DOI (Zenodo) for the frozen data snapshot. -->

## Code availability

The full harness (`run_forecast.py`, `e4_transfer.py`, analysis scripts), the locked
analysis plan with deviations log, the per-claim results ledger, and figure/table
generation code are available at
https://github.com/Muhtasim-Munif-Fahim/cost-aware-tsfm-forecasting under the tagged
release `v1.0-campaign` used for the canonical campaign. <!-- TODO: Zenodo DOI -->

## Author contributions

M.M.M.F.: Conceptualization, Methodology, Software, Formal analysis, Investigation,
Data Curation, Visualization, Writing -- Original Draft. M.R.K.: Conceptualization,
Supervision, Writing -- Review & Editing. Both authors reviewed and approved the
final manuscript.

<!-- Note: this file's own backmatter (Acknowledgements/Author contributions/
Competing interests) is not piped through md2tex -- main.tex carries these
sections directly. Keep the two in sync by hand when either changes. -->

## Competing interests

The authors declare no competing interests.

## Acknowledgements

We thank OpenAQ and its contributing agencies and sensor operators for open access to
the air-quality measurements, and Open-Meteo for the historical weather archive that
provided the temperature targets and meteorological covariates. The authors received
no specific funding for this work.
