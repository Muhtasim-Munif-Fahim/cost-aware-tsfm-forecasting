# Back matter

## Data availability

The datasets analysed during the current study are available in the OpenAQ
repository, https://openaq.org (hourly PM2.5; sensor IDs for all 29 cities are listed
in `cities_manifest.csv` and Supplementary Table S2), the Open-Meteo historical
archive, https://open-meteo.com (temperature and meteorological covariates), and the
UCI Machine Learning Repository,
https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data (Beijing
multi-site air-quality data). The scripts that reproduce the exact extracts used in
this study, together with SHA-256 manifests of every input file, are available in the
code repository referenced in the Code availability statement below.

## Code availability

The full harness (`run_forecast.py`, `e4_transfer.py`, analysis scripts), the locked
analysis plan with deviations log, the per-claim results ledger, and figure/table
generation code are archived at Zenodo under DOI
[10.5281/zenodo.22118240](https://doi.org/10.5281/zenodo.22118240), and are also available at
https://github.com/Muhtasim-Munif-Fahim/cost-aware-tsfm-forecasting under the tagged
release `v2.0-revision` used for the revised manuscript (the earlier tag
`v1.0-campaign` corresponds to the originally submitted version).

<!-- ANONYMIZATION: main.tex swaps this whole statement on \ifanon. The review build
ships the code as Supplementary Software instead of citing a URL (built by
paper/latex/make_code_si.py -> SUBMISSION/Supplementary_Software.zip: 21 scripts in
methods/ and analysis/, plus requirements.txt, README and LICENSE; 24 entries, ~59 kB,
well under the 50 MB SI cap).

The bundle is referee-facing, NOT a copy of the repo. It EXCLUDES results/ (every
*_runconfig.json carries a "hostname" field naming the study machine), the figure and
table generation code, number_audit.py, the LaTeX build scripts, ANALYSIS_PLAN.md and
RESULTS_LEDGER.md. It is also sanitized at build time: module docstrings are replaced
from an authored table and internal-process language is stripped, because the repo
source carries project history ("Reviewer-requested", deviations-log references,
"Phase-N", dated internal decisions) that a journal referee must not read. Three gates
hard-fail the build if anything slips. Keep this statement and the bundle in sync --
if the shipping set changes, this paragraph changes.

The named GitHub statement above is restored for the camera-ready by setting \anonfalse.

ACTION REQUIRED BY THE AUTHORS: set the GitHub repository to PRIVATE for the duration
of review, and public again on acceptance. It is currently live and tagged, so a
reviewer who searches the title would otherwise find it -- and the "will be made
publicly available on publication" wording would not be accurate. -->

<!-- ANONYMIZATION: Acknowledgements, Author contributions and Competing interests
below are suppressed in the \ifanon build and entered in the submission system
instead, per Springer Nature double-anonymous guidance. -->


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
