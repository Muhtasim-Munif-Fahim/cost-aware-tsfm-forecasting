# Reference verification handoff

Date: 2026-08-01

## Scope and source of truth

- The authoritative editable bibliography is `MANUSCRIPT/refs.bib`.
- The upload artifact is the generated single-file `SUBMISSION/manuscript.tex`.
- `SUBMISSION/manuscript.tex` contains 59 embedded `\\bibitem` records. The source `refs.bib` contains 64 records; five are uncited reserve records and are not part of this manuscript's bibliography.

## Verification method

Five disjoint reference batches (1--12, 13--24, 25--36, 37--47, and 48--59) were checked with Consensus search/fetch, followed by a supervising reconciliation pass. Official publisher, proceedings, repository, API, and software-release pages were used where a Consensus record was not the authoritative source. The 59 cited entries were checked for title, author/category, venue, year, pages, DOI/URL, and preprint/software status.

## Material decisions and corrections

- `fahim2026greennas`: restored the final QPAIN 2026 title, full author list, pages 1--6, IEEE venue, DOI `10.1109/QPAIN69676.2026.11545925`, and the arXiv URL.
- `openaq`: reclassified the record as the live OpenAQ API v3 documentation, protected the `API` acronym in the rendered title, and added the access date; it is not a journal article.
- `zippenfenig2023openmeteo`: reclassified the record as computer software/API and versioned it to Open-Meteo 1.4.0 with Zenodo DOI `10.5281/zenodo.14582479`.
- `autogluon2024chronosbolt`: reclassified Chronos-Bolt as a software/model release, retained the exact `amazon/chronos-bolt-small` checkpoint, and linked the official Chronos repository.
- `codecarbon`: replaced the stale early Zenodo record with the observed local software release `v3.2.8`, Zenodo DOI `10.5281/zenodo.20584960`, and the matching GitHub tag.
- `banbury2020benchmarking`: classified the work as the MLSys 2020 workshop proceedings paper and retained the arXiv/ePrint identifier.
- `lacoste2019quantifying`: labelled the record as a non-archival NeurIPS 2019 climate-ML workshop paper rather than a journal article.
- Completed metadata for `das2024timesfm`, `woo2024moirai`, `goswami2024moment`, `kong2025deep`, `ansari2024chronos`, `rasul2023lagllama`, `garza2023timegpt`, `aksu2024gifteval`, `karaouli2025foundational`, `elsken2019nas`, `patterson2021carbon`, `stankeviciute2021conformal`, `xu2021conformal`, `shafer2008conformal`, `elsayed2021really`, and `who2021guidelines`.
- Protected categorical/acronym text for `schwartz2020green` (`Green AI`) and `deb2002nsga2` (`NSGA-II`).
- The introduction's two broad citation claims were narrowed from universal “strongest/dominant” wording to claims supported by the cited literature.

The remaining cited keys were verified without substantive bibliographic correction: `cohen2017estimates`, `snyder2013changing`, `pinder2019opportunities`, `rasp2018neural`, `lam2023graphcast`, `bi2023pangu`, `ke2017lightgbm`, `makridakis2022m5`, `zheng2015forecasting`, `qi2019hybrid`, `tao2019air`, `chang2020lstm`, `salinas2020deepar`, `oreshkin2020nbeats`, `zhou2021informer`, `zoph2017neural`, `liu2018darts`, `oreshkin2021meta`, `liang2024foundation`, `makridakis2018statistical`, `schwartz2020green`, `strubell2019energy`, `rolnick2022tackling`, `abadade2023tinyml`, `hewamalage2023forecast`, `bergmeir2012use`, `zhang2017cautionary`, `garciamartin2019estimation`, `hersbach2020era5`, `hyndman2006another`, `diebold1995comparing`, `harvey1997testing`, `demsar2006statistical`, `vovk2005algorithmic`, `shafer2008conformal`, and `angelopoulos2023conformal`.

## Provenance caveat before submission -- RESOLVED 2026-08-26

**Resolved.** The CodeCarbon version used by the campaign is now established from the
install timestamp rather than inference: the `codecarbon-3.2.8.dist-info` INSTALLER and
RECORD files are stamped `2026-07-13 20:18`, and the canonical campaign runconfig is
stamped `2026-07-14T10:09:37Z` -- roughly fourteen hours later, with no reinstall since
(the dist-info directory has not been rewritten). The campaign therefore ran CodeCarbon
**3.2.8**, and the manuscript's citation of that release is correct as it stands. No bib
entry needs changing.

`src/run_forecast.py` now records `codecarbon` (and `timesfm`) in every future
`_runconfig.json`, so this cannot recur.

Original caveat, retained for the record:



The run manifests record `chronos-forecasting` 2.3.1 but do not record the CodeCarbon version. The local environment inspected during this audit reports CodeCarbon 3.2.8, so the manuscript currently cites the observed 3.2.8 release. Confirm the frozen campaign environment or package lock before upload; if the campaign used a different CodeCarbon version, update that one entry and regenerate the flat file.

## Verification completed

- `analysis/number_audit.py`: 152/152 checks passed.
- `paper/latex/md2tex.py`, main LaTeX x3 with BibTeX, supplement x2, generated `MANUSCRIPT` manuscript/supplement x2 each, and standalone `SUBMISSION/manuscript.tex` x2 completed successfully.
- Final standalone log: no undefined citations, undefined references, LaTeX errors, or emergency stops.
- Remaining log messages are non-fatal layout warnings: two underfull boxes, one underfull vbox, and one overfull methods URL line.
- The repository's Bash build wrapper could not be invoked because Bash is not installed on this Windows host; the equivalent PowerShell-native stages were run.

## Next step

Review the final PDF visually and confirm the CodeCarbon provenance caveat. The corrected upload source is `SUBMISSION/manuscript.tex`; the tracked embedded source is `MANUSCRIPT/manuscript.tex`.
