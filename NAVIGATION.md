# Where things live

Quick map for finding files fast. Full layout + reproduce steps: [README.md](README.md).

## I want to read the paper
| File | What |
|------|------|
| **[MANUSCRIPT/manuscript.pdf](MANUSCRIPT/manuscript.pdf)** | The manuscript (built PDF) |
| **[MANUSCRIPT/supplementary.pdf](MANUSCRIPT/supplementary.pdf)** | Supplementary tables + figure |

> The LaTeX source + journal template are not published here; the PDFs above are the
> readable deliverables. LaTeX source is available from the authors on request.

## I want to edit the paper
Edit prose **only** in `paper/sections/*.md` (canonical, audited). The LaTeX layer
(`.tex`/`.cls`/`.bib`) is generated locally by the `paper/latex/` build scripts and is not
tracked in this repo.

| File | What |
|------|------|
| `paper/sections/*.md` | Canonical prose (title/abstract, intro, results, discussion, methods, backmatter) |
| `paper/latex/md2tex.py`, `make_submission.py`, `build.sh` | Local build pipeline: `sections/*.md` → LaTeX → `MANUSCRIPT/*.pdf` (audit-gated) |
| `paper/RESULTS_LEDGER.md` | Every quantitative claim → artifact + exact command (L-###) |
| `paper/ANALYSIS_PLAN.md` | Pre-specified plan + deviations log |

## I want to run the code
Run everything **from the repo root**, e.g. `python src/run_forecast.py ...`.

| Path | What |
|------|------|
| `src/run_forecast.py` | Main forecasting harness (single / sweep / regime / cities) |
| `src/e4_transfer.py` | E4 transfer-vs-zero-shot crux experiment |
| `src/*_fetch.py`, `src/city_select.py` | Data acquisition (OpenAQ + Open-Meteo) |
| `analysis/` | Post-hoc analyses + `number_audit.py` (the claim-verification gate) |
| `figures/` | Figure scripts → `figures/main/F*.{pdf,png}` |
| `tables/` | `make_tables.py` → `tables/out/*.{tex,md}` |
| `results/v1/` | Canonical result artifacts (the reproducibility record) |

## Regenerate outputs
```bash
python analysis/number_audit.py     # claim gate — must pass (currently 152/152)
python tables/make_tables.py        # -> tables/out/
python figures/fig*.py              # -> figures/main/
bash   paper/latex/build.sh         # local only -> MANUSCRIPT/{manuscript,supplementary}.pdf
```
