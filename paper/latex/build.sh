#!/bin/bash
# Full manuscript build. Run from anywhere. Requires pandoc + MiKTeX on PATH.
#
# Pipeline:
#   1. audit gate (analysis/number_audit.py) -- must pass
#   2. md2tex.py: paper/sections/*.md -> generated/*.tex fragments (incl. abstract)
#   3. compile main + supplement here (paper/latex/) -- validates + produces main.bbl
#   4. make_submission.py: flatten \input + embed bibliography -> single-file sources
#   5. compile the single-file sources -> the submission deliverables in MANUSCRIPT/
#
# Deliverables (journal wants one .tex each): MANUSCRIPT/manuscript.{tex,pdf},
# MANUSCRIPT/supplementary.{tex,pdf}. refs.bib is kept there for the separate-bib option.
set -e
cd "$(dirname "$0")"                       # paper/latex/
ROOT="$(cd ../.. && pwd)"

python "$ROOT/analysis/number_audit.py"
python md2tex.py

# --- working build here: validates + generates main.bbl for embedding ---
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode supplement.tex
pdflatex -interaction=nonstopmode supplement.tex

# --- flatten to single-file submission sources + compile them ---
python make_submission.py                  # -> MANUSCRIPT/{manuscript,supplementary}.tex
# official SciRep class + supporting style/bst files travel with manuscript.tex
cp wlscirep.cls jabbrv.sty jabbrv-ltwa-all.ldf jabbrv-ltwa-en.ldf naturemag-doi.bst "$ROOT/MANUSCRIPT/"
cd "$ROOT/MANUSCRIPT"
# bibliography is embedded (no bibtex step needed for the flattened file)
pdflatex -interaction=nonstopmode manuscript.tex
pdflatex -interaction=nonstopmode manuscript.tex
pdflatex -interaction=nonstopmode supplementary.tex
pdflatex -interaction=nonstopmode supplementary.tex
rm -f ./*.aux ./*.log ./*.out ./*.bbl ./*.blg    # keep MANUSCRIPT/ clean (tex + pdf + bib + class only)

echo "BUILD OK -> MANUSCRIPT/manuscript.pdf, MANUSCRIPT/supplementary.pdf (single-file)"
