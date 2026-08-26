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

# --- prose style gate: this manuscript uses no em dashes anywhere.
# The markdown sections were cleaned once, but prose written later on the LaTeX side
# (captions, back matter) reintroduced them, so the check runs over BOTH families.
# Skips comment lines and dash rulers; matches exactly three hyphens, and the unicode
# em dash in the markdown sources.
python - "$ROOT" <<'PYEOF'
import re, sys, pathlib
root = pathlib.Path(sys.argv[1])
files = sorted(root.glob("paper/sections/*.md")) + [
    root / "paper/latex/main.tex", root / "paper/latex/supplement.tex",
    *sorted((root / "paper/latex/floats").glob("*.tex"))]
ruler, tex_em, uni_em = re.compile(r"-{4,}"), re.compile(r"(?<!-)---(?!-)"), re.compile("—")
bad = []
for p in files:
    if not p.exists():
        continue
    pat = uni_em if p.suffix == ".md" else tex_em
    for i, line in enumerate(p.read_text(encoding="utf-8").split("\n"), 1):
        if line.lstrip().startswith("%") or ruler.search(line):
            continue
        if pat.search(line):
            bad.append(f"  {p.relative_to(root).as_posix()}:{i}: {line.strip()[:88]}")
if bad:
    print("EM DASH CHECK FAILED -- house style is zero em dashes:")
    print("\n".join(bad))
    raise SystemExit(1)
print("em dash check passed (0 in prose)")
PYEOF

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

# --- flat, zippable submission folder (main manuscript only; supplement ships as PDF) ---
cd "$ROOT/paper/latex"
python make_flat_submission.py
cd "$ROOT/SUBMISSION"
# proves the folder is self-contained: nothing outside it is on the search path
pdflatex -interaction=nonstopmode manuscript.tex
pdflatex -interaction=nonstopmode manuscript.tex
# NB: never `rm ./*.pdf` here -- the figures are PDFs. Only the compiled output goes.
rm -f ./*.aux ./*.log ./*.out ./manuscript.pdf
python "$ROOT/paper/latex/make_flat_submission.py" --zip-only

# --- anonymized Supplementary Software bundle (code only; own anonymity gate) ---
python "$ROOT/paper/latex/make_code_si.py"

# --- anonymity gate: fails the build if the double-anonymous switch is on but an
# --- identifying string survived into the deliverables.
#
# Deliberately NOT matching bare surnames: citing one's own prior work in the
# bibliography is required under double-anonymous review, so "Fahim"/"Karim" appear
# there legitimately. What must not appear is the affiliation, the corresponding
# email, the name-bearing repository URL, or any first-person self-citation.
cd "$ROOT"
if grep -q '^\\anontrue' paper/latex/main.tex; then
  echo "anon build: scanning deliverables for identifying strings..."
  if grep -rniE 'rajshahi|mrkarim@|github\.com/Muhtasim|our (IEEE|conference) paper|our own \\cite' \
       MANUSCRIPT/manuscript.tex MANUSCRIPT/supplementary.tex SUBMISSION/manuscript.tex ; then
    echo "ANONYMITY CHECK FAILED -- identifying string above survived into a deliverable" >&2
    exit 1
  fi
  echo "anonymity check passed"
fi

echo "BUILD OK -> MANUSCRIPT/manuscript.pdf, MANUSCRIPT/supplementary.pdf (single-file)"
echo "         -> SUBMISSION/ + SUBMISSION.zip (flat, compiles standalone)"
echo "         -> SUBMISSION/Supplementary_Software.zip (anonymized code bundle)"
