#!/usr/bin/env python
"""Assemble the folder that gets uploaded to Scientific Reports.

Everything here is build output: run build.sh, make_markedup.py and
make_response_letter.py first, then this. Nothing is authored in the package folder,
so it is safe to delete and regenerate at any time.

    python paper/latex/make_revision_package.py [--out Revision_Package]

Layout is numbered in upload order, because the journal's form takes one file at a
time and the ordering is otherwise easy to get wrong:

    01_Manuscript/          the revised manuscript, PDF + single-file LaTeX + bib
    02_Marked_Up/           the tracked-changes copy (goes under "related files":
                            Scientific Reports forbids markup in the manuscript itself)
    03_Response/            point-by-point response, DOCX and PDF
    04_Figures/             F2-F6 as PDF (vector, for production) and PNG (for review).
                            Figure 1 is native TikZ inside manuscript.tex, so it has no
                            standalone file; that is expected, not a missing figure.
    05_LaTeX_Support/       class/style/bst that manuscript.tex needs to compile
    06_Supplementary_Software/  the referee-facing code bundle

A staleness check runs first: if any source is newer than the artifact built from it,
the script refuses rather than shipping a mixed-vintage package.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# (destination subfolder, source path, required?)
ITEMS = [
    ("01_Manuscript", "MANUSCRIPT/manuscript.pdf", True),
    ("01_Manuscript", "MANUSCRIPT/manuscript.tex", True),
    ("01_Manuscript", "MANUSCRIPT/supplementary.pdf", True),
    ("01_Manuscript", "MANUSCRIPT/supplementary.tex", True),
    ("01_Manuscript", "MANUSCRIPT/refs.bib", False),
    ("02_Marked_Up", "MANUSCRIPT/manuscript_marked.pdf", True),
    ("03_Response", "MANUSCRIPT/response_to_reviewers.docx", True),
    ("03_Response", "MANUSCRIPT/response_to_reviewers.pdf", True),
    ("05_LaTeX_Support", "SUBMISSION/wlscirep.cls", True),
    ("05_LaTeX_Support", "SUBMISSION/naturemag-doi.bst", True),
    ("05_LaTeX_Support", "SUBMISSION/jabbrv.sty", True),
    ("05_LaTeX_Support", "SUBMISSION/jabbrv-ltwa-en.ldf", True),
    ("05_LaTeX_Support", "SUBMISSION/jabbrv-ltwa-all.ldf", True),
    ("06_Supplementary_Software", "SUBMISSION/Supplementary_Software.zip", True),
]

FIGURES = ["F2_panel_advantage", "F3_foresight_ablation", "F4_beijing12",
           "F5_decision_maps", "F6_covariate_quality"]

# artifact -> sources it must not be older than
STALENESS = {
    "MANUSCRIPT/manuscript.pdf": ["paper/sections", "paper/latex/main.tex",
                                  "paper/latex/floats"],
    "MANUSCRIPT/manuscript_marked.pdf": ["MANUSCRIPT/manuscript.tex"],
    "MANUSCRIPT/response_to_reviewers.docx": ["Revision/RESPONSE_TO_REVIEWERS.md"],
}

README = """Scientific Reports revision package
Manuscript: Cost-Aware Evaluation of Time-Series Foundation Models for Urban
            Air-Quality and Temperature Forecasting
Submission ID: 02ecdd52-141e-481a-8b7d-401ed7f5af0d

01_Manuscript
    manuscript.pdf / .tex     revised manuscript. The .tex is flattened to a single
                              file with the bibliography embedded, which is the form
                              the journal asks for. refs.bib is included only for the
                              separate-bibliography option.
    supplementary.pdf / .tex  Supplementary Information.

02_Marked_Up
    manuscript_marked.pdf     all changes against the submitted version, additions
                              underlined in blue and deletions struck through in red.
                              Upload under "related files": tracked changes are not
                              permitted inside the manuscript file itself.

03_Response
    response_to_reviewers     point-by-point response, addressing R1.1-R1.4, R2.1-R2.2
                              and R3 individually. DOCX and PDF are the same document.

04_Figures
    F2-F6, PDF (vector, for production) and PNG (for on-screen review).
    There is no F1 file: Figure 1 is drawn in TikZ inside manuscript.tex.

05_LaTeX_Support
    Class, bibliography style and the jabbrv journal-abbreviation package that
    manuscript.tex needs in order to compile.

06_Supplementary_Software
    Referee-facing code bundle. Not a copy of the full repository: it excludes results
    (whose run manifests name the study machine), figure/table generation and the
    build scripts.
"""


def newest(path: Path) -> float:
    """mtime of a file, or of the newest file under a directory."""
    if path.is_dir():
        times = [p.stat().st_mtime for p in path.rglob("*") if p.is_file()]
        return max(times) if times else 0.0
    return path.stat().st_mtime if path.exists() else 0.0


def check_staleness() -> list[str]:
    problems = []
    for artifact, sources in STALENESS.items():
        a = ROOT / artifact
        if not a.exists():
            continue
        for s in sources:
            sp = ROOT / s
            if sp.exists() and newest(sp) > a.stat().st_mtime:
                problems.append(f"  {artifact} is older than {s}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="Revision_Package")
    ap.add_argument("--force", action="store_true",
                    help="package anyway despite stale artifacts")
    args = ap.parse_args()

    stale = check_staleness()
    if stale and not args.force:
        sys.exit(
            "refusing to package: these artifacts are older than their sources\n"
            + "\n".join(stale)
            + "\n\nRebuild first:\n"
              "  bash paper/latex/build.sh\n"
              "  python paper/latex/make_markedup.py\n"
              "  python paper/latex/make_response_letter.py\n"
              "\nor pass --force."
        )
    if stale:
        print("WARNING packaging stale artifacts:\n" + "\n".join(stale) + "\n")

    out = ROOT / args.out
    if out.exists():
        shutil.rmtree(out)

    copied, missing = 0, []
    for sub, rel, required in ITEMS:
        src = ROOT / rel
        if not src.exists():
            (missing if required else []).append(rel)
            continue
        dst_dir = out / sub
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst_dir / src.name)
        copied += 1

    figdir = out / "04_Figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for stem in FIGURES:
        for ext in ("pdf", "png"):
            src = ROOT / "figures" / "main" / f"{stem}.{ext}"
            if src.exists():
                shutil.copy(src, figdir / src.name)
                copied += 1
            elif ext == "pdf":
                missing.append(f"figures/main/{stem}.pdf")

    if missing:
        shutil.rmtree(out, ignore_errors=True)
        sys.exit("missing required files:\n" + "\n".join(f"  {m}" for m in missing))

    (out / "README.txt").write_text(README, encoding="utf-8")

    total = 0
    print(f"{args.out}/")
    for d in sorted(p for p in out.iterdir() if p.is_dir()):
        print(f"  {d.name}/")
        for f in sorted(d.iterdir()):
            kb = f.stat().st_size / 1024
            total += f.stat().st_size
            print(f"    {f.name:<38} {kb:8.1f} kB")
    print(f"  README.txt")
    print(f"\n{copied + 1} files, {total / 1048576:.1f} MB")

    letter = ROOT / "MANUSCRIPT" / "response_to_reviewers.docx"
    if letter.exists():
        import zipfile, re as _re
        try:
            with zipfile.ZipFile(letter) as z:
                body = _re.sub(r"<[^>]+>", "", z.read("word/document.xml").decode("utf-8"))
            for marker in ("PENDING", "TO BE INSERTED", "ZENODO-DOI-PLACEHOLDER"):
                if marker in body:
                    print(f"\nWARNING: the response letter still contains '{marker}'. "
                          "Mint the DOI, run insert_zenodo_doi.py, and rebuild before uploading.")
                    break
        except Exception:
            pass


if __name__ == "__main__":
    main()
