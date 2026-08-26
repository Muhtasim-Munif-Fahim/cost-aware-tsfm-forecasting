#!/usr/bin/env python
"""Render the point-by-point response letter to PDF and DOCX.

    in : Revision/RESPONSE_TO_REVIEWERS.md
    out: MANUSCRIPT/response_to_reviewers.pdf
         MANUSCRIPT/response_to_reviewers.docx

Scientific Reports accepts the response as PDF; the DOCX travels alongside it because
editors routinely want an editable copy they can comment in.

Deliberately does NOT overwrite MANUSCRIPT/response_letter.pdf, which is the letter
from the ORIGINAL submission (2026-07-31) and is still wanted as a record.

Reviewer quotations are set apart visually: pandoc renders markdown blockquotes as
`quote`, which by default is only indented, and an editor skimming the letter should be
able to tell the reviewer's words from ours at a glance. The header below gives them a
rule and grey italics.

Run after the DOI is inserted, so the [PENDING] markers are gone; the script refuses to
build otherwise unless --allow-pending is passed.

Usage: python paper/latex/make_response_letter.py [--allow-pending] [--keep-build]
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "Revision" / "RESPONSE_TO_REVIEWERS.md"
OUT_PDF = ROOT / "MANUSCRIPT" / "response_to_reviewers.pdf"
OUT_DOCX = ROOT / "MANUSCRIPT" / "response_to_reviewers.docx"
BUILD = ROOT / "paper" / "latex" / "_response_build"

HEADER = r"""
\usepackage{mdframed}
\usepackage{xcolor}
\definecolor{quoterule}{gray}{0.55}
\definecolor{quotetext}{gray}{0.30}

% Reviewer quotations: a left rule plus grey italics, so the reviewer's words are
% never mistaken for ours.
\let\oldquote\quote
\let\endoldquote\endquote
\renewenvironment{quote}{%
  \begin{mdframed}[leftline=true,rightline=false,topline=false,bottomline=false,
                   linewidth=2pt,linecolor=quoterule,
                   innerleftmargin=10pt,innerrightmargin=0pt,
                   innertopmargin=4pt,innerbottommargin=4pt,
                   skipabove=8pt,skipbelow=8pt]%
  \itshape\color{quotetext}%
}{\end{mdframed}}

% Keep a reviewer heading with the comment that follows it.
\usepackage{needspace}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-pending", action="store_true",
                    help="build even though the Zenodo DOI is still a placeholder")
    ap.add_argument("--keep-build", action="store_true")
    args = ap.parse_args()

    if not SRC.exists():
        sys.exit(f"missing {SRC.relative_to(ROOT)}")
    text = SRC.read_text(encoding="utf-8")

    pending = [
        f"  line {i}: {ln.strip()[:88]}"
        for i, ln in enumerate(text.splitlines(), 1)
        if "[PENDING]" in ln or "TO BE INSERTED" in ln or "ZENODO-DOI-PLACEHOLDER" in ln
    ]
    if pending and not args.allow_pending:
        sys.exit(
            "the response letter still carries DOI placeholders:\n"
            + "\n".join(pending)
            + "\n\nMint the Zenodo DOI and run paper/latex/insert_zenodo_doi.py first,\n"
              "or pass --allow-pending to build a draft anyway."
        )
    if pending:
        print(f"warning: building with {len(pending)} unresolved placeholder line(s)")

    BUILD.mkdir(parents=True, exist_ok=True)
    header = BUILD / "header.tex"
    header.write_text(HEADER, encoding="utf-8")

    common = [
        "pandoc", str(SRC),
        "--from=markdown+pipe_tables+yaml_metadata_block",
        "--syntax-highlighting=tango",
    ]

    def run(args_, what):
        proc = subprocess.run(common + args_, capture_output=True, text=True, cwd=BUILD)
        if proc.returncode != 0:
            sys.exit(f"pandoc ({what}) failed (rc={proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
        for line in (proc.stderr or "").splitlines():
            if line.strip() and "MiKTeX updates" not in line:
                print(f"  pandoc ({what}): {line.strip()}")

    run([
        "-o", str(BUILD / "response.pdf"),
        "--pdf-engine=pdflatex",
        "-V", "geometry:margin=1in",
        "-V", "fontsize=11pt",
        "-V", "colorlinks=true",
        "-V", "linkcolor=black",
        "-V", "urlcolor=blue",
        "-V", "documentclass=article",
        "-H", str(header),
    ], "pdf")

    # No -H header for docx: that LaTeX preamble means nothing to the docx writer, which
    # styles blockquotes through its own "Block Text" style instead.
    run(["-o", str(BUILD / "response.docx")], "docx")

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    for src, dst in ((BUILD / "response.pdf", OUT_PDF), (BUILD / "response.docx", OUT_DOCX)):
        if not src.exists():
            sys.exit(f"pandoc reported success but produced no {src.suffix} file")
        shutil.copy(src, dst)

    pages = "?"
    try:
        counts = [int(m) for m in re.findall(rb"/Count\s+(\d+)", OUT_PDF.read_bytes())]
        if counts:
            pages = max(counts)
    except Exception:
        pass
    print(f"wrote {OUT_PDF.relative_to(ROOT).as_posix()} "
          f"({pages} pages, {len(text.split()):,} words)")
    print(f"wrote {OUT_DOCX.relative_to(ROOT).as_posix()} "
          f"({OUT_DOCX.stat().st_size / 1024:.0f} kB)")

    if not args.keep_build:
        shutil.rmtree(BUILD, ignore_errors=True)
    else:
        print(f"kept build dir: {BUILD.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()

