#!/usr/bin/env python
"""Build the marked-up ("tracked changes") copy for the Scientific Reports revision.

Scientific Reports forbids tracked changes inside the manuscript file itself, so the
marked-up version travels as a separate PDF under "related files". This builds it.

    old = Submission Files/manuscript.tex   the flattened source as submitted (2026-07-29)
    new = SUBMISSION/manuscript.tex         the flattened source build.sh just produced

Both are single-file flattened sources, which is what latexdiff wants: no \\input to
chase, bibliography already embedded. Run build.sh first so `new` is current.

Two wrinkles this script exists to handle:

1. `latexdiff` needs Algorithm::Diff, which the bundled MiKTeX perl does not have. The
   `latexdiff-so` ("standalone") variant carries that module inline, so we call it.

2. wlscirep.cls keeps `\\begin{abstract}` in the PREAMBLE, before `\\begin{document}`.
   latexdiff only marks up the body, so the revised abstract came through unmarked, and
   latexdiff's own \\DIFadd/\\DIFdel definitions land AFTER \\end{abstract} anyway. We
   therefore hoist a copy of those definitions above the abstract and word-diff the
   abstract ourselves. Both abstracts are pure prose (asserted below), so wrapping runs
   in \\DIFadd/\\DIFdel is safe; the assert fires if that ever stops being true.

The class also declares \\author twice, which sends latexdiff into line-diff mode for the
whole preamble; --replace-context2cmd drops `author` from the context2 list to avoid it.

Usage:  python paper/latex/make_markedup.py [--keep-build]
Output: MANUSCRIPT/manuscript_marked.pdf
"""

from __future__ import annotations

import argparse
import difflib
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "Submission Files" / "manuscript.tex"
NEW = ROOT / "SUBMISSION" / "manuscript.tex"
OUT = ROOT / "MANUSCRIPT" / "manuscript_marked.pdf"
BUILD = ROOT / "paper" / "latex" / "_markedup_build"

# Support files the flattened source needs alongside it, all shipped in SUBMISSION/.
SUPPORT_GLOBS = ("*.pdf", "*.cls", "*.bst", "*.sty", "*.ldf")

ABS_OPEN, ABS_CLOSE = r"\begin{abstract}", r"\end{abstract}"

# Hoisted above the abstract so markup inside it resolves. \providecommand means the
# copy latexdiff inserts later is a no-op, so the two cannot disagree.
ABSTRACT_MARKUP_SUPPORT = r"""
%DIF ABSTRACT MARKUP SUPPORT (added by make_markedup.py)
%DIF wlscirep keeps the abstract in the preamble, ahead of latexdiff's own definitions.
\RequirePackage[normalem]{ulem}
\RequirePackage{color}\definecolor{RED}{rgb}{1,0,0}\definecolor{BLUE}{rgb}{0,0,1}
\providecommand{\DIFadd}[1]{{\protect\color{blue}\uwave{#1}}}
\providecommand{\DIFdel}[1]{{\protect\color{red}\sout{#1}}}
%DIF END ABSTRACT MARKUP SUPPORT
"""


def find_latexdiff_so() -> list[str]:
    """Locate latexdiff-so and return the argv prefix that runs it.

    Prefer the perl script over MiKTeX's .exe launcher: the launcher is a binary stub,
    so handing it to perl makes perl try to parse the PE header. When only the launcher
    is on PATH we invoke it directly instead.
    """
    candidates: list[Path] = []
    plain = shutil.which("latexdiff")
    if plain:
        candidates.append(Path(plain).with_name("latexdiff-so"))
    candidates.append(Path("J:/840/Latex/scripts/latexdiff/latexdiff-so"))
    for cand in candidates:
        if cand.exists() and cand.suffix.lower() != ".exe":
            return ["perl", str(cand)]

    launcher = shutil.which("latexdiff-so")
    if launcher:
        return [launcher] if launcher.lower().endswith(".exe") else ["perl", launcher]

    sys.exit(
        "latexdiff-so not found. It ships with MiKTeX under scripts/latexdiff/.\n"
        "Plain latexdiff is not enough here: it needs the Algorithm::Diff perl module,\n"
        "which this MiKTeX perl does not carry. latexdiff-so bundles it inline."
    )


def extract_abstract(tex: str, label: str) -> str:
    """Return the abstract body, asserting it is prose we can safely mark up."""
    try:
        i = tex.index(ABS_OPEN) + len(ABS_OPEN)
        j = tex.index(ABS_CLOSE)
    except ValueError:
        sys.exit(f"{label}: could not locate the abstract environment")
    body = tex[i:j]
    cmds = set(re.findall(r"\\[a-zA-Z]+", body))
    if cmds or "$" in body:
        sys.exit(
            f"{label}: abstract is no longer pure prose (commands={sorted(cmds)}, "
            f"math={'$' in body}). Word-level markup could break the build; extend "
            "this script deliberately rather than trusting it."
        )
    return body


def word_diff(old: str, new: str) -> str:
    """Word-level diff of two prose blocks, marked with \\DIFadd / \\DIFdel."""
    o, n = old.split(), new.split()
    out: list[str] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, o, n, autojunk=False).get_opcodes():
        if tag in ("delete", "replace"):
            out.append(r"\DIFdel{" + " ".join(o[i1:i2]) + "}")
        if tag in ("insert", "replace"):
            out.append(r"\DIFadd{" + " ".join(n[j1:j2]) + "}")
        if tag == "equal":
            out.extend(o[i1:i2])
    # Rewrap so the emitted source stays diffable by eye, like the rest of the file.
    text, line, lines = " ".join(out), "", []
    for word in text.split():
        if line and len(line) + 1 + len(word) > 92:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    lines.append(line)
    return "\n" + "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--keep-build",
        action="store_true",
        help="keep the scratch build directory (diff .tex, logs) for inspection",
    )
    args = ap.parse_args()

    for p in (OLD, NEW):
        if not p.exists():
            sys.exit(f"missing {p.relative_to(ROOT)}; run paper/latex/build.sh first")
    if NEW.stat().st_mtime < OLD.stat().st_mtime:
        print("warning: SUBMISSION/manuscript.tex is older than the submitted copy")

    latexdiff_cmd = find_latexdiff_so()
    BUILD.mkdir(parents=True, exist_ok=True)
    shutil.copy(OLD, BUILD / "old.tex")
    shutil.copy(NEW, BUILD / "new.tex")

    print(f"latexdiff-so: {' '.join(latexdiff_cmd)}")
    proc = subprocess.run(
        [
            *latexdiff_cmd,
            "--type=UNDERLINE",
            # drop `author` from the context2 list; wlscirep declares it twice, which
            # otherwise forces line-diff mode across the entire preamble.
            "--replace-context2cmd=title,date,institute",
            "old.tex",
            "new.tex",
        ],
        cwd=BUILD,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        sys.exit(f"latexdiff failed (rc={proc.returncode}):\n{proc.stderr}")
    for line in proc.stderr.splitlines():
        if line.strip() and "MiKTeX updates" not in line:
            print(f"  latexdiff: {line.strip()}")

    diff = proc.stdout

    # --- graft the abstract markup latexdiff cannot produce ---
    old_abs = extract_abstract((BUILD / "old.tex").read_text(encoding="utf-8"), "old")
    new_abs = extract_abstract((BUILD / "new.tex").read_text(encoding="utf-8"), "new")
    if old_abs.split() == new_abs.split():
        print("abstract: unchanged, nothing to mark")
    else:
        marked = word_diff(old_abs, new_abs)
        i = diff.index(ABS_OPEN) + len(ABS_OPEN)
        j = diff.index(ABS_CLOSE)
        diff = diff[:i] + marked + diff[j:]
        diff = diff.replace(ABS_OPEN, ABSTRACT_MARKUP_SUPPORT + ABS_OPEN, 1)
        print(
            f"abstract: marked {marked.count(chr(92) + 'DIFadd{')} additions, "
            f"{marked.count(chr(92) + 'DIFdel{')} deletions"
        )

    (BUILD / "diff.tex").write_text(diff, encoding="utf-8")
    print(f"markup totals: {diff.count(chr(92) + 'DIFadd{')} additions, "
          f"{diff.count(chr(92) + 'DIFdel{')} deletions")

    for pattern in SUPPORT_GLOBS:
        for src in NEW.parent.glob(pattern):
            shutil.copy(src, BUILD / src.name)

    for _ in range(3):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "diff.tex"],
            cwd=BUILD,
            capture_output=True,
            text=True,
        )

    pdf = BUILD / "diff.pdf"
    if not pdf.exists():
        log = (BUILD / "diff.log").read_text(encoding="utf-8", errors="replace")
        errs = [ln for ln in log.splitlines() if ln.startswith("!")][:15]
        sys.exit("marked-up copy failed to compile:\n" + "\n".join(errs))

    log = (BUILD / "diff.log").read_text(encoding="utf-8", errors="replace")
    pages = re.search(r"Output written on diff\.pdf \((\d+) pages", log)
    for bad, what in (("undefined", "undefined references/citations"),):
        n = len([ln for ln in log.splitlines() if "Warning" in ln and bad in ln])
        if n:
            print(f"warning: {n} {what} in the marked-up copy")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(pdf, OUT)
    print(f"\nwrote {OUT.relative_to(ROOT).as_posix()} ({pages.group(1) if pages else '?'} pages)")

    if not args.keep_build:
        shutil.rmtree(BUILD, ignore_errors=True)
    else:
        print(f"kept build dir: {BUILD.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
