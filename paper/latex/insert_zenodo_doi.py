#!/usr/bin/env python
"""Insert the minted Zenodo DOI everywhere it belongs, in one pass.

The DOI is the last blocker on the Scientific Reports revision, and it has to land in
five places that are easy to half-finish by hand. Every edit below asserts on its exact
expected text and refuses to guess, so a partial run fails loudly instead of silently
leaving a placeholder in a submitted file.

Places the DOI goes:
  1. paper/sections/05_backmatter.md   Code availability (link label AND target)
  2. Revision/RESPONSE_TO_REVIEWERS.md the editor's DOI-repository request
  3. CITATION.cff                      doi: + identifiers:
  4. README.md                         "will be added on release" -> the real DOI

It also DELETES the note at the top of the response letter that begins "One item
outstanding before submission". That block is addressed to us, not to the editor, and
must not travel with the submitted letter.

    python paper/latex/insert_zenodo_doi.py 10.5281/zenodo.17123456 [--dry-run]

Accepts a bare DOI, a doi.org URL, or a zenodo.org record URL.

Reserve the DOI on the Zenodo draft BEFORE publishing ("Reserve DOI" on the deposit
form). CITATION.cff and README.md ship inside Zenodo_Deposit.zip, so the deposit must
contain files that already carry its own DOI; reserving is what makes that possible.

After this runs:
    bash paper/latex/build.sh
    python paper/latex/make_markedup.py
    python paper/latex/make_response_pdf.py
    python paper/latex/make_zenodo_pack.py       # rebuild the deposit with the DOI in it
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

BACKMATTER = ROOT / "paper" / "sections" / "05_backmatter.md"
LETTER = ROOT / "Revision" / "RESPONSE_TO_REVIEWERS.md"
CITATION = ROOT / "CITATION.cff"
README = ROOT / "README.md"

DOI_RE = re.compile(r"^10\.5281/zenodo\.\d+$")

# --- exact source text each edit expects -------------------------------------------

LETTER_INTERNAL_NOTE = """
> **One item outstanding before submission: the Zenodo DOI (marked `[PENDING]` below) must
> be minted and inserted here and in the manuscript's Code availability statement, which
> currently carries the placeholder `ZENODO-DOI-PLACEHOLDER`. All analyses are complete.**
"""

# Matched as a whole paragraph and re-emitted rewrapped, so the DOI does not leave a
# 130-character line in a file otherwise wrapped at ~90.
LETTER_PENDING_OLD = (
    "`[PENDING]` The analysis code is deposited at Zenodo under DOI `<TO BE INSERTED>` and is\n"
    "linked from the Code Availability statement. The public GitHub repository remains at\n"
    "github.com/Muhtasim-Munif-Fahim/cost-aware-tsfm-forecasting."
)

README_OLD = "A Zenodo archive DOI will be added on release."

CITATION_ANCHOR = 'repository-code: "https://github.com/Muhtasim-Munif-Fahim/cost-aware-tsfm-forecasting"\n'


def normalise(raw: str) -> str:
    """Accept a bare DOI, a doi.org URL, or a zenodo.org record URL."""
    s = raw.strip().rstrip("/")
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s, flags=re.I)
    m = re.search(r"zenodo\.org/(?:records?|record)/(\d+)", s, flags=re.I)
    if m:
        s = f"10.5281/zenodo.{m.group(1)}"
    if s.lower().startswith("doi:"):
        s = s[4:]
    if not DOI_RE.match(s):
        sys.exit(
            f"'{raw}' does not look like a Zenodo DOI.\n"
            "Expected 10.5281/zenodo.<digits>, a doi.org URL, or a zenodo.org record URL."
        )
    return s


class Editor:
    def __init__(self, dry_run: bool) -> None:
        self.dry_run = dry_run
        self.changes: list[str] = []

    def edit(self, path: Path, old: str, new: str, what: str) -> None:
        if not path.exists():
            sys.exit(f"missing {path.relative_to(ROOT)}")
        text = path.read_text(encoding="utf-8")
        n = text.count(old)
        if n == 0:
            sys.exit(
                f"{path.relative_to(ROOT).as_posix()}: expected text for '{what}' not found.\n"
                f"Looked for:\n---\n{old}\n---\n"
                "The file has drifted since this script was written; fix by hand and "
                "update the expected text here."
            )
        text = text.replace(old, new)
        if not self.dry_run:
            path.write_text(text, encoding="utf-8")
        self.changes.append(
            f"  {path.relative_to(ROOT).as_posix()}: {what}"
            + (f" ({n} occurrences)" if n > 1 else "")
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("doi", help="e.g. 10.5281/zenodo.17123456")
    ap.add_argument("--dry-run", action="store_true", help="report edits without writing")
    args = ap.parse_args()

    doi = normalise(args.doi)
    url = f"https://doi.org/{doi}"
    print(f"DOI : {doi}\nURL : {url}\n")

    ed = Editor(args.dry_run)

    # 1. manuscript Code availability -- placeholder is both the link label and the target
    ed.edit(BACKMATTER, "ZENODO-DOI-PLACEHOLDER", doi, "Code availability DOI")

    # 2. response letter: drop the note to ourselves, then answer the editor properly
    ed.edit(LETTER, LETTER_INTERNAL_NOTE, "", "removed internal 'one item outstanding' note")
    ed.edit(
        LETTER,
        LETTER_PENDING_OLD,
        f"The analysis code is deposited at Zenodo under DOI [{doi}]({url}),\n"
        f"and is linked from the Code Availability statement. The public GitHub\n"
        f"repository remains at github.com/Muhtasim-Munif-Fahim/cost-aware-tsfm-forecasting.",
        "editor DOI-repository response",
    )

    # 3. CITATION.cff -- both the plain field and a resolvable identifier entry
    ed.edit(
        CITATION,
        CITATION_ANCHOR,
        CITATION_ANCHOR
        + f'doi: "{doi}"\n'
        + "identifiers:\n"
        + '  - type: doi\n'
        + f'    value: "{doi}"\n'
        + '    description: "Archived snapshot of the analysis code and results"\n',
        "doi + identifiers",
    )

    # 4. README
    ed.edit(README, README_OLD, f"The archived snapshot is at [{doi}]({url}).", "archive DOI")

    print(("would change" if args.dry_run else "changed") + ":")
    print("\n".join(ed.changes))

    leftovers = []
    for p in (BACKMATTER, LETTER, CITATION, README):
        body = p.read_text(encoding="utf-8")
        for marker in ("ZENODO-DOI-PLACEHOLDER", "[PENDING]", "TO BE INSERTED"):
            if marker in body:
                leftovers.append(f"  {p.relative_to(ROOT).as_posix()}: {marker}")
    if leftovers and not args.dry_run:
        print("\nWARNING -- placeholders still present:")
        print("\n".join(leftovers))
    elif not args.dry_run:
        print("\nno placeholders remain in the four edited files")

    print(
        "\nnext:\n"
        "  bash paper/latex/build.sh\n"
        "  python paper/latex/make_markedup.py\n"
        "  python paper/latex/make_response_pdf.py\n"
        "  python paper/latex/make_zenodo_pack.py"
    )
    if args.dry_run:
        print("\n(dry run; nothing written)")


if __name__ == "__main__":
    main()
