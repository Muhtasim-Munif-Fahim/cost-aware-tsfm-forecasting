#!/usr/bin/env python3
"""What may never reach a public artifact (GitHub `main`, the Zenodo deposit, any
future publish channel), and what must be redacted in what does.

This module is policy, not a build script. Every script that assembles a public-facing
artifact -- today `make_zenodo_pack.py` and `make_public_release.py` -- imports
`blocked()` and `self_test()` from here rather than hand-rolling its own deny list.
That rule exists because of what happened without it: `MANUSCRIPT/response_to_reviewers.pdf`
(quotes the referee reports verbatim) nearly shipped in both the Zenodo deposit and the
GitHub push before a NEVER_ICASE pattern was added for it, and two more gaps in the same
class turned up on the very next audit (see NEVER's history below and REDACT).

Two exceptions, deliberately NOT covered by this module and never to be pointed at it:
`make_code_si.py` (the anonymised referee bundle) and `make_revision_package.py` (the
editor's upload folder) are allowlist-first for different, non-public audiences, and
correctly include material this module blocks everywhere else -- the response letter is
*supposed* to be in the editor's package. Do not import this module into either.

Run `python paper/latex/confidentiality.py --self-test` to check the rules against known
paths in both directions. Every build script that ships to a public artifact runs this
automatically before writing anything.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import sys

# Case-SENSITIVE patterns. Must stay case-sensitive: "SUBMISSION*" targets the root
# SUBMISSION.pdf/SUBMISSION.zip artefacts, and folding case would also catch
# paper/latex/make_submission.py, which is tooling that belongs in every public artifact.
#
# The "*/"-prefixed forms exist because fnmatch has no implicit "**/" prefix -- a bare
# "SUBMISSION*" only matches at the start of the full relative path. Without the prefixed
# form, MANUSCRIPT/SUBMISSION.pdf would ship: MANUSCRIPT is an allowed top-level publish
# directory, so the allowlist doesn't stop it, and the root-anchored pattern doesn't
# either. Found live in the working tree as an untracked file before this was fixed.
NEVER = [
    "Revision/*",
    "SUBMISSION*", "*/SUBMISSION*",
    "Submission Files*", "*/Submission Files*",
    "*.docx",
    "*_quarantine_*",
    "*NOT_USED*",
    # archive/pilot_2026-07-13/**: pre-audit pilot output, self-documented by
    # archive/pilot_2026-07-13/README.md as containing known-contaminated results
    # (unfiltered sentinel/impossible PM2.5 values -- a Seoul reading of 10000, negative
    # concentrations in three cities) superseded by the canonical results/v1/ campaign.
    # Never cited by path or number in the manuscript, supplement, or README. It was
    # already tracked on the public GitHub main before this policy existed, which is how
    # it reached a real Zenodo upload undetected -- the top-level allowlist only checks
    # "is this public on GitHub", not "does the manuscript need this", and those are
    # different bars. Kept on GitHub for provenance (its own README explains why); not
    # citable-record material, so excluded from every deposit/release built from here on.
    "archive/*",
    # MANUSCRIPT/*: the built manuscript and supplement PDFs. Not code, not promised by
    # the Code availability statement, and publishing the under-review manuscript is a
    # self-archiving decision that must be taken deliberately rather than inherited from
    # a file list. Excluded from every public artifact.
    "MANUSCRIPT/*",
    # Internal editorial/session handoff notes (bibliography verification process),
    # never cited by the manuscript and not one of the four things the Code
    # Availability statement names (harness, analysis plan, results ledger,
    # figure/table code).
    "paper/REFERENCE_VERIFICATION_HANDOFF.md",
]

# Case-INSENSITIVE patterns, for peer-review correspondence and review-process artefacts.
# Matched without regard to case because the same document exists under several
# spellings: the source is Revision/RESPONSE_TO_REVIEWERS.md, but the rendered copies are
# MANUSCRIPT/response_to_reviewers.{pdf,docx}. A case-sensitive pattern written for the
# source silently let the lowercase renders through, and MANUSCRIPT/ being an allowed
# top-level entry meant the allowlist didn't stop them either.
#
# Anchored to the basename ("name*" / "*/name*") rather than "*name*", so that build
# tooling named after the document it renders still ships: paper/latex/make_response_letter.py
# is a script, not correspondence, and an unanchored "*response_letter*" pattern swallowed it.
NEVER_ICASE = [
    "response_to_reviewers*", "*/response_to_reviewers*",
    "response_letter*", "*/response_letter*",
    "*manuscript_marked*",                       # marked-up copy: a review artefact
    "*reviewers.txt", "*editor.txt",
    "revision_plan*", "*/revision_plan*",
]

# Files that ship, but only after a content transform. Distinct from NEVER: the file
# carries real, useful information (a run manifest's provenance) alongside one field that
# doesn't belong in a public archive. Blocking the whole file would be strictly worse than
# redacting the field, so it is not in NEVER.
#
# src/run_forecast.py writes "hostname": platform.node() into every run manifest for local
# debugging. On this machine that value is "Fahim" -- the author's own Windows username,
# which happens to be their first name. Low severity (the author's full name is already
# public as corresponding author) but still sloppy in a manifest an editor might open, and
# 58 such files were already tracked and published before this was noticed.
REDACT = ["*_runconfig.json"]


def blocked(path: str) -> bool:
    # fnmatchcase, NOT fnmatch: on Windows fnmatch folds case, which would make the
    # pattern intended for the root SUBMISSION.pdf artefact also reject make_submission.py.
    if any(fnmatch.fnmatchcase(path, pat) for pat in NEVER):
        return True
    low = path.lower()
    return any(fnmatch.fnmatchcase(low, pat.lower()) for pat in NEVER_ICASE)


def needs_redact(path: str) -> bool:
    return any(fnmatch.fnmatchcase(path, pat) for pat in REDACT)


def transform(path: str, data: bytes) -> bytes:
    """Return `data` as it should ship: unchanged, unless `path` needs redaction."""
    if not needs_redact(path):
        return data
    cfg = json.loads(data)
    if "hostname" in cfg:
        cfg["hostname"] = "REDACTED"
    # indent=2 keeps the file human-diffable; default=str covers any non-JSON-native
    # value (e.g. a numpy scalar) the original writer may have let through.
    return json.dumps(cfg, indent=2, default=str).encode("utf-8")


# Paths that must be refused, and paths that must survive. The SHIP half exists because
# the obvious over-broad fix (folding case on every pattern) silently drops tooling that
# belongs in the deposit -- that mistake is exactly what a fixture here would catch.
SELF_TEST_BLOCK = [
    "MANUSCRIPT/response_to_reviewers.pdf",      # quotes the referee reports verbatim
    "MANUSCRIPT/response_to_reviewers.docx",
    "MANUSCRIPT/manuscript_marked.pdf",
    "Revision/RESPONSE_TO_REVIEWERS.md",
    "Revision/REVISION_PLAN.md",
    "Revision/Reviewers.txt",
    "Revision/Editor.txt",
    "response_letter.pdf",
    "SUBMISSION.pdf",
    "results/v1/_quarantine_partial_sweep/sweep_nwp_a0_s42_cities.csv",
    # nested SUBMISSION* paths -- the anchoring gap; these sat untracked in MANUSCRIPT/
    # when the gap was found, one allowlist hop away from shipping.
    "MANUSCRIPT/SUBMISSION.pdf",
    "MANUSCRIPT/SUBMISSION (1).pdf",
    "MANUSCRIPT/Submission Files.pdf",
    # pre-audit pilot data, self-documented as contaminated, never cited anywhere
    "archive/pilot_2026-07-13/contaminated_results/pm25_final_panel_cities.csv",
    "archive/pilot_2026-07-13/README.md",
    "paper/REFERENCE_VERIFICATION_HANDOFF.md",
    "MANUSCRIPT/manuscript.pdf",
    "MANUSCRIPT/supplementary.pdf",
]
SELF_TEST_SHIP = [
    "paper/latex/make_submission.py",            # must NOT be caught by "SUBMISSION*"
    "paper/latex/make_zenodo_pack.py",
    "paper/latex/make_public_release.py",
    "paper/latex/confidentiality.py",
    "paper/latex/make_response_letter.py",       # tooling named after the doc it renders
    "paper/latex/make_markedup.py",
    "analysis/nwp_sweep.py",
    "src/run_forecast.py",
    "figures/fig6_covariate_quality.py",
    # explicitly promised by the Code Availability statement -- must NOT get caught
    # by any future broadening of the paper/REFERENCE_VERIFICATION_HANDOFF.md rule
    "paper/ANALYSIS_PLAN.md",
    "paper/METHODS.md",
    "paper/RESULTS_LEDGER.md",
]


def _self_test_redaction() -> int:
    fixture = json.dumps({"hostname": "Fahim", "git_commit": "abc123", "n": 29}).encode()
    out = json.loads(transform("results/v1/x_runconfig.json", fixture))
    ok = (out.get("hostname") == "REDACTED"
          and out.get("git_commit") == "abc123" and out.get("n") == 29)
    if not ok:
        print(f"  FAIL redaction: got {out!r}")
    unrelated = transform("results/v1/x_cities.csv", fixture)
    if unrelated != fixture:
        print("  FAIL redaction: transform touched a file that should pass through untouched")
        ok = False
    return 0 if ok else 1


def self_test() -> int:
    failures = 0
    for rel in SELF_TEST_BLOCK:
        if not blocked(rel):
            print(f"  FAIL would publish: {rel}")
            failures += 1
    for rel in SELF_TEST_SHIP:
        if blocked(rel):
            print(f"  FAIL wrongly refused: {rel}")
            failures += 1
    failures += _self_test_redaction()
    print(f"self-test: {len(SELF_TEST_BLOCK)} refused + {len(SELF_TEST_SHIP)} shipped "
          f"+ redaction, {failures} failure(s)")
    return failures


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if not args.self_test:
        ap.error("this module has no build action of its own; pass --self-test")
    sys.exit(1 if self_test() else 0)
