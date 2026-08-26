#!/usr/bin/env python3
"""Build the Zenodo deposit archive for the revised manuscript.

The editor requires the underlying code to be deposited in a DOI-assigning repository and
linked from Methods or Code Availability. This builds that archive.

It is NOT the same artifact as `make_code_si.py`, and the difference matters:

  * `make_code_si.py` builds an ANONYMISED, referee-facing bundle: a subset of the code
    with internal comments stripped and no repository URL, because Scientific Reports
    forwards Supplementary Information to referees.
  * This builds the CITABLE archive: it mirrors what is public on the `main` branch, is
    not anonymised, and is what the DOI resolves to.

Design: the file list is an explicit ALLOWLIST taken from the public branch, not a
denylist of things to drop. A deposit is permanent and public, so the failure mode of a
denylist -- a new private file appearing and silently shipping -- is unacceptable here.
`Revision/` in particular holds confidential peer-review correspondence (the editor's
decision letter and the referee reports) and must never enter a public archive; an
allowlist makes that structurally impossible rather than a matter of remembering.

What may never ship regardless of the allowlist, and what must be redacted in what does,
is defined once in `confidentiality.py` and shared with `make_public_release.py` (the
GitHub side of this same content set) -- see that module for the policy and its rationale.

The archive is built from the git tree of the public branch where available, so what ships
is exactly what has been published, plus any revision-added files named explicitly below.

Usage:
  python paper/latex/make_zenodo_pack.py [--ref origin/main] [--out Zenodo_Deposit.zip]
  python paper/latex/make_zenodo_pack.py --self-test
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from confidentiality import blocked, needs_redact, self_test, transform  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Files added during the revision that are not yet on the public branch but belong in the
# deposit. Listed explicitly so that adding one is a deliberate act.
REVISION_ADDITIONS = [
    "src/covariate_degradation.py",
    "analysis/nwp_covariate_error.py",
    "analysis/nwp_sweep.py",
    "analysis/dm_dependence_robustness.py",
    "analysis/pandemic_stratified.py",
    "analysis/pandemic_exposure.py",
    "figures/fig6_covariate_quality.py",
    "results/v1/nwp_covariate_error.csv",
    "results/v1/pandemic_stratified.csv",
    "results/v1/pm25_dm_dependence_robustness.csv",
    "results/v1/weather_dm_dependence_robustness.csv",
    "results/v1/pm25_dm_perlead.csv",
    "results/v1/weather_dm_perlead.csv",
    "results/v1/nwp_sweep_summary.csv",
    "results/v1/nwp_sweep_raw.csv",
    # three-foundation-model panel (scale-vs-family control) and its statistics
    "results/v1/pm25_panel/causal_3fm_cities.csv",
    "results/v1/pm25_panel/causal_3fm_dm_panel.csv",
    "results/v1/pm25_panel/causal_3fm_dm_panel_summary.csv",
    "results/v1/weather_panel/causal_3fm_cities.csv",
    "results/v1/weather_panel/causal_3fm_dm_panel.csv",
    "results/v1/weather_panel/causal_3fm_dm_panel_summary.csv",
    "results/v1/energy/tfm_beijing_results.csv",
    "tables/make_revision_tables.py",
]


def _git(args):
    try:
        out = subprocess.run(["git"] + args, cwd=ROOT, capture_output=True,
                             text=True, check=True)
        return [l.strip() for l in out.stdout.splitlines() if l.strip()]
    except subprocess.CalledProcessError:
        return []


def public_top_level(ref: str):
    """Top-level entries published on `ref` -- the allowlist of what may ship at all."""
    return set(_git(["ls-tree", "--name-only", ref]))


def current_tracked():
    """Files tracked in the working tree now.

    Taken from the index rather than from the published tree because the published branch
    lags local renames (e.g. fig6_decision_maps -> fig5_decision_maps); building from the
    stale tree would ship filenames that no longer exist.

    NOTE: this used to say that untracked files never appear here, and that this was a
    second reason `Revision/` could not leak. That is no longer true -- `Revision/` is now
    committed on master. What keeps it out is the top-level allowlist (`Revision` is not
    published on `main`) and the confidentiality module's deny lists. Do not rely on
    trackedness as a control.
    """
    return _git(["ls-files"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="origin/main",
                    help="git ref whose tree defines the public file list")
    ap.add_argument("--out", default=os.path.join(ROOT, "Zenodo_Deposit.zip"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--self-test", action="store_true",
                    help="check confidentiality.py's rules against known-bad and "
                         "known-good paths, plus the redaction transform")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(1 if self_test() else 0)

    if self_test():
        sys.exit("refusing to build: the confidentiality self-test failed")

    allowed_top = public_top_level(args.ref)
    if not allowed_top:
        sys.exit(f"could not read tree of {args.ref}; fetch it first "
                 f"(git fetch origin main) or pass --ref")

    tracked = [f for f in current_tracked()
               if f.split("/")[0] in allowed_top]
    files = list(dict.fromkeys(tracked + REVISION_ADDITIONS))

    shipped, skipped_missing, refused = [], [], []
    for rel in files:
        if blocked(rel):
            refused.append(rel)
            continue
        if not os.path.exists(os.path.join(ROOT, rel)):
            skipped_missing.append(rel)
            continue
        shipped.append(rel)

    print(f"public top-level allowlist ({args.ref}): {len(allowed_top)} entries")
    print(f"tracked files within it : {len(tracked)}")
    print(f"revision additions      : {len(REVISION_ADDITIONS)}")
    print(f"-> shipping             : {len(shipped)}")
    if refused:
        print(f"-> REFUSED by NEVER list: {len(refused)}")
        for r in refused[:10]:
            print(f"     {r}")
    if skipped_missing:
        print(f"-> listed but absent    : {len(skipped_missing)}")
        for r in skipped_missing[:10]:
            print(f"     {r}")

    if args.dry_run:
        print("\ndry run; nothing written")
        return

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in shipped:
            with open(os.path.join(ROOT, rel), "rb") as f:
                data = f.read()
            z.writestr(rel, transform(rel, data))
    size = os.path.getsize(args.out) / 1024**2
    print(f"\nwrote {args.out}  ({size:.1f} MB, {len(shipped)} files)")

    # Post-hoc verification: read the archive back and assert nothing forbidden is in it,
    # and that every file needing redaction actually got it.
    import json
    with zipfile.ZipFile(args.out) as z:
        names = z.namelist()
        bad = [n for n in names if blocked(n)]
        if bad:
            sys.exit(f"REFUSING: forbidden entries found in the built archive: {bad[:10]}")
        unredacted = []
        for n in names:
            if needs_redact(n):
                cfg = json.loads(z.read(n))
                if cfg.get("hostname") not in (None, "REDACTED"):
                    unredacted.append(n)
        if unredacted:
            sys.exit(f"REFUSING: unredacted hostname shipped: {unredacted[:10]}")
    print("verified: no confidential/review-correspondence files, all hostnames redacted")


if __name__ == "__main__":
    main()
