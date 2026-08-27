#!/usr/bin/env python
"""Build the public release as a clean orphan branch, for pushing to `main`.

Per .claude/CLAUDE.md: public releases go out as a clean orphan branch pushed to
`main`; local `master` keeps the full history and is never pushed. This builds that
orphan branch. It does NOT push -- pushing is a separate, deliberate act.

The file list comes from make_zenodo_pack, deliberately, so the GitHub release and the
Zenodo deposit are the same content set by construction rather than by coincidence. The
manuscript's Code availability statement cites both, so they must not drift apart. What
may never ship and what must be redacted is policy shared by both, defined once in
confidentiality.py -- see that module before touching either allowlist.

    python paper/latex/make_public_release.py [--branch public-release] [--dry-run]
    python paper/latex/make_public_release.py --tag v2.0-revision

The commit is authored by whoever git is configured as, carries no AI attribution
trailer (a rule in .claude/CLAUDE.md -- GitHub renders Co-Authored-By as a repo
Contributor, and this repo is cited in the manuscript), and is built with a temporary
index so the working tree and the current branch are never touched.

Tagging is opt-in and separate. The manuscript cites tag `v2.0-revision`, and that tag
should point at the state whose CITATION.cff and README carry the minted Zenodo DOI --
so tag only after insert_zenodo_doi.py has run, not before.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_pack():
    spec = importlib.util.spec_from_file_location(
        "zpack", str(Path(__file__).with_name("make_zenodo_pack.py")))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def git(*args, env=None, check=True, capture=True):
    return subprocess.run(["git", *args], cwd=ROOT, env=env, check=check,
                          capture_output=capture, text=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="origin/main",
                    help="ref whose tree defines the public allowlist")
    ap.add_argument("--branch", default="public-release",
                    help="local branch name to (re)build")
    ap.add_argument("--message", default=None, help="commit subject")
    ap.add_argument("--tag", default=None,
                    help="also create this tag on the new commit (do this only after "
                         "the Zenodo DOI is inserted)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    zp = load_pack()

    if zp.self_test():
        sys.exit("refusing to build: the confidentiality self-test failed")

    allowed_top = zp.public_top_level(args.ref)
    if not allowed_top:
        sys.exit(f"could not read tree of {args.ref}; run: git fetch origin main")

    tracked = [f for f in zp.current_tracked() if f.split("/")[0] in allowed_top]
    files = list(dict.fromkeys(tracked + zp.REVISION_ADDITIONS))

    shipped, refused, missing = [], [], []
    for rel in files:
        if zp.blocked(rel):
            refused.append(rel)
        elif not (ROOT / rel).exists():
            missing.append(rel)
        else:
            shipped.append(rel)

    print(f"allowlist ({args.ref}): {len(allowed_top)} top-level entries")
    print(f"shipping  : {len(shipped)} files")
    print(f"refused   : {len(refused)}")
    for r in refused:
        print(f"    {r}")
    if missing:
        print(f"listed but absent: {len(missing)}")
        for r in missing[:10]:
            print(f"    {r}")

    # Independent re-check of the final list, not of the rules that produced it.
    leaked = [f for f in shipped if zp.blocked(f)]
    if leaked:
        sys.exit(f"REFUSING: blocked paths survived into the ship list: {leaked[:10]}")

    if args.dry_run:
        print("\ndry run; no branch written")
        return

    subject = args.message or "Public release (revision)"

    # Split off files needing a content transform (currently: redact the hostname field
    # in run manifests -- see confidentiality.py) from everything else. The plain files
    # are added in one batch from their tracked working-tree content, exactly as before;
    # the redacted ones are classified by filename alone (no need to read content just to
    # decide), then injected below without ever writing the transformed bytes to disk.
    redact_paths = [rel for rel in shipped if zp.needs_redact(rel)]
    plain_paths = [rel for rel in shipped if rel not in set(redact_paths)]

    # Build the tree through a throwaway index so neither the working tree nor the
    # current branch is disturbed. The orphan commit has no parent, so none of the
    # private master history is reachable from what gets pushed.
    with tempfile.TemporaryDirectory() as td:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(td) / "index"))
        batch = "".join(f"{f}\0" for f in plain_paths)
        proc = subprocess.run(
            # -z must precede --stdin: git rejects any option placed after --stdin.
            ["git", "update-index", "--add", "-z", "--stdin"],
            cwd=ROOT, env=env, input=batch.encode(), capture_output=True)
        if proc.returncode != 0:
            sys.exit(f"git update-index failed:\n{proc.stderr.decode(errors='replace')}")

        # Redacted files: transform in memory, inject the result as a blob via
        # hash-object (writes to the object database from stdin, no working-tree write),
        # then register that blob under the file's original path with --cacheinfo. The
        # tracked original on disk is never touched.
        for rel in redact_paths:
            with open(ROOT / rel, "rb") as f:
                data = f.read()
            data = zp.transform(rel, data)
            h = subprocess.run(["git", "hash-object", "-w", "--stdin"], cwd=ROOT, env=env,
                               input=data, capture_output=True, check=True)
            sha = h.stdout.decode().strip()
            subprocess.run(["git", "update-index", "--add", "--cacheinfo",
                            f"100644,{sha},{rel}"], cwd=ROOT, env=env, check=True)

        tree = git("write-tree", env=env).stdout.strip()
        commit = git("commit-tree", tree, "-m", subject, env=env).stdout.strip()

    git("branch", "-f", args.branch, commit)
    n = len(git("ls-tree", "-r", "--name-only", commit).stdout.splitlines())
    print(f"\nbuilt {args.branch} -> {commit[:10]}  ({n} files, orphan: no parent)")

    # Post-hoc verification against the actual committed tree, not just the plan for it:
    # confirm the redacted paths really carry the placeholder in the built commit.
    import json
    unredacted = []
    for rel in redact_paths:
        content = git("show", f"{commit}:{rel}").stdout
        cfg = json.loads(content)
        if cfg.get("hostname") not in (None, "REDACTED"):
            unredacted.append(rel)
    if unredacted:
        sys.exit(f"REFUSING: unredacted hostname in built commit: {unredacted[:10]}")
    if redact_paths:
        print(f"verified: hostname redacted in all {len(redact_paths)} run manifests")

    if args.tag:
        git("tag", "-f", "-a", args.tag, commit, "-m", subject)
        print(f"tagged {args.tag} -> {commit[:10]}")

    print("\nreview before pushing:")
    print(f"  git ls-tree -r --name-only {args.branch} | less")
    print(f"  git show --stat {args.branch}")
    print("\nthen, to publish:")
    print(f"  git push origin {args.branch}:main" + (" --force-with-lease" if True else ""))
    if args.tag:
        print(f"  git push origin {args.tag}")


if __name__ == "__main__":
    main()
