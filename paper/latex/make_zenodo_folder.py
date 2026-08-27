#!/usr/bin/env python
"""Extract the Zenodo deposit archive to a folder, for drag-and-drop upload.

Zenodo's "New upload" page accepts files dropped individually as well as a zip; some
users find dragging the extracted folder's contents more natural than uploading one
zip. This does not rebuild anything -- it extracts Zenodo_Deposit.zip verbatim, so
the folder is guaranteed to match what make_zenodo_pack.py already built and
verified (confidentiality self-test, no forbidden paths, DOI embedded, hostnames
redacted). If Zenodo_Deposit.zip is stale, rebuild it first:

    python paper/latex/make_zenodo_pack.py

Usage:
    python paper/latex/make_zenodo_folder.py [--out Zenodo_Upload]
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from confidentiality import blocked, self_test  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ZIP = ROOT / "Zenodo_Deposit.zip"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="Zenodo_Upload")
    args = ap.parse_args()

    if self_test():
        sys.exit("refusing to extract: the confidentiality self-test failed")

    if not ZIP.exists():
        sys.exit(f"missing {ZIP.relative_to(ROOT)}; build it first:\n"
                 f"  python paper/latex/make_zenodo_pack.py")

    out = ROOT / args.out
    import shutil
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    with zipfile.ZipFile(ZIP) as z:
        names = z.namelist()
        bad = [n for n in names if blocked(n)]
        if bad:
            sys.exit(f"REFUSING: forbidden entries in {ZIP.name}: {bad[:10]}")
        z.extractall(out)

    n_files = sum(1 for _ in out.rglob("*") if _.is_file())
    n_dirs = sum(1 for _ in out.rglob("*") if _.is_dir())
    total_bytes = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())

    print(f"extracted {len(names)} entries -> {out}")
    print(f"  {n_files} files, {n_dirs} directories, {total_bytes / 1048576:.1f} MB")
    print(f"\nre-verified: no forbidden paths in the extracted folder")
    print(f"\nUpload the CONTENTS of this folder (not the folder itself) to the")
    print(f"reserved Zenodo deposit, then click Publish:")
    print(f"  {out}")


if __name__ == "__main__":
    main()
