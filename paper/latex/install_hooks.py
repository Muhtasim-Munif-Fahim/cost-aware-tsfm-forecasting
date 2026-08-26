#!/usr/bin/env python
"""One-time local setup: point git at the tracked hooks directory, then prove it works.

`.githooks/pre-push` is tracked in the repo, but git only runs hooks from `.git/hooks/`
by default -- an untracked, per-clone directory nothing here writes to. Without this
step (or the equivalent manual `git config`), the hook is silently inert: no error, no
warning, a push just doesn't get checked. Run once per clone/machine.

    python paper/latex/install_hooks.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    subprocess.run(["git", "config", "core.hooksPath", ".githooks"], cwd=ROOT, check=True)
    print("core.hooksPath -> .githooks")

    proc = subprocess.run([sys.executable, "paper/latex/confidentiality.py", "--self-test"],
                          cwd=ROOT)
    if proc.returncode != 0:
        sys.exit("hooksPath is set, but the confidentiality self-test failed -- "
                 "fix that before relying on the hook.")
    print("confidentiality self-test passed; the pre-push hook is now active.")


if __name__ == "__main__":
    main()
