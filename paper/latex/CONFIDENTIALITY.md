# What may never reach a public artifact

This repo publishes two ways: a GitHub `main` branch (via `make_public_release.py`) and
a Zenodo deposit (via `make_zenodo_pack.py`), both citing the same content set. Neither
may ever ship the following, and both enforce it through one shared module,
`paper/latex/confidentiality.py`, rather than each keeping its own list.

## Confidential categories

- **Peer-review correspondence** — everything under `Revision/`: the editor's decision
  letter, the referee reports, the revision plan.
- **Response letters** and the **marked-up manuscript diff** — both quote the referee
  reports or show tracked changes; they go to the editor, never to a public archive.
- **Run-manifest hostnames** — `src/run_forecast.py` records `platform.node()` in every
  `*_runconfig.json` for local debugging. Useful locally, not for a public archive, so it
  is *redacted* (replaced with `"REDACTED"`) rather than the whole file being dropped —
  the rest of the manifest (git commit, package versions, timestamps) is real provenance
  worth keeping.
- **Quarantined or withdrawn analysis** — anything matching `*_quarantine_*` or
  `*NOT_USED*`.
- **Root submission artefacts** — `SUBMISSION*`, `Submission Files*`, at any nesting
  depth (not just the repo root — see the comments in `confidentiality.py` for why that
  distinction mattered).

## The rule

Any script that produces a publish-surface artifact — the public GitHub branch, the
Zenodo deposit, or any future channel — must `import confidentiality` and use its
`blocked()` (what never ships) and `transform()` (what ships redacted), and must run
`self_test()` before writing anything. Never hand-roll a new deny list for a new
publish target.

**Named exception, not a template to copy:** `make_code_si.py` (the anonymised referee
bundle) and `make_revision_package.py` (the editor's upload folder) are allowlist-first
for different, non-public audiences, and correctly *include* material this module blocks
everywhere else — the response letter belongs in the editor's package. If you're writing
a new script that ships to GitHub or Zenodo, follow `make_zenodo_pack.py`'s pattern
instead.

## One-time setup

```
python paper/latex/install_hooks.py
```

This points git at the tracked `.githooks/` directory (git only reads `.git/hooks/` by
default, which nothing here writes to) and runs the confidentiality self-test as an
immediate correctness check. After this, `.githooks/pre-push` runs automatically on
every `git push`: it refuses outright to push `master` to anywhere, and on any push that
looks like it reaches `main`, a tag, or the `public-release` branch, it re-runs the
self-test and scans the actual tree being pushed for forbidden paths.

`git push --no-verify` bypasses the hook. That's for a deliberate, understood exception
— never use it to get past a confidentiality failure.

## Verifying the rules yourself

```
python paper/latex/confidentiality.py --self-test
```
