# `src/deprecated/` — Archived Source Code

This folder is **not** part of the installed `sportstradamus` package. It
lives outside `src/sportstradamus/` so Poetry will not include it in builds.

## Why this exists

Code lands here when it has no caller in any of the project's reachable
entry points:

* The CLI scripts wired in `pyproject.toml` (`prophecize`, `confer`,
  `meditate`, `reflect`, `dashboard`).
* Any module under `src/sportstradamus/scripts/`.

Removing dead code outright loses the design intent. Preserving it here
keeps the implementation searchable and re-introducible without cluttering
the active tree.

## Header protocol

Every file in this folder begins with a header comment in this exact form:

```
# ARCHIVED <YYYY-MM-DD> from src/sportstradamus/<original/path>
# Reason: <one-sentence reason>
# Last live SHA: <short git sha>
# Original imports (now unresolved here):
#   <one import per line>
```

The `Last live SHA` lets a reader run `git show <sha>:<path>` to see the
file in its original context. The `Original imports` block is a hint about
what the file expected from the live package; those imports are not
guaranteed to still resolve.

## Reintroducing archived code

If a feature returns:

1. Move the file (or the relevant function) back into `src/sportstradamus/`.
2. Update the corresponding `[ ] TODO` line in the top-level `README.md`
   to `[x]` and remove it.
3. Add caller(s). Run `ruff check` to confirm imports resolve.

## Provenance

The initial sweep landed on 2026-04-21 as Phase 2 of the maintainability
refactor. See `docs/STYLE_GUIDE.md` §11 for the dead-code policy.
