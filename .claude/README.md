# .claude/ — agentic development tooling

This directory configures the automation this repo uses for agentic
development with Claude Code: quality-gate hooks, specialized review agents,
and a research guardrail. It encodes the same engineering policy human
contributors follow through `docs/STYLE_GUIDE.md`, `pre-commit`, and CI —
automation any contributor may benefit from, and none is required to use.
Project instructions for agent sessions live in `CLAUDE.md` at the repo root.

## Hooks (`hooks/`)

Small scripts wired to tool-lifecycle events. Alongside them, `settings.json`
runs `ruff check --fix` + `ruff format` inline on every edited `.py` file and
asks for confirmation before destructive shell commands.

- `complexity-gate.py` — blocks a `.py` edit that adds a function over the
  STYLE_GUIDE §10 limits (cyclomatic complexity > 10, source length > 200
  lines, nesting > 4) or a dead wrapper (§18.7), or makes an existing
  violation worse. A ratchet: pre-existing debt never blocks an unrelated
  edit. Waive a genuinely irreducible case with `# style: allow-complexity` /
  `-length` / `-nesting` / `-wrapper` / `-all` inside the function plus a
  one-line reason.
- `design-lint.py` — non-blocking nudge when a dashboard edit adds a design
  tell DESIGN.md bans (emoji icons, overused fonts, default red, purple gradients).
- `docs-style.py` — non-blocking nudge when a Markdown edit adds a dated
  build-log block, the drift shape STYLE_GUIDE §16 forbids.
- `mark-dirty.py` — marks the tree code-dirty after any `.py` edit so the
  push gate knows the integration suite is stale.
- `pretask-snapshot.py` / `posttask-diff.py` — snapshot HEAD/branch/dirty
  state before a subagent runs; warn afterward if it moved HEAD or changed
  files outside its assignment.
- `push-gate.py` — asks for confirmation on `git push` when the integration
  suite has not passed since the last `.py` edit.
- `research-gate.py` — blocks a research-gated edit (a `shipped:` flip in
  `stat_meta.json`, or a file matching `research_gated.txt`) until a research
  brief (`/tmp/researcher_*.md`) or a waiver (`.state/research_waiver`) exists.
- `session-start.sh` — provisions remote (web) sessions with the GitHub CLI
  and the Poetry dependency stack; no-op locally.
- `tests/` — pytest coverage for the hooks themselves.

## Agents (`agents/`)

- `refactoring-specialist` — enforces STYLE_GUIDE structure on files touched
  in a session, behavior-preserving; runs before any push, PR, or review.
- `research-analyst` — literature + statistical synthesis for ambiguous
  modeling decisions; writes a cited brief to `/tmp/researcher_*.md`.
- `devel-ship-curator` — assembles a production-delta-only PR onto `devel`
  for a gate-passing cell; never pushes.
- `experiment-graduation-specialist` — generalizes a proven single-cell
  method into the market-agnostic sweep pool, stripping per-cell hardcodes.
- `prompt-engineer` — designs and evaluates prompts for LLM-facing work.

## Wiring and state

`settings.json` binds each hook to its event (post-edit for lint, complexity,
docs, and design; pre-Bash for the push gate; pre-edit for the research gate;
around subagent runs for the snapshot pair; session start for provisioning)
and holds the session permission allowlist. `research_gated.txt` lists the
distribution-family files the research gate covers. `.state/` (gitignored)
holds runtime markers such as `code_dirty` and `integration_green`.
