# Cross-Agent Development Template — Design

- **Date:** 2026-06-17
- **Status:** Approved design, ready for implementation planning
- **Author:** Trevor (with Claude Code)
- **Topic:** A reusable, copy-in template that brings this repo's working Claude Code
  setup to any new or existing repository, and makes it work in both Claude Code and
  GitHub Copilot (VS Code).

## Summary

Generalize the agentic development setup that works well in Sportstradamus into a
single drop-in template. The template is one unified bundle that a user copies into a
target repo and then activates by running a `setup-project` skill inside their agent.
The skill inspects the repo, detects the stack, asks a few questions it cannot infer,
and writes a tailored configuration: shared project rules, a cross-agent hard gate, the
agent adapters, and stack-specific quality gates.

The key enabler — discovered during design — is that **Agent Skills are now a
cross-agent open standard** (`agentskills.io`). A `SKILL.md` folder placed in
`.claude/skills/` is read by **both** Claude Code and GitHub Copilot in VS Code. That
collapses most of the duplication a "two separate setups" approach would have required:
skills and rules are authored once and consumed by both agents, with only thin
per-agent adapter files differing.

## Goals

- One template, copied into a repo, that upgrades both new and existing repos.
- Works in **Claude Code** and **GitHub Copilot (VS Code)** from the same files.
- A single `setup-project` skill that tailors the setup to the target repo's stack.
- First-class support for Python, JavaScript/TypeScript, C++/CMake, and **ROS 2
  (ament/colcon)**, with a generic fallback for anything else.
- Carry forward the parts of the current setup that generalize (the code doctrine,
  quality-gate discipline, refactoring/research subagents, drift detection, push gate),
  and drop the parts that are Sportstradamus-specific.
- Keep enforcement honest across agents: hard gates run agent-independently via
  pre-commit; agent hooks are an in-loop convenience on top.

## Non-goals

- Not vendoring the Superpowers library into the template (avoids drift and licensing
  questions). The template documents how to install Superpowers per agent and relies on
  the cross-agent Agent Skills standard for the user's own skills.
- Not building Copilot-native hooks in v1. Copilot's enforcement comes from pre-commit;
  Claude Code's in-loop hooks are a bonus. Copilot hooks may be a later enhancement.
- Not supporting the Copilot cloud coding agent or Copilot CLI as primary targets in v1.
  They are compatible with the shared skills, but provisioning/CI wiring for them is out
  of scope for the first version.

## Background: the cross-agent mechanism (verified during design)

**Agent Skills** are folders containing a `SKILL.md` with YAML frontmatter:

- Required frontmatter: `name` (lowercase, hyphens/numbers, ≤64 chars, must match the
  parent directory) and `description` (≤1024 chars, says *what it does and when to use
  it*). Optional: `argument-hint`, `user-invocable`, `disable-model-invocation`,
  `context`.
- Invocation: auto-loaded when Copilot/Claude matches the request to the description, or
  invoked manually by typing `/` in chat.

**Discovery locations (Copilot in VS Code):**

- Workspace: `.github/skills/`, `.claude/skills/`, `.agents/skills/` (default;
  configurable via `chat.agentSkillsLocations`).
- User-level: `~/.copilot/skills/`, `~/.claude/skills/`, `~/.agents/skills/`.
- Monorepos: `chat.useCustomizationsInParentRepositories`.

**Relevant Copilot/VS Code settings:**

- `chat.useAgentsMdFile` — makes Copilot read `AGENTS.md`.
- `chat.useAgentSkills` — auto-load skills from the skills directories.
- `github.copilot.chat.skillTool.enabled` — experimental forked-context skill execution.

**Claude Code** reads `.claude/skills/` natively, reads `CLAUDE.md` (canonical, supports
`@path` imports), and current versions also read `AGENTS.md`.

**Consequence:** `.claude/skills/<name>/SKILL.md` is the one location read by both
agents by default, and `SKILL.md`'s `name`/`description` frontmatter is common to both.
So skills are the portable unit. The community Superpowers-for-Copilot ports (faulkdev,
dwaintr marketplace extension) confirm the pattern: they are just `.github/` files plus
`SKILL.md` skills, no VS Code extension strictly required.

> Implementation must verify that a single `SKILL.md` satisfies both agents
> simultaneously (Claude Code may accept additional optional fields; the required
> `name`/`description` are shared). If a field collides, prefer the intersection and
> document any per-agent escape hatch.

## Architecture

A single unified bundle with three conceptual layers.

```
your-repo/
  AGENTS.md                      # all project rules — single source of truth
  CLAUDE.md                      # `@AGENTS.md` import + Claude-only notes
  .pre-commit-config.yaml        # cross-agent HARD gate
  .claude/
    skills/
      setup-project/SKILL.md     # the init command — read by BOTH agents
      <custom>/SKILL.md          # the user's own skills
    settings.json                # permissions, hooks wiring, outputStyle, plugins
    hooks/                       # drift, push-gate, mark-dirty, complexity, research-gate
    agents/                      # refactoring-specialist, research-analyst, prompt-engineer
    output-styles/caveman.md
  .github/
    copilot-instructions.md      # short pointer -> AGENTS.md
    instructions/
      caveman.instructions.md    # applyTo "**" — caveman tone for Copilot
    agents/                      # Copilot agent-profile mirrors of key subagents
  .vscode/
    settings.json                # chat.useAgentsMdFile, chat.useAgentSkills, ...
```

### Layer 1 — Shared core (authored once, both agents read)

- **`AGENTS.md`** — the complete, generalized project doctrine and the quality-gate
  commands. The single source of truth. `CLAUDE.md` is `@AGENTS.md` plus a few
  Claude-only notes; `.github/copilot-instructions.md` is a short "read and follow
  `AGENTS.md`" pointer (and Copilot also auto-reads `AGENTS.md` via
  `chat.useAgentsMdFile`, giving double coverage).
- **`.claude/skills/`** — Agent Skills, read by both. Houses `setup-project` and any
  custom skills the user adds.
- **`.pre-commit-config.yaml`** — the agent-independent hard gate (format, lint,
  complexity ceiling, fast tests). Fires for Claude, Copilot, and human commits alike.
  This is how Copilot gets real enforcement despite having no Claude-style hook loop.

### Layer 2 — Claude adapter (`.claude/`)

- **`settings.json`** — tailored permission allowlist, hook wiring,
  `outputStyle: Caveman`, and the Superpowers plugin enablement
  (`enabledPlugins` + `extraKnownMarketplaces`).
- **`hooks/`** — language-agnostic hooks ported verbatim, plus generalized ones:
  - `pretask-snapshot.py` / `posttask-diff.py` — subagent drift detection (verbatim).
  - destructive-bash guard (inline in `settings.json`, verbatim).
  - `push-gate.py` + `mark-dirty.py` — block push when tests are stale since the last
    code edit (mechanism kept; the "code file" extension set and the test command come
    from the detected stack).
  - auto-format-on-edit (PostToolUse) — runs the detected formatter on the edited file.
  - `complexity-gate.py` — generalized to **lizard** (multi-language), in-loop nudge.
  - `research-gate.py` — generalized research-first gate (config-driven via
    `research_gated.txt`; the Sportstradamus `stat_meta.json` special-case is dropped).
- **`agents/`** — `refactoring-specialist` (generalized to style/readability
  enforcement), `research-analyst` (generalized to "any flagged decision"),
  `prompt-engineer` (kept as-is).
- **`output-styles/caveman.md`** — copied verbatim.

### Layer 3 — Copilot adapter (`.github/` + `.vscode/`)

- **`.github/copilot-instructions.md`** — short pointer to `AGENTS.md`.
- **`.vscode/settings.json`** — `chat.useAgentsMdFile: true`,
  `chat.useAgentSkills: true`, and any skill-location settings needed so `.claude/skills/`
  and `AGENTS.md` are picked up. Checked into the repo so defaults travel with it.
- **`.github/instructions/caveman.instructions.md`** — the caveman rules as an
  always-applied instruction (`applyTo: "**"`); same source text as the Claude
  output-style (shared in the template repo, e.g. via symlink). Copilot honors tone
  instructions, if less rigidly than Claude's native output style. "Delete this file to
  disable" noted at the top.
- **`.github/agents/`** — Copilot agent-profile mirrors of the key subagents
  (refactoring, research) where they add value. Content derived from the same generalized
  source as the Claude subagents.
- **Enforcement:** pre-commit only in v1 (no Claude-style PostToolUse loop). The
  research gate is therefore advisory (instruction) on the Copilot side, not blocking.

## The `setup-project` skill (the init)

A single Agent Skill at `.claude/skills/setup-project/SKILL.md`, runnable in either
agent (`/setup-project`). Re-runnable (safe to run again after the stack changes).

1. **Detect (read-only):**
   - Is this a git repo? If not, offer `git init`.
   - New vs existing (any source files / commit history).
   - Stack(s), via the recipe table below: language, package manager, test runner,
     linter, formatter.
   - Existing `CI`, `.pre-commit-config.yaml`, `CLAUDE.md`, `AGENTS.md`, `README`.
2. **Ask only what cannot be inferred** (one at a time, multiple-choice where possible):
   one-line project purpose, key directories, optional-module toggles, and which globs
   are research-gated.
3. **Write tailored files:**
   - `AGENTS.md` — project name, purpose, detected stack, the gate commands
     (build/test/lint/format), a short package/directory map, and the generalized
     doctrine. For an existing repo, summarize the discovered structure rather than
     writing a blank scaffold.
   - `CLAUDE.md` (`@AGENTS.md` + Claude notes) and `.github/copilot-instructions.md`
     (pointer).
   - `.claude/settings.json` — permission allowlist tailored to the detected tools
     (e.g. add colcon/cmake/clang-* for ROS 2, npm/pnpm for JS); hook wiring; the
     hook command strings (test command for push-gate, formatter for auto-format).
   - `.vscode/settings.json` — Copilot settings above.
   - `.pre-commit-config.yaml` — the stack recipe's hooks.
   - Provisioning script(s) — `session-start.sh` for Claude remote sessions, tailored to
     the package manager (and ROS 2 sourcing where applicable).
   - Seed `research_gated.txt` (commented, empty by default) and prompt for gated globs.
4. **Report** what was written and the next steps: `pre-commit install`, and the
   per-agent Superpowers install command (printed, not run).

The recipe data the skill applies lives as a reference table in the skill folder (e.g.
`references/stacks.md`), so adding a stack is a documentation edit, not code.

## Stack recipes

| Stack | Detect | Build / Test | Lint / Format | Complexity | pre-commit (fast) | Provision |
|---|---|---|---|---|---|---|
| Python | `pyproject.toml`, `setup.py` | `pytest` | `ruff` | ruff rules + lizard | ruff (check+format), lizard ceiling | poetry/uv/pip install |
| JS/TS | `package.json` | `jest`/`vitest`/`npm test` | `eslint` + `prettier` | lizard | eslint, prettier, lizard ceiling | npm/pnpm/yarn install |
| C++/CMake | `CMakeLists.txt` (no `package.xml`) | `cmake` + `ctest` | `clang-format` + `clang-tidy` | clang-tidy + lizard | clang-format, lizard ceiling | cmake configure/build |
| ROS 2 (ament) | `package.xml` (+ `CMakeLists.txt` → ament_cmake; + `setup.py` → ament_python) | `colcon build`; `colcon test && colcon test-result --verbose` | `clang-format` / `ament_uncrustify`; `ament_lint` / `clang-tidy` | clang-tidy + lizard | clang-format, cpplint, cmake/xml lint, lizard ceiling | `source /opt/ros/$ROS_DISTRO/setup.bash`; `rosdep install --from-paths src --ignore-src -y`; `colcon build`; `source install/setup.bash` |
| Generic | none of the above | run the detected/declared test command | run the detected/declared lint+format | lizard | declared lint/test + lizard ceiling | declared install command |

ROS 2 notes: heavy `ament_lint`/`colcon test` linters run in the push-gate/CI, not in
per-commit hooks (too slow). Permissions added for ROS 2: `colcon`, `cmake`, `ctest`,
`clang-format`, `clang-tidy`, `ament_*`, `rosdep`, and sourcing the overlay.
`ROS_DISTRO` is read from the environment (humble/iron/jazzy/rolling).

## Complexity enforcement (default-on for Python and C++)

Lean and config-driven — no port of the 407-line radon hook.

- **Python (soft, bypassable):** ruff rules — `C901` (mccabe, max-complexity = 10),
  `PLR1702` (max nested blocks = 4), `PLR0915` (statements/length), `PLR0913`
  (arguments). Bypass per function with `# noqa: C901`. Already enforced by the
  auto-format-on-edit hook and pre-commit.
- **C++ (soft, bypassable):** clang-tidy `readability-function-size`
  (`NestingThreshold = 4`, plus `LineThreshold`, `StatementThreshold`,
  `ParameterThreshold`) and `readability-function-cognitive-complexity`. Bypass with
  `// NOLINT`.
- **Universal hard ceiling (no bypass), both languages:** `lizard -T
  cyclomatic_complexity=49 -T nloc=200` in pre-commit and the push-gate.

Net policy (matches the current Sportstradamus model): untagged functions ≤ CC 10;
bypass-tagged functions ≤ CC 49; nothing exceeds CC 49; nesting depth ≤ 4; function
length ≤ 200 NLOC. All thresholds are tunable knobs seeded with these defaults.

## Research-driven development (kept, generalized)

The method: for an architecturally significant or hard-to-reverse change, do
literature/analysis first, write a cited brief, then decide.

- **Shared (`AGENTS.md`):** the research-first convention, how `research_gated.txt` and
  the waiver work.
- **Claude:** `research-analyst` subagent (generalized away from distribution-family
  vocabulary) plus `research-gate.py`, which **blocks** an edit to a gated glob when no
  `/tmp/researcher_*.md` brief and no `.claude/.state/research_waiver` exist.
- **Copilot:** a `research` skill (the analyst body) plus an instruction that states the
  convention. Advisory only — no blocking hook in v1.
- **Per-repo:** `setup-project` seeds `research_gated.txt` empty (gate off) and asks
  which globs to gate (e.g. ROS 2 interface definitions `**/msg/*.msg`, `**/srv/*.srv`,
  or a control-loop core directory).

## caveman + superpowers across agents

- **caveman:** the same rule text registered two ways — Claude `outputStyle: Caveman`
  (with `output-styles/caveman.md`), Copilot `.github/instructions/caveman.instructions.md`.
  On by default (the user's preference); removable.
- **superpowers:** documented install, not vendored.
  - Claude: the plugin marketplace (the user's current `enabledPlugins` /
    `extraKnownMarketplaces`).
  - Copilot: the dwaintr marketplace extension or the faulkdev skills port.
  - `setup-project` prints the correct command for whichever agent runs it.

## Carried / generalized / dropped (mapping from the current setup)

- **Carried, generalized:** the code doctrine (less-code, scope discipline,
  no-narration comments, reuse-before-write, deep-functions, type-hints-in-moderation);
  the "run the gates before claiming done" discipline; the refactoring-specialist
  mandate; subagent-driven-development convention; session memory-capture convention;
  subagent-drift check; push-gate; destructive-bash guard; research-driven development.
- **Dropped (Sportstradamus-specific):** `design-lint.py` (Streamlit design tokens),
  `devel-ship-curator` agent, the ML/league-market/ship-gate vocabulary in `CLAUDE.md`,
  the `docs-style.py` §16 reference (the generic "living docs" nudge may survive as an
  optional module without the section reference).

## Template repository structure and distribution

The template is built as **its own repository/directory** (suggested:
`~/agent-project-template/`), not inside Sportstradamus. The user copies the bundle's
contents into a target repo and runs `/setup-project`.

```
agent-project-template/
  README.md                      # what it is, how to use it
  template/                      # the drop-in payload (the layout shown above)
    AGENTS.md                    # generic seed doctrine (pre-tailoring)
    CLAUDE.md
    .pre-commit-config.yaml      # generic seed
    .claude/ ...
    .github/ ...
    .vscode/ ...
  references/                    # stack recipes, etc., consumed by setup-project
```

In a target repo, `AGENTS.md` is the single rules file (no duplication). In the template
itself it is one seed file, tailored by `setup-project` on first run. The real duplication
point is the caveman rule text, which must exist in two formats (Claude `output-style` and
Copilot instruction); that is where a shared source plus a symlink — or a small generation
step in `setup-project` — applies.

Distribution mechanism (copy the folder, clone, or a `degit`-style fetch) is left to the
implementation plan; "copy the `template/` contents in, then run `/setup-project`" is the
contract.

## Risks / to verify during implementation

1. **Single-`SKILL.md`, two agents:** confirm one `SKILL.md` satisfies both Claude Code
   and Copilot frontmatter requirements simultaneously.
2. **AGENTS.md read path in Claude:** confirm whether Claude reads `AGENTS.md` natively
   or only via the `CLAUDE.md` `@import`; ship the `@import` to be safe.
3. **Copilot settings names:** verify the exact current setting keys
   (`chat.useAgentSkills`, `chat.useAgentsMdFile`, `chat.agentSkillsLocations`) against
   the installed Copilot version before committing them.
4. **lizard availability for C++:** confirm lizard's C++ CCN/NLOC accuracy and the
   `-T` threshold exit-code behavior used as the hard gate.
5. **clang-tidy config portability:** confirm `readability-function-size` option names
   and that the shipped `.clang-tidy` works without a compilation database for
   pre-commit (or scope clang-tidy to push-gate/CI where `compile_commands.json` exists).

## Acceptance criteria

- Copying the bundle into a fresh repo and running `/setup-project` in Claude Code, then
  in a separate Copilot Chat session, produces a working setup in both without manual
  edits.
- A Python repo and a ROS 2 (ament_cmake) repo each get correct gate commands, a working
  `.pre-commit-config.yaml`, and a tailored `AGENTS.md`.
- The complexity ceiling (CC 49 / nesting 4 / NLOC 200) is enforced by pre-commit in
  both agents; the soft CC-10 gate is bypassable with `# noqa` / `// NOLINT`.
- A research-gated edit is blocked in Claude Code without a brief/waiver, and the
  convention is present (advisory) for Copilot.
- caveman and the Superpowers install guidance are present for both agents.
