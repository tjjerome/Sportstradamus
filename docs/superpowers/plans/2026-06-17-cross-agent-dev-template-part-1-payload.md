# Cross-Agent Dev Template — Part 1 (Static Payload) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the copy-in template payload that brings the Sportstradamus agentic setup to any repo and works in both Claude Code and GitHub Copilot (VS Code) — minus the smart `setup-project` init, which is Part 2.

**Architecture:** A new standalone git repo at `$HOME/agent-project-template`. The drop-in payload lives under `template/`; the repo also holds its own dev-time `tests/` (pytest, exercising the shipped Python hooks) and `references/`. Shared rules live once in `template/AGENTS.md`; `CLAUDE.md` and `.github/copilot-instructions.md` are pointers. Hard enforcement is a `template/.pre-commit-config.yaml` (agent-independent). Claude Code gets in-loop hooks; Copilot gets the same rules via instructions plus the shared `.claude/skills/` location and pre-commit.

**Tech Stack:** Python 3.11 (hooks + their pytest tests), pre-commit, lizard (multi-language complexity), ruff, clang-tidy, the Agent Skills `SKILL.md` standard, Claude Code `settings.json`/hooks, GitHub Copilot `.github/` customization + `.vscode/settings.json`.

**Source references (read once):**
- Design spec: `/home/trevor/Sportstradamus/docs/superpowers/specs/2026-06-17-cross-agent-dev-template-design.md`
- Hooks to port verbatim live in `/home/trevor/Sportstradamus/.claude/hooks/`
- caveman output-style to copy: `/home/trevor/Sportstradamus/.claude/output-styles/caveman.md`

**Conventions for this plan:**
- `TEMPLATE_ROOT` means `$HOME/agent-project-template`. All work happens there, NOT in the Sportstradamus repo.
- "Copy verbatim from `<src>`" means byte-for-byte; do not edit the file.
- Every task ends with a commit in `TEMPLATE_ROOT`.

---

## File Structure

Created in `TEMPLATE_ROOT`:

```
agent-project-template/
  README.md                              # what it is + how to use (Task 1, finalized Task 14)
  pyproject.toml                         # dev deps (pytest, lizard, ruff, pre-commit) (Task 1)
  references/                            # consumed by Part 2's setup-project (placeholder now)
  tests/
    conftest.py                          # payload-path fixtures (Task 7)
    test_drift_hooks.py                  # Task 7
    test_push_gate.py                    # Task 8
    test_complexity_gate.py              # Task 9
    test_research_gate.py                # Task 10
  template/                              # THE DROP-IN PAYLOAD
    AGENTS.md                            # single source of rules (Task 2)
    CLAUDE.md                            # @AGENTS.md pointer (Task 3)
    .pre-commit-config.yaml              # cross-agent hard gate (Task 11)
    research_gated.txt                   # seed, empty/commented (Task 10)
    .claude/
      settings.json                      # perms, hooks wiring, outputStyle, plugins (Task 6)
      output-styles/caveman.md           # copied verbatim (Task 5)
      hooks/
        pretask-snapshot.py              # verbatim (Task 7)
        posttask-diff.py                 # verbatim (Task 7)
        mark-dirty.py                    # generalized (Task 8)
        push-gate.py                     # generalized (Task 8)
        complexity-gate.py               # lizard, new (Task 9)
        research-gate.py                 # generalized (Task 10)
      agents/
        refactoring-specialist.md        # generalized (Task 12)
        research-analyst.md              # generalized (Task 12)
        prompt-engineer.md               # copied (Task 12)
      skills/.gitkeep                     # shared skills home; setup-project lands here in Part 2 (Task 1)
    .github/
      copilot-instructions.md            # pointer -> AGENTS.md (Task 3)
      instructions/caveman.instructions.md  # symlink -> ../../.claude/output-styles/caveman.md (Task 5)
      agents/
        refactoring-specialist.md        # Copilot mirror (Task 12)
        research-analyst.md              # Copilot mirror (Task 12)
    .vscode/
      settings.json                      # Copilot wiring (Task 4)
```

---

## Task 1: Scaffold the template repo

**Files:**
- Create: `$HOME/agent-project-template/` (git repo)
- Create: `$HOME/agent-project-template/pyproject.toml`
- Create: `$HOME/agent-project-template/README.md` (skeleton)
- Create: `$HOME/agent-project-template/template/.claude/skills/.gitkeep`
- Create: `$HOME/agent-project-template/references/.gitkeep`

- [ ] **Step 1: Create the repo and directory skeleton**

```bash
mkdir -p "$HOME/agent-project-template"
cd "$HOME/agent-project-template"
git init -q
mkdir -p template/.claude/hooks template/.claude/agents template/.claude/output-styles \
         template/.claude/skills template/.github/instructions template/.github/agents \
         template/.vscode references tests
touch template/.claude/skills/.gitkeep references/.gitkeep
```

- [ ] **Step 2: Write `pyproject.toml` (dev deps + pytest config)**

```toml
[project]
name = "agent-project-template"
version = "0.1.0"
description = "Cross-agent (Claude Code + GitHub Copilot) project setup template"
requires-python = ">=3.11"

[project.optional-dependencies]
dev = ["pytest>=8", "lizard>=1.17", "ruff>=0.6", "pre-commit>=3.7"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Write `README.md` skeleton** (finalized in Task 14)

```markdown
# Cross-Agent Project Template

Drop-in setup that gives any repo a consistent agentic workflow in **Claude Code**
and **GitHub Copilot (VS Code)** from the same files.

Status: Part 1 (static payload). The `setup-project` init skill is Part 2.

## Use it (manual, Part 1)

1. Copy the contents of `template/` into the root of your repo.
2. Edit `AGENTS.md`: project name, purpose, and the build/test/lint commands.
3. Install the hard gate: `pip install pre-commit lizard && pre-commit install`.
4. Open the repo in Claude Code or Copilot — both read the shared rules and skills.

(Full how-to in Task 14.)
```

- [ ] **Step 4: Install dev deps and verify pytest runs**

Run: `cd "$HOME/agent-project-template" && pip install -e ".[dev]" 2>/dev/null || pip install pytest lizard ruff pre-commit`
Run: `python -m pytest -q`
Expected: pytest runs and reports "no tests ran" (exit 5) — confirms the harness works.

- [ ] **Step 5: Commit**

```bash
cd "$HOME/agent-project-template"
git add -A
git commit -m "chore: scaffold cross-agent template repo"
```

---

## Task 2: Write the generic `AGENTS.md` doctrine

This is the carried-over doctrine from the Sportstradamus `CLAUDE.md`, stripped of all
project specifics. It is the single source of rules both agents read.

**Files:**
- Create: `template/AGENTS.md`

- [ ] **Step 1: Write `template/AGENTS.md`**

```markdown
# AGENTS.md

Project rules for AI coding agents (Claude Code, GitHub Copilot, and any
skills-compatible agent). This file is the single source of truth. `CLAUDE.md`
imports it; `.github/copilot-instructions.md` points to it.

> Replace the bracketed placeholders below during setup. Keep this file the one
> canonical home for project rules — cross-reference, do not duplicate.

## Project

- **Name:** [PROJECT NAME]
- **Purpose:** [ONE-LINE PURPOSE]
- **Stack:** [DETECTED STACK]
- **Key directories:** [MAP]

## Quality gates — run before claiming done

```bash
[FORMAT COMMAND]
[LINT COMMAND]
[TEST COMMAND]
```

All must pass. A clean run is required before you say a task is finished, and
before any commit or push.

## Writing code here

The default is *less code*, not more. Match the surrounding code; do not impose
textbook patterns over it.

- **Scope.** Only make changes directly requested or clearly necessary. No
  unrequested features, refactors, or "improvements." Notice unrelated work worth
  doing? Say so — do not just do it.
- **Defensive coding.** No error handling, fallbacks, or validation for cases that
  cannot happen. Catch only the specific exceptions you can handle; let everything
  else fail loud. No bare `except`. Prefer try/act over pre-checking conditions that
  are almost always true.
- **Comments explain why, never what.** Delete any comment that restates the line
  under it. No section-divider banners, no `# Note:` spam. Docstrings on public
  functions/classes/modules only, and only when they add what the signature does not.
- **Prefer a few deep functions over many thin ones.** A coherent 40-line function
  beats six 7-line fragments. No wrapper that only renames and forwards a call. No
  new class where a function does the job. No factory/strategy/DI scaffolding until
  three real implementations need it.
- **Reuse before you write.** Grep for an existing helper before adding one. Do not
  reimplement the standard library or the existing stack. Find the same logic in two
  places? Consolidate — do not copy it a third time.
- **Type hints in moderation.** Annotate public signatures and module boundaries;
  skip obvious locals. Avoid `Any`; model real structures.

## Complexity limits (enforced)

- Cyclomatic complexity: untagged functions ≤ 10; with an explicit bypass
  (`# noqa: C901` in Python, `// NOLINT` in C++) ≤ 49; nothing exceeds 49.
- Nesting depth ≤ 4. Function length ≤ 200 lines.
- The pre-commit gate enforces the hard ceilings for every agent and human commit.

## Research-driven development

Before an architecturally significant or hard-to-reverse change (anything matching a
glob in `research_gated.txt`): do the research first, write a short cited brief to
`/tmp/researcher_<topic>.md`, then proceed. In Claude Code the research-gate hook
blocks such an edit until a brief or a one-line `.claude/.state/research_waiver`
exists. Other agents: treat this as a required step even though it is not enforced.

## Subagent / multi-step work

For work touching two or more modules, dispatch one worker per module and review the
diffs, rather than serializing everything through one context. Keep each worker to a
single module's scope.

## Memory

When a unit of work completes, capture any durable, non-obvious, repeatable lesson —
do not force a memory from every session.
```

- [ ] **Step 2: Verify the file is valid Markdown and has no leftover real project names**

Run: `grep -niE "sportstradamus|league|market|streamlit" template/AGENTS.md || echo "clean"`
Expected: `clean`

- [ ] **Step 3: Commit**

```bash
git add template/AGENTS.md
git commit -m "feat: generic AGENTS.md doctrine seed"
```

---

## Task 3: Pointer files (CLAUDE.md, copilot-instructions.md)

**Files:**
- Create: `template/CLAUDE.md`
- Create: `template/.github/copilot-instructions.md`

- [ ] **Step 1: Write `template/CLAUDE.md`**

```markdown
@AGENTS.md

## Claude Code notes

- This project's rules live in AGENTS.md (imported above). Follow them.
- Skills are in `.claude/skills/` (shared with Copilot). Hooks are in `.claude/hooks/`.
- Output style `Caveman` is on by default; switch with `/output-style`.
```

- [ ] **Step 2: Write `template/.github/copilot-instructions.md`**

```markdown
# Copilot instructions

Read and follow `AGENTS.md` at the repository root — it is the single source of
project rules, the quality gates, and the complexity limits. Skills live in
`.claude/skills/` (this repo enables them via `.vscode/settings.json`).
```

- [ ] **Step 3: Verify both files reference AGENTS.md**

Run: `grep -l "AGENTS.md" template/CLAUDE.md template/.github/copilot-instructions.md`
Expected: both paths printed.

- [ ] **Step 4: Commit**

```bash
git add template/CLAUDE.md template/.github/copilot-instructions.md
git commit -m "feat: CLAUDE.md + copilot-instructions pointers to AGENTS.md"
```

---

## Task 4: VS Code / Copilot wiring

**Files:**
- Create: `template/.vscode/settings.json`

- [ ] **Step 1: Write `template/.vscode/settings.json`**

```json
{
  "chat.useAgentsMdFile": true,
  "chat.useAgentSkills": true,
  "github.copilot.chat.codeGeneration.useInstructionFiles": true
}
```

- [ ] **Step 2: Verify it is valid JSON**

Run: `python -m json.tool template/.vscode/settings.json > /dev/null && echo OK`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add template/.vscode/settings.json
git commit -m "feat: VS Code Copilot wiring (AGENTS.md + agent skills)"
```

---

## Task 5: caveman (output-style + shared Copilot instruction)

**Files:**
- Create: `template/.claude/output-styles/caveman.md` (copied verbatim)
- Create: `template/.github/instructions/caveman.instructions.md` (symlink to the above)

- [ ] **Step 1: Copy the caveman output-style verbatim**

```bash
cp /home/trevor/Sportstradamus/.claude/output-styles/caveman.md \
   template/.claude/output-styles/caveman.md
```

- [ ] **Step 2: Create the Copilot instruction as a symlink to the same text**

The caveman rule text must exist for Copilot too. Symlink so it is edited once.

```bash
cd template/.github/instructions
ln -s ../../.claude/output-styles/caveman.md caveman.instructions.md
cd "$HOME/agent-project-template"
```

- [ ] **Step 3: Prepend Copilot frontmatter is NOT possible on a symlink target shared with Claude.**

Instead, verify the symlink resolves and document the apply scope in the README (Task 14). Copilot reads `*.instructions.md`; an `applyTo` frontmatter is optional and, when absent, the instruction applies broadly. Keeping the shared text without frontmatter is intentional so one file serves both agents.

Run: `readlink template/.github/instructions/caveman.instructions.md`
Expected: `../../.claude/output-styles/caveman.md`
Run: `test -f template/.github/instructions/caveman.instructions.md && echo "resolves"`
Expected: `resolves`

- [ ] **Step 4: Commit**

```bash
git add -A template/.claude/output-styles/caveman.md template/.github/instructions/caveman.instructions.md
git commit -m "feat: caveman style for both agents (shared text via symlink)"
```

---

## Task 6: Claude `settings.json` seed

Carries env, the generalized permission allowlist, hook wiring, the destructive-bash
guard (verbatim), the caveman output style, and Superpowers plugin enablement.

**Files:**
- Create: `template/.claude/settings.json`

- [ ] **Step 1: Copy the destructive-bash guard command string verbatim** from
`/home/trevor/Sportstradamus/.claude/settings.json` (the `PreToolUse` → `Bash` hook whose
command begins `bash -c 'CMD=$(jq -r ".tool_input.command")...`). It is language-agnostic;
do not edit it.

- [ ] **Step 2: Write `template/.claude/settings.json`** using that guard string for
`<<DESTRUCTIVE_GUARD_COMMAND>>`:

```json
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "env": {
    "CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR": "1"
  },
  "permissions": {
    "defaultMode": "plan",
    "allow": [
      "Bash(find:*)", "Bash(rg:*)", "Bash(grep:*)", "Bash(ls:*)", "Bash(cat:*)",
      "Bash(wc:*)", "Bash(sed:*)", "Bash(head:*)", "Bash(tail:*)", "Bash(sort:*)",
      "Bash(tree:*)", "Bash(stat:*)", "Bash(echo:*)", "Bash(printf:*)",
      "Bash(mkdir -p:*)", "Bash(jq:*)", "Bash(python -m json.tool:*)",
      "Bash(git status:*)", "Bash(git diff:*)", "Bash(git log:*)",
      "Bash(git branch --show-current:*)", "Bash(git rev-parse:*)",
      "Bash(git add:*)", "Bash(git commit:*)", "Bash(git show:*)",
      "Bash(pre-commit:*)", "Bash(lizard:*)", "Bash(ruff:*)",
      "WebSearch"
    ]
  },
  "hooks": {
    "PreToolUse": [
      { "matcher": "Bash", "hooks": [ { "type": "command", "command": "<<DESTRUCTIVE_GUARD_COMMAND>>" } ] },
      { "matcher": "Bash", "hooks": [ { "type": "command", "command": "python3 \"$CLAUDE_PROJECT_DIR/.claude/hooks/push-gate.py\"" } ] },
      { "matcher": "Task", "hooks": [ { "type": "command", "command": "python3 \"$CLAUDE_PROJECT_DIR/.claude/hooks/pretask-snapshot.py\"" } ] },
      { "matcher": "Edit|Write", "hooks": [ { "type": "command", "command": "python3 \"$CLAUDE_PROJECT_DIR/.claude/hooks/research-gate.py\"" } ] }
    ],
    "PostToolUse": [
      { "matcher": "Edit|Write|MultiEdit", "hooks": [
        { "type": "command", "command": "python3 \"$CLAUDE_PROJECT_DIR/.claude/hooks/complexity-gate.py\"" },
        { "type": "command", "command": "python3 \"$CLAUDE_PROJECT_DIR/.claude/hooks/mark-dirty.py\"" }
      ] },
      { "matcher": "Task", "hooks": [ { "type": "command", "command": "python3 \"$CLAUDE_PROJECT_DIR/.claude/hooks/posttask-diff.py\"" } ] }
    ]
  },
  "outputStyle": "Caveman",
  "extraKnownMarketplaces": {
    "claude-plugins-official": { "source": { "source": "git", "url": "https://github.com/anthropics/claude-plugins-official.git" } }
  },
  "enabledPlugins": { "superpowers@claude-plugins-official": true }
}
```

- [ ] **Step 3: Verify valid JSON**

Run: `python -m json.tool template/.claude/settings.json > /dev/null && echo OK`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add template/.claude/settings.json
git commit -m "feat: Claude settings seed (perms, hooks, caveman, superpowers)"
```

---

## Task 7: Port the subagent-drift hooks (verbatim) + smoke test

These two hooks are language-agnostic. Copy them byte-for-byte, then pin their behavior
with one smoke test so a future edit cannot silently break them.

**Files:**
- Create: `template/.claude/hooks/pretask-snapshot.py` (verbatim)
- Create: `template/.claude/hooks/posttask-diff.py` (verbatim)
- Create: `tests/conftest.py`
- Create: `tests/test_drift_hooks.py`

- [ ] **Step 1: Copy both hooks verbatim**

```bash
cp /home/trevor/Sportstradamus/.claude/hooks/pretask-snapshot.py template/.claude/hooks/
cp /home/trevor/Sportstradamus/.claude/hooks/posttask-diff.py   template/.claude/hooks/
```

- [ ] **Step 2: Write `tests/conftest.py`** (shared fixture pointing at the payload hooks)

```python
import pathlib
import pytest

HOOKS = pathlib.Path(__file__).resolve().parent.parent / "template" / ".claude" / "hooks"


@pytest.fixture
def hooks_dir() -> pathlib.Path:
    return HOOKS
```

- [ ] **Step 3: Write the failing test `tests/test_drift_hooks.py`**

```python
import importlib.util
import json
import subprocess
import sys


def _run(hook_path, payload):
    return subprocess.run(
        [sys.executable, str(hook_path)],
        input=json.dumps(payload), capture_output=True, text=True,
    )


def test_posttask_diff_reports_moved_head(hooks_dir, tmp_path, monkeypatch):
    # A pretask snapshot with a different HEAD than current should surface a drift note.
    state = tmp_path / ".claude" / ".state"
    state.mkdir(parents=True)
    (state / "pretask-sess1.json").write_text(json.dumps(
        {"head": "aaaaaaaa", "branch": "main", "status": ""}))
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path))
    # git in tmp_path returns empty HEAD -> differs from 'aaaaaaaa' -> drift message.
    subprocess.run(["git", "init", "-q"], cwd=tmp_path)
    result = _run(hooks_dir / "posttask-diff.py", {"session_id": "sess1"})
    assert result.returncode == 0
    assert "subagent-drift-check" in result.stdout
```

- [ ] **Step 4: Run the test to verify it passes (verbatim hooks already work)**

Run: `python -m pytest tests/test_drift_hooks.py -q`
Expected: PASS (1 passed). If it fails, the copy was not verbatim — re-copy.

- [ ] **Step 5: Commit**

```bash
git add template/.claude/hooks/pretask-snapshot.py template/.claude/hooks/posttask-diff.py tests/
git commit -m "feat: port subagent-drift hooks verbatim + smoke test"
```

---

## Task 8: Generalize push-gate + mark-dirty (TDD)

The mechanism (block a push when the test gate is stale since the last code edit) is
kept; the only change is broadening "code file" beyond `.py` and genericizing the
message. `decide()` is already pure and stays.

**Files:**
- Create: `template/.claude/hooks/mark-dirty.py`
- Create: `template/.claude/hooks/push-gate.py`
- Create: `tests/test_push_gate.py`

- [ ] **Step 1: Write the failing test `tests/test_push_gate.py`**

```python
import importlib.util
import sys


def _load(hooks_dir, module_name, filename):
    spec = importlib.util.spec_from_file_location(module_name, hooks_dir / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_mark_dirty_recognizes_many_source_suffixes(hooks_dir):
    md = _load(hooks_dir, "mark_dirty", "mark-dirty.py")
    assert md.is_code_file("a/b/c.py")
    assert md.is_code_file("src/foo.cpp")
    assert md.is_code_file("include/foo.hpp")
    assert md.is_code_file("web/app.ts")
    assert not md.is_code_file("README.md")
    assert not md.is_code_file("data.csv")


def test_push_gate_decide(hooks_dir):
    pg = _load(hooks_dir, "push_gate", "push-gate.py")
    assert pg.decide(is_push=False, dirty=10.0, green=None) == "allow"
    assert pg.decide(is_push=True, dirty=None, green=None) == "allow"
    assert pg.decide(is_push=True, dirty=20.0, green=10.0) == "ask"   # edited after green
    assert pg.decide(is_push=True, dirty=5.0, green=10.0) == "allow"  # green after edit
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_push_gate.py -q`
Expected: FAIL (module files do not exist yet).

- [ ] **Step 3: Write `template/.claude/hooks/mark-dirty.py`**

```python
#!/usr/bin/env python3
"""PostToolUse[Edit|Write] hook: mark the tree code-dirty when a source file
changes, so the push-gate knows the test gate is stale."""
from __future__ import annotations

import json
import os
import pathlib
import sys

SOURCE_SUFFIXES = (
    ".py", ".pyi", ".c", ".h", ".cc", ".cpp", ".cxx", ".hh", ".hpp",
    ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java",
)


def is_code_file(path: str) -> bool:
    return path.endswith(SOURCE_SUFFIXES)


def main() -> None:
    payload = json.load(sys.stdin)
    path = payload.get("tool_input", {}).get("file_path", "")
    if is_code_file(path):
        state = pathlib.Path(os.environ.get("CLAUDE_PROJECT_DIR", ".")) / ".claude" / ".state"
        state.mkdir(parents=True, exist_ok=True)
        (state / "code_dirty").touch()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write `template/.claude/hooks/push-gate.py`**

```python
#!/usr/bin/env python3
"""PreToolUse[Bash] hook: ask to confirm a git push when the project's test gate
has not passed since the last source edit."""
from __future__ import annotations

import json
import os
import pathlib
import sys


def is_git_push(command: str) -> bool:
    parts = command.split()
    return "git" in parts and "push" in parts


def decide(is_push: bool, dirty: float | None, green: float | None) -> str:
    if not is_push or dirty is None:
        return "allow"
    if green is None or dirty > green:
        return "ask"
    return "allow"


def _mtime(path: pathlib.Path) -> float | None:
    return path.stat().st_mtime if path.exists() else None


def main() -> None:
    payload = json.load(sys.stdin)
    command = payload.get("tool_input", {}).get("command", "")
    state = pathlib.Path(os.environ.get("CLAUDE_PROJECT_DIR", ".")) / ".claude" / ".state"
    decision = decide(is_git_push(command), _mtime(state / "code_dirty"), _mtime(state / "integration_green"))
    if decision == "ask":
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "ask",
            "permissionDecisionReason": (
                "The test gate has not passed since your last code edit. Run the "
                "project's test command (see AGENTS.md) then "
                "`touch \"$CLAUDE_PROJECT_DIR/.claude/.state/integration_green\"` "
                "before pushing, or confirm to push anyway."
            ),
        }}))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_push_gate.py -q`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add template/.claude/hooks/mark-dirty.py template/.claude/hooks/push-gate.py tests/test_push_gate.py
git commit -m "feat: generalize push-gate + mark-dirty to multi-language sources"
```

---

## Task 9: Complexity-gate via lizard (TDD)

In-loop, non-blocking nudge when an edited file has a function over the hard ceiling.
Pre-commit (Task 11) is the authoritative block; this is the fast editor signal.
Thresholds are env-overridable so tests can trigger them trivially.

**Files:**
- Create: `template/.claude/hooks/complexity-gate.py`
- Create: `tests/test_complexity_gate.py`

- [ ] **Step 1: Write the failing test `tests/test_complexity_gate.py`**

```python
import importlib.util
import shutil
import sys

import pytest


def _load(hooks_dir):
    spec = importlib.util.spec_from_file_location("complexity_gate", hooks_dir / "complexity-gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


COMPLEX_FN = '''
def f(x):
    if x == 1: return 1
    if x == 2: return 2
    if x == 3: return 3
    return 0
'''


@pytest.mark.skipif(shutil.which("lizard") is None, reason="lizard not installed")
def test_violation_when_ceiling_is_low(hooks_dir, tmp_path, monkeypatch):
    mod = _load(hooks_dir)
    src = tmp_path / "sample.py"
    src.write_text(COMPLEX_FN)
    monkeypatch.setenv("COMPLEXITY_CCN_CEILING", "1")
    assert mod.violations(str(src)) != []


@pytest.mark.skipif(shutil.which("lizard") is None, reason="lizard not installed")
def test_no_violation_under_high_ceiling(hooks_dir, tmp_path, monkeypatch):
    mod = _load(hooks_dir)
    src = tmp_path / "sample.py"
    src.write_text(COMPLEX_FN)
    monkeypatch.setenv("COMPLEXITY_CCN_CEILING", "99")
    monkeypatch.setenv("COMPLEXITY_NLOC_CEILING", "9999")
    assert mod.violations(str(src)) == []


def test_non_source_file_ignored(hooks_dir):
    mod = _load(hooks_dir)
    assert mod.violations("notes.md") == []
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_complexity_gate.py -q`
Expected: FAIL (module does not exist yet).

- [ ] **Step 3: Write `template/.claude/hooks/complexity-gate.py`**

```python
#!/usr/bin/env python3
"""PostToolUse[Edit|Write] hook: in-loop complexity nudge via lizard.

Emits a non-blocking reminder (exit 2) when an edited source file has a function
over the hard ceiling (cyclomatic complexity or length). The pre-commit gate is the
authoritative block; this is the fast in-editor signal. Thresholds are
env-overridable (COMPLEXITY_CCN_CEILING / COMPLEXITY_NLOC_CEILING)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

CCN_CEILING = 49
NLOC_CEILING = 200
SOURCE_SUFFIXES = (
    ".py", ".pyi", ".c", ".h", ".cc", ".cpp", ".cxx", ".hh", ".hpp",
    ".ts", ".tsx", ".js", ".jsx",
)


def violations(path: str) -> list[str]:
    if not path.endswith(SOURCE_SUFFIXES) or shutil.which("lizard") is None:
        return []
    ccn = os.environ.get("COMPLEXITY_CCN_CEILING", str(CCN_CEILING))
    nloc = os.environ.get("COMPLEXITY_NLOC_CEILING", str(NLOC_CEILING))
    proc = subprocess.run(
        ["lizard", "-T", f"cyclomatic_complexity={ccn}", "-T", f"nloc={nloc}", "-w", path],
        capture_output=True, text=True,
    )
    return [ln for ln in proc.stdout.splitlines() if ln.strip()]


def main() -> int:
    payload = json.load(sys.stdin)
    path = payload.get("tool_input", {}).get("file_path", "")
    found = violations(path)
    if found:
        sys.stderr.write(
            "[complexity-gate] hard ceiling exceeded (CC>{} or NLOC>{}):\n{}\n"
            "Refactor before commit — pre-commit will block.\n".format(
                CCN_CEILING, NLOC_CEILING, "\n".join(found)
            )
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_complexity_gate.py -q`
Expected: PASS (3 passed; or 1 passed + 2 skipped if lizard is somehow absent — but Task 1 installed it).

- [ ] **Step 5: Commit**

```bash
git add template/.claude/hooks/complexity-gate.py tests/test_complexity_gate.py
git commit -m "feat: lizard-based multi-language complexity-gate hook"
```

---

## Task 10: Generalize research-gate (TDD) + seed config

Drop the Sportstradamus `stat_meta.json` shipped-flip special-case. Keep the
config-driven glob gating + brief/waiver escape hatch.

**Files:**
- Create: `template/.claude/hooks/research-gate.py`
- Create: `template/research_gated.txt`
- Create: `tests/test_research_gate.py`

- [ ] **Step 1: Write the failing test `tests/test_research_gate.py`**

```python
import importlib.util


def _load(hooks_dir):
    spec = importlib.util.spec_from_file_location("research_gate", hooks_dir / "research-gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_is_gated_matches_globs(hooks_dir):
    mod = _load(hooks_dir)
    globs = ["**/msg/*.msg", "src/core/*.py"]
    assert mod.is_gated({"file_path": "robot/msg/Pose.msg"}, globs)
    assert mod.is_gated({"file_path": "src/core/control.py"}, globs)
    assert not mod.is_gated({"file_path": "src/util/log.py"}, globs)


def test_empty_globs_gate_nothing(hooks_dir):
    mod = _load(hooks_dir)
    assert not mod.is_gated({"file_path": "anything.py"}, [])
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_research_gate.py -q`
Expected: FAIL (module does not exist yet).

- [ ] **Step 3: Write `template/.claude/hooks/research-gate.py`**

```python
#!/usr/bin/env python3
"""PreToolUse[Edit|Write] hook: block an edit to a research-gated path when no
research brief and no waiver exist. The agent then writes a brief (to
/tmp/researcher_<topic>.md) or a one-line .claude/.state/research_waiver."""
from __future__ import annotations

import fnmatch
import glob
import json
import os
import pathlib
import sys


def project_dir() -> pathlib.Path:
    return pathlib.Path(os.environ.get("CLAUDE_PROJECT_DIR", "."))


def gated_globs() -> list[str]:
    cfg = project_dir() / "research_gated.txt"
    if not cfg.exists():
        return []
    return [ln.strip() for ln in cfg.read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]


def is_gated(tool_input: dict, globs: list[str]) -> bool:
    path = tool_input.get("file_path", "")
    return any(fnmatch.fnmatch(path, g) for g in globs)


def brief_or_waiver_exists() -> bool:
    waiver = project_dir() / ".claude" / ".state" / "research_waiver"
    return bool(glob.glob("/tmp/researcher_*.md")) or waiver.exists()


def main() -> None:
    payload = json.load(sys.stdin)
    tool_input = payload.get("tool_input", {})
    if is_gated(tool_input, gated_globs()) and not brief_or_waiver_exists():
        sys.stderr.write(
            "research-gated change ({}). Do the research first and write "
            "/tmp/researcher_<topic>.md, OR write a one-line justification to "
            ".claude/.state/research_waiver, then retry this edit.".format(
                tool_input.get("file_path", "?")
            )
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write `template/research_gated.txt`** (empty seed — gate off until populated)

```text
# Globs of architecturally-significant / hard-to-reverse paths whose edits require a
# research brief first (see AGENTS.md "Research-driven development"). One glob per line.
# Empty = gate is off. Examples:
# **/msg/*.msg
# **/srv/*.srv
# src/<core_module>/**
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_research_gate.py -q`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add template/.claude/hooks/research-gate.py template/research_gated.txt tests/test_research_gate.py
git commit -m "feat: generalize research-gate to config-driven glob gating"
```

---

## Task 11: Pre-commit hard gate (generic seed)

The agent-independent enforcement layer. The generic seed runs ruff (Python) when
present and the lizard hard ceiling for all languages. Stack-specific hooks
(clang-format, eslint, etc.) are added by Part 2's `setup-project`; this seed is the
floor that always applies.

**Files:**
- Create: `template/.pre-commit-config.yaml`

- [ ] **Step 1: Write `template/.pre-commit-config.yaml`**

```yaml
# Cross-agent hard gate. Fires for Claude, Copilot, and human commits.
# setup-project (Part 2) adds stack-specific hooks; this is the always-on floor.
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: local
    hooks:
      - id: lizard-hard-ceiling
        name: lizard complexity/length hard ceiling (CC<=49, NLOC<=200)
        entry: lizard -T cyclomatic_complexity=49 -T nloc=200 -w
        language: python
        additional_dependencies: [lizard]
        types_or: [python, c, c++]
        pass_filenames: true
```

- [ ] **Step 2: Verify the config parses**

Run: `cd template && pre-commit validate-config .pre-commit-config.yaml && echo OK; cd ..`
Expected: `OK` (no validation errors).

- [ ] **Step 3: Functional check — the ceiling actually blocks**

Run:
```bash
tmp=$(mktemp -d); cd "$tmp"; git init -q
cp "$HOME/agent-project-template/template/.pre-commit-config.yaml" .
printf 'def f(x):\n%s    return 0\n' "$(for i in $(seq 1 60); do echo "    if x==$i: return $i"; done)" > big.py
git add -A
pre-commit run lizard-hard-ceiling --files big.py; echo "exit=$?"
cd "$HOME/agent-project-template"; rm -rf "$tmp"
```
Expected: the `lizard-hard-ceiling` hook FAILS (non-zero), naming `f` as over the CC ceiling.

- [ ] **Step 4: Commit**

```bash
git add template/.pre-commit-config.yaml
git commit -m "feat: pre-commit hard gate (ruff + lizard ceiling)"
```

---

## Task 12: Agents (generalized) + Copilot mirrors

**Files:**
- Create: `template/.claude/agents/refactoring-specialist.md`
- Create: `template/.claude/agents/research-analyst.md`
- Create: `template/.claude/agents/prompt-engineer.md` (copied)
- Create: `template/.github/agents/refactoring-specialist.md`
- Create: `template/.github/agents/research-analyst.md`

- [ ] **Step 1: Copy the prompt-engineer agent verbatim** (already generic)

```bash
cp /home/trevor/Sportstradamus/.claude/agents/prompt-engineer.md template/.claude/agents/
```

- [ ] **Step 2: Write `template/.claude/agents/refactoring-specialist.md`** (generalized)

```markdown
---
name: refactoring-specialist
description: "Use to enforce the project's code doctrine (AGENTS.md) on files touched this session before pushing or opening a PR. Reviews and refactors for orchestrator flatness, wrapper-function elimination, duplicate-code consolidation, loop sanity, and helper placement — without changing behavior. Runs the formatter/linter on its scope only; the main agent owns the authoritative gate run."
tools: Read, Edit, Write, Bash, Glob, Grep
model: sonnet
---

You enforce AGENTS.md on a handed-in list of files. Do not scan the whole repo on
your own; the main agent hands you the scope. For each file:

- Flatten needless orchestration; inline wrappers that only rename and forward a call.
- Consolidate duplicated logic into one helper; do not copy a third time.
- Keep functions deep and coherent, not split into thin fragments.
- Respect the complexity limits (CC ≤ 10 untagged / ≤ 49 tagged, nesting ≤ 4, ≤ 200 lines).
- Place helpers where they belong (one file, one responsibility).

Do not change behavior. Run the project's formatter/linter on the files you touched.
Report what you changed and anything you intentionally left.
```

- [ ] **Step 3: Write `template/.claude/agents/research-analyst.md`** (generalized)

```markdown
---
name: research-analyst
description: "Use when a decision is architecturally significant or hard to reverse, or a result is ambiguous and the path forward needs evidence. Reads the local context, searches the literature/web, and writes a short cited brief to /tmp/researcher_<topic>.md. Read-only with respect to production code."
tools: Read, Grep, Glob, Bash, WebSearch, WebFetch, Write
model: opus
---

You are a careful analyst. Given a flagged decision:

1. Read the relevant local files and restate the decision and the options.
2. Search the literature/web for evidence; prefer primary sources.
3. Weigh trade-offs honestly; state uncertainty where it exists.
4. Write a cited brief to `/tmp/researcher_<topic>.md` with a clear recommendation.

Do not modify production code. The brief is the deliverable; the main agent decides.
```

- [ ] **Step 4: Create the Copilot agent mirrors** (same generalized bodies, Copilot frontmatter)

Write `template/.github/agents/refactoring-specialist.md` and
`template/.github/agents/research-analyst.md` with the SAME body text as Steps 2–3, but
replace the frontmatter with Copilot's agent-profile form:

```markdown
---
description: "<same description as the Claude agent>"
tools: ['edit', 'search', 'runCommands']
---
```
(Use the matching description per agent; keep the body identical to the Claude version.)

- [ ] **Step 5: Verify all five agent files have frontmatter**

Run: `for f in template/.claude/agents/*.md template/.github/agents/*.md; do head -1 "$f" | grep -q '^---' && echo "ok $f" || echo "MISSING $f"; done`
Expected: `ok` for all five.

- [ ] **Step 6: Commit**

```bash
git add template/.claude/agents template/.github/agents
git commit -m "feat: generalized refactoring + research agents (Claude + Copilot)"
```

---

## Task 13: End-to-end smoke — copy payload into a fixture repo

Prove the payload works when dropped into a fresh repo: pre-commit installs and blocks,
and the Claude hooks fire on simulated payloads.

**Files:**
- Create: `tests/test_payload_smoke.py`

- [ ] **Step 1: Write `tests/test_payload_smoke.py`**

```python
import json
import shutil
import subprocess
import sys
import pathlib

PAYLOAD = pathlib.Path(__file__).resolve().parent.parent / "template"


def test_payload_drops_in_and_hooks_fire(tmp_path):
    repo = tmp_path / "demo"
    repo.mkdir()
    # Copy the drop-in payload (including dotfiles) into the fresh repo.
    subprocess.run(["bash", "-c", f"cp -r {PAYLOAD}/. {repo}/"], check=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    assert (repo / "AGENTS.md").exists()
    assert (repo / ".claude" / "settings.json").exists()

    # research-gate blocks a gated edit with no brief/waiver.
    (repo / "research_gated.txt").write_text("**/*.secret\n")
    res = subprocess.run(
        [sys.executable, str(repo / ".claude" / "hooks" / "research-gate.py")],
        input=json.dumps({"tool_input": {"file_path": "x.secret"}}),
        capture_output=True, text=True,
        env={"CLAUDE_PROJECT_DIR": str(repo), "PATH": __import__("os").environ["PATH"]},
    )
    assert res.returncode == 2 and "research-gated" in res.stderr
```

- [ ] **Step 2: Run it to verify it passes**

Run: `python -m pytest tests/test_payload_smoke.py -q`
Expected: PASS (1 passed).

- [ ] **Step 3: Run the full test suite**

Run: `python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_payload_smoke.py
git commit -m "test: end-to-end payload drop-in smoke"
```

---

## Task 14: Finalize the README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace `README.md` with the full how-to**

```markdown
# Cross-Agent Project Template

Drop-in setup giving any repo a consistent agentic workflow in **Claude Code** and
**GitHub Copilot (VS Code)** from the same files: shared rules (`AGENTS.md`), shared
skills (`.claude/skills/`, read by both agents), a cross-agent hard gate (pre-commit),
Claude in-loop hooks, and the caveman output style.

## What you get

- `AGENTS.md` — the single source of project rules; `CLAUDE.md` and
  `.github/copilot-instructions.md` point to it.
- Complexity limits enforced (CC ≤ 10 untagged / ≤ 49 tagged, nesting ≤ 4, ≤ 200 lines)
  via pre-commit (lizard) + an in-loop Claude hook.
- Subagent-drift detection, a push gate, a destructive-command guard, and a
  research-first gate (Claude hooks).
- caveman style for both agents.

## Install (manual — Part 1)

1. Copy the contents of `template/` into your repo root (include dotfiles):
   `cp -r template/. /path/to/your/repo/`
2. Edit `AGENTS.md`: fill in name, purpose, key dirs, and the format/lint/test commands.
3. `pip install pre-commit lizard && pre-commit install`
4. Open the repo in Claude Code or Copilot. Both read `AGENTS.md` and `.claude/skills/`.

## Superpowers (optional, per agent)

- Claude Code: enabled via `.claude/settings.json` (official plugin marketplace).
- GitHub Copilot: install the dwaintr "Superpowers for Copilot Chat" marketplace
  extension, or the faulkdev skills port.

## Coming in Part 2

A `setup-project` skill that auto-detects your stack (Python / JS-TS / C++ / ROS 2 /
generic), tailors `AGENTS.md`, the permission allowlist, the gate commands, and the
stack-specific pre-commit hooks — so step 2 above is automated.

## Developing this template

`pip install -e ".[dev]"` then `python -m pytest -q`. The tests exercise the shipped
Python hooks under `template/.claude/hooks/`.
```

- [ ] **Step 2: Verify the suite still passes and commit**

Run: `python -m pytest -q`
Expected: all pass.

```bash
git add README.md
git commit -m "docs: finalize template README (Part 1)"
```

---

## Self-Review

**Spec coverage (against the design spec's components):**
- Unified layout, shared core (AGENTS.md, `.claude/skills/`, pre-commit) → Tasks 1, 2, 11; skills home seeded in Task 1 (populated in Part 2).
- Claude adapter (settings, hooks, agents, output-style) → Tasks 5, 6, 7, 8, 9, 10, 12.
- Copilot adapter (copilot-instructions, caveman instruction, agent mirrors, .vscode) → Tasks 3, 4, 5, 12.
- Complexity enforcement (ruff + lizard ceiling; clang-tidy is added by Part 2's C++/ROS2 recipe) → Tasks 9, 11.
- Research-driven development (gate + config + agent) → Tasks 10, 12.
- caveman + superpowers (documented) → Tasks 5, 6, 14.
- **Deferred to Part 2 (by design):** the `setup-project` skill, `detect_stack`, the stack-recipe table, clang-tidy/ROS2 provisioning, tailoring logic. The `references/` dir and `.claude/skills/` home are seeded here.

**Placeholder scan:** `AGENTS.md` intentionally ships bracketed placeholders (`[PROJECT NAME]`, gate commands) — these are template fill-ins the user/Part-2 completes, not plan placeholders. Every code/test step contains complete code. No "TBD"/"similar to Task N".

**Type/name consistency:** `is_code_file` (Task 8), `violations` (Task 9), `is_gated`/`brief_or_waiver_exists`/`gated_globs` (Task 10), `decide`/`is_git_push` (Task 8) are used consistently in their tests. `SOURCE_SUFFIXES` defined per-hook (mark-dirty and complexity-gate intentionally keep their own, slightly different, suffix sets). The `conftest.py` `hooks_dir` fixture is used by every hook test.

**Note on Task 5:** the symlinked caveman instruction has no Copilot frontmatter (one file serves both agents). If, during execution, Copilot is found to require frontmatter on `*.instructions.md`, fall back to a small generation step in Part 2 (copy + prepend frontmatter) rather than symlinking — recorded in the spec's risks.
