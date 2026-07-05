# Cross-Agent Dev Template — Part 2 (setup-project init skill) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline) or superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add the `setup-project` Agent Skill that auto-detects a repo's stack and tailors the Part 1 payload (AGENTS.md, permission allowlist, gate commands, stack-specific pre-commit hooks, provisioning).

**Architecture:** A skill folder at `template/.claude/skills/setup-project/` (read by both Claude Code and Copilot). Detection is a small, testable Python helper (`scripts/detect_stack.py`) emitting JSON; the recipe knowledge is documentation (`references/stacks.md`); the tailoring is the agent following `SKILL.md` prose using that JSON + recipes (no framework, per the spec). First-class stacks: Python, JavaScript/TypeScript, C++/CMake, ROS 2 (ament_cmake / ament_python); generic fallback otherwise.

**Tech Stack:** Python 3.11 (detection + pytest), the Agent Skills `SKILL.md` standard.

**Conventions:** `TEMPLATE_ROOT` = `$HOME/agent-project-template`; all work there. Each task ends with a commit. Reuse the existing `tests/conftest.py` style (add a `skill_dir` fixture).

---

## File Structure (created in `TEMPLATE_ROOT`)

```
template/.claude/skills/setup-project/
  SKILL.md                 # skill body (frontmatter + prose orchestration)  (Task 5)
  scripts/detect_stack.py  # testable detection -> JSON                       (Task 1-3)
  references/stacks.md     # recipe table the skill applies                   (Task 4)
tests/
  test_detect_stack.py     # TDD for detection                               (Task 1-3)
```

---

## Task 1: detect_stack.py — core + Python + generic (TDD)

**Files:**
- Create: `template/.claude/skills/setup-project/scripts/detect_stack.py`
- Create: `tests/test_detect_stack.py`

- [ ] **Step 1: Create the skill dirs**

```bash
R="$HOME/agent-project-template"
mkdir -p "$R/template/.claude/skills/setup-project/scripts" "$R/template/.claude/skills/setup-project/references"
```

- [ ] **Step 2: Write the failing test `tests/test_detect_stack.py`**

```python
import importlib.util
import pathlib

SCRIPT = (pathlib.Path(__file__).resolve().parent.parent
          / "template" / ".claude" / "skills" / "setup-project" / "scripts" / "detect_stack.py")


def _mod():
    spec = importlib.util.spec_from_file_location("detect_stack", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generic_when_no_markers(tmp_path):
    d = _mod().detect(tmp_path)
    assert d["stacks"] == []
    assert d["commands"]["test"].startswith("<")  # placeholder for the user
    assert d["package_manager"] is None


def test_python_poetry(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.poetry]\n")
    (tmp_path / "poetry.lock").write_text("")
    (tmp_path / "app.py").write_text("x = 1\n")
    d = _mod().detect(tmp_path)
    assert d["stacks"] == ["python"]
    assert d["package_manager"] == "poetry"
    assert d["commands"]["lint"] == "ruff check ."
    assert d["has_sources"] is True


def test_python_pip_default(tmp_path):
    (tmp_path / "setup.py").write_text("from setuptools import setup\n")
    d = _mod().detect(tmp_path)
    assert d["stacks"] == ["python"]
    assert d["package_manager"] == "pip"


def test_existing_inventory_flags_prior_tooling(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("# rules\n")
    skill = tmp_path / ".claude" / "skills" / "foo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: foo\n---\n")
    (tmp_path / ".pre-commit-config.yaml").write_text("repos: []\n")
    e = _mod().detect(tmp_path)["existing"]
    assert e["claude_md"] is True
    assert e["claude_skills"] is True
    assert e["precommit"] is True
    assert e["agents_md"] is False
```

- [ ] **Step 3: Run it to verify it fails**

Run: `cd "$HOME/agent-project-template" && python3 -m pytest tests/test_detect_stack.py -q`
Expected: FAIL (script does not exist).

- [ ] **Step 4: Write `template/.claude/skills/setup-project/scripts/detect_stack.py`**

```python
#!/usr/bin/env python3
"""Detect a repo's stack and quality-gate commands for the setup-project skill.

Usage: python3 detect_stack.py [ROOT]  ->  prints JSON to stdout. The skill
consumes this JSON, then tailors AGENTS.md, the permission allowlist, the
pre-commit config, and provisioning per references/stacks.md."""
from __future__ import annotations

import json
import pathlib
import sys

SOURCE_SUFFIXES = (
    ".py", ".pyi", ".c", ".h", ".cc", ".cpp", ".cxx", ".hh", ".hpp",
    ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java",
)

COMMANDS = {
    "python": {"test": "pytest", "lint": "ruff check .", "format": "ruff format ."},
    "javascript": {"test": "npm test", "lint": "eslint .", "format": "prettier --check ."},
    "cpp_cmake": {
        "build": "cmake -S . -B build && cmake --build build",
        "test": "ctest --test-dir build", "lint": "clang-tidy", "format": "clang-format -i",
    },
    "ros2_ament_cmake": {
        "build": "colcon build", "test": "colcon test && colcon test-result --verbose",
        "lint": "clang-tidy / ament_lint", "format": "clang-format -i",
    },
    "ros2_ament_python": {
        "build": "colcon build", "test": "colcon test && colcon test-result --verbose",
        "lint": "ruff check . / ament_flake8", "format": "ruff format .",
    },
    "generic": {
        "test": "<your test command>", "lint": "<your lint command>",
        "format": "<your format command>",
    },
}


def _exists(root: pathlib.Path, *names: str) -> bool:
    return any((root / n).exists() for n in names)


def detect_python_pm(root: pathlib.Path) -> str:
    if (root / "poetry.lock").exists():
        return "poetry"
    if (root / "uv.lock").exists():
        return "uv"
    return "pip"


def detect_js_pm(root: pathlib.Path) -> str:
    if (root / "pnpm-lock.yaml").exists():
        return "pnpm"
    if (root / "yarn.lock").exists():
        return "yarn"
    return "npm"


def detect_stacks(root: pathlib.Path) -> list[str]:
    stacks: list[str] = []
    if (root / "package.xml").exists():
        if _exists(root, "CMakeLists.txt"):
            stacks.append("ros2_ament_cmake")
        if _exists(root, "setup.py", "setup.cfg"):
            stacks.append("ros2_ament_python")
        if not stacks:
            stacks.append("ros2_ament_cmake")
    elif (root / "CMakeLists.txt").exists():
        stacks.append("cpp_cmake")
    if _exists(root, "pyproject.toml", "setup.py") and not any(s.startswith("ros2") for s in stacks):
        stacks.append("python")
    if (root / "package.json").exists():
        stacks.append("javascript")
    return stacks


def has_sources(root: pathlib.Path) -> bool:
    return any(
        p.is_file() and p.suffix in SOURCE_SUFFIXES and ".git" not in p.parts
        for p in root.rglob("*")
    )


def _nonempty_dir(p: pathlib.Path) -> bool:
    return p.is_dir() and any(p.iterdir())


def detect_existing(root: pathlib.Path) -> dict:
    """Inventory pre-existing agent-framework artifacts so the skill can ask the
    user whether to overwrite them or fold the template in."""
    return {
        "claude_md": (root / "CLAUDE.md").exists(),
        "agents_md": (root / "AGENTS.md").exists(),
        "claude_settings": (root / ".claude" / "settings.json").exists(),
        "claude_hooks": _nonempty_dir(root / ".claude" / "hooks"),
        "claude_agents": _nonempty_dir(root / ".claude" / "agents"),
        "claude_skills": _nonempty_dir(root / ".claude" / "skills"),
        "copilot_instructions": (root / ".github" / "copilot-instructions.md").exists(),
        "github_skills": _nonempty_dir(root / ".github" / "skills"),
        "github_prompts": _nonempty_dir(root / ".github" / "prompts"),
        "github_instructions": _nonempty_dir(root / ".github" / "instructions"),
        "github_chatmodes": _nonempty_dir(root / ".github" / "chatmodes"),
        "vscode_settings": (root / ".vscode" / "settings.json").exists(),
        "mcp": (root / ".vscode" / "mcp.json").exists() or (root / ".mcp.json").exists(),
        "precommit": (root / ".pre-commit-config.yaml").exists(),
        "ci": _nonempty_dir(root / ".github" / "workflows"),
    }


def detect(root: str | pathlib.Path) -> dict:
    root = pathlib.Path(root)
    stacks = detect_stacks(root)
    pm = None
    if "python" in stacks or "ros2_ament_python" in stacks:
        pm = detect_python_pm(root)
    elif "javascript" in stacks:
        pm = detect_js_pm(root)
    primary = stacks[0] if stacks else "generic"
    return {
        "root": str(root),
        "is_git": (root / ".git").exists(),
        "has_sources": has_sources(root),
        "stacks": stacks,
        "package_manager": pm,
        "commands": COMMANDS.get(primary, COMMANDS["generic"]),
        "existing": detect_existing(root),
    }


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    print(json.dumps(detect(root), indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd "$HOME/agent-project-template" && python3 -m pytest tests/test_detect_stack.py -q`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
R="$HOME/agent-project-template"
git -C "$R" add template/.claude/skills/setup-project/scripts/detect_stack.py tests/test_detect_stack.py
git -C "$R" commit -m "feat: detect_stack.py core + Python/generic detection"
```

---

## Task 2: detect_stack.py — JavaScript/TypeScript (TDD)

**Files:**
- Modify: `tests/test_detect_stack.py` (add cases)

The implementation already covers JS via `detect_stacks`/`detect_js_pm`; this task pins it with tests.

- [ ] **Step 1: Append tests to `tests/test_detect_stack.py`**

```python
def test_javascript_pnpm(tmp_path):
    (tmp_path / "package.json").write_text('{"scripts":{"test":"vitest"}}')
    (tmp_path / "pnpm-lock.yaml").write_text("")
    d = _mod().detect(tmp_path)
    assert d["stacks"] == ["javascript"]
    assert d["package_manager"] == "pnpm"
    assert d["commands"]["lint"] == "eslint ."


def test_javascript_npm_default(tmp_path):
    (tmp_path / "package.json").write_text("{}")
    d = _mod().detect(tmp_path)
    assert d["package_manager"] == "npm"
```

- [ ] **Step 2: Run the tests**

Run: `cd "$HOME/agent-project-template" && python3 -m pytest tests/test_detect_stack.py -q`
Expected: PASS (5 passed).

- [ ] **Step 3: Commit**

```bash
R="$HOME/agent-project-template"
git -C "$R" add tests/test_detect_stack.py
git -C "$R" commit -m "test: pin JavaScript/TypeScript stack detection"
```

---

## Task 3: detect_stack.py — C++/CMake + ROS 2 ament (TDD)

**Files:**
- Modify: `tests/test_detect_stack.py` (add cases)

Implementation already covers these; pin with tests, including the ROS 2 precedence rule (a `package.xml` makes it ROS 2, not plain Python/CMake).

- [ ] **Step 1: Append tests to `tests/test_detect_stack.py`**

```python
def test_cpp_cmake_plain(tmp_path):
    (tmp_path / "CMakeLists.txt").write_text("project(demo)\n")
    (tmp_path / "main.cpp").write_text("int main(){return 0;}\n")
    d = _mod().detect(tmp_path)
    assert d["stacks"] == ["cpp_cmake"]
    assert d["commands"]["test"] == "ctest --test-dir build"


def test_ros2_ament_cmake(tmp_path):
    (tmp_path / "package.xml").write_text("<package/>\n")
    (tmp_path / "CMakeLists.txt").write_text("project(node)\n")
    d = _mod().detect(tmp_path)
    assert d["stacks"] == ["ros2_ament_cmake"]
    assert d["commands"]["build"] == "colcon build"


def test_ros2_ament_python(tmp_path):
    (tmp_path / "package.xml").write_text("<package/>\n")
    (tmp_path / "setup.py").write_text("from setuptools import setup\n")
    d = _mod().detect(tmp_path)
    assert "ros2_ament_python" in d["stacks"]
    assert "python" not in d["stacks"]  # package.xml -> ROS 2, not plain Python
    assert d["package_manager"] == "pip"
```

- [ ] **Step 2: Run the tests**

Run: `cd "$HOME/agent-project-template" && python3 -m pytest tests/test_detect_stack.py -q`
Expected: PASS (8 passed).

- [ ] **Step 3: Commit**

```bash
R="$HOME/agent-project-template"
git -C "$R" add tests/test_detect_stack.py
git -C "$R" commit -m "test: pin C++/CMake + ROS 2 ament detection and precedence"
```

---

## Task 4: stacks.md recipe reference

**Files:**
- Create: `template/.claude/skills/setup-project/references/stacks.md`

- [ ] **Step 1: Write `references/stacks.md`**

````markdown
# Stack recipes

`detect_stack.py` classifies the repo; the setup-project skill applies the matching
row here. Adding a stack = adding a row (no code change unless a new detection marker
is needed).

| Stack | Detect marker | Build / Test | Lint / Format | pre-commit (fast) | Permissions to add | Provision |
|---|---|---|---|---|---|---|
| python | `pyproject.toml` / `setup.py` | `pytest` | `ruff check .` / `ruff format .` | ruff, ruff-format | `Bash(pytest:*)`, `Bash(ruff:*)` | poetry/uv/pip install |
| javascript | `package.json` | `npm test` (or scripts.test) | `eslint .` / `prettier --check .` | eslint, prettier | `Bash(npm:*)`, `Bash(npx:*)` | npm/pnpm/yarn install |
| cpp_cmake | `CMakeLists.txt` (no `package.xml`) | `cmake -S . -B build && cmake --build build` / `ctest --test-dir build` | `clang-tidy` / `clang-format -i` | clang-format | `Bash(cmake:*)`, `Bash(ctest:*)`, `Bash(clang-format:*)`, `Bash(clang-tidy:*)` | cmake configure/build |
| ros2_ament_cmake | `package.xml` + `CMakeLists.txt` | `colcon build` / `colcon test && colcon test-result --verbose` | `clang-tidy` / `ament_lint` / `clang-format -i` | clang-format, cpplint | `Bash(colcon:*)`, `Bash(cmake:*)`, `Bash(clang-format:*)`, `Bash(clang-tidy:*)`, `Bash(ament_*:*)`, `Bash(rosdep:*)` | `source /opt/ros/$ROS_DISTRO/setup.bash`; `rosdep install --from-paths src --ignore-src -y`; `colcon build`; `source install/setup.bash` |
| ros2_ament_python | `package.xml` + `setup.py` | `colcon build` / `colcon test && colcon test-result --verbose` | `ruff check .` / `ament_flake8` / `ruff format .` | ruff, ruff-format | `Bash(colcon:*)`, `Bash(ruff:*)`, `Bash(ament_*:*)`, `Bash(rosdep:*)` | as ros2_ament_cmake |
| generic | none of the above | the declared test command | the declared lint/format | (lizard ceiling only) | the declared tool commands | the declared install command |

## clang-tidy config (C++ / ROS 2 complexity limits)

Write a `.clang-tidy` at the repo root enabling the function-size + cognitive-complexity
checks so the C++ complexity limits match AGENTS.md:

```yaml
Checks: 'readability-function-size,readability-function-cognitive-complexity'
CheckOptions:
  - key: readability-function-size.NestingThreshold
    value: '4'
  - key: readability-function-size.LineThreshold
    value: '200'
  - key: readability-function-cognitive-complexity.Threshold
    value: '49'
```

The universal lizard ceiling in `.pre-commit-config.yaml` already covers cyclomatic
complexity and length for C++ and Python; clang-tidy adds nesting + cognitive checks.
````

- [ ] **Step 2: Verify the table parses as Markdown (no broken pipes) and mentions every stack**

Run: `grep -cE "^\| (python|javascript|cpp_cmake|ros2_ament_cmake|ros2_ament_python|generic) " "$HOME/agent-project-template/template/.claude/skills/setup-project/references/stacks.md"`
Expected: `6`

- [ ] **Step 3: Commit**

```bash
R="$HOME/agent-project-template"
git -C "$R" add template/.claude/skills/setup-project/references/stacks.md
git -C "$R" commit -m "docs: setup-project stack recipe reference"
```

---

## Task 5: SKILL.md (the skill body)

**Files:**
- Create: `template/.claude/skills/setup-project/SKILL.md`

- [ ] **Step 1: Write `SKILL.md`**

````markdown
---
name: setup-project
description: Tailor this cross-agent template to the current repo. Detects the stack (Python, JavaScript/TypeScript, C++/CMake, ROS 2) and fills AGENTS.md, the permission allowlist, the quality-gate commands, and stack-specific pre-commit hooks. Run once after copying the template into a repo; safe to re-run.
---

# setup-project

Tailor the freshly-copied template to THIS repository. Make the smallest set of edits
that leaves the repo working in both Claude Code and GitHub Copilot.

## 1. Detect

Run the detector and read its JSON:

```bash
python3 .claude/skills/setup-project/scripts/detect_stack.py .
```

It reports `stacks`, `package_manager`, suggested `commands`, `has_sources` (new vs
existing repo), and which config files already exist. If `is_git` is false, offer to run
`git init`.

## 2. Read the recipe

Open `.claude/skills/setup-project/references/stacks.md` and find the row for the primary
detected stack. It lists the build/test/lint/format commands, the permissions to add,
the pre-commit hooks, and the provisioning steps.

## 3. Ask only what you cannot infer

Ask the user, one at a time: the one-line project purpose, the key directories, and which
globs (if any) should be research-gated. For an existing repo, also confirm the detected
test/lint/format commands.

## 4. Tailor the files

- `AGENTS.md` — replace `[PROJECT NAME]`, `[ONE-LINE PURPOSE]`, `[DETECTED STACK]`,
  `[MAP]`, and the three `[... COMMAND]` placeholders with the detected values. For an
  existing repo, summarize the real directory structure in the key-directories map.
- `.claude/settings.json` — add the stack's permission entries to `permissions.allow`.
- `.pre-commit-config.yaml` — add the stack's fast hooks (e.g. clang-format for C++/ROS 2,
  eslint+prettier for JS) above the existing lizard ceiling.
- For C++/ROS 2, also write the `.clang-tidy` from `references/stacks.md`.
- `research_gated.txt` — add any gated globs the user named.
- Provisioning — for ROS 2, write the `source`/`rosdep`/`colcon` steps into a setup script
  the user can run; otherwise note the install command in `AGENTS.md`.

## 5. Report

List exactly what you changed and the next steps: run `pre-commit install`, run the test
gate once, and (optionally) install Superpowers for the agent in use (Claude: official
plugin marketplace; Copilot: the dwaintr extension or faulkdev skills port).

Re-running is safe: re-detect and re-apply; do not duplicate entries that already exist.
````

- [ ] **Step 2: Verify frontmatter + that the body references the detector and recipes**

Run:
```bash
S="$HOME/agent-project-template/template/.claude/skills/setup-project/SKILL.md"
head -1 "$S" | grep -q '^---' && echo "frontmatter ok"
grep -q "detect_stack.py" "$S" && grep -q "stacks.md" "$S" && echo "refs ok"
grep -qE "^name: setup-project$" "$S" && echo "name ok"
```
Expected: `frontmatter ok`, `refs ok`, `name ok`.

- [ ] **Step 3: Commit**

```bash
R="$HOME/agent-project-template"
git -C "$R" add template/.claude/skills/setup-project/SKILL.md
git -C "$R" commit -m "feat: setup-project SKILL.md (detect -> ask -> tailor)"
```

---

## Task 6: Integration smoke + README update

**Files:**
- Create: `tests/test_setup_project_smoke.py`
- Modify: `README.md`

- [ ] **Step 1: Write `tests/test_setup_project_smoke.py`**

```python
import json
import pathlib
import subprocess
import sys

SCRIPT = (pathlib.Path(__file__).resolve().parent.parent
          / "template" / ".claude" / "skills" / "setup-project" / "scripts" / "detect_stack.py")


def test_detector_runs_as_cli_on_a_ros2_fixture(tmp_path):
    (tmp_path / "package.xml").write_text("<package/>\n")
    (tmp_path / "CMakeLists.txt").write_text("project(n)\n")
    out = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path)],
                         capture_output=True, text=True, check=True)
    data = json.loads(out.stdout)
    assert data["stacks"] == ["ros2_ament_cmake"]
    assert data["commands"]["build"] == "colcon build"


def test_skill_and_recipes_present():
    base = SCRIPT.parent.parent
    assert (base / "SKILL.md").exists()
    assert (base / "references" / "stacks.md").exists()
    assert "name: setup-project" in (base / "SKILL.md").read_text()
```

- [ ] **Step 2: Run the full suite**

Run: `cd "$HOME/agent-project-template" && python3 -m pytest -q`
Expected: all pass (Part 1's 9 + Part 2's tests).

- [ ] **Step 3: Update `README.md`** — change the "Coming in Part 2" section to "Automated setup":

Replace the `## Coming in Part 2` section with:

```markdown
## Automated setup (Part 2)

After copying `template/` in, run `/setup-project` in Claude Code or Copilot. It detects
your stack (Python / JS-TS / C++ / ROS 2 / generic), fills `AGENTS.md`, the permission
allowlist, the gate commands, and the stack-specific pre-commit hooks. Re-runnable.

Manual detection check: `python3 .claude/skills/setup-project/scripts/detect_stack.py .`
```

- [ ] **Step 4: Commit**

```bash
R="$HOME/agent-project-template"
git -C "$R" add tests/test_setup_project_smoke.py README.md
git -C "$R" commit -m "test+docs: setup-project smoke + README Part 2"
```

---

## Self-Review

**Spec coverage:** detection (Task 1-3), recipe table incl. ROS 2 + clang-tidy (Task 4),
the detect→ask→tailor skill body (Task 5), smoke + docs (Task 6). The skill is read by both
agents because it lives in `.claude/skills/` (verified in Part 1's smoke). Tailoring stays
agent-driven prose per the spec's "no framework".

**Placeholder scan:** the `generic` stack's `<your test command>` strings are intentional
user fill-ins surfaced by the detector, not plan placeholders. All code/test steps are
complete.

**Type/name consistency:** `detect()`, `detect_stacks()`, `detect_python_pm()`,
`detect_js_pm()`, `has_sources()`, and the `COMMANDS` table keys
(`python`/`javascript`/`cpp_cmake`/`ros2_ament_cmake`/`ros2_ament_python`/`generic`) are
used consistently across the implementation and every test. The `_mod()` test loader
matches the `detect_stack.py` path.
