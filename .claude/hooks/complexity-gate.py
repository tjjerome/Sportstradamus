#!/usr/bin/env python3
"""Claude Code PostToolUse gate: block edits that introduce or worsen
over-threshold function complexity, length, or nesting.

Design — a ratchet that only tightens:
  - Only functions you actually changed since HEAD are checked.
  - A function is blocked only if your edit made the metric WORSE, or the
    function is new. Pre-existing debt never blocks an unrelated edit.

Thresholds map to docs/STYLE_GUIDE.md:
  - cyclomatic complexity  > 10   (§10, "single digits, ~10 ceiling")
  - function source lines  > 200  (§2.8 / §10 mandatory-refactor line; no monoliths)
  - control-flow nesting   > 4    (§10, ">4 levels is a refactor signal")

Escape hatch: put `# style: allow-complexity` (or -length, -nesting, -all)
inside a function to skip that check for it — the documented-suppression
posture of §19. Pair it with a one-line reason.

Exit 0 = pass (silent). Exit 2 = block; the report is sent to Claude on
stderr. Any internal error fails OPEN (exit 0) so a hook bug can never wedge
editing — the one exception is a syntax error in the edited file, which blocks.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys

# --- Tunables ---------------------------------------------------------------
CC_MAX = 10
LEN_MAX = 200
NEST_MAX = 4

# Gate package source only. Scripts, tests, and archived code are looser by
# design — this mirrors the per-file-ignores in pyproject.toml.
INCLUDE_PREFIX = "src/sportstradamus/"
EXCLUDE_PREFIXES = ("src/sportstradamus/scripts/", "src/deprecated/", "tests/")
# ----------------------------------------------------------------------------

ALLOW_RE = re.compile(r"#\s*style:\s*allow-(complexity|length|nesting|all)")
HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", re.M)


def _is_func(node):
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))


def iter_functions(tree):
    """Yield (qualname, node) for every function/method, dotted by enclosure."""
    out = []

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                walk(child, prefix + child.name + ".")
            elif _is_func(child):
                qn = prefix + child.name
                out.append((qn, child))
                walk(child, qn + ".")

    walk(tree, "")
    return out


def complexity(func):
    """McCabe cyclomatic complexity of one function, nested defs excluded."""
    score = 1

    def rec(node):
        nonlocal score
        if isinstance(node, (ast.If, ast.IfExp, ast.For, ast.AsyncFor,
                             ast.While, ast.ExceptHandler)):
            score += 1
        elif isinstance(node, ast.BoolOp):
            score += len(node.values) - 1
        elif isinstance(node, ast.comprehension):
            score += len(node.ifs)
        elif isinstance(node, ast.match_case):
            score += 1
        for child in ast.iter_child_nodes(node):
            if _is_func(child) or isinstance(child, ast.Lambda):
                continue
            rec(child)

    for stmt in func.body:
        rec(stmt)
    return score


def nesting(func):
    """Max control-flow nesting depth, elif-flattened, nested defs excluded."""
    max_d = 0

    def block(stmts, depth):
        for s in stmts:
            descend(s, depth)

    def descend(node, depth):
        nonlocal max_d
        if _is_func(node) or isinstance(node, ast.ClassDef):
            return
        if isinstance(node, ast.If):
            d = depth + 1
            max_d = max(max_d, d)
            block(node.body, d)
            orelse = node.orelse
            while len(orelse) == 1 and isinstance(orelse[0], ast.If):
                block(orelse[0].body, d)  # elif: same depth
                orelse = orelse[0].orelse
            if orelse:
                block(orelse, d)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            d = depth + 1
            max_d = max(max_d, d)
            block(node.body, d)
            if node.orelse:
                block(node.orelse, d)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            d = depth + 1
            max_d = max(max_d, d)
            block(node.body, d)
        elif isinstance(node, ast.Try):
            d = depth + 1
            max_d = max(max_d, d)
            block(node.body, d)
            for h in node.handlers:
                block(h.body, d)
            block(node.orelse, d)
            block(node.finalbody, d)
        elif isinstance(node, ast.Match):
            d = depth + 1
            max_d = max(max_d, d)
            for case in node.cases:
                block(case.body, d)

    block(func.body, 0)
    return max_d


def length(func, lines):
    """Source lines in the function span, excluding blanks and comment lines."""
    start = func.lineno
    end = getattr(func, "end_lineno", func.lineno)
    n = 0
    for i in range(start, end + 1):
        text = lines[i - 1].strip() if i - 1 < len(lines) else ""
        if text and not text.startswith("#"):
            n += 1
    return n


def allowed(func, lines):
    start = func.lineno
    end = getattr(func, "end_lineno", func.lineno)
    found = set(ALLOW_RE.findall("\n".join(lines[start - 1:end])))
    if "all" in found:
        return {"complexity", "length", "nesting"}
    return found


def metrics_for(src):
    tree = ast.parse(src)
    lines = src.splitlines()
    out = {}
    for qn, fn in iter_functions(tree):
        out[qn] = {
            "complexity": complexity(fn),
            "length": length(fn, lines),
            "nesting": nesting(fn),
            "lineno": fn.lineno,
            "end_lineno": getattr(fn, "end_lineno", fn.lineno),
            "allow": allowed(fn, lines),
        }
    return out


def git(args, cwd):
    try:
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                           text=True, timeout=10)
    except Exception:
        return None
    return r.stdout if r.returncode == 0 else None


def changed_ranges(repo, rel):
    out = git(["diff", "--unified=0", "HEAD", "--", rel], repo)
    if out is None:
        return None
    ranges = []
    for m in HUNK_RE.finditer(out):
        start = int(m.group(1))
        count = int(m.group(2) or "1")
        ranges.append((start, start + max(count, 1) - 1))
    return ranges


def read_stdin_path():
    try:
        data = json.load(sys.stdin)
    except Exception:
        return None
    return (data.get("tool_input") or {}).get("file_path")


CHECKS = (
    ("complexity", CC_MAX, "cyclomatic complexity",
     "Extract helpers or replace branching with a lookup table / early "
     "returns (STYLE_GUIDE §10)."),
    ("length", LEN_MAX, "source lines",
     "Extract named _step_* helpers; the orchestrator should read as a list "
     "of steps (STYLE_GUIDE §2.8, §10)."),
    ("nesting", NEST_MAX, "levels of nesting",
     "Flatten with guard clauses and early returns (STYLE_GUIDE §10)."),
)


def main():
    path = read_stdin_path()
    if not path or not path.endswith(".py"):
        return 0

    apath = os.path.abspath(path)
    cwd = os.path.dirname(apath) or "."
    repo = git(["rev-parse", "--show-toplevel"], cwd)
    if not repo:
        return 0  # not a git repo — can't scope safely, fail open
    repo = repo.strip()
    rel = os.path.relpath(apath, repo).replace(os.sep, "/")

    if not rel.startswith(INCLUDE_PREFIX):
        return 0
    if any(rel.startswith(x) for x in EXCLUDE_PREFIXES):
        return 0

    try:
        with open(apath, encoding="utf-8") as f:
            cur_src = f.read()
    except OSError:
        return 0

    try:
        cur = metrics_for(cur_src)
    except SyntaxError as e:
        sys.stderr.write(
            f"[style-gate] {rel} does not parse after the edit: {e.msg} "
            f"(line {e.lineno}). Fix the syntax error before continuing.\n")
        return 2

    head_src = git(["show", f"HEAD:{rel}"], repo)
    if head_src is None:
        base = {}              # new file (untracked, or not yet in HEAD)
        touched = set(cur)
    else:
        try:
            base = metrics_for(head_src)
        except SyntaxError:
            base = {}
        ranges = changed_ranges(repo, rel) or []
        if not ranges:
            return 0           # nothing changed vs HEAD
        touched = set()
        for qn, m in cur.items():
            for lo, hi in ranges:
                if not (m["end_lineno"] < lo or m["lineno"] > hi):
                    touched.add(qn)
                    break

    problems = []
    for qn in sorted(touched):
        m = cur[qn]
        b = base.get(qn)
        for key, limit, label, fix in CHECKS:
            val = m[key]
            if val <= limit or key in m["allow"]:
                continue
            prior = b[key] if b else None
            if prior is not None and val <= prior:
                continue       # over the limit, but you didn't make it worse
            origin = "new" if prior is None else f"was {prior}"
            problems.append(
                f"  {qn}()  {label}: {val}  (limit {limit}, {origin})\n"
                f"      \u2192 {fix}")

    if not problems:
        return 0

    sys.stderr.write(
        f"[style-gate] Blocked: in {rel}, functions you changed exceed the "
        f"docs/STYLE_GUIDE.md limits and are new or got worse:\n"
        + "\n".join(problems)
        + "\n\nFix the functions you touched. If the complexity is genuinely "
        "irreducible (e.g. a numeric kernel), add `# style: allow-<check>` "
        "inside the function with a one-line reason, per §19.\n")
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:  # never wedge editing on a hook bug
        sys.stderr.write(f"[style-gate] internal error, allowing edit: {e}\n")
        sys.exit(0)
