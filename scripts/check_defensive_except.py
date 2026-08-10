#!/usr/bin/env python3
"""Pre-commit guard: block the defensive try-blocks audited out (STYLE_GUIDE §5/§11).

Two patterns are rejected:

* Bare ``except:`` — swallows everything including ``KeyboardInterrupt``; catch
  the specific exception instead (§11).
* Optional-dependency imports — ``except ImportError`` / ``except
  ModuleNotFoundError`` makes a dependency optional (§5). cvxpy / pyyaml /
  tabulate are core deps now and the first-party clv / training.report guards
  were dead code; a dependency is either a core import or it is not used.

Deferred imports for heavy-dependency or cycle isolation are still fine: they
import unconditionally inside the function that needs them and do not catch
``ImportError`` (e.g. the torch-isolation import in ``strategies/``).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_BANNED_IMPORT = frozenset({"ImportError", "ModuleNotFoundError"})


def _caught_names(handler: ast.ExceptHandler) -> set[str]:
    exc = handler.type
    if exc is None:
        return set()
    nodes = exc.elts if isinstance(exc, ast.Tuple) else [exc]
    return {n.id for n in nodes if isinstance(n, ast.Name)}


def _violations(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if node.type is None:
            out.append((node.lineno, "bare `except:` — catch the specific exception (§11)"))
        elif _caught_names(node) & _BANNED_IMPORT:
            out.append(
                (node.lineno, "optional import — make the dependency a core top-level import (§5)")
            )
    return out


def main(paths: list[str]) -> int:
    failed = False
    for raw in paths:
        path = Path(raw)
        if path.suffix != ".py":
            continue
        for lineno, msg in _violations(path):
            print(f"{path}:{lineno}: {msg}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
