"""Blocking duplicate-code gate: pylint R0801 over ``src/sportstradamus/``.

Ruff has no clone detector; pylint's similarity checker fills that gap, and
GitClear's data makes duplicated logic the dominant source of drift in
AI-assisted code. The policy (CLAUDE.md "Reuse before you write"; STYLE_GUIDE
§2.9) is that **any parallelism must be justified**: a block that stays parallel
across leagues/grains must be the *same shape over genuinely different
knowledge* and carry an explicit ``# pylint: disable=duplicate-code`` pragma
with a one-line rationale. pylint honours those pragmas, so a clean run here
means every surviving similarity is pragma-justified — and any new unjustified
clone fails the gate.

The checker config (``min-similarity-lines = 6`` and the ``ignore-*`` flags)
lives in ``[tool.pylint.similarities]`` in ``pyproject.toml``; pylint discovers
it from the repo root, which is why the subprocess runs with ``cwd`` there.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = "src/sportstradamus/"

# pylint exit code is a bitmask; bit 1 (fatal) and bit 32 (usage error) mean the
# checker could not run at all, which must surface as a hard failure rather than
# a misleading "no duplicates found".
_PYLINT_FATAL = 1
_PYLINT_USAGE = 32


def _duplicate_groups(stdout: str) -> list[str]:
    """Extract each R0801 group (header + the ``==module:[a:b]`` references)."""
    groups: list[str] = []
    buf: list[str] = []
    collecting = False
    for line in stdout.splitlines():
        if "R0801" in line and "Similar lines in" in line:
            if buf:
                groups.append("\n".join(buf))
            buf = [line.strip()]
            collecting = True
        elif collecting and line.startswith("=="):
            buf.append(line.strip())
        elif buf:
            groups.append("\n".join(buf))
            buf = []
            collecting = False
    if buf:
        groups.append("\n".join(buf))
    return groups


def test_no_unjustified_duplicate_code() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pylint", "--disable=all", "--enable=duplicate-code", _SRC],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode & (_PYLINT_FATAL | _PYLINT_USAGE):
        raise AssertionError(
            "pylint could not run the duplicate-code check:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    groups = _duplicate_groups(result.stdout)
    assert not groups, (
        f"{len(groups)} unjustified duplicate-code group(s) (pylint R0801) in {_SRC}. "
        "Consolidate genuine clones into a base method or a shared helper; or, if the "
        "parallelism is the same shape over genuinely different knowledge, add "
        "`# pylint: disable=duplicate-code` with a one-line rationale (STYLE_GUIDE §2.9):\n\n"
        + "\n\n".join(groups)
    )
