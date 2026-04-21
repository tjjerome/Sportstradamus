"""Shared pytest fixtures for the golden-snapshot suite.

These tests protect the public CLI surface during the readability refactor.
See docs/STYLE_GUIDE.md §14 and the Phase 1 section of the refactor plan at
/home/trevor/.claude/plans/my-codebase-has-gotten-toasty-bonbon.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Absolute path to the committed golden-snapshot fixtures."""
    return FIXTURES_DIR


def read_snapshot(name: str) -> str:
    """Read a committed snapshot by filename.

    Args:
        name: File name inside ``tests/golden/fixtures/``.

    Returns:
        Decoded file contents. Trailing whitespace is preserved because help
        text uses it.
    """
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")


def write_snapshot(name: str, contents: str) -> None:
    """Overwrite a snapshot. Called only when intentionally regenerating."""
    (FIXTURES_DIR / name).write_text(contents, encoding="utf-8")
