"""Shared pytest fixtures for the golden-snapshot suite.

These tests protect the public CLI surface during the readability refactor.
See docs/STYLE_GUIDE.md §19 and the Phase 1 section of the refactor plan at
/home/trevor/.claude/plans/my-codebase-has-gotten-toasty-bonbon.md.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from sportstradamus.helpers.locks import _LOCK_DIR_ENV

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session", autouse=True)
def isolated_lock_dir(tmp_path_factory) -> None:
    """Point the training-artifact flock at a scratch dir for the whole session.

    tmp_path_factory roots itself per xdist worker, so the CLI tests that exercise a
    confirm walk or a meditate run neither contend with each other nor block a real
    walk that happens to be running on this box.
    """
    os.environ[_LOCK_DIR_ENV] = str(tmp_path_factory.mktemp("locks"))


@pytest.fixture
def fixtures_dir() -> Path:
    """Absolute path to the committed golden-snapshot fixtures."""
    return FIXTURES_DIR


# get_ev inverts a book price with brentq; its EV drifts a few ppm across
# unrelated refactors (final iterate + nonlinear-region sensitivity off the line).
# Pin the row structure exactly and the EV within this relative tolerance, well
# below any change that would move a real betting decision.
_BOOK_EV_RTOL = 1e-4


@pytest.fixture
def assert_player_books_close():
    """Assert two ``merge_player_books`` row lists match, EV at relative tolerance.

    Rows are ``(league, market, date, player, {book: ev}, lines, observed_at)``.
    Everything but the per-book ``ev`` dict must be equal; the ev compares within
    ``_BOOK_EV_RTOL`` so the snapshot tracks real EV changes, not root-finder jitter.
    """

    def _assert(actual, expected):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected, strict=True):
            assert a[:4] == e[:4] and a[5:] == e[5:]
            assert a[4] == pytest.approx(e[4], rel=_BOOK_EV_RTOL)

    return _assert


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
