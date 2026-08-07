"""Shared fixtures for the dev-only diagnostics suite (``pytest -m diagnostics``)."""

from __future__ import annotations

import os

import pytest

from sportstradamus.helpers.locks import _LOCK_DIR_ENV


@pytest.fixture(scope="session", autouse=True)
def isolated_lock_dir(tmp_path_factory) -> None:
    """Point the training-artifact flock at a scratch dir for the whole session.

    tmp_path_factory roots itself per xdist worker, so the tests that exercise a
    confirm walk or a meditate run neither contend with each other nor block a real
    walk that happens to be running on this box.
    """
    os.environ[_LOCK_DIR_ENV] = str(tmp_path_factory.mktemp("locks"))
