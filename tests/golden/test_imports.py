"""Sanity import tests for every reachable module in the package.

If a refactor breaks an import (typo, circular dep, missing re-export from a
new package's ``__init__.py``) this suite catches it before any CLI-level
test even runs. Cheap to run, very useful during the multi-phase split.
"""

from __future__ import annotations

import importlib

import pytest

CORE_MODULES = [
    "sportstradamus.helpers",
    "sportstradamus.spiderLogger",
    "sportstradamus.skew_normal",
    "sportstradamus.feature_selection",
    "sportstradamus.stats",
    "sportstradamus.books",
    "sportstradamus.moneylines",
    "sportstradamus.train",
    "sportstradamus.sportstradamus",
    "sportstradamus.nightly",
    "sportstradamus.dashboard",
    "sportstradamus.dashboard_app",
    "sportstradamus.dashboard_data",
    "sportstradamus.analysis",
]


@pytest.mark.parametrize("module_name", CORE_MODULES)
def test_module_imports(module_name: str) -> None:
    importlib.import_module(module_name)
