"""Golden snapshots of every CLI's ``--help`` output.

These tests catch accidental flag renames, removed entry points, and changes
to documented help text that users or the wider Sportstradamus ecosystem
might depend on. They run via ``pytest tests/golden/`` in CI.

Snapshots live in ``tests/golden/fixtures/``. To regenerate after an
intentional CLI change, set ``REGENERATE_SNAPSHOTS=1`` in the environment
and rerun the suite — the test will overwrite the fixture in place. Without
that flag a mismatch is a hard failure.
"""

from __future__ import annotations

import os

import pytest
from click.testing import CliRunner

from sportstradamus.dashboard import run as dashboard_cli
from sportstradamus.moneylines import confer as confer_cli
from sportstradamus.nightly import run as reflect_cli
from sportstradamus.prediction.cli import main as prophecize_cli
from sportstradamus.training.cli import meditate as meditate_cli
from tests.golden.conftest import read_snapshot, write_snapshot

CLI_CASES = [
    ("prophecize", prophecize_cli, "prophecize_help.txt"),
    ("meditate", meditate_cli, "meditate_help.txt"),
    ("reflect", reflect_cli, "reflect_help.txt"),
    ("dashboard", dashboard_cli, "dashboard_help.txt"),
    ("confer", confer_cli, "confer_help.txt"),
]


@pytest.mark.parametrize(("name", "command", "snapshot"), CLI_CASES)
def test_cli_help_matches_snapshot(name: str, command, snapshot: str) -> None:
    runner = CliRunner()
    result = runner.invoke(command, ["--help"])
    assert result.exit_code == 0, f"{name} --help exited {result.exit_code}: {result.output}"

    if os.environ.get("REGENERATE_SNAPSHOTS") == "1":
        write_snapshot(snapshot, result.output)
        pytest.skip(f"Regenerated snapshot {snapshot}")

    expected = read_snapshot(snapshot)
    assert result.output == expected, (
        f"{name} --help drifted from {snapshot}. "
        "If the change is intentional, rerun with REGENERATE_SNAPSHOTS=1."
    )
