"""Golden snapshots of the umbrella CLI's ``--help`` output, one per command node.

Walks the resolved ``sportstradamus`` command tree — the root, every group, and
every leaf — so a newly mounted command is snapshotted automatically and an
accidental flag rename, unmounted command, or help-text drift is a hard failure.

Snapshots live in ``tests/golden/fixtures/``. To regenerate after an intentional
CLI change, set ``REGENERATE_SNAPSHOTS=1`` in the environment and rerun the
suite — the test overwrites fixtures in place. Without that flag a mismatch
fails.
"""

from __future__ import annotations

import os

import click
import pytest
from click.testing import CliRunner

from sportstradamus.cli import cli
from tests.golden.conftest import read_snapshot, write_snapshot


def _walk(cmd: click.Command, args: tuple[str, ...]):
    yield args
    if isinstance(cmd, click.Group):
        ctx = click.Context(cmd, info_name=args[-1] if args else "sportstradamus")
        for name in sorted(cmd.list_commands(ctx)):
            yield from _walk(cmd.get_command(ctx, name), (*args, name))


_NODES = list(_walk(cli, ()))
_CASES = [
    (node, "help_" + "_".join(("sportstradamus", *node)).replace("-", "_") + ".txt")
    for node in _NODES
]


@pytest.mark.parametrize(
    ("node", "snapshot"),
    _CASES,
    ids=[" ".join(("sportstradamus", *node)) for node in _NODES],
)
def test_cli_help_matches_snapshot(node: tuple[str, ...], snapshot: str) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, [*node, "--help"], prog_name="sportstradamus")
    assert result.exit_code == 0, f"{node} --help exited {result.exit_code}: {result.output}"

    if os.environ.get("REGENERATE_SNAPSHOTS") == "1":
        write_snapshot(snapshot, result.output)
        pytest.skip(f"Regenerated snapshot {snapshot}")

    expected = read_snapshot(snapshot)
    assert result.output == expected, (
        f"sportstradamus {' '.join(node)} --help drifted from {snapshot}. "
        "If the change is intentional, rerun with REGENERATE_SNAPSHOTS=1."
    )
