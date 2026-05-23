"""Fake-mode test for the monthly gate-status cron wrapper (no git, no network)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "gate_status_update.sh"


@pytest.mark.integration
def test_gate_status_dry_run_invokes_generator_and_skips_git(tmp_path):
    env = dict(os.environ)
    env["GATE_STATUS_DRY_RUN"] = "1"
    # Stub the generator so the test touches neither poetry nor the real data.
    env["GENERATE_SHIP_CONFIG_CMD"] = "echo generate-ship-config-stub"
    result = subprocess.run(
        ["bash", str(_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(_REPO),
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "generate-ship-config-stub" in result.stdout
    assert "dry-run" in result.stdout.lower()
