"""Fake-mode tests for the monthly gate-status cron wrapper (no network)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "gate_status_update.sh"
_STAT_META_REL = "src/sportstradamus/data/config/stat_meta.json"


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


def _git(*args: str, cwd: Path, env: dict[str, str]) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True
    ).stdout


@pytest.mark.integration
def test_gate_status_failed_pr_leaves_no_local_branch(tmp_path):
    # Prod's runs of 09-01 and 10-01 each died after creating their branch and left
    # it behind; the remote copy is what a PR needs, so only that one may survive.
    env = dict(os.environ)
    global_config = tmp_path / "gitconfig"
    global_config.write_text("[user]\n\tname = t\n\temail = t@example.com\n")
    env.update(GIT_CONFIG_GLOBAL=str(global_config), GIT_CONFIG_NOSYSTEM="1")
    origin, repo, bin_dir = tmp_path / "origin.git", tmp_path / "repo", tmp_path / "bin"
    _git("init", "--bare", str(origin), cwd=tmp_path, env=env)
    _git("init", "-b", "main", str(repo), cwd=tmp_path, env=env)
    (repo / _STAT_META_REL).parent.mkdir(parents=True)
    (repo / _STAT_META_REL).write_text("{}\n")
    (repo / "scripts").mkdir()
    shutil.copy(_SCRIPT, repo / "scripts" / _SCRIPT.name)
    _git("add", "-A", cwd=repo, env=env)
    _git("commit", "-m", "init", cwd=repo, env=env)
    _git("remote", "add", "origin", str(origin), cwd=repo, env=env)
    _git("push", "origin", "main", cwd=repo, env=env)

    bin_dir.mkdir()
    generator = bin_dir / "generate"
    generator.write_text(
        '#!/bin/sh\nwhile [ $# -gt 0 ]; do [ "$1" = --meta ] && echo changed > "$2"; shift; done\n'
    )
    gh = bin_dir / "gh"
    gh.write_text("#!/bin/sh\nexit 1\n")
    generator.chmod(0o755)
    gh.chmod(0o755)
    env.update(GENERATE_SHIP_CONFIG_CMD=str(generator), PATH=f"{bin_dir}:{env['PATH']}")

    result = subprocess.run(
        ["bash", str(repo / "scripts" / _SCRIPT.name)],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(repo),
        check=False,
    )

    assert result.returncode != 0
    assert _git("branch", "--list", "gate-status/*", cwd=repo, env=env) == ""
    assert not (repo / ".gate-status-main").exists()
    assert "gate-status/main-" in _git("ls-remote", "origin", cwd=repo, env=env)
