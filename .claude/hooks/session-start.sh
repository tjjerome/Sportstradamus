#!/bin/bash
# SessionStart hook: provision a Claude Code on the web session so the three
# quality gates (ruff / pytest golden / integration smoke) can run.
#
# Three independent, remote-only, idempotent responsibilities:
#   1. Install the GitHub CLI so the web console's PR-status indicator
#      (`gh pr checks`) works. Auth is left to the user: set GH_TOKEN in the
#      environment-variables config and `gh` picks it up automatically.
#   2. Stub the gitignored runtime files (creds/keys.json, book_weights.json,
#      goalies.json, …) from tests/ci_fixtures/ — the same provisioning the
#      CI quality job uses. helpers/config.py and helpers/scraping.py read
#      them eagerly at import, so without them the golden/integration suites
#      collapse at collection. Only missing files are stubbed; a tree synced
#      from production is never clobbered.
#   3. Install the Python dependency stack with Poetry. This needs the
#      environment's network policy to allow download.pytorch.org (the pinned
#      CPU-only torch source) on top of the default PyPI allowlist; until that
#      is granted, `poetry install` 403s on the pytorch_cpu source.

set -uo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# --- 1. GitHub CLI (subshelled so its failure can't block dep install) -------
if ! command -v gh >/dev/null 2>&1; then
  (
    set -e
    if ! command -v wget >/dev/null 2>&1; then
      sudo apt-get update -qq
      sudo apt-get install -y -qq wget
    fi
    sudo mkdir -p -m 755 /etc/apt/keyrings
    wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg \
      | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg >/dev/null
    sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
      | sudo tee /etc/apt/sources.list.d/github-cli.list >/dev/null
    sudo apt-get update -qq
    sudo apt-get install -y -qq gh
  ) || echo "gh install failed; the web PR-status indicator may not work." >&2
fi

if [ -z "${GH_TOKEN:-}" ] && [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "gh installed. Set GH_TOKEN in your Claude Code on the web env to authenticate." >&2
fi

# --- 2. Gitignored runtime stubs from the CI fixtures ------------------------
_ROOT="${CLAUDE_PROJECT_DIR:-$(pwd)}"
if [ ! -f "$_ROOT/src/sportstradamus/creds/keys.json" ]; then
  cp "$_ROOT/tests/ci_fixtures/creds/keys.json" "$_ROOT/src/sportstradamus/creds/keys.json"
fi
for _fixture in "$_ROOT"/tests/ci_fixtures/data/*.json; do
  _target="$_ROOT/src/sportstradamus/data/config/$(basename "$_fixture")"
  [ -f "$_target" ] || cp "$_fixture" "$_target"
done

# --- 3. Python dependencies --------------------------------------------------
# Skip when the package and the pinned torch already import (warm container).
if poetry run python -c "import sportstradamus, torch" >/dev/null 2>&1; then
  exit 0
fi

if ! poetry install; then
  echo "poetry install failed." >&2
  echo "If this is the pytorch_cpu source returning 403, allow download.pytorch.org" >&2
  echo "in the environment's Network access policy (Custom or Full, keeping the" >&2
  echo "default package-manager allowlist), then resume to re-run this hook." >&2
fi

exit 0
