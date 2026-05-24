#!/bin/bash
# SessionStart hook: install the GitHub CLI so the Claude Code on the web
# console's PR-status indicator (which shells out to `gh pr checks`) works.
#
# - Skips on local sessions; only runs in remote envs.
# - Idempotent: if `gh` is already on PATH, exits immediately.
# - Auth is left to the user: set GH_TOKEN in your Claude Code on the web
#   environment-variables config and `gh` will pick it up automatically.

set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

if command -v gh >/dev/null 2>&1; then
  exit 0
fi

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

if [ -z "${GH_TOKEN:-}" ] && [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "gh installed. Set GH_TOKEN in your Claude Code on the web env to authenticate." >&2
fi
