#!/usr/bin/env bash
# Monthly gate-status job: regenerate main's ship_config.json from live
# graduation metrics and, if it changed, open a PR for a human to merge.
# main is the public branch, so this NEVER pushes to main directly. Invoked
# via `run_job.sh gate-status` (which adds flock + healthchecks).
#
# Environment (optional):
#   GATE_STATUS_DRY_RUN       non-empty -> generate (or stub) and stop before git.
#   GENERATE_SHIP_CONFIG_CMD  generator command (default: "poetry run generate-ship-config").
#   GATE_STATUS_WORKTREE      worktree dir for the main checkout (default: $PROJECT_DIR/.gate-status-main).
#   SPORTSTRADAMUS_DIR        project root (default: parent of this script).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_DIR="${SPORTSTRADAMUS_DIR:-$(dirname -- "$SCRIPT_DIR")}"
SHIP_CONFIG_REL="src/sportstradamus/data/ship_config.json"
GEN_CMD="${GENERATE_SHIP_CONFIG_CMD:-poetry run generate-ship-config}"

cd "$PROJECT_DIR"

# Dry/fake mode (tests, manual dry runs): generate or stub, then stop before git.
if [[ -n "${GATE_STATUS_DRY_RUN:-}" ]]; then
    $GEN_CMD --branch main --dry-run
    echo "gate-status: dry-run complete (no PR)."
    exit 0
fi

git fetch origin main

TMP_CONFIG="$(mktemp)"
TMP_MAIN="$(mktemp)"
WORKTREE_DIR=""
# Always remove the temp files, and the worktree if one was created, so a
# mid-run failure (push / gh pr create) can't leave a stale worktree that
# collides with next month's `git worktree add`.
cleanup() {
    rm -f "$TMP_CONFIG" "$TMP_MAIN"
    if [[ -n "$WORKTREE_DIR" ]]; then
        git worktree remove --force "$WORKTREE_DIR" 2>/dev/null || true
    fi
}
trap cleanup EXIT

$GEN_CMD --branch main --out "$TMP_CONFIG"

# No change vs main's committed config -> nothing to do.
if git show "origin/main:$SHIP_CONFIG_REL" > "$TMP_MAIN" 2>/dev/null \
        && diff -q "$TMP_MAIN" "$TMP_CONFIG" >/dev/null 2>&1; then
    echo "gate-status: main ship_config unchanged; no PR."
    exit 0
fi

# A change exists -> land it on a fresh branch off origin/main and open a PR.
BRANCH="gate-status/main-$(date -u +%Y%m%d)"
WORKTREE_DIR="${GATE_STATUS_WORKTREE:-$PROJECT_DIR/.gate-status-main}"
git worktree add --force -B "$BRANCH" "$WORKTREE_DIR" origin/main
cp "$TMP_CONFIG" "$WORKTREE_DIR/$SHIP_CONFIG_REL"
git -C "$WORKTREE_DIR" add "$SHIP_CONFIG_REL"
git -C "$WORKTREE_DIR" commit -m "chore(gate): refresh main ship_config from live graduation"
git -C "$WORKTREE_DIR" push -u origin "$BRANCH"
gh pr create --base main --head "$BRANCH" \
    --title "Gate-status: refresh main ship_config from live graduation" \
    --body "Automated monthly regeneration of main's ship_config.json from live graduation metrics (training/graduation.py). Review the active/withheld diff before merging; main is the public branch."
echo "gate-status: opened PR on branch $BRANCH."
# The EXIT trap removes the worktree (here on success, or on any earlier failure).
