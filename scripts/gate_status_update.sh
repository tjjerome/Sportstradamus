#!/usr/bin/env bash
# Monthly gate-status job: regenerate main's stat_meta.json from live
# graduation metrics and, if it changed, open a PR for a human to merge.
# main is the public branch, so this NEVER pushes to main directly. Invoked
# via `run_job.sh gate-status` (which adds flock + healthchecks).
#
# Environment (optional):
#   GATE_STATUS_DRY_RUN       non-empty -> generate (or stub) and stop before git.
#   GENERATE_SHIP_CONFIG_CMD  generator command (default: "poetry run python -m sportstradamus ship config").
#   GATE_STATUS_WORKTREE      worktree dir for the main checkout (default: $PROJECT_DIR/.gate-status-main).
#   SPORTSTRADAMUS_DIR        project root (default: parent of this script).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_DIR="${SPORTSTRADAMUS_DIR:-$(dirname -- "$SCRIPT_DIR")}"
STAT_META_REL="src/sportstradamus/data/config/stat_meta.json"
GEN_CMD="${GENERATE_SHIP_CONFIG_CMD:-poetry run python -m sportstradamus ship config}"

cd "$PROJECT_DIR"

# Dry/fake mode (tests, manual dry runs): generate or stub, then stop before git.
if [[ -n "${GATE_STATUS_DRY_RUN:-}" ]]; then
    $GEN_CMD --branch main --dry-run
    echo "gate-status: dry-run complete (no PR)."
    exit 0
fi

git fetch origin main

WORKTREE_DIR=""
# Always remove the worktree (if one was created) so a mid-run failure (push
# / gh pr create) can't leave a stale worktree that collides with next
# month's `git worktree add`.
cleanup() {
    if [[ -n "$WORKTREE_DIR" ]]; then
        git worktree remove --force "$WORKTREE_DIR" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Work on a worktree off origin/main so the generator mutates the main
# branch's stat_meta.json directly. The generator is in-place: it edits
# the file at $STAT_META_REL inside the worktree.
BRANCH="gate-status/main-$(date -u +%Y%m%d)"
WORKTREE_DIR="${GATE_STATUS_WORKTREE:-$PROJECT_DIR/.gate-status-main}"
git worktree add --force -B "$BRANCH" "$WORKTREE_DIR" origin/main

$GEN_CMD --branch main --meta "$WORKTREE_DIR/$STAT_META_REL"

# No change vs main's committed config -> nothing to do.
if git -C "$WORKTREE_DIR" diff --quiet -- "$STAT_META_REL"; then
    echo "gate-status: main stat_meta unchanged; no PR."
    exit 0
fi

# A change exists -> commit it on the worktree's branch and open a PR.
git -C "$WORKTREE_DIR" add "$STAT_META_REL"
git -C "$WORKTREE_DIR" commit -m "chore(gate): refresh main stat_meta from live graduation"
git -C "$WORKTREE_DIR" push -u origin "$BRANCH"
gh pr create --base main --head "$BRANCH" \
    --title "Gate-status: refresh main stat_meta from live graduation" \
    --body "Automated monthly regeneration of main's stat_meta.json from live graduation metrics (training/graduation.py). Review the shipped: devel <-> main diff before merging; main is the public branch."
echo "gate-status: opened PR on branch $BRANCH."
# The EXIT trap removes the worktree (here on success, or on any earlier failure).
