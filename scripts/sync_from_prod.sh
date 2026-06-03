#!/usr/bin/env bash
# Pull production's odds archive + runtime outputs down to this dev machine and
# fold them in losslessly. Run from the dev box whenever you want to sync up:
#
#   scripts/sync_from_prod.sh             # full sync
#   scripts/sync_from_prod.sh --dry-run   # print the commands without running them
#
# The prod archive is snapshotted under prod's archive flock (so the copy is
# never torn mid-write), pulled here, then merged into dev's archive via
# `merge-archives` (which backs up dev's archive first and never drops the local
# backfill). The runtime parquet/json outputs are rsync'd in update mode
# (added / refreshed, never deleted). Every ssh/rsync uses a connect timeout so
# a powered-off prod fails fast instead of hanging.
#
# Environment (optional overrides):
#   PROD_SSH             ssh target for prod  (default: sportstradamus@192.168.1.84)
#   PROD_DIR             prod project root    (default: /home/sportstradamus/Sportstradamus)
#   SSH_CONNECT_TIMEOUT  seconds before giving up if prod is off (default: 15)
#   RSYNC_TIMEOUT        per-transfer I/O timeout in seconds      (default: 120)
set -euo pipefail

PROD_SSH="${PROD_SSH:-sportstradamus@192.168.1.84}"
PROD_DIR="${PROD_DIR:-/home/sportstradamus/Sportstradamus}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-15}"
RSYNC_TIMEOUT="${RSYNC_TIMEOUT:-120}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
LOCAL_DIR="$(dirname -- "$SCRIPT_DIR")"

DRY_RUN=0
case "${1:-}" in
    -n | --dry-run) DRY_RUN=1 ;;
    "") ;;
    *)
        echo "usage: $(basename "$0") [-n|--dry-run]" >&2
        exit 64
        ;;
esac

ARCHIVE_LOCK="/tmp/sportstradamus-archive.lock"
SNAPSHOT="/tmp/sportstradamus_prod_snapshot.duckdb"
RUNTIME_REL="src/sportstradamus/data/runtime"
SSH_OPTS=(-o "ConnectTimeout=$SSH_CONNECT_TIMEOUT" -o BatchMode=yes)
RSYNC_SSH="ssh -o ConnectTimeout=$SSH_CONNECT_TIMEOUT -o BatchMode=yes"

# Echo every command; run it unless we're in --dry-run.
run() {
    printf '+ %s\n' "$*"
    [[ "$DRY_RUN" -eq 1 ]] && return 0
    "$@"
}

echo ">> sync_from_prod: $PROD_SSH:$PROD_DIR -> $LOCAL_DIR"
[[ "$DRY_RUN" -eq 1 ]] && echo ">> DRY RUN (no changes)"

# 1. Fail fast if prod is off / off the LAN / outbound ssh not set up.
if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '+ %s\n' "ssh ${SSH_OPTS[*]} $PROD_SSH true"
elif ! ssh "${SSH_OPTS[@]}" "$PROD_SSH" true 2>/dev/null; then
    echo "ERROR: prod unreachable at $PROD_SSH (is it on? on the LAN? outbound ssh set up?)" >&2
    exit 1
fi

# 2. Snapshot prod's archive under its flock so the copy is consistent against
#    prod's run_job.sh crons. Local-side expansion of the paths is intended.
# shellcheck disable=SC2029
run ssh "${SSH_OPTS[@]}" "$PROD_SSH" \
    "flock $ARCHIVE_LOCK cp '$PROD_DIR/archive/archive.duckdb' '$SNAPSHOT'"

# 3. Pull the snapshot down.
run rsync -a --partial --timeout="$RSYNC_TIMEOUT" -e "$RSYNC_SSH" \
    "$PROD_SSH:$SNAPSHOT" "$SNAPSHOT"

# 4. Pull data/runtime/ (update, never delete dev-only files). Runtime files are
#    written atomically on prod, so a live read returns complete files.
run mkdir -p "$LOCAL_DIR/$RUNTIME_REL"
run rsync -a --timeout="$RSYNC_TIMEOUT" -e "$RSYNC_SSH" \
    "$PROD_SSH:$PROD_DIR/$RUNTIME_REL/" "$LOCAL_DIR/$RUNTIME_REL/"

# 5. Fold prod's rows into dev's archive losslessly. Run from the repo root so
#    merge-archives' default relative target resolves to dev's archive. Module
#    form avoids needing a fresh `poetry install` to register the entry point.
cd "$LOCAL_DIR"
run poetry run python -m sportstradamus.scripts.merge_archives --source "$SNAPSHOT"

# 6. Clean up the temp snapshots (remote and local).
# shellcheck disable=SC2029
run ssh "${SSH_OPTS[@]}" "$PROD_SSH" "rm -f '$SNAPSHOT'"
run rm -f "$SNAPSHOT"

echo ">> sync complete."
