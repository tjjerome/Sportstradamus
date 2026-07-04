#!/usr/bin/env bash
# One-time deploy helper for the P8 Phase 0 leg-schema retirement
# (docs/superpowers/plans/2026-07-03-p8-phase0-leg-schema-retirement.md).
#
# There is no data file to push from dev to prod here: prod's
# history.parquet / parlay_hist.parquet / current_offers.parquet etc. are
# live, cron-written files and are the only authoritative copies — dev's
# local copies (if any) are stale test data. So unlike sync_to_prod.sh
# (which mirrors dev's trained models onto prod), this script does not
# rsync anything. It runs the plan's own deploy runbook *on* prod, against
# prod's own real files, over ssh:
#
#   1. git pull --ff-only on devel (refuses if prod's tree is dirty or on
#      the wrong branch)
#   2. poetry lock --no-update (resync poetry.lock's content hash — this
#      branch dropped klepto from pyproject.toml but deliberately left
#      poetry.lock for the ship-curator process; poetry install fails
#      without this step) then poetry install
#   3. run the migration script's own --dry-run remotely, print it, ask
#      for confirmation
#   4. run the migration for real (--backfill-close --backfill-alt-lines)
#   5. restart the dashboard service
#
# Usage:
#   scripts/deploy_leg_schema_migration.sh             # run the real deploy
#   scripts/deploy_leg_schema_migration.sh --dry-run   # print every remote
#                                                       # command, touch nothing
#
# Best run in the cron dead zone (between the 20:50 prophecize and 23:00
# reflect) — the migration script's own writer-active guard
# (_refuse_if_writer_active) will refuse to run if a cron job's .tmp file
# is present regardless of clock time, but avoiding the window means you
# won't be waiting on that guard.
#
# Environment (optional overrides):
#   PROD_SSH             ssh target for prod  (default: sportstradamus@192.168.1.84)
#   PROD_DIR             prod project root    (default: /home/sportstradamus/Sportstradamus)
#   SSH_CONNECT_TIMEOUT  seconds before giving up if prod is off (default: 15)
set -euo pipefail

PROD_SSH="${PROD_SSH:-sportstradamus@192.168.1.84}"
PROD_DIR="${PROD_DIR:-/home/sportstradamus/Sportstradamus}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-15}"

DRY_RUN=0
case "${1:-}" in
    -n | --dry-run) DRY_RUN=1 ;;
    "") ;;
    *)
        echo "usage: $(basename "$0") [-n|--dry-run]" >&2
        exit 64
        ;;
esac

SSH_OPTS=(-o "ConnectTimeout=$SSH_CONNECT_TIMEOUT" -o BatchMode=yes)
MIGRATE_CMD="poetry run python -m sportstradamus.scripts.migrate_leg_schema"

# Echo every remote command; run it unless we're in --dry-run.
run() {
    printf '+ %s\n' "$*"
    [[ "$DRY_RUN" -eq 1 ]] && return 0
    # shellcheck disable=SC2029
    ssh "${SSH_OPTS[@]}" "$PROD_SSH" "$*"
}

echo ">> deploy_leg_schema_migration: $PROD_SSH:$PROD_DIR"
[[ "$DRY_RUN" -eq 1 ]] && echo ">> DRY RUN (no changes)"
echo ">> best run in the cron dead zone: between the 20:50 prophecize and 23:00 reflect"

# 1. Fail fast if prod is off / off the LAN / outbound ssh not set up.
if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '+ %s\n' "ssh ${SSH_OPTS[*]} $PROD_SSH true"
elif ! probe_err="$(ssh "${SSH_OPTS[@]}" "$PROD_SSH" true 2>&1)"; then
    if [[ "$probe_err" == *"Permission denied"* ]]; then
        echo "ERROR: prod refused non-interactive auth at $PROD_SSH." >&2
        echo "       This script needs key-based ssh (BatchMode rejects password prompts)." >&2
        echo "       Set it up once:  ssh-keygen -t ed25519   # if you have no key" >&2
        echo "                        ssh-copy-id $PROD_SSH" >&2
    else
        echo "ERROR: prod unreachable at $PROD_SSH (is it on? on the LAN? outbound ssh set up?)" >&2
        echo "       ssh: $probe_err" >&2
    fi
    exit 1
fi

# 2. Refuse if prod's checkout is dirty or not on devel — never pull over
#    someone else's in-progress state.
if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '+ %s\n' "ssh ${SSH_OPTS[*]} $PROD_SSH cd $PROD_DIR && git status --porcelain && git branch --show-current"
else
    # shellcheck disable=SC2029  # intentional: $PROD_DIR expands client-side
    branch="$(ssh "${SSH_OPTS[@]}" "$PROD_SSH" "cd '$PROD_DIR' && git branch --show-current")"
    # shellcheck disable=SC2029
    dirty="$(ssh "${SSH_OPTS[@]}" "$PROD_SSH" "cd '$PROD_DIR' && git status --porcelain")"
    if [[ "$branch" != "devel" ]]; then
        echo "ERROR: prod is on branch '$branch', not 'devel'. Refusing to touch it." >&2
        exit 1
    fi
    if [[ -n "$dirty" ]]; then
        echo "ERROR: prod's checkout has uncommitted changes:" >&2
        printf '%s\n' "$dirty" >&2
        echo "       Refusing to pull over in-progress state. Investigate on prod first." >&2
        exit 1
    fi
fi

# 3. Pull the merged branch. --ff-only: if this isn't a fast-forward,
#    something unexpected happened upstream — stop rather than merge.
run "cd '$PROD_DIR' && git pull --ff-only origin devel"

# 4. Resync poetry.lock's content hash (no dependency version changes),
#    then install. See header note: this branch dropped klepto from
#    pyproject.toml but left poetry.lock for the ship-curator process.
run "cd '$PROD_DIR' && poetry lock --no-update"
run "cd '$PROD_DIR' && poetry install"

# 5. Always preview the real migration first — it's read-only and this is
#    exactly the "--dry-run first" step the plan's runbook calls for. Runs
#    for real even under the wrapper's own --dry-run, since it can't
#    change anything; skip only if prod hasn't been pulled yet (dry-run
#    of this whole script implies steps 3-4 above were also skipped).
if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '+ %s\n' "ssh ${SSH_OPTS[*]} $PROD_SSH cd $PROD_DIR && $MIGRATE_CMD --dry-run"
    echo ">> DRY RUN complete; no remote commands were executed."
    exit 0
fi
echo ">> migration dry-run preview:"
# shellcheck disable=SC2029  # intentional: $PROD_DIR/$MIGRATE_CMD expand client-side
preview="$(ssh "${SSH_OPTS[@]}" "$PROD_SSH" "cd '$PROD_DIR' && $MIGRATE_CMD --dry-run")"
printf '%s\n' "$preview"

# 6. Confirm before the one irreversible step.
echo
read -r -p "Run the real migration on prod (--backfill-close --backfill-alt-lines)? [y/N] " reply < /dev/tty
if [[ ! "$reply" =~ ^[Yy]$ ]]; then
    echo ">> aborted; prod unchanged."
    exit 1
fi
run "cd '$PROD_DIR' && $MIGRATE_CMD --backfill-close --backfill-alt-lines"

# 7. Restart the dashboard so it picks up new-schema snapshots. Needs
#    passwordless sudo for this one command on prod; if it fails, see the
#    troubleshooting table in docs/dashboard_remote_access.md.
run "sudo systemctl restart sportstradamus-dashboard"

echo ">> deploy complete."
