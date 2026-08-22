#!/usr/bin/env bash
# Push this dev machine's locally-produced data UP to the production server. Run
# from the dev box after a local `meditate` and/or a collector run (`ctg-fetch`,
# `savant-fetch`):
#
#   scripts/sync_to_prod.sh             # sync models + serving artifacts + collector snapshots
#   scripts/sync_to_prod.sh --dry-run   # show what would change, touch nothing
#
# Three kinds of pass:
#   * MODELS (mirror, --delete): the models dir is mirrored — prod-only orphans
#     are deleted so a retired cell's pickle doesn't linger. The research subdir
#     models/deterministic/ is excluded (and prod's copy protected from --delete).
#     Before deleting any prod model absent on dev, the script lists them and asks
#     for confirmation; with no orphans it proceeds silently.
#   * COLLECTOR SNAPSHOTS (additive, NO --delete): the Cleaning the Glass (NBA)
#     and Baseball Savant (MLB) dated-snapshot dirs are pushed additively. They
#     live under player_data/ and team_data/ beside prod-only siblings
#     (nba_players_*.csv, affinity_*.csv, gamelogs) — a --delete there would wipe
#     those. Hard rule: NEVER --delete on a player_data/ or team_data/ path.
#     A collector dir absent on dev (catalog not yet populated) is skipped.
#   * SERVING ARTIFACTS (single files, additive): gitignored training outputs the
#     prod serving pipeline reads — book_weights.json and stat_calibration.json
#     (consensus weights + per-cell cv/zi read at serve), model_stats.{parquet,csv}
#     (kelly shrinkage), and each league's corr_* correlation trio (parlay
#     scoring). With prod's `meditate` cron disabled, this script is the ONLY path
#     these take to prod; models synced without them serve against stale
#     calibration (the 2026-08 WNBA book_weights incident). Explicit file list,
#     never a config/-wide pass — config/ also holds prod-local runtime state
#     (goalies.json) that a directory sync would clobber.
#
# NOTE: if prod's weekly `meditate` cron is enabled it rewrites the models dir and
# the serving artifacts above. Don't run this during that window (or at all while
# prod self-trains — the two workflows overwrite each other).
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

MODELS_REL="src/sportstradamus/data/models"
# Collector snapshot dirs — synced additively (NO --delete; see header).
COLLECTOR_RELS=(
    "src/sportstradamus/data/player_data/NBA/cleaningtheglass"
    "src/sportstradamus/data/team_data/NBA/cleaningtheglass"
    "src/sportstradamus/data/player_data/MLB/baseballsavant"
    "src/sportstradamus/data/team_data/MLB/baseballsavant"
)
# Serving artifacts — gitignored files meditate/correlate write on dev and the
# prod pipelines read at serve (see header). Files absent on dev are skipped.
SERVING_FILE_RELS=(
    "src/sportstradamus/data/config/book_weights.json"
    "src/sportstradamus/data/config/stat_calibration.json"
    "src/sportstradamus/data/training/model_stats.parquet"
    "src/sportstradamus/data/training/model_stats.csv"
)
for league_dir in "$LOCAL_DIR"/src/sportstradamus/data/leagues/*/; do
    [[ -d "$league_dir" ]] || continue
    for name in corr_same_team.parquet corr_opposing.parquet corr_metadata.json; do
        [[ -f "$league_dir$name" ]] && SERVING_FILE_RELS+=("${league_dir#"$LOCAL_DIR/"}$name")
    done
done
SSH_OPTS=(-o "ConnectTimeout=$SSH_CONNECT_TIMEOUT" -o BatchMode=yes)
RSYNC_SSH="ssh -o ConnectTimeout=$SSH_CONNECT_TIMEOUT -o BatchMode=yes"

run() {
    printf '+ %s\n' "$*"
    [[ "$DRY_RUN" -eq 1 ]] && return 0
    "$@"
}

# One rsync pass. $1 = repo-relative dir, $2 = 1 to mirror (--delete), 0 additive.
sync_pass() {
    local src_rel="$1" use_delete="$2"
    local src="$LOCAL_DIR/$src_rel"
    local dest="$PROD_SSH:$PROD_DIR/$src_rel"
    if [[ ! -d "$src" ]]; then
        echo ">> skip $src_rel (not present on dev)"
        return 0
    fi
    local filter=(--timeout="$RSYNC_TIMEOUT" -e "$RSYNC_SSH")
    [[ "$use_delete" -eq 1 ]] && filter+=(--delete --exclude='deterministic/')

    echo ">> sync: $src/ -> $dest/"
    # shellcheck disable=SC2029  # expand $src_rel locally into the remote mkdir
    run ssh "${SSH_OPTS[@]}" "$PROD_SSH" "mkdir -p '$PROD_DIR/$src_rel'"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo ">> itemized preview ($src_rel):"
        rsync -ain "${filter[@]}" "$src/" "$dest/"
        return 0
    fi

    # Mirror passes confirm before deleting prod-only files (present on prod,
    # absent on dev). rsync itemizes those as "*deleting <path>".
    if [[ "$use_delete" -eq 1 ]]; then
        local preview deletions
        preview="$(rsync -ain "${filter[@]}" "$src/" "$dest/")"
        deletions="$(printf '%s\n' "$preview" | sed -n 's/^\*deleting[[:space:]]*//p')"
        if [[ -n "$deletions" ]]; then
            echo
            echo "These prod files under $src_rel are NOT on dev and WILL BE DELETED:"
            # shellcheck disable=SC2086  # intentional split: one filename per line
            printf '   %s\n' $deletions
            echo
            read -r -p "Delete the above and mirror dev onto prod? [y/N] " reply < /dev/tty
            if [[ ! "$reply" =~ ^[Yy]$ ]]; then
                echo ">> aborted; prod unchanged."
                exit 1
            fi
        else
            echo ">> no prod-only files to delete; mirroring without confirmation."
        fi
    fi

    run rsync -av "${filter[@]}" "$src/" "$dest/"
}

# One serving-artifact file, additive overwrite. $1 = repo-relative file path.
sync_file() {
    local rel="$1"
    local src="$LOCAL_DIR/$rel"
    if [[ ! -f "$src" ]]; then
        echo ">> skip $rel (not present on dev)"
        return 0
    fi
    local dest_dir
    dest_dir="$(dirname -- "$rel")"
    # shellcheck disable=SC2029  # expand $dest_dir locally into the remote mkdir
    run ssh "${SSH_OPTS[@]}" "$PROD_SSH" "mkdir -p '$PROD_DIR/$dest_dir'"
    run rsync -av --timeout="$RSYNC_TIMEOUT" -e "$RSYNC_SSH" "$src" "$PROD_SSH:$PROD_DIR/$rel"
}

echo ">> sync_to_prod -> $PROD_SSH:$PROD_DIR"
[[ "$DRY_RUN" -eq 1 ]] && echo ">> DRY RUN (no changes)"

# Fail fast if prod is off / off the LAN / outbound ssh not set up.
if ! probe_err="$(ssh "${SSH_OPTS[@]}" "$PROD_SSH" true 2>&1)"; then
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

sync_pass "$MODELS_REL" 1
for rel in "${COLLECTOR_RELS[@]}"; do
    sync_pass "$rel" 0
done
for rel in "${SERVING_FILE_RELS[@]}"; do
    sync_file "$rel"
done

[[ "$DRY_RUN" -eq 1 ]] && echo ">> dry run complete." || echo ">> push complete."
