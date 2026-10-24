#!/usr/bin/env bash
# Push this dev machine's locally-produced data UP to the production server. Run
# from the dev box after a local `meditate` and/or a collector run (`ctg-fetch`,
# `savant-fetch`):
#
#   scripts/sync_to_prod.sh             # sync models + serving artifacts + collector snapshots
#   scripts/sync_to_prod.sh --dry-run   # show what would change, touch nothing
#   scripts/sync_to_prod.sh --league NFL [--dry-run]   # one league only (see below)
#
# Three kinds of pass:
#   * MODELS (mirror, --delete): the models dir is mirrored — prod-only orphans
#     are deleted so a retired cell's pickle doesn't linger. Two subdirs nothing
#     serves from are excluded (and prod's copies protected from --delete): the
#     research models/deterministic/ and models/prior/, the superseded model files
#     a retrain keeps on the box that ran it (training/prior_models.py).
#     Before deleting any prod model absent on dev, the script lists them and asks
#     for confirmation; with no orphans it proceeds silently.
#   * COLLECTOR SNAPSHOTS (additive, NO --delete): the Cleaning the Glass (NBA),
#     Baseball Savant (MLB) and FantasyPoints (NFL) snapshot dirs are pushed
#     additively. They live under player_data/ and team_data/ beside prod-only
#     siblings (nba_players_*.csv, gamelogs, NFL seasons prod pulled and dev
#     never did) — a --delete there would wipe those. Hard rule:
#     NEVER --delete on a player_data/ or team_data/ path. A collector dir absent
#     on dev (catalog not yet populated) is skipped.
#     NFL rides here even though prod runs its own weekly fp-fetch: when that job
#     returns empty (the 2026-09 API change did, for every endpoint), prod's
#     snapshots silently go blank while dev's stay good, and the NFL models trained
#     on dev then meet a feature schema prod cannot produce — which took the whole
#     league out of prophecize for four days.
#   * SERVING ARTIFACTS (single files, additive): gitignored training outputs the
#     prod serving pipeline reads — book_weights.json and stat_calibration.json
#     (consensus weights + per-cell cv/zi read at serve), model_stats.{parquet,csv}
#     (kelly shrinkage), each league's corr_* correlation trio (parlay scoring) and
#     corr_market_summary.parquet (combo-market pricing, the dashboard's heatmap),
#     and MLB's two Savant comp tables (player_data/MLB/affinity_*.csv, which only
#     `meditate` refreshes: a model learns its comp features from the copy on the
#     box that trained it). With prod's `meditate` cron disabled, this script is
#     the ONLY path these take to prod; models synced without them serve against
#     stale calibration (the 2026-08 WNBA book_weights incident). Explicit file list,
#     never a config/-wide pass — config/ also holds prod-local runtime state
#     (goalies.json) that a directory sync would clobber.
#
# NOTE: if prod's weekly `meditate` cron is enabled it rewrites the models dir and
# the serving artifacts above. Don't run this during that window (or at all while
# prod self-trains — the two workflows overwrite each other).
#
# --league LEAGUE sends that league and nothing of any other (when to use it and
# how to undo it: docs/OPERATIONS.md). It is not the three passes narrowed:
#   * the league's model files (models/<LEAGUE>_*) go additively — no --delete,
#     and no collector snapshots at all;
#   * book_weights.json, stat_calibration.json and model_stats.{parquet,csv} each
#     hold every league, so prod's own copies are fetched and only this league's
#     entries replaced (build_league_only_serving.py prints the proof). Nothing is
#     sent unless that proof passes and prod's copies still have the checksums
#     they had when fetched;
#   * every file a send replaces is kept on prod, under
#     league-send-backup/<LEAGUE>-<UTC stamp>/ beside the project root and in the
#     data folder's layout, so the last line printed is one `cp -a` that puts
#     them all back.
# --dry-run still fetches prod's three files (a read) and writes nothing there.
#
# Environment (optional overrides):
#   PROD_SSH             ssh target for prod  (default: sportstradamus@192.168.1.84)
#   PROD_DIR             prod project root    (default: /home/sportstradamus/Sportstradamus)
#   SSH_CONNECT_TIMEOUT  seconds before giving up if prod is off (default: 15)
#   RSYNC_TIMEOUT        per-transfer I/O timeout in seconds      (default: 120)
#   LEAGUE_SEND_REHEARSAL_DIR  --league only: a local stand-in for prod's project
#                        root, holding a copy of its data folder at the same
#                        relative path. Every remote path then points into it and
#                        prod is not contacted.
set -euo pipefail

PROD_SSH="${PROD_SSH:-sportstradamus@192.168.1.84}"
PROD_DIR="${PROD_DIR:-/home/sportstradamus/Sportstradamus}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-15}"
RSYNC_TIMEOUT="${RSYNC_TIMEOUT:-120}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
LOCAL_DIR="$(dirname -- "$SCRIPT_DIR")"

usage() {
    echo "usage: $(basename "$0") [--league LEAGUE] [-n|--dry-run]" >&2
    exit 64
}

DRY_RUN=0
LEAGUE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n | --dry-run) DRY_RUN=1 ;;
        --league)
            # An empty or mistyped league must not fall through to the full sync.
            [[ "${2:-}" =~ ^[[:upper:]]+$ ]] || usage
            LEAGUE="$2"
            shift
            ;;
        *) usage ;;
    esac
    shift
done
# The stand-in covers --league only: the full sync would still reach the real prod.
if [[ -n "${LEAGUE_SEND_REHEARSAL_DIR:-}" && -z "$LEAGUE" ]]; then
    echo "LEAGUE_SEND_REHEARSAL_DIR rehearses --league sends only" >&2
    exit 64
fi

DATA_REL="src/sportstradamus/data"
MODELS_REL="src/sportstradamus/data/models"
# Collector snapshot dirs — synced additively (NO --delete; see header).
COLLECTOR_RELS=(
    "src/sportstradamus/data/player_data/NBA/cleaningtheglass"
    "src/sportstradamus/data/team_data/NBA/cleaningtheglass"
    "src/sportstradamus/data/player_data/MLB/baseballsavant"
    "src/sportstradamus/data/team_data/MLB/baseballsavant"
    "src/sportstradamus/data/player_data/NFL"
    "src/sportstradamus/data/team_data/NFL"
)
# The two lists both modes share, so a league send cannot drift from the full sync.
MLB_COMP_RELS=(
    "src/sportstradamus/data/player_data/MLB/affinity_hittersByHittingProfile_matchScores.csv"
    "src/sportstradamus/data/player_data/MLB/affinity_pitchersBySHV_matchScores.csv"
)
CORR_FILES=(
    corr_same_team.parquet corr_opposing.parquet corr_metadata.json corr_market_summary.parquet
)
# The files that hold every league at once, relative to the data dir: --league
# merges them where the full sync sends them whole (see header).
SHARED_RELS=(config/book_weights.json config/stat_calibration.json training/model_stats.parquet)
# Serving artifacts — gitignored files meditate/correlate write on dev and the
# prod pipelines read at serve (see header). Files absent on dev are skipped.
SERVING_FILE_RELS=(
    "src/sportstradamus/data/config/book_weights.json"
    "src/sportstradamus/data/config/stat_calibration.json"
    "src/sportstradamus/data/training/model_stats.parquet"
    "src/sportstradamus/data/training/model_stats.csv"
    "${MLB_COMP_RELS[@]}"
)
for league_dir in "$LOCAL_DIR"/src/sportstradamus/data/leagues/*/; do
    [[ -d "$league_dir" ]] || continue
    for name in "${CORR_FILES[@]}"; do
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
    [[ "$use_delete" -eq 1 ]] && filter+=(--delete --exclude='deterministic/' --exclude='prior/')

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

# --league: one additive rsync of the local files $2.. into prod's <data dir>/$1.
# The file each one replaces moves to $BACKUP/$1, so the backup folder mirrors the
# data folder and a single `cp -a` restores all of it.
send_keeping_replaced() {
    local sub="$1"
    shift
    local flags=(-ai --backup --backup-dir="$BACKUP/$sub")
    [[ "$DRY_RUN" -eq 1 ]] && flags+=(-n)
    rsync "${flags[@]}" --timeout="$RSYNC_TIMEOUT" -e "$RSYNC_SSH" "$@" "$DEST/$DATA_REL/$sub/"
}

# --league: send $LEAGUE's files and nothing of any other league (see header).
sync_league() {
    local on_prod prod_dir backup_root
    if [[ -n "${LEAGUE_SEND_REHEARSAL_DIR:-}" ]]; then
        echo ">> REHEARSAL: prod is not contacted"
        # Absolute, because rsync reads a relative --backup-dir against the destination.
        DEST="$(cd -- "$LEAGUE_SEND_REHEARSAL_DIR" && pwd)"
        prod_dir="$DEST"
        backup_root="$DEST/league-send-backup"
        on_prod=(bash -c)
    else
        DEST="$PROD_SSH:$PROD_DIR"
        prod_dir="$PROD_DIR"
        backup_root="$(dirname -- "$PROD_DIR")/league-send-backup"
        on_prod=(ssh "${SSH_OPTS[@]}" "$PROD_SSH")
    fi
    BACKUP="$backup_root/$LEAGUE-$(date -u +%Y%m%dT%H%M%SZ)"
    echo ">> sync_to_prod --league $LEAGUE -> $DEST"
    [[ "$DRY_RUN" -eq 1 ]] && echo ">> DRY RUN (nothing is written on prod)"

    local work fetched current rel
    work="$(mktemp -d)"
    # shellcheck disable=SC2064  # expand $work now: the local is gone when the trap fires
    trap "rm -rf '$work'" EXIT
    local base="$work/prod" merged="$work/merged"
    mkdir "$base"
    echo ">> fetch prod's shared files"
    for rel in "${SHARED_RELS[@]}"; do
        rsync -a --timeout="$RSYNC_TIMEOUT" -e "$RSYNC_SSH" "$DEST/$DATA_REL/$rel" "$base/"
    done
    fetched="$(cd "$base" && md5sum "${SHARED_RELS[@]##*/}" | awk '{print $1}')"

    echo ">> merge: $LEAGUE's entries from this box, every other league's from prod"
    if ! (cd "$LOCAL_DIR" && poetry run python "$SCRIPT_DIR/build_league_only_serving.py" \
        --league "$LEAGUE" --here-dir "$LOCAL_DIR/$DATA_REL" \
        --prod-dir "$base" --out-dir "$merged"); then
        echo "STOP: the merge's proof did not pass; nothing sent." >&2
        exit 1
    fi

    current="$("${on_prod[@]}" "cd '$prod_dir/$DATA_REL' && md5sum ${SHARED_RELS[*]}" \
        | awk '{print $1}')"
    if [[ "$current" != "$fetched" ]]; then
        echo "STOP: a shared file changed on prod since the fetch; nothing sent. Run again." >&2
        exit 1
    fi
    echo ">> prod's shared files are unchanged since the fetch"

    local lg="${LEAGUE,,}"
    [[ "$DRY_RUN" -eq 1 ]] || echo ">> the files this replaces are kept on prod in $BACKUP"
    echo ">> $LEAGUE model files"
    send_keeping_replaced models "$LOCAL_DIR/$MODELS_REL/${LEAGUE}_"*
    echo ">> shared files, $LEAGUE's entries replaced"
    send_keeping_replaced config "$merged/book_weights.json" "$merged/stat_calibration.json"
    send_keeping_replaced training "$merged/model_stats.parquet" "$merged/model_stats.csv"
    echo ">> $LEAGUE correlation files"
    (cd "$LOCAL_DIR/$DATA_REL/leagues/$lg" \
        && send_keeping_replaced "leagues/$lg" "${CORR_FILES[@]}")
    if [[ "$LEAGUE" == MLB ]]; then
        echo ">> MLB comp tables"
        (cd "$LOCAL_DIR" && send_keeping_replaced player_data/MLB "${MLB_COMP_RELS[@]}")
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo ">> dry run complete."
    else
        echo ">> sent. To undo, on prod:  cp -a $BACKUP/. $prod_dir/$DATA_REL/"
    fi
}

if [[ -n "$LEAGUE" ]]; then
    sync_league
    exit 0
fi

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
