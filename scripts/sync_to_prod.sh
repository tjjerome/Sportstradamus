#!/usr/bin/env bash
# Push this dev machine's trained model pickles UP to the production server,
# mirroring the models dir (replace existing, delete prod-only orphans). Run
# from the dev box after a local `meditate`:
#
#   scripts/sync_to_prod.sh             # mirror models to prod
#   scripts/sync_to_prod.sh --dry-run   # show what would change, touch nothing
#
# Only the top-level *.mdl files are synced; the models/deterministic/ research
# subdir is excluded (and prod's copy of it is protected from --delete). Before
# deleting any prod model that is NOT present on dev, the script lists those
# files and asks for confirmation; with no such orphans it proceeds silently.
#
# NOTE: prod's weekly `meditate` cron (Fri 01:00) rewrites this same dir. Don't
# run this during that window or you may race its model writes.
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
SRC="$LOCAL_DIR/$MODELS_REL"
DEST="$PROD_SSH:$PROD_DIR/$MODELS_REL"
SSH_OPTS=(-o "ConnectTimeout=$SSH_CONNECT_TIMEOUT" -o BatchMode=yes)
RSYNC_SSH="ssh -o ConnectTimeout=$SSH_CONNECT_TIMEOUT -o BatchMode=yes"
RSYNC_FILTER=(--delete --exclude='deterministic/')

run() {
    printf '+ %s\n' "$*"
    [[ "$DRY_RUN" -eq 1 ]] && return 0
    "$@"
}

echo ">> sync_to_prod: $SRC/ -> $DEST/"
[[ "$DRY_RUN" -eq 1 ]] && echo ">> DRY RUN (no changes)"

# 1. Fail fast if prod is off / off the LAN / outbound ssh not set up.
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

# 2. Make sure the destination dir exists so the first-ever push can't fail.
# shellcheck disable=SC2029
run ssh "${SSH_OPTS[@]}" "$PROD_SSH" "mkdir -p '$PROD_DIR/$MODELS_REL'"

# 3. Dry-run the mirror to find prod models that would be DELETED and not
#    replaced (present on prod, absent on dev). rsync itemizes these as
#    "*deleting <path>"; that set is exactly the orphans we must warn about.
preview="$(rsync -ain "${RSYNC_FILTER[@]}" --timeout="$RSYNC_TIMEOUT" \
    -e "$RSYNC_SSH" "$SRC/" "$DEST/")"
deletions="$(printf '%s\n' "$preview" | sed -n 's/^\*deleting[[:space:]]*//p')"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo ">> itemized preview (no changes made):"
    printf '%s\n' "$preview"
    exit 0
fi

# 4. Confirm only if the mirror would delete prod-only models.
if [[ -n "$deletions" ]]; then
    echo
    echo "These prod models are NOT on dev and WILL BE DELETED:"
    # shellcheck disable=SC2086  # intentional split: one model filename per line
    printf '   %s\n' $deletions
    echo
    read -r -p "Delete the above and mirror dev's models onto prod? [y/N] " reply < /dev/tty
    if [[ ! "$reply" =~ ^[Yy]$ ]]; then
        echo ">> aborted; prod unchanged."
        exit 1
    fi
else
    echo ">> no prod-only models to delete; mirroring without confirmation."
fi

# 5. Real push: mirror dev's models onto prod.
run rsync -av "${RSYNC_FILTER[@]}" --timeout="$RSYNC_TIMEOUT" \
    -e "$RSYNC_SSH" "$SRC/" "$DEST/"

echo ">> push complete."
