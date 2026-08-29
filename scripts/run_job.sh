#!/usr/bin/env bash
# Wrapper for cron-driven Sportstradamus pipelines.
#
# Usage:
#   run_job.sh <job> [extra args passed through to the underlying command]
#
# Jobs (all dispatch through `python -m sportstradamus`, see the case map):
#   prophecize         prediction pipeline -> dashboard snapshots
#   confer             fetch current odds/props
#   close-lines        confer --close-lines
#   meditate           train/retrain models
#   reflect            nightly resolution
#   gate-status        scripts/gate_status_update.sh (monthly: refresh main ship_config, open PR)
#   fp-fetch           fetch fp run (weekly: snapshot Fantasy Points Data Suite)
#   ctg-fetch          fetch ctg run (snapshot Cleaning the Glass NBA tables)
#   savant-fetch       fetch savant run (snapshot Baseball Savant MLB leaderboards)
#   ledger-commit      bet ledger-commit --run-slot {morning,afternoon}
#
# Environment (optional):
#   HEALTHCHECK_URL_<JOB>   per-job healthchecks.io URL (e.g. HEALTHCHECK_URL_PROPHECIZE)
#   HEALTHCHECK_URL         fallback URL used if the per-job one is unset
#   SPORTSTRADAMUS_DIR      project root (default: parent of this script)
#   LOG_DIR                 log directory (default: $SPORTSTRADAMUS_DIR/logs)
#   ARCHIVE_LOCK_TIMEOUT    seconds to wait for the shared archive lock (default: 900)
#   GIT_PULL                set 0 to skip the pre-job devel pull (default: 1)
#
# Healthchecks pings:
#   Every ping carries a `?rid=` run ID and a body whose first line is
#   `job=<job> host=<host> status=…`, so an alert names its own job and its own
#   elapsed time even when several HEALTHCHECK_URL_<JOB> share one check UUID.
#
# Concurrency model:
#   - Per-job flock (-n): a second invocation of the *same* job is skipped.
#   - Pull flock (-n): only one job pulls devel at a time; concurrent jobs skip
#     the pull and run the checkout as-is. A failed pull never blocks the job.
#   - Shared archive flock (-w): all jobs serialize on the DuckDB archive so
#     they don't fight over the single-writer lock. A queued job waits up to
#     ARCHIVE_LOCK_TIMEOUT seconds, then logs FAIL_LOCK and exits 75 (EX_TEMPFAIL).
#
# Exit codes: propagates the wrapped command's exit code.

set -u
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_DIR="${SPORTSTRADAMUS_DIR:-$(dirname -- "$SCRIPT_DIR")}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"
LOCK_DIR="${LOCK_DIR:-/tmp}"

if [[ $# -lt 1 ]]; then
    echo "usage: $(basename "$0") <prophecize|confer|close-lines|meditate|reflect|gate-status|fp-fetch|ctg-fetch|savant-fetch|ledger-commit> [args...]" >&2
    exit 64
fi

JOB="$1"
shift

# Module form (python -m) never consults console-script stubs, so a job that
# fires mid-`poetry install` can't hit a half-written entry point.
UMBRELLA=(poetry run python -m sportstradamus)
case "$JOB" in
    prophecize)   CMD=("${UMBRELLA[@]}" prophecize) ;;
    confer)       CMD=("${UMBRELLA[@]}" confer) ;;
    close-lines)  CMD=("${UMBRELLA[@]}" confer --close-lines) ;;
    meditate)     CMD=("${UMBRELLA[@]}" meditate) ;;
    reflect)      CMD=("${UMBRELLA[@]}" reflect) ;;
    gate-status)  CMD=(bash "$SCRIPT_DIR/gate_status_update.sh") ;;
    fp-fetch)     CMD=("${UMBRELLA[@]}" fetch fp run) ;;
    ctg-fetch)    CMD=("${UMBRELLA[@]}" fetch ctg run) ;;
    savant-fetch) CMD=("${UMBRELLA[@]}" fetch savant run) ;;
    ledger-commit) CMD=("${UMBRELLA[@]}" bet ledger-commit) ;;
    *)
        echo "unknown job: $JOB" >&2
        exit 64
        ;;
esac
CMD+=("$@")

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${JOB}.log"
LOCK_FILE="$LOCK_DIR/sportstradamus-${JOB}.lock"
ARCHIVE_LOCK_FILE="$LOCK_DIR/sportstradamus-archive.lock"
ARCHIVE_LOCK_TIMEOUT="${ARCHIVE_LOCK_TIMEOUT:-900}"
PULL_LOCK_FILE="$LOCK_DIR/sportstradamus-git-pull.lock"
# Nightly-folded dashboard modifier corrections dirty these two tracked files;
# they are reproducible from HEAD + data/runtime/modifier_overrides.json, so
# the pull path resets them and re-folds after (see pull_devel).
MODIFIER_CONFIGS=(
    src/sportstradamus/data/config/banned_combos.json
    src/sportstradamus/data/config/parlay_rake.json
)

# Resolve the healthcheck URL: per-job override wins, fall back to the shared one.
job_upper="$(echo "$JOB" | tr '[:lower:]-' '[:upper:]_')"
hc_var="HEALTHCHECK_URL_${job_upper}"
HC_URL="${!hc_var:-${HEALTHCHECK_URL:-}}"

# Pairs this run's /start with its own terminating ping. Without it healthchecks
# times the run from whichever job pinged /start last, so a shared check reports
# some other job's elapsed time on our failure.
RUN_ID="$(cat /proc/sys/kernel/random/uuid)"
HC_TAG="job=$JOB host=${HOSTNAME%%.*}"

ping_hc() {
    local suffix="$1" body="${2-}"  # "" for success, "/fail" for failure, "/start" for start
    [[ -z "$HC_URL" ]] && return 0
    printf '%s' "$body" | curl -fsS --max-time 10 --retry 3 --data-binary @- \
        -o /dev/null "${HC_URL}${suffix}?rid=${RUN_ID}" || true
}

# The logs are mostly tqdm carriage-return frames, so a raw tail is one
# unreadable progress bar. Split on \r and drop the frames.
log_excerpt() {
    tail -c 262144 "$LOG_FILE" | tr '\r' '\n' \
        | grep -vE 'it/s\]|s/it\]|\?it/s\]' | tail -n 40
}

# Names the live jobs when we lose the archive lock — the holder is one of
# them, and "waited 900s" alone doesn't say who starved us.
running_jobs() {
    local lock name found=()
    for lock in "$LOCK_DIR"/sportstradamus-*.lock; do
        name="${lock##*/sportstradamus-}"
        name="${name%.lock}"
        case "$name" in archive | git-pull | "$JOB") continue ;; esac
        flock -n "$lock" true 2>/dev/null || found+=("$name")
    done
    echo "${found[*]:-none}"
}

log() {
    printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >>"$LOG_FILE"
}

pull_devel() {
    (
        if ! flock -n 7; then
            log "PULL_SKIP job=$JOB reason=pull_in_progress"
            exit 0
        fi
        git checkout -- "${MODIFIER_CONFIGS[@]}" >>"$LOG_FILE" 2>&1 || true
        if ! git pull --ff-only origin devel >>"$LOG_FILE" 2>&1; then
            log "PULL_WARN job=$JOB reason=pull_failed action=run_existing_checkout"
            exit 0
        fi
        # A pull that moves the dependency set must sync the venv before the
        # job runs, or the checkout imports against stale packages.
        if git diff --name-only 'HEAD@{1}..HEAD' 2>/dev/null | grep -qE '^(pyproject\.toml|poetry\.lock)$'; then
            if poetry install --no-interaction >>"$LOG_FILE" 2>&1; then
                log "PULL_INSTALL job=$JOB ok"
            else
                log "PULL_WARN job=$JOB reason=poetry_install_failed"
            fi
        fi
        # Re-apply overlay corrections onto the fresh configs; entries devel
        # now carries are pruned from the overlay.
        if poetry run python -m sportstradamus.scripts.fold_modifier_overrides --prune \
            >>"$LOG_FILE" 2>&1; then
            log "PULL job=$JOB ok"
        else
            log "PULL_WARN job=$JOB reason=fold_failed"
        fi
    ) 7>"$PULL_LOCK_FILE"
}

run() {
    log "START job=$JOB cmd=${CMD[*]}"
    ping_hc "/start" "$HC_TAG status=start
cmd=${CMD[*]}"

    local start_ts end_ts duration status
    start_ts=$(date +%s)
    set +e
    "${CMD[@]}" >>"$LOG_FILE" 2>&1
    status=$?
    set -e
    end_ts=$(date +%s)
    duration=$((end_ts - start_ts))

    if [[ $status -eq 0 ]]; then
        log "OK job=$JOB duration=${duration}s"
        ping_hc "" "$HC_TAG status=ok duration=${duration}s"
    else
        log "FAIL job=$JOB duration=${duration}s exit=$status"
        ping_hc "/fail" "$HC_TAG status=fail duration=${duration}s exit=$status
log=$LOG_FILE
--- last 40 log lines ---
$(log_excerpt)"
    fi
    return $status
}

cd "$PROJECT_DIR"

# flock -n: bail immediately if a previous run of this same job is still going.
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    log "SKIP job=$JOB reason=already_running"
    exit 0
fi

# Self-deploy: refresh the checkout from devel before running (GIT_PULL=0 to
# skip). Never blocks the job — a failed pull runs the existing checkout.
[[ "${GIT_PULL:-1}" == "1" ]] && pull_devel

# Shared archive lock: serialize against any other job touching the DuckDB
# archive so we don't crash on DuckDB's single-writer constraint. Wait up to
# ARCHIVE_LOCK_TIMEOUT seconds for the holder to finish.
exec 8>"$ARCHIVE_LOCK_FILE"
wait_start=$(date +%s)
if ! flock -w "$ARCHIVE_LOCK_TIMEOUT" 8; then
    waited=$(( $(date +%s) - wait_start ))
    log "FAIL_LOCK job=$JOB reason=archive_lock_timeout waited=${waited}s"
    ping_hc "/fail" "$HC_TAG status=fail_lock waited=${waited}s
$JOB never started: gave up waiting for the shared DuckDB archive lock.
jobs running: $(running_jobs)"
    exit 75  # EX_TEMPFAIL
fi
wait_end=$(date +%s)
wait_duration=$((wait_end - wait_start))
[[ $wait_duration -gt 1 ]] && log "WAIT job=$JOB archive_lock_wait=${wait_duration}s"

run
