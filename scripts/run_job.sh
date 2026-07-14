#!/usr/bin/env bash
# Wrapper for cron-driven Sportstradamus pipelines.
#
# Usage:
#   run_job.sh <job> [extra args passed through to the underlying command]
#
# Jobs:
#   prophecize         poetry run prophecize
#   confer             poetry run confer
#   close-lines        poetry run confer --close-lines
#   meditate           poetry run meditate
#   reflect            poetry run reflect
#   gate-status        scripts/gate_status_update.sh (monthly: refresh main ship_config, open PR)
#   fp-fetch           poetry run fp-fetch run (weekly: snapshot Fantasy Points Data Suite)
#   ctg-fetch          poetry run ctg-fetch run (snapshot Cleaning the Glass NBA tables)
#   savant-fetch       poetry run savant-fetch run (snapshot Baseball Savant MLB leaderboards)
#   ledger-commit      poetry run ledger-commit --run-slot {morning,afternoon}
#
# Environment (optional):
#   HEALTHCHECK_URL_<JOB>   per-job healthchecks.io URL (e.g. HEALTHCHECK_URL_PROPHECIZE)
#   HEALTHCHECK_URL         fallback URL used if the per-job one is unset
#   SPORTSTRADAMUS_DIR      project root (default: parent of this script)
#   LOG_DIR                 log directory (default: $SPORTSTRADAMUS_DIR/logs)
#   ARCHIVE_LOCK_TIMEOUT    seconds to wait for the shared archive lock (default: 900)
#   GIT_PULL                set 0 to skip the pre-job devel pull (default: 1)
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

case "$JOB" in
    prophecize)   CMD=(poetry run prophecize) ;;
    confer)       CMD=(poetry run confer) ;;
    close-lines)  CMD=(poetry run confer --close-lines) ;;
    meditate)     CMD=(poetry run meditate) ;;
    reflect)      CMD=(poetry run reflect) ;;
    gate-status)  CMD=(bash "$SCRIPT_DIR/gate_status_update.sh") ;;
    fp-fetch)     CMD=(poetry run fp-fetch run) ;;
    ctg-fetch)    CMD=(poetry run ctg-fetch run) ;;
    savant-fetch) CMD=(poetry run savant-fetch run) ;;
    ledger-commit) CMD=(poetry run ledger-commit) ;;
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

ping_hc() {
    local suffix="$1"  # "" for success, "/fail" for failure, "/start" for start
    [[ -z "$HC_URL" ]] && return 0
    curl -fsS --max-time 10 --retry 3 -o /dev/null "${HC_URL}${suffix}" || true
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
    ping_hc "/start"

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
        ping_hc ""
    else
        log "FAIL job=$JOB duration=${duration}s exit=$status"
        # Send last 50 lines of the log as the failure body so the alert is useful.
        if [[ -n "$HC_URL" ]]; then
            tail -n 50 "$LOG_FILE" | curl -fsS --max-time 10 --retry 3 \
                --data-binary @- -o /dev/null "${HC_URL}/fail" || true
        fi
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
    ping_hc "/fail"
    exit 75  # EX_TEMPFAIL
fi
wait_end=$(date +%s)
wait_duration=$((wait_end - wait_start))
[[ $wait_duration -gt 1 ]] && log "WAIT job=$JOB archive_lock_wait=${wait_duration}s"

run
