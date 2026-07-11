"""Per-spec run results and the transient JSON report for collectors.

The runner records one :class:`RunResult` per endpoint and dumps them to a
tempdir report at the end of a ``run`` / ``backfill``. Kept apart from the
drive loop so the reporting and small I/O utilities (parquet row count, URL
build, error-to-dict) have one home.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit, urlunsplit

import click

# Status values for `RunResult.status` — also the dict keys in the report
# summary. ``ok`` = wrote rows; ``empty`` = wrote a zero-row parquet;
# ``fetch_failed`` = HTTP error / decode error / etc; ``routing_failed`` =
# ``path_for`` couldn't resolve a target for the catalog entry; ``skipped``
# = a non-empty parquet already exists for this cell and the user didn't
# pass ``--refetch``.
RESULT_OK = "ok"
RESULT_EMPTY = "empty"
RESULT_FETCH_FAILED = "fetch_failed"
RESULT_ROUTING_FAILED = "routing_failed"
RESULT_SKIPPED = "skipped"

# Body preview length in the failure report. Long enough to spot HTML error
# pages, JSON error envelopes, or compressed binary blobs at a glance; short
# enough not to inflate the report file when a tool returns a huge body that
# happens to be unparseable.
_REPORT_PREVIEW_CHARS = 500


@dataclass
class RunResult:
    """One spec's outcome from a single ``run`` / ``backfill`` iteration.

    Serialised verbatim into the report file so the human has everything
    needed to diagnose failing specs without re-running.
    """

    name: str
    url: str
    method: str
    status: str
    season: int
    week: int
    period: str | None = None
    rows: int = 0
    path: str | None = None
    error_class: str | None = None
    error_message: str | None = None
    http_status: int | None = None
    response_preview: str | None = None
    request_body: dict | list | None = None


def write_run_report(
    results: list[RunResult],
    *,
    command: str,
    report_prefix: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Dump the per-spec results to ``/tmp/{report_prefix}_report_<ts>_<uniq>.json``.

    The report is transient debug output — it lives under tempdir so users
    (and CI) don't accidentally commit it. The unique suffix keeps two runs
    that fire in the same second (parallel test workers) from writing to the
    same file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fd, name = tempfile.mkstemp(prefix=f"{report_prefix}_report_{timestamp}_", suffix=".json")
    os.close(fd)
    path = Path(name)
    summary = {
        "total": len(results),
        RESULT_OK: sum(1 for r in results if r.status == RESULT_OK),
        RESULT_EMPTY: sum(1 for r in results if r.status == RESULT_EMPTY),
        RESULT_FETCH_FAILED: sum(1 for r in results if r.status == RESULT_FETCH_FAILED),
        RESULT_ROUTING_FAILED: sum(1 for r in results if r.status == RESULT_ROUTING_FAILED),
        RESULT_SKIPPED: sum(1 for r in results if r.status == RESULT_SKIPPED),
    }
    payload: dict[str, Any] = {
        "command": command,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "results": [asdict(r) for r in results],
    }
    # ``extra`` is namespaced under "context" so callers can't silently
    # clobber the top-level "command" / "summary" / "results" fields by
    # passing a conflicting key.
    if extra:
        payload["context"] = extra
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def summarize(
    results: list[RunResult],
    *,
    report_prefix: str,
    command: str,
    extra: dict[str, Any],
) -> tuple[Path, list[RunResult]]:
    """Write the report, echo the ok/skip/empty/failed summary, return failures."""
    report_path = write_run_report(
        results, command=command, report_prefix=report_prefix, extra=extra
    )
    failures = [r for r in results if r.status in (RESULT_FETCH_FAILED, RESULT_ROUTING_FAILED)]
    click.echo("", err=True)
    click.echo(f"Report: {report_path}", err=True)
    click.echo(
        f"Summary: {sum(1 for r in results if r.status == RESULT_OK)} ok, "
        f"{sum(1 for r in results if r.status == RESULT_SKIPPED)} skip, "
        f"{sum(1 for r in results if r.status == RESULT_EMPTY)} empty, "
        f"{len(failures)} failed.",
        err=True,
    )
    return report_path, failures


def existing_parquet_rows(path: Path) -> int:
    """Return the row count of an existing parquet, or 0 if missing / unreadable.

    Reads only parquet metadata (microseconds per file), not the full data,
    so this is cheap enough to call once per cell. A 0 return means "treat as
    not yet fetched" — either the file doesn't exist or it's a previous
    failed run that wrote a header with no rows.
    """
    if not path.is_file():
        return 0
    try:
        import pyarrow.parquet as pq

        return pq.read_metadata(path).num_rows
    except (OSError, ValueError):
        return 0


def truncate_for_preview(body: Any) -> str:
    """JSON-encode ``body`` and truncate to ``_REPORT_PREVIEW_CHARS``.

    Used on the empty-rows path so the report carries enough of the actual
    response to tell apart "filters too restrictive" (rows: []) from
    "different response shape" from "cached empty result".
    """
    try:
        text = json.dumps(body, default=str)
    except (TypeError, ValueError):
        text = repr(body)
    return text[:_REPORT_PREVIEW_CHARS]


def exc_to_err_dict(exc: BaseException) -> dict[str, Any]:
    """Convert an exception into the JSON-friendly fields the report needs."""
    info: dict[str, Any] = {
        "error_class": exc.__class__.__name__,
        "error_message": str(exc),
    }
    response = getattr(exc, "response", None)
    if response is not None:
        info["http_status"] = getattr(response, "status_code", None)
        text = getattr(response, "text", None)
        if isinstance(text, str):
            info["response_preview"] = text[:_REPORT_PREVIEW_CHARS]
    return info


def build_url(url: str, params: dict[str, str]) -> str:
    parsed = urlsplit(url)
    query = urlencode(params) if params else ""
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, query, ""))
