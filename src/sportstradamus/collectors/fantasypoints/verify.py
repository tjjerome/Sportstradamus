"""Spot-check downloaded Fantasy Points parquets against expectations.

Used after a ``fp-fetch run`` / ``backfill`` to confirm the files on disk
actually contain the (season, week, mode) slice the user asked for. The
bug this catches: a request whose week filter didn't bind returns
whole-season data, which still lands at the right path and still looks
like a healthy row count.

Each response row is one aggregate per entity over the selected weeks, so
there is no per-row week column to check. What the rows do carry is
``games`` — how many of that entity's games went into the aggregate — and
that is the invariant the week filter controls: a weekly request yields at
most one game per entity, a season-to-date request through week N at most
N. An unfiltered response fails immediately.

Anything off is reported as a :class:`VerificationIssue`. The CLI wrapper
(``fp-fetch verify``) prints per-spec results and exits non-zero if any
spec yielded an ``error``-severity issue.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from sportstradamus.collectors.catalog import EndpointSpec
from sportstradamus.collectors.fantasypoints.transform import parquet_path_for_spec

# Per-entity game count, present on every tool. The only column that
# reflects how wide a window the response actually covered.
_GAMES_COL = "games"

# Identity columns the stats layer groups and joins on. Their absence is an
# error rather than a warning: a missing one is indistinguishable from an
# empty week downstream, and silently drops every feature built on the file.
_TEAM_IDENTITY = ("teamTeamId", "teamAbbreviation")
_PLAYER_IDENTITY = ("playerPlayerId", "playerFirstName", "playerLastName")

# Maximum number of column names to echo in a "missing column" error
# message — enough context to spot a schema change without flooding the
# terminal.
_MISSING_COL_PREVIEW = 8

Mode = Literal["weekly", "season_to_date", "postseason"]
DEFAULT_MODE: Mode = "weekly"

Severity = Literal["error", "warn", "info"]


@dataclass(frozen=True)
class VerificationIssue:
    """One observation from spot-checking a downloaded parquet.

    Severity meanings:

    - ``error``: file or contents contradict the requested (season, week,
      mode) — almost certainly a download/catalog bug.
    - ``warn``: surprising but not unambiguously wrong (e.g. zero rows).
    - ``info``: noted but expected.
    """

    spec_name: str
    code: str
    severity: Severity
    message: str
    path: str


def _single_issue(
    spec: EndpointSpec, *, code: str, severity: Severity, message: str, path: str
) -> list[VerificationIssue]:
    """Wrap one issue in the list every check site returns."""
    return [
        VerificationIssue(
            spec_name=spec.name, code=code, severity=severity, message=message, path=path
        )
    ]


def verify_spec(
    spec: EndpointSpec,
    *,
    season: int,
    week: int,
    mode: Mode = DEFAULT_MODE,
) -> list[VerificationIssue]:
    """Spot-check the parquet for one catalog entry.

    Args:
        spec: Catalog entry to check.
        season: NFL season year the parquet should cover.
        week: NFL week the parquet should cover (mode-dependent meaning).
        mode: ``weekly`` / ``season_to_date`` / ``postseason``.

    Returns:
        A list of :class:`VerificationIssue`, empty when every check passed.
    """
    try:
        path = parquet_path_for_spec(spec, season=season, week=week, mode=mode)
    except ValueError as exc:
        return _single_issue(
            spec, code="routing_failed", severity="error", message=str(exc), path="<unrouted>"
        )
    if not path.is_file():
        return _single_issue(
            spec,
            code="file_missing",
            severity="error",
            message=(
                f"Expected parquet not found. Re-run fp-fetch run "
                f"--week {week} --season {season} --mode {mode} --only {spec.name}"
            ),
            path=str(path),
        )
    df = pd.read_parquet(path)
    return _check_dataframe(df, spec=spec, season=season, week=week, mode=mode, path=path)


def verify_catalog(
    specs: list[EndpointSpec],
    *,
    season: int,
    week: int,
    mode: Mode = DEFAULT_MODE,
) -> dict[str, list[VerificationIssue]]:
    """Run :func:`verify_spec` for every spec; return ``{name: issues}``.

    Specs with no issues map to an empty list, so callers can both count
    pass/fail and pull issue detail for failures from the same dict.
    """
    return {spec.name: verify_spec(spec, season=season, week=week, mode=mode) for spec in specs}


def max_games_for_mode(*, week: int, mode: Mode) -> int:
    """Return the largest per-entity ``games`` value valid for one (week, mode).

    Mirrors :func:`source.period_params` — a divergence between the request
    builder and this check shows up as a test failure rather than silent
    acceptance of an unfiltered response.
    """
    return week if mode == "season_to_date" else 1


def _check_dataframe(
    df: pd.DataFrame,
    *,
    spec: EndpointSpec,
    season: int,
    week: int,
    mode: Mode,
    path: Path,
) -> list[VerificationIssue]:
    """Run row-count, window and identity checks against one loaded parquet."""
    path_str = str(path)
    if df.empty:
        return _single_issue(
            spec,
            code="file_empty",
            severity="warn",
            message="Parquet has zero rows. FP may not publish this tool for "
            f"(season={season}, week={week}, mode={mode}).",
            path=path_str,
        )
    return [
        *_check_window(df, spec=spec, week=week, mode=mode, path=path_str),
        *_check_identity(df, spec=spec, path=path_str),
    ]


def _check_window(
    df: pd.DataFrame,
    *,
    spec: EndpointSpec,
    week: int,
    mode: Mode,
    path: str,
) -> list[VerificationIssue]:
    if _GAMES_COL not in df.columns:
        return _single_issue(
            spec,
            code="missing_games_column",
            severity="warn",
            message=f"Column {_GAMES_COL!r} missing — cannot verify the week window. "
            f"Columns present: {list(df.columns)[:_MISSING_COL_PREVIEW]}...",
            path=path,
        )
    allowed = max_games_for_mode(week=week, mode=mode)
    observed = pd.to_numeric(df[_GAMES_COL], errors="coerce").max()
    if pd.notna(observed) and observed > allowed:
        return _single_issue(
            spec,
            code="window_too_wide",
            severity="error",
            message=f"Mode {mode!r} for week {week} allows at most {allowed} game(s) per "
            f"entity, but a row aggregates {int(observed)}. The week filter did not "
            "bind — check the catalog entry's params against a browser request.",
            path=path,
        )
    return []


def _check_identity(df: pd.DataFrame, *, spec: EndpointSpec, path: str) -> list[VerificationIssue]:
    """Confirm the columns the stats layer groups and joins on survived."""
    required = _PLAYER_IDENTITY if spec.name.startswith("player_") else _TEAM_IDENTITY
    missing = [col for col in required if col not in df.columns]
    if missing:
        return _single_issue(
            spec,
            code="missing_identity_columns",
            severity="error",
            message=f"Identity columns {missing} absent — every feature built on this "
            "file would silently vanish downstream. The response schema likely "
            "changed; re-capture the endpoint.",
            path=path,
        )
    return []
