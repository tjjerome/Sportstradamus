"""Spot-check downloaded Fantasy Points parquets against expectations.

Used after a ``fp-fetch run`` / ``backfill`` to confirm the files on
disk actually contain the (season, week, mode) slice the user asked
for. The original bug this catches: pre-v2 catalog bodies had no week
filter, so FP returned whole-season data labelled with the wrong week
even though the file landed at the right path.

Per spec the check loads the expected parquet (path comes from
:func:`transform.parquet_path_for_spec`) and asserts:

- File exists and is non-empty.
- ``gameSeason`` column contains exactly the requested season.
- ``gameWeek`` column contains exactly the week set implied by the
  mode (``[N]`` for weekly, ``[1..N]`` for season_to_date, ``[N]``
  for postseason — the regular-vs-postseason distinction is in
  ``gameType`` which we also sample when present).

Anything off is reported as a :class:`VerificationIssue`. The CLI
wrapper (``fp-fetch verify``) prints per-spec results and exits
non-zero if any spec yielded an ``error``-severity issue.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from sportstradamus.fantasypoints.body_substitute import DEFAULT_MODE, Mode
from sportstradamus.fantasypoints.catalog import EndpointSpec
from sportstradamus.fantasypoints.transform import parquet_path_for_spec

# Columns every FP /values response carries on each row. The verify
# checks key off these — if they're absent we report it as an error
# because routing depends on them.
_SEASON_COL = "gameSeason"
_WEEK_COL = "gameWeek"

# Optional column FP includes on most tools — distinguishes REG vs
# POST games. Sampled when present so postseason-mode verification
# can catch a body that didn't actually filter to playoffs.
_GAME_TYPE_COL = "gameType"
_REGULAR_GAME_TYPES = frozenset({"REG", "Regular"})
_POST_GAME_TYPES = frozenset({"POST", "Postseason", "Playoffs"})

# Maximum number of column names to echo in a "missing column" error
# message — enough context to spot typos without flooding the terminal.
_MISSING_COL_PREVIEW = 8

Severity = Literal["error", "warn", "info"]


@dataclass(frozen=True)
class VerificationIssue:
    """One observation from spot-checking a downloaded parquet.

    Severity meanings:

    - ``error``: file or contents contradict the requested (season,
      week, mode) — almost certainly a download/catalog bug.
    - ``warn``: surprising but not unambiguously wrong (e.g. zero
      rows, missing optional column).
    - ``info``: noted but expected (e.g. spec skipped because file
      doesn't exist and the user only asked for an existence check).
    """

    spec_name: str
    code: str
    severity: Severity
    message: str
    path: str


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
        season: NFL season year the parquet should contain.
        week: NFL week the parquet should contain (mode-dependent meaning).
        mode: ``weekly`` / ``season_to_date`` / ``postseason``.

    Returns:
        A list of :class:`VerificationIssue`, possibly empty. An empty
        list means every check passed.
    """
    try:
        path = parquet_path_for_spec(spec, season=season, week=week, mode=mode)
    except ValueError as exc:
        return [
            VerificationIssue(
                spec_name=spec.name,
                code="routing_failed",
                severity="error",
                message=str(exc),
                path="<unrouted>",
            )
        ]
    if not path.is_file():
        return [
            VerificationIssue(
                spec_name=spec.name,
                code="file_missing",
                severity="error",
                message=(
                    f"Expected parquet not found. Re-run fp-fetch run "
                    f"--week {week} --season {season} --mode {mode} --only {spec.name}"
                ),
                path=str(path),
            )
        ]
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

    Specs with no issues map to an empty list, so callers can both
    list pass/fail counts and pull the issue detail for failures from
    the same dict.
    """
    return {spec.name: verify_spec(spec, season=season, week=week, mode=mode) for spec in specs}


def _check_dataframe(
    df: pd.DataFrame,
    *,
    spec: EndpointSpec,
    season: int,
    week: int,
    mode: Mode,
    path: Path,
) -> list[VerificationIssue]:
    """Run column / value / mode checks against one loaded parquet."""
    issues: list[VerificationIssue] = []
    path_str = str(path)
    if df.empty:
        issues.append(
            VerificationIssue(
                spec_name=spec.name,
                code="file_empty",
                severity="warn",
                message="Parquet has zero rows. FP may not publish this tool for "
                f"(season={season}, week={week}, mode={mode}).",
                path=path_str,
            )
        )
        return issues
    issues.extend(_check_season(df, spec=spec, season=season, path=path_str))
    issues.extend(_check_week(df, spec=spec, week=week, mode=mode, path=path_str))
    issues.extend(_check_game_type(df, spec=spec, mode=mode, path=path_str))
    return issues


def _check_season(
    df: pd.DataFrame, *, spec: EndpointSpec, season: int, path: str
) -> list[VerificationIssue]:
    if _SEASON_COL not in df.columns:
        return [
            VerificationIssue(
                spec_name=spec.name,
                code="missing_season_column",
                severity="error",
                message=f"Column {_SEASON_COL!r} missing — cannot verify season. "
                f"Columns present: {list(df.columns)[:_MISSING_COL_PREVIEW]}...",
                path=path,
            )
        ]
    actual_seasons = set(df[_SEASON_COL].dropna().unique().tolist())
    extra = actual_seasons - {season}
    if extra:
        return [
            VerificationIssue(
                spec_name=spec.name,
                code="season_mismatch",
                severity="error",
                message=f"Expected only season {season}, found seasons {sorted(actual_seasons)}.",
                path=path,
            )
        ]
    return []


def _check_week(
    df: pd.DataFrame,
    *,
    spec: EndpointSpec,
    week: int,
    mode: Mode,
    path: str,
) -> list[VerificationIssue]:
    if _WEEK_COL not in df.columns:
        return [
            VerificationIssue(
                spec_name=spec.name,
                code="missing_week_column",
                severity="error",
                message=f"Column {_WEEK_COL!r} missing — cannot verify week.",
                path=path,
            )
        ]
    actual_weeks = set(df[_WEEK_COL].dropna().unique().tolist())
    expected = _expected_weeks(week=week, mode=mode)
    extra = actual_weeks - expected
    if extra:
        return [
            VerificationIssue(
                spec_name=spec.name,
                code="week_mismatch",
                severity="error",
                message=f"Mode {mode!r} for week {week} expects weeks {sorted(expected)}, "
                f"but parquet has weeks {sorted(actual_weeks)} "
                f"(unexpected: {sorted(extra)}). This is the symptom of "
                "the pre-v2 'no week filter in body' bug — re-run "
                "`fp-fetch discover --replace` to refresh the catalog.",
                path=path,
            )
        ]
    return []


def _check_game_type(
    df: pd.DataFrame, *, spec: EndpointSpec, mode: Mode, path: str
) -> list[VerificationIssue]:
    if _GAME_TYPE_COL not in df.columns:
        return []  # Not every tool exposes it. Skip silently.
    actual_types = {str(t) for t in df[_GAME_TYPE_COL].dropna().unique().tolist()}
    if mode == "postseason":
        non_post = actual_types - _POST_GAME_TYPES
        if non_post:
            return [
                VerificationIssue(
                    spec_name=spec.name,
                    code="game_type_not_postseason",
                    severity="error",
                    message=f"Mode 'postseason' expected only playoff game types, "
                    f"found {sorted(actual_types)} (offending: {sorted(non_post)}).",
                    path=path,
                )
            ]
    else:
        non_reg = actual_types - _REGULAR_GAME_TYPES
        if non_reg:
            return [
                VerificationIssue(
                    spec_name=spec.name,
                    code="game_type_not_regular",
                    severity="error",
                    message=f"Mode {mode!r} expected only regular-season game types, "
                    f"found {sorted(actual_types)} (offending: {sorted(non_reg)}).",
                    path=path,
                )
            ]
    return []


def _expected_weeks(*, week: int, mode: Mode) -> set[int]:
    """Return the set of ``gameWeek`` values valid for one (week, mode).

    Mirrors :func:`body_substitute._override_weeks` so any divergence
    between the body builder and the verifier shows up as a test
    failure rather than silent acceptance of bad data.
    """
    if mode == "season_to_date":
        return set(range(1, week + 1))
    return {week}
