"""Date-keyed snapshot loader for the collector framework (CTG / savant).

Read-side analog of the week-keyed
:class:`sportstradamus.stats.nfl_fp_snapshots.SnapshotStore`. The Fantasy Points
NFL snapshots are keyed by integer game-week (``week_NN``) and the NFL consumer
stitches weeks ``1..N`` into a season-to-date view. Cleaning the Glass (NBA) and
Baseball Savant (MLB) instead serve *already-cumulative* season-to-date tables
captured on a run date, so the natural key is the capture date and the only query
is "the most recent snapshot strictly before this game date". That strict ``<`` is
the leakage guard: a snapshot captured the morning of a game may already fold in
that game upstream, so it must never feed a prediction for that same game — the
same rule the NFL loader enforces by asking for ``week - 1``.

Different period type (date vs. week ordinal) and different query shape (single
as-of lookup vs. ``1..N`` windowing) from the week-keyed store, so this is a
separate loader, not a bimodal generalization of it.

Layout on disk::

    data/{base_dir}/{season}/{YYYY-MM-DD}/{tool}.parquet

where ``base_dir`` is e.g. ``player_data/NBA/cleaningtheglass`` or
``team_data/MLB/baseballsavant``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Collection
from datetime import date, datetime
from importlib import resources as pkg_resources
from pathlib import Path

import pandas as pd

from sportstradamus import data
from sportstradamus.helpers.text import remove_accents
from sportstradamus.spiderLogger import logger

_DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _as_date(value: date | datetime) -> date:
    return value.date() if isinstance(value, datetime) else value


class DatedSnapshotStore:
    """Loader for one grain of date-captured collector snapshots.

    Parameterized by ``base_dir`` (the snapshot root under ``data/``, e.g.
    ``"player_data/NBA/cleaningtheglass"``) and ``label`` (used in log lines).
    """

    def __init__(self, base_dir: str, label: str) -> None:
        self.base_dir = base_dir
        self.label = label

    def snapshot_dir(self, season: int, snap_date: date) -> Path | None:
        """Return the ``{YYYY-MM-DD}/`` directory for ``(season, snap_date)``, or ``None``."""
        directory = pkg_resources.files(data) / f"{self.base_dir}/{season}/{snap_date.isoformat()}"
        return Path(str(directory)) if os.path.exists(directory) else None

    def available_dates(self, season: int) -> list[date]:
        """Return the sorted capture dates present on disk for ``season`` (empty if none)."""
        season_dir = pkg_resources.files(data) / f"{self.base_dir}/{season}"
        if not os.path.exists(season_dir):
            return []
        dates: list[date] = []
        for entry in Path(str(season_dir)).iterdir():
            if _DATE_DIR_RE.match(entry.name):
                dates.append(date.fromisoformat(entry.name))
        dates.sort()
        return dates

    def available_tools(self, season: int, snap_date: date) -> list[str]:
        """Return the sorted tool stems (parquet basenames) in the ``(season, snap_date)`` snapshot."""
        directory = self.snapshot_dir(season, snap_date)
        if directory is None:
            return []
        return sorted(p.stem for p in directory.glob("*.parquet"))

    def latest_before(self, season: int, game_date: date | datetime) -> date | None:
        """Return the most recent capture date strictly before ``game_date``, or ``None``.

        The strict ``<`` is the leakage guard — a same-day snapshot may already
        contain the target game.
        """
        cutoff = _as_date(game_date)
        prior = [d for d in self.available_dates(season) if d < cutoff]
        return prior[-1] if prior else None

    def load_snapshot(self, season: int, snap_date: date, tool: str) -> pd.DataFrame | None:
        """Load ``{tool}.parquet`` from the ``(season, snap_date)`` directory.

        Returns the frame, or ``None`` when the directory / file is absent or the
        parquet cannot be parsed.
        """
        directory = self.snapshot_dir(season, snap_date)
        if directory is None:
            return None
        path = directory / f"{tool}.parquet"
        if not os.path.exists(path):
            return None
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            logger.warning(
                "failed to read %s %s for season=%s date=%s: %s",
                self.label,
                tool,
                season,
                snap_date.isoformat(),
                exc,
            )
            return None

    def load_asof(
        self, season: int, game_date: date | datetime, tool: str
    ) -> pd.DataFrame | None:
        """Load ``tool`` from the latest snapshot strictly before ``game_date``.

        ``None`` when no snapshot predates the game — the caller degrades to
        gamelog-only features, so an absent backfill never leaks or crashes.
        """
        snap_date = self.latest_before(season, game_date)
        if snap_date is None:
            return None
        return self.load_snapshot(season, snap_date, tool)


def _asof_snapshot_frames(
    store: DatedSnapshotStore, season: int, snap_date: date, key_col: str
) -> list[pd.DataFrame]:
    """Load every tool in the ``(season, snap_date)`` snapshot that carries ``key_col``."""
    frames = []
    for tool in store.available_tools(season, snap_date):
        frame = store.load_snapshot(season, snap_date, tool)
        if frame is not None and not frame.empty and key_col in frame.columns:
            frames.append(frame)
    return frames


def _merge_frames_on_key(frames: list[pd.DataFrame], key_col: str) -> pd.DataFrame:
    """Outer-merge tool frames horizontally on ``key_col``, dropping later duplicate columns."""
    merged = frames[0]
    for frame in frames[1:]:
        dup = [c for c in frame.columns if c in merged.columns and c != key_col]
        merged = merged.merge(frame.drop(columns=dup), on=key_col, how="outer")
    return merged


def load_asof_features(
    store: DatedSnapshotStore,
    season: int,
    game_date: date | datetime,
    key_col: str,
    meta_cols: Collection[str],
    *,
    key_transform: Callable[[str], str] = remove_accents,
) -> pd.DataFrame | None:
    """Merge every tool in the latest pre-``game_date`` snapshot into one ``{col}_asof`` frame.

    Every tool is read from a single capture date — the most recent strictly
    before ``game_date`` (the leakage guard) — so all columns share one
    as-of view. The tools are horizontally merged on ``key_col``, indexed by
    ``key_transform(key_col)`` (accent-stripped player name for the player
    grain; a team-abbreviation key for the team grain), traded-player deduped
    (the row with the most non-null cells wins), and each surviving non-meta
    column renamed to raw ``{col}_asof``. The base-class profile join then
    prefixes ``Player `` (player grain) or consumes the frame directly (team
    grain), mirroring the NFL FP as-of projection.

    Returns ``None`` when ``key_col`` is unset (join schema not yet pinned for
    the source), no snapshot predates the game, or no tool carries ``key_col``
    — the caller then degrades to gamelog-only features.
    """
    if not key_col:
        return None
    snap_date = store.latest_before(season, game_date)
    if snap_date is None:
        return None
    frames = _asof_snapshot_frames(store, season, snap_date, key_col)
    if not frames:
        return None
    merged = _merge_frames_on_key(frames, key_col)
    merged = merged.loc[merged[key_col].notna()]
    feat_cols = [c for c in merged.columns if c not in set(meta_cols) | {key_col}]
    if merged.empty or not feat_cols:
        return None
    out = merged[feat_cols].copy()
    out.index = merged[key_col].astype(str).apply(key_transform)
    if out.index.has_duplicates:
        ranked = out.assign(_rank=out.notna().sum(axis=1)).sort_values("_rank", ascending=False)
        out = ranked[~ranked.index.duplicated(keep="first")].drop(columns="_rank")
    out.columns = [f"{c}_asof" for c in out.columns]
    return out
