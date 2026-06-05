"""Shared weekly-snapshot parquet loader for the NFL FantasyPoints pipeline.

:mod:`sportstradamus.stats.nfl_fp_weekly` (player-grain) and
:mod:`sportstradamus.stats.nfl_fp_team_weekly` (team-grain) read the same
``{season}/week_{NN}/{kind}.parquet`` snapshot layout and differ only in
three values: the base directory under ``data/`` (``player_data/NFL`` vs
``team_data/NFL``), the logical-kind -> basename ``FILE_KINDS`` map, and the
error-message label. Each module owns one :class:`SnapshotStore` configured
with those three and re-exports its bound methods as the module-level loader
API, so call sites stay ``nfl_fp_weekly.load_window(...)`` unchanged.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from importlib import resources as pkg_resources
from importlib.resources.abc import Traversable
from pathlib import Path

import pandas as pd

from sportstradamus import data
from sportstradamus.spiderLogger import logger

_WEEK_DIR_RE = re.compile(r"^week_(\d{2})$")
_WEEK_DIR_FMT = "week_{week:02d}"


class SnapshotStore:
    """Loader for one grain of FP weekly snapshots (player or team).

    Parameterized by ``base_dir`` (the snapshot root under ``data/``, e.g.
    ``"player_data/NFL"``), the ``file_kinds`` logical-name -> basename map,
    and ``label`` (used in ``unknown <label> file_kind`` errors).
    """

    def __init__(self, base_dir: str, file_kinds: dict[str, str], label: str) -> None:
        self.base_dir = base_dir
        self.file_kinds = file_kinds
        self.label = label

    def snapshot_dir(self, season: int, snapshot_week: int) -> Traversable | None:
        """Return the ``week_{NN}/`` directory for ``(season, snapshot_week)``.

        ``None`` is the expected interim state for weeks not yet downloaded.
        """
        week_dir = _WEEK_DIR_FMT.format(week=snapshot_week)
        directory = pkg_resources.files(data) / f"{self.base_dir}/{season}/{week_dir}"
        return directory if os.path.exists(directory) else None

    def available_snapshots(self, season: int) -> list[int]:
        """Return the sorted snapshot weeks present on disk for ``season`` (empty if none)."""
        season_dir = pkg_resources.files(data) / f"{self.base_dir}/{season}"
        if not os.path.exists(season_dir):
            return []

        weeks: list[int] = []
        for entry in Path(str(season_dir)).iterdir():
            match = _WEEK_DIR_RE.match(entry.name)
            if match:
                weeks.append(int(match.group(1)))
        weeks.sort()
        return weeks

    def load_snapshot(
        self,
        season: int,
        snapshot_week: int,
        file_kind: str,
    ) -> pd.DataFrame | None:
        """Load one parquet from the ``(season, snapshot_week)`` directory.

        Returns the per-game rows for ``snapshot_week`` of ``season``, an
        empty DataFrame when the parquet exists but holds no rows (FP writes
        empty files during in-flight weeks), or ``None`` when the directory
        or file is absent or the parquet cannot be parsed. Use
        :meth:`load_through` / :meth:`load_window` to stitch weeks together.

        Raises:
            ValueError: If ``file_kind`` is not a key in ``file_kinds``.
        """
        if file_kind not in self.file_kinds:
            raise ValueError(
                f"unknown {self.label} file_kind: {file_kind!r}; valid kinds: {sorted(self.file_kinds)}"
            )

        directory = self.snapshot_dir(season, snapshot_week)
        if directory is None:
            return None

        path = directory / self.file_kinds[file_kind]
        if not os.path.exists(path):
            return None

        try:
            return pd.read_parquet(path)
        except Exception as exc:
            logger.warning(
                "failed to read %s for season=%s week=%s: %s",
                file_kind,
                season,
                snapshot_week,
                exc,
            )
            return None

    def load_window(
        self,
        season: int,
        start_week: int,
        end_week: int,
        file_kind: str,
    ) -> pd.DataFrame | None:
        """Concatenate snapshots over ``[start_week, end_week]`` inclusive.

        Stacks every snapshot whose week is in the closed interval; missing
        or empty snapshots are skipped, and the result is ``None`` only when
        no snapshot in the window produced any rows. The caller aggregates
        (for rate stats, ``sum(numerator) / sum(denominator)`` across the
        concatenated rows).

        Raises:
            ValueError: If ``file_kind`` is unknown or ``start_week > end_week``.
        """
        if file_kind not in self.file_kinds:
            raise ValueError(
                f"unknown {self.label} file_kind: {file_kind!r}; valid kinds: {sorted(self.file_kinds)}"
            )
        if start_week > end_week:
            raise ValueError(f"start_week={start_week} must be <= end_week={end_week}")

        frames: list[pd.DataFrame] = []
        for week in self.available_snapshots(season):
            if not start_week <= week <= end_week:
                continue
            df = self.load_snapshot(season, week, file_kind)
            if df is not None and not df.empty:
                frames.append(df)

        if not frames:
            return None
        return pd.concat(frames, ignore_index=True)

    def load_window_or_empty(
        self,
        season: int,
        start_week: int,
        end_week: int,
        file_kind: str,
    ) -> pd.DataFrame:
        """:meth:`load_window` with loader errors swallowed and ``None`` normalized to empty.

        The aggregate recipes want a guaranteed DataFrame: an inverted range or
        any loader exception yields an empty frame (logged, tagged with
        :attr:`label`) rather than raising, and a no-rows window becomes an
        empty frame instead of ``None``.
        """
        if start_week > end_week:
            return pd.DataFrame()
        try:
            df = self.load_window(season, start_week, end_week, file_kind)
        except Exception as exc:
            logger.warning(
                "%s load_window(%s, %s, %s, %s) failed: %s",
                self.label,
                season,
                start_week,
                end_week,
                file_kind,
                exc,
            )
            return pd.DataFrame()
        return df if df is not None else pd.DataFrame()

    def load_through(
        self,
        season: int,
        last_week: int,
        file_kind: str,
    ) -> pd.DataFrame | None:
        """Concatenate snapshots ``1..last_week`` inclusive (season-to-date).

        Anchored-at-week-1 form of :meth:`load_window`. The leakage-safe
        time-capsule pattern for a game in week ``W`` is
        ``load_through(season, W - 1, kind)`` — the target game is excluded.

        Raises:
            ValueError: If ``file_kind`` is unknown or ``last_week < 1``.
        """
        if last_week < 1:
            raise ValueError(f"last_week={last_week} must be >= 1")
        return self.load_window(season, 1, last_week, file_kind)

    def load_all_snapshots(
        self,
        season: int,
        file_kind: str,
    ) -> dict[int, pd.DataFrame]:
        """Return ``{snapshot_week: DataFrame}`` for every parsing snapshot of ``file_kind``.

        Un-concatenated, for diagnostics and snapshot-drift analysis;
        production paths use :meth:`load_window` / :meth:`load_through`.
        """
        out: dict[int, pd.DataFrame] = {}
        for week in self.available_snapshots(season):
            df = self.load_snapshot(season, week, file_kind)
            if df is not None and not df.empty:
                out[week] = df
        return out

    def snapshot_inventory(self, seasons: Iterable[int] | None = None) -> pd.DataFrame:
        """Inventory which snapshots+files exist on disk.

        Args:
            seasons: Iterable of season years, or ``None`` to scan every
                numeric subdirectory under ``base_dir``.

        Returns:
            DataFrame with columns ``["season", "snapshot_week",
            *file_kinds]``; each kind column is ``True`` when that parquet
            exists in the snapshot directory. Empty (correct columns, zero
            rows) when no snapshots are found. Used by operator tools to
            confirm a download is complete before swapping production over.
        """
        if seasons is None:
            root = pkg_resources.files(data) / self.base_dir
            if not os.path.exists(root):
                return pd.DataFrame(columns=["season", "snapshot_week", *self.file_kinds])
            season_entries = sorted(
                int(entry.name) for entry in Path(str(root)).iterdir() if entry.name.isdigit()
            )
        else:
            season_entries = sorted(seasons)

        rows: list[dict[str, object]] = []
        for season in season_entries:
            for week in self.available_snapshots(season):
                directory = self.snapshot_dir(season, week)
                if directory is None:
                    continue
                row: dict[str, object] = {"season": season, "snapshot_week": week}
                for kind, basename in self.file_kinds.items():
                    row[kind] = os.path.exists(directory / basename)
                rows.append(row)

        return pd.DataFrame(rows, columns=["season", "snapshot_week", *self.file_kinds])
