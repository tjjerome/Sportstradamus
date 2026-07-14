#!/usr/bin/env python3
"""One-shot era backfill for pre-stamp prediction history.

Rows written before the train-time ``model_version`` stamp (WS-1 Stage 0)
carry no model identity. This script reconstructs an approximate identity —
an *era key* — for those rows so era-aware live reads have something to join
on until stamped rows age in and supersede it.

Per ``(League, Market)`` cell, rows sharing the calibration fingerprint
``(Dist, CV, Temperature, Disp Cal, Step)`` form one era, labelled ``era.{n}``
by first-seen date (the all-NaN pre-schema block sorts first as ``era.0``).
On real history these fingerprints occupy non-overlapping time windows — a
retrain shifts the recency-CV and dispersion knobs, so a distinct tuple is a
clean change-point. Each era is annotated with the most recent
``stat_meta.json`` git flip date at or before its first row, giving a
human-readable pointer to the config change that most likely opened it.

The output ``data/runtime/history_eras.parquet`` is keyed by
``(League, Market, Date, Player)`` so a consumer joins it back onto history
one-to-one. Read-only w.r.t. everything except that output parquet.

Usage
-----
    poetry run backfill-history-eras
    poetry run backfill-history-eras --output /tmp/eras.parquet
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm

from sportstradamus.helpers.io import (
    HISTORY_PATH,
    _atomic_write_parquet,
    read_history,
)

# The calibration knobs that define a model era. A change in any of them across
# consecutive predictions marks a new era; rounding absorbs float-repr jitter.
_FINGERPRINT_COLS = ["Dist", "CV", "Temperature", "Disp Cal", "Step"]
_FINGERPRINT_ROUND = 9

# Join key back onto history — one era row per (cell, date, player) offer group.
_ERA_KEY_COLS = ["League", "Market", "Date", "Player"]

_STAT_META_PATH = "src/sportstradamus/data/config/stat_meta.json"

_OUTPUT_PATH = Path(HISTORY_PATH).with_name("history_eras.parquet")


def _stat_meta_flip_dates() -> list[str]:
    """``stat_meta.json`` commit dates (ISO ``YYYY-MM-DD``), newest-first.

    Empty when git or the file is unavailable — era labelling degrades to no
    flip annotation rather than failing the backfill.
    """
    result = subprocess.run(
        ["git", "log", "--date=short", "--format=%cd", "--", _STAT_META_PATH],
        capture_output=True,
        text=True,
        check=False,
    )
    return [line for line in result.stdout.splitlines() if line]


def _nearest_flip(first_date: str, flip_dates: list[str]) -> str | None:
    """Most recent stat_meta flip date at or before ``first_date`` (ISO strings)."""
    candidates = [d for d in flip_dates if d <= first_date]
    return max(candidates) if candidates else None


def _cell_eras(cell: pd.DataFrame, flip_dates: list[str]) -> pd.DataFrame:
    """Assign an era key to every row of one ``(League, Market)`` cell.

    Each distinct calibration fingerprint is one era, numbered by first-seen
    date. Returns one row per ``(League, Market, Date, Player)`` join key
    (deduped — offer rows that share a key were scored by the same model, so
    their era is identical) with the era label, its fingerprint, window, and
    nearest config-flip annotation.
    """
    cell = cell.drop_duplicates(_ERA_KEY_COLS)
    fp = cell[_FINGERPRINT_COLS].round(_FINGERPRINT_ROUND)
    # Sentinel makes NaN==NaN so the all-NaN pre-schema block collapses to one era.
    key = fp.fillna("∅").astype(str).agg("|".join, axis=1)

    dates = pd.to_datetime(cell["Date"])
    first_seen = dates.groupby(key).transform("min")
    # Order eras by first-seen date, then by fingerprint so two distinct tuples
    # opening on the same boundary day stay separate eras.
    order = first_seen.dt.strftime("%Y-%m-%d") + "|" + key
    era_rank = order.rank(method="dense").astype(int) - 1

    out = cell[_ERA_KEY_COLS].copy()
    out["era_id"] = [f"era.{i}" for i in era_rank]
    for col in _FINGERPRINT_COLS:
        out[col] = cell[col].to_numpy()
    out["era_first_date"] = first_seen.dt.strftime("%Y-%m-%d").to_numpy()
    out["era_last_date"] = dates.groupby(key).transform("max").dt.strftime("%Y-%m-%d").to_numpy()
    out["label_flip_date"] = [_nearest_flip(d, flip_dates) for d in out["era_first_date"]]
    return out


@click.command()
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Override the output parquet path (defaults to data/runtime/history_eras.parquet).",
)
def backfill_history_eras(output: Path | None) -> None:
    """Segment pre-stamp history into model eras and write the join parquet."""
    out_path = output if output is not None else _OUTPUT_PATH
    history = read_history()
    if history.empty:
        click.echo("history.parquet is empty; nothing to backfill.")
        return

    flip_dates = _stat_meta_flip_dates()
    cells = list(history.groupby(["League", "Market"], sort=False))
    era_frames = [
        _cell_eras(cell, flip_dates)
        for _, cell in tqdm(cells, desc="Segmenting history eras", unit="cell")
    ]
    eras = pd.concat(era_frames, ignore_index=True)
    _atomic_write_parquet(eras, out_path)

    n_eras = eras.groupby(["League", "Market"])["era_id"].nunique().sum()
    click.echo(
        f"Wrote {len(eras)} era rows across {len(cells)} cells "
        f"({n_eras} distinct eras) to {out_path}."
    )


if __name__ == "__main__":
    backfill_history_eras()
