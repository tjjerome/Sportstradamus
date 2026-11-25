"""The money tables, offline: what each play's legs and the engine's own entries returned.

Scores every registered play over a window of game dates (``plays.play_record``) from a
history, this box's or a frozen copy, and the engine's graded entries over the same window
(``plays.entry_record``) from the parlay history's day files. Prints both tables and writes
them to a sandbox directory: ``data/runtime`` belongs to the nightly job. Hand-run to read a
change to a play, or a model version, in money before it ships
(docs/handoffs/profit-engine.md).
"""

from datetime import datetime
from pathlib import Path

import click
import numpy as np
import pandas as pd

from sportstradamus import plays, realized
from sportstradamus.helpers.io import HISTORY_PATH, PARLAY_HIST_DIR, _atomic_write_parquet

# Fixed, so a replay of the same history prints the same bounds.
REPLAY_SEED = 0
# The parlay history's columns the entry table reads. Its nested ``legs`` column costs
# gigabytes to load, so the read is projected to these.
_ENTRY_COLS = [
    "Date",
    "League",
    "Platform",
    "Bet Size",
    "Boost",
    "Model EV",
    "Legs Resolved",
    "Misses",
]
_WINDOW_COLS = ["first_date", "last_date"]


@click.command()
@click.option(
    "--history",
    "history_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path(str(HISTORY_PATH)),
    help="The history to score, e.g. a frozen copy (default: this box's "
    "data/runtime/history.parquet).",
)
@click.option(
    "--entries",
    "entries_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path(str(PARLAY_HIST_DIR)),
    help="Directory of the parlay history's day files (default: this box's "
    "data/runtime/parlay_hist).",
)
@click.option(
    "--since",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=realized.COUNT_ONCE_FROM.strftime("%Y-%m-%d"),
    show_default=True,
    help="First game date scored. A prop counts once from COUNT_ONCE_FROM, and history "
    "holds both sides' payouts and the opponent that names a leg's game from 2026-10-05.",
)
@click.option(
    "--until",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Last game date scored (default: every settled leg).",
)
@click.option(
    "--out",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("/tmp/play_replay"),
    show_default=True,
    help="Directory play_record.parquet and entry_record.parquet land in.",
)
def main(
    history_path: Path, entries_dir: Path, since: datetime, until: datetime | None, out: Path
) -> None:
    """Score each play's legs, and the engine's own entries, in money over a window of game dates."""
    last = until or pd.Timestamp.max
    rng = np.random.default_rng(REPLAY_SEED)
    # Count-once reads every row of a prop, so the window is cut after the offers are settled.
    offers = realized.settled_offers(pd.read_parquet(history_path))
    offers = offers[pd.to_datetime(offers["Date"]).between(since, last)]
    play_table = plays.play_record(plays.posted_sides(offers), rng)
    entries = pd.concat(
        [pd.read_parquet(day, columns=_ENTRY_COLS) for day in sorted(entries_dir.glob("*.parquet"))]
    )
    entries = entries[pd.to_datetime(entries["Date"]).between(since, last)]
    entry_table = plays.entry_record(entries, rng)

    out.mkdir(parents=True, exist_ok=True)
    _atomic_write_parquet(play_table, out / "play_record.parquet")
    _atomic_write_parquet(entry_table, out / "entry_record.parquet")
    power_cols = [col for col in play_table.columns if col.startswith("power")]
    tables = {
        "Legs of each play": play_table.drop(columns=[*_WINDOW_COLS, *power_cols]),
        "Power entries those legs could build, one leg a game": play_table[
            ["play", "league", "kind", *power_cols]
        ],
        "The engine's own entries": entry_table.drop(columns=_WINDOW_COLS),
    }
    for title, table in tables.items():
        click.echo(f"\n{title}")
        click.echo(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    click.echo(f"\nwrote play_record.parquet and entry_record.parquet to {out}")
