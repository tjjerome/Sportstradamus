"""The money tables, offline: what each play's legs and the engine's own entries returned.

Scores every registered play over a window of game dates (``plays.play_record``) from a
history, this box's or a frozen copy, and the engine's graded entries over the same window
(``plays.entry_record``) from the parlay history's day files. Prints both tables and writes
them to a sandbox directory: ``data/runtime`` belongs to the nightly job. Hand-run to read a
change to a play, or a model version, in money before it ships
(docs/handoffs/profit-engine.md). ``--walk-forward`` prices each game date's EV from fits on
earlier dates only and adds the fit's own checks, the reads behind its acceptance tests and
kill lines (docs/archive/researcher_ev_fit.md, 3).
"""

from datetime import datetime
from pathlib import Path

import click
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sportstradamus import live_fit, plays, realized
from sportstradamus.helpers.io import HISTORY_PATH, PARLAY_HIST_DIR, _atomic_write_parquet
from sportstradamus.training.scorecard import _CI_HIGH_PCT, _CI_LOW_PCT, _GATE1_N_BOOT

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
# The research brief reads a log-loss gain in nats a leg times this.
_GAIN_SCALE = 10_000


def _gap_to_rule(
    sides: pd.DataFrame, play: str, rng: np.random.Generator
) -> tuple[float, float, float]:
    """``play`` less ``model_edge`` in return a leg, with 95 % bounds.

    Each resample draws whole game dates once and scores both plays on them, so what the two
    share on a date cancels in the bound (the brief's AT5 and K4).
    """
    picked = pd.DataFrame(
        {name: sides["Unit"].where(plays.PLAYS[name](sides)) for name in (play, "model_edge")}
    )
    # Per game date: the play's summed return and legs, then the rule's.
    by_date = picked.groupby(sides["Date"]).agg(["sum", "count"]).to_numpy()

    def gap(totals: np.ndarray) -> np.ndarray:
        play_units, play_legs, rule_units, rule_legs = totals.T
        return play_units / play_legs - rule_units / rule_legs

    draws = rng.integers(0, len(by_date), (_GATE1_N_BOOT, len(by_date)))
    lo, hi = np.percentile(gap(by_date[draws].sum(axis=1)), [_CI_LOW_PCT, _CI_HIGH_PCT])
    return float(gap(by_date.sum(axis=0))), lo, hi


def _fit_check(
    walked: pd.DataFrame, sides: pd.DataFrame, rng: np.random.Generator, out: Path
) -> None:
    """The fit check: the walked fit's gain and band tables, the legs its lower bound clears
    and its rank plays' gaps.
    """
    gain = live_fit.fit_gain(walked, rng)
    bands = live_fit.ev_bands(walked, rng)
    _atomic_write_parquet(gain, out / "fit_gain.parquet")
    _atomic_write_parquet(bands, out / "ev_bands.parquet")
    cleared = (walked["Fit Prob Low"] * walked["Payout"] >= 1).sum()
    gaps = {play: _gap_to_rule(sides, play, rng) for play in ("ev_rank", "ev_pick")}
    dates = walked["Date"]
    click.echo(
        f"\nFit check: {dates.nunique()} game dates, {dates.min()} to {dates.max()}, each "
        "priced from fits on earlier dates only"
    )
    click.echo(
        f"\nThe fit's log-loss gain over the app's price, x {_GAIN_SCALE:,} (above 0: better)"
    )
    click.echo(gain.to_string(index=False, float_format=lambda x: f"{x * _GAIN_SCALE:+.1f}"))
    click.echo("\nWhat the model's side paid on a unit staked, by band of EV")
    click.echo(bands.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    click.echo(
        f"\nLegs the lower bound clears (Fit Prob Low x payout at least 1): {cleared:,} of "
        f"{len(walked):,}"
    )
    for play, (gap, gap_lo, gap_hi) in gaps.items():
        click.echo(
            f"{play} minus model_edge, return a leg: {gap:+.3f} [{gap_lo:+.3f}, {gap_hi:+.3f}]"
        )
    click.echo(f"\nwrote fit_gain.parquet and ev_bands.parquet to {out}")


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
@click.option(
    "--walk-forward",
    is_flag=True,
    help="Price each game date's EV from fits on earlier dates only, score the plays on the "
    "dates so priced, and add the fit's own checks (fit_gain.parquet, ev_bands.parquet). "
    "Needs a history that carries the book stamp.",
)
def main(
    history_path: Path,
    entries_dir: Path,
    since: datetime,
    until: datetime | None,
    out: Path,
    walk_forward: bool,
) -> None:
    """Score each play's legs, and the engine's own entries, in money over a window of game dates."""
    last = until or pd.Timestamp.max
    # Asked of the file's schema: the history itself takes a minute and gigabytes to load.
    if walk_forward and "Book Class" not in pq.read_schema(history_path).names:
        raise click.UsageError(
            "--walk-forward needs the book stamp, and this history has no Book Class column."
        )
    # Count-once reads every row of a prop, so the window is cut after the offers are settled.
    offers = realized.settled_offers(pd.read_parquet(history_path))
    offers = offers[pd.to_datetime(offers["Date"]).between(since, last)]
    if walk_forward:
        # A scoring run's prices go first: the walk joins its own under the same names, and
        # its picks replace the run's.
        offers = live_fit.walk_forward(offers.drop(columns=live_fit.SCORE_COLS, errors="ignore"))
        offers["Pick"] = live_fit.picks(offers)
    sides = plays.posted_sides(offers)
    # Each table draws from its own generator, so a play added to the registry moves no bound
    # of the entry table or the fit check.
    play_table = plays.play_record(sides, np.random.default_rng(plays.RECORD_SEED))
    entries = pd.concat(
        [pd.read_parquet(day, columns=_ENTRY_COLS) for day in sorted(entries_dir.glob("*.parquet"))]
    )
    entries = entries[pd.to_datetime(entries["Date"]).between(since, last)]
    entry_table = plays.entry_record(entries, np.random.default_rng(plays.RECORD_SEED))

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
    if walk_forward:
        _fit_check(offers, sides, np.random.default_rng(plays.RECORD_SEED), out)
