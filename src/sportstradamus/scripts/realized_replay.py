"""The offline realized-money replay: the live recommendation rule on held-out rows at market lines.

``realized.recommended`` (``Win Prob x Payout - 1 >= 0.05``) runs over each cell's test-set rows as
the Receipts ledger runs it over ``history.parquet``, at Underdog's even-pick payout, on the rows that
hold a market line: pick'em rows (``QuoteAuthenticity == "pickem"``) and sportsbook rows priced within
``NEAR_EVEN_BAND`` of even. Synthetic ``max(Avg10, 0.5)`` lines, EV-derived lines and lopsided
sportsbook quotes stay out: the live board posts market lines, and the gates' confident bins are
mostly the pipeline's own lines (docs/ship_gate.md, line types). Every scope reports the recommended
cohort's read, hit and gap (``read - hit``, positive = the read runs high, the sign of
``live_gap_per_cell.parquet``) with date-block bootstrap bounds, beside the same numbers from the
live record; ``gap_delta`` is how far the replay reads from live. The tolerance the owner asked for
is read on the ``l2`` rows of leagues whose live sample holds ``LIVE_MIN_LEGS`` recommended legs.
The league rows also carry the L2 gap-form column (``implied_live_hit_lb``, ``l2_flag``).

A report, never a gate (owner 2026-10-30, docs/handoffs/live-gap-2.md §5 item 5).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import click
import numpy as np
import pandas as pd

from sportstradamus import realized
from sportstradamus.helpers.archive import archive_market
from sportstradamus.helpers.distributions import UNDERDOG_BOOST_BASELINE
from sportstradamus.helpers.io import (
    REALIZED_REPLAY_PATH,
    _atomic_write_csv,
    _atomic_write_parquet,
    read_history,
)
from sportstradamus.helpers.training_quotes import AUTHENTIC, PICKEM
from sportstradamus.prediction.offer_records import _MAX_CONFIDENCE, UNQUOTED_BOOK_DISAGREEMENT_MAX
from sportstradamus.scripts.tail_information import key_rng
from sportstradamus.scripts.tail_scorecard import TEST_SETS_DIR
from sportstradamus.training.labels import over_label
from sportstradamus.training.scorecard import _bootstrap_mean_ci_clustered

# A sportsbook row priced within this of even is a market-centre line (the L2 frozen definition).
NEAR_EVEN_BAND = 0.06
# The live legs the replay is read against: Underdog's even pick pays 1.83; Sleeper's 1.78 sits outside.
LIVE_PAYOUT_BAND = (1.80, 1.90)
# L2 floors: a league row carries a flag only over this many recommended rows and dates.
L2_MIN_ROWS = 300
L2_MIN_DATES = 20
# A league's live sample must hold this many recommended even-pick legs before its gap_delta
# counts toward the tolerance.
LIVE_MIN_LEGS = 100
# Pick'em rows count toward L2 only in cells whose test rows go Over within this band (frozen L2).
L2_OVER_RATE_BAND = (0.40, 0.60)
# The one-sided 90 % bounds the brief pre-registered, as the bootstrap's (low, high) percentiles.
_PCT_90 = (10.0, 90.0)
# One generator per scope key (key_rng), so a --league run reproduces the full run's bounds.
REPLAY_SEED = 1729
BREAK_EVEN = 1 / UNDERDOG_BOOST_BASELINE
_TEST_COLUMNS = ["Player", "Date", "Result", "Line", "P", "Odds", "QuoteAuthenticity", "Book_EV"]
_COHORT = ("n", "dates", "read", "hit", "hit_lb", "gap", "gap_lo", "gap_hi", "roi")
REPLAY_COLUMNS = (
    "scope",
    "league",
    "market",
    *_COHORT,
    *(f"live_{col}" for col in _COHORT),
    "gap_delta",
    "agrees",
    "implied_live_hit_lb",
    "l2_flag",
    "offline_before",
    "live_since",
    "live_until",
    "computed_at",
)


def market_line_rows(test: pd.DataFrame) -> pd.DataFrame:
    """The test rows that hold a market line, tagged ``line_type`` pickem / near_even."""
    authenticity = test["QuoteAuthenticity"]
    near_even = authenticity.eq(AUTHENTIC) & ((1 - test["Odds"]) - 0.5).abs().le(NEAR_EVEN_BAND)
    line_type = np.select([authenticity.eq(PICKEM), near_even], ["pickem", "near_even"], None)
    rows = test.assign(line_type=line_type)
    return rows[rows["line_type"].notna()]


def replay_offers(test: pd.DataFrame, league: str, market: str, version: str) -> pd.DataFrame:
    """One live-shaped offer per market-line test row, graded by ``realized.settled_offers``.

    ``Boost`` 1.0 on Underdog is the 1.83 even pick, so ``recommended`` runs unchanged. ``Result``
    is the test row's outcome against its line (a push drops out downstream, as live). The side is
    the fused ``P``'s, its ``Win Prob`` capped as ``finalize_records`` caps it. A pick'em row whose
    capped ``Win Prob`` sits more than ``UNQUOTED_BOOK_DISAGREEMENT_MAX`` from its ``Market Prob``
    drops as the board drops it: the one ``finalize_records`` gate binding at one line per player
    (the three-per-player trim and the boost cap cannot).
    """
    rows = market_line_rows(test)
    over = rows["P"].to_numpy() >= 0.5
    actual, line = rows["Result"].to_numpy(float), rows["Line"].to_numpy(float)
    book_over = 1 - rows["Odds"].to_numpy(float)
    offers = pd.DataFrame(
        {
            "League": league,
            "Market": market,
            "Player": rows["Player"].to_numpy(),
            "Date": rows["Date"].to_numpy(),
            "Line": line,
            "Bet": np.where(over, "Over", "Under"),
            "Win Prob": np.minimum(np.where(over, rows["P"], 1 - rows["P"]), _MAX_CONFIDENCE),
            "Market Prob": np.where(over, book_over, 1 - book_over),
            "Boost": 1.0,
            "Platform": "Underdog",
            "Alt Line": False,
            "Model Version": version,
            "Market Projection": rows["Book_EV"]
            .where(rows["QuoteAuthenticity"].eq(AUTHENTIC))
            .to_numpy(),
            "Result": np.select([actual > line, actual < line], ["Over", "Under"], "Push"),
            "line_type": rows["line_type"].to_numpy(),
        }
    )
    phantom = offers["line_type"].eq("pickem") & (
        (offers["Win Prob"] - offers["Market Prob"]).abs() > UNQUOTED_BOOK_DISAGREEMENT_MAX
    )
    return realized.settled_offers(offers[~phantom])


def cohort(offers: pd.DataFrame, rng: np.random.Generator) -> dict:
    """The recommended rows' read, hit and gap with date-block bounds; NaN where nothing qualifies."""
    rec = offers[offers["Recommended"]]
    dates = rec["Date"].to_numpy()
    gap, gap_lo, gap_hi = _bootstrap_mean_ci_clustered(
        (rec["Win Prob"] - rec["Hit"]).to_numpy(float), dates, rng, percentiles=_PCT_90
    )
    hit, hit_lb, _ = _bootstrap_mean_ci_clustered(
        rec["Hit"].to_numpy(float), dates, rng, percentiles=_PCT_90
    )
    return {
        "n": len(rec),
        "dates": int(rec["Date"].nunique()),
        "read": float(rec["Win Prob"].mean()),
        "hit": hit,
        "hit_lb": hit_lb,
        "gap": gap,
        "gap_lo": gap_lo,
        "gap_hi": gap_hi,
        "roi": float(rec["Unit"].mean()),
    }


def live_recommended(
    history: pd.DataFrame, since: pd.Timestamp, until: pd.Timestamp | None
) -> pd.DataFrame:
    """The live record's recommended even-pick legs in the window, with the archive cell key."""
    offers = realized.settled_offers(history)
    dates = pd.to_datetime(offers["Date"])
    keep = offers["Recommended"] & offers["Payout"].between(*LIVE_PAYOUT_BAND) & (dates >= since)
    if until is not None:
        keep &= dates < until
    offers = offers[keep]
    return offers.assign(
        cell=[
            archive_market(lg, mk)
            for lg, mk in zip(offers["League"], offers["Market"], strict=True)
        ]
    )


def load_offline(
    test_sets_dir: Path, leagues: tuple[str, ...], before: pd.Timestamp
) -> tuple[pd.DataFrame, list[str]]:
    """Every cell's market-line offers dated before ``before``, plus the dumps that could not be read."""
    frames, skipped = [], []
    for path in sorted(test_sets_dir.glob("*.csv")):
        league, slug = path.stem.split("_", 1)
        if leagues and league not in leagues:
            continue
        columns = set(pd.read_csv(path, nrows=0).columns)
        if missing := sorted(set(_TEST_COLUMNS) - columns):
            skipped.append(f"{path.stem}: no {', '.join(missing)}")
            continue
        test = pd.read_csv(path, usecols=_TEST_COLUMNS)
        test = test[pd.to_datetime(test["Date"]) < before]
        market = archive_market(league, slug.replace("-", " "))
        offers = replay_offers(test, league, market, path.stem)
        if offers.empty:
            skipped.append(f"{path.stem}: no market-line row before {before.date()}")
            continue
        over_rate = float(over_label(test["Result"], test["Line"]).mean())
        frames.append(offers.assign(cell=market, over_rate=over_rate))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), skipped


def _scope_row(
    scope: str, league: str, market: str, offline: pd.DataFrame, live: pd.DataFrame
) -> dict:
    rng = key_rng(REPLAY_SEED, f"{scope}/{league}/{market}")
    off, liv = cohort(offline, rng), cohort(live, rng)
    row = {"scope": scope, "league": league, "market": market, **off}
    row.update({f"live_{key}": value for key, value in liv.items()})
    row["gap_delta"] = off["gap"] - liv["gap"]
    row["agrees"] = (
        (off["hit"] > BREAK_EVEN) == (liv["hit"] > BREAK_EVEN) if off["n"] and liv["n"] else None
    )
    row["implied_live_hit_lb"] = np.nan
    row["l2_flag"] = None
    return row


def l2_rows(offline: pd.DataFrame) -> pd.DataFrame:
    """The frozen L2 population: near-even sportsbook rows, and pick'em rows of balanced cells."""
    balanced = offline["over_rate"].between(*L2_OVER_RATE_BAND)
    return offline[
        offline["line_type"].eq("near_even") | (offline["line_type"].eq("pickem") & balanced)
    ]


def replay_table(offline: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    """One row per cell, league and L2 league scope; the live columns read the same scope live."""
    rows = []
    for (league, cell), part in offline.groupby(["League", "cell"]):
        live_part = live[(live["League"] == league) & (live["cell"] == cell)]
        rows.append(_scope_row("cell", league, cell, part, live_part))
    for league, part in offline.groupby("League"):
        live_part = live[live["League"] == league]
        rows.append(_scope_row("league", league, "", part, live_part))
        row = _scope_row("l2", league, "", l2_rows(part), live_part)
        if row["n"] >= L2_MIN_ROWS and row["dates"] >= L2_MIN_DATES and row["live_n"]:
            # The brief's "live claimed mean + the (hit - claimed) gap's one-sided 90 % lower bound".
            row["implied_live_hit_lb"] = row["live_read"] - row["gap_hi"]
            row["l2_flag"] = bool(row["implied_live_hit_lb"] > BREAK_EVEN)
        rows.append(row)
    return pd.DataFrame(rows)


def tolerance_rows(table: pd.DataFrame) -> pd.DataFrame:
    """The ``l2`` rows of leagues whose live sample holds ``LIVE_MIN_LEGS`` recommended legs."""
    return table[(table["scope"] == "l2") & (table["live_n"] >= LIVE_MIN_LEGS)]


@click.command()
@click.option("--league", "leagues", multiple=True, help="Replay only these leagues (repeatable).")
@click.option(
    "--test-sets-dir",
    type=click.Path(path_type=Path),
    default=TEST_SETS_DIR,
    help="Directory of {LEAGUE}_{market}.csv dumps (default: the package's data/test_sets).",
)
@click.option(
    "--offline-before",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=realized.COUNT_ONCE_FROM.strftime("%Y-%m-%d"),
    show_default=True,
    help="Test rows dated before this are the offline sample.",
)
@click.option(
    "--live-since",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=realized.COUNT_ONCE_FROM.strftime("%Y-%m-%d"),
    show_default=True,
    help="Live legs dated from this day count (count-once holds from COUNT_ONCE_FROM).",
)
@click.option(
    "--live-until",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Live legs dated before this day count (default: every settled leg).",
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=Path("/tmp/realized_replay.csv"),
    show_default=True,
)
@click.option(
    "--parquet",
    type=click.Path(path_type=Path),
    default=REALIZED_REPLAY_PATH,
    help="Where the scope rows land (default: data/training/realized_replay.parquet).",
)
def main(leagues, test_sets_dir, offline_before, live_since, live_until, out, parquet) -> None:
    """Replay the live recommendation rule on held-out market-line rows and read it against live."""
    before, since = pd.Timestamp(offline_before), pd.Timestamp(live_since)
    until = pd.Timestamp(live_until) if live_until else None
    offline, skipped = load_offline(test_sets_dir, tuple(lg.upper() for lg in leagues), before)
    for reason in skipped:
        click.echo(f"skipped {reason}")
    if offline.empty:
        raise click.ClickException("no market-line test row to replay")
    table = replay_table(offline, live_recommended(read_history(), since, until))
    table = table.assign(
        offline_before=before,
        live_since=since,
        live_until=until,
        computed_at=pd.Timestamp(datetime.now(UTC)).tz_localize(None),
    )[list(REPLAY_COLUMNS)].astype({"agrees": "boolean", "l2_flag": "boolean"})
    _atomic_write_csv(table, out)
    _atomic_write_parquet(table, parquet)
    leagues_out = table[table["scope"] != "cell"].set_index(["scope", "league"])
    shown = [
        "n",
        "dates",
        "read",
        "hit",
        "hit_lb",
        "gap",
        "gap_lo",
        "gap_hi",
        "live_n",
        "live_hit",
        "live_gap",
        "gap_delta",
        "agrees",
        "implied_live_hit_lb",
        "l2_flag",
    ]
    click.echo(leagues_out[shown].to_string(float_format=lambda x: f"{x:.3f}"))
    tolerance = tolerance_rows(table)
    deltas = ", ".join(
        f"{league} {delta:.3f}"
        for league, delta in zip(tolerance["league"], tolerance["gap_delta"], strict=True)
    )
    click.echo(
        f"tolerance reached: {deltas} (l2 rows, live_n >= {LIVE_MIN_LEGS}); "
        f"max |gap_delta| {tolerance['gap_delta'].abs().max():.3f}; sides of {BREAK_EVEN:.4f} "
        f"agree in {int(tolerance['agrees'].eq(True).sum())} of {len(tolerance)}"
    )
    below = table[(table["scope"] == "l2") & (table["live_n"] < LIVE_MIN_LEGS)]
    if not below.empty:
        click.echo(
            "below the live floor: "
            + ", ".join(
                f"{league} ({n})"
                for league, n in zip(below["league"], below["live_n"], strict=True)
            )
        )
    click.echo(f"wrote {len(table)} rows to {out} and {parquet}")
