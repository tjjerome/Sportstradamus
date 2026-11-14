"""The offline realized-money replay: the live rule on held-out rows at posted rungs and market lines.

The ``posted`` scope prices each dump's rows at the Underdog and Sleeper rungs the archive ladder
held for them (``tail_pricing.price_cell``, the dump's own pickle from ``--models-dir``), keeps the
rungs each prop still held at its last poll (``still_posted``), runs the live board's rule on them
(``tail_scorecard.replay_live_rule``) and keeps the recommended even-pick legs dated in the live
window. The live side is the live record's recommended even-pick legs on the same rows (league,
cell, player, date), and ``live_same_version`` is the share of them the dumps' own pickle read: a
held-out dump against live mixes model versions (a version serves only after its matrix ends), so
the replay's representativeness is read where that share is 1, on forward dumps.

The ``cell``, ``league`` and ``l2`` scopes are the first replay, kept until L2's pre-registered read
finishes: ``realized.recommended`` (``Win Prob x Payout - 1 >= 0.05``) over each cell's test rows
dated before ``--offline-before`` at Underdog's even-pick payout, on the rows that hold a market line:
pick'em rows (``QuoteAuthenticity == "pickem"``) and sportsbook rows priced within
``NEAR_EVEN_BAND`` of even. The league rows also carry the L2 gap-form column
(``implied_live_hit_lb``, ``l2_flag``).

Every scope reports the recommended cohort's read, hit and gap (``read - hit``, positive = the read
runs high, the sign of ``live_gap_per_cell.parquet``) with date-block bootstrap bounds, beside the
same numbers from the live record; ``gap_delta`` is how far the replay reads from live. The
tolerance is read on the ``posted`` league rows whose live sample holds ``LIVE_MIN_LEGS`` legs.

A report, never a gate (owner 2026-10-30, docs/handoffs/live-gap-2.md §5 item 5).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import click
import numpy as np
import pandas as pd

from sportstradamus import realized
from sportstradamus.helpers import UNDERDOG_BOOST_BASELINE, archive_market
from sportstradamus.helpers.distributions import DFS_FAIR_PICK_PROB
from sportstradamus.helpers.io import (
    REALIZED_REPLAY_PATH,
    _atomic_write_csv,
    _atomic_write_parquet,
    read_history,
)
from sportstradamus.helpers.training_quotes import AUTHENTIC, PICKEM
from sportstradamus.prediction.offer_records import _MAX_CONFIDENCE, UNQUOTED_BOOK_DISAGREEMENT_MAX
from sportstradamus.scripts.tail_information import key_rng
from sportstradamus.scripts.tail_pricing import SERVED_MODELS_DIR, open_archive, price_dumps
from sportstradamus.scripts.tail_scorecard import (
    DEFAULT_ARCHIVE,
    TEST_SETS_DIR,
    replay_live_rule,
    with_live_sides,
)
from sportstradamus.spiderLogger import logger
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
# The rungs one poll writes for a prop share a stamp to well inside this. A rung last seen
# earlier is a line the app had moved off, which count-once never counts live (keeping them
# doubled NFL's legs and lifted their hit .537 -> .583 on 2026-11-13: the line had moved toward
# the model).
SAME_POLL = pd.Timedelta(minutes=5)
_PROP_KEY = ["League", "Market", "Player", "Date", "Platform"]
_ROW_KEY = ["League", "cell", "Player", "Date"]
_TEST_COLUMNS = ["Player", "Date", "Result", "Line", "P", "Odds", "QuoteAuthenticity", "Book_EV"]
_COHORT_COLUMNS = ("n", "dates", "read", "hit", "hit_lb", "gap", "gap_lo", "gap_hi", "roi")
REPLAY_COLUMNS = (
    "scope",
    "league",
    "market",
    *_COHORT_COLUMNS,
    *(f"live_{col}" for col in _COHORT_COLUMNS),
    "gap_delta",
    "agrees",
    "live_same_version",
    "implied_live_hit_lb",
    "l2_flag",
    "offline_before",
    "live_since",
    "live_until",
    "computed_at",
)
_REPORT_COLUMNS = [
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
    "live_same_version",
    "implied_live_hit_lb",
    "l2_flag",
]


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
    the fused ``P``'s, its ``Win Prob`` capped as ``finalize_records`` caps it. A pick'em row's
    ``Market Prob`` is the payout-implied ``DFS_FAIR_PICK_PROB``, as the board prices an even pick
    (the dump's ``Odds`` there is the decoded book leg), and one whose capped ``Win Prob`` sits more
    than ``UNQUOTED_BOOK_DISAGREEMENT_MAX`` from that price drops as the board drops it: the one
    ``finalize_records`` gate binding at one line per player (the three-per-player trim and the
    boost cap cannot).
    """
    rows = market_line_rows(test)
    pickem = rows["line_type"].to_numpy() == "pickem"
    over = rows["P"].to_numpy() >= 0.5
    actual, line = rows["Result"].to_numpy(float), rows["Line"].to_numpy(float)
    book_over = 1 - rows["Odds"].to_numpy(float)
    win_prob = np.minimum(np.where(over, rows["P"], 1 - rows["P"]), _MAX_CONFIDENCE)
    offers = pd.DataFrame(
        {
            "League": league,
            "Market": market,
            "Player": rows["Player"].to_numpy(),
            "Date": rows["Date"].to_numpy(),
            "Line": line,
            "Bet": np.where(over, "Over", "Under"),
            "Win Prob": win_prob,
            "Market Prob": np.where(
                pickem, DFS_FAIR_PICK_PROB, np.where(over, book_over, 1 - book_over)
            ),
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
    phantom = pickem & (np.abs(win_prob - DFS_FAIR_PICK_PROB) > UNQUOTED_BOOK_DISAGREEMENT_MAX)
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


def _recommended_even_picks(
    offers: pd.DataFrame, since: pd.Timestamp, until: pd.Timestamp | None
) -> pd.DataFrame:
    """The recommended legs paying ``LIVE_PAYOUT_BAND`` dated in ``[since, until)``."""
    dates = pd.to_datetime(offers["Date"])
    keep = offers["Recommended"] & offers["Payout"].between(*LIVE_PAYOUT_BAND) & (dates >= since)
    if until is not None:
        keep &= dates < until
    return offers[keep]


def live_recommended(
    history: pd.DataFrame, since: pd.Timestamp, until: pd.Timestamp | None
) -> pd.DataFrame:
    """The live record's recommended even-pick legs in the window, with the archive cell key."""
    offers = _recommended_even_picks(realized.settled_offers(history), since, until)
    return offers.assign(
        cell=[
            archive_market(lg, mk)
            for lg, mk in zip(offers["League"], offers["Market"], strict=True)
        ]
    )


def _keyed(frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` with its archive cell and a day-string ``Date``: the key offline and live share."""
    return frame.assign(
        cell=[
            archive_market(lg, mk) for lg, mk in zip(frame["League"], frame["Market"], strict=True)
        ],
        Date=pd.to_datetime(frame["Date"]).dt.strftime("%Y-%m-%d"),
    )


def still_posted(rungs: pd.DataFrame) -> pd.Series:
    """Rungs their prop still held at its last poll: the offline twin of count-once."""
    last = rungs.groupby(_PROP_KEY)["last_poll"].transform("max")
    return (last - rungs["last_poll"]) <= SAME_POLL


def posted_legs(
    paths: list[Path],
    models_dir: Path,
    archive: Path,
    history: pd.DataFrame,
    since: pd.Timestamp,
    until: pd.Timestamp | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The dumps' rows at the rungs still posted, through the live board's rule.

    Returns the recommended even-pick legs dated in ``[since, until)``, and every priced row's
    (League, cell, Player, Date) with the version of the pickle that priced it.
    """
    con = open_archive(archive)
    rungs, knobs, _, _, excluded = price_dumps(paths, con, since.strftime("%Y-%m-%d"), models_dir)
    con.close()
    for stem, reason in excluded.items():
        click.echo(f"posted scope: excluded {stem}: {reason}")
    if rungs.empty:
        return pd.DataFrame(), pd.DataFrame(columns=[*_ROW_KEY, "version"])
    rungs, _ = with_live_sides(rungs, history)
    rungs = rungs.assign(version=[knobs[stem]["model_version"] for stem in rungs["cell"]])
    rows = _keyed(rungs[["League", "Market", "Player", "Date", "version"]])
    # finalize_records warns on every slate it trims, which offline is nearly all of them.
    logger.setLevel(logging.ERROR)
    replayed = pd.concat(
        [
            replay_live_rule(part, part["p_serving"], knobs[stem])
            for stem, part in rungs[still_posted(rungs)].groupby("cell")
        ],
        ignore_index=True,
    )
    return _keyed(_recommended_even_picks(replayed, since, until)), rows.drop_duplicates(_ROW_KEY)


def posted_table(legs: pd.DataFrame, rows: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    """League and cell rows of the ``posted`` scope: offline legs against live legs on the same rows.

    A live leg counts only on a priced row; ``live_same_version`` is the share of those legs the
    row's own pickle read.
    """
    live = _keyed(live).merge(rows[[*_ROW_KEY, "version"]], on=_ROW_KEY)
    live = live.assign(same_version=live["Model Version"].eq(live["version"]))
    records = []
    for league, cells in rows.groupby("League")["cell"]:
        off_league, live_league = legs["League"].eq(league), live["League"].eq(league)
        scopes = [("", off_league, live_league)]
        scopes += [
            (cell, off_league & legs["cell"].eq(cell), live_league & live["cell"].eq(cell))
            for cell in sorted(cells.unique())
        ]
        for market, off_mask, live_mask in scopes:
            row = _scope_row("posted", league, market, legs[off_mask], live[live_mask])
            row["live_same_version"] = live.loc[live_mask, "same_version"].mean()
            records.append(row)
    return pd.DataFrame(records)


def load_offline(paths: list[Path], before: pd.Timestamp) -> tuple[pd.DataFrame, list[str]]:
    """Every cell's market-line offers dated before ``before``, plus the dumps that could not be read."""
    frames, skipped = [], []
    for path in paths:
        league, slug = path.stem.split("_", 1)
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
    row["live_same_version"] = np.nan
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
        if (
            row["n"] >= L2_MIN_ROWS
            and row["dates"] >= L2_MIN_DATES
            and row["live_n"] >= LIVE_MIN_LEGS
        ):
            # The brief's "live claimed mean + the (hit - claimed) gap's one-sided 90 % lower bound".
            row["implied_live_hit_lb"] = row["live_read"] - row["gap_hi"]
            row["l2_flag"] = bool(row["implied_live_hit_lb"] > BREAK_EVEN)
        rows.append(row)
    return pd.DataFrame(rows)


def tolerance_rows(table: pd.DataFrame) -> pd.DataFrame:
    """The ``posted`` league rows whose live sample holds ``LIVE_MIN_LEGS`` recommended legs."""
    return table[
        (table["scope"] == "posted") & (table["market"] == "") & (table["live_n"] >= LIVE_MIN_LEGS)
    ]


def _echo_report(table: pd.DataFrame) -> None:
    leagues_out = table[table["market"] == ""].set_index(["scope", "league"])
    click.echo(leagues_out[_REPORT_COLUMNS].to_string(float_format=lambda x: f"{x:.3f}"))
    tolerance = tolerance_rows(table)
    click.echo(
        f"tolerance (posted rows, live_n >= {LIVE_MIN_LEGS}): "
        + ", ".join(
            f"{row.league} gap_delta {row.gap_delta:+.3f}, same version {row.live_same_version:.2f}"
            for row in tolerance.itertuples()
        )
    )
    posted = table[(table["scope"] == "posted") & (table["market"] == "")]
    click.echo(
        f"hit_lb against break-even {BREAK_EVEN:.4f} (posted rows): "
        + ", ".join(f"{row.league} {row.hit_lb:.3f} on {row.n}" for row in posted.itertuples())
    )
    flagged = table[(table["scope"] == "l2") & table["l2_flag"].notna()]
    matches = int((flagged["l2_flag"] == (flagged["live_hit"] > BREAK_EVEN)).sum())
    click.echo(f"l2_flag matches live's side in {matches} of {len(flagged)}")
    below = posted[posted["live_n"] < LIVE_MIN_LEGS]
    if not below.empty:
        click.echo(
            "below the live floor: "
            + ", ".join(
                f"{league} ({n})"
                for league, n in zip(below["league"], below["live_n"], strict=True)
            )
        )


@click.command()
@click.option("--league", "leagues", multiple=True, help="Replay only these leagues (repeatable).")
@click.option(
    "--test-sets-dir",
    type=click.Path(path_type=Path),
    default=TEST_SETS_DIR,
    help="Directory of {LEAGUE}_{market}.csv dumps (default: the package's data/test_sets).",
)
@click.option(
    "--models-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=SERVED_MODELS_DIR,
    help="The dumps' own pickles, which price the posted scope (default: the package's "
    "data/models; a sandbox run's models beside its dumps).",
)
@click.option(
    "--archive",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_ARCHIVE,
    help="Odds archive whose ladder prices the posted scope, opened read-only (default: "
    "$SPORTSTRADAMUS_ARCHIVE_DB or archive/archive.duckdb).",
)
@click.option(
    "--offline-before",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=realized.COUNT_ONCE_FROM.strftime("%Y-%m-%d"),
    show_default=True,
    help="Test rows dated before this are the cell, league and l2 scopes' offline sample.",
)
@click.option(
    "--live-since",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=realized.COUNT_ONCE_FROM.strftime("%Y-%m-%d"),
    show_default=True,
    help="Live legs, and the posted scope's offline legs, dated from this day count "
    "(count-once holds from COUNT_ONCE_FROM).",
)
@click.option(
    "--live-until",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Live legs, and the posted scope's offline legs, dated before this day count "
    "(default: every settled leg).",
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
def main(
    leagues: tuple[str, ...],
    test_sets_dir: Path,
    models_dir: Path,
    archive: Path,
    offline_before: datetime,
    live_since: datetime,
    live_until: datetime | None,
    out: Path,
    parquet: Path,
) -> None:
    """Replay the live rule on held-out rows at posted rungs and market lines; read it against live."""
    before, since = pd.Timestamp(offline_before), pd.Timestamp(live_since)
    until = pd.Timestamp(live_until) if live_until else None
    wanted = {lg.upper() for lg in leagues}
    paths = [
        path
        for path in sorted(test_sets_dir.glob("*.csv"))
        if not wanted or path.stem.split("_", 1)[0] in wanted
    ]
    history = read_history()
    live = live_recommended(history, since, until)
    offline, skipped = load_offline(paths, before)
    for reason in skipped:
        click.echo(f"skipped {reason}")
    legs, rows = posted_legs(paths, models_dir, archive, history, since, until)
    tables = [posted_table(legs, rows, live)]
    if not offline.empty:
        tables.append(replay_table(offline, live))
    table = pd.concat(tables, ignore_index=True)
    if table.empty:
        raise click.ClickException("no test row to replay at a market line or a posted rung")
    table = table.assign(
        offline_before=before,
        live_since=since,
        live_until=until,
        computed_at=pd.Timestamp(datetime.now(UTC)).tz_localize(None),
    )[list(REPLAY_COLUMNS)].astype({"agrees": "boolean", "l2_flag": "boolean"})
    _atomic_write_csv(table, out)
    _atomic_write_parquet(table, parquet)
    _echo_report(table)
    click.echo(f"wrote {len(table)} rows to {out} and {parquet}")
