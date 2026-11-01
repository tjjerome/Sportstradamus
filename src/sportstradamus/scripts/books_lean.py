"""X2, the books'-lean paper test: do the sportsbooks' raw rungs at the posted line beat the apps?

Model-free and report-only (the owner's test; never a serving rule). On every settled leg the
board posted since ``COUNT_ONCE_FROM``, the sportsbooks' ladder rungs at the leg's line as of its
``Scored At`` give a mean over-probability (``rung_book_leg``); a lean of ``LEAN_MIN`` or more
picks that side, a lean past ``LEAN_MAX`` is a stale or one-sided ladder and sits out, and the
pick is graded at the payout the app posted for that side (``Payout Over`` / ``Payout Under``; a
side the app never posted sits out). Rare-outcome cells sit out (their rungs are one-sided). The
read is anytime-valid: a normal-mixture confidence sequence (Howard, Ramdas, McAuliffe and Sekhon
2021, with Hoeffding's variance proxy t/4) tuned at ``CS_DESIGN_LEGS`` legs bounds the mean excess
over break-even at every look, beside a date-block 90 % lower bound; the secondary arms read at
the Bonferroni-split level. Pre-registered in docs/archive/researcher_model_class_and_edge.md
(X2). Hand-run weekly after ``sync_from_prod.sh`` and the archive merge; every look is appended to
the looks ledger so the sequence is honest about how often it was read.
"""

from __future__ import annotations

import math
import os
from datetime import UTC, datetime
from pathlib import Path

import click
import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from sportstradamus import realized
from sportstradamus.helpers.archive import LADDER_POLL_TOLERANCE_S, archive_market
from sportstradamus.helpers.io import (
    BOOKS_LEAN_LEDGER_PATH,
    BOOKS_LEAN_LOOKS_PATH,
    _atomic_write_parquet,
    read_history,
    read_parquet_safe,
)
from sportstradamus.helpers.training_quotes import DFS_PLATFORM_BOOKS
from sportstradamus.prediction.book_quotes import rung_book_leg
from sportstradamus.scripts.tail_information import key_rng
from sportstradamus.scripts.tail_pricing import open_archive
from sportstradamus.training.scorecard import _bootstrap_mean_ci_clustered

# The lean that picks a side, and the lean past which a ladder is stale or one-sided (brief X2).
LEAN_MIN = 0.55
LEAN_MAX = 0.65
# Cells whose rungs are one-sided (anytime scorer, a rare count); the lean there is the vig.
RARE_MARKETS = {
    "MLB": frozenset(
        {
            "home runs",
            "stolen bases",
            "doubles",
            "triples",
            "singles",
            "walks",
            "rbi",
            "runs",
            "hits",
        }
    ),
    "NHL": frozenset({"goals", "powerPlayPoints"}),
    "NFL": frozenset({"tds"}),
}
# A rung polled within this many hours of the scoring is "fresh" (secondary arm).
FRESH_RUNG_H = 1.0
SECONDARY_ARMS = ("books3", "fresh", "model_agrees")
# The sequence's level and the sample it is tightest at (the brief's accept floor).
CS_ALPHA = 0.10
CS_DESIGN_LEGS = 1000
ACCEPT_MIN_LEGS = 1000
ACCEPT_MIN_DATES = 60
KILL_MIN_LEGS = 400
KILL_HIT = 0.53
# The one-sided 90 % lower bound the brief pre-registered, as the bootstrap's percentiles.
_PCT_90 = (10.0, 90.0)
# One generator per stratum (key_rng), so a stratum's bounds hold still as other strata appear.
LEAN_SEED = 1729
_DEFAULT_ARCHIVE = os.environ.get("SPORTSTRADAMUS_ARCHIVE_DB", "archive/archive.duckdb")
LOOK_COLUMNS = (
    "look_at",
    "arm",
    "platform",
    "league_group",
    "n",
    "dates",
    "hit",
    "breakeven",
    "excess",
    "excess_lb90",
    "cs_lo",
    "cs_hi",
    "alpha",
    "verdict",
)
# Archive.get_ladder_rungs's read, joined to the legs frame with each leg's own decision time.
_LEAN_SQL = """
WITH scope AS (
    SELECT k.rid, l.book, l.line, l.p_over, l.observed_at
    FROM legs k JOIN ladder l ON l.league = k.league AND l.market = k.market
        AND l.game_date = k.game_date AND l.entity = k.entity
    WHERE NOT list_contains($dfs, l.book) AND l.observed_at <= k.scored_at
), polls AS (SELECT rid, book, max(observed_at) AS last_poll FROM scope GROUP BY ALL)
SELECT s.rid, s.book, s.line, arg_max(s.p_over, s.observed_at) AS p_over,
    max(s.observed_at) AS observed_at
FROM scope s JOIN polls p USING (rid, book)
WHERE s.observed_at >= p.last_poll - to_seconds($tolerance)
GROUP BY ALL ORDER BY rid, book, line
"""


def qualified_legs(history: pd.DataFrame) -> pd.DataFrame:
    """One settled leg per prop since ``COUNT_ONCE_FROM`` with a scoring stamp, outside the rare cells.

    The main line stands for a prop posted at several (``Alt Line`` False first); the archive cell
    name joins the ladder.
    """
    legs = realized.settled_offers(history)
    since = pd.to_datetime(legs["Date"]) >= realized.COUNT_ONCE_FROM
    legs = legs[since & legs["Scored At"].notna()]
    legs = legs.sort_values("Alt Line", kind="stable").drop_duplicates(subset=realized._PROP_KEY)
    cells = [archive_market(lg, mk) for lg, mk in zip(legs["League"], legs["Market"], strict=True)]
    legs = legs.assign(cell=cells)
    rare = [
        cell in RARE_MARKETS.get(league, ())
        for league, cell in zip(legs["League"], legs["cell"], strict=True)
    ]
    return legs[~np.array(rare, dtype=bool)].sort_values(["Date", "Player"]).reset_index(drop=True)


def ladder_leans(con: duckdb.DuckDBPyConnection, legs: pd.DataFrame) -> pd.DataFrame:
    """Per leg: the sportsbooks' rung leg at its line as of ``Scored At``, the book count, the rung age.

    ``age_h`` is the hours from the newest rung's poll to the scoring; NaN with ``p_books`` where
    no sportsbook brackets the line.
    """
    scored_at = pd.to_datetime(legs["Scored At"]).reset_index(drop=True)
    keys = pd.DataFrame(
        {
            "rid": np.arange(len(legs)),
            "league": legs["League"].to_numpy(),
            "market": legs["cell"].to_numpy(),
            "game_date": pd.to_datetime(legs["Date"]).dt.date.to_numpy(),
            "entity": legs["Player"].to_numpy(),
            "scored_at": scored_at.to_numpy(),
        }
    )
    con.register("legs", keys)
    params = {"dfs": sorted(DFS_PLATFORM_BOOKS), "tolerance": LADDER_POLL_TOLERANCE_S}
    rungs = con.execute(_LEAN_SQL, params).df()
    lines = legs["Line"].to_numpy(float)
    p_books, age_h = np.full(len(legs), np.nan), np.full(len(legs), np.nan)
    n_books = np.zeros(len(legs), dtype=int)
    for rid, group in rungs.groupby("rid"):
        p_books[rid], n_books[rid] = rung_book_leg(group, lines[rid])
        age_h[rid] = (scored_at[rid] - group["observed_at"].max()) / pd.Timedelta(hours=1)
    return pd.DataFrame({"p_books": p_books, "n_books": n_books, "age_h": age_h}, index=legs.index)


def paper_sides(legs: pd.DataFrame) -> pd.DataFrame:
    """The paper pick per leg: the lean's side inside the band, priced by that side's own posted payout.

    ``Payout Over`` / ``Payout Under`` are the record's decimal payouts of each side at the last
    scoring, so a pick against the board's ``Bet`` is priced as the app posted it, never derived.
    """
    p = legs["p_books"]
    side = np.select([p >= LEAN_MIN, p <= 1 - LEAN_MIN], ["Over", "Under"], None)
    payout = np.where(side == "Over", legs["Payout Over"], legs["Payout Under"])
    legs = legs.assign(side=side, payout=payout)
    inside = legs["side"].notna() & p.between(1 - LEAN_MAX, LEAN_MAX)
    # The record writes 0 for a side the app never posted; such a pick sits out.
    picks = legs[inside & (legs["payout"] > 0)].copy()
    picks["breakeven"] = 1 / picks["payout"]
    picks["hit"] = picks["side"].eq(picks["Result"])
    picks["excess"] = picks["hit"].astype(float) - picks["breakeven"]
    picks["books3"] = picks["n_books"] >= 3
    picks["fresh"] = picks["age_h"] <= FRESH_RUNG_H
    picks["model_agrees"] = picks["side"].eq(picks["Bet"])
    picks["league_group"] = np.where(picks["League"].eq("NHL"), "NHL", "other")
    return picks


def _boundary(v: float, rho: float, alpha: float) -> float:
    return math.sqrt((v + rho) * math.log((v + rho) / (rho * alpha**2)))


def mixture_rho(design_legs: int, alpha: float) -> float:
    """The mixture scale that makes the boundary tightest at the design sample (variance t/4)."""
    v0 = design_legs / 4
    # The boundary's derivative in rho vanishes where log((v0 + rho) / (rho alpha^2)) = v0 / rho;
    # that residual rises monotonically in rho, so the bracket holds its one root.
    return brentq(lambda rho: math.log((v0 + rho) / (rho * alpha**2)) - v0 / rho, 1e-3, 1e6)


def sequence_bounds(excess: np.ndarray, alpha: float) -> tuple[float, float]:
    """The confidence sequence for the mean excess after these legs: mean +/- u(t/4) / t."""
    t = len(excess)
    radius = _boundary(t / 4, mixture_rho(CS_DESIGN_LEGS, alpha), alpha) / t
    return float(excess.mean() - radius), float(excess.mean() + radius)


def verdict(n: int, dates: int, hit: float, excess_lb90: float, cs_lo: float, cs_hi: float) -> str:
    """The brief's rule: kill reads first, accept needs the sample floors or the sequence, else open."""
    if cs_hi < 0 or (n >= KILL_MIN_LEGS and hit <= KILL_HIT):
        return "kill"
    if (n >= ACCEPT_MIN_LEGS and dates >= ACCEPT_MIN_DATES and excess_lb90 > 0) or cs_lo > 0:
        return "accept"
    return "open"


def look_rows(picks: pd.DataFrame, look_at: pd.Timestamp) -> pd.DataFrame:
    """One row per stratum of this look.

    The primary arm pooled and by platform x league group, then each secondary arm pooled at the
    Bonferroni-split level.
    """
    strata = [("primary", "all", "all", picks, CS_ALPHA)]
    strata += [
        ("primary", platform, group, part, CS_ALPHA)
        for (platform, group), part in picks.groupby(["Platform", "league_group"])
    ]
    strata += [
        (arm, "all", "all", picks[picks[arm]], CS_ALPHA / len(SECONDARY_ARMS))
        for arm in SECONDARY_ARMS
    ]
    rows = []
    for arm, platform, group, part, alpha in strata:
        if part.empty:
            continue
        excess = part["excess"].to_numpy(float)
        rng = key_rng(LEAN_SEED, f"{arm}/{platform}/{group}")
        mean, lb, _ = _bootstrap_mean_ci_clustered(
            excess, part["Date"].to_numpy(), rng, percentiles=_PCT_90
        )
        cs_lo, cs_hi = sequence_bounds(excess, alpha)
        n, dates, hit = len(part), int(part["Date"].nunique()), float(part["hit"].mean())
        rows.append(
            {
                "look_at": look_at,
                "arm": arm,
                "platform": platform,
                "league_group": group,
                "n": n,
                "dates": dates,
                "hit": hit,
                "breakeven": float(part["breakeven"].mean()),
                "excess": mean,
                "excess_lb90": lb,
                "cs_lo": cs_lo,
                "cs_hi": cs_hi,
                "alpha": alpha,
                "verdict": verdict(n, dates, hit, lb, cs_lo, cs_hi),
            }
        )
    return pd.DataFrame(rows, columns=list(LOOK_COLUMNS))


@click.command()
@click.option(
    "--archive",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_ARCHIVE,
    help="Odds archive, opened read-only (default: $SPORTSTRADAMUS_ARCHIVE_DB or "
    "archive/archive.duckdb).",
)
@click.option(
    "--ledger",
    type=click.Path(dir_okay=False, path_type=Path),
    default=BOOKS_LEAN_LEDGER_PATH,
    help="This look's graded legs (default: data/training/books_lean_ledger.parquet).",
)
@click.option(
    "--looks",
    type=click.Path(dir_okay=False, path_type=Path),
    default=BOOKS_LEAN_LOOKS_PATH,
    help="Every look's rows, appended (default: data/training/books_lean_looks.parquet).",
)
def main(archive: Path, ledger: Path, looks: Path) -> None:
    """Grade the sportsbooks' lean at the posted line on the settled legs (X2, report only)."""
    con = open_archive(archive)
    legs = qualified_legs(read_history())
    leans = ladder_leans(con, legs)
    picks = paper_sides(legs.join(leans))
    look_at = pd.Timestamp(datetime.now(UTC)).tz_localize(None)
    table = look_rows(picks, look_at)
    previous = read_parquet_safe(looks)
    _atomic_write_parquet(picks, ledger)
    _atomic_write_parquet(pd.concat([previous, table], ignore_index=True), looks)
    with_leg = int(leans["p_books"].notna().sum())
    click.echo(f"legs {len(legs)} | with a rung leg {with_leg} | picks {len(picks)}")
    click.echo(
        table.drop(columns="look_at").to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )
    looks_before = 0 if previous.empty else previous["look_at"].nunique()
    click.echo(f"look {looks_before + 1}; wrote {ledger} and {looks}")
