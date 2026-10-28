"""Level archived sportsbook rows at training read time (I6d).

Two defects sit in the archive's historical book prices. The WS1 migration decoded every row
written before 2026-06-28 under the 06-28 shape, so a row encoded under another shape reads off
level; and a live writer defect priced duplicated same-side quotes at exactly 0.5. The archive
stays raw (``merge_archives`` keeps ``MAX(under_prob)``, so a repair that lowers a value is
undone at the next sync); the training reader levels each book-row here instead, before the
consensus forms.

The rules and constants are frozen by docs/handoffs/i6d-decode-preregistration.md (cited as
§n); the per-block verdicts are ``data/config/book_decode.json``, which
``scripts/fit_book_decode.py`` writes.
"""

from __future__ import annotations

import dataclasses
import datetime
import importlib.resources as pkg_resources
import json
from collections.abc import Sequence
from functools import cache
from typing import NamedTuple

import numpy as np

from sportstradamus import data
from sportstradamus.helpers.distributions import get_odds
from sportstradamus.helpers.training_quotes import (
    DFS_PLATFORM_BOOKS,
    LADDER_REPAIRED,
    REDECODED,
    ArchivedBookQuote,
    _positive,
    _probability,
    archive_ev_is_usable,
)

LIVE = "live"
STORED = "stored"
# §2: backfills stamp rows exactly on these UTC hours; any other observation is a live poll.
_STAMP_HOURS = (0, 1, 2, 23)
# §3: production's first run_job PULL (prophecize). From then every job ran devel HEAD, which
# carries the WS1 writer; the 06-28 deploy itself was never logged.
WS1_NATIVE_FROM = datetime.datetime(2026, 7, 11, 14, 50, 3)
# §4: the producer run is not recorded, so a season stands in for it; the month a season opens.
_SEASON_START_MONTH = {"NFL": 8, "NBA": 9, "NHL": 9, "MLB": 1, "WNBA": 1}
# §5, §11: a rung this far from the row's own over-probability marks the duplicate-side defect.
LADDER_DISAGREEMENT = 0.05
# §5, §11: an exact even price, which the duplicate-side defect writes.
EVEN_PRICE_TOLERANCE = 1e-9
# §11: without a rung, other books this sure of the under expose an even price as the defect.
_OTHER_BOOKS_UNDER_FLOOR = 0.70
# §9, §10: the verdicts that decode; a demoted block's rows are not a price.
_ACCEPTED_VERDICTS = ("accept", "inherit")


class BlockVerdict(NamedTuple):
    """One migrated block's decode; ``family`` None means its rows are not a price."""

    family: str | None
    cv: float | None


def stamp_layer(observed_at: datetime.datetime) -> str:
    """``"HH:00"`` for a backfill stamp, ``"live"`` for a poll (§2)."""
    on_the_hour = (observed_at.minute, observed_at.second, observed_at.microsecond) == (0, 0, 0)
    if on_the_hour and observed_at.hour in _STAMP_HOURS:
        return f"{observed_at.hour:02d}:00"
    return LIVE


def is_native(
    layer: str, observed_at: datetime.datetime, ladder_over_probability: float | None
) -> bool:
    """Whether the WS1 writer wrote the row (§3)."""
    if layer == "00:00":
        return False
    if layer == LIVE:
        return observed_at >= WS1_NATIVE_FROM
    return ladder_over_probability is not None


def block_key(league: str, game_date: datetime.date, layer: str) -> tuple[str, str]:
    """The decode table's ``(layer, block)`` for a migrated row (§4)."""
    if layer in ("00:00", LIVE):
        return LIVE, f"{game_date:%Y-%m}"
    start = _SEASON_START_MONTH[league]
    return layer, str(game_date.year if game_date.month >= start else game_date.year - 1)


def decode_under(
    family: str, line: float | np.ndarray, ev: float | np.ndarray, cv: float | None
) -> float | np.ndarray:
    """The pre-registered ungated decode of a stored ``ev`` (§7); vectorizes over arrays."""
    if family == "SkewNormal":
        return get_odds(line, ev, "SkewNormal", cv=cv, step=0.5)
    if family == "NegBin":
        # r explicit: get_odds reads NegBin at cv == 1 with no r as Poisson.
        return get_odds(line, ev, "NegBin", cv=cv, r=1.0 / cv, step=1.0)
    if family == "Poisson":
        return get_odds(line, ev, "Poisson", step=1.0)
    if family == "Gamma":
        return get_odds(line, ev, "Gamma", cv=cv, step=0.5)
    raise ValueError(f"no pre-registered decode for {family!r}")


@cache
def decode_table() -> dict[tuple[str, str, str, str], BlockVerdict]:
    """``(league, market, layer, block)`` -> verdict, from the committed table (§12)."""
    path = pkg_resources.files(data) / "config" / "book_decode.json"
    return {
        (block["league"], block["market"], block["layer"], block["block"]): BlockVerdict(
            block["family"] if block["verdict"] in _ACCEPTED_VERDICTS else None, block["cv"]
        )
        for block in json.loads(path.read_text())["blocks"]
    }


def ladder_repaired(row: ArchivedBookQuote) -> ArchivedBookQuote:
    """The row priced at its same-observation rung where the two disagree (§11, Class II)."""
    rung = row.ladder_over_probability
    if rung is None or not _probability(row.under_probability):
        return row
    if abs((1.0 - row.under_probability) - rung) <= LADDER_DISAGREEMENT:
        return row
    return dataclasses.replace(row, under_probability=1.0 - rung, levelling=LADDER_REPAIRED)


def _redecoded(row: ArchivedBookQuote, verdict: BlockVerdict) -> ArchivedBookQuote | None:
    """The migrated row under its block's validated decode, or None when it is not a price."""
    if verdict.family is None:
        return None
    if verdict.family == STORED:
        under = row.under_probability
    elif _positive(row.line) and archive_ev_is_usable(row.ev, row.line):
        under = float(decode_under(verdict.family, row.line, row.ev, verdict.cv))
    else:
        return None
    if not _probability(under):
        return None
    return dataclasses.replace(row, under_probability=under, levelling=REDECODED)


def _reads_duplicate_side(row: ArchivedBookQuote, rows: Sequence[ArchivedBookQuote]) -> bool:
    """An exact even price the other books on its line put well under (§11, no rung)."""
    if (
        not _probability(row.under_probability)
        or abs(row.under_probability - 0.5) >= EVEN_PRICE_TOLERANCE
    ):
        return False
    others = [
        other.under_probability
        for other in rows
        if other is not row
        and other.book not in DFS_PLATFORM_BOOKS
        and other.line == row.line
        and _probability(other.under_probability)
    ]
    return bool(others) and float(np.median(others)) > _OTHER_BOOKS_UNDER_FLOOR


def level_book_rows(
    league: str, market: str, game_date: datetime.date, rows: Sequence[ArchivedBookQuote]
) -> list[ArchivedBookQuote]:
    """One player-game's latest-per-book rows, levelled; rows that are not a price drop out.

    DFS platform rows pass through: their price is the platform's even payout, never book
    evidence. A migrated row whose block has no verdict stays as stored; the harness writes one
    for every archived block, so only test fixtures meet that rule.
    """
    table = decode_table()
    staged: list[tuple[ArchivedBookQuote, bool]] = []  # (row, open to the even-price rule)
    for row in rows:
        if row.book in DFS_PLATFORM_BOOKS:
            staged.append((row, False))
            continue
        layer = stamp_layer(row.observed_at)
        if is_native(layer, row.observed_at, row.ladder_over_probability):
            staged.append((ladder_repaired(row), row.ladder_over_probability is None))
            continue
        verdict = table.get((league, market, *block_key(league, game_date, layer)))
        if verdict is None:
            staged.append((row, False))
            continue
        levelled = _redecoded(row, verdict)
        if levelled is not None:
            staged.append((levelled, False))
    kept = [row for row, _ in staged]
    return [
        row for row, even_rule in staged if not (even_rule and _reads_duplicate_side(row, kept))
    ]
