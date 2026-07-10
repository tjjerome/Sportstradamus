"""Credit-budget pacing governor for The Odds API.

The plus key (``odds_api_plus``) carries ``cycle_credits`` (20,000) per
monthly billing cycle, resetting on ``cycle_reset_day``. The quota headers
on every response — ``X-Requests-Remaining`` / ``X-Requests-Used`` /
``X-Requests-Last`` — are the source of truth for budget left; the ledger
below never decides remaining credits, it only feeds cost *estimation*.

Every governed response appends one JSON line to
``helpers.io.ODDS_USAGE_LEDGER_PATH`` with the schema::

    {ts, run, kind, key, endpoint, sport_key, league, cost, used, remaining, status}

``run``/``kind`` come from :func:`set_run_context` ("broad" confer vs
"close_lines"); ``key`` is the keys.json entry *name*, never the secret.

Floor invariant: the close-lines pass is the per-game floor and is never
governed. Before a broad confer run, :func:`broad_run_allowance` reserves
the projected floor spend to cycle end (recent close-lines cost/day x days
left x ``floor_safety_factor``), splits what remains across the broad slots
left in the cycle, and admits leagues in ``league_priority`` order while
their estimated per-run cost (recent-ledger mean per league, else the
config's ``league_seed_costs`` estimate, else ``default_league_run_cost``)
still fits the slot share. Config lives in
``data/config/odds_api_budget.json`` — loaded eagerly here, not by
``helpers.config``.

League activity is derived from the free Odds API events feed instead of
hand-maintained season-start constants. Each broad run tiers every league
of interest by days to its next scheduled game (:func:`classify_tier`):
``live`` once a game is within ``promote_within_days`` — sticky through
mid-season gaps (NFL weekdays, all-star breaks) while any game stays inside
``poll_horizon_days`` — and ``preseason`` from ``poll_horizon_days`` out,
admitted after every live league. Leagues with nothing inside the horizon
are idle: not polled, kept in the snapshot at
``helpers.io.LEAGUE_ACTIVITY_PATH`` only as an ``idle`` record remembering
the last seen game. :func:`league_is_live` serves the snapshot to the
prophecize/meditate/pickem-build offer loops, and
:func:`update_window_open` gates ``Stats.update`` itself (10 days before
the next game to 10 days after the last; ``SPORTSTRADAMUS_FORCE_UPDATE=1``
bypasses). Season-start constants remain only as the stale-snapshot
fallback and for training semantics.
"""

from __future__ import annotations

import calendar
import importlib.resources as pkg_resources
import json
import os
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlsplit

from sportstradamus import creds, data
from sportstradamus.helpers.io import ODDS_USAGE_LEDGER_PATH, read_league_activity

with (pkg_resources.files(data) / "config" / "odds_api_budget.json").open() as infile:
    BUDGET_CFG: dict = json.load(infile)

# Ledger records label calls by key NAME so the secret value never hits disk.
with (pkg_resources.files(creds) / "keys.json").open() as infile:
    _KEY_NAMES = {v: k for k, v in json.load(infile).items() if k.startswith("odds_api")}

_SPORT_KEY_LEAGUE = {
    "basketball_nba": "NBA",
    "americanfootball_nfl": "NFL",
    "basketball_wnba": "WNBA",
    "baseball_mlb": "MLB",
    "icehockey_nhl": "NHL",
}

# ~1.5 cycles of ledger history: covers the estimate window plus a full-cycle
# retrospective while keeping the file small.
_LEDGER_RETENTION_DAYS = 45

# Broad confer refreshes the activity snapshot ~5x/day; older than this means
# confer is down, so the update gates fall back to season-start constants.
_ACTIVITY_STALE_DAYS = 3

# Legacy pre-season update window, kept as the activity-snapshot fallback.
_SEASON_START_LEAD_DAYS = 7

# Gamelog refresh runs from this many days before a league's next scheduled
# game to this many days after its last one (owner-set: 10 either side).
_UPDATE_WINDOW_DAYS = 10

# "unknown" so backfill scripts that import moneylines still record ledger
# lines without tagging a run kind.
_run = {
    "kind": "unknown",
    "run_id": datetime.now(UTC).isoformat(),
    "calls": 0,
    "cost": 0,
}


def set_run_context(kind: str) -> None:
    """Stamp the run context ("broad" or "close_lines") and zero the totals."""
    _run.update(kind=kind, run_id=datetime.now(UTC).isoformat(), calls=0, cost=0)


def run_summary() -> dict:
    """Return ``{kind, run_id, calls, cost}`` for the end-of-run log line."""
    return dict(_run)


def _endpoint_kind(path: str) -> str:
    hist = "/historical/" in path or "/odds-history" in path
    prefix = "hist_" if hist else ""
    trimmed = path.rstrip("/")
    if "/events/" in path and trimmed.endswith("/odds"):
        return prefix + "event_odds"
    if trimmed.endswith("/events"):
        return prefix + "events"
    if trimmed.endswith(("/odds", "/odds-history")):
        return prefix + "odds"
    return prefix + "sports_index"


def _sport_key(path: str) -> str | None:
    parts = path.split("/")
    try:
        sport = parts[parts.index("sports") + 1]
    except (ValueError, IndexError):
        return None
    return sport or None


def record_response(url: str, params: dict | None, res) -> None:
    """Append one ledger line for an Odds API ``requests.Response``.

    No-op when ``X-Requests-Remaining`` is absent — error/malformed responses
    carry no quota headers, so there is nothing to account.
    """
    remaining = res.headers.get("X-Requests-Remaining")
    if remaining is None:
        return
    path = urlsplit(url).path
    sport_key = _sport_key(path)
    cost = int(res.headers.get("X-Requests-Last", 0))
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "run": _run["run_id"],
        "kind": _run["kind"],
        "key": _KEY_NAMES.get((params or {}).get("apiKey"), "unknown"),
        "endpoint": _endpoint_kind(path),
        "sport_key": sport_key,
        "league": _SPORT_KEY_LEAGUE.get(sport_key),
        "cost": cost,
        "used": int(res.headers.get("X-Requests-Used", 0)),
        "remaining": int(remaining),
        "status": res.status_code,
    }
    ledger = Path(str(ODDS_USAGE_LEDGER_PATH))
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a") as f:
        f.write(json.dumps(record) + "\n")
    _run["calls"] += 1
    _run["cost"] += cost


def cycle_bounds(now: datetime, reset_day: int) -> tuple[datetime, datetime]:
    """Return ``(cycle_start, cycle_end)`` around ``now`` for a monthly reset.

    ``reset_day`` is clamped per month (day 31 in February resolves to the
    28th/29th), and the clamped day is what ``now`` is compared against.
    """
    year, month = now.year, now.month
    if now.day < min(reset_day, calendar.monthrange(year, month)[1]):
        year, month = (year - 1, 12) if month == 1 else (year, month - 1)
    next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
    return (
        _month_anchor(year, month, reset_day, now.tzinfo),
        _month_anchor(next_year, next_month, reset_day, now.tzinfo),
    )


def _month_anchor(year: int, month: int, reset_day: int, tz) -> datetime:
    day = min(reset_day, calendar.monthrange(year, month)[1])
    return datetime(year, month, day, tzinfo=tz)


def _read_ledger(ledger: Path) -> list[tuple[datetime, str, dict]]:
    """Parse the JSONL ledger into ``(ts, raw_line, record)`` triples.

    A torn final line (process killed mid-append) is skipped; any other
    malformed content fails loud.
    """
    if not ledger.is_file():
        return []
    with ledger.open() as f:
        lines = f.readlines()
    entries = []
    for i, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                continue
            raise
        entries.append((datetime.fromisoformat(record["ts"]), line, record))
    return entries


def _league_run_means(recent: list[dict]) -> dict[str, float]:
    """Mean summed cost per league across broad runs (null-league rows excluded)."""
    run_totals: dict[tuple[str, str], float] = defaultdict(float)
    for r in recent:
        if r["kind"] == "broad" and r["league"]:
            run_totals[(r["run"], r["league"])] += r["cost"]
    per_league: dict[str, list[float]] = defaultdict(list)
    for (_, league), total in run_totals.items():
        per_league[league].append(total)
    return {league: sum(runs) / len(runs) for league, runs in per_league.items()}


def estimate_costs(now: datetime, cfg: dict) -> tuple[float, dict[str, float]]:
    """Estimate spend rates from the usage ledger.

    Returns ``(floor_per_day, league_costs)``: recent close-lines burn per day
    (floored at ``floor_min_per_day``) and the mean broad-run cost per league,
    both over the last ``estimate_window_days`` of ``odds_api_plus`` records.
    ``league_seed_costs`` from the config bootstrap leagues the ledger hasn't
    seen yet (fresh deploy, season start); a ledger mean always wins over its
    seed. Opportunistically prunes records older than the retention horizon.
    """
    ledger = Path(str(ODDS_USAGE_LEDGER_PATH))
    entries = _read_ledger(ledger)

    window_start = now - timedelta(days=cfg["estimate_window_days"])
    recent = [r for ts, _, r in entries if ts >= window_start and r["key"] == "odds_api_plus"]
    close_total = sum(r["cost"] for r in recent if r["kind"] == "close_lines")
    floor_per_day = max(close_total / cfg["estimate_window_days"], cfg["floor_min_per_day"])
    league_costs = {**cfg.get("league_seed_costs", {}), **_league_run_means(recent)}

    keep_cutoff = now - timedelta(days=_LEDGER_RETENTION_DAYS)
    if any(ts < keep_cutoff for ts, _, _ in entries):
        tmp = ledger.with_suffix(ledger.suffix + ".tmp")
        with tmp.open("w") as f:
            f.writelines(line for ts, line, _ in entries if ts >= keep_cutoff)
        tmp.replace(ledger)

    return floor_per_day, league_costs


def classify_tier(days_to_next: float | None, previous_tier: str | None, cfg: dict) -> str | None:
    """Tier a league by days to its next scheduled game.

    ``live`` within ``promote_within_days`` (in-progress games count — a
    negative gap is within), or whenever the league was already live and
    still has a game inside ``poll_horizon_days``: the sticky bit keeps NFL
    weekdays and all-star breaks from demoting a mid-season league.
    ``preseason`` inside ``poll_horizon_days``; ``None`` (idle) beyond it or
    when no game is scheduled at all.
    """
    if days_to_next is None or days_to_next > cfg["poll_horizon_days"]:
        return None
    if days_to_next <= cfg["promote_within_days"] or previous_tier == "live":
        return "live"
    return "preseason"


def _fresh_leagues() -> dict | None:
    """The activity snapshot's leagues map when fresh, else ``None``.

    ``None`` sends callers to their season-start-constant fallback: the
    snapshot is missing (fresh deploy) or older than ``_ACTIVITY_STALE_DAYS``
    (confer is down), and a stale feed must never freeze the pipelines.
    """
    activity = read_league_activity()
    updated = activity.get("updated")
    if updated is None:
        return None
    if datetime.now(UTC) - datetime.fromisoformat(updated) > timedelta(days=_ACTIVITY_STALE_DAYS):
        return None
    return activity["leagues"]


def league_is_live(league: str, season_start) -> bool:
    """Gate for the per-league offer loops (prophecize, meditate, pickem-build).

    A league is live when the feed-derived activity snapshot tiers it
    ``live`` or ``preseason`` (both poll — preseason starts
    ``poll_horizon_days`` before opening night). A missing or stale snapshot
    falls back to the legacy ``season_start`` window.
    """
    leagues = _fresh_leagues()
    if leagues is not None:
        return leagues.get(league, {}).get("tier") in ("live", "preseason")
    return datetime.today().date() > season_start - timedelta(days=_SEASON_START_LEAD_DAYS)


def _feed_ts(commence: str) -> datetime:
    return datetime.fromisoformat(commence.replace("Z", "+00:00"))


def update_window_open(league: str, season_start) -> bool:
    """Season gate consulted by ``Stats.update`` before hitting league APIs.

    Open within ``_UPDATE_WINDOW_DAYS`` of a scheduled game on either side:
    from 10 days before the league's next game (season approach) to 10 days
    after its last seen one (post-season ingest tail). Set
    ``SPORTSTRADAMUS_FORCE_UPDATE=1`` to bypass for offseason rebuilds and
    backfills. A league the fresh snapshot has never seen is closed; a
    missing or stale snapshot falls back to the ``season_start`` constant
    (which has no season-end memory).
    """
    if os.environ.get("SPORTSTRADAMUS_FORCE_UPDATE"):
        return True
    leagues = _fresh_leagues()
    if leagues is None:
        return datetime.today().date() > season_start - timedelta(days=_UPDATE_WINDOW_DAYS)
    record = leagues.get(league, {})
    window = timedelta(days=_UPDATE_WINDOW_DAYS)
    now = datetime.now(UTC)
    next_game = record.get("next_game")
    if next_game is not None and _feed_ts(next_game) - now <= window:
        return True
    last_game = record.get("last_game")
    return last_game is not None and now - _feed_ts(last_game) <= window


class BudgetDecision(NamedTuple):
    """Outcome of the broad-run budget check."""

    allowed_leagues: tuple[str, ...]
    reason: str  # "ok" | "partial" | "floor_reserve" | "no_fit" | "idle"
    reserve: float
    per_slot: float
    spendable: float


def broad_run_allowance(
    now: datetime,
    remaining: int,
    league_tiers: dict[str, str],
    floor_per_day: float,
    league_costs: dict[str, float],
    cfg: dict,
) -> BudgetDecision:
    """Decide which leagues a broad confer run may fetch within budget.

    ``remaining`` is the live ``X-Requests-Remaining`` header value. The
    close-lines floor to cycle end (x ``floor_safety_factor``) is reserved
    first; the rest is split across the broad slots left in the cycle and
    handed out while each league's estimated run cost still fits — live
    leagues in ``league_priority`` order, then preseason leagues likewise
    (about-to-start leagues ride the bottom of the list until opening night).
    """
    admit_order = [lg for lg in cfg["league_priority"] if league_tiers.get(lg) == "live"] + [
        lg for lg in cfg["league_priority"] if league_tiers.get(lg) == "preseason"
    ]
    if not admit_order:
        return BudgetDecision((), "idle", 0.0, 0.0, 0.0)
    _, cycle_end = cycle_bounds(now, cfg["cycle_reset_day"])
    days_left = (cycle_end - now).total_seconds() / 86400
    reserve = floor_per_day * days_left * cfg["floor_safety_factor"]
    spendable = remaining - reserve
    if spendable <= 0:
        return BudgetDecision((), "floor_reserve", reserve, 0.0, spendable)
    per_slot = spendable / max(cfg["broad_slots_per_day"] * days_left, 1.0)
    allowance = per_slot
    allowed: list[str] = []
    for league in admit_order:
        cost = league_costs.get(league, cfg["default_league_run_cost"])
        if cost <= allowance:
            allowed.append(league)
            allowance -= cost
    if len(allowed) == len(admit_order):
        reason = "ok"
    elif allowed:
        reason = "partial"
    else:
        reason = "no_fit"
    return BudgetDecision(tuple(allowed), reason, reserve, per_slot, spendable)
