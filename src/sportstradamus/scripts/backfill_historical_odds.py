"""Backfill real two-sided book EVs for markets whose archived ``ev`` is a
degenerate consensus-line seed.

The live odds archive stores only de-vigged ``ev`` (raw American prices are
never persisted). The NFL passing family (``passing yards``, ``attempts``,
``completions``, ``passing tds``, ``interceptions`` …) was seeded into the old
klepto archive with the median line stamped as *every* book's ``ev`` — so the
reconstructed book probability is a flat ``0.5`` coin flip and the ship gates
grade the model against a non-existent book. See
``project_passing_book_degenerate`` for the diagnosis.

This driver re-fetches real historical two-sided prices through the existing
:func:`sportstradamus.moneylines.get_props` historical path and writes them
with a game-day ``01:00`` ``observed_at`` so point-in-time training reads
prefer them over the midnight seed rows.

Validate before spending: ``--dry-run`` makes the same paid API calls but
routes the parsed result into a capturing stub (no archive write) and prints
the per-book ``ev`` spread per market. A non-zero cross-book std means real
de-vig came back; a zero std means the source is degenerate too and a backfill
will not help.

    poetry run python -m sportstradamus.scripts.backfill_historical_odds \
        --league NFL --markets "passing yards,attempts,completions" \
        --start 2023-10-29 --end 2023-10-29 --max-dates 1 --dry-run
"""

import datetime
import importlib.resources as pkg_resources
import json
from pathlib import Path

import click
import duckdb
import numpy as np
import pytz

from sportstradamus import data
from sportstradamus.helpers import Archive
from sportstradamus.moneylines import OddsAPIAuthError, get_moneylines, get_props

# Matches Archive's default DB location (resolved relative to the repo cwd).
ARCHIVE_DB = Path("archive/archive.duckdb")
# Per-date resume log: maps a job signature to its completed game-dates so a
# credit-exhausted run can be re-launched and skip what already landed.
PROGRESS_PATH = Path("archive/backfill_progress.json")

# Odds API sport keys for the leagues whose player props we archive.
ODDS_API_SPORT_KEYS = {
    "NFL": "americanfootball_nfl",
    "NBA": "basketball_nba",
    "WNBA": "basketball_wnba",
    "NHL": "icehockey_nhl",
    "MLB": "baseball_mlb",
}


class _CaptureArchive:
    """Stand-in archive for ``--dry-run``: records writes, never persists."""

    def __init__(self):
        self.book_evs = {}

    def merge_player_books(self, league, market, date, player, book_evs, lines, observed_at=None):
        self.book_evs.setdefault(market, {})[(date, player)] = dict(book_evs)

    def set_team_books(self, *args, **kwargs):
        pass

    def write(self, *args, **kwargs):
        pass


def _load_keys_and_props(league, markets=None):
    """Return ``(apikey, props_subset)`` filtered to the requested markets."""
    with open(pkg_resources.files("sportstradamus.creds") / "keys.json") as f:
        apikey = json.load(f)
    with open(pkg_resources.files(data) / "config" / "stat_map.json") as f:
        full = json.load(f)["Odds API"][league]
    if markets is None:
        return apikey, {league: full}
    want = {m.strip() for m in markets.split(",")}
    subset = {key: name for key, name in full.items() if name in want}
    missing = want - set(subset.values())
    if missing:
        raise click.ClickException(
            f"markets not in stat_map[Odds API][{league}]: {sorted(missing)}"
        )
    return apikey, {league: subset}


def _game_dates(league, props, start, end):
    """Distinct archived game-dates for the league in ``[start, end]``."""
    want = [m.strip() for m in props[league].values()]
    con = duckdb.connect(str(ARCHIVE_DB), read_only=True)
    rows = con.execute(
        "SELECT DISTINCT game_date FROM odds "
        "WHERE league=? AND market = ANY(?) AND game_date BETWEEN ? AND ? "
        "ORDER BY game_date",
        [league, want, start, end],
    ).fetchall()
    con.close()
    return [r[0] for r in rows]


def _report_dry_run(capture):
    """Print per-book ev spread so the probe shows if real de-vig returned."""
    for market, entries in capture.book_evs.items():
        spreads = [np.std(list(be.values())) for be in entries.values() if len(be) >= 2]
        if not spreads:
            click.echo(f"  {market}: no multi-book entries fetched")
            continue
        n_real = sum(s > 1e-9 for s in spreads)
        click.echo(
            f"  {market}: {len(spreads)} player-entries, "
            f"mean cross-book std={np.mean(spreads):.3f}, "
            f"{n_real}/{len(spreads)} with real de-vig spread"
        )
        sample = next(iter(entries.items()))
        click.echo(f"    sample {sample[0][1]}: {sample[1]}")


def _job_sig(league, props):
    """Stable key for the resume log: league + the exact market set."""
    return f"{league}|{','.join(sorted(props[league].values()))}"


def _load_done(job_sig):
    """Completed game-dates (ISO strings) for this job from the resume log."""
    if not PROGRESS_PATH.exists():
        return set()
    return set(json.loads(PROGRESS_PATH.read_text()).get(job_sig, []))


def _mark_done(job_sig, iso_date):
    """Append one completed game-date to the resume log (durable per date)."""
    log = json.loads(PROGRESS_PATH.read_text()) if PROGRESS_PATH.exists() else {}
    log.setdefault(job_sig, [])
    if iso_date not in log[job_sig]:
        log[job_sig].append(iso_date)
    PROGRESS_PATH.write_text(json.dumps(log, indent=2))


def _as_of(d, snapshot_hour):
    return pytz.utc.localize(datetime.datetime(d.year, d.month, d.day, snapshot_hour))


def _probe(apikey, props, league, sport_key, dates, snapshot_hour):
    """Dry run: fetch + parse into a capturing stub, print ev spread, no write."""
    archive = _CaptureArchive()
    for d in dates:
        click.echo(f"  fetching {d} (as-of {_as_of(d, snapshot_hour):%Y-%m-%d %H:%MZ})")
        get_props(
            archive, apikey, props, date=_as_of(d, snapshot_hour), sport=league, key=sport_key
        )
    click.echo("DRY RUN — per-market book-ev spread:")
    _report_dry_run(archive)


def _backfill(apikey, props, league, sport_key, dates, snapshot_hour):
    """Fetch each date, flushing + checkpointing per date so a credit-out exit
    is safe and the same command resumes from where it stopped."""
    job_sig = _job_sig(league, props)
    done = _load_done(job_sig)
    remaining = [d for d in dates if d.isoformat() not in done]
    click.echo(f"{league}: {len(done)} done, {len(remaining)} to fetch this run")

    archive = Archive()
    for i, d in enumerate(remaining, 1):
        click.echo(
            f"  [{i}/{len(remaining)}] {d} (as-of {_as_of(d, snapshot_hour):%Y-%m-%d %H:%MZ})"
        )
        try:
            get_moneylines(
                archive, apikey, date=_as_of(d, snapshot_hour), sport=league, key=sport_key
            )
            get_props(
                archive, apikey["odds_api_plus"], props, date=_as_of(d, snapshot_hour), sport=league, key=sport_key
            )
        except OddsAPIAuthError as e:
            click.echo(f"STOP (401): {e}")
            click.echo(
                f"Out of credits/auth at {d}. {len(done)} date(s) saved; partial {d} discarded. "
                "Refill credits and re-run the SAME command to resume."
            )
            return
        archive.write()
        _mark_done(job_sig, d.isoformat())
        done.add(d.isoformat())
    click.echo(f"backfill complete: {len(done)} date(s) total.")


@click.command()
@click.option("--league", default="NFL", help="League whose props to backfill.")
@click.option("--markets", required=False, default=None, help="Comma-separated market names (stat_map values).")
@click.option("--start", required=True, help="First game-date, YYYY-MM-DD.")
@click.option("--end", required=True, help="Last game-date, YYYY-MM-DD.")
@click.option("--snapshot-hour", default=6, help="UTC hour for the as-of API snapshot.")
@click.option("--max-dates", default=0, help="Process at most N game-dates this run (0 = all).")
@click.option("--check-all", is_flag=True, help="Check all dates even if not in archive.")
@click.option("--dry-run", is_flag=True, help="Fetch + parse but do not write (prints ev spread).")
def main(league, markets, start, end, snapshot_hour, max_dates, check_all, dry_run):
    """Re-fetch real historical book EVs for degenerate-seed markets.

    Resumable: completed game-dates are logged to ``PROGRESS_PATH`` after each
    date's write, so a run that hits the Odds API credit limit (HTTP 401) exits
    cleanly and the same invocation picks up where it left off.
    """
    apikey, props = _load_keys_and_props(league, markets)
    sport_key = ODDS_API_SPORT_KEYS[league]
    if check_all:
        dates = [datetime.datetime.strptime(start, "%Y-%m-%d") + datetime.timedelta(days=i) for i in range((datetime.datetime.strptime(end, "%Y-%m-%d") - datetime.datetime.strptime(start, "%Y-%m-%d")).days + 1)]
    else:
        dates = _game_dates(league, props, start, end)
    click.echo(
        f"{league}: {len(dates)} archived game-date(s) {start}..{end}, markets={list(props[league].values())}"
    )

    if dry_run:
        _probe(apikey, props, league, sport_key, dates[: max_dates or len(dates)], snapshot_hour)
        return

    if max_dates:
        job_sig = _job_sig(league, props)
        done = _load_done(job_sig)
        dates = [d for d in dates if d.isoformat() not in done][:max_dates]
    _backfill(apikey, props, league, sport_key, dates, snapshot_hour)


if __name__ == "__main__":
    main()
