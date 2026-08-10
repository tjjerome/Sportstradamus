"""Characterization test for the shared NBA/WNBA upcoming-games schedule scrape.

`fetch_upcoming_games` was extracted from a per-league block that lived
near-byte-identical in StatsNBA.update and StatsWNBA.update (differing only in
LeagueID 00 vs 10 and the season value). This test pins it against a verbatim
reference copy of that block across a recorded internationalbroadcasterschedule
response, the dedup precedence (first date seen wins), and the endpoint-failure
path (``Scrape.get`` returns ``{}`` on exhausted retries, so indexing raises
``KeyError`` and the loop yields no games). The reference keeps the old broad
``except Exception`` so the narrowed ``(KeyError, IndexError, TypeError)`` catch
in the extracted helper is proven equivalent on the realistic failure shape.
"""

from datetime import date, timedelta
from urllib.parse import parse_qs, urlparse

import pytest

from sportstradamus.stats import base
from sportstradamus.stats.base import fetch_upcoming_games

_TODAY = date(2024, 12, 1)


def _payload(games):
    return {"resultSets": [{}, {"CompleteGameList": games}]}


def _old_inline(league_id, season, today):
    """Verbatim copy of the pre-consolidation NBA/WNBA upcoming-games block."""
    upcoming_games = {}
    try:
        ug_url = f"https://stats.nba.com/stats/internationalbroadcasterschedule?LeagueID={league_id}&Season={season}&RegionID=1&Date={today.strftime('%m/%d/%Y')}&EST=Y"

        ug_res = base.scraper.get(ug_url)["resultSets"][1]["CompleteGameList"]

        next_day = today + timedelta(days=1)
        ug_url = f"https://stats.nba.com/stats/internationalbroadcasterschedule?LeagueID={league_id}&Season={season}&RegionID=1&Date={next_day.strftime('%m/%d/%Y')}&EST=Y"

        ug_res.extend(base.scraper.get(ug_url)["resultSets"][1]["CompleteGameList"])

        next_day = next_day + timedelta(days=1)
        ug_url = f"https://stats.nba.com/stats/internationalbroadcasterschedule?LeagueID={league_id}&Season={season}&RegionID=1&Date={next_day.strftime('%m/%d/%Y')}&EST=Y"

        ug_res.extend(base.scraper.get(ug_url)["resultSets"][1]["CompleteGameList"])

        for game in ug_res:
            if game["vtAbbreviation"] not in upcoming_games:
                upcoming_games[game["vtAbbreviation"]] = {
                    "Opponent": game["htAbbreviation"],
                    "Home": False,
                }
            if game["htAbbreviation"] not in upcoming_games:
                upcoming_games[game["htAbbreviation"]] = {
                    "Opponent": game["vtAbbreviation"],
                    "Home": True,
                }

    except Exception:
        pass
    return upcoming_games


def _install_scraper(monkeypatch, payloads_by_date):
    """Stateless url->payload stub so the helper and the reference agree."""

    def _get(url, *args, **kwargs):
        date_str = parse_qs(urlparse(url).query)["Date"][0]
        return payloads_by_date.get(date_str, {})

    monkeypatch.setattr(base.scraper, "get", _get)


# BOS appears on date0 as away and date1 as home: first-seen (date0) must win.
_RECORDED = {
    "12/01/2024": _payload([{"vtAbbreviation": "BOS", "htAbbreviation": "NYK"}]),
    "12/02/2024": _payload([{"vtAbbreviation": "LAL", "htAbbreviation": "BOS"}]),
    "12/03/2024": _payload([{"vtAbbreviation": "MIA", "htAbbreviation": "LAL"}]),
}

_EXPECTED = {
    "BOS": {"Opponent": "NYK", "Home": False},
    "NYK": {"Opponent": "BOS", "Home": True},
    "LAL": {"Opponent": "BOS", "Home": False},
    "MIA": {"Opponent": "LAL", "Home": False},
}


@pytest.mark.parametrize(
    ("league_id", "season"),
    [("00", "2024"), ("10", 2024)],  # NBA passes a str slice, WNBA an int year
)
def test_matches_reference_and_expected(monkeypatch, league_id, season):
    _install_scraper(monkeypatch, _RECORDED)
    result = fetch_upcoming_games(league_id, season, _TODAY)
    assert result == _EXPECTED
    assert result == _old_inline(league_id, season, _TODAY)


def test_endpoint_failure_yields_empty(monkeypatch):
    # Scrape.get returns {} after exhausting retries -> ["resultSets"] KeyErrors.
    _install_scraper(monkeypatch, {})
    assert fetch_upcoming_games("00", "2024", _TODAY) == {}
    assert fetch_upcoming_games("00", "2024", _TODAY) == _old_inline("00", "2024", _TODAY)


def test_partial_fetch_failure_yields_empty(monkeypatch):
    # First date succeeds, second returns {}: mid-fetch KeyError aborts before the
    # game loop, so no games are recorded -- matching the old single-try block.
    partial = {"12/01/2024": _RECORDED["12/01/2024"]}
    _install_scraper(monkeypatch, partial)
    assert fetch_upcoming_games("00", "2024", _TODAY) == {}
    assert fetch_upcoming_games("00", "2024", _TODAY) == _old_inline("00", "2024", _TODAY)


def test_empty_game_list_yields_empty(monkeypatch):
    _install_scraper(monkeypatch, dict.fromkeys(_RECORDED, _payload([])))
    assert fetch_upcoming_games("00", "2024", _TODAY) == {}
