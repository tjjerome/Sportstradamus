"""Golden pins for team_assets.json + the theme.team_colors loader.

Resolved current-team code lists per league (verified against
data/leagues/{lg}/teamlog.parquet + gamelog.parquet via
scripts/build_team_assets.py, and against abbreviations.json for MLB, which has
no teamlog/gamelog in this repo). These are NOT a naive recency window over the
raw parquets: NHL's raw teamlog carries a stale code (ARI, superseded by UTA
after the 2024 relocation to Utah) and four Feb-2025 4 Nations Face-Off
international codes (CAN, FIN, SWE, USA) that are not NHL club franchises;
WNBA's raw teamlog carries both PDX and POR for the same Portland franchise
(the historical incident that motivated this loader's safe-fallback contract).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from sportstradamus.dashboard.theme import GRAY, GRAY_SECONDARY, team_colors

_REPO = Path(__file__).resolve().parents[2]
_ASSETS_PATH = _REPO / "src" / "sportstradamus" / "data" / "config" / "team_assets.json"

_LEAGUE_CODES = {
    "NBA": (
        "ATL",
        "BKN",
        "BOS",
        "CHA",
        "CHI",
        "CLE",
        "DAL",
        "DEN",
        "DET",
        "GSW",
        "HOU",
        "IND",
        "LAC",
        "LAL",
        "MEM",
        "MIA",
        "MIL",
        "MIN",
        "NO",
        "NYK",
        "OKC",
        "ORL",
        "PHI",
        "PHX",
        "POR",
        "SAC",
        "SAS",
        "TOR",
        "UTA",
        "WAS",
    ),
    "NFL": (
        "ARI",
        "ATL",
        "BAL",
        "BUF",
        "CAR",
        "CHI",
        "CIN",
        "CLE",
        "DAL",
        "DEN",
        "DET",
        "GB",
        "HOU",
        "IND",
        "JAX",
        "KC",
        "LAC",
        "LAR",
        "LV",
        "MIA",
        "MIN",
        "NE",
        "NO",
        "NYG",
        "NYJ",
        "PHI",
        "PIT",
        "SEA",
        "SF",
        "TB",
        "TEN",
        "WAS",
    ),
    "NHL": (
        "ANA",
        "BOS",
        "BUF",
        "CAR",
        "CBJ",
        "CGY",
        "CHI",
        "COL",
        "DAL",
        "DET",
        "EDM",
        "FLA",
        "LA",
        "MIN",
        "MTL",
        "NJ",
        "NSH",
        "NYI",
        "NYR",
        "OTT",
        "PHI",
        "PIT",
        "SEA",
        "SJ",
        "STL",
        "TB",
        "TOR",
        "UTA",
        "VAN",
        "VGK",
        "WAS",
        "WPG",
    ),
    "MLB": (
        "ARI",
        "ATL",
        "BAL",
        "BOS",
        "CHC",
        "CWS",
        "CIN",
        "CLE",
        "COL",
        "DET",
        "HOU",
        "KC",
        "LAA",
        "LAD",
        "MIA",
        "MIL",
        "MIN",
        "NYM",
        "NYY",
        "ATH",
        "PHI",
        "PIT",
        "SD",
        "SF",
        "SEA",
        "STL",
        "TB",
        "TEX",
        "TOR",
        "WAS",
    ),
    "WNBA": (
        "ATL",
        "CHI",
        "CON",
        "DAL",
        "GSV",
        "IND",
        "LAS",
        "LVA",
        "MIN",
        "NYL",
        "POR",
        "PHX",
        "SEA",
        "TOR",
        "WAS",
    ),
}

_EXPECTED_COUNTS = {"NBA": 30, "NFL": 32, "NHL": 32, "MLB": 30, "WNBA": 15}

_HEX_RE = re.compile(r"^#[0-9A-F]{6}$")


def _raw_assets() -> dict:
    with _ASSETS_PATH.open() as f:
        return json.load(f)


def test_per_league_exact_counts() -> None:
    assets = _raw_assets()
    for league, expected in _EXPECTED_COUNTS.items():
        assert len(assets[league]) == expected, (
            f"{league} has {len(assets[league])} entries, expected {expected}"
        )


def test_every_current_team_resolves_to_a_non_fallback_color() -> None:
    for league, codes in _LEAGUE_CODES.items():
        for code in codes:
            primary, secondary = team_colors(league, code)
            assert (primary, secondary) != (GRAY, GRAY_SECONDARY), (
                f"{league}/{code} resolved to the fallback pair"
            )


def test_all_values_are_rrggbb_hex() -> None:
    assets = _raw_assets()
    for league, teams in assets.items():
        for code, entry in teams.items():
            for key in ("primary", "secondary"):
                value = entry[key]
                assert _HEX_RE.match(value), (
                    f"{league}/{code}/{key} = {value!r} is not #RRGGBB uppercase hex"
                )


def test_fallback_pinned_for_unknown_league_and_unknown_code() -> None:
    assert team_colors("XFL", "ANY") == (GRAY, GRAY_SECONDARY)
    assert team_colors("NBA", "ZZZ") == (GRAY, GRAY_SECONDARY)


def test_team_fills_never_gold() -> None:
    assert "#C9A227" not in _ASSETS_PATH.read_text(), (
        "gold #C9A227 must never appear in team_assets.json"
    )
    assets = _raw_assets()
    for league, teams in assets.items():
        for code, entry in teams.items():
            assert entry["primary"] != "#C9A227", f"{league}/{code} primary is gold"
            assert entry["secondary"] != "#C9A227", f"{league}/{code} secondary is gold"


def test_no_same_league_duplicate_color_pairs() -> None:
    assets = _raw_assets()
    for league, teams in assets.items():
        pairs = [(entry["primary"], entry["secondary"]) for entry in teams.values()]
        assert len(pairs) == len(set(pairs)), (
            f"{league} has duplicate (primary, secondary) pairs across teams"
        )
