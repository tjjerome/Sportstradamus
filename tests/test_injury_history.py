"""Injury history pull: MLB and NHL transaction parsing and the per-season parquet writes."""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import requests
from click.testing import CliRunner

from sportstradamus.collectors import injury_history as ih

# (transaction id, a real feed description, the parsed action, list_days and injury).
_IL_MOVES = [
    (
        825521,
        "Los Angeles Dodgers placed 1B Freddie Freeman on the 10-day injured list retroactive"
        " to March 31, 2025. Right ankle sprain.",
        "placed",
        10,
        "Right ankle sprain.",
    ),
    (
        824102,
        "New York Yankees transferred RHP JT Brubaker from the 15-day injured list to the"
        " 60-day injured list. Left rib fractures.",
        "transferred",
        60,
        "Left rib fractures.",
    ),
    (
        825894,
        "Baltimore Orioles activated SS Gunnar Henderson from the 10-day injured list.",
        "activated",
        10,
        None,
    ),
    (
        831773,
        "Atlanta Braves transferred SS Nacho Alvarez Jr. from the 10-day injured list to the"
        " 60-day injured list.",
        "transferred",
        60,
        None,
    ),
    (
        868235,
        "Los Angeles Angels placed C Logan O'Hoppe on the 7-day injured list. Concussion.",
        "placed",
        7,
        "Concussion.",
    ),
    (
        112433,
        "St. Louis Cardinals placed 2B Skip Schumaker on the 15-day disabled list retroactive"
        " to March 26, 2012. Torn right oblique.",
        "placed",
        15,
        "Torn right oblique.",
    ),
]

# Status changes on other lists, and an activation that names no list at all.
_OTHER_LISTS = {
    824457: "Seattle Mariners placed 2B Jorge Polanco on the paternity list.",
    827825: "Atlanta Braves activated LHP Aaron Bummer from the bereavement list.",
    830102: "Arizona Diamondbacks placed RHP Joe Elbis on the restricted list.",
    824190: "Milwaukee Brewers activated RHP Abner Uribe.",
}

_GARCIA = "San Diego Padres activated RHP Luis García from the 10-day injured list."
_MEARS = "Milwaukee Brewers activated RHP Nick Mears from the 15-day injured list."


def _txn(txn_id: int, description: str, player: int | None) -> dict:
    """One feed transaction; ``player=None`` is a copy that names no one."""
    txn = {
        "id": txn_id,
        "toTeam": {"id": 147, "name": "New York Yankees"},
        "date": "2025-04-08",
        "effectiveDate": "2025-04-08",
        "resolutionDate": "2025-04-08",
        "typeCode": "SC",
        "typeDesc": "Status Change",
        "description": description,
    }
    if player is not None:
        txn["person"] = {"id": player, "fullName": "Feed Player"}
    return txn


_FEED = {
    "transactions": [
        *(_txn(txn_id, text, player=txn_id + 1) for txn_id, text, *_ in _IL_MOVES),
        *(_txn(txn_id, text, player=txn_id + 1) for txn_id, text in _OTHER_LISTS.items()),
        # The 2022 feed's shape: one transaction repeated, only one copy naming the player.
        _txn(615681, _GARCIA, player=None),
        _txn(615681, _GARCIA, player=472610),
        _txn(615681, _GARCIA, player=None),
        # And one move entered twice under two ids.
        _txn(826901, _MEARS, player=683232),
        _txn(826898, _MEARS, player=683232),
    ]
}

_NFL_REPORT = pd.DataFrame(
    {
        "season": [2025, 2025],
        "week": [1, 1],
        "team": ["ARI", "ARI"],
        "gsis_id": ["00-0039521", "00-0037141"],
        "report_status": ["Out", "Questionable"],
        "practice_status": ["Did Not Participate In Practice", "Limited Participation in Practice"],
    }
)

_IR, _LTIR = "injured reserve", "long-term injured reserve"

# Real descriptions from ESPN's 2023 and 2026 NHL transaction replies, each with the moves it
# parses to as (action, list, position, player_name).
_NHL_MOVES = [
    (
        "Recalled D Henry Thrun from Manitoba (AHL). Placed D Dylan DeMelo on injured reserve.",
        [("Placed", _IR, "D", "Dylan DeMelo")],
    ),
    # A match from "Placed" must stop at the sentence end, not run on to "injured reserve".
    (
        "Placed F Chase Stillman on Waivers. Activated D Jacob Bernard-Docker from injured reserve.",
        [("Activated", _IR, "D", "Jacob Bernard-Docker")],
    ),
    (
        "Reinstated D Cam York from injured reserve. Placed D Rasmus Ristolainen on injured"
        " reserve, retroactive to Oct. 24. Recalled F Jacob Gaucher from Lehigh Valley (AHL).",
        [("Reinstated", _IR, "D", "Cam York"), ("Placed", _IR, "D", "Rasmus Ristolainen")],
    ),
    (
        "Recalled D Dennis Gilbert from Rochester (AHL). Placed G Alex Lyon and D Olen Zellweger"
        " on injured reserve. Signed D Louis Crevier to a four-year contract. Transferred F"
        " Matthew Savoie to long-term injured reserve.",
        [
            ("Placed", _IR, "G", "Alex Lyon"),
            ("Placed", _IR, "D", "Olen Zellweger"),
            ("Transferred", _LTIR, "F", "Matthew Savoie"),
        ],
    ),
    (
        "Reassigned C Roger McQueen and C Cruz Lucius to San Diego (AHL). Recalled D Noah Warren,"
        " D Ian Mitchell and D Tristan Luneau from San Diego. Placed D Drew Helleson on injured"
        " reserve. Moved RW Troy Terry from the non-roster list to long-term injured reserve.",
        [("Placed", _IR, "D", "Drew Helleson"), ("Moved", _LTIR, "RW", "Troy Terry")],
    ),
    (
        "Assigned C Zach Aston-Reese to Grand Rapids (AHL). Reassigned F Jonatan Berggren to Grand"
        " Rapids. Activated C J.T. Compher and LW Klim Kostin from injured reserve.",
        [("Activated", _IR, "C", "J.T. Compher"), ("Activated", _IR, "LW", "Klim Kostin")],
    ),
    (
        "Recalled F Rutger McGroarty from Wilkes-Barre/Scranton (AHL). Placed D Jack St. Ivany on"
        " injured reserve.",
        [("Placed", _IR, "D", "Jack St. Ivany")],
    ),
    ("Activated from injured reserve.", []),
    (
        "Reinstated Fs Owen Tippett and Travis Konecny from injured reserve.",
        [("Reinstated", _IR, "F", "Owen Tippett"), ("Reinstated", _IR, "F", "Travis Konecny")],
    ),
    (
        "Placed C Ben Kindel, C Blake Lizotte, RW Justin Brazeau and D Ryan Graves on injured"
        " reserve.",
        [
            ("Placed", _IR, "C", "Ben Kindel"),
            ("Placed", _IR, "C", "Blake Lizotte"),
            ("Placed", _IR, "RW", "Justin Brazeau"),
            ("Placed", _IR, "D", "Ryan Graves"),
        ],
    ),
    (
        "Placed D Rasmus Sandin on long-term injured reserve and D Matt Roy on injured reserve."
        " Recalled D Justin Holl from Hershey (AHL).",
        [("Placed", _LTIR, "D", "Rasmus Sandin"), ("Placed", _IR, "D", "Matt Roy")],
    ),
    (
        "Reinstated D Jacob Bryson from injured reserve and placed G Ukko-Pekka Luukkonen on"
        " injured reserve.",
        [("Reinstated", _IR, "D", "Jacob Bryson"), ("Placed", _IR, "G", "Ukko-Pekka Luukkonen")],
    ),
    (
        "Recalled Conor Geekie from Syracuse (AHL). Placed Dominic James on injured reserve.",
        [("Placed", _IR, None, "Dominic James")],
    ),
    (
        "Recalled C Brayden Yager from Manitoba (AHL). Placed D Josh Morrissey on injured reserve"
        " and RW Walker Duehr on waivers.",
        [("Placed", _IR, "D", "Josh Morrissey")],
    ),
    (
        "Activated LW Dylan Holloway off long-term injured reserve and reassigned him to"
        " Bakersfield (AHL).",
        [("Activated", _LTIR, "LW", "Dylan Holloway")],
    ),
    (
        "Recalled D Louis Crevier from Rockford (AHL). Reinstated D Jarred Tinordi to the active"
        " roster from injured reserve. Assigned Ds Wyatt Kaiser and Isaak Phillips to Rockford.",
        [("Reinstated", _IR, "D", "Jarred Tinordi")],
    ),
    (
        "Activated RW/C Mackie Samoskevich from injured reserve. Placed RW Jacob Melanson on"
        " injured reserve.",
        [("Activated", _IR, "RW/C", "Mackie Samoskevich"), ("Placed", _IR, "RW", "Jacob Melanson")],
    ),
    (
        "Activated D Matt Benning from injured reserve and C Nico Sturm from non-roster injured"
        " reserve.",
        [
            ("Activated", _IR, "D", "Matt Benning"),
            ("Activated", "non-roster injured reserve", "C", "Nico Sturm"),
        ],
    ),
]


def _espn_page(index: int, stamp: str, team: str, moves: list) -> dict:
    """One page of ESPN's two-page reply, every transaction stamped ``stamp`` for ``team``."""
    transactions = [
        {"date": stamp, "description": text, "team": {"abbreviation": team}} for text, _ in moves
    ]
    return {"pageIndex": index, "pageCount": 2, "transactions": transactions}


_HALF = len(_NHL_MOVES) // 2
_ESPN_PAGES = [
    _espn_page(1, "2026-11-12T08:00Z", "WPG", _NHL_MOVES[:_HALF]),
    _espn_page(2, "2026-10-12T07:00Z", "PIT", _NHL_MOVES[_HALF:]),
]

# The row count the game_absences stub reports holding.
_ABSENCE_ROWS = 812


def _stub_sources(monkeypatch, tmp_path) -> dict[str, list]:
    """Serve the fixtures, stub game_absences and write under ``tmp_path``; return the calls."""
    requested: dict[str, list] = {"MLB": [], "NFL": [], "ESPN": [], "absences": []}

    def fake_get(url, **_):
        if url.startswith("https://site.api.espn.com/"):
            requested["ESPN"].append(url)
            reply = _ESPN_PAGES[int(url.rsplit("page=", 1)[1]) - 1]
        else:
            requested["MLB"].append(url)
            reply = _FEED
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(reply).encode()
        return response

    def fake_import_injuries(years):
        requested["NFL"].extend(years)
        return _NFL_REPORT.copy()

    def fake_update_season(league, season, path):
        requested["absences"].append((league, season, path))
        return _ABSENCE_ROWS

    monkeypatch.setattr(ih, "PLAYER_DATA_BASE", tmp_path)
    monkeypatch.setattr(ih.requests, "get", fake_get)
    monkeypatch.setattr(ih.nfl, "import_injuries", fake_import_injuries)
    monkeypatch.setattr(ih.game_absences, "update_season", fake_update_season)
    return requested


def _pull(monkeypatch, tmp_path, args: list[str]):
    """Run the command against the stubs; return its result and the requests it made."""
    requested = _stub_sources(monkeypatch, tmp_path)
    result = CliRunner().invoke(ih.injury_history, args)
    assert result.exit_code == 0, result.output
    return result, requested


def _mlb_2025(monkeypatch, tmp_path) -> pd.DataFrame:
    _pull(monkeypatch, tmp_path, ["--league", "MLB", "--season", "2025"])
    return pd.read_parquet(tmp_path / "MLB" / "injuries" / "2025.parquet")


def test_injured_list_moves_parse_to_action_list_length_and_injury(monkeypatch, tmp_path):
    moves = _mlb_2025(monkeypatch, tmp_path)

    expected = pd.DataFrame(
        [(txn_id, *parsed) for txn_id, _, *parsed in _IL_MOVES],
        columns=["id", "action", "list_days", "injury"],
    ).astype({"list_days": "Int64"})
    parsed = moves[moves["id"].isin(expected["id"])][expected.columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(parsed, expected)


def test_other_lists_and_bare_activations_drop_and_repeats_collapse(monkeypatch, tmp_path):
    moves = _mlb_2025(monkeypatch, tmp_path)

    assert sorted(moves["id"]) == sorted([txn_id for txn_id, *_ in _IL_MOVES] + [615681, 826901])
    assert moves.loc[moves["id"] == 615681, "player_id"].tolist() == [472610]


def test_command_writes_one_parquet_per_league_season(monkeypatch, tmp_path):
    result, requested = _pull(
        monkeypatch, tmp_path, ["--league", "mlb", "--league", "NFL", "--season", "2025"]
    )

    assert requested["MLB"] == [ih._MLB_TRANSACTIONS_URL.format(season=2025)]
    assert requested["NFL"] == [2025]
    assert requested["ESPN"] == requested["absences"] == []
    mlb = pd.read_parquet(tmp_path / "MLB" / "injuries" / "2025.parquet")
    assert (mlb.dtypes[["player_id", "team_id", "list_days"]] == "Int64").all()
    nfl = pd.read_parquet(tmp_path / "NFL" / "injuries" / "2025.parquet")
    pd.testing.assert_frame_equal(nfl, _NFL_REPORT)
    assert result.stdout.strip() == f"injury-history: MLB/2025={len(_IL_MOVES) + 2} NFL/2025=2"


def test_per_game_leagues_resume_in_game_absences_and_nhl_adds_injured_reserve(
    monkeypatch, tmp_path
):
    leagues = ("NBA", "WNBA", "NHL")
    result, requested = _pull(
        monkeypatch,
        tmp_path,
        ["--league", "nba", "--league", "WNBA", "--league", "NHL", "--season", "2025"],
    )

    assert requested["absences"] == [
        (league, 2025, tmp_path / league / "injuries" / "2025.parquet") for league in leagues
    ]
    # The stub writes nothing, so any file there would be the command overwriting a resumed pull.
    assert not any((tmp_path / league / "injuries").exists() for league in leagues)
    assert requested["ESPN"] == [ih._NHL_TRANSACTIONS_URL.format(year=2025, page=p) for p in (1, 2)]
    assert requested["MLB"] == requested["NFL"] == []
    ir_rows = sum(len(moves) for _, moves in _NHL_MOVES)
    assert result.stdout.strip() == (
        f"injury-history: NBA/2025={_ABSENCE_ROWS} WNBA/2025={_ABSENCE_ROWS}"
        f" NHL/2025={_ABSENCE_ROWS} NHL-IR/2025={ir_rows}"
    )


def _nhl_2026(monkeypatch, tmp_path) -> tuple[pd.DataFrame, dict[str, list]]:
    _, requested = _pull(monkeypatch, tmp_path, ["--league", "NHL", "--season", "2026"])
    return pd.read_parquet(tmp_path / "NHL" / "injured_reserve" / "2026.parquet"), requested


def test_injured_reserve_clauses_parse_to_one_row_per_player(monkeypatch, tmp_path):
    moves, _ = _nhl_2026(monkeypatch, tmp_path)

    expected = pd.DataFrame(
        [(text, *move) for text, parsed in _NHL_MOVES for move in parsed],
        columns=["description", "action", "list", "position", "player_name"],
    ).astype("string")
    pd.testing.assert_frame_equal(moves[expected.columns], expected)


def test_injured_reserve_reads_every_page_into_day_stamped_rows(monkeypatch, tmp_path):
    moves, requested = _nhl_2026(monkeypatch, tmp_path)

    assert requested["ESPN"] == [ih._NHL_TRANSACTIONS_URL.format(year=2026, page=p) for p in (1, 2)]
    assert list(moves.dtypes.items()) == list(ih._IR_DTYPES.items())
    stamps = moves[["date", "team"]].drop_duplicates().values.tolist()
    assert stamps == [[pd.Timestamp("2026-11-12"), "WPG"], [pd.Timestamp("2026-10-12"), "PIT"]]


def test_default_seasons_run_through_each_leagues_current_season(monkeypatch, tmp_path):
    class _March2024(date):
        @classmethod
        def today(cls):
            return cls(2024, 3, 1)

    monkeypatch.setattr(ih, "date", _March2024)
    # The NFL's 2024 season has not kicked off in March, so nflverse has no 2024 file yet.
    monkeypatch.setattr(ih.nflr, "get_current_season", lambda: 2023)

    _, requested = _pull(monkeypatch, tmp_path, [])

    seasons = (2022, 2023, 2024)
    assert requested["NFL"] == [2022, 2023]
    assert requested["MLB"] == [ih._MLB_TRANSACTIONS_URL.format(season=s) for s in seasons]
    assert requested["absences"] == [
        (league, s, tmp_path / league / "injuries" / f"{s}.parquet")
        for league in ("NBA", "WNBA", "NHL")
        for s in seasons
    ]
    assert requested["ESPN"] == [
        ih._NHL_TRANSACTIONS_URL.format(year=s, page=p) for s in seasons for p in (1, 2)
    ]
