"""Injury history pull: MLB injured-list parsing and the per-season parquet writes."""

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


def _stub_sources(monkeypatch, tmp_path) -> dict[str, list]:
    """Serve ``_FEED`` and ``_NFL_REPORT`` and write under ``tmp_path``; return the requests."""
    requested: dict[str, list] = {"MLB": [], "NFL": []}

    def fake_get(url, **_):
        requested["MLB"].append(url)
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(_FEED).encode()
        return response

    def fake_import_injuries(years):
        requested["NFL"].extend(years)
        return _NFL_REPORT.copy()

    monkeypatch.setattr(ih, "PLAYER_DATA_BASE", tmp_path)
    monkeypatch.setattr(ih.requests, "get", fake_get)
    monkeypatch.setattr(ih.nfl, "import_injuries", fake_import_injuries)
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
    mlb = pd.read_parquet(tmp_path / "MLB" / "injuries" / "2025.parquet")
    assert (mlb.dtypes[["player_id", "team_id", "list_days"]] == "Int64").all()
    nfl = pd.read_parquet(tmp_path / "NFL" / "injuries" / "2025.parquet")
    pd.testing.assert_frame_equal(nfl, _NFL_REPORT)
    assert result.stdout.strip() == f"injury-history: MLB/2025={len(_IL_MOVES) + 2} NFL/2025=2"


def test_default_seasons_run_through_each_leagues_current_season(monkeypatch, tmp_path):
    class _March2024(date):
        @classmethod
        def today(cls):
            return cls(2024, 3, 1)

    monkeypatch.setattr(ih, "date", _March2024)
    # The NFL's 2024 season has not kicked off in March, so nflverse has no 2024 file yet.
    monkeypatch.setattr(ih.nflr, "get_current_season", lambda: 2023)

    _, requested = _pull(monkeypatch, tmp_path, [])

    assert requested["NFL"] == [2022, 2023]
    assert requested["MLB"] == [
        ih._MLB_TRANSACTIONS_URL.format(season=s) for s in (2022, 2023, 2024)
    ]
