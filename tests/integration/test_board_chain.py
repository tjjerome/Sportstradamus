"""One board from the entry search to the dashboard's pages, through prophecize's own writers.

``test_end_to_end`` hands ``prophecize`` stand-in frames and intercepts its writes, so nothing
there passes what the entry search really returns to the snapshot writers, the history upserts
or the pages that read them. Here only the scrape and the model's scoring are stand-ins: the
scored records take the real EV stamp, entry search and story menu, the run writes its
snapshots and histories to scratch files, and the dashboard renders those files.
"""

from __future__ import annotations

import datetime
import shutil
from itertools import permutations
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from click.testing import CliRunner
from streamlit.testing.v1 import AppTest

from sportstradamus.dashboard import data as dashboard_data
from sportstradamus.helpers import UNDERDOG_BOOST_BASELINE, io
from sportstradamus.prediction import cli as prediction_cli
from sportstradamus.prediction import ev_stamp, persist, scoring
from sportstradamus.prediction.offer_records import SCORED_RECORD_COLS
from sportstradamus.stats.wnba import StatsWNBA
from tests.golden.test_correlation_helpers import _stats_for, _stub_corr_sources
from tests.golden.test_dashboard_render_smoke import _APP, _hero_title
from tests.golden.test_ev_stamp import _model_led_fit
from tests.integration.test_end_to_end import (
    _stub_prophecize_inputs,
    _stub_stats_loaders,
    _synthetic_offers,
)

pytestmark = pytest.mark.integration

# Every correlation key the board's legs resolve to: a guard, forward or center ranked first or
# second on his team, on either side of the game, all on points.
_TOKENS = [f"{side}{pos}{rank}.PTS" for side in ("", "_OPP_") for pos in "GFC" for rank in (1, 2)]

# The model's chance on each leg, in the stand-in board's order (LVA's four players, NYL's
# three, SEA's three, PHX's two). The fit prices a leg at it, so each team's first player is
# its pick, and no two entries price alike.
_WIN_PROBS = [0.58, 0.53, 0.52, 0.51, 0.57, 0.505, 0.5, 0.56, 0.515, 0.495, 0.55, 0.49]


def _scored_records(game_date: str) -> list[dict]:
    """Twelve WNBA legs over two games, as ``model_prob`` hands a platform's board to the stamp.

    Each carries Underdog's own label for the market and its even multiplier, and the position
    code the search ranks by usage; ``Game`` and the depth label are the search's to add.
    """
    scored = _synthetic_offers().reindex(columns=SCORED_RECORD_COLS)
    scored["Date"] = game_date
    scored["Market"] = "Points"
    scored["Player position"] = [slot % 3 + 1 for slot in range(len(scored))]
    scored[["Boost", "Boost_Over", "Boost_Under"]] = UNDERDOG_BOOST_BASELINE
    scored["Win Prob"] = _WIN_PROBS
    scored["Model Version"] = "v1"
    return scored.to_dict("records")


def test_a_scored_board_reaches_the_dashboard_through_prophecizes_own_writers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixtures_dir: Path, reset_archive_singleton
) -> None:
    repo_root = Path.cwd()
    (tmp_path / "archive").mkdir()
    shutil.copy(fixtures_dir / "legacy_archive.duckdb", tmp_path / "archive" / "archive.duckdb")
    monkeypatch.chdir(tmp_path)

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    # The run writes each snapshot where the dashboard reads it, under its real file name.
    for module in (persist, dashboard_data):
        for name in [n for n in vars(module) if n.startswith("CURRENT_") and n.endswith("_PATH")]:
            monkeypatch.setattr(module, name, runtime / getattr(io, name).name)
    for name in ("HISTORY_PATH", "PARLAY_HIST_PATH", "PARLAY_HIST_DIR"):
        monkeypatch.setattr(io, name, runtime / getattr(io, name).name)
    monkeypatch.setattr(ev_stamp, "LIVE_FIT_PATH", runtime / "live_fit.parquet")
    _model_led_fit(runtime / "live_fit.parquet", league="WNBA")

    game_date = datetime.date.today().isoformat()
    records = _scored_records(game_date)
    _stub_prophecize_inputs(monkeypatch)
    _stub_stats_loaders(monkeypatch)
    monkeypatch.setattr(
        prediction_cli,
        "get_ud",
        lambda capture_dir=None, scored_at=None: {"WNBA": {"Points": records}},
    )
    monkeypatch.setattr(scoring, "_match_league_offers", lambda *league_pass: records)
    # The search ranks a team's players by the usage profile profile_market builds.
    usage = _stats_for([leg["Player"] for leg in records], "MIN short", "USG_PCT short")
    monkeypatch.setattr(
        StatsWNBA,
        "profile_market",
        lambda self, _stat: setattr(self, "playerProfile", usage.playerProfile),
    )
    _stub_corr_sources(monkeypatch, dict.fromkeys(permutations(_TOKENS, 2), 0.2))

    result = CliRunner().invoke(prediction_cli.main, [], catch_exceptions=False)
    assert result.exit_code == 0, result.output

    games = {"LVA/NYL", "PHX/SEA"}
    offers = pd.read_parquet(runtime / "current_offers.parquet")
    assert len(offers) == len(records)
    assert set(offers["Game"]) == games
    # The board names the market, and the stamp picked each team's leg the fit likes most.
    assert offers["Market"].eq("PTS").all()
    assert set(offers.loc[offers["Pick"], "Team"]) == {"LVA", "NYL", "SEA", "PHX"}

    parlays = pd.read_parquet(runtime / "current_parlays.parquet")
    assert set(parlays["Game"]) == games
    assert parlays["EV"].notna().all()
    stories = pd.read_parquet(runtime / "current_game_stories.parquet")
    assert set(stories["Game"]) == games

    # Both histories took the run's rows: every leg, and every entry on its day's partition.
    history = pd.read_parquet(runtime / "history.parquet")
    assert len(history) == len(records)
    assert history["Pick"].sum() == offers["Pick"].sum()
    assert len(io.read_parlay_hist()) == len(parlays)

    # The pages resolve nothing against the scratch directory the run's archive lives in.
    monkeypatch.chdir(repo_root)
    st.cache_data.clear()
    at = AppTest.from_file(str(_APP), default_timeout=30)
    at.run()
    assert not at.exception
    assert _hero_title(at) == "Tonight"
    cards = [m.body for m in at.markdown if m.body.startswith('<div class="tonight-card')]
    assert len(cards) == len(games)

    at.switch_page("surfaces/board.py")
    at.run()
    assert not at.exception
    assert _hero_title(at) == "Today's Predictions"

    at.switch_page("surfaces/games.py")
    at.run()
    assert not at.exception
    assert at.selectbox(key="game_select").options == [
        f"LVA vs NYL · {game_date}",
        f"SEA vs PHX · {game_date}",
    ]
