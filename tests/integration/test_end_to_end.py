"""End-to-end smoke test for the ``confer -> meditate -> prophecize`` flow.

The goal is *not* to validate model quality or reproduce a live run end to
end; it is to confirm that the three CLIs still wire up to one another
and that none of the orchestration code has been broken by a refactor.

Two modes, controlled by the ``SPORTSTRADAMUS_INTEGRATION_REAL_APIS``
environment variable:

* **Fake (default).** The Odds API, ``nba_api``, Underdog, and Sleeper are
  all replaced with stubs / canned fixtures. Runs in well under 90 seconds.
* **Real.** Set ``SPORTSTRADAMUS_INTEGRATION_REAL_APIS=1`` to opt in to
  live network calls (``confer`` still goes through ``--fixture-dir``,
  but ``meditate`` and ``prophecize`` get real ``StatsWNBA`` data).
  Allowed to take longer.

The test never writes data: every disk-write touchpoint
(``Archive.write``, model pickle writes, history files) is intercepted.
We exercise import paths and callback wiring only.

Marked ``integration`` so the default ``pytest`` collection skips it; opt
in with ``pytest -m integration``.
"""

from __future__ import annotations

import datetime
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

_REAL_APIS = os.environ.get("SPORTSTRADAMUS_INTEGRATION_REAL_APIS") == "1"


@pytest.mark.integration
def test_pipeline_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixtures_dir: Path,
    reset_archive_singleton,
    preserve_data_files,
) -> None:
    """Touch every stage of the pipeline; assert the wiring is intact.

    Per phase we invoke the CLI, intercept any disk write, and verify the
    orchestration callbacks fired with sensible payloads. ``confer`` is
    the one phase whose production code path runs unmodified — its
    ``--fixture-dir`` flag is the only fixture-mode hook in production.
    """
    # ----- shared scaffolding -----
    (tmp_path / "archive").mkdir(parents=True)
    # Seed a pre-sample_ts archive so Archive.__init__ exercises the
    # ALTER TABLE migration path on top of an existing on-disk DB. A
    # greenfield CREATE TABLE IF NOT EXISTS hides schema-migration bugs.
    shutil.copy(
        fixtures_dir / "legacy_archive.duckdb",
        tmp_path / "archive" / "archive.duckdb",
    )
    monkeypatch.chdir(tmp_path)

    from sportstradamus.helpers.archive import Archive

    runner = CliRunner()

    # ----- Phase 1: confer (REAL flag path, fixture-fed) -----
    from sportstradamus.moneylines import confer

    result = runner.invoke(
        confer,
        ["--fixture-dir", str(fixtures_dir / "odds_api")],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, f"confer failed: {result.output}"

    archive_obj = Archive()
    pts_df = archive_obj.to_pandas("WNBA", "PTS")
    book_cols = [c for c in pts_df.columns if c != "Line"]
    offers_with_ev = int(pts_df[book_cols].notna().any(axis=1).sum()) if not pts_df.empty else 0
    assert offers_with_ev >= 10, (
        f"confer wrote EV for only {offers_with_ev} player-prop offers; "
        f"expected >= 10. archive contents: {pts_df!r}"
    )

    # Re-run confer to confirm the archive append-only property: a second
    # poll should add new ``observed_at`` rows rather than overwrite the
    # first poll's rows.
    first_poll_rows = int(
        archive_obj._connection.execute("SELECT COUNT(*) FROM odds").fetchone()[0]
    )
    result = runner.invoke(
        confer,
        ["--fixture-dir", str(fixtures_dir / "odds_api")],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, f"second confer failed: {result.output}"
    second_poll_rows = int(
        archive_obj._connection.execute("SELECT COUNT(*) FROM odds").fetchone()[0]
    )
    assert second_poll_rows > first_poll_rows, (
        f"second confer poll did not append new rows: {first_poll_rows} -> {second_poll_rows}"
    )

    sample_player = next(iter(pts_df.index))[1] if not pts_df.empty else None
    if sample_player is not None:
        history = archive_obj.get_ev_history("WNBA", "PTS", "2026-05-08", sample_player)
        if not history.empty:
            assert history["observed_at"].is_monotonic_increasing, (
                "get_ev_history must return rows ordered by observed_at"
            )

    # ----- Phase 2: meditate (CLI invoked; ML stubbed; no writes) -----
    from sportstradamus.training import cli as training_cli
    from sportstradamus.training import markets as markets_module

    # Restrict to one market; skip the extended per-market loop.
    monkeypatch.setattr(markets_module, "ALL_MARKETS", {"WNBA": ["PTS"]})
    monkeypatch.setattr(training_cli, "ALL_MARKETS", {"WNBA": ["PTS"]})

    # Gate-driven meditate skips cells withheld in stat_meta; an empty
    # ship-config makes every cell active (mirrors --deterministic) so this
    # smoke check is robust to which cells happen to be withheld on the
    # committed branch.
    monkeypatch.setattr(training_cli, "load_ship_config", lambda *a, **kw: {})

    if not _REAL_APIS:
        _stub_stats_loaders(monkeypatch)

    monkeypatch.setattr(training_cli, "fit_book_weights", lambda *a, **kw: {})
    monkeypatch.setattr(training_cli, "correlate", lambda *a, **kw: None)

    train_market_calls: list[tuple[str, str]] = []

    def stub_train_market(league, market, *args, **kwargs):
        # No pickle write — this is a smoke test; we only verify the CLI
        # reached the per-market call site with the right league/market pair.
        train_market_calls.append((league, market))

    monkeypatch.setattr(training_cli, "train_market", stub_train_market)

    # ``meditate`` rewrites ``data/book_weights.json`` mid-run; the
    # ``preserve_data_files`` fixture restores the original bytes on
    # teardown so the test leaves no on-disk side effects.

    from sportstradamus.training.cli import meditate

    result = runner.invoke(meditate, ["--league", "WNBA"], catch_exceptions=False)
    assert result.exit_code == 0, f"meditate failed: {result.output}"
    assert ("WNBA", "PTS") in train_market_calls, (
        f"train_market was not invoked for WNBA:PTS. calls={train_market_calls}"
    )

    # ----- Phase 3: prophecize (CLI invoked; parquet snapshot + scrapers mocked) -----
    from sportstradamus.prediction import cli as prediction_cli

    monkeypatch.setattr(prediction_cli, "get_ud", dict)
    monkeypatch.setattr(prediction_cli, "get_sleeper", dict)

    captured: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    def stub_process_offers(offer_dict, book, stats, **kwargs):
        offers = _synthetic_offers()
        parlays = _synthetic_parlays(book)
        captured[book] = (offers, parlays)
        return offers, parlays

    monkeypatch.setattr(prediction_cli, "process_offers", stub_process_offers)

    snapshot_calls: list[dict] = []

    def stub_write_current_offers(
        offers, parlays, leagues, platforms, contest_variant="power", stats_dict=None
    ):
        snapshot_calls.append(
            {
                "offers": offers,
                "parlays": parlays,
                "leagues": list(leagues),
                "platforms": list(platforms),
                "contest_variant": contest_variant,
            }
        )

    monkeypatch.setattr(prediction_cli, "write_current_offers", stub_write_current_offers)

    game_corr_calls: list = []
    monkeypatch.setattr(prediction_cli, "write_current_game_corr", game_corr_calls.append)

    game_context_calls: list = []
    monkeypatch.setattr(prediction_cli, "write_current_game_context", game_context_calls.append)

    game_stories_calls: list = []
    monkeypatch.setattr(prediction_cli, "write_current_game_stories", game_stories_calls.append)

    # Skip writing prediction history to data/history.dat.
    def _noop_write(_df):
        return None

    def _empty_df():
        return pd.DataFrame()

    monkeypatch.setattr(prediction_cli, "write_history", _noop_write)
    monkeypatch.setattr(prediction_cli, "write_parlay_hist", _noop_write)
    monkeypatch.setattr(prediction_cli, "read_history", _empty_df)
    monkeypatch.setattr(prediction_cli, "read_parlay_hist", _empty_df)

    if not _REAL_APIS:
        _stub_stats_loaders(monkeypatch)

    from sportstradamus.prediction.cli import main as prophecize_main

    result = runner.invoke(prophecize_main, [], catch_exceptions=False)
    assert result.exit_code == 0, f"prophecize failed: {result.output}"

    # The parquet snapshot writer was reached but no real disk write fired.
    assert snapshot_calls, "write_current_offers was never invoked"

    # The orchestration produced offers with EV and at least one parlay candidate.
    assert captured, "process_offers was never invoked"
    underdog_offers, _ = captured.get("Underdog", (pd.DataFrame(), pd.DataFrame()))
    assert len(underdog_offers) >= 10, f"expected >= 10 offers with EV, got {len(underdog_offers)}"
    assert underdog_offers["Projection"].notna().sum() >= 10, (
        "fewer than 10 offers had a populated Projection column"
    )
    parlay_total = sum(len(p) for _, p in captured.values())
    assert parlay_total >= 1, "no parlay candidates were returned"

    # The P2 narrative layer attached its columns and the corr-slice writer fired.
    snapshot_offers = snapshot_calls[0]["offers"]
    snapshot_parlays = snapshot_calls[0]["parlays"]
    assert "Why" in snapshot_offers.columns, "attach_offer_why did not add the Why column"
    assert "Position" in snapshot_offers.columns, "Position depth label did not flow to offers"
    assert "Thesis" in snapshot_parlays.columns, (
        "attach_parlay_theses did not add the Thesis column"
    )
    assert game_corr_calls, "write_current_game_corr was never invoked"

    # Game context is built once and the same frame fed to the writer: one row per
    # (League, Game, Date) with a classified shape.
    assert game_context_calls, "write_current_game_context was never invoked"
    context = game_context_calls[0]
    assert not context.empty, "build_game_context produced no rows from the offers frame"
    assert set(context["Game"]) == {"LVA/NYL", "PHX/SEA"}, (
        f"unexpected games: {set(context['Game'])}"
    )
    assert context["shape"].notna().all(), "every game context row carries a classified shape"

    # Story-menu writer fired with a column-stable frame. It is empty here: the
    # stubbed process_offers populates no story_sink, so the generator yields no
    # stories (story generation itself is covered by tests/golden/test_story_menu).
    from sportstradamus.prediction.stories.menu import _STORY_COLS

    assert game_stories_calls, "write_current_game_stories was never invoked"
    assert list(game_stories_calls[0].columns) == _STORY_COLS


# --- helpers --------------------------------------------------------------


def _stub_stats_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace per-league ``Stats`` I/O with no-ops in fake mode.

    ``meditate`` and ``prophecize`` instantiate every supported league's
    ``Stats`` class at startup and call ``load`` / ``update`` on the
    relevant ones; in fake mode we don't want any of those calls hitting
    ``nba_api``, ``nfl_data_py``, or local CSV caches. The per-league
    update gate is pinned open so the run never consults the host's real
    ``league_activity.json`` snapshot (or the season calendar).
    """
    import sportstradamus.stats.nba as nba_module
    import sportstradamus.stats.nfl as nfl_module
    import sportstradamus.stats.wnba as wnba_module
    from sportstradamus.helpers import odds_budget

    monkeypatch.setattr(odds_budget, "league_is_live", lambda league, season_start: True)
    monkeypatch.setattr(odds_budget, "update_window_open", lambda league, season_start: True)

    for mod in (nba_module, nfl_module, wnba_module):
        cls_name = {
            nba_module: "StatsNBA",
            nfl_module: "StatsNFL",
            wnba_module: "StatsWNBA",
        }[mod]
        cls = getattr(mod, cls_name)
        monkeypatch.setattr(cls, "load", lambda self: None)
        monkeypatch.setattr(cls, "update", lambda self: None)
        if hasattr(cls, "update_player_comps"):
            monkeypatch.setattr(cls, "update_player_comps", lambda self: None)
        if hasattr(cls, "trim_gamelog"):
            monkeypatch.setattr(cls, "trim_gamelog", lambda self: datetime.date(2026, 5, 1))


_PLAYER_LINES = [
    ("A'Ja Wilson", "LVA", "NYL", 22.5, 24.1),
    ("Jackie Young", "LVA", "NYL", 16.5, 17.8),
    ("Kelsey Plum", "LVA", "NYL", 18.5, 19.2),
    ("Chelsea Gray", "LVA", "NYL", 13.5, 14.0),
    ("Sabrina Ionescu", "NYL", "LVA", 19.5, 20.6),
    ("Breanna Stewart", "NYL", "LVA", 20.5, 21.7),
    ("Jonquel Jones", "NYL", "LVA", 14.5, 15.3),
    ("Skylar Diggins-Smith", "SEA", "PHX", 17.5, 18.0),
    ("Nneka Ogwumike", "SEA", "PHX", 15.5, 16.2),
    ("Jewell Loyd", "SEA", "PHX", 19.5, 20.8),
    ("Kahleah Copper", "PHX", "SEA", 18.5, 19.4),
    ("Brittney Griner", "PHX", "SEA", 16.5, 17.1),
]


# Team → implied win probability and half-total, so ``build_game_context`` has a
# real ``Moneyline``/``O/U`` to classify shape from (LVA and SEA the favorites).
_TEAM_CONTEXT = {
    "LVA": (0.62, 86.0),
    "NYL": (0.38, 80.0),
    "SEA": (0.55, 83.0),
    "PHX": (0.45, 81.0),
}


def _synthetic_offers() -> pd.DataFrame:
    """Mirror the column contract that ``prediction/cli.py`` consumes.

    Carries the post-``find_correlation`` columns the P2 narrative layer reads —
    ``Game`` (canonical sorted key), ``O/U`` half-total, ``Moneyline`` win
    probability, ``Position`` depth label, ``DVPOA`` — so ``build_game_context``
    produces real context rows rather than the empty-frame fallback.
    """
    rows = []
    for i, (player, team, opp, line, model_ev) in enumerate(_PLAYER_LINES):
        win_prob, half_total = _TEAM_CONTEXT[team]
        rows.append(
            {
                "League": "WNBA",
                "Date": "2026-05-08",
                "Game": "/".join(sorted([team, opp])),
                "Team": team,
                "Opponent": opp,
                "Player": player,
                "Market": "PTS",
                "Line": line,
                "Boost": 1.0,
                "Bet": "Over",
                "Projection": model_ev,
                "Model Param": line,
                "Market Projection": line,
                "Win Prob": 0.55,
                "Market Prob": 0.50,
                "Model EV": 1.05,
                "Market EV": 1.0,
                "O/U": half_total,
                "Moneyline": win_prob,
                "DVPOA": 0.06,
                "Dist": "Gamma",
                "CV": 1.0,
                "Gate": 0,
                "Temperature": 1.0,
                "Disp Cal": 1.0,
                "Step": 0.5,
                "Player position": "G",
                "Position": f"G{i % 3 + 1}",
                "Kelly": 1.0,
            }
        )
    return pd.DataFrame(rows)


def _synthetic_parlays(book: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Platform": book,
                "League": "WNBA",
                "Date": "2026-05-08",
                "Game": "LVA@NYL",
                "Family": "WNBA-PTS",
                "Model EV": 1.45,
                "Market EV": 1.10,
                "Rec Bet": 5.0,
                "Fun": 0.8,
                "P": 0.42,
                "PB": 0.30,
                "Legs": (
                    ("A'Ja Wilson", "PTS", 22.5, "Over"),
                    ("Sabrina Ionescu", "PTS", 19.5, "Over"),
                ),
            }
        ]
    )
