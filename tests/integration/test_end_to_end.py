"""End-to-end smoke test for the ``confer -> meditate -> prophecize`` flow.

The goal is *not* to validate model quality or reproduce a live run end to
end; it is to confirm that the three CLIs still wire up to one another
and that none of the orchestration code has been broken by a refactor.

Two modes, controlled by the ``SPORTSTRADAMUS_INTEGRATION_REAL_APIS``
environment variable:

* **Fake (default).** The Odds API, ``nba_api``, Google Sheets, Underdog,
  and Sleeper are all replaced with stubs / canned fixtures. Runs in
  well under 90 seconds.
* **Real.** Set ``SPORTSTRADAMUS_INTEGRATION_REAL_APIS=1`` to opt in to
  live network calls (``confer`` still goes through ``--fixture-dir``,
  but ``meditate`` and ``prophecize`` get real ``StatsWNBA`` data).
  Allowed to take longer.

The test never writes data: every disk-write touchpoint
(``Archive.write``, model pickle writes, history files, Google Sheets) is
intercepted. We exercise import paths and callback wiring only.

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

    # ----- Phase 2: meditate (CLI invoked; ML stubbed; no writes) -----
    from sportstradamus.training import cli as training_cli
    from sportstradamus.training import markets as markets_module

    # Restrict to one market; skip the extended per-market loop.
    monkeypatch.setattr(markets_module, "ALL_MARKETS", {"WNBA": ["PTS"]})
    monkeypatch.setattr(training_cli, "ALL_MARKETS", {"WNBA": ["PTS"]})

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
    assert (
        ("WNBA", "PTS") in train_market_calls
    ), f"train_market was not invoked for WNBA:PTS. calls={train_market_calls}"

    # ----- Phase 3: prophecize (CLI invoked; parquet snapshot + scrapers mocked) -----
    from sportstradamus.prediction import cli as prediction_cli

    monkeypatch.setattr(prediction_cli, "get_ud", lambda: {})
    monkeypatch.setattr(prediction_cli, "get_sleeper", lambda: {})

    captured: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    def stub_process_offers(offer_dict, book, stats, **kwargs):
        offers = _synthetic_offers()
        parlays = _synthetic_parlays(book)
        captured[book] = (offers, parlays)
        return offers, parlays

    monkeypatch.setattr(prediction_cli, "process_offers", stub_process_offers)

    snapshot_calls: list[dict] = []

    def stub_write_current_offers(offers, parlays, leagues, platforms, contest_variant="power"):
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
    assert (
        underdog_offers["Model EV"].notna().sum() >= 10
    ), "fewer than 10 offers had a populated Model EV column"
    parlay_total = sum(len(p) for _, p in captured.values())
    assert parlay_total >= 1, "no parlay candidates were returned"


# --- helpers --------------------------------------------------------------


def _stub_stats_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace per-league ``Stats`` I/O with no-ops in fake mode.

    ``meditate`` and ``prophecize`` instantiate every supported league's
    ``Stats`` class at startup and call ``load`` / ``update`` on the
    relevant ones; in fake mode we don't want any of those calls hitting
    ``nba_api``, ``nfl_data_py``, or local CSV caches.
    """
    import sportstradamus.stats.nba as nba_module
    import sportstradamus.stats.nfl as nfl_module
    import sportstradamus.stats.wnba as wnba_module

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


def _synthetic_offers() -> pd.DataFrame:
    """Mirror the column contract that ``prediction/cli.py`` consumes."""
    rows = []
    for player, team, opp, line, model_ev in _PLAYER_LINES:
        rows.append(
            {
                "League": "WNBA",
                "Date": "2026-05-08",
                "Team": team,
                "Opponent": opp,
                "Player": player,
                "Market": "PTS",
                "Line": line,
                "Boost": 1.0,
                "Bet": "Over",
                "Model EV": model_ev,
                "Model Param": line,
                "Books EV": line,
                "Model P": 0.55,
                "Books P": 0.50,
                "Model": 1.05,
                "Books": 1.0,
                "Dist": "Gamma",
                "CV": 1.0,
                "Gate": 0,
                "Temperature": 1.0,
                "Disp Cal": 1.0,
                "Step": 0.5,
                "Player position": "G",
                "K": 1.0,
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
                "Books EV": 1.10,
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
