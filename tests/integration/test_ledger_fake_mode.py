"""Fake-mode CLI coverage for the simulated-bettor ledger commit (Stage 4).

``docs/handoffs/sleeper-parity.md`` §6 Stage 4 acceptance criterion, verbatim:
"a fake-mode run produces platform-tagged Sleeper ledger rows." This file is
that proof, driven through the real ``ledger_commit`` CLI rather than a unit
test of any one module.

``underdog_pickem.live_load`` is this codebase's sanctioned short-circuit
seam for tests (see its docstring and ``construct_entries``'s
``parlay_dfs``/``offers_df`` injection params) -- faking it here, rather than
its internals (``get_ud``/``get_sleeper``/``process_offers``), avoids feeding
a hand-built ``offers_df`` through the real ``find_correlation`` beam search,
which depends on committed ``leagues/{league}/corr_same_team.parquet`` /
``corr_opposing.parquet`` matrices that aren't practical to fixture for a
small synthetic slate. Faking at this seam still exercises the real
``ledger_commit`` CLI, ``build_candidate_universe``'s real per-platform loop,
``construct_entries``'s real ranking/sizing, ``_ledger_cross_game``'s real
beam search + pooled-curve pricing, ``run_commit``'s real draw/budget/dedup
logic, and ``_ledger_store``'s real JSONL write -- everything except the
scrape + correlation search themselves. ``test_ledger_cross_platform.py``
(golden suite) already patches this same seam for platform-merge coverage;
this file's addition is driving it through the CLI and asserting on the
JSONL that actually lands on disk.

Unlike ``test_end_to_end.py``, which intercepts every write and never
touches disk, this file also monkeypatches ``_ledger_store.entries_path``
to resolve under ``tmp_path`` -- the whole point here is asserting real
ledger rows get written to disk and reading them back.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from sportstradamus.strategies import _ledger_store, ledger
from sportstradamus.strategies.underdog_pickem import PickemConfig

DATE = datetime.date(2026, 7, 13)

_LEAGUE_BY_PLATFORM = {"Underdog": "NBA", "Sleeper": "WNBA"}


def _canonical_leg(player: str, platform: str, league: str) -> dict:
    return {
        "player": player,
        "team": "BOS",
        "market": "PTS",
        "stat": "PTS",
        "bet": "Over",
        "line": 22.5,
        "league": league,
        "game": "BOS/LAL",
        "date": DATE.isoformat(),
        "platform": platform,
        "win_prob": 0.6,
        "boost": 1.0,
        "push_prob": 0.0,
        "kelly": 0.05,
    }


def _offers_df(platform: str) -> pd.DataFrame:
    """Minimal offers frame carrying the columns downstream code reads.

    Realism only matters as far as ``_ledger_cross_game.build_cross_game_candidates``
    (which runs for real against whatever ``live_load`` returns) -- its
    ``filter_legs`` gate needs ``Win Prob``/``Market Prob`` edges to clear
    ``_SHARED_CONFIG``'s thresholds, and its beam search needs >=2 distinct
    games to produce a cross-game combo.
    """
    league = _LEAGUE_BY_PLATFORM[platform]
    rows = [
        {
            "League": league,
            "Date": DATE.isoformat(),
            "Game": "BOS/LAL",
            "Team": "BOS",
            "Opponent": "LAL",
            "Player": f"{platform} Player A",
            "Market": "Points",
            "Line": 22.5,
            "Boost": 1.0,
            "Bet": "Over",
            "Platform": platform,
            "Win Prob": 0.58,
            "Market Prob": 0.545,
        },
        {
            "League": league,
            "Date": DATE.isoformat(),
            "Game": "NYK/MIA",
            "Team": "NYK",
            "Opponent": "MIA",
            "Player": f"{platform} Player B",
            "Market": "Points",
            "Line": 18.5,
            "Boost": 1.0,
            "Bet": "Over",
            "Platform": platform,
            "Win Prob": 0.56,
            "Market Prob": 0.53,
        },
    ]
    return pd.DataFrame(rows)


# (Model EV, Boost) chosen so each platform deterministically WINS a
# different persona's draw -- see _fake_live_load's docstring for why an
# accidental tie or one-sided dominance between platforms is a fixture bug,
# not something to leave to per-replicate RNG luck.
_PRICING_BY_PLATFORM = {
    "Underdog": (1.05, 2.0),  # tight payout -> higher joint_prob, wins "safe"
    "Sleeper": (1.20, 3.0),  # looser payout -> higher ev, wins "high_ev"/"kelly_growth"
}


def _fake_live_load(
    config: PickemConfig, platform: str
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Stand-in for the real ``live_load``: no scrape, no correlation search.

    Returns one same-game ``power`` parlay so ``construct_entries`` has
    something to rank, plus a real ``offers_df`` for
    ``_ledger_cross_game.build_cross_game_candidates`` to beam-search over.

    Underdog and Sleeper are priced so each platform wins a different
    persona's ``draw_entries`` weighting outright (Underdog: higher
    ``joint_prob`` -> "safe"; Sleeper: higher ``ev`` -> "high_ev"/
    "kelly_growth") -- with only two candidates in the pool, a tied or
    one-sided-dominant pricing would make the draw deterministically favor
    one platform every replicate (stable ``argsort`` breaks exact ties by
    position; a dominant candidate's percentile-weighted draw probability
    swamps the other's). Splitting which persona favors which platform makes
    both platforms committed by construction, not RNG luck.
    """
    model_ev, boost = _PRICING_BY_PLATFORM[platform]
    player = f"{platform} Same-Game Player"
    row = {
        "League": _LEAGUE_BY_PLATFORM[platform],
        "Date": DATE.isoformat(),
        "Game": "BOS/LAL",
        "Bet Size": 2,
        "Boost": boost,
        "Model EV": model_ev,
        "Leg 1": f"{player} Over 22.5 points - 60.0%, 3.0x",
        "Leg 2": f"{platform} Second Leg Over 10.5 rebounds - 58.0%, 3.0x",
        "legs": [
            _canonical_leg(player, platform, _LEAGUE_BY_PLATFORM[platform]),
            _canonical_leg(f"{platform} Second Leg", platform, _LEAGUE_BY_PLATFORM[platform]),
        ],
    }
    parlay_dfs = {"power": pd.DataFrame([row]), "flex": pd.DataFrame()}
    return parlay_dfs, _offers_df(platform)


@pytest.fixture
def _fake_mode_ledger(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect the ledger JSONL store under ``tmp_path`` and fake ``live_load``.

    Shared by both tests in this file so the CLI-invocation shape (fake
    scrape + on-disk store) isn't duplicated. Returns the entries directory
    so a test can inspect it directly if needed.
    """
    entries_dir = tmp_path / "entries"
    monkeypatch.setattr(
        _ledger_store, "entries_path", lambda date: entries_dir / f"{date.isoformat()}.jsonl"
    )
    monkeypatch.setattr(ledger, "live_load", _fake_live_load)
    return entries_dir


@pytest.mark.integration
def test_ledger_commit_fake_mode_produces_platform_tagged_sleeper_rows(
    _fake_mode_ledger: Path,
) -> None:
    """Stage 4 acceptance criterion: a fake-mode run through the real CLI
    produces committed ledger rows tagged with both platforms, not just a
    schema field that happens to exist."""
    runner = CliRunner()

    result = runner.invoke(
        ledger.ledger_commit,
        ["--run-slot", "morning", "--date", DATE.isoformat()],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    records = _ledger_store.read_records(DATE)
    platforms = {rec["platform"] for rec in records}
    assert "Sleeper" in platforms, f"no Sleeper-tagged rows committed; platforms={platforms}"
    assert "Underdog" in platforms, f"no Underdog-tagged rows committed; platforms={platforms}"


@pytest.mark.integration
def test_ledger_commit_fake_mode_idempotent_on_repeat_invocation(
    _fake_mode_ledger: Path,
) -> None:
    """Same (run_slot, date) invoked twice must not double-commit."""
    runner = CliRunner()
    args = ["--run-slot", "morning", "--date", DATE.isoformat()]

    first = runner.invoke(ledger.ledger_commit, args, catch_exceptions=False)
    assert first.exit_code == 0

    second = runner.invoke(ledger.ledger_commit, args, catch_exceptions=False)

    assert second.exit_code == 0
    assert "committed 0 new entries" in second.output
