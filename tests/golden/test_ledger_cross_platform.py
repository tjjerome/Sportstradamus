"""Pin the MERGE design for cross-platform candidate combination in
strategies.ledger: Sleeper and Underdog candidates share one universe and
one persona budget (docs/handoffs/sleeper-parity.md Stage 4 ledger line),
rather than tracking independent per-platform bankrolls/budgets. See
test_ledger.py for the rest of this module's coverage (idempotency, record
schema, kelly-growth resizing) -- this file is scoped to the platform-merge
behavior only.
"""

from __future__ import annotations

import datetime
from decimal import Decimal

import pandas as pd
import pytest

from sportstradamus.helpers import UNDERDOG_BOOST_BASELINE
from sportstradamus.strategies import _ledger_selection, ledger, underdog_pickem
from sportstradamus.strategies._ledger_selection import LedgerCandidate
from sportstradamus.strategies.underdog_pickem import RecommendedEntry

DATE = datetime.date(2026, 7, 12)


def _redirect_store(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        ledger._ledger_store,
        "entries_path",
        lambda date: tmp_path / f"{date.isoformat()}.jsonl",
    )


def _canonical_leg(player: str, platform: str = "Underdog") -> dict:
    return {
        "player": player,
        "team": "",
        "market": "",
        "stat": "PTS",
        "bet": "Over",
        "line": 4.5,
        "league": "NBA",
        "game": "",
        "date": DATE.isoformat(),
        "platform": platform,
        "win_prob": 0.6,
        "boost": 1.0,
        "push_prob": 0.0,
        "kelly": 0.0,
    }


def _candidate(
    cand_id: str,
    players: frozenset[str],
    *,
    platform: str,
    joint_prob: float,
    ev: float = 0.1,
    stake: Decimal = Decimal("10"),
) -> LedgerCandidate:
    return LedgerCandidate(
        id=cand_id,
        contest_variant="power",
        entry_size=2,
        legs=tuple(f"{p} Over 4.5 rebounds - 60.0%, 1.6x" for p in players),
        canonical_legs=tuple(_canonical_leg(p, platform=platform) for p in players),
        players=players,
        game_span=1,
        joint_prob=joint_prob,
        payout_multiplier=3.0,
        ev=ev,
        stake=stake,
        platform=platform,
    )


def _recommended_entry(cand_id: str, player: str, platform: str) -> RecommendedEntry:
    return RecommendedEntry(
        id=cand_id,
        contest_variant="power",
        entry_size=2,
        legs=(f"{player} Over 4.5 rebounds - 60.0%, 1.6x",),
        joint_prob=0.5,
        payout_multiplier=3.0,
        ev=0.1,
        recommended_stake=Decimal("10"),
        canonical_legs=(_canonical_leg(player, platform=platform),),
        platform=platform,
    )


# --- build_candidate_universe combines both platforms' scrapes -----------------


def test_build_candidate_universe_combines_both_platforms(monkeypatch) -> None:
    def _fake_live_load(config, platform):
        return {}, pd.DataFrame(columns=["Boost"])

    def _fake_construct_entries(date, bankroll, config, *, parlay_dfs, platform):
        prefix = "ud" if platform == "Underdog" else "sl"
        return [_recommended_entry(f"{prefix}-same-1", f"{prefix.upper()} Same Player", platform)]

    def _fake_cross_game(offers_df, config, date, run_slot, *, platform):
        prefix = "ud" if platform == "Underdog" else "sl"
        return [
            _candidate(
                f"{prefix}-cross-1",
                frozenset({f"{prefix.upper()} Cross Player"}),
                platform=platform,
                joint_prob=0.5,
            )
        ]

    monkeypatch.setattr(ledger, "live_load", _fake_live_load)
    monkeypatch.setattr(ledger, "construct_entries", _fake_construct_entries)
    monkeypatch.setattr(ledger._ledger_cross_game, "build_cross_game_candidates", _fake_cross_game)

    pools = ledger.build_candidate_universe(DATE, "morning")

    universe = pools["safe"]
    assert pools["high_ev"] is universe
    assert pools["kelly_growth"] is universe
    assert len(universe) == 4
    by_id = {c.id: c for c in universe}
    assert set(by_id) == {"ud-same-1", "ud-cross-1", "sl-same-1", "sl-cross-1"}
    for cand_id, candidate in by_id.items():
        expected_platform = "Underdog" if cand_id.startswith("ud-") else "Sleeper"
        assert candidate.platform == expected_platform


# --- even_picks draws from Underdog's even picks alone -------------------------


def _same_game_parlay(players: tuple[str, str], boosts: tuple[float, float], platform: str) -> dict:
    return {
        "League": "NBA",
        "Game": "BOS/LAL",
        "Bet Size": 2,
        "Boost": 3.5 * boosts[0] * boosts[1],
        "Boost Pairs": (1.0,),
        "Model EV": 1.3,
        "legs": [
            _canonical_leg(player, platform=platform) | {"boost": boost}
            for player, boost in zip(players, boosts, strict=True)
        ],
    }


def _offer(player: str, game: str, boost: float, platform: str) -> dict:
    team, opponent = game.split("/")
    return {
        "Player": player,
        "Team": team,
        "Opponent": opponent,
        "Market": "Rebounds",
        "League": "NBA",
        "Game": game,
        "Platform": platform,
        "Date": DATE.isoformat(),
        "Line": 4.5,
        "Bet": "Over",
        "Win Prob": 0.60,
        "Market Prob": 0.57,
        "Boost": boost,
    }


def _mixed_multiplier_frames(platform: str) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """What ``live_load`` hands back: same-game parlays and offers, some all even picks,
    some discounted or boosted. The Sleeper frames carry 1.0 multipliers too, so a pool
    that admitted them on the multiplier alone would show it.
    """
    if platform == "Sleeper":
        parlays = [_same_game_parlay(("SL A", "SL B"), (1.0, 1.0), platform)]
        offers = [
            _offer("SL X", "NYK/MIA", 1.0, platform),
            _offer("SL Y", "DAL/PHX", 1.0, platform),
        ]
    else:
        parlays = [
            _same_game_parlay(("Even A", "Even B"), (1.0, 1.0), platform),
            _same_game_parlay(("Even C", "Discount D"), (1.0, 0.87), platform),
            _same_game_parlay(("Boost E", "Boost F"), (1.2, 1.2), platform),
        ]
        offers = [
            _offer("Even X", "NYK/MIA", 1.0, platform),
            _offer("Even Y", "DAL/PHX", 1.0, platform),
            _offer("Boost Z", "GSW/SAC", 1.1, platform),
            _offer("Discount W", "CHI/DET", 0.9, platform),
        ]
    return {"power": pd.DataFrame(parlays), "flex": pd.DataFrame()}, pd.DataFrame(offers)


def _load_mixed_multiplier_frames(monkeypatch) -> dict:
    """Fake ``live_load`` with the frames above, at full model trust so every cross-game
    combo clears the shared EV floor. Returns the frames by platform."""
    frames = {platform: _mixed_multiplier_frames(platform) for platform in ledger._PLATFORMS}
    monkeypatch.setattr(ledger, "live_load", lambda config, platform: frames[platform])
    monkeypatch.setattr(
        ledger._ledger_cross_game, "resolve_market_shrinkage", lambda *a: (1.0, "training")
    )
    return frames


def test_even_picks_pool_holds_only_underdog_entries_of_even_picks(monkeypatch) -> None:
    _load_mixed_multiplier_frames(monkeypatch)

    pools = ledger.build_candidate_universe(DATE, "morning")

    pool = pools["even_picks"]
    assert {c.players for c in pool} == {
        frozenset({"Even A", "Even B"}),  # same-game
        frozenset({"Even X", "Even Y"}),  # cross-game
    }
    assert all(c.platform == "Underdog" for c in pool)
    assert all(leg["boost"] == 1.0 for c in pool for leg in c.canonical_legs)


def test_shared_universe_is_every_candidate_whatever_its_multipliers(monkeypatch) -> None:
    """The even_picks pool is built beside the shared universe, never carved out of it:
    the other personas see what the builders give on the full frames."""
    frames = _load_mixed_multiplier_frames(monkeypatch)
    full_frame_candidates = [
        candidate
        for platform in ledger._PLATFORMS
        for candidate in ledger._platform_candidates(*frames[platform], DATE, "morning", platform)
    ]

    pools = ledger.build_candidate_universe(DATE, "morning")

    assert pools["safe"] == full_frame_candidates
    multipliers = {leg["boost"] for c in pools["safe"] for leg in c.canonical_legs}
    assert {0.87, 0.9, 1.0, 1.1, 1.2} <= multipliers
    assert "Sleeper" in {c.platform for c in pools["safe"]}


# --- run_commit draws from one shared pool/budget per persona ------------------


def _six_candidate_universe() -> list[LedgerCandidate]:
    """3 Underdog + 3 Sleeper candidates, distinct players (no Jaccard penalty
    against each other) and slightly staggered joint_prob (breaks exact score
    ties so the weighted draw isn't degenerate) -- all comfortably above the
    "safe" persona's implicit bar since its scorer is bare joint_prob.
    """
    underdog = [
        _candidate(
            f"ud-{i}", frozenset({f"UD Player {i}"}), platform="Underdog", joint_prob=0.5 + i * 0.01
        )
        for i in range(3)
    ]
    sleeper = [
        _candidate(
            f"sl-{i}", frozenset({f"SL Player {i}"}), platform="Sleeper", joint_prob=0.5 + i * 0.01
        )
        for i in range(3)
    ]
    return underdog + sleeper


def _force_full_draw(monkeypatch) -> None:
    """draw_entries stops early on a DRAW_CONTINUE_PROB coin flip after every
    successful draw (see _ledger_selection.py), so a 6-candidate pool doesn't
    reliably fill a 5-slot budget under the real per-replicate RNG stream --
    most replicates stop after 1-2 draws, which would leave the "6 candidates
    contend for 5 shared slots" claim unproven by accident. Pushing the
    threshold above 1.0 makes `rng.random() >= DRAW_CONTINUE_PROB` always
    False, i.e. never stop early, so the draw runs until the budget (5) or the
    pool (6) is exhausted -- deterministic given DATE's fixed RNG seed.
    """
    monkeypatch.setattr(_ledger_selection, "DRAW_CONTINUE_PROB", 2.0)


def _patch_universe(monkeypatch, universe: list[LedgerCandidate]) -> None:
    pools = dict.fromkeys(_ledger_selection.PERSONAS, universe) | {"even_picks": []}
    monkeypatch.setattr(ledger, "build_candidate_universe", lambda date, run_slot: pools)


def _safe_persona_records(tmp_path) -> list[dict]:
    path = tmp_path / f"{DATE.isoformat()}.jsonl"
    records = ledger._ledger_store.read_records(DATE)
    assert path.exists()  # sanity: we're reading the redirected file, not the real archive
    return [rec for rec in records if rec["persona"] == "safe" and rec["replicate_id"] == 0]


def test_run_commit_persona_budget_shared_across_platforms(monkeypatch, tmp_path) -> None:
    _redirect_store(monkeypatch, tmp_path)
    _force_full_draw(monkeypatch)
    _patch_universe(monkeypatch, _six_candidate_universe())

    ledger.run_commit(DATE, "morning")

    records = _safe_persona_records(tmp_path)
    assert len(records) == _ledger_selection.MAX_ENTRIES_PER_DAY
    platforms_drawn = {rec["platform"] for rec in records}
    assert platforms_drawn == {"Underdog", "Sleeper"}, (
        "expected both platforms to compete for and fill the one shared 5-slot "
        f"budget; got {platforms_drawn}"
    )


# --- committed records carry the correct platform after the shared draw --------


def test_committed_records_carry_correct_platform_after_shared_draw(monkeypatch, tmp_path) -> None:
    _redirect_store(monkeypatch, tmp_path)
    _force_full_draw(monkeypatch)
    universe = _six_candidate_universe()
    id_to_platform = {c.id: c.platform for c in universe}
    _patch_universe(monkeypatch, universe)

    ledger.run_commit(DATE, "morning")

    records = _safe_persona_records(tmp_path)
    assert len(records) == _ledger_selection.MAX_ENTRIES_PER_DAY
    for rec in records:
        assert rec["platform"] == id_to_platform[rec["id"]]


# --- live_load stamps Platform on the scraped offers ---------------------------


def _live_load_offers(monkeypatch, platform: str, boost: float) -> pd.DataFrame:
    """``live_load``'s offers frame with the Stats loaders, the scrape and the search stubbed.

    ``boost`` is what ``process_offers`` hands back for the one offer.
    """

    class _StubStats:
        season_start = None

        def load(self) -> None:
            pass

    for name in ("StatsNBA", "StatsNFL", "StatsWNBA", "StatsMLB", "StatsNHL"):
        monkeypatch.setattr(f"sportstradamus.stats.{name}", _StubStats)
    monkeypatch.setattr(underdog_pickem.odds_budget, "league_is_live", lambda *a: False)
    monkeypatch.setattr("sportstradamus.books.get_ud", list)
    monkeypatch.setattr("sportstradamus.books.get_sleeper", list)
    monkeypatch.setattr(
        "sportstradamus.prediction.scoring.process_offers",
        lambda *a, **k: (pd.DataFrame([{"Player": "A", "Market": "PTS", "Boost": boost}]), None),
    )
    monkeypatch.setattr(underdog_pickem, "_parlays_per_variant", lambda *a: {})

    _, offers_df = underdog_pickem.live_load(ledger._SHARED_CONFIG, platform)
    return offers_df


def test_live_load_stamps_platform_on_offers(monkeypatch) -> None:
    """``live_load`` is the ledger's only offers source, and ``build_leg`` /
    Sleeper flex pricing read ``Platform`` — a column ``process_offers`` never
    stamps (prophecize adds it to its own copies in prediction/cli.py). The
    fake ``live_load``s elsewhere in the suite include the column in their
    fixtures, which is exactly how the missing stamp shipped unnoticed."""
    offers_df = _live_load_offers(monkeypatch, "Underdog", boost=UNDERDOG_BOOST_BASELINE)

    assert (offers_df["Platform"] == "Underdog").all()


def test_live_load_returns_the_raw_pick_multiplier(monkeypatch) -> None:
    """``process_offers`` scales an Underdog ``Boost`` by the per-pick baseline and
    leaves Sleeper's alone. The cross-game builder prices on this frame and the
    committed record stores its legs, so both need the multiplier the platform
    quotes: an even pick comes back as exactly 1.0."""
    even_pick = _live_load_offers(monkeypatch, "Underdog", boost=UNDERDOG_BOOST_BASELINE)
    discounted = _live_load_offers(monkeypatch, "Underdog", boost=0.87 * UNDERDOG_BOOST_BASELINE)
    sleeper = _live_load_offers(monkeypatch, "Sleeper", boost=1.9)

    assert even_pick["Boost"].tolist() == [1.0]
    assert discounted["Boost"].tolist() == pytest.approx([0.87])
    assert sleeper["Boost"].tolist() == [1.9]
