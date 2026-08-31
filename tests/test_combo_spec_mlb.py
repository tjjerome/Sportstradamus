"""Combo-spec tests for the four MLB fantasy component-sum markets.

Pins the spec weight tables against `_mlb_fantasy_props`, the binomial-ladder
hit-type split, the in-sample quality-start functional, post-hook determinism
under POST_RNG_SEED, and the SP-win moneyline map (fitted logistic + measured
bucket-table fallback) per the combo-sum pricing brief §8.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from sportstradamus.helpers.combined_markets import POST_RNG_SEED
from sportstradamus.stats import mlb as mlb_mod
from sportstradamus.stats.mlb import (
    FANTASY_COMBO_MARKETS,
    FANTASY_HIT_TYPE_WEIGHTS,
    LEAGUE_HBP_PER_GAME,
    LEAGUE_HIT_SHARES,
    SP_WIN_RATIO_BUCKETS,
    StatsMLB,
    _multinomial_split,
)

QUOTE_DATE = "2026-08-30"

MLB_LOG_STRINGS = {"player": "playerName", "date": "gameDate", "team": "team"}


def make_stats(**attrs):
    """Bare StatsMLB carrying only what the tested methods read."""
    stats = object.__new__(StatsMLB)
    stats.league = "MLB"
    stats.log_strings = MLB_LOG_STRINGS
    for key, value in attrs.items():
        setattr(stats, key, value)
    return stats


def empty_gamelog():
    return pd.DataFrame(columns=["playerName", "gameDate", "team"])


def test_spec_tables_match_fantasy_props():
    # Spec construction is pure config: no gamelog/profile attributes are set here,
    # so any instance-data access at spec-build time fails loudly.
    stats = make_stats()
    for market in sorted(FANTASY_COMBO_MARKETS):
        spec = stats._fantasy_combo_spec(market, "Any Player")
        props = dict(StatsMLB._mlb_fantasy_props(market))
        if "pitcher" in market:
            assert dict(spec.marginals) == {
                sub: props[sub] for sub in ("pitcher strikeouts", "runs allowed", "pitching outs")
            }
            assert spec.bernoulli == (("pitcher win", props["pitcher win"]),)
            assert spec.sampled == ()
            assert spec.analytics == ("win_map",)
            assert spec.post_builder.keywords == {"weight": props["quality start"]}
        else:
            assert dict(spec.marginals) == {
                sub: props[sub] for sub in ("walks", "rbi", "runs", "stolen bases")
            }
            assert spec.bernoulli == ()
            assert spec.sampled == ("hits",)
            assert spec.analytics == ("hit_shares", "hbp")
            assert dict(FANTASY_HIT_TYPE_WEIGHTS[market]) == {
                sub: props[sub] for sub in ("singles", "doubles", "triples", "home runs")
            }


def test_underdog_props_tables_pinned():
    assert dict(StatsMLB._mlb_fantasy_props("hitter fantasy points underdog")) == {
        "singles": 3,
        "doubles": 6,
        "triples": 8,
        "home runs": 10,
        "walks": 3,
        "rbi": 2,
        "runs": 2,
        "stolen bases": 4,
    }
    assert dict(StatsMLB._mlb_fantasy_props("pitcher fantasy points underdog")) == {
        "pitcher win": 5,
        "pitcher strikeouts": 3,
        "runs allowed": -3,
        "pitching outs": 1,
        "quality start": 5,
    }


def test_spec_none_outside_the_four():
    stats = make_stats()
    for market in (
        "hitter fantasy points parlay",
        "pitcher fantasy points parlay",
        "hits+runs+rbi",
        "pitcher strikeouts",
    ):
        assert stats._fantasy_combo_spec(market, "Any Player") is None


def test_multinomial_split_sums_and_means():
    hits = np.random.default_rng(7).integers(0, 6, size=20000)
    split = _multinomial_split(np.random.default_rng(POST_RNG_SEED), hits, LEAGUE_HIT_SHARES)
    assert np.array_equal(sum(split), hits)
    for share, counts in zip(LEAGUE_HIT_SHARES, split, strict=True):
        assert counts.min() >= 0
        assert counts.mean() == pytest.approx(hits.mean() * share, abs=0.05)


def test_quality_start_post_exact():
    draws = {
        "pitching outs": np.array([18.0, 17.9, 21.0, 18.0, 12.0]),
        "runs allowed": np.array([3.0, 0.0, 4.0, 2.0, 1.0]),
        "pitcher strikeouts": np.zeros(5),
        "pitcher win": np.zeros(5),
    }
    stats = make_stats()
    for market, weight in (
        ("pitcher fantasy points underdog", 5.0),
        ("pitcher fantasy score", 4.0),
    ):
        post = stats._fantasy_combo_spec(market, "Any Player").post_builder(
            stats, "Any Pitcher", QUOTE_DATE
        )
        assert np.array_equal(post(draws), weight * np.array([1.0, 0.0, 0.0, 1.0, 0.0]))


def hitter_window(games, home_runs_per_game):
    return pd.DataFrame(
        {
            "playerName": ["Slugger"] * games,
            "hits": [home_runs_per_game] * games,
            "singles": [0] * games,
            "doubles": [0] * games,
            "triples": [0] * games,
            "home runs": [home_runs_per_game] * games,
        }
    )


def test_hitter_post_deterministic_and_offset():
    stats = make_stats(short_gamelog=hitter_window(30, 1))
    spec = stats._fantasy_combo_spec("hitter fantasy points underdog", "Any Hitter")
    hits = np.random.default_rng(11).integers(0, 6, size=8192).astype(float)
    draws = {"hits": hits, "walks": np.zeros(8192)}
    term_a = spec.post_builder(stats, "Slugger", QUOTE_DATE)(draws)
    term_b = spec.post_builder(stats, "Slugger", QUOTE_DATE)(draws)
    assert np.array_equal(term_a, term_b)
    # 30 window hits >= the shares floor and all are home runs, so the split is
    # degenerate and the term is exactly 10*H plus the 3*HBP offset (+0.135).
    assert np.allclose(term_a, 10.0 * hits + 3.0 * LEAGUE_HBP_PER_GAME)


def test_hitter_post_league_share_fallback():
    # 3 window hits sit below _HIT_SHARES_MIN_HITS, so the league shares split
    # the draws: E[term] = H * sum(w*s) + 2*HBP offset for the score variant.
    stats = make_stats(short_gamelog=hitter_window(3, 1))
    spec = stats._fantasy_combo_spec("hitter fantasy score", "Any Hitter")
    hits = np.full(20000, 4.0)
    term = spec.post_builder(stats, "Slugger", QUOTE_DATE)(draws={"hits": hits})
    weights = [w for _, w in FANTASY_HIT_TYPE_WEIGHTS["hitter fantasy score"]]
    expected = 4.0 * sum(w * s for w, s in zip(weights, LEAGUE_HIT_SHARES, strict=True))
    assert term.mean() == pytest.approx(expected + 2.0 * LEAGUE_HBP_PER_GAME, abs=0.15)


def fake_archive(p_team, seen=None):
    def get_moneyline(league, date, team):
        if seen is not None:
            seen.append((league, date, team))
        return p_team

    return SimpleNamespace(get_moneyline=get_moneyline)


def test_bernoulli_p_bucket_table(monkeypatch):
    # No "moneyline" column -> the logistic fit is infeasible -> bucket table.
    stats = make_stats(
        gamelog=empty_gamelog(),
        upcoming_games={"NYY": {"Pitcher": "Gerrit Cole"}},
        _sp_win_curve=None,
    )
    for p_team, ratio in ((0.5, 0.589), (0.70, 0.684), (0.35, 0.430)):
        monkeypatch.setattr(mlb_mod, "archive", fake_archive(p_team))
        p = stats._combo_bernoulli_p("pitcher win", "Gerrit Cole", QUOTE_DATE)
        assert p == pytest.approx(p_team * ratio)
    # Brief §8b: overall P(team win) 0.500 -> P(SP win) 0.293; the 0.45-0.50
    # bucket ratio 0.589 lands 0.5 * 0.589 = 0.2945 on that measurement.
    monkeypatch.setattr(mlb_mod, "archive", fake_archive(0.5))
    stats._sp_win_curve = None
    p = stats._combo_bernoulli_p("pitcher win", "Gerrit Cole", QUOTE_DATE)
    assert p == pytest.approx(0.293, abs=0.005)
    assert stats._sp_win_curve == ()


def synthetic_starter_gamelog(n=3000):
    """Starter rows drawn from the brief's logistic: logit p_SP = a + 1.0 * logit p_team."""
    rng = np.random.default_rng(0)
    a_true = logit(0.293)
    p_team = rng.uniform(0.25, 0.75, n)
    won = rng.random(n) < expit(a_true + logit(p_team))
    quoted = pd.DataFrame(
        {
            "playerName": "Some Starter",
            "gameDate": "2026-06-01",
            "team": "BOS",
            "starting pitcher": True,
            "pitcher win": won.astype(int),
            "moneyline": p_team,
        }
    )
    unquoted = quoted.head(200).assign(moneyline=0.5)  # archive miss default, excluded
    return pd.concat([quoted, unquoted], ignore_index=True)


def test_bernoulli_p_fitted_curve(monkeypatch):
    stats = make_stats(
        gamelog=synthetic_starter_gamelog(),
        upcoming_games={"NYY": {"Pitcher": "Probable Guy"}},
        _sp_win_curve=None,
    )
    monkeypatch.setattr(mlb_mod, "archive", fake_archive(0.5))
    p = stats._combo_bernoulli_p("pitcher win", "Probable Guy", QUOTE_DATE)
    # The fitted map at p_team = 0.5 is expit(a): the brief's 0.293-vs-0.500
    # relationship, recovered from the synthetic gamelog within sampling error.
    assert p == pytest.approx(0.293, abs=0.02)
    a, b = stats._sp_win_curve
    assert a == pytest.approx(logit(0.293), abs=0.15)
    assert b == pytest.approx(1.0, abs=0.25)


def test_bernoulli_p_unresolved_team_is_nan():
    stats = make_stats(gamelog=empty_gamelog(), upcoming_games={}, _sp_win_curve=None)
    assert np.isnan(stats._combo_bernoulli_p("pitcher win", "Unknown Arm", QUOTE_DATE))


def test_pitcher_team_resolution(monkeypatch):
    gamelog = pd.DataFrame(
        {
            "playerName": ["Settled Starter"],
            "gameDate": ["2026-06-01"],
            "team": ["BOS"],
        }
    )
    stats = make_stats(
        gamelog=gamelog,
        upcoming_games={"NYY2": {"Pitcher": "Nightcap Guy"}},
        _sp_win_curve=(),
    )
    assert stats._pitcher_team("Settled Starter", "2026-06-01") == "BOS"
    # Doubleheader keys carry a game-number suffix the archive does not use.
    seen = []
    monkeypatch.setattr(mlb_mod, "archive", fake_archive(0.5, seen))
    stats._combo_bernoulli_p("pitcher win", "Nightcap Guy", QUOTE_DATE)
    assert seen == [("MLB", QUOTE_DATE, "NYY")]


def test_bucket_edges_cover_the_unit_interval():
    edges = [edge for edge, _ in SP_WIN_RATIO_BUCKETS]
    assert edges == sorted(edges)
    assert edges[-1] == 1.0
