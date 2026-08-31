"""Golden: combo-spec weights == the DFS platforms' published scoring rules.

Weights are pinned against the platform scoring tables declared below, never
against the stored gamelog settlement columns: the MLB gamelog's ``hitByPitch``
field never populates, so its fantasy settlement columns silently omit the HBP
term (combo-sum pricing brief §8e) and would pin a biased truth.

NFL fantasy composites deliberately declare no spec: their scoring spans a
dozen thinly-quoted markets, so they stay on the model path.

Platform rule sources: underdogfantasy.com Rules -> Scoring and the
prizepicks.com scoring chart (2026 season), cross-checked against the repo's
settled gamelog formulas where those are trustworthy (every league but MLB's
HBP term).
"""

import pytest

from sportstradamus.helpers import combo_props
from sportstradamus.stats.mlb import (
    FANTASY_HBP_WEIGHTS,
    FANTASY_HIT_TYPE_WEIGHTS,
    LEAGUE_HIT_SHARES,
    StatsMLB,
)
from sportstradamus.stats.nba import NBA_FANTASY_WEIGHTS, StatsNBA
from sportstradamus.stats.nfl import StatsNFL
from sportstradamus.stats.nhl import StatsNHL
from sportstradamus.stats.wnba import StatsWNBA

# Simple-sum combos: each combo market settles as the plain sum of its
# components, so combo_quote assigns every component weight 1.0.
EXPECTED_COMBO_PROPS = {
    "qb tds": ["passing tds", "rushing tds"],
    "qb yards": ["passing yards", "rushing yards"],
    "yards": ["receiving yards", "rushing yards"],
    # nfl.py's update() settles tds = rushing_tds + receiving_tds, alongside
    # qb_tds = rushing_tds + passing_tds for the QB market.
    "tds": ["receiving tds", "rushing tds"],
    "hits+runs+rbi": ["hits", "runs", "rbi"],
    "PRA": ["PTS", "REB", "AST"],
    "RA": ["REB", "AST"],
    "PR": ["PTS", "REB"],
    "PA": ["PTS", "AST"],
    "BLST": ["BLK", "STL"],
    "sogBS": ["shots", "blocked"],
}

# NBA/WNBA: PrizePicks and Underdog use identical basketball weights.
NBA_PLATFORM_SCORING = {"PTS": 1, "REB": 1.2, "AST": 1.5, "BLK": 3, "STL": 3, "TOV": -1}

# NHL Underdog: skater goals 6 / assists 4 / SOG 1 / blocked 1 / hits 0.5 /
# power-play points 0.5; goalie win 6 / save 0.6 / goal against -3. "sogBS" is
# the shots+blocked sum market (both sides weight 1); "win"/"Moneyline" both
# name the goalie-win component.
UNDERDOG_NHL_SKATER = {
    "goals": 6,
    "assists": 4,
    "shots": 1,
    "blocked": 1,
    "sogBS": 1,
    "hits": 0.5,
    "powerPlayPoints": 0.5,
}
UNDERDOG_NHL_GOALIE = {"saves": 0.6, "goalsAgainst": -3, "win": 6, "Moneyline": 6}
# NHL PrizePicks skater: goals 8 / assists 5 / SOG 1.5 / blocked 1.5.
PRIZEPICKS_NHL_SKATER = {"goals": 8, "assists": 5, "shots": 1.5, "blocked": 1.5, "sogBS": 1.5}

# MLB hit-type weights — Underdog 1B 3 / 2B 6 / 3B 8 / HR 10, PrizePicks
# 1B 3 / 2B 5 / 3B 8 / HR 10 — are deliberately NOT pinned per component here:
# the spec may quote hit types directly or split a quoted ``hits`` by compound
# multinomial (brief §8d), and the linear weight per name differs between the
# two encodings. Same for pitcher innings (Underdog settles 3 per COMPLETED
# inning, so any linear outs weight is a proxy choice; brief §8a) and quality
# start (a post functional of outs/runs). Those weights are pinned against the
# MLB lane's exported module constants below.
MLB_ENCODING_DEPENDENT = {
    "hits",
    "singles",
    "doubles",
    "triples",
    "home runs",
    "hit by pitch",
    "hbp",
    "pitching outs",
    "quality start",
}
UNDERDOG_MLB_HITTER = {"walks": 3, "rbi": 2, "runs": 2, "stolen bases": 4}
PRIZEPICKS_MLB_HITTER = {"walks": 2, "rbi": 2, "runs": 2, "stolen bases": 5}
UNDERDOG_MLB_PITCHER = {
    "pitcher strikeouts": 3,
    "runs allowed": -3,  # platform settles EARNED runs -3; quoted proxy is total runs
    "earned runs allowed": -3,
    "pitcher win": 5,
    "win": 5,
}
PRIZEPICKS_MLB_PITCHER = {
    "pitcher strikeouts": 3,
    "runs allowed": -3,
    "earned runs allowed": -3,
    "pitcher win": 6,
    "win": 6,
    "pitching outs": 1,  # PrizePicks settles exactly 1 per out (3/inning + 1/extra out)
}

# NFL: PrizePicks pass yds 0.04 / pass TD 4 / INT -1 / rush+rec yds 0.1 /
# rush+rec TD 6 / reception 1 / fumble lost -1 / 2pt conversion 2; Underdog
# identical except reception 0.5 and fumble lost -2.
PRIZEPICKS_NFL = {
    "passing yards": 0.04,
    "passing tds": 4,
    "interceptions": -1,
    "yards": 0.1,
    "rushing yards": 0.1,
    "receiving yards": 0.1,
    "tds": 6,
    "rushing tds": 6,
    "receiving tds": 6,
    "receptions": 1,
    "fumbles lost": -1,
}
UNDERDOG_NFL = PRIZEPICKS_NFL | {"receptions": 0.5, "fumbles lost": -2}


def _spec(cls, market):
    spec = object.__new__(cls)._fantasy_combo_spec(market)
    assert spec is not None, f"{cls.__name__} declares no combo spec for {market!r}"
    return spec


def _check_platform_weights(spec, table, league_market, allowed_extra=frozenset()):
    declared = dict(spec.marginals) | dict(spec.bernoulli)
    assert declared, f"{league_market}: spec declares no weighted components"
    for name, weight in declared.items():
        if name in table:
            assert weight == pytest.approx(table[name]), f"{league_market}: {name}"
        else:
            assert name in allowed_extra, f"{league_market}: unexpected component {name!r}"
    for name in spec.sampled:
        assert name in set(table) | allowed_extra, f"{league_market}: unexpected sampled {name!r}"


def test_combo_props_registry():
    assert combo_props == EXPECTED_COMBO_PROPS


def test_nba_weight_table_matches_platform_scoring():
    assert dict(NBA_FANTASY_WEIGHTS) == NBA_PLATFORM_SCORING


@pytest.mark.parametrize("cls", [StatsNBA, StatsWNBA])
@pytest.mark.parametrize("market", ["fantasy points underdog", "fantasy points prizepicks"])
def test_nba_family_fantasy_weights(cls, market):
    spec = _spec(cls, market)
    assert dict(spec.marginals) == NBA_PLATFORM_SCORING
    assert spec.sampled == ()
    assert spec.bernoulli == ()
    assert spec.post_builder is None


NHL_MARKET_TABLES = {
    "skater fantasy points underdog": UNDERDOG_NHL_SKATER,
    "goalie fantasy points underdog": UNDERDOG_NHL_GOALIE,
    "fantasy points prizepicks": PRIZEPICKS_NHL_SKATER,
}


@pytest.mark.parametrize("market", sorted(NHL_MARKET_TABLES))
def test_nhl_fantasy_weights(market):
    spec = _spec(StatsNHL, market)
    _check_platform_weights(spec, NHL_MARKET_TABLES[market], f"NHL {market}")


MLB_MARKET_TABLES = {
    "hitter fantasy points underdog": UNDERDOG_MLB_HITTER,
    "pitcher fantasy points underdog": UNDERDOG_MLB_PITCHER,
    "hitter fantasy score": PRIZEPICKS_MLB_HITTER,
    "pitcher fantasy score": PRIZEPICKS_MLB_PITCHER,
}


@pytest.mark.parametrize("market", sorted(MLB_MARKET_TABLES))
def test_mlb_fantasy_weights(market):
    spec = _spec(StatsMLB, market)
    _check_platform_weights(
        spec, MLB_MARKET_TABLES[market], f"MLB {market}", allowed_extra=MLB_ENCODING_DEPENDENT
    )


# The hit types reach the sum through the compound-multinomial split of a
# quoted ``hits`` draw, so their weights live in the post-term table rather
# than in spec.marginals.
UNDERDOG_MLB_HIT_TYPES = {"singles": 3, "doubles": 6, "triples": 8, "home runs": 10}
PRIZEPICKS_MLB_HIT_TYPES = {"singles": 3, "doubles": 5, "triples": 8, "home runs": 10}


@pytest.mark.parametrize(
    ("market", "table"),
    [
        ("hitter fantasy points underdog", UNDERDOG_MLB_HIT_TYPES),
        ("hitter fantasy score", PRIZEPICKS_MLB_HIT_TYPES),
    ],
)
def test_mlb_hit_type_post_term_weights(market, table):
    assert dict(FANTASY_HIT_TYPE_WEIGHTS[market]) == table
    assert sum(LEAGUE_HIT_SHARES) == pytest.approx(1.0)
    assert len(LEAGUE_HIT_SHARES) == len(table)


def test_mlb_hbp_offset_weights():
    # Both platforms score a hit by pitch like a walk. No book quotes HBP, so
    # the spec carries it as a deterministic mean offset instead of a
    # component; pin it against the same walk weight the marginals use.
    assert FANTASY_HBP_WEIGHTS["hitter fantasy points underdog"] == UNDERDOG_MLB_HITTER["walks"]
    assert FANTASY_HBP_WEIGHTS["hitter fantasy score"] == PRIZEPICKS_MLB_HITTER["walks"]


@pytest.mark.parametrize("market", ["fantasy points underdog", "fantasy points prizepicks"])
def test_nfl_fantasy_declares_no_combo_spec(market):
    # NFL fantasy composites are deliberately out of the component-sum lane:
    # their scoring spans a dozen markets whose books quote thinly, so they stay
    # on the model path with the platform's own payout as the market reference.
    # NFL's simple combos (qb tds / qb yards / yards) ride combo_props instead.
    assert object.__new__(StatsNFL)._fantasy_combo_spec(market) is None
