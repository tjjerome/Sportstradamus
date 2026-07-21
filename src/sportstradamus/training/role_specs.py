"""Per-(league, market) role columns for the role×position group-CDF strategy.

Single source of truth read by three consumers: the strategy applicability gate
(`model_strategy_registry.Applicability.matches`), the training role-context
builder (`structural_context.build_two_part_context`), and the pipeline
prune-preservation step. A continuous cell is eligible for the two-part
role×position method iff it has a registered ``RoleSpec`` whose columns its
matrix carries.

The role score tiers players within position by a conditional-expectation
surrogate ``Π(volume) × Σ_component(usage × efficiency)``. Column choices are a
small, market-appropriate pick from each league's shared schema: basketball uses
the stat's team share (``PCT_*``) × player rate; NFL receiving/rushing use
target/carry share × yards-per; NHL uses ice-time share × per-60 rate. Composite
markets (scrimmage yards, PRA/PR/RA/PA) sum their components, which is the
expected value of the fused stat by linearity. Position **codes are league-wide**
(one roster per league); the context builder tiers only the codes a market's
train rows actually carry.
"""

from __future__ import annotations

from dataclasses import dataclass

from sportstradamus.training.structural_strategies import ROLE_COLUMNS

# Mirrors each league's ``Stats.positions`` list (position code == index + 1).
# A league physical constant duplicated here so this module stays import-light
# (importing the Stats subclasses would construct an Archive); parity with the
# Stats classes is pinned by ``tests/golden/test_role_specs.py``.
_POSITION_LABELS: dict[str, tuple[str, ...]] = {
    "NFL": ("QB", "WR", "RB", "TE"),
    "NBA": ("P", "C", "F", "W", "B"),
    "WNBA": ("G", "F", "C"),
    "NHL": ("C", "W", "D", "G"),
}


def league_position_codes(league: str) -> tuple[int, ...]:
    """1-indexed position codes for a league's roster (code == label index + 1)."""
    return tuple(range(1, len(_POSITION_LABELS[league]) + 1))


def position_label(league: str, code: int) -> str:
    """Human-readable label for a 1-indexed league position code."""
    return _POSITION_LABELS[league][code - 1]


@dataclass(frozen=True)
class RoleSpec:
    """Market-appropriate role columns for one cell's role-score tiering.

    ``efficiency_cols`` is either empty (volume/attempt markets, where the
    outcome *is* the usage) or aligned one-to-one with ``usage_share_cols``
    (each share pairs with its efficiency as one summed component).
    """

    volume_cols: tuple[str, ...]
    usage_share_cols: tuple[str, ...]
    efficiency_cols: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Fail at import if a registry row's column tuples are misaligned."""
        if self.efficiency_cols and len(self.efficiency_cols) != len(self.usage_share_cols):
            raise ValueError(f"efficiency_cols must be empty or match usage_share_cols: {self}")
        if len(self.usage_share_cols) > 1 and not self.efficiency_cols:
            raise ValueError(f"composite role spec needs per-share efficiency: {self}")
        if not self.volume_cols or not self.usage_share_cols:
            raise ValueError(f"role spec requires volume and usage columns: {self}")

    @property
    def role_columns(self) -> tuple[str, ...]:
        return (*self.volume_cols, *self.usage_share_cols, *self.efficiency_cols)

    @property
    def all_columns(self) -> tuple[str, ...]:
        return (*self.role_columns, "Player position")


# NBA/WNBA share one column schema; a market picks its team-share × player-rate
# pair, and any continuous cell without a named pair falls back to the fused
# usage rate × player impact (``USG_PCT`` × ``PIE``).
_BASKETBALL_VOLUME = ("Team PACE",)
_BASKETBALL_DEFAULT: tuple[tuple[str, ...], tuple[str, ...]] = (
    ("Player USG_PCT",),
    ("Player PIE",),
)
_BASKETBALL_FACTORS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "PTS": (("Player PCT_FGA",), ("Player TS_PCT",)),
    "REB": (("Player PCT_REB",), ("Player REB_PCT",)),
    "AST": (("Player PCT_AST",), ("Player AST_RATIO",)),
    "DREB": (("Player PCT_REB",), ("Player DREB_PCT",)),
    "FGA": (("Player PCT_FGA",), ()),
    "FG3A": (("Player PCT_FG3A",), ()),
    "FGM": (("Player PCT_FGA",), ("Player EFG_PCT",)),
    "PR": (("Player PCT_FGA", "Player PCT_REB"), ("Player TS_PCT", "Player REB_PCT")),
    "PA": (("Player PCT_FGA", "Player PCT_AST"), ("Player TS_PCT", "Player AST_RATIO")),
    "RA": (("Player PCT_REB", "Player PCT_AST"), ("Player REB_PCT", "Player AST_RATIO")),
    "PRA": (
        ("Player PCT_FGA", "Player PCT_REB", "Player PCT_AST"),
        ("Player TS_PCT", "Player REB_PCT", "Player AST_RATIO"),
    ),
}

# NHL shares one schema across every continuous cell: team shot-attempt volume ×
# ice-time share × shots-per-60. Goalie cells carry no shooting rate, so the role
# score degenerates and the method self-kills to the pooled fallback.
_NHL_DEFAULT = RoleSpec(("Team Corsi",), ("Player TimeShare",), ("Player Shot60",))

# NFL role columns differ by market (rushing vs. receiving vs. passing) but every
# NFL matrix carries all of them, so the pick is per-market. Receiving derives
# from the legacy flat ROLE_COLUMNS (plays, pass_rate, target share, yards per
# target) to keep its role score and blob byte-identical to the shipped baseline.
_NFL_PASS_VOLUME = ("Team plays_per_game", "Team pass_rate")
_NFL_RUN_VOLUME = ("Team plays_per_game",)
_NFL_QB_USAGE = ("Player snap pct",)
_NFL_PASS_EFFICIENCY = ("Player pass yards per attempt",)
_NFL_CELLS: dict[str, RoleSpec] = {
    "receiving yards": RoleSpec(ROLE_COLUMNS[:2], ROLE_COLUMNS[2:3], ROLE_COLUMNS[3:4]),
    "targets": RoleSpec(_NFL_PASS_VOLUME, ("Player target share",)),
    "receptions": RoleSpec(_NFL_PASS_VOLUME, ("Player target share",)),
    "passing yards": RoleSpec(_NFL_PASS_VOLUME, _NFL_QB_USAGE, _NFL_PASS_EFFICIENCY),
    "qb yards": RoleSpec(_NFL_PASS_VOLUME, _NFL_QB_USAGE, _NFL_PASS_EFFICIENCY),
    "passing first downs": RoleSpec(_NFL_PASS_VOLUME, _NFL_QB_USAGE, _NFL_PASS_EFFICIENCY),
    "attempts": RoleSpec(_NFL_PASS_VOLUME, _NFL_QB_USAGE),
    "completions": RoleSpec(_NFL_PASS_VOLUME, _NFL_QB_USAGE),
    "sacks taken": RoleSpec(_NFL_PASS_VOLUME, _NFL_QB_USAGE),
    "rushing yards": RoleSpec(
        _NFL_RUN_VOLUME, ("Player carry share",), ("Player yards per carry",)
    ),
    "carries": RoleSpec(_NFL_RUN_VOLUME, ("Player carry share",)),
    "yards": RoleSpec(
        _NFL_RUN_VOLUME,
        ("Player carry share", "Player target share"),
        ("Player yards per carry", "Player yards per target"),
    ),
    "fantasy points prizepicks": RoleSpec(_NFL_RUN_VOLUME, _NFL_QB_USAGE),
    "fantasy points underdog": RoleSpec(_NFL_RUN_VOLUME, _NFL_QB_USAGE),
}
ROLE_SPEC_BY_CELL: dict[tuple[str, str], RoleSpec] = {
    ("NFL", market): spec for market, spec in _NFL_CELLS.items()
}


def role_spec_for(league: str, market: str) -> RoleSpec | None:
    override = ROLE_SPEC_BY_CELL.get((league, market))
    if override is not None:
        return override
    if league in ("NBA", "WNBA"):
        usage, efficiency = _BASKETBALL_FACTORS.get(market, _BASKETBALL_DEFAULT)
        return RoleSpec(_BASKETBALL_VOLUME, usage, efficiency)
    if league == "NHL":
        return _NHL_DEFAULT
    return None
