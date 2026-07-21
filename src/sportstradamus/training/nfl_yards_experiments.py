"""Pure routing and support guards for the surviving NFL-yards candidates."""

from __future__ import annotations

from collections.abc import Mapping

AUTO = "auto"
NONE = "none"
RECEIVING_ROLE_POSITION_TWO_PART_GROUPCDF_FIXEDLINEAR = (
    "receiving-role-position-two-part-groupcdf-fixedlinear-v3"
)
RUSHING_AFFINE_GROUPCDF_BOOK_POOL = "rushing-qb-rb-affine-groupcdf-bookpool-v1"

NFL_YARDS_EXPERIMENTS: dict[str, tuple[str, str]] = {
    RECEIVING_ROLE_POSITION_TWO_PART_GROUPCDF_FIXEDLINEAR: ("NFL", "receiving yards"),
    RUSHING_AFFINE_GROUPCDF_BOOK_POOL: ("NFL", "rushing yards"),
}
EXPERIMENT_CHOICES: tuple[str, ...] = (AUTO, NONE, *NFL_YARDS_EXPERIMENTS)
RUSHING_EXPERT_EXPERIMENTS: frozenset[str] = frozenset({RUSHING_AFFINE_GROUPCDF_BOOK_POOL})

ROLE_COLUMNS: tuple[str, ...] = (
    "Team plays_per_game",
    "Team pass_rate",
    "Player target share",
    "Player yards per target",
)
RECEIVING_POSITIONS: dict[int, str] = {2: "WR", 3: "RB", 4: "TE"}
RUSHING_POSITIONS: dict[int, str] = {1: "QB", 3: "RB"}
RECEIVING_SUPPORT: dict[str, int] = {
    "train_rows": 4000,
    "validation_rows": 400,
    "test_rows": 400,
    "train_players": 100,
    "validation_players": 100,
}
RUSHING_SUPPORT: dict[str, int] = {
    "train_rows": 1500,
    "validation_rows": 300,
    "test_rows": 300,
    "train_players": 60,
    "validation_players": 50,
}


def validate_experiment_selection(
    experiment: str,
    *,
    league: str,
    market_selection: str | None,
) -> None:
    """Reject an unresolved or explicit selector outside its one registered cell."""
    if experiment == NONE:
        return
    if experiment == AUTO:
        raise ValueError("NFL-yards experiment must be resolved per cell before training")
    if experiment not in NFL_YARDS_EXPERIMENTS:
        raise ValueError(f"unknown NFL-yards experiment {experiment!r}")

    registered_league, registered_market = NFL_YARDS_EXPERIMENTS[experiment]
    selected_markets = (
        []
        if market_selection is None
        else [part.strip() for part in market_selection.split(",") if part.strip()]
    )
    if league != registered_league or selected_markets != [registered_market]:
        raise ValueError(
            f"{experiment} is registered only for --league {registered_league} "
            f"--market {registered_market!r}"
        )


def resolve_experiment_selection(flag_value: str, cell: Mapping[str, object]) -> str:
    """Honor a persisted structural method only when the CLI selector is ``auto``."""
    experiment = cell.get("nfl_yards_experiment", NONE) if flag_value == AUTO else flag_value
    if not isinstance(experiment, str):
        raise ValueError("nfl_yards_experiment must be a string")
    return experiment


def validate_receiving_recipe(
    experiment: str,
    *,
    league: str,
    market: str,
    distribution: str,
    target_normalization: str,
    posthoc: str,
    dist_training_loss: str,
    blending: str,
    hpo_selection: str,
    sn_param: str,
    stabilization: str,
) -> None:
    """Enforce the fixed paired-control recipe registered for a yards candidate."""
    if experiment == NONE:
        return
    expected_cell = NFL_YARDS_EXPERIMENTS.get(experiment)
    if expected_cell != (league, market):
        raise ValueError(f"{experiment} is not registered for {league}/{market}")

    recipe = {
        "distribution": (distribution, "SkewNormal"),
        "target_normalization": (target_normalization, "ratio_meanyr"),
        "posthoc": (posthoc, "none"),
        "dist_training_loss": (dist_training_loss, "crps"),
        "blending": (blending, "nll"),
        "hpo_selection": (hpo_selection, "loss"),
        "sn_param": (sn_param, "direct"),
        "stabilization": (stabilization, "None"),
    }
    mismatches = [
        f"{key}={actual!r} (requires {expected!r})"
        for key, (actual, expected) in recipe.items()
        if actual != expected
    ]
    if mismatches:
        raise ValueError(f"{experiment} fixed-control recipe mismatch: {', '.join(mismatches)}")
