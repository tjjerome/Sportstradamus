"""Pure routing and support guards for the surviving NFL-yards candidates."""

from __future__ import annotations

from collections.abc import Mapping

AUTO = "auto"
NONE = "none"
TWO_PART_STRATEGY = (
    "role-position-two-part-groupcdf-fixedlinear-v3"
)
AFFINE_STRATEGY = "affine-groupcdf-bookpool-v1"

NFL_YARDS_EXPERIMENTS: dict[str, tuple[str, str]] = {
    TWO_PART_STRATEGY: ("NFL", "receiving yards"),
    AFFINE_STRATEGY: ("NFL", "rushing yards"),
}
EXPERIMENT_CHOICES: tuple[str, ...] = (AUTO, NONE, *NFL_YARDS_EXPERIMENTS)
AFFINE_EXPERT_EXPERIMENTS: frozenset[str] = frozenset({AFFINE_STRATEGY})

ROLE_COLUMNS: tuple[str, ...] = (
    "Team plays_per_game",
    "Team pass_rate",
    "Player target share",
    "Player yards per target",
)
TWO_PART_POSITIONS: dict[int, str] = {2: "WR", 3: "RB", 4: "TE"}
AFFINE_POSITIONS: dict[int, str] = {1: "QB", 3: "RB"}
TWO_PART_SUPPORT: dict[str, int] = {
    "train_rows": 4000,
    "validation_rows": 400,
    "test_rows": 400,
    "train_players": 100,
    "validation_players": 100,
}
AFFINE_SUPPORT: dict[str, int] = {
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
    experiment = cell.get("structural_calibration", NONE) if flag_value == AUTO else flag_value
    if not isinstance(experiment, str):
        raise ValueError("structural_calibration must be a string")
    return experiment


def validate_two_part_recipe(
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
