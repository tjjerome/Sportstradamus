"""The ``posthoc`` axis is a signed control, and the unlocked distribution class stays honest.

Part E of the sweep expansion is a verification pass: the identity machinery already carries
``posthoc`` end to end, so these tests pin the properties the wider search now depends on rather
than adding any machinery to produce them.
"""

import pytest

from sportstradamus.training.model_strategy import (
    CellContext,
    controls_json,
    corner_fingerprint,
    get_strategy,
    registered_strategies,
    strategy_cli_args,
    strategy_controls,
    validate_model_recipe,
    validate_strategy_selection,
)
from sportstradamus.training.model_strategy.registry import CAP_DETERMINISTIC_TRAIN
from sportstradamus.training.posthoc import CDF_STAGE, POSTHOC_SLUGS, STRUCTURAL_STAGE

_MATRIX_HASH = "matrix-sha"


def _searchable_specs():
    return [spec for spec in registered_strategies() if spec.axes]


def _model_filedict(spec, controls: dict[str, str]) -> dict[str, object]:
    """A filedict that agrees with ``controls`` on every field the recipe validator reads."""
    model = {
        "distribution": controls["dist"],
        "sn_param": controls.get("sn_param"),
        "zinb_mode": controls.get("zinb_mode"),
        "posthoc": controls["posthoc"],
    }
    normalization = controls.get("normalization")
    model["target_normalization"] = normalization or "none"
    model["normalized"] = normalization in ("ratio_meanyr", "ratio_projvol")
    model.update(spec.fixed_persist)
    return model


@pytest.mark.parametrize("spec", _searchable_specs(), ids=lambda spec: spec.slug)
def test_posthoc_is_a_signed_axis_on_every_searchable_family(spec):
    assert "posthoc" in spec.axes
    assert spec.persist["posthoc"] == "posthoc"
    assert set(spec.axes["posthoc"]) <= POSTHOC_SLUGS - STRUCTURAL_STAGE


@pytest.mark.parametrize("spec", _searchable_specs(), ids=lambda spec: spec.slug)
def test_cdf_recalibration_is_offered_to_skewnormal_alone(spec):
    # pipeline gates the whole-CDF stage on ``dist == "SkewNormal"``; on any other family the
    # corner would train identically to ``none`` while carrying its own fingerprint.
    offered = set(spec.axes["posthoc"]) & CDF_STAGE
    assert offered == (CDF_STAGE if spec.slug == "SkewNormal" else set())


def test_corners_differing_only_in_posthoc_share_a_signature_but_not_a_fingerprint():
    spec = get_strategy("SkewNormal")
    corners = strategy_controls(spec)
    base = corners[0]
    other = next(c for c in corners if c["posthoc"] != base["posthoc"] and _same_but_posthoc(c, base))

    assert corner_fingerprint(spec, base, _MATRIX_HASH) != corner_fingerprint(
        spec, other, _MATRIX_HASH
    )
    assert controls_json(base) != controls_json(other)
    # The spec did not change, so every artifact of the family stays resolvable.
    assert spec.canonical_signature == get_strategy("SkewNormal").canonical_signature


def _same_but_posthoc(left: dict[str, str], right: dict[str, str]) -> bool:
    return {k: v for k, v in left.items() if k != "posthoc"} == {
        k: v for k, v in right.items() if k != "posthoc"
    }


@pytest.mark.parametrize("spec", _searchable_specs(), ids=lambda spec: spec.slug)
def test_persisted_posthoc_drift_is_rejected(spec):
    controls = strategy_controls(spec)[0]
    validate_model_recipe(_model_filedict(spec, controls), spec, controls)

    served = next(slug for slug in spec.axes["posthoc"] if slug != controls["posthoc"])
    drifted = _model_filedict(spec, controls) | {"posthoc": served}
    with pytest.raises(ValueError, match="signed model recipe mismatch: posthoc"):
        validate_model_recipe(drifted, spec, controls)


@pytest.mark.parametrize(
    "spec", [s for s in _searchable_specs() if s.fixed_persist], ids=lambda spec: spec.slug
)
def test_count_corner_signs_target_normalization_none(spec):
    # A count corner can now win on a cell whose stat_meta names a continuous normalization;
    # fixed_persist writes the clearing edit, and the validator holds it to that.
    assert spec.fixed_persist == {"target_normalization": "none"}
    controls = strategy_controls(spec)[0]
    assert "normalization" not in controls

    stale = _model_filedict(spec, controls) | {"target_normalization": "ratio_meanyr"}
    with pytest.raises(ValueError, match="target_normalization"):
        validate_model_recipe(stale, spec, controls)


@pytest.mark.parametrize("spec", _searchable_specs(), ids=lambda spec: spec.slug)
def test_class_unlocked_families_still_carry_a_normalization_control(spec):
    # validate_model_recipe demands target_normalization="none" from any spec that lists "count"
    # among its classes and carries no normalization control. Every family lists "count" now, so
    # dropping SkewNormal's or Mixture's normalization axis would silently start demanding "none".
    assert "count" in spec.applicability.distribution_classes
    if spec.fixed_persist:
        assert spec.fixed_persist["target_normalization"] == "none"
    else:
        assert "normalization" in spec.axes or "normalization" in spec.fixed_controls


@pytest.mark.parametrize("spec", registered_strategies(), ids=lambda spec: spec.slug)
def test_every_registered_corner_forwards_through_the_cli(spec):
    cell = CellContext(league="NBA", market="PTS", distribution=spec.slug, distribution_class="x")
    for controls in strategy_controls(spec):
        args = strategy_cli_args(cell, spec, controls)
        assert len(args) == 2 * len(controls)


def test_count_family_admission_falls_open_without_matrix_facts():
    # train_market builds its CellContext from the run's own columns and never reads the target's
    # lattice or mean, so a corner an operator forced and signed must keep validating.
    spec = get_strategy("NegBin")
    assert spec.applicability.requires_integer_target
    cell = CellContext(
        league="NFL",
        market="passing yards",
        distribution="NegBin",
        distribution_class="count",
        data_columns=frozenset(),
        matrix_sha256=_MATRIX_HASH,
    )

    validate_strategy_selection(
        cell,
        spec,
        strategy_controls(spec)[0],
        required_capabilities={CAP_DETERMINISTIC_TRAIN},
    )

    known_continuous = CellContext(**{**vars(cell), "target_is_integer": False})
    with pytest.raises(ValueError, match="not applicable/capable"):
        validate_strategy_selection(
            known_continuous,
            spec,
            strategy_controls(spec)[0],
            required_capabilities={CAP_DETERMINISTIC_TRAIN},
        )
