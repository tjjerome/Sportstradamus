"""Data-driven non-regression pins for calibration methods graduated into the posthoc pool.

Every method graduated into the single-valued ``posthoc`` calibration pool
(:data:`sportstradamus.training.posthoc.STRUCTURAL_STAGE`) pins its passing pilot cell
here as **one row** of :data:`PILOT_PINS` — never a bespoke per-pilot test function and
never a market-specific production branch. The test drives each pilot through the same
generic pool path every other cell uses: selecting the method by its slug in the ``posthoc``
field resolves the structural strategy identity, its frozen recipe, and its applicability for
that exact ``(league, market)``. A method whose pilot needs an ``if league/market == …``
branch to stay green has a broken generalization — fix the method so the pilot's behavior
emerges from its own data, then pin the emergent result here.

Later graduation stages extend a row's ``expected`` with the pilot's known-good artifact
(blob hash / decoded CDF endpoints / ship verdict); Stage A ships the harness with the
pilot rows still empty — the two NFL pilots land in Stages B (two-part) and C (affine).
"""

from __future__ import annotations

from typing import TypedDict

import pytest

from sportstradamus.training.model_strategy_registry import (
    SWEEP_CAPABILITIES,
    CellContext,
    distribution_class,
    get_strategy,
    strategies_for_cell,
    strategy_controls,
)
from sportstradamus.training.posthoc import POSTHOC_SLUGS, STRUCTURAL_STAGE
from sportstradamus.training.role_specs import role_spec_for


class PilotPin(TypedDict, total=False):
    """One graduated method's pilot cell and its expected pool-path behavior."""

    league: str
    market: str
    method: str
    distribution: str
    # Filled by later stages: the pilot's known-good calibration blob hash / verdict.
    expected: dict


# One row per graduated method's passing pilot. Stage C (affine → NFL/rushing yards)
# adds its row once the market-agnostic conversion lands.
PILOT_PINS: tuple[PilotPin, ...] = (
    {
        "league": "NFL",
        "market": "receiving yards",
        "method": "role-position-two-part-groupcdf-fixedlinear-v3",
        "distribution": "SkewNormal",
    },
)


def _pilot_cell(pin: PilotPin) -> CellContext:
    """The pilot's registry context with the role columns its real matrix carries."""
    spec = role_spec_for(pin["league"], pin["market"])
    columns = frozenset(spec.all_columns) if spec is not None else frozenset({"Player position"})
    return CellContext(
        pin["league"],
        pin["market"],
        pin["distribution"],
        distribution_class(pin["distribution"]),
        columns,
        "pilot-matrix-sha",
    )


def test_graduated_methods_are_pool_members():
    """Both structural methods are single-valued, mutually-exclusive members of the posthoc pool."""
    assert STRUCTURAL_STAGE <= POSTHOC_SLUGS
    for slug in STRUCTURAL_STAGE:
        spec = get_strategy(slug)
        assert spec.is_structural
        # The method's own slug is the value of the single ``posthoc`` field it rides.
        assert spec.fixed_controls["posthoc"] == slug


@pytest.mark.parametrize("pin", PILOT_PINS, ids=lambda pin: f"{pin['league']}-{pin['market']}")
def test_pilot_reproduces_through_the_pool(pin: PilotPin):
    """Selecting the pilot's method by slug in ``posthoc`` resolves its structural identity — no
    market-specific branch. Later stages assert the pilot's known-good artifact against
    ``pin['expected']``.
    """
    method = pin["method"]
    assert method in STRUCTURAL_STAGE
    spec = get_strategy(method)
    cell = _pilot_cell(pin)
    # The pool selects this method for the pilot cell through the generic applicability path.
    assert spec in strategies_for_cell(cell, required_capabilities=SWEEP_CAPABILITIES)
    # Its frozen recipe carries the method's slug in the single ``posthoc`` field.
    assert strategy_controls(spec) == (dict(spec.fixed_controls),)
    assert spec.fixed_controls["posthoc"] == method
