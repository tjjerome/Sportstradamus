"""What ``canonical_signature`` covers, and — more importantly — what it must not.

The signature answers "did this strategy's implementation change?". It once hashed the whole
spec, so adding one value to a shared search axis rotated every family's signature and, through
``corner_fingerprint``, orphaned every model pickle, board row, and ledger row at once. These pin
the scope so that cannot recur: a search-space edit is invisible to the hash, a version bump is
not. Corner membership (``strategy_controls``) is what rejects an artifact whose controls left the
pool, and it needs no help from the signature to do it.
"""

import dataclasses

import pytest

from sportstradamus.training.model_strategy import (
    corner_fingerprint,
    get_strategy,
    registered_strategies,
    strategy_controls,
)

_MATRIX_HASH = "matrix-sha"


def _searchable_specs():
    return [spec for spec in registered_strategies() if spec.axes]


@pytest.mark.parametrize("spec", _searchable_specs(), ids=lambda spec: spec.slug)
def test_widening_a_search_axis_leaves_the_signature_and_fingerprints_alone(spec):
    axis = next(iter(spec.axes))
    wider = dataclasses.replace(
        spec, axes={**spec.axes, axis: (*spec.axes[axis], "a-value-nobody-has-trained")}
    )
    corner = strategy_controls(spec)[0]

    assert wider.canonical_signature == spec.canonical_signature
    assert corner_fingerprint(wider, corner, _MATRIX_HASH) == corner_fingerprint(
        spec, corner, _MATRIX_HASH
    )


@pytest.mark.parametrize("spec", _searchable_specs(), ids=lambda spec: spec.slug)
def test_cli_and_persist_maps_are_not_signed(spec):
    flag = next(iter(spec.cli_flags))
    renamed = dataclasses.replace(spec, cli_flags={**spec.cli_flags, flag: "--renamed"})
    repersisted = dataclasses.replace(spec, persist={**spec.persist, "dist": "elsewhere"})

    assert renamed.canonical_signature == spec.canonical_signature
    assert repersisted.canonical_signature == spec.canonical_signature


@pytest.mark.parametrize("spec", registered_strategies(), ids=lambda spec: spec.slug)
def test_a_version_bump_rotates_the_signature(spec):
    bumped = dataclasses.replace(spec, implementation_version=spec.implementation_version + 1)
    reschemad = dataclasses.replace(
        spec, artifact_schema_version=spec.artifact_schema_version + 1
    )

    assert bumped.canonical_signature != spec.canonical_signature
    assert reschemad.canonical_signature != spec.canonical_signature


def test_families_keep_distinct_signatures():
    signatures = {spec.slug: spec.canonical_signature for spec in registered_strategies()}
    assert len(set(signatures.values())) == len(signatures)


@pytest.mark.parametrize("spec", _searchable_specs(), ids=lambda spec: spec.slug)
def test_membership_still_rejects_a_corner_whose_axis_value_was_pulled(spec):
    axis, values = next((axis, vals) for axis, vals in spec.axes.items() if len(vals) > 1)
    corner = next(c for c in strategy_controls(spec) if c[axis] == values[0])
    narrowed = dataclasses.replace(spec, axes={**spec.axes, axis: values[1:]})

    # The signature no longer notices, so this membership test is the whole guard.
    assert narrowed.canonical_signature == get_strategy(spec.slug).canonical_signature
    assert corner not in strategy_controls(narrowed)
