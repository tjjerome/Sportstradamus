"""Tests for archived ``deprecated.parlay_families.assign_parlay_families``.

Moved here from ``tests/golden/test_parlay_families_characterization.py`` and
``tests/golden/test_parlay_search.py`` when Task 0.2 of the P8
leg-schema-retirement plan archived the family-clustering code (its only
production call site — the ``Family`` column build in ``correlation.py`` —
was deleted). See ``src/deprecated/parlay_families.py`` and
``src/deprecated/README.md``.

Pure deterministic clustering (no RNG). The cluster *partition* is the
invariant; the integer label ids are not stable across sklearn/BLAS versions,
so the multi-family cases compare membership via :func:`_families` rather than
raw labels. Cases exercise the early-return (n below the clustering floor), the
pairwise RKHS-cosine distance build, the silhouette k-selection across k=2 and
k=3 winners, and the final collapse-to-single decision.
"""

from __future__ import annotations

import numpy as np

from deprecated.parlay_families import assign_parlay_families


def _families(labels):
    """Partition leg indices by family label, dropping the arbitrary label ids."""
    groups: dict[int, set[int]] = {}
    for idx, label in enumerate(labels):
        groups.setdefault(label, set()).add(idx)
    return frozenset(frozenset(g) for g in groups.values())


def _block_corr(blocks, off=0.8, n=6):
    C = np.eye(n)
    for blk in blocks:
        for i in blk:
            for j in blk:
                if i != j:
                    C[i, j] = off
    return C


def _grouped_c(groups: list[list[int]], intra: float, cross: float) -> np.ndarray:
    """Leg correlation matrix: ``intra`` within a group, ``cross`` between."""
    n = sum(len(g) for g in groups)
    of_group = {leg: gi for gi, g in enumerate(groups) for leg in g}
    c = np.eye(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                c[i, j] = intra if of_group[i] == of_group[j] else cross
    return c


def test_below_clustering_floor_returns_single():
    out = assign_parlay_families([(0, 1), (0, 2), (1, 2), (3, 4)], np.eye(6))
    assert out.tolist() == [1, 1, 1, 1]


def test_two_blocks_split_into_two_families():
    C = _block_corr([[0, 1, 2], [3, 4, 5]])
    bet_ids = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]
    assert _families(assign_parlay_families(bet_ids, C).tolist()) == _families([1, 1, 1, 2, 2, 2])


def test_uniform_blob_still_splits_by_shared_legs():
    C = np.full((6, 6), 0.5)
    np.fill_diagonal(C, 1.0)
    bet_ids = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]
    assert _families(assign_parlay_families(bet_ids, C).tolist()) == _families([1, 1, 1, 2, 2, 2])


def test_three_blocks_select_three_families():
    C = _block_corr([[0, 1], [2, 3], [4, 5]], off=0.9)
    bet_ids = [(0,), (1,), (2,), (3,), (4,), (5,)]
    assert _families(assign_parlay_families(bet_ids, C).tolist()) == _families([1, 1, 2, 2, 3, 3])


def test_identical_parlays_collapse_to_single():
    C = np.full((6, 6), 0.8)
    np.fill_diagonal(C, 1.0)
    bet_ids = [(0, 1)] * 6
    assert assign_parlay_families(bet_ids, C).tolist() == [1, 1, 1, 1, 1, 1]


def test_assign_families_splits_two_independent_groups() -> None:
    """Two leg blocks with ~0 cross-correlation → exactly two families."""
    c = _grouped_c([[0, 1, 2], [3, 4, 5]], intra=0.8, cross=0.0)
    bet_ids = [
        (0, 1),
        (0, 2),
        (1, 2),
        (0, 1, 2),  # group A
        (3, 4),
        (3, 5),
        (4, 5),
        (3, 4, 5),  # group B
    ]
    labels = assign_parlay_families(bet_ids, c)

    assert len(labels) == 8
    assert len(set(labels)) == 2
    assert len(set(labels[:4])) == 1  # all A parlays together
    assert len(set(labels[4:])) == 1  # all B parlays together
    assert set(labels[:4]).isdisjoint(set(labels[4:]))


def test_assign_families_collapses_when_unseparable() -> None:
    """No separable structure (every parlay mutually identical) → one family.

    All parlays share the same legs, so every pairwise distance is 0 and the
    best silhouette cannot clear ``_PARLAY_MIN_SILHOUETTE`` — the honest
    answer is a single family, the exact degeneracy the audit flagged.
    """
    c = _grouped_c([[0, 1, 2]], intra=0.8, cross=0.8)
    bet_ids = [(0, 1, 2)] * 8
    labels = assign_parlay_families(bet_ids, c)
    assert len(labels) == 8
    assert set(labels) == {1}


def test_assign_families_handles_non_psd_without_nan() -> None:
    """Indefinite C (negative off-diagonals) yields valid labels, no nan."""
    c = _grouped_c([[0, 1, 2], [3, 4, 5]], intra=0.6, cross=0.0)
    c[2, 5] = c[5, 2] = -0.95
    c[1, 4] = c[4, 1] = -0.9
    assert np.min(np.linalg.eigvalsh(c)) < 0, "test setup invalid"
    bet_ids = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (0, 3), (2, 5)]

    labels = assign_parlay_families(bet_ids, c)
    assert len(labels) == 8
    assert np.all(np.isfinite(labels))
    assert labels.min() >= 1
    assert 1 <= len(set(labels)) <= 4


def test_assign_families_tiny_input_is_single_family() -> None:
    """At/under the clustering minimum (5), everything is family 1."""
    c = np.eye(3)
    bet_ids = [(0, 1), (1, 2), (0, 2), (0, 1), (1, 2)]
    labels = assign_parlay_families(bet_ids, c)
    assert len(labels) == 5
    assert set(labels) == {1}


def test_assign_families_is_deterministic() -> None:
    """Identical inputs produce identical labels across calls."""
    c = _grouped_c([[0, 1, 2], [3, 4, 5]], intra=0.8, cross=0.0)
    bet_ids = [(0, 1), (0, 2), (1, 2), (0, 1, 2), (3, 4), (3, 5), (4, 5), (3, 4, 5)]
    a = assign_parlay_families(bet_ids, c)
    b = assign_parlay_families(bet_ids, c)
    assert np.array_equal(a, b)
