"""Characterization pins for ``prediction.parlay.assign_parlay_families``.

Pure deterministic clustering (no RNG): golden labels captured from the
pre-refactor implementation. Cases exercise the early-return (n below the
clustering floor), the pairwise RKHS-cosine distance build, the
silhouette k-selection across k=2 and k=3 winners, and the final
collapse-to-single decision.
"""

import numpy as np

from sportstradamus.prediction.parlay import assign_parlay_families


def _block_corr(blocks, off=0.8, n=6):
    C = np.eye(n)
    for blk in blocks:
        for i in blk:
            for j in blk:
                if i != j:
                    C[i, j] = off
    return C


def test_below_clustering_floor_returns_single():
    out = assign_parlay_families([(0, 1), (0, 2), (1, 2), (3, 4)], np.eye(6))
    assert out.tolist() == [1, 1, 1, 1]


def test_two_blocks_split_into_two_families():
    C = _block_corr([[0, 1, 2], [3, 4, 5]])
    bet_ids = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]
    assert assign_parlay_families(bet_ids, C).tolist() == [1, 1, 1, 2, 2, 2]


def test_uniform_blob_still_splits_by_shared_legs():
    C = np.full((6, 6), 0.5)
    np.fill_diagonal(C, 1.0)
    bet_ids = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]
    assert assign_parlay_families(bet_ids, C).tolist() == [2, 2, 2, 1, 1, 1]


def test_three_blocks_select_three_families():
    C = _block_corr([[0, 1], [2, 3], [4, 5]], off=0.9)
    bet_ids = [(0,), (1,), (2,), (3,), (4,), (5,)]
    assert assign_parlay_families(bet_ids, C).tolist() == [2, 2, 3, 3, 1, 1]


def test_identical_parlays_collapse_to_single():
    C = np.full((6, 6), 0.8)
    np.fill_diagonal(C, 1.0)
    bet_ids = [(0, 1)] * 6
    assert assign_parlay_families(bet_ids, C).tolist() == [1, 1, 1, 1, 1, 1]
