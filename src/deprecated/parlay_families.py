# ARCHIVED 2026-07-03 from src/sportstradamus/prediction/parlay.py
# Reason: orphaned when Task 0.2 of the P8 leg-schema-retirement plan deleted
#   its only call site (the Family column build in correlation.py).
# Last live SHA: f222044
# Original imports (now unresolved here):
#   from scipy.cluster.hierarchy import fcluster, linkage
#   from scipy.spatial.distance import squareform
#   from sklearn.metrics import silhouette_score
#   from sportstradamus.prediction.parlay import _nearest_psd

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

from sportstradamus.prediction.parlay import _nearest_psd

# Upper bound on independence families per game. Auto-selected count (by
# silhouette) may be anywhere in [1, this]; keeps the dashboard picker short.
_PARLAY_MAX_FAMILIES: int = 4

# Silhouette floor: a k>=2 split must separate at least this well, otherwise
# the survivors are one blob and we honestly report a single family.
_PARLAY_MIN_SILHOUETTE: float = 0.15

# Below this many survivor parlays, clustering is noise — all one family.
_PARLAY_MIN_FOR_CLUSTERING: int = 5

# Self-kernel floor: a parlay whose 1ᵀC1 is at/below this is treated as having
# no usable correlation signal (distance 1 to everything) instead of dividing
# by ~0.
_KERNEL_SELF_FLOOR: float = 1e-9


def _parlay_distance_matrix(idx, Cpsd, self_k, n):
    """Pairwise RKHS-cosine distance between parlays; non-finite entries floored to 1."""
    dist = np.zeros((n, n))
    for i in range(n):
        bi = idx[i]
        for j in range(i + 1, n):
            denom = np.sqrt(self_k[i] * self_k[j])
            sim = Cpsd[np.ix_(bi, idx[j])].sum() / denom if denom > 0 else np.nan
            d = 1.0 - np.clip(sim, -1.0, 1.0)
            if not np.isfinite(d):
                d = 1.0
            dist[i, j] = dist[j, i] = d
    return dist


def _select_family_count(z, dist, n, max_families):
    """Silhouette-select the family count k in [1, max_families], smallest k on ties."""
    best_k, best_score = 1, -1.0
    for k in range(2, min(max_families, n - 1) + 1):
        labels = fcluster(z, k, criterion="maxclust")
        if len(set(labels)) < 2:
            continue
        # Ascending k with strict-improvement keeps the smallest k on ties.
        score = silhouette_score(dist, labels, metric="precomputed")
        if score > best_score:
            best_score, best_k = score, k
    return best_k, best_score


def assign_parlay_families(
    bet_ids: list[tuple[int, ...]],
    C: np.ndarray,
    *,
    max_families: int = _PARLAY_MAX_FAMILIES,
) -> np.ndarray:
    """Cluster a game's survivor parlays into independence families.

    Each parlay is the indicator vector ``1_p`` over its legs; parlay
    similarity is the RKHS cosine ``(1_pᵀ C 1_q) / sqrt((1_pᵀ C 1_p)·(1_qᵀ C
    1_q))`` — the diagonal enters numerator-self and denominator consistently,
    so it is a true bounded cosine (unlike the legacy mean-with-diagonal vs.
    mean-without). ``C`` is projected to the nearest PSD first so every
    self-kernel is non-negative and the cosine is well defined; non-finite
    distances are floored to 1. Families are formed by average-linkage
    hierarchical clustering and the cluster count is silhouette-selected in
    ``[1, max_families]`` — weak separation honestly collapses to one family.

    Args:
        bet_ids: One tuple of integer leg indices into ``C`` per parlay.
        C: Per-game leg×leg signed correlation matrix (unit diagonal).
        max_families: Upper bound on the auto-selected family count.

    Returns:
        np.ndarray: Integer labels ``1..k`` (length ``len(bet_ids)``).
    """
    n = len(bet_ids)
    single = np.ones(n, dtype=int)
    if n <= _PARLAY_MIN_FOR_CLUSTERING:
        return single

    Cpsd = _nearest_psd(np.asarray(C, dtype=float))
    idx = [np.asarray(b, dtype=int) for b in bet_ids]
    self_k = np.array([Cpsd[np.ix_(b, b)].sum() for b in idx])
    self_k = np.where(np.isfinite(self_k) & (self_k > _KERNEL_SELF_FLOOR), self_k, np.nan)

    dist = _parlay_distance_matrix(idx, Cpsd, self_k, n)

    condensed = squareform(dist, checks=False)
    if not np.all(np.isfinite(condensed)):
        return single

    z = linkage(condensed, method="average")
    best_k, best_score = _select_family_count(z, dist, n, max_families)

    if best_k >= 2 and best_score >= _PARLAY_MIN_SILHOUETTE:
        return fcluster(z, best_k, criterion="maxclust").astype(int)
    return single
