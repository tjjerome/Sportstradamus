"""Combo-market pricing kernel: scrambled-Sobol NORTA for weighted component sums.

DFS combo markets — ``qb yards`` = passing yards + rushing yards, ``qb tds``,
the multi-stat fantasy scores — settle on a weighted sum of player stats that
no sportsbook quotes directly. This kernel prices ``P(sum < line)`` honestly
from the priced components: a scrambled-Sobol point set pushed through a
Gaussian copula (NORTA) couples up to :data:`MAX_COMPONENTS` marginals, each
inverted under the exact ``helpers.distributions.get_odds`` parameterization
for its family, and the weighted draw vector is sorted once so every offered
alt-line reads off the same sample. Pairwise correlation comes from the
``same_player`` scope of ``corr_market_summary.parquet`` via
:func:`load_same_player_rho`; a missing pair means independence. On the
motivating qb-yards example the same-player residual pass/rush correlation is
+0.154 (measured; not negative as previously documented here), so independence
understates the combined variance — the copula puts it back. Mechanism, draw
counts, and the floor-inversion rule follow the combo-sum pricing research
brief (max measured tail error 0.0004-0.0053 on every real combo spec).
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from importlib import resources as pkg_resources

import numpy as np
import pandas as pd
from scipy.stats import gamma, nbinom, norm, poisson, qmc, skewnorm

from sportstradamus import data
from sportstradamus.helpers.distributions import (
    _dp_log_pmf_grid,
    _dp_mu_from_mean,
    skewnormal_loc_from_mean,
)

# Frozen scramble seed: identical inputs must price identically across calls,
# cron processes, and the backtest (RQMC practice per the brief; never derived
# from the clock).
_SOBOL_SEED = 20260830

# Draw counts must be powers of two (Sobol balance property): 8192 for d <= 4,
# 16384 for d >= 5 per the brief's error tables. No antithetic on top — the
# scramble already balances and antithetic measured worse at every n.
_SOBOL_SMALL_D = 4
_SOBOL_LOG2_SMALL = 13
_SOBOL_LOG2_LARGE = 14

# The QMC error tables were measured to d = 8-9; past that the scrambled-net
# advantage must be re-measured, so the kernel refuses rather than extrapolates.
MAX_COMPONENTS = 9

# Copula-uniform clip: keeps norm.ppf finite at the scrambled corners and the
# scipy ppfs off their q = 0 / q = 1 degeneracies (which return -1 / inf).
_U_EPS = 1e-12

# NORTA feasibility clamp on any pairwise Gaussian rho (Ghosh & Henderson
# 2003): some marginal/correlation combinations are unattainable, so clamp
# instead of raising.
_RHO_CLAMP = 0.99

# Frozen seed for spec-builder post hooks that need their own draws (e.g. the
# MLB compound-multinomial hit-type split): construct a fresh
# ``np.random.default_rng(POST_RNG_SEED)`` inside the hook on every quote so
# repeated quotes reproduce bit-identically.
POST_RNG_SEED = 20260831

# Copied from prediction/joint.py (_nearest_psd + psd_or_none, Higham 2002
# eigenvalue clip): helpers must not import prediction, so consolidation into
# one home is deferred to the joint.py retirement phase.
# pylint: disable=duplicate-code
_PSD_EIG_TOLERANCE = 1e-4


def _psd_repaired(corr: np.ndarray) -> np.ndarray:
    """Return ``corr`` eigenvalue-clip repaired to PSD with unit diagonal.

    Pairwise-assembled correlation matrices carry no PSD guarantee; sub-floor
    matrices are symmetrized, eigenvalues clipped at the floor, and the
    diagonal rescaled back to 1.
    """
    if np.min(np.linalg.eigvalsh(corr)) >= _PSD_EIG_TOLERANCE:
        return corr
    corr = (corr + corr.T) / 2
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.clip(eigvals, _PSD_EIG_TOLERANCE, None)
    repaired = (eigvecs * eigvals) @ eigvecs.T
    scale = 1.0 / np.sqrt(np.diag(repaired))
    return repaired * scale[:, None] * scale[None, :]


@dataclass(frozen=True)
class ComboComponent:
    """One weighted component of a combo market, in ``get_odds`` parameter language.

    ``mean`` is the base-distribution mean (the value ``get_ev`` returns);
    ``gate`` is the zero-inflation gate applied on top of it, exactly as
    ``get_odds`` applies it. ``weight`` may be negative (TOV -1, runs allowed
    -3), fractional, or 0.0 — a zero-weight component is still sampled (it
    participates in the copula and is visible to the ``post`` hook by name)
    but contributes nothing to the weighted sum. ``dist`` additionally admits
    ``"Bernoulli"`` (mean = success probability; used for win components),
    which has no ``get_odds`` counterpart.
    """

    market: str
    weight: float
    mean: float
    dist: str
    cv: float
    sigma: float | None = None
    skew: float | None = None
    gate: float | None = None


@dataclass
class ComboSum:
    """Priced weighted-sum distribution from one NORTA sample.

    ``mean`` is the exact weighted sum of component marginal means, plus the
    sample mean of the ``post`` term when a hook was supplied; ``sd`` and
    ``under_prob`` read the sorted draws.
    """

    mean: float
    sd: float
    draws_sorted: np.ndarray = field(repr=False)

    def under_prob(self, line: float) -> float:
        """P(sum < line), splitting exact push mass evenly like ``get_odds``."""
        lo = np.searchsorted(self.draws_sorted, line, side="left")
        hi = np.searchsorted(self.draws_sorted, line, side="right")
        return float((lo + hi) / (2 * self.draws_sorted.size))


# get_odds applies a zero-inflation gate only for these families (ZINB but not
# NegBin, ZAGamma but not Gamma; SkewNormal/Normal gate on the historical zero
# rate). Mirror its dispatch exactly.
_GATED_FAMILIES = frozenset({"ZINB", "ZAGamma", "SkewNormal", "Normal"})


def _effective_gate(c: ComboComponent) -> float | None:
    """The gate ``get_odds`` would actually apply for this component's encoding."""
    if c.gate is None or c.dist not in _GATED_FAMILIES:
        return None
    if c.dist == "ZINB" and c.cv == 1:
        return None  # the legacy cv == 1 encoding prices as bare Poisson, gate ignored
    return c.gate


def _family_ppf(u: np.ndarray, c: ComboComponent) -> np.ndarray:
    """Base-family quantile at ``u`` under ``get_odds``'s exact parameterization."""
    # style: allow-complexity — flat per-family dispatch mirroring get_odds' own
    # branch shape; splitting it would scatter one parameterization contract.
    if c.dist == "Poisson" or (c.dist in ("NegBin", "ZINB") and c.cv == 1):
        return poisson.ppf(u, c.mean)
    if c.dist in ("NegBin", "ZINB"):
        r = 1.0 / c.cv
        return nbinom.ppf(u, r, r / (r + c.mean))
    if c.dist == "DPO":
        phi = np.array([1.0 / (1.0 + c.cv * c.mean)])
        grid, log_pmf = _dp_log_pmf_grid(_dp_mu_from_mean(np.array([c.mean]), phi), phi)
        cdf = np.clip(np.cumsum(np.exp(log_pmf[:, 0])), 0.0, 1.0)
        return grid[np.minimum(np.searchsorted(cdf, u), grid.size - 1)]
    if c.dist in ("SkewNormal", "Normal"):
        sigma = c.sigma if c.sigma is not None else c.mean * c.cv
        alpha = c.skew if c.dist == "SkewNormal" and c.skew is not None else 0.0
        loc = float(skewnormal_loc_from_mean(c.mean, sigma, alpha))
        if alpha == 0.0:
            return norm.ppf(u, loc=loc, scale=sigma)
        return skewnorm.ppf(u, alpha, loc=loc, scale=sigma)
    if c.dist in ("Gamma", "ZAGamma"):
        alpha = 1.0 / c.cv**2
        return gamma.ppf(u, alpha, scale=c.mean / alpha)
    if c.dist == "Bernoulli":
        return (u > 1.0 - c.mean).astype(float)
    raise ValueError(f"combo kernel cannot invert family {c.dist!r}")


def _component_ppf(u: np.ndarray, c: ComboComponent) -> np.ndarray:
    """Marginal quantile u -> x for one component.

    Discrete families use the generalized (floor) inverse ``min{k : F(k) >= u}``
    so draws stay on the settlement lattice — randomized inversion belongs to
    diagnostics, never to the sampler (brief §7). Zero inflation mirrors
    ``get_odds``'s ``F(x) = gate + (1 - gate) * F_base(x)``: the gate's mass
    lands on the zero atom and the remainder rescales through the base family.
    """
    gate = _effective_gate(c)
    if gate is None:
        return _family_ppf(u, c)
    x = np.zeros_like(u)
    live = u > gate
    x[live] = _family_ppf((u[live] - gate) / (1.0 - gate), c)
    return x


def _marginal_mean(c: ComboComponent) -> float:
    """E[X] of the component marginal (the gate deflates the base mean)."""
    gate = _effective_gate(c)
    return (1.0 - gate) * c.mean if gate is not None else c.mean


def _component_sd(c: ComboComponent) -> float:
    """Family-convention sd, used only to rank components for Sobol dimensions."""
    if c.dist == "Bernoulli":
        return float(np.sqrt(max(c.mean * (1.0 - c.mean), 0.0)))
    if c.dist in ("SkewNormal", "Normal"):
        return float(c.sigma) if c.sigma is not None else abs(c.mean) * c.cv
    if c.dist in ("Gamma", "ZAGamma"):
        return abs(c.mean) * c.cv
    if c.dist == "Poisson" or (c.dist in ("NegBin", "ZINB") and c.cv == 1):
        return float(np.sqrt(max(c.mean, 0.0)))
    # NegBin r = 1/cv and DPO phi = 1/(1 + cv*ev) share var = mean*(1 + cv*mean).
    return float(np.sqrt(max(c.mean * (1.0 + c.cv * c.mean), 0.0)))


_Z_CACHE: dict[int, np.ndarray] = {}


def _base_gaussian(d: int) -> np.ndarray:
    """Gaussianized scrambled-Sobol point set for dimension ``d``, cached per d."""
    if d not in _Z_CACHE:
        m = _SOBOL_LOG2_SMALL if d <= _SOBOL_SMALL_D else _SOBOL_LOG2_LARGE
        u = qmc.Sobol(d=d, scramble=True, seed=_SOBOL_SEED).random_base2(m)
        _Z_CACHE[d] = norm.ppf(np.clip(u, _U_EPS, 1.0 - _U_EPS))
    return _Z_CACHE[d]


def combo_sum_quote(
    components: Sequence[ComboComponent],
    rho: Callable[[str, str], float],
    post: Callable[[dict[str, np.ndarray]], np.ndarray] | None = None,
) -> ComboSum:
    """Price a weighted component sum by scrambled-Sobol NORTA (the brief's one rail).

    The pairwise ``rho(market_a, market_b)`` matrix (symmetric, unit diagonal,
    entries clamped for NORTA feasibility) is PSD-repaired, Cholesky-coloured
    onto the cached Sobol Gaussians — the highest ``|weight * sd|`` component
    occupies dimension 0, where the Sobol sequence equidistributes best — and
    each column is mapped through its component's marginal quantile.

    ``post`` receives the per-component draw vectors keyed by market name and
    returns one additive term vector of the same length — the home of
    deterministic functionals of sampled components (the MLB quality-start
    indicator on the sampled outs/runs pair, the compound-multinomial hit-type
    split of a sampled ``hits``). Hooks that need randomness must build their
    own ``np.random.default_rng(POST_RNG_SEED)`` per call so quotes stay
    reproducible.
    """
    comps = sorted(components, key=lambda c: -abs(c.weight) * _component_sd(c))
    d = len(comps)
    if not 0 < d <= MAX_COMPONENTS:
        raise ValueError(f"combo kernel supports 1..{MAX_COMPONENTS} components, got {d}")
    corr = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            r = float(np.clip(rho(comps[i].market, comps[j].market), -_RHO_CLAMP, _RHO_CLAMP))
            corr[i, j] = corr[j, i] = r
    z = _base_gaussian(d) @ np.linalg.cholesky(_psd_repaired(corr)).T
    u = np.clip(norm.cdf(z), _U_EPS, 1.0 - _U_EPS)

    draws = {c.market: _component_ppf(u[:, k], c) for k, c in enumerate(comps)}
    total = np.zeros(u.shape[0])
    for c in comps:
        total += c.weight * draws[c.market]
    mean = sum(c.weight * _marginal_mean(c) for c in comps)
    if post is not None:
        extra = np.asarray(post(draws), dtype=float)
        total = total + extra
        mean += float(extra.mean())
    total.sort()
    return ComboSum(mean=float(mean), sd=float(total.std()), draws_sorted=total)


_RHO_CACHE: dict[str, dict[tuple[str, str], float]] = {}


def load_same_player_rho(league: str) -> Callable[[str, str], float]:
    """Same-player market-pair rho lookup for :func:`combo_sum_quote`.

    Reads the ``scope == "same_player"`` rows of the league's
    ``corr_market_summary.parquet`` (written by ``training/correlate.py``) into
    a symmetric pair table, cached per league. A missing file, scope, or pair
    reads as 0 (independence): the measured cost of one dropped pair is at
    most ~0.012 of tail probability and typically under 0.005.
    """
    key = league.lower()
    if key not in _RHO_CACHE:
        path = pkg_resources.files(data) / "leagues" / key / "corr_market_summary.parquet"
        pairs: dict[tuple[str, str], float] = {}
        try:
            summary = pd.read_parquet(str(path))
        except FileNotFoundError:
            summary = pd.DataFrame(columns=["market_a", "market_b", "rho_mean", "scope"])
        for row in summary[summary["scope"] == "same_player"].itertuples(index=False):
            pairs[(row.market_a, row.market_b)] = float(row.rho_mean)
            pairs[(row.market_b, row.market_a)] = float(row.rho_mean)
        _RHO_CACHE[key] = pairs
    pairs = _RHO_CACHE[key]

    def rho(market_a: str, market_b: str) -> float:
        if market_a == market_b:
            return 1.0
        return pairs.get((market_a, market_b), 0.0)

    return rho
