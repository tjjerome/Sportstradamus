"""Distributional math for model/bookmaker fusion.

Three related jobs live here:

1. **Odds-space conversion**: ``odds_to_prob`` / ``prob_to_odds`` /
   ``no_vig_odds`` convert between American odds and raw probabilities and
   strip the bookmaker's hold ("vig") so both sides sum to 1.
2. **Distribution inversion**: ``get_ev`` solves for the base-distribution
   mean that matches a (line, under-probability) pair under a given
   distribution family. ``get_odds`` runs the forward direction — given a
   mean, return P(outcome < line). Both handle zero-inflation gates the
   same way: the gate contributes ``gate`` to the CDF at zero and the
   remaining ``(1-gate)`` is the base distribution.
3. **Model/book fusion**: ``fused_loc`` blends the model's per-observation
   distribution parameters with the bookmaker's implied distribution
   using either a log-opinion pool (NegBin, DPO) or a precision-weighted
   blend (Gamma, SkewNormal) with weight ``w`` on the model. See CLAUDE.md
   for the math and the diagnostic block that validates this.

``set_model_start_values`` prepares LightGBMLSS start values from the
training matrix's per-player historical moments; kept here because it
shares the distribution-family dispatch with the inversion code.
``apply_temperature`` / ``apply_cdf_recal`` apply a fitted calibration map at
serve time; the fitters themselves live in ``training.posthoc``.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import beta as beta_fn
from scipy.special import expit, gammaln, logit, logsumexp, xlogy
from scipy.stats import gamma, nbinom, norm, poisson, skewnorm

# Grid resolution for the continuous CRPS integrands (SkewNormal, gated Gamma).
# 500 points over the support is the resolution the dispersion-calibration CRPS
# settled on; the arg-min over a scalar weight/scale is invariant to it.
_CRPS_GRID_POINTS = 500

# Minimum support size for the NegBin CRPS sum: even when observed values and the
# predictive mean are small, we sum at least this many terms to cover the tail.
_NEGBIN_CRPS_K_FLOOR = 30

# Weight on the lower-bound tail violation in ``fit_distro``'s objective: the
# clamp cares far more about a left-tail breach (mass below ``lower_bound``)
# than a right-tail one, so its penalty is scaled up relative to the upper tail.
LOWER_TAIL_PENALTY = 100

# Above this raw value softplus(x) ≈ x to machine precision; ``_softplus_inv``
# returns x directly past it to dodge expm1 overflow.
SOFTPLUS_LINEAR_THRESHOLD = 20

# Historical zero rate above which a cell is treated as zero-inflated: the
# SkewNormal gate is published, NegBin escalates to ZINB, and the book gate
# joins the blend. Below it the zero mass is noise. Shared by the training and
# prediction decode paths so the two cannot drift on this cutoff.
GATE_PUBLISH_THRESHOLD = 0.02

# Historical zero rate above which a SkewNormal cell trains on nonzero rows and
# decodes its location/scale against ``MeanYr_nonzero`` instead of ``MeanYr``.
# Shared by both pipelines for the same reason as GATE_PUBLISH_THRESHOLD.
NONZERO_DENOM_GATE = 0.05

# Logit-space clip keeping temperature scaling clear of the 0/1 singularities.
LOGIT_CLIP_EPS = 1e-6

# Double Poisson support-grid + parameter bounds. Mirror the torch constants in
# ``sportstradamus.double_poisson`` (kept separate so this module's import graph
# stays torch-free); cross-consistency is pinned by the numpy-vs-torch pmf
# golden test.
_DP_KMAX_MIN = 30
_DP_GRID_SIGMAS = 12.0
_DP_KMAX_CAP = 1024
_DP_PHI_FLOOR = 0.02
_DP_PHI_CEILING = 25.0
# Newton (multiplicative fixed-point) inversion of the exact DP mean back to the
# mu parameter: relative tolerance and iteration cap before the brentq fallback.
_DP_NEWTON_RTOL = 1e-6
_DP_NEWTON_MAX_ITER = 8
# Moment start values seed phi at clip(mean/var, floor, ceiling): a moment
# estimate outside this band is small-sample noise, not a real dispersion read.
_DP_PHI_SEED_FLOOR = 0.25
_DP_PHI_SEED_CEILING = 4.0
# SkewNormal precision-pool floor on the model scale, as a fraction of the book scale.
# The pool weights legs by 1/sigma^2, so a GBDT sigma far below the book's seizes the
# blend regardless of w and saturates served P to 0/1 — three such rows carried 98% of
# NFL passing yards' Gate-1 point estimate (docs/archive/researcher_passing_g1.md §6;
# 0.8 was the best measured kappa, with g4/cov50 unchanged and g5 improved).
_BLEND_MODEL_SCALE_FLOOR = 0.8


def odds_to_prob(odds):
    """Convert American odds to an implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    odds = -odds
    return odds / (odds + 100)


def prob_to_odds(p):
    """Convert a probability to American odds (integer-rounded)."""
    if p < 0.5:
        return int(np.round((1 - p) / p * 100))
    return int(np.round((p / (1 - p)) * -100))


# Power de-vig applies only to lopsided quotes; on the body (near even money) it
# is numerically identical to the proportional method, so gating it there is free.
_DEVIG_LOPSIDED_FLAG = 0.3
# Bracket ceiling for the power-exponent root-find; k stays near 1 for real vigs.
_DEVIG_POWER_MAX = 50.0


def _power_devig_exponent(o: float, u: float) -> float:
    """Solve ``o**k + u**k == 1`` for the de-vig power ``k`` (Clarke et al. 2017)."""
    if o + u <= 1:
        return 1.0
    return brentq(lambda k: o**k + u**k - 1.0, 1.0, _DEVIG_POWER_MAX, xtol=1e-10)


def no_vig_odds(over, under=None, method="proportional"):
    """Return ``[p_over, p_under]`` with the bookmaker's hold removed.

    Accepts either American odds (``|x| >= 100``) or decimal odds. When
    ``under`` is omitted, the caller is treated as offering a one-sided
    line; we fabricate an under from a conservative 6.5% vig assumption
    so the two-sided math still works.

    ``method="power"`` switches lopsided two-sided quotes (``|p_over - 0.5| >
    _DEVIG_LOPSIDED_FLAG``) to the power (logarithmic) de-vig, which corrects
    favourite-longshot bias (Clarke, Kovalchik & Ingram 2017); the body and
    every one-sided line stay on the proportional default (identical at even
    money).
    """
    o = odds_to_prob(over) if np.abs(over) >= 100 else 1 / over
    if under is None or under <= 0:
        juice = 1.0652
        u = juice - o
        return [o / juice, u / juice]

    u = odds_to_prob(under) if np.abs(under) >= 100 else 1 / under
    if method == "power" and np.abs(o / (o + u) - 0.5) > _DEVIG_LOPSIDED_FLAG:
        k = _power_devig_exponent(o, u)
        return [o**k, u**k]

    juice = o + u
    return [o / juice, u / juice]


# Per-pick fair payout for an unboosted Underdog Power pick: power[4] ** (1/4)
# = 10 ** 0.25 ≈ 1.778. Used by model_prob to convert the raw promo multiplier
# from the Underdog API into a payout-inclusive value so ``Model = P * Boost``
# yields true per-$1 expected return; consumers that want the raw modifier
# (dashboard display, ``nightly.py`` profit-sim, parlay-search arithmetic in
# ``correlation.py``) divide it back out.
UNDERDOG_BOOST_BASELINE: float = 1.78

# An unboosted symmetric DFS pick pays the same either way, so it prices even.
DFS_FAIR_PICK_PROB = 0.5


def dfs_boost_probs(odds_over, odds_under):
    """Payout-implied ``[p_over, p_under]`` from full decimal DFS payout odds.

    A two-sided offer devigs proportionally, so a symmetric pair prices at
    exactly ``DFS_FAIR_PICK_PROB`` per side and standard picks are unchanged.
    A one-sided offer — the missing side passed as 0 or NaN — is the
    platform's own claim on a moved or discounted line: the offered side
    stores its raw breakeven ``1/odds`` with no hold shave and the other side
    its complement, never a fabricated fair 50/50. Neither side offered
    prices even.
    """
    imp_over = 1.0 / odds_over if odds_over > 0 else None
    imp_under = 1.0 / odds_under if odds_under > 0 else None
    if imp_over is not None and imp_under is not None:
        juice = imp_over + imp_under
        return [imp_over / juice, imp_under / juice]
    if imp_over is not None:
        return [imp_over, 1.0 - imp_over]
    if imp_under is not None:
        return [1.0 - imp_under, imp_under]
    return [DFS_FAIR_PICK_PROB, DFS_FAIR_PICK_PROB]


# Cap the SkewNormal implied mean at this multiple of the line — the archive's
# own blown-row threshold (``BLOWN_LINE_FACTOR``). Below the corresponding
# under-prob the inversion would exceed it, so the line is used instead.
SN_MAX_MEAN_FACTOR = 5.0

# A SkewNormal's standardized skewness is bounded to |gamma| < 0.99527 (Azzalini 1985 [82]).
# Clamp a target skew just inside it so the moment-match's alpha stays finite; a count cell's
# low-mean right-skew that exceeds the bound is met in variance and finished by the integer PMF
# correction (WS2 research brief, /tmp/researcher_ws2_book_shape.md Findings 2-3).
_SN_SKEW_MAX = (
    0.9952  # a hair below the 0.99527 bound on purpose: at the bound delta=±1 and alpha=±inf
)
_SN_SKEW_C23 = ((4 - np.pi) / 2) ** (2 / 3)


def skewnormal_loc_from_mean(
    mean: np.ndarray | float,
    sigma: np.ndarray | float,
    skew: np.ndarray | float,
) -> np.ndarray:
    """Derive the scipy SkewNormal ``loc`` parameter from the distribution mean.

    ``loc = mean - sigma * delta * sqrt(2/pi)``
    where ``delta = skew / sqrt(1 + skew**2)``.

    This is the exact inversion used in ``_skewnormal_odds``, ``fused_loc``,
    and ``get_ev``'s internal residual — one authoritative formula so the
    betting path, the training dump, and the Gate-4 fit all derive the same loc.

    Args:
        mean: Distributional mean (E[X]).
        sigma: Scale parameter (positive).
        skew: Shape / skewness parameter (alpha).

    Returns:
        ``loc`` parameter for ``scipy.stats.skewnorm(skew, loc=loc, scale=sigma)``.
    """
    mean = np.asarray(mean, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    skew = np.asarray(skew, dtype=float)
    delta = skew / np.sqrt(1.0 + skew**2)
    return mean - sigma * delta * np.sqrt(2.0 / np.pi)


def skewnormal_params_from_moments(
    var: np.ndarray | float, skew: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the SkewNormal ``(scale, shape)`` reproducing a target variance and skewness.

    The mean is pinned separately (:func:`skewnormal_loc_from_mean`), so this returns only
    ``(sigma, alpha)``. Inverts the standardized-skewness map (Azzalini 1985 [82]): with
    ``delta = alpha / sqrt(1 + alpha**2)`` and ``t = delta**2 * 2/pi``,
    ``gamma = ((4-pi)/2) * t**1.5 / (1-t)**1.5``; letting ``b = |gamma|**(2/3) / ((4-pi)/2)**(2/3)``
    gives ``t = b/(1+b)``, hence ``alpha = delta / sqrt(1-delta**2)`` and
    ``sigma = sqrt(var * (1+b))``. ``skew`` is clamped to ``+/-_SN_SKEW_MAX`` (just inside the
    SkewNormal bound) so ``alpha`` stays finite — a target past the bound is met in variance with
    the maximal feasible skew. This is the WS2 book-shape moment-match feeding
    ``book_skewnormal_shape``; ``skew == 0`` yields ``alpha == 0, sigma == sqrt(var)`` (Normal).

    Args:
        var: Target variance (positive). Broadcast-compatible with ``skew``.
        skew: Target standardized skewness (dimensionless). Clamped to
            ``±_SN_SKEW_MAX`` before inversion.

    Returns:
        ``(sigma, alpha)`` — scale (same units as ``sqrt(var)``) and shape
        parameter for ``scipy.stats.skewnorm(alpha, loc=..., scale=sigma)``.
    """
    gamma = np.clip(np.asarray(skew, dtype=float), -_SN_SKEW_MAX, _SN_SKEW_MAX)
    b = np.abs(gamma) ** (2.0 / 3.0) / _SN_SKEW_C23
    delta = np.copysign(np.sqrt(np.pi / 2.0 * (b / (1.0 + b))), gamma)
    alpha = delta / np.sqrt(np.clip(1.0 - delta**2, 1e-12, None))
    sigma = np.sqrt(np.asarray(var, dtype=float) * (1.0 + b))
    return sigma, alpha


def _crps_grid_bound(y: np.ndarray, mean: np.ndarray) -> float:
    """Upper integration limit for a continuous CRPS grid: ``max(2·max(y), 4·mean(mean))``.

    Extends past the realized upper tail and the predictive mean so the grid never
    truncates the tail CRPS is chosen to weight (researcher_crps_blending.md §4).
    """
    return max(float(y.max()) * 2, float(np.mean(mean)) * 4)


def negbin_crps(y, r, p, gate=None) -> np.ndarray:
    """Per-row CRPS of a (gated) NegBin predictive: ``Σ_k (F(k) − 1{y≤k})²``.

    Sums over the integer support. When ``gate`` is given the predictive is
    zero-inflated and the gated CDF ``gate + (1−gate)·F_base`` carries the
    zero spike (no separate ``y==0`` term).
    """
    y = np.asarray(y, dtype=float)
    r = np.asarray(r, dtype=float)
    p = np.asarray(p, dtype=float)
    mean = r * (1 - p) / p
    k_max = int(max(y.max() * 2, np.mean(mean) * 4, _NEGBIN_CRPS_K_FLOOR))
    k_vals = np.arange(k_max + 1)
    cdf = nbinom.cdf(k_vals[:, None], r[None, :], p[None, :])
    if gate is not None:
        gate = np.atleast_1d(np.asarray(gate, dtype=float))
        cdf = gate[None, :] + (1 - gate[None, :]) * cdf
    indicator = (y[None, :] <= k_vals[:, None]).astype(float)
    return np.sum((cdf - indicator) ** 2, axis=0)


def _dp_log_pmf_grid(mu, phi):
    """Normalized DP log-pmf over the support grid; returns ``(grid, log_pmf)``.

    Efron's (1986) double Poisson kernel, exactly normalized by logsumexp over
    a truncated grid (terms decay super-factorially past the mode, so the grid
    spans ``mu + _DP_GRID_SIGMAS·sd`` capped at ``_DP_KMAX_CAP``). Canonical
    shapes: ``mu``/``phi`` come in scalar or 1-D and leave broadcast to a
    common ``(n,)``; output is ``grid (K,)``, ``log_pmf (K, n)``.
    """
    mu = np.clip(np.atleast_1d(np.asarray(mu, dtype=float)), 1e-6, None)
    phi = np.clip(np.atleast_1d(np.asarray(phi, dtype=float)), _DP_PHI_FLOOR, _DP_PHI_CEILING)
    mu, phi = np.broadcast_arrays(mu, phi)
    span = np.max(mu + _DP_GRID_SIGMAS * np.sqrt(mu / phi))
    kmax = int(np.clip(span + 1, _DP_KMAX_MIN, _DP_KMAX_CAP))
    grid = np.arange(kmax + 1, dtype=float)
    k = grid[:, None]
    log_kernel = (
        0.5 * np.log(phi)[None, :]
        - (phi * mu)[None, :]
        + (phi[None, :] - 1.0) * k
        + (1.0 - phi[None, :]) * xlogy(k, k)
        - gammaln(k + 1.0)
        + phi[None, :] * k * np.log(mu)[None, :]
    )
    return grid, log_kernel - logsumexp(log_kernel, axis=0, keepdims=True)


def _dp_mean(mu, phi):
    """Exact predictive mean of the double Poisson; ``mu`` is only approximate.

    Returns shape ``(n,)`` for any scalar/1-D input combination.
    """
    grid, log_pmf = _dp_log_pmf_grid(mu, phi)
    return np.sum(grid[:, None] * np.exp(log_pmf), axis=0)


def _dp_mu_from_mean(mean, phi):
    """Invert the exact DP mean back to the ``mu`` parameter; returns ``(n,)``.

    Multiplicative fixed-point (Newton on the log scale — d mean/d log mu ≈
    mean near the solution, since mean ≈ mu); rows that miss the tolerance
    fall back to a per-row brentq bracket.
    """
    mean = np.clip(np.atleast_1d(np.asarray(mean, dtype=float)), 1e-6, None)
    phi = np.broadcast_to(np.atleast_1d(np.asarray(phi, dtype=float)), mean.shape)
    mu = mean.copy()

    def _converged(current):
        err = np.abs(_dp_mean(current, phi) - mean)
        return (err < np.abs(mean) * _DP_NEWTON_RTOL) | (err < 1e-9)

    for _ in range(_DP_NEWTON_MAX_ITER):
        ratio = mean / np.clip(_dp_mean(mu, phi), 1e-9, None)
        if np.all(np.abs(ratio - 1.0) < _DP_NEWTON_RTOL):
            return mu
        mu = np.clip(mu * np.clip(ratio, 0.5, 2.0), 1e-6, None)
    for i in np.flatnonzero(~_converged(mu)):
        lo, hi = 1e-6, max(mean[i] * 4, 1.0)

        def _gap(m, p=phi[i], t=mean[i]):
            return float(_dp_mean(m, p)[0]) - t

        f_lo, f_hi = _gap(lo), _gap(hi)
        if f_lo * f_hi > 0:
            # Target mean sits outside the achievable range (e.g. get_ev's
            # 1e-6 floor probe, below the exact mean at the mu floor) —
            # clamp to the nearer endpoint instead of failing the bracket.
            mu[i] = lo if abs(f_lo) <= abs(f_hi) else hi
        else:
            mu[i] = brentq(_gap, lo, hi, xtol=1e-8)
    return mu


def _dp_cdf_pmf(x, mu, phi):
    """Row-aligned DP ``(CDF(x), PMF(x))`` from one shared grid evaluation.

    ``x`` is a scalar or an array broadcastable to the ``(n,)`` parameter
    shape. CDF sums the normalized pmf up to ``floor(x)``; PMF is the point
    mass at ``x`` (zero off the integer lattice or below zero).
    """
    grid, log_pmf = _dp_log_pmf_grid(mu, phi)
    pmf_grid = np.exp(log_pmf)
    cdf_grid = np.clip(np.cumsum(pmf_grid, axis=0), 0.0, 1.0)
    n = pmf_grid.shape[1]
    x_arr = np.broadcast_to(np.asarray(x, dtype=float), (n,))

    floor_idx = np.floor(x_arr).astype(int)
    cdf = np.where(floor_idx < 0, 0.0, cdf_grid[np.clip(floor_idx, 0, len(grid) - 1), np.arange(n)])
    on_lattice = np.isclose(x_arr - np.round(x_arr), 0.0) & (x_arr >= 0)
    pmf = np.where(
        on_lattice,
        pmf_grid[np.clip(np.round(x_arr).astype(int), 0, len(grid) - 1), np.arange(n)],
        0.0,
    )
    return cdf, pmf


def _dp_ppf(q, mu, phi):
    """Row-aligned DP quantile: smallest k with ``CDF(k) >= q``."""
    grid, log_pmf = _dp_log_pmf_grid(mu, phi)
    cdf_grid = np.clip(np.cumsum(np.exp(log_pmf), axis=0), 0.0, 1.0)
    q_arr = np.broadcast_to(np.asarray(q, dtype=float), (cdf_grid.shape[1],))
    hit = cdf_grid >= q_arr[None, :]
    idx = np.where(hit.any(axis=0), hit.argmax(axis=0), len(grid) - 1)
    return grid[idx]


def dp_pmf_curve(xs, ev, phi):
    """DP pmf on an integer grid for one ``(exact mean, phi)`` pair.

    Chart/diagnostic surface: inverts the mean once, then evaluates the
    normalized pmf at each ``xs``.
    """
    mu = _dp_mu_from_mean(np.array([float(ev)]), np.array([float(phi)]))[0]
    n = len(xs)
    _, pmf = _dp_cdf_pmf(np.asarray(xs, dtype=float), np.full(n, mu), np.full(n, float(phi)))
    return pmf


def dp_crps(y, mu, phi) -> np.ndarray:
    """Per-row CRPS of a Double Poisson predictive: ``Σ_k (F(k) − 1{y≤k})²``.

    ``mu``/``phi`` are the natural DP parameters (not the exact mean), matching
    the raw LightGBMLSS param frame the dispersion calibration re-scores.
    """
    y = np.asarray(y, dtype=float)
    grid, log_pmf = _dp_log_pmf_grid(mu, phi)
    cdf_grid = np.clip(np.cumsum(np.exp(log_pmf), axis=0), 0.0, 1.0)
    mean = np.sum(grid[:, None] * np.exp(log_pmf), axis=0)
    k_max = int(max(y.max() * 2, np.mean(mean) * 4, _NEGBIN_CRPS_K_FLOOR))
    k_vals = np.arange(k_max + 1)
    # Past the internal grid the CDF has converged to 1 (the grid spans 12 sd);
    # clip the index rather than re-summing a longer tail.
    cdf = cdf_grid[np.clip(k_vals, 0, len(grid) - 1), :]
    indicator = (y[None, :] <= k_vals[:, None]).astype(float)
    return np.sum((cdf - indicator) ** 2, axis=0)


def gamma_crps(y, alpha, scale, gate=None) -> np.ndarray:
    """Per-row CRPS of a (gated) Gamma predictive.

    Non-gated uses the closed form (Scheuerer & Möller); the zero-inflated case
    integrates the gated CDF on a grid because the spike has no closed form.
    """
    y = np.asarray(y, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    scale = np.asarray(scale, dtype=float)
    mean = alpha * scale
    if gate is not None:
        gate = np.atleast_1d(np.asarray(gate, dtype=float))
        x_grid = np.linspace(0, _crps_grid_bound(y, mean), _CRPS_GRID_POINTS)
        dx = x_grid[1] - x_grid[0]
        cdf = gamma.cdf(x_grid[:, None], alpha[None, :], scale=scale[None, :])
        cdf = gate[None, :] + (1 - gate[None, :]) * cdf
        indicator = (y[None, :] <= x_grid[:, None]).astype(float)
        return np.sum((cdf - indicator) ** 2, axis=0) * dx
    f_y = gamma.cdf(y, alpha, scale=scale)
    f_y_a1 = gamma.cdf(y, alpha + 1, scale=scale)
    return y * (2 * f_y - 1) - mean * (2 * f_y_a1 - 1) - scale / beta_fn(0.5, alpha)


def skewnorm_crps(y, loc, scale, alpha, gate=None) -> np.ndarray:
    """Per-row CRPS of a (gated) SkewNormal predictive by trapezoid grid integration.

    The skew-normal has no published closed-form CRPS, so integrate
    ``(F(x) − 1{y≤x})²`` over a grid using the analytic CDF. When ``gate`` is given
    the gated CDF ``gate + (1−gate)·F_base`` carries the zero spike.
    """
    y = np.asarray(y, dtype=float)
    loc = np.asarray(loc, dtype=float)
    scale = np.asarray(scale, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    delta = alpha / np.sqrt(1.0 + alpha**2)
    mean = loc + scale * delta * np.sqrt(2.0 / np.pi)
    x_grid = np.linspace(0, _crps_grid_bound(y, mean), _CRPS_GRID_POINTS)
    dx = x_grid[1] - x_grid[0]
    cdf = skewnorm.cdf(x_grid[:, None], alpha[None, :], loc=loc[None, :], scale=scale[None, :])
    if gate is not None:
        gate = np.atleast_1d(np.asarray(gate, dtype=float))
        cdf = gate[None, :] + (1 - gate[None, :]) * cdf
    indicator = (y[None, :] <= x_grid[:, None]).astype(float)
    return np.sum((cdf - indicator) ** 2, axis=0) * dx


def get_ev(line, under, cv=1, dist="SkewNormal", gate=None, skew_alpha=None, sigma=None):
    """Invert the book's ``(line, under-prob)`` to the implied mean.

    The exact numerical inverse of :func:`get_odds`: returns the mean ``ev`` for
    which ``get_odds(line, ev, dist, …) == under``, so the book-EV round trip
    shared by the archive write and ``book_fallback_prob`` read is self-consistent
    for every family and gate (the zero-inflation gate is re-added on decode, so
    it must not be stripped here — it is baked into the inverted ``get_odds``).

    The implied mean is capped at ``SN_MAX_MEAN_FACTOR × line``: a book price that
    would imply a larger mean (its under-prob sits below the capped distribution's
    floor — e.g. a high-zero-rate count at a low line) clamps to that ceiling
    rather than running away. A non-monotone bracket (e.g. a negative spread line)
    returns the neutral line.

    Args:
        line: The bookmaker's line.
        under: The bookmaker's implied probability that the outcome is under the line.
        cv: Coefficient of variation (shape source when alpha/r are not supplied).
        dist: Distribution family — ``"Gamma"``/``"ZAGamma"``/``"NegBin"``/
            ``"ZINB"``/``"DPO"``/``"Poisson"``/``"SkewNormal"``/``"Normal"``
            (symmetric, ev == line at an even-money price; game lines pin here).
        gate: Zero-inflation probability; ``None`` disables ZI handling.
        skew_alpha: SkewNormal skewness; ``None`` → 0 (symmetric).
        sigma: SkewNormal scale held fixed across the inversion; ``None`` → ``ev*cv``.
            WS2 passes the per-cell fitted scale evaluated once at the quoted line, so
            the bracket stays monotone (the shape does not move with the solved mean).

    Returns:
        The mean that reproduces the book's ``under`` under :func:`get_odds`.
    """
    under = float(np.clip(under, 1e-6, 1 - 1e-6))
    step = 1.0 if dist in ("NegBin", "ZINB", "Poisson", "DPO") else 0.5

    def p_under(ev):
        return get_odds(
            line, ev, dist, cv=cv, step=step, gate=gate, skew_alpha=skew_alpha, sigma=sigma
        )

    lo = 1e-6
    hi = max(SN_MAX_MEAN_FACTOR * float(line), 1.0)
    p_lo, p_hi = p_under(lo), p_under(hi)
    if p_lo <= p_hi:  # non-monotone bracket (e.g. a negative line) — no inversion
        return float(line)
    if under >= p_lo:  # implied mean at or below the floor
        return lo
    if under <= p_hi:  # implied mean beyond the cap — clamp instead of running away
        return hi
    return float(brentq(lambda ev: p_under(ev) - under, lo, hi, xtol=1e-8))


def _negbin_odds(line, ev, cv, r, gate, dist):
    if r is None:
        r = 1 / cv
    p = r / (r + ev)
    base_cdf = nbinom.cdf(line, r, p)
    base_pmf = nbinom.pmf(line, r, p)
    if gate is not None and dist == "ZINB":
        # ZI-CDF: gate + (1 - gate) * base_CDF
        base_cdf = gate + (1 - gate) * base_cdf
        base_pmf = (1 - gate) * base_pmf
    return base_cdf - base_pmf / 2


def _dp_odds(line, ev, cv, phi):
    """P(under ``line``) for a Double Poisson with exact mean ``ev``.

    ``phi=None`` is the book fallback: match the NegBin book variance
    convention ``var = ev + cv·ev²`` via ``phi = 1/(1 + cv·ev)``. The mean is
    Newton-inverted to the ``mu`` parameter internally so every caller speaks
    mean-space, like the other families.
    """
    ev_arr = np.atleast_1d(np.asarray(ev, dtype=float))
    phi_arr = 1.0 / (1.0 + cv * ev_arr) if phi is None else np.asarray(phi, dtype=float)
    mu = _dp_mu_from_mean(ev_arr, phi_arr)
    cdf, pmf = _dp_cdf_pmf(line, mu, phi_arr)
    out = cdf - pmf / 2
    return float(out[0]) if np.ndim(ev) == 0 else out


def _skewnormal_odds(high, low, ev, cv, sigma, skew_alpha, gate):
    sigma_val = sigma if sigma is not None else ev * cv
    a = skew_alpha if skew_alpha is not None else 0.0
    delta = a / np.sqrt(1 + a**2)
    loc_sn = ev - sigma_val * delta * np.sqrt(2 / np.pi)
    cdf_high = skewnorm.cdf(high, a, loc=loc_sn, scale=sigma_val)
    cdf_low = skewnorm.cdf(low, a, loc=loc_sn, scale=sigma_val)
    if gate is not None:
        cdf_high = gate + (1 - gate) * cdf_high
        cdf_low = gate + (1 - gate) * cdf_low
    push = cdf_high - cdf_low
    return cdf_high - push / 2


def _gamma_odds(high, low, ev, cv, alpha, gate, dist):
    if alpha is None:
        alpha = 1 / cv**2

    cdf_high = gamma.cdf(high, alpha, scale=ev / alpha)
    cdf_low = gamma.cdf(low, alpha, scale=ev / alpha)
    if gate is not None and dist == "ZAGamma":
        # ZA-CDF: gate + (1 - gate) * base_CDF
        cdf_high = gate + (1 - gate) * cdf_high
        cdf_low = gate + (1 - gate) * cdf_low
    push = cdf_high - cdf_low
    return cdf_high - push / 2


# Mixture-normal quantile bisection: a ±8-sigma bracket around the component
# locations contains any practical quantile, and 80 halvings resolve it far
# past the 4-decimal CDF round-trip the golden test pins.
_MIXNORM_BRACKET_SIGMAS = 8.0
_MIXNORM_BISECT_ITERS = 80


def _mixnorm_cdf(y, w1, loc1, scale1, loc2, scale2):
    cdf1 = norm.cdf(y, loc=loc1, scale=scale1)
    cdf2 = norm.cdf(y, loc=loc2, scale=scale2)
    return w1 * cdf1 + (1 - w1) * cdf2


def _mixnorm_ppf(q, w1, loc1, scale1, loc2, scale2):
    """Scalar-q mixture quantile by per-row bisection (no closed-form inverse)."""
    span = _MIXNORM_BRACKET_SIGMAS * np.maximum(scale1, scale2)
    lo = np.minimum(loc1, loc2) - span
    hi = np.maximum(loc1, loc2) + span
    for _ in range(_MIXNORM_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        below = _mixnorm_cdf(mid, w1, loc1, scale1, loc2, scale2) < q
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def _mixnorm_odds(high, low, w1, loc1, scale1, loc2, scale2):
    cdf_high = _mixnorm_cdf(high, w1, loc1, scale1, loc2, scale2)
    cdf_low = _mixnorm_cdf(low, w1, loc1, scale1, loc2, scale2)
    push = cdf_high - cdf_low
    return cdf_high - push / 2


def get_odds(
    line,
    ev,
    dist,
    cv=1,
    alpha=None,
    r=None,
    gate=None,
    step=1,
    sigma=None,
    skew_alpha=None,
    phi=None,
    mix=None,
):
    """Return the raw probability that the outcome falls below ``line``.

    Inverse of ``get_ev``. Temperature scaling is applied elsewhere, at
    the over/under decision layer, not here.

    Args:
        line: The line / cutoff value.
        ev: Expected value (base-distribution mean).
        dist: Distribution family (same options as ``get_ev``, plus ``"DPO"``).
        cv: Coefficient of variation, used when ``alpha``/``r``/``phi`` are not supplied.
        alpha: Gamma shape; derived as ``1/cv²`` if ``None``.
        r: NegBin dispersion; derived as ``1/cv`` if ``None``.
        gate: Zero-inflation probability; ``None`` disables ZI handling.
        step: Bin width for the discrete half-point correction.
        sigma: SkewNormal scale; derived as ``ev*cv`` if ``None``.
        skew_alpha: SkewNormal skewness; defaults to ``0``.
        phi: Double Poisson precision; derived as ``1/(1+cv·ev)`` if ``None``.
        mix: Mixture params dict with keys ``w1``, ``loc1``, ``scale1``,
            ``loc2``, ``scale2`` (arrays); required when ``dist == "Mixture"``.

    Returns:
        Probability of outcome being under ``line``.
    """
    high = np.floor((line + step) / step) * step
    low = np.ceil((line - step) / step) * step

    # Poisson (discrete count data).
    # NegBin without model params falls back to Poisson only when cv==1 (old encoding);
    # when cv!=1 the archive EV was Gaussian-encoded by get_ev, so fall through to the
    # Gaussian/Gamma branch for a consistent round-trip.

    if dist == "Poisson" or (dist in ("NegBin", "ZINB") and r is None and cv == 1):
        return poisson.cdf(line, ev) - poisson.pmf(line, ev) / 2
    if dist in ("NegBin", "ZINB"):
        return _negbin_odds(line, ev, cv, r, gate, dist)
    if dist == "DPO":
        return _dp_odds(line, ev, cv, phi)
    if dist == "SkewNormal":
        return _skewnormal_odds(high, low, ev, cv, sigma, skew_alpha, gate)
    if dist == "Normal":
        # skew_alpha is deliberately ignored: game lines must invert symmetrically
        # because the no-vig median price IS the implied value. Passing any skew
        # here would make the EV depend on the book's vig direction.
        return _skewnormal_odds(high, low, ev, cv, sigma, 0.0, gate)
    if dist == "Mixture":
        return _mixnorm_odds(
            high, low, mix["w1"], mix["loc1"], mix["scale1"], mix["loc2"], mix["scale2"]
        )
    return _gamma_odds(high, low, ev, cv, alpha, gate, dist)


def _dp_push_pmf(line_arr, ev_arr, cv, phi):
    """Row-aligned Double Poisson point mass at the line, for ``get_push_prob``."""
    line_1d, ev_1d = (np.atleast_1d(a) for a in np.broadcast_arrays(line_arr, ev_arr))
    phi_1d = 1.0 / (1.0 + cv * ev_1d) if phi is None else np.asarray(phi, dtype=float)
    _, pmf = _dp_cdf_pmf(line_1d, _dp_mu_from_mean(ev_1d, phi_1d), phi_1d)
    return pmf if np.ndim(line_arr) or np.ndim(ev_arr) else float(pmf[0])


def get_push_prob(
    line, ev, dist, cv=1, alpha=None, r=None, gate=None, sigma=None, skew_alpha=None, phi=None
):
    """Return P(stat == line) for a market.

    Push probability is non-zero only when ``line`` is an integer **and** the
    distribution family is discrete (NegBin / ZINB / DPO / Poisson). Continuous
    families (Gamma, ZAGamma, SkewNormal, Normal, Mixture) return 0 because the
    point mass at any single value is zero. Used by parlay scoring to handle the
    Underdog "push drops one leg" rule.

    Args:
        line: The bookmaker line. Push only fires for integer-valued lines.
        ev: Base-distribution mean (post fused_loc / dispersion calibration).
        dist: Distribution family ("NegBin", "ZINB", "DPO", "Poisson", "Gamma",
            "ZAGamma", "SkewNormal", "Normal"). Anything not recognized as
            discrete returns 0.
        cv: CV used as fallback when ``r``/``phi`` is not supplied.
        alpha: Unused; included so the call signature parallels ``get_odds``.
        r: NegBin dispersion. Falls back to ``1/cv``.
        gate: Zero-inflation gate; ZI shrinks the base pmf by ``(1 - gate)``.
            For an integer line at 0, the gate contributes its full mass.
        sigma: Unused; parallels ``get_odds``.
        skew_alpha: Unused; parallels ``get_odds``.
        phi: Double Poisson precision. Falls back to ``1/(1+cv·ev)``.

    Returns:
        np.ndarray | float: Push probability, broadcast to the shape of
            ``line`` / ``ev``.
    """
    del alpha, sigma, skew_alpha  # unused but accepted for parity with get_odds
    line_arr = np.asarray(line, dtype=float)
    ev_arr = np.asarray(ev, dtype=float)
    is_integer = np.isclose(line_arr - np.round(line_arr), 0.0)
    nonneg = line_arr >= 0

    if dist == "Poisson" or (dist in ("NegBin", "ZINB") and r is None and cv == 1):
        pmf = poisson.pmf(np.round(line_arr), ev_arr)
    elif dist == "DPO":
        pmf = _dp_push_pmf(line_arr, ev_arr, cv, phi)
    elif dist in ("NegBin", "ZINB"):
        r_val = r if r is not None else 1 / cv
        p = r_val / (r_val + ev_arr)
        pmf = nbinom.pmf(np.round(line_arr), r_val, p)
        if gate is not None and dist == "ZINB":
            # ZI shrinks the continuous part of the pmf; the gate adds mass at 0.
            pmf = (1 - np.asarray(gate)) * pmf
            zero_mask = np.isclose(line_arr, 0.0)
            pmf = np.where(zero_mask, pmf + np.asarray(gate), pmf)
    else:
        # Continuous distributions have zero point mass at any single line.
        return np.zeros_like(line_arr)

    return np.where(is_integer & nonneg, pmf, 0.0)


@dataclass
class DecodedParams:
    """Base-distribution mean plus the shape parameters ``get_odds`` consumes.

    Only the fields relevant to the family are populated: ``r`` for
    NegBin/ZINB, ``alpha`` for Gamma/ZAGamma, ``sigma``/``skew`` for SkewNormal,
    ``phi`` for DPO. ``gate`` is the per-row zero-inflation gate (ZINB/ZAGamma)
    or the broadcast historical gate (SkewNormal), and is ``None`` when the
    cell is not gated.
    """

    ev: np.ndarray
    r: np.ndarray | None = None
    alpha: np.ndarray | None = None
    sigma: np.ndarray | None = None
    skew: np.ndarray | None = None
    gate: np.ndarray | None = None
    phi: np.ndarray | None = None


def decode_predictive_mean(prob_params, dist, *, sn_loc=None, sn_scale=None, hist_gate=0.0):
    """Decode raw LightGBMLSS ``predict(pred_type="parameters")`` output.

    Single source of truth for the train-side and predict-side decode of a
    distribution's base mean and shape, so the two pipelines cannot drift on the
    parameterization. ``get_odds`` consumes the returned shape directly.

    SkewNormal requires ``sn_loc``/``sn_scale`` already run through the
    ``training.baselines`` registry by the caller — ``helpers`` must not import
    ``training`` — and publishes the gate from ``hist_gate`` above
    :data:`GATE_PUBLISH_THRESHOLD`. The count families carry their own per-row
    gate column and ignore ``hist_gate``.

    Args:
        prob_params: Frame/dict-like with the raw distribution columns:
            ``total_count``/``probs``/``gate`` (NegBin/ZINB), ``mu``/``phi``
            (DPO — the mean is decoded exactly by series, not read off ``mu``),
            ``concentration``/``rate``/``gate`` (Gamma/ZAGamma), or ``alpha``
            (SkewNormal skewness).
        dist: Distribution family name.
        sn_loc: SkewNormal decoded location (required for SkewNormal).
        sn_scale: SkewNormal decoded scale (required for SkewNormal).
        hist_gate: SkewNormal historical zero rate; published as the gate when
            above the threshold. Ignored for count families.

    Returns:
        DecodedParams with ``ev`` plus the family's shape fields populated.
    """
    if dist in ("NegBin", "ZINB"):
        r = np.asarray(prob_params["total_count"], dtype=float)
        p = np.asarray(prob_params["probs"], dtype=float)
        gate = np.asarray(prob_params["gate"], dtype=float) if dist == "ZINB" else None
        return DecodedParams(ev=r * p / (1 - p), r=r, gate=gate)

    if dist == "DPO":
        mu = np.asarray(prob_params["mu"], dtype=float)
        phi = np.asarray(prob_params["phi"], dtype=float)
        return DecodedParams(ev=_dp_mean(mu, phi), phi=phi)

    if dist in ("Gamma", "ZAGamma"):
        alpha = np.asarray(prob_params["concentration"], dtype=float)
        beta = np.asarray(prob_params["rate"], dtype=float)
        gate = np.asarray(prob_params["gate"], dtype=float) if dist == "ZAGamma" else None
        return DecodedParams(ev=alpha / beta, alpha=alpha, gate=gate)

    sn_loc = np.asarray(sn_loc, dtype=float)
    sn_scale = np.asarray(sn_scale, dtype=float)
    skew = np.asarray(prob_params["alpha"], dtype=float)
    delta = skew / np.sqrt(1 + skew**2)
    ev = sn_loc + sn_scale * delta * np.sqrt(2 / np.pi)
    gate = np.full_like(ev, float(hist_gate)) if hist_gate > GATE_PUBLISH_THRESHOLD else None
    return DecodedParams(ev=ev, sigma=sn_scale, skew=skew, gate=gate)


def apply_temperature(p_over, temperature):
    """Temperature-scale an over-probability in logit space.

    Shared by the prediction and training calibration paths. Returns ``p_over``
    unchanged when ``temperature`` is ``None`` (no calibration fitted).
    """
    if temperature is None:
        return p_over
    clipped = np.clip(p_over, LOGIT_CLIP_EPS, 1 - LOGIT_CLIP_EPS)
    return expit(logit(clipped) / temperature)


def apply_cdf_recal(blob: dict | None, u):
    """Apply a §6.1 Rung C whole-CDF recalibration map to CDF value(s) ``u = F(point)``.

    ``blob`` is fit by :func:`sportstradamus.training.posthoc.fit_isotonic_pit` /
    ``select_pit_recal`` on the validation PIT; shared here (not left in
    ``training.posthoc``) because both prediction (:mod:`sportstradamus.prediction.model_prob`,
    the dashboard deep-dive) and training (``pipeline``, ``scorecard``) apply the same fitted
    map. Identity when ``blob`` is ``None`` (no map fitted, or a pre-Rung-C pickle).
    """
    u = np.asarray(u, dtype=float)
    if blob is None:
        return u
    return np.clip(np.interp(u, blob["x"], blob["y"]), 0.0, 1.0)


def fit_distro(mean, std, lower_bound, upper_bound, lower_tol=0.1, upper_tol=0.001):
    """Solve for a scaling factor ``w`` that pulls (mean, std) into the bounds.

    Used for sanity-clamping distribution moments before feeding downstream
    consumers. The objective penalizes over/under-shooting the specified
    tail probabilities plus any deviation of ``w`` away from 1.
    """

    def objective(w, m, s):
        v = w if w >= 1 else 1 / w
        if s > 0:
            return (
                LOWER_TAIL_PENALTY * max((norm.cdf(lower_bound, w * m, v * s) - lower_tol), 0)
                + max((norm.sf(upper_bound, w * m, v * s) - upper_tol), 0)
                + np.power(1 - v, 2)
            )
        return (
            LOWER_TAIL_PENALTY * max((poisson.cdf(lower_bound, w * m) - lower_tol), 0)
            + max((poisson.sf(upper_bound, w * m) - upper_tol), 0)
            + np.power(1 - v, 2)
        )

    res = minimize(objective, [1], args=(mean, std), bounds=[(0.5, 2)], tol=1e-3, method="TNC")
    return res.x[0]


def fused_loc(
    w,
    ev_a,
    ev_b,
    cv,
    dist,
    *,
    r=None,
    alpha=None,
    sigma=None,
    skew_alpha=None,
    gate_model=None,
    gate_book=None,
    book_sigma=None,
    book_skew_alpha=None,
    phi=None,
):
    """Blend model and bookmaker distribution parameters with weight ``w``.

    The blend is a logarithmic opinion pool (Genest & Zidek 1986) for
    NegBin / DPO and a precision-weighted blend for Gamma / SkewNormal:

    * **NegBin**: geometric mean of both means *and* dispersion parameters.
      The model provides per-observation ``r``; the book's ``r`` is derived
      as ``1/cv``. Both ``μ`` and ``r`` are blended in log space with the
      same weight ``w``.
    * **DPO**: same log-pool on the exact mean and the precision ``phi``;
      the book's ``phi`` is ``1/(1+cv·ev_b)`` (matching the NegBin book
      variance convention ``var = ev + cv·ev²``). Returns the blended
      *mean* — ``get_odds`` Newton-inverts mean → ``mu`` internally.
    * **Gamma**: precision-weighted blend. The model provides
      per-observation ``alpha``; the book's ``alpha`` is ``1/cv²``.
      Returns ``(alpha, beta, gate_blend)``.
    * **SkewNormal**: precision-weighted blend of ``loc`` / ``sigma``,
      linear blend of ``alpha``. The book side uses the fitted
      ``(book_sigma, book_skew_alpha)`` when supplied, else the legacy
      symmetric constant-CV normal (``sigma = ev*cv``, ``alpha = 0``).
      Returns ``(ev, sigma, alpha, gate_blend)``.

    When ``gate_model`` and ``gate_book`` are supplied (zero-inflated
    distributions), the gate itself is blended linearly and appended as
    the final tuple element. ``ev_a`` and ``ev_b`` must be *base*
    distribution means (before gate deflation).

    Args:
        w: Weight on the model prediction, in ``[0, 1]``.
        ev_a: Model's base-distribution mean.
        ev_b: Bookmaker's base-distribution mean.
        cv: Coefficient of variation for the book side.
        dist: ``"NegBin"``, ``"Gamma"``, or ``"SkewNormal"``.
        r: NegBin per-observation dispersion from the model.
        alpha: Gamma shape from the model.
        sigma: SkewNormal per-observation scale from the model.
        skew_alpha: SkewNormal per-observation skewness from the model.
        gate_model: Model's per-observation zero-inflation gate.
        gate_book: Historical zero-inflation gate for the book side.
        book_sigma: Per-observation book SkewNormal scale (WS2 fitted shape,
            evaluated at the quoted line). ``None`` → legacy ``ev_b*cv``.
        book_skew_alpha: Per-observation book SkewNormal skewness. ``None`` → 0
            (symmetric), paired with the legacy ``book_sigma`` default.
        phi: DPO per-observation precision from the model.

    Returns:
        NegBin → ``(r_blend, p, gate_blend)``,
        DPO → ``(blended_mean, phi_blend, gate_blend)``,
        Gamma → ``(alpha, beta, gate_blend)``,
        SkewNormal → ``(blended_ev, blended_sigma, blended_alpha, gate_blend)``.
        ``gate_blend`` is ``None`` when no gate parameters are supplied.
    """
    gate_blend = None
    if gate_model is not None and gate_book is not None:
        gate_blend = w * np.asarray(gate_model, dtype=float) + (1 - w) * gate_book
    elif gate_book is not None and gate_book > 0:
        # No model gate (hurdle model) — use book gate directly.
        gate_blend = gate_book

    if dist == "DPO":
        ev_a = np.clip(np.asarray(ev_a, dtype=float), 1e-9, None)
        ev_b = np.clip(np.asarray(ev_b, dtype=float), 1e-9, None)
        mean_blend = np.exp(w * np.log(ev_a) + (1 - w) * np.log(ev_b))
        phi_book = 1.0 / (1.0 + cv * ev_b)
        phi_blend = np.exp(
            w * np.log(np.clip(phi, _DP_PHI_FLOOR, _DP_PHI_CEILING)) + (1 - w) * np.log(phi_book)
        )
        return mean_blend, phi_blend, gate_blend

    if dist == "NegBin":
        mu = np.exp(
            w * np.log(np.clip(ev_a, 1e-9, None)) + (1 - w) * np.log(np.clip(ev_b, 1e-9, None))
        )
        r_blend = np.exp(w * np.log(np.clip(r, 1e-9, None)) + (1 - w) * np.log(1 / cv))
        p = r_blend / (r_blend + mu)
        return r_blend, p, gate_blend

    if dist == "SkewNormal":
        ev_a = np.clip(np.asarray(ev_a, dtype=float), 1e-9, None)
        ev_b = np.clip(np.asarray(ev_b, dtype=float), 1e-9, None)
        model_sigma = np.clip(np.asarray(sigma, dtype=float), 1e-6, None)
        model_skew = np.asarray(skew_alpha, dtype=float)

        if book_sigma is None:
            book_scale = ev_b * cv
            book_skew = np.zeros_like(ev_b)
        else:
            book_scale = np.asarray(book_sigma, dtype=float)
            book_skew = np.asarray(book_skew_alpha, dtype=float)
        book_scale = np.clip(book_scale, 1e-6, None)

        model_delta = model_skew / np.sqrt(1 + model_skew**2)
        model_loc = ev_a - model_sigma * model_delta * np.sqrt(2 / np.pi)
        book_delta = book_skew / np.sqrt(1 + book_skew**2)
        book_loc = ev_b - book_scale * book_delta * np.sqrt(2 / np.pi)

        # The floor applies to the pool only: model_loc above decodes the model's own
        # mean identity from its true sigma, which the floor must not shift. At w = 1
        # (no authentic book quote) there is no book leg to seize, and the pure-model
        # identity fused == model must hold exactly.
        pooled_model_sigma = np.where(
            np.asarray(w) >= 1.0,
            model_sigma,
            np.maximum(model_sigma, _BLEND_MODEL_SCALE_FLOOR * book_scale),
        )
        prec_m = 1.0 / pooled_model_sigma**2
        prec_b = 1.0 / book_scale**2
        total_prec = w * prec_m + (1 - w) * prec_b
        blended_loc = (w * model_loc * prec_m + (1 - w) * book_loc * prec_b) / total_prec
        blended_sigma = 1.0 / np.sqrt(total_prec)
        blended_skew = w * model_skew + (1 - w) * book_skew

        bl_delta = blended_skew / np.sqrt(1 + blended_skew**2)
        blended_ev = blended_loc + blended_sigma * bl_delta * np.sqrt(2 / np.pi)

        return blended_ev, blended_sigma, blended_skew, gate_blend

    ev_a = np.clip(np.asarray(ev_a, dtype=float), 1e-9, None)
    ev_b = np.clip(np.asarray(ev_b, dtype=float), 1e-9, None)
    model_alpha = np.clip(np.asarray(alpha, dtype=float), 1e-9, None)
    book_alpha = 1 / cv**2
    inv_var_m = model_alpha / ev_a**2
    inv_var_b = book_alpha / ev_b**2
    total_inv_var = w * inv_var_m + (1 - w) * inv_var_b
    blended_mean = (w * ev_a * inv_var_m + (1 - w) * ev_b * inv_var_b) / total_inv_var
    blended_alpha = blended_mean**2 * total_inv_var
    blended_beta = blended_mean * total_inv_var
    return blended_alpha, blended_beta, gate_blend


def _softplus_inv(x):
    x = np.asarray(x, dtype=float)
    return np.where(
        x > SOFTPLUS_LINEAR_THRESHOLD,
        x,
        np.log(np.expm1(np.clip(x, 1e-4, SOFTPLUS_LINEAR_THRESHOLD))),
    )


# Mirrors skew_normal_centered._GAMMA1_MAX (the tanh response bound on the
# centered gamma1 head); duplicated so this module's import graph stays
# torch-free, cross-pinned by tests/golden/test_centered_sn_seed.py.
_SN_CENTERED_GAMMA1_BOUND = 0.99
# Exactly-0 gamma1 has an infinite cube-root gradient in the CP→DP inversion
# (R2 brief risk iii), so the centered seed starts slightly off-zero; positive
# because X carries no skew-sign information.
_SN_CENTERED_GAMMA1_SEED = 0.05


def _skewnormal_start_values(mu, std, n, offset_mode, normalized, sn_param):
    if offset_mode:
        # Additive centered residual target (y - baseline): residual mean
        # ≈ 0 per row; scale starts at per-player STDYr. Explicit per-row
        # broadcast — must be (n,) not scalar (a degenerate scalar 0 was
        # a confirmed seeding regression in the overconfidence
        # investigation; the (n, 3) shape guard in the unit test pins
        # this).
        loc = np.zeros(n)
        scale = std.copy()
    elif normalized:
        cv_player = np.clip(std / mu, 0.01, 10)
        loc = np.ones(n)
        scale = cv_player  # scale ≈ CV since mean ≈ 1.0.
    else:
        loc = mu.copy()
        scale = std.copy()
    log_scale = np.log(np.clip(scale, 1e-6, None))
    if sn_param == "centered":
        # The direct alpha seed is 0, so loc/scale above are exactly the
        # predictive mean/sd — the centered (mean, sd) heads reuse them as is.
        gamma1_raw = np.arctanh(_SN_CENTERED_GAMMA1_SEED / _SN_CENTERED_GAMMA1_BOUND)
        return np.column_stack([loc, log_scale, np.full(n, gamma1_raw)])
    alpha_skew = np.zeros(n)  # Start symmetric.
    return np.column_stack([loc, log_scale, alpha_skew])


def _negbin_start_values(mu, std, hist_gate, dist, _r_upper):
    r_init = np.clip(mu**2 / np.clip(std**2 - mu, 1e-6, None), 0.5, _r_upper)
    probs = np.clip(mu / (mu + r_init), 0.01, 0.99)
    if dist == "ZINB":
        nb_zeros = nbinom.pmf(0, r_init, probs)
        hist_gate = np.clip(hist_gate - nb_zeros, 0, 0.99)
        mu = mu / (1 - hist_gate)
        r_init = np.clip(mu**2 / np.clip(std**2 - mu, 1e-6, None), 0.5, _r_upper)
        probs = np.clip(mu / (mu + r_init), 0.01, 0.99)
    sv = np.column_stack([r_init, logit(probs)])
    return sv, hist_gate


def _dp_start_values(mu, std):
    """Raw-space (softplus) DP start values from per-player moments."""
    phi = np.clip(mu / np.clip(std**2, 1e-6, None), _DP_PHI_SEED_FLOOR, _DP_PHI_SEED_CEILING)
    return np.column_stack([_softplus_inv(mu), _softplus_inv(phi)])


def _gamma_start_values(mu, std, hist_gate, dist, _a_upper):
    if dist == "ZAGamma":
        mu = mu / (1 - hist_gate)
    alpha = np.clip((mu / std) ** 2, 0.1, _a_upper)
    beta = np.clip(alpha / np.clip(mu, 1e-6, None), 0.01, 50)
    return np.column_stack([_softplus_inv(alpha), _softplus_inv(beta)])


# Mixture seeds: the base component starts dominant at this softmax weight (the
# raw logits are the two logs), and the component scales straddle the seeding
# spread so the pair starts separated instead of collapsed onto one Gaussian.
_MIX_BASE_WEIGHT = 0.75
_MIX_SCALE_FACTORS = (0.8, 1.2)


def _mixture_start_values(mu, std, n, offset_mode, normalized):
    if offset_mode:
        # Same centered-residual seeding as the SkewNormal offset mode: base
        # component at residual 0, boom component one historical std above.
        loc = np.zeros(n)
        boom_loc = std
        spread = std
    elif normalized:
        cv_player = np.clip(std / mu, 0.01, 10)
        loc = np.ones(n)
        boom_loc = 1.0 + cv_player
        spread = cv_player
    else:
        loc = mu
        boom_loc = mu + std
        spread = std
    base_f, boom_f = _MIX_SCALE_FACTORS
    log_scale = np.log(np.clip(base_f * spread, 1e-6, None))
    boom_log_scale = np.log(np.clip(boom_f * spread, 1e-6, None))
    logit_base = np.full(n, np.log(_MIX_BASE_WEIGHT))
    logit_boom = np.full(n, np.log(1.0 - _MIX_BASE_WEIGHT))
    return np.column_stack([loc, boom_loc, log_scale, boom_log_scale, logit_base, logit_boom])


def set_model_start_values(
    model,
    dist,
    X_data,
    shape_ceiling=None,
    normalized=False,
    offset_mode=False,
    sn_param: str = "direct",
):
    """Initialize LightGBMLSS start values from per-player historical moments.

    Values live in the model's raw (pre-response-function) space. Response
    functions per distribution:

    * NegBin / ZINB: ``total_count`` → ReLU, ``probs`` → sigmoid,
      ``gate`` → sigmoid.
    * Gamma / ZAGamma: ``concentration`` → softplus, ``rate`` → softplus,
      ``gate`` → sigmoid.
    * DPO: ``mu`` → softplus, ``phi`` → softplus.
    * SkewNormal: ``loc`` → identity, ``scale`` → exp, ``alpha`` → identity;
      with ``sn_param="centered"``: ``mean`` → identity, ``sd`` → exp,
      ``gamma1`` → ``0.99·tanh``.
    * Mixture (2-component Gaussian): ``loc_1``/``loc_2`` → identity,
      ``scale_1``/``scale_2`` → exp, ``mix_prob_1``/``mix_prob_2`` →
      deterministic softmax over the raw logit pair.

    Args:
        model: The LightGBMLSS model whose ``start_values`` gets assigned.
        dist: Distribution name — ``"NegBin"``, ``"ZINB"``, ``"DPO"``,
            ``"Gamma"``, ``"ZAGamma"``, ``"SkewNormal"``, or ``"Mixture"``.
        X_data: DataFrame; must contain ``"MeanYr"``, ``"STDYr"``, and
            ``"ZeroYr"`` columns.
        shape_ceiling: Upper bound on shape during training. When ``None``,
            a conservative default is used (50 for NegBin, 100 for Gamma).
        normalized: If ``True``, targets are already normalized to
            ``Result/MeanYr ≈ 1.0`` and start values are set for that space.
        offset_mode: SkewNormal / Mixture only. If ``True``, targets are an
            additive centered residual (``y - baseline``), so ``loc`` is
            seeded at zero per row and ``scale`` at per-player STDYr
            (residual dispersion ≈ per-player std). Mutually exclusive with
            ``normalized``; ignored for the other distributions.
        sn_param: SkewNormal-only head parametrization — ``"direct"``
            (loc, scale, alpha) or ``"centered"`` (``CenteredSkewNormal``'s
            mean, sd, gamma1). The direct alpha seed is 0, so the direct
            loc/scale seeds are already the predictive mean/sd; the centered
            variant reuses them and seeds gamma1 just off zero
            (``_SN_CENTERED_GAMMA1_SEED``). Ignored for every other
            distribution.
    """
    sv = X_data[["MeanYr", "STDYr", "ZeroYr"]].to_numpy()
    n = len(sv)

    mu = np.clip(sv[:, 0], 1e-6, None)
    std = np.clip(sv[:, 1], 1e-6, None)
    hist_gate = np.clip(sv[:, 2], 0, 0.99)

    _r_upper = shape_ceiling if shape_ceiling is not None else 50
    _a_upper = shape_ceiling if shape_ceiling is not None else 100

    if dist == "SkewNormal":
        sv = _skewnormal_start_values(mu, std, n, offset_mode, normalized, sn_param)
    elif dist in ["NegBin", "ZINB"]:
        sv, hist_gate = _negbin_start_values(mu, std, hist_gate, dist, _r_upper)
    elif dist == "DPO":
        sv = _dp_start_values(mu, std)
    elif dist in ["Gamma", "ZAGamma"]:
        sv = _gamma_start_values(mu, std, hist_gate, dist, _a_upper)
    elif dist == "Mixture":
        sv = _mixture_start_values(mu, std, n, offset_mode, normalized)

    if dist in ["ZINB", "ZAGamma"]:
        gate_val = np.clip(hist_gate, 0.01, 0.99)
        sv = np.column_stack([sv, np.full(n, logit(gate_val))])

    model.start_values = sv
