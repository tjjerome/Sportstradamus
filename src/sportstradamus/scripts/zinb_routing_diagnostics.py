#!/usr/bin/env python3
"""Score every ZINB (league, market) cell and recommend a count-model family.

Task B1.2. Read-only with respect to the model pickles, ``model_prob``, and the
training pipeline: this script only reads cached training-data parquets and
computes descriptive statistics on the marginal count target (``Result``).

The routing recommendation must be precomputable from training data alone so it
survives a temporal split — every statistic here is intercept-only (no
covariates). Per cell we fit two intercept-only count models with ``statsmodels``
(a Negative Binomial and a Zero-Inflated Negative Binomial), measure
zero-inflation two ways with bootstrap CIs, run a zero-inflation score test, and
compute a Schwarz-corrected Vuong statistic comparing a Hurdle-NB to the ZINB.
The decision tree in :func:`_route` maps those statistics to one of
``plain_nb`` / ``hurdle_nb_ztnb`` / ``mzinb`` / ``cmp``.

Output: one ``data/zinb_routing/{LEAGUE}_diagnostics.parquet`` per league
(repo-root ``data/``, not the package data dir), one row per (league, market).

Usage
-----
    poetry run zinb-routing-diagnostics
    poetry run zinb-routing-diagnostics --league NBA --bootstrap 2000
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import click
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
from statsmodels.discrete.discrete_model import NegativeBinomial
from tqdm import tqdm

from sportstradamus import data
from sportstradamus.helpers.config import stat_dist
from sportstradamus.training.markets import ALL_MARKETS

logger = logging.getLogger(__name__)

# Leagues B1.2 routes; MLB/NHL have no ZINB markets so they are out of scope.
_ROUTED_LEAGUES = ("NBA", "WNBA", "NFL")
# The exact stat_dist label that marks a count cell as a ZINB candidate.
_ZINB_LABEL = "ZINB"
# Guard from the plan: the dynamic cell derivation must yield exactly this many.
_EXPECTED_CELL_COUNT = 23
# The count target column shared with tests/integration/test_zinb_hurdle_live_path.py.
_TARGET_COLUMN = "Result"

# Default bootstrap resample count for the zero-inflation index CIs. 1000 keeps
# the percentile CI stable without making the 23-cell pass slow.
_DEFAULT_BOOTSTRAP = 1000
# Default RNG seed so reruns of the descriptive pass are reproducible.
_DEFAULT_SEED = 1729
# Percentile CI coverage: 95% two-sided.
_CI_LOW_PCT = 2.5
_CI_HIGH_PCT = 97.5

# --- Routing thresholds (applied in order in _route). STYLE_GUIDE §9. ---
# var/mean below this is underdispersion -> Conway-Maxwell-Poisson territory;
# neither NB nor ZINB can model under-dispersion (their variance >= mean).
_UNDERDISPERSION_RATIO = 1.3
# When the excess-zero CI contains 0 and dispersion is only mild (close to
# Poisson), a plain NB suffices -- a hurdle/ZINB would add parameters for no
# gain. Slightly above _UNDERDISPERSION_RATIO so the [1.3, 1.5) band is "mild".
_MILD_DISPERSION_RATIO = 1.5

# statsmodels NB optimizer iteration cap. 200 is generous for an intercept-only
# fit; the Newton/BFGS default sometimes needs more than the library default.
_NB_MAXITER = 200
# Zero-truncated NB lower support floor used when reconstructing the hurdle
# positive-count density; guards log(1 - nb0) when nb0 -> 1.
_NB_ZERO_MASS_CEIL = 1.0 - 1e-12

ROUTING_CHOICES: tuple[str, str, str, str] = ("plain_nb", "hurdle_nb_ztnb", "mzinb", "cmp")

REQUIRED_COLUMNS: tuple[str, ...] = (
    "league",
    "market",
    "n",
    "observed_mean",
    "variance",
    "var_mean_ratio",
    "zero_rate",
    "cond_pos_mean",
    "ziP_index",
    "ziP_ci_lo",
    "ziP_ci_hi",
    "ziNB_index",
    "ziNB_ci_lo",
    "ziNB_ci_hi",
    "wilson_einbeck_pvalue",
    "vuong_schwarz_stat",
    "routing_recommendation",
)


class ZeroInflationIndices(NamedTuple):
    """Point estimates and bootstrap CIs for the two zero-inflation indices.

    All values are dimensionless. ``zip_*`` is the Puig-Valero index relative to
    a Poisson with the same mean; ``zinb_*`` is the excess-zero measure relative
    to a fitted intercept-only Negative Binomial (Blasco-Moreno et al. 2019).
    """

    zip_index: float
    zip_ci_lo: float
    zip_ci_hi: float
    zinb_index: float
    zinb_ci_lo: float
    zinb_ci_hi: float


def zinb_cells() -> list[tuple[str, str]]:
    """Return every (league, market) whose stat_dist label is exactly ``ZINB``.

    Derived dynamically from ``stat_dist.json`` and ``ALL_MARKETS`` so the set
    tracks config drift. Asserts the plan's guard count of 23.

    Returns:
        Sorted-by-config (league, market) pairs over NBA, WNBA, and NFL.
    """
    cells: list[tuple[str, str]] = []
    for league in _ROUTED_LEAGUES:
        for market in ALL_MARKETS[league]:
            if stat_dist.get(league, {}).get(market) == _ZINB_LABEL:
                cells.append((league, market))
    if len(cells) != _EXPECTED_CELL_COUNT:
        raise click.ClickException(
            f"Expected {_EXPECTED_CELL_COUNT} ZINB cells, found {len(cells)}: {cells}"
        )
    return cells


def _training_data_root() -> Path:
    """Return the cached training-data parquet directory inside the package."""
    return Path(str(pkg_resources.files(data) / "training_data"))


def _market_stem(market: str) -> str:
    """Map a market name to its parquet filename stem (spaces -> hyphens).

    NFL markets carry spaces in ``ALL_MARKETS`` (e.g. ``"sacks taken"``) but the
    cached parquets are hyphenated (``NFL_sacks-taken.parquet``).
    """
    return market.replace(" ", "-")


def _load_target(league: str, market: str) -> np.ndarray | None:
    """Load the marginal count target for one cell, or ``None`` if absent.

    Returns:
        Finite non-negative ``Result`` counts as a float array, or ``None`` when
        the parquet is missing (logged as a warning so the cell is skipped).
    """
    path = _training_data_root() / f"{league}_{_market_stem(market)}.parquet"
    if not path.is_file():
        logger.warning("Missing training parquet for %s %s: %s", league, market, path)
        return None
    frame = pd.read_parquet(path, engine="pyarrow", columns=[_TARGET_COLUMN])
    y = frame[_TARGET_COLUMN].to_numpy(dtype=float)
    return y[np.isfinite(y)]


def _fit_nb_zero_prob(y: np.ndarray) -> float:
    """Return the zero probability of an intercept-only NB fitted to ``y``.

    Fits a Negative Binomial with only an intercept, then evaluates its implied
    P(Y = 0). NaN if the optimizer fails (the caller treats that as "no NB
    baseline available").

    Args:
        y: Non-negative integer-valued counts.

    Returns:
        P(Y = 0) under the fitted NB, in [0, 1]; NaN on fit failure.
    """
    design = np.ones((len(y), 1))
    try:
        result = NegativeBinomial(y, design).fit(disp=0, maxiter=_NB_MAXITER)
    except Exception as exc:  # statsmodels raises a grab-bag of LinAlg/convergence errors.
        logger.warning("NB intercept-only fit failed: %s", exc)
        return float("nan")
    mu = float(np.exp(result.params[0]))
    alpha = float(result.params[-1])
    if not (math.isfinite(mu) and math.isfinite(alpha)) or alpha <= 0 or mu <= 0:
        return float("nan")
    # statsmodels NB2 parameterization: size = 1/alpha, prob = size / (size + mu).
    size = 1.0 / alpha
    prob = size / (size + mu)
    return float(stats.nbinom.pmf(0, size, prob))


def _zip_index(y: np.ndarray) -> float:
    """Puig-Valero zero-inflation index relative to Poisson: 1 + ln(p0)/mean.

    Zero (or NaN if undefined) means "as many zeros as a Poisson with this
    mean"; positive means excess zeros. Guards mean > 0 and p0_obs > 0.
    """
    mean = float(y.mean())
    p0_obs = float((y == 0).mean())
    if mean <= 0 or p0_obs <= 0:
        return float("nan")
    return 1.0 + math.log(p0_obs) / mean


def _zinb_index(y: np.ndarray, nb_zero_prob: float) -> float:
    """Excess-zero measure relative to a fitted NB (Blasco-Moreno et al. 2019).

    ``(p0_obs - p0_NB) / (1 - p0_NB)``: the fraction of the non-NB-zero mass that
    shows up as extra observed zeros. 0 means the NB already explains the zeros;
    positive means structural excess zeros beyond what dispersion alone buys.
    """
    p0_obs = float((y == 0).mean())
    if not math.isfinite(nb_zero_prob) or nb_zero_prob >= _NB_ZERO_MASS_CEIL:
        return float("nan")
    return (p0_obs - nb_zero_prob) / (1.0 - nb_zero_prob)


def _bootstrap_ci(
    y: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float]:
    """Percentile bootstrap CI for ``statistic`` by resampling ``y`` with replacement.

    Args:
        y: The sample to resample.
        statistic: Maps a resample to a scalar; non-finite draws are dropped.
        rng: Seeded generator (``np.random.default_rng``) for reproducibility.
        n_boot: Number of resamples.

    Returns:
        ``(ci_lo, ci_hi)`` at the 2.5/97.5 percentiles; ``(nan, nan)`` if every
        resample produced a non-finite value.
    """
    n = len(y)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[i] = statistic(y[idx])
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(finite, [_CI_LOW_PCT, _CI_HIGH_PCT])
    return float(lo), float(hi)


def _zero_inflation_indices(
    y: np.ndarray, rng: np.random.Generator, n_boot: int
) -> ZeroInflationIndices:
    """Compute both zero-inflation indices with bootstrap percentile CIs.

    Both indices and both CIs are computed on the same marginal sample. The NB
    baseline used by the ziNB index is refit inside each bootstrap resample so
    the CI reflects NB-fit uncertainty, not just the empirical zero rate.

    Args:
        y: Marginal count target.
        rng: Seeded generator for the resampling.
        n_boot: Number of bootstrap resamples.

    Returns:
        A :class:`ZeroInflationIndices` with point estimates and 95% CIs.
    """
    zip_point = _zip_index(y)
    zinb_point = _zinb_index(y, _fit_nb_zero_prob(y))

    def _zinb_stat(sample: np.ndarray) -> float:
        return _zinb_index(sample, _fit_nb_zero_prob(sample))

    zip_lo, zip_hi = _bootstrap_ci(y, _zip_index, rng, n_boot)
    zinb_lo, zinb_hi = _bootstrap_ci(y, _zinb_stat, rng, n_boot)
    return ZeroInflationIndices(
        zip_index=zip_point,
        zip_ci_lo=zip_lo,
        zip_ci_hi=zip_hi,
        zinb_index=zinb_point,
        zinb_ci_lo=zinb_lo,
        zinb_ci_hi=zinb_hi,
    )


def _wilson_einbeck_pvalue(y: np.ndarray) -> float:
    """Score-test p-value for zero-inflation against a Poisson null.

    Wilson & Einbeck (Statistical Modelling, 2019) derive a score test for the
    zero-inflation parameter. Their intercept-only score statistic reduces to a
    comparison of the observed zero count against the Poisson-expected zero
    count, studentized by the score-test variance. We use the closed-form
    homogeneous-Poisson score statistic (van den Broek 1995, the special case
    the Wilson-Einbeck test generalizes):

        S = (n0 - n*e^{-mu})^2 / (n*e^{-mu}*(1 - e^{-mu}) - n*mu*e^{-2mu})

    where ``mu`` is the sample mean (the Poisson MLE), ``n0`` the observed zero
    count, and ``n`` the sample size. S is chi-squared with 1 df under the null
    of no zero-inflation; we return the upper-tail p-value. Small p => reject the
    Poisson, evidence for zero-inflation.

    Args:
        y: Marginal count target.

    Returns:
        Upper-tail chi-squared(1) p-value; NaN if the variance term is non-positive
        (degenerate, e.g. mu = 0).
    """
    n = len(y)
    mu = float(y.mean())
    if mu <= 0:
        return float("nan")
    n_zero = float((y == 0).sum())
    exp_mu = math.exp(-mu)
    expected_zero = n * exp_mu
    variance = n * exp_mu * (1.0 - exp_mu) - n * mu * exp_mu**2
    if variance <= 0:
        return float("nan")
    score = (n_zero - expected_zero) ** 2 / variance
    return float(stats.chi2.sf(score, df=1))


def _hurdle_nb_loglikeobs(y: np.ndarray) -> tuple[np.ndarray, int] | None:
    """Per-observation log-likelihood of an intercept-only Hurdle-NB.

    A hurdle decomposes into a Bernoulli zero/positive gate and a zero-truncated
    NB on the positives. statsmodels has no hurdle model, so we build the density
    directly: ``log p_zero`` for zeros and
    ``log(1 - p_zero) + logpmf_NB(k) - log(1 - pmf_NB(0))`` for positives, where
    the NB is fitted only on the positive counts. The parameter count is 3
    (gate probability + NB intercept + NB dispersion).

    Args:
        y: Marginal count target.

    Returns:
        ``(loglikeobs, k_params)`` or ``None`` if the positive-count NB fit fails
        or there are no positive observations.
    """
    is_zero = y == 0
    positives = y[~is_zero]
    if positives.size == 0:
        return None
    p_zero = float(is_zero.mean())
    design = np.ones((len(positives), 1))
    try:
        result = NegativeBinomial(positives, design).fit(disp=0, maxiter=_NB_MAXITER)
    except Exception as exc:
        logger.warning("Hurdle positive-NB fit failed: %s", exc)
        return None
    mu = float(np.exp(result.params[0]))
    alpha = float(result.params[-1])
    if not (math.isfinite(mu) and math.isfinite(alpha)) or alpha <= 0 or mu <= 0:
        return None
    size = 1.0 / alpha
    prob = size / (size + mu)
    nb_zero = float(stats.nbinom.pmf(0, size, prob))
    if nb_zero >= _NB_ZERO_MASS_CEIL:
        return None

    loglik = np.empty(len(y), dtype=float)
    # log p_zero is only finite when at least one zero exists; the gate is the
    # empirical zero rate so p_zero > 0 whenever is_zero.any().
    loglik[is_zero] = math.log(p_zero) if p_zero > 0 else 0.0
    trunc = stats.nbinom.logpmf(positives, size, prob) - math.log(1.0 - nb_zero)
    loglik[~is_zero] = math.log(1.0 - p_zero) + trunc
    # 3 free parameters: gate probability + NB intercept + NB dispersion alpha.
    return loglik, 3


def _zinb_loglikeobs(y: np.ndarray) -> tuple[np.ndarray, int] | None:
    """Per-observation log-likelihood of an intercept-only ZINB and its k_params.

    Fitted with L-BFGS rather than statsmodels' default BFGS: the default
    Hessian-based step diverges to NaN log-likelihoods on several of the real
    count cells (e.g. NBA_FG3M), whereas L-BFGS converges cleanly to the same
    optimum. Returns ``None`` (Vuong becomes NaN) if the fit still degenerates.
    """
    design = np.ones((len(y), 1))
    try:
        model = ZeroInflatedNegativeBinomialP(y, design, exog_infl=design, p=2)
        result = model.fit(disp=0, maxiter=_NB_MAXITER, method="lbfgs")
        loglik = np.asarray(model.loglikeobs(np.asarray(result.params)))
    except Exception as exc:
        logger.warning("ZINB loglikeobs fit failed: %s", exc)
        return None
    if not np.isfinite(loglik).all():
        logger.warning("ZINB loglikeobs produced non-finite values; skipping Vuong.")
        return None
    return loglik, len(result.params)


def _vuong_schwarz(y: np.ndarray) -> float:
    """Schwarz(BIC)-corrected Vuong statistic for Hurdle-NB vs ZINB.

    Both are non-nested intercept-only count models. Following Wilson (2015,
    "The misuse of the Vuong test for non-nested models...") we apply the
    Schwarz correction to the likelihood-ratio numerator and studentize by the
    per-observation log-likelihood-ratio standard deviation:

        m_i  = ll_hurdle_i - ll_zinb_i           (per-obs LR contributions)
        LR*  = sum(m_i) - (k_zinb - k_hurdle)/2 * ln(n)
        V    = LR* / (sqrt(n) * sd(m_i))

    Sign convention: **positive V favors the first model (Hurdle-NB)**; negative
    favors the ZINB. NaN if either fit fails or the per-obs LR has zero variance.

    Args:
        y: Marginal count target.

    Returns:
        The corrected Vuong statistic (approximately standard normal under the
        null that the two models fit equally well).
    """
    hurdle = _hurdle_nb_loglikeobs(y)
    zinb = _zinb_loglikeobs(y)
    if hurdle is None or zinb is None:
        return float("nan")
    ll_hurdle, k_hurdle = hurdle
    ll_zinb, k_zinb = zinb
    diff = ll_hurdle - ll_zinb
    sd = float(diff.std(ddof=1))
    if sd <= 0:
        return float("nan")
    n = len(y)
    schwarz = (k_zinb - k_hurdle) / 2.0 * math.log(n)
    numerator = float(diff.sum()) - schwarz
    return numerator / (math.sqrt(n) * sd)


def _route(stats_row: dict) -> str:
    """Map a per-cell statistics row to a recommended model family.

    Decision tree (applied in order):
      1. ``var_mean_ratio < _UNDERDISPERSION_RATIO`` -> ``cmp`` (under-dispersed;
         NB/ZINB cannot model variance below the mean).
      2. else if the ziNB bootstrap CI excludes 0 -> ``mzinb`` (robust structural
         zero-inflation beyond NB dispersion).
      3. else if the ziNB CI contains 0 and dispersion is mild
         (``var_mean_ratio < _MILD_DISPERSION_RATIO``) -> ``plain_nb``.
      4. else -> ``hurdle_nb_ztnb`` (no structural inflation but enough dispersion
         that a hurdle's separate gate + zero-truncated NB is the safer default).

    Args:
        stats_row: Needs ``var_mean_ratio``, ``ziNB_ci_lo``, ``ziNB_ci_hi``.

    Returns:
        One of :data:`ROUTING_CHOICES`.
    """
    var_mean_ratio = stats_row["var_mean_ratio"]
    if var_mean_ratio < _UNDERDISPERSION_RATIO:
        return "cmp"
    ci_lo = stats_row["ziNB_ci_lo"]
    ci_hi = stats_row["ziNB_ci_hi"]
    ci_excludes_zero = math.isfinite(ci_lo) and math.isfinite(ci_hi) and (ci_lo > 0 or ci_hi < 0)
    if ci_excludes_zero:
        return "mzinb"
    if var_mean_ratio < _MILD_DISPERSION_RATIO:
        return "plain_nb"
    return "hurdle_nb_ztnb"


def _compute_cell_stats(
    league: str, market: str, y: np.ndarray, rng: np.random.Generator, n_boot: int
) -> dict:
    """Compute the full diagnostics row for one (league, market) marginal target.

    Args:
        league: League code (e.g. ``"NBA"``).
        market: Market name as it appears in ``ALL_MARKETS``.
        y: Finite non-negative count target.
        rng: Seeded generator for the bootstrap CIs.
        n_boot: Bootstrap resample count.

    Returns:
        A dict keyed by :data:`REQUIRED_COLUMNS`.
    """
    mean = float(y.mean())
    variance = float(y.var(ddof=1))
    var_mean_ratio = variance / mean if mean > 0 else float("nan")
    positives = y[y > 0]
    indices = _zero_inflation_indices(y, rng, n_boot)
    row = {
        "league": league,
        "market": market,
        "n": len(y),
        "observed_mean": mean,
        "variance": variance,
        "var_mean_ratio": var_mean_ratio,
        "zero_rate": float((y == 0).mean()),
        "cond_pos_mean": float(positives.mean()) if positives.size else float("nan"),
        "ziP_index": indices.zip_index,
        "ziP_ci_lo": indices.zip_ci_lo,
        "ziP_ci_hi": indices.zip_ci_hi,
        "ziNB_index": indices.zinb_index,
        "ziNB_ci_lo": indices.zinb_ci_lo,
        "ziNB_ci_hi": indices.zinb_ci_hi,
        "wilson_einbeck_pvalue": _wilson_einbeck_pvalue(y),
        "vuong_schwarz_stat": _vuong_schwarz(y),
    }
    row["routing_recommendation"] = _route(row)
    return row


def _write_league_parquet(rows: list[dict], league: str, out_dir: Path) -> Path:
    """Write one league's diagnostics rows to ``{out_dir}/{league}_diagnostics.parquet``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=list(REQUIRED_COLUMNS))
    path = out_dir / f"{league}_diagnostics.parquet"
    frame.to_parquet(path, engine="pyarrow", index=False)
    return path


@click.command()
@click.option(
    "--league",
    type=click.Choice(_ROUTED_LEAGUES),
    default=None,
    help="Restrict to one league (default: all three ZINB leagues).",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=Path("data/zinb_routing"),
    show_default=True,
    help="Directory for the per-league diagnostics parquets.",
)
@click.option(
    "--bootstrap",
    default=_DEFAULT_BOOTSTRAP,
    show_default=True,
    help="Bootstrap resamples for the zero-inflation index CIs.",
)
@click.option(
    "--seed",
    default=_DEFAULT_SEED,
    show_default=True,
    help="RNG seed for the bootstrap resampling.",
)
def main(league: str | None, out_dir: Path, bootstrap: int, seed: int) -> None:
    """Score ZINB cells on the marginal count target and recommend a model family."""
    if bootstrap <= 0:
        raise click.UsageError("--bootstrap must be a positive integer.")
    cells = zinb_cells()
    if league is not None:
        cells = [(lg, market) for lg, market in cells if lg == league]
    if not cells:
        click.echo(f"No ZINB cells for league filter {league!r}.")
        return

    rng = np.random.default_rng(seed)
    rows_by_league: dict[str, list[dict]] = {}
    for cell_league, market in tqdm(cells, desc="Scoring ZINB cells"):
        y = _load_target(cell_league, market)
        if y is None or y.size == 0:
            continue
        row = _compute_cell_stats(cell_league, market, y, rng, bootstrap)
        rows_by_league.setdefault(cell_league, []).append(row)

    out_dir = Path(out_dir)
    for cell_league, rows in rows_by_league.items():
        path = _write_league_parquet(rows, cell_league, out_dir)
        click.echo(f"Wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
