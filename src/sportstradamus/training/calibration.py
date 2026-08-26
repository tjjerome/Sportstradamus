"""Model calibration: book weight fitting, model-blend weight, distribution selection."""

import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.stats import fit, gamma, nbinom, norm, poisson, skew, skewnorm

from sportstradamus.helpers import (
    dp_crps,
    fused_loc,
    gamma_crps,
    get_logger,
    negbin_crps,
    skewnorm_crps,
    skewnormal_loc_from_mean,
    stat_cv,
)
from sportstradamus.helpers.distributions import _dp_cdf_pmf, _dp_mu_from_mean

# Per-observation log-likelihood clamp for the DPO blend-weight fit — mirrors the
# -20 outlier clamp in fit_model_weight so w is fit on the same scale.
_DPO_WEIGHT_LOGLIK_CLAMP: float = -20.0

logger = get_logger(__name__)

# Blend-weight bounds for the model-vs-book optimization. 0.05 keeps a minimum
# bookmaker signal even when the model is dominant; 0.9 keeps a minimum model
# signal even when the bookmaker is dominant. Both ends prevent degenerate
# all-one-source blends that give log-likelihood no room to improve.
_MODEL_WEIGHT_MIN: float = 0.05
_MODEL_WEIGHT_MAX: float = 0.9

# crps_1se grid: 35 points give 0.025 steps over the weight bounds — fine enough that
# the 1-SE pick is grid-limited, coarse enough that the n×35 loss matrix stays cheap.
_ONE_SE_GRID_POINTS: int = 35
_ONE_SE_GRID: np.ndarray = np.linspace(_MODEL_WEIGHT_MIN, _MODEL_WEIGHT_MAX, _ONE_SE_GRID_POINTS)
# κ = one standard error — the single global shrinkage intensity; never tuned per cell
# or league (a per-cell κ reopens the selection channel the brief warns about).
_ONE_SE_KAPPA: float = 1.0
# Player-clustered paired bootstrap of the loss difference: fixed draws + seed keep the
# fitted weight deterministic run-to-run (seed matches the brief's probe).
_ONE_SE_BOOTSTRAP_DRAWS: int = 2000
_ONE_SE_BOOTSTRAP_SEED: int = 1729
# Below this many distinct clusters the clustered SE is too noisy to trust; the fit
# collapses to the plain grid argmin instead.
_ONE_SE_MIN_CLUSTERS: int = 10

# Resolution threshold for choosing NegBin over Gamma: the per-player ratio
# step/nonzero_mean measures how "count-like" (integer-stepped, low-mean) the
# distribution is.  Above 0.2 the stat fits a count model better than a
# continuous one (empirically validated on NBA/NFL/NHL markets).
_NEGBIN_RESOLUTION_THRESHOLD: float = 0.2

# Zero-inflation threshold for upgrading NegBin → ZINB.  Excess zeros above
# 10% of total observations are more than Poisson mixing can absorb in NegBin;
# the ZI component prevents the shape parameter from bloating to compensate.
_ZINB_ZERO_INFLATION_THRESHOLD: float = 0.1

# Zero-inflation threshold for upgrading Gamma → ZAGamma (zero-augmented).
# Gamma is continuous and can't model a genuine spike at 0; above 5% excess
# zeros the ZAGamma component is needed to avoid calibration drift at the
# over/under threshold near the bookmaker line.
_ZAGAMMA_ZERO_INFLATION_THRESHOLD: float = 0.05

# Minimum graded (book line vs realized result) samples before per-book weights
# are fit. With fewer, the fit overfits a handful of games, so the caller keeps
# the prior weights instead.
_MIN_SAMPLES_FOR_BOOK_FIT: int = 9

# Minimum graded rows in a single line bin before that bin's conditional moments are
# trusted for the book-shape fit. Below it the bin's variance/skew are too noisy.
_MIN_ROWS_PER_SHAPE_BIN: int = 120

# Minimum usable line bins before a per-cell shape curve is fit. Two points pin a line
# with zero residual freedom; three is the floor for a fit with any room to disagree.
_MIN_BINS_FOR_SHAPE_FIT: int = 3


def _extract_result_and_test_df(market, df, stat_data):
    """Attach realized results to ``df`` and split off the per-book test frame.

    Returns ``(result, test_df)`` — the realized outcome (win→{0,1} for
    Moneyline, else float) and ``df`` with its ``Result`` column dropped.
    """
    if market == "Moneyline":
        log = stat_data.teamlog[
            [
                stat_data.log_strings["team"],
                stat_data.log_strings["date"],
                stat_data.log_strings["win"],
            ]
        ].copy()
        log[stat_data.log_strings["date"]] = log[stat_data.log_strings["date"]].str[:10]
        df["Result"] = log.drop_duplicates(
            [stat_data.log_strings["date"], stat_data.log_strings["team"]]
        ).set_index([stat_data.log_strings["date"], stat_data.log_strings["team"]])[
            stat_data.log_strings["win"]
        ]
        df.dropna(subset="Result", inplace=True)
        result = (df["Result"] == "W").astype(int)
        test_df = df.drop(columns="Result")

    elif market == "Totals":
        log = stat_data.teamlog[
            [
                stat_data.log_strings["team"],
                stat_data.log_strings["date"],
                stat_data.log_strings["score"],
            ]
        ].copy()
        log[stat_data.log_strings["date"]] = log[stat_data.log_strings["date"]].str[:10]
        df["Result"] = log.drop_duplicates(
            [stat_data.log_strings["date"], stat_data.log_strings["team"]]
        ).set_index([stat_data.log_strings["date"], stat_data.log_strings["team"]])[
            stat_data.log_strings["score"]
        ]
        df.dropna(subset="Result", inplace=True)
        result = df["Result"].astype(float)
        test_df = df.drop(columns="Result")

    elif market == "1st 1 innings":
        log = stat_data.gamelog.loc[
            stat_data.gamelog["starting pitcher"],
            ["opponent", stat_data.log_strings["date"], "1st inning runs allowed"],
        ].copy()
        log[stat_data.log_strings["date"]] = log[stat_data.log_strings["date"]].str[:10]
        df["Result"] = log.drop_duplicates([stat_data.log_strings["date"], "opponent"]).set_index(
            [stat_data.log_strings["date"], "opponent"]
        )["1st inning runs allowed"]
        df.dropna(subset="Result", inplace=True)
        result = df["Result"].astype(float)
        test_df = df.drop(columns="Result")

    else:
        log = stat_data.gamelog[
            [stat_data.log_strings["player"], stat_data.log_strings["date"], market]
        ].copy()
        log[stat_data.log_strings["date"]] = log[stat_data.log_strings["date"]].str[:10]
        df["Result"] = log.drop_duplicates(
            [stat_data.log_strings["date"], stat_data.log_strings["player"]]
        ).set_index([stat_data.log_strings["date"], stat_data.log_strings["player"]])[market]
        df.dropna(subset="Result", inplace=True)
        result = df["Result"].astype(float)
        test_df = df.drop(columns="Result")

    return result, test_df


def _make_book_objective(market, dist, cv):
    """Build the per-observation negative-log-likelihood objective for the fit.

    Moneyline uses log-loss over de-vigged win probabilities; count families
    (NegBin/ZINB/Poisson/DPO) use the Poisson logpmf of a weighted geometric-mean
    projection; everything else uses a precision-weighted Normal logpdf.
    """
    if market == "Moneyline":
        from sklearn.metrics import log_loss

        def objective(w, x, y):
            prob = np.exp(
                np.ma.average(np.ma.MaskedArray(np.log(x), mask=np.isnan(x)), weights=w, axis=1)
            )
            return log_loss(y[~np.ma.getmask(prob)], np.ma.getdata(prob)[~np.ma.getmask(prob)])

        return objective

    if dist in ["NegBin", "ZINB", "Poisson", "DPO"]:

        def objective(w, x, y):
            proj = np.array(
                np.exp(
                    np.ma.average(np.ma.MaskedArray(np.log(x), mask=np.isnan(x)), weights=w, axis=1)
                )
            )
            return -np.mean(poisson.logpmf(y.astype(int), proj))

        return objective

    def objective(w, x, y):
        s = np.ma.MaskedArray(x * cv, mask=np.isnan(x))
        std = np.sqrt(1 / np.ma.average(np.power(s, -2), weights=w, axis=1))
        proj = np.array(
            np.ma.average(
                np.ma.MaskedArray(x * np.power(s, -2), mask=np.isnan(x)), weights=w, axis=1
            )
            * std
            * std
        )
        return -np.mean(norm.logpdf(y, proj, std))

    return objective


def fit_book_weights(league: str, market: str, stat_data, archive, book_weights: dict) -> dict:
    """Fit optimal weights for multiple sportsbooks using historical accuracy."""
    warnings.simplefilter("ignore", UserWarning)
    from sportstradamus.training.config import load_distribution_config

    logger.info("Fitting Book Weights - %s, %s", league, market)
    df = archive.to_pandas(league, market)
    df = df[[col for col in df.columns if col != "pinnacle"]]
    if len([col for col in df.columns if col not in ["Line", "Result", "Over"]]) == 0:
        return {}
    cv = stat_cv[league].get(market, 1)
    stat_dist = load_distribution_config()
    dist = stat_dist.get(league, {}).get(market, "Poisson")

    result, test_df = _extract_result_and_test_df(market, df, stat_data)
    objective = _make_book_objective(market, dist, cv)

    if "Line" in test_df.columns:
        test_df.drop(columns=["Line"], inplace=True)

    x = test_df.loc[~test_df.isna().all(axis=1)].to_numpy()
    x[x < 0] = np.nan
    y = result.loc[~test_df.isna().all(axis=1)].to_numpy()
    if len(x) > _MIN_SAMPLES_FOR_BOOK_FIT:
        prev_weights = book_weights.get(league, {}).get(market, {})
        guess = {}
        for book in test_df.columns:
            guess.update({book: prev_weights.get(book, 1)})

        guess = list(guess.values())
        guess = np.clip(guess / np.sum(guess), 0.005, 0.75)
        res = minimize(
            objective,
            guess,
            args=(x, y),
            bounds=[(0.001, 1)] * len(test_df.columns),
            tol=1e-8,
            method="TNC",
        )

        return {k: res.x[i] for i, k in enumerate(test_df.columns)}
    return {}


def fit_book_shape(league: str, market: str, results, lines) -> dict | None:
    """Fit the book's conditional ``(variance, skewness)`` curves from a cell's history.

    Bins realized ``results`` by the book ``line`` they were graded against, keeps the bins
    clearing ``_MIN_ROWS_PER_SHAPE_BIN``, and over each bin's conditional mean ``μ`` fits:

    * ``var = a·μ^b`` (Taylor's power law) by sqrt(n)-weighted least squares in log space.
      ``b`` is free to cross the Poisson line ``var = μ`` — sub-Poisson count cells (WNBA
      DREB, ``b ≈ 0.34``) are real and must not be forced to ``b ≥ 1``.
    * ``γ = skew_c + skew_d·μ`` (linear) under the same weighting. Evaluation clamps γ to the
      SkewNormal-admissible band downstream (in :func:`book_skewnormal_shape` via
      ``skewnormal_params_from_moments``), so the raw linear is never extrapolated into an
      infeasible skew.

    Returns ``{a, b, skew_c, skew_d, n_bins}``, or ``None`` when fewer than
    ``_MIN_BINS_FOR_SHAPE_FIT`` bins clear the row floor (the caller keeps the cell's
    constant-CV symmetric shape). ``train_market`` calls this for each SkewNormal cell and
    persists the result via :func:`~sportstradamus.training.config.save_book_shape_config`.
    """
    results = np.asarray(results, dtype=float)
    lines = np.asarray(lines, dtype=float)
    mu, var, bin_skew, weight = [], [], [], []
    for line in np.unique(lines):
        bin_results = results[lines == line]
        if len(bin_results) < _MIN_ROWS_PER_SHAPE_BIN:
            continue
        mu.append(bin_results.mean())
        var.append(bin_results.var(ddof=1))
        bin_skew.append(float(skew(bin_results)))
        weight.append(np.sqrt(len(bin_results)))

    if len(mu) < _MIN_BINS_FOR_SHAPE_FIT:
        logger.info(
            "Book shape fit - %s, %s: %d usable line bins (< %d) — constant-CV fallback",
            league,
            market,
            len(mu),
            _MIN_BINS_FOR_SHAPE_FIT,
        )
        return None

    mu, var, bin_skew, weight = map(np.asarray, (mu, var, bin_skew, weight))
    b, log_a = np.polyfit(np.log(mu), np.log(var), 1, w=weight)
    skew_d, skew_c = np.polyfit(mu, bin_skew, 1, w=weight)
    return {
        "a": float(np.exp(log_a)),
        "b": float(b),
        "skew_c": float(skew_c),
        "skew_d": float(skew_d),
        "n_bins": len(mu),
    }


def _minimize_weight(objective) -> float:
    """Minimize a per-weight blend objective over ``[_MODEL_WEIGHT_MIN, _MODEL_WEIGHT_MAX]`` (TNC)."""
    res = minimize(
        objective, 0.5, bounds=[(_MODEL_WEIGHT_MIN, _MODEL_WEIGHT_MAX)], tol=1e-8, method="TNC"
    )
    return res.x[0]


def fit_model_weight(
    model_ev,
    odds_ev,
    result,
    dist,
    model_alpha=None,
    model_r=None,
    cv=None,
    model_sigma=None,
    model_skew_alpha=None,
    gate_model=None,
    gate_book=None,
) -> float:
    """Optimize the single blend weight between model predictions and
    bookmaker lines by maximizing clamped log-likelihood on validation data.

    Log-likelihood is clamped at -20 per observation to prevent outlier
    domination while preserving per-observation conditional discrimination.

    Returns a single float w in [0.05, 0.9].

    - NegBin: uses the logarithmic opinion pool — geometric mean of
      both μ and r with a single weight w.  The book's r is 1/cv.
    - Gamma: precision-weighted blend using model alpha and book
      alpha (1/cv²).
    - SkewNormal: precision-weighted blend of loc/sigma, linear blend of alpha.

    When gate_model/gate_book are provided, the likelihood accounts for
    zero-inflation: P(y) = gate*I(y=0) + (1-gate)*base_pdf(y).
    """
    result = np.asarray(result, dtype=float)
    model_ev = np.asarray(model_ev, dtype=float)
    odds_ev = np.asarray(odds_ev, dtype=float)
    has_gate = gate_model is not None and gate_book is not None
    has_hurdle_gate = gate_book is not None and gate_model is None

    if dist == "SkewNormal":
        model_sigma_arr = np.asarray(model_sigma, dtype=float)
        model_skew_arr = np.asarray(model_skew_alpha, dtype=float)

        def objective(w):
            bl_ev, bl_sigma, bl_alpha, g_blend = fused_loc(
                w,
                model_ev,
                odds_ev,
                cv,
                "SkewNormal",
                sigma=model_sigma_arr,
                skew_alpha=model_skew_arr,
                gate_model=gate_model,
                gate_book=gate_book,
            )

            delta = bl_alpha / np.sqrt(1 + bl_alpha**2)
            bl_loc = bl_ev - bl_sigma * delta * np.sqrt(2 / np.pi)

            base_logpdf = np.clip(
                skewnorm.logpdf(result, bl_alpha, loc=bl_loc, scale=bl_sigma), -20, 0
            )

            if (has_gate or has_hurdle_gate) and g_blend is not None:
                loglik = np.where(
                    result == 0,
                    np.log(np.clip(g_blend, 1e-12, None)),
                    np.log(np.clip(1 - g_blend, 1e-12, None)) + base_logpdf,
                )
                return -np.mean(loglik)
            return -np.mean(base_logpdf)

        return _minimize_weight(objective)

    if dist == "NegBin":
        model_r_arr = np.asarray(model_r, dtype=float)
        result_int = result.astype(int)

        def objective(w):
            r_blend, p_blend, g_blend = fused_loc(
                w,
                model_ev,
                odds_ev,
                cv,
                "NegBin",
                r=model_r_arr,
                gate_model=gate_model,
                gate_book=gate_book,
            )
            base_logpmf = np.clip(nbinom.logpmf(result_int, r_blend, p_blend), -20, 0)
            if has_gate:
                loglik = np.where(
                    result_int == 0,
                    np.log(np.clip(g_blend + (1 - g_blend) * np.exp(base_logpmf), 1e-12, None)),
                    np.log(np.clip(1 - g_blend, 1e-12, None)) + base_logpmf,
                )
                return -np.mean(loglik)
            return -np.mean(base_logpmf)

        return _minimize_weight(objective)
    model_alpha_arr = np.asarray(model_alpha, dtype=float)

    def objective(w):
        alpha_bl, beta_bl, g_blend = fused_loc(
            w,
            model_ev,
            odds_ev,
            cv,
            "Gamma",
            alpha=model_alpha_arr,
            gate_model=gate_model,
            gate_book=gate_book,
        )
        base_logpdf = np.clip(gamma.logpdf(result, alpha_bl, scale=1 / beta_bl), -20, 0)
        if has_gate:
            loglik = np.where(
                result == 0,
                np.log(np.clip(g_blend, 1e-12, None)),
                np.log(np.clip(1 - g_blend, 1e-12, None)) + base_logpdf,
            )
            return -np.mean(loglik)
        return -np.mean(base_logpdf)

    return _minimize_weight(objective)


def _crps_loss_vector(
    model_ev,
    odds_ev,
    result,
    dist,
    model_alpha=None,
    model_r=None,
    cv=None,
    model_sigma=None,
    model_skew_alpha=None,
    gate_model=None,
    gate_book=None,
):
    """Build the per-family per-observation CRPS of the blended predictive as a function of ``w``.

    Shared by :func:`fit_model_weight_crps` (which minimizes the mean) and
    :func:`fit_model_weight_crps_1se` (which needs the whole loss vector at each grid point).
    """
    result = np.asarray(result, dtype=float)
    model_ev = np.asarray(model_ev, dtype=float)
    odds_ev = np.asarray(odds_ev, dtype=float)

    if dist == "SkewNormal":
        model_sigma_arr = np.asarray(model_sigma, dtype=float)
        model_skew_arr = np.asarray(model_skew_alpha, dtype=float)

        def loss_vector(w):
            bl_ev, bl_sigma, bl_alpha, g_blend = fused_loc(
                w,
                model_ev,
                odds_ev,
                cv,
                "SkewNormal",
                sigma=model_sigma_arr,
                skew_alpha=model_skew_arr,
                gate_model=gate_model,
                gate_book=gate_book,
            )
            bl_loc = skewnormal_loc_from_mean(bl_ev, bl_sigma, bl_alpha)
            return skewnorm_crps(result, bl_loc, bl_sigma, bl_alpha, gate=g_blend)

        return loss_vector

    if dist == "NegBin":
        model_r_arr = np.asarray(model_r, dtype=float)

        def loss_vector(w):
            r_blend, p_blend, g_blend = fused_loc(
                w,
                model_ev,
                odds_ev,
                cv,
                "NegBin",
                r=model_r_arr,
                gate_model=gate_model,
                gate_book=gate_book,
            )
            return negbin_crps(result, r_blend, p_blend, gate=g_blend)

        return loss_vector

    model_alpha_arr = np.asarray(model_alpha, dtype=float)

    def loss_vector(w):
        alpha_bl, beta_bl, g_blend = fused_loc(
            w,
            model_ev,
            odds_ev,
            cv,
            "Gamma",
            alpha=model_alpha_arr,
            gate_model=gate_model,
            gate_book=gate_book,
        )
        return gamma_crps(result, alpha_bl, 1 / beta_bl, gate=g_blend)

    return loss_vector


def fit_model_weight_crps(
    model_ev,
    odds_ev,
    result,
    dist,
    model_alpha=None,
    model_r=None,
    cv=None,
    model_sigma=None,
    model_skew_alpha=None,
    gate_model=None,
    gate_book=None,
) -> float:
    """Fit the model↔book blend weight by minimizing mean CRPS of the blended predictive.

    Mirrors :func:`fit_model_weight` — same ``fused_loc`` blend, same ``[0.05, 0.9]`` bounds, same
    zero-inflation gate plumbing — but swaps the clamped log-likelihood for the strictly-proper,
    bounded CRPS. CRPS needs no ``-20`` clamp, so it drops the clamp's left-tail bias on ``w``
    (the robustness lever for heavy-tailed cells; see ``/tmp/researcher_crps_blending.md``). The
    gated CDF carries the zero spike, so no separate ``y==0`` term is added.

    Returns a single float w in [0.05, 0.9].
    """
    loss_vector = _crps_loss_vector(
        model_ev,
        odds_ev,
        result,
        dist,
        model_alpha=model_alpha,
        model_r=model_r,
        cv=cv,
        model_sigma=model_sigma,
        model_skew_alpha=model_skew_alpha,
        gate_model=gate_model,
        gate_book=gate_book,
    )
    return _minimize_weight(lambda w: np.mean(loss_vector(w)))


def _one_se_weight(losses: np.ndarray, clusters) -> float:
    """Smallest ``_ONE_SE_GRID`` weight whose mean loss sits within ``_ONE_SE_KAPPA`` SEs of the argmin.

    ``losses`` is the n×G per-observation loss matrix over ``_ONE_SE_GRID``. For each grid
    weight below the argmin, the SE is the player-clustered paired bootstrap of the per-row
    loss difference against the argmin column — whole clusters resampled with replacement,
    ``scorecard._bootstrap_mean_ci_clustered``'s idiom. The rule only restricts: with no
    usable clusters (``None`` or fewer than ``_ONE_SE_MIN_CLUSTERS`` distinct), a non-finite
    SE, or no smaller weight inside the band, it returns the plain argmin.
    """
    mean_loss = losses.mean(axis=0)
    star = int(np.argmin(mean_loss))
    if clusters is None:
        return float(_ONE_SE_GRID[star])
    _, codes = np.unique(np.asarray(clusters), return_inverse=True)
    n_clusters = int(codes.max()) + 1
    if n_clusters < _ONE_SE_MIN_CLUSTERS:
        return float(_ONE_SE_GRID[star])

    rng = np.random.default_rng(_ONE_SE_BOOTSTRAP_SEED)
    picks = rng.integers(0, n_clusters, size=(_ONE_SE_BOOTSTRAP_DRAWS, n_clusters))
    counts = np.bincount(codes, minlength=n_clusters)
    rows_per_draw = counts[picks].sum(axis=1)
    for j in range(star):
        diff = losses[:, j] - losses[:, star]
        cluster_sums = np.bincount(codes, weights=diff, minlength=n_clusters)
        se = float(np.std(cluster_sums[picks].sum(axis=1) / rows_per_draw, ddof=1))
        if not np.isfinite(se):
            return float(_ONE_SE_GRID[star])
        if mean_loss[j] <= mean_loss[star] + _ONE_SE_KAPPA * se:
            return float(_ONE_SE_GRID[j])
    return float(_ONE_SE_GRID[star])


def fit_model_weight_crps_1se(
    model_ev,
    odds_ev,
    result,
    dist,
    model_alpha=None,
    model_r=None,
    cv=None,
    model_sigma=None,
    model_skew_alpha=None,
    gate_model=None,
    gate_book=None,
    clusters=None,
) -> float:
    """Fit the blend weight by a one-standard-error parsimony rule on the CRPS path.

    Same per-observation loss as :func:`fit_model_weight_crps`, evaluated over ``_ONE_SE_GRID``
    instead of TNC-minimized: take the argmin, then return the smallest grid weight whose mean
    CRPS is within ``_ONE_SE_KAPPA`` player-clustered paired-bootstrap SEs of it — Diebold–Pauly
    shrinkage of the estimated combination weight toward the book
    (``/tmp/researcher_blend_weight_slug.md``). On a no-edge cell the flat loss path drives ``w``
    to the floor; on a real-edge cell the band is tight and ``w`` stays near the argmin.

    ``clusters`` is the per-row player identity (date for team markets); without it the fit
    falls back to the plain grid argmin.

    Returns a single float w in [0.05, 0.9].
    """
    loss_vector = _crps_loss_vector(
        model_ev,
        odds_ev,
        result,
        dist,
        model_alpha=model_alpha,
        model_r=model_r,
        cv=cv,
        model_sigma=model_sigma,
        model_skew_alpha=model_skew_alpha,
        gate_model=gate_model,
        gate_book=gate_book,
    )
    losses = np.column_stack([loss_vector(w) for w in _ONE_SE_GRID])
    return _one_se_weight(losses, clusters)


def fit_dpo_weight(model_ev, book_ev, result, model_phi, cv, blending, clusters=None) -> float:
    """Blend-weight fit for the DPO family, mirroring :func:`fit_model_weight`.

    Not routed through :func:`fit_blend_weight` because that dispatcher's unknown-family
    fallback is the Gamma branch — silently wrong for DPO. Same ``_minimize_weight`` bounds
    and per-observation clamp; the objective is the blended DP log-pmf (``nll``),
    :func:`~sportstradamus.helpers.distributions.dp_crps` (``crps``), or the same 1-SE
    grid rule as :func:`fit_model_weight_crps_1se` over the ``dp_crps`` losses
    (``crps_1se``, which is what ``clusters`` feeds).
    """
    result = np.asarray(result, dtype=float)

    def _blended(w):
        mean_bl, phi_bl, _ = fused_loc(w, model_ev, book_ev, cv, "DPO", phi=model_phi)
        return _dp_mu_from_mean(mean_bl, phi_bl), phi_bl

    if blending == "crps_1se":
        losses = np.column_stack([dp_crps(result, *_blended(w)) for w in _ONE_SE_GRID])
        return _one_se_weight(losses, clusters)

    if blending == "crps":

        def objective(w):
            mu_bl, phi_bl = _blended(w)
            return float(np.mean(dp_crps(result, mu_bl, phi_bl)))

    else:

        def objective(w):
            mu_bl, phi_bl = _blended(w)
            _, pmf = _dp_cdf_pmf(result, mu_bl, phi_bl)
            with np.errstate(divide="ignore"):
                logpmf = np.log(pmf)
            return float(-np.mean(np.clip(logpmf, _DPO_WEIGHT_LOGLIK_CLAMP, 0.0)))

    return _minimize_weight(objective)


# Per-cell blend strategy: how the model and book distributions are combined.
# Each entry owns its weight-fitting objective and its weight bounds, so a strategy
# can change the objective (as `crps_1se` does) and/or the bounds (e.g. drop the
# 0.05 floor) without touching the others — but never a gate-scored functional like
# Brier-at-line (assay-sensitivity; /tmp/researcher_blend_weight_slug.md). Default
# `nll` reproduces the historical behavior exactly.
DEFAULT_BLENDING: str = "nll"
BLENDING_SLUGS: frozenset[str] = frozenset({"nll", "crps", "crps_1se"})


def fit_blend_weight(blending: str, *args, **kwargs) -> float:
    """Dispatch to the blend strategy's weight fitter. ``nll`` is the legacy
    clamped-log-likelihood objective in :func:`fit_model_weight`; ``crps`` is the
    bounded strictly-proper objective in :func:`fit_model_weight_crps`; ``crps_1se``
    is the 1-SE parsimony rule in :func:`fit_model_weight_crps_1se`.
    """
    if blending not in BLENDING_SLUGS:
        raise ValueError(f"Unknown blending slug {blending!r}; valid: {sorted(BLENDING_SLUGS)}")
    if blending == "crps_1se":
        return fit_model_weight_crps_1se(*args, **kwargs)
    # Only the 1-SE rule uses cluster identity; the TNC fitters keep their signatures.
    kwargs.pop("clusters", None)
    if blending == "crps":
        return fit_model_weight_crps(*args, **kwargs)
    return fit_model_weight(*args, **kwargs)


def select_distribution(player_stats) -> tuple[str, float]:
    """Recommend a distribution family by inspecting per-player data properties.

    Returns (dist_name, p_zero) where dist_name is one of NegBin/ZINB/Gamma/ZAGamma
    and p_zero is the estimated excess zero-inflation rate.
    """
    warnings.filterwarnings("ignore", "overflow", RuntimeWarning)

    sample = player_stats.first()
    is_integer = all(v == int(v) for v in sample)
    if is_integer:
        uniques = (
            player_stats.apply(lambda x: x.unique().tolist())
            .explode()
            .drop_duplicates()
            .sort_values()
        )
        step = uniques.diff().dropna().min() if len(uniques) > 1 else 1
    else:
        step = 0

    if not is_integer or step != 1:
        dist = "Gamma"
    else:

        def _player_resolution(x):
            nz = x[x > 0]
            return step / nz.mean() if len(nz) > 0 else np.nan

        resolutions = player_stats.apply(_player_resolution).dropna()
        resolution = resolutions.median()
        dist = "NegBin" if resolution > _NEGBIN_RESOLUTION_THRESHOLD else "Gamma"
        logger.info("  Resolution: %.4f (%s)", resolution, dist)

    observed_zeros = player_stats.agg(lambda x: x.eq(0).mean())

    if dist in ["NegBin", "ZINB"]:

        def _nb_mom(x):
            mu, var = x.mean(), x.var()
            if var <= mu:
                var = mu + 1e-6
            p = np.clip(mu / var, 1e-3, 1 - 1e-3)
            n = np.clip(mu * p / (1 - p), 0.1, 50)
            return (n, p)

        nb_fit = player_stats.apply(_nb_mom)
        base_zero_prob = nb_fit.apply(lambda row: nbinom.pmf(0, row[0], row[1]))
        p_zero = float(((observed_zeros - base_zero_prob) / (1 - base_zero_prob)).clip(0, 1).mean())
        if p_zero > _ZINB_ZERO_INFLATION_THRESHOLD:
            dist = "ZINB"
    else:
        gam_fit = player_stats.apply(
            lambda x: fit(gamma, x[x > 0].astype(float), {"a": (0, 50), "scale": (0, 500)}).params
        )
        base_zero_prob = gam_fit.apply(lambda row: gamma.cdf(0.99, row[0], scale=row[2]))
        p_zero = float(((observed_zeros - base_zero_prob) / (1 - base_zero_prob)).clip(0, 1).mean())
        if p_zero > _ZAGAMMA_ZERO_INFLATION_THRESHOLD:
            dist = "ZAGamma"

    data_type = f"integer (step={int(step)})" if is_integer else "continuous"
    logger.info("  Data type: %s", data_type)
    logger.info("  Zero inflation - %.4f", p_zero)
    logger.info("  Selected: %s", dist)

    return dist, p_zero
