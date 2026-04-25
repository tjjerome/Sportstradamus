"""Model calibration: book weight fitting, model-blend weight, distribution selection."""

import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.stats import fit, gamma, nbinom, norm, poisson, skewnorm

from sportstradamus.helpers import fused_loc, stat_cv


def fit_book_weights(league: str, market: str, stat_data, archive, book_weights: dict) -> dict:
    """Fit optimal weights for multiple sportsbooks using historical accuracy."""
    warnings.simplefilter("ignore", UserWarning)
    from sportstradamus.training.config import load_distribution_config

    print(f"Fitting Book Weights - {league}, {market}")
    df = archive.to_pandas(league, market)
    df = df[[col for col in df.columns if col != "pinnacle"]]
    if len([col for col in df.columns if col not in ["Line", "Result", "Over"]]) == 0:
        return {}
    cv = stat_cv[league].get(market, 1)
    stat_dist = load_distribution_config()
    dist = stat_dist.get(league, {}).get(market, "Poisson")

    if market == "Moneyline":
        log = stat_data.teamlog[
            [
                stat_data.log_strings["team"],
                stat_data.log_strings["date"],
                stat_data.log_strings["win"],
            ]
        ]
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
        ]
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
        ]
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
        ]
        log[stat_data.log_strings["date"]] = log[stat_data.log_strings["date"]].str[:10]
        df["Result"] = log.drop_duplicates(
            [stat_data.log_strings["date"], stat_data.log_strings["player"]]
        ).set_index([stat_data.log_strings["date"], stat_data.log_strings["player"]])[market]
        df.dropna(subset="Result", inplace=True)
        result = df["Result"].astype(float)
        test_df = df.drop(columns="Result")

    if market == "Moneyline":
        from sklearn.metrics import log_loss

        def objective(w, x, y):
            prob = np.exp(
                np.ma.average(np.ma.MaskedArray(np.log(x), mask=np.isnan(x)), weights=w, axis=1)
            )
            return log_loss(y[~np.ma.getmask(prob)], np.ma.getdata(prob)[~np.ma.getmask(prob)])

    elif dist in ["NegBin", "ZINB", "Poisson"]:

        def objective(w, x, y):
            proj = np.array(
                np.exp(
                    np.ma.average(np.ma.MaskedArray(np.log(x), mask=np.isnan(x)), weights=w, axis=1)
                )
            )
            return -np.mean(poisson.logpmf(y.astype(int), proj))

    else:

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

    if "Line" in test_df.columns:
        test_df.drop(columns=["Line"], inplace=True)

    x = test_df.loc[~test_df.isna().all(axis=1)].to_numpy()
    x[x < 0] = np.nan
    y = result.loc[~test_df.isna().all(axis=1)].to_numpy()
    if len(x) > 9:
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
    else:
        return {}


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
                gate_book=gate_book,
            )

            delta = bl_alpha / np.sqrt(1 + bl_alpha**2)
            bl_loc = bl_ev - bl_sigma * delta * np.sqrt(2 / np.pi)

            base_logpdf = np.clip(
                skewnorm.logpdf(result, bl_alpha, loc=bl_loc, scale=bl_sigma), -20, 0
            )

            if has_hurdle_gate and g_blend is not None:
                loglik = np.where(
                    result == 0,
                    np.log(np.clip(g_blend, 1e-12, None)),
                    np.log(np.clip(1 - g_blend, 1e-12, None)) + base_logpdf,
                )
                return -np.mean(loglik)
            return -np.mean(base_logpdf)

        res = minimize(objective, 0.5, bounds=[(0.05, 0.9)], tol=1e-8, method="TNC")
        return res.x[0]

    elif dist == "NegBin":
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

        res = minimize(objective, 0.5, bounds=[(0.05, 0.9)], tol=1e-8, method="TNC")
        return res.x[0]
    else:
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

        res = minimize(objective, 0.5, bounds=[(0.05, 0.9)], tol=1e-8, method="TNC")
        return res.x[0]


def select_distribution(player_stats):
    """Recommend a distribution family by inspecting per-player data properties.

    Returns (dist_name, p_zero) where dist_name is one of NegBin/ZINB/Gamma/ZAGamma
    and p_zero is the estimated excess zero-inflation rate.
    """
    import warnings

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
        dist = "NegBin" if resolution > 0.2 else "Gamma"
        print(f"  Resolution: {resolution:.4f} ({'NegBin' if resolution > 0.2 else 'Gamma'})")

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
        if p_zero > 0.1:
            dist = "ZINB"
    else:
        gam_fit = player_stats.apply(
            lambda x: fit(gamma, x[x > 0].astype(float), {"a": (0, 50), "scale": (0, 500)}).params
        )
        base_zero_prob = gam_fit.apply(lambda row: gamma.cdf(0.99, row[0], scale=row[2]))
        p_zero = float(((observed_zeros - base_zero_prob) / (1 - base_zero_prob)).clip(0, 1).mean())
        if p_zero > 0.05:
            dist = "ZAGamma"

    print(f"  Data type: {f'integer (step={int(step)})' if is_integer else 'continuous'}")
    print(f"  Zero inflation - {p_zero:.4f}")
    print(f"  Selected: {dist}")

    return dist, p_zero
