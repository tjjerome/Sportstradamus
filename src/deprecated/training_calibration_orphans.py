# ARCHIVED 2026-10-16 from src/sportstradamus/training/calibration.py
# Reason: select_distribution had no caller in src/sportstradamus/, tests/ or scripts/ (the live
#         family choice is training/pipeline.py's _step_select_distribution), so it is archived
#         with the three thresholds only it read.
# Last live SHA: 62a49e89
# Original imports (now unresolved here):
#   import warnings
#   import numpy as np
#   from scipy.stats import fit, gamma, nbinom
#   from sportstradamus.helpers import get_logger
#   logger = get_logger(__name__)

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
