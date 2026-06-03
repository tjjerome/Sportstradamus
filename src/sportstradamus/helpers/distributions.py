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
   using either a log-opinion pool (NegBin) or a precision-weighted
   blend (Gamma, SkewNormal) with weight ``w`` on the model. See CLAUDE.md
   for the math and the diagnostic block that validates this.

``set_model_start_values`` prepares LightGBMLSS start values from the
training matrix's per-player historical moments; kept here because it
shares the distribution-family dispatch with the inversion code.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit
from scipy.stats import gamma, nbinom, norm, poisson, skewnorm

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


def no_vig_odds(over, under=None):
    """Return ``[p_over, p_under]`` with the bookmaker's hold removed.

    Accepts either American odds (``|x| >= 100``) or decimal odds. When
    ``under`` is omitted, the caller is treated as offering a one-sided
    line; we fabricate an under from a conservative 6.5% vig assumption
    so the two-sided math still works.
    """
    o = odds_to_prob(over) if np.abs(over) >= 100 else 1 / over
    if under is None or under <= 0:
        juice = 1.0652
        u = juice - o
    else:
        u = odds_to_prob(under) if np.abs(under) >= 100 else 1 / under

        juice = o + u

    return [o / juice, u / juice]


# Cap the SkewNormal implied mean at this multiple of the line — the archive's
# own blown-row threshold (``BLOWN_LINE_FACTOR``). Below the corresponding
# under-prob the inversion would exceed it, so the line is used instead.
SN_MAX_MEAN_FACTOR = 5.0


def _skewnormal_ev(line, under, cv, skew_alpha):
    """Solve for the SkewNormal mean reproducing the book's ``under`` at ``line``.

    The scale grows with the mean, so ``cdf(line, mean)`` asymptotes to
    ``Phi(-1/cv)`` as ``mean → ∞`` rather than to 0. An under-prob near that floor
    inverts to a mean many times the line — and can hand ``brentq`` a same-sign
    bracket — so an under that would imply a mean above ``SN_MAX_MEAN_FACTOR ×
    line`` collapses to the neutral line. The cutoff ``Phi((1-F)/(F·cv))`` is the
    under-prob at which ``cdf(line, F·line) = under``.
    """
    line = float(line)
    a = float(skew_alpha) if skew_alpha is not None else 0.0
    f = SN_MAX_MEAN_FACTOR
    if a == 0.0 and under <= norm.cdf((1 - f) / (f * cv)):
        return line

    def _sn_residual(mean):
        sigma = mean * cv
        delta = a / np.sqrt(1 + a**2)
        loc_sn = mean - sigma * delta * np.sqrt(2 / np.pi)
        try:
            return float(skewnorm.cdf(line, a, loc=loc_sn, scale=sigma)) - under
        except (ValueError, RuntimeWarning):
            return np.nan

    lo = 1e-6
    hi = max(2 * line / max(1 - under, 0.01), 1.0)
    while hi < 1e8 and _sn_residual(hi) > 0:
        hi = min(hi * 2, 1e8)
    # brentq needs endpoints straddling zero; if they don't (``a != 0`` asymptote
    # or a numerical edge) the bracket is degenerate. nan-safe chained compare.
    if not (_sn_residual(lo) > 0 > _sn_residual(hi)):
        return line
    return float(brentq(_sn_residual, lo, hi, xtol=1e-8))


def get_ev(line, under, cv=1, dist="SkewNormal", gate=None, skew_alpha=None):
    """Invert the book's (line, under-prob) to recover the implied mean.

    For zero-inflated distributions (ZINB/ZAGamma) with ``gate`` supplied,
    the book's CDF is decomposed as ``gate + (1-gate) * base_CDF`` and the
    function solves for the *base* distribution mean so ``fused_loc``
    receives comparable parameters on both sides.

    Args:
        line: The bookmaker's line.
        under: The bookmaker's implied probability that the outcome is
            under the line.
        cv: Coefficient of variation. Used to derive shape: ``1/cv`` for
            NegBin, ``1/cv²`` for Gamma.
        dist: Distribution family — one of ``"Gamma"``, ``"ZAGamma"``,
            ``"NegBin"``, ``"ZINB"``, ``"Poisson"``, ``"SkewNormal"``.
        gate: Historical zero-inflation probability. ``None`` means no ZI.
        skew_alpha: Skewness parameter for SkewNormal; ``None`` → 0 (symmetric).

    Returns:
        The base-distribution mean that reproduces the book's ``under``.
    """
    under = np.clip(under, 1e-6, 1 - 1e-6)

    # For ZI distributions, strip out the zero-inflation component so we solve
    # for the base distribution mean: gate + (1-gate)*base_CDF = under
    # ⇒ base_CDF = (under - gate) / (1 - gate)
    if gate is not None and gate > 0 and dist in ("ZINB", "ZAGamma", "SkewNormal"):
        base_cdf = (under - gate) / (1 - gate)
        # An under-prob at or below the zero-inflation mass leaves no room for the
        # base distribution below the line, so the inversion runs away to a huge
        # mean (the book and the model's zero rate disagree). Use the neutral line.
        if base_cdf <= 0:
            return float(line)
        under = np.clip(base_cdf, 1e-6, 1 - 1e-6)

    # CDF is monotonically decreasing in mean (1→0), so a bracket is always valid:
    #   at lo≈0 the CDF≈1 > under, at hi→∞ the CDF→0 < under.
    lo = 1e-6
    hi = max(2 * line / max(1 - under, 0.01), 1.0)

    if dist in ("NegBin", "ZINB", "Poisson"):
        line = np.ceil(float(line) - 1)
        if cv == 1:

            def _pois_residual(mean):
                return float(poisson.cdf(line, mean)) - under

            while _pois_residual(hi) > 0:
                hi *= 2
            return float(brentq(_pois_residual, lo, hi, xtol=1e-8))
        r = 1.0 / cv

        def _nb_residual(mean):
            p = r / (r + mean)
            return float(nbinom.cdf(line, r, p)) - under

        while _nb_residual(hi) > 0:
            hi *= 2
        return float(brentq(_nb_residual, lo, hi, xtol=1e-8))

    if dist == "SkewNormal":
        return _skewnormal_ev(line, under, cv, skew_alpha)

    # Gamma / ZAGamma (default for all continuous distributions)
    line = float(line)
    alpha = 1.0 / (cv**2)

    def _gamma_residual(mean):
        return float(gamma.cdf(line, alpha, scale=mean / alpha)) - under

    while _gamma_residual(hi) > 0:
        hi *= 2
    return float(brentq(_gamma_residual, lo, hi, xtol=1e-8))


def get_odds(
    line, ev, dist, cv=1, alpha=None, r=None, gate=None, step=1, sigma=None, skew_alpha=None
):
    """Return the raw probability that the outcome falls below ``line``.

    Inverse of ``get_ev``. Temperature scaling is applied elsewhere, at
    the over/under decision layer, not here.

    Args:
        line: The line / cutoff value.
        ev: Expected value (base-distribution mean).
        dist: Distribution family (same options as ``get_ev``).
        cv: Coefficient of variation, used when ``alpha``/``r`` are not supplied.
        alpha: Gamma shape; derived as ``1/cv²`` if ``None``.
        r: NegBin dispersion; derived as ``1/cv`` if ``None``.
        gate: Zero-inflation probability; ``None`` disables ZI handling.
        step: Bin width for the discrete half-point correction.
        sigma: SkewNormal scale; derived as ``ev*cv`` if ``None``.
        skew_alpha: SkewNormal skewness; defaults to ``0``.

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

    if dist == "SkewNormal":
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

    if alpha is None:
        alpha = 1 / cv**2

    # Gamma / ZAGamma CDF
    cdf_high = gamma.cdf(high, alpha, scale=ev / alpha)
    cdf_low = gamma.cdf(low, alpha, scale=ev / alpha)
    if gate is not None and dist == "ZAGamma":
        # ZA-CDF: gate + (1 - gate) * base_CDF
        cdf_high = gate + (1 - gate) * cdf_high
        cdf_low = gate + (1 - gate) * cdf_low
    push = cdf_high - cdf_low
    return cdf_high - push / 2


def get_push_prob(line, ev, dist, cv=1, alpha=None, r=None, gate=None, sigma=None, skew_alpha=None):
    """Return P(stat == line) for a market.

    Push probability is non-zero only when ``line`` is an integer **and** the
    distribution family is discrete (NegBin / ZINB / Poisson). Continuous
    families (Gamma, ZAGamma, SkewNormal, Normal) return 0 because the point
    mass at any single value is zero. Used by parlay scoring to handle the
    Underdog "push drops one leg" rule.

    Args:
        line: The bookmaker line. Push only fires for integer-valued lines.
        ev: Base-distribution mean (post fused_loc / dispersion calibration).
        dist: Distribution family ("NegBin", "ZINB", "Poisson", "Gamma",
            "ZAGamma", "SkewNormal", "Normal"). Anything not recognized as
            discrete returns 0.
        cv: CV used as fallback when ``r`` is not supplied.
        alpha: Unused; included so the call signature parallels ``get_odds``.
        r: NegBin dispersion. Falls back to ``1/cv``.
        gate: Zero-inflation gate; ZI shrinks the base pmf by ``(1 - gate)``.
            For an integer line at 0, the gate contributes its full mass.
        sigma: Unused; parallels ``get_odds``.
        skew_alpha: Unused; parallels ``get_odds``.

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
    NegBin/ZINB, ``alpha`` for Gamma/ZAGamma, ``sigma``/``skew`` for SkewNormal.
    ``gate`` is the per-row zero-inflation gate (ZINB/ZAGamma) or the broadcast
    historical gate (SkewNormal), and is ``None`` when the cell is not gated.
    """

    ev: np.ndarray
    r: np.ndarray | None = None
    alpha: np.ndarray | None = None
    sigma: np.ndarray | None = None
    skew: np.ndarray | None = None
    gate: np.ndarray | None = None


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
            ``total_count``/``probs``/``gate`` (NegBin/ZINB),
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
):
    """Blend model and bookmaker distribution parameters with weight ``w``.

    The blend is a logarithmic opinion pool (Genest & Zidek 1986) for
    NegBin and a precision-weighted blend for Gamma / SkewNormal:

    * **NegBin**: geometric mean of both means *and* dispersion parameters.
      The model provides per-observation ``r``; the book's ``r`` is derived
      as ``1/cv``. Both ``μ`` and ``r`` are blended in log space with the
      same weight ``w``.
    * **Gamma**: precision-weighted blend. The model provides
      per-observation ``alpha``; the book's ``alpha`` is ``1/cv²``.
      Returns ``(alpha, beta, gate_blend)``.
    * **SkewNormal**: precision-weighted blend of ``loc`` / ``sigma``,
      linear blend of ``alpha``. Book side uses ``alpha=0`` (symmetric
      Normal). Returns ``(ev, sigma, alpha, gate_blend)``.

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

    Returns:
        NegBin → ``(r_blend, p, gate_blend)``,
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

        # Book side: symmetric normal (alpha=0), sigma = ev * cv.
        book_sigma = np.clip(ev_b * cv, 1e-6, None)

        # Derive loc from EV: loc = EV - sigma * delta * sqrt(2/pi).
        model_delta = model_skew / np.sqrt(1 + model_skew**2)
        model_loc = ev_a - model_sigma * model_delta * np.sqrt(2 / np.pi)
        book_loc = ev_b  # alpha=0 → delta=0 → loc = EV.

        # Precision-weighted blend.
        prec_m = 1.0 / model_sigma**2
        prec_b = 1.0 / book_sigma**2
        total_prec = w * prec_m + (1 - w) * prec_b
        blended_loc = (w * model_loc * prec_m + (1 - w) * book_loc * prec_b) / total_prec
        blended_sigma = 1.0 / np.sqrt(total_prec)
        blended_skew = w * model_skew  # book alpha=0, so blend reduces to w * model.

        # Compute blended EV from blended params.
        bl_delta = blended_skew / np.sqrt(1 + blended_skew**2)
        blended_ev = blended_loc + blended_sigma * bl_delta * np.sqrt(2 / np.pi)

        return blended_ev, blended_sigma, blended_skew, gate_blend

    # Gamma — precision-weighted blend.
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


def set_model_start_values(
    model, dist, X_data, shape_ceiling=None, normalized=False, offset_mode=False
):
    """Initialize LightGBMLSS start values from per-player historical moments.

    Values live in the model's raw (pre-response-function) space. Response
    functions per distribution:

    * NegBin / ZINB: ``total_count`` → ReLU, ``probs`` → sigmoid,
      ``gate`` → sigmoid.
    * Gamma / ZAGamma: ``concentration`` → softplus, ``rate`` → softplus,
      ``gate`` → sigmoid.
    * SkewNormal: ``loc`` → identity, ``scale`` → exp, ``alpha`` → identity.

    Args:
        model: The LightGBMLSS model whose ``start_values`` gets assigned.
        dist: Distribution name — ``"NegBin"``, ``"ZINB"``, ``"Gamma"``,
            ``"ZAGamma"``, or ``"SkewNormal"``.
        X_data: DataFrame; must contain ``"MeanYr"``, ``"STDYr"``, and
            ``"ZeroYr"`` columns.
        shape_ceiling: Upper bound on shape during training. When ``None``,
            a conservative default is used (50 for NegBin, 100 for Gamma).
        normalized: If ``True``, targets are already normalized to
            ``Result/MeanYr ≈ 1.0`` and start values are set for that space.
        offset_mode: SkewNormal-only. If ``True``, targets are an additive
            centered residual (``y - baseline``), so ``loc`` is seeded at
            zero per row and ``scale`` at per-player STDYr (residual
            dispersion ≈ per-player std). Mutually exclusive with
            ``normalized``; ignored for non-SkewNormal distributions.
    """
    from scipy.special import logit

    def _softplus_inv(x):
        x = np.asarray(x, dtype=float)
        return np.where(
            x > SOFTPLUS_LINEAR_THRESHOLD,
            x,
            np.log(np.expm1(np.clip(x, 1e-4, SOFTPLUS_LINEAR_THRESHOLD))),
        )

    sv = X_data[["MeanYr", "STDYr", "ZeroYr"]].to_numpy()
    n = len(sv)

    mu = np.clip(sv[:, 0], 1e-6, None)
    std = np.clip(sv[:, 1], 1e-6, None)
    hist_gate = np.clip(sv[:, 2], 0, 0.99)

    _r_upper = shape_ceiling if shape_ceiling is not None else 50
    _a_upper = shape_ceiling if shape_ceiling is not None else 100

    if dist == "SkewNormal":
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
            # Targets ≈ 1.0 for all players. Use global start values.
            cv_player = np.clip(std / mu, 0.01, 10)
            loc = np.ones(n)
            scale = cv_player  # scale ≈ CV since mean ≈ 1.0.
        else:
            loc = mu.copy()
            scale = std.copy()
        alpha_skew = np.zeros(n)  # Start symmetric.
        # loc: identity → raw = value.
        # scale: exp → raw = log(value).
        # alpha: identity → raw = value.
        sv = np.column_stack([loc, np.log(np.clip(scale, 1e-6, None)), alpha_skew])

    elif dist in ["NegBin", "ZINB"]:
        # r = mu² / (var - mu); ReLU response → raw = value (identity for r>0).
        r_init = np.clip(mu**2 / np.clip(std**2 - mu, 1e-6, None), 0.5, _r_upper)
        # PyTorch probs = mu / (mu + r); sigmoid response → raw = logit(probs).
        probs = np.clip(mu / (mu + r_init), 0.01, 0.99)
        if dist == "ZINB":
            nb_zeros = nbinom.pmf(0, r_init, probs)
            hist_gate = np.clip(hist_gate - nb_zeros, 0, 0.99)
            mu = mu / (1 - hist_gate)
            r_init = np.clip(mu**2 / np.clip(std**2 - mu, 1e-6, None), 0.5, _r_upper)
            probs = np.clip(mu / (mu + r_init), 0.01, 0.99)
        sv = np.column_stack([r_init, logit(probs)])

    elif dist in ["Gamma", "ZAGamma"]:
        if dist == "ZAGamma":
            mu = mu / (1 - hist_gate)
        alpha = np.clip((mu / std) ** 2, 0.1, _a_upper)
        beta = np.clip(alpha / np.clip(mu, 1e-6, None), 0.01, 50)
        # softplus response → raw = softplus_inv(value).
        sv = np.column_stack([_softplus_inv(alpha), _softplus_inv(beta)])

    if dist in ["ZINB", "ZAGamma"]:
        gate_val = np.clip(hist_gate, 0.01, 0.99)
        sv = np.column_stack([sv, np.full(n, logit(gate_val))])

    model.start_values = sv
