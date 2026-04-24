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

import warnings

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.stats import gamma, nbinom, norm, poisson, skewnorm


def odds_to_prob(odds):
    """Convert American odds to an implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        odds = -odds
        return odds / (odds + 100)


def prob_to_odds(p):
    """Convert a probability to American odds (integer-rounded)."""
    if p < 0.5:
        return int(np.round((1 - p) / p * 100))
    else:
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


def get_ev(line, under, cv=1, dist="Gamma", gate=None, skew_alpha=None):
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

    under = np.clip(under, 1e-6, 1 - 1e-6)

    # For ZI distributions, strip out the zero-inflation component so we solve
    # for the base distribution mean: gate + (1-gate)*base_CDF = under
    # ⇒ base_CDF = (under - gate) / (1 - gate)
    if gate is not None and gate > 0 and dist in ("ZINB", "ZAGamma", "SkewNormal"):
        under = np.clip((under - gate) / (1 - gate), 1e-6, 1 - 1e-6)

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

    elif dist == "SkewNormal":
        line = float(line)
        a = float(skew_alpha) if skew_alpha is not None else 0.0

        def _sn_residual(mean):
            sigma = mean * cv
            delta = a / np.sqrt(1 + a**2)
            loc_sn = mean - sigma * delta * np.sqrt(2 / np.pi)
            try:
                return float(skewnorm.cdf(line, a, loc=loc_sn, scale=sigma)) - under
            except (ValueError, RuntimeWarning):
                return np.nan

        # Expand hi until residual becomes negative, with safety cap.
        max_hi = 1e8
        while hi < max_hi and _sn_residual(hi) > 0:
            hi = min(hi * 2, max_hi)
        # If we hit the cap and residual is still positive, the bracket is
        # degenerate — fall back to returning the line itself as the mean.
        if _sn_residual(hi) > 0 or np.isnan(_sn_residual(hi)):
            return float(line)
        return float(brentq(_sn_residual, lo, hi, xtol=1e-8))

    else:
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

    elif dist in ("NegBin", "ZINB"):
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

    elif dist == "SkewNormal":
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

    else:
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
                100 * max((norm.cdf(lower_bound, w * m, v * s) - lower_tol), 0)
                + max((norm.sf(upper_bound, w * m, v * s) - upper_tol), 0)
                + np.power(1 - v, 2)
            )
        else:
            return (
                100 * max((poisson.cdf(lower_bound, w * m) - lower_tol), 0)
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

    elif dist == "SkewNormal":
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

    else:  # Gamma — precision-weighted blend.
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


def set_model_start_values(model, dist, X_data, shape_ceiling=None, normalized=False):
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
    """
    from scipy.special import logit

    def _softplus_inv(x):
        x = np.asarray(x, dtype=float)
        return np.where(x > 20, x, np.log(np.expm1(np.clip(x, 1e-4, 20))))

    sv = X_data[["MeanYr", "STDYr", "ZeroYr"]].to_numpy()
    n = len(sv)

    mu = np.clip(sv[:, 0], 1e-6, None)
    std = np.clip(sv[:, 1], 1e-6, None)
    hist_gate = np.clip(sv[:, 2], 0, 0.99)

    _r_upper = shape_ceiling if shape_ceiling is not None else 50
    _a_upper = shape_ceiling if shape_ceiling is not None else 100

    if dist == "SkewNormal":
        if normalized:
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
