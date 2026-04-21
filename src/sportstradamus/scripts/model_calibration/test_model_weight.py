"""
Diagnostic script to test different fit_model_weight objectives.
Loads saved test/validation data and evaluates multiple scoring rules
to understand why model_weight is being pushed to the lower bound.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import beta as beta_fn
from scipy.stats import gamma as gamma_dist
from scipy.stats import nbinom

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sportstradamus.helpers import fused_loc, get_odds

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "sportstradamus", "data")


def load_market(league, market):
    """Load test set CSV and extract model/book parameters."""
    import json

    path = os.path.join(DATA_DIR, "test_sets", f"{league}_{market}.csv")
    df = pd.read_csv(path, index_col=0)

    result = df["Result"].to_numpy().astype(float)
    line = df["Line"].to_numpy().astype(float)
    book_ev = df["Blended_EV"].to_numpy().astype(float)
    model_ev = df["EV"].to_numpy().astype(float)

    gate = df["Gate"].to_numpy().astype(float) if "Gate" in df.columns else None
    alpha = df["Alpha"].to_numpy().astype(float) if "Alpha" in df.columns else None
    # R and NB_P are the actual NegBin params; P is the calibrated probability (not a dist param)
    r = df["R"].to_numpy().astype(float) if "R" in df.columns else None
    nb_p = df["NB_P"].to_numpy().astype(float) if "NB_P" in df.columns else None

    with open(os.path.join(DATA_DIR, "stat_cv.json")) as f:
        stat_cv = json.load(f)
    cv = stat_cv[league][market]

    with open(os.path.join(DATA_DIR, "stat_dist.json")) as f:
        stat_dist = json.load(f)
    dist = stat_dist[league][market]

    with open(os.path.join(DATA_DIR, "stat_zi.json")) as f:
        stat_zi = json.load(f)
    hist_gate = stat_zi.get(league, {}).get(market, 0.0)

    return {
        "result": result,
        "line": line,
        "book_ev": book_ev,
        "model_ev": model_ev,
        "gate": gate,
        "alpha": alpha,
        "r": r,
        "nb_p": nb_p,
        "cv": cv,
        "dist": dist,
        "hist_gate": hist_gate,
    }


def fit_dispersion_on_raw_model(data):
    """Find optimal dispersion scaling c for the RAW model (before blending).
    Returns c_opt and a new data dict with corrected shape parameters."""
    result = data["result"]
    dist = data["dist"]
    model_ev = data["model_ev"]
    gate = data["gate"]

    if dist in ("NegBin", "ZINB"):
        r_raw = data["r"]
        # p from r and ev: p = r / (r + ev)
        r_raw / (r_raw + model_ev)

        def loss(c):
            r_cal = r_raw * c
            p_cal = r_cal / (r_cal + model_ev)  # mean-preserving
            result_int = result.astype(int)
            k_max = int(max(result_int.max() * 2, np.mean(model_ev) * 4, 30))
            k_vals = np.arange(k_max + 1)
            cdf = nbinom.cdf(k_vals[:, None], r_cal[None, :], p_cal[None, :])
            if gate is not None:
                cdf = gate[None, :] + (1 - gate[None, :]) * cdf
            indicator = (result_int[None, :] <= k_vals[:, None]).astype(float)
            crps = np.mean(np.sum((cdf - indicator) ** 2, axis=0))
            reg = 0.01 * np.log(c) ** 2
            return crps + reg

        res = minimize_scalar(loss, bounds=(0.1, 10.0), method="bounded")
        c_opt = res.x

        d = dict(data)
        d["r"] = r_raw * c_opt
        return c_opt, d

    else:  # Gamma / ZAGamma
        alpha_raw = data["alpha"]

        def loss(c):
            alpha_cal = alpha_raw * c
            scale_cal = model_ev / alpha_cal  # mean-preserving
            if gate is not None:
                x_max = max(result.max() * 2, np.mean(model_ev) * 4)
                x_grid = np.linspace(0, x_max, 500)
                dx = x_grid[1] - x_grid[0]
                cdf_grid = gamma_dist.cdf(
                    x_grid[:, None], alpha_cal[None, :], scale=scale_cal[None, :]
                )
                cdf_grid = gate[None, :] + (1 - gate[None, :]) * cdf_grid
                indicator = (result[None, :] <= x_grid[:, None]).astype(float)
                crps = np.mean(np.sum((cdf_grid - indicator) ** 2, axis=0) * dx)
            else:
                F_y = gamma_dist.cdf(result, alpha_cal, scale=scale_cal)
                F_y_a1 = gamma_dist.cdf(result, alpha_cal + 1, scale=scale_cal)
                mu = model_ev
                crps = np.mean(
                    result * (2 * F_y - 1)
                    - mu * (2 * F_y_a1 - 1)
                    - scale_cal / beta_fn(0.5, alpha_cal)
                )
            reg = 0.01 * np.log(c) ** 2
            return crps + reg

        res = minimize_scalar(loss, bounds=(0.1, 10.0), method="bounded")
        c_opt = res.x

        d = dict(data)
        d["alpha"] = alpha_raw * c_opt
        return c_opt, d


def eval_accuracy(w, data, use_dispersion_cal=False):
    """Compute accuracy at a given model weight."""
    dist = data["dist"]
    base_dist = "NegBin" if dist in ("NegBin", "ZINB") else "Gamma"
    zi_kwargs = {}
    if dist in ("ZINB", "ZAGamma") and data["hist_gate"] > 0:
        zi_kwargs = dict(gate_model=data["gate"], gate_book=data["hist_gate"])

    if base_dist == "NegBin":
        r_blend, p_blend, gate_blend = fused_loc(
            w, data["model_ev"], data["book_ev"], data["cv"], "NegBin", r=data["r"], **zi_kwargs
        )
        weighted_mean = r_blend * (1 - p_blend) / p_blend
        under = get_odds(data["line"], weighted_mean, dist, r=r_blend, gate=gate_blend)
    else:
        alpha_blend, beta_blend, gate_blend = fused_loc(
            w,
            data["model_ev"],
            data["book_ev"],
            data["cv"],
            "Gamma",
            alpha=data["alpha"],
            **zi_kwargs,
        )
        weighted_mean = alpha_blend / beta_blend
        under = get_odds(data["line"], weighted_mean, dist, alpha=alpha_blend, gate=gate_blend)

    under = np.clip(under, 0, 1)
    over = 1 - under
    y_class = (data["result"] >= data["line"]).astype(int)
    y_pred = (over > 0.5).astype(int)
    mask = np.maximum(over, under) > 0.54
    acc = np.mean(y_class[mask] == y_pred[mask]) if mask.sum() > 0 else np.nan
    return acc, mask.mean()


def objective_nll(w, data, clamp=-20):
    """Negative log-likelihood (clamped)."""
    dist = data["dist"]
    base_dist = "NegBin" if dist in ("NegBin", "ZINB") else "Gamma"
    zi_kwargs = {}
    has_gate = dist in ("ZINB", "ZAGamma") and data["hist_gate"] > 0
    if has_gate:
        zi_kwargs = dict(gate_model=data["gate"], gate_book=data["hist_gate"])

    if base_dist == "NegBin":
        r_blend, p_blend, g_blend = fused_loc(
            w, data["model_ev"], data["book_ev"], data["cv"], "NegBin", r=data["r"], **zi_kwargs
        )
        result_int = data["result"].astype(int)
        base_logpmf = np.clip(nbinom.logpmf(result_int, r_blend, p_blend), clamp, 0)
        if has_gate:
            loglik = np.where(
                result_int == 0,
                np.log(np.clip(g_blend + (1 - g_blend) * np.exp(base_logpmf), 1e-12, None)),
                np.log(np.clip(1 - g_blend, 1e-12, None)) + base_logpmf,
            )
            return -np.mean(loglik)
        return -np.mean(base_logpmf)
    else:
        alpha_bl, beta_bl, g_blend = fused_loc(
            w,
            data["model_ev"],
            data["book_ev"],
            data["cv"],
            "Gamma",
            alpha=data["alpha"],
            **zi_kwargs,
        )
        base_logpdf = np.clip(
            gamma_dist.logpdf(data["result"], alpha_bl, scale=1 / beta_bl), clamp, 0
        )
        if has_gate:
            loglik = np.where(
                data["result"] == 0,
                np.log(np.clip(g_blend, 1e-12, None)),
                np.log(np.clip(1 - g_blend, 1e-12, None)) + base_logpdf,
            )
            return -np.mean(loglik)
        return -np.mean(base_logpdf)


def objective_nll_unclamped(w, data):
    """Unclamped NLL for comparison."""
    return objective_nll(w, data, clamp=-1e10)


def objective_crps(w, data):
    """CRPS objective."""
    dist = data["dist"]
    base_dist = "NegBin" if dist in ("NegBin", "ZINB") else "Gamma"
    zi_kwargs = {}
    has_gate = dist in ("ZINB", "ZAGamma") and data["hist_gate"] > 0
    if has_gate:
        zi_kwargs = dict(gate_model=data["gate"], gate_book=data["hist_gate"])

    if base_dist == "NegBin":
        r_blend, p_blend, g_blend = fused_loc(
            w, data["model_ev"], data["book_ev"], data["cv"], "NegBin", r=data["r"], **zi_kwargs
        )
        result_int = data["result"].astype(int)
        mu_blend = r_blend * (1 - p_blend) / p_blend
        k_max = int(max(result_int.max() * 2, np.mean(mu_blend) * 4, 30))
        k_vals = np.arange(k_max + 1)
        cdf = nbinom.cdf(k_vals[:, None], r_blend[None, :], p_blend[None, :])
        if has_gate and g_blend is not None:
            cdf = g_blend[None, :] + (1 - g_blend[None, :]) * cdf
        indicator = (result_int[None, :] <= k_vals[:, None]).astype(float)
        return np.mean(np.sum((cdf - indicator) ** 2, axis=0))
    else:
        alpha_bl, beta_bl, g_blend = fused_loc(
            w,
            data["model_ev"],
            data["book_ev"],
            data["cv"],
            "Gamma",
            alpha=data["alpha"],
            **zi_kwargs,
        )
        scale_bl = 1 / beta_bl
        mu_bl = alpha_bl * scale_bl
        if has_gate and g_blend is not None:
            x_max = max(data["result"].max() * 2, np.mean(mu_bl) * 4)
            x_grid = np.linspace(0, x_max, 500)
            dx = x_grid[1] - x_grid[0]
            cdf_grid = gamma_dist.cdf(x_grid[:, None], alpha_bl[None, :], scale=scale_bl[None, :])
            cdf_grid = g_blend[None, :] + (1 - g_blend[None, :]) * cdf_grid
            indicator = (data["result"][None, :] <= x_grid[:, None]).astype(float)
            return np.mean(np.sum((cdf_grid - indicator) ** 2, axis=0) * dx)
        else:
            F_y = gamma_dist.cdf(data["result"], alpha_bl, scale=scale_bl)
            F_y_a1 = gamma_dist.cdf(data["result"], alpha_bl + 1, scale=scale_bl)
            crps_vals = (
                data["result"] * (2 * F_y - 1)
                - mu_bl * (2 * F_y_a1 - 1)
                - scale_bl / beta_fn(0.5, alpha_bl)
            )
            return np.mean(crps_vals)


def objective_brier(w, data):
    """Brier score on over/under classification."""
    dist = data["dist"]
    base_dist = "NegBin" if dist in ("NegBin", "ZINB") else "Gamma"
    zi_kwargs = {}
    has_gate = dist in ("ZINB", "ZAGamma") and data["hist_gate"] > 0
    if has_gate:
        zi_kwargs = dict(gate_model=data["gate"], gate_book=data["hist_gate"])

    if base_dist == "NegBin":
        r_blend, p_blend, gate_blend = fused_loc(
            w, data["model_ev"], data["book_ev"], data["cv"], "NegBin", r=data["r"], **zi_kwargs
        )
        weighted_mean = r_blend * (1 - p_blend) / p_blend
        under = get_odds(data["line"], weighted_mean, dist, r=r_blend, gate=gate_blend)
    else:
        alpha_blend, beta_blend, gate_blend = fused_loc(
            w,
            data["model_ev"],
            data["book_ev"],
            data["cv"],
            "Gamma",
            alpha=data["alpha"],
            **zi_kwargs,
        )
        weighted_mean = alpha_blend / beta_blend
        under = get_odds(data["line"], weighted_mean, dist, alpha=alpha_blend, gate=gate_blend)

    under = np.clip(under, 0, 1)
    over = 1 - under
    y_class = (data["result"] >= data["line"]).astype(float)
    return np.mean((over - y_class) ** 2)


def sweep_weights(data, objectives, w_range=np.linspace(0.05, 0.9, 50)):
    """Evaluate each objective across a range of weights."""
    results = {name: [] for name in objectives}
    accs = []
    coverages = []
    for w in w_range:
        for name, fn in objectives.items():
            results[name].append(fn(w, data))
        acc, cov = eval_accuracy(w, data)
        accs.append(acc)
        coverages.append(cov)
    return w_range, results, accs, coverages


def optimize_weight(data, objective_fn, name=""):
    """Find optimal weight for a given objective."""
    res = minimize(
        lambda w: objective_fn(w[0], data), [0.5], bounds=[(0.05, 0.9)], tol=1e-8, method="TNC"
    )
    return res.x[0]


def debias_data(data):
    """Create a copy of data with model EVs scaled to match book EV mean.
    Preserves the model's per-observation relative ordering while
    removing the global mean bias."""
    d = dict(data)
    scale = np.mean(data["book_ev"]) / np.mean(data["model_ev"])
    d["model_ev"] = data["model_ev"] * scale
    # For NegBin: r stays the same, only mean changes (adjusts p implicitly via fused_loc)
    # For Gamma: alpha stays the same, only mean changes (adjusts beta implicitly)
    return d, scale


def eval_accuracy_2w(w_mean, w_shape, data):
    """Accuracy with separate mean and shape weights.
    w_mean controls the blended mean, w_shape controls the blended shape."""
    dist = data["dist"]
    base_dist = "NegBin" if dist in ("NegBin", "ZINB") else "Gamma"
    has_gate = dist in ("ZINB", "ZAGamma") and data["hist_gate"] > 0
    if has_gate:
        dict(gate_model=data["gate"], gate_book=data["hist_gate"])

    if base_dist == "NegBin":
        # Blend mean with w_mean, shape with w_shape
        mu = np.exp(
            w_mean * np.log(np.clip(data["model_ev"], 1e-9, None))
            + (1 - w_mean) * np.log(np.clip(data["book_ev"], 1e-9, None))
        )
        r_blend = np.exp(w_shape * np.log(data["r"]) + (1 - w_shape) * np.log(1 / data["cv"]))
        r_blend / (r_blend + mu)
        gate_blend = None
        if has_gate:
            gate_blend = (
                w_mean * np.asarray(data["gate"], dtype=float) + (1 - w_mean) * data["hist_gate"]
            )
        under = get_odds(data["line"], mu, dist, r=r_blend, gate=gate_blend)
    else:
        # Gamma: blend mean with w_mean, alpha with w_shape
        model_alpha = np.asarray(data["alpha"], dtype=float)
        book_alpha = 1 / data["cv"] ** 2
        # Mean blend
        inv_var_m = model_alpha / np.asarray(data["model_ev"], dtype=float) ** 2
        inv_var_b = book_alpha / np.asarray(data["book_ev"], dtype=float) ** 2
        total_inv_var = w_mean * inv_var_m + (1 - w_mean) * inv_var_b
        blended_mean = (
            w_mean * data["model_ev"] * inv_var_m + (1 - w_mean) * data["book_ev"] * inv_var_b
        ) / total_inv_var
        # Shape blend (separate weight)
        blended_alpha = w_shape * model_alpha + (1 - w_shape) * book_alpha
        blended_alpha / blended_mean
        gate_blend = None
        if has_gate:
            gate_blend = (
                w_mean * np.asarray(data["gate"], dtype=float) + (1 - w_mean) * data["hist_gate"]
            )
        under = get_odds(data["line"], blended_mean, dist, alpha=blended_alpha, gate=gate_blend)

    under = np.clip(under, 0, 1)
    over = 1 - under
    y_class = (data["result"] >= data["line"]).astype(int)
    y_pred = (over > 0.5).astype(int)
    mask = np.maximum(over, under) > 0.54
    acc = np.mean(y_class[mask] == y_pred[mask]) if mask.sum() > 0 else np.nan
    return acc, mask.mean()


def objective_brier_2w(params, data):
    """Brier score with separate mean and shape weights."""
    w_mean, w_shape = params
    dist = data["dist"]
    base_dist = "NegBin" if dist in ("NegBin", "ZINB") else "Gamma"
    has_gate = dist in ("ZINB", "ZAGamma") and data["hist_gate"] > 0

    if base_dist == "NegBin":
        mu = np.exp(
            w_mean * np.log(np.clip(data["model_ev"], 1e-9, None))
            + (1 - w_mean) * np.log(np.clip(data["book_ev"], 1e-9, None))
        )
        r_blend = np.exp(w_shape * np.log(data["r"]) + (1 - w_shape) * np.log(1 / data["cv"]))
        r_blend / (r_blend + mu)
        gate_blend = None
        if has_gate:
            gate_blend = (
                w_mean * np.asarray(data["gate"], dtype=float) + (1 - w_mean) * data["hist_gate"]
            )
        under = get_odds(data["line"], mu, dist, r=r_blend, gate=gate_blend)
    else:
        model_alpha = np.asarray(data["alpha"], dtype=float)
        book_alpha = 1 / data["cv"] ** 2
        inv_var_m = model_alpha / np.asarray(data["model_ev"], dtype=float) ** 2
        inv_var_b = book_alpha / np.asarray(data["book_ev"], dtype=float) ** 2
        total_inv_var = w_mean * inv_var_m + (1 - w_mean) * inv_var_b
        blended_mean = (
            w_mean * data["model_ev"] * inv_var_m + (1 - w_mean) * data["book_ev"] * inv_var_b
        ) / total_inv_var
        blended_alpha = w_shape * model_alpha + (1 - w_shape) * book_alpha
        blended_alpha / blended_mean
        gate_blend = None
        if has_gate:
            gate_blend = (
                w_mean * np.asarray(data["gate"], dtype=float) + (1 - w_mean) * data["hist_gate"]
            )
        under = get_odds(data["line"], blended_mean, dist, alpha=blended_alpha, gate=gate_blend)

    under = np.clip(under, 0, 1)
    over = 1 - under
    y_class = (data["result"] >= data["line"]).astype(float)
    return np.mean((over - y_class) ** 2)


def objective_nll_2w(params, data, clamp=-20):
    """Clamped NLL with separate mean and shape weights."""
    w_mean, w_shape = params
    dist = data["dist"]
    base_dist = "NegBin" if dist in ("NegBin", "ZINB") else "Gamma"
    has_gate = dist in ("ZINB", "ZAGamma") and data["hist_gate"] > 0

    if base_dist == "NegBin":
        mu = np.exp(
            w_mean * np.log(np.clip(data["model_ev"], 1e-9, None))
            + (1 - w_mean) * np.log(np.clip(data["book_ev"], 1e-9, None))
        )
        r_blend = np.exp(w_shape * np.log(data["r"]) + (1 - w_shape) * np.log(1 / data["cv"]))
        p_blend = r_blend / (r_blend + mu)
        result_int = data["result"].astype(int)
        base_logpmf = np.clip(nbinom.logpmf(result_int, r_blend, p_blend), clamp, 0)
        if has_gate:
            gate_blend = (
                w_mean * np.asarray(data["gate"], dtype=float) + (1 - w_mean) * data["hist_gate"]
            )
            loglik = np.where(
                result_int == 0,
                np.log(np.clip(gate_blend + (1 - gate_blend) * np.exp(base_logpmf), 1e-12, None)),
                np.log(np.clip(1 - gate_blend, 1e-12, None)) + base_logpmf,
            )
            return -np.mean(loglik)
        return -np.mean(base_logpmf)
    else:
        model_alpha = np.asarray(data["alpha"], dtype=float)
        book_alpha = 1 / data["cv"] ** 2
        inv_var_m = model_alpha / np.asarray(data["model_ev"], dtype=float) ** 2
        inv_var_b = book_alpha / np.asarray(data["book_ev"], dtype=float) ** 2
        total_inv_var = w_mean * inv_var_m + (1 - w_mean) * inv_var_b
        blended_mean = (
            w_mean * data["model_ev"] * inv_var_m + (1 - w_mean) * data["book_ev"] * inv_var_b
        ) / total_inv_var
        blended_alpha = w_shape * model_alpha + (1 - w_shape) * book_alpha
        blended_beta = blended_alpha / blended_mean
        base_logpdf = np.clip(
            gamma_dist.logpdf(data["result"], blended_alpha, scale=1 / blended_beta), clamp, 0
        )
        if has_gate:
            gate_blend = (
                w_mean * np.asarray(data["gate"], dtype=float) + (1 - w_mean) * data["hist_gate"]
            )
            loglik = np.where(
                data["result"] == 0,
                np.log(np.clip(gate_blend, 1e-12, None)),
                np.log(np.clip(1 - gate_blend, 1e-12, None)) + base_logpdf,
            )
            return -np.mean(loglik)
        return -np.mean(base_logpdf)


def main():
    markets = [
        ("NBA", "PTS"),  # ZAGamma, w=0.05, big degradation
        ("NBA", "MIN"),  # Gamma, w=0.05
        ("NBA", "FGM"),  # NegBin, w=0.149, big degradation
        ("NBA", "REB"),  # NegBin, w=0.061
        ("NBA", "FTM"),  # NegBin, w=0.90, good market
        ("NBA", "STL"),  # NegBin, w=0.588, good market
    ]

    objectives = {
        "NLL_clamped": lambda w, d: objective_nll(w, d, clamp=-20),
        "CRPS": objective_crps,
        "Brier": objective_brier,
    }

    for league, market in markets:
        print(f"\n{'='*70}")
        print(f" {league} {market}")
        print(f"{'='*70}")

        try:
            data = load_market(league, market)
        except Exception as e:
            print(f"  Error loading: {e}")
            continue

        print(f"  dist={data['dist']}, cv={data['cv']:.4f}, hist_gate={data['hist_gate']:.4f}")
        print(f"  n={len(data['result'])}, mean_result={data['result'].mean():.2f}")
        print(
            f"  mean_model_ev={data['model_ev'].mean():.2f}, mean_book_ev={data['book_ev'].mean():.2f}"
        )
        if data["alpha"] is not None:
            print(
                f"  mean_model_alpha={data['alpha'].mean():.2f}, book_alpha={1/data['cv']**2:.2f}"
            )
        if data["r"] is not None:
            print(f"  mean_model_r={data['r'].mean():.2f}, book_r={1/data['cv']:.2f}")

        # --- CURRENT PIPELINE: blend first, then dispersion cal ---
        print("\n  === CURRENT: blend → dispersion_cal ===")
        print(f"  {'Objective':<20} {'Opt w':>8}  {'Acc@opt':>8}  {'Cov@opt':>8}")
        print(f"  {'-'*50}")
        for name, fn in objectives.items():
            try:
                w_opt = optimize_weight(data, fn, name)
                acc, cov = eval_accuracy(w_opt, data)
                print(f"  {name:<20} {w_opt:>8.3f}  {acc:>8.4f}  {cov:>8.3f}")
            except Exception as e:
                print(f"  {name:<20} ERROR: {e}")

        # --- NEW PIPELINE: dispersion cal on raw model first, then blend ---
        try:
            c_opt, data_dc = fit_dispersion_on_raw_model(data)
            print(f"\n  === NEW: dispersion_cal(c={c_opt:.3f}) → blend ===")
            if data_dc["alpha"] is not None:
                print(
                    f"  corrected_alpha={data_dc['alpha'].mean():.2f} (was {data['alpha'].mean():.2f})"
                )
            if data_dc["r"] is not None:
                print(f"  corrected_r={data_dc['r'].mean():.2f} (was {data['r'].mean():.2f})")
            print(f"  {'Objective':<20} {'Opt w':>8}  {'Acc@opt':>8}  {'Cov@opt':>8}")
            print(f"  {'-'*50}")
            for name, fn in objectives.items():
                try:
                    w_opt = optimize_weight(data_dc, fn, name)
                    acc, cov = eval_accuracy(w_opt, data_dc)
                    print(f"  {name:<20} {w_opt:>8.3f}  {acc:>8.4f}  {cov:>8.3f}")
                except Exception as e:
                    print(f"  {name:<20} ERROR: {e}")

            # Accuracy sweep with dispersion-corrected data
            print("\n  Accuracy at specific weights (dispersion-first):")
            print(f"  {'w':>6}  {'Acc':>8}  {'Coverage':>8}")
            for w in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
                acc, cov = eval_accuracy(w, data_dc)
                print(f"  {w:>6.2f}  {acc:>8.4f}  {cov:>8.3f}")
        except Exception as e:
            print(f"\n  Dispersion-first ERROR: {e}")

        # --- Reference: accuracy sweep with original data ---
        print("\n  Accuracy at specific weights (original):")
        print(f"  {'w':>6}  {'Acc':>8}  {'Coverage':>8}")
        for w in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
            acc, cov = eval_accuracy(w, data)
            print(f"  {w:>6.2f}  {acc:>8.4f}  {cov:>8.3f}")


if __name__ == "__main__":
    main()
