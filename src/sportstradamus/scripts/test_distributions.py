"""
Experiment: Compare distribution families and loss functions for LightGBMLSS.

Tests:
1. LogNormal vs SkewNormal vs current (Gamma/NegBin) on normalized and raw targets
2. NLL vs CRPS loss functions
3. Low-mean markets (NBA BLK, NFL tds) vs high-mean markets (NBA PTS, NFL passing-yards)

Preprocessing:
- For continuous-base distributions (Gamma/ZAGamma): remove ALL zeros (they come from ZA gate)
- For count-base distributions (NegBin): remove stat_zi fraction of zeros (structural only)
- All experiments per market use the same preprocessed dataset

Metrics:
- NLL on test set (absolute space)
- ev_meanyr_corr vs result_meanyr_corr (compression diagnostic)
- Accuracy / Over% at >0.54 confidence
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy.stats import gamma as gamma_dist, lognorm, nbinom
from scipy.special import logit
import lightgbm as lgb
from lightgbmlss.model import LightGBMLSS
from lightgbmlss.distributions.Gamma import Gamma
from lightgbmlss.distributions.LogNormal import LogNormal
from lightgbmlss.distributions.ZAGamma import ZAGamma
from lightgbmlss.distributions.NegativeBinomial import NegativeBinomial
from lightgbmlss.distributions.ZALN import ZALN
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sportstradamus.skew_normal import SkewNormal
from sportstradamus.helpers import set_model_start_values

warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "training_data"

# ── Market configurations ───────────────────────────────────────────────
# continuous=True  → base dist is continuous (Gamma), ALL zeros are structural
# continuous=False → base dist is count (NegBin), zeros can be natural

MARKETS = {
    # NBA
    "NBA_PTS":      {"file": "NBA_PTS.csv",            "dist": "ZAGamma", "zi": 0.065, "continuous": True},
    "NBA_PRA":      {"file": "NBA_PRA.csv",            "dist": "Gamma",   "zi": 0.026, "continuous": True},
    "NBA_BLK":      {"file": "NBA_BLK.csv",            "dist": "NegBin",  "zi": 0.007, "continuous": False},
    "NBA_AST":      {"file": "NBA_AST.csv",            "dist": "NegBin",  "zi": 0.013, "continuous": False},
    # NFL
    "NFL_pass_yds": {"file": "NFL_passing-yards.csv",  "dist": "ZAGamma", "zi": 0.066, "continuous": True},
    "NFL_fantasy-points-underdog":  {"file": "NFL_fantasy-points-underdog.csv",        "dist": "Gamma",  "zi": 0.0289,   "continuous": True},
    "NFL_receptions":  {"file": "NFL_receptions.csv",     "dist": "NegBin",  "zi": 0.0098,   "continuous": False},
    "NFL_tds":      {"file": "NFL_tds.csv",            "dist": "NegBin",  "zi": 0.006, "continuous": False},
}

EPSILON = 0.01  # Clip value for zeros in continuous distributions


# ── Helpers ─────────────────────────────────────────────────────────────

def preprocess_zi(y, zi_gate, is_continuous_base, rng=None):
    """Remove structural zeros from data.

    For continuous-base distributions (Gamma/ZAGamma):
        ALL zeros come from the ZA gate → remove all zeros.
    For count-base distributions (NegBin):
        Some zeros are structural (gate), some natural → remove zi_gate fraction.

    Returns boolean mask of observations to keep.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    keep = np.ones(len(y), dtype=bool)
    zero_mask = y == 0

    if is_continuous_base:
        # All zeros are structural in continuous distributions
        keep[zero_mask] = False
    elif zi_gate > 0:
        # Remove zi_gate fraction of zeros (structural portion)
        zero_indices = np.where(zero_mask)[0]
        n_remove = int(len(zero_indices) * zi_gate)
        if n_remove > 0:
            remove_indices = rng.choice(zero_indices, size=n_remove, replace=False)
            keep[remove_indices] = False

    return keep


def load_market(market_file, zi_gate, is_continuous_base):
    """Load training CSV, apply ZI preprocessing, split data."""
    df = pd.read_csv(DATA_DIR / market_file, index_col=0)

    meta_cols = ["Line", "Odds", "EV", "Result", "Player", "Date", "Archived"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    # Apply ZI preprocessing before splitting
    y_all = df["Result"].values.copy()
    keep = preprocess_zi(y_all, zi_gate, is_continuous_base)
    df = df[keep].reset_index(drop=True)

    X = df[feature_cols].copy()
    y = df["Result"].values.copy()
    meta = df[meta_cols].copy()

    # Temporal split: 70% train, 15% val, 15% test
    n = len(df)
    n_train = int(n * 0.7)
    X_train, y_train = X.iloc[:n_train].copy(), y[:n_train]
    X_rem, y_rem = X.iloc[n_train:].copy(), y[n_train:]
    meta_rem = meta.iloc[n_train:]

    X_val, X_test, y_val, y_test, meta_val, meta_test = train_test_split(
        X_rem, y_rem, meta_rem, test_size=0.5, random_state=25
    )

    categories = ["Home", "Player position", "Player team"]
    for c in categories:
        if c in X_train.columns:
            for df_ in [X_train, X_val, X_test]:
                df_[c] = df_[c].astype("category")

    # Ensure numpy-native dtypes (not nullable Int64/Float64) for scipy compat
    for df_ in [X_train, X_val, X_test]:
        for c in df_.columns:
            if hasattr(df_[c].dtype, 'numpy_dtype'):
                df_[c] = df_[c].astype(df_[c].dtype.numpy_dtype)

    return X_train, X_val, X_test, y_train, y_val, y_test, meta_val, meta_test


def make_global_start_values(dist_label, y_train_norm):
    """Create global (non-per-observation) start value setter for normalized targets.

    For normalized data (y = Result / MeanYr ≈ 1.0), the mean is the same for all
    observations, so per-obs start values are unnecessary. Use a single global value.
    """
    pos = y_train_norm[y_train_norm > EPSILON]
    mu = np.mean(pos) if len(pos) > 0 else 1.0
    sigma_emp = np.std(pos) if len(pos) > 0 else 0.5
    cv = np.clip(sigma_emp / mu, 0.01, 10)

    if dist_label in ("LogNormal", "ZALN"):
        sigma = np.sqrt(np.log1p(cv ** 2))
        loc = np.log(mu) - sigma ** 2 / 2
        sv = np.array([[loc, np.log(np.clip(sigma, 1e-6, None))]])
        if dist_label == "ZALN":
            gate_frac = np.mean(y_train_norm <= 0)
            gate_frac = np.clip(gate_frac, 0.01, 0.99)
            sv = np.hstack([sv, [[logit(gate_frac)]]])
    elif dist_label == "SkewNormal":
        sv = np.array([[mu, np.log(np.clip(sigma_emp, 1e-6, None)), 0.0]])
    else:
        raise ValueError(f"No global start values for {dist_label}")

    def _set(model, X_data):
        model.start_values = sv

    return _set


def compute_metrics(ev, y_true, lines, sigma_or_shape, dist_type, gate=None):
    """Compute over/under accuracy metrics in absolute space."""
    results = {}
    results["model_ev_mean"] = float(np.mean(ev))
    results["result_mean"] = float(np.mean(y_true))

    # Compute P(under line)
    if dist_type in ("LogNormal", "ZALN"):
        sigma = sigma_or_shape
        mu_param = np.log(np.clip(ev, 1e-9, None)) - sigma ** 2 / 2
        under_probs = lognorm.cdf(lines, s=sigma, scale=np.exp(mu_param))
        if gate is not None and dist_type == "ZALN":
            under_probs = gate + (1 - gate) * under_probs
    elif dist_type in ("Gamma", "ZAGamma"):
        alpha = sigma_or_shape
        under_probs = gamma_dist.cdf(lines, alpha, scale=ev / alpha)
        if gate is not None and dist_type == "ZAGamma":
            under_probs = gate + (1 - gate) * under_probs
    elif dist_type in ("NegBin",):
        r = sigma_or_shape
        p = r / (r + np.clip(ev, 1e-9, None))
        under_probs = nbinom.cdf(lines, r, p)
    elif dist_type == "SkewNormal":
        scale, alpha_skew = sigma_or_shape
        from scipy.stats import skewnorm
        delta = alpha_skew / np.sqrt(1 + alpha_skew ** 2)
        loc_sn = ev - scale * delta * np.sqrt(2 / np.pi)
        under_probs = skewnorm.cdf(lines, alpha_skew, loc=loc_sn, scale=scale)
    else:
        raise ValueError(f"Unknown dist_type: {dist_type}")

    over_probs = 1 - under_probs
    pred_over = (over_probs > 0.5).astype(int)
    actual_over = (y_true >= lines).astype(int)
    conf_mask = np.maximum(under_probs, over_probs) > 0.54

    if conf_mask.sum() > 10:
        results["accuracy"] = float(np.mean(pred_over[conf_mask] == actual_over[conf_mask]))
        results["over_pct"] = float(np.mean(pred_over[conf_mask]))
    else:
        results["accuracy"] = float("nan")
        results["over_pct"] = float("nan")

    results["n_confident"] = int(conf_mask.sum())
    results["n_total"] = int(len(y_true))

    return results


# ── Main experiment runner ──────────────────────────────────────────────

def run_experiment(name, X_train, X_val, X_test, y_train, y_val, y_test,
                   meta_test, dist_obj, dist_label, loss_fn,
                   normalize=False, start_values_fn=None):
    """Run a single experiment configuration."""
    y_labels = y_train.copy()

    # Normalize targets if requested
    if normalize:
        meanyr_train = np.clip(X_train["MeanYr"].values, 0.5, None)
        y_labels = y_labels / meanyr_train

    # For continuous distributions, clip remaining zeros to epsilon
    if dist_label in ("LogNormal", "Gamma", "SkewNormal"):
        y_labels = np.clip(y_labels, EPSILON, None)

    X_tr = X_train.copy()

    model = LightGBMLSS(dist_obj)

    if start_values_fn is not None:
        start_values_fn(model, X_tr)

    dtrain = lgb.Dataset(X_tr, label=y_labels)

    # Hyperparameter search (reduced scope for speed)
    hp = {
        "feature_pre_filter": ["none", [False]],
        "num_threads": ["none", [8]],
        "max_depth": ["none", [-1]],
        "max_bin": ["none", [127]],
        "num_leaves": ["int", {"low": 8, "high": 63, "log": False}],
        "lambda_l1": ["float", {"low": 1e-6, "high": 10, "log": True}],
        "lambda_l2": ["float", {"low": 1e-6, "high": 10, "log": True}],
        "min_child_samples": ["int", {"low": 30, "high": 150, "log": False}],
        "learning_rate": ["float", {"low": 0.01, "high": 0.15, "log": True}],
        "feature_fraction": ["float", {"low": 0.4, "high": 1.0, "log": False}],
        "bagging_fraction": ["float", {"low": 0.4, "high": 1.0, "log": False}],
        "bagging_freq": ["none", [1]],
    }

    try:
        opt_params = model.hyper_opt(
            hp, dtrain,
            num_boost_round=500,
            nfold=4,
            early_stopping_rounds=30,
            max_minutes=5,
            n_trials=50,
            silence=True,
        )
    except Exception as e:
        return {"name": name, "error": str(e)}

    model.train(opt_params, dtrain, num_boost_round=opt_params["opt_rounds"])

    # Predict on test set
    if start_values_fn is not None:
        start_values_fn(model, X_test)
    preds = model.predict(X_test, pred_type="parameters")

    lines = meta_test["Line"].values
    y_t = y_test.copy()

    # ── Extract EV and shape params, compute metrics in absolute space ──

    if dist_label in ("LogNormal", "ZALN"):
        loc = preds["loc"].values
        scale = preds["scale"].values
        ev_norm = np.exp(loc + scale ** 2 / 2)
        if normalize:
            meanyr_test = np.clip(X_test["MeanYr"].values, 0.5, None)
            ev = ev_norm * meanyr_test
            loc_abs = loc + np.log(meanyr_test)
        else:
            ev = ev_norm
            loc_abs = loc
        sigma = scale
        gate = preds["gate"].values if "gate" in preds.columns else None

        metrics = compute_metrics(ev, y_t, lines, sigma, dist_label, gate=gate)

        # NLL in absolute space
        if gate is not None:
            nll_vals = np.where(
                y_t == 0,
                -np.log(np.clip(gate, 1e-12, None)),
                -np.log(np.clip(1 - gate, 1e-12, None))
                + (-lognorm.logpdf(np.clip(y_t, 1e-9, None), s=sigma,
                                   scale=np.exp(loc_abs)))
            )
        else:
            valid = y_t > 0
            nll_vals = -lognorm.logpdf(np.clip(y_t[valid], 1e-9, None),
                                       s=sigma[valid], scale=np.exp(loc_abs[valid]))
        metrics["nll"] = float(np.mean(np.clip(nll_vals, -20, 20)))
        metrics["model_sigma_mean"] = float(np.mean(sigma))
        empirical_cv = np.std(y_t[y_t > 0]) / np.mean(y_t[y_t > 0]) if np.sum(y_t > 0) > 1 else 1.0
        metrics["empirical_sigma"] = float(np.sqrt(np.log1p(empirical_cv ** 2)))

    elif dist_label in ("Gamma", "ZAGamma"):
        alpha = preds["concentration"].values
        beta = preds["rate"].values
        ev_gamma = alpha / beta
        if normalize:
            meanyr_test = np.clip(X_test["MeanYr"].values, 0.5, None)
            ev = ev_gamma * meanyr_test
            beta_abs = beta / meanyr_test
        else:
            ev = ev_gamma
            beta_abs = beta
        gate = preds["gate"].values if "gate" in preds.columns else None

        metrics = compute_metrics(
            ev, y_t, lines, alpha,
            "ZAGamma" if gate is not None else "Gamma", gate=gate
        )

        valid = y_t > 0
        if gate is not None:
            nll_vals = np.where(
                y_t == 0,
                -np.log(np.clip(gate, 1e-12, None)),
                -np.log(np.clip(1 - gate, 1e-12, None))
                + (-gamma_dist.logpdf(np.clip(y_t, 1e-9, None), alpha,
                                      scale=1 / beta_abs))
            )
        else:
            nll_vals = -gamma_dist.logpdf(np.clip(y_t[valid], 1e-9, None),
                                          alpha[valid], scale=1 / beta_abs[valid])
        metrics["nll"] = float(np.mean(np.clip(nll_vals, -20, 20)))
        metrics["model_alpha_mean"] = float(np.mean(alpha))

    elif dist_label == "NegBin":
        r = preds["total_count"].values
        p = preds["probs"].values
        ev_nb = r * p / (1 - p)
        if normalize:
            meanyr_test = np.clip(X_test["MeanYr"].values, 0.5, None)
            ev = ev_nb * meanyr_test
        else:
            ev = ev_nb
        gate = None

        metrics = compute_metrics(ev, y_t, lines, r, "NegBin")

        result_int = np.clip(y_t.astype(int), 0, None)
        p_eval = r / (r + np.clip(ev, 1e-9, None))
        nll_vals = -np.clip(nbinom.logpmf(result_int, r, p_eval), -20, 0)
        metrics["nll"] = float(np.mean(nll_vals))
        metrics["model_r_mean"] = float(np.mean(r))

    elif dist_label == "SkewNormal":
        loc = preds["loc"].values
        scale = preds["scale"].values
        alpha_skew = preds["alpha"].values
        delta = alpha_skew / np.sqrt(1 + alpha_skew ** 2)
        ev_sn = loc + scale * delta * np.sqrt(2 / np.pi)
        if normalize:
            meanyr_test = np.clip(X_test["MeanYr"].values, 0.5, None)
            ev = ev_sn * meanyr_test
            loc_abs = loc * meanyr_test
            scale_abs = scale * meanyr_test
        else:
            ev = ev_sn
            loc_abs = loc
            scale_abs = scale

        metrics = compute_metrics(ev, y_t, lines, (scale_abs, alpha_skew), "SkewNormal")

        from scipy.stats import skewnorm
        nll_vals = -skewnorm.logpdf(y_t, alpha_skew, loc=loc_abs, scale=scale_abs)
        metrics["nll"] = float(np.mean(np.clip(nll_vals, -20, 20)))
        metrics["model_alpha_skew_mean"] = float(np.mean(alpha_skew))

    # ── Compression diagnostic ──
    meanyr = X_test["MeanYr"].values
    metrics["ev_meanyr_corr"] = float(np.corrcoef(meanyr, ev - meanyr)[0, 1])
    metrics["result_meanyr_corr"] = float(np.corrcoef(meanyr, y_t - meanyr)[0, 1])
    metrics["compression_ratio"] = (
        abs(metrics["ev_meanyr_corr"]) / max(abs(metrics["result_meanyr_corr"]), 1e-6)
    )

    # Hyperparameters selected
    metrics["lr"] = opt_params.get("learning_rate", None)
    metrics["rounds"] = opt_params.get("opt_rounds", None)
    metrics["leaves"] = opt_params.get("num_leaves", None)

    metrics["name"] = name
    metrics["loss_fn"] = loss_fn
    metrics["dist"] = dist_label
    metrics["normalized"] = normalize

    return metrics


# ── Main ────────────────────────────────────────────────────────────────

def main():
    all_results = []

    for market_name, cfg in tqdm(MARKETS.items(), desc="Markets", unit="market"):
        print(f"\n{'=' * 60}")
        print(f"  {market_name}  (current dist: {cfg['dist']})")
        print(f"{'=' * 60}")

        X_train, X_val, X_test, y_train, y_val, y_test, meta_val, meta_test = \
            load_market(cfg["file"], cfg["zi"], cfg["continuous"])

        global_mean = np.mean(y_train)
        zero_rate = np.mean(y_train == 0)
        print(f"  After ZI preprocessing: mean={global_mean:.2f}, "
              f"zero_rate={zero_rate:.3f}, N_train={len(y_train)}")

        # Normalized targets for computing global start values
        meanyr_train = np.clip(X_train["MeanYr"].values, 0.5, None)
        y_train_norm = y_train / meanyr_train

        experiments = []

        # ── 1. Baseline (unnormalized, current distribution) ────────────
        cur_dist = cfg["dist"]
        if cur_dist == "ZAGamma":
            # Zeros already removed for continuous-base → use Gamma (non-ZA)
            experiments.append((
                "Baseline Gamma (no ZA)",
                Gamma(stabilization="None", loss_fn="nll", response_fn="softplus"),
                "Gamma", "nll", False,
                lambda m, X: set_model_start_values(m, "Gamma", X),
            ))
        elif cur_dist == "Gamma":
            experiments.append((
                "Baseline Gamma",
                Gamma(stabilization="None", loss_fn="nll", response_fn="softplus"),
                "Gamma", "nll", False,
                lambda m, X: set_model_start_values(m, "Gamma", X),
            ))
        elif cur_dist == "NegBin":
            experiments.append((
                "Baseline NegBin",
                NegativeBinomial(stabilization="None", loss_fn="nll"),
                "NegBin", "nll", False,
                lambda m, X: set_model_start_values(m, "NegBin", X),
            ))

        # ── 2. LogNormal normalized NLL ─────────────────────────────────
        ln_sv = make_global_start_values("LogNormal", y_train_norm)
        experiments.append((
            "LogNormal norm NLL",
            LogNormal(stabilization="None", loss_fn="nll"),
            "LogNormal", "nll", True, ln_sv,
        ))

        # ── 3. LogNormal normalized CRPS ────────────────────────────────
        experiments.append((
            "LogNormal norm CRPS",
            LogNormal(stabilization="None", loss_fn="crps"),
            "LogNormal", "crps", True, ln_sv,
        ))

        # ── 4. SkewNormal normalized NLL ────────────────────────────────
        sn_sv = make_global_start_values("SkewNormal", y_train_norm)
        experiments.append((
            "SkewNormal norm NLL",
            SkewNormal(stabilization="None", loss_fn="nll"),
            "SkewNormal", "nll", True, sn_sv,
        ))

        # ── 5. SkewNormal normalized CRPS ───────────────────────────────
        experiments.append((
            "SkewNormal norm CRPS",
            SkewNormal(stabilization="None", loss_fn="crps"),
            "SkewNormal", "crps", True, sn_sv,
        ))

        # ── 6. Current dist normalized (only for continuous base) ───────
        # NegBin can't handle non-integer normalized targets, so skip
        if cfg["continuous"]:
            experiments.append((
                "Gamma norm NLL",
                Gamma(stabilization="None", loss_fn="nll", response_fn="softplus"),
                "Gamma", "nll", True,
                lambda m, X: set_model_start_values(m, "Gamma", X),
            ))

        # ── Run all experiments for this market ─────────────────────────
        for exp_name, dist_obj, dist_label, loss_fn, normalize, sv_fn in experiments:
            full_name = f"{market_name} | {exp_name}"
            print(f"\n  ▸ {exp_name} ...", flush=True)
            try:
                result = run_experiment(
                    full_name, X_train, X_val, X_test,
                    y_train, y_val, y_test,
                    meta_test, dist_obj, dist_label, loss_fn,
                    normalize=normalize, start_values_fn=sv_fn,
                )
                all_results.append(result)

                if "error" in result:
                    print(f"    ERROR: {result['error']}")
                else:
                    print(f"    NLL={result['nll']:.3f}  "
                          f"Acc={result.get('accuracy', float('nan')):.3f}  "
                          f"Over%={result.get('over_pct', float('nan')):.3f}")
                    print(f"    EV_mean={result['model_ev_mean']:.2f}  "
                          f"Result_mean={result['result_mean']:.2f}")
                    print(f"    ev_corr={result['ev_meanyr_corr']:.3f}  "
                          f"result_corr={result['result_meanyr_corr']:.3f}  "
                          f"compress_ratio={result['compression_ratio']:.2f}x")
                    lr = result.get('lr', 0)
                    print(f"    HP: lr={lr:.4f}  "
                          f"rounds={result.get('rounds', '?')}  "
                          f"leaves={result.get('leaves', '?')}")
            except Exception as e:
                print(f"    EXCEPTION: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({"name": full_name, "error": str(e)})

    # ── Summary table ───────────────────────────────────────────────────
    print(f"\n\n{'=' * 130}")
    print("SUMMARY")
    print(f"{'=' * 130}")
    header = (f"{'Name':<45} {'Dist':<12} {'Loss':<6} {'Norm':<5} "
              f"{'NLL':>7} {'Acc':>6} {'Over%':>6} "
              f"{'EV_corr':>8} {'R_corr':>8} {'Cmpr':>6}")
    print(header)
    print("-" * 130)
    for r in all_results:
        if "error" in r:
            print(f"{r['name']:<45} ERROR: {r.get('error', '?')[:70]}")
        else:
            print(
                f"{r['name']:<45} "
                f"{r.get('dist', '?'):<12} "
                f"{r.get('loss_fn', '?'):<6} "
                f"{'Yes' if r.get('normalized') else 'No':<5} "
                f"{r.get('nll', float('nan')):>7.3f} "
                f"{r.get('accuracy', float('nan')):>6.3f} "
                f"{r.get('over_pct', float('nan')):>6.3f} "
                f"{r.get('ev_meanyr_corr', float('nan')):>8.3f} "
                f"{r.get('result_meanyr_corr', float('nan')):>8.3f} "
                f"{r.get('compression_ratio', float('nan')):>6.2f}x"
            )

    # Save results to JSON for later analysis
    out_path = DATA_DIR.parent / "experiment_results.json"
    serializable = []
    for r in all_results:
        sr = {}
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)):
                sr[k] = float(v)
            else:
                sr[k] = v
        serializable.append(sr)
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
