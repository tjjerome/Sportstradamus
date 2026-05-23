"""HurdleZINB: two-stage replacement for the joint ZINB distributional model.

The joint ZINB's gate is unidentified under NLL (it converges to ~half the true
structural-zero rate), producing inflated "Over" probabilities in zero-heavy
markets (FG3M, OREB, PF, …).  HurdleZINB fixes this by separating the two
learning problems:

Stage 1 — a calibrated binary LightGBM classifier estimates q = P(Y = 0)
directly, without competing with the count distribution's NLL.

Stage 2 — a NegBin LightGBMLSS fitted on the strictly-positive subset
(Y > 0) captures the count shape.

The structural-inflation gate π is then *derived* from the ZINB identity
rather than learned:

    q  = π + (1 − π) · NB(0)
    ⇒  π = clip((q − NB(0)) / (1 − NB(0)), 0, 1)

where NB(0) = (1 − p)^r  (PyTorch NegativeBinomial convention).

The returned ``gate`` column is π.  By construction
P(Y = 0) = π + (1 − π) · NB(0) = q, matching the calibrated classifier
exactly — no ZINB/hurdle double-count.

See ``docs/OVERCONFIDENCE_INVESTIGATION.md`` and
``docs/plans/2026-05-18-fga-fg3m-overconfidence-fix.md`` for the full
analysis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Numeric guard: keep denominator (1 − NB(0)) away from zero.
_NB0_DENOM_FLOOR: float = 1e-12

# Clip the binary classifier's output away from 0/1 before using it in the
# ZINB identity — pure-0 or pure-1 probabilities are artefacts of boosting
# on the boundary and would collapse the gate formula.
_Q_CLIP_LO: float = 1e-6
_Q_CLIP_HI: float = 1.0 - 1e-6

# Floor on (1 − p(0)) inside the zero-truncated NB loss so log() stays finite
# when p(0) → 1 (degenerate total_count → 0 or probs → 0 mid-boosting). Same
# guard role as _NB0_DENOM_FLOOR on the predict() side. STYLE_GUIDE §9.
_ZTNB_P0_FLOOR: float = 1e-12

# LightGBM param keys accepted by plain lgb.train. All LightGBMLSS-specific
# keys (e.g. "opt_rounds") are excluded — passing them to lgb.train raises
# an error.
_LGB_ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        "num_leaves",
        "learning_rate",
        "feature_fraction",
        "bagging_fraction",
        "bagging_freq",
        "min_child_samples",
        "min_child_weight",
        "max_bin",
        "max_depth",
        "lambda_l1",
        "lambda_l2",
        "path_smooth",
        "num_threads",
        "verbosity",
        "feature_pre_filter",
        "monotone_constraints",
        # Determinism kwargs from seed_everything — must also be forwarded.
        "seed",
        "bagging_seed",
        "feature_fraction_seed",
        "deterministic",
        "force_row_wise",
    }
)


class _ZeroTruncatedNB:
    """torch ``NegativeBinomial`` whose ``log_prob`` is zero-truncated (ZTNB).

    Validated building block for a future marginalized-ZINB (MZINB) head — it is
    intentionally NOT wired into :meth:`HurdleZINB.fit`. Stage B1 found that
    swapping the Stage-2 loss to this ZTNB does recover an unbiased count
    component, but the resulting ``NB(0)`` then exceeds the calibrated zero rate
    ``q`` on most rows, which the frozen derived-π identity cannot represent (it
    clips π to 0, breaking the reconstruction). The joint/hurdle decode therefore
    stays on the full-support NB; see the Stage-B1 outcome + Track-B rescope in
    ``docs/operation_ship_references.md`` §6.

    Mechanics (for MZINB reuse): LightGBMLSS computes its loss as
    ``-Σ distribution(**params).log_prob(y)``, so assigning this class to a
    distribution's ``.distribution`` factory makes the optimizer maximize

        log p_ZTNB(y) = log p_NB(y) − log(1 − p_NB(0))

    on the strictly-positive subset. All non-loss attributes (``sample``,
    ``rsample``, ``mean``, …) delegate to the wrapped NB. Module-level + picklable
    by reference. Covered by ``tests/test_ztnb_loss.py``.
    """

    def __init__(self, total_count=None, probs=None, **kwargs: object) -> None:
        # Lazy import: keep module import light (matches fit()'s lazy imports).
        from torch.distributions import NegativeBinomial

        self._nb = NegativeBinomial(total_count=total_count, probs=probs, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # Delegate everything except loss to the wrapped NB. Guard ``_nb`` so a
        # pre-init / unpickled access raises cleanly instead of recursing.
        if name == "_nb":
            raise AttributeError(name)
        return getattr(self._nb, name)

    def log_prob(self, value: Any) -> Any:
        import torch

        log_pk = self._nb.log_prob(value)
        log_p0 = self._nb.log_prob(torch.zeros_like(value))
        # 1 − p(0), floored away from 0 so log() stays finite (see _ZTNB_P0_FLOOR).
        # expm1(log_p0) is the stable form of exp(log_p0) − 1.
        one_minus_p0 = torch.clamp(-torch.expm1(log_p0), min=_ZTNB_P0_FLOOR)
        return log_pk - torch.log(one_minus_p0)


class HurdleZINB:
    """Two-stage hurdle model that replaces the joint ZINB.

    The public prediction surface (``total_count``, ``probs``, ``gate``)
    matches the LightGBMLSS ZINB column contract so downstream consumers in
    ``prediction/model_prob.py`` require no changes.

    Class attribute ``is_hurdle = True`` lets the pipeline and model_prob
    distinguish this object from a plain ``LightGBMLSS`` without ``isinstance``
    coupling.

    Attributes:
        is_hurdle: Always ``True``; pipeline guard.
        clf: Trained binary LightGBM Booster (Stage 1). ``None`` before fit.
        nb: Trained NegBin LightGBMLSS model (Stage 2). ``None`` before fit.
    """

    is_hurdle: bool = True

    def __init__(self) -> None:
        self.clf = None
        self.nb = None
        self._shape_ceiling: float | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        hp: dict,
        rounds: int,
        *,
        shape_ceiling: float | None = None,
        seed: int | None = None,
    ) -> HurdleZINB:
        """Train both stages. Returns self.

        Stage 1 fits a binary LightGBM classifier on (y == 0) to estimate
        q = P(Y = 0).  Stage 2 fits a NegBin LightGBMLSS on the positive
        subset (y > 0).

        Args:
            X: Feature matrix (DataFrame). Must contain ``MeanYr``, ``STDYr``,
                and ``ZeroYr`` columns (required by ``set_model_start_values``).
            y: Target array. May contain zeros (structural + sampling).
            hp: LightGBM hyperparameter dict produced by Optuna. LightGBMLSS-
                specific keys (e.g. ``opt_rounds``) are silently filtered
                before being passed to ``lgb.train``.
            rounds: Number of boosting rounds for both stages.
            shape_ceiling: Upper bound on the NegBin ``total_count`` shape
                parameter. Forwarded to ``set_model_start_values`` and the
                ``_BoundedResponseFn`` wrapper.
            seed: If not ``None``, pins all RNGs and merges LightGBM
                determinism kwargs into both stage param dicts. For offline
                eval / debug only — never use in production.

        Returns:
            ``self`` (for chaining).
        """
        import lightgbm as lgb
        from lightgbmlss.distributions.NegativeBinomial import NegativeBinomial
        from lightgbmlss.model import LightGBMLSS

        from sportstradamus.helpers import set_model_start_values
        from sportstradamus.training.hyperparams import _BoundedResponseFn
        from sportstradamus.training.pipeline import seed_everything

        self._shape_ceiling = shape_ceiling

        # Determinism: pin all RNGs and collect lgb kwargs ONCE, then merge
        # into both stage param dicts.
        det_kwargs: dict = {}
        if seed is not None:
            det_kwargs = seed_everything(seed)

        # --- Stage 1: binary classifier on zero/non-zero label ---
        binary_labels = (y == 0).astype(int)
        clf_params = {k: v for k, v in hp.items() if k in _LGB_ALLOWED_KEYS}
        clf_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            **clf_params,
            **det_kwargs,
        }
        dtrain_clf = lgb.Dataset(X, label=binary_labels)
        self.clf = lgb.train(clf_params, dtrain_clf, num_boost_round=rounds)

        # --- Stage 2: NegBin LightGBMLSS on positive subset only ---
        pos_mask = y > 0
        X_pos = X[pos_mask]
        y_pos = y[pos_mask]

        nb_dist = NegativeBinomial(stabilization="None", loss_fn="nll")
        if shape_ceiling is not None:
            nb_dist.param_dict["total_count"] = _BoundedResponseFn(
                nb_dist.param_dict["total_count"], shape_ceiling
            )

        self.nb = LightGBMLSS(nb_dist)
        set_model_start_values(self.nb, "NegBin", X_pos, shape_ceiling=shape_ceiling)

        nb_params = {k: v for k, v in hp.items() if k in _LGB_ALLOWED_KEYS}
        nb_params = {
            "verbosity": -1,
            **nb_params,
            **det_kwargs,
        }
        nb_params["opt_rounds"] = rounds
        dtrain_nb = lgb.Dataset(X_pos, label=y_pos.astype(float))
        self.nb.train(nb_params, dtrain_nb, num_boost_round=rounds)

        return self

    def set_model_start_values(self, X: pd.DataFrame) -> None:
        """Delegate to the internal NegBin model for uniform pipeline call sites.

        Assigns ``self.nb.start_values`` — same as calling
        ``set_model_start_values(nb, "NegBin", X, ...)`` directly.

        Args:
            X: Feature matrix (DataFrame). Must contain ``MeanYr``, ``STDYr``,
                and ``ZeroYr`` columns.
        """
        if self.nb is None:
            return
        from sportstradamus.helpers import set_model_start_values

        set_model_start_values(self.nb, "NegBin", X, shape_ceiling=self._shape_ceiling)

    def predict(self, X: pd.DataFrame, pred_type: str = "parameters") -> pd.DataFrame:
        """Return distribution parameters indexed like X.

        Computes the derived-π gate via the ZINB identity so the returned
        ``gate`` column is the structural-inflation probability π, not the
        raw classifier output q.  Downstream consumers in
        ``prediction/model_prob.py`` can use ``total_count``, ``probs``,
        and ``gate`` unchanged.

        The ZINB identity:
            q  = π + (1 − π) · NB(0)
            π  = clip((q − NB(0)) / (1 − NB(0)), 0, 1)

        where NB(0) = (1 − p)^r  (PyTorch NegativeBinomial convention,
        mean = r · p / (1 − p)).

        Args:
            X: Feature matrix. Must match the training schema.
            pred_type: Forwarded to the internal NegBin model. Only
                ``"parameters"`` is meaningful for downstream use.

        Returns:
            DataFrame with columns ``["total_count", "probs", "gate"]``,
            indexed identically to ``X``.
        """
        if self.clf is None or self.nb is None:
            raise RuntimeError("HurdleZINB.predict() called before fit()")

        # Stage 1: q = P(Y = 0) from calibrated binary classifier.
        q_raw = self.clf.predict(X)
        q = np.clip(q_raw, _Q_CLIP_LO, _Q_CLIP_HI)

        # Stage 2: NegBin parameters on full X (not just positives).
        self.set_model_start_values(X)
        nb_preds = self.nb.predict(X, pred_type="parameters")
        nb_preds.index = X.index

        total_count = nb_preds["total_count"].to_numpy()
        probs = nb_preds["probs"].to_numpy()

        # NB(0) in PyTorch convention: (1 − p)^r
        nb0 = (1 - probs) ** total_count

        # Derived-π gate from the ZINB identity.
        denom = np.clip(1.0 - nb0, _NB0_DENOM_FLOOR, None)
        gate = np.clip((q - nb0) / denom, 0.0, 1.0)

        return pd.DataFrame(
            {"total_count": total_count, "probs": probs, "gate": gate},
            index=X.index,
        )
