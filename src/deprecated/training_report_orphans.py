# ARCHIVED 2026-11-26 from src/sportstradamus/training/report.py
# Reason: Its one caller, underdog_pickem.resolve_market_shrinkage, was archived when
#         pick'em and the paper ledger came to read the fitted number (policy_v5). The
#         dashboard's slip rail reads model_stats' kelly_shrinkage column itself
#         (dashboard.components.slip_builder.slip_shrinkage). Its three tests were
#         deleted from tests/golden/test_training_metrics.py (git keeps them).
# Last live SHA: 41d03f93
# Original imports (now unresolved here):
#   import pandas as pd
#   from sportstradamus.training.report import MODEL_STATS_PATH

# From src/sportstradamus/training/report.py.
def get_market_calibration(league: str, market: str) -> dict[str, float]:
    """Return ``{kelly_shrinkage, brier_skill_score, model_weight}`` for one market.

    Reads ``model_stats.parquet``. Returns NaNs when the row is missing so the
    Kelly resolution chain can fall through to its next source. Kelly imports
    this getter — never reaches into the parquet directly.
    """
    nan_result = {
        "kelly_shrinkage": float("nan"),
        "brier_skill_score": float("nan"),
        "model_weight": float("nan"),
    }
    if not MODEL_STATS_PATH.is_file():
        return nan_result
    df = pd.read_parquet(MODEL_STATS_PATH)
    if df.empty:
        return nan_result
    sub = df[(df["league"] == league) & (df["market"] == market)]
    if sub.empty:
        return nan_result
    row = sub.iloc[0]
    return {
        "kelly_shrinkage": float(row.get("kelly_shrinkage", float("nan"))),
        "brier_skill_score": float(row.get("brier_skill_score", float("nan"))),
        "model_weight": float(row.get("model_weight", float("nan"))),
    }
