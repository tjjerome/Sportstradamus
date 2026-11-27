# ARCHIVED 2026-11-27 from src/sportstradamus/training/correlate.py
# Reason: The summary's same_team and opposing scopes pool the league's games
#         (_league_pair_summary, docs/handoffs/profit-engine.md 5.3), so nothing averages
#         the stored per-team blocks.
# Last live SHA: 87739585
# Original imports (now unresolved here):
#   import pandas as pd

def _market_summary(blocks: pd.Series, scope: str) -> pd.DataFrame:
    """Aggregate (team, market_a, market_b) → R across teams to market-pair means.

    Position prefixes strip (B1.AST → AST) so the pair grid is market-level;
    pairs pool with an unweighted mean over the pooled rows. ``n_teams`` is the
    pooled (team × slot-prefix) row count — the legacy name the dashboard's
    thin-pair mask reads — and ``n_teams_distinct`` counts the distinct teams
    behind those rows, so downstream shrinkage can tell 30 teams from 30
    batting-slot pairs on 4 teams.
    """
    columns = ["market_a", "market_b", "rho_mean", "n_teams", "scope", "n_teams_distinct"]
    if blocks.empty:
        return pd.DataFrame(columns=columns)

    df = blocks.rename("R").reset_index()
    df.columns = ["team", "market_a", "market_b", "R"]
    for market_col in ("market_a", "market_b"):
        df[market_col] = df[market_col].str.split(".").str[-1]
    out = (
        df.groupby(["market_a", "market_b"])
        .agg(
            rho_mean=("R", "mean"),
            n_teams=("R", "size"),
            n_teams_distinct=("team", "nunique"),
        )
        .reset_index()
    )
    out["scope"] = scope
    return out[columns]


\n