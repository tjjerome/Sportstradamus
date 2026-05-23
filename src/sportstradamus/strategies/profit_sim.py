"""Kelly-sized profit / ROI / Sharpe / drawdown backtest.

Phase 2 of the new ship-gate plan
([docs/ship_gate.md](../../../docs/ship_gate.md)) — lifted from
``pages/6_📊_Stats_Profit_Sim.py`` so the same Monte-Carlo strategy backtest
that powers the dashboard can also size the offline supersede gate
(S3 — paired Sharpe) and the live devel→main gate (positive Kelly ROI;
+0.5% / 2-wk vs incumbent). Behavior is preserved relative to the page;
the page now imports these symbols.

Public surface:

- :func:`compute_payout` — per-row payout multiplier for a winning bet.
- :func:`simulate_strategy` — Monte-Carlo backtest of a filter-and-rank
  strategy on an exploded-offer DataFrame.
- :func:`simulate_kelly_all` — deterministic single-run, no-threshold,
  Kelly-sized variant used by the ship gate (two models bet the same
  universe, so paired Sharpe is well-defined).
- :func:`summarize_runs` — collapse a simulation result into
  ``{mean_final, roi, max_drawdown, sharpe, win_rate}``.

Input DataFrame contract (from
:func:`sportstradamus.dashboard_data.get_filtered_history`):
the exploded per-offer frame, indexed by row, with columns
``Player, Market, Platform, Boost, Model P, Books, Hit`` plus the
``ranking`` column (``K`` / ``Model P`` / ``Model``) and the caller's
``prob_col`` (defaults to ``Model P`` in the dashboard). A ``_date``
column is materialized inside the simulator from ``Date`` if absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 100 MC runs is the dashboard default — enough to compute a stable 10-90 band
# without making the Streamlit page sluggish. Gate callers override to 1.
N_MONTE_CARLO_DEFAULT: int = 100

# Quarter-Kelly hedge against win-prob estimation noise (matches the live
# pickem sizer in ``strategies/kelly.py``); the dashboard uses straight Kelly
# capped at 5% per leg, so we preserve that cap here.
_KELLY_CAP_FRACTION: float = 0.05

# Default Monte-Carlo seed; the page used 42, so we keep it for parity.
_DEFAULT_RNG_SEED: int = 42

# Kelly-all cap on ``max_bets_day`` — high enough that the page's MC sampling
# branch never fires for the gate, so the run is fully deterministic.
_KELLY_ALL_MAX_BETS_DAY: int = 10_000_000

# Underdog / default sportsbook -110 vig (100/110 payout per dollar staked).
_DEFAULT_AMERICAN_MINUS_110_PAYOUT: float = 100.0 / 110.0

# Ranking column mapping — translates the user-facing ranking label into the
# exploded-frame column name the simulator sorts on.
RANKING_MAP: dict[str, str] = {
    "Kelly": "K",
    "Probability": "Model P",
    "EV": "Model",
}


def compute_payout(row: pd.Series) -> float:
    """Return the payout multiplier for a winning $1 bet on ``row``.

    Underdog / unspecified platforms pay -110 (100/110 per dollar staked);
    Sleeper pays 1× (the platform tracks its own juice elsewhere). All
    platforms get the per-row ``Boost`` multiplier applied last.
    """
    platform = row.get("Platform", "")
    boost = row.get("Boost", 1)
    if platform == "Sleeper":
        return float(boost)
    return _DEFAULT_AMERICAN_MINUS_110_PAYOUT * float(boost)


def simulate_strategy(
    df: pd.DataFrame,
    *,
    prob_col: str,
    ranking: str,
    min_model_p: float,
    min_books_p: float,
    max_bets_day: int,
    sizing_pct: float,
    use_kelly: bool,
    initial_bankroll: float,
    n_mc: int = N_MONTE_CARLO_DEFAULT,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Monte-Carlo backtest a filter-and-rank strategy on exploded offers.

    Filters ``df`` to rows with ``prob_col >= min_model_p`` and
    ``Books >= min_books_p``, then for each MC run walks the date axis and
    on each date picks at most ``max_bets_day`` bets via probability-
    weighted sampling (when more eligible than the cap) or by deduplicating
    on the base player (when at or below the cap). Bets are sized as
    fixed-percent of bankroll or Kelly (capped at 5%) per ``use_kelly``.

    Returns a long-format DataFrame ``[date, run, bankroll, daily_pnl]``,
    one row per (date, run). Empty input or no eligible rows return an
    empty DataFrame.
    """
    rank_col = RANKING_MAP.get(ranking, "K")

    eligible = df.loc[(df[prob_col] >= min_model_p) & (df["Books"].fillna(0) >= min_books_p)].copy()
    if eligible.empty:
        return pd.DataFrame()

    eligible = eligible.sort_values(rank_col, ascending=False).drop_duplicates(
        subset=["Player", "Market", "_date"], keep="first"
    )
    eligible["_player_base"] = eligible["Player"].str.split(r" \+ | vs\. ").str[0]
    eligible["Payout"] = eligible.apply(compute_payout, axis=1)
    eligible = eligible.sort_values("_date")
    dates = sorted(eligible["_date"].unique())

    if rng is None:
        rng = np.random.default_rng(_DEFAULT_RNG_SEED)

    all_runs: list[dict[str, object]] = []
    for run_i in range(n_mc):
        bankroll = float(initial_bankroll)
        for date in dates:
            day_bets = eligible.loc[eligible["_date"] == date]
            if day_bets.empty:
                all_runs.append(
                    {"date": date, "run": run_i, "bankroll": bankroll, "daily_pnl": 0.0}
                )
                continue

            day_bets = _pick_day_bets(day_bets, rank_col, max_bets_day, rng)
            daily_pnl = _settle_day(day_bets, prob_col, use_kelly, sizing_pct, bankroll)
            bankroll = max(bankroll + daily_pnl, 0.0)
            all_runs.append(
                {"date": date, "run": run_i, "bankroll": bankroll, "daily_pnl": daily_pnl}
            )

    return pd.DataFrame(all_runs)


def simulate_kelly_all(
    df: pd.DataFrame,
    *,
    prob_col: str,
    initial_bankroll: float,
) -> pd.DataFrame:
    """Deterministic single-run Kelly-all backtest used by the ship gate.

    Equivalent to :func:`simulate_strategy` with no thresholds, no
    Monte-Carlo sampling, Kelly sizing, and ``ranking="EV"``. Two models
    invoked against the same input frame bet on the same universe of
    events, which is what makes the paired Sharpe (S3) and live ROI
    comparisons well-defined.
    """
    return simulate_strategy(
        df,
        prob_col=prob_col,
        ranking="EV",
        min_model_p=0.0,
        min_books_p=0.0,
        max_bets_day=_KELLY_ALL_MAX_BETS_DAY,
        sizing_pct=1.0,
        use_kelly=True,
        initial_bankroll=initial_bankroll,
        n_mc=1,
    )


def extract_sim_returns(sim_df: pd.DataFrame, initial_bankroll: float) -> np.ndarray:
    """Per-event return series ``daily_pnl / start-of-event bankroll``.

    Mirrors :func:`summarize_runs`'s Sharpe computation so callers that
    want the raw per-event return series (e.g. Memmel paired-Sharpe inference)
    use the same denominator that ``summarize_runs`` uses. Returns an empty
    array for an empty simulation result.

    Args:
        sim_df: Result frame from :func:`simulate_kelly_all` or
            :func:`simulate_strategy` — columns ``run``, ``daily_pnl``,
            ``bankroll``.
        initial_bankroll: Starting bankroll used in the simulation; fills the
            shift-by-1 NaN at the first row of each run.

    Returns:
        1-D float array of per-event returns, all runs concatenated.
    """
    if sim_df.empty:
        return np.array([], dtype=float)
    series = sim_df.groupby("run", group_keys=False)[["daily_pnl", "bankroll"]].apply(
        lambda x: x["daily_pnl"].values
        / np.maximum(
            x["bankroll"].shift(1).fillna(initial_bankroll).values,
            1,
        )
    )
    return np.concatenate(series.values)


def summarize_runs(result: pd.DataFrame, initial_bankroll: float) -> dict[str, float]:
    """Collapse a :func:`simulate_strategy` result into headline metrics.

    Returns ``{mean_final, roi, max_drawdown, sharpe, win_rate}``.
    ``sharpe`` is plain ``mean(returns) / std(returns)`` (not annualized;
    matches the dashboard's "Sharpe Ratio" column). ``max_drawdown`` is
    the mean across MC runs of each run's peak-to-trough drawdown.
    ``win_rate`` is the fraction of (run, date) cells with positive PnL.
    """
    if result.empty:
        return {
            "mean_final": float(initial_bankroll),
            "roi": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "win_rate": 0.0,
        }

    final_runs = result.groupby("run").last()
    final_bankrolls = final_runs["bankroll"].values
    mean_final = float(final_bankrolls.mean())
    roi = (mean_final - float(initial_bankroll)) / float(initial_bankroll)

    drawdowns: list[float] = []
    for run_i in result["run"].unique():
        run_bankroll = result.loc[result["run"] == run_i, "bankroll"].to_numpy()
        if len(run_bankroll) == 0:
            continue
        peak = np.maximum.accumulate(run_bankroll)
        dd = (peak - run_bankroll) / np.where(peak > 0, peak, 1)
        drawdowns.append(float(dd.max()))
    max_drawdown = float(np.mean(drawdowns)) if drawdowns else 0.0

    daily_returns = result.groupby("run", group_keys=False)[["daily_pnl", "bankroll"]].apply(
        lambda x: x["daily_pnl"].values
        / np.maximum(x["bankroll"].shift(1).fillna(initial_bankroll).values, 1)
    )
    all_returns = np.concatenate(daily_returns.values)
    sharpe = float(np.mean(all_returns) / np.std(all_returns)) if np.std(all_returns) > 0 else 0.0

    daily_outcomes = result.groupby(["run", "date"])["daily_pnl"].sum().reset_index()
    win_rate = float((daily_outcomes["daily_pnl"] > 0).mean())

    return {
        "mean_final": mean_final,
        "roi": roi,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "win_rate": win_rate,
    }


# --------------------------------------------------------------------------- #
# Internal helpers


def _pick_day_bets(
    day_bets: pd.DataFrame, rank_col: str, max_bets_day: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Return the subset of ``day_bets`` that ``simulate_strategy`` actually wagers.

    When the eligible count exceeds ``max_bets_day``, sample without
    replacement weighted by ``rank_col`` and reject duplicates by base
    player. When at or below the cap, dedupe by base player keeping the
    best by ``rank_col``.
    """
    if len(day_bets) > max_bets_day:
        weights = day_bets[rank_col].clip(lower=0.01).to_numpy()
        weights = weights / weights.sum()
        chosen: list[int] = []
        used_players: set[str] = set()
        pool_idx = list(range(len(day_bets)))
        pool_weights = weights.copy()
        while len(chosen) < max_bets_day and pool_idx:
            pool_weights_norm = pool_weights / pool_weights.sum()
            pick = rng.choice(len(pool_idx), p=pool_weights_norm)
            actual_idx = pool_idx[pick]
            player_base = day_bets.iloc[actual_idx]["_player_base"]
            if player_base not in used_players:
                chosen.append(actual_idx)
                used_players.add(player_base)
            pool_idx.pop(pick)
            pool_weights = np.delete(pool_weights, pick)
            if pool_weights.sum() == 0:
                break
        return day_bets.iloc[chosen]

    return day_bets.sort_values(rank_col, ascending=False).drop_duplicates(
        subset=["_player_base"], keep="first"
    )


def _settle_day(
    day_bets: pd.DataFrame,
    prob_col: str,
    use_kelly: bool,
    sizing_pct: float,
    bankroll: float,
) -> float:
    """Apply the bet-sizing rule to ``day_bets`` and return the day's PnL."""
    daily_pnl = 0.0
    for _, bet in day_bets.iterrows():
        if use_kelly:
            payout_mult = bet["Payout"]
            if payout_mult <= 1:
                continue
            kelly_f = (bet[prob_col] * payout_mult - 1) / (payout_mult - 1)
            kelly_f = max(0.0, min(kelly_f, _KELLY_CAP_FRACTION))
            bet_size = bankroll * kelly_f
        else:
            bet_size = bankroll * float(sizing_pct) / 100.0

        if bet["Hit"]:
            daily_pnl += bet_size * bet["Payout"]
        else:
            daily_pnl -= bet_size
    return daily_pnl
