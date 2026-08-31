"""Backtest combo-market pricing: legacy mean+generic-cv tail vs component-sum quotes.

As-of replay over historical gamedays. For every graded (player, gameday) with an
archived combo line, both constructions price P(over line) from the same
information cutoff:

* legacy: ``check_combo_markets`` scalar mean read through the combo cell's
  ``(dist, cv)`` — the ``combo_ev_inversion`` tail every consumer sees today,
* kernel: ``Stats.combo_quote`` — admission-gated component-sum distribution.

Outcomes join from the gamelog directly (never history.parquet, whose legacy rows
need filtering). Rows are kept on participation, not on a positive result: a
zero is a settled under for every one of these markets, and dropping zeros
selects the sample toward overs. Per the research brief, calibration is reported
on ALL quotes produced, not only servable ones, so selection stays observable.

    poetry run python -m sportstradamus.scripts.backtest_combo_quotes \
        --league MLB --markets "hitter fantasy points underdog" --days 150
"""

import datetime

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from sportstradamus.helpers import get_odds
from sportstradamus.helpers.config import stat_cv, stat_dist
from sportstradamus.scripts.inject_backfilled_odds import (
    _COMMENCE_STANDIN,
    _load_league,
    _target_at,
)
from sportstradamus.stats.base import archive, is_mlb_pitcher_market

# Acceptance buckets from the lane plan: the flip gate compares |claimed - hit| in the
# directional extremes, n >= 50 per bucket.
_EXTREME_HI = 0.75
_EXTREME_LO = 0.25
_EXTREME_MIN_N = 50
_DECILE_EDGES = np.linspace(0.0, 1.0, 11)


def _game_time(game_date: datetime.date) -> datetime.datetime:
    """Serving's read cutoff: everything quoted before first pitch, nothing after.

    The DFS platforms post their boards hours after the training lookback has
    closed (measured: Underdog's MLB fantasy rows land ~13:50 UTC against a
    12:00 training cutoff), so replaying a serving consumer at the training
    cutoff would grade an empty board.
    """
    return datetime.datetime.combine(game_date, datetime.time()) + _COMMENCE_STANDIN


def _grade_market(stat_data, market: str, days: int, serving: bool) -> pd.DataFrame:
    """One row per (player, gameday) carrying line, result, and both claimed P(over)."""
    ls = stat_data.log_strings
    gamelog = stat_data.gamelog.drop_duplicates(subset=[ls["game"], ls["player"]], keep="last")
    if market not in gamelog.columns:
        raise click.UsageError(f"{stat_data.league} gamelog has no column {market!r}")
    game_dates = pd.to_datetime(gamelog[ls["date"]]).dt.date
    cutoff = datetime.date.today() - datetime.timedelta(days=days)
    gamelog = gamelog.loc[(game_dates > cutoff) & (game_dates < datetime.date.today())]
    # Participation, not a positive result: MLB reads the mound or the plate
    # depending on the market, exactly as get_training_matrix does.
    usage_stat = stat_data.usage_stat
    if stat_data.league == "MLB":
        usage_stat = "batters faced" if is_mlb_pitcher_market(market) else ls["usage"]
    gamelog = gamelog.loc[pd.to_numeric(gamelog[usage_stat], errors="coerce") > 0]

    cv = stat_cv.get(stat_data.league, {}).get(market, 1)
    dist = stat_dist.get(stat_data.league, {}).get(market, "Gamma")

    records = []
    gamedays = gamelog.groupby(pd.to_datetime(gamelog[ls["date"]]).dt.date)
    for game_date, players_df in tqdm(
        gamedays, unit="gameday", desc=f"{stat_data.league} {market}", total=len(gamedays)
    ):
        date = game_date.strftime("%Y-%m-%d")
        at = _game_time(game_date) if serving else _target_at(game_date)
        results = players_df.set_index(ls["player"])[market]
        results = results[~results.index.duplicated()]
        players = list(results.index)

        quote_inputs = archive.get_training_quote_inputs(
            stat_data.league, market, date, players, at=at
        )
        lines = {
            player: legacy_line
            for player, (_, legacy_line) in quote_inputs.items()
            if legacy_line and legacy_line > 0
        }
        if not lines:
            continue

        stat_data.window_short_logs(game_date)
        kernel_quotes = stat_data.combo_quote(market, list(lines), date, at, lines=lines)

        for player, line in lines.items():
            actual = results[player]
            if not np.isfinite(actual) or actual == line:
                continue
            try:
                legacy_ev = stat_data.check_combo_markets(market, player, date)
            except ValueError:
                # The legacy gamelog fill-in inverts a NaN empirical for a player
                # with a quoted component but no trailing games. Counted as an
                # absent legacy quote so one fragile row cannot end the run.
                legacy_ev = np.nan
            legacy_over = np.nan
            if legacy_ev and np.isfinite(legacy_ev) and legacy_ev > 0:
                legacy_over = 1.0 - float(get_odds(line, legacy_ev, dist, cv=cv))
            quote = kernel_quotes.get(player)
            records.append(
                {
                    "League": stat_data.league,
                    "Market": market,
                    "Date": date,
                    "Player": player,
                    "Line": line,
                    "Actual": actual,
                    "Over": actual > line,
                    "legacy_ev": legacy_ev if legacy_ev else np.nan,
                    "legacy_over": legacy_over,
                    "kernel_over": quote.over_probability if quote else np.nan,
                    "kernel_ev": quote.ev if quote else np.nan,
                    "kernel_sd": quote.sum_sd if quote else np.nan,
                    "kernel_books": quote.book_count if quote else 0,
                }
            )
    return pd.DataFrame(records)


def _path_report(frame: pd.DataFrame, col: str) -> list[str]:
    """Coverage, Brier, extreme-bucket calibration, and decile table for one path."""
    quoted = frame.loc[frame[col].notna()]
    lines = [f"  coverage {len(quoted)}/{len(frame)}"]
    if quoted.empty:
        return lines
    over = quoted["Over"].to_numpy(dtype=float)
    p = quoted[col].to_numpy(dtype=float)
    lines.append(
        f"  Brier {np.mean((p - over) ** 2):.4f}  claimed {p.mean():.3f}  hit {over.mean():.3f}"
    )
    for label, mask in (
        (f">= {_EXTREME_HI}", p >= _EXTREME_HI),
        (f"<= {_EXTREME_LO}", p <= _EXTREME_LO),
    ):
        n = int(mask.sum())
        if n:
            gap = abs(p[mask].mean() - over[mask].mean())
            flag = "" if n >= _EXTREME_MIN_N else " (n below gate)"
            lines.append(
                f"  P(over) {label}: n={n} claimed={p[mask].mean():.3f} "
                f"hit={over[mask].mean():.3f} |gap|={gap:.3f}{flag}"
            )
    buckets = pd.cut(p, _DECILE_EDGES, include_lowest=True)
    decile = quoted.groupby(buckets, observed=True).agg(
        n=(col, "size"), claimed=(col, "mean"), hit=("Over", "mean")
    )
    lines.append(decile.round(3).to_string())
    return lines


def _verdict(frame: pd.DataFrame) -> str:
    """Pre-agreed flip gate: kernel not worse on Brier, better |claimed-hit| in extremes."""
    both = frame.loc[frame["legacy_over"].notna() & frame["kernel_over"].notna()]
    if both.empty:
        return "verdict: NO OVERLAP (paths quote disjoint rows)"
    over = both["Over"].to_numpy(dtype=float)
    briers = {
        col: np.mean((both[col].to_numpy(dtype=float) - over) ** 2)
        for col in ("legacy_over", "kernel_over")
    }
    checks = [briers["kernel_over"] <= briers["legacy_over"] + 1e-9]
    for lo, hi in ((_EXTREME_HI, 1.01), (-0.01, _EXTREME_LO)):
        p = both["kernel_over"].to_numpy(dtype=float)
        mask = (p >= lo) & (p <= hi)
        lp = both["legacy_over"].to_numpy(dtype=float)
        lmask = (lp >= lo) & (lp <= hi)
        if mask.sum() >= _EXTREME_MIN_N and lmask.sum() >= _EXTREME_MIN_N:
            kgap = abs(p[mask].mean() - over[mask].mean())
            lgap = abs(lp[lmask].mean() - over[lmask].mean())
            checks.append(kgap <= lgap)
    status = "GREEN" if all(checks) else "NOT GREEN"
    return (
        f"verdict: {status} (paired n={len(both)}; Brier legacy {briers['legacy_over']:.4f} "
        f"vs kernel {briers['kernel_over']:.4f}; {sum(checks)}/{len(checks)} checks pass)"
    )


@click.command()
@click.option("--league", required=True, help="League to backtest.")
@click.option("--markets", required=True, help="Comma-separated combo/fantasy market names.")
@click.option("--days", default=150, show_default=True, help="Gameday lookback window.")
@click.option(
    "--as-of",
    type=click.Choice(["serving", "training"]),
    default="serving",
    show_default=True,
    help="Quote read cutoff: first pitch (serving) or first pitch minus the training lookback.",
)
@click.option(
    "--out",
    default="/tmp/backtest_combo_quotes.csv",
    show_default=True,
    help="Row-level CSV output path.",
)
def main(league, markets, days, as_of, out):
    """Grade legacy combo means vs component-sum quotes over historical gamedays."""
    stat_data = _load_league(league)
    frames = []
    for market in (m.strip() for m in markets.split(",")):
        frame = _grade_market(stat_data, market, days, as_of == "serving")
        if frame.empty:
            click.echo(f"{league} {market}: no gradeable rows (no archived lines?)")
            continue
        frames.append(frame)
        click.echo(f"\n=== {league} {market} (n={len(frame)}) ===")
        click.echo("legacy (check_combo_markets mean + generic-cv tail):")
        click.echo("\n".join(_path_report(frame, "legacy_over")))
        click.echo("kernel (component-sum quote):")
        click.echo("\n".join(_path_report(frame, "kernel_over")))
        click.echo(_verdict(frame))
    if frames:
        pd.concat(frames).to_csv(out, index=False)
        click.echo(f"\nrow-level output -> {out}")


if __name__ == "__main__":
    main()
