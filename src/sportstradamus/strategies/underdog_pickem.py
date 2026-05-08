"""Underdog Pick'em entry-construction orchestrator (Phase 3 §3.3 + §3.5 Rivals).

Pure orchestrator: re-scores Underdog offers via
:func:`prediction.scoring.process_offers`, re-runs
:func:`prediction.correlation.find_correlation` per contest variant, then
applies the Pick'em-specific filters, Kelly sizing, ranking, and YAML emit.
Rank/dedupe and YAML emit live in :mod:`strategies._pickem_emit` to keep
this module under the CLAUDE.md 300-line ceiling.
"""

from __future__ import annotations

import datetime
import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from sportstradamus.helpers.logging import get_logger
from sportstradamus.strategies._pickem_emit import emit_yaml, rank_and_dedupe
from sportstradamus.strategies.kelly import (
    DEFAULT_KELLY_FRACTION,
    MAX_FRACTION_OF_BANKROLL,
    fractional_kelly_stake,
    resolve_shrinkage,
)

_logger = get_logger("pickem-build")

# Roadmap §3.5: Rivals payout structure caps the variant at 2-3 leg entries
# regardless of what the user sets in ``PickemConfig.entry_sizes``.
_RIVALS_LEG_SIZES: tuple[int, ...] = (2, 3)

_RECOMMENDATIONS_DIR = Path("data") / "recommendations"


@dataclass(frozen=True)
class PickemConfig:
    """Tunable thresholds for :func:`construct_entries` (§6.a)."""

    min_model_edge: float = 0.020
    min_sharp_edge: float = 0.015
    disagreement_threshold: float = 0.04
    min_correlation: float = 0.10
    min_ev: float = 0.05
    entry_sizes: tuple[int, ...] = (3, 5)
    contest_variants: tuple[str, ...] = ("power", "flex", "rivals")
    top_k: int = 20
    max_overlap: int = 2
    kelly_fraction: float = DEFAULT_KELLY_FRACTION
    max_stake_pct_bankroll: float = MAX_FRACTION_OF_BANKROLL


@dataclass(frozen=True)
class RecommendedEntry:
    """One ranked entry emitted to the recommendations YAML."""

    id: str
    contest_variant: str
    entry_size: int
    legs: tuple[str, ...]
    joint_prob: float
    payout_multiplier: float
    ev: float
    recommended_stake: Decimal
    shrinkage_source: str = "training"
    shrinkage: float = 1.0
    extras: dict[str, Any] = field(default_factory=dict)


def _filter_legs(offers: pd.DataFrame, config: PickemConfig) -> pd.DataFrame:
    """Apply leg-level filters: model & sharp coverage + edge + disagreement."""
    if offers.empty:
        return offers
    required = {"Model P", "Books P"}
    if not required.issubset(offers.columns):
        msg = f"_filter_legs needs columns {required}; got {set(offers.columns)}"
        raise ValueError(msg)

    df = offers.dropna(subset=["Model P", "Books P"]).copy()
    keep = (
        ((df["Model P"] - 0.5) >= config.min_model_edge)
        & ((df["Books P"] - 0.5) >= config.min_sharp_edge)
        & ((df["Model P"] - df["Books P"]).abs() <= config.disagreement_threshold)
    )
    return df.loc[keep].copy()


def _filter_parlays(
    parlay_df: pd.DataFrame, variant: str, entry_sizes: Iterable[int], config: PickemConfig
) -> pd.DataFrame:
    """Apply post-search size + EV filters; Rivals is forced to 2/3 legs."""
    if parlay_df.empty:
        return parlay_df
    sizes = set(_RIVALS_LEG_SIZES if variant == "rivals" else entry_sizes)
    df = parlay_df.copy()
    df = df[df["Bet Size"].isin(sizes)]
    df = df[(df["Model EV"] - 1.0) >= config.min_ev]
    return df.reset_index(drop=True)


def _validate_rivals_coverage(
    parlay_df: pd.DataFrame, offers: pd.DataFrame
) -> pd.DataFrame:
    """Drop Rivals candidates where one matchup side is missing from offers."""
    if parlay_df.empty or offers.empty:
        return parlay_df
    players = offers["Player"].astype(str).tolist() if "Player" in offers.columns else []
    keep_rows = []
    for _, row in parlay_df.iterrows():
        legs = [row.get(f"Leg {i}", "") for i in range(1, int(row["Bet Size"]) + 1)]
        ok = True
        for desc in legs:
            if "vs." not in str(desc):
                continue
            sides = [s.strip() for s in str(desc).split("vs.")[:2]]
            covered = sum(
                1 for side in sides
                if any(side and side.split()[0] in p for p in players)
            )
            if covered < 2:
                _logger.warning("rivals candidate dropped: one-sided (%s)", desc)
                ok = False
                break
        if ok:
            keep_rows.append(row)
    return pd.DataFrame(keep_rows).reset_index(drop=True) if keep_rows else parlay_df.iloc[0:0]


def _row_to_entry(
    row: pd.Series,
    variant: str,
    bankroll: Decimal,
    config: PickemConfig,
    shrinkage_info: tuple[float, str],
) -> RecommendedEntry:
    payout = float(row["Boost"])
    model_ev = float(row["Model EV"])
    joint_prob = model_ev / payout if payout > 0 else 0.0
    ev = model_ev - 1.0
    bet_size = int(row["Bet Size"])
    legs = tuple(str(row.get(f"Leg {i}", "")) for i in range(1, bet_size + 1))
    shrinkage, source = shrinkage_info

    stake = fractional_kelly_stake(
        bankroll=bankroll,
        win_prob=joint_prob,
        payout_multiplier=Decimal(repr(payout)),
        fraction=config.kelly_fraction,
        model_shrinkage=shrinkage,
        max_fraction_of_bankroll=config.max_stake_pct_bankroll,
    )

    digest = hashlib.sha1("|".join(legs).encode()).hexdigest()[:12]
    return RecommendedEntry(
        id=digest,
        contest_variant=variant,
        entry_size=bet_size,
        legs=legs,
        joint_prob=joint_prob,
        payout_multiplier=payout,
        ev=ev,
        recommended_stake=stake,
        shrinkage=shrinkage,
        shrinkage_source=source,
        extras={"league": str(row.get("League", "")), "game": str(row.get("Game", ""))},
    )


def _resolve_market_shrinkage(league: str, market: str) -> tuple[float, str]:
    """Resolve a Kelly shrinkage value plus a tag identifying its source."""
    try:
        from sportstradamus import clv as _clv
        from sportstradamus.training.report import get_market_calibration
    except ImportError:
        return 1.0, "fallback"

    try:
        live_bss, live_n = _clv.get_segment_calibration(league, market)
    except Exception:
        live_bss, live_n = float("nan"), 0
    try:
        train_metrics = get_market_calibration(league, market)
    except Exception:
        train_metrics = {"brier_skill_score": float("nan")}
    train_bss = train_metrics.get("brier_skill_score", float("nan"))

    shrinkage = resolve_shrinkage(training_bss=train_bss, live_bss=live_bss, live_n=live_n)
    has_train = not _isna(train_bss)
    has_live = not _isna(live_bss) and live_n > 0
    if not has_train and not has_live:
        return shrinkage, "fallback"
    if has_train and has_live:
        return shrinkage, "blended"
    return shrinkage, "training" if has_train else "clv_segment"


def _isna(x: Any) -> bool:
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return x is None


def construct_entries(
    date: datetime.date,
    bankroll: Decimal,
    config: PickemConfig | None = None,
    *,
    parlay_dfs: dict[str, pd.DataFrame] | None = None,
    offers_df: pd.DataFrame | None = None,
) -> list[RecommendedEntry]:
    """Build ranked, sized Underdog Pick'em entries for ``date``.

    ``parlay_dfs`` / ``offers_df`` injection short-circuits the live scraper
    so tests and offline reruns do not hit the network.
    """
    config = config or PickemConfig()
    if parlay_dfs is None:
        parlay_dfs, offers_df = _live_load(config)
    offers_df = pd.DataFrame() if offers_df is None else offers_df
    filtered_offers = _filter_legs(offers_df, config) if not offers_df.empty else offers_df

    entries: list[RecommendedEntry] = []
    for variant in config.contest_variants:
        parlays = parlay_dfs.get(variant, pd.DataFrame())
        if parlays.empty:
            continue
        parlays = _filter_parlays(parlays, variant, config.entry_sizes, config)
        if variant == "rivals":
            parlays = _validate_rivals_coverage(parlays, filtered_offers)
        for _, row in parlays.iterrows():
            league = str(row.get("League", ""))
            market = ""
            if not filtered_offers.empty and "Market" in filtered_offers.columns:
                market = str(filtered_offers["Market"].iloc[0])
            shrinkage_info = _resolve_market_shrinkage(league, market)
            entries.append(_row_to_entry(row, variant, bankroll, config, shrinkage_info))

    return rank_and_dedupe(entries, config)


def _live_load(config: PickemConfig) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Re-run the prophecize loader and search per variant. Heavy."""
    from sportstradamus.books import get_ud
    from sportstradamus.prediction.correlation import find_correlation
    from sportstradamus.prediction.scoring import process_offers
    from sportstradamus.stats import StatsNBA, StatsNFL, StatsWNBA

    stats: dict[str, Any] = {}
    today = datetime.date.today()
    for cls, key in ((StatsNBA, "NBA"), (StatsNFL, "NFL"), (StatsWNBA, "WNBA")):
        s = cls()
        s.load()
        if today > (s.season_start - datetime.timedelta(days=7)):
            s.update()
            stats[key] = s
    offers_df, _ = process_offers(
        get_ud(), "Underdog", stats, contest_variant=config.contest_variants[0]
    )
    scored = offers_df.to_dict("records") if not offers_df.empty else []
    parlay_dfs = {
        v: find_correlation(scored, stats, "Underdog", contest_variant=v)[1]
        for v in config.contest_variants
    }
    return parlay_dfs, offers_df


def _build_cli() -> Any:
    import click

    @click.command()
    @click.option("--date", default="today", help="Slate date (YYYY-MM-DD or 'today').")
    @click.option("--bankroll", type=str, required=True, help="Bankroll in dollars.")
    @click.option("--out", "out_path", type=click.Path(dir_okay=False), default=None,
                  help="Override output YAML path.")
    def pickem_build(date: str, bankroll: str, out_path: str | None) -> None:
        """Build today's Underdog Pick'em entries and emit recommendations YAML."""
        slate_date = (
            datetime.date.today() if date == "today" else datetime.date.fromisoformat(date)
        )
        config = PickemConfig()
        entries = construct_entries(slate_date, Decimal(bankroll), config)
        target = Path(out_path) if out_path else _RECOMMENDATIONS_DIR / f"{slate_date.isoformat()}.yaml"
        path = emit_yaml(entries, slate_date, Decimal(bankroll), config, target)
        click.echo(f"wrote {len(entries)} entries -> {path}")

    return pickem_build


def main() -> None:
    _build_cli()()


if __name__ == "__main__":
    main()
