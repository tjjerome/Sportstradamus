"""Pick'em orchestrator for Underdog and Sleeper (Phase 3 §3.3 + §3.5 Rivals).

Pure orchestrator over ``prediction.scoring.process_offers`` and
``prediction.correlation.find_correlation`` (one search per contest
variant). Rank/dedupe and YAML emit live in :mod:`._pickem_emit` to keep
this file under the CLAUDE.md 300-line ceiling.
"""

from __future__ import annotations

import datetime
import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import click
import pandas as pd

from sportstradamus.helpers import odds_budget, stat_map
from sportstradamus.helpers.logging import get_logger
from sportstradamus.leg_schema import leg_label
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

# prophecize's hourly snapshot sizes stakes at this nominal bankroll purely to
# drive rank_and_dedupe ordering (bankroll-invariant below the per-bet cap); the
# dashboard re-sizes every entry against the user's actual bankroll.
REFERENCE_BANKROLL = Decimal("1000")

# Sleeper has no Rivals/H2H game mode; callers building a Sleeper PickemConfig
# override contest_variants with this instead of the Underdog-shaped default.
PLATFORM_CONTEST_VARIANTS: dict[str, tuple[str, ...]] = {
    "Underdog": ("power", "flex", "rivals"),
    "Sleeper": ("power", "flex"),
}


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
    canonical_legs: tuple[dict, ...] = ()
    shrinkage_source: str = "training"
    shrinkage: float = 1.0
    extras: dict[str, Any] = field(default_factory=dict)
    platform: str = "Underdog"


def filter_legs(offers: pd.DataFrame, config: PickemConfig) -> pd.DataFrame:
    """Apply leg-level filters: model & sharp coverage + edge + disagreement."""
    if offers.empty:
        return offers
    required = {"Win Prob", "Market Prob"}
    if not required.issubset(offers.columns):
        msg = f"filter_legs needs columns {required}; got {set(offers.columns)}"
        raise ValueError(msg)

    df = offers.dropna(subset=["Win Prob", "Market Prob"]).copy()
    keep = (
        ((df["Win Prob"] - 0.5) >= config.min_model_edge)
        & ((df["Market Prob"] - 0.5) >= config.min_sharp_edge)
        & ((df["Win Prob"] - df["Market Prob"]).abs() <= config.disagreement_threshold)
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


def _rivals_row_covered(row: pd.Series, players: list[str]) -> bool:
    for leg in row["legs"]:
        desc = str(leg["player"])
        if "vs." not in desc:
            continue
        sides = [s.strip() for s in desc.split("vs.")[:2]]
        covered = sum(1 for side in sides if any(side and side.split()[0] in p for p in players))
        if covered < 2:
            _logger.warning("rivals candidate dropped: one-sided (%s)", desc)
            return False
    return True


def _validate_rivals_coverage(parlay_df: pd.DataFrame, offers: pd.DataFrame) -> pd.DataFrame:
    if parlay_df.empty or offers.empty:
        return parlay_df
    players = offers["Player"].astype(str).tolist() if "Player" in offers.columns else []
    keep_rows = [row for _, row in parlay_df.iterrows() if _rivals_row_covered(row, players)]
    return pd.DataFrame(keep_rows).reset_index(drop=True) if keep_rows else parlay_df.iloc[0:0]


def _row_to_entry(
    row: pd.Series,
    variant: str,
    bankroll: Decimal,
    config: PickemConfig,
    shrinkage_info: tuple[float, str],
    platform: str = "Underdog",
) -> RecommendedEntry:
    payout = float(row["Boost"])
    model_ev = float(row["Model EV"])
    joint_prob = model_ev / payout if payout > 0 else 0.0
    ev = model_ev - 1.0
    bet_size = int(row["Bet Size"])
    canonical_legs = tuple(row["legs"])
    legs = tuple(leg_label(leg) for leg in canonical_legs)
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
        canonical_legs=canonical_legs,
        shrinkage=shrinkage,
        shrinkage_source=source,
        extras={"league": str(row.get("League", "")), "game": str(row.get("Game", ""))},
        platform=platform,
    )


def resolve_market_shrinkage(league: str, market: str) -> tuple[float, str]:
    """Resolve Kelly shrinkage for one ``(league, market)`` cell.

    Blends live CLV-segment calibration with offline training calibration per
    ``kelly.resolve_shrinkage``'s explicit>blended>training>clv_segment>fallback
    chain; the returned source label tells a caller which rung of that chain
    fired for this cell.

    Returns:
        ``(shrinkage, source)`` where ``source`` is one of ``"blended"``,
        ``"training"``, ``"clv_segment"``, or ``"fallback"``.
    """
    # Lazy imports keep torch (via training.report) out of the dashboard import
    # path — do NOT hoist clv or get_market_calibration to module level.
    from sportstradamus import clv as _clv
    from sportstradamus.training.report import get_market_calibration

    live_bss, live_n = _clv.get_segment_calibration(league, market)
    train_bss = get_market_calibration(league, market)["brier_skill_score"]

    shrinkage = resolve_shrinkage(training_bss=train_bss, live_bss=live_bss, live_n=live_n)
    has_train = not pd.isna(train_bss)
    has_live = not pd.isna(live_bss) and live_n > 0
    if not has_train and not has_live:
        return shrinkage, "fallback"
    if has_train and has_live:
        return shrinkage, "blended"
    return shrinkage, "training" if has_train else "clv_segment"


def construct_entries(
    date: datetime.date,
    bankroll: Decimal,
    config: PickemConfig | None = None,
    *,
    parlay_dfs: dict[str, pd.DataFrame] | None = None,
    offers_df: pd.DataFrame | None = None,
    platform: str = "Underdog",
) -> list[RecommendedEntry]:
    """Build ranked, sized Pick'em entries for ``date`` on ``platform``.

    ``parlay_dfs`` / ``offers_df`` injection short-circuits the live scraper
    so tests and offline reruns do not hit the network.
    """
    config = config or PickemConfig()
    if parlay_dfs is None:
        parlay_dfs, offers_df = live_load(config, platform)
    offers_df = pd.DataFrame() if offers_df is None else offers_df
    filtered_offers = filter_legs(offers_df, config) if not offers_df.empty else offers_df

    entries: list[RecommendedEntry] = []
    for variant in config.contest_variants:
        parlays = parlay_dfs.get(variant, pd.DataFrame())
        if parlays.empty:
            continue
        parlays = _filter_parlays(parlays, variant, config.entry_sizes, config)
        if variant == "rivals":
            parlays = _validate_rivals_coverage(parlays, filtered_offers)
        entries.extend(_variant_entries(parlays, variant, bankroll, config, platform))

    return rank_and_dedupe(entries, config)


def _variant_entries(
    parlays: pd.DataFrame,
    variant: str,
    bankroll: Decimal,
    config: PickemConfig,
    platform: str = "Underdog",
) -> list[RecommendedEntry]:
    entries: list[RecommendedEntry] = []
    for _, row in parlays.iterrows():
        league = str(row.get("League", ""))
        shrinkage_info = _parlay_shrinkage(row, league, platform)
        entries.append(_row_to_entry(row, variant, bankroll, config, shrinkage_info, platform))
    return entries


def _parlay_shrinkage(row: pd.Series, league: str, platform: str) -> tuple[float, str]:
    """Kelly shrinkage for one parlay: the most conservative of its leg markets.

    A parlay cashes only if every leg hits, so it inherits the trust of its
    least-calibrated market — resolve per distinct leg market and keep the min.
    Leg ``market`` is the raw platform name (a ``stat_map[platform]`` key);
    mapping it gives the canonical cell key (``REB``, ``BLST``, …) that
    ``resolve_market_shrinkage`` resolves against. Combo / H2H names with no
    mapping are skipped; a parlay with none left falls back to full trust.
    """
    markets = {stat_map[platform].get(leg["market"]) for leg in row["legs"]} - {None}
    if not markets:
        return 1.0, "fallback"
    return min((resolve_market_shrinkage(league, m) for m in markets), key=lambda r: r[0])


def leg_player(leg: str) -> str:
    """Player name from a leg label ``"{player} Over|Under {line} {market}"``."""
    for token in (" Over ", " Under "):
        if token in leg:
            return leg.split(token, 1)[0].strip()
    return ""


def _parlays_per_variant(
    scored_offers_df: pd.DataFrame,
    stats: dict[str, Any],
    config: PickemConfig,
    platform: str = "Underdog",
) -> dict[str, pd.DataFrame]:
    """Run ``find_correlation`` once per contest variant on already-scored offers.

    The expensive scrape + model scoring happened upstream; this only enumerates
    correlated parlays per payout variant, which touches no archive. Candidates
    come from the ``filter_legs`` pool — the same edge/disagreement gates as the
    leg universe — so garbage-probability legs never seed a parlay. An empty
    pool short-circuits: ``find_correlation`` KeyErrors on a rowless frame.
    """
    from sportstradamus.prediction.correlation import find_correlation

    pool = filter_legs(scored_offers_df, config)
    if pool.empty:
        return {v: pd.DataFrame() for v in config.contest_variants}
    scored = pool.to_dict("records")
    return {
        v: find_correlation(scored, stats, platform, contest_variant=v)[1]
        for v in config.contest_variants
    }


def build_entries_from_scored(
    date: datetime.date,
    bankroll: Decimal,
    scored_offers_df: pd.DataFrame,
    stats: dict[str, Any],
    config: PickemConfig | None = None,
    *,
    platform: str = "Underdog",
) -> list[RecommendedEntry]:
    """Build ranked entries from offers already scored by ``process_offers``.

    Skips the scrape and model scoring — the caller supplies ``scored_offers_df``
    — and runs only the per-variant correlation search before ranking and
    sizing. The ``prophecize`` snapshot hook uses this so the hourly run feeds
    the dashboard without a second scrape or archive lock.
    """
    config = config or PickemConfig()
    parlay_dfs = _parlays_per_variant(scored_offers_df, stats, config, platform)
    return construct_entries(
        date,
        bankroll,
        config,
        parlay_dfs=parlay_dfs,
        offers_df=scored_offers_df,
        platform=platform,
    )


def live_load(
    config: PickemConfig, platform: str = "Underdog"
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Re-run the prophecize loader and search per variant. Heavy."""
    from sportstradamus.books import get_sleeper, get_ud
    from sportstradamus.prediction.scoring import process_offers
    from sportstradamus.stats import StatsMLB, StatsNBA, StatsNFL, StatsNHL, StatsWNBA

    loaders = {"Underdog": get_ud, "Sleeper": get_sleeper}

    stats: dict[str, Any] = {}
    for cls, key in (
        (StatsNBA, "NBA"),
        (StatsNFL, "NFL"),
        (StatsWNBA, "WNBA"),
        (StatsMLB, "MLB"),
        (StatsNHL, "NHL"),
    ):
        s = cls()
        s.load()
        if odds_budget.league_is_live(key, s.season_start):
            s.update()
            stats[key] = s
    offers_df, _ = process_offers(
        loaders[platform](), platform, stats, contest_variant=config.contest_variants[0]
    )
    # process_offers doesn't stamp Platform — prophecize adds it to its own copies
    # (prediction/cli.py), and build_leg/parlay pricing require the column here too.
    offers_df["Platform"] = platform
    return _parlays_per_variant(offers_df, stats, config, platform), offers_df


@click.command()
@click.option("--date", default="today", help="Slate date (YYYY-MM-DD or 'today').")
@click.option("--bankroll", type=str, required=True, help="Bankroll in dollars.")
@click.option(
    "--platform",
    type=click.Choice(["Underdog", "Sleeper"], case_sensitive=False),
    default="Underdog",
    show_default=True,
    help="DFS platform to build Pick'em entries for.",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False),
    default=None,
    help="Override output YAML path.",
)
def pickem_build(date: str, bankroll: str, platform: str, out_path: str | None) -> None:
    """Build today's Pick'em entries for ``platform`` and emit recommendations YAML."""
    slate_date = datetime.date.today() if date == "today" else datetime.date.fromisoformat(date)
    config = PickemConfig(contest_variants=PLATFORM_CONTEST_VARIANTS[platform])
    entries = construct_entries(slate_date, Decimal(bankroll), config, platform=platform)
    target = (
        Path(out_path)
        if out_path
        else _RECOMMENDATIONS_DIR / f"{slate_date.isoformat()}_{platform.lower()}.yaml"
    )
    path = emit_yaml(entries, slate_date, Decimal(bankroll), config, target)
    click.echo(f"wrote {len(entries)} entries -> {path}")


if __name__ == "__main__":
    pickem_build()
