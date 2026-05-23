#!/usr/bin/env python3
"""Year-over-year Pearson stability diagnostic for NFL comp features.

Phase 1G gate before ``optimize_comp_weights.py --save``. For every feature
listed in ``playerCompStats.json[NFL][position]``, we compute within-player
Pearson r across the (2023->2024) and (2024->2025) season-pair pool and
emit a per-(feature, position) decision row to ``data/comp_feature_yoy.csv``.

Decision rules (per plan §1G / research brief §4):
- ``r >= 0.55``                     -> ``keep``
- ``0.40 <= r < 0.55``              -> ``transform_3yr_rolling``
- ``r < 0.40``                      -> ``drop`` (unless archetypal -> keep)
- ``n_pairs < MIN_PAIRS`` or NaN r  -> ``unvalidated``

The "archetypal" override pins a small set of alignment / snap-share / block
columns that the literature treats as archetype differentiators even when
they swing year over year with scheme changes; see plan §1G and research
brief §4 for the justification.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from sportstradamus import data
from sportstradamus.stats.nfl_fp_loader import apply_rolling_transforms, load_one_year

# Seasons participating in the diagnostic. FP data starts in 2023 so two
# consecutive-pair transitions ((2023, 2024), (2024, 2025)) are all we get.
YEARS: tuple[int, ...] = (2023, 2024, 2025)
YEAR_PAIRS: tuple[tuple[int, int], ...] = ((2023, 2024), (2024, 2025))

# Per research-brief §4: filter to seasons where the player accrued enough
# games for the per-feature aggregate to mean anything. Eight games is the
# standard half-season threshold used in NFL stickiness studies.
MIN_GAMES_PER_SEASON: int = 8

# Minimum (prev, next) pair count for an r to be considered statistically
# usable. Below this the (feature, position) row is reported as unvalidated
# rather than letting noise drive a drop.
MIN_PAIRS: int = 30

# Threshold above which a feature is sticky enough to use as-is.
R_KEEP_THRESHOLD: float = 0.55

# Threshold above which a feature is borderline-sticky and can be salvaged
# by averaging over a 3-year window (variance reduction ~sqrt(3)).
R_TRANSFORM_THRESHOLD: float = 0.40

# Positions the comp profile splits on. FB/HB are remapped to RB inside the
# loader so the per-position rows match the playerCompStats.json blocks.
COMP_POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE")

# Loader-emitted canonical names. ``load_one_year`` renames the raw FP
# ``G`` / ``POS`` headers and collapses HB/FB before returning, so the
# diagnostic now keys on the same column names ``build_comp_profile``
# consumes downstream.
GAMES_COLUMN: str = "player_game_count"
POSITION_COLUMN: str = "position"

# Archetypal features: scheme-dependent so Y/Y r is low, but the literature
# treats alignment / snap / block proxies as archetype differentiators that
# stay informative for KNN comp neighborhoods. Override drop -> keep_archetypal
# for any (feature, position) that matches one of these names.
ARCHETYPAL_FEATURES: frozenset[str] = frozenset(
    {
        "rec_routes_WIDE_RTE_pct",
        "rec_routes_SLOT_RTE_pct",
        "rec_routes_INLINE_RTE_pct",
        "rec_routes_BACK_RTE_pct",
        "off_snaps_Snap_pct",
        "off_snaps_overall_Snap_pct",
        "block_pct_proxy",
    }
)

# Metadata-shaped features that are definitionally Y/Y stable (age has
# perfect r per season, height/bmi are constant) so the Pearson check is
# uninformative. Tag them ``keep_metadata`` rather than ``keep`` so the
# decision table is honest about why they survived the gate.
_TRIVIAL_FEATURES: frozenset[str] = frozenset({"age", "height", "bmi"})

# (feature, position) pairs that get a manual KEEP override even if
# ``n_pairs < MIN_PAIRS``. RBs often have too few games of receiving sample
# to satisfy the test, but the features are literature-supported and we
# want them in the RB comp profile.
RB_RECEIVING_KEEP_OVERRIDE: frozenset[tuple[str, str]] = frozenset(
    {
        ("rec_adv_YPRR", "RB"),
        ("rec_adv_TPRR", "RB"),
        ("rec_adv_MTF_REC", "RB"),
        ("rec_adv_TGT_pct", "RB"),
    }
)

# Default output path. Lives at the repo data/ root rather than under
# src/sportstradamus/data/ -- the CSV is a one-shot diagnostic artifact, not
# a packaged data file (per plan §1G "persist the result, don't re-run from
# CLI; re-run only when the FP file inventory changes").
DEFAULT_OUTPUT_PATH: str = "data/comp_feature_yoy.csv"

# CSV column order for the decision table.
OUTPUT_COLUMNS: tuple[str, ...] = ("feature", "position", "r", "n_pairs", "decision")


def load_candidate_features() -> dict[str, list[str]]:
    """Return ``{position: [feature, ...]}`` from ``playerCompStats.json[NFL]``."""
    config_path = pkg_resources.files(data) / "config/playerCompStats.json"
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    nfl_cfg = cfg.get("NFL", {})
    return {pos: list(nfl_cfg.get(pos, {}).keys()) for pos in COMP_POSITIONS}


def load_stacked_seasons(min_games: int) -> pd.DataFrame:
    """Stack FP loader output across YEARS, filter on min games.

    Returns a DataFrame whose row index is the accent-stripped player name
    (the loader's natural key) plus ``season`` and ``position`` columns.
    ``load_one_year`` already collapses HB/FB to RB and renames the FP
    headers, so per-position groupbys match the comp profile taxonomy
    without further normalization.
    """
    frames: list[pd.DataFrame] = []
    for year in YEARS:
        per_year = load_one_year(year)
        if per_year.empty:
            click.echo(f"[warn] loader returned empty frame for {year}", err=True)
            continue
        per_year = per_year.copy()
        per_year["season"] = year
        frames.append(per_year)

    if not frames:
        raise RuntimeError("no FP seasons loaded; check data/player_data/NFL/")

    stacked = pd.concat(frames, copy=False)

    if GAMES_COLUMN not in stacked.columns:
        raise RuntimeError(f"loader output missing {GAMES_COLUMN!r} column")
    if POSITION_COLUMN not in stacked.columns:
        raise RuntimeError(f"loader output missing {POSITION_COLUMN!r} column")

    # Apply the 3-year rolling smooth before the min-games filter so the
    # Phase 1G transform features get the same smoothing here as they will
    # in production ``build_comp_profile`` (which sees every player-season
    # the loader emits, not just those above the games threshold).
    stacked = apply_rolling_transforms(stacked)

    # Min-games filter mirrors the half-season cutoff used in stickiness lit.
    stacked = stacked.loc[stacked[GAMES_COLUMN] >= min_games].copy()

    return stacked


def collect_pairs(
    stacked: pd.DataFrame, feature: str, position: str | None = None
) -> pd.DataFrame:
    """Build a (prev, next) DataFrame of within-player Y/Y values for one feature.

    When ``position`` is given, rows are restricted to players whose
    ``position`` column matches; with ``position=None`` the diagnostic
    pools across positions (used for the per-feature roll-up).
    """
    if feature not in stacked.columns:
        return pd.DataFrame(columns=["prev", "next"])

    frame = stacked
    if position is not None:
        frame = frame.loc[frame["position"] == position]
    if frame.empty:
        return pd.DataFrame(columns=["prev", "next"])

    cols = [feature, "season", "position"]
    frame = frame[cols].copy()
    frame["player"] = frame.index

    pairs: list[tuple[float, float]] = []
    grouped = frame.groupby(["player", "position"], sort=False)
    for (_player, _pos), grp in grouped:
        season_to_value = grp.set_index("season")[feature]
        for y_prev, y_next in YEAR_PAIRS:
            if y_prev in season_to_value.index and y_next in season_to_value.index:
                pairs.append((season_to_value.loc[y_prev], season_to_value.loc[y_next]))

    pair_df = pd.DataFrame(pairs, columns=["prev", "next"])
    return pair_df.dropna()


def compute_pearson(pair_df: pd.DataFrame) -> float:
    """Return Pearson r of (prev, next) pairs, or NaN if undefined."""
    if len(pair_df) < 2:
        return float("nan")
    if pair_df["prev"].std() == 0 or pair_df["next"].std() == 0:
        return float("nan")
    return float(pair_df["prev"].corr(pair_df["next"]))


def decide(
    feature: str, position: str, r: float, n_pairs: int, min_pairs: int
) -> str:
    """Map (feature, position, r, n_pairs) onto a decision label per plan §1G thresholds."""
    if feature in _TRIVIAL_FEATURES:
        return "keep_metadata"
    # Manual override pinned before the n_pairs/NaN gate so RB receiving
    # features keep their slot even when the per-position sample is too thin
    # for a reliable Pearson r.
    if (feature, position) in RB_RECEIVING_KEEP_OVERRIDE:
        return "keep_override"
    if np.isnan(r) or n_pairs < min_pairs:
        return "unvalidated"
    if r >= R_KEEP_THRESHOLD:
        return "keep"
    if r >= R_TRANSFORM_THRESHOLD:
        return "transform_3yr_rolling"
    if feature in ARCHETYPAL_FEATURES:
        return "keep_archetypal"
    return "drop"


def build_results_table(
    stacked: pd.DataFrame,
    features_by_position: dict[str, list[str]],
    min_pairs: int,
) -> pd.DataFrame:
    """Compute one row per (feature, position) the config opts into."""
    rows: list[dict[str, object]] = []
    missing: set[str] = set()

    feature_position_pairs: list[tuple[str, str]] = []
    for position, feats in features_by_position.items():
        for feature in feats:
            feature_position_pairs.append((feature, position))

    for feature, position in tqdm(
        feature_position_pairs, desc="Y/Y r per (feature, position)", unit="cell"
    ):
        if feature not in stacked.columns:
            missing.add(feature)
            rows.append(
                {
                    "feature": feature,
                    "position": position,
                    "r": float("nan"),
                    "n_pairs": 0,
                    "decision": decide(feature, position, float("nan"), 0, min_pairs),
                }
            )
            continue

        pair_df = collect_pairs(stacked, feature, position=position)
        n_pairs = len(pair_df)
        r = compute_pearson(pair_df)
        rows.append(
            {
                "feature": feature,
                "position": position,
                "r": r,
                "n_pairs": n_pairs,
                "decision": decide(feature, position, r, n_pairs, min_pairs),
            }
        )

    if missing:
        click.echo(
            f"[warn] {len(missing)} feature(s) not found in loader output: "
            f"{sorted(missing)}",
            err=True,
        )

    return pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS))


def format_per_position_table(results: pd.DataFrame, position: str) -> str:
    """Render the (feature, r, decision) breakdown for one position as text."""
    sub = results.loc[results["position"] == position].copy()
    if sub.empty:
        return f"\n=== {position}: no features in playerCompStats.json[NFL][{position}] ===\n"

    sub = sub.sort_values(
        by=["decision", "r"],
        ascending=[True, False],
        kind="stable",
        na_position="last",
    )

    lines = [f"\n=== {position} ({len(sub)} features) ==="]
    lines.append(f"  {'feature':<45} {'r':>8} {'n':>6}  decision")
    lines.append(f"  {'-' * 45} {'-' * 8} {'-' * 6}  {'-' * 22}")
    for _, row in sub.iterrows():
        r_str = "   nan" if pd.isna(row["r"]) else f"{row['r']:>8.4f}"
        lines.append(
            f"  {row['feature']:<45} {r_str} {row['n_pairs']:>6}  {row['decision']}"
        )
    return "\n".join(lines)


def format_tally(results: pd.DataFrame) -> str:
    """Render the per-position decision tally + grand total."""
    pivot = (
        results.assign(_one=1)
        .pivot_table(
            index="position",
            columns="decision",
            values="_one",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(index=list(COMP_POSITIONS))
        .fillna(0)
        .astype(int)
    )

    # Show columns in policy order even when one didn't fire this run.
    column_order = [
        "keep",
        "keep_metadata",
        "keep_override",
        "transform_3yr_rolling",
        "keep_archetypal",
        "drop",
        "unvalidated",
    ]
    for col in column_order:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[column_order]
    pivot["total"] = pivot.sum(axis=1)
    pivot.loc["GRAND"] = pivot.sum(axis=0)

    return "\n=== Decision tally ===\n" + pivot.to_string()


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--output",
    default=DEFAULT_OUTPUT_PATH,
    show_default=True,
    help="Output CSV path for the per-(feature, position) decision table.",
)
@click.option(
    "--min-games-per-season",
    default=MIN_GAMES_PER_SEASON,
    show_default=True,
    type=int,
    help="Min FP-CSV games column to include a player-season in the pool.",
)
@click.option(
    "--min-pairs",
    default=MIN_PAIRS,
    show_default=True,
    type=int,
    help="Min (prev, next) pair count before a row escapes the unvalidated bucket.",
)
def main(output: str, min_games_per_season: int, min_pairs: int) -> None:
    """Phase 1G: Y/Y stability diagnostic for NFL comp candidate features."""
    features_by_position = load_candidate_features()
    n_features_total = sum(len(v) for v in features_by_position.values())
    click.echo(
        f"Loaded {n_features_total} (feature, position) candidates across "
        f"{len(COMP_POSITIONS)} positions."
    )

    stacked = load_stacked_seasons(min_games=min_games_per_season)
    click.echo(
        f"Stacked frame: {len(stacked):,} player-season rows after "
        f"{GAMES_COLUMN} >= {min_games_per_season} filter."
    )

    results = build_results_table(stacked, features_by_position, min_pairs=min_pairs)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False, float_format="%.4f")
    click.echo(f"\nWrote {len(results)} rows to {output_path}.")

    for position in COMP_POSITIONS:
        click.echo(format_per_position_table(results, position))

    click.echo(format_tally(results))


if __name__ == "__main__":
    main()
