"""Tail scorecard: the live recommendation rule replayed on held-out test rows at DFS rungs.

g1 scores the population where the model agrees with the market, but the legs the live
rule recommends are the tail where it does not, and they read well above their hit rate.
``tail_pricing`` prices every held-out test row at each Underdog and Sleeper rung the
archive's ladder held for it; this command hands each slate to the live ``finalize_records``
the way ``model_prob`` does, grades the survivors with ``realized.settled_offers`` and
reports the selected-tail gap (read - hit) per cell, per league and overall with
day-clustered CIs, beside the live gap over the same window split by model-version age, and
the information test (``tail_information``). Spec:
``docs/archive/researcher_train_serve_skew.md`` section 6.

A training-time diagnostic and acceptance input for the NFL lane (I6d), tail
recalibration (I6c, any rank-2 calibration change) and the version-age test (I6g); never a
ship gate, and nothing here demotes or withholds a model. Every input is read-only, the
archive opened ``read_only``; the one write is the sandbox CSV at ``--out``.

Known optimism: test rows carry training-matrix features, and three measured skews
(serve-path parity, in-game quotes in the enriched game lines, MLB's retrain-day comp
snapshot) make them kinder than serving, so read the selected gap as a lower bound on live.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import logging
import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from tqdm import tqdm

from sportstradamus import data
from sportstradamus.helpers import UNDERDOG_BOOST_BASELINE, platform_payout
from sportstradamus.helpers.io import read_history
from sportstradamus.prediction.offer_records import (
    _MAX_CONFIDENCE,
    SCORED_RECORD_COLS,
    book_over_prob,
    finalize_records,
)
from sportstradamus.realized import cohort_summary, settled_offers
from sportstradamus.scripts.tail_information import DAY_RESAMPLES, information_table
from sportstradamus.scripts.tail_pricing import (
    SCORED_PLATFORMS,
    CellExcludedError,
    open_archive,
    price_cell,
)
from sportstradamus.spiderLogger import logger
from sportstradamus.training.scorecard import _bootstrap_mean_ci_clustered

TEST_SETS_DIR = Path(str(pkg_resources.files(data) / "test_sets"))
_DEFAULT_ARCHIVE = os.environ.get("SPORTSTRADAMUS_ARCHIVE_DB", "archive/archive.duckdb")
_DEFAULT_OUT = Path("/tmp/tail_scorecard.csv")

# Posted payout x stored price: 1 on a one-sided rung, at most ~0.89 on a two-sided one
# (a 12%+ hold devigged away).
_ONE_SIDED_TOL = 0.02

# Below this many selected legs a gap is descriptive only (spec section 6, Power).
DESCRIPTIVE_MIN_SELECTED = 100

# Spec section 6 (reproduction line): a version's first four served days against the rest.
FRESH_VERSION_DAYS = 3

_BOOTSTRAP_SEED = 0  # fixed so a rerun on the same inputs prints the same CIs

# |Line - served mean| in served SDs (spec section 6 splits).
_Z_BANDS = [0, 0.25, 0.5, 0.75, 1.0, 1.5, np.inf]

# finalize_records projects these into its export schema without reading them.
_PASSENGERS = dict.fromkeys(
    [
        "Team",
        "Opponent",
        "Push Prob",
        "Quote Source",
        "Quote Authenticity",
        "Quote Books",
        "Quote Line",
        "Quote Observed At",
    ]
)

_RUNG_KEY = ["League", "Date", "Player", "Market", "Line", "Platform"]
_SCOPE_COLUMNS = {"league": "League", "cell": "cell"}
_HEADLINE_COLUMNS = [
    "scope",
    "key",
    "test_rows",
    "rung_rows",
    "n",
    "read",
    "hit",
    "market",
    "gap",
    "gap_lo",
    "gap_hi",
    "bulk_gap",
    "training_gap",
    "posted_share",
    "live_n",
    "live_gap",
    "live_gap_lo",
    "live_gap_hi",
    "descriptive",
]


def side_boosts(rungs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each rung's raw boost per side, in the convention ``finalize_records`` takes them.

    ``Live Bet`` / ``Live Boost`` are history's chosen side at the rung and its raw boost (0
    when never posted), NaN where history never scored the rung. The ladder stores the
    devigged price, or a one-sided rung's raw breakeven (``dfs_boost_probs``), so posted
    payout x stored price reads 1 on a one-sided rung, where the side opposite an unposted
    pick pays its raw breakeven and the side opposite a posted pick was never offered, and
    1/overround on a two-sided rung; that recovers which sides a rung posted. A side history
    never priced pays 1 / (price x overround), and ``main`` flags it ``assumed``.

    Returns:
        The rungs with ``Boost_Over``, ``Boost_Under`` and ``One Sided``, and the overround
        (median over each platform's two-sided rungs off Underdog's standard rung, which
        pays the baseline both ways, 2 / baseline) and its rung count per ``Platform``.
    """
    per_boost = platform_payout(1.0, rungs["Platform"])  # full payout per unit of raw boost
    live_payout = rungs["Live Boost"] * per_boost
    price = {"Over": rungs["p_dfs"], "Under": 1 - rungs["p_dfs"]}
    held = live_payout * price["Over"].where(rungs["Live Bet"].eq("Over"), price["Under"])
    one_sided = live_payout.eq(0) | (held - 1).abs().lt(_ONE_SIDED_TOL)
    # Underdog's standard rung pays the baseline both ways, which devigs to exactly 0.5.
    standard = rungs["Platform"].eq("Underdog") & np.isclose(rungs["p_dfs"], 0.5)
    measured = rungs["Live Bet"].notna() & ~one_sided & ~standard
    platform = rungs["Platform"][measured]
    overround = (1 / held[measured]).groupby(platform).agg(["median", "size"])
    rung_overround = np.where(
        standard,
        2 / UNDERDOG_BOOST_BASELINE,
        overround["median"].reindex(rungs["Platform"]).to_numpy(),
    )
    boosts = {
        f"Boost_{side}": np.select(
            [rungs["Live Bet"].eq(side), one_sided],
            [rungs["Live Boost"], np.where(live_payout.eq(0), 1 / (p * per_boost), 0.0)],
            1 / (p * rung_overround * per_boost),
        )
        for side, p in price.items()
    }
    return rungs.assign(**boosts, **{"One Sided": one_sided}), overround


def replay_live_rule(rungs: pd.DataFrame, p_over: pd.Series, knobs: dict) -> pd.DataFrame:
    """The live recommendation rule on one cell's rungs, graded the way ``realized`` grades.

    Each (platform, date) slate goes to ``offer_records.finalize_records`` as ``model_prob``
    builds a board, so the 0.90 clip, argmax side, unquoted-disagreement gate, boost cap and
    three-per-player trim are the live code; ``realized.settled_offers`` then grades the
    survivors (pushes void, unposted sides dropped) and flags ``Recommended``.

    Args:
        rungs: One cell's rungs: ``League``, ``Market``, ``Platform``, ``Date``, ``Player``,
            ``Line``, raw ``Boost_Over`` / ``Boost_Under``, ``Market Projection`` (NaN when
            unquoted), ``Projection`` and the realized ``Actual``.
        p_over: The served over-probability per rung.
        knobs: The cell's pickle knobs.

    Returns:
        The surviving rungs with the live ``Bet``, ``Win Prob`` and ``Market Prob``, plus
        ``settled_offers``' ``Payout``, ``Hit`` and ``Recommended``.
    """
    league, market = rungs["League"].iat[0], rungs["Market"].iat[0]
    dist, cv, step = knobs["distribution"], knobs["cv"], knobs["step"]
    board = rungs.assign(
        **{"Model Under": 1 - p_over, "Model Weight": knobs["weight"]}, **_PASSENGERS
    )
    board["Model Over"] = 1 - board["Model Under"]
    # Serving decodes the book leg with the cell's zero rate on the zero-inflated family only.
    zero_rate = knobs["hist_gate"] if dist == "ZINB" else None
    board["Market EV"] = book_over_prob(board, dist, cv, step, zero_rate, league, market)
    kept = []
    for (platform, _), slate in board.groupby(["Platform", "Date"]):
        records = finalize_records(
            slate.copy(),
            league,
            platform,
            dist,
            cv,
            step,
            knobs["temperature"],
            knobs["dispersion_cal"],
            knobs["model_version"],
            knobs["pit_recal_blob"],
        )
        live_picks = pd.DataFrame(records, columns=SCORED_RECORD_COLS)
        # Typed, so a slate that keeps nothing concatenates like one that does.
        live_picks = live_picks[["Player", "Line", "Bet", "Win Prob", "Market Prob"]].astype(
            {"Line": float, "Win Prob": float, "Market Prob": float}
        )
        kept.append(slate.merge(live_picks, on=["Player", "Line"]))
    kept = pd.concat(kept, ignore_index=True)
    return settled_offers(
        kept.assign(
            Boost=kept["Boost_Over"].where(kept["Bet"].eq("Over"), kept["Boost_Under"]),
            Result=np.select(
                [kept["Actual"] > kept["Line"], kept["Actual"] < kept["Line"]],
                ["Over", "Under"],
                "Push",
            ),
            **{"Model Version": knobs["model_version"]},
        )
    )


def _tail(selected: pd.DataFrame, rng: np.random.Generator) -> dict[str, float]:
    stats = cohort_summary(selected)
    over_read = (selected["Win Prob"] - selected["Hit"]).to_numpy(float)
    gap, lo, hi = _bootstrap_mean_ci_clustered(
        over_read, selected["Date"].to_numpy(), rng, DAY_RESAMPLES
    )
    return {
        "n": stats["n"],
        "read": stats["pred_rate"],
        "hit": stats["hit_rate"],
        "market": stats["book_rate"],
        "payout": stats["payout"],
        "gap": gap,
        "gap_lo": lo,
        "gap_hi": hi,
        "overstatement": over_read.sum(),
    }


def _bulk(rungs: pd.DataFrame) -> dict[str, float]:
    """Every settled rung, selected or not: served log-loss and the argmax side's read - hit."""
    graded = rungs[rungs["Actual"] != rungs["Line"]]
    over = (graded["Actual"] > graded["Line"]).to_numpy(float)
    p = graded["p_serving"].to_numpy(float)
    read_over, read_under = np.minimum(p, _MAX_CONFIDENCE), np.minimum(1 - p, _MAX_CONFIDENCE)
    hit = np.where(read_over >= read_under, over, 1 - over)
    return {
        "bulk_log_loss": log_loss(over, p, labels=[0, 1]),
        "bulk_gap": float(np.mean(np.maximum(read_over, read_under) - hit)),
    }


def _bands(tail: pd.DataFrame) -> dict[str, np.ndarray]:
    """The band each selected leg falls in, per split (spec section 6, Outputs)."""
    toward_bet = np.where(
        tail["Bet"].eq("Over"), tail["consensus"] - tail["Line"], tail["Line"] - tail["consensus"]
    )
    z = (tail["Line"] - tail["Projection"]).abs() / tail["served_sd"]
    return {
        "side": tail["Bet"].to_numpy(),
        "quote": tail["QuoteAuthenticity"].to_numpy(),
        "rung": np.where(tail["Line"].eq(tail["Test Line"]), "main", "alt"),
        "consensus distance": np.select(
            [
                np.isnan(toward_bet),
                toward_bet <= -1,
                toward_bet < 0,
                toward_bet == 0,
                toward_bet <= 1,
            ],
            ["none", "<= -1", "(-1, 0)", "0", "(0, 1]"],
            "> 1",
        ),
        "|z|": pd.cut(z, _Z_BANDS, right=False).astype(str).to_numpy(),
        "sportsbook rung": np.where(tail["n_books"] > 0, "quoted", "none"),
        "payout": tail["Payout Source"].to_numpy(),
    }


def _in_scope(frame: pd.DataFrame, scope: str, key: str) -> pd.DataFrame:
    return frame if scope == "overall" else frame[frame[_SCOPE_COLUMNS[scope]].eq(key)]


def _summarize(
    rungs: pd.DataFrame,
    selected: pd.DataFrame,
    trained: pd.DataFrame,
    live: pd.DataFrame,
    cells: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """One row per scope (overall, league, cell), and per split band above cell level.

    The split bands cut the selected tail; the ``live version age`` split cuts the live
    recommended legs by ``Version Age``.
    """
    leagues = sorted(cells["League"].unique())
    scopes = [("overall", "all"), *(("league", lg) for lg in leagues)]
    records = []
    for scope, key in [*scopes, *(("cell", cell) for cell in cells["cell"])]:
        tail = _in_scope(selected, scope, key)
        training_tail = _in_scope(trained, scope, key)
        live_legs = _in_scope(live, scope, key)
        live_tail = _tail(live_legs, rng)
        scope_cells = _in_scope(cells, scope, key)
        scope_rungs = _in_scope(rungs, scope, key)
        records.append(
            {
                "scope": scope,
                "key": key,
                "split": "all",
                "band": "all",
                "test_rows": scope_cells["test_rows"].sum(),
                "recon_err": scope_cells["recon_err"].max(),
                "rung_rows": len(scope_rungs),
                **_tail(tail, rng),
                "descriptive": len(tail) < DESCRIPTIVE_MIN_SELECTED,
                "posted_share": tail["Payout Source"].eq("posted").mean(),
                "training_n": len(training_tail),
                "training_gap": (training_tail["Win Prob"] - training_tail["Hit"]).mean(),
                **{f"live_{stat}": live_tail[stat] for stat in ("n", "gap", "gap_lo", "gap_hi")},
                **_bulk(scope_rungs),
            }
        )
        if scope == "cell":
            continue
        cuts = [(split, tail, bands) for split, bands in _bands(tail).items()]
        cuts.append(("live version age", live_legs, live_legs["Version Age"]))
        for split, legs, bands in cuts:
            records += [
                {"scope": scope, "key": key, "split": split, "band": band, **_tail(group, rng)}
                for band, group in legs.groupby(bands)
            ]
    return pd.DataFrame(records)


def _history_since(start: str) -> pd.DataFrame:
    """Live history from ``start``, each row tagged with how old its model version was.

    Age counts from the version's first served date over all of history, so the window cut
    comes after it.
    """
    history = read_history()
    # A retrain is league-wide and its date heads every version string it serves.
    retrain = history["Model Version"].str.extract(r"^(\d{8})\.", expand=False)
    served = pd.to_datetime(history["Date"])
    age = (served - served.groupby([history["League"], retrain]).transform("min")).dt.days
    history["Version Age"] = np.select(
        [age.le(FRESH_VERSION_DAYS), age.notna()],
        [f"days 0-{FRESH_VERSION_DAYS}", f"days {FRESH_VERSION_DAYS + 1}+"],
        None,
    )
    return history[history["Date"] >= start]


def _print_tables(
    summary: pd.DataFrame, information: pd.DataFrame, overround: pd.DataFrame
) -> None:
    headline = summary[summary["split"].eq("all")][_HEADLINE_COLUMNS]
    splits = summary[summary["scope"].eq("overall") & summary["split"].ne("all")]
    split_columns = ["split", "band", "n", "read", "hit", "market", "gap", "gap_lo", "gap_hi"]
    # A --league run whose cells all miss the fit floors has no estimate columns to drop.
    pooled = information[information["scope"].ne("cell")].drop(
        columns=["split", "lin_b_market", "logit_b_market"], errors="ignore"
    )
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(headline.to_string(index=False, float_format="%.3f"))
        print(splits[split_columns].to_string(index=False, float_format="%.3f"))
        print("information test (cell fixed effects; ll_gain in nats per row):")
        print(pooled.to_string(index=False, float_format="%.4f"))
        print("overround (1 / posted payout x stored price, two-sided rungs off the standard):")
        print(overround.to_string(float_format="%.4f"))
        print(f"Underdog standard rung: {2 / UNDERDOG_BOOST_BASELINE:.4f}")


@click.command()
@click.option(
    "--league",
    "leagues",
    multiple=True,
    help="Score only these leagues (repeatable). Default: every test set.",
)
@click.option(
    "--archive",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_ARCHIVE,
    help="Odds archive, opened read-only (default: $SPORTSTRADAMUS_ARCHIVE_DB or "
    "archive/archive.duckdb).",
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=_DEFAULT_OUT,
    show_default=True,
    help="Sandbox CSV of every scope and split; never model_stats.parquet.",
)
def main(leagues: tuple[str, ...], archive: Path, out: Path) -> None:
    """Replay the live recommendation rule on held-out test rows at the ladder's DFS rungs.

    A training-time diagnostic, never a ship gate. Prints the selected-tail gap per cell,
    league and overall beside the live gap over the same window, then the information test,
    and writes every scope and split to OUT.
    """
    con = open_archive(archive)
    first_rung_date = con.execute(
        "SELECT min(game_date) FROM ladder WHERE list_contains($platforms, book)",
        {"platforms": list(SCORED_PLATFORMS)},
    ).fetchone()[0]
    start = str(first_rung_date)
    paths = sorted(
        path
        for path in TEST_SETS_DIR.glob("*.csv")
        if not leagues or path.stem.split("_", 1)[0] in leagues
    )
    priced, knobs, cells, informative, excluded = [], {}, [], [], {}
    for path in tqdm(paths, desc="pricing cells"):
        try:
            cell_rungs, knobs[path.stem], diag, info = price_cell(path, con, start)
        except CellExcludedError as exc:
            excluded[path.stem] = str(exc)
            continue
        priced.append(cell_rungs)
        cells.append(diag)
        informative.append(info)
    con.close()

    history = _history_since(start)
    live_sides = history.drop_duplicates(_RUNG_KEY)[[*_RUNG_KEY, "Bet", "Boost"]]
    live_sides = live_sides.rename(columns={"Bet": "Live Bet", "Boost": "Live Boost"})
    rungs = pd.concat(priced, ignore_index=True).merge(live_sides, on=_RUNG_KEY, how="left")
    rungs, overround = side_boosts(rungs)

    # finalize_records warns on every slate it trims, which offline is nearly all of them.
    logger.setLevel(logging.ERROR)
    replays = {"p_serving": [], "p_training": []}
    by_cell = rungs.groupby("cell")
    for cell, cell_rungs in tqdm(by_cell, desc="replaying slates", total=by_cell.ngroups):
        for arm, frames in replays.items():
            frames.append(replay_live_rule(cell_rungs, cell_rungs[arm], knobs[cell]))
    selected, trained = (pd.concat(frames, ignore_index=True) for frames in replays.values())
    selected, trained = selected[selected["Recommended"]], trained[trained["Recommended"]]
    posted = selected["Live Bet"].eq(selected["Bet"]) | selected["One Sided"]
    selected = selected.assign(**{"Payout Source": np.where(posted, "posted", "assumed")})

    settled = settled_offers(history)
    windows = rungs.groupby(["League", "Market", "cell"], as_index=False)["Date"].max()
    live = settled[settled["Recommended"]].merge(
        windows, on=["League", "Market"], suffixes=("", " End")
    )
    live = live[live["Date"].le(live["Date End"])]

    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    summary = _summarize(rungs, selected, trained, live, pd.DataFrame(cells), rng)
    information = information_table(pd.concat(informative, ignore_index=True), rng)
    skipped = pd.DataFrame(
        {
            "scope": "cell",
            "key": list(excluded),
            "split": "excluded",
            "note": list(excluded.values()),
        }
    )
    pd.concat([summary, information, skipped], ignore_index=True).to_csv(out, index=False)

    _print_tables(summary, information, overround)
    for cell, reason in excluded.items():
        print(f"excluded {cell}: {reason}")
    print(f"wrote {out}")
