"""Price a cell's held-out test rows at every DFS rung the archive ladder held for them.

The pricing half of the tail scorecard; ``tail_scorecard`` replays the live rule on what this
builds. A test row is re-served from its dump's model-only parameters and the cell pickle's
serving knobs. The training convention must first rebuild the dump's own ``P`` (the
self-check that the chain is the pickle's); the serving convention then prices each Underdog
and Sleeper rung the ladder held for the row. Every input is read-only, the archive opened
``read_only``.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from sportstradamus.helpers import (
    GATE_PUBLISH_THRESHOLD,
    apply_cdf_recal,
    apply_temperature,
    fused_loc,
    get_odds,
)
from sportstradamus.helpers.distributions import _DP_PHI_CEILING, DecodedParams, predictive_std
from sportstradamus.helpers.io import model_pickle_path
from sportstradamus.helpers.training_quotes import AUTHENTIC, DFS_PLATFORM_BOOKS
from sportstradamus.scripts.tail_information import information_rows
from sportstradamus.training.model_strategy import BASE_STRUCTURAL_STRATEGY
from sportstradamus.training.posthoc import (
    PROB_STAGE,
    apply_posthoc,
    correct_fused_mean,
    served_mean,
)

# The DFS boards the live recommender prices; the other DFS books never post a leg here.
SCORED_PLATFORMS = ("Underdog", "Sleeper")

# What re-serving a test row and its information test need: the shared columns, then each
# family's model-only fusion inputs. A dump without them predates the shape persist.
_TEST_COLUMNS = [
    "Player",
    "Date",
    "Result",
    "Line",
    "EV",
    "Book_EV",
    "P",
    "P_standalone",
    "Odds",
    "QuoteAuthenticity",
    "StructuralStrategy",
]
_FAMILY_COLUMNS = {
    "SkewNormal": ["SN_Sigma_model", "SN_Alpha_model"],
    "NegBin": ["R_model"],
    "ZINB": ["R_model", "Gate_model"],
    "DPO": ["DP_PHI_model"],
}
_KNOBS = (
    "distribution",
    "weight",
    "cv",
    "dispersion_cal",
    "skew_cal",
    "hist_gate",
    "step",
    "temperature",
    "posthoc",
    "posthoc_blob",
    "pit_recal_blob",
    "model_version",
)

# The prototype rebuilt P to 1e-13 on every cell; past 1e-6 the chain is not the pickle's.
RECONSTRUCTION_TOL = 1e-6

# Training prices its test set without passing step, so get_odds runs at its default.
TRAINING_STEP = 1.0

_RUNG_SQL = """
WITH rung AS (
    SELECT k.rid, l.book AS platform, l.line, arg_max(l.p_over, l.observed_at) AS p_dfs,
        max(l.observed_at) AS last_poll
    FROM test_rows k JOIN ladder l ON l.league = $league AND l.market = $market
        AND l.game_date = k.game_date AND l.entity = k.player
    WHERE list_contains($platforms, l.book)
    GROUP BY ALL
), books AS (
    SELECT r.rid, r.platform, r.line, count(DISTINCT l.book) AS n_books
    FROM rung r JOIN test_rows k USING (rid) JOIN ladder l ON l.league = $league
        AND l.market = $market AND l.game_date = k.game_date AND l.entity = k.player
        AND l.line = r.line AND l.observed_at <= r.last_poll
    WHERE NOT list_contains($dfs, l.book)
    GROUP BY ALL
), book_lines AS (
    SELECT r.rid, r.platform, r.line, arg_max(o.line, o.observed_at) AS book_line
    FROM rung r JOIN test_rows k USING (rid) JOIN odds o ON o.league = $league
        AND o.market = $market AND o.game_date = k.game_date AND o.entity = k.player
        AND o.observed_at <= r.last_poll AND o.line IS NOT NULL
    WHERE NOT list_contains($dfs, o.book)
    GROUP BY r.rid, r.platform, r.line, o.book
), consensus AS (
    SELECT rid, platform, line, median(book_line) AS consensus FROM book_lines GROUP BY ALL
)
SELECT rung.*, coalesce(books.n_books, 0) AS n_books, consensus.consensus
FROM rung LEFT JOIN books USING (rid, platform, line)
    LEFT JOIN consensus USING (rid, platform, line)
-- finalize_records breaks Distance ties by board order, so the board order must not vary.
ORDER BY rid, platform, line
"""


class CellExcludedError(Exception):
    """A cell the scorecard cannot score; the message is the reason it prints."""


def open_archive(path: Path) -> duckdb.DuckDBPyConnection:
    """Open the odds archive read-only and pin the process's Archive singleton to it.

    ``finalize_records`` reads ``archive.default_totals``, which constructs the Archive
    singleton; the two environment switches point it at this file in read-only mode, so
    nothing this run opens can take DuckDB's writer lock (DuckDB also refuses to open a
    file this process already holds in a second mode).
    """
    os.environ["SPORTSTRADAMUS_ARCHIVE_DB"] = str(path)
    os.environ["SPORTSTRADAMUS_ARCHIVE_READ_ONLY"] = "1"
    return duckdb.connect(str(path), read_only=True)


def ladder_rungs(
    con: duckdb.DuckDBPyConnection, league: str, market: str, test_rows: pd.DataFrame
) -> pd.DataFrame:
    """Every Underdog/Sleeper rung the ladder held for each test row, at its last poll.

    ``test_rows`` carries ``rid``, ``game_date`` and ``player``. Per rung: the platform's
    stored over-price at its last poll (``p_dfs``), that poll's time, how many sportsbooks
    had a rung at the same line by then (``n_books``) and the decision-time ``consensus``:
    the median over sportsbooks of each one's latest line observed by the last poll.
    """
    con.register("test_rows", test_rows)
    params = {
        "league": league,
        "market": market,
        "platforms": list(SCORED_PLATFORMS),
        "dfs": sorted(DFS_PLATFORM_BOOKS),
    }
    return con.execute(_RUNG_SQL, params).df()


def _training_weight(rows: pd.DataFrame, knobs: dict) -> np.ndarray:
    # Training blends authentic rows at the fitted weight and every other row at 1 (the model
    # alone); serving blends every row at the fitted weight.
    return np.where(rows["QuoteAuthenticity"].eq(AUTHENTIC), knobs["weight"], 1.0)


def serve(
    rows: pd.DataFrame, knobs: dict, w_row: np.ndarray, ev_book: np.ndarray, step: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-serve test rows from their persisted model-only parameters at each row's ``Line``.

    The pickle's chain in serving order: ``fused_loc`` at per-row model weight ``w_row``
    against book base mean ``ev_book``, ``correct_fused_mean``, the dispersion / skew
    calibration (DPO phi capped at its ceiling), ``get_odds`` at ``step``, the whole-CDF
    recal, temperature, then a PROB_STAGE posthoc. Training and serving differ only in
    ``w_row``, ``ev_book`` and ``step``.

    Returns:
        ``(p_over, mean, sd)``: the calibrated over-probability, and the served
        distribution's mean and SD with its zero-inflation gate folded in.
    """
    dist, cv = knobs["distribution"], knobs["cv"]
    slug, blob, shape_cal = knobs["posthoc"], knobs["posthoc_blob"], knobs["dispersion_cal"]
    ev_model = rows["EV"].to_numpy(float)
    line = rows["Line"].to_numpy(float)
    if dist == "SkewNormal":
        gate_kw = {}
        if knobs["hist_gate"] > GATE_PUBLISH_THRESHOLD:
            # The CSV's Gate is training's blend w_train * gate_model against a 0 book gate;
            # dividing the training weight back out recovers the gate either arm re-blends.
            gate_model = rows["Gate"].to_numpy(float) / _training_weight(rows, knobs)
            gate_kw = {"gate_model": gate_model, "gate_book": 0.0}
        base, sigma, skew, gate = fused_loc(
            w_row,
            ev_model,
            ev_book,
            cv,
            dist,
            sigma=rows["SN_Sigma_model"].to_numpy(float),
            skew_alpha=rows["SN_Alpha_model"].to_numpy(float),
            **gate_kw,
        )
        base = correct_fused_mean(slug, blob, base, gate)
        shape = DecodedParams(ev=base, sigma=sigma * shape_cal, skew=skew + knobs["skew_cal"])
        under = get_odds(
            line, base, dist, step=step, sigma=shape.sigma, skew_alpha=shape.skew, gate=gate
        )
    elif dist == "DPO":
        phi_model = rows["DP_PHI_model"].to_numpy(float)
        base, phi, gate = fused_loc(w_row, ev_model, ev_book, cv, dist, phi=phi_model)
        base = correct_fused_mean(slug, blob, base, gate)
        shape = DecodedParams(ev=base, phi=np.minimum(phi * shape_cal, _DP_PHI_CEILING))
        under = get_odds(line, base, dist, step=step, phi=shape.phi)
    else:
        gate_kw = (
            {"gate_model": rows["Gate_model"].to_numpy(float), "gate_book": knobs["hist_gate"]}
            if dist == "ZINB"
            else {}
        )
        r_model = rows["R_model"].to_numpy(float)
        r, p, gate = fused_loc(w_row, ev_model, ev_book, cv, "NegBin", r=r_model, **gate_kw)
        base = correct_fused_mean(slug, blob, r * (1 - p) / p, gate)
        shape = DecodedParams(ev=base, r=r * shape_cal)
        under = get_odds(line, base, dist, step=step, r=shape.r, gate=gate)
    under = apply_cdf_recal(knobs["pit_recal_blob"], under)
    p_over = apply_temperature(1 - under, knobs["temperature"])
    if slug in PROB_STAGE:
        p_over = apply_posthoc(slug, blob, p_over)
    gate = 0.0 if gate is None else np.asarray(gate, dtype=float)
    # A zero-inflated family is a point mass at 0 of weight gate beside the base distribution.
    sd = np.sqrt((1 - gate) * predictive_std(dist, shape) ** 2 + gate * (1 - gate) * base**2)
    return p_over, served_mean(base, gate), sd


def reconstruction_error(rows: pd.DataFrame, knobs: dict) -> float:
    """Max |P_rebuilt - P| over test rows re-served under the training convention.

    Training blends with ``_training_weight`` against ``Book_EV`` at get_odds' default step;
    NaN (a row the chain cannot price) counts as a failure.
    """
    rebuilt, _, _ = serve(
        rows, knobs, _training_weight(rows, knobs), rows["Book_EV"].to_numpy(float), TRAINING_STEP
    )
    return float(np.max(np.abs(rebuilt - rows["P"].to_numpy(float))))


def price_cell(
    path: Path, con: duckdb.DuckDBPyConnection, start: str
) -> tuple[pd.DataFrame, dict, dict, pd.DataFrame]:
    """One cell's test rows priced at every ladder rung they had, under both conventions.

    Returns the rung rows (the test row's columns, then the rung's platform, line, stored
    price, last poll, sportsbook coverage and consensus, each convention's over-price and
    the served mean and SD), the pickle knobs, the cell's window test-row count and
    reconstruction error, and its information-test rows.

    Raises:
        CellExcludedError: no pickle, an unreplayed family, a test set without the model-only
            columns, a structural strategy, a failed reconstruction, or no rung.
    """
    league, slug = path.stem.split("_", 1)
    market = slug.replace("-", " ")
    try:
        with model_pickle_path(league, market).open("rb") as infile:
            filedict = pickle.load(infile)
    except FileNotFoundError:
        raise CellExcludedError("no model pickle") from None
    knobs = {key: filedict[key] for key in _KNOBS}
    dist = knobs["distribution"]
    if dist not in _FAMILY_COLUMNS:
        raise CellExcludedError(f"{dist} is not replayed")
    needed = [*_TEST_COLUMNS, *_FAMILY_COLUMNS[dist]]
    if dist == "SkewNormal" and knobs["hist_gate"] > GATE_PUBLISH_THRESHOLD:
        needed.append("Gate")
    if missing := sorted(set(needed) - set(pd.read_csv(path, nrows=0).columns)):
        raise CellExcludedError(f"not reconstructable, re-dump (no {', '.join(missing)})")
    rows = pd.read_csv(path, usecols=needed)
    if set(rows["StructuralStrategy"].dropna()) - {BASE_STRUCTURAL_STRATEGY}:
        raise CellExcludedError("structural strategy, skipped")
    error = reconstruction_error(rows, knobs)
    if not error <= RECONSTRUCTION_TOL:
        raise CellExcludedError(f"reconstruction max |dP| {error:.1e} > {RECONSTRUCTION_TOL:.0e}")

    rows = rows[rows["Date"] >= start].reset_index(drop=True)
    keys = pd.DataFrame(
        {
            "rid": rows.index,
            "game_date": pd.to_datetime(rows["Date"]).dt.date,
            "player": rows["Player"],
        }
    )
    found = ladder_rungs(con, league, market, keys)
    if found.empty:
        raise CellExcludedError("no ladder rung in the scored window")
    test_rows = rows.iloc[found["rid"]].rename(columns={"Line": "Test Line", "Result": "Actual"})
    priced = pd.concat(
        [
            test_rows.reset_index(drop=True),
            found.rename(columns={"platform": "Platform", "line": "Line"}),
        ],
        axis=1,
    )
    authentic = priced["QuoteAuthenticity"].eq(AUTHENTIC)
    serving_book = priced["Book_EV"].where(authentic, priced["EV"]).to_numpy(float)
    priced["p_serving"], priced["Projection"], priced["served_sd"] = serve(
        priced, knobs, np.full(len(priced), knobs["weight"]), serving_book, knobs["step"]
    )
    priced["p_training"] = serve(
        priced,
        knobs,
        _training_weight(priced, knobs),
        priced["Book_EV"].to_numpy(float),
        TRAINING_STEP,
    )[0]
    priced["Market Projection"] = priced["Book_EV"].where(authentic)
    diag = {"cell": path.stem, "League": league, "test_rows": len(rows), "recon_err": error}
    info = information_rows(con, league, market, rows, found)
    return (
        priced.assign(cell=path.stem, League=league, Market=market),
        knobs,
        diag,
        info.assign(cell=path.stem, League=league),
    )
