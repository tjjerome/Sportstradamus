"""Pins the refit-carry knob: a refit pickle serves boosters that have seen every matrix row.

A cell whose stat_meta sets ``refit: "all"`` (``train_market``'s ``refit`` keyword here) refits
its served booster(s) on every matrix row once the run is done (``training/refit.py``). A tiny
frozen matrix trains through ``train_market`` deterministically, knob off and on, as a NegBin
cell served through the plain mean anchor and as a gated, ratio-normalised SkewNormal cell
without one. Level 2 of ``Player position`` first appears past the 70 % mark, so only a refit
has seen it. These pins hold that:

- the knob off writes none of the refit's four keys and serves the boosters the knob-on pickle
  hashes as its siblings;
- the refit fits every row at the siblings' round counts, bit-identically run to run;
- a knob-on pickle equals the knob-off one but for its boosters and the four keys, identity
  included, and its test-set dump is byte-identical;
- model_prob and feature_parity serve the refit alike whichever levels a batch holds, and each
  refit booster has learned the late level;
- the knob refuses a hurdle cell, an affine-expert strategy and a holdout-blind run;
- the sweep's and confirm's meditate runs are knob-off, so a knob cell's holdout-blind trial trains.
"""

from __future__ import annotations

import hashlib
import importlib
import pickle
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest

from sportstradamus.helpers import config
from sportstradamus.helpers.io import market_file_slug
from sportstradamus.scripts import feature_parity
from sportstradamus.training import pipeline, ship_config
from sportstradamus.training.lineage import write_matrix_manifest
from sportstradamus.training.model_strategy import (
    BASE_STRUCTURAL_STRATEGY,
    MODEL_STRATEGY_MODEL_KEY,
    meditate_command,
)
from sportstradamus.training.refit import resolve_refit
from sportstradamus.training.structural_strategies import AFFINE_STRATEGY

# The package __init__ re-exports the model_prob *function*, shadowing the submodule.
mp = importlib.import_module("sportstradamus.prediction.model_prob")

LEAGUE, MARKET = "NFL", "refit carry"
SLUG = market_file_slug(LEAGUE, MARKET)
FEATURES = ["MeanYr", "STDYr", "ZeroYr", "MeanYr_nonzero", "Feature A", "Home", "Player position"]
N_PLAYERS, N_DATES = 30, 40
# The first LATE_PLAYERS players move to position LATE_LEVEL from date LATE_FROM on, past the
# 70 % mark at date 28, and score LATE_SHIFT more: what only a refit on every row can learn.
LATE_PLAYERS, LATE_FROM, LATE_LEVEL, LATE_SHIFT = 15, 30, 2, 5.0
# Late-level rows, and as many others, on a serving slate.
SLATE_ROWS = 12
# The SkewNormal cell's zero rate: past NONZERO_DENOM_GATE, so its model fits positive rows only.
ZERO_RATE = 0.15
REFIT_KEYS = {"refit_through", "refit_rows", "refit_rounds", "sibling_booster_sha256"}
BOOSTER_KEYS = {"model", "mean_anchor"}
CELLS = {
    "count": {"dist": "NegBin", "target_normalization": "none", "mean_anchor": "plain"},
    "continuous": {
        "dist": "SkewNormal",
        "target_normalization": "ratio_meanyr",
        "mean_anchor": "none",
    },
}
# A run the knob can refit: no hurdle, the base strategy, not holdout-blind.
SERVABLE = {
    "use_hurdle": False,
    "structural_strategy": BASE_STRUCTURAL_STRATEGY,
    "holdout_blind": False,
}
REFUSED = f"^{LEAGUE} {MARKET}: refit 'all' refused: .*"


class _FrozenClock(datetime):
    """One ``trained_at``, and so one ``model_version``, for every run in the module."""

    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 11, 23, 12, tzinfo=UTC)


class _Stats:
    def __init__(self, matrix: pd.DataFrame):
        self.league = LEAGUE
        self.log_strings = {"player": "Player"}
        self.gamelog = matrix[["Player", "Result"]].rename(columns={"Result": MARKET})
        self.model_dependency_inventory = {}

    def get_stat_columns(self, _market):
        return list(FEATURES)


class _Run(NamedTuple):
    filedict: dict
    dump: bytes


def _matrix(kind: str) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    player = np.tile(np.arange(N_PLAYERS), N_DATES)
    week = np.repeat(np.arange(N_DATES), N_PLAYERS)
    late = (player < LATE_PLAYERS) & (week >= LATE_FROM)
    base = (1.0 if kind == "count" else 2.0) * rng.uniform(3.0, 9.0, N_PLAYERS)[player]
    home = rng.integers(0, 2, player.size)
    mean = base * (1 + 0.1 * home) + LATE_SHIFT * late
    if kind == "count":
        result = rng.poisson(mean).astype(float)
    else:
        positive = np.maximum(0.5, rng.normal(mean, 2.0))
        result = np.where(rng.random(player.size) < ZERO_RATE, 0.0, positive)
    line = np.round(2 * base) / 2 + 0.5
    priced = player % 3 > 0
    return pd.DataFrame(
        {
            "Player": [f"P{i:02d}" for i in player],
            "Date": pd.date_range("2024-09-01", periods=N_DATES, freq="7D")[week],
            "Result": result,
            "MeanYr": 0.85 * base,
            "STDYr": rng.uniform(2.0, 3.0, player.size),
            "ZeroYr": 0.02 if kind == "count" else ZERO_RATE,
            "MeanYr_nonzero": base,
            "Feature A": rng.normal(size=player.size),
            "Home": home,
            "Player position": np.where(late, LATE_LEVEL, player % 2),
            "Line": line,
            "Odds": np.clip(rng.normal(0.5, 0.08, player.size), 0.2, 0.8),
            "EV": line + rng.normal(0.0, 0.5, player.size),
            "QuoteSource": np.where(priced, "book_direct", "pickem_platform"),
            "QuoteAuthenticity": np.where(priced, "authentic", "pickem"),
            "QuoteBookCount": np.where(priced, 3, 0),
        }
    )


def _train(out: Path, kind: str, **knobs) -> _Run:
    """One deterministic ``train_market`` run on the kind's frozen matrix, into a sandbox."""
    out.mkdir(parents=True)
    matrix = _matrix(kind)
    path = out / f"{SLUG}.parquet"
    matrix.to_parquet(path)
    stats = _Stats(matrix)
    write_matrix_manifest(
        path,
        matrix,
        stats,
        league=LEAGUE,
        market=MARKET,
        cutoff_date="2024-01-01",
        repo_root=Path.cwd(),
    )
    artifacts = out / "artifacts"
    pipeline.train_market(
        LEAGUE,
        MARKET,
        stats,
        None,
        pd.Timestamp("2024-01-01").date(),
        force=True,
        deterministic=True,
        matrix_input=path,
        artifact_output=artifacts,
        **{**CELLS[kind], **knobs},
    )
    return _Run(
        pickle.loads((artifacts / f"{SLUG}.mdl").read_bytes()),
        (artifacts / "test_sets" / f"{SLUG}.csv").read_bytes(),
    )


@pytest.fixture(scope="module", autouse=True)
def _sandboxed_pipeline():
    """Keeps every run off the gitignored calibration maps, and on one clock."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pipeline, "stat_cv", {LEAGUE: {}})
        patch.setattr(pipeline, "stat_zi", {})
        patch.setattr(pipeline, "datetime", _FrozenClock)
        yield


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict[tuple[str, str], _Run]:
    """Each cell trained knob off and knob on, and the anchored count cell knob on again."""
    root = tmp_path_factory.mktemp("refit")
    arms = {("count", "again"): {"refit": "all"}}
    for kind in CELLS:
        arms |= {(kind, "off"): {}, (kind, "on"): {"refit": "all"}}
    return {
        (kind, arm): _train(root / kind / arm, kind, **knobs) for (kind, arm), knobs in arms.items()
    }


def _model_strings(filedict: dict) -> dict[str, str]:
    """The pickle's served boosters, keyed as ``refit_rounds`` and its sibling hashes key them."""
    boosters = {"model": filedict["model"].booster}
    if "mean_anchor" in filedict:
        boosters["mean_anchor"] = filedict["mean_anchor"]
    return {key: booster.model_to_string() for key, booster in boosters.items()}


@pytest.mark.parametrize("kind", CELLS)
def test_knob_off_writes_no_refit_keys_and_serves_the_hashed_siblings(runs, kind):
    off, on = runs[kind, "off"].filedict, runs[kind, "on"].filedict
    assert not REFIT_KEYS & off.keys()
    assert on["sibling_booster_sha256"] == {
        key: hashlib.sha256(text.encode()).hexdigest() for key, text in _model_strings(off).items()
    }


@pytest.mark.parametrize("kind", CELLS)
def test_refit_fits_every_row_at_the_siblings_round_counts(runs, kind):
    off, on = runs[kind, "off"].filedict, runs[kind, "on"].filedict
    rounds = {"model": on["params"]["opt_rounds"]}
    if "mean_anchor" in on:
        rounds["mean_anchor"] = on["diagnostics"]["plain_rounds"]
    assert on["refit_rounds"] == rounds
    assert on["refit_rows"] == N_PLAYERS * N_DATES
    assert on["refit_through"] == _matrix(kind)["Date"].max().date().isoformat()
    assert _model_strings(on).keys() == _model_strings(off).keys() == rounds.keys()


def test_two_refits_are_bit_identical(runs):
    assert _model_strings(runs["count", "on"].filedict) == _model_strings(
        runs["count", "again"].filedict
    )


@pytest.mark.parametrize("kind", CELLS)
def test_refit_pickle_differs_only_in_its_boosters_and_refit_keys(runs, kind):
    off, on = runs[kind, "off"], runs[kind, "on"]
    assert on.filedict[MODEL_STRATEGY_MODEL_KEY] == off.filedict[MODEL_STRATEGY_MODEL_KEY]
    assert on.filedict["model_version"] == off.filedict["model_version"]
    rest_on = {k: v for k, v in on.filedict.items() if k not in BOOSTER_KEYS | REFIT_KEYS}
    rest_off = {k: v for k, v in off.filedict.items() if k not in BOOSTER_KEYS}
    assert pickle.dumps(rest_on) == pickle.dumps(rest_off)
    assert on.dump == off.dump


def _slate(matrix: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.Index]:
    """Late-level rows then others, uncast as serving receives them, and the late rows' labels."""
    late = matrix["Player position"].eq(LATE_LEVEL)
    late_rows = matrix.index[late][:SLATE_ROWS]
    return matrix.loc[late_rows.append(matrix.index[~late][-SLATE_ROWS:]), columns], late_rows


@pytest.mark.parametrize("kind", CELLS)
def test_serving_paths_score_the_refit_alike_whichever_levels_a_batch_holds(runs, kind):
    on = runs[kind, "on"].filedict
    matrix, gate = _matrix(kind), on["hist_gate"]
    slate, late_rows = _slate(matrix, on["expected_columns"])
    served = feature_parity.score(on, LEAGUE, MARKET, slate, gate)

    # Training casts the categoricals over every matrix row; a batch of the late level alone
    # holds none of the others.
    trained_cast = matrix[slate.columns].astype({"Home": "category", "Player position": "category"})
    pd.testing.assert_frame_equal(
        feature_parity.score(on, LEAGUE, MARKET, trained_cast.loc[slate.index], gate), served
    )
    pd.testing.assert_frame_equal(
        feature_parity.score(on, LEAGUE, MARKET, slate.loc[late_rows], gate),
        served.loc[late_rows],
    )

    rows = slate.copy()
    structural = mp._resolve_serving_strategy(on, LEAGUE, MARKET).structural_strategy
    market = mp._build_prob_params(
        on,
        MARKET,
        SimpleNamespace(volume_stats=()),
        rows,
        on["distribution"],
        on["normalized"],
        on["target_normalization"],
        structural,
    )
    # model_prob's frame carries more than the model's columns, in its own order.
    mp._decode_model_params(
        market,
        on["distribution"],
        rows[rows.columns[::-1]].assign(**{"Market Projection": 1.0}),
        gate,
        on["offset_meta"],
        on["target_normalization"],
        on.get("mean_anchor"),
    )
    pd.testing.assert_frame_equal(market[served.columns], served)


@pytest.mark.parametrize("kind", CELLS)
def test_every_served_booster_learns_the_level_its_sibling_never_saw(runs, kind):
    on, off = runs[kind, "on"].filedict, runs[kind, "off"].filedict
    slate, late_rows = _slate(_matrix(kind), on["expected_columns"])
    pairs = [(on, off)]
    if "mean_anchor" in on:
        # The anchor serves this cell's mean; without it the distributional model serves its own.
        pairs.append(tuple({k: v for k, v in f.items() if k != "mean_anchor"} for f in (on, off)))
    for refit, sibling in pairs:
        shift = (
            feature_parity.score(refit, LEAGUE, MARKET, slate, on["hist_gate"])["Projection"]
            - feature_parity.score(sibling, LEAGUE, MARKET, slate, on["hist_gate"])["Projection"]
        )
        assert shift[late_rows].mean() > LATE_SHIFT / 2
        assert shift.drop(late_rows).abs().mean() < LATE_SHIFT / 5


@pytest.mark.parametrize(
    ("flags", "reason"),
    [
        ({"use_hurdle": True}, "hurdle"),
        ({"structural_strategy": AFFINE_STRATEGY}, "affine-expert"),
        ({"holdout_blind": True}, "holdout-blind"),
    ],
)
def test_refit_refuses_a_run_it_cannot_serve(flags, reason):
    run = {**SERVABLE, **flags}
    with pytest.raises(ValueError, match=REFUSED + reason):
        resolve_refit(LEAGUE, MARKET, "all", **run)
    assert resolve_refit(LEAGUE, MARKET, "none", **run) is False


@pytest.mark.parametrize(
    ("knobs", "reason"),
    [
        ({"holdout_blind": True}, "holdout-blind"),
        ({"dist": "ZINB", "zinb_mode": "hurdle"}, "hurdle"),
    ],
)
def test_train_market_refuses_before_any_fit(tmp_path, knobs, reason):
    with pytest.raises(ValueError, match=REFUSED + reason):
        _train(tmp_path / "refused", "count", refit="all", **knobs)
    assert not (tmp_path / "refused" / "artifacts").exists()


def test_the_cell_knob_decides_unless_overridden(monkeypatch):
    monkeypatch.setattr(config, "stat_meta", {LEAGUE: {MARKET: {"refit": "all"}}})
    assert resolve_refit(LEAGUE, MARKET, None, **SERVABLE) is True
    assert resolve_refit(LEAGUE, MARKET, "none", **SERVABLE) is False
    assert resolve_refit(LEAGUE, "other", None, **SERVABLE) is False


def test_sweep_and_confirm_runs_are_knob_off(monkeypatch):
    monkeypatch.setattr(config, "stat_meta", {LEAGUE: {MARKET: {"refit": "all"}}})
    command = meditate_command(LEAGUE, MARKET, "--holdout-blind")
    override = command[command.index("--refit") + 1]
    assert resolve_refit(LEAGUE, MARKET, override, **{**SERVABLE, "holdout_blind": True}) is False


def test_unknown_refit_rejected():
    cell = {"shipped": "withheld", "dist": "NegBin", "target_normalization": "none"}
    ship_config._validate_cell(LEAGUE, MARKET, {**cell, "refit": "all"})
    with pytest.raises(ValueError, match="unknown refit"):
        ship_config._validate_cell(LEAGUE, MARKET, {**cell, "refit": "weekly"})
