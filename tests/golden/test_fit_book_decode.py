"""The I6d decode harness's pure parts, against docs/handoffs/i6d-decode-preregistration.md.

Synthetic frames throughout; no archive opens. Pins the reference cleaning (§5), the
chronological split (§6), the candidate order and the code-history family (§7), the fit and
acceptance (§8, §9), inheritance (§10), and the CLI's refusal to write a partial table.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from sportstradamus.helpers.book_levelling import decode_under
from sportstradamus.scripts.fit_book_decode import (
    LOG_CV_BOUNDS,
    _candidate_order,
    _evaluate,
    _halves,
    _inherit,
    _references,
    _test_block,
    family_on,
    main,
)

_KEY = ["game_date", "entity", "book", "line"]


def _dated(dates: list[str], per_date: list[int]) -> pd.DataFrame:
    """Matched rows, latest game date first, so a split that skips the sort fails."""
    return pd.DataFrame({"game_date": pd.to_datetime(np.repeat(dates, per_date))}).iloc[::-1]


@pytest.mark.parametrize(
    ("dates", "fit_dates"),
    [
        (["2025-11-01", "2025-11-02", "2025-11-03", "2025-11-04"], 2),
        (["2025-11-01", "2025-11-02", "2025-11-03", "2025-11-04", "2025-11-05"], 3),
    ],
)
def test_the_fit_half_is_the_first_half_of_the_matched_dates(dates, fit_dates):
    fit, val = _halves(_dated(dates, [40] * len(dates)))
    assert sorted(fit["game_date"].unique()) == list(pd.to_datetime(dates[:fit_dates]))
    assert sorted(val["game_date"].unique()) == list(pd.to_datetime(dates[fit_dates:]))


def test_a_half_under_the_floor_leaves_the_block_untestable():
    assert _halves(_dated(["2025-11-01", "2025-11-02"], [40, 29])) is None


def test_the_code_history_family_tries_second_and_once():
    assert _candidate_order("NegBin") == ["stored", "NegBin", "SkewNormal", "Poisson", "Gamma"]
    assert _candidate_order(None) == ["stored", "SkewNormal", "NegBin", "Poisson", "Gamma"]


def test_the_code_history_family_is_the_last_config_on_or_before_the_date():
    history = [
        (datetime.date(2026, 2, 23), {"NFL": {"tds": "ZINB", "yards": "DPO"}}),
        (datetime.date(2026, 5, 23), {"NFL": {"tds": "SkewNormal"}}),
    ]
    assert family_on(history, "NFL", "tds", datetime.date(2026, 3, 1)) == "NegBin"
    assert family_on(history, "NFL", "tds", datetime.date(2026, 6, 17)) == "SkewNormal"
    assert family_on(history, "NFL", "receptions", datetime.date(2026, 3, 1)) == "SkewNormal"
    assert family_on(history, "NFL", "yards", datetime.date(2026, 3, 1)) is None
    assert family_on(history, "NFL", "tds", datetime.date(2026, 2, 22)) is None


def test_the_references_drop_a_disagreeing_rung_and_an_even_price():
    key = {"game_date": pd.Timestamp("2025-11-02"), "entity": "a", "book": "fanduel", "line": 0.5}
    rows = pd.DataFrame(
        [
            {**key, "layer": "01:00", "native": True, "under_prob": 0.40, "p_over": 0.60},
            {**key, "layer": "01:00", "native": True, "under_prob": 0.44, "p_over": 0.58},
            {**key, "layer": "01:00", "native": True, "under_prob": 0.30, "p_over": 0.40},
            {**key, "layer": "01:00", "native": True, "under_prob": 0.50, "p_over": 0.50},
            {**key, "layer": "01:00", "native": False, "under_prob": 0.90, "p_over": None},
            {**key, "layer": "23:00", "native": True, "under_prob": 0.20, "p_over": 0.80},
        ]
    )
    references = _references(rows)
    assert references["01:00"][tuple(key.values())] == pytest.approx(0.42)
    assert references["23:00"][tuple(key.values())] == pytest.approx(0.20)


def _block(stored_shift: float, cv: float = 0.6) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Migrated rows on six game dates, and the native 01:00 twins a NegBin decode prices."""
    rng = np.random.default_rng(1729)
    n = 120
    keys = pd.DataFrame(
        {
            "game_date": pd.to_datetime(np.repeat(pd.date_range("2025-11-01", periods=6), 20)),
            "entity": [f"player {i}" for i in range(n)],
            "book": "fanduel",
            "line": rng.choice([0.5, 1.5, 2.5], n),
            "ev": rng.uniform(0.8, 3.0, n),
        }
    )
    reference = decode_under("NegBin", keys["line"].to_numpy(), keys["ev"].to_numpy(), cv)
    native = keys.assign(under_prob=reference, p_over=1 - reference, layer="01:00", native=True)
    return keys.assign(under_prob=reference + stored_shift), _references(native)


def test_a_level_stored_price_is_accepted_without_a_parameter():
    rows, references = _block(stored_shift=0.001)
    record = _test_block(rows, references, None)
    assert record["tested"] and record["reference_layer"] == "01:00"
    assert (record["verdict"], record["family"], record["cv"]) == ("accept", "stored", None)
    assert record["candidates"]["stored"]["log_cv"] is None


def test_an_off_level_block_is_accepted_at_its_code_history_family():
    rows, references = _block(stored_shift=0.2)
    record = _test_block(rows, references, "NegBin")
    assert list(record["candidates"]) == ["stored", "NegBin"]
    assert record["candidates"]["stored"] is None or not record["candidates"]["stored"]["passes"]
    assert (record["verdict"], record["family"]) == ("accept", "NegBin")
    assert record["cv"] == pytest.approx(0.6, abs=1e-3)
    assert record["candidates"]["NegBin"]["interior"] is True


def test_a_fit_at_its_bound_does_not_pass():
    # Lines a half-point either side of ev: the decode leans off .5 symmetrically, so the bars
    # hold at the widest cv and only the bound rule can refuse it.
    rows = pd.DataFrame({"line": np.tile([2.5, 1.5], 40), "ev": 2.0, "reference": 0.5})
    outcome = _evaluate("SkewNormal", rows.iloc[:40], rows.iloc[40:])
    assert abs(outcome["mean_diff"]) <= 0.01 and outcome["median_abs_diff"] <= 0.03
    assert outcome["log_cv"] > LOG_CV_BOUNDS[1] - 1e-3
    assert outcome["interior"] is False
    assert outcome["passes"] is False


def _record(layer: str, block: str, verdict=None, family=None, cv=None) -> dict:
    return {
        "league": "NFL",
        "market": "tds",
        "layer": layer,
        "block": block,
        "tested": verdict is not None,
        "verdict": verdict,
        "family": family,
        "cv": cv,
    }


def _settled(blocks: list[dict], layer: str, block: str) -> dict:
    return next(b for b in _inherit(blocks) if (b["layer"], b["block"]) == (layer, block))


def test_an_untested_season_inherits_its_nearest_tested_season():
    blocks = [_record("01:00", "2023"), _record("01:00", "2024", "accept", "NegBin", 0.59)]
    settled = _settled(blocks, "01:00", "2023")
    assert (settled["verdict"], settled["family"], settled["cv"]) == ("inherit", "NegBin", 0.59)
    assert _settled(blocks, "01:00", "2024") == blocks[1]


def test_a_tied_distance_takes_the_later_season():
    blocks = [
        _record("01:00", "2023", "accept", "SkewNormal", 1.02),
        _record("01:00", "2024"),
        _record("01:00", "2025", "accept", "NegBin", 0.55),
    ]
    settled = _settled(blocks, "01:00", "2024")
    assert (settled["verdict"], settled["family"], settled["cv"]) == ("inherit", "NegBin", 0.55)


@pytest.mark.parametrize(
    ("blocks", "layer", "block"),
    [
        (
            [
                _record("01:00", "2023"),
                _record("01:00", "2024", "demote"),
                _record("01:00", "2026", "accept", "NegBin", 0.5),
            ],
            "01:00",
            "2023",
        ),
        (
            [_record("01:00", "2023"), _record("02:00", "2024", "accept", "NegBin", 0.5)],
            "01:00",
            "2023",
        ),
        (
            [_record("live", "2025-11"), _record("live", "2025-12", "accept", "stored")],
            "live",
            "2025-11",
        ),
    ],
    ids=["demoted nearest source", "no tested season on the layer", "live epoch"],
)
def test_an_untested_block_without_an_accepted_source_demotes(blocks, layer, block):
    assert _settled(blocks, layer, block)["verdict"] == "demote"


def test_a_league_run_needs_its_own_out_path(tmp_path):
    archive = tmp_path / "archive.duckdb"
    archive.touch()
    result = CliRunner().invoke(main, ["--league", "NHL", "--archive", str(archive)])
    assert result.exit_code != 0
    assert "--out" in result.output and "unlevelled" in result.output
