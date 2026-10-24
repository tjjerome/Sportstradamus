"""Decision-time history columns: schema membership, upsert over older rows, parquet round-trip.

Also the ``Alt Line`` flag the writer stamps beside ``Consensus Line``.
"""

import numpy as np
import pandas as pd

from sportstradamus.helpers import config, io
from sportstradamus.history_schema import DECISION_COLS, HISTORY_COLS, OFFER_LEVEL_COLS
from sportstradamus.prediction.cli import _stamp_alt_line, _upsert_history
from sportstradamus.prediction.offer_records import SCORED_RECORD_COLS


def _row(line: float, platform: str, **extra) -> dict:
    base = dict.fromkeys(HISTORY_COLS, np.nan)
    base.update(
        {
            "Player": "Player A",
            "League": "NBA",
            "Date": "2026-10-04",
            "Market": "points",
            "Line": line,
            "Platform": platform,
            "Bet": "Over",
            "Win Prob": 0.6,
            "Market Prob": 0.55,
            "Boost": 1.0,
        }
    )
    base.update(extra)
    return base


def test_decision_cols_sit_between_offer_cols_and_actual():
    assert HISTORY_COLS[-1] == "Actual"
    assert HISTORY_COLS[-1 - len(DECISION_COLS) : -1] == DECISION_COLS
    assert not set(DECISION_COLS) & set(OFFER_LEVEL_COLS)


def test_scored_record_supplies_every_decision_col_the_writer_does_not_stamp():
    # cli.py stamps Scored At, First Scored At and Consensus Line itself; everything else must
    # come off the scored record, or the writer's strict column selection raises.
    stamped = {"Scored At", "First Scored At", "Consensus Line"}
    assert set(DECISION_COLS) - stamped <= set(SCORED_RECORD_COLS)


def test_upsert_over_pre_schema_rows_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "HISTORY_PATH", tmp_path / "history.parquet")
    old = pd.DataFrame([_row(24.5, "Underdog")]).drop(columns=DECISION_COLS)
    stamp = pd.Timestamp("2026-10-04T18:30:00")
    new = pd.DataFrame(
        [
            _row(
                24.5,
                "Sleeper",
                **{
                    "Scored At": stamp,
                    "First Scored At": stamp,
                    "Quote Source": "book_direct",
                    "Quote Books": 3.0,
                    "Quote Observed At": stamp,
                    "Model Weight": 0.4,
                    "Market Prob Implied": False,
                },
            )
        ]
    )
    io.write_history(_upsert_history(old, new))
    back = io.read_history().set_index("Platform")

    # concat appends the columns the older rows lacked; readers go by name, not order.
    assert set(back.columns) == set(HISTORY_COLS) - {"Platform"}
    assert back.loc["Sleeper", "Quote Source"] == "book_direct"
    assert back.loc["Sleeper", "Scored At"] == stamp
    assert back.loc["Sleeper", "First Scored At"] == stamp
    # A flag beside the older rows' NaN survives the parquet write as a flag.
    np.testing.assert_equal(back.loc["Sleeper", "Market Prob Implied"], False)
    assert pd.isna(back.loc["Underdog", "Quote Source"])
    assert pd.isna(back.loc["Underdog", "Scored At"])
    assert pd.isna(back.loc["Underdog", "First Scored At"])
    assert pd.isna(back.loc["Underdog", "Market Prob Implied"])
    assert np.isnan(back.loc["Underdog", "Model Weight"])


_FIRST = pd.Timestamp("2026-10-03T13:50:00")
_RUN = pd.Timestamp("2026-10-04T18:50:00")


def test_a_rescore_keeps_the_stamp_its_key_was_first_stored_with():
    stored = pd.DataFrame(
        [
            # reflect closed this one, so the dedup keeps it over its re-score.
            _row(26.5, "Underdog", **{"Close Market Prob": 0.52, "First Scored At": _FIRST}),
            _row(24.5, "Underdog", **{"First Scored At": _FIRST}),
            # Stored before the column existed.
            _row(25.5, "Underdog"),
        ]
    )
    # Listed apart from the stored order, with a new rung among the re-scores.
    rescored = pd.DataFrame(
        [
            _row(line, "Underdog", **{"Win Prob": 0.7, "First Scored At": _RUN})
            for line in (25.5, 23.5, 24.5, 26.5)
        ]
    )

    upserted = _upsert_history(stored, rescored).set_index("Line")

    assert sorted(upserted.index) == [23.5, 24.5, 25.5, 26.5]
    # The re-scored row is the one kept, under the stamp its key was stored with.
    assert upserted.loc[24.5, ["Win Prob", "First Scored At"]].tolist() == [0.7, _FIRST]
    assert upserted.loc[26.5, ["Close Market Prob", "First Scored At"]].tolist() == [0.52, _FIRST]
    assert upserted.at[25.5, "Win Prob"] == 0.7
    assert pd.isna(upserted.at[25.5, "First Scored At"])
    assert upserted.at[23.5, "First Scored At"] == _RUN


def test_a_history_older_than_the_column_stamps_only_new_keys():
    stored = pd.DataFrame([_row(24.5, "Underdog")]).drop(columns="First Scored At")
    rescored = pd.DataFrame(
        [_row(line, "Underdog", **{"First Scored At": _RUN}) for line in (23.5, 24.5)]
    )

    stamps = _upsert_history(stored, rescored).set_index("Line")["First Scored At"]

    assert pd.isna(stamps[24.5])
    assert stamps[23.5] == _RUN


def test_alt_line_is_judged_against_the_reference_line(monkeypatch):
    """An entry no sportsbook posts is still judged, at its own line of record."""
    monkeypatch.setitem(config.stat_dist, "XLG", {"count": "NegBin", "yards": "Gamma"})
    offers = pd.DataFrame(
        {
            "League": "XLG",
            "Market": ["count", "count", "yards", "yards", "count"],
            "Line": [5.0, 5.5, 52.5, 53.5, 9.5],
            "Reference Line": [4.5, 4.5, 50.5, 50.5, np.nan],
            "Consensus Line": np.nan,
        }
    )

    flags = _stamp_alt_line(offers)["Alt Line"]

    # Count lines move in half points and tolerate 0.75; continuous lines tolerate 2.5.
    # With no reference line at all there is nothing to be an alternate of.
    assert flags.tolist() == [False, True, False, True, False]
