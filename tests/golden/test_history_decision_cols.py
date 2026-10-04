"""Decision-time history columns: schema membership, upsert over older rows, parquet round-trip."""

import numpy as np
import pandas as pd

from sportstradamus.helpers import io
from sportstradamus.history_schema import DECISION_COLS, HISTORY_COLS, OFFER_LEVEL_COLS
from sportstradamus.prediction.cli import _upsert_history
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
    # cli.py stamps Scored At and Consensus Line itself; everything else must come off
    # the scored record, or the writer's strict column selection raises.
    assert set(DECISION_COLS) - {"Scored At", "Consensus Line"} <= set(SCORED_RECORD_COLS)


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
                    "Quote Source": "book_direct",
                    "Quote Books": 3.0,
                    "Quote Observed At": stamp,
                    "Model Weight": 0.4,
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
    assert pd.isna(back.loc["Underdog", "Quote Source"])
    assert pd.isna(back.loc["Underdog", "Scored At"])
    assert np.isnan(back.loc["Underdog", "Model Weight"])
