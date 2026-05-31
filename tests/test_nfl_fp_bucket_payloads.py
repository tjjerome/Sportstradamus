"""Characterization tests for `_iter_bucket_payloads`.

The four nfl_fp separation/coverage aggregators shared an identical JSON
``bucket`` parsing preamble that was extracted into one generator. These tests
pin the generator's skip semantics and assert it yields exactly what the old
inline scaffolding did, so the aggregators (whose divergent bodies were left
untouched) produce identical output by construction.
"""

import json

import numpy as np
import pandas as pd

from sportstradamus.stats.nfl_fp_aggregation import PLAYER_GROUP_COL
from sportstradamus.stats.nfl_fp_weekly_aggregate import _iter_bucket_payloads


def _old_inline(df):
    """The scaffolding the four aggregators shared before consolidation."""
    out = []
    for player_id, bucket_str in zip(df[PLAYER_GROUP_COL], df["bucket"], strict=True):
        if not isinstance(bucket_str, str) or not bucket_str:
            continue
        try:
            payload = json.loads(bucket_str)
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        out.append((player_id, payload))
    return out


def _frame(rows):
    return pd.DataFrame(rows, columns=[PLAYER_GROUP_COL, "bucket"])


def test_yields_only_valid_dict_payloads():
    df = _frame(
        [
            ("p1", json.dumps({"man": {"v": 1}})),
            ("p2", ""),
            ("p3", None),
            ("p4", np.nan),
            ("p5", "not json"),
            ("p6", json.dumps([1, 2, 3])),
            ("p7", json.dumps("scalar")),
            ("p8", json.dumps({"zone": {"v": 2}})),
        ]
    )
    assert list(_iter_bucket_payloads(df)) == [
        ("p1", {"man": {"v": 1}}),
        ("p8", {"zone": {"v": 2}}),
    ]


def test_matches_pre_consolidation_scaffolding():
    df = _frame(
        [
            ("a", json.dumps({"x": {"value": 0.4, "weight": 10}})),
            ("b", ""),
            ("c", "garbage{"),
            ("d", json.dumps(42)),
            ("e", json.dumps({"y": {"value": 0.1, "weight": 3}})),
            ("f", 3.14),
        ]
    )
    assert list(_iter_bucket_payloads(df)) == _old_inline(df)


def test_empty_frame_yields_nothing():
    assert list(_iter_bucket_payloads(_frame([]))) == []
