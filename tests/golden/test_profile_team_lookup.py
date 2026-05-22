"""Regression: profile team lookups tolerate mid-season expansion teams.

A new franchise (the 2025 Golden State Valkyries 'GSV', the 2026 Toronto Tempo
'TOR') appears in the gamelog before it accrues the teamlog history that builds
``teamProfile`` / ``defenseProfile``. A plain ``profile.loc[teams]`` lookup then
raised ``KeyError: ['TOR']`` and crashed WNBA training/prediction (formerly
papered over by per-team hardcoded ``"GSV"`` guards). ``_profile_rows_for_teams``
substitutes an all-NaN row for any absent team — the honest "no history"
representation, which LightGBM consumes as a missing feature.
"""

import numpy as np
import pandas as pd

from sportstradamus.stats.base import _profile_rows_for_teams


def test_present_teams_returned_in_request_order():
    """All-present keys behave exactly like ``.loc`` (order + duplicates kept)."""
    prof = pd.DataFrame({"def_a": [1.0, 2.0], "def_b": [3.0, 4.0]}, index=["KC", "BUF"])
    out = _profile_rows_for_teams(prof, ["BUF", "KC", "BUF"])
    assert out["def_a"].tolist() == [2.0, 1.0, 2.0]
    assert out["def_b"].tolist() == [4.0, 3.0, 4.0]


def test_missing_team_yields_nan_row_not_keyerror():
    """An absent team (e.g. the 2026 expansion 'TOR') yields NaN, not KeyError."""
    prof = pd.DataFrame({"def_a": [1.0]}, index=["KC"])
    out = _profile_rows_for_teams(prof, ["KC", "TOR"])
    assert out.loc["KC", "def_a"] == 1.0
    assert np.isnan(out.loc["TOR", "def_a"])
    assert list(out.index) == ["KC", "TOR"]
