"""An NHL skater market trains on skaters only, time on ice included.

The time-on-ice matrix held goalie rows, priced at 23 to 37 minutes on the stand-in line
while the goalies played a full game (study §14 item 11).
"""

import pandas as pd

from sportstradamus.stats import StatsNHL


def test_time_on_ice_matrix_keeps_skaters_only():
    gamelog = pd.DataFrame({"position": ["C", "W", "D", "G"]})

    kept = StatsNHL()._market_position_filter(gamelog, "timeOnIce")

    assert kept["position"].tolist() == ["C", "W", "D"]
