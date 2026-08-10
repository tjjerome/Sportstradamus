"""Measure the MLB volume-normalization constants from the backfilled data:
the home/away batting-order plate-appearance curves and the within-slot PA spread
from the gamelog, plus the league OBP baseline from the teamlog (statsapi true-OBP
scale, so it matches the ``teamProfile["OBP"]`` the offense adjustment reads). Prints
values to paste into ``stats/mlb.py``. Re-run after a backfill to refresh them.
"""

from importlib import resources

import pandas as pd

from sportstradamus import data


def main():
    g = pd.read_parquet(resources.files(data) / "leagues" / "mlb" / "gamelog.parquet")
    g["PA"] = pd.to_numeric(g["plateAppearances"], errors="coerce").fillna(0)
    sb = g[g["starting batter"]].copy()

    home = [round(sb.loc[(sb.battingOrder == s) & sb.home, "PA"].mean(), 3) for s in range(1, 10)]
    away = [round(sb.loc[(sb.battingOrder == s) & ~sb.home, "PA"].mean(), 3) for s in range(1, 10)]
    std = [round(sb.loc[sb.battingOrder == s, "PA"].std(), 3) for s in range(1, 10)]
    teamlog = pd.read_parquet(resources.files(data) / "leagues" / "mlb" / "teamlog.parquet")
    lg_obp = pd.to_numeric(teamlog["OBP"], errors="coerce").mean()

    print(f"SLOT_PA_HOME = {tuple(home)}")
    print(f"SLOT_PA_AWAY = {tuple(away)}")
    print(f"SLOT_STD     = {tuple(std)}")
    print(f"LG_AVG_OBP (teamlog scale, matches teamProfile) = {lg_obp:.4f}")


if __name__ == "__main__":
    main()
