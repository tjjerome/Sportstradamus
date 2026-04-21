from datetime import datetime

import click
from tqdm import tqdm

from sportstradamus.stats import StatsMLB, StatsNBA, StatsNFL, StatsNHL, StatsWNBA


@click.command()
@click.option(
    "--league",
    type=click.Choice(["All", "NFL", "NBA", "MLB", "NHL", "WNBA"]),
    default="All",
    help="Select league to train on",
)
def main(league):
    """
    Resets the stat structures for all leagues. This is useful for when we want to reprocess the data from scratch, or if we want to clear out any old data that may be causing issues.
    """

    if league in ["All", "NFL"]:
        NFL = StatsNFL()
        seasons = [
            (datetime(2021, 9, 9).date(), None),
            (datetime(2022, 9, 8).date(), None),
            (datetime(2023, 9, 7).date(), None),
            (datetime(2024, 9, 5).date(), None),
            (datetime(2025, 9, 4).date(), None),
        ]
        for season_start, _ in tqdm(seasons, desc="NFL seasons"):
            NFL.season_start = season_start
            NFL.update()
        NFL.trim_gamelog()
    elif league in ["All", "NHL"]:
        NHL = StatsNHL()
        seasons = [
            (datetime(2023, 10, 10).date(), None),
            (datetime(2024, 10, 4).date(), None),
            (datetime(2025, 10, 7).date(), None),
        ]
        for season_start, _ in tqdm(seasons, desc="NHL seasons"):
            NHL.season_start = season_start
            NHL.update()
        NHL.trim_gamelog()
    elif league in ["All", "NBA"]:
        NBA = StatsNBA()
        seasons = [
            (datetime(2023, 10, 24).date(), "2023-24"),
            (datetime(2024, 10, 22).date(), "2024-25"),
            (datetime(2025, 10, 21).date(), "2025-26"),
        ]
        for season_start, season in tqdm(seasons, desc="NBA seasons"):
            NBA.season_start = season_start
            NBA.season = season
            NBA.update()
        NBA.trim_gamelog()
    elif league in ["All", "WNBA"]:
        WNBA = StatsWNBA()
        seasons = [
            (datetime(2023, 5, 19).date(), None),
            (datetime(2024, 5, 14).date(), None),
            (datetime(2025, 5, 16).date(), None),
        ]
        for season_start, _ in tqdm(seasons, desc="WNBA seasons"):
            WNBA.season_start = season_start
            WNBA.update()
        WNBA.trim_gamelog()
    elif league in ["All", "MLB"]:
        MLB = StatsMLB()
        seasons = [
            (datetime(2023, 3, 30).date(), None),
            (datetime(2024, 3, 20).date(), None),
            (datetime(2025, 3, 18).date(), None),
        ]
        for season_start, _ in tqdm(seasons, desc="MLB seasons"):
            MLB.season_start = season_start
            MLB.update()
        MLB.trim_gamelog()


if __name__ == "__main__":
    main()
