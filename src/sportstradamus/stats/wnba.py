"""StatsWNBA: WNBA player stats, inherits from StatsNBA."""

from datetime import date, datetime, timedelta

import nba_api.stats.endpoints as nba
import numpy as np
import pandas as pd

from sportstradamus.helpers import remove_accents
from sportstradamus.helpers.io import read_gamelog, write_gamelog
from sportstradamus.stats import nba_client, nba_season_tables
from sportstradamus.stats.base import fetch_upcoming_games
from sportstradamus.stats.nba import StatsNBA
from sportstradamus.stats.nba_client import NBAStatsError

# Regulation minutes of a WNBA game (four 10-minute quarters), the scale of every *_40 rate.
_REGULATION_MINUTES: int = 40

# The comp inputs the season-to-date bio and shot-location tables hold, read as of the eve of a
# comp pool's date (StatsNBA._rosters_as_of).
_DATED_COMP_FIELDS = ("USG_PCT", "TS_PCT", "RA_PCT", "ITP_PCT", "MR_PCT", "C3_PCT", "B3_PCT")

# Each zone share's column among the shot table's eight attempt columns: restricted area, the rest
# of the paint, mid-range, left and right corner threes, above-the-break threes, backcourt, and
# both corners' threes again as one zone. A share's denominator is the first six, every attempt
# but the backcourt's counted once.
_ZONE_SHARE_COLUMNS = {"RA_PCT": 0, "ITP_PCT": 1, "MR_PCT": 2, "C3_PCT": 7, "B3_PCT": 5}
_COUNTED_ZONES = 6

# stats.wnba.com abbreviations differ from our canonical set; remapped wherever a
# team code enters (player index, shot locations, player logs, team logs).
_WNBA_TEAM_ABBR_MAP = {
    "CONN": "CON",
    "NY": "NYL",
    "LA": "LAS",
    "LV": "LVA",
    "PHO": "PHX",
    "WSH": "WAS",
    "PDX": "POR",  # Portland — canonical POR (parity with the NBA), not the airport code
}


def _remap_abbr(series: pd.Series) -> pd.Series:
    return series.map(lambda x: _WNBA_TEAM_ABBR_MAP.get(x, x))


def _shot_zone_shares(shots: pd.DataFrame) -> pd.DataFrame:
    """Each player's share of her attempts by zone, beside her accent-free name and our team code.

    A player without an attempt is left out.
    """
    attempts = shots["FGA"].fillna(0)
    counted = attempts.iloc[:, :_COUNTED_ZONES].sum(axis=1)
    shares = attempts.iloc[:, list(_ZONE_SHARE_COLUMNS.values())].div(counted, axis=0)
    shares.columns = list(_ZONE_SHARE_COLUMNS)
    shares["PLAYER_NAME"] = shots["PLAYER_NAME"].map(remove_accents)
    shares["TEAM_ABBREVIATION"] = _remap_abbr(shots["TEAM_ABBREVIATION"])
    return shares.loc[counted > 0]


# Calendar month (inclusive) at which playoff logs are fetched in addition to regular season.
_PLAYOFF_MONTH = 9
# Days between latest_date and today that also triggers a playoff-log fetch mid-season.
_PLAYOFF_LAG_DAYS = 150
# WNBA playoff series formats: first round best-of-3, semifinals + finals best-of-5.
# The round isn't in the game ID, so it's inferred from the count of distinct prior
# opponents this postseason (each prior round = one more opponent already faced).
_WNBA_FIRST_ROUND_WINS = 2
_WNBA_LATER_ROUND_WINS = 3
# A WNBA postseason spans ~6 weeks; this window brackets it (and stays well short of a
# year) so the round inference only counts opponents from the current postseason.
_POSTSEASON_LOOKBACK_DAYS = 75


class StatsWNBA(StatsNBA):
    """WNBA player statistics, routed to stats.wnba.com via nba_client."""

    regulation_minutes = _REGULATION_MINUTES

    def __init__(self):
        super().__init__()
        self.league = "WNBA"
        self.positions = ["G", "F", "C"]
        self._set_season_start(datetime(2026, 5, 15).date())
        self.default_total = 81.667

        self.gamelog.columns = [stat.replace("_48", "_40") for stat in self.gamelog.columns]
        self.stat_types = [stat.replace("_48", "_40") for stat in self.stat_types]

    def _set_season_start(self, day):
        self.season_start = day
        self.season = day.year

    def _series_games_to_win(self, playoff_teamlog, team, opp, date):
        """Wins to clinch this WNBA series: best-of-3 in the first round, best-of-5 in
        the semifinals + finals. The round is inferred from the count of distinct
        opponents this team has already faced this postseason (a prior-round proxy,
        since the round isn't encoded in the game ID).
        """
        team_col, opp_col, date_col = (
            self.log_strings["team"],
            self.log_strings["opponent"],
            self.log_strings["date"],
        )
        tl_dates = pd.to_datetime(playoff_teamlog[date_col]).dt.date
        this_postseason = playoff_teamlog.loc[
            (playoff_teamlog[team_col] == team)
            & (tl_dates < date)
            & (tl_dates >= date - timedelta(days=_POSTSEASON_LOOKBACK_DAYS))
        ]
        prior_rounds = this_postseason.loc[this_postseason[opp_col] != opp, opp_col].nunique()
        return _WNBA_FIRST_ROUND_WINS if prior_rounds == 0 else _WNBA_LATER_ROUND_WINS

    def load(self) -> None:
        """Load data from files."""
        wnba_data = read_gamelog("wnba")
        self.players = wnba_data["players"]
        self.gamelog = wnba_data["gamelog"]
        self.teamlog = wnba_data["teamlog"]

    def _update(self) -> None:
        """Fetch new WNBA game logs from stats.wnba.com and merge into stored data."""
        today = datetime.today().date()
        player_df = self._fetch_player_index()
        bios, shots = self._fetch_player_bios(today - timedelta(days=1))
        player_df = self._build_wnba_player_profiles(player_df, bios, shots)

        # Drop records with incomplete advanced stats so they can be re-fetched
        if "OFF_RATING" in self.gamelog.columns:
            self.gamelog = self.gamelog.dropna(subset=["OFF_RATING"])

        latest_date = self.season_start
        if not self.gamelog.empty:
            latest_date = pd.to_datetime(self.gamelog[self.log_strings["date"]]).max().date()
            latest_date = max(latest_date, self.season_start)

        self.upcoming_games, self.upcoming_games_by_date = fetch_upcoming_games(
            self.league, self.season_start.year, today
        )

        (nba_gamelog, adv_gamelog, usg_gamelog, teamlog, sco_teamlog, adv_teamlog) = (
            self._fetch_wnba_logs(latest_date, today)
        )

        stat_df = self._assemble_wnba_player_rows(nba_gamelog, adv_gamelog, usg_gamelog, player_df)
        if not stat_df.empty:
            self._enrich_team_markets(
                stat_df,
                date_col=self.log_strings["date"],
                team_col="TEAM_ABBREVIATION",
            )
            self.gamelog = (
                pd.concat([stat_df[self.gamelog.columns], self.gamelog])
                .sort_values(self.log_strings["date"])
                .reset_index(drop=True)
            )

        team_df = self._assemble_wnba_team_rows(teamlog, sco_teamlog, adv_teamlog)
        if not team_df.empty:
            self.teamlog = (
                pd.concat([team_df[self.teamlog.columns], self.teamlog])
                .sort_values(self.log_strings["date"])
                .reset_index(drop=True)
            )

        self.gamelog.drop_duplicates(subset=["PLAYER_ID", "GAME_ID"], keep="last", inplace=True)
        self.teamlog.drop_duplicates(subset=["TEAM_ID", "GAME_ID"], keep="last", inplace=True)

        if self._full_clean_due():
            self._enrich_team_markets(
                self.gamelog,
                date_col=self.log_strings["date"],
                team_col="TEAM_ABBREVIATION",
            )
            self.gamelog["GAME_DATE"] = self.gamelog["GAME_DATE"].astype(str)
            self.teamlog["GAME_DATE"] = self.teamlog["GAME_DATE"].astype(str)

        write_gamelog("wnba", self.gamelog, self.teamlog, self.players)

    def _fetch_player_index(self) -> pd.DataFrame:
        """Player index (id, name, team, position) from the WNBA stats endpoint."""
        pos_map = {"G-F": "G", "F-G": "F", "F-C": "C"}
        try:
            player_df = nba_client.fetch(
                nba.playerindex.PlayerIndex,
                host=nba_client.WNBA_HOST,
                season=self.season_start.year,
                league_id="10",
                historical_nullable=1,
            ).get_normalized_dict()["PlayerIndex"]
            player_df = pd.DataFrame(player_df).rename(
                columns={"POSITION": "POS", "Position": "POS", "PERSON_ID": "PLAYER_ID"}
            )
            player_df.TEAM_ABBREVIATION = _remap_abbr(player_df.TEAM_ABBREVIATION)
            player_df.POS = player_df.POS.apply(lambda x: pos_map.get(x, x))
        except NBAStatsError:
            player_df = pd.DataFrame(
                columns=["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "POS"]
            )
        return player_df

    def _fetch_player_bios(self, day: date) -> tuple[pd.DataFrame, pd.DataFrame]:
        """The season's bio and shot-location tables through ``day`` (empty on API error)."""
        try:
            return nba_season_tables.read(self.league, self.season, day)
        except NBAStatsError:
            return pd.DataFrame(), pd.DataFrame()

    def _build_wnba_player_profiles(self, player_df, bios, shots) -> pd.DataFrame:
        """Merge bios + shot-zone shares into per-team player profiles in self.players.

        Returns the player frame augmented with profile columns (or unchanged when
        the bios table holds nobody).
        """
        if bios.empty:
            return player_df
        bios = bios.assign(PLAYER_NAME=bios.PLAYER_NAME.map(remove_accents))
        player_df = player_df.merge(bios, on="PLAYER_ID", suffixes=(None, "_y")).merge(
            _shot_zone_shares(shots),
            on=["PLAYER_NAME", "TEAM_ABBREVIATION"],
            suffixes=(None, "_y"),
        )
        player_df.PLAYER_WEIGHT = player_df.PLAYER_WEIGHT.astype(float)
        player_df.POS = player_df.POS.str[0]
        player_df.index = player_df.PLAYER_NAME
        player_df["PLAYER_BMI"] = (
            player_df["PLAYER_WEIGHT"]
            / player_df["PLAYER_HEIGHT_INCHES"]
            / player_df["PLAYER_HEIGHT_INCHES"]
        )
        wnba_numeric_cols = ["AGE", "PLAYER_HEIGHT_INCHES", "PLAYER_BMI", *_DATED_COMP_FIELDS]
        player_dict = player_df.groupby("TEAM_ABBREVIATION")[["POS", *wnba_numeric_cols]].apply(
            lambda x: x
        )
        player_dict[wnba_numeric_cols] = player_dict[wnba_numeric_cols].replace(
            [np.nan, np.inf, -np.inf], 0
        )

        player_dict = {
            level: player_dict.xs(level).T.to_dict() for level in player_dict.index.levels[0]
        }
        self._merge_season_roster(self.season_start.year, player_dict)
        return player_df

    def _fetch_wnba_logs(self, latest_date, today):
        """Fetch player + team game logs (regular + conditional playoffs) as six lists.

        All-or-nothing: any endpoint failure yields six empty lists, never a
        partial mix of populated and empty logs.
        """
        params = {
            "season_nullable": self.season_start.year,
            "league_id_nullable": "10",
            "date_from_nullable": latest_date.strftime("%m/%d/%Y"),
            "date_to_nullable": today.strftime("%m/%d/%Y"),
        }

        def _player_logs(measure: str | None = None) -> list:
            extra = {} if measure is None else {"measure_type_player_game_logs_nullable": measure}
            return nba_client.fetch(
                nba.playergamelogs.PlayerGameLogs, host=nba_client.WNBA_HOST, **(params | extra)
            ).get_normalized_dict()["PlayerGameLogs"]

        def _team_logs(measure: str | None = None) -> list:
            extra = {} if measure is None else {"measure_type_player_game_logs_nullable": measure}
            return nba_client.fetch(
                nba.teamgamelogs.TeamGameLogs, host=nba_client.WNBA_HOST, **(params | extra)
            ).get_normalized_dict()["TeamGameLogs"]

        try:
            nba_gamelog = _player_logs()
            adv_gamelog = _player_logs("Advanced")
            usg_gamelog = _player_logs("Usage")
            teamlog = _team_logs()
            sco_teamlog = _team_logs("Scoring")
            adv_teamlog = _team_logs("Advanced")

            if (today.month >= _PLAYOFF_MONTH) or (today - latest_date).days > _PLAYOFF_LAG_DAYS:
                params.update({"season_type_nullable": "Playoffs"})
                nba_gamelog.extend(_player_logs())
                adv_gamelog.extend(_player_logs("Advanced"))
                usg_gamelog.extend(_player_logs("Usage"))
                teamlog.extend(_team_logs())
                sco_teamlog.extend(_team_logs("Scoring"))
                adv_teamlog.extend(_team_logs("Advanced"))
        except NBAStatsError:
            return [], [], [], [], [], []
        return nba_gamelog, adv_gamelog, usg_gamelog, teamlog, sco_teamlog, adv_teamlog

    def _assemble_wnba_player_rows(self, nba_gamelog, adv_gamelog, usg_gamelog, player_df):
        """Merge base + advanced + usage player logs into the gamelog-shaped frame."""
        if not (nba_gamelog and adv_gamelog and usg_gamelog):
            return pd.DataFrame()

        nba_gamelog.sort(key=lambda x: (x["GAME_ID"], x["PLAYER_ID"]))
        adv_gamelog.sort(key=lambda x: (x["GAME_ID"], x["PLAYER_ID"]))
        usg_gamelog.sort(key=lambda x: (x["GAME_ID"], x["PLAYER_ID"]))

        stat_df = (
            pd.DataFrame(nba_gamelog)
            .merge(pd.DataFrame(adv_gamelog), on=["PLAYER_ID", "GAME_ID"], suffixes=[None, "_y"])
            .merge(pd.DataFrame(usg_gamelog), on=["PLAYER_ID", "GAME_ID"], suffixes=[None, "_y"])
        )
        stat_df = stat_df[[col for col in stat_df.columns if "_y" not in col]]
        stat_df.drop_duplicates(subset=["PLAYER_ID", "GAME_ID"], keep="last", inplace=True)
        stat_df.PLAYER_NAME = stat_df.PLAYER_NAME.apply(remove_accents)
        stat_df = stat_df.loc[stat_df["MIN"] > 1]
        stat_df = stat_df.loc[~stat_df.TEAM_ABBREVIATION.isna()]
        if not self.gamelog.empty:
            existing = set(
                self.gamelog[["PLAYER_ID", "GAME_ID"]].itertuples(index=False, name=None)
            )
            stat_df = stat_df[
                ~stat_df.apply(lambda x: (x["PLAYER_ID"], x["GAME_ID"]) in existing, axis=1)
            ]

        stat_df["HOME"] = stat_df.MATCHUP.str.contains(" vs. ")
        stat_df = stat_df.merge(player_df[["PLAYER_ID", "POS"]], on="PLAYER_ID")
        stat_df["OPP"] = stat_df.MATCHUP.apply(lambda x: x[x.rfind(" ") :].strip())

        stat_df["PRA"] = stat_df["PTS"] + stat_df["REB"] + stat_df["AST"]
        stat_df["PR"] = stat_df["PTS"] + stat_df["REB"]
        stat_df["PA"] = stat_df["PTS"] + stat_df["AST"]
        stat_df["RA"] = stat_df["REB"] + stat_df["AST"]
        stat_df["BLST"] = stat_df["BLK"] + stat_df["STL"]
        stat_df["fantasy points prizepicks"] = (
            stat_df["PTS"]
            + stat_df["REB"] * 1.2
            + stat_df["AST"] * 1.5
            + stat_df["BLST"] * 3
            - stat_df["TOV"]
        )
        stat_df["fantasy points underdog"] = (
            stat_df["PTS"]
            + stat_df["REB"] * 1.2
            + stat_df["AST"] * 1.5
            + stat_df["BLST"] * 3
            - stat_df["TOV"]
        )
        stat_df["fantasy points parlay"] = stat_df["PRA"] + stat_df["BLST"] * 2 - stat_df["TOV"]
        stat_df["FTR"] = (stat_df["FTM"] / stat_df["FGA"]).where(stat_df["FGA"] > 0, 0)
        stat_df["FG3_RATIO"] = (stat_df["FG3A"] / stat_df["FGA"]).where(stat_df["FGA"] > 0, 0)
        stat_df["BLK_PCT"] = (stat_df["BLK"] / stat_df["BLKA"]).where(stat_df["BLKA"] > 0, 0)
        stat_df["FGA_40"] = stat_df["FGA"] / stat_df["MIN"] * _REGULATION_MINUTES
        stat_df["FG3A_40"] = stat_df["FG3A"] / stat_df["MIN"] * _REGULATION_MINUTES
        stat_df["REB_40"] = stat_df["REB"] / stat_df["MIN"] * _REGULATION_MINUTES
        stat_df["OREB_40"] = stat_df["OREB"] / stat_df["MIN"] * _REGULATION_MINUTES
        stat_df["DREB_40"] = stat_df["DREB"] / stat_df["MIN"] * _REGULATION_MINUTES
        stat_df["AST_40"] = stat_df["AST"] / stat_df["MIN"] * _REGULATION_MINUTES
        stat_df["TOV_40"] = stat_df["TOV"] / stat_df["MIN"] * _REGULATION_MINUTES
        stat_df["BLKA_40"] = stat_df["BLKA"] / stat_df["MIN"] * _REGULATION_MINUTES
        stat_df["STL_40"] = stat_df["STL"] / stat_df["MIN"] * _REGULATION_MINUTES
        stat_df.TEAM_ABBREVIATION = _remap_abbr(stat_df.TEAM_ABBREVIATION)
        stat_df.OPP = _remap_abbr(stat_df.OPP)

        stat_df["GAME_DATE"] = pd.to_datetime(stat_df["GAME_DATE"]).astype(str)
        return stat_df

    def _assemble_wnba_team_rows(self, teamlog, sco_teamlog, adv_teamlog):
        """Merge base + scoring + advanced team logs and pair home/away OPP_ columns."""
        if not (teamlog and sco_teamlog and adv_teamlog):
            return pd.DataFrame()

        teamlog.sort(key=lambda x: (x["GAME_ID"], x["TEAM_ID"]))
        sco_teamlog.sort(key=lambda x: (x["GAME_ID"], x["TEAM_ID"]))
        adv_teamlog.sort(key=lambda x: (x["GAME_ID"], x["TEAM_ID"]))

        team_df = (
            pd.DataFrame(teamlog)
            .merge(pd.DataFrame(adv_teamlog), on=["TEAM_ID", "GAME_ID"], suffixes=[None, "_y"])
            .merge(pd.DataFrame(sco_teamlog), on=["TEAM_ID", "GAME_ID"], suffixes=[None, "_y"])
        )
        team_df = team_df[[col for col in team_df.columns if "_y" not in col]]
        team_df = team_df.loc[~team_df.TEAM_ABBREVIATION.isna()]

        team_df["HOME"] = team_df.MATCHUP.str.contains(" vs. ")
        team_df["OPP"] = team_df.MATCHUP.apply(lambda x: x[x.rfind(" ") :].strip())
        team_df["FTR"] = (team_df["FTM"] / team_df["FGA"]).where(team_df["FGA"] > 0, 0)
        team_df["BLK_RATIO"] = (team_df["BLK"] / team_df["BLKA"]).where(team_df["BLKA"] > 0, 0)
        team_df.TEAM_ABBREVIATION = _remap_abbr(team_df.TEAM_ABBREVIATION)
        team_df.OPP = _remap_abbr(team_df.OPP)

        stats = [stat for stat in self.teamlog.columns if "OPP_" in stat]
        home_teams = team_df.loc[team_df.HOME]
        home_teams.index = home_teams.GAME_ID
        away_teams = team_df.loc[~team_df.HOME]
        away_teams.index = away_teams.GAME_ID
        home_teams = home_teams.join(away_teams.add_prefix("OPP_")[stats])
        away_teams = away_teams.join(home_teams.add_prefix("OPP_")[stats])
        team_df = pd.concat([home_teams, away_teams], ignore_index=True)

        team_df["GAME_DATE"] = pd.to_datetime(team_df["GAME_DATE"]).astype(str)
        return team_df

    def _comp_season_pair(self) -> tuple[int, int]:
        return self.season_start.year - 1, self.season_start.year

    def _current_season_key(self, target_game_date: date) -> int:
        """WNBA override: integer-year season keys.

        WNBA seasons run May–Oct of a single calendar year, so the season
        containing ``target_game_date`` is just its calendar year. ``int``
        return type lets the inherited :meth:`StatsNBA._player_seasons_through`
        compare against the int keys in ``self.players``.
        """
        return target_game_date.year

    # Cleaning the Glass covers NBA only; WNBA has no such source, so both hooks
    # fall back to the base no-op instead of inheriting StatsNBA's CTG wiring.
    def _join_fp_player_features(self, date):
        return None

    def _join_fp_team_features(self, date):
        return None, None

    # WNBA's absences are unscreened, so the hook falls back to the base no-op instead of
    # inheriting StatsNBA's absence columns.
    def _availability_context(self, stats, date, teams, opponents):
        pass

    def _dated_comp_fields(self, bios: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
        """``_DATED_COMP_FIELDS`` from a day's bio and shot tables, by accent-free player name.

        A player the bios list without an attempt reads zero shares.
        """
        bios = bios.assign(PLAYER_NAME=bios.PLAYER_NAME.map(remove_accents))
        tables = bios.merge(
            _shot_zone_shares(shots), on="PLAYER_NAME", suffixes=(None, "_y"), how="outer"
        ).drop_duplicates("PLAYER_NAME")
        return tables.set_index("PLAYER_NAME")[list(_DATED_COMP_FIELDS)].fillna(0)
