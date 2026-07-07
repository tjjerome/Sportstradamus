"""Stats base class: shared data loading, feature engineering, and prediction API."""

import importlib.resources as pkg_resources
import json
import os.path
import pickle
import warnings
from datetime import date, datetime, timedelta
from io import StringIO
from time import sleep

import line_profiler
import nba_api.stats.endpoints as nba
import nfl_data_py as nfl
import nflreadpy as nflr
import numpy as np
import pandas as pd
import requests
import statsapi as mlb
from scipy.stats import iqr, norm, poisson
from sklearn.neighbors import BallTree
from tqdm import tqdm

from sportstradamus import data
from sportstradamus.helpers import (
    LazyArchive,
    Scrape,
    abbreviations,
    combo_props,
    feature_filter,
    get_ev,
    get_mlb_pitchers,
    get_odds,
    remove_accents,
    set_model_start_values,
    stat_cv,
    stat_dist,
)
from sportstradamus.helpers.archive import TRAINING_LOOKBACK
from sportstradamus.helpers.io import read_gamelog
from sportstradamus.spiderLogger import logger

# Safety ceiling for comp z-scores. Anything beyond ±5σ is already noise;
# ±10 is generous and acts as a guard against near-zero (but non-NaN) stds
# produced by 2-comp players whose means nearly coincide. See profile_market.
_COMP_Z_CLIP: float = 10.0

# Empirical-Bayes prior count for the ``comps mean (EB)`` shrinkage. The
# shrinkage weight is ``K / (K + n_eff)`` where ``n_eff`` is the sum of
# distance-weighted comp memberships (Efron & Morris 1975 JASA). At the
# NFL-weekly granularity Albert (2008) and the comp-research brief
# recommend K in the 8-12 band; 10 is the middle of that band.
_COMP_EB_PRIOR_K: float = 10.0

# Empirical-Bayes prior count for the career expanding-mean shrinkage toward the
# per-position baseline (same K/(K+n) Efron-Morris form as _COMP_EB_PRIOR_K). K is
# the game count at which a player's own career mean and the position prior carry
# equal weight. Cross-validated 2026-06-20 (held-out purged GroupKFold OOF +
# deterministic model confirm): NFL -> 1 (the fitted gates prefer the lighter
# shrinkage; carries min_gate_slack +2.96 vs K=10), other leagues keep 10 (WNBA's
# gates prefer the heavier prior, NBA near-inert). Per-league override below.
_EXPANDING_EB_PRIOR_K: float = 10.0
_EXPANDING_EB_PRIOR_K_BY_LEAGUE: dict[str, float] = {"NFL": 1.0}

# Quantile bands for the ``comps p25`` / ``comps p75`` signals. Use the
# distance-weighted comp distribution rather than the raw comp set: a
# closer comp gets more vote, matching how ``comps mean`` is built.
_COMP_QUANTILE_LO: float = 0.25
_COMP_QUANTILE_HI: float = 0.75

# Half-life (days) for recency-weighting comp (comp, opponent) outcomes in the
# matchup-conditional comp lookup. A comp's game decays as
# ``exp(-days_ago / halflife)`` so recent matchups dominate the raw/z estimate.
# Cross-validated 2026-06-20 (held-out OOF): inert -- ``Player comps z recent`` is a
# recency-reweighted convex combination, so H only re-weights within the mean (OOF
# R² spread <= 0.0006 across H in {14..180}, all leagues); kept at 45.
_COMP_RECENCY_HALFLIFE_DAYS: float = 45.0

# Clip bounds for per-player projection scale ratios in get_volume_stats.
# Floor prevents shrinking a player below 10% of their mean; cap prevents
# inflating beyond 10×. Shared across NBA, WNBA, NHL, and NFL volume models.
_VOLUME_SCALE_RATIO_FLOOR: float = 0.1
_VOLUME_SCALE_RATIO_CAP: float = 10.0

# Players with fewer than this many games get a zero trend — the vectorized
# reimplementation of helpers.text.get_trends, which returns zeros below the
# same floor (the trailing-window slope is too noisy on <3 points).
_MIN_GAMES_FOR_TRENDS: int = 3

# Participation floor for defense/usage profiling: keep only players whose
# market hit-rate (per-game value clipped to [0, 1], averaged) clears this.
# Below it the player barely sees the market and only adds profile noise.
_MIN_PARTICIPATION_RATE: float = 0.1

# Month (1-12) on/after which a league's season is labeled by the starting
# calendar year. Aug+ rolls the season year forward for fall/winter leagues.
_SEASON_ROLLOVER_MONTH: int = 8

# Days ahead (including today) that fetch_upcoming_games scrapes. Three days
# covers today + tomorrow + day-after so late-night runs don't miss next-day
# tip-offs.
_SCHEDULE_LOOKAHEAD_DAYS: int = 3

# A "blowout" is a game lopsided enough to suppress a starter's volume. Each
# threshold is the |implied spread| (own minus opponent implied total, league
# units) at which per-player-normalized starter playing time first drops >~4%
# and keeps falling. Derived from the gamelog: the effect is FAVORITE-ONLY (the
# favorite rests starters; underdog starters play full/more minutes chasing) —
# the signed Spread feature carries that asymmetry, so this stays |spread|-based
# as a sign-agnostic regime flag. NBA/WNBA knee ~10 pts (≈ p80 spread); NFL ~11
# (snaps resist past p80). NHL shows no playing-time effect (TimeShare flat/rising
# in blowouts) so its value is a nominal lopsidedness marker (~p90, expect Blowout
# inert and reverted at validation); MLB has no local line history to fit, so 2.5
# runs is a reasoned default, not derived.
_BLOWOUT_SPREAD_THRESHOLD: dict[str, float] = {
    "NBA": 10.0,
    "WNBA": 10.0,
    "NFL": 11.0,
    "NHL": 1.5,
    "MLB": 2.5,
}

# Postseason decode for the leagues whose native game ID encodes the season type
# (verified against the cached gamelogs): NBA/WNBA carry it in the 3rd character,
# NHL in characters 5-6. The value is ``(slice_start, slice_stop, postseason_codes)``.
# Postseason codes: NBA playoffs(4) + play-in(5); WNBA playoffs(4) only -- its 5 is
# the Commissioner's Cup, not a bracket game; NHL playoffs(03). Every other code is
# regular or exhibition (preseason, All-Star, NBA Cup, 4 Nations) and flags 0, which
# is exactly the exclusion we want. NFL/MLB carry no such ID and override _playoff_flag.
_PLAYOFF_GAMEID_CODE: dict[str, tuple[int, int, frozenset[str]]] = {
    "NBA": (2, 3, frozenset({"4", "5"})),
    "WNBA": (2, 3, frozenset({"4"})),
    "NHL": (4, 6, frozenset({"03"})),
}

# Series context (playoff contention). The within-series W-L record is tallied from
# prior playoff teamlog games against the same opponent inside this lookback window;
# a best-of-7 spans ~2 weeks, so 30 days isolates the current series from any
# prior-season meeting against the same team (two teams meet in at most one series
# per postseason, so no same-postseason cross-round collision).
_SERIES_LOOKBACK_DAYS: int = 30
# Wins to clinch the default playoff series: NBA and NHL are best-of-7 every round.
# WNBA (round-dependent) and MLB (round-dependent) override _series_games_to_win.
_SERIES_DEFAULT_GAMES_TO_WIN: int = 4
_SERIES_FEATURE_COLS = ("SeriesWins", "SeriesLosses", "FacingElimination", "CanClinch")


# Columns of the empty per-player feature frame get_stats seeds before populating
# rows -- and the frame it returns when no requested player has game history.
_BASE_STAT_COLUMNS = [
    "Avg1",
    "Avg3",
    "Avg5",
    "Avg10",
    "AvgYr",
    "AvgH2H",
    "Mean10",
    "MeanYr",
    "MeanYr_nonzero",
    "MeanYr_expanding_shifted",
    "MeanYr_expanding_eb",
    "MeanYr_expanding_vsopp",
    "MeanH2H",
    "STD10",
    "STDYr",
    "ZeroYr",
    "DaysOff",
    "DaysIntoSeason",
    "GamesPlayed",
    "H2HPlayed",
    "Home",
    "Moneyline",
    "Total",
    "OppTotal",
    "Spread",
    "GameTotal",
    "Blowout",
    "Playoff",
    "SeriesWins",
    "SeriesLosses",
    "FacingElimination",
    "CanClinch",
]
# LazyArchive defers DuckDB lock acquisition until the first attribute
# access so processes that merely import this module — most importantly
# the long-lived Streamlit dashboard — do not hold the archive lock for
# their entire lifetime. See LazyArchive docstring in helpers/archive.py.
archive = LazyArchive()
scraper = Scrape()

# When True, forces a full reload of all cached CSVs on the next load() /
# update() call. Flipped manually by operators to clear corrupt gamelog state.
clean_data = False
pd.set_option("future.no_silent_downcasting", True)


def _profile_rows_for_teams(profile: pd.DataFrame, team_keys: pd.Index | list[str]) -> pd.DataFrame:
    """Look up ``profile`` rows for ``team_keys``, tolerating any absent team.

    A mid-season expansion franchise (the 2025 Golden State Valkyries, the 2026
    Toronto Tempo) shows up in the gamelog before it accrues the teamlog history
    that builds ``teamProfile`` / ``defenseProfile``; a plain ``.loc`` lookup then
    raises ``KeyError``. ``reindex`` substitutes an all-NaN row, which LightGBM
    consumes as a missing feature — the honest representation for a team with no
    history. Replaces the former per-team hardcoded ``"GSV"`` guards.
    """
    return profile.reindex(team_keys)


def _zscore_within_player(df: pd.DataFrame, player_col: str, market: str) -> pd.Series:
    """Z-score each row's ``market`` value against that player's own mean and std.

    Players with zero spread map to NaN before the divide and then fill 0, so a
    player with a flat history contributes a neutral score rather than a divide-by-
    zero. Casts ``df[market]`` to float in place; returns the z-scored series.
    """
    df[market] = df[market].astype(float)
    groups = df.groupby(player_col)[market]
    std = groups.transform("std").replace(0, np.nan)
    return ((df[market] - groups.transform("mean")) / std).fillna(0)


# Substrings that mark an MLB market as pitcher-side ("earned runs allowed",
# "pitches thrown", "1st inning allowed", ...) rather than batter-side. Shared
# by every MLB pitcher/batter branch point across base.py and mlb.py.
_MLB_PITCHER_MARKET_TOKENS: tuple[str, ...] = ("allowed", "pitch")


def is_mlb_pitcher_market(market: str) -> bool:
    """Return True if ``market`` is an MLB pitcher-side stat, False for batter-side."""
    return any(token in market for token in _MLB_PITCHER_MARKET_TOKENS)


def flatten_offers(offers: dict | list, market: str) -> dict | list:
    """Flatten a nested ``{group: {player: ...}}`` offers mapping to ``{player: ...}``.

    Merges every group's player dict, then re-applies the market-keyed group last
    so its entries win. A non-dict ``offers`` (already flat) is returned unchanged.
    """
    if not isinstance(offers, dict):
        return offers
    flat: dict = {}
    for players in offers.values():
        flat.update(players)
    flat.update(offers.get(market, {}))
    return flat


def scale_team_volume_to_budget(
    player_profile: pd.DataFrame,
    market: str,
    *,
    budget_mean: float,
    typical_rotation: int,
    avg_unmodeled_min: float,
    per_player_floor: float,
    per_player_cap: float,
) -> None:
    """Rescale per-player SkewNormal volume projections to a team minutes budget.

    For each team, distributes the (budget - modeled-total) deficit across the
    team's modeled players precision-weighted by variance, clips each player to
    ``per_player_cap``, and rescales loc/scale together to preserve every
    player's coefficient of variation. Mutates ``player_profile`` in place.
    Shared by the NBA and NHL volume models, which differ only in the per-league
    ``budget_mean`` derivation and the rotation/floor/cap parameters.

    Two real-world constraints complicate a simple minutes cap:

    1. Overtime: games sometimes go beyond regulation. With probability p_ot per
       period, each OT adds ot_per_period player-minutes. Modelling this as a
       geometric process gives:
           budget_mean = reg_minutes + ot_per_period * p_ot / (1 - p_ot)

    2. Unmodeled players: incomplete roster coverage (rookies, fringe benchwarmers)
       means some minutes go to players we have no projection for. We reserve:
           unmodeled_reserve = max(0, typical_rotation - N) * avg_unmodeled_min

    The precision-weighted adjustment is the minimum-information-loss solution to:
        minimise  sum_i (d_i / sigma_i)^2
        subject to  sum_i (mu_i + d_i) = target
        -> d_i = sigma_i^2 / sum_j(sigma_j^2) * (target - total)
    Players we are most uncertain about absorb the correction; confidently
    projected players are left largely untouched.

    Args:
        player_profile: DataFrame with columns ``"team"``, ``"proj {market} loc"``,
            ``"proj {market} scale"``, and optionally ``"proj {market} alpha"``.
            Rows with ``team == 0`` are skipped (unassigned players).
            Writes back ``"proj {market} loc"``, ``"proj {market} scale"``,
            ``"proj {market} mean"``, and ``"proj {market} std"`` in place.
        market: Stat name used to construct the column key (e.g. ``"MIN"``).
        budget_mean: Expected total team volume (minutes or TOI) per game,
            including overtime expectation.  Units match ``per_player_cap``.
        typical_rotation: Median number of modeled players per team per game;
            used to size the unmodeled reserve.
        avg_unmodeled_min: Mean volume for players outside the modeled tier;
            units match ``budget_mean``.
        per_player_floor: Minimum per-player volume used to set the lower target
            (``N * per_player_floor``).  Units match ``budget_mean``.
        per_player_cap: Hard ceiling clipped onto each player's new mean before
            ratio rescaling.
    """
    teams = player_profile.loc[player_profile["team"] != 0].groupby("team")
    for _team, team_df in teams:
        loc = team_df[f"proj {market} loc"].copy()
        scale = team_df[f"proj {market} scale"].copy()
        sn_alpha = (
            team_df[f"proj {market} alpha"].copy()
            if f"proj {market} alpha" in team_df.columns
            else pd.Series(0, index=loc.index, dtype=loc.dtype)
        )
        N = len(team_df)

        delta = sn_alpha / np.sqrt(1 + sn_alpha**2)
        true_means = loc + scale * delta * np.sqrt(2 / np.pi)
        true_vars = scale**2

        total = true_means.sum()
        if total <= 0:
            continue

        unmodeled_count = max(0, typical_rotation - N)
        unmodeled_reserve = unmodeled_count * avg_unmodeled_min

        upper_target = budget_mean - unmodeled_reserve
        lower_target = N * per_player_floor
        target = max(upper_target, lower_target)

        deficit = target - total
        total_var = true_vars.sum()
        if total_var > 0:
            adjustments = true_vars / total_var * deficit
        else:
            adjustments = true_means / total * deficit

        new_means = (true_means + adjustments).clip(lower=0, upper=per_player_cap)

        ratio = new_means / true_means.replace(0, np.nan)
        ratio = ratio.fillna(1.0).clip(
            lower=_VOLUME_SCALE_RATIO_FLOOR, upper=_VOLUME_SCALE_RATIO_CAP
        )
        new_scale = scale * ratio
        new_loc = new_means - new_scale * delta * np.sqrt(2 / np.pi)

        player_profile.loc[loc.index, f"proj {market} loc"] = new_loc
        player_profile.loc[scale.index, f"proj {market} scale"] = new_scale
        player_profile.loc[loc.index, f"proj {market} mean"] = new_means
        player_profile.loc[scale.index, f"proj {market} std"] = new_scale


def fetch_upcoming_games(league_id: str, season: str | int, today: date) -> dict[str, dict]:
    """Map each team to its next matchup from the stats.nba.com schedule.

    Scrapes the ``internationalbroadcasterschedule`` endpoint for ``today`` and
    the next two days. Shared by the NBA and WNBA ``update`` paths, which differ
    only in ``league_id`` ("00" vs "10") and the ``season`` value interpolated
    into the URL. Returns ``{team_abbr: {"Opponent": str, "Home": bool}}`` keyed
    by the first date a team is seen.

    ``Scrape.get`` swallows network errors and returns ``{}`` on exhausted
    retries, so an unavailable endpoint surfaces here as a ``KeyError`` while
    indexing the empty response; that and the sibling ``IndexError`` /
    ``TypeError`` from a malformed payload yield an empty mapping. This is the
    historical bare-except behavior, narrowed to the exceptions the block can
    actually raise (§ E722).
    """
    upcoming_games: dict[str, dict] = {}
    try:
        ug_res = []
        for offset in range(_SCHEDULE_LOOKAHEAD_DAYS):
            day = today + timedelta(days=offset)
            ug_url = (
                "https://stats.nba.com/stats/internationalbroadcasterschedule"
                f"?LeagueID={league_id}&Season={season}&RegionID=1"
                f"&Date={day.strftime('%m/%d/%Y')}&EST=Y"
            )
            ug_res.extend(scraper.get(ug_url)["resultSets"][1]["CompleteGameList"])

        for game in ug_res:
            if game["vtAbbreviation"] not in upcoming_games:
                upcoming_games[game["vtAbbreviation"]] = {
                    "Opponent": game["htAbbreviation"],
                    "Home": False,
                }
            if game["htAbbreviation"] not in upcoming_games:
                upcoming_games[game["htAbbreviation"]] = {
                    "Opponent": game["vtAbbreviation"],
                    "Home": True,
                }
    except (KeyError, IndexError, TypeError):
        pass
    return upcoming_games


class Stats:
    """A parent class for handling and analyzing sports statistics.

    Attributes:
        gamelog (list): A list of dictionaries representing game logs.
        archive (dict): A dictionary containing archived statistics.
        players (dict): A dictionary containing player specific data.
        season_start (datetime): The start date of the  season.
        playerStats (dict): Dictionary containing player statistics.
        edges (list): List of edges.
        dvp_index (dict): Dictionary containing DVP index data.

    Methods:
        parse_game(game): Parses a game and updates the gamelog.
        load(): Loads game logs from a file.
        update(): Updates the gamelog with new game data.
        bucket_stats(market): Groups statistics into buckets based on market type.
        get_stats(offer, game_date): Retrieves statistics for a given offer and game date.
        get_training_matrix(market): Retrieves the training data matrix and target labels for a specified market.
    """

    def __init__(self):
        self.gamelog = pd.DataFrame()
        self.teamlog = pd.DataFrame()
        self.archive = {}
        self.players = {}
        self.season_start = datetime(year=1900, month=1, day=1).date()
        self.playerStats = {}
        self.edges = {}
        self.positions = []
        self.dvp_index = {}
        self.dvpoa_latest_date = datetime(year=1900, month=1, day=1).date()
        self.bucket_latest_date = datetime(year=1900, month=1, day=1).date()
        self.bucket_market = ""
        self.profile_latest_date = datetime(year=1900, month=1, day=1).date()
        self.profiled_market = ""
        self.volume_stats = []
        self.playerProfile = pd.DataFrame()
        self.defenseProfile = pd.DataFrame()
        self.teamProfile = pd.DataFrame()
        self.upcoming_games = {}
        self.usage_stat = ""
        self.tiebreaker_stat = ""
        self.comps = {}
        # Per-week comp cache (training matrix path). ``_ensure_comps(date)``
        # routes the active ``self.comps`` to the entry keyed by the
        # ``(season, week-of-Wednesday)`` bucket that contains ``date``, and
        # rebuilds on cache miss. ``date=None`` (inference / cron) keeps the
        # legacy lazy-once snapshot.
        self._comps_by_week: dict = {}
        self._current_comps_key = None
        # Per-comp-pool cache of the static (player, comp, position, dist,
        # weight) pairs DataFrame. Keyed by ``_current_comps_key`` so it
        # invalidates in lockstep with ``self.comps``. Lets ``base_profile``
        # and ``get_stats`` skip the Python tuple-loop rebuild per gameday.
        self._comp_pairs_cache: dict = {}

    def parse_game(self, game):
        """Parse a single game API response and update ``self.gamelog``.

        Args:
            game (dict): A dictionary representing a game.

        Returns:
            None
        """

    def load(self):
        """Read cached gamelog / teamlog / players artifacts for this league.

        The on-disk shape is uniform across leagues (one parquet/JSON bundle
        per league, see :mod:`sportstradamus.helpers.io`). Subclasses with
        league-specific post-processing (NFL roster validation, MLB
        affinity CSVs) override and either replace this body entirely or
        call ``super().load()`` first. The league key is derived from
        ``self.league`` so a single base implementation serves NBA, WNBA,
        and NHL — see :ref:`STYLE_GUIDE §2.6 <three-occurrences>`.
        """
        league_data = read_gamelog(self.league.lower())
        self.gamelog = league_data["gamelog"]
        self.teamlog = league_data["teamlog"]
        self.players = league_data["players"]

    def update(self):
        """Fetch new game data from the league API and append to ``self.gamelog``.

        Returns:
            None
        """

    def _enrich_team_markets(
        self,
        df: pd.DataFrame,
        *,
        date_col: str = "gameday",
        team_col: str = "team",
        mask: "np.ndarray | None" = None,
    ) -> None:
        """In-place: populate ``moneyline`` / ``totals`` columns from the archive.

        Two bulk DuckDB queries (one per market) replace the per-row
        ``archive.get_moneyline`` / ``archive.get_total`` ``DataFrame.apply``
        loops that previously dominated ``update()`` wall time — point
        lookups across a full-history gamelog ran into the millions of
        serial queries and took minutes. The bulk path is two queries plus
        an in-Python dict lookup per row.

        Args:
            df: Frame to enrich. Must have a date column and a team column;
                values are written back via ``df.loc[mask, col]``.
            date_col: Name of the date column (``gameday`` for NFL/MLB
                shaped frames, ``gameDate`` for MLB game logs,
                ``GAME_DATE`` for NBA-shaped frames).
            team_col: Name of the team column.
            mask: Optional boolean mask scoping enrichment to a subset.
                ``None`` enriches the whole frame.

        Fallback values match the per-row helpers: ``0.5`` for moneyline,
        :attr:`Archive.default_totals` per league for totals.
        """
        if df.empty:
            return
        if mask is None:
            mask = np.ones(len(df), dtype=bool)
        if not mask.any():
            return
        subset = df.loc[mask]
        unique_dates = subset[date_col].unique()
        ml_map = archive.get_team_market_map(self.league, "Moneyline", dates=unique_dates)
        tot_map = archive.get_team_market_map(self.league, "Totals", dates=unique_dates)
        default_total = archive.default_totals.get(self.league, 1)
        keys = list(zip(subset[date_col].astype(str).str[:10], subset[team_col], strict=False))
        df.loc[mask, "moneyline"] = [ml_map.get(k, 0.5) for k in keys]
        df.loc[mask, "totals"] = [tot_map.get(k, default_total) for k in keys]

    @staticmethod
    def _build_comps(knn, profile_df, min_comps=5, max_comps=20):
        """Build comp lists with distances using a hybrid k-NN + radius approach.
        Guarantees at least min_comps and caps at max_comps per player.
        Results are sorted by distance (closest first).
        """
        n_players = len(profile_df)
        # +1 to k-NN query to account for self being excluded
        k_floor = min(min_comps + 1, n_players)
        k_ceil = min(max_comps, n_players - 1)

        # Step 1: k-NN to get guaranteed minimum comps (includes self at distance 0)
        d_knn, i_knn = knn.query(profile_df.values, k=k_floor)
        # Step 2: radius from median of k-NN max distances
        r = np.quantile(np.max(d_knn, axis=1), 0.5)
        i_rad, d_rad = knn.query_radius(
            profile_df.values, r, sort_results=True, return_distance=True
        )

        players = profile_df.index
        comps = {}
        for j in range(n_players):
            # Merge k-NN and radius results, deduplicate, exclude self
            seen = {}
            for idx, dist in zip(i_knn[j], d_knn[j], strict=False):
                if idx != j:
                    seen[idx] = dist
            for idx, dist in zip(i_rad[j], d_rad[j], strict=False):
                if idx != j and (idx not in seen or dist < seen[idx]):
                    seen[idx] = dist
            sorted_pairs = sorted(seen.items(), key=lambda x: x[1])[:k_ceil]
            comps[players[j]] = {
                "comps": [str(players[idx]) for idx, _ in sorted_pairs],
                "distances": [round(float(dist), 4) for _, dist in sorted_pairs],
            }
        return comps

    def update_player_comps(self, year=None):
        return

    def _compute_comps(self, target_game_date: "date | None" = None) -> None:
        """Build comps from loaded data. Override in subclass.

        Args:
            target_game_date: Optional date the comps should reflect "as of".
                Subclasses with point-in-time profile assembly (currently NFL)
                use it to bound the comp pool to games strictly before this
                date. Subclasses that don't accept it must still tolerate the
                keyword argument so the training-matrix caller can pass it
                uniformly; for those leagues the comps remain today()-bound
                and the legacy lookahead leakage is not yet addressed.
        """

    def _comp_target_key(self, date: "date | None") -> "int | None":
        """Cache key for the comp pool active at ``date``.

        Default heuristic: the week index of the most-recent Wednesday on or
        before ``date``. Wednesday is the operator-confirmed retrain boundary;
        within a Wed→Wed window all gamedays share the same comp pool,
        matching live retrain cadence. Subclasses with schedule semantics
        (NFL) may override to return a (season, schedule-week) tuple.

        Returns ``None`` when ``date`` is ``None`` -- the inference path keys
        on the lazy-once snapshot instead.
        """
        if date is None:
            return None
        wed_offset = (date.weekday() - 2) % 7  # Monday=0..Sunday=6; Wednesday=2
        most_recent_wed = date - timedelta(days=wed_offset)
        return most_recent_wed.toordinal() // 7

    def _ensure_comps(self, date: "date | None" = None) -> None:
        """Materialize ``self.comps`` for the comp pool active at ``date``.

        Two modes:

        * ``date is None`` (inference, cron, dashboards): lazy-once. Build
          comps if ``self.comps`` is empty, otherwise reuse the cached
          snapshot for the lifetime of the ``Stats`` instance.
        * ``date`` provided (training matrix path): route ``self.comps`` to
          the per-week cache entry containing ``date``. Cache key comes from
          :meth:`_comp_target_key`. First visit to a (season, week) bucket
          calls :meth:`_compute_comps` with the date so per-league code can
          bound the comp pool point-in-time; subsequent visits to the same
          bucket are O(1) swaps. This eliminates the look-ahead leakage where
          a 2023 training row's comps "know" the 2025 player population.
        """
        # TODO(comp-leakage-mlb): StatsNFL, StatsNBA, StatsWNBA, and StatsNHL
        # all honor ``target_game_date`` and gate their comp pool on the
        # appropriate season-cut. StatsMLB still relies on the today() affinity
        # match-score CSVs (``affinity_pitchersBySHV_matchScores.csv`` /
        # ``affinity_hittersByHittingProfile_matchScores.csv``) loaded in
        # ``StatsMLB.load`` and has no ``_compute_comps`` override; the
        # training matrix path therefore reuses that fixed snapshot across
        # every gameday. A true fix needs historical baseballsavant exports
        # (the published CSV is current-state only) so this leakage stays
        # open until that data source is wired in.
        if date is None:
            if not self.comps:
                self._compute_comps()
            return

        target_key = self._comp_target_key(date)
        if self._current_comps_key == target_key and self.comps:
            return

        cached = self._comps_by_week.get(target_key)
        if cached is not None:
            self.comps = cached
            self._current_comps_key = target_key
            return

        self._compute_comps(target_game_date=date)
        self._comps_by_week[target_key] = self.comps
        self._current_comps_key = target_key

    @staticmethod
    def _flatten_comp_records(comps: dict) -> list[tuple]:
        """Flatten ``{position: {player: {comps, distances}}}`` to
        (player, comp, position, dist, weight) records, weight = 1/(1+dist).

        Skips non-dict levels and MLB-style list comp structures (which yield no
        records here). The Python tuple-loop runs once per comp-pool swap via the
        ``_comp_pairs`` cache, not once per gameday.
        """
        records = []
        for position, pos_comps in comps.items():
            if not isinstance(pos_comps, dict):
                continue
            for player, comp_data in pos_comps.items():
                if not isinstance(comp_data, dict):
                    continue
                comps_list = comp_data.get("comps")
                dist_list = comp_data.get("distances")
                if not comps_list or dist_list is None:
                    continue
                for comp, dist in zip(comps_list, dist_list, strict=False):
                    if comp == player:
                        continue
                    d = float(dist)
                    records.append((player, comp, position, d, 1.0 / (1.0 + d)))
        return records

    def _comp_pairs(self) -> pd.DataFrame:
        """Return the static (player, comp, position, dist, weight) pairs DataFrame.

        Cached per ``_current_comps_key`` so the Python tuple-loop that flattens
        ``self.comps`` only runs once per comp-pool swap (i.e. once per training
        week), not once per gameday. ``base_profile`` and ``get_stats`` consume
        this to derive the per-day comp features via vectorized ops.

        MLB-style comp structures (``{"pitchers": {pid: [id1, id2, ...]}}`` --
        lists rather than ``{"comps": ..., "distances": ...}`` dicts) yield an
        empty frame here; the MLB code path in ``get_stats`` operates directly
        on ``self.comps`` and does not call into this helper.
        """
        key = self._current_comps_key
        cached = self._comp_pairs_cache.get(key)
        if cached is not None:
            return cached

        records = self._flatten_comp_records(self.comps)
        if not records:
            pairs = pd.DataFrame(columns=["player", "comp", "position", "dist", "weight"])
        else:
            pairs = pd.DataFrame.from_records(
                records, columns=["player", "comp", "position", "dist", "weight"]
            )
        self._comp_pairs_cache[key] = pairs
        return pairs

    @staticmethod
    def _vectorized_weighted_quantiles(
        df_sorted: pd.DataFrame, qs: list[float]
    ) -> dict[float, pd.Series]:
        """Per-player distance-weighted quantile of ``comp_mean``.

        Replaces the per-group ``DataFrame.groupby.apply(_weighted_q)`` recipe
        with a single sort + cumsum + np.interp pass. ``df_sorted`` must hold
        ``['player', 'comp_mean', 'weight']`` already ordered by
        ``('player', 'comp_mean')``. Returns ``{q: Series(indexed by player)}``.
        """
        if df_sorted.empty:
            return {q: pd.Series(dtype=float) for q in qs}

        grouper = df_sorted.groupby("player", sort=False)
        group_sizes = grouper.size()
        players_idx = group_sizes.index
        sizes = group_sizes.to_numpy()
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        ends = starts + sizes

        cum_w = grouper["weight"].cumsum().to_numpy()
        total_w = grouper["weight"].transform("sum").to_numpy()
        cum_norm = np.divide(cum_w, total_w, out=np.zeros_like(cum_w), where=total_w > 0)
        vals = df_sorted["comp_mean"].to_numpy()

        results: dict[float, pd.Series] = {}
        for q in qs:
            out = np.full(len(players_idx), np.nan, dtype=np.float64)
            for i, (s, e) in enumerate(zip(starts, ends, strict=False)):
                if total_w[s] <= 0:
                    continue
                out[i] = np.interp(q, cum_norm[s:e], vals[s:e])
            results[q] = pd.Series(out, index=players_idx)
        return results

    def save_comps(self):
        """Write current comps to the league's JSON file for inspection."""
        if not self.comps:
            return
        filepath = pkg_resources.files(data) / "leagues" / self.league.lower() / "comps.json"
        with open(filepath, "w") as f:
            json.dump(self.comps, f, indent=4)

    def _join_fp_team_features(self, date) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """League hook: return ``(team_features, defense_features)`` for ``date``.

        Default no-op returns ``(None, None)``. Override on a league subclass
        to plug an external team-grain data source into ``teamProfile`` /
        ``defenseProfile``. The frames must be indexed by the same key the
        gamelog uses for the team / opponent columns (abbreviation for NFL).
        ``base_profile`` join-left-merges them after the existing teamlog
        aggregation so absent teams degrade to NaN-then-zero downstream.
        """
        return None, None

    def _join_fp_player_features(self, date) -> pd.DataFrame | None:
        """League hook: return season-to-date per-player FP feature frame for ``date``.

        Default no-op returns ``None``. Override on a league subclass to plug an
        external player-grain data source into ``playerstats``. The frame must
        be indexed by the same key ``base_profile``'s ``playerstats`` uses
        (``log_strings['player']`` value, post-``remove_accents``). Column
        names should be RAW ``{name}_asof`` (no ``Player`` prefix) — the
        ``add_prefix("Player ")`` step downstream in ``get_stats`` /
        ``get_feature_columns`` prepends the namespace once, producing the
        ``Player {name}_asof`` keys the feature filter expects.
        """
        return None

    def _profile_stat_types(self):
        if self.league in ("NBA", "WNBA"):
            stat_types = self.stat_types
            team_stat_types = self.team_stat_types
        elif self.league == "NFL":
            stat_types = (
                self.stat_types["passing"]
                + self.stat_types["rushing"]
                + self.stat_types["receiving"]
            )
            team_stat_types = list(
                set(self.stat_types["offense"]) | set(self.stat_types["defense"])
            )
        elif self.league == "MLB":
            stat_types = self.stat_types["pitching"] + self.stat_types["batting"]
            team_stat_types = (
                self.stat_types["fielding"]
                + self.stat_types["pitching"]
                + self.stat_types["batting"]
            )
        elif self.league == "NHL":
            stat_types = self.stat_types["skater"] + self.stat_types["goalie"]
            team_stat_types = self.team_stat_types
        return stat_types, team_stat_types

    def _player_recent_slope(self, frame, value_cols):
        """Per-player OLS slope of each value col over the player's last 5 games.

        Rank-indexed least-squares slope, zeroed for players with fewer than
        ``_MIN_GAMES_FOR_TRENDS`` total games. Shared by the efficiency-stat
        ``growth`` profile columns and the comp-trend market signal.
        """
        _pc = self.log_strings["player"]
        last5 = frame.groupby(_pc).tail(5).copy()
        last5["_rank"] = last5.groupby(_pc).cumcount()
        grp = last5.groupby(_pc)
        n = grp.size()
        sum_r = grp["_rank"].sum()
        sum_r2 = (last5["_rank"] ** 2).groupby(last5[_pc]).sum()
        denom = (n * sum_r2 - sum_r**2).replace(0, np.nan)
        valid = [c for c in value_cols if c in last5.columns]
        sum_y = grp[valid].sum()
        iy = last5[valid].multiply(last5["_rank"], axis=0)
        sum_iy = iy.groupby(last5[_pc]).sum()
        slopes = (sum_iy.multiply(n, axis=0) - sum_y.multiply(sum_r, axis=0)).div(denom, axis=0)
        total = frame.groupby(_pc).size()
        slopes.loc[total < _MIN_GAMES_FOR_TRENDS] = 0.0
        return slopes.fillna(0).infer_objects(copy=False)

    def _build_player_profile_stats(self, stat_types, date):
        _filled_gl = self.short_gamelog.fillna(0).infer_objects(copy=False)
        _player_col = self.log_strings["player"]
        # Intersect stat_types with available gamelog columns so partial fixtures
        # (e.g. tests/integration's cached parquets that don't reload the full
        # gamelog) don't KeyError. Production runs have all expected columns
        # after Stats.update(); this is purely defensive for fixture mode.
        _avail_stats = [s for s in stat_types if s in _filled_gl.columns]
        playerstats = _filled_gl.groupby(_player_col)[_avail_stats].mean(numeric_only=True)

        _last5 = _filled_gl.groupby(_player_col).tail(5)
        playershortstats = (
            _last5.groupby(_last5[_player_col])[_avail_stats]
            .mean()
            .fillna(0)
            .infer_objects(copy=False)
            .add_suffix(" short", 1)
        )

        playertrends = self._player_recent_slope(_filled_gl, stat_types).add_suffix(" growth", 1)

        playerstats = playerstats.join(playershortstats)
        playerstats = playerstats.join(playertrends)

        # League hook: external player-grain feature source (e.g. NFL
        # season-to-date FP aggregates from `load_through_one_year`). Override
        # ``_join_fp_player_features`` to return a frame indexed by the same
        # player-name key as ``playerstats``, with columns already prefixed
        # ``Player {name}_asof`` — joined left so missing players degrade to
        # NaN-then-zero downstream.
        fp_player_features = self._join_fp_player_features(date)
        if fp_player_features is not None and not fp_player_features.empty:
            playerstats = playerstats.join(fp_player_features, how="left")
        return playerstats

    def _build_team_defense_profile(self, team_stat_types, date):
        _team_col = self.log_strings["team"]
        _team_last10 = self.short_teamlog.groupby(_team_col).tail(10)
        teamstats = _team_last10.groupby(_team_last10[_team_col])[team_stat_types].mean()

        # League hook: per-league augmentation of teamstats / defenseProfile with
        # external-data-source features (e.g. NFL FP team-grain weekly snapshots).
        # Default no-op returns ``None``; NFL overrides return abbreviation-indexed
        # ``(team_features, defense_features)`` frames merged in below.
        fp_team_features, fp_defense_features = self._join_fp_team_features(date)
        if fp_team_features is not None and not fp_team_features.empty:
            teamstats = teamstats.join(fp_team_features, how="left")

        self.defenseProfile = (
            self.defenseProfile.join(teamstats, how="right").fillna(0).infer_objects(copy=False)
        )
        self.defenseProfile.index.name = self.log_strings["opponent"]
        if fp_defense_features is not None and not fp_defense_features.empty:
            self.defenseProfile = (
                self.defenseProfile.join(fp_defense_features, how="left")
                .fillna(0)
                .infer_objects(copy=False)
            )

        self.teamProfile = teamstats

    @line_profiler.profile
    def base_profile(self, date=datetime.today().date()):
        if isinstance(date, str):
            date = datetime.strptime(date, "%Y-%m-%d").date()
        elif isinstance(date, datetime):
            date = date.date()
        if date == self.profile_latest_date:
            return

        self.profile_latest_date = date

        self.playerProfile = pd.DataFrame(
            columns=[
                "team",
                "age",
                "position",
                "depth",
                "z",
                "position z",
                "home",
                "moneyline gain",
                "totals gain",
            ]
        )
        self.defenseProfile = pd.DataFrame(
            columns=[
                "avg",
                "home",
                "moneyline gain",
                "totals gain",
                "position",
                "comps",
                *self.positions,
            ]
        )

        one_year_ago = date - timedelta(days=300)

        gameDates = pd.to_datetime(self.gamelog[self.log_strings["date"]]).dt.date
        self.short_gamelog = self.gamelog[(one_year_ago <= gameDates) & (gameDates < date)].copy()
        _pc = self.log_strings["player"]
        self.short_gamelog[_pc] = self.short_gamelog[_pc].apply(remove_accents)
        gameDates = pd.to_datetime(self.teamlog[self.log_strings["date"]]).dt.date
        self.short_teamlog = self.teamlog[(one_year_ago <= gameDates) & (gameDates < date)].copy()

        stat_types, team_stat_types = self._profile_stat_types()
        playerstats = self._build_player_profile_stats(stat_types, date)
        self._build_team_defense_profile(team_stat_types, date)

        self.playerProfile = (
            self.playerProfile.join(playerstats, how="right").fillna(0).infer_objects(copy=False)
        )
        self.playerProfile.drop_duplicates(inplace=True)

    def _apply_position_z(self, market):
        for position in self.positions:
            positionLogs = self.short_gamelog.loc[
                self.short_gamelog[self.log_strings["position"]] == position
            ]
            positionGroups = positionLogs.groupby(self.log_strings["player"])
            positionAvg = positionGroups[market].mean().mean()
            positionStd = positionGroups[market].mean().std()
            if positionAvg == 0 or positionStd == 0:
                continue
            idx = list(
                set(positionGroups.groups.keys()).intersection(set(self.playerProfile.index))
            )
            self.playerProfile.loc[idx, "position z"] = (
                (positionGroups[market].mean() - positionAvg).div(positionStd).astype(float)
            )
            positionGroups = positionLogs.groupby(
                [self.log_strings["opponent"], self.log_strings["game"]]
            )
            positionGames = positionGroups[market].sum()
            positionGroups = positionGames.groupby(self.log_strings["opponent"])
            leagueavg = positionGroups.mean().mean()
            if leagueavg == 0:
                self.defenseProfile[position] = 0
            else:
                self.defenseProfile[position] = positionGroups.mean().div(leagueavg) - 1

    def _apply_comp_features(self, date, market, _all_mean, _all_trend):
        # Pass ``date`` so the training-matrix iteration rebuilds the comp
        # pool whenever the advancing date crosses into a new Wednesday-keyed
        # week (see :meth:`_ensure_comps`). The inference / cron path passes
        # today() and hits the lazy-once snapshot path.
        self._ensure_comps(date=date)
        _pairs = self._comp_pairs()
        if not _pairs.empty:
            # Per-day dynamic step: bind market-specific ``_all_mean`` to the
            # cached static pairs. The Python tuple-loop that flattens
            # ``self.comps`` lives in ``_comp_pairs`` and only runs on
            # comp-pool swap (per training week), not per gameday.
            _cp_df_p = _pairs.loc[
                _pairs["player"].isin(_all_mean.index), ["player", "comp", "weight"]
            ].copy()
            _cp_df_p["comp_mean"] = _cp_df_p["comp"].map(_all_mean)
            _cp_df_p = _cp_df_p.dropna(subset=["comp_mean"])
            if not _cp_df_p.empty:
                _wsum_p = (
                    (_cp_df_p["comp_mean"] * _cp_df_p["weight"]).groupby(_cp_df_p["player"]).sum()
                )
                _wcnt_p = _cp_df_p["weight"].groupby(_cp_df_p["player"]).sum()
                _comp_wmean = (_wsum_p / _wcnt_p).reindex(self.playerProfile.index)
                self.playerProfile["comps mean"] = _comp_wmean

                # Empirical-Bayes shrinkage of the comp mean toward the
                # position baseline. ``_comp_wmean`` is the distance-
                # weighted estimator; ``n_eff`` (weight mass) substitutes
                # for sample size in the K/(K+n) shrinkage factor. The
                # baseline is the population mean of comp means rather
                # than the global market mean -- that's the right anchor
                # because the comp pool already filters on the position's
                # usage band.
                _baseline = float(_cp_df_p["comp_mean"].mean())
                _shrinkage = _wcnt_p / (_wcnt_p + _COMP_EB_PRIOR_K)
                _comp_eb = (_shrinkage * _comp_wmean + (1.0 - _shrinkage) * _baseline).reindex(
                    self.playerProfile.index
                )
                self.playerProfile["comps mean (EB)"] = _comp_eb

                # Distance-weighted comp p25 / p75: np.interp(q, cum_w_norm, sorted_vals)
                # vectorized across all players via :meth:`_vectorized_weighted_quantiles`.
                _sorted = (
                    _cp_df_p[["player", "comp_mean", "weight"]]
                    .sort_values(["player", "comp_mean"])
                    .reset_index(drop=True)
                )
                _qs = self._vectorized_weighted_quantiles(
                    _sorted, [_COMP_QUANTILE_LO, _COMP_QUANTILE_HI]
                )
                self.playerProfile["comps p25"] = _qs[_COMP_QUANTILE_LO].reindex(
                    self.playerProfile.index
                )
                self.playerProfile["comps p75"] = _qs[_COMP_QUANTILE_HI].reindex(
                    self.playerProfile.index
                )

                # Wide comp spread flags an unreliable ``comps mean`` anchor.
                _dev = _cp_df_p["comp_mean"] - _cp_df_p["player"].map(_comp_wmean)
                _wvar = (_dev**2 * _cp_df_p["weight"]).groupby(_cp_df_p["player"]).sum() / _wcnt_p
                self.playerProfile["comps std"] = np.sqrt(_wvar).reindex(self.playerProfile.index)

                # Distance-weighted mean of comps' recent market slope -- mirrors
                # comps mean but over the per-player trend (_all_trend) rather
                # than the level (_all_mean).
                _cg = _cp_df_p.assign(comp_trend=_cp_df_p["comp"].map(_all_trend)).dropna(
                    subset=["comp_trend"]
                )
                _g_wsum = (_cg["comp_trend"] * _cg["weight"]).groupby(_cg["player"]).sum()
                _g_wcnt = _cg["weight"].groupby(_cg["player"]).sum()
                self.playerProfile["comps trend"] = (_g_wsum / _g_wcnt).reindex(
                    self.playerProfile.index
                )

    def _begin_profile_market(self, market, date) -> "date | None":
        """Normalize ``date``, refresh the base profile for ``market``, and return it.

        Returns ``None`` when ``(market, date)`` is already the active profile and
        the caller should return early; otherwise runs :meth:`base_profile`, records
        ``market`` as current, and returns the normalized ``date`` for the caller to
        reuse downstream (the comp path needs a real ``date``, not a ``str``). The
        subclass ``profile_market`` overrides share this guard.
        """
        if isinstance(date, str):
            date = datetime.strptime(date, "%Y-%m-%d").date()
        elif isinstance(date, datetime):
            date = date.date()
        if market == self.profiled_market and date == self.profile_latest_date:
            return None

        self.base_profile(date)
        self.profiled_market = market
        return date

    def profile_market(self, market, date=datetime.today().date()):
        date = self._begin_profile_market(market, date)
        if date is None:
            return

        # Replace slow .filter(lambda) with aggregation-based pre-filtering
        _pc = self.log_strings["player"]
        _grp_pre = self.short_gamelog.groupby(_pc)[market]
        _agg = _grp_pre.agg(count="count", clipped_mean=lambda x: x.clip(0, 1).mean())
        _valid_players = _agg[
            (_agg["clipped_mean"] > _MIN_PARTICIPATION_RATE) & (_agg["count"] > 1)
        ].index
        _filtered = self.short_gamelog[self.short_gamelog[_pc].isin(_valid_players)]
        playerGroups = _filtered.groupby(_pc)

        leagueavg = playerGroups[market].mean().mean()
        leaguestd = playerGroups[market].mean().std()
        if np.isnan(leagueavg) or np.isnan(leaguestd):
            return

        self.playerProfile[
            [
                "z",
                "home",
                "moneyline gain",
                "totals gain",
                "position z",
                "comps mean",
                "comps z",
                "comps raw",
                "comps z recent",
                "comps std",
                "comps trend",
            ]
        ] = 0.0
        self.playerProfile["z"] = (playerGroups[market].mean() - leagueavg).div(leaguestd)

        # Vectorized home split — avoids per-group .apply()
        _home_col = self.log_strings["home"]
        _home_mask = _filtered[_home_col].astype(bool)
        _home_mean = _filtered.loc[_home_mask].groupby(_pc)[market].mean()
        _all_mean = playerGroups[market].mean()
        _all_trend = self._player_recent_slope(_filtered, [market])[market]
        self.playerProfile["home"] = _home_mean / _all_mean - 1

        defenseGroups = self.short_gamelog.groupby(
            [self.log_strings["opponent"], self.log_strings["game"]]
        )
        defenseGames = defenseGroups[[market, self.log_strings["home"], "moneyline", "totals"]].agg(
            {
                market: "sum",
                self.log_strings["home"]: lambda x: np.mean(x) > 0.5,
                "moneyline": "mean",
                "totals": "mean",
            }
        )
        defenseGroups = defenseGames.groupby(self.log_strings["opponent"])

        self.defenseProfile[
            [
                "avg",
                "home",
                "moneyline gain",
                "totals gain",
                "position",
                "comp n",
                "comp distance",
            ]
        ] = 0.0
        leagueavg = defenseGroups[market].mean().mean()
        leaguestd = defenseGroups[market].mean().std()
        self.defenseProfile["avg"] = defenseGroups[market].mean().div(leagueavg) - 1

        _def_home_mask = defenseGames[self.log_strings["home"]].astype(bool)
        _def_home_mean = (
            defenseGames.loc[_def_home_mask]
            .groupby(self.log_strings["opponent"])[market]
            .mean()
            .clip(0.1)
        )
        _def_all_mean = defenseGroups[market].mean().clip(0.1)
        self.defenseProfile["home"] = _def_home_mean / _def_all_mean - 1

        self._apply_position_z(market)

        # slope = (N*Σ(X*Y) - ΣX*ΣY) / (N*ΣX² - (ΣX)²)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            _ml_filled = _filtered["moneyline"].fillna(0.5).astype(float)
            _ml_mean_p = _ml_filled.groupby(_filtered[_pc]).transform("mean")
            _mkt_vals = _filtered[market].astype(float)
            _mkt_mean_p = _mkt_vals.groupby(_filtered[_pc]).transform("mean")
            _X_ml = _ml_filled / 0.5 - _ml_mean_p
            _Y_p = _mkt_vals / _mkt_mean_p.replace(0, np.nan) - 1
            _n_p = _filtered.groupby(_pc).size()
            _SXY = (_X_ml * _Y_p).groupby(_filtered[_pc]).sum()
            _SX = _X_ml.groupby(_filtered[_pc]).sum()
            _SY = _Y_p.groupby(_filtered[_pc]).sum()
            _SX2 = (_X_ml**2).groupby(_filtered[_pc]).sum()
            _denom = (_n_p * _SX2 - _SX**2).replace(0, np.nan)
            self.playerProfile["moneyline gain"] = (_n_p * _SXY - _SX * _SY) / _denom

            _tot_filled = _filtered["totals"].fillna(self.default_total).astype(float)
            _tot_mean_p = _tot_filled.groupby(_filtered[_pc]).transform("mean")
            _X_tot = _tot_filled / self.default_total - _tot_mean_p
            _SXY_t = (_X_tot * _Y_p).groupby(_filtered[_pc]).sum()
            _SX_t = _X_tot.groupby(_filtered[_pc]).sum()
            _SX2_t = (_X_tot**2).groupby(_filtered[_pc]).sum()
            _denom_t = (_n_p * _SX2_t - _SX_t**2).replace(0, np.nan)
            self.playerProfile["totals gain"] = (_n_p * _SXY_t - _SX_t * _SY) / _denom_t

            _opp_col = self.log_strings["opponent"]
            _dg = defenseGames.reset_index()
            _def_ml = _dg["moneyline"].fillna(0.5).astype(float)
            _def_ml_mean = _def_ml.groupby(_dg[_opp_col]).transform("mean")
            _def_mkt = _dg[market].astype(float)
            _def_mkt_mean = _def_mkt.groupby(_dg[_opp_col]).transform("mean")
            _X_def_ml = _def_ml / 0.5 - _def_ml_mean
            _Y_def = _def_mkt / _def_mkt_mean.replace(0, np.nan) - 1
            _n_def = _dg.groupby(_opp_col).size()
            _SXY_def = (_X_def_ml * _Y_def).groupby(_dg[_opp_col]).sum()
            _SX_def = _X_def_ml.groupby(_dg[_opp_col]).sum()
            _SY_def = _Y_def.groupby(_dg[_opp_col]).sum()
            _SX2_def = (_X_def_ml**2).groupby(_dg[_opp_col]).sum()
            _denom_def = (_n_def * _SX2_def - _SX_def**2).replace(0, np.nan)
            self.defenseProfile["moneyline gain"] = (
                _n_def * _SXY_def - _SX_def * _SY_def
            ) / _denom_def

            _def_tot = _dg["totals"].fillna(self.default_total).astype(float)
            _def_tot_mean = _def_tot.groupby(_dg[_opp_col]).transform("mean")
            _X_def_tot = _def_tot / self.default_total - _def_tot_mean
            _SXY_dt = (_X_def_tot * _Y_def).groupby(_dg[_opp_col]).sum()
            _SX_dt = _X_def_tot.groupby(_dg[_opp_col]).sum()
            _SX2_dt = (_X_def_tot**2).groupby(_dg[_opp_col]).sum()
            _denom_dt = (_n_def * _SX2_dt - _SX_dt**2).replace(0, np.nan)
            self.defenseProfile["totals gain"] = (_n_def * _SXY_dt - _SX_dt * _SY_def) / _denom_dt

        self._apply_comp_features(date, market, _all_mean, _all_trend)

        self.defenseProfile.fillna(0.0, inplace=True)
        self.teamProfile.fillna(0.0, inplace=True)
        self.playerProfile.fillna(0.0, inplace=True)

    def _roster_player_df(self, season):
        if self.league == "NBA":
            season = "-".join([str(season), str(season - 1999)])

        if "NBA" in self.league:
            player_df = []
            season_players = (
                self.players.get(season)
                if self.players.get(season)
                else self.players.get(season - 1)
            )
            for team, roster in season_players.items():
                roster = [
                    {self.log_strings["player"]: k, self.log_strings["team"]: team} | v
                    for k, v in roster.items()
                ]
                player_df.extend(roster)

            player_df = pd.DataFrame(player_df)
        elif self.league == "NHL":
            player_df = pd.DataFrame(self.players[season].values())
        elif self.league == "NFL":
            player_df = self.players.reset_index().rename(columns={"name": "player display name"})
        else:
            player_df = self.players
        return player_df

    def get_depth(self, offers, date=datetime.today().date()):
        players = list(offers.keys()) if isinstance(offers, dict) else [x["Player"] for x in offers]

        for player in players:
            if " + " in player.replace(" vs. ", " + "):
                split_players = player.replace(" vs. ", " + ").split(" + ")
                players.append(split_players[0])
                players.append(split_players[1])

        players = set(players)
        season = (
            date.year
            if ((date.month >= _SEASON_ROLLOVER_MONTH) or (self.league in ["WNBA", "MLB"]))
            else date.year - 1
        )

        player_df = self._roster_player_df(season)

        if date < datetime.today().date():
            old_player_df = self.gamelog.loc[
                (pd.to_datetime(self.gamelog[self.log_strings["date"]]).dt.date == date)
                & self.gamelog[self.log_strings["player"]].isin(list(players)),
                [
                    self.log_strings["player"],
                    self.log_strings["team"],
                    self.log_strings["position"],
                ],
            ]
            player_df = old_player_df.merge(
                player_df, on=self.log_strings["player"], suffixes=[None, "_obs"]
            )
            player_df.drop(
                columns=[col for col in player_df.columns if "_obs" in col], inplace=True
            )

        player_df.index = player_df[self.log_strings["player"]]
        player_df.drop(columns=self.log_strings["player"], inplace=True)
        player_df = player_df.loc[list(set(player_df.index) & players)]
        player_df = player_df.loc[~player_df.index.duplicated()]

        self.profile_market(self.usage_stat, date=date)
        usage = pd.DataFrame(self.playerProfile[[self.usage_stat + " short", self.tiebreaker_stat]])
        player_df = player_df.join(usage, how="left").fillna(0).infer_objects(copy=False)
        ranks = (
            player_df.sort_values(self.tiebreaker_stat, ascending=False)
            .groupby([self.log_strings["team"], self.log_strings["position"]])
            .rank(ascending=False, method="first")[self.usage_stat + " short"]
            .astype(int)
        )

        self.playerProfile["depth"] = ranks.to_dict()
        self.playerProfile["position"] = player_df[self.log_strings["position"]].apply(
            lambda x: self.positions.index(x) + 1
        )
        self.playerProfile["team"] = player_df[self.log_strings["team"]]
        if self.log_strings.get("age", "") in player_df.columns:
            self.playerProfile["age"] = player_df[self.log_strings["age"]]

        self.playerProfile.fillna(0, inplace=True)

    def get_stats(self, market, offers, date=datetime.today().date()):
        self.profile_market(market, date)
        self._ensure_comps(date=date)
        stats = pd.DataFrame(columns=_BASE_STAT_COLUMNS)

        players, teams, opponents = self._resolve_offer_entities(offers)
        playergames = self.short_gamelog.loc[
            self.short_gamelog[self.log_strings["player"]].isin(players)
        ]
        if playergames.empty:
            return stats

        stats = self._rolling_features(stats, playergames, market, date)
        stats, teams, opponents, pitchers = self._game_context(
            stats, offers, market, date, teams, opponents
        )
        self._h2h_features(stats, market, opponents, pitchers)
        stats, defstats = self._join_profiles(stats, teams, opponents, pitchers)
        # _join_profiles drops players off the depth chart, which can leave a
        # gameday with zero rows (e.g. a recent date whose players aren't on the
        # depth chart yet); the feature builders below index per-row and raise on
        # an empty frame, so return early -- the day contributes no training rows.
        if stats.empty:
            return stats
        self._expanding_features(stats, market, date, opponents)
        self._interaction_features(stats, defstats)
        self._comp_features(stats, defstats, market, opponents, date)
        stats = self._join_defense_and_parks(stats, defstats, teams)
        return stats.fillna(0).infer_objects(copy=False)

    def _resolve_offer_entities(self, offers):
        """Resolve an ``offers`` dict-or-list into deduped ``(players, teams, opponents)``.

        Combo legs (``"A + B"`` / ``"A vs. B"``) are split into their two component
        players, fanning out the ``"/"``-joined team/opponent strings; players with a
        blank team or opponent are dropped.
        """
        if isinstance(offers, dict):
            players = list(offers.keys())
            teams = {k: v["Team"] for k, v in offers.items()}
            opponents = {k: v["Opponent"] for k, v in offers.items()}
        else:
            players = [x["Player"] for x in offers]
            teams = {x["Player"]: x["Team"] for x in offers}
            opponents = {x["Player"]: x["Opponent"] for x in offers}

        players = list(set(players))
        for player in players.copy():
            if " + " in player.replace(" vs. ", " + "):
                if player not in teams:
                    continue
                split_players = player.replace(" vs. ", " + ").split(" + ")
                players.remove(player)
                players.append(split_players[0])
                players.append(split_players[1])

                split_teams = teams.pop(player).split("/")
                if len(split_teams) == 1:
                    split_teams = split_teams * 2

                teams[split_players[0]] = split_teams[0]
                teams[split_players[1]] = split_teams[1]

                split_opponents = opponents.pop(player).split("/")
                if len(split_opponents) == 1:
                    split_opponents = split_opponents * 2

                opponents[split_players[0]] = split_opponents[0]
                opponents[split_players[1]] = split_opponents[1]
            elif teams[player] == "" or opponents[player] == "":
                players.remove(player)
                teams.pop(player)
                opponents.pop(player)
        return players, teams, opponents

    def _rolling_features(self, stats, playergames, market, date):
        """Populate the tail-window rolling stats (Avg/Mean/STD/ratios) for each player."""
        player_col = self.log_strings["player"]

        # Vectorized tail-based stats — avoids per-group .apply(lambda)
        pg_last10 = playergames.groupby(player_col).tail(10)
        pg_last5 = playergames.groupby(player_col).tail(5)
        pg_last3 = playergames.groupby(player_col).tail(3)
        pg_last1 = playergames.groupby(player_col).tail(1)

        stats["Avg1"] = pg_last1.groupby(player_col)[market].median()
        stats["Avg3"] = pg_last3.groupby(player_col)[market].median()
        stats["Avg5"] = pg_last5.groupby(player_col)[market].median()
        stats["Avg10"] = pg_last10.groupby(player_col)[market].median()
        stats["AvgYr"] = playergames.groupby(player_col)[market].median()
        stats["Mean10"] = pg_last10.groupby(player_col)[market].mean()
        stats["MeanYr"] = playergames.groupby(player_col)[market].mean()
        nonzero_games = playergames.loc[playergames[market] > 0]
        stats["MeanYr_nonzero"] = nonzero_games.groupby(player_col)[market].mean()
        stats["MeanYr_nonzero"] = stats["MeanYr_nonzero"].fillna(stats["MeanYr"].clip(lower=0.5))
        stats["STD10"] = pg_last10.groupby(player_col)[market].std()
        stats["STDYr"] = playergames.groupby(player_col)[market].std()
        last_date = pd.to_datetime(playergames.groupby(player_col)[self.log_strings["date"]].last())
        stats["DaysOff"] = (pd.Timestamp(date) - last_date).dt.days
        stats["DaysIntoSeason"] = (date - self.season_start).days
        stats["GamesPlayed"] = playergames.groupby(player_col)[market].count()
        zero_counts = playergames.loc[playergames[market] == 0].groupby(player_col).size()
        stats["ZeroYr"] = zero_counts.reindex(stats.index, fill_value=0) / stats["GamesPlayed"]

        meanyr_safe = stats["MeanYr"].clip(lower=0.5)
        stats["Avg5_Ratio"] = stats["Avg5"] / meanyr_safe
        stats["Avg10_Ratio"] = stats["Avg10"] / meanyr_safe
        stats["Mean10_Ratio"] = stats["Mean10"] / meanyr_safe
        stats["STD_Ratio"] = stats["STDYr"] / meanyr_safe

        return stats.loc[~stats.index.duplicated()]

    def _game_context(self, stats, offers, market, date, teams, opponents):
        """Attach the game-script context and resolve the game-day matchup.

        Sets Home / Moneyline / Total (own team) plus OppTotal (opponent implied
        total) and the derived Spread / GameTotal / Blowout. Historical (``date``
        in the past) reads the realized ``gamelog`` row and rebinds ``teams`` /
        ``opponents`` (and MLB ``pitchers``) from it, taking OppTotal from the same
        frame; upcoming reads ``upcoming_games`` plus the archived moneyline/total
        and an extra ``get_total`` for the opponent. Returns the (possibly rebound)
        ``(stats, teams, opponents, pitchers)``; ``pitchers`` is ``None`` for
        non-MLB leagues.
        """
        pitchers = None
        if date < datetime.today().date():
            todays_games = self.gamelog.loc[
                pd.to_datetime(self.gamelog[self.log_strings["date"]]).dt.date == date
            ]
            todays_games.index = todays_games[self.log_strings["player"]]
            todays_games = todays_games.loc[~todays_games.index.duplicated()]
            stats["Home"] = todays_games[self.log_strings["home"]]
            stats["Moneyline"] = todays_games["moneyline"]
            stats["Total"] = todays_games["totals"]

            teams = todays_games[self.log_strings["team"]].to_dict()
            opponents = todays_games[self.log_strings["opponent"]].to_dict()
            team_total = todays_games.groupby(self.log_strings["team"])["totals"].first()
            stats["OppTotal"] = pd.Series(opponents).map(team_total)
            if self.league == "MLB":
                pitchers = todays_games["opponent pitcher"].to_dict()
        else:
            if self.league == "MLB":
                pitchers = {
                    x: self.upcoming_games.get(teams[x], {}).get("Opponent Pitcher")
                    for x in stats.index
                }
                battingOrder = {
                    x: self.upcoming_games.get(teams[x], {}).get("Batting Order").index(x) + 1
                    if x in self.upcoming_games.get(teams[x], {}).get("Batting Order", [])
                    else 0
                    for x in stats.index
                }
                self.playerProfile.depth = battingOrder

            dates = {x["Player"]: x["Date"] for x in offers}
            for player in list(dates.keys()):
                if " + " in player.replace(" vs. ", " + "):
                    split_players = player.replace(" vs. ", " + ").split(" + ")
                    dates[split_players[0]] = dates[player]
                    dates[split_players[1]] = dates[player]
                    dates.pop(player)

            stats["Home"] = [self.upcoming_games.get(teams[x], {}).get("Home") for x in stats.index]
            stats["Moneyline"] = [
                archive.get_moneyline(self.league, dates[x], teams[x]) for x in stats.index
            ]
            stats["Total"] = [
                archive.get_total(self.league, dates[x], teams[x]) for x in stats.index
            ]
            stats["OppTotal"] = [
                archive.get_total(self.league, dates[x], opponents[x]) for x in stats.index
            ]
        stats["Spread"] = stats["Total"] - stats["OppTotal"]
        stats["GameTotal"] = stats["Total"] + stats["OppTotal"]
        stats["Blowout"] = (stats["Spread"].abs() > _BLOWOUT_SPREAD_THRESHOLD[self.league]).astype(
            int
        )
        self._schedule_context(stats, date, teams)
        self._playoff_flag(stats, date, teams)
        self._series_context(stats, date, teams, opponents)
        return stats, teams, opponents, pitchers

    def _h2h_features(self, stats, market, opponents, pitchers):
        """Attach head-to-head Avg/Mean/Played against this opponent (MLB: vs the pitcher)."""
        if self.league == "MLB" and not is_mlb_pitcher_market(market):
            h2hgames = self.short_gamelog.loc[
                self.short_gamelog["opponent pitcher"]
                == self.short_gamelog[self.log_strings["player"]].map(pitchers)
            ].groupby(self.log_strings["player"])
        else:
            h2hgames = self.short_gamelog.loc[
                self.short_gamelog[self.log_strings["opponent"]]
                == self.short_gamelog[self.log_strings["player"]].map(opponents)
            ].groupby(self.log_strings["player"])
        stats["AvgH2H"] = h2hgames[market].median()
        stats["MeanH2H"] = h2hgames[market].mean()
        stats["H2HPlayed"] = h2hgames[market].count()

    def _schedule_context(self, stats, date, teams):
        """Hook: attach league-specific schedule-context columns. No-op on base.

        NFL overrides this to emit the Weekday feature (game date day-of-week) from
        the gamelog (historical) or ``upcoming_games`` (upcoming). Called at the end
        of :meth:`_game_context` so it shares the same date branch.
        """

    def _playoff_flag(self, stats, date, teams):
        """Flag postseason games via the native game-ID season-type code.

        Handles the three leagues whose game ID encodes the season type
        (:data:`_PLAYOFF_GAMEID_CODE`); NFL and MLB override with their own signal.
        Historical reads the realized gamelog row's game ID per player. The upcoming
        schedule carries no game ID, so the serve branch infers the phase from the
        season's latest completed game -- a games-based read, not a calendar window,
        so it needs no per-league regular-season length (which drifts year to year).
        """
        start, stop, codes = _PLAYOFF_GAMEID_CODE[self.league]
        game_col = self.log_strings["game"]
        gl_dates = pd.to_datetime(self.gamelog[self.log_strings["date"]])
        if date < datetime.today().date():
            todays = (
                self.gamelog.loc[gl_dates.dt.date == date]
                .drop_duplicates(self.log_strings["player"])
                .set_index(self.log_strings["player"])
            )
            game_codes = todays[game_col].astype(str).str.slice(start, stop).reindex(stats.index)
            stats["Playoff"] = game_codes.isin(codes).astype(int)
        else:
            latest_code = str(self.gamelog.loc[gl_dates.idxmax(), game_col])[start:stop]
            stats["Playoff"] = int(latest_code in codes)

    def _series_context(self, stats, date, teams, opponents):
        """Attach playoff-series contention: within-series W-L record plus the
        FacingElimination / CanClinch flags (model track series-context feature).

        Tallies the record from prior playoff teamlog games against this opponent
        (:meth:`_tally_series_record`), then derives the flags from a per-league
        wins-to-clinch (:meth:`_series_games_to_win`: best-of-7 here, round-dependent
        for WNBA/MLB). Regular-season games carry no series, so the whole block
        short-circuits to zero when nobody on the slate is in the postseason. NFL
        (single-elimination, no multi-game series) overrides this entirely.
        """
        if not stats["Playoff"].any():
            stats[list(_SERIES_FEATURE_COLS)] = 0
            return
        playoff_teamlog = self._playoff_teamlog()
        wins, losses = self._tally_series_record(
            playoff_teamlog, stats.index, date, teams, opponents
        )
        matchups = {(teams.get(p), opponents.get(p)) for p in stats.index}
        games_to_win = {
            m: self._series_games_to_win(playoff_teamlog, m[0], m[1], date) for m in matchups
        }
        gtw = pd.Series(
            {p: games_to_win[(teams.get(p), opponents.get(p))] for p in stats.index}, dtype=float
        )
        in_playoffs = stats["Playoff"] == 1
        stats["SeriesWins"] = wins
        stats["SeriesLosses"] = losses
        stats["FacingElimination"] = ((losses == gtw - 1) & in_playoffs).astype(int)
        stats["CanClinch"] = ((wins == gtw - 1) & in_playoffs).astype(int)

    def _tally_series_record(self, playoff_teamlog, index, date, teams, opponents):
        """Per-player (wins, losses) in the current playoff series, from the teamlog.

        Counts this team's prior wins/losses against the matchup opponent within
        :data:`_SERIES_LOOKBACK_DAYS` (the same-postseason series window). Shared by
        the game-ID leagues and MLB; they differ only in how ``playoff_teamlog`` is
        scoped (:meth:`_playoff_teamlog`).
        """
        team_col, opp_col, date_col = (
            self.log_strings["team"],
            self.log_strings["opponent"],
            self.log_strings["date"],
        )
        tl_dates = pd.to_datetime(playoff_teamlog[date_col]).dt.date
        window = playoff_teamlog.loc[
            (tl_dates < date) & (tl_dates >= date - timedelta(days=_SERIES_LOOKBACK_DAYS))
        ]
        wins = window.loc[window["WL"] == "W"].groupby([team_col, opp_col]).size()
        losses = window.loc[window["WL"] == "L"].groupby([team_col, opp_col]).size()
        series_wins = pd.Series(
            {p: int(wins.get((teams.get(p), opponents.get(p)), 0)) for p in index}, dtype=int
        )
        series_losses = pd.Series(
            {p: int(losses.get((teams.get(p), opponents.get(p)), 0)) for p in index}, dtype=int
        )
        return series_wins, series_losses

    def _playoff_teamlog(self):
        """Teamlog rows that are playoff games, for the game-ID leagues. MLB overrides
        (its game ID carries no code) to scope by the stamped ``game type``.
        """
        start, stop, codes = _PLAYOFF_GAMEID_CODE[self.league]
        game_codes = self.teamlog[self.log_strings["game"]].astype(str).str.slice(start, stop)
        return self.teamlog.loc[game_codes.isin(codes)]

    def _series_games_to_win(self, playoff_teamlog, team, opp, date):
        """Wins needed to clinch the series. Best-of-7 for NBA/NHL; WNBA and MLB
        override for their round-dependent formats.
        """
        return _SERIES_DEFAULT_GAMES_TO_WIN

    def _expanding_features(self, stats, market, date, opponents):
        """Career expanding-mean player features (M-1), leakage-safe.

        Reads the FULL gamelog (strict ``< date``) per requested player -- distinct
        from the 1-year-window ``MeanYr`` in :meth:`_rolling_features`. Emits the
        career mean, its head-to-head-vs-opponent variant, and an Empirical-Bayes
        shrinkage of the career mean toward the per-position baseline where the
        career window is sparse (the small-n NFL case the build pre-registers).
        """
        player_col = self.log_strings["player"]
        opp_col = self.log_strings["opponent"]
        gld = pd.to_datetime(self.gamelog[self.log_strings["date"]]).dt.date
        career = self.gamelog.loc[gld < date, [player_col, opp_col, market]].copy()
        career[player_col] = career[player_col].apply(remove_accents)
        career = career.loc[career[player_col].isin(stats.index)]

        career_grp = career.groupby(player_col)[market]
        stats["MeanYr_expanding_shifted"] = career_grp.mean().reindex(stats.index)
        n_career = career_grp.count().reindex(stats.index).fillna(0)
        stats["MeanYr_expanding_n"] = n_career

        h2h = career.loc[career[opp_col] == career[player_col].map(opponents)]
        h2h_grp = h2h.groupby(player_col)[market]
        stats["MeanYr_expanding_vsopp"] = h2h_grp.mean().reindex(stats.index)
        stats["MeanYr_expanding_vsopp_n"] = h2h_grp.count().reindex(stats.index).fillna(0)

        expanding = stats["MeanYr_expanding_shifted"]
        pos_baseline = expanding.groupby(stats["Player position"]).transform("mean")
        pos_baseline = pos_baseline.fillna(expanding.mean())
        k = _EXPANDING_EB_PRIOR_K_BY_LEAGUE.get(self.league, _EXPANDING_EB_PRIOR_K)
        shrink = n_career / (n_career + k)
        stats["MeanYr_expanding_eb"] = (
            shrink * expanding.fillna(pos_baseline) + (1.0 - shrink) * pos_baseline
        )

    def _interaction_features(self, stats, defstats):
        """Opponent-defense × player interaction columns (M-1, non-MLB).

        Products of leakage-safe profile columns: the player's career level scaled
        by the opponent's overall defense ratio, and the player's standardized level
        scaled by the opponent's defense strength vs the player's own position. MLB
        is excluded -- its ``defstats`` carries a pitcher overlay, not the team-defense
        ``avg`` / per-position strength these products assume.
        """
        if self.league == "MLB":
            return
        stats["PlayerExp_x_DefAvg"] = stats["MeanYr_expanding_eb"] * defstats["avg"]
        stats["PlayerZ_x_DefPos"] = stats["Player z"] * defstats["position"]

    def _join_profiles(self, stats, teams, opponents, pitchers):
        """Join player / team / defense profiles; return ``(stats, defstats)``.

        Non-MLB drops players off the depth chart and projects the per-position
        defense column; MLB overlays the opposing pitcher's profile onto defense.
        """
        stats = stats.join(self.playerProfile.add_prefix("Player "))
        if self.league != "MLB":
            stats = stats.loc[stats["Player depth"] > 0]

        teamstats = _profile_rows_for_teams(self.teamProfile, stats.index.map(teams)).add_prefix(
            "Team "
        )
        teamstats.index = stats.index
        stats = stats.join(teamstats)
        defstats = _profile_rows_for_teams(self.defenseProfile, stats.index.map(opponents)).astype(
            float
        )
        if self.league == "MLB":
            defstats.loc[
                [x in self.pitcherProfile.index for x in stats.index.map(pitchers)],
                self.pitcherProfile.columns,
            ] = self.pitcherProfile.loc[
                [x for x in stats.index.map(pitchers) if x in self.pitcherProfile.index]
            ].values
        else:
            # The depth filter above can leave zero rows; guard the per-row
            # position diag (the drop still runs so the empty frame keeps the
            # same schema as populated gamedays).
            if len(stats):
                defstats["position"] = np.diag(defstats.iloc[:, stats["Player position"] + 5])
            defstats.drop(columns=self.positions, inplace=True)

        defstats.index = stats.index
        return stats, defstats

    def _comp_features(self, stats, defstats, market, opponents, date):
        """Attach the matchup-conditional comp signal to ``stats`` / ``defstats``.

        The residual signal writes to ``stats["Player comps z"]`` (was the misnamed
        ``defstats["comps"]`` under the ``Defense `` prefix). The count + uniqueness
        signals stay on the defstats side under their legacy names (``Defense comp n``,
        ``Defense comp distance``) so cached training matrices and the SHAP-rebuild
        filter survive without a regeneration pass.
        """
        stats["Player comps z"] = 0.0
        if self.league == "MLB":
            self._mlb_comp_features(stats, defstats, market, opponents)
        else:
            self._nonmlb_comp_features(stats, defstats, market, opponents, date)

    def _mlb_comp_features(self, stats, defstats, market, opponents):
        """MLB per-player comp-z: z-score each comp's outcome vs this opponent."""
        player_col = self.log_strings["player"]
        opp_col = self.log_strings["opponent"]
        is_pitch_market = is_mlb_pitcher_market(market)
        for player in stats.index:
            compGames = self._mlb_comp_games(player, market, is_pitch_market, stats)
            if compGames is None:
                continue
            scores = _zscore_within_player(compGames, player_col, market)
            compGames[market] = scores
            opp_comp_games = compGames.loc[compGames[opp_col] == opponents[player], market]
            stats.loc[player, "Player comps z"] = opp_comp_games.mean()
            defstats.loc[player, "comp n"] = opp_comp_games.count()

    @staticmethod
    def _mlb_comp_ids(pool: dict, pid) -> list:
        """Comp id list for ``pid`` from a discretized ``{"comps": [...]}`` comp pool.

        Falls back to ``[pid]`` (self-comp) when the pool has no entry, matching the
        legacy bare-list ``pool.get(pid, [pid])`` behavior now that MLB affinity pools
        carry the ``{"comps": ..., "distances": ...}`` shape shared with the other
        leagues' KNN comps.
        """
        return pool.get(pid, {"comps": [pid]})["comps"]

    def _mlb_comp_games(self, player, market, is_pitch_market, stats):
        """Comp games backing one MLB player's comp-z, or ``None`` to skip the player.

        For hitter markets also writes ``stats["Pitcher comps"]`` (the player's mean
        vs their comparable opposing pitchers) before returning.
        """
        player_col = self.log_strings["player"]
        playerGames = self.short_gamelog.loc[self.short_gamelog[player_col] == player]
        if playerGames.empty:
            return None
        pid = playerGames["playerId"].mode()[0]
        if is_pitch_market:
            comps = self._mlb_comp_ids(self.comps["pitchers"], pid)
            compGames = self.short_gamelog.loc[
                self.short_gamelog["playerId"].isin(comps) & self.short_gamelog["starting pitcher"]
            ].copy()
            return None if compGames.empty else compGames

        comps = self._mlb_comp_ids(self.comps["hitters"], pid)
        compGames = self.short_gamelog.loc[
            self.short_gamelog["playerId"].isin(comps) & self.short_gamelog["starting batter"]
        ].copy()
        if compGames.empty:
            return None

        pitch_id = self.short_gamelog.loc[
            self.short_gamelog[player_col] == player, "opponent pitcher id"
        ]
        if pitch_id.empty:
            return None

        pitch_id = pitch_id.mode()[0]
        pitchComps = self._mlb_comp_ids(self.comps["pitchers"], pitch_id)
        pitchGames = playerGames.loc[playerGames["opponent pitcher id"].isin(pitchComps)]
        if pitchGames.empty or pitchGames[market].mean() == 0:
            stats.loc[player, "Pitcher comps"] = 0
        else:
            stats.loc[player, "Pitcher comps"] = (
                playerGames[market].mean() / pitchGames[market].mean()
            )
        return compGames

    def _nonmlb_comp_features(self, stats, defstats, market, opponents, date):
        """Non-MLB matchup-conditional comp lookup (vectorized).

        The static ``(player, comp, position, dist, weight)`` table is built once per
        comp-pool swap by :meth:`_comp_pairs`; here we attach the per-call opponent
        and merge against the per-day z-scored gamelog to get each comp's outcome vs
        this opponent, z-scored against the comp's own baseline and distance-weighted.
        QW-5 adds a recency-weighted raw-scale mean and a recency-weighted z sibling.
        """
        opp_col = self.log_strings["opponent"]
        player_col = self.log_strings["player"]
        date_col = self.log_strings["date"]
        gl_z = self.short_gamelog[[player_col, opp_col, market, date_col]].copy()
        gl_z["_mkt_zscore"] = _zscore_within_player(gl_z, player_col, market)
        _days_ago = (pd.Timestamp(date) - pd.to_datetime(gl_z[date_col])).dt.days
        gl_z["_recency_w"] = np.exp(-_days_ago / _COMP_RECENCY_HALFLIFE_DAYS)

        pairs = self._comp_pairs()
        if pairs.empty:
            return
        pos_idx = stats["Player position"].astype(int) - 1
        valid = (pos_idx >= 0) & (pos_idx < len(self.positions))
        if not valid.any():
            return

        expected_pos = pd.Series(
            np.asarray(self.positions)[pos_idx[valid].to_numpy()],
            index=stats.index[valid],
            name="expected_pos",
        )
        # Keep only pairs whose position matches the stats-side position assignment.
        # Preserves the original lookup semantics (``self.comps[stats_position].get(
        # player, ...)``) for players whose ``Player position`` disagrees with the
        # comp-pool assignment.
        cp_df = pairs.merge(
            expected_pos.rename_axis("target_player").reset_index(),
            left_on="player",
            right_on="target_player",
            how="inner",
        )
        cp_df = cp_df[cp_df["position"] == cp_df["expected_pos"]]
        if cp_df.empty:
            return

        cp_df["opp"] = cp_df["player"].map(opponents)
        # Mean distance to comps (player-side uniqueness signal -- not matchup-
        # dependent). Carried on defstats only because cached training matrices know
        # it under the ``Defense `` prefix.
        defstats["comp distance"] = cp_df.groupby("player")["dist"].mean().reindex(defstats.index)
        merged = cp_df.merge(
            gl_z[[player_col, opp_col, market, "_mkt_zscore", "_recency_w"]],
            left_on=["comp", "opp"],
            right_on=[player_col, opp_col],
            how="inner",
            suffixes=("", "_gl"),
        )
        merged["weighted_z"] = merged["_mkt_zscore"] * merged["weight"]
        comp_wsum = merged.groupby("player")["weighted_z"].sum()
        comp_wcount = merged.groupby("player")["weight"].sum()
        stats["Player comps z"] = (comp_wsum / comp_wcount).reindex(stats.index).fillna(0.0)
        # Count of (comp, opponent) game observations backing each ``Player comps z``
        # estimate. Stays on defstats for cached-matrix compatibility.
        defstats["comp n"] = merged.groupby("player").size().reindex(defstats.index)

        # Emitted alongside ``Player comps z`` (which stays distance-only) so
        # the A/B picks the winner rather than silently regressing the shipped feature.
        merged["_rw"] = merged["weight"] * merged["_recency_w"]
        _rw_sum = merged.groupby("player")["_rw"].sum()
        merged["_weighted_raw"] = merged[market] * merged["_rw"]
        stats["Player comps raw"] = (
            (merged.groupby("player")["_weighted_raw"].sum() / _rw_sum)
            .reindex(stats.index)
            .fillna(0.0)
        )
        merged["_weighted_z_recent"] = merged["_mkt_zscore"] * merged["_rw"]
        stats["Player comps z recent"] = (
            (merged.groupby("player")["_weighted_z_recent"].sum() / _rw_sum)
            .reindex(stats.index)
            .fillna(0.0)
        )

    def _join_defense_and_parks(self, stats, defstats, teams):
        """Join the defense profile and, for MLB, the per-team park factors."""
        stats = stats.join(defstats.add_prefix("Defense "))

        if self.league == "MLB":
            stats = stats.join(
                pd.DataFrame(
                    {
                        x: {
                            f"PF {k}": v for k, v in self.park_factors.get(teams.get(x), ()).items()
                        }
                        for x in stats.index
                    }
                ).T
            )
            stats["Player depth"] = stats["Player position"]

        return stats

    def load_volume_model_params(
        self, offers, market, date, rename_map=None, position_filter=None
    ) -> bool:
        """Join a volume model's predicted distribution parameters into playerProfile.

        Shared head of ``StatsMLB`` / ``StatsNHL`` / ``StatsNFL`` ``get_volume_stats``:
        flattens ``offers``, profiles the market, loads the cached
        ``models/{league}_{market}.mdl`` pickle, predicts the per-player
        distribution parameters, and joins them into ``self.playerProfile`` under
        ``rename_map`` (the per-league difference besides ``market``). Returns
        ``True`` on success, ``False`` when the gamelog is empty or the pickle is
        absent so callers with post-join work (NHL/NFL budget scaling) can abort.

        Args:
            offers: Sportsbook offers dict or list passed through from the caller.
            market: The volume stat market name (e.g. ``"carries"``).
            date: Game date used for profiling and stats lookup.
            rename_map: Column rename applied to the raw model output before the
                join, e.g. ``{"loc": "proj carries loc", ...}``. Defaults to the
                SkewNormal ``loc``/``scale``/``alpha`` map (``proj {market} loc``
                etc.) shared by NFL and NHL; MLB passes its own ``mean``/``std`` map.
            position_filter: When not ``None``, restrict predicted rows to players
                whose ``"Player position"`` value is in this collection before the
                join. NFL passes a per-market list of depth-chart position numbers
                (``1`` = QB, ``2`` = WR, ``3`` = RB, ``4`` = TE). MLB and NHL
                pass ``None`` and skip the filter.
        """
        if rename_map is None:
            rename_map = {
                "loc": f"proj {market} loc",
                "scale": f"proj {market} scale",
                "alpha": f"proj {market} alpha",
            }

        flat_offers = flatten_offers(offers, market)

        self.profile_market(market, date)
        self.get_depth(flat_offers, date)
        playerStats = self.get_stats(market, flat_offers, date)

        if playerStats.empty:
            logger.warning(f"Gamelog missing - {date}")
            return False

        filename = "_".join([self.league, market]).replace(" ", "-")
        filepath = pkg_resources.files(data) / f"models/{filename}.mdl"
        if self._volume_model_cache is None:
            self._volume_model_cache = {}
        if filename not in self._volume_model_cache:
            if os.path.isfile(filepath):
                with open(filepath, "rb") as infile:
                    self._volume_model_cache[filename] = pickle.load(infile)
            else:
                logger.warning(f"{filename} missing")
                return False

        filedict = self._volume_model_cache[filename]
        playerStats = playerStats[filedict["expected_columns"]]
        model = filedict["model"]
        dist = filedict["distribution"]

        categories = ["Home", "Player position"]
        if "Player position" not in playerStats.columns:
            categories.remove("Player position")
        for c in categories:
            playerStats[c] = playerStats[c].astype("category")

        set_model_start_values(model, dist, playerStats)

        prob_params = model.predict(playerStats, pred_type="parameters")
        prob_params.index = playerStats.index

        prob_params.sort_index(inplace=True)
        playerStats.sort_index(inplace=True)
        if position_filter is not None:
            prob_params = prob_params.loc[playerStats["Player position"].isin(position_filter)]

        # gate is a ZI-model artifact; callers consume only the distribution parameters.
        prob_params.drop(columns=["gate"], inplace=True, errors="ignore")
        self.playerProfile = self.playerProfile.join(
            prob_params.rename(columns=rename_map), lsuffix="_obs"
        )
        self.playerProfile.drop(
            columns=[col for col in self.playerProfile.columns if "_obs" in col], inplace=True
        )
        return True

    def get_volume_stats(self, offers, date=datetime.today().date()):
        return

    def get_stat_columns(self, market):
        """Return the candidate feature columns for ``market``.

        Returns the full unfiltered candidate set: Common + league Common +
        all playerProfile, teamProfile, and defenseProfile columns. The
        Filtered SHAP-ranked bucket was removed in the 2026-05-27 no-filter
        rewire (researcher Option C; Akhiat & Touchanti 2024 arXiv:2411.05937 —
        tree ensembles statistically tie FS-filtered vs no-FS across 960
        XGBoost experiments). ``feature_filter.json`` still ships the Common /
        Always buckets; the Filtered bucket is gone from disk.
        """
        league_filter = feature_filter.get(self.league, {})
        self.base_profile()
        cols = (
            feature_filter.get("Common", [])
            + league_filter.get("Common", [])
            + list(self.playerProfile.add_prefix("Player ").columns)
            + list(self.teamProfile.add_prefix("Team ").columns)
            + list(self.defenseProfile.add_prefix("Defense ").columns)
        )
        if "Player team" in cols:
            cols.remove("Player team")
        for pos in self.positions:
            cols.remove(f"Defense {pos}")
        if self.league == "MLB":
            cols.extend(
                ["PF R", "PF OBP", "PF H", "PF 1B", "PF 2B", "PF 3B", "PF HR", "PF BB", "PF K"]
            )

        return sorted(set(cols))

    def _market_position_filter(self, gamelog: pd.DataFrame, market: str) -> pd.DataFrame:
        """Hook: restrict per-gameday training rows to positions eligible for ``market``.

        Base implementation keeps all rows. Position-locked leagues (NFL) override
        this to drop players whose position never accrues the stat, which would
        otherwise enter the matrix as all-zero rows that depress the marginal mean
        and inflate the zero fraction (docs/gbdt_mean_regression_plan.md Stage A1.6).
        """
        return gamelog

    def _dispatch_volume_stats(self, offers, gameDate, market):
        if market in self.volume_stats:
            self.get_depth(offers, gameDate)
        elif self.league == "MLB":
            self.get_volume_stats(
                offers,
                gameDate,
                pitcher=is_mlb_pitcher_market(market),
            )
        elif self.league == "NHL":
            self.get_volume_stats(
                offers,
                gameDate,
                pitcher=any(string in market for string in ["Against", "saves", "goalie"]),
            )
        else:
            self.get_volume_stats(offers, gameDate)

    def _resolve_player_market_odds(self, stats, market, date, target_at):
        evs = []
        lines = []
        odds = []
        archived = []
        for player in stats.index:
            a = True
            ev = archive.get_ev(self.league, market, date, player, at=target_at)
            line = archive.get_line(self.league, market, date, player, at=target_at)
            if np.isnan(ev):
                ev = self.check_combo_markets(market, player, date)
            if line <= 0:
                a = False
                line = np.max([stats.loc[player, "Avg10"], 0.5])
            if ev <= 0:
                ev = get_ev(
                    line,
                    0.5,
                    stat_cv[self.league].get(market, 1),
                    dist=stat_dist.get(self.league, {}).get(market, "Gamma"),
                )

            lines.append(line)
            _cv = stat_cv[self.league].get(market, 1)
            _dist = stat_dist.get(self.league, {}).get(market, "Gamma")
            # Shape-free: reads the stored under_prob directly rather than inverting ev through
            # get_odds, so the training feature is distribution-family-agnostic and Jensen-free.
            # Falls back to get_odds when under_prob was not stored (pre-migration rows).
            under = archive.get_composite_under_prob(
                self.league, market, date, player, at=target_at
            )
            if np.isnan(under):
                under = get_odds(line, ev, _dist, cv=_cv)
            odds.append(1 - under)
            evs.append(ev)
            archived.append(a)
        return lines, odds, evs, archived

    def get_training_matrix(self, market, cutoff_date=None):
        """Retrieves the training data matrix and target labels for a specified market.

        Args:
            market (str): The market type to retrieve training data for.

        Returns:
            X (pd.DataFrame): The training data matrix.
            y (pd.DataFrame): The target labels.
        """
        matrix = []
        mlb_bookless = []

        if cutoff_date is None:
            cutoff_date = (datetime.today() - timedelta(days=850)).date()

        gamelog = self.gamelog.drop_duplicates(
            subset=[self.log_strings["game"], self.log_strings["player"]], keep="last"
        ).copy()
        gamelog[self.log_strings["date"]] = pd.to_datetime(
            gamelog[self.log_strings["date"]]
        ).dt.date
        gamelog = gamelog.loc[
            (gamelog[self.log_strings["date"]] > cutoff_date)
            & (gamelog[self.log_strings["date"]] < datetime.today().date())
        ]
        gamelog = self._market_position_filter(gamelog, market)
        # MLB never set self.usage_stat (its usage path used to be skipped entirely), so read
        # the participation column explicitly: pitcher markets accrue on the mound (batters
        # faced), hitter markets at the plate (log_strings["usage"] == plateAppearances).
        usage_stat = self.usage_stat
        if self.league == "MLB":
            usage_stat = (
                "batters faced" if is_mlb_pitcher_market(market) else self.log_strings["usage"]
            )
        usage_cutoff = gamelog[usage_stat].quantile(0.15)

        gamedays = gamelog.groupby(self.log_strings["date"])
        offerKeys = {
            self.log_strings["player"]: "Player",
            self.log_strings["team"]: "Team",
            self.log_strings["opponent"]: "Opponent",
            self.log_strings["home"]: "Home",
            market: "Result",
        }
        for gameDate, players in tqdm(
            gamedays, unit="gameday", desc="Gathering Training Data", total=len(gamedays)
        ):
            offers_df = players.loc[:, offerKeys.keys()].rename(columns=offerKeys)
            offers_df["Result"] = offers_df["Result"].clip(0, None)

            offers_df.index = offers_df["Player"]
            offers_df = offers_df.loc[~offers_df.index.duplicated()]
            offers = offers_df.to_dict("records")

            self._dispatch_volume_stats(offers, gameDate, market)

            stats = self.get_stats(market, offers, gameDate)
            usage = players[usage_stat]
            usage.index = players[self.log_strings["player"]]
            usage = usage.loc[stats.index]
            usage = usage[~usage.index.duplicated(keep="first")]

            date = gameDate.strftime("%Y-%m-%d")

            # 8 PM UTC stand-in for true commence_time. Per-league actual
            # start times can be wired in later; this default sits well
            # after every league's typical kickoff window so target_at
            # below stays >= the legacy single-point archive sentinel
            # (game_date midnight) and back-compat lookups never miss.
            game_time = datetime.combine(gameDate, datetime.min.time()) + timedelta(hours=20)
            target_at = game_time - TRAINING_LOOKBACK

            lines, odds, evs, archived = self._resolve_player_market_odds(
                stats, market, date, target_at
            )

            stats["Line"] = lines
            stats["Odds"] = odds
            stats["EV"] = evs
            stats = stats.join(offers_df["Result"])
            stats["Player"] = stats.index
            stats["Date"] = date
            stats["GameTime"] = game_time
            stats["Archived"] = archived

            if stats["Home"].dtype == bool:
                if self.league == "MLB":
                    # Book-first: keep only real-odds rows so book-quoted MLB cells stay
                    # Archived-only. Hold participation rows aside as a fallback used only
                    # when the market turns out to have no archived odds at all (below).
                    matrix.extend(stats.loc[stats["Archived"]].to_dict("records"))
                    if not matrix:
                        mlb_bookless.extend(stats.loc[usage > usage_cutoff].to_dict("records"))
                else:
                    matrix.extend(
                        stats.loc[stats["Archived"] | (usage > usage_cutoff)].to_dict("records")
                    )

        if not matrix:
            # No archived odds anywhere for this MLB market — assemble the book-less cell
            # (pitches thrown, pitcher fantasy score) on median-fallback lines instead of
            # returning an empty frame. Empty for every other league (mlb_bookless stays []).
            matrix = mlb_bookless

        return (
            pd.DataFrame(matrix).fillna(0).infer_objects(copy=False).replace([np.inf, -np.inf], 0)
        )

    def trim_gamelog(self):
        """Trims the gamelog to the most recent 21500 rows of data plus one year."""
        ROWS = 50000 if self.league == "NFL" else 21500

        usable_gamelog = self.gamelog[
            self.gamelog[self.log_strings["usage"]]
            >= self.gamelog[self.log_strings["usage"]].quantile(0.15)
        ]
        cutoff_date = pd.to_datetime(usable_gamelog[self.log_strings["date"]]).iloc[
            0
        ].date() + timedelta(days=200)
        usable_gamelog = usable_gamelog.loc[
            pd.to_datetime(usable_gamelog[self.log_strings["date"]]).dt.date > cutoff_date
        ]
        ROWS = min(ROWS, len(usable_gamelog))

        last_training_day = pd.to_datetime(
            usable_gamelog.iloc[-ROWS][self.log_strings["date"]]
        ).date()
        cutoff_date = last_training_day - timedelta(days=300)
        self.gamelog = self.gamelog.loc[
            pd.to_datetime(self.gamelog[self.log_strings["date"]]).dt.date > cutoff_date
        ].copy()

        return last_training_day

    def _convert_to_market_dist(self, v, subline, sub_cv, sub_dist, dist, cv):
        if sub_dist != dist and not np.isnan(v):
            return get_ev(subline, get_odds(subline, v, sub_dist, cv=sub_cv), cv=cv, dist=dist)
        return v

    def _submarket_ev(self, submarket, date, player, dist, cv):
        sub_cv = stat_cv.get(self.league, {}).get(submarket, 1)
        sub_dist = stat_dist.get(self.league, {}).get(submarket, "Gamma")
        v = archive.get_ev(self.league, submarket, date, player)
        subline = archive.get_line(self.league, submarket, date, player)
        v = self._convert_to_market_dist(v, subline, sub_cv, sub_dist, dist, cv)
        return v, subline, sub_cv, sub_dist

    def _combo_market_ev(self, market, date, player, dist, cv):
        ev = 0
        for submarket in combo_props.get(market, []):
            v, _, _, _ = self._submarket_ev(submarket, date, player, dist, cv)
            if np.isnan(v) or v == 0:
                return 0
            ev += v
        return ev

    def _fantasy_default_contribution(
        self, submarket, weight, v, subline, sub_cv, sub_dist, player_games
    ):
        if not (np.isnan(v) or v == 0):
            return v * weight, True
        if subline == 0 and not player_games.empty:
            subline = np.floor(player_games.iloc[-10:][submarket].median()) + 0.5
        if subline != 0:
            under = (player_games[submarket] < subline).mean()
            return get_ev(subline, under, sub_cv, dist=sub_dist) * weight, False
        return 0, False
