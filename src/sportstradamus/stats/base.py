"""Stats base class: shared data loading, feature engineering, and prediction API."""

import importlib.resources as pkg_resources
import json
import os.path
import pickle
import warnings
from datetime import datetime, timedelta
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

# Quantile bands for the ``comps p25`` / ``comps p75`` signals. Use the
# distance-weighted comp distribution rather than the raw comp set: a
# closer comp gets more vote, matching how ``comps mean`` is built.
_COMP_QUANTILE_LO: float = 0.25
_COMP_QUANTILE_HI: float = 0.75

# Clip bounds for per-player projection scale ratios in get_volume_stats.
# Floor prevents shrinking a player below 10% of their mean; cap prevents
# inflating beyond 10×. Shared across NBA, WNBA, NHL, and NFL volume models.
_VOLUME_SCALE_RATIO_FLOOR: float = 0.1
_VOLUME_SCALE_RATIO_CAP: float = 10.0

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
        ml_map = archive.get_team_market_map(
            self.league, "Moneyline", dates=unique_dates
        )
        tot_map = archive.get_team_market_map(
            self.league, "Totals", dates=unique_dates
        )
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
        pass

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

        records = []
        for position, pos_comps in self.comps.items():
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

        _filled_gl = self.short_gamelog.fillna(0).infer_objects(copy=False)
        _player_col = self.log_strings["player"]
        # Intersect stat_types with available gamelog columns so partial fixtures
        # (e.g. tests/integration's cached parquets that don't reload the full
        # gamelog) don't KeyError. Production runs have all expected columns
        # after Stats.update(); this is purely defensive for fixture mode.
        _avail_stats = [s for s in stat_types if s in _filled_gl.columns]
        playerstats = _filled_gl.groupby(_player_col)[_avail_stats].mean(numeric_only=True)

        # Vectorized tail(5).mean() — avoids per-group .apply()
        _last5 = _filled_gl.groupby(_player_col).tail(5)
        playershortstats = (
            _last5.groupby(_last5[_player_col])[_avail_stats]
            .mean()
            .fillna(0)
            .infer_objects(copy=False)
            .add_suffix(" short", 1)
        )

        # Vectorized trend (slope of last 5 games) — replaces per-group polyfit
        # slope = (N*Σ(rank*y) - Σrank*Σy) / (N*Σrank² - (Σrank)²)
        _last5t = _last5.copy()
        _last5t["_rank"] = _last5t.groupby(_player_col).cumcount()
        _grp_t = _last5t.groupby(_player_col)
        _n_t = _grp_t.size()
        _sum_r = _grp_t["_rank"].sum()
        _sum_r2 = (_last5t["_rank"] ** 2).groupby(_last5t[_player_col]).sum()
        _denom_t = (_n_t * _sum_r2 - _sum_r**2).replace(0, np.nan)
        _valid_stats = [s for s in stat_types if s in _last5t.columns]
        _sum_y = _grp_t[_valid_stats].sum()
        _iy = _last5t[_valid_stats].multiply(_last5t["_rank"], axis=0)
        _sum_iy = _iy.groupby(_last5t[_player_col]).sum()
        playertrends = (_sum_iy.multiply(_n_t, axis=0) - _sum_y.multiply(_sum_r, axis=0)).div(
            _denom_t, axis=0
        )
        # Original get_trends returns zeros for players with < 3 total games
        _total_games = _filled_gl.groupby(_player_col).size()
        playertrends.loc[_total_games < 3] = 0.0
        playertrends = playertrends.fillna(0).infer_objects(copy=False).add_suffix(" growth", 1)

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

        # Vectorized tail(10).mean() for team stats
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

        self.playerProfile = (
            self.playerProfile.join(playerstats, how="right").fillna(0).infer_objects(copy=False)
        )
        self.playerProfile.drop_duplicates(inplace=True)

    @line_profiler.profile
    def profile_market(self, market, date=datetime.today().date()):
        if isinstance(date, str):
            date = datetime.strptime(date, "%Y-%m-%d").date()
        elif isinstance(date, datetime):
            date = date.date()
        if market == self.profiled_market and date == self.profile_latest_date:
            return

        self.base_profile(date)
        self.profiled_market = market

        # Replace slow .filter(lambda) with aggregation-based pre-filtering
        _pc = self.log_strings["player"]
        _grp_pre = self.short_gamelog.groupby(_pc)[market]
        _agg = _grp_pre.agg(count="count", clipped_mean=lambda x: x.clip(0, 1).mean())
        _valid_players = _agg[(_agg["clipped_mean"] > 0.1) & (_agg["count"] > 1)].index
        _filtered = self.short_gamelog[self.short_gamelog[_pc].isin(_valid_players)]
        playerGroups = _filtered.groupby(_pc)

        leagueavg = playerGroups[market].mean().mean()
        leaguestd = playerGroups[market].mean().std()
        if np.isnan(leagueavg) or np.isnan(leaguestd):
            return

        self.playerProfile[
            ["z", "home", "moneyline gain", "totals gain", "position z", "comps mean", "comps z"]
        ] = 0.0
        self.playerProfile["z"] = (playerGroups[market].mean() - leagueavg).div(leaguestd)

        # Vectorized home split — avoids per-group .apply()
        _home_col = self.log_strings["home"]
        _home_mask = _filtered[_home_col].astype(bool)
        _home_mean = _filtered.loc[_home_mask].groupby(_pc)[market].mean()
        _all_mean = playerGroups[market].mean()
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

        # Vectorized defense home split
        _def_home_mask = defenseGames[self.log_strings["home"]].astype(bool)
        _def_home_mean = (
            defenseGames.loc[_def_home_mask]
            .groupby(self.log_strings["opponent"])[market]
            .mean()
            .clip(0.1)
        )
        _def_all_mean = defenseGroups[market].mean().clip(0.1)
        self.defenseProfile["home"] = _def_home_mean / _def_all_mean - 1

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

        # Vectorized polyfit — replaces per-group np.polyfit with batch
        # slope = (N*Σ(X*Y) - ΣX*ΣY) / (N*ΣX² - (ΣX)²)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # --- Player moneyline gain ---
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

            # --- Player totals gain ---
            _tot_filled = _filtered["totals"].fillna(self.default_total).astype(float)
            _tot_mean_p = _tot_filled.groupby(_filtered[_pc]).transform("mean")
            _X_tot = _tot_filled / self.default_total - _tot_mean_p
            _SXY_t = (_X_tot * _Y_p).groupby(_filtered[_pc]).sum()
            _SX_t = _X_tot.groupby(_filtered[_pc]).sum()
            _SX2_t = (_X_tot**2).groupby(_filtered[_pc]).sum()
            _denom_t = (_n_p * _SX2_t - _SX_t**2).replace(0, np.nan)
            self.playerProfile["totals gain"] = (_n_p * _SXY_t - _SX_t * _SY) / _denom_t

            # --- Defense moneyline gain ---
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

            # --- Defense totals gain ---
            _def_tot = _dg["totals"].fillna(self.default_total).astype(float)
            _def_tot_mean = _def_tot.groupby(_dg[_opp_col]).transform("mean")
            _X_def_tot = _def_tot / self.default_total - _def_tot_mean
            _SXY_dt = (_X_def_tot * _Y_def).groupby(_dg[_opp_col]).sum()
            _SX_dt = _X_def_tot.groupby(_dg[_opp_col]).sum()
            _SX2_dt = (_X_def_tot**2).groupby(_dg[_opp_col]).sum()
            _denom_dt = (_n_def * _SX2_dt - _SX_dt**2).replace(0, np.nan)
            self.defenseProfile["totals gain"] = (_n_def * _SXY_dt - _SX_dt * _SY_def) / _denom_dt

        # Player comps mean: distance-weighted average of comps' market means.
        # Player comps mean (EB): Efron-Morris shrinkage of comps mean toward
        #   the position baseline. Reduces tail noise on low-sample players.
        # Player comps p25 / p75: distance-weighted quantiles of comp means.
        #   Lets the GBDT discriminate "tight cluster, confident estimate" vs
        #   "wide cluster, uncertain estimate" without emitting a redundant std.
        # The static ``comps z`` (player vs comp-pool level ranking) was
        # removed -- it carries no matchup signal and is redundant with
        # ``comps mean`` next to ``MeanYr`` for the GBDT. The matchup-conditional
        # ``Player comps z`` is computed per-prediction in ``get_stats``.
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

                # Distance-weighted comp p25 / p75: numpy's standard weighted-
                # quantile recipe (np.interp(q, cum_w_norm, sorted_vals))
                # vectorized across all players in a single sort + cumsum pass
                # via :meth:`_vectorized_weighted_quantiles`. Replaces the
                # per-group ``DataFrame.groupby.apply`` recipe that dominated
                # the per-gameday cost on training matrix generation.
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

        self.defenseProfile.fillna(0.0, inplace=True)
        self.teamProfile.fillna(0.0, inplace=True)
        self.playerProfile.fillna(0.0, inplace=True)

    def get_depth(self, offers, date=datetime.today().date()):
        players = list(offers.keys()) if isinstance(offers, dict) else [x["Player"] for x in offers]

        for player in players:
            if " + " in player.replace(" vs. ", " + "):
                split_players = player.replace(" vs. ", " + ").split(" + ")
                players.append(split_players[0])
                players.append(split_players[1])

        players = set(players)
        season = (
            date.year if ((date.month >= 8) or (self.league in ["WNBA", "MLB"])) else date.year - 1
        )
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
        stats = pd.DataFrame(
            columns=[
                "Avg1",
                "Avg3",
                "Avg5",
                "Avg10",
                "AvgYr",
                "AvgH2H",
                "Mean10",
                "MeanYr",
                "MeanYr_nonzero",
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
            ]
        )
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

        playergames = self.short_gamelog.loc[
            self.short_gamelog[self.log_strings["player"]].isin(players)
        ]

        if playergames.empty:
            return stats

        _player_col = self.log_strings["player"]

        # Vectorized tail-based stats — avoids per-group .apply(lambda)
        _pg_last10 = playergames.groupby(_player_col).tail(10)
        _pg_last5 = playergames.groupby(_player_col).tail(5)
        _pg_last3 = playergames.groupby(_player_col).tail(3)
        _pg_last1 = playergames.groupby(_player_col).tail(1)

        stats["Avg1"] = _pg_last1.groupby(_player_col)[market].median()
        stats["Avg3"] = _pg_last3.groupby(_player_col)[market].median()
        stats["Avg5"] = _pg_last5.groupby(_player_col)[market].median()
        stats["Avg10"] = _pg_last10.groupby(_player_col)[market].median()
        stats["AvgYr"] = playergames.groupby(_player_col)[market].median()
        stats["Mean10"] = _pg_last10.groupby(_player_col)[market].mean()
        stats["MeanYr"] = playergames.groupby(_player_col)[market].mean()
        _nonzero_games = playergames.loc[playergames[market] > 0]
        stats["MeanYr_nonzero"] = _nonzero_games.groupby(_player_col)[market].mean()
        stats["MeanYr_nonzero"] = stats["MeanYr_nonzero"].fillna(stats["MeanYr"].clip(lower=0.5))
        stats["STD10"] = _pg_last10.groupby(_player_col)[market].std()
        stats["STDYr"] = playergames.groupby(_player_col)[market].std()
        _last_date = pd.to_datetime(
            playergames.groupby(_player_col)[self.log_strings["date"]].last()
        )
        stats["DaysOff"] = (pd.Timestamp(date) - _last_date).dt.days
        stats["DaysIntoSeason"] = (date - self.season_start).days
        stats["GamesPlayed"] = playergames.groupby(_player_col)[market].count()
        _zero_counts = playergames.loc[playergames[market] == 0].groupby(_player_col).size()
        stats["ZeroYr"] = _zero_counts.reindex(stats.index, fill_value=0) / stats["GamesPlayed"]

        # Scale-invariant ratio features: relative to player baseline
        _meanyr_safe = stats["MeanYr"].clip(lower=0.5)
        stats["Avg5_Ratio"] = stats["Avg5"] / _meanyr_safe
        stats["Avg10_Ratio"] = stats["Avg10"] / _meanyr_safe
        stats["Mean10_Ratio"] = stats["Mean10"] / _meanyr_safe
        stats["STD_Ratio"] = stats["STDYr"] / _meanyr_safe

        stats = stats.loc[~stats.index.duplicated()]

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

        if self.league == "MLB" and not any([string in market for string in ["allowed", "pitch"]]):
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
            defstats["position"] = np.diag(defstats.iloc[:, stats["Player position"] + 5])
            defstats.drop(columns=self.positions, inplace=True)

        defstats.index = stats.index

        # The matchup-conditional residual signal now writes to
        # ``stats["Player comps z"]`` (was the misnamed ``defstats["comps"]``
        # under the ``Defense `` prefix). The count + uniqueness signals stay
        # on the defstats side under their legacy names (``Defense comp n``,
        # ``Defense comp distance``) so cached training matrices and the
        # SHAP-rebuild filter survive without a regeneration pass.
        #
        # Pre-compute per-player zscore of market column for comps lookups.
        _opp_col = self.log_strings["opponent"]
        _player_col = self.log_strings["player"]
        _gl_z = self.short_gamelog[[_player_col, _opp_col, market]].copy()
        _gl_z[market] = _gl_z[market].astype(float)
        _gl_groups = _gl_z.groupby(_player_col)[market]
        _std = _gl_groups.transform("std").replace(0, np.nan)
        _gl_z["_mkt_zscore"] = ((_gl_z[market] - _gl_groups.transform("mean")) / _std).fillna(0)
        stats["Player comps z"] = 0.0

        if self.league == "MLB":
            _is_pitch_market = any(s in market for s in ["allowed", "pitch"])
            for player, row in stats.iterrows():
                playerGames = self.short_gamelog.loc[self.short_gamelog[_player_col] == player]
                if playerGames.empty:
                    continue
                pid = playerGames["playerId"].mode()[0]
                if _is_pitch_market:
                    comps = self.comps["pitchers"].get(pid, [pid])
                    compGames = self.short_gamelog.loc[
                        self.short_gamelog["playerId"].isin(comps)
                        & self.short_gamelog["starting pitcher"]
                    ]
                    if compGames.empty:
                        continue
                else:
                    comps = self.comps["hitters"].get(pid, [pid])
                    compGames = self.short_gamelog.loc[
                        self.short_gamelog["playerId"].isin(comps)
                        & self.short_gamelog["starting batter"]
                    ]
                    if compGames.empty:
                        continue

                    pitch_id = self.short_gamelog.loc[
                        self.short_gamelog[_player_col] == player, "opponent pitcher id"
                    ]
                    if pitch_id.empty:
                        continue

                    pitch_id = pitch_id.mode()[0]
                    pitchComps = self.comps["pitchers"].get(pitch_id, [pitch_id])
                    pitchGames = playerGames.loc[
                        playerGames["opponent pitcher id"].isin(pitchComps)
                    ]
                    if pitchGames.empty or pitchGames[market].mean() == 0:
                        stats.loc[player, "Pitcher comps"] = 0
                    else:
                        stats.loc[player, "Pitcher comps"] = (
                            playerGames[market].mean() / pitchGames[market].mean()
                        )

                compGames[market] = compGames[market].astype(float)
                _comp_groups = compGames.groupby(_player_col)[market]
                _comp_std = _comp_groups.transform("std").replace(0, np.nan)
                scores = ((compGames[market] - _comp_groups.transform("mean")) / _comp_std).fillna(
                    0
                )

                _gl_groups = _gl_z.groupby(_player_col)[market]
                _gl_std = _gl_groups.transform("std").replace(0, np.nan)
                _gl_z["_mkt_zscore"] = (
                    (_gl_z[market] - _gl_groups.transform("mean")) / _gl_std
                ).fillna(0)
                scores.index = scores.index.droplevel(0)
                compGames[market] = scores
                opp_comp_games = compGames.loc[compGames[_opp_col] == opponents[player], market]
                stats.loc[player, "Player comps z"] = opp_comp_games.mean()
                defstats.loc[player, "comp n"] = opp_comp_games.count()
        else:
            # Non-MLB matchup-conditional comp lookup. The static
            # ``(player, comp, position, dist, weight)`` table is built once
            # per comp-pool swap by :meth:`_comp_pairs`; here we only attach
            # the per-call opponent and merge against the per-day ``_gl_z``.
            _pairs = self._comp_pairs()
            if not _pairs.empty:
                _pos_idx = stats["Player position"].astype(int) - 1
                _valid = (_pos_idx >= 0) & (_pos_idx < len(self.positions))
                if _valid.any():
                    _expected_pos = pd.Series(
                        np.asarray(self.positions)[_pos_idx[_valid].to_numpy()],
                        index=stats.index[_valid],
                        name="expected_pos",
                    )
                    # Keep only pairs whose position matches the stats-side
                    # position assignment. Preserves the original lookup
                    # semantics (``self.comps[stats_position].get(player, ...)``)
                    # for players whose ``Player position`` disagrees with the
                    # comp-pool assignment.
                    _cp_df = _pairs.merge(
                        _expected_pos.rename_axis("target_player").reset_index(),
                        left_on="player",
                        right_on="target_player",
                        how="inner",
                    )
                    _cp_df = _cp_df[_cp_df["position"] == _cp_df["expected_pos"]]
                    if not _cp_df.empty:
                        _cp_df["opp"] = _cp_df["player"].map(opponents)
                        # Mean distance to comps (player-side uniqueness signal --
                        # not matchup-dependent). Carried on defstats only because
                        # cached training matrices know it under the ``Defense `` prefix.
                        defstats["comp distance"] = (
                            _cp_df.groupby("player")["dist"].mean().reindex(defstats.index)
                        )
                        # Inner-join comp games vs the opponent. The result is the
                        # matchup-conditional residual signal: each comp's outcome
                        # vs this opponent z-scored against the comp's own per-player
                        # baseline, distance-weighted across the comp set.
                        _merged = _cp_df.merge(
                            _gl_z[[_player_col, _opp_col, "_mkt_zscore"]],
                            left_on=["comp", "opp"],
                            right_on=[_player_col, _opp_col],
                            how="inner",
                            suffixes=("", "_gl"),
                        )
                        _merged["weighted_z"] = _merged["_mkt_zscore"] * _merged["weight"]
                        _comp_wsum = _merged.groupby("player")["weighted_z"].sum()
                        _comp_wcount = _merged.groupby("player")["weight"].sum()
                        _comp_means = _comp_wsum / _comp_wcount
                        stats["Player comps z"] = _comp_means.reindex(stats.index).fillna(0.0)
                        # Count of (comp, opponent) game observations backing each
                        # ``Player comps z`` estimate. Stays on defstats for cached-
                        # matrix compatibility.
                        defstats["comp n"] = (
                            _merged.groupby("player").size().reindex(defstats.index)
                        )

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

        return stats.fillna(0).infer_objects(copy=False)

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

    def get_training_matrix(self, market, cutoff_date=None):
        """Retrieves the training data matrix and target labels for a specified market.

        Args:
            market (str): The market type to retrieve training data for.

        Returns:
            X (pd.DataFrame): The training data matrix.
            y (pd.DataFrame): The target labels.
        """
        matrix = []

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
        if self.league != "MLB":
            usage_cutoff = gamelog[self.usage_stat].quantile(0.15)

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

            if market in self.volume_stats:
                self.get_depth(offers, gameDate)
            elif self.league == "MLB":
                self.get_volume_stats(
                    offers,
                    gameDate,
                    pitcher=any([string in market for string in ["allowed", "pitch"]]),
                )
            elif self.league == "NHL":
                self.get_volume_stats(
                    offers,
                    gameDate,
                    pitcher=any([string in market for string in ["Against", "saves", "goalie"]]),
                )
            else:
                self.get_volume_stats(offers, gameDate)

            stats = self.get_stats(market, offers, gameDate)
            if self.league != "MLB":
                usage = players[self.usage_stat]
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
                odds.append(1 - get_odds(line, ev, _dist, cv=_cv))
                evs.append(ev)
                archived.append(a)

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
                    matrix.extend(stats.loc[stats["Archived"]].to_dict("records"))
                else:
                    matrix.extend(
                        stats.loc[stats["Archived"] | (usage > usage_cutoff)].to_dict("records")
                    )

        M = pd.DataFrame(matrix).fillna(0).infer_objects(copy=False).replace([np.inf, -np.inf], 0)

        return M

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
