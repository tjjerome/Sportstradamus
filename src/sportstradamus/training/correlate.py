"""Per-league player-stat correlation matrix builder.

Produces *stratified* Spearman correlation matrices on residualized per-game
stats with sample-size-aware shrinkage, plus a metadata sidecar for
reproducibility.

Outputs (per league):

* ``data/{LEAGUE}_corr_same_team.csv`` — within-team pair correlations
  indexed by ``(team, market_a, market_b)``.
* ``data/{LEAGUE}_corr_opposing.csv`` — cross-team pair correlations
  indexed by ``(team, team_market, opp_market)``. Both sides keyed by their
  raw (un-prefixed) market names; the file structure encodes the
  team-vs-opponent relationship.
* ``data/correlations/{LEAGUE}_corr_metadata.json`` — date range covered,
  per-team observation counts, generation timestamp, git SHA.

The intermediate per-game record at ``data/training_data/{LEAGUE}_corr.csv``
is still written (warm-start cache) but no longer used by the prediction
pipeline.
"""

import importlib.resources as pkg_resources
import json
import subprocess
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

from sportstradamus import data

# Lookback window for game inclusion. ~1 calendar year covers a full regular
# season + post-season for the major leagues.
LOOKBACK_DAYS: int = 300

# Rolling window for per-player residualization. Eight games trades off bias
# (longer windows mask form swings) and variance (shorter windows leak
# per-game noise into the "mean").
ROLLING_WINDOW_GAMES: int = 8

# Minimum prior-game observations required for a residual to be defined.
# Below this, the residual is left as NaN rather than fit on near-zero history.
MIN_ROLLING_OBSERVATIONS: int = 3

# Minimum shared-game count for a pair to keep its raw correlation. Pairs with
# fewer shared games are shrunk toward zero proportionally to the deficit;
# below this floor a single noisy game can dominate the estimate.
MIN_OVERLAP_FOR_FULL_WEIGHT: int = 30

# Pairs with absolute (post-shrinkage) correlation below this magnitude are
# dropped from the output — keeps the on-disk matrix sparse.
CORR_MAGNITUDE_FLOOR: float = 0.05

_TRACKED_STATS: dict[str, dict] = {
    "NFL": {
        "QB": [
            "passing yards",
            "rushing yards",
            "qb yards",
            "fantasy points prizepicks",
            "fantasy points underdog",
            "passing tds",
            "rushing tds",
            "qb tds",
            "completions",
            "carries",
            "interceptions",
            "attempts",
            "sacks taken",
            "longest completion",
            "longest rush",
            "passing first downs",
            "first downs",
            "fumbles lost",
            "completion percentage",
        ],
        "RB": [
            "rushing yards",
            "receiving yards",
            "yards",
            "fantasy points prizepicks",
            "fantasy points underdog",
            "tds",
            "rushing tds",
            "receiving tds",
            "carries",
            "receptions",
            "targets",
            "longest rush",
            "longest reception",
            "first downs",
            "fumbles lost",
        ],
        "WR": [
            "receiving yards",
            "yards",
            "fantasy points prizepicks",
            "fantasy points underdog",
            "tds",
            "receiving tds",
            "receptions",
            "targets",
            "longest reception",
            "first downs",
            "fumbles lost",
        ],
        "TE": [
            "receiving yards",
            "yards",
            "fantasy points prizepicks",
            "fantasy points underdog",
            "tds",
            "receiving tds",
            "receptions",
            "targets",
            "longest reception",
            "first downs",
            "fumbles lost",
        ],
    },
    "NHL": {
        "G": ["saves", "goalsAgainst", "goalie fantasy points underdog"],
        "C": [
            "points",
            "shots",
            "sogBS",
            "fantasy points prizepicks",
            "skater fantasy points underdog",
            "blocked",
            "hits",
            "goals",
            "assists",
            "faceOffWins",
            "timeOnIce",
        ],
        "W": [
            "points",
            "shots",
            "sogBS",
            "fantasy points prizepicks",
            "skater fantasy points underdog",
            "blocked",
            "hits",
            "goals",
            "assists",
            "faceOffWins",
            "timeOnIce",
        ],
        "D": [
            "points",
            "shots",
            "sogBS",
            "fantasy points prizepicks",
            "skater fantasy points underdog",
            "blocked",
            "hits",
            "goals",
            "assists",
            "faceOffWins",
            "timeOnIce",
        ],
    },
    "NBA": {
        "C": [
            "PTS",
            "REB",
            "AST",
            "PRA",
            "PR",
            "RA",
            "PA",
            "FG3M",
            "fantasy points prizepicks",
            "fantasy points underdog",
            "TOV",
            "BLK",
            "STL",
            "BLST",
            "FG3A",
            "FTM",
            "FGM",
            "FGA",
            "OREB",
            "DREB",
            "PF",
            "MIN",
        ],
        "P": [
            "PTS",
            "REB",
            "AST",
            "PRA",
            "PR",
            "RA",
            "PA",
            "FG3M",
            "fantasy points prizepicks",
            "fantasy points underdog",
            "TOV",
            "BLK",
            "STL",
            "BLST",
            "FG3A",
            "FTM",
            "FGM",
            "FGA",
            "OREB",
            "DREB",
            "PF",
            "MIN",
        ],
        "B": [
            "PTS",
            "REB",
            "AST",
            "PRA",
            "PR",
            "RA",
            "PA",
            "FG3M",
            "fantasy points prizepicks",
            "fantasy points underdog",
            "TOV",
            "BLK",
            "STL",
            "BLST",
            "FG3A",
            "FTM",
            "FGM",
            "FGA",
            "OREB",
            "DREB",
            "PF",
            "MIN",
        ],
        "F": [
            "PTS",
            "REB",
            "AST",
            "PRA",
            "PR",
            "RA",
            "PA",
            "FG3M",
            "fantasy points prizepicks",
            "fantasy points underdog",
            "TOV",
            "BLK",
            "STL",
            "BLST",
            "FG3A",
            "FTM",
            "FGM",
            "FGA",
            "OREB",
            "DREB",
            "PF",
            "MIN",
        ],
        "W": [
            "PTS",
            "REB",
            "AST",
            "PRA",
            "PR",
            "RA",
            "PA",
            "FG3M",
            "fantasy points prizepicks",
            "fantasy points underdog",
            "TOV",
            "BLK",
            "STL",
            "BLST",
            "FG3A",
            "FTM",
            "FGM",
            "FGA",
            "OREB",
            "DREB",
            "PF",
            "MIN",
        ],
    },
    "WNBA": {
        "G": [
            "PTS",
            "REB",
            "AST",
            "PRA",
            "PR",
            "RA",
            "PA",
            "FG3M",
            "fantasy points prizepicks",
            "fantasy points underdog",
            "TOV",
            "BLK",
            "STL",
            "BLST",
            "FG3A",
            "FTM",
            "FGM",
            "FGA",
            "OREB",
            "DREB",
            "PF",
            "MIN",
        ],
        "F": [
            "PTS",
            "REB",
            "AST",
            "PRA",
            "PR",
            "RA",
            "PA",
            "FG3M",
            "fantasy points prizepicks",
            "fantasy points underdog",
            "TOV",
            "BLK",
            "STL",
            "BLST",
            "FG3A",
            "FTM",
            "FGM",
            "FGA",
            "OREB",
            "DREB",
            "PF",
            "MIN",
        ],
        "C": [
            "PTS",
            "REB",
            "AST",
            "PRA",
            "PR",
            "RA",
            "PA",
            "FG3M",
            "fantasy points prizepicks",
            "fantasy points underdog",
            "TOV",
            "BLK",
            "STL",
            "BLST",
            "FG3A",
            "FTM",
            "FGM",
            "FGA",
            "OREB",
            "DREB",
            "PF",
            "MIN",
        ],
    },
    "MLB": {
        "P": [
            "pitcher strikeouts",
            "pitching outs",
            "pitches thrown",
            "hits allowed",
            "runs allowed",
            "1st inning runs allowed",
            "1st inning hits allowed",
            "pitcher fantasy score",
            "pitcher fantasy points underdog",
            "walks allowed",
        ],
        "B": [
            "hitter fantasy score",
            "hitter fantasy points underdog",
            "hits+runs+rbi",
            "total bases",
            "walks",
            "stolen bases",
            "hits",
            "runs",
            "rbi",
            "batter strikeouts",
            "singles",
            "doubles",
            "triples",
            "home runs",
        ],
    },
}


def _git_sha() -> str:
    """Return the short git SHA of the current HEAD, or ``"unknown"`` if not in a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def _residualize_gamelog(
    gamelog: pd.DataFrame,
    player_col: str,
    date_col: str,
    stat_cols: list[str],
) -> pd.DataFrame:
    """Subtract each player's leak-free rolling-N-game mean from each stat column.

    For each player, sort by date and replace ``stat[t]`` with
    ``stat[t] - mean(stat[t-ROLLING_WINDOW_GAMES : t-1])``. The first few games
    of each player's history (less than ``MIN_ROLLING_OBSERVATIONS`` priors)
    yield NaN residuals, which are propagated through the correlation step
    via pairwise-complete observation counting.

    Args:
        gamelog: Per-player per-game DataFrame.
        player_col: Column name holding the player identifier.
        date_col: Column name holding the game date (string or date).
        stat_cols: Stat columns to residualize. Missing columns are skipped.

    Returns:
        A new DataFrame with the same shape and index where each ``stat_cols``
        column has been replaced by its per-player residual.
    """
    present = [c for c in stat_cols if c in gamelog.columns]
    if not present:
        return gamelog.copy()

    out = gamelog.copy()
    order = out[[player_col, date_col]].astype({player_col: "string", date_col: "string"})
    sort_idx = order.sort_values([player_col, date_col], kind="stable").index

    sorted_view = out.loc[sort_idx, [player_col, *present]]
    grouped = sorted_view.groupby(player_col, sort=False)
    for stat in present:
        prior_mean = grouped[stat].transform(
            lambda s: (
                s.shift(1)
                .rolling(ROLLING_WINDOW_GAMES, min_periods=MIN_ROLLING_OBSERVATIONS)
                .mean()
            )
        )
        # Align back to the original index order.
        out.loc[sort_idx, stat] = (sorted_view[stat] - prior_mean).to_numpy()
    return out


def _build_team_game_records(
    league: str,
    log,
    latest_date: datetime,
) -> list[dict]:
    """Iterate the gamelog and emit one record per (team, game) with residualized stats.

    Args:
        league: League key into ``_TRACKED_STATS``.
        log: ``Stats``-shaped instance with ``gamelog``, ``log_strings``,
            ``profile_market``, and ``playerProfile``.
        latest_date: Floor on game dates — earlier games are skipped.

    Returns:
        A list of dicts ready for ``pd.json_normalize``. Each dict has
        ``TEAM``, ``DATE``, plus position-keyed stat values for the team and
        ``_OPP_``-prefixed values for the opposing team.
    """
    stats = _TRACKED_STATS[league]
    log_str = log.log_strings

    flat_stats = sorted({s for col_lists in stats.values() for s in col_lists})
    residualized = _residualize_gamelog(
        log.gamelog,
        log_str["player"],
        log_str["date"],
        flat_stats,
    )

    games = residualized[log_str["game"]].unique()
    records: list[dict] = []

    for gameId in tqdm(games):
        game_df = residualized.loc[residualized[log_str["game"]] == gameId]
        gameDate = datetime.fromisoformat(game_df.iloc[0][log_str["date"]])
        if gameDate < latest_date or len(game_df[log_str["team"]].unique()) != 2:
            continue
        [home_team, away_team] = tuple(
            game_df.sort_values(log_str["home"], ascending=False)[log_str["team"]].unique()
        )

        if league == "MLB":
            bat_df = game_df.loc[game_df["starting batter"]]
            bat_df.position = "B" + bat_df.battingOrder.astype(str)
            bat_df.index = bat_df.position
            pitch_df = game_df.loc[game_df["starting pitcher"]]
            pitch_df.position = "P"
            pitch_df.index = pitch_df.position
            game_df = pd.concat([bat_df, pitch_df])
        else:
            log.profile_market(log_str["usage"], date=gameDate)
            usage = pd.DataFrame(
                log.playerProfile[
                    [f"{log_str.get('usage')} short", f"{log_str.get('usage_sec')} short"]
                ]
            )
            usage.reset_index(inplace=True)
            game_df = game_df.merge(usage, how="left")
            game_df = game_df.loc[game_df[log_str["position"]].apply(lambda x: isinstance(x, str))]
            # Usage columns are non-stat metadata; safe to fill NaN here.
            for usage_col in (f"{log_str.get('usage')} short", f"{log_str.get('usage_sec')} short"):
                if usage_col in game_df.columns:
                    game_df[usage_col] = game_df[usage_col].fillna(0)
            ranks = (
                game_df.sort_values(f"{log_str.get('usage_sec')} short", ascending=False)
                .groupby([log_str["team"], log_str["position"]])
                .rank(ascending=False, method="first")[f"{log_str.get('usage')} short"]
                .astype(int)
            )
            game_df[log_str["position"]] = game_df[log_str["position"]] + ranks.astype(str)
            game_df.index = game_df[log_str["position"]]

        homeStats: dict = {}
        awayStats: dict = {}
        for position in stats:
            homeStats.update(
                game_df.loc[
                    (game_df[log_str["team"]] == home_team)
                    & game_df[log_str["position"]].str.contains(position),
                    stats[position],
                ].to_dict("index")
            )
            awayStats.update(
                game_df.loc[
                    (game_df[log_str["team"]] == away_team)
                    & game_df[log_str["position"]].str.contains(position),
                    stats[position],
                ].to_dict("index")
            )

        records.append(
            {"TEAM": home_team}
            | {"DATE": gameDate.date()}
            | homeStats
            | {"_OPP_" + k: v for k, v in awayStats.items()}
        )
        records.append(
            {"TEAM": away_team}
            | {"DATE": gameDate.date()}
            | awayStats
            | {"_OPP_" + k: v for k, v in homeStats.items()}
        )
    return records


def _shrink_correlations(
    corr: pd.DataFrame,
    overlap: pd.DataFrame,
) -> pd.DataFrame:
    """Pull pair correlations toward zero in proportion to the overlap deficit.

    For pairs with at least ``MIN_OVERLAP_FOR_FULL_WEIGHT`` shared games, the
    correlation is unchanged. Below that floor, the correlation is multiplied
    by ``overlap / MIN_OVERLAP_FOR_FULL_WEIGHT`` (a credibility weight that
    hits zero when no games overlap).

    Args:
        corr: Square pairwise correlation DataFrame.
        overlap: Square pairwise overlap-count DataFrame, same shape and labels.

    Returns:
        Correlation DataFrame with shrinkage applied in place of zero-padding
        for low-sample pairs.
    """
    weight = (overlap / MIN_OVERLAP_FOR_FULL_WEIGHT).clip(upper=1.0)
    return corr * weight


def _stratify_team_pairs(team_corr: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Split a team's stacked pair series into same-team and opposing series.

    Args:
        team_corr: Series indexed by ``(market_a, market_b)`` for one team's
            pairs. Markets prefixed with ``_OPP_`` denote opponent stats.

    Returns:
        ``(same_team, opposing)`` where:
            * ``same_team`` keeps pairs where neither marker has the ``_OPP_``
              prefix; the opposite-side ``(_OPP_a, _OPP_b)`` pairs are dropped
              because they are the same team's opponent's same-team
              correlations and belong to ``same_team`` keyed under the
              opponent's row, not this team's.
            * ``opposing`` keeps pairs with exactly one ``_OPP_`` side,
              normalized so the team's market is the first index level and
              the opp's market (without the ``_OPP_`` prefix) is the second.
    """
    is_opp_a = team_corr.index.get_level_values(0).str.startswith("_OPP_")
    is_opp_b = team_corr.index.get_level_values(1).str.startswith("_OPP_")

    same_mask = ~is_opp_a & ~is_opp_b
    cross_mask = is_opp_a ^ is_opp_b

    same = team_corr[same_mask]

    cross = team_corr[cross_mask].copy()
    cross_idx_a = cross.index.get_level_values(0).to_numpy()
    cross_idx_b = cross.index.get_level_values(1).to_numpy()
    cross_is_opp_a = is_opp_a[cross_mask]

    team_market = np.where(cross_is_opp_a, cross_idx_b, cross_idx_a)
    opp_market = np.where(cross_is_opp_a, cross_idx_a, cross_idx_b)
    opp_market = np.array([s.removeprefix("_OPP_") for s in opp_market])

    cross.index = pd.MultiIndex.from_arrays([team_market, opp_market], names=team_corr.index.names)
    cross = cross[~cross.index.duplicated(keep="first")]
    return same, cross


def correlate(league: str, stat_data, force: bool = False) -> None:
    """Build stratified per-league correlation matrices and metadata sidecar.

    Process per league:

    1. Load (or warm-start) the per-game record cache; trim to the
       ``LOOKBACK_DAYS`` window.
    2. Replace each stat with its per-player ``ROLLING_WINDOW_GAMES``-game
       residual.
    3. Build per-team matrices, computing pairwise Spearman correlations
       and pairwise overlap counts.
    4. Apply shrinkage proportional to the overlap deficit below
       ``MIN_OVERLAP_FOR_FULL_WEIGHT``.
    5. Stratify into same-team and opposing pair series; drop pairs below
       ``CORR_MAGNITUDE_FLOOR``.
    6. Write the two CSVs and the metadata JSON sidecar.

    Args:
        league: One of NFL/NBA/MLB/NHL/WNBA.
        stat_data: A loaded ``Stats`` instance with ``gamelog``,
            ``log_strings``, ``profile_market``, and ``playerProfile``.
        force: When True, rebuilds the per-game record cache from scratch
            instead of reusing the prior run's CSV.
    """
    print(f"Correlating {league}...")
    log = stat_data

    training_data_dir = pkg_resources.files(data) / "training_data"
    training_data_dir.mkdir(parents=True, exist_ok=True)
    raw_filepath = training_data_dir / f"{league}_corr.csv"
    if raw_filepath.is_file() and not force:
        matrix = pd.read_csv(raw_filepath, index_col=0)
        if "DATE" in matrix.columns:
            matrix["DATE"] = pd.to_datetime(matrix["DATE"], format="mixed")
            latest_date = matrix["DATE"].max()
            matrix = matrix.loc[datetime.today() - timedelta(days=LOOKBACK_DAYS) <= matrix["DATE"]]
        else:
            matrix = pd.DataFrame()
            latest_date = datetime.today() - timedelta(days=LOOKBACK_DAYS)
    else:
        matrix = pd.DataFrame()
        latest_date = datetime.today() - timedelta(days=LOOKBACK_DAYS)

    new_records = _build_team_game_records(league, log, latest_date)
    matrix = pd.concat([matrix, pd.json_normalize(new_records)], ignore_index=True)
    matrix.to_csv(raw_filepath)

    matrix_for_corr = matrix.drop(columns="DATE", errors="ignore")

    same_team_blocks: dict = {}
    opposing_blocks: dict = {}
    per_team_obs: dict[str, int] = {}

    teams_iter = matrix_for_corr["TEAM"].unique() if "TEAM" in matrix_for_corr.columns else []
    for team in teams_iter:
        team_matrix = matrix_for_corr.loc[team == matrix_for_corr["TEAM"]].drop(columns="TEAM")
        # Drop columns that are entirely NaN (residuals never defined for any
        # game) — they cannot contribute to a correlation estimate.
        team_matrix = team_matrix.loc[:, team_matrix.notna().any(axis=0)]
        if team_matrix.shape[1] < 2 or len(team_matrix) < 2:
            continue
        team_matrix = team_matrix.reindex(sorted(team_matrix.columns), axis=1)

        per_team_obs[str(team)] = len(team_matrix)

        # Pairwise overlap = number of games where both columns have a defined residual.
        present = team_matrix.notna().astype(int)
        overlap = present.T @ present

        c_spearman = team_matrix.corr(method="spearman")
        c_remap = 2 * np.sin(np.pi / 6 * c_spearman)
        c_shrunk = _shrink_correlations(c_remap, overlap)

        c_stack = c_shrunk.unstack().dropna()
        c_stack = c_stack.loc[c_stack.abs() > CORR_MAGNITUDE_FLOOR]
        c_stack = c_stack.reindex(c_stack.abs().sort_values(ascending=False).index)

        same, opposing = _stratify_team_pairs(c_stack)
        if not same.empty:
            same_team_blocks[team] = same
        if not opposing.empty:
            opposing_blocks[team] = opposing

    same_path = pkg_resources.files(data) / f"{league}_corr_same_team.csv"
    opposing_path = pkg_resources.files(data) / f"{league}_corr_opposing.csv"
    if same_team_blocks:
        pd.concat(same_team_blocks).to_csv(same_path)
    else:
        pd.DataFrame(columns=["R"]).to_csv(same_path)
    if opposing_blocks:
        pd.concat(opposing_blocks).to_csv(opposing_path)
    else:
        pd.DataFrame(columns=["R"]).to_csv(opposing_path)

    metadata_dir = pkg_resources.files(data) / "correlations"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / f"{league}_corr_metadata.json"
    dates = (
        pd.to_datetime(matrix.DATE)
        if "DATE" in matrix.columns
        else pd.Series(dtype="datetime64[ns]")
    )
    metadata = {
        "league": league,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "lookback_days": LOOKBACK_DAYS,
        "rolling_window_games": ROLLING_WINDOW_GAMES,
        "min_overlap_for_full_weight": MIN_OVERLAP_FOR_FULL_WEIGHT,
        "corr_magnitude_floor": CORR_MAGNITUDE_FLOOR,
        "date_range": {
            "start": dates.min().date().isoformat() if not dates.empty else None,
            "end": dates.max().date().isoformat() if not dates.empty else None,
        },
        "total_team_game_observations": len(matrix),
        "per_team_observations": per_team_obs,
    }
    with open(metadata_path, "w") as fh:
        json.dump(metadata, fh, indent=2)
