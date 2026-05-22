# Correlate Methodology Audit

Audit of `src/sportstradamus/training/correlate.py` (`correlate(league, stat_data, force=False)`),
the function that produces the per-league `{LEAGUE}_corr.csv` matrices consumed by the
prediction pipeline's `find_correlation`.

This document is a read-only methodology audit. No code changes are proposed here.

## Top-line findings

- **Single data source.** `correlate` reads only from the in-memory `Stats.gamelog`
  passed in as `stat_data`; it does not touch the `Archive`, the trained model
  pickles, or any cached training-matrix CSV. The only on-disk read is the function's
  own previous output (`training_data/{LEAGUE}_corr.csv`).
- **Hard-coded ~300 day window.** The lookback is a fixed `timedelta(days=300)` from
  `datetime.today()`; there is no league-aware season cutoff and no configurable knob.
- **Raw values, no residuals, no decay.** Correlations are computed directly on the
  per-game raw stat values. There is no per-player de-meaning, no opponent/venue
  adjustment, and no time-decay weighting — every retained game counts equally.
- **Spearman + Fisher-style remap.** Pairwise correlation is `team_matrix.corr(method="spearman", min_periods=int(len(team_matrix) * 0.75))` followed by
  `c = 2 * np.sin(np.pi / 6 * c_spearman)` (the Spearman→Pearson remap for elliptical
  distributions). Pairs with `|c| <= 0.05` are dropped; no shrinkage is applied to
  surviving pairs.
- **Per-team matrices, no same-team / opposing-team stratification.** A single matrix
  is built per `team`, mixing that team's own player columns and `_OPP_*` columns
  (the same game's opposing team's stats) in one pass. Same-team and opposing-team
  pair correlations therefore live in the *same* matrix but are not labelled as such
  in the output, and there is no separate "cross-game" correlation.
- **Two CSV outputs, only one is consumed.** The intermediate per-game record is
  written to `data/training_data/{LEAGUE}_corr.csv` (with a `DATE` column), and the
  final correlation table is written to `data/{LEAGUE}_corr.csv` (no metadata
  columns, no sample sizes). Only the latter is read by `find_correlation`.
- **Minimum-overlap threshold is implicit.** `min_periods=int(len(team_matrix) * 0.75)`
  means a pair needs values in ≥75% of that team's rows; otherwise the cell becomes
  `NaN` and is dropped. No partial-overlap pair gets a shrunk estimate — it gets
  thrown out.
- **Column-level pre-filter.** Before correlation, columns where ≥50% of values are
  exactly zero are dropped (`team_matrix.loc[:, ((team_matrix == 0).mean() < 0.5)]`).
  Combined with `fillna(0)` upstream, this conflates "did not play" with "scored
  zero" and silently kills any sparse stat.
- **Two callers; one is a re-export.** `correlate` is invoked exactly once at
  `training/cli.py:135` inside the `meditate` per-league loop. It is also re-exported
  from `training/__init__.py` but no other module calls it. It is import-clean for
  standalone use given a loaded `Stats` instance.
- **Last meaningful change was Mar 2026.** Commit `85e0acf` ("Refactor correlation as
  beam search, use spearman correlation coefficients") introduced the current Spearman
  + sin-remap formulation. The Apr 2026 commit `c305a14` was a pure file-move from
  `src/sportstradamus/correlation.py` (the old monolith) into `training/correlate.py`
  and a deprecation-archive cleanup; no methodology changes since March.

---

## 1. Inputs

### 1a. Where does the data come from?

The function reads from the `Stats` instance passed in as `stat_data`:

```python
206  def correlate(league: str, stat_data, force: bool = False) -> None:
207      """Calculate feature correlations with outcomes for feature engineering."""
208      print(f"Correlating {league}...")
209      stats = _TRACKED_STATS[league]
210      log = stat_data
211      log_str = log.log_strings
...
223      games = log.gamelog[log_str["game"]].unique()
...
227          game_df = log.gamelog.loc[log.gamelog[log_str["game"]] == gameId]
```

So the input is exclusively `stat_data.gamelog` (the per-player per-game DataFrame
populated by `Stats.load()` / `Stats.update()`). No call is made to:

- `Archive` (no `from sportstradamus.helpers import archive`, no `archive.get_*`).
- Trained model pickles in `data/models/`.
- Cached training matrices in `data/training_data/` *other than* the prior correlate
  output (see below).

The only on-disk read is the function's own previous output, used as a warm-start cache:

```python
213      filepath = pkg_resources.files(data) / f"training_data/{league}_corr.csv"
214      if filepath.is_file() and not force:
215          matrix = pd.read_csv(filepath, index_col=0)
216          matrix.DATE = pd.to_datetime(matrix.DATE, format="mixed")
217          latest_date = matrix.DATE.max()
218          matrix = matrix.loc[datetime.today() - timedelta(days=300) <= matrix.DATE]
219      else:
220          matrix = pd.DataFrame()
221          latest_date = datetime.today() - timedelta(days=300)
```

### 1b. Time window

A flat 300-day rolling window from "today":

```python
218          matrix = matrix.loc[datetime.today() - timedelta(days=300) <= matrix.DATE]
...
221          latest_date = datetime.today() - timedelta(days=300)
...
228          gameDate = datetime.fromisoformat(game_df.iloc[0][log_str["date"]])
229          if gameDate < latest_date or len(game_df[log_str["team"]].unique()) != 2:
230              continue
```

Notes on the window:

- 300 days is a magic number defined inline (no module-level constant); it violates
  CLAUDE.md §"No magic numbers".
- It is keyed off `datetime.today()`, so the window is wall-clock dependent and
  therefore non-reproducible if rerun on a different date with the same `Stats` data.
- When the cache exists, `latest_date` is the *max* date already in the cache, so on
  warm-start runs only newly-arrived games (since the last run) are processed; the
  300-day floor is enforced only during the cache-prune (line 218) and the empty-cache
  initial seed (line 221), not during incremental top-up.

### 1c. Per-game eligibility

Two filters drop games entirely:

```python
229          if gameDate < latest_date or len(game_df[log_str["team"]].unique()) != 2:
230              continue
```

Games involving exactly two teams are required (so neutral-site / oddly-tagged rows
are silently dropped).

---

## 2. Computation

### 2a. Raw values vs. residuals

Correlations are computed on **raw per-game stat values**, not residuals. There is no
per-player mean subtraction, no opponent-adjustment, no pace-adjustment, no
home/away normalization. The cell that ends up in the team matrix is the literal
`game_df[stat]` value:

```python
266              homeStats.update(
267                  game_df.loc[
268                      (game_df[log_str["team"]] == home_team)
269                      & game_df[log_str["position"]].str.contains(position),
270                      stats[position],
271                  ].to_dict("index")
272              )
```

The only transformation before correlation is `fillna(0)` (line 299) and the
column-level zero-density filter (line 302).

### 2b. Time-decay weighting

None. Each retained game contributes equally to the team-level pairwise correlation:

```python
304          c_spearman = team_matrix.corr(
305              method="spearman", min_periods=int(len(team_matrix) * 0.75)
306          ).unstack()
```

`pandas.DataFrame.corr` does not accept weights, and no manual exponential / linear
decay is applied.

### 2c. Correlation method and post-processing

Spearman, then Fisher-style sin remap, then a flat magnitude floor:

```python
304          c_spearman = team_matrix.corr(
305              method="spearman", min_periods=int(len(team_matrix) * 0.75)
306          ).unstack()
307          c = 2 * np.sin(np.pi / 6 * c_spearman)
308          c = c.reindex(c.abs().sort_values(ascending=False).index).dropna()
309          c = c.loc[c.abs() > 0.05]
310          big_c.update({team: c})
```

`2 * sin(pi/6 * rho)` is the standard Spearman-to-Pearson conversion for elliptically
distributed data. The remap was the substantive change in commit `85e0acf`.

---

## 3. Pairwise filtering

### 3a. Minimum-overlap threshold

Yes, an implicit one via pandas `min_periods`:

```python
305              method="spearman", min_periods=int(len(team_matrix) * 0.75)
```

A pair must have non-NaN values in ≥75% of that team's retained rows. This is a
hard threshold: pairs below the cutoff become `NaN` and are then dropped by
`.dropna()` on line 308.

Note this interacts with the upstream `fillna(0)` on line 299: because NaNs are
already replaced with zeros before the corr call, in practice the `min_periods`
threshold rarely fires on its own — every cell already has a numeric value. The
real effect happens earlier, via the column-level zero-density filter:

```python
302              team_matrix = team_matrix.loc[:, ((team_matrix == 0).mean() < 0.5)]
```

Columns whose values are zero ≥50% of the time are dropped before corr is even called.

### 3b. Shrinkage toward zero

None. Pairs that survive `min_periods` and the `|c| > 0.05` floor are written
verbatim — there is no Ledoit-Wolf, no James-Stein, no sample-size-aware shrinkage,
and no Bayesian prior. The `0.05` threshold is a hard cutoff, not a soft taper:

```python
309          c = c.loc[c.abs() > 0.05]
```

---

## 4. Stratification

### 4a. Same-team vs opposing-team

The function builds *one* matrix per team that mixes both:

```python
281          game_data.append(
282              {"TEAM": home_team}
283              | {"DATE": gameDate.date()}
284              | homeStats
285              | {"_OPP_" + k: v for k, v in awayStats.items()}
286          )
287          game_data.append(
288              {"TEAM": away_team}
289              | {"DATE": gameDate.date()}
290              | awayStats
291              | {"_OPP_" + k: v for k, v in homeStats.items()}
292          )
```

Each game produces two rows — one keyed on the home team (with its own players as
direct columns and the away team's players as `_OPP_*` columns) and the symmetric
row for the away team. The per-team correlation matrix on line 304 then computes
correlations across **all** columns simultaneously, so a single output entry can be:

- Same-team: e.g. `RB1 rushing yards` × `WR1 receiving yards`.
- Opposing-team: e.g. `QB passing yards` × `_OPP_CB targets allowed`.
- "Self-opposing": e.g. `_OPP_RB1 rushing yards` × `_OPP_WR1 receiving yards` —
  these are correlations among the *opponent's* players, conditioned on which team
  they faced this team.

The output naming convention (`_OPP_` prefix) lets the consumer distinguish them by
string-matching the column name, but the matrix is *not* split into separate
same-team / opposing-team / cross-game files. There is also no notion of cross-game
correlation at all — every observation is a single game.

### 4b. Per-team granularity

The per-team `groupby` is the only stratification axis:

```python
300      for team in matrix.TEAM.unique():
301          team_matrix = matrix.loc[team == matrix.TEAM].drop(columns="TEAM")
```

This means the output is indexed by `(team, market_a, market_b)` — there is no
league-pooled estimate, no position-pair pooled estimate, and no aggregation across
teams. Sample size per team is bounded by ~one season of games for that team
(within the 300-day window).

### 4c. Position bucketing

Positions are folded into the column names *before* correlation, by appending a
within-team rank:

```python
253              game_df = game_df.fillna(0).infer_objects(copy=False)
254              ranks = (
255                  game_df.sort_values(f"{log_str.get('usage_sec')} short", ascending=False)
256                  .groupby([log_str["team"], log_str["position"]])
257                  .rank(ascending=False, method="first")[f"{log_str.get('usage')} short"]
258                  .astype(int)
259              )
260              game_df[log_str["position"]] = game_df[log_str["position"]] + ranks.astype(str)
261              game_df.index = game_df[log_str["position"]]
```

So an NFL row's `WR1 receiving yards` is the team's WR1-by-snap-share's receiving
yards on that game. MLB uses batting order rather than usage rank:

```python
235          if league == "MLB":
236              bat_df = game_df.loc[game_df["starting batter"]]
237              bat_df.position = "B" + bat_df.battingOrder.astype(str)
238              bat_df.index = bat_df.position
239              pitch_df = game_df.loc[game_df["starting pitcher"]]
240              pitch_df.position = "P"
241              pitch_df.index = pitch_df.position
242              game_df = pd.concat([bat_df, pitch_df])
```

This implicitly assumes the WR1 / B3 / etc. slot is a stable identity across games,
which is approximately true for full seasons but breaks down with mid-season role
changes (the "WR1" column is whichever player happened to lead snap share that
specific game).

---

## 5. Output

### 5a. Two CSVs

Intermediate per-game record (used as the warm-start cache):

```python
294      matrix = pd.concat([matrix, pd.json_normalize(game_data)], ignore_index=True)
295      matrix.to_csv(filepath)   # data/training_data/{LEAGUE}_corr.csv
```

Final correlation table:

```python
312      pd.concat(big_c).to_csv(pkg_resources.files(data) / f"{league}_corr.csv")
```

### 5b. Schema

**Intermediate** (`training_data/{LEAGUE}_corr.csv`): one row per `(team, game_date)`
pair with columns `TEAM`, `DATE`, plus `<position><rank>.<stat>` and
`_OPP_<position><rank>.<stat>` produced by `pd.json_normalize` flattening the
nested `homeStats` / `awayStats` dicts. No sample-size column, no league column
(implied by filename).

**Final** (`{LEAGUE}_corr.csv`): `pd.concat(big_c)` of per-team Series stacked into a
3-level MultiIndex `(team, market_a, market_b)` with one value column. The consumer
relabels these axes:

```python
# prediction/correlation.py:120
120          c = pd.read_csv(pkg_resources.files(data) / (f"{league}_corr.csv"), index_col=[0, 1, 2])
121          c.rename_axis(["team", "market", "correlation"], inplace=True)
122          c.columns = ["R"]
```

So the on-disk schema is effectively `team, market, correlation, R` with **no**
metadata: no date range, no sample size per pair, no `n` for the underlying team
matrix, no league tag, no version stamp, no creation timestamp inside the CSV
itself.

### 5c. What is *not* persisted

- The number of games each team's matrix was built from.
- Which 300-day window was active at write time.
- Which Spearman value preceded the sin remap (only the post-remap value is kept).
- Which pairs were dropped by the `min_periods` / 0.05 / 50%-zero filters.

---

## 6. Consumers

`grep`-confirmed callers:

```
src/sportstradamus/training/__init__.py:13:from sportstradamus.training.correlate import correlate
src/sportstradamus/training/__init__.py:24:    "correlate",
src/sportstradamus/training/cli.py:15:from sportstradamus.training.correlate import correlate
src/sportstradamus/training/cli.py:135:        correlate(lg, stat_data, force)
```

The single in-process caller is `meditate`'s per-league loop:

```python
# training/cli.py
134          stat_data.update_player_comps()
135          correlate(lg, stat_data, force)
136          league_start_date = stat_data.trim_gamelog()
```

The `__init__.py` re-export means it can be imported as `from sportstradamus.training
import correlate` and called standalone, given a loaded `Stats` instance. There is
no standalone CLI wrapper.

Downstream, the final `{LEAGUE}_corr.csv` is read by the prediction pipeline:

```python
# prediction/correlation.py:120
120          c = pd.read_csv(pkg_resources.files(data) / (f"{league}_corr.csv"), index_col=[0, 1, 2])
```

The intermediate `training_data/{LEAGUE}_corr.csv` is read **only** by `correlate`
itself on the next run as a warm-start cache (no other grep hits).

---

## 7. Recency / commit history

`git log --follow --oneline src/sportstradamus/training/correlate.py`:

```
c305a14 Phase 4c: split train.py (2,766L) into training/ package
18253f8 Phase 2: archive orphaned modules to src/deprecated/
871657e Phase 1: mechanical ruff format + safe autofixes
85e0acf Refactor correlation as beam search, use spearman correlation coefficients
7e22242 add mixture2G model
4ba3f01 filter out nba g league stats
```

Walking the file's lineage:

- **`4ba3f01` / `7e22242`** — content predates the current methodology; touched the
  tracked-stats dict.
- **`85e0acf`** (Mar 17 2026) — the last meaningful methodology change. Switched the
  correlation method to Spearman with the `2*sin(pi/6*rho)` remap and reframed the
  prediction-side consumer as a beam search. Diff stat:
  `correlation.py | 7 +-` and `sportstradamus.py | 259 +-`. The correlate-side
  change was small (the corr-method swap); most of the churn was in the consumer.
- **`871657e`** — formatter pass only (`ruff format` + safe autofixes). No logic.
- **`18253f8`** — archived an orphaned copy to `src/deprecated/`. No live-file
  changes.
- **`c305a14`** (Apr 25 2026) — the package split. The diff header
  `similarity index 58% / copy from src/deprecated/correlation.py / copy to
  src/sportstradamus/training/correlate.py` shows this was a file move +
  housekeeping (removal of an archive header and top-of-file `Stats*().load();
  .update()` side-effects, expansion of NHL `L`/`R` positions to a single `W`
  bucket, formatter compaction of NBA/WNBA stat lists). No change to the
  correlation math itself.

So the methodology has been stable since March 2026. There is no in-tree commit
that flags a known issue with the current implementation, but the deprecation
header that `c305a14` removed is itself a flag worth noting:

> ARCHIVED 2026-04-21 from src/sportstradamus/correlation.py
> Reason: Generates {LEAGUE}_corr.csv; the CSVs in repo are stale and no script invokes this.
> Last live SHA: 871657e

That is, between `18253f8` (archive) and `c305a14` (resurrect into the package),
the function had been deemed orphaned with stale outputs. It was reinstated as part
of the `meditate` rewrite; there is no commit message confirming the staleness was
resolved or that the methodology was re-validated against fresh data.

No `{LEAGUE}_corr.csv` files are checked into `src/sportstradamus/data/` at the
audit point (`find src/sportstradamus/data -name "*corr*"` returns only
`corr_modifiers.json`), so the prediction-side consumer relies entirely on a fresh
`meditate` run having produced the artefacts.
