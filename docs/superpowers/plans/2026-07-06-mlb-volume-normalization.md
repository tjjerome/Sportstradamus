# MLB Volume Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace MLB's trained `plateAppearances` volume model with a structural batting-order plate-appearance projector plus a bounded team offense adjustment, so hitter markets get sane, missing-player-robust volume features with no training.

**Architecture:** `StatsMLB.get_volume_stats` splits into two tracks on the existing `pitcher` flag. Pitchers keep the model path unchanged (`load_volume_model_params("pitches thrown")`, no normalization). Hitters route to a new structural projector: `get_depth` already resolves each hitter's batting slot into `playerProfile["depth"]` (actual slot for settled games, posted-lineup or modal slot for upcoming), so the projector maps that slot through a measured home/away PA curve, scales it by a bounded per-team offense multiplier, and writes `proj plateAppearances mean/std` into `playerProfile` — the same output contract the pitcher track produces. Because the projection is structural (a lookup times a scalar, no model fit), it needs no training and populates matrices immediately. `plateAppearances` is retired as a trained cell but stays a gamelog column.

**Tech Stack:** Python 3.11, pandas/numpy, pytest (golden suite via pytest-xdist), the project's `Stats` class hierarchy and DuckDB `Archive`.

**Key design facts (verified against current code):**
- `get_depth` (`stats/mlb.py:1065-1112`) sets `playerProfile["depth"]` = 1-based batting slot (0/NaN when unresolved) and calls `base_profile(date)`, which builds `teamProfile` with last-10 park-neutral team OBP.
- The matrix loop calls `_dispatch_volume_stats` (`stats/base.py:2073`) **before** `get_stats` (`:2075`); proj columns left in `playerProfile` survive into features (proven by NBA_PTS carrying `Player proj MIN *`).
- `archive.get_total(league, date, team)` returns **per-team** implied runs (base.py:1393-1394 "Total (own team)"), default `4.671` when unquoted.
- Gamelog carries `battingOrder`, `home` (bool), `starting batter`, `starting pitcher`, `opponent pitcher` (opposing starter name), `hits allowed`, `walks allowed`, `batters faced`, `playerName`, `gameDate`, `team`. The per-batter `OBP` column is `(H+BB)/atBats` (noisy, up to 5.5) and is **~0 on pitcher rows** (it holds the pitcher's own batting), so opposing-starter OBP-allowed is computed as `(hits allowed + walks allowed) / batters faced` ≈ 0.31 (all three columns fully populated). The league OBP baseline comes from the **teamlog** `OBP` column (statsapi true OBP, mean 0.315), matching `teamProfile["OBP"]` that the adjustment reads.
- `plateAppearances` stays a gamelog column (`stats/mlb.py:389`) and `log_strings["usage"]` (`:148`); only its **trained-market** registration is removed.

**Staging note:** Tasks 1-2 deliver the core structural projector (slot curve + home/away + fallback, offense adjustment stubbed at 1.0). That alone unblocks matrix assembly and satisfies the primary missing-player-robustness intent. Tasks 3-4 layer on the bounded offense adjustment (the spec's OBP+market blend). Tasks 5-6 retire the cell and verify. The offense-adjustment layer (Task 3) is the separable, optional-to-defer piece.

---

## Files

- **Modify** `src/sportstradamus/stats/mlb.py`
  - Add module-level constants (slot curves, std, league baselines, blend weights, clip band).
  - Replace `get_volume_stats` (`:938-949`) with the two-track version.
  - Add `_project_plate_appearances`, `_mlb_offense_adjustment`, `_obp_factor`, `_market_factor`.
  - Change `self.volume_stats` (`:142`) to `["pitches thrown"]`.
- **Modify** `src/sportstradamus/training/markets.py` — remove `"plateAppearances"` from `ALL_MARKETS["MLB"]` (`:72`).
- **Modify** `src/sportstradamus/data/config/stat_meta.json` — delete the MLB `"plateAppearances"` cell (`:93-98`).
- **Modify** `src/sportstradamus/training/baselines.py:77` — update the stale volume-stats comment.
- **Create** `scripts/measure_mlb_volume_constants.py` — committed, auditable measurement of the constants.
- **Create** `tests/golden/test_mlb_volume_normalization.py` — unit tests (no training).

**Do NOT touch** (PA remains a gamelog column): `stats/mlb.py:148,389,527,577,586`, `prediction/cli.py:86` (`_VOLUME_STAT["MLB"]` is a gamelog-column reference for the volume-trend chart).

---

### Task 1: Measurement script + module constants

**Files:**
- Create: `scripts/measure_mlb_volume_constants.py`
- Modify: `src/sportstradamus/stats/mlb.py` (module top, after existing imports/constants)
- Test: `tests/golden/test_mlb_volume_normalization.py`

- [ ] **Step 1: Write the committed measurement script**

Create `scripts/measure_mlb_volume_constants.py`:

```python
"""Measure the MLB volume-normalization constants from the backfilled gamelog:
the home/away batting-order plate-appearance curves, the within-slot PA spread,
and the park-neutral league OBP baseline. Prints values to paste into
``stats/mlb.py``. Re-run after a gamelog backfill to refresh them.
"""

from importlib import resources

import pandas as pd

import sportstradamus.data as data


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
```

- [ ] **Step 2: Run the measurement script to confirm the constants**

Run: `poetry run python scripts/measure_mlb_volume_constants.py`
Expected (from the ~2-season backfill; small drift acceptable):
```
SLOT_PA_HOME = (4.404, 4.291, 4.208, 4.11, 3.971, 3.831, 3.688, 3.533, 3.358)
SLOT_PA_AWAY = (4.584, 4.488, 4.377, 4.284, 4.149, 4.008, 3.862, 3.705, 3.543)
SLOT_STD     = (0.712, 0.698, 0.685, 0.662, 0.693, 0.731, 0.764, 0.784, 0.799)
LG_AVG_OBP (teamlog scale, matches teamProfile) = 0.3149
```

- [ ] **Step 3: Write the failing test for the constants**

Create `tests/golden/test_mlb_volume_normalization.py`:

```python
"""Unit tests for MLB structural plate-appearance volume normalization.

The projector is structural (no trained model), so every path is testable
without training: slot->curve mapping, home/away selection, the unresolved-slot
fallback, and the bounded team offense adjustment.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from sportstradamus.stats import mlb


def test_slot_constants_are_well_formed():
    for curve in (mlb.SLOT_PA_HOME, mlb.SLOT_PA_AWAY, mlb.SLOT_STD):
        assert len(curve) == 9
        assert all(v > 0 for v in curve)
    # Leadoff bats more than the nine-hole; away teams bat a full ninth so out-PA them.
    assert mlb.SLOT_PA_HOME[0] > mlb.SLOT_PA_HOME[-1]
    assert mlb.SLOT_PA_AWAY[0] > mlb.SLOT_PA_AWAY[-1]
    assert all(a >= h for a, h in zip(mlb.SLOT_PA_AWAY, mlb.SLOT_PA_HOME))
    assert mlb.SLOT_PA_ALL == tuple(
        (h + a) / 2 for h, a in zip(mlb.SLOT_PA_HOME, mlb.SLOT_PA_AWAY)
    )
    lo, hi = mlb.OFFENSE_ADJ_CLIP
    assert 0 < lo < 1 < hi
    assert 0 < mlb.LG_AVG_OBP < 0.5
    assert mlb.OBP_ADJ_WEIGHT + mlb.MARKET_ADJ_WEIGHT == pytest.approx(1.0)
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `poetry run pytest tests/golden/test_mlb_volume_normalization.py::test_slot_constants_are_well_formed -v`
Expected: FAIL with `AttributeError: module 'sportstradamus.stats.mlb' has no attribute 'SLOT_PA_HOME'`

- [ ] **Step 5: Add the constants to `stats/mlb.py`**

Add near the top of `src/sportstradamus/stats/mlb.py`, after the existing imports/module constants (match the file's existing constant style):

```python
# Batting-order plate-appearance structure, measured from the backfilled ~2-season
# MLB gamelog (scripts/measure_mlb_volume_constants.py). The batting slot fixes a
# hitter's PA regardless of who fills it, so a missing starter's PAs go to his
# replacement in the same slot -- no team budget redistribution is needed. Away
# teams bat a full ninth every game; home teams skip the bottom 9th when leading,
# so away slots carry ~0.18 PA more. Index 0 = leadoff (slot 1) .. index 8 = slot 9.
SLOT_PA_HOME = (4.404, 4.291, 4.208, 4.110, 3.971, 3.831, 3.688, 3.533, 3.358)
SLOT_PA_AWAY = (4.584, 4.488, 4.377, 4.284, 4.149, 4.008, 3.862, 3.705, 3.543)
SLOT_PA_ALL = tuple((h + a) / 2 for h, a in zip(SLOT_PA_HOME, SLOT_PA_AWAY))
# Within-slot game-to-game PA spread (extra innings, blowouts, early removal): the
# genuinely unpredictable part of PA, so it stays in std and is not offense-adjusted.
SLOT_STD = (0.712, 0.698, 0.685, 0.662, 0.693, 0.731, 0.764, 0.784, 0.799)
# Fallback for a priced hitter whose slot cannot be resolved (no lineup, no history):
# mean starting-batter PA across slots, with a deliberately wide spread.
SLOT_PA_LEAGUE_AVG = sum(SLOT_PA_ALL) / len(SLOT_PA_ALL)
SLOT_STD_UNKNOWN = 1.0

# Team offense adjustment: a bounded per-team PA multiplier (nominal 1.0). OBP is the
# mechanistic driver (more base-runners -> more lineup turnover -> more PA); the
# book-implied team total is a sanity anchor. Clipped to +/-8% because the predictable
# offense signal is only ~1-2 PA on a ~36 PA base -- the rest is unpredictable (-> std).
LG_AVG_OBP = 0.315  # mean team OBP (teamlog scale; matches teamProfile["OBP"] the offense adjustment reads)
LG_AVG_TEAM_TOTAL = 4.671  # mirrors archive default_totals["MLB"] so unquoted games -> neutral 1.0
OBP_ADJ_WEIGHT = 0.70
MARKET_ADJ_WEIGHT = 0.30
OFFENSE_ADJ_CLIP = (0.92, 1.08)
_OBP_POLE_GUARD = 0.5  # cap expected OBP so the 1/(1-OBP) PA law stays finite
_TEAM_OBP_WINDOW = 10  # recent starts for opposing-starter OBP-allowed (matches teamProfile last-10)
```

If Step 2 printed values that differ from these, use the printed values.

- [ ] **Step 6: Run the test to verify it passes**

Run: `poetry run pytest tests/golden/test_mlb_volume_normalization.py::test_slot_constants_are_well_formed -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add scripts/measure_mlb_volume_constants.py src/sportstradamus/stats/mlb.py tests/golden/test_mlb_volume_normalization.py
git commit -m "feat(mlb-volume): measured slot-PA constants + measurement script"
```

---

### Task 2: Core structural projector (offense adjustment stubbed at 1.0)

**Files:**
- Modify: `src/sportstradamus/stats/mlb.py` (add `_project_plate_appearances`; add a temporary `_mlb_offense_adjustment` returning 1.0)
- Test: `tests/golden/test_mlb_volume_normalization.py`

- [ ] **Step 1: Write the failing test for the projector's slot mapping**

Append to `tests/golden/test_mlb_volume_normalization.py`:

```python
def _bare_mlb():
    stats = mlb.StatsMLB.__new__(mlb.StatsMLB)
    stats.league = "MLB"
    stats.log_strings = {
        "team": "team", "opponent": "opponent", "player": "playerName",
        "date": "gameDate", "home": "home",
    }
    return stats


def test_projector_maps_slot_to_home_away_curve_and_fallback(monkeypatch):
    stats = _bare_mlb()
    game_day = date(2024, 5, 1)
    # get_depth output: A leadoff, B nine-hole, C unresolved (bench/no history).
    stats.playerProfile = pd.DataFrame(
        {"team": ["NYY", "NYY", "BOS"], "depth": [1.0, 9.0, 0.0]},
        index=["A", "B", "C"],
    )
    # gamelog supplies the home/away flag for the settled game day.
    stats.gamelog = pd.DataFrame(
        {
            "playerName": ["A", "B", "C"],
            "gameDate": ["2024-05-01"] * 3,
            "home": [True, False, True],
        }
    )
    monkeypatch.setattr(stats, "get_depth", lambda offers, d: None)
    monkeypatch.setattr(stats, "_mlb_offense_adjustment", lambda teams, offers, d, hm: {})

    offers = [
        {"Player": "A", "Team": "NYY", "Opponent": "BOS"},
        {"Player": "B", "Team": "NYY", "Opponent": "BOS"},
        {"Player": "C", "Team": "BOS", "Opponent": "NYY"},
    ]
    stats._project_plate_appearances(offers, game_day)
    pp = stats.playerProfile

    assert pp.at["A", "proj plateAppearances mean"] == pytest.approx(mlb.SLOT_PA_HOME[0])
    assert pp.at["B", "proj plateAppearances mean"] == pytest.approx(mlb.SLOT_PA_AWAY[8])
    assert pp.at["C", "proj plateAppearances mean"] == pytest.approx(mlb.SLOT_PA_LEAGUE_AVG)
    assert pp.at["A", "proj plateAppearances std"] == pytest.approx(mlb.SLOT_STD[0])
    assert pp.at["C", "proj plateAppearances std"] == pytest.approx(mlb.SLOT_STD_UNKNOWN)


def test_projector_applies_team_offense_multiplier(monkeypatch):
    stats = _bare_mlb()
    stats.playerProfile = pd.DataFrame({"team": ["NYY"], "depth": [1.0]}, index=["A"])
    stats.gamelog = pd.DataFrame(
        {"playerName": ["A"], "gameDate": ["2024-05-01"], "home": [True]}
    )
    monkeypatch.setattr(stats, "get_depth", lambda offers, d: None)
    monkeypatch.setattr(stats, "_mlb_offense_adjustment", lambda *a, **k: {"NYY": 1.05})
    stats._project_plate_appearances(
        [{"Player": "A", "Team": "NYY", "Opponent": "BOS"}], date(2024, 5, 1)
    )
    assert stats.playerProfile.at["A", "proj plateAppearances mean"] == pytest.approx(
        mlb.SLOT_PA_HOME[0] * 1.05
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `poetry run pytest tests/golden/test_mlb_volume_normalization.py -k projector -v`
Expected: FAIL with `AttributeError: 'StatsMLB' object has no attribute '_project_plate_appearances'`

- [ ] **Step 3: Implement the projector + a temporary 1.0 adjustment stub**

In `src/sportstradamus/stats/mlb.py`, add these methods to `StatsMLB` (place near `get_volume_stats`):

```python
def _project_plate_appearances(self, offers, date=datetime.today().date()):
    """Structural batting-order plate-appearance projection (no trained model).

    ``get_depth`` resolves each hitter's slot into ``playerProfile["depth"]``
    (actual for settled games, posted-lineup or modal slot upcoming). This maps
    that slot through the measured home/away PA curve, scales by a bounded
    per-team offense multiplier, and writes ``proj plateAppearances mean/std``
    into ``playerProfile`` -- the same contract the pitcher volume model produces.
    """
    self.get_depth(offers, date)
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d").date()
    profile = self.playerProfile

    if date < datetime.today().date():
        day = self.gamelog[pd.to_datetime(self.gamelog["gameDate"]).dt.date == date]
        home_map = (
            day.drop_duplicates("playerName").set_index("playerName")["home"].to_dict()
        )
    else:
        home_map = {
            p: self.upcoming_games.get(t, {}).get("Home")
            for p, t in profile["team"].items()
        }

    teams = set(profile["team"].dropna())
    adjustment = self._mlb_offense_adjustment(teams, offers, date, home_map)

    means, stds = {}, {}
    for player in profile.index:
        raw = profile.at[player, "depth"]
        slot = int(raw) if pd.notna(raw) and 1 <= raw <= 9 else 0
        if slot == 0:
            means[player] = SLOT_PA_LEAGUE_AVG
            stds[player] = SLOT_STD_UNKNOWN
            continue
        is_home = home_map.get(player)
        curve = SLOT_PA_ALL if is_home is None else SLOT_PA_HOME if is_home else SLOT_PA_AWAY
        team = profile.at[player, "team"]
        means[player] = curve[slot - 1] * adjustment.get(team, 1.0)
        stds[player] = SLOT_STD[slot - 1]

    profile["proj plateAppearances mean"] = pd.Series(means)
    profile["proj plateAppearances std"] = pd.Series(stds)
    profile.fillna(
        {
            "proj plateAppearances mean": SLOT_PA_LEAGUE_AVG,
            "proj plateAppearances std": SLOT_STD_UNKNOWN,
        },
        inplace=True,
    )

def _mlb_offense_adjustment(self, teams, offers, date, home_map):
    return {}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `poetry run pytest tests/golden/test_mlb_volume_normalization.py -k projector -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add src/sportstradamus/stats/mlb.py tests/golden/test_mlb_volume_normalization.py
git commit -m "feat(mlb-volume): structural batting-order PA projector"
```

---

### Task 3: Team offense adjustment (OBP + market blend)

**Files:**
- Modify: `src/sportstradamus/stats/mlb.py` (replace the `_mlb_offense_adjustment` stub; add `_obp_factor`, `_market_factor`)
- Test: `tests/golden/test_mlb_volume_normalization.py`

- [ ] **Step 1: Write the failing test for the adjustment**

Append to `tests/golden/test_mlb_volume_normalization.py`:

```python
class _FakeArchive:
    """Stand-in for the module-level ``archive`` singleton. The real one is a
    LazyArchive proxy (``__slots__ = ()``) that opens a DuckDB connection on any
    attribute access, so tests swap the module name rather than patch a method."""

    def __init__(self, total):
        self._total = total

    def get_total(self, league, date, team):
        return self._total


def test_offense_adjustment_blends_obp_and_market(monkeypatch):
    stats = _bare_mlb()
    stats.park_factors = {"NYY": {"OBP": 1.00}, "BOS": {"OBP": 1.00}}
    # teamProfile carries last-10 team OBP (built by base_profile downstream).
    stats.teamProfile = pd.DataFrame({"OBP": [0.320]}, index=["NYY"])
    # gamelog: opposing starter "Ace" has recent starts (OBP-allowed from the
    # counting columns), plus the batter row naming that starter for the game day.
    stats.gamelog = pd.DataFrame(
        {
            "playerName": ["Ace", "Ace", "A"],
            "team": ["BOS", "BOS", "NYY"],
            "gameDate": ["2024-04-01", "2024-04-08", "2024-05-01"],
            "starting pitcher": [True, True, False],
            "hits allowed": [5, 5, 0],
            "walks allowed": [3, 3, 0],
            "batters faced": [25, 25, 0],
            "opponent pitcher": ["", "", "Ace"],
        }
    )
    monkeypatch.setattr(mlb, "archive", _FakeArchive(4.90))

    offers = [{"Player": "A", "Team": "NYY", "Opponent": "BOS"}]
    home_map = {"A": True}
    adj = stats._mlb_offense_adjustment({"NYY"}, offers, date(2024, 5, 1), home_map)

    team_obp = 0.320
    park_obp = 1.00  # NYY home -> NYY park
    starter_obp = (5 + 3 + 5 + 3) / (25 + 25)  # pooled (H+BB)/BF over recent starts
    obp_exp = team_obp * park_obp * (starter_obp / mlb.LG_AVG_OBP)
    obp_factor = (1 - mlb.LG_AVG_OBP) / (1 - obp_exp)
    market_factor = 4.90 / mlb.LG_AVG_TEAM_TOTAL
    expected = float(
        np.clip(
            mlb.OBP_ADJ_WEIGHT * obp_factor + mlb.MARKET_ADJ_WEIGHT * market_factor,
            *mlb.OFFENSE_ADJ_CLIP,
        )
    )
    assert adj["NYY"] == pytest.approx(expected)
    # inputs chosen so the blend lands strictly inside the clip band, so this
    # verifies the blend math rather than the clip masking it.
    lo, hi = mlb.OFFENSE_ADJ_CLIP
    assert lo < adj["NYY"] < hi


def test_offense_adjustment_degrades_without_obp_history(monkeypatch):
    stats = _bare_mlb()
    stats.park_factors = {}
    stats.teamProfile = pd.DataFrame({"OBP": []}, index=pd.Index([], name="team"))
    stats.gamelog = pd.DataFrame(
        {
            "playerName": [], "team": [], "gameDate": [], "starting pitcher": [],
            "hits allowed": [], "walks allowed": [], "batters faced": [],
            "opponent pitcher": [],
        }
    )
    # Unquoted game -> get_total returns its default, so market_factor is neutral.
    monkeypatch.setattr(mlb, "archive", _FakeArchive(mlb.LG_AVG_TEAM_TOTAL))
    adj = stats._mlb_offense_adjustment(
        {"NYY"}, [{"Player": "A", "Team": "NYY", "Opponent": "BOS"}], date(2024, 5, 1), {}
    )
    assert adj["NYY"] == pytest.approx(1.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `poetry run pytest tests/golden/test_mlb_volume_normalization.py -k offense_adjustment -v`
Expected: FAIL (the stub returns `{}`, so `adj["NYY"]` raises `KeyError`)

- [ ] **Step 3: Implement the adjustment**

In `src/sportstradamus/stats/mlb.py`, replace the stub `_mlb_offense_adjustment` with:

```python
def _mlb_offense_adjustment(self, teams, offers, date, home_map):
    """Bounded per-team PA multiplier (nominal 1.0).

    Blends an OBP-driven factor (team on-base talent x opposing-starter
    OBP-allowed x park) with a market anchor (book-implied team runs), then
    clips to +/-8%. A team missing OBP history falls back to the market factor
    alone; an unquoted game yields a neutral market factor via the archive default.
    """
    records = offers if isinstance(offers, list) else list(offers.values())
    opponent_of = {r["Team"]: r["Opponent"] for r in records}
    home_by_team = {
        r["Team"]: home_map.get(r["Player"]) for r in records
    }

    if date < datetime.today().date():
        day = self.gamelog[pd.to_datetime(self.gamelog["gameDate"]).dt.date == date]
        starter_of = {}
        for team in teams:
            named = day.loc[day[self.log_strings["team"]] == team, "opponent pitcher"]
            starter_of[team] = named.mode().iloc[0] if not named.mode().empty else None
    else:
        starter_of = {
            team: self.upcoming_games.get(team, {}).get("Opponent Pitcher")
            for team in teams
        }

    adjustment = {}
    for team in teams:
        obp = self._obp_factor(
            team, opponent_of.get(team), home_by_team.get(team), starter_of.get(team), date
        )
        market = self._market_factor(team, date)
        blended = (
            OBP_ADJ_WEIGHT * obp + MARKET_ADJ_WEIGHT * market if obp is not None else market
        )
        adjustment[team] = float(np.clip(blended, *OFFENSE_ADJ_CLIP))
    return adjustment

def _obp_factor(self, team, opponent, is_home, opp_starter, date):
    """Expected-OBP PA factor: team talent x park x opposing-starter OBP-allowed.

    Returns ``None`` when the team has no recent OBP so the caller can fall back
    to the market anchor alone.
    """
    if team not in self.teamProfile.index:
        return None
    obp_exp = self.teamProfile.at[team, "OBP"]
    park_team = team if is_home else opponent
    obp_exp *= self.park_factors.get(park_team, {}).get("OBP", 1.0)
    if opp_starter:
        # The gamelog OBP column is ~0 on pitcher rows (it holds the pitcher's own
        # batting), so compute OBP-allowed from the populated counting columns.
        starts = self.gamelog[
            (self.gamelog["playerName"] == opp_starter)
            & self.gamelog["starting pitcher"]
            & (pd.to_datetime(self.gamelog["gameDate"]).dt.date < date)
        ].tail(_TEAM_OBP_WINDOW)
        faced = pd.to_numeric(starts["batters faced"], errors="coerce").sum()
        if faced > 0:
            on_base = (
                pd.to_numeric(starts["hits allowed"], errors="coerce").sum()
                + pd.to_numeric(starts["walks allowed"], errors="coerce").sum()
            )
            obp_exp *= (on_base / faced) / LG_AVG_OBP
    obp_exp = min(obp_exp, _OBP_POLE_GUARD)
    return (1 - LG_AVG_OBP) / (1 - obp_exp)

def _market_factor(self, team, date):
    """Book-implied team runs relative to league average (neutral 1.0 when unquoted)."""
    date_str = date.strftime("%Y-%m-%d") if not isinstance(date, str) else date
    return archive.get_total(self.league, date_str, team) / LG_AVG_TEAM_TOTAL
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `poetry run pytest tests/golden/test_mlb_volume_normalization.py -k offense_adjustment -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add src/sportstradamus/stats/mlb.py tests/golden/test_mlb_volume_normalization.py
git commit -m "feat(mlb-volume): bounded team offense adjustment (OBP + market)"
```

---

### Task 4: Two-track `get_volume_stats`

**Files:**
- Modify: `src/sportstradamus/stats/mlb.py` (`get_volume_stats`, `:938-949`)
- Test: `tests/golden/test_mlb_volume_normalization.py`

- [ ] **Step 1: Write the failing test for the two-track split**

Append to `tests/golden/test_mlb_volume_normalization.py`:

```python
def test_get_volume_stats_routes_hitter_to_structural(monkeypatch):
    stats = _bare_mlb()
    calls = {}
    monkeypatch.setattr(
        stats, "_project_plate_appearances",
        lambda offers, d: calls.__setitem__("hitter", (offers, d)),
    )
    monkeypatch.setattr(
        stats, "load_volume_model_params",
        lambda *a, **k: calls.__setitem__("pitcher", (a, k)),
    )
    offers = [{"Player": "A", "Team": "NYY", "Opponent": "BOS"}]

    stats.get_volume_stats(offers, date(2024, 5, 1), pitcher=False)
    assert "hitter" in calls and "pitcher" not in calls

    calls.clear()
    stats.get_volume_stats(offers, date(2024, 5, 1), pitcher=True)
    assert "pitcher" in calls and "hitter" not in calls
    # pitcher track still loads the "pitches thrown" model
    assert calls["pitcher"][0][1] == "pitches thrown"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `poetry run pytest tests/golden/test_mlb_volume_normalization.py -k routes_hitter -v`
Expected: FAIL — the current `get_volume_stats` calls `load_volume_model_params` for the hitter case too (`market="plateAppearances"`), so `"hitter"` is never set.

- [ ] **Step 3: Rewrite `get_volume_stats`**

In `src/sportstradamus/stats/mlb.py`, replace the whole method (`:938-949`) with:

```python
def get_volume_stats(self, offers, date=datetime.today().date(), pitcher=False):
    if not pitcher:
        self._project_plate_appearances(offers, date)
        return
    market = "pitches thrown"
    self.load_volume_model_params(
        offers,
        market,
        date,
        {
            "loc": f"proj {market} mean",
            "rate": f"proj {market} mean",
            "scale": f"proj {market} std",
        },
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `poetry run pytest tests/golden/test_mlb_volume_normalization.py -k routes_hitter -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/sportstradamus/stats/mlb.py tests/golden/test_mlb_volume_normalization.py
git commit -m "feat(mlb-volume): split get_volume_stats into pitcher/hitter tracks"
```

---

### Task 5: Retire `plateAppearances` as a trained cell

**Files:**
- Modify: `src/sportstradamus/training/markets.py:72`
- Modify: `src/sportstradamus/data/config/stat_meta.json:93-98`
- Modify: `src/sportstradamus/stats/mlb.py:142`
- Modify: `src/sportstradamus/training/baselines.py:77`
- Test: `tests/golden/test_mlb_volume_normalization.py`

- [ ] **Step 1: Write the failing test for retirement + routing**

Append to `tests/golden/test_mlb_volume_normalization.py`:

```python
def test_plate_appearances_retired_but_pitches_thrown_kept():
    from sportstradamus.training.markets import ALL_MARKETS

    assert "plateAppearances" not in ALL_MARKETS["MLB"]
    assert "pitches thrown" in ALL_MARKETS["MLB"]


def test_dispatch_routes_markets_correctly(monkeypatch):
    stats = _bare_mlb()
    stats.volume_stats = ["pitches thrown"]
    seen = {}
    monkeypatch.setattr(stats, "get_depth", lambda offers, d: seen.__setitem__("depth", True))
    monkeypatch.setattr(
        stats, "get_volume_stats",
        lambda offers, d, pitcher: seen.__setitem__("volume", pitcher),
    )
    offers = [{"Player": "A", "Team": "NYY", "Opponent": "BOS"}]

    # a trained volume market -> get_depth only
    stats._dispatch_volume_stats(offers, date(2024, 5, 1), "pitches thrown")
    assert seen == {"depth": True}

    # a hitter market -> get_volume_stats(pitcher=False)
    seen.clear()
    stats._dispatch_volume_stats(offers, date(2024, 5, 1), "total bases")
    assert seen == {"volume": False}

    # a pitcher market -> get_volume_stats(pitcher=True)
    seen.clear()
    stats._dispatch_volume_stats(offers, date(2024, 5, 1), "hits allowed")
    assert seen == {"volume": True}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `poetry run pytest tests/golden/test_mlb_volume_normalization.py -k "retired or dispatch_routes" -v`
Expected: FAIL — `"plateAppearances"` is still in `ALL_MARKETS["MLB"]`.

- [ ] **Step 3: Remove `plateAppearances` from `ALL_MARKETS["MLB"]`**

In `src/sportstradamus/training/markets.py`, delete the line `    "plateAppearances",` (`:72`) — the first element of the `"MLB"` list.

- [ ] **Step 4: Delete the MLB `plateAppearances` cell in `stat_meta.json`**

In `src/sportstradamus/data/config/stat_meta.json`, remove the `"plateAppearances"` key and its object under `"MLB"` (`:93-98`):

```json
    "plateAppearances": {
        "dist": "SkewNormal",
        "shipped": "withheld",
        "target_normalization": "none",
        "posthoc": "none"
    },
```

Ensure the preceding entry's trailing comma / JSON validity is preserved (delete the block including its trailing comma; if it was the last entry, remove the comma from the new last entry instead).

- [ ] **Step 5: Update `self.volume_stats`**

In `src/sportstradamus/stats/mlb.py:142`, change:

```python
self.volume_stats = ["plateAppearances", "pitches thrown"]
```
to:
```python
self.volume_stats = ["pitches thrown"]
```

- [ ] **Step 6: Fix the stale comment in `baselines.py`**

In `src/sportstradamus/training/baselines.py:77`, update the comment so it no longer lists `plateAppearances` as a trained volume stat. Read the surrounding lines first, then edit the comment to read (adjust wording to fit the sentence):

```python
# (volume_stats: MLB pitches thrown, ... -- plateAppearances is now a structural
# projection, not a trained cell)
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `poetry run pytest tests/golden/test_mlb_volume_normalization.py -k "retired or dispatch_routes" -v`
Expected: PASS

- [ ] **Step 8: Validate the JSON parses**

Run: `poetry run python -c "import json, pathlib; json.loads(pathlib.Path('src/sportstradamus/data/config/stat_meta.json').read_text()); print('ok')"`
Expected: `ok`

- [ ] **Step 9: Commit**

```bash
git add src/sportstradamus/training/markets.py src/sportstradamus/data/config/stat_meta.json src/sportstradamus/stats/mlb.py src/sportstradamus/training/baselines.py tests/golden/test_mlb_volume_normalization.py
git commit -m "feat(mlb-volume): retire plateAppearances as a trained cell"
```

---

### Task 6: Integration — assemble a hitter matrix + full gates

**Files:**
- No source changes; verification only.

- [ ] **Step 1: Assemble one hitter matrix against an isolated archive copy**

The isolated-archive pattern avoids the DuckDB write lock (never blocks background jobs). Run:

```bash
cp archive/archive.duckdb /tmp/mlb_vol_check.duckdb
SPORTSTRADAMUS_ARCHIVE_DB=/tmp/mlb_vol_check.duckdb \
  poetry run meditate --league MLB --market "total bases" \
  --bypass-withholding --matrix-only --target-normalization auto
```
Expected: exits 0; writes `src/sportstradamus/data/training_data/MLB_total-bases.parquet`.

- [ ] **Step 2: Verify the proj columns are populated with no model pickle**

Run:

```bash
poetry run python -c "
import pandas as pd
from importlib import resources
import sportstradamus.data as data
df = pd.read_parquet(resources.files(data) / 'training_data' / 'MLB_total-bases.parquet')
cols = [c for c in df.columns if 'proj plateAppearances' in c]
print('proj cols:', cols)
sub = df[cols]
print('non-null frac:', sub.notna().mean().to_dict())
print('mean std:', sub.std().to_dict())
assert cols, 'proj plateAppearances columns missing'
assert (sub.std() > 0).all(), 'proj columns are constant -- projector not varying by slot'
print('OK')
"
```
Expected: two `Player proj plateAppearances mean/std` columns, non-null, non-constant, `OK`. No `MLB_plateAppearances.mdl` is needed.

- [ ] **Step 3: Run ruff**

Run: `poetry run ruff check src/sportstradamus/`
Expected: clean. Fix any lint in the touched files.

- [ ] **Step 4: Run the golden suite**

Run: `poetry run pytest tests/golden/`
Expected: all pass (including the new `test_mlb_volume_normalization.py`).

- [ ] **Step 5: Run the integration suite**

Run: `poetry run pytest -m integration -n0 && touch "$CLAUDE_PROJECT_DIR/.claude/.state/integration_green"`
Expected: all pass; the state file is touched.

- [ ] **Step 6: Clean up the scratch archive**

Run: `rm -f /tmp/mlb_vol_check.duckdb`

- [ ] **Step 7: Commit any lint fixes**

```bash
git add -A
git commit -m "test(mlb-volume): verify hitter matrix populates proj columns"
```

---

## Post-implementation (main session, not a plan task)

After the plan is implemented, the main session owns the wrap-up per the repo conventions:
1. Run the `refactoring-specialist` subagent on every touched `.py` (`stats/mlb.py`, `training/markets.py`, `training/baselines.py`, `scripts/measure_mlb_volume_constants.py`, `tests/golden/test_mlb_volume_normalization.py`) **before** any push/PR/review.
2. Re-run the three gates once after the specialist returns (ruff, golden, integration).
3. Append a ledger line to `docs/handoffs/mlb-nhl-activation.md` §10; resume the paused Phase 5 matrix assembly for the remaining hitter cells now that the projector populates their volume features.
4. No `shipped:` flip; SkewNormal families unchanged. **Never push `devel` directly.**

---

## Self-review

- **Spec coverage:** two tracks (Tasks 2/4) ✓; PA retired as a cell (Task 5) ✓; `proj plateAppearances mean = SLOT_PA[home|away][slot] × offense_adjustment`, `std = SLOT_STD[slot]` (Task 2) ✓; three slot paths — known/modal-unknown handled by reused `get_depth`, no-history fallback in Task 2 ✓; offense adjustment OBP+market blend, clip, degradation (Task 3) ✓; measured constants + auditable script (Task 1) ✓; unit-testable without training (all tasks) ✓; `pitches thrown` kept (Task 5) ✓.
- **Deviations from spec (flagged for the reviewer):** (a) slot resolution **reuses `get_depth`** rather than a new resolver — for unknown live orders it uses the player's *modal* slot (get_depth's behavior) instead of the spec's probability-weighted slot mix; matrix assembly (the immediate goal) uses the *actual* historical slot, so this only affects live serving and is simpler/DRY. (b) `market_factor` uses `get_total(team)` **directly** (confirmed per-team implied runs) instead of splitting a game total by moneyline — identical outcome, simpler. Both reduce code per the repo's less-code mandate.
- **Placeholder scan:** none — every step has concrete code, commands, and expected output.
- **Type consistency:** `_project_plate_appearances(offers, date)`, `_mlb_offense_adjustment(teams, offers, date, home_map)`, `_obp_factor(team, opponent, is_home, opp_starter, date)`, `_market_factor(team, date)` names/signatures are used consistently across tasks and tests; column names `proj plateAppearances mean/std`, `depth`, `OBP`, `starting pitcher`, `opponent pitcher`, `home` match the verified code facts.
