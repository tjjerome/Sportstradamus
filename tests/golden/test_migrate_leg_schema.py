"""Golden tests for the one-shot Sheets-era leg-schema migration (P8 Task 0.8).

Synthetic ``tmp_path``-scoped fixtures only — this never touches real
``data/runtime/*.parquet``. Each frame type gets: (a) a correctness check that
the old shape converts to the new canonical shape, (b) an idempotency check
(a second run is a byte-identical no-op), (c) a ``--dry-run`` check (nothing
written). ``history.parquet``'s ``Close Market Prob > 1`` quarantine gets its
own dedicated test.
"""

from __future__ import annotations

import pandas as pd
from click.testing import CliRunner

import sportstradamus.scripts.migrate_leg_schema as migrate_mod
from sportstradamus.leg_schema import LEG_FIELDS
from sportstradamus.scripts.migrate_leg_schema import main


def _invoke(runtime_dir, *extra_args):
    runner = CliRunner()
    return runner.invoke(main, ["--runtime-dir", str(runtime_dir), *extra_args])


def _file_bytes(path):
    return path.read_bytes()


# ---------------------------------------------------------------------------
# parlay_hist.parquet: Leg 1..6 display strings -> legs structs
# ---------------------------------------------------------------------------


def _seed_parlay_hist(path):
    df = pd.DataFrame(
        [
            {
                "League": "NBA",
                "Game": "BOS/MIA",
                "Date": "2026-01-15",
                "Platform": "Underdog",
                "Legs": 2,
                "Misses": 0,
                "Leg 1": "Jayson Tatum Over 28.5 PTS - 59.0%, 1.03x",
                "Leg 2": "Jimmy Butler Under 22.5 PTS - 55.0%, 1.05x",
                "Leg 3": None,
                "Leg 4": None,
                "Leg 5": None,
                "Leg 6": None,
                "Leg Probs": [0.61, 0.56],
                "Fun": 1.2,
                "Family": "core",
            }
        ]
    )
    df.to_parquet(path, engine="pyarrow", index=False)


def test_migrate_parlay_hist_converts_legs_to_structs(tmp_path):
    path = tmp_path / "parlay_hist.parquet"
    _seed_parlay_hist(path)

    result = _invoke(tmp_path)
    assert result.exit_code == 0, result.output

    out = pd.read_parquet(path)
    assert "Leg 1" not in out.columns
    assert "Leg Probs" not in out.columns
    assert "Fun" not in out.columns
    assert "Family" not in out.columns
    assert "Legs Resolved" in out.columns
    assert out["Legs Resolved"].iloc[0] == 2  # renamed from the old leg-count "Legs"

    legs = out["legs"].iloc[0]
    assert len(legs) == 2
    assert {leg["player"] for leg in legs} == {"Jayson Tatum", "Jimmy Butler"}
    for leg in legs:
        assert set(leg.keys()) == set(LEG_FIELDS)
    tatum = next(leg for leg in legs if leg["player"] == "Jayson Tatum")
    assert tatum["bet"] == "Over"
    assert tatum["line"] == 28.5
    assert tatum["market"] == "PTS"
    assert tatum["win_prob"] == 0.61  # folded in from Leg Probs[0]


def test_migrate_parlay_hist_idempotent(tmp_path):
    path = tmp_path / "parlay_hist.parquet"
    _seed_parlay_hist(path)

    first = _invoke(tmp_path)
    assert first.exit_code == 0, first.output
    after_first = _file_bytes(path)

    second = _invoke(tmp_path)
    assert second.exit_code == 0, second.output
    after_second = _file_bytes(path)

    assert after_first == after_second
    assert "already migrated" in second.output


def test_migrate_parlay_hist_dry_run_writes_nothing(tmp_path):
    path = tmp_path / "parlay_hist.parquet"
    _seed_parlay_hist(path)
    before_mtime = path.stat().st_mtime_ns
    before_bytes = _file_bytes(path)

    result = _invoke(tmp_path, "--dry-run")
    assert result.exit_code == 0, result.output
    assert "would convert" in result.output

    assert path.stat().st_mtime_ns == before_mtime
    assert _file_bytes(path) == before_bytes
    # Old schema markers still present — nothing was migrated.
    assert "Leg 1" in pd.read_parquet(path).columns


# ---------------------------------------------------------------------------
# history.parquet: nested Offers tuples -> flat rows
# ---------------------------------------------------------------------------


# On-disk field order for the pre-Task-0.7 Offers struct (mirrors the migration
# script's own frozen ``_OFFER_STRUCT_FIELDS`` — parquet stores each offer as a
# dict, never a raw tuple, so pyarrow can write a homogeneous list<struct> column).
_OFFER_STRUCT_FIELDS = (
    "line",
    "boost",
    "platform",
    "bet",
    "model_p",
    "books_p",
    "close_books_p",
    "market_clv",
    "model_clv",
)


def _offer_struct(
    line, boost, platform, bet, model_p, books_p, close_p=None, market_clv=None, model_clv=None
):
    """One on-disk Offers-struct dict — mirrors the retired ``io._offer_tuple_to_dict``."""
    values = (line, boost, platform, bet, model_p, books_p, close_p, market_clv, model_clv)
    return dict(zip(_OFFER_STRUCT_FIELDS, values, strict=True))


def _seed_history(path, *, extra_offers=()):
    offers = [
        _offer_struct(28.5, 1.03, "Underdog", "Over", 0.59, 0.54, 0.57, 0.03, 0.05),
        # Legacy 6-field offer (pre-CLV) on the same prediction — 'nan' platform
        # string as the old writer sometimes stamped it; closing trio NaN-padded
        # the same way the retired writer padded a bare 6-tuple before dict conversion.
        _offer_struct(28.5, 1.00, "nan", "Over", 0.60, 0.55),
        *extra_offers,
    ]
    df = pd.DataFrame(
        [
            {
                "Player": "Jayson Tatum",
                "League": "NBA",
                "Team": "BOS",
                "Date": "2026-01-15",
                "Market": "PTS",
                "Projection": 27.2,
                "Market Projection": 27.0,
                "Dist": "SkewNormal",
                "CV": 0.22,
                "Model Param": 5.1,
                "Gate": None,
                "Temperature": 1.0,
                "Disp Cal": 1.0,
                "Step": 0.5,
                "Offers": offers,
                "Actual": 31.0,
            }
        ]
    )
    df.to_parquet(path, engine="pyarrow", index=False)


def test_migrate_history_explodes_offers_and_normalizes_platform(tmp_path):
    path = tmp_path / "history.parquet"
    _seed_history(path)

    result = _invoke(tmp_path)
    assert result.exit_code == 0, result.output

    out = pd.read_parquet(path)
    assert "Offers" not in out.columns
    assert len(out) == 2  # one row per offer
    assert set(out["Platform"]) == {"Underdog", None}
    assert (out["Player"] == "Jayson Tatum").all()  # prediction-level cols duplicated

    full = out.loc[out["Boost"] == 1.03].iloc[0]
    assert full["Close Market Prob"] == 0.57
    assert full["Market CLV"] == 0.03

    legacy = out.loc[out["Boost"] == 1.00].iloc[0]
    assert pd.isna(legacy["Close Market Prob"])  # legacy 6-tuple NaN-padded


def test_migrate_history_idempotent(tmp_path):
    path = tmp_path / "history.parquet"
    _seed_history(path)

    first = _invoke(tmp_path)
    assert first.exit_code == 0, first.output
    after_first = _file_bytes(path)

    second = _invoke(tmp_path)
    assert second.exit_code == 0, second.output
    after_second = _file_bytes(path)

    assert after_first == after_second
    assert "already migrated" in second.output


def test_migrate_history_dry_run_writes_nothing(tmp_path):
    path = tmp_path / "history.parquet"
    _seed_history(path)
    before_mtime = path.stat().st_mtime_ns
    before_bytes = _file_bytes(path)

    result = _invoke(tmp_path, "--dry-run")
    assert result.exit_code == 0, result.output
    assert "would explode" in result.output

    assert path.stat().st_mtime_ns == before_mtime
    assert _file_bytes(path) == before_bytes
    assert "Offers" in pd.read_parquet(path).columns


def test_migrate_history_quarantines_close_prob_over_one(tmp_path):
    path = tmp_path / "history.parquet"
    _seed_history(
        path,
        extra_offers=[
            # Corrupted archive reads: Close Market Prob > 1 is impossible for a
            # probability. Two of these; one well-formed row (from the base
            # fixture) should survive untouched.
            _offer_struct(28.5, 1.10, "Sleeper", "Under", 0.40, 0.45, 1.35, 0.10, 0.12),
            _offer_struct(28.5, 1.20, "Sleeper", "Over", 0.62, 0.58, 2.00, -0.05, 0.02),
        ],
    )

    result = _invoke(tmp_path)
    assert result.exit_code == 0, result.output
    assert "2 closing-trio quarantined" in result.output

    out = pd.read_parquet(path)
    bad_rows = out.loc[out["Boost"].isin([1.10, 1.20])]
    assert len(bad_rows) == 2
    assert bad_rows["Close Market Prob"].isna().all()
    assert bad_rows["Market CLV"].isna().all()
    assert bad_rows["Model CLV"].isna().all()

    good_row = out.loc[out["Boost"] == 1.03].iloc[0]
    assert good_row["Close Market Prob"] == 0.57  # untouched, in range


class _FakeArchive:
    """Stub ``Archive.get_line`` keyed on ``(league, market, date, player)``."""

    def __init__(self, lines: dict):
        self._lines = lines

    def get_line(self, league, market, date, player):
        return self._lines.get((league, market, date, player))


def test_backfill_alt_lines_routes_tolerance_by_distribution_family(monkeypatch):
    """Count and continuous stats use different tolerances, exactly like
    ``prediction.cli._stamp_alt_line`` — a diff that clears the (wrong) flat
    0.5 tolerance but not the real per-family one must not be flagged."""
    monkeypatch.setattr(
        migrate_mod,
        "stat_dist",
        {"NBA": {"AST": "ZINB"}, "NFL": {"Yds": "Gamma"}},
    )
    flat = pd.DataFrame(
        [
            # Count stat: diff 0.6 clears the old flat 0.5 tolerance but not
            # the real count tolerance (0.75) -> must NOT be flagged alt.
            {
                "League": "NBA",
                "Market": "AST",
                "Date": "2026-01-15",
                "Player": "Jayson Tatum",
                "Line": 10.0,
                "Alt Line": None,
            },
            # Continuous stat: diff 1.6 clears the old flat 0.5 tolerance but
            # not the real continuous tolerance (2.5) -> must NOT be flagged.
            {
                "League": "NFL",
                "Market": "Yds",
                "Date": "2026-01-15",
                "Player": "Some Rusher",
                "Line": 100.0,
                "Alt Line": None,
            },
            # Count stat, genuinely alt: diff 2.0 clears even the real 0.75
            # count tolerance.
            {
                "League": "NBA",
                "Market": "AST",
                "Date": "2026-01-15",
                "Player": "Jimmy Butler",
                "Line": 10.0,
                "Alt Line": None,
            },
        ]
    )
    archive = _FakeArchive(
        {
            ("NBA", "AST", "2026-01-15", "Jayson Tatum"): 10.6,
            ("NFL", "Yds", "2026-01-15", "Some Rusher"): 101.6,
            ("NBA", "AST", "2026-01-15", "Jimmy Butler"): 12.0,
        }
    )

    out = migrate_mod._backfill_alt_lines(flat, archive)

    by_player = out.set_index("Player")["Alt Line"]
    assert not by_player["Jayson Tatum"]
    assert not by_player["Some Rusher"]
    assert by_player["Jimmy Butler"]


# ---------------------------------------------------------------------------
# user_slips.parquet: per-leg "desc" strings -> legs structs
# ---------------------------------------------------------------------------


def _seed_user_slips(path):
    df = pd.DataFrame(
        [
            {
                "slip_id": "abc-123",
                "saved_at": "2026-01-15T18:30:00",
                "builder_type": "manual",
                "platform": "Underdog",
                "League": "NBA",
                "Game": "BOS/MIA",
                "legs": [
                    {"desc": "Jayson Tatum Over 28.5 PTS - 59.0%, 1.03x"},
                    {"desc": "Jimmy Butler Under 22.5 PTS - 55.0%, 1.05x"},
                ],
                "bet_size": 2,
                "status": "pending",
            }
        ]
    )
    df.to_parquet(path, engine="pyarrow", index=False)


def test_migrate_user_slips_converts_desc_to_structs(tmp_path):
    path = tmp_path / "user_slips.parquet"
    _seed_user_slips(path)

    result = _invoke(tmp_path)
    assert result.exit_code == 0, result.output

    out = pd.read_parquet(path)
    legs = out["legs"].iloc[0]
    assert len(legs) == 2
    for leg in legs:
        assert set(leg.keys()) == set(LEG_FIELDS)
        assert "desc" not in leg
    tatum = next(leg for leg in legs if leg["player"] == "Jayson Tatum")
    assert tatum["bet"] == "Over"
    assert tatum["line"] == 28.5
    assert tatum["platform"] == "Underdog"
    assert tatum["date"] == "2026-01-15"


def test_migrate_user_slips_idempotent(tmp_path):
    path = tmp_path / "user_slips.parquet"
    _seed_user_slips(path)

    first = _invoke(tmp_path)
    assert first.exit_code == 0, first.output
    after_first = _file_bytes(path)

    second = _invoke(tmp_path)
    assert second.exit_code == 0, second.output
    after_second = _file_bytes(path)

    assert after_first == after_second
    assert "already migrated" in second.output


def test_migrate_user_slips_dry_run_writes_nothing(tmp_path):
    path = tmp_path / "user_slips.parquet"
    _seed_user_slips(path)
    before_mtime = path.stat().st_mtime_ns
    before_bytes = _file_bytes(path)

    result = _invoke(tmp_path, "--dry-run")
    assert result.exit_code == 0, result.output
    assert "would convert desc legs" in result.output

    assert path.stat().st_mtime_ns == before_mtime
    assert _file_bytes(path) == before_bytes
    stale_legs = pd.read_parquet(path)["legs"].iloc[0]
    assert "desc" in stale_legs[0]


# ---------------------------------------------------------------------------
# current_offers.parquet: Team Correlation / Opp Correlation strings -> structs
# ---------------------------------------------------------------------------


def _seed_current_offers(path):
    df = pd.DataFrame(
        [
            {
                "Player": "Jayson Tatum",
                "Market": "PTS",
                "Team Correlation": "Jaylen Brown Over 24.5 PTS (1.15x)",
                "Opp Correlation": "Jimmy Butler Under 22.5 PTS (0.90x), Bam Adebayo Over 12.5 REB (1.05x)",
            }
        ]
    )
    df.to_parquet(path, engine="pyarrow", index=False)


def test_migrate_current_offers_converts_correlation_strings_to_structs(tmp_path):
    path = tmp_path / "current_offers.parquet"
    _seed_current_offers(path)

    result = _invoke(tmp_path)
    assert result.exit_code == 0, result.output

    out = pd.read_parquet(path)
    assert "Team Correlation" not in out.columns
    assert "Opp Correlation" not in out.columns

    same = out["Corr Same"].iloc[0]
    assert same == [
        {"player": "Jaylen Brown", "market": "PTS", "bet": "Over", "line": 24.5, "mult": 1.15}
    ]

    opp = out["Corr Opp"].iloc[0]
    assert len(opp) == 2
    assert opp[0]["player"] == "Jimmy Butler"
    assert opp[0]["mult"] == 0.90
    assert opp[1]["player"] == "Bam Adebayo"


def test_migrate_current_offers_idempotent(tmp_path):
    path = tmp_path / "current_offers.parquet"
    _seed_current_offers(path)

    first = _invoke(tmp_path)
    assert first.exit_code == 0, first.output
    after_first = _file_bytes(path)

    second = _invoke(tmp_path)
    assert second.exit_code == 0, second.output
    after_second = _file_bytes(path)

    assert after_first == after_second
    assert "already migrated" in second.output


def test_migrate_current_offers_dry_run_writes_nothing(tmp_path):
    path = tmp_path / "current_offers.parquet"
    _seed_current_offers(path)
    before_mtime = path.stat().st_mtime_ns
    before_bytes = _file_bytes(path)

    result = _invoke(tmp_path, "--dry-run")
    assert result.exit_code == 0, result.output
    assert "would convert correlation cols" in result.output

    assert path.stat().st_mtime_ns == before_mtime
    assert _file_bytes(path) == before_bytes
    assert "Team Correlation" in pd.read_parquet(path).columns


# ---------------------------------------------------------------------------
# current_game_stories.parquet: per-leg "desc" strings -> legs structs
# ---------------------------------------------------------------------------


def _seed_current_game_stories(path):
    df = pd.DataFrame(
        [
            {
                "Platform": "Underdog",
                "Game": "BOS/MIA",
                "League": "NBA",
                "Date": "2026-01-15",
                "objective": "Bankroll Builder",
                "legs": [
                    {"desc": "Jayson Tatum Over 28.5 PTS - 59.0%, 1.03x"},
                    {"desc": "Jimmy Butler Under 22.5 PTS - 55.0%, 1.05x"},
                ],
                "kelly_stake": 0.02,
            }
        ]
    )
    df.to_parquet(path, engine="pyarrow", index=False)


def test_migrate_current_game_stories_converts_desc_to_structs(tmp_path):
    path = tmp_path / "current_game_stories.parquet"
    _seed_current_game_stories(path)

    result = _invoke(tmp_path)
    assert result.exit_code == 0, result.output

    out = pd.read_parquet(path)
    legs = out["legs"].iloc[0]
    assert len(legs) == 2
    for leg in legs:
        assert set(leg.keys()) == set(LEG_FIELDS)
        assert "desc" not in leg
    tatum = next(leg for leg in legs if leg["player"] == "Jayson Tatum")
    assert tatum["bet"] == "Over"
    assert tatum["line"] == 28.5
    assert tatum["market"] == "PTS"
    assert tatum["platform"] == "Underdog"
    assert tatum["date"] == "2026-01-15"


def test_migrate_current_game_stories_idempotent(tmp_path):
    path = tmp_path / "current_game_stories.parquet"
    _seed_current_game_stories(path)

    first = _invoke(tmp_path)
    assert first.exit_code == 0, first.output
    after_first = _file_bytes(path)

    second = _invoke(tmp_path)
    assert second.exit_code == 0, second.output
    after_second = _file_bytes(path)

    assert after_first == after_second
    assert "already migrated" in second.output


def test_migrate_current_game_stories_dry_run_writes_nothing(tmp_path):
    path = tmp_path / "current_game_stories.parquet"
    _seed_current_game_stories(path)
    before_mtime = path.stat().st_mtime_ns
    before_bytes = _file_bytes(path)

    result = _invoke(tmp_path, "--dry-run")
    assert result.exit_code == 0, result.output
    assert "would convert desc legs" in result.output

    assert path.stat().st_mtime_ns == before_mtime
    assert _file_bytes(path) == before_bytes
    stale_legs = pd.read_parquet(path)["legs"].iloc[0]
    assert "desc" in stale_legs[0]


# ---------------------------------------------------------------------------
# current_parlays.parquet: Leg 1..6 display strings -> legs structs
# ---------------------------------------------------------------------------


def _seed_current_parlays(path):
    df = pd.DataFrame(
        [
            {
                "Game": "BOS/MIA",
                "Date": "2026-01-15",
                "League": "NBA",
                "Platform": "Underdog",
                "Model EV": 1.12,
                "Market EV": 1.05,
                "Boost": 1.0,
                "Rec Bet": 0.5,
                "Leg 1": "Jayson Tatum Over 28.5 PTS - 59.0%, 1.03x",
                "Leg 2": "Jimmy Butler Under 22.5 PTS - 55.0%, 1.05x",
                "Leg 3": "",
                "Leg 4": "",
                "Leg 5": "",
                "Leg 6": "",
                "Legs": (
                    "Jayson Tatum Over 28.5 PTS - 59.0%, 1.03x, "
                    "Jimmy Butler Under 22.5 PTS - 55.0%, 1.05x"
                ),
                "P": 0.61,
                "PB": 0.56,
                "Bet Size": 2,
                "Leg Probs": (0.61, 0.56),
                "Indep P": 0.34,
            }
        ]
    )
    df.to_parquet(path, engine="pyarrow", index=False)


def test_migrate_current_parlays_converts_legs_to_structs(tmp_path):
    path = tmp_path / "current_parlays.parquet"
    _seed_current_parlays(path)

    result = _invoke(tmp_path)
    assert result.exit_code == 0, result.output

    out = pd.read_parquet(path)
    assert "Leg 1" not in out.columns
    assert "Leg Probs" not in out.columns
    assert "Legs" not in out.columns

    legs = out["legs"].iloc[0]
    assert len(legs) == 2
    assert {leg["player"] for leg in legs} == {"Jayson Tatum", "Jimmy Butler"}
    for leg in legs:
        assert set(leg.keys()) == set(LEG_FIELDS)
    tatum = next(leg for leg in legs if leg["player"] == "Jayson Tatum")
    assert tatum["bet"] == "Over"
    assert tatum["line"] == 28.5
    assert tatum["market"] == "PTS"
    assert tatum["win_prob"] == 0.61  # folded in from Leg Probs[0]


def test_migrate_current_parlays_idempotent(tmp_path):
    path = tmp_path / "current_parlays.parquet"
    _seed_current_parlays(path)

    first = _invoke(tmp_path)
    assert first.exit_code == 0, first.output
    after_first = _file_bytes(path)

    second = _invoke(tmp_path)
    assert second.exit_code == 0, second.output
    after_second = _file_bytes(path)

    assert after_first == after_second
    assert "already migrated" in second.output


def test_migrate_current_parlays_dry_run_writes_nothing(tmp_path):
    path = tmp_path / "current_parlays.parquet"
    _seed_current_parlays(path)
    before_mtime = path.stat().st_mtime_ns
    before_bytes = _file_bytes(path)

    result = _invoke(tmp_path, "--dry-run")
    assert result.exit_code == 0, result.output
    assert "would convert" in result.output

    assert path.stat().st_mtime_ns == before_mtime
    assert _file_bytes(path) == before_bytes
    assert "Leg 1" in pd.read_parquet(path).columns


# ---------------------------------------------------------------------------
# Absent files: every step reports "absent" and the run still exits clean.
# ---------------------------------------------------------------------------


def test_migrate_reports_absent_files_and_exits_clean(tmp_path):
    result = _invoke(tmp_path)
    assert result.exit_code == 0, result.output
    assert result.output.count("absent") == 6
