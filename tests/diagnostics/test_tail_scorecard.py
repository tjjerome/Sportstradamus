"""Diagnostics tests for the tail scorecard (R3 train/serve-skew brief, section 6).

(a) The reconstruction self-check rebuilds the persisted test-set probability from real
    rows of one cell per family branch, and trips on a perturbed knob.
(b) The live-rule replay recommends, row for row, what ``finalize_records`` keeps on the
    same slates handed over the way ``model_prob`` builds them, graded at realized's rule.
(c) The archive is opened read-only, and the rung, consensus and decision-time quote reads
    take the right rows from it.

Self-contained: a few real test-set rows per cell with the knobs copied out of its pickle
(never the booster), a synthetic slate fixture, and a scratch DuckDB file.
"""

from __future__ import annotations

import io
import os
from datetime import date

import duckdb
import numpy as np
import pandas as pd
import pytest

from sportstradamus.prediction import offer_records
from sportstradamus.prediction.offer_records import book_over_prob, finalize_records
from sportstradamus.realized import RECOMMENDED_EDGE_MIN, RECOMMENDED_PAYOUT_MAX
from sportstradamus.scripts.tail_information import information_rows
from sportstradamus.scripts.tail_pricing import (
    RECONSTRUCTION_TOL,
    ladder_rungs,
    open_archive,
    reconstruction_error,
)
from sportstradamus.scripts.tail_scorecard import replay_live_rule

pytestmark = pytest.mark.diagnostics

# Real rows dated 2026-08-30 or later (authentic and DFS-only where the cell has both) and
# the pickle's serving knobs: gated SkewNormal, SkewNormal with the whole-CDF recal, an
# ungated SkewNormal with a mean-stage corrector, ZINB and DPO with Platt, plain NegBin.
_CELLS = {
    "NFL_receptions": (
        {
            "distribution": "SkewNormal",
            "weight": 0.7974565459471956,
            "cv": 0.5698460793009616,
            "dispersion_cal": 1.7322762183842166,
            "skew_cal": 7.431492691019174,
            "hist_gate": 0.1286014950286668,
            "step": 1.0,
            "temperature": 1.380956125871687,
            "posthoc": "none",
            "posthoc_blob": None,
            "pit_recal_blob": None,
        },
        """\
Line,EV,Book_EV,P,QuoteAuthenticity,SN_Sigma_model,SN_Alpha_model,Gate
2.5,3.4832193359923278,2.773028505647773,0.51987712386015583,authentic,1.9473510682582853,-0.1559797525405883,0.10255410402920601
6.5,5.4557982577910522,6.2061788430905453,0.3454144530536265,authentic,2.454376285726374,-0.29960605502128601,0.10255410402920601
2.5,2.2792182100683847,2.3998808066807902,0.38851656388803357,authentic,1.1708894371986389,0.035180088132619802,0.10255410402920601
0.5,1.6438427834343421,0.49999999866467548,0.67075996212001321,synthetic,0.998418629169464,0.4927065372467041,0.1286014950286668
0.5,1.8907566172632515,0.60036492707890599,0.63694137007260188,pickem,1.5257035493850708,0.45040622353553772,0.1286014950286668
""",
    ),
    "NFL_rushing-yards": (
        {
            "distribution": "SkewNormal",
            "weight": 0.6015348094870825,
            "cv": 0.8879270280627408,
            "dispersion_cal": 1.0,
            "skew_cal": 0.0,
            "hist_gate": 0.11077121236515859,
            "step": 1.0,
            "temperature": 1.421989349499424,
            "posthoc": "cdf_recal_isotonic",
            "posthoc_blob": None,
            "pit_recal_blob": {
                "kind": "isotonic_pit",
                "x": [
                    0.0,
                    0.10162856926435151,
                    0.1780566412628567,
                    0.23727934765810557,
                    0.30759225427281983,
                    0.3845429663514626,
                    0.47388826875010226,
                    0.5893378676675393,
                    0.7263817150788797,
                    0.8800080177132692,
                    0.9872240956741196,
                    1.0,
                ],
                "y": [0.0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0],
                "lam": 1.0,
            },
        },
        """\
Line,EV,Book_EV,P,QuoteAuthenticity,SN_Sigma_model,SN_Alpha_model,Gate
57.5,47.897546082441899,57.331361677973753,0.42259428063704318,authentic,26.720149675011641,0.81112098693847656,0.066632740126728804
5.5,12.026485050342719,5.2824967068907656,0.51487013571957008,authentic,7.8043080866336823,-0.22230434417724609,0.066632740126728804
17.5,22.330401117719383,17.447691573790042,0.45332888240858821,authentic,17.180868126451969,-0.81728535890579224,0.066632740126728804
0.5,7.538354323126673,0.50000000000000067,0.84528935144625872,synthetic,5.4827980995178223,2.123969554901123,0.11077121236515849
6.5,25.356439747055248,6.4999999999999956,0.7216306310257633,pickem,18.620749306678771,0.91627311706542958,0.11077121236515849
""",
    ),
    "MLB_hits-allowed": (
        {
            "distribution": "SkewNormal",
            "weight": 0.7627174944742802,
            "cv": 0.4145833414887848,
            "dispersion_cal": 1.4057511698638674,
            "skew_cal": 1.4212884699460102,
            "hist_gate": 0.00517207081758504,
            "step": 1.0,
            "temperature": 1.1237600664870495,
            "posthoc": "roe_mean",
            "posthoc_blob": {"kind": "affine", "a": 1.13066488398066, "b": 0.7976059564733968},
            "pit_recal_blob": None,
        },
        """\
Line,EV,Book_EV,P,QuoteAuthenticity,SN_Sigma_model,SN_Alpha_model
3.5,4.6657713172025401,5.5634876678958296,0.72938128966142879,authentic,1.3076788038015366,0.020566735416650699
3.5,5.2445999626097946,8.4851956874933165,0.73134210612422601,authentic,1.7352867309863751,-0.0038410853594540999
2.5,4.8530082954001053,4.728609916907307,0.85316368282159871,authentic,1.9914967155456544,0.0096226669847964998
2.5,4.3919266243000292,5.3580972144210364,0.8038215795730892,authentic,2.2513269782066345,-0.0142015125602483
2.5,4.4778572230871312,8.5076574030623284,0.78350681350324214,authentic,1.5568193197250366,0.054674658924341202
""",
    ),
    "MLB_runs-allowed": (
        {
            "distribution": "ZINB",
            "weight": 0.6005871812405703,
            "cv": 0.6020892104073043,
            "dispersion_cal": 1.1546645110601315,
            "skew_cal": 0.0,
            "hist_gate": 0.14517980107115533,
            "step": 1.0,
            "temperature": 1.2708754188851972,
            "posthoc": "prob_recal_platt",
            "posthoc_blob": {"kind": "platt", "a": 0.4597929776721205, "b": 0.03640849794550808},
            "pit_recal_blob": None,
        },
        """\
Line,EV,Book_EV,P,QuoteAuthenticity,R_model,Gate_model
2.5,3.565764347338138,3.0498353126874158,0.54393749191431839,authentic,100,0
2.5,3.5608499746012257,2.7787726012189826,0.51906485234861344,authentic,7.4620208740234366,0
2.5,2.6286082037442844,2.8984701195348075,0.48794283980347719,authentic,5.9096426963806152,0
1.5,2.7274717396300368,2.5156129252191186,0.54616448092038516,authentic,5.3399248123168954,0.043815035185328297
2.5,4.1239783769736462,3.3059406796832298,0.55485977802465947,authentic,15.732693672180176,0.0084316824040827007
""",
    ),
    "MLB_total-bases": (
        {
            "distribution": "NegBin",
            "weight": 0.45258799860149684,
            "cv": 0.6976169765317217,
            "dispersion_cal": 0.7599127903740163,
            "skew_cal": 0.0,
            "hist_gate": 0.4070809680310726,
            "step": 1.0,
            "temperature": 1.0533173723394567,
            "posthoc": "none",
            "posthoc_blob": None,
            "pit_recal_blob": None,
        },
        """\
Line,EV,Book_EV,P,QuoteAuthenticity,R_model
1.5,1.1646123490867339,1.6812522959149228,0.3587072816431095,authentic,1.5757714509963989
0.5,1.3318526056075342,1.3027130066431245,0.57342778079341494,authentic,1.4128826856613159
0.5,0.74151064863161709,0.77200842499846012,0.41176369704676069,authentic,0.61871105432510376
0.5,1.4083879679123006,1.2068475548686022,0.52205255312146803,authentic,0.57433664798736572
0.5,1.4563705794061332,1.2827376340561587,0.5715442488067265,authentic,1.1710128784179688
""",
    ),
    "NFL_carries": (
        {
            "distribution": "DPO",
            "weight": 0.8249209112330032,
            "cv": 0.5800291822465613,
            "dispersion_cal": 1.0336604899653683,
            "skew_cal": 0.0,
            "hist_gate": 0.05468881060224794,
            "step": 1.0,
            "temperature": 1.507818711065186,
            "posthoc": "prob_recal_platt",
            "posthoc_blob": {"kind": "platt", "a": 0.22269894719457062, "b": -0.044335177183788745},
            "pit_recal_blob": None,
        },
        """\
Line,EV,Book_EV,P,QuoteAuthenticity,DP_PHI_model
15.5,11.442756158486926,18.386149786903708,0.45467466101762322,authentic,0.47712421417236328
3.5,3.9440877262046712,4.477377165805458,0.49478018675711077,authentic,0.7132028341293335
13.5,13.567644849850398,14.556333416729911,0.48772802828725781,authentic,0.64484673738479614
1,4.3868716844943574,1.3243233958704808,0.51236225406937086,synthetic,0.078296706080436707
2.5,2.4177285307390322,3.0388139938224872,0.47889060055618837,pickem,0.81425297260284424
""",
    ),
}


@pytest.mark.parametrize("cell", sorted(_CELLS))
def test_reconstruction_rebuilds_the_persisted_probability(cell):
    knobs, csv = _CELLS[cell]
    rows = pd.read_csv(io.StringIO(csv))
    assert reconstruction_error(rows, knobs) <= RECONSTRUCTION_TOL
    # A 10% temperature error must trip the check, or the tolerance pins nothing.
    perturbed = knobs | {"temperature": knobs["temperature"] * 1.1}
    assert reconstruction_error(rows, perturbed) > RECONSTRUCTION_TOL


_LEAGUE, _MARKET = "MLB", "total bases"
_REPLAY_KNOBS = {
    "distribution": "NegBin",
    "weight": 0.45,
    "cv": 0.7,
    "step": 1.0,
    "temperature": 1.05,
    "dispersion_cal": 0.76,
    "hist_gate": 0.41,
    "model_version": "fixture",
    "pit_recal_blob": None,
}

# Served over-price and raw boosts as model_prob hands them over (Underdog multiplier,
# Sleeper decimal; 0 = side not posted); a blank book mean is an unquoted leg. Alpha's
# four Underdog rungs on 09-10 lose the one farthest from the standard payout, and its
# 09-11 rung counts toward a separate slate; Beta 1.5 disagrees with its payout-implied
# price by more than 0.15; Beta 4.5's chosen side was never posted; Gamma 2.0 pushes;
# Delta's boost is over the cap; Epsilon pays above the recommended ceiling.
_SLATES = """\
Platform,Date,Player,Line,p_over,Boost_Over,Boost_Under,Market Projection,Actual
Underdog,2026-09-10,Alpha,1.5,0.66,1.0,1.0,2.0,3
Underdog,2026-09-10,Alpha,2.5,0.40,1.05,0.95,2.0,1
Underdog,2026-09-10,Alpha,0.5,0.80,0.9,1.1,2.0,1
Underdog,2026-09-10,Alpha,3.5,0.30,0.8,1.3,2.0,3
Underdog,2026-09-11,Alpha,1.5,0.95,1.0,1.0,2.0,0
Underdog,2026-09-10,Beta,1.5,0.86,1.0,1.0,,2
Underdog,2026-09-10,Beta,2.5,0.55,1.2,0.85,,2
Underdog,2026-09-10,Beta,4.5,0.25,2.2,0,,1
Underdog,2026-09-10,Gamma,2.0,0.70,1.0,1.0,2.0,2
Underdog,2026-09-10,Delta,5.5,0.55,2.1,0,2.0,6
Sleeper,2026-09-10,Alpha,1.5,0.66,1.82,1.82,2.0,3
Sleeper,2026-09-10,Epsilon,3.5,0.55,2.6,0,2.0,4
Sleeper,2026-09-10,Eta,1.5,0.62,1.75,1.95,,1
"""

# Every column finalize_records projects that the replay does not otherwise set.
_BOARD_PASSENGERS = dict.fromkeys(
    [
        "Team",
        "Opponent",
        "Push Prob",
        "Quote Source",
        "Quote Authenticity",
        "Quote Books",
        "Quote Line",
        "Quote Observed At",
    ]
)


class _StubArchive:
    """``finalize_records`` reads only ``default_totals`` (the O/U passenger)."""

    default_totals = {_LEAGUE: 4.671}


def _recommended_by_finalize(slates: pd.DataFrame) -> pd.DataFrame:
    """The live path: model_prob's board per slate, finalize_records, realized's rule."""
    knobs = _REPLAY_KNOBS
    picked = []
    for (platform, _), slate in slates.groupby(["Platform", "Date"]):
        board = slate.drop(columns=["p_over", "Actual"]).assign(
            League=_LEAGUE,
            Market=_MARKET,
            Projection=np.nan,
            **{"Model Weight": knobs["weight"]},
            **_BOARD_PASSENGERS,
        )
        board["Model Under"] = 1 - slate["p_over"]
        board["Model Over"] = 1 - board["Model Under"]
        board["Market EV"] = book_over_prob(
            board, knobs["distribution"], knobs["cv"], knobs["step"], None, _LEAGUE, _MARKET
        )
        records = pd.DataFrame(
            finalize_records(
                board,
                _LEAGUE,
                platform,
                knobs["distribution"],
                knobs["cv"],
                knobs["step"],
                knobs["temperature"],
                knobs["dispersion_cal"],
                knobs["model_version"],
            )
        ).merge(slate[["Player", "Line", "Actual"]], on=["Player", "Line"])
        payout = records["Boost"]
        recommended = (
            (records["Actual"] != records["Line"])
            & (records["Win Prob"] * payout - 1 >= RECOMMENDED_EDGE_MIN)
            & (payout > 1)
            & (payout <= RECOMMENDED_PAYOUT_MAX)
        )
        picked.append(records[recommended].assign(Platform=platform, Payout=payout))
    return pd.concat(picked, ignore_index=True)


def test_replay_recommends_what_finalize_records_keeps(monkeypatch):
    monkeypatch.setattr(offer_records, "archive", _StubArchive())
    slates = pd.read_csv(io.StringIO(_SLATES))
    rungs = slates.drop(columns="p_over").assign(League=_LEAGUE, Market=_MARKET, Projection=np.nan)

    replayed = replay_live_rule(rungs, slates["p_over"], _REPLAY_KNOBS)
    replayed = replayed[replayed["Recommended"]]
    expected = _recommended_by_finalize(slates)

    key = ["Platform", "Date", "Player", "Line"]
    got = replayed.sort_values(key).reset_index(drop=True)
    want = expected.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(got[[*key, "Bet"]], want[[*key, "Bet"]])
    for column in ("Win Prob", "Market Prob", "Payout"):
        np.testing.assert_allclose(got[column], want[column], rtol=1e-12)
    # The fixture exercises every branch: the per-slate trim, the phantom gate, an unposted
    # side, a push, the boost cap and the payout ceiling each remove a leg.
    assert set(got[key].itertuples(index=False, name=None)) == {
        ("Sleeper", "2026-09-10", "Alpha", 1.5),
        ("Sleeper", "2026-09-10", "Eta", 1.5),
        ("Underdog", "2026-09-10", "Alpha", 0.5),
        ("Underdog", "2026-09-10", "Alpha", 1.5),
        ("Underdog", "2026-09-10", "Beta", 2.5),
        ("Underdog", "2026-09-11", "Alpha", 1.5),
    }
    assert got.loc[got["Date"].eq("2026-09-11"), "Win Prob"].item() == pytest.approx(0.9)


# Alpha's NFL receptions rungs on 2026-09-14. Underdog 4.5 is polled twice (the later price
# wins) and last polled at 12:00; Sleeper 4.5 last at 11:00; Underdog 5.5 at 09:00. FanDuel
# quotes 4.5 at 11:30 (before Underdog's last poll, after Sleeper's); DraftKings's 13:00
# rung is too late; PrizePicks is DFS, never a sportsbook rung; Bravo and 09-21 are not
# test rows.
_LADDER_CSV = """\
game_date,entity,book,line,p_over,observed_at
2026-09-14,Alpha,Underdog,4.5,0.50,2026-09-14 10:00:00
2026-09-14,Alpha,Underdog,4.5,0.55,2026-09-14 12:00:00
2026-09-14,Alpha,Sleeper,4.5,0.48,2026-09-14 11:00:00
2026-09-14,Alpha,Underdog,5.5,0.40,2026-09-14 09:00:00
2026-09-14,Alpha,FanDuel,4.5,0.52,2026-09-14 11:30:00
2026-09-14,Alpha,DraftKings,4.5,0.50,2026-09-14 13:00:00
2026-09-14,Alpha,PrizePicks,4.5,0.50,2026-09-14 11:00:00
2026-09-14,Bravo,Underdog,3.5,0.50,2026-09-14 11:00:00
2026-09-21,Alpha,Underdog,4.5,0.50,2026-09-21 11:00:00
"""
# Decision-time consensus: by 12:00 FanDuel's last line is 5.0 and DraftKings's 4.5; by
# 11:00 they are 4.5 and 4.0; by 09:00 only FanDuel has quoted. PrizePicks and a blank
# (team-market) line never count. The decision-time quote at the test row's 4.5 by the
# 12:00 last poll is FanDuel's 08:00 (under .46) and DraftKings's 11:30 (.44): the 12:30
# quote is too late and PrizePicks's .10 is DFS.
_ODDS_CSV = """\
game_date,entity,book,ev,observed_at,under_prob,line
2026-09-14,Alpha,FanDuel,4.4,2026-09-14 08:00:00,0.46,4.5
2026-09-14,Alpha,FanDuel,4.9,2026-09-14 11:45:00,0.55,5.0
2026-09-14,Alpha,FanDuel,4.4,2026-09-14 12:30:00,0.30,4.5
2026-09-14,Alpha,DraftKings,3.9,2026-09-14 10:30:00,0.40,4.0
2026-09-14,Alpha,DraftKings,4.4,2026-09-14 11:30:00,0.44,4.5
2026-09-14,Alpha,PrizePicks,6.4,2026-09-14 10:00:00,0.5,6.5
2026-09-14,Alpha,PrizePicks,4.4,2026-09-14 10:00:00,0.10,4.5
2026-09-14,Alpha,Caesars,4.4,2026-09-14 08:00:00,0.5,
"""
_INFO_TEST_ROW = {
    "Player": ["Alpha"],
    "Date": ["2026-09-14"],
    "Result": [6.0],
    "Line": [4.5],
    "Odds": [0.40],
    "P_standalone": [0.6],
    "QuoteAuthenticity": ["authentic"],
}


def test_archive_opens_read_only(tmp_path, monkeypatch):
    for switch in ("SPORTSTRADAMUS_ARCHIVE_DB", "SPORTSTRADAMUS_ARCHIVE_READ_ONLY"):
        monkeypatch.setenv(switch, "")  # restored at teardown; open_archive pins both
    path = tmp_path / "archive.duckdb"
    with duckdb.connect(str(path)) as writer:
        for table, rows in (("ladder", _LADDER_CSV), ("odds", _ODDS_CSV)):
            (tmp_path / f"{table}.csv").write_text(rows)
            writer.execute(
                f"CREATE TABLE {table} AS SELECT 'NFL' AS league, 'receptions' AS market, * "
                f"FROM read_csv('{tmp_path / table}.csv')"
            )
    test_rows = pd.DataFrame({"rid": [0], "game_date": [date(2026, 9, 14)], "player": ["Alpha"]})

    con = open_archive(path)
    try:
        modes = con.execute("SELECT readonly FROM duckdb_databases() WHERE NOT internal").fetchall()
        assert modes == [(True,)]
        # The Archive singleton finalize_records constructs opens the same file read-only too.
        assert os.environ["SPORTSTRADAMUS_ARCHIVE_DB"] == str(path)
        assert os.environ["SPORTSTRADAMUS_ARCHIVE_READ_ONLY"] == "1"
        # DuckDB will not open a file this process holds in a second mode: no writer fits.
        with pytest.raises(duckdb.ConnectionException):
            duckdb.connect(str(path))
        rungs = ladder_rungs(con, "NFL", "receptions", test_rows)
        info = information_rows(con, "NFL", "receptions", pd.DataFrame(_INFO_TEST_ROW), rungs)
    finally:
        con.close()

    got = rungs.sort_values(["platform", "line"])
    assert got[["platform", "line"]].values.tolist() == [
        ["Sleeper", 4.5],
        ["Underdog", 4.5],
        ["Underdog", 5.5],
    ]
    assert got["p_dfs"].tolist() == [0.48, 0.55, 0.40]
    assert got["last_poll"].tolist() == [
        pd.Timestamp(2026, 9, 14, 11),
        pd.Timestamp(2026, 9, 14, 12),
        pd.Timestamp(2026, 9, 14, 9),
    ]
    assert got["n_books"].tolist() == [0, 1, 0]
    assert got["consensus"].tolist() == [4.25, 4.75, 4.5]
    # Decision time is the latest platform poll (Underdog 4.5 at 12:00), not Sleeper's.
    row = info.iloc[0]
    assert len(info) == 1
    assert row["last_poll"] == pd.Timestamp(2026, 9, 14, 12)
    assert row["consensus"] == 4.75
    assert row["p_decision"] == pytest.approx(0.55)
    assert row["p_training"] == pytest.approx(0.60)
    assert row["y"] == 1.0
