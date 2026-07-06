"""Unit tests for the Operation Ship 75 strategy sweep (``training.model_strategy_sweep``).

Two layers are covered. The per-corner *primitive* ``_score_corner`` is the honest scorer: it loads
one trained deterministic dump and runs the production :func:`scorecard.gate_row` on its served
(validation-fit-calibrated) predictive — no test re-fit; the heavy dump load + scorecard gate are
monkeypatched so the test pins only the plumbing. The *orchestration* (family-grid enumeration,
board assembly, verdict formatting) is exercised with the heavy per-corner ``meditate`` train+score
monkeypatched out, so no model trains: the Optuna GridSampler study visits every corner of the
cell's family grid once, scores each by the honest gate, and ranks the board by ship slack.
"""

import subprocess

import pandas as pd

from sportstradamus.training import model_strategy_sweep as sweep


def _canned_row(*, ship, g4_pit_ks):
    """A scorecard ship-row where Gate 4 binds the slack (the other gates pass with headroom)."""
    return {
        "ship": ship,
        "g1_brier_diff_ci_hi": -0.01,
        "g1_brier_skill_score": 0.04,
        "g1_pass": True,
        "g2_star_z": 0.1,
        "g2_pass": True,
        "g3_bench_z": 0.2,
        "g3_pass": True,
        "g4_pit_ks": g4_pit_ks,
        "g4_pit_ks_max": 0.05,
        "g4_pass": g4_pit_ks < 0.05,
        "g5_ece_debiased": 0.03,
        "g5_pass": True,
        "g6_pass": True,
        "central50_coverage": 0.49,
        "n_rows": 1500,
    }


def _fake_row(family, corner, slack):
    """A board row for one scored corner — the shape ``_run_and_score`` returns."""
    return {
        **corner,
        "family": family,
        "slack": slack,
        "ships": slack > 0,
        "g1_pass": True,
        "g1_brier_diff_ci_hi": -0.01,
        "g1_brier_skill": 0.03,
        "g2_pass": True,
        "g2_star_z": 0.1,
        "g3_pass": True,
        "g3_bench_z": 0.2,
        "g4_pass": True,
        "g4_pit_ks": 0.04,
        "g4_pit_ks_max": 0.05,
        "g5_pass": True,
        "g5_ece_debiased": 0.03,
        "g6_pass": True,
        "central50_coverage": 0.49,
        "dispersion_cal": 1.0,
        "skew_cal": 0.0,
        "n": 2000,
    }


def _fake_run_and_score(league, market, family, corner):
    """One honest row per corner; slack favors a known best corner per family (no model trains)."""
    if family == "SkewNormal":
        slack = (
            {"ratio_meanyr": 0.0, "centered_additive_mean10": 0.20, "centered_additive_eb_meanyr_k10": -0.10}[
                corner["normalization"]
            ]
            + (0.05 if corner["dist_training_loss"] == "nll" else 0.0)
            + (0.02 if corner["blending_loss_fn"] == "crps" else 0.0)
        )
    else:  # ZINB
        slack = (
            {"joint": 0.0, "hurdle": 0.15}[corner["zinb_mode"]]
            + (0.05 if corner["count_dispersion_objective"] == "pit_ks" else 0.0)
            + (0.02 if corner["blending_loss_fn"] == "crps" else 0.0)
        )
    return [_fake_row(family, corner, slack)]


# --- family registry -------------------------------------------------------------------------


def test_family_registry_grids_and_persist_maps():
    """Two families with the agreed axes, corner counts, persist fields, and non-persistable defaults."""
    import math

    sn, zinb = sweep._FAMILIES["SkewNormal"], sweep._FAMILIES["ZINB"]
    assert math.prod(len(v) for v in sn.axes.values()) == 12
    assert math.prod(len(v) for v in zinb.axes.values()) == 8
    assert sn.persist == {"normalization": "target_normalization", "blending_loss_fn": "blending"}
    assert zinb.persist == {
        "zinb_mode": "zinb_mode",
        "count_dispersion_objective": "count_dispersion_objective",
        "blending_loss_fn": "blending",
    }
    # Only SkewNormal's dist-loss is non-persistable (the family default ships); every ZINB axis persists.
    assert sn.defaults == {"dist_training_loss": "crps"}
    assert zinb.defaults == {}


def test_dump_subdir_matches_meditate_keying():
    """The dump subdir mirrors pipeline.py: SN uses the normalization; a ZINB corner uses the
    ratio_meanyr count fallback, with a ``_hurdle`` suffix in hurdle mode.
    """
    assert sweep._dump_subdir({"normalization": "centered_additive_mean10"}) == "centered_additive_mean10"
    assert sweep._dump_subdir({"zinb_mode": "joint"}) == "ratio_meanyr"
    assert sweep._dump_subdir({"zinb_mode": "hurdle"}) == "ratio_meanyr_hurdle"


def test_decode_strategy_is_norm_for_sn_and_fallback_for_count():
    assert sweep._decode_strategy({"normalization": "ratio_meanyr"}) == "ratio_meanyr"
    assert sweep._decode_strategy({"zinb_mode": "hurdle"}) == "ratio_meanyr"


# --- honest per-corner scorer ----------------------------------------------------------------


def test_score_corner_runs_production_gate_for_sn(monkeypatch):
    captured = {}

    def fake_gate_row(df, pred_col, **kwargs):
        captured.update(kwargs)
        return _canned_row(ship=True, g4_pit_ks=0.03)

    monkeypatch.setattr(sweep, "load_test_set", lambda path, col: pd.DataFrame())
    monkeypatch.setattr(sweep.pd, "read_pickle", lambda path: {"dispersion_cal": 1.27, "skew_cal": 2.3})
    monkeypatch.setattr(sweep, "gate_row", fake_gate_row)
    monkeypatch.setattr(sweep, "apply_thresholds", lambda row: row)

    corner = {"normalization": "centered_additive_mean10", "dist_training_loss": "crps", "blending_loss_fn": "nll"}
    row = sweep._score_corner("WNBA", "AST", corner)

    # The honest gate decodes under the trial's normalization — no test re-fit of calibration.
    assert captured["decode_strategy"] == "centered_additive_mean10"
    assert row["normalization"] == "centered_additive_mean10"
    assert row["ships"] is True
    assert row["slack"] == (0.05 - 0.03) / 0.05  # Gate 4 binds
    # All six gates surfaced (g6 included).
    assert row["g1_pass"] is True and row["g6_pass"] is True
    assert row["dispersion_cal"] == 1.27 and row["skew_cal"] == 2.3
    assert row["n"] == 1500


def test_score_corner_decodes_zinb_with_ratio_meanyr_and_no_skew(monkeypatch):
    """A ZINB corner scores through the same gate with ``decode_strategy=ratio_meanyr``; a count
    dump has no ``skew_cal`` so it surfaces as 0.0.
    """
    captured = {}

    def fake_gate_row(df, pred_col, **kwargs):
        captured.update(kwargs)
        return _canned_row(ship=True, g4_pit_ks=0.02)

    monkeypatch.setattr(sweep, "load_test_set", lambda path, col: pd.DataFrame())
    monkeypatch.setattr(sweep.pd, "read_pickle", lambda path: {"dispersion_cal": 1.0})  # no skew_cal
    monkeypatch.setattr(sweep, "gate_row", fake_gate_row)
    monkeypatch.setattr(sweep, "apply_thresholds", lambda row: row)

    corner = {"zinb_mode": "hurdle", "count_dispersion_objective": "crps", "blending_loss_fn": "nll"}
    row = sweep._score_corner("MLB", "pitcher strikeouts", corner)
    assert captured["decode_strategy"] == "ratio_meanyr"
    assert row["zinb_mode"] == "hurdle"
    assert row["skew_cal"] == 0.0


def test_verdict_and_failed_gates_include_g6():
    assert sweep._verdict({"ships": True}) == "SHIP"
    killed = {"ships": False, "g1_pass": True, "g4_pass": False, "g6_pass": False}
    assert sweep._failed_gates(killed) == ["g4", "g6"]
    assert sweep._verdict(killed) == "KILL: g4 g6"


def test_run_and_score_tags_family_and_uses_honest_gate(monkeypatch):
    monkeypatch.setattr(sweep, "_run_deterministic_meditate", lambda *a, **k: None)
    captured = {}

    def fake_score(league, market, corner):
        captured["corner"] = corner
        return {**corner, "slack": 0.1, "ships": True}

    monkeypatch.setattr(sweep, "_score_corner", fake_score)
    corner = {"normalization": "ratio_meanyr", "dist_training_loss": "nll", "blending_loss_fn": "crps"}
    rows = sweep._run_and_score("WNBA", "AST", "SkewNormal", corner)
    assert captured["corner"] == corner
    assert rows[0]["family"] == "SkewNormal"
    assert rows[0]["slack"] == 0.1


# --- per-cell study over the family grid -----------------------------------------------------


def test_search_cell_enumerates_sn_grid_and_ranks(monkeypatch):
    monkeypatch.setattr(sweep, "_cell_family", lambda lg, mkt: "SkewNormal")
    monkeypatch.setattr(sweep, "_run_and_score", _fake_run_and_score)
    board = sweep.search_cell("WNBA", "AST")

    assert len(board) == 12  # 3 norms × 2 dist-loss × 2 blend
    # Best corner: centered-mean10 + nll + crps blend (0.20 + 0.05 + 0.02).
    assert board.iloc[0]["normalization"] == "centered_additive_mean10"
    assert board.iloc[0]["dist_training_loss"] == "nll"
    assert board.iloc[0]["blending_loss_fn"] == "crps"
    assert board["slack"].is_monotonic_decreasing
    assert (board["family"] == "SkewNormal").all()
    assert (board["league"] == "WNBA").all() and (board["market"] == "AST").all()
    # g6 is surfaced on the board.
    assert "g6_pass" in board.columns


def test_search_cell_enumerates_zinb_grid(monkeypatch):
    monkeypatch.setattr(sweep, "_cell_family", lambda lg, mkt: "ZINB")
    monkeypatch.setattr(sweep, "_run_and_score", _fake_run_and_score)
    board = sweep.search_cell("MLB", "pitcher strikeouts")

    assert len(board) == 8  # 2 modes × 2 disp-objective × 2 blend
    # Best ZINB corner: hurdle + pit_ks + crps (0.15 + 0.05 + 0.02).
    assert board.iloc[0]["zinb_mode"] == "hurdle"
    assert board.iloc[0]["count_dispersion_objective"] == "pit_ks"
    assert board.iloc[0]["blending_loss_fn"] == "crps"
    # SN-only axes are blank on a ZINB board (the schema is a shared superset).
    assert board["normalization"].isna().all()


def test_run_deterministic_meditate_builds_corner_flags(monkeypatch, tmp_path):
    """Each corner axis is forwarded as its flag; a ZINB corner omits --target-normalization and the
    output is captured to a per-corner log under the research log root.
    """
    monkeypatch.setattr(sweep, "_DETERMINISTIC_LOG_ROOT", tmp_path)
    calls = []
    monkeypatch.setattr(
        sweep.subprocess,
        "run",
        lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0),
    )

    sn = {"normalization": "ratio_meanyr", "dist_training_loss": "nll", "blending_loss_fn": "crps"}
    sweep._run_deterministic_meditate("WNBA", "AST", sn)
    last = calls[-1]
    assert last[last.index("--target-normalization") + 1] == "ratio_meanyr"
    assert last[last.index("--dist-training-loss") + 1] == "nll"
    assert last[last.index("--blending-loss-fn") + 1] == "crps"
    assert sweep._log_path("WNBA", "AST", sn).exists()

    zinb = {"zinb_mode": "hurdle", "count_dispersion_objective": "pit_ks", "blending_loss_fn": "nll"}
    sweep._run_deterministic_meditate("MLB", "pitcher strikeouts", zinb)
    z = calls[-1]
    assert "--target-normalization" not in z
    assert z[z.index("--zinb-mode") + 1] == "hurdle"
    assert z[z.index("--count-dispersion-objective") + 1] == "pit_ks"


# --- board derivation from stat_meta ---------------------------------------------------------


def test_select_board_cells_withheld_default_shipped_flag_and_data_filter(monkeypatch):
    """Default board = withheld × registered-family × trainable (`ALL_MARKETS`) × has-cached-data.
    `--include-shipped` adds already-shipped cells; a cell without a cached matrix is set aside as
    missing-data (warned, not swept); unswept families and non-market stems are dropped.
    """
    fake = {
        "WNBA": {
            "AST": {"dist": "SkewNormal", "shipped": "withheld"},  # default board
            "PTS": {"dist": "SkewNormal", "shipped": "devel"},  # only with --include-shipped
            "STL": {"dist": "ZINB", "shipped": "withheld"},  # withheld but no cached matrix → missing
        },
        "MLB": {
            "pitcher strikeouts": {"dist": "ZINB", "shipped": "withheld"},  # default board
            "hits allowed": {"dist": "Gamma", "shipped": "withheld"},  # excluded: unswept family
            "1st inning hits allowed": {"dist": "ZINB", "shipped": "withheld"},  # excluded: not a market
        },
    }
    fake_markets = {"WNBA": ["AST", "PTS", "STL"], "MLB": ["pitcher strikeouts", "hits allowed"]}
    has_data = {("WNBA", "AST"), ("WNBA", "PTS"), ("MLB", "pitcher strikeouts")}
    monkeypatch.setattr(sweep, "load_stat_meta", lambda path: fake)
    monkeypatch.setattr(sweep, "ALL_MARKETS", fake_markets)
    monkeypatch.setattr(sweep, "_has_training_data", lambda lg, mkt: (lg, mkt) in has_data)

    sweepable, missing = sweep._select_board_cells()
    assert set(sweepable) == {("WNBA", "AST"), ("MLB", "pitcher strikeouts")}
    assert missing == [("WNBA", "STL")]  # withheld + registry, but no cached matrix
    # --include-shipped pulls in the devel cell (it has data); STL still missing.
    incl, _ = sweep._select_board_cells(include_shipped=True)
    assert ("WNBA", "PTS") in incl
    # League filter narrows to that league's sweepable cells.
    sweepable_wnba, _ = sweep._select_board_cells("WNBA")
    assert sweepable_wnba == [("WNBA", "AST")]


def test_run_board_mode_warns_missing_data_and_passes_include_shipped(monkeypatch, capsys):
    """The board runner warns per cell skipped for a missing matrix and forwards --include-shipped."""
    captured = {}

    def fake_select(league, include_shipped):
        captured["incl"] = include_shipped
        return [("WNBA", "AST")], [("WNBA", "STL")]

    monkeypatch.setattr(sweep, "_select_board_cells", fake_select)
    monkeypatch.setattr(sweep, "_corner_count", lambda cells: 12)
    monkeypatch.setattr(
        sweep, "run_board", lambda cells, out: pd.DataFrame({"league": ["WNBA"], "market": ["AST"], "ships": [False]})
    )
    monkeypatch.setattr(sweep, "_print_board_rollup", lambda b: None)

    sweep._run_board_mode("WNBA", True, "/tmp/board.csv")
    assert captured["incl"] is True
    out = capsys.readouterr().out
    assert "skip WNBA STL: no cached training matrix" in out
    assert "1 skipped (no cached matrix)" in out


# --- family-aware actionable summary ---------------------------------------------------------


def test_stat_meta_edit_is_family_aware():
    sn = {"family": "SkewNormal", "normalization": "centered_additive_mean10", "blending_loss_fn": "crps"}
    assert sweep._stat_meta_edit(sn) == "target_normalization=centered_additive_mean10, blending=crps"
    zinb = {"family": "ZINB", "zinb_mode": "hurdle", "count_dispersion_objective": "pit_ks", "blending_loss_fn": "nll"}
    assert (
        sweep._stat_meta_edit(zinb)
        == "zinb_mode=hurdle, count_dispersion_objective=pit_ks, blending=nll"
    )


def test_repro_note_flags_nondefault_sn_dist_loss_only():
    assert "nll" in sweep._repro_note({"family": "SkewNormal", "dist_training_loss": "nll"})
    assert sweep._repro_note({"family": "SkewNormal", "dist_training_loss": "crps"}) == ""
    # A count family has no dist_training_loss key → no note (NaN is not a str).
    assert sweep._repro_note({"family": "ZINB", "dist_training_loss": float("nan")}) == ""


# --- CLI -------------------------------------------------------------------------------------


def test_cli_runs_a_single_cell(monkeypatch, tmp_path):
    from click.testing import CliRunner

    monkeypatch.setattr(sweep, "_cell_family", lambda lg, mkt: "SkewNormal")
    monkeypatch.setattr(sweep, "_run_and_score", _fake_run_and_score)
    out = str(tmp_path / "board.csv")
    result = CliRunner().invoke(sweep.main, ["--league", "WNBA", "--market", "AST", "--out", out])
    assert result.exit_code == 0, result.output
    assert "centered_additive_mean10" in result.output


def test_cli_confirm_invokes_run_confirm(monkeypatch, tmp_path):
    """``--confirm`` hands the ranked board to the confirm loop (which is mocked here)."""
    from click.testing import CliRunner

    from sportstradamus.training import model_strategy_confirm

    monkeypatch.setattr(sweep, "_cell_family", lambda lg, mkt: "SkewNormal")
    monkeypatch.setattr(sweep, "_run_and_score", _fake_run_and_score)
    seen = {}
    monkeypatch.setattr(model_strategy_confirm, "run_confirm", lambda board, *, yes: seen.update(n=len(board), yes=yes))
    out = str(tmp_path / "board.csv")
    result = CliRunner().invoke(
        sweep.main, ["--league", "WNBA", "--market", "AST", "--out", out, "--confirm", "--yes"]
    )
    assert result.exit_code == 0, result.output
    assert seen == {"n": 12, "yes": True}


def test_cli_single_cell_upserts_into_existing_board(monkeypatch, tmp_path):
    """A single-cell run merges into the board file — replacing that cell's prior rows, keeping the
    others — so it refreshes the living board instead of clobbering it.
    """
    from click.testing import CliRunner

    monkeypatch.setattr(sweep, "_cell_family", lambda lg, mkt: "SkewNormal")
    monkeypatch.setattr(sweep, "_run_and_score", _fake_run_and_score)
    out = str(tmp_path / "board.csv")
    runner = CliRunner()

    def run(league, market):
        return runner.invoke(sweep.main, ["--league", league, "--market", market, "--out", out])

    assert run("WNBA", "AST").exit_code == 0
    assert run("NBA", "FGA").exit_code == 0
    board = pd.read_csv(out)
    assert set(zip(board["league"], board["market"], strict=True)) == {("WNBA", "AST"), ("NBA", "FGA")}
    n_two_cells = len(board)
    assert run("WNBA", "AST").exit_code == 0  # re-run replaces the cell's rows, not appends
    assert len(pd.read_csv(out)) == n_two_cells
