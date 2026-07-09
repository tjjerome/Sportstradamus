"""Unit tests for the Operation Ship 75 confirm-and-ship loop (``training.model_strategy_confirm``).

The loop persists a swept winner to stat_meta.json, retrains it at full HPO, and keeps it (devel)
or reverts. No model trains and no real stat_meta is touched: the ``meditate`` subprocess and the
stat_meta / pickle / model_stats IO are monkeypatched, so the tests pin the decision logic — which
corner is a persistable candidate, and that a failure reverts the stat_meta entry *and* prunes the
pickle (reverting stat_meta alone would leave a failed cell serving).
"""

import pandas as pd
import pytest

from sportstradamus.training import model_strategy_confirm as mc


def _sn_row(norm, dist_loss, blend, ships, slack):
    return {
        "league": "WNBA",
        "market": "AST",
        "family": "SkewNormal",
        "normalization": norm,
        "dist_training_loss": dist_loss,
        "blending_loss_fn": blend,
        "ships": ships,
        "slack": slack,
    }


def _sn_original():
    return {"dist": "SkewNormal", "shipped": "withheld", "target_normalization": "none", "blending": "nll"}


# --- candidate selection ---------------------------------------------------------------------


def test_candidate_picks_best_persistable_corner():
    """The best-slack corner wins under nll (non-persistable dist-loss); the candidate is the next-
    best corner that is fully persistable (crps), not the top row.
    """
    board = pd.DataFrame(
        [
            _sn_row("centered_additive_mean10", "nll", "crps", True, 0.30),  # top slack, but nll
            _sn_row("centered_additive_mean10", "crps", "crps", True, 0.25),  # persistable
            _sn_row("ratio_meanyr", "crps", "nll", False, -0.10),
        ]
    )
    cand = mc._candidate(board)
    assert cand["status"] == "candidate"
    assert cand["edits"] == {"target_normalization": "centered_additive_mean10", "blending": "crps"}
    assert cand["slack"] == 0.25


def test_candidate_ranks_only_when_only_nonpersistable_ships():
    board = pd.DataFrame(
        [
            _sn_row("centered_additive_mean10", "nll", "crps", True, 0.30),  # only ships under nll
            _sn_row("centered_additive_mean10", "crps", "crps", False, -0.05),
        ]
    )
    assert mc._candidate(board)["status"] == "ranks_only"


def test_candidate_none_when_nothing_ships():
    assert mc._candidate(pd.DataFrame([_sn_row("ratio_meanyr", "crps", "nll", False, -0.2)])) is None


def test_candidate_zinb_is_fully_persistable():
    """Every ZINB axis persists (empty defaults), so its top shipping corner is always the candidate."""
    row = {
        "league": "MLB",
        "market": "pitcher strikeouts",
        "family": "ZINB",
        "zinb_mode": "hurdle",
        "count_dispersion_objective": "pit_ks",
        "blending_loss_fn": "crps",
        "dist_training_loss": float("nan"),  # mirrors the reindexed board (SN-only column blank)
        "ships": True,
        "slack": 0.2,
    }
    cand = mc._candidate(pd.DataFrame([row]))
    assert cand["edits"] == {
        "zinb_mode": "hurdle",
        "count_dispersion_objective": "pit_ks",
        "blending": "crps",
    }


# --- meditate subprocess primitive ------------------------------------------------------------


def test_run_meditate_true_on_clean_exit_false_on_error(monkeypatch, tmp_path):
    """`_run_meditate` reports subprocess success/failure only — it does not read the ship verdict."""
    monkeypatch.setattr(mc, "_CONFIRM_LOG_ROOT", tmp_path)
    monkeypatch.setattr(mc.subprocess, "run", lambda *a, **k: None)
    assert mc._run_meditate("NBA", "PTS") is True

    def boom(*a, **k):
        raise mc.subprocess.CalledProcessError(1, "meditate")

    monkeypatch.setattr(mc.subprocess, "run", boom)
    assert mc._run_meditate("NBA", "PTS") is False


# --- persist / confirm / revert --------------------------------------------------------------


def test_confirm_one_pass_keeps_devel(monkeypatch):
    meta = {"WNBA": {"AST": _sn_original()}}
    writes = []
    monkeypatch.setattr(mc, "_atomic_write_meta", lambda m: writes.append("w"))
    monkeypatch.setattr(mc, "_confirm_meditate", lambda lg, mkt: True)
    pruned = []
    monkeypatch.setattr(mc, "prune_model_pickle", lambda lg, mkt: pruned.append((lg, mkt)))

    cand = {
        "league": "WNBA",
        "market": "AST",
        "edits": {"target_normalization": "centered_additive_mean10", "blending": "crps"},
    }
    assert mc._confirm_one(meta, cand) == ("WNBA", "AST", "SHIPPED", [])
    # Persisted: the edits + shipped=devel; no revert, no prune.
    assert meta["WNBA"]["AST"]["shipped"] == "devel"
    assert meta["WNBA"]["AST"]["target_normalization"] == "centered_additive_mean10"
    assert meta["WNBA"]["AST"]["blending"] == "crps"
    assert pruned == []
    assert len(writes) == 1  # one persist write, no revert write


def test_confirm_one_fail_reverts_stat_meta_and_prunes_pickle(monkeypatch):
    """The safety-critical path: a ship=False confirm restores the original stat_meta entry AND
    prunes the pickle — reverting stat_meta alone would leave the failed cell serving.
    """
    original = _sn_original()
    meta = {"WNBA": {"AST": dict(original)}}
    monkeypatch.setattr(mc, "_atomic_write_meta", lambda m: None)
    monkeypatch.setattr(mc, "_confirm_meditate", lambda lg, mkt: False)
    pruned = []
    monkeypatch.setattr(mc, "prune_model_pickle", lambda lg, mkt: pruned.append((lg, mkt)) or True)
    monkeypatch.setattr(mc, "_failed_gates_after", lambda lg, mkt: ["g4"])

    cand = {
        "league": "WNBA",
        "market": "AST",
        "edits": {"target_normalization": "centered_additive_mean10", "blending": "crps"},
    }
    assert mc._confirm_one(meta, cand) == ("WNBA", "AST", "REVERTED", ["g4"])
    assert meta["WNBA"]["AST"] == original  # fully reverted
    assert pruned == [("WNBA", "AST")]  # pickle pruned — the cell cannot serve


# --- end-to-end loop -------------------------------------------------------------------------


def test_run_confirm_yes_persists_and_confirms(monkeypatch, capsys):
    board = pd.DataFrame(
        [
            _sn_row("centered_additive_mean10", "crps", "crps", True, 0.25),
            _sn_row("ratio_meanyr", "crps", "nll", False, -0.1),
        ]
    )
    meta = {"WNBA": {"AST": _sn_original()}}
    monkeypatch.setattr(mc, "load_stat_meta", lambda path: meta)
    monkeypatch.setattr(mc, "_backup_stat_meta", lambda: mc.pathlib.Path("/tmp/stat_meta.bak.json"))
    monkeypatch.setattr(mc, "_atomic_write_meta", lambda m: None)
    monkeypatch.setattr(mc, "_confirm_meditate", lambda lg, mkt: True)
    monkeypatch.setattr(mc, "prune_model_pickle", lambda lg, mkt: False)

    mc.run_confirm(board, yes=True)
    assert meta["WNBA"]["AST"]["shipped"] == "devel"
    assert meta["WNBA"]["AST"]["target_normalization"] == "centered_additive_mean10"
    assert "SHIPPED" in capsys.readouterr().out


def test_run_confirm_mixed_board_routes_withheld_and_shipped(monkeypatch, capsys):
    """One --confirm run auto-ships the withheld cell via _confirm_one and supersession-tests the live
    cell via _supersede_one — a single combined report."""
    board = pd.DataFrame(
        [
            _sn_row("centered_additive_mean10", "crps", "crps", True, 0.25),  # WNBA AST (withheld)
            {
                "league": "NBA",
                "market": "PTS",
                "family": "SkewNormal",
                "normalization": "centered_additive_mean10",
                "dist_training_loss": "crps",
                "blending_loss_fn": "crps",
                "ships": True,
                "slack": 0.30,
            },
        ]
    )
    meta = {
        "WNBA": {"AST": _sn_original()},  # withheld
        "NBA": {"PTS": {"dist": "SkewNormal", "shipped": "devel",
                        "target_normalization": "ratio_meanyr", "blending": "nll"}},
    }
    monkeypatch.setattr(mc, "load_stat_meta", lambda path: meta)
    monkeypatch.setattr(mc, "_backup_stat_meta", lambda: mc.pathlib.Path("/tmp/stat_meta.bak.json"))
    calls = {"confirm": [], "supersede": []}
    monkeypatch.setattr(mc, "_confirm_one", lambda m, c: calls["confirm"].append(c["market"]) or ("WNBA", "AST", "SHIPPED", []))
    monkeypatch.setattr(mc, "_supersede_one", lambda m, c: calls["supersede"].append(c["market"]) or ("NBA", "PTS", "SUPERSEDED", []))

    mc.run_confirm(board, yes=True)
    assert calls["confirm"] == ["AST"]
    assert calls["supersede"] == ["PTS"]
    out = capsys.readouterr().out
    assert "SHIPPED" in out and "SUPERSEDED" in out


def test_activation_gate_empty_post_go():
    """MLB+NHL D1/D2 went GO 2026-07-09 — production gates no league. The guard machinery
    stays for the next onboarding; the tests below monkeypatch it to stay covered."""
    assert mc._ACTIVATION_GATED_LEAGUES == ()


def test_run_confirm_skips_activation_gated_league(monkeypatch, capsys):
    """A withheld board-passer in a gated league is announced and dropped — never persisted or
    retrained — while a covered-league candidate in the same run still confirms."""
    monkeypatch.setattr(mc, "_ACTIVATION_GATED_LEAGUES", ("MLB", "NHL"))
    mlb_row = {
        "league": "MLB",
        "market": "total bases",
        "family": "ZINB",
        "zinb_mode": "hurdle",
        "count_dispersion_objective": "pit_ks",
        "blending_loss_fn": "crps",
        "ships": True,
        "slack": 0.04,
    }
    board = pd.DataFrame([_sn_row("centered_additive_mean10", "crps", "crps", True, 0.25), mlb_row])
    meta = {
        "WNBA": {"AST": _sn_original()},
        "MLB": {"total bases": {"dist": "ZINB", "shipped": "withheld"}},
    }
    monkeypatch.setattr(mc, "load_stat_meta", lambda path: meta)
    monkeypatch.setattr(mc, "_backup_stat_meta", lambda: mc.pathlib.Path("/tmp/stat_meta.bak.json"))
    confirmed = []
    monkeypatch.setattr(
        mc, "_confirm_one", lambda m, c: confirmed.append(c["market"]) or ("WNBA", "AST", "SHIPPED", [])
    )

    mc.run_confirm(board, yes=True)
    assert confirmed == ["AST"]
    assert meta["MLB"]["total bases"]["shipped"] == "withheld"
    assert "ACTIVATION-GATED MLB total bases" in capsys.readouterr().out


def test_run_confirm_all_gated_returns_before_backup(monkeypatch, capsys):
    """When every candidate is activation-gated the loop exits before the backup/persist step."""
    monkeypatch.setattr(mc, "_ACTIVATION_GATED_LEAGUES", ("MLB", "NHL"))
    nhl_row = {
        "league": "NHL",
        "market": "saves",
        "family": "SkewNormal",
        "normalization": "centered_additive_mean10",
        "dist_training_loss": "crps",
        "blending_loss_fn": "nll",
        "ships": True,
        "slack": 0.04,
    }
    board = pd.DataFrame([nhl_row])
    meta = {"NHL": {"saves": {"dist": "SkewNormal", "shipped": "withheld"}}}
    monkeypatch.setattr(mc, "load_stat_meta", lambda path: meta)
    touched = []
    monkeypatch.setattr(mc, "_backup_stat_meta", lambda: touched.append("backup"))

    mc.run_confirm(board, yes=True)
    assert touched == []
    out = capsys.readouterr().out
    assert "ACTIVATION-GATED NHL saves" in out
    assert "no confirmable candidates" in out


def test_run_confirm_ranks_only_never_persists(monkeypatch, capsys):
    """A cell whose only shipping corner is non-persistable is reported and skipped — the loop never
    reaches the backup/persist step.
    """
    board = pd.DataFrame(
        [
            _sn_row("centered_additive_mean10", "nll", "crps", True, 0.30),
            _sn_row("centered_additive_mean10", "crps", "crps", False, -0.05),
        ]
    )
    touched = []
    monkeypatch.setattr(mc, "_backup_stat_meta", lambda: touched.append("backup"))
    monkeypatch.setattr(mc, "load_stat_meta", lambda path: touched.append("load") or {})

    mc.run_confirm(board, yes=True)
    assert touched == []  # never persisted anything
    assert "RANKS-ONLY" in capsys.readouterr().out


# --- cell-artifact snapshot / restore ----------------------------------------------------------


def test_snapshot_restore_round_trips_all_artifacts(monkeypatch, tmp_path):
    """Snapshot copies the incumbent artifacts aside; restore puts them back byte-identical and
    restores the stat_meta entry — the safety primitive the supersede path relies on.
    """
    arts = [tmp_path / name for name in ("NBA_PTS.mdl", "NBA_PTS.csv", "model_stats.parquet")]
    for a in arts:
        a.write_text("incumbent")
    monkeypatch.setattr(mc, "_cell_artifacts", lambda lg, mkt: arts)
    monkeypatch.setattr(mc, "_CONFIRM_LOG_ROOT", tmp_path / "logs")

    backup = mc._snapshot_cell("NBA", "PTS")
    assert (backup / "NBA_PTS.csv").read_text() == "incumbent"  # the S2/S3 baseline copy

    for a in arts:  # a candidate meditate overwrites every artifact in place
        a.write_text("CANDIDATE")
    meta = {"NBA": {"PTS": {"target_normalization": "candidate"}}}
    monkeypatch.setattr(mc, "_atomic_write_meta", lambda m: None)

    mc._restore_cell("NBA", "PTS", backup, meta, {"target_normalization": "ratio_meanyr"})
    assert all(a.read_text() == "incumbent" for a in arts)
    assert meta["NBA"]["PTS"] == {"target_normalization": "ratio_meanyr"}


def test_cell_artifacts_covers_serve_read_files():
    """The restore set must include every file serving reads back — the pickle (carries the
    calibrators), stat_calibration, and book_weights — else a HOLD could leave the incumbent serving
    the candidate's config. The test-set CSV (the S2/S3 baseline) is here too.
    """
    names = {p.name for p in mc._cell_artifacts("NBA", "PTS")}
    assert {"NBA_PTS.mdl", "NBA_PTS.csv", "stat_calibration.json", "book_weights.json"} <= names


# --- supersede orchestration --------------------------------------------------------------------


def _shipped_meta():
    return {
        "NBA": {
            "PTS": {
                "dist": "SkewNormal",
                "shipped": "devel",
                "target_normalization": "ratio_meanyr",
                "blending": "nll",
            }
        }
    }


def _supersede_cand():
    return {
        "league": "NBA",
        "market": "PTS",
        "family": "SkewNormal",
        "status": "candidate",
        "edits": {"target_normalization": "centered_additive_mean10", "blending": "crps"},
        "slack": 0.2,
    }


def _verdict(*, ship, s1=True, s2=True, s3=True):
    return {
        "s1_pass": s1,
        "s2_n": 120,
        "s2_mean": 0.02,
        "s2_ci_lo": 0.005 if s2 else -0.004,
        "s2_ci_hi": 0.03,
        "s2_pass": s2,
        "s3_sharpe_baseline": 1.10,
        "s3_sharpe_candidate": 1.40,
        "s3_memmel_z": 2.33 if s3 else 0.5,
        "s3_pass": s3,
        "ship": ship,
    }


def _patch_supersede_io(monkeypatch, *, verdict, meditate_ok=True):
    """Patch the heavy IO of _supersede_one; return (restored, pruned) spy lists."""
    monkeypatch.setattr(mc, "_snapshot_cell", lambda lg, mkt: mc.pathlib.Path("/tmp/bk"))
    monkeypatch.setattr(mc, "_run_meditate", lambda lg, mkt: meditate_ok)
    monkeypatch.setattr(mc, "load_test_set", lambda path, col: pd.DataFrame())
    monkeypatch.setattr(mc, "supersede_verdict", lambda *a, **k: verdict)
    monkeypatch.setattr(mc, "_atomic_write_meta", lambda m: None)
    restored, pruned = [], []
    monkeypatch.setattr(mc, "_restore_cell", lambda lg, mkt, bk, m, orig: restored.append((lg, mkt, orig)))
    monkeypatch.setattr(mc, "prune_model_pickle", lambda lg, mkt: pruned.append((lg, mkt)))
    return restored, pruned


def test_supersede_hold_restores_incumbent_and_never_prunes(monkeypatch, capsys):
    """The safety-critical path: a HOLD verdict restores the incumbent and never prunes the live pickle."""
    meta = _shipped_meta()
    restored, pruned = _patch_supersede_io(monkeypatch, verdict=_verdict(ship=False, s3=False))
    result = mc._supersede_one(meta, _supersede_cand())
    assert result[:3] == ("NBA", "PTS", "HELD")
    assert result[3] == ["S3"]  # only S3 failed
    assert restored[0][:2] == ("NBA", "PTS")
    assert pruned == []  # live cell keeps serving
    assert "S3" in capsys.readouterr().out  # the comparison was printed


def test_supersede_pass_and_yes_keeps_candidate(monkeypatch):
    meta = _shipped_meta()
    restored, pruned = _patch_supersede_io(monkeypatch, verdict=_verdict(ship=True))
    monkeypatch.setattr(mc.click, "confirm", lambda *a, **k: True)
    result = mc._supersede_one(meta, _supersede_cand())
    assert result[:3] == ("NBA", "PTS", "SUPERSEDED")
    assert restored == []  # winning candidate kept in place
    assert pruned == []


def test_supersede_pass_but_no_restores_incumbent(monkeypatch):
    meta = _shipped_meta()
    restored, _ = _patch_supersede_io(monkeypatch, verdict=_verdict(ship=True))
    monkeypatch.setattr(mc.click, "confirm", lambda *a, **k: False)
    result = mc._supersede_one(meta, _supersede_cand())
    assert result[:3] == ("NBA", "PTS", "HELD")
    assert result[3] == ["declined"]
    assert restored[0][:2] == ("NBA", "PTS")


def test_supersede_meditate_error_restores_incumbent(monkeypatch):
    meta = _shipped_meta()
    restored, pruned = _patch_supersede_io(monkeypatch, verdict=_verdict(ship=True), meditate_ok=False)
    result = mc._supersede_one(meta, _supersede_cand())
    assert result[:3] == ("NBA", "PTS", "HELD")
    assert result[3] == ["retrain error"]
    assert restored[0][:2] == ("NBA", "PTS")
    assert pruned == []


def test_supersede_restores_on_verdict_exception(monkeypatch):
    """A crash mid-verdict still restores the incumbent via the finally guard, then re-raises."""
    meta = _shipped_meta()
    restored, _ = _patch_supersede_io(monkeypatch, verdict=_verdict(ship=True))

    def boom(*a, **k):
        raise RuntimeError("verdict blew up")

    monkeypatch.setattr(mc, "supersede_verdict", boom)
    with pytest.raises(RuntimeError):
        mc._supersede_one(meta, _supersede_cand())
    assert restored[0][:2] == ("NBA", "PTS")
