"""Unit tests for the Operation Ship 75 confirm-and-ship loop (``training.model_strategy_confirm``).

The loop persists a swept winner to stat_meta.json, retrains it at full HPO, and keeps it (devel)
or reverts. No model trains and no real stat_meta is touched: the ``meditate`` subprocess and the
stat_meta / pickle / model_stats IO are monkeypatched, so the tests pin the decision logic — which
corner is a persistable candidate, and that a failure reverts the stat_meta entry *and* prunes the
pickle (reverting stat_meta alone would leave a failed cell serving).
"""

import pandas as pd

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
