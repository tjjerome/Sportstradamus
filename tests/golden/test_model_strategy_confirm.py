"""Unit tests for the Operation Ship 75 confirm-and-ship loop (``training.model_strategy_confirm``).

The loop persists a swept winner to stat_meta.json, retrains it at full HPO, and keeps it (devel)
or reverts. No model trains and no real stat_meta is touched: the ``meditate`` subprocess and the
stat_meta / pickle / model_stats IO are monkeypatched, so the tests pin the decision logic — which
corner is a persistable candidate, and that a failure reverts the stat_meta entry *and* prunes the
pickle (reverting stat_meta alone would leave a failed cell serving).
"""

import json

import pandas as pd
import pytest

from sportstradamus.training import model_strategy_confirm as mc
from sportstradamus.training.model_strategy_artifacts import (
    build_artifact_identity,
)
from sportstradamus.training.model_strategy_registry import (
    CellContext,
    controls_json,
    distribution_class,
    get_strategy,
    registered_strategies,
)
from sportstradamus.training.role_specs import role_spec_for
from sportstradamus.training.structural_strategies import AFFINE_STRATEGY

_MATRIX_SHA = "matrix-123"
_MATRIX_COLUMNS = frozenset(
    column
    for spec in registered_strategies()
    for column in spec.applicability.required_data_columns
)


@pytest.fixture(autouse=True)
def _fixed_cell_context(monkeypatch):
    def context(league, market):
        dist = "ZINB" if league == "MLB" else "SkewNormal"
        columns = set(_MATRIX_COLUMNS)
        spec = role_spec_for(league, market)
        if spec is not None:
            columns |= set(spec.all_columns)
        return CellContext(
            league,
            market,
            dist,
            distribution_class(dist),
            frozenset(columns),
            _MATRIX_SHA,
        )

    monkeypatch.setattr(mc, "_cell_context", context)


def _fake_meta_disk(monkeypatch, tmp_path):
    """Route ``_atomic_write_meta`` to a tmp ``_STAT_META`` file.

    The ship path re-reads the cell from disk after the meditate subprocess
    (``_sync_cell_from_disk`` — the subprocess may have pinned ``hpo_selection``),
    so a bare no-op write mock would make the sync read the real repo file and
    clobber the in-memory candidate. Returns the write-call list.
    """
    meta_file = tmp_path / "stat_meta.json"
    monkeypatch.setattr(mc, "_STAT_META", meta_file)
    writes = []
    monkeypatch.setattr(
        mc,
        "_atomic_write_meta",
        lambda m: writes.append("w") or meta_file.write_text(json.dumps(m)),
    )
    return writes


def _signed_row(league, market, strategy_slug, controls, ships, slack):
    spec = get_strategy(strategy_slug)
    legacy = (
        {"validation_audit": {"split_fingerprint_sha256": "split-123"}}
        if spec.is_structural
        else None
    )
    identity = build_artifact_identity(
        spec.slug, league, market, controls, legacy, matrix_hash=_MATRIX_SHA
    )
    return {
        "league": league,
        "market": market,
        "family": spec.family,
        "strategy_slug": identity.strategy_slug,
        "structural_strategy": identity.structural_strategy,
        "strategy_signature": identity.signature,
        "strategy_implementation_version": identity.implementation_version,
        "artifact_schema_version": identity.artifact_schema_version,
        "strategy_status": identity.status,
        "controls_json": controls_json(controls),
        "corner_fingerprint": identity.corner_fingerprint,
        "matrix_hash": _MATRIX_SHA,
        "split_fingerprint": identity.split_fingerprint,
        **controls,
        "ships": ships,
        "slack": slack,
    }


def _sn_row(norm, dist_loss, blend, ships, slack, sn_param="direct"):
    return _signed_row(
        "WNBA",
        "AST",
        "SkewNormal",
        {
            "dist": "SkewNormal",
            "normalization": norm,
            "dist_training_loss": dist_loss,
            "sn_param": sn_param,
            "blending_loss_fn": blend,
        },
        ships,
        slack,
    )


def _sn_original():
    return {
        "dist": "SkewNormal",
        "shipped": "withheld",
        "target_normalization": "none",
        "blending": "nll",
    }


def _zinb_row(mode, disp, blend, ships, slack):
    """A ZINB count corner, reindexed to the full board schema (SN-only columns blank)."""
    return {
        **_signed_row(
            "MLB",
            "pitcher strikeouts",
            "ZINB",
            {
                "dist": "ZINB",
                "zinb_mode": mode,
                "count_dispersion_objective": disp,
                "blending_loss_fn": blend,
            },
            ships,
            slack,
        ),
        "normalization": float("nan"),
        "dist_training_loss": float("nan"),
        "sn_param": float("nan"),
    }


def _negbin_row(disp, blend, ships, slack):
    """A plain-NegBin count corner: no ``zinb_mode`` (its persist map omits it), other columns blank."""
    return {
        **_signed_row(
            "MLB",
            "pitcher strikeouts",
            "NegBin",
            {
                "dist": "NegBin",
                "count_dispersion_objective": disp,
                "blending_loss_fn": blend,
            },
            ships,
            slack,
        ),
        "zinb_mode": float("nan"),
        "normalization": float("nan"),
        "dist_training_loss": float("nan"),
        "sn_param": float("nan"),
    }


def _structural_row(slug, market, ships=True, slack=0.4):
    return _signed_row("NFL", market, slug, dict(get_strategy(slug).fixed_controls), ships, slack)


# --- candidate selection ---------------------------------------------------------------------


def test_candidate_picks_top_slack_corner():
    """The top-slack shipping corner is the candidate, and every SkewNormal axis — including
    dist_training_loss (S4) — lands in the persisted edits.
    """
    board = pd.DataFrame(
        [
            _sn_row("centered_additive_mean10", "nll", "crps", True, 0.30),
            _sn_row("centered_additive_mean10", "crps", "crps", True, 0.25, sn_param="centered"),
            _sn_row("ratio_meanyr", "crps", "nll", False, -0.10),
        ]
    )
    cand = mc._candidate(board)
    assert cand["edits"] == {
        "dist": "SkewNormal",
        "target_normalization": "centered_additive_mean10",
        "dist_training_loss": "nll",
        "sn_param": "direct",
        "blending": "crps",
        "structural_strategy": "none",
    }
    assert cand["slack"] == 0.30


def test_candidate_none_when_nothing_ships():
    assert (
        mc._candidate(pd.DataFrame([_sn_row("ratio_meanyr", "crps", "nll", False, -0.2)])) is None
    )


def test_candidate_skips_mixture_until_serving_lands():
    """Serve-iff-ship: a Mixture corner may top the board but must not become the confirm
    candidate while model_prob has no Mixture branch — the next-best non-Mixture shipping
    corner wins instead, and an all-Mixture slice yields no candidate.
    """
    mix_row = _signed_row(
        "WNBA",
        "AST",
        "Mixture",
        {"dist": "Mixture", "normalization": "ratio_meanyr"},
        True,
        0.9,
    )
    sn_row = _sn_row("centered_additive_mean10", "crps", "crps", True, 0.2)
    cand = mc._candidate(pd.DataFrame([mix_row, sn_row]))
    assert cand["family"] == "SkewNormal"
    assert mc._candidate(pd.DataFrame([mix_row])) is None


def test_candidate_zinb_is_fully_persistable():
    """Every ZINB axis persists (empty defaults), so its top shipping corner is always the candidate;
    the persisted edits include the swept ``dist`` (which pins the winning count family).
    """
    row = _zinb_row("hurdle", "pit_ks", "crps", True, 0.2)
    cand = mc._candidate(pd.DataFrame([row]))
    assert cand["edits"] == {
        "dist": "ZINB",
        "zinb_mode": "hurdle",
        "count_dispersion_objective": "pit_ks",
        "blending": "crps",
        "structural_strategy": "none",
    }


def test_candidate_cross_family_negbin_wins():
    """A count cell's slice mixes ZINB and NegBin corners; the top-slack shipping-and-persistable one
    is NegBin, so the candidate flips the cell's family and its edits carry NO ``zinb_mode``.
    """
    board = pd.DataFrame(
        [
            _zinb_row("hurdle", "pit_ks", "crps", True, 0.10),
            _negbin_row("crps", "nll", True, 0.22),  # top slack, ships, NegBin
            _zinb_row("joint", "crps", "nll", False, -0.05),
        ]
    )
    cand = mc._candidate(board)
    assert cand["family"] == "NegBin"
    assert cand["edits"] == {
        "dist": "NegBin",
        "count_dispersion_objective": "crps",
        "blending": "nll",
        "structural_strategy": "none",
    }
    assert "zinb_mode" not in cand["edits"]  # the flip never reads the NaN zinb_mode column
    assert cand["slack"] == 0.22


def test_candidate_cross_family_zinb_wins():
    """Same ZINB+NegBin mix but a ZINB corner has the top slack, so the candidate stays ZINB and its
    edits include ``dist=ZINB`` and ``zinb_mode`` — NegBin's blank columns are never consulted.
    """
    board = pd.DataFrame(
        [
            _zinb_row("hurdle", "pit_ks", "crps", True, 0.30),  # top slack, ships, ZINB
            _negbin_row("crps", "nll", True, 0.18),
        ]
    )
    cand = mc._candidate(board)
    assert cand["family"] == "ZINB"
    assert cand["edits"] == {
        "dist": "ZINB",
        "zinb_mode": "hurdle",
        "count_dispersion_objective": "pit_ks",
        "blending": "crps",
        "structural_strategy": "none",
    }
    assert cand["slack"] == 0.30


def test_candidate_structural_method_persists_full_recipe_and_identity():
    slug = AFFINE_STRATEGY
    cand = mc._candidate(pd.DataFrame([_structural_row(slug, "rushing yards")]))
    assert cand["strategy_slug"] == slug
    assert cand["structural_strategy"] == slug
    assert cand["edits"] == {
        "dist": "SkewNormal",
        "target_normalization": "ratio_meanyr",
        "dist_training_loss": "crps",
        "sn_param": "direct",
        "blending": "nll",
        "hpo_selection": "loss",
        "posthoc": "none",
        "structural_strategy": slug,
    }


def test_candidate_base_winner_clears_structural_method_without_stringifying_nan():
    row = _signed_row(
        "NFL",
        "receiving yards",
        "SkewNormal",
        {
            "dist": "SkewNormal",
            "normalization": "ratio_meanyr",
            "dist_training_loss": "crps",
            "sn_param": "direct",
            "blending_loss_fn": "nll",
        },
        True,
        0.2,
    )
    cand = mc._candidate(pd.DataFrame([row]))
    assert cand["structural_strategy"] == "none"
    assert cand["edits"]["structural_strategy"] == "none"
    assert "nan" not in cand["edits"].values()


def test_candidate_rejects_missing_required_structural_axis():
    slug = AFFINE_STRATEGY
    row = _structural_row(slug, "rushing yards")
    row["hpo_selection"] = float("nan")
    with pytest.raises(ValueError, match="hpo_selection contradicts controls_json"):
        mc._candidate(pd.DataFrame([row]))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("strategy_slug", None, "missing strategy_slug"),
        ("strategy_slug", "not-registered", "unknown model strategy"),
        ("structural_strategy", None, "missing structural_strategy"),
        ("strategy_signature", "stale", "stale or mismatched strategy identity"),
        ("strategy_implementation_version", 999, "stale or mismatched strategy identity"),
        ("artifact_schema_version", 999, "stale or mismatched strategy identity"),
        ("strategy_status", "killed_fallback", "stale or mismatched strategy identity"),
        ("matrix_hash", "old-matrix", "stale or mismatched strategy identity"),
        ("corner_fingerprint", "stale", "stale strategy corner fingerprint"),
    ],
)
def test_candidate_rejects_missing_or_stale_signed_identity(field, value, match):
    row = _sn_row("ratio_meanyr", "crps", "nll", True, 0.2)
    row[field] = value
    with pytest.raises(ValueError, match=match):
        mc._candidate(pd.DataFrame([row]))


def test_candidate_rejects_noncanonical_controls_json():
    row = _sn_row("ratio_meanyr", "crps", "nll", True, 0.2)
    row["controls_json"] = json.dumps(json.loads(row["controls_json"]))
    with pytest.raises(ValueError, match="stale or noncanonical strategy controls"):
        mc._candidate(pd.DataFrame([row]))


# --- meditate subprocess primitive ------------------------------------------------------------


def test_run_meditate_true_on_clean_exit_false_on_error(monkeypatch, tmp_path):
    """`_run_meditate` reports subprocess success/failure only — it does not read the ship verdict."""
    monkeypatch.setattr(mc, "_CONFIRM_LOG_ROOT", tmp_path)
    commands = []
    monkeypatch.setattr(
        mc,
        "_run_meditate_with_lock_retry",
        lambda cmd, path, timeout: commands.append(cmd),
    )
    ordinary = mc._candidate(
        pd.DataFrame(
            [
                _signed_row(
                    "NBA",
                    "PTS",
                    "SkewNormal",
                    {
                        "dist": "SkewNormal",
                        "normalization": "ratio_meanyr",
                        "dist_training_loss": "crps",
                        "sn_param": "direct",
                        "blending_loss_fn": "nll",
                    },
                    True,
                    0.2,
                )
            ]
        )
    )
    assert mc._run_meditate("NBA", "PTS", ordinary) is True
    # NBA/PTS is a registered role-registry cell, so the role×position two-part strategy is
    # applicable there; the confirm full-HPO command stamps the structural selector as "auto"
    # (meditate resolves it from stat_meta, which is "none" for this base corner).
    assert commands[0][commands[0].index("--structural-strategy") + 1] == "auto"

    structural = mc._candidate(
        pd.DataFrame(
            [
                _structural_row(
                    AFFINE_STRATEGY,
                    "rushing yards",
                )
            ]
        )
    )
    assert mc._run_meditate("NFL", "rushing yards", structural) is True
    command = commands[-1]
    assert command[command.index("--structural-strategy") + 1] == "auto"
    assert command[command.index("--stabilization") + 1] == "None"

    def boom(*a, **k):
        raise mc.subprocess.CalledProcessError(1, "meditate")

    monkeypatch.setattr(mc, "_run_meditate_with_lock_retry", boom)
    assert mc._run_meditate("NBA", "PTS", ordinary) is False


def test_ship_verdict_is_bound_to_reported_structural_method(monkeypatch):
    slug = AFFINE_STRATEGY
    expected = build_artifact_identity(
        slug,
        "NFL",
        "rushing yards",
        dict(get_strategy(slug).fixed_controls),
        matrix_hash=_MATRIX_SHA,
    )
    stats = pd.DataFrame(
        {
            "league": ["NFL"],
            "market": ["rushing yards"],
            "strategy_slug": [slug],
            "structural_strategy": [slug],
            "strategy_signature": [expected.signature],
            "strategy_implementation_version": [expected.implementation_version],
            "artifact_schema_version": [expected.artifact_schema_version],
            "strategy_status": ["active"],
            "strategy_controls_json": [expected.controls_json],
            "strategy_corner_fingerprint": [expected.corner_fingerprint],
            "strategy_matrix_hash": [expected.matrix_hash],
            "strategy_split_fingerprint": [expected.split_fingerprint],
            "ship": [True],
        }
    )
    monkeypatch.setattr(mc.pd, "read_parquet", lambda *args, **kwargs: stats)

    assert mc._ship_from_model_stats("NFL", "rushing yards", expected) is True
    stats.loc[0, "strategy_controls_json"] = "{}"
    assert mc._ship_from_model_stats("NFL", "rushing yards", expected) is False
    stats.loc[0, "strategy_controls_json"] = expected.controls_json
    wrong_controls = {
        "dist": "SkewNormal",
        "normalization": "ratio_meanyr",
        "dist_training_loss": "crps",
        "sn_param": "direct",
        "blending_loss_fn": "nll",
    }
    wrong = build_artifact_identity(
        "SkewNormal",
        "NFL",
        "rushing yards",
        wrong_controls,
        matrix_hash=_MATRIX_SHA,
    )
    assert mc._ship_from_model_stats("NFL", "rushing yards", wrong) is False


def test_withheld_confirm_fails_closed_before_model_stats_when_artifacts_do_not_match(monkeypatch):
    candidate = mc._candidate(pd.DataFrame([_sn_row("ratio_meanyr", "crps", "nll", True, 0.2)]))
    monkeypatch.setattr(mc, "_run_meditate", lambda *args: True)
    monkeypatch.setattr(mc, "_produced_artifacts_match", lambda *args: False)
    monkeypatch.setattr(
        mc,
        "_ship_from_model_stats",
        lambda *args: pytest.fail("unverified artifacts must never reach the ship verdict"),
    )
    assert mc._confirm_meditate("WNBA", "AST", candidate) is False


# --- persist / confirm / revert --------------------------------------------------------------


def test_confirm_one_pass_keeps_devel(monkeypatch, tmp_path):
    meta = {"WNBA": {"AST": _sn_original()}}
    writes = _fake_meta_disk(monkeypatch, tmp_path)
    monkeypatch.setattr(mc, "_snapshot_cell", lambda lg, mkt: tmp_path / "snapshot")
    monkeypatch.setattr(mc, "_confirm_meditate", lambda lg, mkt, candidate: True)
    pruned = []
    monkeypatch.setattr(mc, "prune_model_pickle", lambda lg, mkt: pruned.append((lg, mkt)))

    cand = {
        "league": "WNBA",
        "market": "AST",
        "edits": {
            "target_normalization": "centered_additive_mean10",
            "sn_param": "centered",
            "blending": "crps",
        },
    }
    assert mc._confirm_one(meta, cand) == ("WNBA", "AST", "SHIPPED", [])
    # Persisted: the edits + shipped=devel; no revert, no prune. A shipping centered corner
    # writes sn_param="centered" — the warm cron retrain reads it back from stat_meta.
    assert meta["WNBA"]["AST"]["shipped"] == "devel"
    assert meta["WNBA"]["AST"]["target_normalization"] == "centered_additive_mean10"
    assert meta["WNBA"]["AST"]["sn_param"] == "centered"
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
    monkeypatch.setattr(mc, "_snapshot_cell", lambda lg, mkt: mc.pathlib.Path("/tmp/snapshot"))
    restored = []

    def restore(lg, mkt, backup, current_meta, prior):
        restored.append((lg, mkt, backup))
        current_meta[lg][mkt] = prior

    monkeypatch.setattr(mc, "_restore_cell", restore)
    monkeypatch.setattr(mc, "_confirm_meditate", lambda lg, mkt, candidate: False)
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
    assert restored == [("WNBA", "AST", mc.pathlib.Path("/tmp/snapshot"))]
    assert pruned == [("WNBA", "AST")]  # pickle pruned — the cell cannot serve


def test_confirm_one_exception_restores_once_after_single_candidate_transition(
    monkeypatch, tmp_path
):
    original = _sn_original()
    meta = {"WNBA": {"AST": dict(original)}}
    states = []
    monkeypatch.setattr(
        mc,
        "_atomic_write_meta",
        lambda current: states.append(json.loads(json.dumps(current))),
    )
    monkeypatch.setattr(mc, "_snapshot_cell", lambda lg, mkt: tmp_path / "snapshot")

    def restore(lg, mkt, backup, current_meta, prior):
        current_meta[lg][mkt] = prior
        mc._atomic_write_meta(current_meta)

    monkeypatch.setattr(mc, "_restore_cell", restore)
    monkeypatch.setattr(
        mc,
        "_confirm_meditate",
        lambda lg, mkt, candidate: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    pruned = []
    monkeypatch.setattr(mc, "prune_model_pickle", lambda lg, mkt: pruned.append((lg, mkt)))
    candidate = {
        "league": "WNBA",
        "market": "AST",
        "edits": {"target_normalization": "ratio_meanyr", "blending": "crps"},
    }

    with pytest.raises(RuntimeError, match="boom"):
        mc._confirm_one(meta, candidate)

    assert len(states) == 2
    assert states[0]["WNBA"]["AST"]["shipped"] == "devel"
    assert states[1]["WNBA"]["AST"] == original
    assert meta["WNBA"]["AST"] == original
    assert pruned == [("WNBA", "AST")]


# --- end-to-end loop -------------------------------------------------------------------------


def test_run_confirm_yes_persists_and_confirms(monkeypatch, capsys, tmp_path):
    board = pd.DataFrame(
        [
            _sn_row("centered_additive_mean10", "crps", "crps", True, 0.25),
            _sn_row("ratio_meanyr", "crps", "nll", False, -0.1),
        ]
    )
    meta = {"WNBA": {"AST": _sn_original()}}
    monkeypatch.setattr(mc, "load_stat_meta", lambda path: meta)
    monkeypatch.setattr(mc, "_backup_stat_meta", lambda: mc.pathlib.Path("/tmp/stat_meta.bak.json"))
    _fake_meta_disk(monkeypatch, tmp_path)
    monkeypatch.setattr(mc, "_snapshot_cell", lambda lg, mkt: tmp_path / "snapshot")
    monkeypatch.setattr(mc, "_confirm_meditate", lambda lg, mkt, candidate: True)
    monkeypatch.setattr(mc, "prune_model_pickle", lambda lg, mkt: False)

    mc.run_confirm(board, yes=True)
    assert meta["WNBA"]["AST"]["shipped"] == "devel"
    assert meta["WNBA"]["AST"]["target_normalization"] == "centered_additive_mean10"
    assert meta["WNBA"]["AST"]["sn_param"] == "direct"  # the swept default persists explicitly too
    assert "SHIPPED" in capsys.readouterr().out


def test_run_confirm_mixed_board_routes_withheld_and_shipped(monkeypatch, capsys):
    """One --confirm run auto-ships the withheld cell via _confirm_one and supersession-tests the live
    cell via _supersede_one — a single combined report."""
    board = pd.DataFrame(
        [
            _sn_row("centered_additive_mean10", "crps", "crps", True, 0.25),  # WNBA AST (withheld)
            _signed_row(
                "NBA",
                "PTS",
                "SkewNormal",
                {
                    "dist": "SkewNormal",
                    "normalization": "centered_additive_mean10",
                    "dist_training_loss": "crps",
                    "sn_param": "direct",
                    "blending_loss_fn": "crps",
                },
                True,
                0.30,
            ),
        ]
    )
    meta = {
        "WNBA": {"AST": _sn_original()},  # withheld
        "NBA": {
            "PTS": {
                "dist": "SkewNormal",
                "shipped": "devel",
                "target_normalization": "ratio_meanyr",
                "blending": "nll",
            }
        },
    }
    monkeypatch.setattr(mc, "load_stat_meta", lambda path: meta)
    monkeypatch.setattr(mc, "_backup_stat_meta", lambda: mc.pathlib.Path("/tmp/stat_meta.bak.json"))
    calls = {"confirm": [], "supersede": []}
    monkeypatch.setattr(
        mc,
        "_confirm_one",
        lambda m, c: calls["confirm"].append(c["market"]) or ("WNBA", "AST", "SHIPPED", []),
    )
    monkeypatch.setattr(
        mc,
        "_supersede_one",
        lambda m, c: calls["supersede"].append(c["market"]) or ("NBA", "PTS", "SUPERSEDED", []),
    )

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
    mlb_row = _signed_row(
        "MLB",
        "total bases",
        "ZINB",
        {
            "dist": "ZINB",
            "zinb_mode": "hurdle",
            "count_dispersion_objective": "pit_ks",
            "blending_loss_fn": "crps",
        },
        True,
        0.04,
    )
    board = pd.DataFrame([_sn_row("centered_additive_mean10", "crps", "crps", True, 0.25), mlb_row])
    meta = {
        "WNBA": {"AST": _sn_original()},
        "MLB": {"total bases": {"dist": "ZINB", "shipped": "withheld"}},
    }
    monkeypatch.setattr(mc, "load_stat_meta", lambda path: meta)
    monkeypatch.setattr(mc, "_backup_stat_meta", lambda: mc.pathlib.Path("/tmp/stat_meta.bak.json"))
    confirmed = []
    monkeypatch.setattr(
        mc,
        "_confirm_one",
        lambda m, c: confirmed.append(c["market"]) or ("WNBA", "AST", "SHIPPED", []),
    )

    mc.run_confirm(board, yes=True)
    assert confirmed == ["AST"]
    assert meta["MLB"]["total bases"]["shipped"] == "withheld"
    assert "ACTIVATION-GATED MLB total bases" in capsys.readouterr().out


def test_run_confirm_all_gated_returns_before_backup(monkeypatch, capsys):
    """When every candidate is activation-gated the loop exits before the backup/persist step."""
    monkeypatch.setattr(mc, "_ACTIVATION_GATED_LEAGUES", ("MLB", "NHL"))
    nhl_row = _signed_row(
        "NHL",
        "saves",
        "SkewNormal",
        {
            "dist": "SkewNormal",
            "normalization": "centered_additive_mean10",
            "dist_training_loss": "crps",
            "sn_param": "direct",
            "blending_loss_fn": "nll",
        },
        True,
        0.04,
    )
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


def test_restore_removes_candidate_artifact_absent_from_snapshot(monkeypatch, tmp_path):
    artifact = tmp_path / "NFL_receiving-yards.mdl"
    monkeypatch.setattr(mc, "_cell_artifacts", lambda lg, mkt: [artifact])
    monkeypatch.setattr(mc, "_CONFIRM_LOG_ROOT", tmp_path / "logs")
    backup = mc._snapshot_cell("NFL", "receiving yards")
    artifact.write_text("candidate")
    meta = {"NFL": {"receiving yards": {"shipped": "devel"}}}
    monkeypatch.setattr(mc, "_atomic_write_meta", lambda current: None)

    mc._restore_cell("NFL", "receiving yards", backup, meta, {"shipped": "withheld"})

    assert not artifact.exists()
    assert meta["NFL"]["receiving yards"] == {"shipped": "withheld"}


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
    return mc._candidate(
        pd.DataFrame(
            [
                _signed_row(
                    "NBA",
                    "PTS",
                    "SkewNormal",
                    {
                        "dist": "SkewNormal",
                        "normalization": "centered_additive_mean10",
                        "dist_training_loss": "crps",
                        "sn_param": "direct",
                        "blending_loss_fn": "crps",
                    },
                    True,
                    0.2,
                )
            ]
        )
    )


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


def _patch_supersede_io(monkeypatch, *, verdict, meditate_ok=True, model_stats_ok=True):
    """Patch the heavy IO of _supersede_one; return (restored, pruned) spy lists."""
    monkeypatch.setattr(mc, "_snapshot_cell", lambda lg, mkt: mc.pathlib.Path("/tmp/bk"))
    monkeypatch.setattr(mc, "_run_meditate", lambda lg, mkt, candidate: meditate_ok)
    monkeypatch.setattr(mc, "_produced_artifacts_match", lambda lg, mkt, candidate: True)
    monkeypatch.setattr(
        mc,
        "_ship_from_model_stats",
        lambda lg, mkt, expected: model_stats_ok,
    )
    monkeypatch.setattr(mc, "load_test_set", lambda path, col: pd.DataFrame())
    monkeypatch.setattr(mc, "supersede_verdict", lambda *a, **k: verdict)
    monkeypatch.setattr(mc, "_atomic_write_meta", lambda m: None)
    restored, pruned = [], []
    monkeypatch.setattr(
        mc, "_restore_cell", lambda lg, mkt, bk, m, orig: restored.append((lg, mkt, orig))
    )
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
    restored, pruned = _patch_supersede_io(
        monkeypatch, verdict=_verdict(ship=True), meditate_ok=False
    )
    result = mc._supersede_one(meta, _supersede_cand())
    assert result[:3] == ("NBA", "PTS", "HELD")
    assert result[3] == ["retrain error"]
    assert restored[0][:2] == ("NBA", "PTS")
    assert pruned == []


def test_supersede_artifact_identity_failure_restores_incumbent(monkeypatch):
    meta = _shipped_meta()
    restored, pruned = _patch_supersede_io(monkeypatch, verdict=_verdict(ship=True))
    monkeypatch.setattr(mc, "_produced_artifacts_match", lambda *args: False)
    result = mc._supersede_one(meta, _supersede_cand())
    assert result == ("NBA", "PTS", "HELD", ["artifact identity"])
    assert restored[0][:2] == ("NBA", "PTS")
    assert pruned == []


def test_supersede_model_stats_identity_or_ship_failure_restores_before_verdict_and_prompt(
    monkeypatch,
):
    meta = _shipped_meta()
    cand = _supersede_cand()
    restored, pruned = _patch_supersede_io(
        monkeypatch,
        verdict=_verdict(ship=True),
        model_stats_ok=False,
    )
    checked = []
    monkeypatch.setattr(
        mc,
        "_ship_from_model_stats",
        lambda lg, mkt, expected: checked.append((lg, mkt, expected)) or False,
    )
    monkeypatch.setattr(
        mc,
        "load_test_set",
        lambda *args, **kwargs: pytest.fail("model_stats failure must stop before test-set load"),
    )
    monkeypatch.setattr(
        mc,
        "supersede_verdict",
        lambda *args, **kwargs: pytest.fail("model_stats failure must stop before S1/S2/S3"),
    )
    monkeypatch.setattr(
        mc.click,
        "confirm",
        lambda *args, **kwargs: pytest.fail("model_stats failure must never prompt"),
    )

    result = mc._supersede_one(meta, cand)

    assert result == ("NBA", "PTS", "HELD", ["model_stats identity/ship"])
    assert checked == [("NBA", "PTS", mc._candidate_identity(cand))]
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
