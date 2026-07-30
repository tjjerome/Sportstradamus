"""Unit tests for the Operation Ship 75 confirm-and-ship loop (``training.model_strategy.confirm``).

Each cell nominates its top swept corners plus its seeds and its incumbent, then retrains them at
full HPO in order and keeps the first that ships (devel). No model trains and no real stat_meta is
touched: the ``meditate`` subprocess and the stat_meta / pickle / model_stats IO are monkeypatched,
so the tests pin the decision logic — which corners a cell nominates and in what order, that the
walk stops at the first win, and that a failure reverts the stat_meta entry *and* prunes the pickle
(reverting stat_meta alone would leave a failed cell serving).
"""

import json
import signal

import pandas as pd
import pytest

from sportstradamus.training.model_strategy import (
    CellContext,
    build_artifact_identity,
    controls_json,
    distribution_class,
    get_strategy,
    registered_strategies,
)
from sportstradamus.training.model_strategy import confirm as mc
from sportstradamus.training.model_strategy.sweep import EVAL_SPLIT_CROSSFIT
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


@pytest.fixture(autouse=True)
def _sandbox_nominee_ledger(monkeypatch, tmp_path):
    """Keep the confirm walk's research ledger out of the repo's ``research/`` directory."""
    monkeypatch.setattr(mc, "_NOMINEE_LEDGER_PATH", tmp_path / "confirm_nominee_gates.csv")


@pytest.fixture(autouse=True)
def _no_known_good_corners(monkeypatch):
    """Empty the seed/incumbent lane by default, so a test's board is its whole nominee list.

    The real ``_known_good_corners`` reads the repo's committed stat_meta.json; the seed-lane tests
    below patch explicit corners back in.
    """
    monkeypatch.setattr(mc, "_known_good_corners", lambda context: [])


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
        "eval_split": EVAL_SPLIT_CROSSFIT,
        **controls,
        "ships": ships,
        "slack": slack,
    }


def _sn_controls(norm, dist_loss, blend, sn_param="direct", posthoc="none"):
    return {
        "dist": "SkewNormal",
        "normalization": norm,
        "dist_training_loss": dist_loss,
        "sn_param": sn_param,
        "blending_loss_fn": blend,
        "posthoc": posthoc,
    }


def _sn_row(norm, dist_loss, blend, ships, slack, sn_param="direct", posthoc="none"):
    return _signed_row(
        "WNBA",
        "AST",
        "SkewNormal",
        _sn_controls(norm, dist_loss, blend, sn_param, posthoc),
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


def _mixture_row(league, market, slack):
    """A Mixture corner: it ranks in the sweep but has no serve capability, so it never nominates."""
    return _signed_row(
        league,
        market,
        "Mixture",
        {
            "dist": "Mixture",
            "normalization": "ratio_meanyr",
            "dist_training_loss": "nll",
            "blending_loss_fn": "nll",
            "posthoc": "none",
        },
        True,
        slack,
    )


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
                "posthoc": "none",
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
                "posthoc": "none",
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


def _nominate(*rows):
    """Every nominee for the single-cell board made of ``rows``, in the order confirm will try them."""
    return mc._nominees(pd.DataFrame(list(rows)))


def _seed_corners(monkeypatch, *controls):
    monkeypatch.setattr(
        mc,
        "_known_good_corners",
        lambda context: [(get_strategy(c["dist"]), c) for c in controls],
    )


# --- nomination lane -------------------------------------------------------------------------


def test_nominees_rank_by_slack_and_persist_the_whole_recipe():
    """The top-slack corner leads the list, and every SkewNormal axis — including
    dist_training_loss (S4) — lands in the persisted edits.
    """
    nominated = _nominate(
        _sn_row("centered_additive_mean10", "nll", "crps", True, 0.30),
        _sn_row("centered_additive_mean10", "crps", "crps", True, 0.25, sn_param="centered"),
        _sn_row("ratio_meanyr", "crps", "nll", False, -0.10),
    )
    assert nominated[0]["edits"] == {
        "dist": "SkewNormal",
        "target_normalization": "centered_additive_mean10",
        "dist_training_loss": "nll",
        "sn_param": "direct",
        "blending": "crps",
        "posthoc": "none",
    }
    assert [n["slack"] for n in nominated] == [0.30, 0.25, -0.10]
    assert nominated[0]["source"] == "board slack +0.300"


def test_nominees_ignore_the_ships_flag():
    """``ships`` stays on the board as a human-facing signal only — confirm never reads it.

    Fixed-HP deterministic scoring will not ship a recipe that only passes under full HPO, so
    requiring it left the popular NFL passing cells (zero shipping rows) unconfirmable. The top-slack
    admissible corner is nominated regardless; the six gates still decide, after the retrain.
    """
    nominated = _nominate(
        _sn_row("ratio_meanyr", "crps", "nll", False, -0.20),
        _sn_row("centered_additive_mean10", "crps", "crps", False, -0.05),
    )
    assert [n["slack"] for n in nominated] == [-0.05, -0.20]
    assert nominated[0]["edits"]["target_normalization"] == "centered_additive_mean10"


def test_nominees_cap_the_board_lane_at_top_k():
    board = [
        _sn_row("ratio_meanyr", "crps", "crps", False, 0.05),
        _sn_row("centered_additive_mean10", "crps", "crps", False, 0.40),
        _sn_row("centered_additive_eb_meanyr_k10", "crps", "crps", False, 0.30),
        _sn_row("ratio_projvol", "crps", "crps", False, 0.20),
    ]
    nominated = _nominate(*board)
    assert len(nominated) == mc.CONFIRM_TOP_K
    assert [n["slack"] for n in nominated] == [0.40, 0.30, 0.20]


def test_nominees_drop_legacy_holdout_rows():
    """Legacy rows carry ``eval_split`` NA — they were scored against the ship holdout, so they may
    never nominate however high their slack ranks.
    """
    legacy = _sn_row("centered_additive_mean10", "crps", "crps", True, 0.90)
    legacy["eval_split"] = pd.NA
    nominated = _nominate(legacy, _sn_row("ratio_meanyr", "crps", "nll", False, 0.10))
    assert [n["slack"] for n in nominated] == [0.10]


def test_nominees_all_legacy_board_still_nominates_the_seed(monkeypatch):
    """The seed lane does not read ``eval_split``: a cell whose whole board predates the cross-fit
    frame still gets its known-good recipe, and nothing from the board.
    """
    _seed_corners(monkeypatch, _sn_controls("ratio_projvol", "nll", "nll"))
    legacy = _sn_row("centered_additive_mean10", "crps", "crps", True, 0.90)
    legacy["eval_split"] = pd.NA
    nominated = _nominate(legacy)
    assert [n["source"] for n in nominated] == ["seed/incumbent"]
    assert nominated[0]["edits"]["target_normalization"] == "ratio_projvol"


def test_nominees_order_board_then_seed_then_incumbent(monkeypatch):
    """Ordering is the argument each source can make: the board is the search's own ranked
    recommendation on out-of-fold data, seeds are independent full-HPO evidence the deterministic
    ranking cannot see, and the incumbent goes last so an unlucky list never downgrades a cell.
    """
    _seed_corners(
        monkeypatch,
        _sn_controls("ratio_projvol", "nll", "nll"),
        _sn_controls("centered_additive_eb_meanyr_k10", "crps", "crps"),
    )
    nominated = _nominate(_sn_row("centered_additive_mean10", "crps", "crps", False, 0.05))
    assert [n["source"] for n in nominated] == [
        "board slack +0.050",
        "seed/incumbent",
        "seed/incumbent",
    ]
    assert [n["edits"]["target_normalization"] for n in nominated] == [
        "centered_additive_mean10",
        "ratio_projvol",
        "centered_additive_eb_meanyr_k10",
    ]


def test_nominees_dedupe_a_seed_that_repeats_a_board_corner(monkeypatch):
    """A cell that already ships the recipe its board topped must retrain it once, not twice."""
    _seed_corners(monkeypatch, _sn_controls("centered_additive_mean10", "crps", "crps"))
    nominated = _nominate(_sn_row("centered_additive_mean10", "crps", "crps", False, 0.12))
    assert [n["source"] for n in nominated] == ["board slack +0.120"]


def test_nominees_skip_mixture_until_serving_lands():
    """Serve-iff-ship: a Mixture corner may top the board but must not be nominated while model_prob
    has no Mixture branch — the next-best non-Mixture corner is nominated instead, and an
    all-Mixture slice nominates nothing.
    """
    mix_row = _mixture_row("WNBA", "AST", 0.9)
    sn_row = _sn_row("centered_additive_mean10", "crps", "crps", True, 0.2)
    assert [n["family"] for n in _nominate(mix_row, sn_row)] == ["SkewNormal"]
    assert _nominate(mix_row) == []


def test_nominees_zinb_is_fully_persistable():
    """Every ZINB axis persists (empty defaults), so its top corner always nominates; the persisted
    edits include the swept ``dist`` (which pins the winning count family).
    """
    nominated = _nominate(_zinb_row("hurdle", "pit_ks", "crps", True, 0.2))
    assert nominated[0]["edits"] == {
        "dist": "ZINB",
        "zinb_mode": "hurdle",
        "count_dispersion_objective": "pit_ks",
        "blending": "crps",
        "posthoc": "none",
        # A count winner can land on a continuous-configured cell, so the family pins the slug
        # back to none rather than leaving one ship_config._validate_cell rejects.
        "target_normalization": "none",
    }


def test_nominees_cross_family_negbin_leads():
    """A count cell's slice mixes ZINB and NegBin corners; the top-slack persistable one is NegBin,
    so the lead nominee flips the cell's family and its edits carry NO ``zinb_mode``.
    """
    nominated = _nominate(
        _zinb_row("hurdle", "pit_ks", "crps", True, 0.10),
        _negbin_row("crps", "nll", False, 0.22),  # top slack, NegBin
        _zinb_row("joint", "crps", "nll", False, -0.05),
    )
    assert nominated[0]["family"] == "NegBin"
    assert nominated[0]["edits"] == {
        "dist": "NegBin",
        "count_dispersion_objective": "crps",
        "blending": "nll",
        "posthoc": "none",
        "target_normalization": "none",
    }
    assert "zinb_mode" not in nominated[0]["edits"]  # the flip never reads the NaN zinb_mode column
    assert nominated[0]["slack"] == 0.22


def test_nominees_cross_family_zinb_leads():
    """Same ZINB+NegBin mix but a ZINB corner has the top slack, so the lead nominee stays ZINB and
    its edits include ``dist=ZINB`` and ``zinb_mode`` — NegBin's blank columns are never consulted.
    """
    nominated = _nominate(
        _zinb_row("hurdle", "pit_ks", "crps", True, 0.30),  # top slack, ZINB
        _negbin_row("crps", "nll", True, 0.18),
    )
    assert nominated[0]["family"] == "ZINB"
    assert nominated[0]["edits"] == {
        "dist": "ZINB",
        "zinb_mode": "hurdle",
        "count_dispersion_objective": "pit_ks",
        "blending": "crps",
        "posthoc": "none",
        "target_normalization": "none",
    }
    assert nominated[0]["slack"] == 0.30


def test_nominees_structural_method_persists_full_recipe_and_identity():
    slug = AFFINE_STRATEGY
    cand = _nominate(_structural_row(slug, "rushing yards"))[0]
    assert cand["strategy_slug"] == slug
    assert cand["structural_strategy"] == slug
    # The method rides the single-valued ``posthoc`` calibration pool: its slug IS the
    # persisted field value, and no separate structural_strategy edit is written.
    assert cand["edits"] == {
        "dist": "SkewNormal",
        "target_normalization": "ratio_meanyr",
        "dist_training_loss": "crps",
        "sn_param": "direct",
        "blending": "nll",
        "hpo_selection": "loss",
        "posthoc": slug,
    }


def test_nominees_base_winner_clears_structural_method_without_stringifying_nan():
    row = _signed_row(
        "NFL",
        "receiving yards",
        "SkewNormal",
        _sn_controls("ratio_meanyr", "crps", "nll"),
        True,
        0.2,
    )
    cand = _nominate(row)[0]
    assert cand["structural_strategy"] == "none"
    # A base winner writes no structural edit — its ``posthoc`` stays whatever the cell had.
    assert "structural_strategy" not in cand["edits"]
    assert "nan" not in cand["edits"].values()


def test_nominees_reject_missing_required_structural_axis():
    slug = AFFINE_STRATEGY
    row = _structural_row(slug, "rushing yards")
    row["hpo_selection"] = float("nan")
    with pytest.raises(ValueError, match="hpo_selection contradicts controls_json"):
        _nominate(row)


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
def test_nominees_reject_missing_or_stale_signed_identity(field, value, match):
    row = _sn_row("ratio_meanyr", "crps", "nll", True, 0.2)
    row[field] = value
    with pytest.raises(ValueError, match=match):
        _nominate(row)


def test_nominees_reject_noncanonical_controls_json():
    row = _sn_row("ratio_meanyr", "crps", "nll", True, 0.2)
    row["controls_json"] = json.dumps(json.loads(row["controls_json"]))
    with pytest.raises(ValueError, match="stale or noncanonical strategy controls"):
        _nominate(row)


def test_candidates_are_one_nominee_list_per_cell():
    """``_candidates`` groups the board by cell; a cell whose every corner is unconfirmable drops out
    rather than contributing an empty walk.
    """
    board = pd.DataFrame(
        [
            _sn_row("centered_additive_mean10", "crps", "crps", False, 0.25),
            _zinb_row("hurdle", "pit_ks", "crps", False, 0.10),
            _mixture_row("NBA", "PTS", 0.90),
        ]
    )
    grouped = mc._candidates(board)
    assert [(n[0]["league"], n[0]["market"]) for n in grouped] == [
        ("WNBA", "AST"),
        ("MLB", "pitcher strikeouts"),
    ]
    assert [len(n) for n in grouped] == [1, 1]


def test_split_shippable_partitions_cells_not_nominees():
    """The release surface is a property of the cell, so a withheld cell's whole nominee list routes
    to ``_confirm_one`` and a live cell's to the supersession test.
    """
    withheld = [{"league": "WNBA", "market": "AST"}, {"league": "WNBA", "market": "AST"}]
    live = [{"league": "NBA", "market": "PTS"}]
    meta = {"WNBA": {"AST": {"shipped": "withheld"}}, "NBA": {"PTS": {"shipped": "devel"}}}
    assert mc._split_shippable([withheld, live], meta) == ([withheld], [live])


# --- meditate subprocess primitive ------------------------------------------------------------


def test_run_meditate_reports_why_the_subprocess_failed(monkeypatch, tmp_path):
    """`_run_meditate` reports subprocess success/failure only — it does not read the ship verdict.

    A signal death and an ordinary non-zero exit must not report the same reason: the confirm
    walk's verdict line is the only place a native abort is ever named.
    """
    monkeypatch.setattr(mc, "_CONFIRM_LOG_ROOT", tmp_path)
    commands = []
    monkeypatch.setattr(
        mc,
        "_run_meditate_with_lock_retry",
        lambda cmd, path, timeout: commands.append(cmd),
    )
    ordinary = _nominate(
        _signed_row(
            "NBA",
            "PTS",
            "SkewNormal",
            _sn_controls("ratio_meanyr", "crps", "nll"),
            True,
            0.2,
        )
    )[0]
    assert mc._run_meditate("NBA", "PTS", ordinary) == ""
    # The retired --structural-strategy axis emits no selector flag; every base-family control is
    # persisted, so the full-HPO confirm forces none of them and reads the freshly-written cell.
    assert "--structural-strategy" not in commands[0]
    assert "--posthoc" not in commands[0]

    structural = _nominate(_structural_row(AFFINE_STRATEGY, "rushing yards"))[0]
    assert mc._run_meditate("NFL", "rushing yards", structural) == ""
    command = commands[-1]
    assert "--structural-strategy" not in command
    assert command[command.index("--stabilization") + 1] == "None"
    # ``posthoc`` is now persisted (it carries the method slug), so full-HPO confirm skips
    # --posthoc and lets the just-persisted stat_meta posthoc select the structural lane.
    assert "--posthoc" not in command

    def boom(*a, **k):
        raise mc.subprocess.CalledProcessError(1, "meditate")

    monkeypatch.setattr(mc, "_run_meditate_with_lock_retry", boom)
    assert mc._run_meditate("NBA", "PTS", ordinary) == "exit 1"

    def abort(*a, **k):
        raise mc.subprocess.CalledProcessError(-signal.SIGABRT, "meditate")

    monkeypatch.setattr(mc, "_run_meditate_with_lock_retry", abort)
    assert mc._run_meditate("NBA", "PTS", ordinary) == "native abort (SIGABRT)"


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
    wrong = build_artifact_identity(
        "SkewNormal",
        "NFL",
        "rushing yards",
        _sn_controls("ratio_meanyr", "crps", "nll"),
        matrix_hash=_MATRIX_SHA,
    )
    assert mc._ship_from_model_stats("NFL", "rushing yards", wrong) is False


def test_a_gate_miss_is_reported_as_the_gate_not_a_missing_artifact(monkeypatch):
    """Serve-iff-ship prunes the pickle of a cell that failed a gate, so the artifact leg fails too.

    Checking artifact identity first reported a plain Gate-4 miss as a missing serving artifact --
    the real NBA FGA confirm did exactly that.
    """
    candidate = _nominate(_sn_row("ratio_meanyr", "crps", "nll", True, 0.2))[0]
    monkeypatch.setattr(mc, "_run_meditate", lambda *args: "")
    monkeypatch.setattr(mc, "_retrained_matrix_hash", lambda lg, mkt: _MATRIX_SHA)
    monkeypatch.setattr(mc, "_failed_gates_after", lambda lg, mkt: ["g4"])
    monkeypatch.setattr(
        mc,
        "_produced_artifacts_match",
        lambda *args: pytest.fail("a gate miss must not be diagnosed through the pruned pickle"),
    )

    assert mc._confirm_meditate("WNBA", "AST", candidate) == ["g4"]


def test_confirm_accepts_a_retrain_whose_force_update_moved_the_matrix(monkeypatch):
    """The board's matrix hash is stale by construction, so the ship check must not require it.

    ``_run_meditate`` passes ``--force``, which updates the league gamelogs and appends gamedays; the
    training matrix is rewritten and re-hashed. The first overnight run rejected every gate-passing
    count-family nominee on this alone, and reported it as a bare "(retrain error)" because no gate
    had failed.
    """
    retrained_hash = "matrix-after-the-force-update"
    candidate = _nominate(_sn_row("ratio_meanyr", "crps", "nll", True, 0.2))[0]
    assert candidate["matrix_hash"] != retrained_hash

    reported = build_artifact_identity(
        candidate["strategy_slug"],
        candidate["league"],
        candidate["market"],
        candidate["controls"],
        matrix_hash=retrained_hash,
    )
    stats = pd.DataFrame(
        {
            "league": [candidate["league"]],
            "market": [candidate["market"]],
            "strategy_slug": [reported.strategy_slug],
            "structural_strategy": [reported.structural_strategy],
            "strategy_signature": [reported.signature],
            "strategy_implementation_version": [reported.implementation_version],
            "artifact_schema_version": [reported.artifact_schema_version],
            "strategy_status": ["active"],
            "strategy_controls_json": [reported.controls_json],
            "strategy_corner_fingerprint": [reported.corner_fingerprint],
            "strategy_matrix_hash": [retrained_hash],
            "strategy_split_fingerprint": [reported.split_fingerprint],
            "ship": [True],
        }
    )
    monkeypatch.setattr(mc.pd, "read_parquet", lambda *args, **kwargs: stats)
    monkeypatch.setattr(mc, "_run_meditate", lambda *args: "")
    monkeypatch.setattr(mc, "_failed_gates_after", lambda lg, mkt: [])

    seen = {}

    def artifacts_match(league, market, cand, expected):
        seen["matrix_hash"] = expected.matrix_hash
        return True

    monkeypatch.setattr(mc, "_produced_artifacts_match", artifacts_match)

    assert mc._confirm_meditate(candidate["league"], candidate["market"], candidate) == []
    # The pickle leg is held to the retrain's matrix too, so a stale artifact still fails there.
    assert seen["matrix_hash"] == retrained_hash


def test_withheld_confirm_fails_closed_before_model_stats_when_artifacts_do_not_match(monkeypatch):
    candidate = _nominate(_sn_row("ratio_meanyr", "crps", "nll", True, 0.2))[0]
    monkeypatch.setattr(mc, "_run_meditate", lambda *args: "")
    monkeypatch.setattr(mc, "_retrained_matrix_hash", lambda lg, mkt: _MATRIX_SHA)
    monkeypatch.setattr(mc, "_failed_gates_after", lambda lg, mkt: [])
    monkeypatch.setattr(mc, "_produced_artifacts_match", lambda *args: False)
    monkeypatch.setattr(
        mc,
        "_ship_from_model_stats",
        lambda *args: pytest.fail("unverified artifacts must never reach the ship verdict"),
    )
    assert mc._confirm_meditate("WNBA", "AST", candidate) == ["artifact identity"]


# --- persist / confirm / revert --------------------------------------------------------------


def test_confirm_one_pass_keeps_devel(monkeypatch, tmp_path):
    meta = {"WNBA": {"AST": _sn_original()}}
    writes = _fake_meta_disk(monkeypatch, tmp_path)
    monkeypatch.setattr(mc, "_snapshot_cell", lambda lg, mkt: tmp_path / "snapshot")
    monkeypatch.setattr(mc, "_confirm_meditate", lambda lg, mkt, candidate: [])
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
    monkeypatch.setattr(mc, "_confirm_meditate", lambda lg, mkt, candidate: ["g4"])
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


# --- nominee walk ------------------------------------------------------------------------------


def _walk_stubs(*sources):
    return [{"league": "WNBA", "market": "AST", "source": s} for s in sources]


def test_walk_nominees_stops_at_the_first_win():
    """Once a nominee ships, the cell's remaining ~1h retrains are skipped and the report names the
    deciding nominee.
    """
    tried = []

    def attempt(meta, cand):
        tried.append(cand["source"])
        shipped = len(tried) > 1
        return ("WNBA", "AST", "SHIPPED" if shipped else "REVERTED", [] if shipped else ["g4"])

    nominated = _walk_stubs("board slack +0.300", "board slack +0.200", "board slack +0.100")
    assert mc._walk_nominees({}, nominated, attempt) == (
        "WNBA",
        "AST",
        "SHIPPED",
        [],
        "2/3 board slack +0.200",
    )
    assert tried == ["board slack +0.300", "board slack +0.200"]


def test_walk_nominees_reports_the_last_outcome_when_every_nominee_fails():
    nominated = _walk_stubs("board slack +0.100", "seed/incumbent")
    assert mc._walk_nominees(
        {}, nominated, lambda meta, cand: ("WNBA", "AST", "REVERTED", ["g4"])
    ) == ("WNBA", "AST", "REVERTED", ["g4"], "2/2 seed/incumbent")


def test_walk_nominees_reverts_a_loser_before_the_next_nominee_persists(monkeypatch, tmp_path):
    """The walk needs no revert machinery of its own: ``_confirm_one``'s finally already restored the
    loser's stat_meta entry and pruned its pickle before the winner was persisted.
    """
    original = _sn_original()
    meta = {"WNBA": {"AST": dict(original)}}
    _fake_meta_disk(monkeypatch, tmp_path)
    monkeypatch.setattr(mc, "_snapshot_cell", lambda lg, mkt: tmp_path / "snapshot")
    monkeypatch.setattr(mc, "_cell_artifacts", lambda lg, mkt: [])
    monkeypatch.setattr(mc, "_failed_gates_after", lambda lg, mkt: ["g4"])
    pruned = []
    monkeypatch.setattr(mc, "prune_model_pickle", lambda lg, mkt: pruned.append((lg, mkt)))
    confirmed = []
    monkeypatch.setattr(
        mc,
        "_confirm_meditate",
        lambda lg, mkt, cand: confirmed.append(cand["source"]) or ([] if len(confirmed) > 1 else ["g4"]),
    )
    nominated = [
        {
            "league": "WNBA",
            "market": "AST",
            "source": "board slack +0.300",
            "edits": {"target_normalization": "ratio_projvol", "sn_param": "centered"},
        },
        {
            "league": "WNBA",
            "market": "AST",
            "source": "seed/incumbent",
            "edits": {"target_normalization": "centered_additive_mean10", "blending": "crps"},
        },
    ]

    result = mc._walk_nominees(meta, nominated, mc._confirm_one)

    assert result == ("WNBA", "AST", "SHIPPED", [], "2/2 seed/incumbent")
    assert pruned == [("WNBA", "AST")]  # only the loser was darkened
    assert meta["WNBA"]["AST"]["shipped"] == "devel"
    assert meta["WNBA"]["AST"]["target_normalization"] == "centered_additive_mean10"
    assert "sn_param" not in meta["WNBA"]["AST"]  # the loser's edits are gone


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
    confirmed = []
    monkeypatch.setattr(
        mc,
        "_confirm_meditate",
        lambda lg, mkt, cand: confirmed.append(cand["source"]) or [],
    )
    monkeypatch.setattr(mc, "prune_model_pickle", lambda lg, mkt: False)

    mc.run_confirm(board, yes=True)
    assert meta["WNBA"]["AST"]["shipped"] == "devel"
    assert meta["WNBA"]["AST"]["target_normalization"] == "centered_additive_mean10"
    assert meta["WNBA"]["AST"]["sn_param"] == "direct"  # the swept default persists explicitly too
    assert confirmed == ["board slack +0.250"]  # the runner-up is never retrained
    out = capsys.readouterr().out
    assert "WNBA AST — 2 nominee(s)" in out
    assert "1. [board slack +0.250]" in out
    assert "SHIPPED" in out
    assert "1/2 board slack +0.250" in out  # the report's nominee column


def test_run_confirm_prompt_quotes_the_worst_case_retrain_count(monkeypatch, capsys):
    """The operator is quoted the ceiling — every cell's whole nominee list, not one per cell."""
    board = pd.DataFrame(
        [
            _sn_row("centered_additive_mean10", "crps", "crps", False, 0.25),
            _sn_row("ratio_meanyr", "crps", "nll", False, -0.1),
            _signed_row(
                "NBA",
                "PTS",
                "SkewNormal",
                _sn_controls("centered_additive_mean10", "crps", "crps"),
                False,
                0.3,
            ),
        ]
    )
    meta = {"WNBA": {"AST": _sn_original()}, "NBA": {"PTS": {"shipped": "devel"}}}
    monkeypatch.setattr(mc, "load_stat_meta", lambda path: meta)
    monkeypatch.setattr(
        mc, "_backup_stat_meta", lambda: pytest.fail("an aborted run must not back up stat_meta")
    )
    prompts = []
    monkeypatch.setattr(mc.click, "confirm", lambda text: prompts.append(text) or False)

    mc.run_confirm(board)
    assert "up to 3 full-HPO retrains" in prompts[0]
    assert "aborted" in capsys.readouterr().out


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
                _sn_controls("centered_additive_mean10", "crps", "crps"),
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


def test_run_confirm_reports_nothing_confirmable_when_every_corner_is_unservable(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        mc, "load_stat_meta", lambda path: pytest.fail("an empty board must not read stat_meta")
    )
    mc.run_confirm(pd.DataFrame([_mixture_row("WNBA", "AST", 0.9)]), yes=True)
    assert "no confirmable nominees on the board." in capsys.readouterr().out


def test_activation_gate_empty_post_go():
    """MLB+NHL D1/D2 went GO 2026-07-09 — production gates no league. The guard machinery
    stays for the next onboarding; the tests below monkeypatch it to stay covered."""
    assert mc._ACTIVATION_GATED_LEAGUES == ()


def test_run_confirm_skips_activation_gated_league(monkeypatch, capsys):
    """A withheld board-passer in a gated league is announced and dropped — never persisted or
    retrained — while a covered-league nominee in the same run still confirms."""
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
            "posthoc": "none",
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
    """When every cell is activation-gated the loop exits before the backup/persist step."""
    monkeypatch.setattr(mc, "_ACTIVATION_GATED_LEAGUES", ("MLB", "NHL"))
    nhl_row = _signed_row(
        "NHL",
        "saves",
        "SkewNormal",
        _sn_controls("centered_additive_mean10", "crps", "nll"),
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
    return _nominate(
        _signed_row(
            "NBA",
            "PTS",
            "SkewNormal",
            _sn_controls("centered_additive_mean10", "crps", "crps"),
            True,
            0.2,
        )
    )[0]


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
    monkeypatch.setattr(
        mc, "_run_meditate", lambda lg, mkt, candidate: "" if meditate_ok else "exit 1"
    )
    monkeypatch.setattr(mc, "_retrained_matrix_hash", lambda lg, mkt: _MATRIX_SHA)
    monkeypatch.setattr(mc, "_produced_artifacts_match", lambda *args: True)
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
    assert result[3] == ["retrain exit 1"]
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


def test_nominee_ledger_stays_parseable_when_model_stats_widens(monkeypatch, tmp_path):
    """A later, wider model_stats row must not strand the ledger behind a narrower header."""
    ledger = tmp_path / "confirm_nominee_gates.csv"
    monkeypatch.setattr(mc, "_NOMINEE_LEDGER_PATH", ledger)
    rows = iter(
        [
            pd.Series({"league": "NBA", "market": "PTS", "g4_iqr_ratio": 0.91}),
            pd.Series(
                {"league": "NBA", "market": "PTS", "g4_iqr_ratio": 0.88, "g7_new_gate": 1.0}
            ),
        ]
    )
    monkeypatch.setattr(mc, "_cell_row", lambda league, market, columns: next(rows))
    candidate = {"strategy_slug": "sn-centered", "source": "board"}

    mc._record_nominee_gates("NBA", "PTS", candidate)
    mc._record_nominee_gates("NBA", "PTS", candidate)

    recorded = pd.read_csv(ledger)
    assert len(recorded) == 2
    assert list(recorded["g4_iqr_ratio"]) == [0.91, 0.88]
    assert recorded["g7_new_gate"].isna().tolist() == [True, False]
    assert set(recorded["strategy_slug"]) == {"sn-centered"}


def test_confirm_nominee_cap_truncates_each_cell_after_dedup(monkeypatch):
    """``--confirm-nominees N`` keeps a cell's first N nominees; unset keeps every one."""
    board = pd.DataFrame(
        [
            _sn_row("centered_additive_mean10", "crps", "crps", False, 0.25),
            _sn_row("centered_additive_mean10", "crps", "nll", False, 0.20),
            _sn_row("centered_additive_mean10", "nll", "nll", False, 0.15),
        ]
    )

    uncapped = mc._candidates(board)
    capped = mc._candidates(board, 2)

    assert [len(cell) for cell in uncapped] == [3]
    assert [len(cell) for cell in capped] == [2]
    assert [n["source"] for n in capped[0]] == [n["source"] for n in uncapped[0][:2]]
