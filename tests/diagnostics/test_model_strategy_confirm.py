"""Unit tests for the Operation Ship 75 confirm-and-ship loop (``training.model_strategy.confirm``).

Each cell nominates its top swept corners plus its seeds and its incumbent, runs every nominee
blind at full HPO in a research sandbox, and refits only the top-ranked passer canonically, reading
its ship holdout once. No model trains and no real stat_meta is touched: the ``meditate``
subprocess and the stat_meta / pickle / model_stats IO are monkeypatched, so the tests pin the
decision logic — which corners a cell nominates and in what order, which blind run the walk refits,
that the test half is read at most once per cell, that blind runs never write stat_meta, and that a
failure reverts the stat_meta entry *and* prunes the pickle (reverting stat_meta alone would leave
a failed cell serving).
"""

import json
import signal

import pandas as pd
import pytest

from sportstradamus.stats.model_dependencies import DEPENDENCY_NAMESPACE
from sportstradamus.training.lineage import validate_matrix_manifest
from sportstradamus.training.model_strategy import (
    CellContext,
    build_artifact_identity,
    controls_json,
    distribution_class,
    get_strategy,
    registered_strategies,
    sweep,
)
from sportstradamus.training.model_strategy import confirm as mc
from sportstradamus.training.model_strategy import confirm_nominees as mcn
from sportstradamus.training.model_strategy import confirm_runs as mcr
from sportstradamus.training.model_strategy.sweep import EVAL_SPLIT_CROSSFIT, EVAL_SPLIT_TEST
from sportstradamus.training.role_specs import role_spec_for
from sportstradamus.training.structural_strategies import AFFINE_STRATEGY, TWO_PART_STRATEGY

pytestmark = pytest.mark.diagnostics

_MATRIX_SHA = "matrix-123"
_MATRIX_COLUMNS = frozenset(
    column
    for spec in registered_strategies()
    for column in spec.applicability.required_data_columns
)
# The real functions, captured before the autouse stubs below replace the module attributes.
_REAL_PIN_CELL_MATRIX = mcr.pin_cell_matrix
_REAL_RETRY_WANTED = mcr._retry_calibrated_wanted


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

    monkeypatch.setattr(sweep, "_cell_context", context)


@pytest.fixture(autouse=True)
def _sandbox_writes(monkeypatch, tmp_path):
    """Keep every file the walk writes — stat_meta.json, the ledger, blind sandboxes and logs — off
    the repo's own copies, even on a path a test left unstubbed.
    """
    monkeypatch.setattr(mcr, "STAT_META", tmp_path / "stat_meta.json")
    monkeypatch.setattr(sweep, "NOMINEE_LEDGER_PATH", tmp_path / "confirm_nominee_gates.csv")
    monkeypatch.setattr(mcr, "_CONFIRM_LOG_ROOT", tmp_path / "confirm")


@pytest.fixture(autouse=True)
def _stub_matrix_pin(monkeypatch):
    """Keep the per-walk matrix pin off the real training_data and research dirs.

    The pin tests below call the captured real function (``_REAL_PIN_CELL_MATRIX``) against tmp
    paths instead.
    """
    monkeypatch.setattr(mcr, "pin_cell_matrix", lambda league, market: None)


@pytest.fixture(autouse=True)
def _no_seed_corners(monkeypatch):
    """Empty the seed/incumbent lane by default, so a test's board is its whole nominee list.

    The real ``_seed_corners`` reads the repo's committed stat_meta.json; the seed-lane tests
    below patch explicit corners back in.
    """
    monkeypatch.setattr(mcn, "_seed_corners", lambda context: [])


@pytest.fixture(autouse=True)
def _no_calibrated_retry(monkeypatch):
    """Keep the walk's calibrated fallback out of unrelated tests; a retry test restores it."""
    monkeypatch.setattr(mcr, "_retry_calibrated_wanted", lambda candidate, row, cell: False)


@pytest.fixture(autouse=True)
def _no_real_meditate(monkeypatch):
    """A walk trains every nominee before any attempt, so a path a test left unstubbed would launch
    a real full-HPO meditate; it fails the test instead.
    """
    monkeypatch.setattr(
        mcr,
        "_run_meditate_with_lock_retry",
        lambda *args, **kwargs: pytest.fail("a test reached the real meditate subprocess"),
    )


@pytest.fixture(autouse=True)
def _no_incumbent(monkeypatch):
    """The real lookup reads the repo's committed stat_meta.json; the live-walk tests name one."""
    monkeypatch.setattr(mc, "_incumbent_fingerprint", lambda league, market: None)


def _fake_meta_disk(monkeypatch, tmp_path):
    """Route ``atomic_write_meta`` to a tmp ``STAT_META`` file and return the write-call list.

    Keeps every persist — a walk writes only its pick's edits and their revert — off the repo's
    committed stat_meta.json.
    """
    meta_file = tmp_path / "stat_meta.json"
    monkeypatch.setattr(mcr, "STAT_META", meta_file)
    writes = []
    monkeypatch.setattr(
        mcr,
        "atomic_write_meta",
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


def _zinb_row(mode, disp, blend, ships, slack, posthoc="none"):
    """A ZINB count corner, reindexed to the full board schema (SN-only columns blank)."""
    return {
        **_signed_row(
            "MLB",
            "pitcher strikeouts",
            "ZINB",
            {
                "dist": "ZINB",
                "dist_training_loss": "nll",
                "zinb_mode": mode,
                "count_dispersion_objective": disp,
                "blending_loss_fn": blend,
                "posthoc": posthoc,
            },
            ships,
            slack,
        ),
        "normalization": float("nan"),
        "sn_param": float("nan"),
    }


def _negbin_row(disp, blend, ships, slack, league="MLB", market="pitcher strikeouts"):
    """A plain-NegBin count corner: no ``zinb_mode`` (its persist map omits it), other columns blank."""
    return {
        **_signed_row(
            league,
            market,
            "NegBin",
            {
                "dist": "NegBin",
                "dist_training_loss": "nll",
                "count_dispersion_objective": disp,
                "blending_loss_fn": blend,
                "posthoc": "none",
            },
            ships,
            slack,
        ),
        "zinb_mode": float("nan"),
        "normalization": float("nan"),
        "sn_param": float("nan"),
    }


def _structural_row(slug, market, ships=True, slack=0.4):
    return _signed_row("NFL", market, slug, dict(get_strategy(slug).fixed_controls), ships, slack)


def _nominate(*rows, live=False, **kw):
    """Every nominee for the single-cell board made of ``rows``, in the order confirm will try them."""
    return mcn._nominees(pd.DataFrame(list(rows)), live=live, **kw)


def _weighted_row(norm, slack, weight):
    """A SkewNormal board row carrying the deterministic corner's fitted blend weight."""
    return {**_sn_row(norm, "crps", "crps", True, slack), "model_weight": weight}


def _withheld_meta(board):
    """Every cell on ``board`` marked withheld — the fresh lane, which ranks on slack."""
    return {
        league: {market: {"shipped": mc.WITHHELD} for market in sub["market"]}
        for league, sub in board.groupby("league")
    }


def _seed_corners(monkeypatch, *controls):
    monkeypatch.setattr(
        mcn,
        "_seed_corners",
        lambda context: [(get_strategy(c["dist"]), c) for c in controls],
    )


# --- nomination lane -------------------------------------------------------------------------


def test_nominees_rank_by_slack_and_persist_the_whole_recipe():
    """The top-slack corner leads the list, every SkewNormal axis — including dist_training_loss
    (S4) — lands in the persisted edits, and the lane never dips below zero to fill the top-K.
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
    assert [n["slack"] for n in nominated] == [0.30, 0.25]
    assert nominated[0]["source"] == "board slack +0.300"
    assert all(n["board_rank"] == 0.30 for n in nominated)


def test_nominees_skip_a_cell_with_no_positive_board_rank(monkeypatch):
    """A cell whose best admissible corner is non-positive nominates nothing — not even its seed.

    ``ships`` is still never read (a positive-rank corner nominates whatever its ships flag says),
    but the board-confidence bar replaces the old walk-everything contract: full HPO occasionally
    rescued a board-negative recipe (NBA FTM shipped at board −1.86), and the operator traded that
    tail for not burning retrains on cells the board predicts to fail.
    """
    _seed_corners(monkeypatch, _sn_controls("ratio_projvol", "nll", "nll"))
    nominated = _nominate(
        _sn_row("ratio_meanyr", "crps", "nll", False, -0.20),
        _sn_row("centered_additive_mean10", "crps", "crps", True, -0.05),
    )
    assert nominated == []


def test_nominees_cap_the_board_lane_at_top_k():
    board = [
        _sn_row("ratio_meanyr", "crps", "crps", False, 0.05),
        _sn_row("centered_additive_mean10", "crps", "crps", False, 0.40),
        _sn_row("centered_additive_eb_meanyr_k10", "crps", "crps", False, 0.30),
        _sn_row("ratio_projvol", "crps", "crps", False, 0.20),
    ]
    nominated = _nominate(*board)
    assert len(nominated) == mcn.CONFIRM_TOP_K
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
        "dist_training_loss": "nll",
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
        "dist_training_loss": "nll",
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
        "dist_training_loss": "nll",
        "zinb_mode": "hurdle",
        "count_dispersion_objective": "pit_ks",
        "blending": "crps",
        "posthoc": "none",
        "target_normalization": "none",
    }
    assert nominated[0]["slack"] == 0.30


def _integer_target_context(monkeypatch):
    """A WNBA-style continuous cell whose target sits on the integer lattice (e.g. WNBA AST)."""

    def context(league, market):
        columns = set(_MATRIX_COLUMNS)
        spec = role_spec_for(league, market)
        if spec is not None:
            columns |= set(spec.all_columns)
        return CellContext(
            league,
            market,
            "SkewNormal",
            distribution_class("SkewNormal"),
            frozenset(columns),
            _MATRIX_SHA,
            target_is_integer=True,
        )

    monkeypatch.setattr(sweep, "_cell_context", context)


def _all_sn_top(*, count_slack=None):
    """Four SkewNormal corners outranking an optional count corner — the burned-walk board shape."""
    rows = [
        _sn_row("centered_additive_mean10", "crps", "crps", False, 0.40),
        _sn_row("centered_additive_eb_meanyr_k10", "crps", "crps", False, 0.30),
        _sn_row("ratio_meanyr", "crps", "crps", False, 0.20),
        _sn_row("ratio_projvol", "crps", "crps", False, 0.10),
    ]
    if count_slack is not None:
        rows.append(_negbin_row("crps", "nll", False, count_slack, league="WNBA", market="AST"))
    return rows


def test_nominees_promote_the_best_count_corner_over_a_near_identical_continuous_one(monkeypatch):
    """A board whose leaders are four SkewNormal corners still walks the count family second.

    Slot 1 is the board's own leader. Slot 2 goes to the best corner from a family nothing has
    walked yet — here the 0.05 NegBin, not the 0.02 one and not the next SkewNormal — because a
    second SkewNormal corner would re-learn the divergence the first one already showed. Slot 3
    then takes the highest-ranked corner whose Gate-4 mechanism is still unseen.
    """
    _integer_target_context(monkeypatch)
    nominated = _nominate(
        *_all_sn_top(count_slack=0.05),
        _negbin_row("pit_ks", "nll", False, 0.02, league="WNBA", market="AST"),
    )
    assert [n["family"] for n in nominated] == ["SkewNormal", "NegBin", "SkewNormal"]
    assert [n["slack"] for n in nominated] == [0.40, 0.05, 0.30]


def test_nominees_interleave_noop_when_count_already_in_top_slots(monkeypatch):
    _integer_target_context(monkeypatch)
    nominated = _nominate(
        _sn_row("centered_additive_mean10", "crps", "crps", False, 0.40),
        _negbin_row("crps", "nll", False, 0.30, league="WNBA", market="AST"),
        _sn_row("ratio_meanyr", "crps", "crps", False, 0.20),
    )
    assert [n["family"] for n in nominated] == ["SkewNormal", "NegBin", "SkewNormal"]


def test_nominees_interleave_noop_without_a_count_board_row(monkeypatch):
    _integer_target_context(monkeypatch)
    nominated = _nominate(*_all_sn_top())
    assert [n["family"] for n in nominated] == ["SkewNormal"] * mcn.CONFIRM_TOP_K


def test_nominees_interleave_noop_on_non_integer_target():
    """The count-class backup is integer-target gated; mechanism diversity is not.

    The default fixture context carries no integer-lattice fact, so the backup adds nothing and
    the lane stays at ``CONFIRM_TOP_K``. Diversity still orders those slots, so the count corner
    reaches slot 2 on its own — the backup's job here is only the extra insert it declines to make.
    """
    nominated = _nominate(*_all_sn_top(count_slack=0.05))
    assert len(nominated) == mcn.CONFIRM_TOP_K
    assert [n["family"] for n in nominated] == ["SkewNormal", "NegBin", "SkewNormal"]


def test_nominees_interleave_never_borrows_a_non_positive_count_corner(monkeypatch):
    """The count-class backup honors the same board-confidence bar as the lane it patches."""
    _integer_target_context(monkeypatch)
    nominated = _nominate(*_all_sn_top(count_slack=-0.05))
    assert [n["family"] for n in nominated] == ["SkewNormal"] * mcn.CONFIRM_TOP_K


def test_nominees_sort_by_discounted_slack_when_present_else_slack():
    """The board lane ranks by the sweep's confirm-priced ``discounted_slack`` when the column
    exists, so the two changes can land in either order; the source label keeps the raw slack.
    """
    raw_leader = _sn_row("ratio_meanyr", "crps", "nll", False, 0.40)
    discount_leader = _sn_row("centered_additive_mean10", "crps", "crps", False, 0.30)
    assert [n["slack"] for n in _nominate(raw_leader, discount_leader)] == [0.40, 0.30]

    raw_leader["discounted_slack"] = -0.10
    discount_leader["discounted_slack"] = 0.25
    nominated = _nominate(raw_leader, discount_leader)
    # The discounted rank also carries the confidence bar: the raw leader's −0.10 drops it.
    assert [n["slack"] for n in nominated] == [0.30]
    assert nominated[0]["edits"]["target_normalization"] == "centered_additive_mean10"
    assert nominated[0]["source"] == "board slack +0.300"
    assert nominated[0]["board_rank"] == 0.25


def test_nominees_veto_on_veto_slack_and_rank_on_discounted_slack():
    """brief R3: admissibility is priced at the milder q50 ``veto_slack`` while ordering stays on
    the q75 ``discounted_slack``, so a cell whose corners all go rank-negative under the
    confirm-priced discount still walks when the median price says they clear the gates.
    """
    rescued = _sn_row("ratio_meanyr", "crps", "nll", False, 0.10)
    rescued["discounted_slack"], rescued["veto_slack"] = -0.05, 0.04
    runner_up = _sn_row("centered_additive_mean10", "crps", "crps", False, 0.08)
    runner_up["discounted_slack"], runner_up["veto_slack"] = -0.12, 0.02

    nominated = _nominate(rescued, runner_up)
    board = [n for n in nominated if n["source"].startswith("board")]
    # Both corners survive the q50 veto and keep their q75 ordering; board_rank stays the
    # rank-column max so rescued cells sort last in the walk order.
    assert [n["slack"] for n in board] == [0.10, 0.08]
    assert nominated[0]["board_rank"] == pytest.approx(-0.05)

    # A row negative under both prices never nominates even beside a rescued one.
    dead = _sn_row("centered_additive_eb_meanyr_k10", "crps", "crps", False, 0.30)
    dead["discounted_slack"], dead["veto_slack"] = -0.40, -0.20
    with_dead = _nominate(rescued, dead)
    assert [n["slack"] for n in with_dead if n["source"].startswith("board")] == [0.10]

    # All rows dead under the q50 veto: the cell nominates nothing, not even its seed.
    rescued_dead = dict(rescued)
    rescued_dead["discounted_slack"], rescued_dead["veto_slack"] = -0.05, -0.01
    assert _nominate(rescued_dead, dead) == []

    # Legacy boards without the veto column keep the rank-column veto.
    no_veto = _sn_row("ratio_meanyr", "crps", "nll", False, 0.10)
    no_veto["discounted_slack"] = -0.05
    assert _nominate(no_veto) == []


def test_nominees_drop_book_riding_corners_under_min_model_weight():
    """Gate 1 is free at the book, so on a weak cell the best slack is the lowest blend weight and
    the board structurally prefers corners that ride the sportsbook. ``min_model_weight`` filters
    those out of the lane without touching the ordering: ``board_rank`` still reads the board's own
    best corner, so the walk order over cells is unchanged.
    """
    book_rider = _weighted_row("ratio_meanyr", 0.30, 0.05)
    fitted = _weighted_row("centered_additive_mean10", 0.20, 0.90)

    unfiltered = _nominate(book_rider, fitted)
    assert [n["slack"] for n in unfiltered if n["source"].startswith("board")] == [0.30, 0.20]
    assert _nominate(book_rider, fitted, min_model_weight=None) == unfiltered

    filtered = _nominate(book_rider, fitted, min_model_weight=0.3)
    assert [n["slack"] for n in filtered if n["source"].startswith("board")] == [0.20]
    assert all(n["board_rank"] == 0.30 for n in filtered)


@pytest.mark.parametrize("missing", [float("nan"), pd.NA], ids=["nan", "pd.NA"])
def test_nominees_keep_bookless_and_pre_weight_corners_under_min_model_weight(missing):
    """Exactly 1.0 means the cell had no authentic validation quotes to blend against — it is not
    riding a book — and a missing weight (NaN, or the ``pd.NA`` a disk-read board backfills) is a
    board swept before the column existed. Neither is evidence of a book-rider, so the floor lets
    both through.
    """
    bookless = _weighted_row("ratio_meanyr", 0.30, 1.0)
    pre_weight = _weighted_row("centered_additive_mean10", 0.20, missing)
    kept = _nominate(bookless, pre_weight, min_model_weight=0.3)
    assert [n["slack"] for n in kept if n["source"].startswith("board")] == [0.30, 0.20]


def test_nominees_fall_back_to_seeds_when_every_positive_corner_rides_the_book(monkeypatch):
    """The floor is a lane filter, not a cell veto: a cell whose only admissible corners are
    book-riders still nominates its seeds and incumbent, which retrain and are judged at ship time.
    """
    _seed_corners(monkeypatch, _sn_controls("ratio_projvol", "nll", "nll"))
    nominated = _nominate(_weighted_row("ratio_meanyr", 0.30, 0.05), min_model_weight=0.3)
    assert [n["source"] for n in nominated] == ["seed/incumbent"]
    assert nominated[0]["board_rank"] == 0.30


def test_nominees_break_exact_rank_ties_by_fingerprint():
    """brief L5: with equal rank values the lane order is fingerprint order, not row order —
    the same board must nominate the same lane on every run.
    """
    rows = [
        _sn_row("ratio_meanyr", "nll", "crps_1se", False, 0.06, posthoc="prob_recal_platt"),
        _sn_row("ratio_meanyr", "nll", "nll", False, 0.06),
    ]
    forward = [n["corner_fingerprint"] for n in _nominate(*rows)]
    backward = [n["corner_fingerprint"] for n in _nominate(*reversed(rows))]
    assert forward == backward
    board_lane = [f for f in forward if f in {r["corner_fingerprint"] for r in rows}]
    assert board_lane == sorted(board_lane)


def test_nominees_drop_ledger_decided_corners_at_selection():
    """brief L4: a corner with a full-HPO verdict on this matrix never spends a lane slot —
    the lane fills with undecided corners the walk can actually run. A verdict from an older
    matrix does not match.
    """
    decided = _sn_row("centered_additive_mean10", "crps", "crps", False, 0.40)
    stale = _sn_row("ratio_projvol", "crps", "crps", False, 0.35)
    fresh = [
        _sn_row("ratio_meanyr", "crps", "nll", False, 0.30),
        _sn_row("centered_additive_eb_meanyr_k10", "crps", "crps", False, 0.20),
        _sn_row("ratio_meanyr", "nll", "crps", False, 0.10),
    ]
    pd.DataFrame(
        [
            {
                "strategy_corner_fingerprint": decided["corner_fingerprint"],
                "strategy_matrix_hash": _MATRIX_SHA,
            },
            {
                "strategy_corner_fingerprint": stale["corner_fingerprint"],
                "strategy_matrix_hash": "old-matrix",
            },
        ]
    ).to_csv(sweep.NOMINEE_LEDGER_PATH, index=False)

    nominated = _nominate(decided, stale, *fresh)
    board = [n for n in nominated if n["source"].startswith("board")]
    assert len(board) == mcn.CONFIRM_TOP_K
    slacks = [n["slack"] for n in board]
    assert decided["corner_fingerprint"] not in {n["corner_fingerprint"] for n in nominated}
    assert slacks[0] == 0.35  # the stale-matrix corner leads — its verdict no longer binds


def test_next_slot_promotes_a_shaping_corner_at_an_exact_tie_only():
    """brief L2: when the lane holds no PIT-reshaping corner and the diversity pick is exactly
    tied (``_SHAPING_SLOT_TIE_TOL``) with a remaining unseen-mechanism corner that reshapes the
    PIT, take the reshaping corner. A lower-ranked reshaper is never promoted, and a lane that
    already holds a reshaper keeps the plain rule.
    """

    def cand(norm, dl, bl, posthoc, slack):
        row = _sn_row(norm, dl, bl, False, slack, posthoc=posthoc)
        return {
            "strategy_slug": "SkewNormal",
            "family": "SkewNormal",
            "controls": _sn_controls(norm, dl, bl, "direct", posthoc),
            "corner_fingerprint": row["corner_fingerprint"],
        }

    platt = cand("ratio_meanyr", "nll", "crps_1se", "prob_recal_platt", 0.06)
    plain = cand("ratio_meanyr", "nll", "nll", "none", 0.06)
    shaped = cand("ratio_meanyr", "nll", "crps", "cdf_recal_isotonic", 0.06)
    pool, ranks = [platt, plain, shaped], [0.06, 0.06, 0.06]
    lane_state = ({"SkewNormal"}, {mcn._gate4_mechanism(platt)})

    # Exact tie, no reshaper in the lane: the reshaper wins the slot over the plain corner.
    pick = mcn._next_slot(pool, ranks, [1, 2], *lane_state)
    assert pick == 2

    # The reshaper ranks strictly lower: the plain rule stands.
    pick = mcn._next_slot(pool, [0.06, 0.06, 0.05], [1, 2], *lane_state)
    assert pick == 1

    # The lane already carries a reshaper: no preference fires even over a tied reshaper.
    seen = ({"SkewNormal"}, {mcn._gate4_mechanism(platt), mcn._gate4_mechanism(shaped)})
    shaped_crps = cand("ratio_meanyr", "crps", "nll", "cdf_recal_isotonic", 0.06)
    pick = mcn._next_slot([platt, plain, shaped_crps], [0.06, 0.06, 0.06], [1, 2], *seen)
    assert pick == 1


def test_walk_nominees_skips_ledger_decided_nominees_before_any_blind_run(monkeypatch, capsys):
    """brief L4 backstop: a nominee the ledger already decided on this matrix never runs blind, the
    cell's verdict comes from the nominees that did, and an all-decided cell reads SKIPPED.
    """
    monkeypatch.setattr(mcn, "decided_pairs", lambda: {("fp-2", _MATRIX_SHA)})
    nominated = _walk_stubs("board slack +0.300", "board slack +0.200")
    measured = _stub_blind(monkeypatch, {})
    attempt, _ = _attempt_spy("REVERTED", ["g4"])

    result = mc._walk_nominees({}, nominated, attempt)
    assert measured == ["board slack +0.300"]
    assert result == ("WNBA", "AST", "REVERTED", ["g4"], "1/2 board slack +0.300")
    assert "1 nominee(s) already decided on this matrix" in capsys.readouterr().out

    result = mc._walk_nominees({}, nominated[1:], _never_read)
    assert result == ("WNBA", "AST", "SKIPPED", ["prior verdict on this matrix"], "-")


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


def test_a_live_cell_ranks_its_nominees_on_the_margin_over_the_incumbent():
    """The two lanes rank on different quantities because they answer different questions.

    A withheld cell has to clear the gates outright, so it ranks on slack. A live cell has to beat
    the recipe already serving, so it ranks on ``margin_vs_incumbent`` — and the orders differ, as
    the highest-slack corner here is the one that loses to the incumbent.
    """
    loses = {**_sn_row("centered_additive_mean10", "crps", "crps", True, 0.30)}
    loses["margin_vs_incumbent"] = -0.05
    beats = {**_sn_row("centered_additive_mean10", "nll", "crps", True, 0.10)}
    beats["margin_vs_incumbent"] = 0.08

    live = _nominate(loses, beats, live=True)
    fresh = _nominate(loses, beats, live=False)

    assert [n["slack"] for n in live[:1]] == [0.10]  # ranked by margin, so the +0.08 corner leads
    assert [n["slack"] for n in fresh[:1]] == [0.30]  # ranked by slack, so the 0.30 corner leads
    assert loses["corner_fingerprint"] not in {n["corner_fingerprint"] for n in live}


def test_a_board_swept_before_margins_existed_keeps_the_old_live_ranking():
    """Margins are a property of the sweep that wrote the board, so their absence is not a NaN.

    Read back off disk a pre-margin board carries the column as ``pd.NA``; treating that as an
    unknown margin would silently stop every live cell from nominating on a board that never had a
    baseline to measure against.
    """
    legacy = pd.DataFrame([_sn_row("centered_additive_mean10", "crps", "crps", True, 0.30)]).assign(
        margin_vs_incumbent=pd.NA
    )
    meta = {"WNBA": {"AST": {"shipped": "devel"}}}

    grouped = mcn.candidates(legacy, meta)

    assert [n[0]["slack"] for n in grouped] == [0.30]


def test_a_live_cell_with_no_measured_baseline_nominates_nobody():
    """An unmeasured incumbent makes every margin unknown, and unknown is not evidence of better.

    The fresh lane is unaffected: its bar never referenced the incumbent.
    """
    row = {**_sn_row("centered_additive_mean10", "crps", "crps", True, 0.30)}
    row["margin_vs_incumbent"] = float("nan")

    assert _nominate(row, live=True) == []
    assert _nominate(row, live=False)


def test_candidates_are_one_nominee_list_per_cell():
    """``candidates`` groups the board by cell; a cell whose every corner is unconfirmable drops out
    rather than contributing an empty walk.
    """
    board = pd.DataFrame(
        [
            _sn_row("centered_additive_mean10", "crps", "crps", False, 0.25),
            _zinb_row("hurdle", "pit_ks", "crps", False, 0.10),
            _mixture_row("NBA", "PTS", 0.90),
        ]
    )
    grouped = mcn.candidates(board, _withheld_meta(board))
    assert [(n[0]["league"], n[0]["market"]) for n in grouped] == [
        ("WNBA", "AST"),
        ("MLB", "pitcher strikeouts"),
    ]
    assert [len(n) for n in grouped] == [1, 1]


def test_candidates_walk_the_strongest_cells_first():
    """Cells order by best board rank descending, so a deadline cut lands on the weakest tail."""
    board = pd.DataFrame(
        [
            _sn_row("centered_additive_mean10", "crps", "crps", False, 0.10),
            _zinb_row("hurdle", "pit_ks", "crps", False, 0.50),
        ]
    )
    grouped = mcn.candidates(board, _withheld_meta(board))
    assert [(n[0]["league"], n[0]["market"]) for n in grouped] == [
        ("MLB", "pitcher strikeouts"),
        ("WNBA", "AST"),
    ]
    assert [n[0]["board_rank"] for n in grouped] == [0.50, 0.10]


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
    """`_run_meditate` reports subprocess success/failure only — it does not read a verdict.

    A signal death and an ordinary non-zero exit must not report the same reason: the confirm
    walk's verdict line is the only place a native abort is ever named.
    """
    cand = {"league": "NBA", "market": "PTS"}

    def run():
        return mcr._run_meditate(cand, [], log_path=tmp_path / "meditate.log", label="run")

    monkeypatch.setattr(mcr, "_run_meditate_with_lock_retry", lambda cmd, path, timeout: None)
    assert run() == ""

    def boom(*a, **k):
        raise mcr.subprocess.CalledProcessError(1, "meditate")

    monkeypatch.setattr(mcr, "_run_meditate_with_lock_retry", boom)
    assert run() == "exit 1"

    def abort(*a, **k):
        raise mcr.subprocess.CalledProcessError(-signal.SIGABRT, "meditate")

    monkeypatch.setattr(mcr, "_run_meditate_with_lock_retry", abort)
    assert run() == "native abort (SIGABRT)"
    assert cand["elapsed_s"] >= 0  # timed even when the run fails


def test_run_blind_forwards_every_control_and_never_writes_stat_meta(monkeypatch, tmp_path):
    """A blind run reads nothing from a persisted edit: every control rides as its flag, beside the
    flags ``--artifact-output`` requires, and stat_meta.json is never written for it.
    """
    monkeypatch.setattr(mcr, "_CONFIRM_LOG_ROOT", tmp_path)
    monkeypatch.setattr(
        mcr, "atomic_write_meta", lambda meta: pytest.fail("a blind run wrote stat_meta")
    )
    commands, scored = [], []
    monkeypatch.setattr(
        mcr, "_run_meditate_with_lock_retry", lambda cmd, path, timeout: commands.append(cmd)
    )
    score = _scorer()
    monkeypatch.setattr(
        mcr, "_score_corner", lambda *args: scored.append(args[-2:]) or score(*args)
    )
    controls = _sn_controls("ratio_meanyr", "crps", "nll", posthoc="prob_recal_platt")
    ordinary = _nominate(_signed_row("NBA", "PTS", "SkewNormal", controls, True, 0.2))[0]

    run = mcr._run_blind(ordinary, "loss")

    command = commands[0]
    slug = mc.market_file_slug("NBA", "PTS")
    sandbox = tmp_path / "blind" / slug / f"{ordinary['corner_fingerprint']}__loss"
    assert command[command.index("--artifact-output") + 1] == str(sandbox)
    assert command[command.index("--dependency-namespace") + 1] == DEPENDENCY_NAMESPACE
    for flag in ("--force", "--frozen-matrix-dir", "--holdout-blind", "--bypass-withholding"):
        assert flag in command
    assert command[command.index("--posthoc") + 1] == "prob_recal_platt"
    assert command[command.index("--target-normalization") + 1] == "ratio_meanyr"
    assert command[command.index("--hpo-selection") + 1] == "loss"
    # The board's scorer reads this run's own dump and pickle, the pair --params-from reuses.
    assert run.dump == sandbox / "test_sets" / f"{slug}.csv"
    assert run.model == sandbox / f"{slug}.mdl"
    assert scored == [(run.dump, run.model)]
    assert run.failed == []

    structural = _nominate(_structural_row(AFFINE_STRATEGY, "rushing yards"))[0]
    mcr._run_blind(structural, "loss")
    command = commands[-1]
    # A structural spec pins hpo_selection among its controls, so it rides once, as a control.
    assert command.count("--hpo-selection") == 1
    assert command[command.index("--posthoc") + 1] == AFFINE_STRATEGY
    assert command[command.index("--stabilization") + 1] == "None"


def test_run_blind_never_scores_a_dump_an_earlier_walk_left(monkeypatch, tmp_path):
    """A sandbox is keyed by corner, so an earlier walk's dump sits where this run writes; a run that
    exits clean without writing one must fail rather than be scored on the stale file.
    """
    monkeypatch.setattr(mcr, "_CONFIRM_LOG_ROOT", tmp_path)
    monkeypatch.setattr(mcr, "_run_meditate_with_lock_retry", lambda cmd, path, timeout: None)
    score = _scorer()

    def score_if_written(league, market, corner, spec, csv_path, mdl_path):
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        return score(league, market, corner, spec, csv_path, mdl_path)

    monkeypatch.setattr(mcr, "_score_corner", score_if_written)
    cand = _nominate(_sn_row("ratio_meanyr", "crps", "nll", True, 0.2))[0]
    stale = mcr._run_blind(cand, "loss").dump
    stale.parent.mkdir(parents=True)
    stale.write_text("an earlier walk's dump")

    assert mcr._run_blind(cand, "loss").failed == ["artifact identity"]


def test_run_canonical_refits_from_the_blind_pickle_and_reads_persisted_controls(
    monkeypatch, tmp_path
):
    """The canonical refit trains the ranked booster (``--params-from`` its blind pickle) on the
    pinned matrix, and reads every persisted control back from the just-written stat_meta.
    """
    commands = []
    monkeypatch.setattr(
        mcr, "_run_meditate_with_lock_retry", lambda cmd, path, timeout: commands.append(cmd)
    )
    monkeypatch.setattr(mcr, "_cell_row", lambda league, market, columns: None)
    controls = _sn_controls("ratio_meanyr", "crps", "nll", posthoc="prob_recal_platt")
    ordinary = _nominate(_signed_row("NBA", "PTS", "SkewNormal", controls, True, 0.2))[0]
    pickle = tmp_path / "blind" / "NBA_PTS.mdl"

    assert mcr.run_canonical("NBA", "PTS", {**ordinary, "params_from": pickle}) == ("", False)
    command = commands[0]
    assert command[command.index("--params-from") + 1] == str(pickle)
    assert "--force" in command and "--frozen-matrix-dir" in command
    assert "--holdout-blind" not in command and "--artifact-output" not in command
    # Every base-family control is persisted, so the refit forces none of them.
    assert "--posthoc" not in command and "--target-normalization" not in command

    structural = _nominate(_structural_row(AFFINE_STRATEGY, "rushing yards"))[0]
    mcr.run_canonical("NFL", "rushing yards", {**structural, "params_from": pickle})
    command = commands[-1]
    assert "--structural-strategy" not in command
    assert command[command.index("--stabilization") + 1] == "None"
    # ``posthoc`` carries the method slug and is persisted, so stat_meta selects the lane.
    assert "--posthoc" not in command


def test_pin_cell_matrix_freezes_the_frame_every_nominee_trains_on(monkeypatch, tmp_path):
    """The walk copies the cached matrix + a manifest the production validator accepts, and every
    blind run of the cell, and its canonical refit, trains on that same frozen dir.
    """
    monkeypatch.setattr(mcr, "_CONFIRM_LOG_ROOT", tmp_path / "confirm")
    source = tmp_path / "WNBA_AST.parquet"
    pd.DataFrame({"Result": [3.0, 5.0], "MeanYr": [4.1, 4.4]}).to_parquet(source)
    monkeypatch.setattr(mcr, "_training_matrix_path", lambda league, market: source)

    frozen_dir = _REAL_PIN_CELL_MATRIX("WNBA", "AST")

    frozen = frozen_dir / "WNBA_AST.parquet"
    assert frozen.read_bytes() == source.read_bytes()
    validate_matrix_manifest(frozen, pd.read_parquet(frozen))
    manifest = json.loads(frozen.with_suffix(".manifest.json").read_text())
    assert set(manifest) == {
        "builder_version",
        "schema_version",
        "row_count",
        "feature_schema",
        "matrix_sha256",
    }

    commands = []
    monkeypatch.setattr(
        mcr, "_run_meditate_with_lock_retry", lambda cmd, path, timeout: commands.append(cmd)
    )
    monkeypatch.setattr(mcr, "_score_corner", _scorer())
    monkeypatch.setattr(mcr, "_cell_row", lambda league, market, columns: None)
    first, second = (
        _nominate(row)[0]
        for row in (
            _sn_row("ratio_meanyr", "crps", "nll", True, 0.2),
            _sn_row("centered_additive_mean10", "crps", "crps", True, 0.1),
        )
    )
    mcr._run_blind(first, "loss")
    mcr._run_blind(second, "loss")
    mcr.run_canonical("WNBA", "AST", {**second, "params_from": tmp_path / "blind.mdl"})
    dirs = [cmd[cmd.index("--frozen-matrix-dir") + 1] for cmd in commands]
    assert dirs == [str(frozen_dir)] * 3
    assert all("--force" in cmd for cmd in commands)  # frozen input still needs the skip override


def test_pin_cell_matrix_fails_loud_without_a_cached_matrix(monkeypatch, tmp_path):
    """A swept cell always has a cached parquet; a missing one is a broken invariant, not a skip."""
    monkeypatch.setattr(mcr, "_CONFIRM_LOG_ROOT", tmp_path)
    monkeypatch.setattr(
        mcr, "_training_matrix_path", lambda league, market: tmp_path / "absent.parquet"
    )
    with pytest.raises(FileNotFoundError):
        _REAL_PIN_CELL_MATRIX("WNBA", "AST")


def test_walk_nominees_pins_the_matrix_once_per_cell(monkeypatch):
    pinned = []
    monkeypatch.setattr(
        mcr, "pin_cell_matrix", lambda league, market: pinned.append((league, market))
    )
    _stub_blind(monkeypatch, {})
    nominated = _walk_stubs("board slack +0.300", "board slack +0.200")
    mc._walk_nominees({}, nominated, lambda meta, cand: ("WNBA", "AST", "REVERTED", ["g4"]))
    assert pinned == [("WNBA", "AST")]


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
    monkeypatch.setattr(mcr.pd, "read_parquet", lambda *args, **kwargs: stats)

    assert mcr._ship_from_model_stats("NFL", "rushing yards", expected) is True
    stats.loc[0, "strategy_controls_json"] = "{}"
    assert mcr._ship_from_model_stats("NFL", "rushing yards", expected) is False
    stats.loc[0, "strategy_controls_json"] = expected.controls_json
    wrong = build_artifact_identity(
        "SkewNormal",
        "NFL",
        "rushing yards",
        _sn_controls("ratio_meanyr", "crps", "nll"),
        matrix_hash=_MATRIX_SHA,
    )
    assert mcr._ship_from_model_stats("NFL", "rushing yards", wrong) is False


def test_a_gate_miss_is_reported_as_the_gate_not_a_missing_artifact(monkeypatch):
    """Serve-iff-ship prunes the pickle of a cell that failed a gate, so the artifact leg fails too.

    Checking artifact identity first reported a plain Gate-4 miss as a missing serving artifact --
    the real NBA FGA confirm did exactly that.
    """
    candidate = _nominate(_sn_row("ratio_meanyr", "crps", "nll", True, 0.2))[0]
    monkeypatch.setattr(mcr, "run_canonical", lambda lg, mkt, cand: ("", False))
    monkeypatch.setattr(mcr, "_retrained_matrix_hash", lambda lg, mkt: _MATRIX_SHA)
    monkeypatch.setattr(mcr, "_failed_gates_after", lambda lg, mkt: ["g4"])
    monkeypatch.setattr(
        mcr,
        "_produced_artifacts_match",
        lambda *args: pytest.fail("a gate miss must not be diagnosed through the pruned pickle"),
    )

    assert mcr.confirm_meditate("WNBA", "AST", candidate) == ["g4"]


def test_confirm_meditate_prepends_diverged_to_the_failed_gates(monkeypatch):
    """A diverged fit (dispersion calibrator on its floor) is named ahead of the gates it fails, so
    the report sends triage to the fit rather than the anonymous gate list — but a diverged flag
    alone never blocks a ship (the gates already decide).
    """
    candidate = _nominate(_sn_row("ratio_meanyr", "crps", "nll", True, 0.2))[0]
    monkeypatch.setattr(mcr, "run_canonical", lambda lg, mkt, cand: ("", True))
    monkeypatch.setattr(mcr, "_retrained_matrix_hash", lambda lg, mkt: _MATRIX_SHA)
    monkeypatch.setattr(mcr, "_failed_gates_after", lambda lg, mkt: ["g1", "g4"])

    assert mcr.confirm_meditate("WNBA", "AST", candidate) == ["diverged", "g1", "g4"]

    monkeypatch.setattr(mcr, "_failed_gates_after", lambda lg, mkt: [])
    monkeypatch.setattr(mcr, "_produced_artifacts_match", lambda *args: True)
    monkeypatch.setattr(mcr, "_ship_from_model_stats", lambda *args: True)
    assert mcr.confirm_meditate("WNBA", "AST", candidate) == []


def test_record_ledger_row_marks_dispersion_floor_as_diverged(monkeypatch, tmp_path):
    """dispersion_cal at/below the floor margin classifies the fit diverged — returned to the
    caller and recorded as the ledger's ``diverged`` column.
    """
    ledger = tmp_path / "ledger.csv"
    monkeypatch.setattr(sweep, "NOMINEE_LEDGER_PATH", ledger)
    candidate = {"strategy_slug": "SkewNormal", "source": "board slack +0.300"}

    flags = [
        mcr._record_ledger_row(
            candidate, {"league": "WNBA", "market": "AST", "dispersion_cal": cal}, EVAL_SPLIT_TEST
        )
        for cal in (0.1, 0.1005, 0.2)
    ]

    assert flags == [True, True, False]
    assert pd.read_csv(ledger)["diverged"].tolist() == [True, True, False]


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
    monkeypatch.setattr(mcr.pd, "read_parquet", lambda *args, **kwargs: stats)
    monkeypatch.setattr(mcr, "run_canonical", lambda lg, mkt, cand: ("", False))
    monkeypatch.setattr(mcr, "_failed_gates_after", lambda lg, mkt: [])

    seen = {}

    def artifacts_match(league, market, cand, expected):
        seen["matrix_hash"] = expected.matrix_hash
        return True

    monkeypatch.setattr(mcr, "_produced_artifacts_match", artifacts_match)

    assert mcr.confirm_meditate(candidate["league"], candidate["market"], candidate) == []
    # The pickle leg is held to the retrain's matrix too, so a stale artifact still fails there.
    assert seen["matrix_hash"] == retrained_hash


def test_withheld_confirm_fails_closed_before_model_stats_when_artifacts_do_not_match(monkeypatch):
    candidate = _nominate(_sn_row("ratio_meanyr", "crps", "nll", True, 0.2))[0]
    monkeypatch.setattr(mcr, "run_canonical", lambda lg, mkt, cand: ("", False))
    monkeypatch.setattr(mcr, "_retrained_matrix_hash", lambda lg, mkt: _MATRIX_SHA)
    monkeypatch.setattr(mcr, "_failed_gates_after", lambda lg, mkt: [])
    monkeypatch.setattr(mcr, "_produced_artifacts_match", lambda *args: False)
    monkeypatch.setattr(
        mcr,
        "_ship_from_model_stats",
        lambda *args: pytest.fail("unverified artifacts must never reach the ship verdict"),
    )
    assert mcr.confirm_meditate("WNBA", "AST", candidate) == ["artifact identity"]


# --- persist / confirm / revert --------------------------------------------------------------


def test_confirm_one_pass_keeps_devel(monkeypatch, tmp_path):
    meta = {"WNBA": {"AST": _sn_original()}}
    writes = _fake_meta_disk(monkeypatch, tmp_path)
    monkeypatch.setattr(mcr, "snapshot_cell", lambda lg, mkt: tmp_path / "snapshot")
    monkeypatch.setattr(mcr, "confirm_meditate", lambda lg, mkt, candidate: [])
    monkeypatch.setattr(mcr, "served_weight", lambda path: 0.4)
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
    monkeypatch.setattr(mcr, "atomic_write_meta", lambda m: None)
    monkeypatch.setattr(mcr, "snapshot_cell", lambda lg, mkt: mc.pathlib.Path("/tmp/snapshot"))
    restored = []

    def restore(lg, mkt, backup, current_meta, prior):
        restored.append((lg, mkt, backup))
        current_meta[lg][mkt] = prior

    monkeypatch.setattr(mcr, "restore_cell", restore)
    monkeypatch.setattr(mcr, "confirm_meditate", lambda lg, mkt, candidate: ["g4"])
    pruned = []
    monkeypatch.setattr(mc, "prune_model_pickle", lambda lg, mkt: pruned.append((lg, mkt)) or True)
    monkeypatch.setattr(mcr, "_failed_gates_after", lambda lg, mkt: ["g4"])

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
        mcr,
        "atomic_write_meta",
        lambda current: states.append(json.loads(json.dumps(current))),
    )
    monkeypatch.setattr(mcr, "snapshot_cell", lambda lg, mkt: tmp_path / "snapshot")

    def restore(lg, mkt, backup, current_meta, prior):
        current_meta[lg][mkt] = prior
        mcr.atomic_write_meta(current_meta)

    monkeypatch.setattr(mcr, "restore_cell", restore)
    monkeypatch.setattr(
        mcr,
        "confirm_meditate",
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


def test_blind_retry_runs_calibrated_and_both_rows_reach_the_ledger(monkeypatch, tmp_path):
    """A g4-only near miss reruns blind under calibrated selection. The pin rides the retry's own
    edits and its own sandbox, never stat_meta, and both runs land in the ledger.
    """
    monkeypatch.setattr(mcr, "_retry_calibrated_wanted", _REAL_RETRY_WANTED)
    monkeypatch.setattr(
        mcr, "atomic_write_meta", lambda meta: pytest.fail("a blind run wrote stat_meta")
    )
    commands = []
    monkeypatch.setattr(
        mcr, "_run_meditate_with_lock_retry", lambda cmd, path, timeout: commands.append(cmd)
    )
    monkeypatch.setattr(
        mcr, "_score_corner", _in_turn(_scorer(failed=("g4",)), _scorer(slack=0.02))
    )
    cand = _nominate(_sn_row("ratio_meanyr", "crps", "nll", True, 0.2))[0]
    meta = {"WNBA": {"AST": _sn_original()}}

    first, retry = mcr.measure_blind(meta, cand)

    assert [cmd[cmd.index("--hpo-selection") + 1] for cmd in commands] == ["loss", "calibrated"]
    assert first.failed == ["g4"] and retry.failed == []
    assert retry.candidate["edits"]["hpo_selection"] == "calibrated"
    assert retry.candidate["source"] == "board slack +0.200 +calibrated-retry"
    assert retry.model != first.model
    assert "hpo_selection" not in cand["edits"]  # the nominee itself is untouched
    assert meta == {"WNBA": {"AST": _sn_original()}}
    ledger = pd.read_csv(sweep.NOMINEE_LEDGER_PATH)
    assert ledger["hpo_selection"].tolist() == ["loss", "calibrated"]
    assert ledger["eval_split"].tolist() == [EVAL_SPLIT_CROSSFIT] * 2


def test_a_ranked_retry_ships_and_persists_its_pin(monkeypatch, tmp_path):
    """When the calibrated retry is the run that ranks, the walk refits that recipe from the
    retry's own pickle, and the pin reaches stat_meta with the cell's other edits.
    """
    meta = {"WNBA": {"AST": _sn_original()}}
    writes = _fake_meta_disk(monkeypatch, tmp_path)
    monkeypatch.setattr(mcr, "snapshot_cell", lambda lg, mkt: tmp_path / "snapshot")
    cand = _nominate(_sn_row("ratio_meanyr", "crps", "nll", True, 0.2))[0]
    retry = _calibrated(cand)
    monkeypatch.setattr(
        mcr, "measure_blind", lambda m, c: [_blind_run(c, failed=["g4"]), _blind_run(retry)]
    )
    refits = []
    monkeypatch.setattr(mcr, "confirm_meditate", lambda lg, mkt, c: refits.append(c) or [])
    monkeypatch.setattr(mcr, "served_weight", lambda path: 0.4)
    monkeypatch.setattr(
        mc, "prune_model_pickle", lambda lg, mkt: pytest.fail("a shipped retry must not prune")
    )

    result = mc._walk_nominees(meta, [cand], mc._confirm_one)

    assert result == ("WNBA", "AST", "SHIPPED", [], "1/1 board slack +0.200")
    assert refits[0]["params_from"] == _blind_run(retry).model
    assert meta["WNBA"]["AST"]["hpo_selection"] == "calibrated"
    on_disk = json.loads((tmp_path / "stat_meta.json").read_text())
    assert on_disk["WNBA"]["AST"]["hpo_selection"] == "calibrated"
    assert len(writes) == 1  # the pick's one persist; no blind run wrote anything


def test_a_ranked_retry_whose_refit_fails_reverts_its_pin(monkeypatch, tmp_path):
    """A retry that ranks but fails its ship holdout reverts like any loser — the pin leaves
    stat_meta with the rest of its edits, and the pickle is pruned.
    """
    original = _sn_original()
    meta = {"WNBA": {"AST": dict(original)}}
    _fake_meta_disk(monkeypatch, tmp_path)
    monkeypatch.setattr(mcr, "snapshot_cell", lambda lg, mkt: tmp_path / "snapshot")
    monkeypatch.setattr(mcr, "_cell_artifacts", lambda lg, mkt: [])
    cand = _nominate(_sn_row("ratio_meanyr", "crps", "nll", True, 0.2))[0]
    monkeypatch.setattr(
        mcr,
        "measure_blind",
        lambda m, c: [_blind_run(c, failed=["g4"]), _blind_run(_calibrated(c))],
    )
    monkeypatch.setattr(mcr, "confirm_meditate", lambda lg, mkt, c: ["g4"])
    pruned = []
    monkeypatch.setattr(mc, "prune_model_pickle", lambda lg, mkt: pruned.append((lg, mkt)) or True)

    result = mc._walk_nominees(meta, [cand], mc._confirm_one)

    assert result == ("WNBA", "AST", "REVERTED", ["g4"], "1/1 board slack +0.200")
    assert meta["WNBA"]["AST"] == original  # the pin reverted with everything else
    assert pruned == [("WNBA", "AST")]


def test_confirm_one_reverts_a_gate_passing_cell_below_min_model_weight(monkeypatch, tmp_path):
    """The walk held the pick to the floor on its blind pickle, but the canonical refit fits its own
    blend, so the floor is checked again on the pickle the cell would actually serve. A 6/6 cell
    that lands back on the book reverts like any other loser.
    """
    original = _sn_original()
    meta = {"WNBA": {"AST": dict(original)}}
    _fake_meta_disk(monkeypatch, tmp_path)
    monkeypatch.setattr(mcr, "snapshot_cell", lambda lg, mkt: tmp_path / "snapshot")
    monkeypatch.setattr(mcr, "_cell_artifacts", lambda lg, mkt: [])
    monkeypatch.setattr(mcr, "confirm_meditate", lambda lg, mkt, cand: [])
    monkeypatch.setattr(mcr, "served_weight", lambda path: 0.1)
    pruned = []
    monkeypatch.setattr(mc, "prune_model_pickle", lambda lg, mkt: pruned.append((lg, mkt)))

    cand = {"league": "WNBA", "market": "AST", "edits": {"target_normalization": "ratio_meanyr"}}
    assert mc._confirm_one(meta, cand, min_model_weight=0.3) == (
        "WNBA",
        "AST",
        "REVERTED",
        ["w=0.100<0.3"],
    )
    assert meta["WNBA"]["AST"] == original
    assert pruned == [("WNBA", "AST")]
    # Without the knob the same 6/6 cell still ships — the floor is opt-in.
    assert mc._confirm_one(meta, cand) == ("WNBA", "AST", "SHIPPED", [])


# --- nominee walk ------------------------------------------------------------------------------


def _walk_stubs(*sources):
    return [
        {
            "league": "WNBA",
            "market": "AST",
            "source": source,
            "corner_fingerprint": f"fp-{n}",
            "matrix_hash": _MATRIX_SHA,
        }
        for n, source in enumerate(sources, start=1)
    ]


def _blind_run(cand, *, slack=0.1, weight=0.5, failed=()):
    """A scored blind run of ``cand``, in the sandbox its fingerprint and selection key."""
    selection = cand.get("edits", {}).get("hpo_selection", "loss")
    sandbox = mc.pathlib.Path("/sandbox") / f"{cand['corner_fingerprint']}__{selection}"
    row = {"slack": slack, "model_weight": weight}
    return mcr.BlindRun(cand, sandbox / "dump.csv", sandbox / "model.mdl", list(failed), row)


def _calibrated(cand):
    """``cand`` as the walk's calibrated retry runs it."""
    edits = {**cand["edits"], "hpo_selection": "calibrated"}
    return {**cand, "source": f"{cand['source']} +calibrated-retry", "edits": edits}


def _stub_blind(monkeypatch, runs):
    """Serve each nominee one blind run (``runs`` maps source → ``_blind_run`` kwargs); returns the
    order the walk measured them in.
    """
    measured = []

    def measure(meta, cand):
        measured.append(cand["source"])
        return [_blind_run(cand, **runs.get(cand["source"], {}))]

    monkeypatch.setattr(mcr, "measure_blind", measure)
    return measured


def _attempt_spy(outcome="SHIPPED", failed=()):
    attempted = []

    def attempt(meta, cand):
        attempted.append(cand)
        return (cand["league"], cand["market"], outcome, list(failed))

    return attempt, attempted


def _never_read(meta, cand):
    pytest.fail("the ship holdout must stay unread")


def _in_turn(*functions):
    """Each call goes to the next of ``functions``."""
    queue = iter(functions)

    def call(*args):
        return next(queue)(*args)

    return call


def _scorer(*, failed=(), slack=0.1, weight=0.5):
    """A stand-in for the board's scorer: the gate row it would read off a blind dump."""

    def score(league, market, corner, spec, csv_path, mdl_path):
        return {
            **corner,
            "slack": slack,
            "ships": not failed,
            **{f"{gate}_pass": gate not in failed for gate in sweep._GATES},
            "g1_brier_diff_ci_hi": 0.003,
            "g4_pit_ks": 0.055 if "g4" in failed else 0.04,
            "g4_pit_ks_max": 0.05,
            "model_weight": weight,
            "dispersion_cal": 1.0,
            "n": 373,
            "matrix_hash": _MATRIX_SHA,
            "split_fingerprint": None,
            "eval_split": EVAL_SPLIT_CROSSFIT,
        }

    return score


def test_walk_nominees_refits_the_top_blind_passer_from_its_sandbox_pickle(monkeypatch):
    """Every nominee runs blind; the passer with the most blind slack — not the first nominee, and
    not a higher-slack nominee that failed a gate — is the one refit, from its own sandbox pickle.
    """
    nominated = _walk_stubs("board slack +0.300", "board slack +0.200", "seed/incumbent")
    measured = _stub_blind(
        monkeypatch,
        {
            "board slack +0.300": {"slack": 0.30, "failed": ["g4"]},
            "board slack +0.200": {"slack": 0.05},
            "seed/incumbent": {"slack": 0.12},
        },
    )
    attempt, attempted = _attempt_spy()

    result = mc._walk_nominees({}, nominated, attempt)

    assert measured == [n["source"] for n in nominated]
    assert [cand["source"] for cand in attempted] == ["seed/incumbent"]
    assert attempted[0]["params_from"] == _blind_run(nominated[2]).model
    assert result == ("WNBA", "AST", "SHIPPED", [], "3/3 seed/incumbent")


def test_walk_nominees_reads_the_test_half_once_and_stops_on_its_failure(monkeypatch):
    """A refit that fails its ship holdout ends the cell: reading the runner-up there too would
    make the cell's verdict a selection on the test half again.
    """
    nominated = _walk_stubs("board slack +0.300", "board slack +0.200")
    _stub_blind(
        monkeypatch, {"board slack +0.300": {"slack": 0.2}, "board slack +0.200": {"slack": 0.1}}
    )
    attempt, attempted = _attempt_spy("REVERTED", ["g4"])

    assert mc._walk_nominees({}, nominated, attempt) == (
        "WNBA",
        "AST",
        "REVERTED",
        ["g4"],
        "1/2 board slack +0.300",
    )
    assert len(attempted) == 1


def test_walk_nominees_holds_a_cell_with_no_blind_passer(monkeypatch):
    nominated = _walk_stubs("board slack +0.300", "seed/incumbent")
    _stub_blind(monkeypatch, {n["source"]: {"failed": ["g4"]} for n in nominated})
    assert mc._walk_nominees({}, nominated, _never_read) == (
        "WNBA",
        "AST",
        "HELD",
        ["no blind pass"],
        "-",
    )


def test_walk_nominees_holds_the_weight_floor_on_the_blind_pickle(monkeypatch):
    """The floor reads each blind run's own fitted blend weight, so a book-riding passer cannot take
    the refit from a model-driven one with less slack; unset, slack alone decides.
    """
    nominated = _walk_stubs("board slack +0.300", "board slack +0.200")
    _stub_blind(
        monkeypatch,
        {
            "board slack +0.300": {"slack": 0.3, "weight": 0.1},
            "board slack +0.200": {"slack": 0.2, "weight": 0.5},
        },
    )
    attempt, attempted = _attempt_spy()
    mc._walk_nominees({}, nominated, attempt, min_model_weight=0.3)
    mc._walk_nominees({}, nominated, attempt)
    assert [c["source"] for c in attempted] == ["board slack +0.200", "board slack +0.300"]


def _live_verdicts(monkeypatch, passing):
    """S1/S2/S3 on blind dumps: pass only for contenders whose sandbox is in ``passing``; returns
    the (baseline, contender) sandbox pairs compared.
    """
    monkeypatch.setattr(mc, "load_test_set", lambda path, col: path)
    compared = []

    def verdict(baseline, candidate, col, *, league, market):
        compared.append((baseline.parent.name, candidate.parent.name))
        ships = candidate.parent.name in passing
        return _verdict(ship=ships, s2=ships)

    monkeypatch.setattr(mc, "supersede_verdict", verdict)
    return compared


def test_live_walk_ranks_contenders_against_the_incumbent_blind_run(monkeypatch):
    """A live cell's contenders meet the incumbent's own blind run on S1/S2/S3, the incumbent's
    baseline never contends for the refit itself, and a failing contender's higher slack is moot.
    """
    nominated = _walk_stubs("board slack +0.300", "board slack +0.200", "seed/incumbent")
    monkeypatch.setattr(mc, "_incumbent_fingerprint", lambda league, market: "fp-3")
    _stub_blind(
        monkeypatch,
        {
            "board slack +0.300": {"slack": 0.3},
            "board slack +0.200": {"slack": 0.2},
            "seed/incumbent": {"slack": 0.4},
        },
    )
    compared = _live_verdicts(monkeypatch, {"fp-2__loss"})
    attempt, attempted = _attempt_spy("SUPERSEDED")

    result = mc._walk_nominees({}, nominated, attempt, live=True)

    assert compared == [("fp-3__loss", "fp-1__loss"), ("fp-3__loss", "fp-2__loss")]
    assert [c["source"] for c in attempted] == ["board slack +0.200"]
    assert result == ("WNBA", "AST", "SUPERSEDED", [], "2/3 board slack +0.200")


def test_live_walk_lets_the_incumbents_calibrated_retry_contend(monkeypatch):
    """The incumbent's first blind run measures the recipe the cell serves and stays the baseline;
    its calibrated retry is a different recipe, and contends like any other.
    """
    nominated = _walk_stubs("board slack +0.300", "seed/incumbent")
    monkeypatch.setattr(mc, "_incumbent_fingerprint", lambda league, market: "fp-2")
    incumbent = nominated[1]
    runs = {
        "board slack +0.300": [_blind_run(nominated[0], failed=["g4"])],
        "seed/incumbent": [
            _blind_run(incumbent, failed=["g4"]),
            _blind_run(_calibrated({**incumbent, "edits": {}})),
        ],
    }
    monkeypatch.setattr(mcr, "measure_blind", lambda meta, cand: runs[cand["source"]])
    compared = _live_verdicts(monkeypatch, {"fp-2__calibrated"})
    attempt, attempted = _attempt_spy("SUPERSEDED")

    result = mc._walk_nominees({}, nominated, attempt, live=True)

    assert compared == [("fp-2__loss", "fp-2__calibrated")]
    assert attempted[0]["edits"] == {"hpo_selection": "calibrated"}
    assert result[2:] == ("SUPERSEDED", [], "2/2 seed/incumbent")


def test_live_walk_holds_a_cell_without_a_scored_incumbent_baseline(monkeypatch):
    """No baseline, no ranking: a live cell whose incumbent recipe is not among its undecided
    nominees runs nothing, and one whose incumbent's own blind run never scored reads no test half.
    """
    nominated = _walk_stubs("board slack +0.300", "seed/incumbent")
    monkeypatch.setattr(mc, "_incumbent_fingerprint", lambda league, market: "fp-gone")
    monkeypatch.setattr(mcr, "measure_blind", lambda meta, cand: pytest.fail("nothing may run"))
    assert mc._walk_nominees({}, nominated, _never_read, live=True) == (
        "WNBA",
        "AST",
        "HELD",
        ["no incumbent baseline"],
        "-",
    )

    monkeypatch.setattr(mc, "_incumbent_fingerprint", lambda league, market: "fp-2")
    unscored = mcr.BlindRun(nominated[1], None, None, ["retrain exit 1"], None)
    monkeypatch.setattr(
        mcr,
        "measure_blind",
        lambda meta, cand: [unscored if cand is nominated[1] else _blind_run(cand, slack=0.3)],
    )
    assert mc._walk_nominees({}, nominated, _never_read, live=True) == (
        "WNBA",
        "AST",
        "HELD",
        ["incumbent baseline", "retrain exit 1"],
        "-",
    )


def test_live_walk_runs_the_incumbent_baseline_first_and_ranks_in_nomination_order(monkeypatch):
    """The incumbent's blind run is the baseline every contender meets, so it trains first; ranking
    still reads the nominees in nomination order, so a slack tie goes to the earlier nominee — here
    the board corner, over the incumbent's calibrated retry that ran before it.
    """
    nominated = _walk_stubs("board slack +0.300", "seed/incumbent")
    monkeypatch.setattr(mc, "_incumbent_fingerprint", lambda league, market: "fp-2")
    incumbent = {**nominated[1], "edits": {}}
    runs = {
        "board slack +0.300": [_blind_run(nominated[0], slack=0.2)],
        "seed/incumbent": [
            _blind_run(incumbent, failed=["g4"]),
            _blind_run(_calibrated(incumbent), slack=0.2),
        ],
    }
    measured = []
    monkeypatch.setattr(
        mcr,
        "measure_blind",
        lambda meta, cand: measured.append(cand["source"]) or runs[cand["source"]],
    )
    compared = _live_verdicts(monkeypatch, {"fp-1__loss", "fp-2__calibrated"})
    attempt, _ = _attempt_spy("SUPERSEDED")

    result = mc._walk_nominees({}, nominated, attempt, live=True)

    assert measured == ["seed/incumbent", "board slack +0.300"]
    assert compared == [("fp-2__loss", "fp-1__loss"), ("fp-2__loss", "fp-2__calibrated")]
    assert result == ("WNBA", "AST", "SUPERSEDED", [], "1/2 board slack +0.300")


def test_a_broken_incumbent_baseline_holds_the_cell_before_its_other_nominees_train(
    monkeypatch, capsys
):
    """Every contender is read against the incumbent's blind run, so a baseline that never scores
    holds the cell at once: the nominees nominated ahead of it never start their full-HPO runs.
    """
    nominated = _walk_stubs("board slack +0.300", "board slack +0.200", "seed/incumbent")
    monkeypatch.setattr(mc, "_incumbent_fingerprint", lambda league, market: "fp-3")
    unscored = mcr.BlindRun(nominated[2], None, None, ["retrain exit 1"], None)
    measured = []

    def measure(meta, cand):
        measured.append(cand["source"])
        return [unscored] if cand["source"] == "seed/incumbent" else [_blind_run(cand)]

    monkeypatch.setattr(mcr, "measure_blind", measure)

    assert mc._walk_nominees({}, nominated, _never_read, live=True) == (
        "WNBA",
        "AST",
        "HELD",
        ["incumbent baseline", "retrain exit 1"],
        "-",
    )
    assert measured == ["seed/incumbent"]
    assert "2 other nominee(s) not run" in capsys.readouterr().out


def test_walk_persists_only_its_pick_and_no_blind_run_writes_stat_meta(monkeypatch, tmp_path):
    """Every nominee trains blind before anything touches stat_meta.json; the one write is the
    pick's persist, so a loser's edits never reach the file at all.
    """
    original = _sn_original()
    meta = {"WNBA": {"AST": dict(original)}}
    writes = _fake_meta_disk(monkeypatch, tmp_path)
    monkeypatch.setattr(mcr, "snapshot_cell", lambda lg, mkt: tmp_path / "snapshot")
    runs = []
    monkeypatch.setattr(
        mcr, "_run_meditate_with_lock_retry", lambda cmd, path, timeout: runs.append(len(writes))
    )
    monkeypatch.setattr(mcr, "_score_corner", _in_turn(_scorer(slack=0.05), _scorer(slack=0.25)))
    monkeypatch.setattr(mcr, "confirm_meditate", lambda lg, mkt, cand: [])
    monkeypatch.setattr(mcr, "served_weight", lambda path: 0.4)
    nominated = _nominate(
        _sn_row("ratio_meanyr", "crps", "nll", True, 0.3),
        _sn_row("centered_additive_mean10", "crps", "crps", True, 0.2),
    )

    result = mc._walk_nominees(meta, nominated, mc._confirm_one)

    assert result == ("WNBA", "AST", "SHIPPED", [], "2/2 board slack +0.200")
    assert runs == [0, 0]  # both blind runs trained before any stat_meta write
    assert len(writes) == 1
    assert meta["WNBA"]["AST"]["target_normalization"] == "centered_additive_mean10"
    assert meta["WNBA"]["AST"]["shipped"] == "devel"


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
    monkeypatch.setattr(mcr, "backup_stat_meta", lambda: mc.pathlib.Path("/tmp/stat_meta.bak.json"))
    _fake_meta_disk(monkeypatch, tmp_path)
    monkeypatch.setattr(mcr, "snapshot_cell", lambda lg, mkt: tmp_path / "snapshot")
    measured = _stub_blind(monkeypatch, {})
    confirmed = []
    monkeypatch.setattr(
        mcr,
        "confirm_meditate",
        lambda lg, mkt, cand: confirmed.append(cand["source"]) or [],
    )
    monkeypatch.setattr(mcr, "served_weight", lambda path: 0.4)
    monkeypatch.setattr(mc, "prune_model_pickle", lambda lg, mkt: False)

    mc.run_confirm(board, yes=True)
    assert meta["WNBA"]["AST"]["shipped"] == "devel"
    assert meta["WNBA"]["AST"]["target_normalization"] == "centered_additive_mean10"
    assert meta["WNBA"]["AST"]["sn_param"] == "direct"  # the swept default persists explicitly too
    assert measured == confirmed == ["board slack +0.250"]  # the negative rank never nominates
    out = capsys.readouterr().out
    assert "WNBA AST — 1 nominee(s)" in out
    assert "1. [board slack +0.250]" in out
    assert "SHIPPED" in out
    assert "1/1 board slack +0.250" in out  # the report's nominee column


def test_run_confirm_prompt_quotes_every_blind_run(monkeypatch, capsys):
    """The operator is quoted every nominee's blind run, not one per cell."""
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
        mcr, "backup_stat_meta", lambda: pytest.fail("an aborted run must not back up stat_meta")
    )
    prompts = []
    monkeypatch.setattr(mc.click, "confirm", lambda text: prompts.append(text) or False)

    mc.run_confirm(board)
    assert "2 full-HPO sandbox runs" in prompts[0]  # the negative-rank corner never nominates
    assert "aborted" in capsys.readouterr().out


def _walk_first(monkeypatch):
    """Route each cell straight to its lane's attempt with its first nominee — the run-level tests
    below pin lane routing, not the walk.
    """
    monkeypatch.setattr(
        mc,
        "_walk_nominees",
        lambda meta, nominated, attempt, **kw: (*attempt(meta, nominated[0]), "1/1"),
    )


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
    monkeypatch.setattr(mcr, "backup_stat_meta", lambda: mc.pathlib.Path("/tmp/stat_meta.bak.json"))
    _walk_first(monkeypatch)
    calls = {"confirm": [], "supersede": []}
    monkeypatch.setattr(
        mc,
        "_confirm_one",
        lambda m, c, **kw: calls["confirm"].append(c["market"]) or ("WNBA", "AST", "SHIPPED", []),
    )
    monkeypatch.setattr(
        mc,
        "_supersede_one",
        lambda m, c, **kw: (
            calls["supersede"].append(c["market"]) or ("NBA", "PTS", "SUPERSEDED", [])
        ),
    )

    mc.run_confirm(board, yes=True)
    assert calls["confirm"] == ["AST"]
    assert calls["supersede"] == ["PTS"]
    out = capsys.readouterr().out
    assert "SHIPPED" in out and "SUPERSEDED" in out


def test_run_confirm_fresh_only_skips_the_live_lane(monkeypatch, capsys):
    """``fresh_only`` drops the supersession lane before any retrain: live-cell promotions prompt
    per cell, so an unattended run would burn their retrains into guaranteed HELDs."""
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
        "WNBA": {"AST": _sn_original()},
        "NBA": {"PTS": {"dist": "SkewNormal", "shipped": "devel"}},
    }
    monkeypatch.setattr(mc, "load_stat_meta", lambda path: meta)
    monkeypatch.setattr(mcr, "backup_stat_meta", lambda: mc.pathlib.Path("/tmp/stat_meta.bak.json"))
    _walk_first(monkeypatch)
    confirmed = []
    monkeypatch.setattr(
        mc,
        "_confirm_one",
        lambda m, c, **kw: confirmed.append(c["market"]) or ("WNBA", "AST", "SHIPPED", []),
    )
    monkeypatch.setattr(
        mc,
        "_supersede_one",
        lambda m, c, **kw: pytest.fail("fresh_only must never walk the live lane"),
    )

    mc.run_confirm(board, yes=True, fresh_only=True)
    assert confirmed == ["AST"]
    assert "fresh-only: skipping 1 live cell(s)" in capsys.readouterr().out


def test_run_confirm_deadline_skips_cells_not_yet_started(monkeypatch, capsys):
    """The budget is checked between cells: a cell that starts in time finishes its walk, cells past
    the deadline record SKIPPED instead of retraining, and the best board rank walks first."""
    board = pd.DataFrame(
        [
            _sn_row("centered_additive_mean10", "crps", "crps", False, 0.25),
            _zinb_row("hurdle", "pit_ks", "crps", False, 0.50),
        ]
    )
    meta = {
        "WNBA": {"AST": _sn_original()},
        "MLB": {
            "pitcher strikeouts": {
                "dist": "ZINB",
                "dist_training_loss": "nll",
                "shipped": "withheld",
            }
        },
    }
    monkeypatch.setattr(mc, "load_stat_meta", lambda path: meta)
    monkeypatch.setattr(mcr, "backup_stat_meta", lambda: mc.pathlib.Path("/tmp/stat_meta.bak.json"))
    _walk_first(monkeypatch)
    confirmed = []
    monkeypatch.setattr(
        mc,
        "_confirm_one",
        lambda m, c, **kw: (
            confirmed.append((c["league"], c["market"]))
            or (c["league"], c["market"], "SHIPPED", [])
        ),
    )
    ticks = [0.0, 1800.0, 7200.0]  # deadline calc, cell-1 check (inside), cell-2 check (past)
    monkeypatch.setattr(mc.time, "monotonic", lambda: ticks.pop(0) if ticks else 7200.0)

    mc.run_confirm(board, yes=True, deadline_hours=1.0)
    assert confirmed == [("MLB", "pitcher strikeouts")]  # the stronger cell got the budget
    out = capsys.readouterr().out
    assert "WNBA AST" in out and "SKIPPED" in out and "deadline" in out
    assert "1 skipped" in out


def test_walk_nominees_skips_a_corner_the_ledger_already_decided(monkeypatch):
    """A corner with a full-HPO verdict on the identical matrix never reaches the walk — brief L4
    drops it at lane selection, so the slot goes to a corner that can actually run. A ledger row
    from an older matrix does not match, so a cache regen voids the skip. (The walk keeps its own
    skip as a concurrent-run backstop — pinned separately.)"""
    decided, fresh = (
        _sn_row("centered_additive_mean10", "crps", "crps", False, 0.30),
        _sn_row("ratio_meanyr", "crps", "nll", False, 0.25),
    )
    stale = _sn_row("ratio_projvol", "crps", "crps", False, 0.20)
    pd.DataFrame(
        [
            {
                "strategy_corner_fingerprint": decided["corner_fingerprint"],
                "strategy_matrix_hash": _MATRIX_SHA,
            },
            {
                "strategy_corner_fingerprint": stale["corner_fingerprint"],
                "strategy_matrix_hash": "old-matrix",
            },
        ]
    ).to_csv(sweep.NOMINEE_LEDGER_PATH, index=False)
    meta = {"WNBA": {"AST": _sn_original()}}
    measured = _stub_blind(monkeypatch, {})
    attempt, attempted = _attempt_spy("REVERTED", ["g4"])

    result = mc._walk_nominees(
        meta, mcn.candidates(pd.DataFrame([decided, fresh, stale]), meta)[0], attempt
    )
    # The decided leader never nominates; the stale-matrix row runs blind beside the fresh one.
    assert measured == ["board slack +0.250", "board slack +0.200"]
    assert len(attempted) == 1
    assert result[2] == "REVERTED"


def test_run_confirm_reports_nothing_confirmable_when_every_corner_is_unservable(
    monkeypatch, capsys
):
    """Nothing confirmable ⇒ stat_meta is read to assign lanes but never backed up or written."""
    monkeypatch.setattr(
        mcr, "backup_stat_meta", lambda: pytest.fail("an empty board must not touch stat_meta")
    )
    monkeypatch.setattr(
        mcr,
        "atomic_write_meta",
        lambda meta: pytest.fail("an empty board must not write stat_meta"),
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
            "dist_training_loss": "nll",
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
        "MLB": {
            "total bases": {"dist": "ZINB", "dist_training_loss": "nll", "shipped": "withheld"}
        },
    }
    monkeypatch.setattr(mc, "load_stat_meta", lambda path: meta)
    monkeypatch.setattr(mcr, "backup_stat_meta", lambda: mc.pathlib.Path("/tmp/stat_meta.bak.json"))
    _walk_first(monkeypatch)
    confirmed = []
    monkeypatch.setattr(
        mc,
        "_confirm_one",
        lambda m, c, **kw: confirmed.append(c["market"]) or ("WNBA", "AST", "SHIPPED", []),
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
    monkeypatch.setattr(mcr, "backup_stat_meta", lambda: touched.append("backup"))

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
    monkeypatch.setattr(mcr, "_cell_artifacts", lambda lg, mkt: arts)
    monkeypatch.setattr(mcr, "_CONFIRM_LOG_ROOT", tmp_path / "logs")

    backup = mcr.snapshot_cell("NBA", "PTS")
    assert (backup / "NBA_PTS.csv").read_text() == "incumbent"  # the S2/S3 baseline copy

    for a in arts:  # a candidate meditate overwrites every artifact in place
        a.write_text("CANDIDATE")
    meta = {"NBA": {"PTS": {"target_normalization": "candidate"}}}
    monkeypatch.setattr(mcr, "atomic_write_meta", lambda m: None)

    mcr.restore_cell("NBA", "PTS", backup, meta, {"target_normalization": "ratio_meanyr"})
    assert all(a.read_text() == "incumbent" for a in arts)
    assert meta["NBA"]["PTS"] == {"target_normalization": "ratio_meanyr"}


def test_restore_removes_candidate_artifact_absent_from_snapshot(monkeypatch, tmp_path):
    artifact = tmp_path / "NFL_receiving-yards.mdl"
    monkeypatch.setattr(mcr, "_cell_artifacts", lambda lg, mkt: [artifact])
    monkeypatch.setattr(mcr, "_CONFIRM_LOG_ROOT", tmp_path / "logs")
    backup = mcr.snapshot_cell("NFL", "receiving yards")
    artifact.write_text("candidate")
    meta = {"NFL": {"receiving yards": {"shipped": "devel"}}}
    monkeypatch.setattr(mcr, "atomic_write_meta", lambda current: None)

    mcr.restore_cell("NFL", "receiving yards", backup, meta, {"shipped": "withheld"})

    assert not artifact.exists()
    assert meta["NFL"]["receiving yards"] == {"shipped": "withheld"}


def test_cell_artifacts_covers_serve_read_files():
    """The restore set must include every file serving reads back — the pickle (carries the
    calibrators), stat_calibration, and book_weights — else a HOLD could leave the incumbent serving
    the candidate's config. The test-set CSV (the S2/S3 baseline) is here too.
    """
    names = {p.name for p in mcr._cell_artifacts("NBA", "PTS")}
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
    monkeypatch.setattr(mcr, "snapshot_cell", lambda lg, mkt: mc.pathlib.Path("/tmp/bk"))
    monkeypatch.setattr(mcr, "served_weight", lambda path: 0.4)
    monkeypatch.setattr(
        mcr, "run_canonical", lambda lg, mkt, cand: ("" if meditate_ok else "exit 1", False)
    )
    monkeypatch.setattr(mcr, "_retrained_matrix_hash", lambda lg, mkt: _MATRIX_SHA)
    monkeypatch.setattr(mcr, "_failed_gates_after", lambda lg, mkt: [])
    monkeypatch.setattr(mcr, "_produced_artifacts_match", lambda *args: True)
    monkeypatch.setattr(
        mcr,
        "_ship_from_model_stats",
        lambda lg, mkt, expected: model_stats_ok,
    )
    monkeypatch.setattr(mc, "load_test_set", lambda path, col: pd.DataFrame())
    monkeypatch.setattr(mc, "supersede_verdict", lambda *a, **k: verdict)
    monkeypatch.setattr(mcr, "atomic_write_meta", lambda m: None)
    restored, pruned = [], []
    monkeypatch.setattr(
        mcr, "restore_cell", lambda lg, mkt, bk, m, orig: restored.append((lg, mkt, orig))
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


def test_supersede_prepends_diverged_on_a_held_verdict(monkeypatch):
    meta = _shipped_meta()
    _patch_supersede_io(monkeypatch, verdict=_verdict(ship=False, s3=False))
    monkeypatch.setattr(mcr, "run_canonical", lambda lg, mkt, cand: ("", True))
    result = mc._supersede_one(meta, _supersede_cand())
    assert result[:3] == ("NBA", "PTS", "HELD")
    assert result[3] == ["diverged", "S3"]


def test_supersede_pass_and_yes_keeps_candidate(monkeypatch):
    meta = _shipped_meta()
    restored, pruned = _patch_supersede_io(monkeypatch, verdict=_verdict(ship=True))
    monkeypatch.setattr(mc.click, "confirm", lambda *a, **k: True)
    result = mc._supersede_one(meta, _supersede_cand())
    assert result[:3] == ("NBA", "PTS", "SUPERSEDED")
    assert restored == []  # winning candidate kept in place
    assert pruned == []


def test_live_walk_promotes_a_ranked_retry_with_its_pin(monkeypatch):
    """The live lane gets the same fallback: a calibrated retry that ranks carries its pin in its
    edits, so the persisted entry and the promote prompt both show it.
    """
    meta = _shipped_meta()
    restored, pruned = _patch_supersede_io(monkeypatch, verdict=_verdict(ship=True))
    prompts = []
    monkeypatch.setattr(mc.click, "confirm", lambda msg, **k: prompts.append(msg) or True)
    cand = _supersede_cand()
    incumbent = {**cand, "source": "seed/incumbent", "corner_fingerprint": "fp-incumbent"}
    monkeypatch.setattr(mc, "_incumbent_fingerprint", lambda league, market: "fp-incumbent")
    runs = {
        cand["source"]: [_blind_run(cand, failed=["g4"]), _blind_run(_calibrated(cand))],
        "seed/incumbent": [_blind_run(incumbent)],
    }
    monkeypatch.setattr(mcr, "measure_blind", lambda m, c: runs[c["source"]])

    result = mc._walk_nominees(meta, [cand, incumbent], mc._supersede_one, live=True)

    assert result[:3] == ("NBA", "PTS", "SUPERSEDED")
    assert meta["NBA"]["PTS"]["hpo_selection"] == "calibrated"
    assert "hpo_selection=calibrated" in prompts[0]
    assert restored == [] and pruned == []


def test_supersede_auto_promote_swaps_without_prompting(monkeypatch, capsys):
    """``auto_promote`` answers the promote prompt so an unattended run can swap live cells; the
    S1/S2/S3 verdict still decides, and the swap is announced."""
    meta = _shipped_meta()
    restored, pruned = _patch_supersede_io(monkeypatch, verdict=_verdict(ship=True))
    monkeypatch.setattr(
        mc.click, "confirm", lambda *a, **k: pytest.fail("auto_promote must not prompt")
    )
    result = mc._supersede_one(meta, _supersede_cand(), auto_promote=True)
    assert result[:3] == ("NBA", "PTS", "SUPERSEDED")
    assert restored == [] and pruned == []
    assert "auto-promoting NBA PTS" in capsys.readouterr().out


def test_supersede_auto_promote_still_holds_a_losing_verdict(monkeypatch):
    """The flag removes the human veto, not the test: a failing S1/S2/S3 still restores the
    incumbent, so an unattended run can never swap in a worse model."""
    meta = _shipped_meta()
    restored, pruned = _patch_supersede_io(monkeypatch, verdict=_verdict(ship=False, s3=False))
    result = mc._supersede_one(meta, _supersede_cand(), auto_promote=True)
    assert result[:3] == ("NBA", "PTS", "HELD")
    assert restored[0][:2] == ("NBA", "PTS") and pruned == []


def test_walk_lanes_threads_the_run_flags_into_both_lanes(monkeypatch):
    """``run_confirm``'s flags have to reach the walk and the attempt functions; only the live lane
    ranks against an incumbent and promotes.
    """
    seen = {}
    monkeypatch.setattr(
        mc, "_supersede_one", lambda m, c, **kw: seen.update(live=kw) or ("NBA", "PTS", "HELD", [])
    )
    monkeypatch.setattr(
        mc,
        "_confirm_one",
        lambda m, c, **kw: seen.update(fresh=kw) or ("WNBA", "AST", "SHIPPED", []),
    )

    def walk(meta, nominated, attempt, **kw):
        seen[f"walk {nominated[0]['market']}"] = kw
        return (*attempt(meta, nominated[0]), "1/1")

    monkeypatch.setattr(mc, "_walk_nominees", walk)
    fresh = [[{"league": "WNBA", "market": "AST", "source": "seed/incumbent"}]]
    live = [[{"league": "NBA", "market": "PTS", "source": "board slack +0.300"}]]
    mc._walk_lanes({}, fresh, live, None, True, 0.3)
    assert seen == {
        "walk AST": {"live": False, "min_model_weight": 0.3},
        "walk PTS": {"live": True, "min_model_weight": 0.3},
        "fresh": {"min_model_weight": 0.3},
        "live": {"auto_promote": True, "min_model_weight": 0.3},
    }


def _patch_weights(monkeypatch, *, incumbent, candidate):
    """``served_weight`` for the supersession pair — the snapshot dir holds the incumbent's pickle."""
    monkeypatch.setattr(
        mcr, "served_weight", lambda path: incumbent if "bk" in path.parts else candidate
    )


def test_supersede_waives_s2_s3_when_the_incumbent_rides_the_book(monkeypatch, capsys):
    """A w=0.05 incumbent and its w=0.45 challenger both sit on the book at the line, so their
    paired Brier difference is ~0 by construction and S2/S3 can never fire. Where the floor
    separates them, S1 plus that separation is the whole evidence available for the swap.
    """
    meta = _shipped_meta()
    restored, pruned = _patch_supersede_io(
        monkeypatch, verdict=_verdict(ship=False, s2=False, s3=False)
    )
    _patch_weights(monkeypatch, incumbent=0.05, candidate=0.45)
    result = mc._supersede_one(meta, _supersede_cand(), auto_promote=True, min_model_weight=0.3)
    assert result[:3] == ("NBA", "PTS", "SUPERSEDED")
    assert restored == [] and pruned == []
    out = capsys.readouterr().out
    assert "w 0.050→0.450" in out
    assert "S2/S3 waived" in out


def test_supersede_waiver_is_scoped_to_the_pair_it_is_justified_for(monkeypatch):
    """An incumbent already above the floor keeps the full S1+S2+S3 bar, a failing S1 is never
    waived, and with the knob off nothing changes — the waiver only buys the book-riding swap.
    """
    meta = _shipped_meta()
    losing = _verdict(ship=False, s2=False, s3=False)
    _patch_supersede_io(monkeypatch, verdict=losing)

    _patch_weights(monkeypatch, incumbent=0.5, candidate=0.45)
    fitted_incumbent = mc._supersede_one(
        meta, _supersede_cand(), auto_promote=True, min_model_weight=0.3
    )
    assert fitted_incumbent[:3] == ("NBA", "PTS", "HELD")
    assert fitted_incumbent[3] == ["S2", "S3"]

    _patch_weights(monkeypatch, incumbent=0.05, candidate=0.45)
    knob_off = mc._supersede_one(meta, _supersede_cand(), auto_promote=True)
    assert knob_off[:3] == ("NBA", "PTS", "HELD")
    assert knob_off[3] == ["S2", "S3"]

    monkeypatch.setattr(
        mc, "supersede_verdict", lambda *a, **k: _verdict(ship=False, s1=False, s2=False, s3=False)
    )
    no_s1 = mc._supersede_one(meta, _supersede_cand(), auto_promote=True, min_model_weight=0.3)
    assert no_s1[:3] == ("NBA", "PTS", "HELD")
    assert no_s1[3] == ["S1", "S2", "S3"]


def test_supersede_holds_a_candidate_below_min_model_weight(monkeypatch):
    """The floor is a ship criterion on this lane too: a candidate that rides the book is held even
    with all three supersession legs passing."""
    meta = _shipped_meta()
    restored, pruned = _patch_supersede_io(monkeypatch, verdict=_verdict(ship=True))
    _patch_weights(monkeypatch, incumbent=0.5, candidate=0.2)
    result = mc._supersede_one(meta, _supersede_cand(), auto_promote=True, min_model_weight=0.3)
    assert result[:3] == ("NBA", "PTS", "HELD")
    assert result[3] == ["w=0.200<0.3"]
    assert restored[0][:2] == ("NBA", "PTS")
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


def test_supersede_names_the_gate_a_live_refit_fails(monkeypatch):
    """A live refit that misses a gate is held under that gate's name, as on the withheld lane.

    ``ship`` is the six gates' conjunction, so the refit's model_stats leg fails too; checking it
    first reported a plain gate miss on a live cell as "model_stats identity/ship".
    """
    meta = _shipped_meta()
    restored, pruned = _patch_supersede_io(
        monkeypatch, verdict=_verdict(ship=True), model_stats_ok=False
    )
    monkeypatch.setattr(mcr, "_failed_gates_after", lambda lg, mkt: ["g4"])
    monkeypatch.setattr(
        mc,
        "supersede_verdict",
        lambda *a, **k: pytest.fail("a gate miss must stop before S1/S2/S3"),
    )

    assert mc._supersede_one(meta, _supersede_cand()) == ("NBA", "PTS", "HELD", ["g4"])
    assert restored[0][:2] == ("NBA", "PTS")
    assert pruned == []

    monkeypatch.setattr(mcr, "run_canonical", lambda lg, mkt, cand: ("", True))
    assert mc._supersede_one(meta, _supersede_cand())[3] == ["diverged", "g4"]


def test_supersede_artifact_identity_failure_restores_incumbent(monkeypatch):
    meta = _shipped_meta()
    restored, pruned = _patch_supersede_io(monkeypatch, verdict=_verdict(ship=True))
    monkeypatch.setattr(mcr, "_produced_artifacts_match", lambda *args: False)
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
        mcr,
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
    assert checked == [("NBA", "PTS", mcr._candidate_identity(cand))]
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
    """A later, wider gate row must not strand the ledger behind a narrower header."""
    ledger = tmp_path / "confirm_nominee_gates.csv"
    monkeypatch.setattr(sweep, "NOMINEE_LEDGER_PATH", ledger)
    candidate = {"strategy_slug": "sn-centered", "source": "board"}

    mcr._record_ledger_row(
        candidate, {"league": "NBA", "market": "PTS", "g4_iqr_ratio": 0.91}, EVAL_SPLIT_CROSSFIT
    )
    mcr._record_ledger_row(
        candidate,
        {"league": "NBA", "market": "PTS", "g4_iqr_ratio": 0.88, "g7_new_gate": 1.0},
        EVAL_SPLIT_TEST,
    )

    recorded = pd.read_csv(ledger)
    assert len(recorded) == 2
    assert list(recorded["g4_iqr_ratio"]) == [0.91, 0.88]
    assert recorded["g7_new_gate"].isna().tolist() == [True, False]
    assert set(recorded["strategy_slug"]) == {"sn-centered"}
    assert recorded["eval_split"].tolist() == [EVAL_SPLIT_CROSSFIT, EVAL_SPLIT_TEST]


def test_ledger_tags_blind_rows_crossfit_and_the_canonical_refit_test(monkeypatch):
    """Each blind run lands one cross-fit row under the model_stats names the ledger's readers
    expect, controls folded into ``strategy_controls_json``; the cell's one canonical refit lands
    its model_stats row as the test-half read.
    """
    monkeypatch.setattr(mcr, "_run_meditate_with_lock_retry", lambda cmd, path, timeout: None)
    monkeypatch.setattr(mcr, "_score_corner", _scorer(slack=0.15))
    cand = _nominate(_sn_row("ratio_meanyr", "crps", "nll", True, 0.2))[0]
    run = mcr._run_blind(cand, "loss")
    stats_row = pd.Series({"league": "WNBA", "market": "AST", "ship": True, "dispersion_cal": 1.0})
    monkeypatch.setattr(mcr, "_cell_row", lambda league, market, columns: stats_row)

    assert mcr.run_canonical("WNBA", "AST", {**cand, "params_from": run.model}) == ("", False)

    ledger = pd.read_csv(sweep.NOMINEE_LEDGER_PATH)
    assert ledger["eval_split"].tolist() == [EVAL_SPLIT_CROSSFIT, EVAL_SPLIT_TEST]
    blind = ledger.iloc[0]
    assert blind["strategy_corner_fingerprint"] == cand["corner_fingerprint"]
    assert blind["strategy_matrix_hash"] == _MATRIX_SHA
    assert blind["strategy_signature"] == cand["strategy_signature"]
    assert blind["strategy_controls_json"] == controls_json(cand["controls"])
    assert bool(blind["ship"]) and blind["n_validation"] == 373
    assert blind["slack"] == 0.15 and blind["board_slack"] == 0.2
    assert blind["hpo_selection"] == "loss" and blind["elapsed_s"] >= 0
    assert not {"ships", "n", "matrix_hash", "normalization"} & set(ledger.columns)


def test_confirm_median_seconds_times_blind_runs_only():
    """A canonical refit reuses its blind run's parameters, and a legacy row was a test-half
    retrain, so neither may stand in for a blind full-HPO search in the estimate.
    """
    pd.DataFrame(
        [
            {"league": "WNBA", "market": "AST", "elapsed_s": 3600.0, "eval_split": split}
            for split in (EVAL_SPLIT_CROSSFIT, EVAL_SPLIT_CROSSFIT)
        ]
        + [
            {"league": "WNBA", "market": "AST", "elapsed_s": 60.0, "eval_split": EVAL_SPLIT_TEST},
            {"league": "WNBA", "market": "AST", "elapsed_s": 50.0, "eval_split": None},
        ]
    ).to_csv(sweep.NOMINEE_LEDGER_PATH, index=False)
    assert mcr.confirm_median_seconds("WNBA", "AST") == 3600.0
    assert mcr.confirm_median_seconds() == 3600.0


def test_ledger_echoes_board_gates_for_board_nominees_only(monkeypatch, tmp_path):
    """Board nominees carry their board row's gate values into the ledger (``board_*`` columns), so
    board↔confirm comparisons read one row; seed/incumbent nominees leave them NaN.
    """
    ledger = tmp_path / "ledger.csv"
    monkeypatch.setattr(sweep, "NOMINEE_LEDGER_PATH", ledger)
    row = _sn_row("centered_additive_mean10", "crps", "crps", False, 0.25)
    row.update(
        {
            "g1_brier_diff_ci_hi": 0.01,
            "g2_star_z": 1.5,
            "g3_bench_z": 0.8,
            "g4_pit_ks": 0.06,
            "g4_pit_ks_max": 0.05,
            "g5_ece_debiased": 0.04,
            "n": 373,
            "swept_at": "2026-07-29T00:00:00+00:00",
        }
    )
    _seed_corners(monkeypatch, _sn_controls("ratio_projvol", "nll", "nll"))
    board_cand, seed_cand = _nominate(row)
    gates = {"league": "WNBA", "market": "AST", "dispersion_cal": 1.0}

    mcr._record_ledger_row(board_cand, gates, EVAL_SPLIT_CROSSFIT)
    mcr._record_ledger_row(seed_cand, gates, EVAL_SPLIT_CROSSFIT)

    recorded = pd.read_csv(ledger)
    board_row = recorded.iloc[0]
    assert board_row["board_slack"] == 0.25
    assert board_row["board_g1_brier_diff_ci_hi"] == 0.01
    assert board_row["board_g2_star_z"] == 1.5
    assert board_row["board_g3_bench_z"] == 0.8
    assert board_row["board_g4_pit_ks"] == 0.06
    assert board_row["board_g4_pit_ks_max"] == 0.05
    assert board_row["board_g5_ece_debiased"] == 0.04
    assert board_row["board_n"] == 373
    assert board_row["board_eval_split"] == EVAL_SPLIT_CROSSFIT
    assert board_row["board_swept_at"] == "2026-07-29T00:00:00+00:00"
    seed_row = recorded.iloc[1]
    assert seed_row[[f"board_{f}" for f in ("slack", "g2_star_z", "eval_split")]].isna().all()
    assert recorded["diverged"].tolist() == [False, False]


def test_confirm_nominee_cap_truncates_each_cell_after_dedup(monkeypatch):
    """``--confirm-nominees N`` keeps a cell's first N nominees; unset keeps every one."""
    board = pd.DataFrame(
        [
            _sn_row("centered_additive_mean10", "crps", "crps", False, 0.25),
            _sn_row("centered_additive_mean10", "crps", "nll", False, 0.20),
            _sn_row("centered_additive_mean10", "nll", "nll", False, 0.15),
        ]
    )

    uncapped = mcn.candidates(board, _withheld_meta(board))
    capped = mcn.candidates(board, _withheld_meta(board), 2)

    assert [len(cell) for cell in uncapped] == [3]
    assert [len(cell) for cell in capped] == [2]
    assert [n["source"] for n in capped[0]] == [n["source"] for n in uncapped[0][:2]]


# --- mechanism-diverse nomination ---------------------------------------------------------


def test_nominees_spend_the_lane_on_distinct_gate4_mechanisms_not_neighbouring_corners():
    """Three near-identical ZINB corners cannot crowd out a positive DPO row below them.

    The leader keeps slot 1. Slot 2 goes to the best unseen family (DPO) and slot 3 to the best
    corner whose Gate-4 mechanism is still unseen — here the ``pit_ks`` dispersion, not the
    second ``crps`` ZINB that would fail Gate 4 exactly as the leader did.
    """
    nominated = _nominate(
        _zinb_row("joint", "crps", "nll", False, 0.40),
        # Same predictive shape as the leader; only its over-probability corrector differs.
        _zinb_row("joint", "crps", "nll", False, 0.35, posthoc="prob_recal_platt"),
        _zinb_row("joint", "pit_ks", "nll", False, 0.15),
        _negbin_row("crps", "nll", False, 0.25),
    )

    assert [(n["strategy_slug"], n["slack"]) for n in nominated] == [
        ("ZINB", 0.40),
        ("NegBin", 0.25),
        ("ZINB", 0.15),
    ]


def test_nominees_treat_probability_only_recalibrators_as_the_leaders_gate4_mechanism():
    """``prob_recal_*`` does not touch the predictive PIT, so it is not a Gate-4 alternative.

    With only probability-stage variants of the leader available, the lane falls back to plain
    rank order — and a mean-stage corrector, which does reshape what Gate 4 sees, outranks them.
    """
    leader = _sn_controls("ratio_meanyr", "crps", "nll")
    assert mcn._gate4_mechanism({"strategy_slug": "SkewNormal", "controls": leader}) == (
        mcn._gate4_mechanism(
            {
                "strategy_slug": "SkewNormal",
                "controls": {**leader, "posthoc": "prob_recal_platt"},
            }
        )
    )
    assert mcn._gate4_mechanism({"strategy_slug": "SkewNormal", "controls": leader}) != (
        mcn._gate4_mechanism(
            {"strategy_slug": "SkewNormal", "controls": {**leader, "posthoc": "roe_mean"}}
        )
    )

    nominated = _nominate(
        _sn_row("ratio_meanyr", "crps", "nll", False, 0.40),
        _sn_row("ratio_meanyr", "crps", "nll", False, 0.30, posthoc="prob_recal_platt"),
        _sn_row("ratio_meanyr", "crps", "nll", False, 0.20, posthoc="roe_mean"),
    )

    assert [n["slack"] for n in nominated] == [0.40, 0.20, 0.30]


def test_nominees_keep_plain_rank_order_when_no_corner_argues_anything_new():
    nominated = _nominate(
        _sn_row("ratio_meanyr", "crps", "nll", False, 0.40),
        _sn_row("ratio_meanyr", "crps", "nll", False, 0.30, posthoc="prob_recal_isotonic"),
        _sn_row("ratio_meanyr", "crps", "nll", False, 0.20, posthoc="prob_recal_platt"),
        _sn_row("ratio_meanyr", "crps", "nll", False, 0.10, posthoc="none"),
    )

    assert [n["slack"] for n in nominated] == [0.40, 0.30, 0.20]


def test_nonpositive_cells_still_receive_no_confirm_walk_under_diversity():
    assert _nominate(_sn_row("ratio_meanyr", "crps", "nll", False, -0.01)) == []


# --- structural nomination ----------------------------------------------------------------


def test_structural_corners_confirm_only_through_a_board_row_carrying_a_split_fingerprint():
    """A board row brings the split fingerprint a synthesized seed cannot have.

    A structural artifact's fingerprint only exists after the retrain, so nominating the recipe
    straight from the seed register would sign a candidate whose identity check must fail.
    """
    context = sweep._cell_context("NFL", "receiving yards")
    spec = get_strategy(TWO_PART_STRATEGY)
    controls = dict(spec.fixed_controls)

    synthesized = mcn._corner_candidate(
        context, spec, "NFL", "receiving yards", controls, slack=float("nan"), split=None
    )
    from_board = mcn._corner_candidate(
        context, spec, "NFL", "receiving yards", controls, slack=0.4, split="split-123"
    )

    assert synthesized is None
    assert from_board is not None and from_board["split_fingerprint"] == "split-123"

    nominated = _nominate(_structural_row(TWO_PART_STRATEGY, "receiving yards", slack=0.4))
    assert [n["strategy_slug"] for n in nominated] == [TWO_PART_STRATEGY]


def test_board_row_whose_identity_predates_the_current_spec_fails_closed():
    """A stale implementation version is a hard error, never a silently walked nominee."""
    row = _structural_row(TWO_PART_STRATEGY, "receiving yards")
    row["strategy_implementation_version"] -= 1

    with pytest.raises(ValueError, match="stale or mismatched strategy identity"):
        _nominate(row)

    rebuilt = _structural_row(TWO_PART_STRATEGY, "receiving yards")
    rebuilt["matrix_hash"] = "matrix-rebuilt"
    with pytest.raises(ValueError, match="stale or mismatched strategy identity"):
        _nominate(rebuilt)
