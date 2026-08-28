"""Operation Ship 75 confirm-and-ship loop: persist a swept winner, retrain at full HPO, keep or revert.

The strategy sweep (:mod:`sportstradamus.training.model_strategy.sweep`) *ranks* corners on fixed-HP
deterministic trials — it never ships. This module turns a ranked board into shipped cells:

1. For each cell, nominate the top :data:`CONFIRM_TOP_K` corners **across the cell's swept
   families** by ship slack — a count cell's slice carries ZINB, NegBin and DPO corners ranked
   together — then its seed corners and its incumbent. Every swept axis has a ``stat_meta.json``
   field, and each nominee's own family's ``persist`` map builds the edits (so a NegBin corner
   outranking ZINB persists ``dist=NegBin`` and flips the cell's family).
2. Prompt, then per nominee in order: write its persist fields + ``shipped="devel"`` to
   ``stat_meta.json`` (so the confirm ``meditate`` reads the exact config being shipped), run a
   **full-HPO** ``meditate`` on the walk's pinned training matrix, and read the official ``ship``
   verdict from ``model_stats.parquet``.
3. A pass keeps the cell on devel and ends its walk; a failure **auto-reverts** — restore the
   original stat_meta entry *and* :func:`prune_model_pickle` — and the next nominee is tried. The
   pickle prune is mandatory: inference loads pickles by path and never consults ``shipped``, so
   reverting stat_meta alone would leave a failed cell serving.

The walk also owns the g4-only calibrated fallback: a nominee failing ship on a near-miss Gate 4
alone is retrained once under ``hpo_selection: "calibrated"``, and a shipping retry persists that
pin with the cell's other edits. The weekly ``meditate`` never explores — it trains each cell once
with whatever this walk persisted.

Everything is local and uncommitted — the human reviews the ``shipped: devel`` diff and commits it.
A whole-file ``stat_meta.json`` backup is taken before any write as the crash/abort safety net.
"""

import functools
import json
import math
import pathlib
import shutil
import subprocess
import time
from copy import deepcopy

import click
import pandas as pd
import tabulate

from sportstradamus.helpers.io import (
    MODEL_STATS_PATH,
    market_file_slug,
    model_pickle_path,
    prune_model_pickle,
)
from sportstradamus.training.lineage import (
    MATRIX_BUILDER_VERSION,
    MATRIX_SCHEMA_VERSION,
    file_sha,
)
from sportstradamus.training.model_strategy.identity import (
    ArtifactIdentity,
    build_artifact_identity,
    corner_fingerprint,
    validate_strategy_artifacts,
)
from sportstradamus.training.model_strategy.progress import (
    elapsed_ticker,
    human_seconds,
    wrapped,
)
from sportstradamus.training.model_strategy.registry import (
    BASE_STRUCTURAL_STRATEGY,
    CAP_CONFIRM,
    CAP_FULL_HPO,
    CAP_SCORE,
    CAP_SERVE,
    controls_json,
    distribution_class,
    get_strategy,
    meditate_command,
    parse_controls,
    strategies_for_cell,
    strategy_controls,
    strategy_full_hpo_cli_args,
    strategy_persistence_edits,
)
from sportstradamus.training.model_strategy.sweep import (
    _DIVERGED_DISPERSION_CAL,
    _GATES,
    _SHIP_PRED_COL,
    _TEST_SETS_ROOT,
    EVAL_SPLIT_CROSSFIT,
    NOMINEE_LEDGER_PATH,
    _cell_context,
    _failure_reason,
    _run_meditate_with_lock_retry,
    _seed_corners,
    _training_matrix_path,
)
from sportstradamus.training.posthoc import PROB_STAGE
from sportstradamus.training.scorecard import (
    _GATE1_NONINF_MARGIN,
    _supersede_headline,
    load_test_set,
    supersede_verdict,
)
from sportstradamus.training.ship_config import STAT_META_PATH, WITHHELD, load_stat_meta

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_STAT_META = pathlib.Path(str(STAT_META_PATH))
_CONFIRM_LOG_ROOT = _REPO_ROOT / "research" / "logs" / "confirm"
_SHIPPED_DEVEL = "devel"
# Board-row gate values echoed onto each board nominee (as ``board_*`` ledger columns), so a
# board↔confirm comparison reads one ledger row instead of joining two CSVs. Seed/incumbent
# nominees have no board row and leave them NaN.
_BOARD_ECHO_FIELDS: tuple[str, ...] = (
    "slack",
    "g1_brier_diff_ci_hi",
    "g2_star_z",
    "g3_bench_z",
    "g4_pit_ks",
    "g4_pit_ks_max",
    "g5_ece_debiased",
    "n",
    "eval_split",
    "swept_at",
)
# Leagues whose withheld cells confirm may never auto-flip: a board-passing cell is announced and
# skipped until the owner's activation gates (D1/D2) go GO, then the league is removed here in the
# same PR that ships its first cells. Empty since the MLB+NHL GO (2026-07-09,
# docs/handoffs/mlb-nhl-activation.md); machinery stays for the next league onboarding.
_ACTIVATION_GATED_LEAGUES: tuple[str, ...] = ()
# A full-HPO meditate confirm is ~1 h; a large cell can run longer, so a 4 h ceiling keeps a hung
# run from blocking the loop forever. A timeout is treated as a failure (the cell auto-reverts).
_CONFIRM_TIMEOUT_S = 4 * 3600
# Confirm outcomes that leave a shippable cell on devel (vs REVERTED / HELD, which change nothing).
_WIN_OUTCOMES: tuple[str, ...] = ("SHIPPED", "SUPERSEDED")
_CONFIRM_CAPABILITIES = frozenset({CAP_CONFIRM, CAP_FULL_HPO, CAP_SCORE, CAP_SERVE})
# Board rows a cell nominates for full-HPO confirmation. Deterministic fixed-HP scoring does not
# ship the recipes that only pass under real HPO, so nominating the top few — rather than requiring
# a deterministic `ships` — is what makes confirmation reachable at all for the popular NFL passing
# markets, whose boards carry zero shipping rows.
CONFIRM_TOP_K: int = 3
# Controls that change the shape of the predictive distribution, and therefore the PIT Gate 4
# scores. Two corners agreeing on all of them (and on their family) fail Gate 4 identically.
_GATE4_MECHANISM_CONTROLS: tuple[str, ...] = (
    "normalization",
    "dist_training_loss",
    "sn_param",
    "zinb_mode",
    "count_dispersion_objective",
    "blending_loss_fn",
)
# Correctors that move a single over-probability rather than the predictive PIT, so Gate 4
# cannot tell them from ``none``.
_PIT_NEUTRAL_POSTHOC: frozenset[str] = frozenset({"none", *PROB_STAGE})
# brief L2: exact-rank ties only. Wider bands demote corners the board genuinely ranks higher.
_SHAPING_SLOT_TIE_TOL: float = 1e-9
# The exact model_stats identity columns the official ship verdict is bound to.
_SHIP_IDENTITY_COLUMNS = [
    "league",
    "market",
    "strategy_slug",
    "structural_strategy",
    "strategy_signature",
    "strategy_implementation_version",
    "artifact_schema_version",
    "strategy_status",
    "strategy_controls_json",
    "strategy_corner_fingerprint",
    "strategy_matrix_hash",
    "strategy_split_fingerprint",
    "ship",
]
# The six ship-gate booleans in model_stats.parquet (nullable BooleanDtype; pd.NA means the
# scorecard never ran, which counts as not passing here).
_GATE_COLS: tuple[str, ...] = tuple(f"{g}_pass" for g in _GATES)
# brief R3: separates the 9 addressable near-miss ledger rows from the 4 structural
# failures at excess >= 0.049.
_G4_RETRY_MAX_EXCESS: float = 0.010
# brief R3: noise band on the g1 non-inferiority CI bound.
_G1_RETRY_NOISE_BAND: float = 0.002
# The model_stats columns the calibrated-retry predicate reads.
_RETRY_COLUMNS = [
    "league",
    "market",
    "distribution",
    "ship",
    "g4_pit_ks",
    "g4_pit_ks_max",
    "g1_brier_diff_ci_hi",
    *_GATE_COLS,
]


def _rank_column(admissible: pd.DataFrame, *, live: bool) -> str:
    """The board value that decides this cell's nominations.

    A withheld cell is ranked on absolute gate headroom, because passing the gates outright is
    exactly the bar it has to clear. A live cell's bar is a different question — is this corner
    better than the recipe already serving? — so it ranks on ``margin_vs_incumbent``. Ranking a
    supersession on slack measures the candidate against the book and the gates while never
    comparing it to the incumbent it would replace. ``live`` is only ever set on a board that
    carries margins at all (see :func:`_candidates`), so the column is present whenever it is read.
    """
    if live:
        return "margin_vs_incumbent"
    return "discounted_slack" if "discounted_slack" in admissible.columns else "slack"


def _nominees(sub: pd.DataFrame, *, live: bool = False) -> list[dict]:
    """Ordered confirm nominees for one cell: top board corners, then seeds, then the incumbent.

    Only board-confident corners nominate: a corner needs a positive admissibility value — the
    milder ``veto_slack`` pricing on a withheld cell that carries it, else the same value
    :func:`_rank_column` picks for ordering — and a cell whose best admissible corner is
    non-positive returns no nominees at all. On a live cell that also means a cell whose baseline
    never scored (``margin_vs_incumbent`` all ``NaN``) nominates nobody: an unmeasured incumbent
    makes every margin unknown, and an unknown margin is not evidence a corner is better. This
    trades the rare full-HPO rescue of a board-negative recipe (NBA FTM shipped one at board
    −1.86) for not spending walks on cells the board predicts to fail; the operator chose the
    wall clock.
    Legacy boards with no cross-fit rows keep their seed lane — there the board has no opinion.

    Ordering is the argument each source can make. Board rows first, ranked by the same value, with
    the best count-class corner interleaved second on an integer-target cell whose top corners are
    all continuous-family (:func:`_count_class_backup`). Seeds next: independent full-HPO held-out
    evidence the deterministic ranking demonstrably cannot see. The incumbent last, so a cell is
    never downgraded by an unlucky list. Rows scored against the ship holdout (legacy
    ``eval_split``) never nominate. Every nominee carries the cell's best board rank as
    ``board_rank`` so :func:`_candidates` can walk the strongest cells first.
    """
    lg, mkt = sub["league"].iloc[0], sub["market"].iloc[0]
    context = _cell_context(lg, mkt)
    admissible = sub[sub["eval_split"].astype("string").eq(EVAL_SPLIT_CROSSFIT)]
    rank_column = _rank_column(admissible, live=live)
    # A board read back off disk carries pd.NA in any column its sweep predates, and `pd.NA > 0` is
    # neither True nor False — every rank comparison below would raise on it.
    # brief L5: the stable fingerprint tie-break makes an exact-tie lane a fact about the board,
    # not about row order.
    ranked = admissible.assign(
        **{rank_column: pd.to_numeric(admissible[rank_column], errors="coerce")}
    ).sort_values([rank_column, "corner_fingerprint"], ascending=[False, True], kind="stable")
    if len(admissible):
        board_rank = float(ranked[rank_column].max())
        lane_rows = _veto_admissible(ranked, rank_column, live=live)
        if lane_rows is None:
            return []
        ranked = lane_rows
    else:
        board_rank = float("-inf")
    nominated = _board_lane(ranked, rank_column, context, lg, mkt)
    backup = _count_class_backup(ranked, nominated, context, lg, mkt)
    if backup is not None:
        nominated.insert(1, backup)
    for spec, controls in _seed_corners(context):
        cand = _corner_candidate(context, spec, lg, mkt, controls, slack=math.nan, split=None)
        if cand is not None:
            nominated.append({**cand, "source": "seed/incumbent"})
    seen: set[str] = set()
    return [
        {**cand, "board_rank": board_rank}
        for cand in nominated
        if not (cand["corner_fingerprint"] in seen or seen.add(cand["corner_fingerprint"]))
    ]


def _veto_admissible(ranked: pd.DataFrame, rank_column: str, *, live: bool) -> pd.DataFrame | None:
    """The rows eligible for lane slots, or ``None`` when the whole cell fails the nomination bar.

    brief R3: admissibility is priced at the milder median ``veto_slack`` while the ordering (and
    ``board_rank``, the walk order) stays on the conservative rank column, so a cell whose corners
    fail only under the conservative price still walks — last. Boards that predate the veto
    pricing, and the live lane's margin, keep the rank column as the bar. brief L4: a corner the
    ledger has already decided on this matrix cannot usefully walk again — the walk would skip
    it — so it never spends a lane slot either.
    """
    veto = ranked[rank_column]
    if not live and "veto_slack" in ranked.columns:
        veto = pd.to_numeric(ranked["veto_slack"], errors="coerce").fillna(veto)
    if not veto.max() > 0:
        return None
    ranked = ranked[veto > 0]
    decided = _decided_pairs()
    if decided:
        pairs = zip(
            ranked["corner_fingerprint"].astype(str),
            ranked["matrix_hash"].astype(str),
            strict=True,
        )
        ranked = ranked[[pair not in decided for pair in pairs]]
    return ranked


def _decided_pairs() -> set[tuple[str, str]]:
    """Every ``(corner_fingerprint, matrix_hash)`` the ledger holds a full-HPO verdict for."""
    if not NOMINEE_LEDGER_PATH.exists():
        return set()
    ledger = pd.read_csv(NOMINEE_LEDGER_PATH)
    return set(
        zip(
            ledger["strategy_corner_fingerprint"].astype(str),
            ledger["strategy_matrix_hash"].astype(str),
            strict=True,
        )
    )


def _gate4_mechanism(candidate: dict) -> tuple[str, ...]:
    """What a corner does to the predictive PIT — the thing Gate 4 actually scores.

    Two corners sharing this key fail Gate 4 the same way, so walking both spends a full-HPO
    hour to learn one thing. ``posthoc`` only joins the key when it reshapes the distribution:
    ``none`` and the probability-only recalibrators leave the predictive PIT untouched, so they
    are one mechanism, not three.
    """
    controls = candidate["controls"]
    shaping = controls.get("posthoc", "none")
    return (
        candidate["strategy_slug"],
        *(controls.get(name, "") for name in _GATE4_MECHANISM_CONTROLS),
        "" if shaping in _PIT_NEUTRAL_POSTHOC else shaping,
    )


def _board_lane(
    ranked: pd.DataFrame, rank_column: str, context, league: str, market: str
) -> list[dict]:
    """The cell's veto-admissible board corners, ordered for mechanism diversity, capped at K.

    ``ranked`` arrives already filtered to the corners that clear the nomination bar (see
    :func:`_nominees`), best rank first. The leader always goes first — the board's own opinion is
    the best single guess. After that a corner earns its slot by being *different*: first the
    highest-ranked unseen family, then the highest-ranked unseen Gate-4 mechanism — with a
    PIT-reshaping corner taking an exactly tied slot (:func:`_next_slot`) — and only then the
    next corner by rank. Without it a cell whose top three rows are near-identical ZINB corners
    spends its whole walk relearning one verdict while a positive DPO or structural row never
    runs.
    """
    pool: list[dict] = []
    rank_values: list[float] = []
    for _, row in ranked.iterrows():
        cand = _board_candidate_row(row, context, league, market)
        if cand is not None:
            pool.append({**cand, "source": f"board slack {cand['slack']:+.3f}"})
            rank_values.append(float(row[rank_column]))

    lane: list[dict] = []
    families: set[str] = set()
    mechanisms: set[tuple[str, ...]] = set()
    remaining = list(range(len(pool)))
    while remaining and len(lane) < CONFIRM_TOP_K:
        pick = (
            _next_slot(pool, rank_values, remaining, families, mechanisms) if lane else remaining[0]
        )
        remaining.remove(pick)
        lane.append(pool[pick])
        families.add(pool[pick]["strategy_slug"])
        mechanisms.add(_gate4_mechanism(pool[pick]))
    return lane


def _next_slot(
    pool: list[dict],
    rank_values: list[float],
    remaining: list[int],
    families: set[str],
    mechanisms: set[tuple[str, ...]],
) -> int:
    """The highest-ranked remaining corner that argues something new, else the next by rank.

    brief L2: when the lane holds no PIT-reshaping corner and the pick is a non-reshaping corner
    exactly tied (:data:`_SHAPING_SLOT_TIE_TOL`) with an unseen-mechanism corner that does
    reshape, the reshaper takes the slot — the ranking statistic cannot separate the two (slack
    binds on a mean/skill gate on ~97% of top rows), and confirm Gate 4 demonstrably can.
    """
    unseen_family = (index for index in remaining if pool[index]["strategy_slug"] not in families)
    pick = next(unseen_family, None)
    if pick is not None:
        return pick
    unseen = [index for index in remaining if _gate4_mechanism(pool[index]) not in mechanisms]
    pick = unseen[0] if unseen else remaining[0]
    lane_reshapes = any(mechanism[-1] for mechanism in mechanisms)
    if not lane_reshapes and not _gate4_mechanism(pool[pick])[-1]:
        tied_reshaper = next(
            (
                index
                for index in unseen
                if _gate4_mechanism(pool[index])[-1]
                and abs(rank_values[index] - rank_values[pick]) <= _SHAPING_SLOT_TIE_TOL
            ),
            None,
        )
        if tied_reshaper is not None:
            return tied_reshaper
    return pick


def _count_class_backup(
    ranked: pd.DataFrame, nominated: list[dict], context, league: str, market: str
) -> dict | None:
    """The best count-class board corner to slot second on an integer-target cell, or ``None``.

    An integer-target cell's top corners are often all SkewNormal — the family whose full-HPO fit
    can diverge — while stable count-family corners rank just below; whole walks burned on that
    pattern. Interleaving the best count corner at slot 2 costs nothing when the leader confirms
    and saves the walk when it diverges. No-op when the cell's target is not on the integer
    lattice, nothing was nominated, a count corner already sits in the top slots, or no
    veto-admissible count-class row remains (``ranked`` is pre-filtered to the nomination bar).
    Insert-only: no continuous nominee is dropped.
    """
    if not context.target_is_integer or not nominated:
        return None
    if any(distribution_class(cand["family"]) == "count" for cand in nominated):
        return None
    for _, row in ranked.iterrows():
        if distribution_class(str(row["family"])) != "count":
            continue
        cand = _board_candidate_row(row, context, league, market)
        if cand is not None:
            return {**cand, "source": f"board slack {cand['slack']:+.3f}"}
    return None


def _corner_candidate(
    context, spec, league: str, market: str, controls: dict[str, str], *, slack: float, split
) -> dict | None:
    """The confirm-candidate dict for one registered corner, or ``None`` if it cannot confirm.

    A family without the confirm capabilities (Mixture ranks but never serves) drops out here, as
    does a corner whose spec is not applicable to the cell — a seed or incumbent can outlive the
    admission gates that once let it in. So does a structural corner with no board row behind it:
    a structural artifact's split fingerprint only exists after the retrain, so a synthesized
    nominee would carry ``None`` and fail its own identity check.
    """
    if spec not in strategies_for_cell(context) or not spec.capabilities >= _CONFIRM_CAPABILITIES:
        return None
    if spec.is_structural and split is None:
        return None
    identity = build_artifact_identity(
        spec.slug, league, market, controls, matrix_hash=context.matrix_sha256
    )
    return {
        "league": league,
        "market": market,
        "family": spec.family,
        "strategy_slug": spec.slug,
        "structural_strategy": identity.structural_strategy,
        "strategy_signature": identity.signature,
        "strategy_implementation_version": identity.implementation_version,
        "artifact_schema_version": identity.artifact_schema_version,
        "strategy_status": identity.status,
        "matrix_hash": identity.matrix_hash,
        "split_fingerprint": split,
        "controls": controls,
        "corner_fingerprint": identity.corner_fingerprint,
        "edits": strategy_persistence_edits(context, spec, controls),
        "slack": slack,
    }


def _board_candidate_row(row: pd.Series, context, league: str, market: str) -> dict | None:
    """Validate one board row's full identity contract; ``None`` if not confirm-capable.

    Raises on any stale/mismatched identity; returns the confirm-candidate dict for a clean,
    confirm-capable corner, or ``None`` when the corner's family lacks the confirm capabilities so
    the caller can fall through to the next-best row.
    """
    family = _required_text(row, "family", league, market)
    strategy_slug = _required_text(row, "strategy_slug", league, market)
    structural = _required_text(row, "structural_strategy", league, market)
    spec = get_strategy(strategy_slug)
    expected_structural = spec.slug if spec.is_structural else BASE_STRUCTURAL_STRATEGY
    if spec.family != family or structural != expected_structural:
        raise ValueError(f"{league} {market}: strategy/family identity mismatch")
    if spec not in strategies_for_cell(context):
        raise ValueError(f"{league} {market}: strategy {spec.slug!r} is not enrolled/applicable")
    controls = parse_controls(row.get("controls_json"))
    if controls_json(controls) != row["controls_json"] or controls not in strategy_controls(spec):
        raise ValueError(f"{league} {market}: stale or noncanonical strategy controls")
    _validate_control_columns(row, controls, league, market)
    identity = build_artifact_identity(
        spec.slug, league, market, controls, matrix_hash=context.matrix_sha256
    )
    split = _validate_board_identity(row, spec, identity, league, market)
    slack = float(row["slack"])
    if not math.isfinite(slack):
        raise ValueError(f"{league} {market}: candidate slack must be finite")
    cand = _corner_candidate(context, spec, league, market, controls, slack=slack, split=split)
    if cand is not None:
        cand["board_gates"] = {f"board_{field}": row.get(field) for field in _BOARD_ECHO_FIELDS}
    return cand


def _required_text(row: pd.Series, field: str, league: str, market: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value or pd.isna(value):
        raise ValueError(f"{league} {market}: candidate has missing {field}")
    return value


def _validate_control_columns(
    row: pd.Series, controls: dict[str, str], league: str, market: str
) -> None:
    for name, expected in controls.items():
        if name not in row.index:
            continue
        actual = row.get(name)
        if pd.isna(actual) or str(actual) != expected:
            raise ValueError(f"{league} {market}: control column {name} contradicts controls_json")


def _int_matches(actual: object, expected: int) -> bool:
    """Whether a board cell's integer field equals ``expected`` (NaN / non-numeric → False)."""
    try:
        return not pd.isna(actual) and float(actual) == expected
    except (TypeError, ValueError):
        return False


def _validate_split_contract(spec, split: object, league: str, market: str) -> None:
    if spec.split_fingerprint_path and (not isinstance(split, str) or not split):
        raise ValueError(f"{league} {market}: missing strategy split fingerprint")
    if not spec.split_fingerprint_path and split is not None:
        raise ValueError(f"{league} {market}: unexpected strategy split fingerprint")


def _validate_board_identity(
    row: pd.Series,
    spec,
    identity: ArtifactIdentity,
    league: str,
    market: str,
) -> str | None:
    core_matches = (
        _required_text(row, "strategy_slug", league, market) == identity.strategy_slug
        and _required_text(row, "strategy_signature", league, market) == identity.signature
        and _required_text(row, "strategy_status", league, market) == "active"
        and _required_text(row, "matrix_hash", league, market) == identity.matrix_hash
        and _int_matches(
            row.get("strategy_implementation_version"), identity.implementation_version
        )
        and _int_matches(row.get("artifact_schema_version"), identity.artifact_schema_version)
    )
    if not core_matches:
        raise ValueError(f"{league} {market}: stale or mismatched strategy identity")
    fingerprint = _required_text(row, "corner_fingerprint", league, market)
    if fingerprint != identity.corner_fingerprint:
        raise ValueError(f"{league} {market}: stale strategy corner fingerprint")
    split = row.get("split_fingerprint")
    split = None if pd.isna(split) else split
    _validate_split_contract(spec, split, league, market)
    return split


def _candidates(
    board: pd.DataFrame, meta: dict, max_nominees: int | None = None
) -> list[list[dict]]:
    """Each cell's ordered nominee list, strongest cells first; nothing confirmable drops out.

    ``max_nominees`` truncates each list after dedup, trading the tail of a cell's walk for wall
    clock. The ordering :func:`_nominees` establishes is what makes that trade sound — the corners
    most likely to ship come first. ``meta`` decides each cell's lane, which decides what its
    nominees are ranked on.

    Cells sort by their best board rank descending (legacy seed-only cells last) so a deadline cut
    lands on the weakest tail, not on cells the board likes that happened to group late. The two
    lanes rank on different quantities, but a stable sort over the mixed list still leaves each
    lane correctly ordered within itself, which is all :func:`_walk_lanes` walks.

    Whether a board carries incumbent margins is a property of the sweep that wrote it, so it is
    decided once here rather than per cell: on a board swept before margins existed every cell keeps
    the old slack ranking, and on one swept with them a live cell whose own baseline never scored
    gets ``NaN`` — unknown — instead of silently falling back to the wrong quantity.
    """
    priced = "margin_vs_incumbent" in board and board["margin_vs_incumbent"].notna().any()
    cells = [
        nominated[:max_nominees]
        for (league, market), sub in board.groupby(["league", "market"], sort=False)
        if (
            nominated := _nominees(
                sub, live=priced and meta[league][market].get("shipped") != WITHHELD
            )
        )
    ]
    cells.sort(key=lambda nominated: nominated[0]["board_rank"], reverse=True)
    return cells


def _atomic_write_meta(meta: dict) -> None:
    """Write stat_meta.json atomically at its native 4-space indent so the review diff stays minimal."""
    tmp = _STAT_META.with_suffix(".json.tmp")
    with tmp.open("w") as fh:
        json.dump(meta, fh, indent=4)
        fh.write("\n")
    tmp.replace(_STAT_META)


def _backup_stat_meta() -> pathlib.Path:
    """Copy the whole stat_meta.json to a timestamped sibling — the crash/abort recovery point."""
    backup = _STAT_META.with_name(f"stat_meta.{time.strftime('%Y%m%dT%H%M%S')}.bak.json")
    shutil.copy2(_STAT_META, backup)
    return backup


def _cell_artifacts(league: str, market: str) -> list[pathlib.Path]:
    """Every serve-read artifact a full-HPO ``meditate`` rewrites for one cell — the restore set.

    Restoring these puts the incumbent back exactly as it served: the model pickle (which carries all
    calibrators), the test-set CSV (also the snapshotted S2/S3 baseline), ``model_stats`` (parquet +
    csv mirror), the two SHAP CSVs, and the two config files read at serve time — ``stat_calibration``
    and ``book_weights`` (shared, its per-cell key reverted by the whole-file restore). The shared
    files are safe to whole-file restore because a cell's snapshot→restore window is isolated (only
    that cell's ``meditate`` runs between them, so the revert touches only that cell's row/key).

    Deliberately excluded: the per-cell training-matrix cache (``training_data/{slug}.parquet``) and
    the per-league caches (gamelog, ``comps.json``, correlation matrices). Those are training inputs,
    never read at serve time — and a confirm retrain trains from the walk's frozen snapshot
    (``--frozen-matrix-dir``) without touching any of them.
    """
    slug = market_file_slug(league, market)
    training_dir = MODEL_STATS_PATH.parent
    config_dir = _STAT_META.parent
    return [
        model_pickle_path(league, market),
        _TEST_SETS_ROOT / f"{slug}.csv",
        MODEL_STATS_PATH,
        MODEL_STATS_PATH.with_suffix(".csv"),
        training_dir / "feature_importances.csv",
        training_dir / "feature_correlations.csv",
        config_dir / "stat_calibration.json",
        config_dir / "book_weights.json",
    ]


def _snapshot_cell(league: str, market: str) -> pathlib.Path:
    """Copy the incumbent's artifacts to a per-cell backup dir and return it (the S2/S3 baseline lives there)."""
    backup = _CONFIRM_LOG_ROOT / "incumbent_backup" / market_file_slug(league, market)
    shutil.rmtree(backup, ignore_errors=True)
    backup.mkdir(parents=True, exist_ok=True)
    for art in _cell_artifacts(league, market):
        if art.exists():
            shutil.copy2(art, backup / art.name)
    return backup


def _restore_cell(
    league: str, market: str, backup: pathlib.Path, meta: dict, original: dict
) -> None:
    """Restore every snapshotted artifact byte-identical and put the original stat_meta entry back.

    Copies files from ``backup`` over the canonical paths — it never prunes — so a live cell that loses
    the supersession test keeps serving exactly what it served before.
    """
    for art in _cell_artifacts(league, market):
        saved = backup / art.name
        if saved.exists():
            shutil.copy2(saved, art)
        elif art.exists():
            art.unlink()
    meta[league][market] = original
    _atomic_write_meta(meta)


def _cell_row(league: str, market: str, columns: list[str] | None) -> pd.Series | None:
    """One cell's row of the requested model_stats.parquet columns, or ``None`` when absent.

    ``None`` columns reads the whole row.
    """
    stats = pd.read_parquet(MODEL_STATS_PATH, columns=columns)
    hit = stats[(stats["league"] == league) & (stats["market"] == market)]
    return None if hit.empty else hit.iloc[0]


def _split_matches(actual: object, expected: str | None) -> bool:
    """Whether a model_stats split fingerprint matches the expected value (both-absent counts)."""
    if expected is None:
        return bool(pd.isna(actual))
    return actual == expected


def _ship_from_model_stats(league: str, market: str, expected: ArtifactIdentity) -> bool:
    """The official ship verdict bound to the exact reported strategy contract."""
    try:
        row = _cell_row(league, market, _SHIP_IDENTITY_COLUMNS)
    except (OSError, ValueError, KeyError, TypeError):
        return False
    if row is None or not bool(row["ship"]) or row["strategy_status"] != "active":
        return False
    checks = {
        "strategy_slug": expected.strategy_slug,
        "structural_strategy": expected.structural_strategy,
        "strategy_signature": expected.signature,
        "strategy_implementation_version": expected.implementation_version,
        "artifact_schema_version": expected.artifact_schema_version,
        "strategy_controls_json": expected.controls_json,
        # Not strategy_matrix_hash: expected is rebound to the hash this row reported, so comparing
        # them here would assert nothing. The fingerprint below re-derives from that hash instead.
        "strategy_corner_fingerprint": expected.corner_fingerprint,
    }
    if any(row[column] != value for column, value in checks.items()):
        return False
    return _split_matches(row["strategy_split_fingerprint"], expected.split_fingerprint)


def _record_nominee_gates(league: str, market: str, candidate: dict) -> bool:
    """Record the just-retrained nominee's whole model_stats row in the research ledger.

    This is the only moment that row exists. A losing nominee gets its pickle pruned, and
    ``report()`` rebuilds model_stats from the pickles on disk, so by the time the walk reports
    its verdict the gate values behind it are gone — leaving no way to compare a cell's board
    numbers against its own confirm. A board nominee's row also carries its board-side gate values
    (``board_*``, from ``_board_candidate_row``); seed/incumbent rows leave them NaN. Ledger only;
    nothing production reads it.

    Returns whether the fit diverged — ``dispersion_cal`` on its floor
    (:data:`_DIVERGED_DISPERSION_CAL`) — which is also recorded as the ``diverged`` column.
    """
    row = _cell_row(league, market, None)
    if row is None:
        return False
    dispersion = row.get("dispersion_cal")
    diverged = bool(pd.notna(dispersion) and float(dispersion) <= _DIVERGED_DISPERSION_CAL)
    ledger = pd.DataFrame(
        [
            {
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "elapsed_s": candidate.get("elapsed_s"),
                "strategy_slug": candidate["strategy_slug"],
                "source": candidate["source"],
                "diverged": diverged,
                **candidate.get("board_gates", {}),
                **row.to_dict(),
            }
        ]
    )
    if NOMINEE_LEDGER_PATH.exists():
        # Rewrite rather than append: model_stats widens as gates are added, and appending a
        # wider row under a header written by a narrower one yields a CSV no reader can parse.
        ledger = pd.concat([pd.read_csv(NOMINEE_LEDGER_PATH), ledger], ignore_index=True)
    NOMINEE_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(NOMINEE_LEDGER_PATH, index=False)
    return diverged


def _failed_gates_after(league: str, market: str) -> list[str]:
    """The gates a just-confirmed cell fails, read from model_stats — diagnostics for the report."""
    row = _cell_row(league, market, ["league", "market", *_GATE_COLS])
    if row is None:
        return ["(no model_stats row)"]
    return [g for g in _GATES if not bool(row[f"{g}_pass"])]


def _frozen_matrix_dir(league: str, market: str) -> pathlib.Path:
    """The per-cell dir a walk pins its training matrix into (shared by pin and retrain)."""
    return _CONFIRM_LOG_ROOT / "frozen_matrix" / market_file_slug(league, market)


def _pin_cell_matrix(league: str, market: str) -> pathlib.Path:
    """Copy the cell's cached training matrix + lineage manifest into the walk's frozen dir.

    Every nominee of the walk then retrains via ``--frozen-matrix-dir`` on this one frame — an
    unpinned ``--force`` rewrote the matrix between nominees, so they scored different frames
    (n_validation drifted 373/363/358 within one walk). Two side effects of meditate's frozen-input
    mode are deliberate: the run cold-starts its HPO (warm pickle params are ignored, so every
    nominee gets the same full search budget) and it skips the per-market book-weight refits (the
    same isolation as the board's deterministic runs, keeping board and confirm aligned). A swept
    cell always has a cached parquet — the sweep requires one — so a missing source fails loud
    here. The snapshot stays on disk after the walk (research/ is gitignored), matching the
    ``incumbent_backup`` pattern.
    """
    source = _training_matrix_path(league, market)
    frozen_dir = _frozen_matrix_dir(league, market)
    shutil.rmtree(frozen_dir, ignore_errors=True)
    frozen_dir.mkdir(parents=True)
    frozen = frozen_dir / source.name
    shutil.copy2(source, frozen)
    matrix = pd.read_parquet(frozen)
    # Exactly the five fields lineage.validate_matrix_manifest checks on the frozen path.
    manifest = {
        "builder_version": MATRIX_BUILDER_VERSION,
        "schema_version": MATRIX_SCHEMA_VERSION,
        "row_count": len(matrix),
        "feature_schema": list(matrix.columns),
        "matrix_sha256": file_sha(frozen),
    }
    frozen.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return frozen_dir


def _run_meditate(league: str, market: str, candidate: dict) -> str:
    """Full-HPO retrain of a cell from its just-persisted stat_meta strategy; "" iff meditate exits clean.

    Trains on the walk's pinned matrix (``--frozen-matrix-dir``, see :func:`_pin_cell_matrix`) so
    every nominee of a cell scores the same frame. ``--force`` is still required: a frozen-input
    run appends no new rows, and without it a cell whose pickle already exists skips silently and
    never rewrites its outputs. A failure returns its :func:`_failure_reason` rather than a bare
    flag — a native abort inside the fit and an ordinary non-zero exit call for completely
    different triage, and reporting both as one "retrain error" is what sent the last investigation
    looking for a Python exception that never existed. The caller must not trust a possibly-stale
    model_stats row either way; a transient archive-lock clash is retried first
    (:func:`_run_meditate_with_lock_retry`).
    """
    spec = get_strategy(candidate["strategy_slug"])
    context = _cell_context(league, market)
    cmd = meditate_command(
        league,
        market,
        "--force",
        "--frozen-matrix-dir",
        str(_frozen_matrix_dir(league, market)),
        *strategy_full_hpo_cli_args(context, spec, candidate["controls"]),
    )
    log_path = _CONFIRM_LOG_ROOT / f"{market_file_slug(league, market)}.log"
    expected = _confirm_median_seconds(league, market)
    click.echo(f"  retraining {league} {market} — full HPO, {_expectation(expected)} …")
    started = time.monotonic()
    try:
        with elapsed_ticker(f"  {league} {market} full-HPO retrain", expected):
            _run_meditate_with_lock_retry(cmd, log_path, timeout=_CONFIRM_TIMEOUT_S)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return _failure_reason(exc)
    finally:
        # Carried on the candidate rather than a module global: the same dict already threads the
        # walk, and _record_nominee_gates is the next thing to read it.
        candidate["elapsed_s"] = round(time.monotonic() - started, 1)
    click.echo(f"  retrained {league} {market} in {human_seconds(candidate['elapsed_s'])}")
    return ""


def _expectation(expected: float | None) -> str:
    """How long this cell's retrain should take, and the wall-clock time that lands at.

    A redirected run gets no ticker, so the finish time is what makes a stalled retrain obvious in a
    log — an elapsed count would need the reader to know when the line was written.
    """
    if expected is None:
        return "no timing history for this cell"
    finish = time.strftime("%H:%M", time.localtime(time.time() + expected))
    return f"~{human_seconds(expected)}, expect ~{finish}"


def _confirm_median_seconds(league: str | None = None, market: str | None = None) -> float | None:
    """Median full-HPO retrain seconds for one cell from the nominee ledger, else across all cells.

    Naming no cell asks for the all-cells median, which is what sizes the up-front confirm prompt.

    Each ledger row now times exactly one meditate subprocess (a calibrated retry records its own
    row), but a retrain that errored never reaches the ledger, so the estimate is blind to the
    slowest failures — hence it is always shown as approximate. Median guards the estimate against
    that skew.
    """
    if not NOMINEE_LEDGER_PATH.exists():
        return None
    ledger = pd.read_csv(NOMINEE_LEDGER_PATH)
    if "elapsed_s" not in ledger.columns:
        return None
    ledger = ledger.assign(elapsed_s=pd.to_numeric(ledger["elapsed_s"], errors="coerce")).dropna(
        subset=["elapsed_s"]
    )
    if ledger.empty:
        return None
    cell = ledger[(ledger["league"] == league) & (ledger["market"] == market)]
    scope = ledger if league is None or cell.empty else cell
    return float(scope["elapsed_s"].median())


def _candidate_identity(candidate: dict, matrix_hash: str | None = None) -> ArtifactIdentity:
    """The nominated recipe as an identity, rebound to ``matrix_hash`` when a retrain moved the matrix.

    ``_run_meditate`` passes ``--force``, which updates the league gamelogs and appends gamedays, so a
    confirm retrain routinely lands on a different training matrix than the search scored — every cell
    of the first overnight run did. What confirmation tests is the *recipe*; the board's pre-retrain
    matrix hash is stale by construction, and the corner fingerprint binds it, so comparing either
    against the retrain rejects recipes the six gates just passed. The same-retrain invariant that
    actually matters — pickle and model_stats agreeing — is enforced in :func:`_produced_artifacts_match`.
    """
    matrix = candidate["matrix_hash"] if matrix_hash is None else matrix_hash
    return ArtifactIdentity(
        strategy_slug=candidate["strategy_slug"],
        structural_strategy=candidate["structural_strategy"],
        signature=candidate["strategy_signature"],
        implementation_version=candidate["strategy_implementation_version"],
        artifact_schema_version=candidate["artifact_schema_version"],
        league=candidate["league"],
        market=candidate["market"],
        status=candidate["strategy_status"],
        controls_json=controls_json(candidate["controls"]),
        corner_fingerprint=(
            candidate["corner_fingerprint"]
            if matrix_hash is None
            else corner_fingerprint(
                get_strategy(candidate["strategy_slug"]), candidate["controls"], matrix
            )
        ),
        matrix_hash=matrix,
        split_fingerprint=candidate["split_fingerprint"],
    )


def _retrained_matrix_hash(league: str, market: str) -> str | None:
    """The matrix hash the just-finished retrain reported, or ``None`` when it wrote no row."""
    try:
        row = _cell_row(league, market, ["league", "market", "strategy_matrix_hash"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if row is None or not isinstance(row["strategy_matrix_hash"], str):
        return None
    return row["strategy_matrix_hash"]


def _produced_artifacts_match(
    league: str, market: str, candidate: dict, expected: ArtifactIdentity
) -> bool:
    """Fail closed unless the retrain produced the exact signed board strategy and cell.

    ``expected`` carries the matrix hash model_stats reported, so a pickle written against a different
    matrix — a stale artifact the retrain never replaced — fails here.
    """
    spec = get_strategy(candidate["strategy_slug"])
    csv_path = _TEST_SETS_ROOT / f"{market_file_slug(league, market)}.csv"
    try:
        frame = pd.read_csv(csv_path, keep_default_na=False)
        model = pd.read_pickle(model_pickle_path(league, market))
        actual = validate_strategy_artifacts(
            spec,
            candidate["controls"],
            frame,
            model,
            league=league,
            market=market,
            matrix_hash=expected.matrix_hash,
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return actual == expected


def _gate_passed(value) -> bool:
    """NA-safe read of a nullable-boolean gate cell (``bool(pd.NA)`` raises)."""
    return pd.notna(value) and bool(value)


def _g4_only_retry_wanted(
    row: pd.Series | None, cell_hpo_selection: str, cell_zinb_mode: str
) -> bool:
    """Whether a cell earns the one-shot calibrated-HPO retry.

    Fires only for cells trained under ``loss`` selection whose fresh
    model_stats row failed ship on a near-miss Gate 4, alone or with a
    within-band Gate 1 — the dispersion-calibration failures calibrated trial
    selection targets. Any other failing gate blocks the retry, g6 explicitly
    included (deferred pending the Experiment C measurement, brief R3).
    Hurdle-ZINB and Mixture are excluded: those paths have no calibrated
    trial-selection closure.
    """
    if row is None or cell_hpo_selection != "loss":
        return False
    if row["distribution"] == "Mixture":
        return False
    if row["distribution"] == "ZINB" and cell_zinb_mode == "hurdle":
        return False
    if _gate_passed(row["ship"]) or _gate_passed(row["g4_pass"]):
        return False
    if not row["g4_pit_ks"] - row["g4_pit_ks_max"] <= _G4_RETRY_MAX_EXCESS:
        return False
    return _cofailures_within_band(row)


def _cofailures_within_band(row: pd.Series) -> bool:
    """Non-g4 gate failures are admissible only as exactly {g1} inside its noise band."""
    others_failing = [c for c in _GATE_COLS if c != "g4_pass" and not _gate_passed(row[c])]
    if others_failing == ["g1_pass"]:
        return bool(row["g1_brier_diff_ci_hi"] - _GATE1_NONINF_MARGIN <= _G1_RETRY_NOISE_BAND)
    return not others_failing


def _retry_calibrated_wanted(candidate: dict, cell: dict) -> bool:
    """Whether the just-retrained nominee earns the walk's calibrated-HPO fallback.

    ``cell`` is the nominee's stat_meta entry *after* its edits — exactly what the meditate
    subprocess resolved its knobs from. Structural corners never retry (their spec pins
    ``hpo_selection``, and the structural stage has no calibrated trial-selection closure).
    """
    if get_strategy(candidate["strategy_slug"]).is_structural:
        return False
    try:
        row = _cell_row(candidate["league"], candidate["market"], _RETRY_COLUMNS)
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return _g4_only_retry_wanted(
        row, cell.get("hpo_selection", "loss"), cell.get("zinb_mode", "joint")
    )


def _pin_calibrated(meta: dict, candidate: dict) -> None:
    """Pin ``hpo_selection: "calibrated"`` for the retry — in stat_meta (the rerun subprocess
    reads it back per-cell, the same path the weekly cron uses) and in the nominee's edits (so
    the persisted diff, the supersede promote prompt, and the report tell the truth).
    """
    lg, mkt = candidate["league"], candidate["market"]
    click.echo(f"  {lg} {mkt}: g4-only ship fail — retrying with hpo_selection=calibrated")
    meta[lg][mkt]["hpo_selection"] = "calibrated"
    candidate["edits"]["hpo_selection"] = "calibrated"
    _atomic_write_meta(meta)


def _retrain_with_calibrated_retry(
    league: str, market: str, candidate: dict, meta: dict
) -> tuple[str, bool]:
    """Retrain a nominee, with the walk's one-shot calibrated fallback on a g4-only near-miss.

    Runs the nominee's meditate and records its ledger row; when the fresh model_stats row earns
    the retry (:func:`_retry_calibrated_wanted`), pins the knob and reruns the identical command,
    recording the retry as its own ledger row. Returns ``(error, diverged)`` for the run whose
    model_stats row stands; a failed retry leaves the pin for the caller's revert to strip.
    """
    error = _run_meditate(league, market, candidate)
    if error:
        return error, False
    diverged = _record_nominee_gates(league, market, candidate)
    if not _retry_calibrated_wanted(candidate, meta[league][market]):
        return "", diverged
    _pin_calibrated(meta, candidate)
    retry = {**candidate, "source": f"{candidate['source']} +calibrated-retry"}
    error = _run_meditate(league, market, retry)
    if error:
        return error, diverged
    return "", _record_nominee_gates(league, market, retry)


def _confirm_meditate(league: str, market: str, candidate: dict, meta: dict) -> list[str]:
    """Withheld-path confirm: retrain, then the reasons the cell did not ship (empty means it did).

    Reasons rather than a bool because the legs fail for very different operator-facing causes, and a
    bare False reads on the report as a gate failure with no gate named.
    """
    error, diverged = _retrain_with_calibrated_retry(league, market, candidate, meta)
    if error:
        return [f"retrain {error}"]
    # A diverged fit is named ahead of whatever it fails: the gates already reject it, but
    # "diverged g1 g4" sends triage to the fit rather than the anonymous gate list.
    prefix = ["diverged"] if diverged else []
    matrix_hash = _retrained_matrix_hash(league, market)
    if matrix_hash is None:
        return [*prefix, "no model_stats row"]
    # Gates first: serve-iff-ship leaves no pickle behind for a cell that failed one, so checking
    # artifact identity ahead of them reports a missing serving artifact as the cause of a plain
    # Gate-4 miss. The two identity legs still gate the ship — they just answer a later question.
    failed = _failed_gates_after(league, market)
    if failed:
        return prefix + failed
    expected = _candidate_identity(candidate, matrix_hash)
    if not _produced_artifacts_match(league, market, candidate, expected):
        return [*prefix, "artifact identity"]
    if not _ship_from_model_stats(league, market, expected):
        return [*prefix, "model_stats identity/ship"]
    return []


def _confirm_one(meta: dict, cand: dict) -> tuple[str, str, str, list[str]]:
    """Persist one candidate, confirm at full HPO, and keep it (devel) or revert (stat_meta + pickle).

    The pickle prune on failure is what actually dark-outs the cell — inference loads pickles by
    path and ignores ``shipped``, so a reverted stat_meta entry alone would still serve.
    """
    lg, mkt = cand["league"], cand["market"]
    original = deepcopy(meta[lg][mkt])
    backup = _snapshot_cell(lg, mkt)
    keep = False
    try:
        meta[lg][mkt].update(cand["edits"])
        meta[lg][mkt]["shipped"] = _SHIPPED_DEVEL
        _atomic_write_meta(meta)
        failed = _confirm_meditate(lg, mkt, cand, meta)
        if not failed:
            keep = True
            click.secho(f"  SHIPPED (devel) {lg} {mkt}", fg="green")
            return (lg, mkt, "SHIPPED", [])
        click.secho(f"  REVERTED {lg} {mkt} — failed {' '.join(failed)}", fg="red")
        return (lg, mkt, "REVERTED", failed)
    finally:
        if not keep:
            _restore_cell(lg, mkt, backup, meta, original)
            prune_model_pickle(lg, mkt)


def _failed_legs(verdict: dict) -> list[str]:
    """The supersession legs a HOLD verdict failed (for the report's failed-gates column)."""
    return [leg for leg in ("S1", "S2", "S3") if not verdict[f"{leg.lower()}_pass"]]


def _supersede_one(
    meta: dict, cand: dict, *, auto_promote: bool = False
) -> tuple[str, str, str, list[str]]:
    """Supersession-test one live cell: snapshot, retrain the candidate in place, require its exact
    artifact identity and official six-gate ``model_stats`` ship, then run S1/S2/S3 and promote it
    (on a passing verdict + operator yes) or restore the incumbent byte-identical.

    A live-cell swap needs the test to pass AND an explicit promote confirmation; every other exit —
    HOLD, decline, retrain error, or an exception — hits the ``finally`` restore, which copies the
    snapshot back and never prunes, so the incumbent keeps serving. ``auto_promote`` answers that
    confirmation yes so an unattended run can swap live cells: the S1/S2/S3 verdict and the six
    gates still decide, only the human veto is gone. Every swap is still local and uncommitted, so
    the review of the ``stat_meta.json`` diff is where a promotion is actually accepted.
    """
    lg, mkt = cand["league"], cand["market"]
    slug = market_file_slug(lg, mkt)
    original = deepcopy(meta[lg][mkt])
    backup = _snapshot_cell(lg, mkt)
    keep = False
    try:
        meta[lg][mkt].update(cand["edits"])  # shipped left as-is; the cell stays live
        _atomic_write_meta(meta)
        error, diverged = _retrain_with_calibrated_retry(lg, mkt, cand, meta)
        if error:
            return (lg, mkt, "HELD", [f"retrain {error}"])
        prefix = ["diverged"] if diverged else []
        matrix_hash = _retrained_matrix_hash(lg, mkt)
        if matrix_hash is None:
            return (lg, mkt, "HELD", [*prefix, "no model_stats row"])
        expected = _candidate_identity(cand, matrix_hash)
        if not _produced_artifacts_match(lg, mkt, cand, expected):
            return (lg, mkt, "HELD", [*prefix, "artifact identity"])
        if not _ship_from_model_stats(lg, mkt, expected):
            return (lg, mkt, "HELD", [*prefix, "model_stats identity/ship"])
        baseline = load_test_set(backup / f"{slug}.csv", _SHIP_PRED_COL)
        candidate = load_test_set(_TEST_SETS_ROOT / f"{slug}.csv", _SHIP_PRED_COL)
        verdict = supersede_verdict(baseline, candidate, _SHIP_PRED_COL, league=lg, market=mkt)
        click.echo("  " + _supersede_headline(verdict))
        if not verdict["ship"]:
            return (lg, mkt, "HELD", prefix + _failed_legs(verdict))
        edits = ", ".join(f"{k}={v}" for k, v in cand["edits"].items())
        if auto_promote:
            click.secho(f"  auto-promoting {lg} {mkt} to {edits}", fg="yellow")
        elif not click.confirm(f"  Promote {lg} {mkt} to {edits}?"):
            return (lg, mkt, "HELD", ["declined"])
        keep = True
        return (lg, mkt, "SUPERSEDED", [])
    finally:
        if not keep:
            _restore_cell(lg, mkt, backup, meta, original)


def _split_shippable(
    ready: list[list[dict]], meta: dict, *, fresh_only: bool = False
) -> tuple[list[list[dict]], list[list[dict]]]:
    """Partition *cells* by current release surface: withheld (fresh) vs already-shipped (live).

    Withheld cells auto-ship on a clean 6/6 (a fresh cell has no live pickle, so a failed confirm's
    revert+prune restores its dark state). Live cells route to the supersession test, which restores
    the incumbent byte-identical on a loss rather than pruning it. ``fresh_only`` drops the live
    lane entirely: supersession promotions always prompt per cell, so an unattended run would burn
    each live cell's retrains into guaranteed HELDs.
    """
    fresh, shipped = [], []
    for nominated in ready:
        lg, mkt = nominated[0]["league"], nominated[0]["market"]
        target = fresh if meta[lg][mkt].get("shipped") == WITHHELD else shipped
        target.append(nominated)
    if fresh_only and shipped:
        click.secho(
            f"  fresh-only: skipping {len(shipped)} live cell(s) — supersession promotions "
            "prompt per cell, so they need an attended run.",
            fg="yellow",
        )
        shipped = []
    return fresh, shipped


def _drop_activation_gated(fresh: list[list[dict]]) -> list[list[dict]]:
    """Announce and drop withheld cells in activation-gated leagues — they never auto-ship."""
    kept = []
    for nominated in fresh:
        lg, mkt = nominated[0]["league"], nominated[0]["market"]
        if lg in _ACTIVATION_GATED_LEAGUES:
            click.secho(
                f"  ACTIVATION-GATED {lg} {mkt} — withheld {lg} cells ship "
                "only after the D1/D2 owner gate; skipping.",
                fg="yellow",
            )
        else:
            kept.append(nominated)
    return kept


def _announce_plan(fresh: list[list[dict]], shipped: list[list[dict]]) -> None:
    """List the run's plan: each cell's nominees in the order they will be tried, with their source."""
    for label, group, verb in (
        ("withheld cell", fresh, "persist + confirm"),
        ("live cell", shipped, "supersession-test (S1/S2/S3)"),
    ):
        if group:
            click.secho(f"\n{len(group)} {label}(s) to {verb}:", bold=True)
            for nominated in group:
                head = nominated[0]
                click.echo(f"  {head['league']} {head['market']} — {len(nominated)} nominee(s):")
                for n, cand in enumerate(nominated, start=1):
                    edits = ", ".join(f"{k}={v}" for k, v in cand["edits"].items())
                    click.echo(wrapped(f"{n}. [{cand['source']}] {edits}", "    ", "       "))


def _walk_nominees(
    meta: dict, nominated: list[dict], attempt
) -> tuple[str, str, str, list[str], str]:
    """Retrain each nominee in order until one wins; report the deciding attempt and how deep it went.

    The cell's training matrix is pinned once here (:func:`_pin_cell_matrix`), so every nominee
    retrains and scores on the same frame. No new revert machinery is needed: ``_confirm_one``'s
    ``finally`` already restores stat_meta and prunes the pickle on every non-win, and
    ``_supersede_one``'s restores byte-identically. The loop just stops at the first win.

    Nomination already drops ledger-decided corners (brief L4), but a concurrent walk can append a
    verdict between selection and this loop, so a decided nominee is still skipped here — without
    letting a trailing skip overwrite the verdict of a corner that actually ran.
    """
    outcome = ("", "", "REVERTED", ["no nominee"], "-")
    _pin_cell_matrix(nominated[0]["league"], nominated[0]["market"])
    decided = _decided_pairs()
    walked = False
    for n, cand in enumerate(nominated, start=1):
        click.secho(
            f"\n  nominee {n}/{len(nominated)} [{cand['source']}] {cand['league']} {cand['market']}",
            bold=True,
        )
        attempt_label = f"{n}/{len(nominated)} {cand['source']}"
        if decided and (str(cand["corner_fingerprint"]), str(cand["matrix_hash"])) in decided:
            click.secho(
                "    ledger already holds this corner's verdict on this matrix — skipping",
                fg="yellow",
            )
            if not walked:
                outcome = (
                    cand["league"],
                    cand["market"],
                    "SKIPPED",
                    ["prior verdict on this matrix"],
                    attempt_label,
                )
            continue
        walked = True
        lg, mkt, verdict, failed = attempt(meta, cand)
        outcome = (lg, mkt, verdict, failed, attempt_label)
        if verdict in _WIN_OUTCOMES:
            break
    return outcome


def _walk_lanes(
    meta: dict,
    fresh: list[list[dict]],
    shipped: list[list[dict]],
    deadline_hours: float | None,
    auto_promote: bool = False,
) -> list[tuple[str, str, str, list[str], str]]:
    """Walk both lanes cell by cell; past the deadline, remaining cells record SKIPPED instead.

    The deadline clock starts here — after the operator prompt — and is checked between cells, never
    mid-walk: a cell that starts before the deadline finishes its walk, so the budget can overrun by
    at most one cell. :func:`_candidates` ordered the cells best-first, which is what makes the cut
    land on the weakest tail.
    """
    supersede = functools.partial(_supersede_one, auto_promote=auto_promote)
    deadline = None if deadline_hours is None else time.monotonic() + deadline_hours * 3600
    results = []
    for lane, attempt in ((fresh, _confirm_one), (shipped, supersede)):
        for nominated in lane:
            if deadline is not None and time.monotonic() > deadline:
                lead = nominated[0]
                results.append((lead["league"], lead["market"], "SKIPPED", ["deadline"], "-"))
                continue
            results.append(_walk_nominees(meta, nominated, attempt))
    return results


def run_confirm(
    board: pd.DataFrame,
    *,
    yes: bool = False,
    max_nominees: int | None = None,
    deadline_hours: float | None = None,
    fresh_only: bool = False,
    auto_promote: bool = False,
) -> None:
    """Confirm the sweep's winners: auto-ship withheld cells on a clean 6/6, supersession-test live cells.

    ``board`` is the in-memory sweep result (a row per corner). Each cell nominates its top corners
    plus any seed and its incumbent; the nominees are retrained in order until one ships. Withheld
    cells are persisted + retrained and kept on a clean 6/6 (else reverted+pruned). Already-shipped
    cells (present when the sweep ran ``--include-shipped``) run the S1/S2/S3 test and swap the live
    cell only on a passing verdict AND an operator yes; any loss restores the incumbent. ``yes``
    skips only the upfront gate — live-cell promotions prompt individually unless ``auto_promote``
    answers them. ``max_nominees`` caps each cell's walk at its first N nominees;
    ``deadline_hours`` skips cells not yet started when the budget runs out; ``fresh_only`` drops
    the live lane for unattended runs.
    """
    meta = load_stat_meta(_STAT_META)
    ready = _candidates(board, meta, max_nominees)
    if not ready:
        click.echo("no confirmable nominees on the board.")
        return

    fresh, shipped = _split_shippable(ready, meta, fresh_only=fresh_only)
    fresh = _drop_activation_gated(fresh)
    if not fresh and not shipped:
        click.echo("no confirmable candidates after the activation gate.")
        return
    _announce_plan(fresh, shipped)
    retrains = sum(len(n) for n in fresh) + sum(len(n) for n in shipped)
    each = _confirm_median_seconds()
    budget = f" (~{human_seconds(each)} each, <={human_seconds(retrains * each)})" if each else ""
    prompt = (
        f"\nConfirm {len(fresh)} withheld and supersession-test {len(shipped)} live cell(s) — up to "
        f"{retrains} full-HPO retrains{budget}, stopping at each cell's first shipping nominee? "
        "Withheld failures auto-revert; live promotions prompt"
    )
    if not yes and not click.confirm(prompt):
        click.echo("aborted; stat_meta.json unchanged.")
        return

    backup = _backup_stat_meta()
    click.echo(f"stat_meta.json backed up to {backup}")
    _print_confirm_report(_walk_lanes(meta, fresh, shipped, deadline_hours, auto_promote), backup)


def _print_confirm_report(
    results: list[tuple[str, str, str, list[str], str]], backup: pathlib.Path
) -> None:
    """Final table (cell → nominee → outcome → failing gates) + tally, backup path, commit reminder."""
    rows = [
        [f"{lg} {mkt}", nominee, outcome, " ".join(failed) or "-"]
        for lg, mkt, outcome, failed, nominee in results
    ]
    click.secho("\nconfirm results", bold=True)
    click.echo(
        tabulate.tabulate(
            rows, headers=["cell", "nominee", "outcome", "failed gates"], tablefmt="github"
        )
    )
    n_win = sum(1 for r in results if r[2] in _WIN_OUTCOMES)
    n_skip = sum(1 for r in results if r[2] == "SKIPPED")
    skipped = f", {n_skip} skipped" if n_skip else ""
    click.secho(
        f"\n{n_win} shipped/superseded (devel), {len(results) - n_win - n_skip} reverted/held"
        f"{skipped}. Backup: {backup}",
        fg="green" if n_win else "yellow",
    )
    if n_win:
        click.echo("Review the stat_meta.json diff and commit the shipped/superseded cells.")
