"""Confirm-walk nomination: which corners each cell's walk tries, and in what order.

:func:`candidates` reads the sweep board into one nominee list per cell, strongest cells first, for
:func:`sportstradamus.training.model_strategy.confirm.run_confirm` to walk: the cell's top board
corners chosen for mechanism diversity, then its seed corners, then its incumbent. Each board row's
signed identity is checked on the way in, so a stale or mismatched row raises rather than walks.
"""

import math

import pandas as pd

# _cell_context and NOMINEE_LEDGER_PATH are read off sweep at call time, not imported: other
# confirm modules read them too, and one binding means one monkeypatch covers every caller.
from sportstradamus.training.model_strategy import sweep
from sportstradamus.training.model_strategy.identity import (
    ArtifactIdentity,
    build_artifact_identity,
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
    parse_controls,
    strategies_for_cell,
    strategy_controls,
    strategy_persistence_edits,
)
from sportstradamus.training.model_strategy.sweep import EVAL_SPLIT_CROSSFIT, _seed_corners
from sportstradamus.training.posthoc import PROB_STAGE
from sportstradamus.training.ship_config import WITHHELD

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
_CONFIRM_CAPABILITIES = frozenset({CAP_CONFIRM, CAP_FULL_HPO, CAP_SCORE, CAP_SERVE})
# Board rows a cell nominates for full-HPO confirmation. Deterministic fixed-HP scoring does not
# ship the recipes that only pass under real HPO, so nominating the top few — rather than requiring
# a deterministic `ships` — is what makes confirmation reachable at all for the popular NFL passing
# markets, whose boards carry zero shipping rows.
_CONFIRM_TOP_K: int = 3
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


def _rank_column(admissible: pd.DataFrame, *, live: bool) -> str:
    """The board value that decides this cell's nominations.

    A withheld cell is ranked on absolute gate headroom, because passing the gates outright is
    exactly the bar it has to clear. A live cell's bar is a different question — is this corner
    better than the recipe already serving? — so it ranks on ``margin_vs_incumbent``. Ranking a
    supersession on slack measures the candidate against the book and the gates while never
    comparing it to the incumbent it would replace. ``live`` is only ever set on a board that
    carries margins at all (see :func:`candidates`), so the column is present whenever it is read.
    """
    if live:
        return "margin_vs_incumbent"
    return "discounted_slack" if "discounted_slack" in admissible.columns else "slack"


def _nominees(
    sub: pd.DataFrame, *, live: bool = False, min_model_weight: float | None = None
) -> list[dict]:
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
    ``board_rank`` so :func:`candidates` can walk the strongest cells first.
    """
    lg, mkt = sub["league"].iloc[0], sub["market"].iloc[0]
    context = sweep._cell_context(lg, mkt)
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
        lane_rows = _veto_admissible(
            ranked, rank_column, live=live, min_model_weight=min_model_weight
        )
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
    # First nominee per corner wins, so a seed repeating a board corner retrains once, as its row.
    by_fingerprint = {}
    for cand in nominated:
        by_fingerprint.setdefault(cand["corner_fingerprint"], {**cand, "board_rank": board_rank})
    return list(by_fingerprint.values())


def _veto_admissible(
    ranked: pd.DataFrame, rank_column: str, *, live: bool, min_model_weight: float | None = None
) -> pd.DataFrame | None:
    """The rows eligible for lane slots, or ``None`` when the whole cell fails the nomination bar.

    brief R3: admissibility is priced at the milder median ``veto_slack`` while the ordering (and
    ``board_rank``, the walk order) stays on the conservative rank column, so a cell whose corners
    fail only under the conservative price still walks — last. Boards that predate the veto
    pricing, and the live lane's margin, keep the rank column as the bar. brief L4: a corner the
    ledger has already decided on this matrix cannot usefully walk again — the walk would skip
    it — so it never spends a lane slot either.

    ``min_model_weight`` drops book-riding corners from the lane (why: :func:`confirm.run_confirm`).
    It is a lane filter, not a cell veto: a cell whose top corner rides the book still nominates
    its remaining positive corners, and when none remain, its seeds and incumbent.
    """
    veto = ranked[rank_column]
    if not live and "veto_slack" in ranked.columns:
        veto = pd.to_numeric(ranked["veto_slack"], errors="coerce").fillna(veto)
    if not veto.max() > 0:
        return None
    ranked = ranked[veto > 0]
    decided = decided_pairs()
    if decided:
        pairs = zip(
            ranked["corner_fingerprint"].astype(str),
            ranked["matrix_hash"].astype(str),
            strict=True,
        )
        ranked = ranked[[pair not in decided for pair in pairs]]
    if min_model_weight is not None:
        # NaN (a board swept before the weight column existed) and exactly 1.0 (a cell with no
        # authentic validation quotes to blend against) are not evidence of a book-rider.
        weight = pd.to_numeric(ranked["model_weight"], errors="coerce")
        ranked = ranked[~(weight < min_model_weight)]
    return ranked


def decided_pairs() -> set[tuple[str, str]]:
    """Every ``(corner_fingerprint, matrix_hash)`` the ledger holds a full-HPO row for.

    Blind and test-half rows alike: either run decided the corner on that matrix.
    """
    if not sweep.NOMINEE_LEDGER_PATH.exists():
        return set()
    ledger = pd.read_csv(sweep.NOMINEE_LEDGER_PATH)
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
    while remaining and len(lane) < _CONFIRM_TOP_K:
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


def candidates(
    board: pd.DataFrame,
    meta: dict,
    max_nominees: int | None = None,
    *,
    min_model_weight: float | None = None,
) -> list[list[dict]]:
    """Each cell's ordered nominee list, strongest cells first; nothing confirmable drops out.

    ``max_nominees`` truncates each list after dedup, trading the tail of a cell's walk for wall
    clock. The ordering :func:`_nominees` establishes is what makes that trade sound — the corners
    most likely to ship come first. ``meta`` decides each cell's lane, which decides what its
    nominees are ranked on.

    Cells sort by their best board rank descending (legacy seed-only cells last) so a deadline cut
    lands on the weakest tail, not on cells the board likes that happened to group late. The two
    lanes rank on different quantities, but a stable sort over the mixed list still leaves each
    lane correctly ordered within itself, which is all :func:`confirm._walk_lanes` walks.

    Whether a board carries incumbent margins is a property of the sweep that wrote it, so it is
    decided once here rather than per cell: on a board swept before margins existed every cell keeps
    the old slack ranking, and on one swept with them a live cell whose own baseline never scored
    gets ``NaN`` — unknown — instead of silently falling back to the wrong quantity.
    """
    priced = "margin_vs_incumbent" in board and board["margin_vs_incumbent"].notna().any()
    cells = []
    for (league, market), sub in board.groupby(["league", "market"], sort=False):
        live = priced and meta[league][market].get("shipped") != WITHHELD
        nominated = _nominees(sub, live=live, min_model_weight=min_model_weight)
        if nominated:
            cells.append(nominated[:max_nominees])
    cells.sort(key=lambda nominated: nominated[0]["board_rank"], reverse=True)
    return cells
