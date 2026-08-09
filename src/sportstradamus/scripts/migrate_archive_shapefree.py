"""One-time migration: backfill the shape-free ``(under_prob, line)`` columns on ``odds``.

For every player-prop ``odds`` row with a NULL ``under_prob``, store:

* ``line``       — the row's consensus line (``floor(2*median(distinct line))/2`` over the
                   ``lines`` table for its ``league/market/game_date/entity``, the exact
                   :meth:`Archive.get_line` semantics), and
* ``under_prob`` — ``get_odds(line, ev, <current per-cell shape>)``: the de-vigged
                   under-probability the stored ``ev`` implies under the cell's shape.

Lossless for the WS1 read round-trip by construction — ``get_ev`` is the exact numerical
inverse of ``get_odds`` — so re-encoding the stored ``under_prob`` at the same shape
reproduces ``ev`` (asserted on a per-cell sample *before* the row is written). Idempotent:
only NULL-``under_prob`` rows are touched. Team-only markets and cells with no configured
distribution are skipped — their readers keep falling back to ``ev``. Rows with no usable
line in the ``lines`` table are left NULL.

Run on a COPY of the archive first::

    poetry run python -m sportstradamus.scripts.migrate_archive_shapefree --db-path /path/to/copy.duckdb
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import click
import duckdb
import numpy as np
import pandas as pd
from tqdm import tqdm

from sportstradamus.helpers.archive import _TEAM_ONLY_MARKETS
from sportstradamus.helpers.config import book_gate, stat_cv, stat_dist, stat_std
from sportstradamus.helpers.distributions import SN_MAX_MEAN_FACTOR, get_ev, get_odds

_DEFAULT_DB_PATH = Path("archive/archive.duckdb")
_ROUNDTRIP_SAMPLE = 256
_ROUNDTRIP_TOL = 1e-6
# get_ev brackets its inversion on [1e-6, max(SN_MAX_MEAN_FACTOR*line, 1)]; mirror its mean
# floor so the invertibility bracket matches exactly (distributions.py get_ev).
_MEAN_FLOOR = 1e-6
# Local-injectivity probe: perturb the decoded mean by this relative step; if the book's
# under-prob moves less than _SLOPE_MIN_DELTA, get_odds is locally flat (a near-degenerate cv
# collapses a whole mean range onto one under) and the quote is not recoverable.
_SLOPE_REL_STEP = 1e-3
_SLOPE_MIN_DELTA = 1e-9
# Mirror of training.pipeline._SKEWNORMAL_CV_FLOOR: SkewNormal cvs are clamped UP to this floor.
# A cv sitting at the floor was clamped, not computed, so it is not the cv the ev was encoded at
# — its decode is distrusted in favour of the std-recovered cv (the floored cells, e.g. MLB
# doubles, would otherwise partial-migrate at the degenerate floor). sogBS (cv 0.052, above the
# floor) is not caught here — it is rescued by the strict round-trip failing instead.
_SKEWNORMAL_CV_FLOOR = 0.02


def _consensus_lines(lines_df: pd.DataFrame) -> pd.Series:
    """:meth:`Archive.get_line` consensus per ``(game_date, entity)``: floor-½ of the median."""
    return (
        lines_df.drop_duplicates(["game_date", "entity", "line"])
        .groupby(["game_date", "entity"])["line"]
        .apply(lambda s: float(np.floor(2 * np.median(s)) / 2))
        .rename("line")
    )


def _decode_under(
    line: np.ndarray, ev: np.ndarray, dist: str, cv: float, gate: float, step: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Decode ``under_prob`` at one shape ``cv``.

    Returns ``(valid_mask, under[valid], roundtrip_err)`` where ``roundtrip_err`` is the max
    abs ``get_ev(get_odds(ev)) - ev`` over a sample of the surviving rows (``inf`` if none
    survive). ``valid_mask`` drops every row whose stored ``ev`` has no faithful devig under
    this shape:

    * **Invertibility bracket** — ``get_ev`` only brentq-inverts an under-prob strictly between
      the prob at its mean floor (``1e-6``) and at its cap ``max(SN_MAX_MEAN_FACTOR*line, 1)``;
      outside it clamps, so DFS-clamped / runaway rows (``ev`` above the cap) and degenerate
      near-zero means (below the floor) are unrecoverable.
    * **Local injectivity** — a near-degenerate cv makes ``get_odds`` a step function in ``ev``
      (the half-point window swallows the near-spike, collapsing a whole mean range onto one
      under), so require a resolvable slope: perturbing the mean must move under.

    Decode uses ``get_ev``'s own ``step`` (1.0 for count families, else 0.5) so ``under_prob`` is
    its exact inverse — the half-point bin width changes the SkewNormal CDF.
    """
    under = np.asarray(get_odds(line, ev, dist, cv=cv, gate=gate, step=step), dtype=float)
    p_floor = np.asarray(
        get_odds(line, np.full_like(line, _MEAN_FLOOR), dist, cv=cv, gate=gate, step=step),
        dtype=float,
    )
    p_cap = np.asarray(
        get_odds(
            line, np.maximum(SN_MAX_MEAN_FACTOR * line, 1.0), dist, cv=cv, gate=gate, step=step
        ),
        dtype=float,
    )
    under_pert = np.asarray(
        get_odds(line, ev * (1.0 + _SLOPE_REL_STEP), dist, cv=cv, gate=gate, step=step), dtype=float
    )
    valid = (
        np.isfinite(under)
        & (under > p_cap)
        & (under < p_floor)
        & (np.abs(under - under_pert) > _SLOPE_MIN_DELTA)
    )
    if not valid.any():
        return valid, under[valid], float("inf")

    line_v, ev_v, under_v = line[valid], ev[valid], under[valid]
    sample = np.linspace(0, len(under_v) - 1, min(_ROUNDTRIP_SAMPLE, len(under_v))).astype(int)
    err = max(
        abs(get_ev(float(line_v[i]), float(under_v[i]), cv, dist=dist, gate=gate) - float(ev_v[i]))
        for i in sample
    )
    return valid, under_v, err


def _migrate_cell(
    con: duckdb.DuckDBPyConnection, league: str, market: str
) -> tuple[int, float, int]:
    """Backfill one player-prop cell. Returns ``(rows_updated, max_roundtrip_err, rows_dropped)``.

    Decodes ``under_prob`` for every NULL row, drops rows whose stored ``ev`` has no faithful
    devig, verifies the round-trip on a sample, and only then writes. A SkewNormal cell whose
    live cv has been floored retries the decode at the std-recovered cv (see :func:`_decode_under`
    and the candidate-cv comment below); a cell that still fails the round-trip writes nothing.
    """
    dist = stat_dist[league][market]
    cv = stat_cv.get(league, {}).get(market, 1)
    gate = book_gate(league, market, dist)

    odds_df = con.execute(
        "SELECT rowid AS rid, game_date, entity, ev FROM odds "
        "WHERE league=? AND market=? AND under_prob IS NULL AND ev IS NOT NULL",
        [league, market],
    ).fetchdf()
    if odds_df.empty:
        return 0, 0.0, 0
    lines_df = con.execute(
        "SELECT game_date, entity, line FROM lines WHERE league=? AND market=?",
        [league, market],
    ).fetchdf()
    if lines_df.empty:
        return 0, 0.0, 0

    merged = odds_df.merge(_consensus_lines(lines_df), on=["game_date", "entity"], how="inner")
    merged = merged[(merged["line"] > 0) & np.isfinite(merged["ev"]) & (merged["ev"] > 0)]
    if merged.empty:
        return 0, 0.0, 0
    n_candidates = len(merged)

    line = merged["line"].to_numpy(dtype=float)
    ev = merged["ev"].to_numpy(dtype=float)
    step = 1.0 if dist in ("NegBin", "ZINB", "Poisson") else 0.5

    # stat_cv is the live shape and decodes first. Some low-count SkewNormal cells have their cv
    # corrupted in stat_calibration.json long after their ev was written at the real, wider cv —
    # which collapses get_odds onto the half-point spike. Decode them at the std-recovered cv
    # (= std / median ev) instead, since cv == std/mean by construction and the sane std survives.
    # Two corruption signatures, both routed to the recovered cv:
    #   * cv clamped to the floor (e.g. MLB doubles, cv==0.02): the floored cv is distrusted even
    #     when a few window-edge rows round-trip at it, so it cannot partial-migrate at the spike.
    #   * cv merely too small (e.g. NHL sogBS, cv 0.052): the strict round trip fails outright.
    # A non-floored cv that round-trips IS the encoding cv — incl small-but-real ones like NHL
    # shots (0.075) and MLB stolen bases (0.497), whose conservatively-dropped even-money rows just
    # fall back to ev — and is used as-is; the recovery never re-shapes a faithful decode.
    candidates = [cv]
    std = stat_std.get(league, {}).get(market)
    if dist == "SkewNormal" and std and std > 0:
        cv_recovered = float(std / np.median(ev))
        if cv_recovered > cv:
            candidates.append(cv_recovered)
    floored_live = dist == "SkewNormal" and cv <= _SKEWNORMAL_CV_FLOOR

    best_err, best_drop = float("inf"), n_candidates
    for cand in candidates:
        valid, under, err = _decode_under(line, ev, dist, cand, gate, step)
        if err <= _ROUNDTRIP_TOL and not (cand == cv and floored_live):
            sub = merged[valid]
            stg = pd.DataFrame({"rid": sub["rid"].to_numpy(), "u": under, "l": line[valid]})
            con.register("stg_shapefree", stg)
            con.execute(
                "UPDATE odds SET under_prob = stg_shapefree.u, line = stg_shapefree.l "
                "FROM stg_shapefree WHERE odds.rowid = stg_shapefree.rid"
            )
            con.unregister("stg_shapefree")
            return len(sub), err, n_candidates - len(sub)
        if err < best_err:
            best_err, best_drop = err, n_candidates - int(valid.sum())
    # best_err stays inf only when every candidate dropped all rows (no surviving row to round
    # trip) — a clean "all un-invertible" outcome, not an ill-conditioned skip. Report 0.0 so the
    # caller leaves those rows on ev fallback without flagging the cell. A finite best_err means a
    # surviving row failed the round trip: a real skip.
    return 0, (0.0 if best_err == float("inf") else best_err), best_drop


@click.command(help=__doc__)
@click.option(
    "--db-path",
    default=str(_DEFAULT_DB_PATH),
    show_default=True,
    help="Target DuckDB archive file.",
)
@click.option("--no-backup", is_flag=True, help="Skip the .bak copy before mutating the DB.")
def main(db_path: str, no_backup: bool) -> None:
    db = Path(db_path)
    if not db.is_file():
        sys.exit(f"DB file not found: {db}")

    con = duckdb.connect(str(db))
    cols = {
        r[0]
        for r in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name='odds'"
        ).fetchall()
    }
    for col in ("under_prob", "line"):
        if col not in cols:
            con.execute(f"ALTER TABLE odds ADD COLUMN {col} DOUBLE")

    cells = [
        (lg, mk)
        for lg, mk in con.execute(
            "SELECT DISTINCT league, market FROM odds WHERE under_prob IS NULL AND ev IS NOT NULL"
        ).fetchall()
        if mk in stat_dist.get(lg, {}) and mk not in _TEAM_ONLY_MARKETS
    ]
    if not cells:
        con.close()
        print(f"{db}: no un-migrated player-prop cells — nothing to do.")
        return

    if not no_backup:
        backup = db.with_suffix(f".duckdb.bak-{int(time.time())}")
        con.close()
        shutil.copy2(db, backup)
        print(f"Backed up {db} -> {backup}")
        con = duckdb.connect(str(db))

    total, dropped, worst = 0, 0, 0.0
    skipped: list[str] = []
    for lg, mk in tqdm(cells, desc="cells"):
        n, err, n_drop = _migrate_cell(con, lg, mk)
        if err > _ROUNDTRIP_TOL:
            # Ill-conditioned cell the per-row filter could not fully clean (e.g. a
            # near-degenerate cv): write nothing, leave the whole cell on ev fallback,
            # keep going — never abort the run for one bad cell.
            skipped.append(f"{lg}/{mk} (err {err:.1e})")
            continue
        total, dropped, worst = total + n, dropped + n_drop, max(worst, err)

    con.execute("CHECKPOINT")
    con.close()
    print(
        f"Backfilled {total} odds rows across {len(cells) - len(skipped)} cells "
        f"({dropped} un-invertible rows left NULL -> ev fallback). "
        f"Max round-trip abs err {worst:.2e}."
    )
    if skipped:
        print(
            f"Skipped {len(skipped)} ill-conditioned cells (kept on ev fallback): "
            + ", ".join(skipped)
        )


if __name__ == "__main__":
    main()
