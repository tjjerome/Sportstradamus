"""Fit the I6d decode table: under which shape each migrated archive block's ``ev`` was written.

The protocol is frozen in docs/handoffs/i6d-decode-preregistration.md (cited as §n); this
script implements it and may not change a rule without a new pre-registration. It reads prices
only, never an outcome. It opens the archive read-only and writes
``data/config/book_decode.json``, which ``helpers/book_levelling.py`` applies at the training
read. Run it as:

    SPORTSTRADAMUS_ARCHIVE_READ_ONLY=1 poetry run python -m sportstradamus.scripts.fit_book_decode
"""

from __future__ import annotations

import datetime
import importlib.resources as pkg_resources
import json
import math
import os
import subprocess
from collections.abc import Sequence
from functools import partial
from operator import itemgetter
from pathlib import Path

import click
import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from tqdm import tqdm

from sportstradamus import data
from sportstradamus.helpers.book_levelling import (
    EVEN_PRICE_TOLERANCE,
    LADDER_DISAGREEMENT,
    LIVE,
    STORED,
    block_key,
    decode_under,
    is_native,
    ladder_repaired,
    stamp_layer,
)
from sportstradamus.helpers.training_quotes import (
    DFS_PLATFORM_BOOKS,
    ArchivedBookQuote,
    archive_ev_is_usable,
    is_positive,
)

# §9 (owner, 2026-10-27): a decode is level when its validation half sits this close to the
# reference, in mean difference and in median absolute difference.
MEAN_DIFF_MAX = 0.01
MEDIAN_ABS_DIFF_MAX = 0.03
# §8: the one parameter a candidate fits, log cv, is searched inside these bounds.
LOG_CV_BOUNDS = (-5.0, 1.5)
# §9: a fit this close to a bound is not interior, which leaves (-4.999, 1.499).
_INTERIOR_MARGIN = 1e-3
# §8: a fit reads at most this many fit-half rows, drawn with this seed.
_FIT_SAMPLE = 6000
_SEED = 1729
# §6: each half needs this many matched rows for the block to be testable.
MIN_HALF_ROWS = 30
# §7: the candidates after stored and the code-history family, in order.
_GENERIC_FAMILIES = ("SkewNormal", "NegBin", "Poisson", "Gamma")
# §7, §8: the candidates that fit a parameter; stored and Poisson have none.
_PARAMETRIC = frozenset({"SkewNormal", "NegBin", "Gamma"})
# §7: a configured family's decode; any other family skips the code-history candidate.
_DECODE_FAMILY = {
    "ZINB": "NegBin",
    "NegBin": "NegBin",
    "ZAGamma": "Gamma",
    "Gamma": "Gamma",
    "SkewNormal": "SkewNormal",
    "Poisson": "Poisson",
}
# §7: the writer's family for a market its config does not name (7a624294:moneylines.py:490-494).
_UNCONFIGURED_FAMILY = "SkewNormal"
# §7: the committed distribution config over its history, newest path first.
_DIST_CONFIG_PATHS = (
    "src/sportstradamus/data/config/stat_meta.json",
    "src/sportstradamus/data/config/stat_dist.json",
    "src/sportstradamus/data/stat_dist.json",
)
# §7: a backfill block reads the config of the day before ladder capture began (92f918ab).
_BACKFILL_CONFIG_DATE = datetime.date(2026, 6, 17)
# §4: close rows fall after the training read's cutoff, so they are never levelled.
_CLOSE_LAYER = "23:00"
# §5: the native layers a block is referenced against, in the order it tries them.
_REFERENCE_LAYERS = ("01:00", _CLOSE_LAYER)
# §5: a migrated row matches a reference row on these, the line exactly equal.
_KEY = ["game_date", "entity", "book", "line"]

_PREREGISTRATION = "docs/handoffs/i6d-decode-preregistration.md"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_ARCHIVE = os.environ.get("SPORTSTRADAMUS_ARCHIVE_DB") or str(
    _REPO_ROOT / "archive" / "archive.duckdb"
)
_TABLE_PATH = Path(str(pkg_resources.files(data) / "config" / "book_decode.json"))

# §1: every sportsbook row with its same-observation, same-line rung, joined as the training
# read's ``Archive.get_training_quote_inputs`` joins it.
_ROWS_SQL = """
SELECT o.game_date, o.entity, o.book, o.line, o.ev, o.under_prob, o.observed_at, l.p_over
FROM odds o
LEFT JOIN (
    SELECT game_date, entity, book, line, observed_at, AVG(p_over) AS p_over
    FROM ladder WHERE league = $league AND market = $market
    GROUP BY ALL
) l ON l.game_date = o.game_date AND l.entity = o.entity AND l.book = o.book
   AND l.line = o.line AND l.observed_at = o.observed_at
WHERE o.league = $league AND o.market = $market AND NOT list_contains($dfs, o.book)
-- The fit sample is drawn by position, so the row order must not vary between runs.
ORDER BY ALL
"""

DistHistory = Sequence[tuple[datetime.date, dict[str, dict[str, str]]]]


def _load(connection: duckdb.DuckDBPyConnection, league: str, market: str) -> pd.DataFrame:
    """One cell's sportsbook rows with their stamp layer (§2), producer class (§3) and block (§4).

    Every row outside the close layer carries its ``block_layer``/``block``, native or not, so a
    block's ladder share reads beside its migrated rows.
    """
    rows = connection.execute(
        _ROWS_SQL, {"league": league, "market": market, "dfs": sorted(DFS_PLATFORM_BOOKS)}
    ).df()
    rungs = [None if np.isnan(rung) else rung for rung in rows["p_over"]]
    rows["layer"] = [stamp_layer(observed_at) for observed_at in rows["observed_at"]]
    rows["native"] = [
        is_native(layer, observed_at, rung)
        for layer, observed_at, rung in zip(rows["layer"], rows["observed_at"], rungs, strict=True)
    ]
    blocks = [
        block_key(league, game_date, layer) if layer != _CLOSE_LAYER else (None, None)
        for game_date, layer in zip(rows["game_date"], rows["layer"], strict=True)
    ]
    rows["block_layer"] = [layer for layer, _ in blocks]
    rows["block"] = [block for _, block in blocks]
    return rows


def _references(rows: pd.DataFrame) -> dict[str, pd.Series]:
    """Per reference layer, the mean cleaned native ``under_prob`` of each ``_KEY`` (§5)."""
    under = rows["under_prob"]
    # A row with no price or no rung compares False here, so it never references.
    agrees = (1.0 - under - rows["p_over"]).abs() <= LADDER_DISAGREEMENT
    even = (under - 0.5).abs() < EVEN_PRICE_TOLERANCE
    clean = rows[rows["native"] & agrees & ~even]
    return {
        layer: clean[clean["layer"] == layer].groupby(_KEY)["under_prob"].mean().rename("reference")
        for layer in _REFERENCE_LAYERS
    }


def _halves(matched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """The fit half, the first ⌈n/2⌉ matched game dates, and the validation half (§6).

    None when either half holds fewer than ``MIN_HALF_ROWS`` rows: untestable against them.
    """
    dates = np.sort(matched["game_date"].unique())
    in_fit = matched["game_date"].isin(dates[: math.ceil(len(dates) / 2)])
    fit, val = matched[in_fit], matched[~in_fit]
    if min(len(fit), len(val)) < MIN_HALF_ROWS:
        return None
    return fit, val


def _candidate_order(code_history: str | None) -> list[str]:
    """§7: stored, the code-history family, then the generic families, each tried once."""
    return list(
        dict.fromkeys(family for family in (STORED, code_history, *_GENERIC_FAMILIES) if family)
    )


def family_on(history: DistHistory, league: str, market: str, on: datetime.date) -> str | None:
    """The code-history candidate for a cell on a date (§7): its configured family's decode.

    ``history`` is every committed config, oldest first, as ``{league: {market: family}}``.
    None skips the candidate: the date precedes the first config, or the family has no decode.
    """
    configs = [config for committed, config in history if committed <= on]
    if not configs:
        return None
    return _DECODE_FAMILY.get(configs[-1].get(league, {}).get(market, _UNCONFIGURED_FAMILY))


def _dist_history() -> DistHistory:
    """Every commit of the distribution config (§7), oldest first, read from git."""
    git = partial(subprocess.run, cwd=_REPO_ROOT, capture_output=True, text=True, check=True)
    log = git(["git", "log", "--reverse", "--format=%H %cs", "--", *_DIST_CONFIG_PATHS]).stdout
    history = []
    for line in log.splitlines():
        sha, committed = line.split()
        present = git(["git", "ls-tree", "--name-only", sha, "--", *_DIST_CONFIG_PATHS]).stdout
        path = next(newest for newest in _DIST_CONFIG_PATHS if newest in present.split())
        config = json.loads(git(["git", "show", f"{sha}:{path}"]).stdout)
        if path == _DIST_CONFIG_PATHS[0]:
            # stat_meta nests the family as each cell's ``dist``; a cell without one is unnamed.
            config = {
                league: {market: cell["dist"] for market, cell in cells.items() if "dist" in cell}
                for league, cells in config.items()
            }
        history.append((datetime.date.fromisoformat(committed), config))
    return history


def _usable(name: str, rows: pd.DataFrame) -> pd.DataFrame:
    """The rows a candidate can price, by the read path's own tests (``book_levelling``, §11)."""
    if name == STORED:
        return rows[rows["under_prob"].gt(0) & rows["under_prob"].lt(1)]
    decodable = [
        is_positive(line) and archive_ev_is_usable(ev, line)
        for ev, line in zip(rows["ev"], rows["line"], strict=True)
    ]
    return rows[np.array(decodable, dtype=bool)]


def _evaluate(name: str, fit: pd.DataFrame, val: pd.DataFrame) -> dict | None:
    """One candidate's numbers on the validation half (§8, §9).

    A parametric candidate fits log cv by the mean squared difference from the reference on at
    most ``_FIT_SAMPLE`` fit-half rows, and passes only at an interior fit. None when the
    validation half, or a parametric candidate's fit half, holds fewer than ``MIN_HALF_ROWS``
    rows the candidate can price (§6, §7).
    """
    fit, val = _usable(name, fit), _usable(name, val)
    if len(val) < MIN_HALF_ROWS or (name in _PARAMETRIC and len(fit) < MIN_HALF_ROWS):
        return None
    log_cv = None
    if name in _PARAMETRIC:
        sample = fit.sample(min(len(fit), _FIT_SAMPLE), random_state=_SEED)
        line, ev, reference = (sample[c].to_numpy(float) for c in ("line", "ev", "reference"))
        log_cv = float(
            minimize_scalar(
                lambda trial: np.mean(
                    (decode_under(name, line, ev, np.exp(trial)) - reference) ** 2
                ),
                bounds=LOG_CV_BOUNDS,
                method="bounded",
            ).x
        )
    if name == STORED:
        decoded = val["under_prob"].to_numpy(float)
    else:
        cv = None if log_cv is None else math.exp(log_cv)
        decoded = decode_under(name, val["line"].to_numpy(float), val["ev"].to_numpy(float), cv)
    diff = decoded - val["reference"].to_numpy(float)
    mean_diff, median_abs_diff = float(diff.mean()), float(np.median(np.abs(diff)))
    interior = None
    if log_cv is not None:
        interior = (
            LOG_CV_BOUNDS[0] + _INTERIOR_MARGIN < log_cv < LOG_CV_BOUNDS[1] - _INTERIOR_MARGIN
        )
    level = abs(mean_diff) <= MEAN_DIFF_MAX and median_abs_diff <= MEDIAN_ABS_DIFF_MAX
    return {
        "mean_diff": mean_diff,
        "median_abs_diff": median_abs_diff,
        "log_cv": log_cv,
        "interior": interior,
        "val_rows": len(val),
        "passes": level and interior is not False,
    }


def _test_block(
    block_rows: pd.DataFrame, references: dict[str, pd.Series], code_history: str | None
) -> dict:
    """One migrated block's record: ``accept`` at the first candidate that passes, else ``demote``.

    The block is tested against the first reference layer it is testable against (§5, §6), its
    candidates in order (§7, §9). An untestable block comes back ``tested`` False with no
    verdict, for ``_inherit`` to settle (§10).
    """
    record = {
        "verdict": None,
        "family": None,
        "cv": None,
        "tested": False,
        "rows": len(block_rows),
        "null_under_prob": int(block_rows["under_prob"].isna().sum()),
        "reference_layer": None,
        "matched": {},
    }
    for layer in _REFERENCE_LAYERS:
        matched = block_rows.join(references[layer], on=_KEY, how="inner")
        record["matched"][layer] = len(matched)
        halves = _halves(matched)
        if halves is None:
            continue
        fit, val = halves
        record.update(
            verdict="demote",
            tested=True,
            reference_layer=layer,
            fit_dates=fit["game_date"].nunique(),
            fit_rows=len(fit),
            val_dates=val["game_date"].nunique(),
            val_rows=len(val),
            candidates={},
        )
        for name in _candidate_order(code_history):
            outcome = _evaluate(name, fit, val)
            record["candidates"][name] = outcome
            if outcome is not None and outcome["passes"]:
                cv = None if outcome["log_cv"] is None else math.exp(outcome["log_cv"])
                record.update(verdict="accept", family=name, cv=cv)
                break
        return record
    return record


def _inherit(blocks: list[dict]) -> list[dict]:
    """Every block with a verdict: each untested one settled by §10.

    A backfill block takes the verdict of the nearest tested season of its (league, market,
    layer), the later season on a tie; a demoted source, no tested season, or being a live
    epoch demotes it.
    """
    seasons: dict[tuple[str, str, str], list[dict]] = {}
    for block in blocks:
        # Live epochs never source a verdict, so an untested one finds no season and demotes.
        if block["tested"] and block["layer"] != LIVE:
            seasons.setdefault((block["league"], block["market"], block["layer"]), []).append(block)
    settled = []
    for block in blocks:
        sources = seasons.get((block["league"], block["market"], block["layer"]), [])
        if block["tested"]:
            settled.append(block)
        elif not sources:
            settled.append({**block, "verdict": "demote"})
        else:
            season = int(block["block"])
            source = min(
                sources,
                key=lambda other: (abs(int(other["block"]) - season), -int(other["block"])),
            )
            verdict = (
                {"verdict": "demote"}
                if source["verdict"] == "demote"
                else {"verdict": "inherit", "family": source["family"], "cv": source["cv"]}
            )
            settled.append({**block, **verdict, "inherited_from": source["block"]})
    return settled


def _ladder_consistency(rows: pd.DataFrame) -> pd.DataFrame:
    """M0(b) per (book, layer): the share of priced native rows with a rung that disagree with it.

    Disagreeing is more than ``LADDER_DISAGREEMENT`` apart, read as stored and after the read
    path's ``ladder_repaired``; after the repair the share must be 0.
    """
    native = rows[rows["native"] & rows["p_over"].notna() & rows["under_prob"].notna()]
    # In ArchivedBookQuote's field order: book, ev, under, line, observed_at, rung.
    quotes = native[["book", "ev", "under_prob", "line", "observed_at", "p_over"]]
    repaired = np.array(
        [
            ladder_repaired(ArchivedBookQuote(*quote)).under_probability
            for quote in quotes.itertuples(index=False, name=None)
        ],
        dtype=float,
    )
    rung = native["p_over"]
    return (
        native.assign(
            before=(1.0 - native["under_prob"] - rung).abs() > LADDER_DISAGREEMENT,
            after=(1.0 - repaired - rung).abs() > LADDER_DISAGREEMENT,
        )
        .groupby(["book", "layer"])
        .agg(rows=("before", "size"), before=("before", "mean"), after=("after", "mean"))
    )


def _fit_cell(
    connection: duckdb.DuckDBPyConnection, league: str, market: str, history: DistHistory
) -> tuple[list[dict], pd.DataFrame]:
    """One cell's migrated-block records, before inheritance, and its M0(b) consistency."""
    rows = _load(connection, league, market)
    references = _references(rows)
    blocks = []
    levelled = rows[rows["layer"] != _CLOSE_LAYER]
    for (layer, block), group in levelled.groupby(["block_layer", "block"]):
        migrated = group[~group["native"]]
        if migrated.empty:
            continue
        # §7: a live epoch reads the config of its first day, a backfill block one fixed day.
        on = datetime.date.fromisoformat(f"{block}-01") if layer == LIVE else _BACKFILL_CONFIG_DATE
        record = _test_block(migrated, references, family_on(history, league, market, on))
        blocks.append(
            {
                "league": league,
                "market": market,
                "layer": layer,
                "block": block,
                **record,
                # The block's layer and season (or month) with a rung, native rows included: near
                # 1, its migrated rows are likely native rows whose rung is missing.
                "ladder_share": float(group["p_over"].notna().mean()),
            }
        )
    return blocks, _ladder_consistency(rows)


@click.command()
@click.option(
    "--league",
    "leagues",
    multiple=True,
    help="Fit only these leagues (repeatable); needs --out.",
)
@click.option(
    "--archive",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_ARCHIVE,
    help="Odds archive, opened read-only (default: $SPORTSTRADAMUS_ARCHIVE_DB or the "
    "repository's archive/archive.duckdb).",
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Where to write the table (default: the committed data/config/book_decode.json).",
)
def main(leagues: tuple[str, ...], archive: Path, out: Path | None) -> None:
    """Fit and validate the decode of every migrated archive block against native prices.

    Prices only, never an outcome (§14). Prints each league's blocks and migrated rows by
    verdict and family, then M0(b): the native rows that disagree with their ladder rung, as
    stored and after the read path's repair.
    """
    if leagues and out is None:
        raise click.UsageError(
            "--league needs --out: a partial table would leave every other league unlevelled"
        )
    connection = duckdb.connect(str(archive), read_only=True)
    pairs = [
        (league, market)
        for league, market in connection.execute(
            "SELECT DISTINCT league, market FROM odds ORDER BY ALL"
        ).fetchall()
        if not leagues or league in leagues
    ]
    if not pairs:
        raise click.BadParameter(f"no archive rows for {', '.join(leagues)}", param_hint="--league")
    history = _dist_history()
    blocks, consistency = [], {}
    for league, market in tqdm(pairs, desc="fitting decode blocks"):
        cell_blocks, consistency[league, market] = _fit_cell(connection, league, market, history)
        blocks += cell_blocks
    connection.close()
    blocks = sorted(_inherit(blocks), key=itemgetter("league", "market", "layer", "block"))

    out = out or _TABLE_PATH
    # One block per line keeps the committed table's diff readable.
    body = ",\n".join(f"    {json.dumps(block)}" for block in blocks)
    generated_at = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    out.write_text(
        f'{{\n  "preregistration": "{_PREREGISTRATION}",\n  "generated_at": "{generated_at}",\n'
        f'  "blocks": [\n{body}\n  ]\n}}\n'
    )
    print(f"wrote {len(blocks)} blocks to {out}")

    table = pd.DataFrame(blocks)
    by_verdict = table.groupby(["league", "verdict", "family"], dropna=False).agg(
        blocks=("block", "size"), migrated_rows=("rows", "sum")
    )
    print(by_verdict.to_string())
    m0b = pd.concat(consistency, names=["league", "market"])
    print("M0(b): native rows with a rung that disagree with it, as stored and after the repair")
    print(m0b[m0b["before"] > 0].sort_values("before", ascending=False).to_string())
    unrepaired = m0b[m0b["after"] > 0]
    if unrepaired.empty:
        print("M0(b) after the repair: 0 in every (market, book, layer)")
    else:
        print("M0(b) after the repair is not 0 in:")
        print(unrepaired.to_string())


if __name__ == "__main__":
    main()
