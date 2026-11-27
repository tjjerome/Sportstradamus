"""``ship priced``: judge a candidate against a baseline on the rows we price.

The priced population (CLAUDE.md, "Judging model changes") in two parts. P1 is each cell's
held-out rows an app or a sportsbook priced (``QuoteAuthenticity`` pickem or authentic), at
their own ``Line``. P2 is the Underdog and Sleeper rungs each prop still held at its last
poll: the rungs the ``realized-replay`` posted scope prices, before its 5 %-edge rule. Each arm
reads its real served probability and mean: the dump re-served through the arm's own pickle
under the serving convention (``tail_pricing.serve_live``, and ``price_dumps`` at the
rungs); a refit pickle scores no row dated on or before its ``refit_through``
(``tail_pricing.out_of_sample``). The arms meet on the rows both scored at the same line.
With ``--baseline-book`` the baseline is the sportsbook itself, on authentic P1 rows only:
P(over) ``1 - Odds`` (the dump's ``Odds`` is the book's under-probability) and mean
``Book_EV``, for the cells production serves at the book consensus.

Statistics, candidate minus baseline, each a mean over rows with a 95 % interval from
``N_RESAMPLES`` resamples of whole game dates; negative is better:

- ``p1_dmse``: squared error of the served mean, each row's change in % of its own cell's
  baseline MSE, so cells of any scale pool on one;
- ``p1_dbrier``, ``p2_dbrier``: Brier at the line in units of 1e-4, pushes dropped;
- ``toward_book`` (information, never in a verdict): how much closer the candidate's P sits
  to the book's ``1 - Odds`` on authentic P1 rows, in probability points.

The verdict, fixed before any read: a league's primary statistic is ``p2_dbrier`` when its
P2 rungs span ``MIN_DATES`` game dates, else ``p1_dmse``, and each cell reads its league's
primary on its own rows. A scope whose primary spans fewer dates is NEUTRAL. Otherwise it is
KILL when any judged statistic spanning ``MIN_DATES`` dates has its interval wholly above
zero, KEEP when the primary's lies wholly below, and NEUTRAL else.
A value both arms serve alike, to float rounding (``SAME_VALUE_RTOL``), is no change.
Beside each interval, ``deff_player``: the design effect of the change's clustering by player
across dates, 1 + (sum n_i^2 / N - 1) * ICC over player-date means, the one-way ICC truncated
at 0 (information, never in a verdict).

Beside ``--out``, ``<stem>.by_quote_class.csv`` reads each cell apart by quote class,
authentic or not (``quote_class_table``); a report that never changes a verdict.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from sportstradamus.helpers.training_quotes import AUTHENTIC, PRICED_AUTHENTICITY
from sportstradamus.scripts.realized_replay import still_posted
from sportstradamus.scripts.tail_information import key_rng
from sportstradamus.scripts.tail_pricing import (
    CellExcludedError,
    dump_paths,
    load_cell,
    open_archive,
    out_of_sample,
    price_dumps,
    serve_live,
)
from sportstradamus.training.scorecard import _cluster_bootstrap_means

# Whole-game-date resamples behind every interval.
N_RESAMPLES = 500
# The 95 % two-sided bounds, as the bootstrap's (low, high) percentiles.
CI_PERCENTILES = (2.5, 97.5)
# A date bootstrap this thin runs narrow, so a primary population over fewer dates decides
# nothing.
MIN_DATES = 20
# Brier changes read in units of 1e-4.
BRIER_SCALE = 1e4
# Two serving paths computing the same value differ by float rounding (~1e-16 relative); a
# real change sits many orders above it.
SAME_VALUE_RTOL = 1e-12
# Paired with each scope key's crc32 (``key_rng``), so a --league run reproduces the full
# run's intervals.
DEFAULT_SEED = 1729
POOLED = "ALL"
# The statistics a verdict reads, then the information-only one; also the report order.
JUDGED = ("p1_dmse", "p1_dbrier", "p2_dbrier")
STATISTICS = (*JUDGED, "toward_book")

_ROW_KEY = ["Player", "Date"]
_RUNG_KEY = ["League", "cell", "Player", "Date", "Platform", "Line"]
_P2_COLUMNS = [*_RUNG_KEY, "Result", "QuoteAuthenticity", "p_cand", "p_base"]
_CHANGE_COLUMNS = ["League", "cell", "Player", "Date", "statistic", "change"]
_TABLE_COLUMNS = [
    "league",
    "cell",
    "statistic",
    "n_rows",
    "n_dates",
    "estimate",
    "lo",
    "hi",
    "deff_player",
    "primary",
    "verdict",
]
# The quote-class split keeps the main table's columns, with each row's class after its cell.
_SPLIT_COLUMNS = [*_TABLE_COLUMNS[:2], "quotes", *_TABLE_COLUMNS[2:]]


def p1_rows(
    candidate: pd.DataFrame, baseline: pd.DataFrame, classes: tuple[str, ...]
) -> tuple[pd.DataFrame, int]:
    """One cell's P1 rows: the rows both arms scored, at the line they share, of ``classes``.

    Each arm carries the dump's columns plus its served ``p`` (P(over) at the row's line) and
    ``mean``. The arms join on (Player, Date); the outcome, line, quote class and ``Odds``
    are the candidate's.

    Returns:
        The rows with ``p_cand``/``p_base`` and ``mean_cand``/``mean_base``, and how many
        joined rows were dropped because their ``Line`` differs across the arms.
    """
    rows = candidate.merge(
        baseline[[*_ROW_KEY, "Line", "p", "mean"]],
        on=_ROW_KEY,
        suffixes=("_cand", "_base"),
        validate="one_to_one",
    )
    same_line = rows["Line_cand"].eq(rows["Line_base"])
    rows = rows[same_line].rename(columns={"Line_cand": "Line"})
    priced = rows["QuoteAuthenticity"].isin(classes) & rows["Line"].notna()
    return rows[priced], int((~same_line).sum())


def p2_rungs(candidate: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """The P2 rungs: those each prop still held at its last poll, priced by both arms.

    The arms' ``price_dumps`` rungs, each cut by ``still_posted`` as ``realized-replay``
    cuts them, inner-joined on (cell, Player, Date, Platform, Line); the outcome and the
    prop's ``QuoteAuthenticity`` (its dump row's) are the candidate's.
    """
    if candidate.empty or baseline.empty:
        return pd.DataFrame(columns=_P2_COLUMNS)
    cand, base = (rungs[still_posted(rungs)] for rungs in (candidate, baseline))
    joined = cand[[*_RUNG_KEY, "Actual", "QuoteAuthenticity", "p_serving"]].merge(
        base[[*_RUNG_KEY, "p_serving"]],
        on=_RUNG_KEY,
        suffixes=("_cand", "_base"),
        validate="one_to_one",
    )
    return joined.rename(
        columns={"Actual": "Result", "p_serving_cand": "p_cand", "p_serving_base": "p_base"}
    )[_P2_COLUMNS]


def _brier_changes(rows: pd.DataFrame, statistic: str) -> pd.DataFrame:
    settled = rows[rows["Result"] != rows["Line"]]
    over = (settled["Result"] > settled["Line"]).to_numpy(float)
    # A frame with no rungs holds object columns, which isclose cannot read.
    p_cand, p_base = (settled[arm].to_numpy(float) for arm in ("p_cand", "p_base"))
    change = (p_cand - over) ** 2 - (p_base - over) ** 2
    same_p = np.isclose(p_cand, p_base, rtol=SAME_VALUE_RTOL, atol=0.0)
    return settled.assign(statistic=statistic, change=np.where(same_p, 0.0, BRIER_SCALE * change))


def row_changes(p1: pd.DataFrame, p2: pd.DataFrame) -> pd.DataFrame:
    """Every row's change, candidate minus baseline, under each statistic.

    Args:
        p1: P1 rows (``p1_rows``) of any number of cells, with ``League`` and ``cell``.
        p2: P2 rungs (``p2_rungs``).

    Returns:
        ``League``, ``cell``, ``Player``, ``Date``, ``statistic`` (ordered as ``STATISTICS``)
        and ``change``: ``p1_dmse`` in % of the row's own cell's baseline MSE, the Brier
        statistics in 1e-4 on settled rows, ``toward_book`` in points on authentic rows.
    """
    sq_cand = (p1["Result"] - p1["mean_cand"]) ** 2
    sq_base = (p1["Result"] - p1["mean_base"]) ** 2
    cell_mse = sq_base.groupby(p1["cell"]).transform("mean")
    # A probability-only candidate serves the baseline's mean through another code path, a
    # rounding error away on every row: that noise reads as an interval wholly above zero, a KILL.
    same_mean = np.isclose(p1["mean_cand"], p1["mean_base"], rtol=SAME_VALUE_RTOL, atol=0.0)
    book = 1 - p1["Odds"]
    toward = (p1["p_base"] - book).abs() - (p1["p_cand"] - book).abs()
    authentic = p1["QuoteAuthenticity"].eq(AUTHENTIC)
    parts = [
        p1.assign(
            statistic="p1_dmse",
            change=np.where(same_mean, 0.0, 100 * (sq_cand - sq_base) / cell_mse),
        ),
        _brier_changes(p1, "p1_dbrier"),
        _brier_changes(p2, "p2_dbrier"),
        p1[authentic].assign(statistic="toward_book", change=100 * toward[authentic]),
    ]
    changes = pd.concat(
        [part[_CHANGE_COLUMNS] for part in parts if not part.empty], ignore_index=True
    )
    return changes.assign(statistic=pd.Categorical(changes["statistic"], STATISTICS))


def _player_deff(rows: pd.DataFrame) -> float:
    """1 + (sum n_i^2 / N - 1) * ICC: the one-way ANOVA ICC by player of player-date means.

    The date bootstrap already draws a date's rows together, so the rows a player holds on one
    date (a prop's rungs) enter as their mean, and the design effect measures only how a
    player's change persists across dates: per rung, NFL receiving yards' P2 read 7.1 where its
    player-dates read 1.0. The unbalanced form, with the n0 average group size and the ICC
    floored at 0; 1 when there is no clustering to measure (one player, or one date each).
    """
    units = rows.groupby(["Player", "Date"], observed=True)["change"].mean()
    change = units.to_numpy(float)
    _, codes, sizes = np.unique(
        units.index.get_level_values("Player"), return_inverse=True, return_counts=True
    )
    total, groups = len(change), len(sizes)
    if groups < 2 or total == groups:
        return 1.0
    means = np.bincount(codes, weights=change) / sizes
    between = (sizes * (means - change.mean()) ** 2).sum() / (groups - 1)
    within = ((change - means[codes]) ** 2).sum() / (total - groups)
    n0 = (total - (sizes**2).sum() / total) / (groups - 1)
    denom = between + (n0 - 1) * within
    icc = max(0.0, (between - within) / denom) if denom > 0 else 0.0
    return 1 + ((sizes**2).sum() / total - 1) * icc


def scope_table(changes: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Each statistic per cell and pooled per league (cell ``ALL``), with its interval.

    Per scope: rows, game dates, the mean change, its 95 % interval and standard error
    (``se``) from ``N_RESAMPLES`` resamples of whole game dates, each scope drawing from its
    own generator (``key_rng``), and ``deff_player`` (:func:`_player_deff`).
    """
    records = []
    for league, league_rows in changes.groupby("League"):
        for cell, scope in [(POOLED, league_rows), *league_rows.groupby("cell")]:
            for statistic, rows in scope.groupby("statistic", observed=True):
                change = rows["change"].to_numpy(float)
                draws = _cluster_bootstrap_means(
                    change,
                    rows["Date"].to_numpy(),
                    key_rng(seed, f"{league}/{cell}/{statistic}"),
                    N_RESAMPLES,
                )
                lo, hi = np.percentile(draws, CI_PERCENTILES)
                records.append(
                    {
                        "league": league,
                        "cell": cell,
                        "statistic": statistic,
                        "n_rows": len(rows),
                        "n_dates": rows["Date"].nunique(),
                        "estimate": change.mean(),
                        "lo": lo,
                        "hi": hi,
                        "se": draws.std(ddof=1),
                        "deff_player": _player_deff(rows),
                    }
                )
    return pd.DataFrame(records)


def _verdict(scope: pd.DataFrame) -> str:
    lead = scope[scope["primary"]]
    if lead.empty or lead["n_dates"].iat[0] < MIN_DATES:
        return f"NEUTRAL (fewer than {MIN_DATES} dates)"
    # A thin statistic's narrow interval would kill on noise, so it is reported, never judged.
    judged = scope["statistic"].isin(JUDGED) & scope["n_dates"].ge(MIN_DATES)
    if scope.loc[judged, "lo"].gt(0).any():
        return "KILL"
    return "KEEP" if lead["hi"].iat[0] < 0 else "NEUTRAL"


def judge(table: pd.DataFrame) -> pd.DataFrame:
    """``scope_table`` rows with each league's primary statistic flagged and every verdict.

    A league's pooled (``ALL``) P2 dates pick its primary; each cell reads that primary on its
    own rows. The verdict rule is the module docstring's.
    """
    pooled_p2 = table[table["cell"].eq(POOLED) & table["statistic"].eq("p2_dbrier")]
    p2_dates = table["league"].map(pooled_p2.set_index("league")["n_dates"])
    primary = np.where(p2_dates.ge(MIN_DATES), "p2_dbrier", "p1_dmse")
    table = table.assign(primary=table["statistic"].eq(primary))
    verdicts = {scope: _verdict(rows) for scope, rows in table.groupby(["league", "cell"])}
    scopes = zip(table["league"], table["cell"], strict=True)
    return table.assign(verdict=[verdicts[scope] for scope in scopes])


def quote_class_table(
    p1: pd.DataFrame, p2: pd.DataFrame, table: pd.DataFrame, seed: int
) -> pd.DataFrame:
    """Each cell's reads split by quote class, ``authentic`` or ``not authentic``; report-only.

    A P1 row is classed by its own ``QuoteAuthenticity``, a P2 rung by its prop's: an authentic
    prop holds a sportsbook quote, which the book pull and the reference anchor act on. Each
    class is ``scope_table`` on its own rows under ``seed``, per cell, without ``toward_book``
    (authentic rows only, so already the main table's). ``primary`` and ``verdict`` are the
    main ``table``'s: the split never changes a verdict.
    """
    p1_authentic = p1["QuoteAuthenticity"].eq(AUTHENTIC)
    p2_authentic = p2["QuoteAuthenticity"].eq(AUTHENTIC)
    classes = {
        "authentic": (p1[p1_authentic], p2[p2_authentic]),
        "not authentic": (p1[~p1_authentic], p2[~p2_authentic]),
    }
    split = pd.concat(
        [
            scope_table(row_changes(rows, rungs), seed).assign(quotes=quotes)
            for quotes, (rows, rungs) in classes.items()
            if not (rows.empty and rungs.empty)
        ],
        ignore_index=True,
    )
    split = split[split["cell"].ne(POOLED) & split["statistic"].ne("toward_book")]
    verdicts = table[["league", "cell", "statistic", "primary", "verdict"]]
    split = split.merge(verdicts, on=["league", "cell", "statistic"])
    return split.sort_values(["league", "cell"], kind="stable", ignore_index=True)[_SPLIT_COLUMNS]


def _served_cells(
    cells: list[str], dumps: Path, models: Path, start: str, arm: str
) -> dict[str, pd.DataFrame]:
    """Each cell's dump rows dated from ``start`` with the arm's served ``p`` and ``mean``.

    A refit pickle's in-sample rows are dropped first (``out_of_sample``). A cell ``load_cell``
    refuses, or with every row in sample, is printed with its reason and left out.
    """
    served = {}
    for cell in tqdm(cells, desc=f"serving the {arm}"):
        try:
            rows, knobs, _ = load_cell(dumps / f"{cell}.csv", models)
            rows = out_of_sample(rows[rows["Date"] >= start], knobs["refit_through"], cell)
        except CellExcludedError as exc:
            click.echo(f"excluded {cell} ({arm}): {exc}")
            continue
        p_over, mean, _ = serve_live(rows, knobs)
        served[cell] = rows.assign(p=p_over, mean=mean)
    return served


def _posted(
    cells: list[str], arms: list[tuple[Path, Path]], archive: Path, start: str
) -> pd.DataFrame:
    """Both arms' ladder rungs on ``cells`` (``price_dumps``), joined as P2 (``p2_rungs``)."""
    con = open_archive(archive)
    rungs, missed = [], {}
    for dumps, models in arms:
        priced, _, _, _, excluded = price_dumps(
            [dumps / f"{cell}.csv" for cell in cells], con, start, models
        )
        rungs.append(priced)
        missed |= excluded
    con.close()
    for cell, reason in sorted(missed.items()):
        click.echo(f"{cell}: no P2 ({reason})")
    return p2_rungs(*rungs)


def _dump_cells(dumps: Path, leagues: set[str]) -> set[str]:
    return {path.stem for path in dump_paths(dumps, leagues)}


_DIRECTORY = click.Path(exists=True, file_okay=False, path_type=Path)


@click.command()
@click.option(
    "--candidate-test-sets",
    type=_DIRECTORY,
    required=True,
    help="The candidate's {LEAGUE}_{market}.csv dumps.",
)
@click.option(
    "--candidate-models", type=_DIRECTORY, required=True, help="The candidate's own pickles."
)
@click.option(
    "--baseline-test-sets", type=_DIRECTORY, help="The baseline's dumps (with --baseline-models)."
)
@click.option("--baseline-models", type=_DIRECTORY, help="The baseline's own pickles.")
@click.option(
    "--baseline-book",
    is_flag=True,
    help="Judge against the sportsbook instead (P(over) 1 - Odds, mean Book_EV) on authentic "
    "P1 rows, with no P2: for the cells production serves at the book consensus.",
)
@click.option(
    "--archive",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Odds archive whose ladder prices P2, opened read-only (required unless --baseline-book).",
)
@click.option(
    "--start",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    required=True,
    help="Score rows dated from this day.",
)
@click.option("--league", "leagues", multiple=True, help="Score only these leagues (repeatable).")
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("/tmp/priced_score.csv"),
    show_default=True,
    help="One row per league, cell (ALL: the league's cells pooled) and statistic.",
)
@click.option("--seed", default=DEFAULT_SEED, show_default=True, help="Bootstrap seed.")
def main(
    candidate_test_sets: Path,
    candidate_models: Path,
    baseline_test_sets: Path | None,
    baseline_models: Path | None,
    baseline_book: bool,
    archive: Path | None,
    start: datetime,
    leagues: tuple[str, ...],
    out: Path,
    seed: int,
) -> None:
    """Judge a candidate against a baseline on the rows we price: KEEP, KILL or NEUTRAL.

    P1 is the held-out rows an app or sportsbook priced, at their own line; P2 the Underdog
    and Sleeper rungs still posted at the prop's last poll. Each arm reads its own served
    probability and mean. Candidate minus baseline, negative better, with 95 % intervals over
    whole game dates: p1_dmse (% of the cell's baseline MSE), p1_dbrier and p2_dbrier (1e-4,
    pushes dropped), toward_book (points closer to the book's P; information only). A
    league's primary is p2_dbrier when its rungs span 20 game dates, else p1_dmse.
    """
    baseline_dirs = (baseline_test_sets, baseline_models)
    one_baseline = not any(baseline_dirs) if baseline_book else all(baseline_dirs)
    if not one_baseline:
        raise click.UsageError(
            "give --baseline-test-sets with --baseline-models, or --baseline-book alone"
        )
    if archive is None and not baseline_book:
        raise click.UsageError("--archive is required unless --baseline-book")
    day = start.strftime("%Y-%m-%d")
    wanted = {league.upper() for league in leagues}
    candidate_cells = _dump_cells(candidate_test_sets, wanted)
    baseline_cells = candidate_cells if baseline_book else _dump_cells(baseline_test_sets, wanted)
    for cell in sorted(candidate_cells ^ baseline_cells):
        click.echo(f"excluded {cell}: one arm has no dump")
    cells = sorted(candidate_cells & baseline_cells)

    candidate = _served_cells(cells, candidate_test_sets, candidate_models, day, "candidate")
    if baseline_book:
        baseline = {
            cell: rows.assign(p=1 - rows["Odds"], mean=rows["Book_EV"])
            for cell, rows in candidate.items()
        }
        classes = (AUTHENTIC,)
    else:
        baseline = _served_cells(cells, baseline_test_sets, baseline_models, day, "baseline")
        classes = PRICED_AUTHENTICITY
    scored = sorted(candidate.keys() & baseline.keys())
    if not scored:
        raise click.ClickException("no cell both arms can serve")
    p1_frames = []
    for cell in scored:
        rows, dropped = p1_rows(candidate[cell], baseline[cell], classes)
        if dropped:
            click.echo(f"{cell}: dropped {dropped} rows whose Line differs across the arms")
        p1_frames.append(rows.assign(League=cell.split("_", 1)[0], cell=cell))
    p1 = pd.concat(p1_frames, ignore_index=True)
    p2 = (
        pd.DataFrame(columns=_P2_COLUMNS)
        if baseline_book
        else _posted(scored, [(candidate_test_sets, candidate_models), baseline_dirs], archive, day)
    )
    if p1.empty and p2.empty:
        raise click.ClickException("no priced row or posted rung to score")

    table = judge(scope_table(row_changes(p1, p2), seed))[_TABLE_COLUMNS]
    for cell in sorted(set(scored) - set(table["cell"])):
        click.echo(f"excluded {cell}: no priced row or posted rung")
    click.echo(table.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    table.to_csv(out, index=False)
    click.echo(f"wrote {len(table)} rows to {out}")
    split = quote_class_table(p1, p2, table, seed)
    split_out = out.with_name(f"{out.stem}.by_quote_class.csv")
    split.to_csv(split_out, index=False)
    click.echo(f"wrote {len(split)} rows split by quote class to {split_out}")
