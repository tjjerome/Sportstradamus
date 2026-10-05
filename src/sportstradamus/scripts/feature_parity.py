"""Feature parity: the serve-time feature log replayed against the training matrix.

``prediction.feature_log`` keeps, for every player-game ``model_prob`` scores, the frame the
model was fed and what it answered. This command replays that log and reports, per cell and
model version:

- **Self-check.** Each ``model_prob`` batch (the rows sharing a ``Scored At``) is re-scored
  through the file that carries its ``Model Version``, the served pickle or its copy under
  ``models/prior/``, and compared with the outputs the rows logged. A version whose rows do
  not reproduce is reported and left out of everything below; so is one with no file, one
  whose file serving would refuse today, and a volume cell, which is served from the volume
  projector and not from the logged frame. One kind of row is judged on its own: with
  ``STDYr`` 0 a row is seeded from the median of its batch, so it reproduces only while its
  whole batch is still in the log, and a miss drops that row alone.
- **Parity.** A logged row whose (cell, player, game date) is in the cell's cached training
  matrix is compared with that row feature by feature, and both feature vectors are scored
  through the same file. ``flag`` marks a cell whose probability difference (serve minus
  training features) has a standard deviation above ``PARITY_SD_ALARM``.
- **Version re-score** (``--versions``). Each logged row is re-scored under every other
  version of its cell on disk as that version would have served it, through its own file at
  its own zero-rate gate: the paired same-leg difference between versions, with the days
  between the two version stamps. A version that needs a feature the log does not hold, or
  whose file serving would refuse, is named with the reason and not scored.
- **Coverage.** Each cell, platform and game date with settled model-served legs in
  ``history.parquet`` and no log rows, so a log that fails silently shows.

Every probability is the model's own: ``get_odds`` on the decoded outputs at the pickle's
step, before the book blend and the calibration fitted on top of it, so a difference is the
features' doing (between versions, the model's). The line it is read at:

- Parity uses the training-matrix row's ``Line``. Every matched row has one, it is the line
  the training set labels and prices that player-game at, and the log holds no offer line
  (a player-game is posted at several).
- The version re-score runs on rows the matrix has not reached yet, so it uses the log's
  ``Quote Line``, the consensus line when the row was scored. A row no book quoted has none
  and is left out.

Dev-side and read-only: it never runs in production, no served probability depends on it,
and its one write is the sandbox CSV at ``--out``.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import pickle
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from sportstradamus import data
from sportstradamus.helpers.archive import archive_market
from sportstradamus.helpers.io import (
    FEATURE_LOG_DIR,
    MODELS_DIR,
    VOLUME_STATS,
    market_file_slug,
    read_history,
    resolve_model_version,
)
from sportstradamus.prediction.model_prob import (
    _BOOK_FALLBACK_VERSION,
    _build_prob_params,
    _decode_model_params,
    _model_over_and_push,
    _resolve_serving_strategy,
)

TRAINING_DATA_DIR = Path(str(pkg_resources.files(data) / "training_data"))
_DEFAULT_OUT = Path("/tmp/feature_parity.csv")

# The honest-receipts brief's alarm (I6e): a cell whose probability difference between serve
# and training features has a standard deviation above one percentage point.
PARITY_SD_ALARM = 0.01

# Model outputs are float32 and LightGBM's summation order moves with the batch, so a
# re-score agrees to float32 rounding (worst on the replay logs: 9e-8 relative), not bit for
# bit. 1e-6 is about ten of those steps and four orders below the alarm.
REPRODUCE_TOL = 1e-6

# The matrix build and the serve path reach the same feature through different arithmetic;
# below this they differ by float noise, above it by value.
FEATURE_TOL = 1e-6

# Differing features printed per cell and version; the CSV keeps every one.
_WORST_FEATURES = 5

# Sends _build_prob_params down its booster branch; volume cells are excluded before it.
_NO_VOLUME = SimpleNamespace(volume_stats=())

# Coverage is read per platform: each platform's rows come from its own model_prob call, so
# one platform's log write can fail while the other's lands.
_COVERAGE_KEY = ["League", "Market", "Platform", "Date"]

_SUMMARY_COLUMNS = [
    "cell",
    "League",
    "version",
    "log_rows",
    "reproduced",
    "matched",
    "dp_mean",
    "dp_sd",
    "flag",
    "note",
]
_FEATURE_COLUMNS = ["cell", "version", "feature", "differ_share", "mean_abs_diff"]
_PAIR_COLUMNS = [
    "cell",
    "version",
    "other_version",
    "days_between",
    "n",
    "dp_mean",
    "dp_sd",
    "note",
]


def model_files(cell: str) -> dict[str, dict]:
    """Every model file on disk for a cell, keyed by the version it carries.

    The served pickle is read last, so it answers for a version a prior copy also carries.
    """
    models = Path(str(MODELS_DIR))
    files = {}
    for path in [*sorted((models / "prior").glob(f"{cell}__*.mdl")), *models.glob(f"{cell}.mdl")]:
        with path.open("rb") as infile:
            filedict = pickle.load(infile)
        files[resolve_model_version(str(path), filedict)] = filedict
    return files


def score(
    filedict: dict, league: str, market: str, frame: pd.DataFrame, hist_gate: float
) -> pd.DataFrame:
    """The model's raw and decoded outputs for one batch of feature rows, as serving builds them.

    Args:
        frame: The file's ``expected_columns`` for the batch's rows.
        hist_gate: The zero-rate gate the SkewNormal decode reads.
    """
    frame = frame.copy()  # _build_prob_params casts the categoricals in place
    dist, norm = filedict["distribution"], filedict["target_normalization"]
    structural = _resolve_serving_strategy(filedict, league, market).structural_strategy
    outputs = _build_prob_params(
        filedict, market, _NO_VOLUME, frame, dist, filedict["normalized"], norm, structural
    )
    _decode_model_params(outputs, dist, frame, hist_gate, filedict["offset_meta"], norm)
    return outputs


def over_probability(outputs: pd.DataFrame, filedict: dict, line: pd.Series) -> np.ndarray:
    """The model's own over-probability at ``line`` from its decoded pre-blend outputs."""
    priced = outputs.assign(Line=line.to_numpy(float))
    base_mean = outputs["Projection"].to_numpy(float)
    return _model_over_and_push(
        priced, filedict["distribution"], filedict["cv"], filedict["step"], base_mean
    )[0]


def check_version(
    rows: pd.DataFrame, filedict: dict, others: dict[str, dict], train: pd.DataFrame
) -> tuple[dict, pd.DataFrame, list[dict]]:
    """One model version's logged rows of a cell: self-check, parity and version re-score.

    Args:
        rows: The version's logged rows.
        filedict: The model file that carries the version.
        others: The cell's other model files to re-score the rows under, by version.
        train: The training-matrix rows of the cell's matched logged rows, indexed like them.

    Returns:
        The summary record (``note`` set when the version does not reproduce), the features
        that differ on matched rows, and one record per other version.
    """
    league, market, version = (rows[key].iat[0] for key in ("League", "Market", "Model Version"))
    expected = filedict["expected_columns"]
    scoreable, pairs = {}, []
    for other, other_file in others.items():
        if set(other_file["expected_columns"]) <= set(expected):
            scoreable[other] = other_file
        else:
            pairs.append({"other_version": other, "note": "needs features the log does not hold"})

    served, trained = [], []
    rescored = {other: [] for other in scoreable}
    for _, batch in rows.groupby("Scored At"):
        gate = batch["Hist Gate"].iat[0]
        frame = batch[expected]
        served.append(score(filedict, league, market, frame, gate))
        hit = batch.index.intersection(train.index)
        if len(hit):
            # The training features take the row's place in its own batch, so the batch
            # seeding is the same on both sides and identical features score identically.
            swapped = pd.concat([frame.drop(hit), train.loc[hit].reindex(columns=expected)])
            swapped = swapped.loc[frame.index]
            trained.append(score(filedict, league, market, swapped, gate).loc[hit])
        for other, other_file in scoreable.items():
            other_frame = batch[other_file["expected_columns"]]
            # The rows logged the runtime gate of their own version. Another version served
            # at the zero rate its own training wrote, which its file keeps.
            rescored[other].append(
                score(other_file, league, market, other_frame, other_file["hist_gate"])
            )

    served = pd.concat(served).loc[rows.index]
    # reindex, not a column pick: an output the rows never logged compares as a miss.
    close = np.isclose(
        served.to_numpy(float),
        rows.reindex(columns=served.columns).to_numpy(float),
        rtol=REPRODUCE_TOL,
        atol=REPRODUCE_TOL,
        equal_nan=True,
    )
    reproduced = pd.Series(close.all(axis=1), index=rows.index)
    record = {"log_rows": len(rows), "reproduced": reproduced.mean()}
    # A row with dispersion history is seeded from its own features, so a miss on one means
    # the file is not the model that served it.
    if not reproduced[rows["STDYr"].gt(0)].all():
        missed = int((~reproduced).sum())
        note = f"{missed} of {len(rows)} rows' logged outputs not reproduced"
        return record | {"note": note}, pd.DataFrame(), []

    kept = train.index.intersection(rows.index[reproduced])
    record["matched"] = len(kept)
    features = pd.DataFrame()
    if len(kept):
        line = train.loc[kept, "Line"]
        dp = pd.Series(
            over_probability(served.loc[kept], filedict, line)
            - over_probability(pd.concat(trained).loc[kept], filedict, line)
        )
        record |= {"dp_mean": dp.mean(), "dp_sd": dp.std(), "flag": dp.std() > PARITY_SD_ALARM}
        serve_x = rows.loc[kept, expected].astype(float)
        train_x = train.loc[kept].reindex(columns=expected).astype(float)
        differ = ~np.isclose(
            serve_x.to_numpy(),
            train_x.to_numpy(),
            rtol=FEATURE_TOL,
            atol=FEATURE_TOL,
            equal_nan=True,
        )
        features = pd.DataFrame(
            {
                "feature": expected,
                "differ_share": differ.mean(axis=0),
                "mean_abs_diff": (serve_x - train_x).abs().mean().to_numpy(),
            }
        )
        features = features[features["differ_share"] > 0]

    quoted = rows.index[reproduced & rows["Quote Line"].notna()]
    line = rows.loc[quoted, "Quote Line"]
    own = over_probability(served.loc[quoted], filedict, line)
    for other, outputs in rescored.items():
        dp = pd.Series(
            over_probability(pd.concat(outputs).loc[quoted], scoreable[other], line) - own
        )
        pairs.append(
            {
                "other_version": other,
                "days_between": (pd.Timestamp(other[:8]) - pd.Timestamp(version[:8])).days,
                "n": len(dp),
                "dp_mean": dp.mean(),
                "dp_sd": dp.std(),
            }
        )
    return record, features, pairs


def check_cell(
    logged: pd.DataFrame, compare_versions: bool
) -> tuple[list[dict], list[dict], list[dict]]:
    """One cell's logged rows, version by version.

    Returns:
        Summary records per model version (``note`` set on an excluded one), the differing
        features per version, and the version-pair records.
    """
    league, market = logged["League"].iat[0], logged["Market"].iat[0]
    cell = market_file_slug(league, market)
    by_version = logged.groupby("Model Version")
    if market in VOLUME_STATS[league]:
        note = "volume cell: served from the volume projector, not from the logged frame"
        tag = {"cell": cell, "League": league}
        records = [
            tag | {"version": version, "log_rows": len(rows), "note": note}
            for version, rows in by_version
        ]
        return records, [], []
    files, refused = {}, {}
    for version, filedict in model_files(cell).items():
        try:
            _resolve_serving_strategy(filedict, league, market)
        except ValueError as error:
            # Serving's own check: a file it would not serve today (a strategy identity the
            # code has since moved away from) is not one a re-score can be trusted through.
            refused[version] = f"serving refuses the model file ({error})"
        else:
            files[version] = filedict
    matrix = pd.read_parquet(TRAINING_DATA_DIR / f"{cell}.parquet").set_index(["Player", "Date"])
    keys = pd.MultiIndex.from_arrays([logged["Player"], pd.to_datetime(logged["Date"])])
    matched = keys.isin(matrix.index)
    train = matrix.loc[keys[matched]].set_axis(logged.index[matched])

    records, features, pairs = [], [], []
    for version, rows in by_version:
        tag = {"cell": cell, "League": league, "version": version}
        if version not in files:
            note = refused.get(version, "no model file on disk carries this version")
            records.append(tag | {"log_rows": len(rows), "note": note})
            continue
        others = {v: f for v, f in files.items() if v != version} if compare_versions else {}
        record, differing, version_pairs = check_version(rows, files[version], others, train)
        records.append(tag | record)
        features += differing.assign(**tag).to_dict("records")
        pairs += [tag | pair for pair in version_pairs]
        if compare_versions and "note" not in record:
            pairs += [tag | {"other_version": v, "note": note} for v, note in refused.items()]
    return records, features, pairs


def coverage_gaps(
    logged_days: pd.DataFrame, leagues: tuple[str, ...], start: str, end: str | None
) -> pd.DataFrame:
    """Each cell, platform and game date with settled model-served legs in history and no log rows.

    Args:
        logged_days: The cells, platforms and game dates the log holds rows for.
        leagues: Leagues to check; empty for every league in history.
        start: First game date checked.
        end: Last game date checked, or ``None`` for no upper bound.
    """
    history = read_history()
    served = (
        history["Actual"].notna()
        & history["Model Version"].ne(_BOOK_FALLBACK_VERSION)
        & history["Date"].ge(start)
    )
    if end:
        served &= history["Date"].le(end)
    if leagues:
        served &= history["League"].isin(leagues)
    legs = history[served]
    # History names a leg by its model cell except on NBA and WNBA, whose "... underdog"
    # markets the "... prizepicks" cell serves; archive_market folds them. Not normalize_market:
    # Sleeper's stat map chains bat_walks -> walks -> walks allowed.
    cell_market = pd.Series(map(archive_market, legs["League"], legs["Market"]), index=legs.index)
    counts = legs.assign(Market=cell_market).groupby(_COVERAGE_KEY).size()
    merged = counts.rename("legs").reset_index().merge(logged_days, how="left", indicator=True)
    return merged[merged["_merge"].eq("left_only")].drop(columns="_merge")


@click.command()
@click.option(
    "--league",
    "leagues",
    multiple=True,
    help="Check only these leagues (repeatable). Default: every league in the log.",
)
@click.option(
    "--start",
    type=click.DateTime(["%Y-%m-%d"]),
    help="First game date. Default: the log's earliest partition.",
)
@click.option(
    "--end", type=click.DateTime(["%Y-%m-%d"]), help="Last game date. Default: no upper bound."
)
@click.option(
    "--log-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path(str(FEATURE_LOG_DIR)),
    help="Feature log root, read only (default: data/runtime/feature_log).",
)
@click.option(
    "--versions",
    is_flag=True,
    help="Also re-score each logged row under every other version of its cell on disk.",
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=_DEFAULT_OUT,
    show_default=True,
    help="Sandbox CSV of every table.",
)
def main(
    leagues: tuple[str, ...],
    start: datetime | None,
    end: datetime | None,
    log_dir: Path,
    versions: bool,
    out: Path,
) -> None:
    """Check the serve-time feature log against the training matrices.

    A dev-side diagnostic. Re-scores each logged row to confirm the log reproduces what was
    served, compares it with its training-matrix row (features and the model's own
    probability), lists settled cells and dates the log missed, and writes every table to
    OUT.
    """
    # ISO dates order as text, so partition names and history dates are compared unparsed.
    first, last = (day and str(day.date()) for day in (start, end))
    cells: dict[str, list[Path]] = {}
    for path in sorted(log_dir.glob("date=*/*.parquet")):
        day = path.parent.name.removeprefix("date=")
        in_window = (not first or day >= first) and (not last or day <= last)
        if in_window and (not leagues or path.stem.split("_", 1)[0] in leagues):
            cells.setdefault(path.stem, []).append(path)
    if not cells:
        raise click.ClickException(f"no feature log rows under {log_dir} in that window")

    records, features, pairs, logged_days = [], [], [], []
    for cell in tqdm(sorted(cells), desc="re-scoring cells"):
        logged = pd.concat([pd.read_parquet(path) for path in cells[cell]], ignore_index=True)
        logged_days.append(logged[_COVERAGE_KEY].drop_duplicates())
        cell_records, cell_features, cell_pairs = check_cell(logged, versions)
        records += cell_records
        features += cell_features
        pairs += cell_pairs
    logged_days = pd.concat(logged_days, ignore_index=True)
    summary = pd.DataFrame(records, columns=_SUMMARY_COLUMNS)
    differing = pd.DataFrame(features, columns=_FEATURE_COLUMNS).sort_values(
        ["cell", "version", "differ_share", "mean_abs_diff"], ascending=[True, True, False, False]
    )
    compared = pd.DataFrame(pairs, columns=_PAIR_COLUMNS)
    gaps = coverage_gaps(logged_days, leagues, first or logged_days["Date"].min(), last)
    tables = {"summary": summary, "feature": differing, "version": compared, "coverage": gaps}
    written = pd.concat({name: table for name, table in tables.items() if not table.empty})
    written.rename_axis(["table", None]).reset_index(level="table").to_csv(out, index=False)

    shown = [
        (
            "self-check and parity (dp: serve minus training features, at the matrix line):",
            summary[summary["note"].isna()].drop(columns="note").astype({"matched": int}),
        ),
        (
            "features that differ most (share of matched rows, mean |serve - matrix|):",
            differing.groupby(["cell", "version"]).head(_WORST_FEATURES),
        ),
        (
            "settled legs in history with no log rows (every date is in the CSV):",
            gaps.groupby(["League", "Market", "Platform"], as_index=False).agg(
                dates=("Date", "nunique"),
                first=("Date", "min"),
                last=("Date", "max"),
                legs=("legs", "sum"),
            ),
        ),
    ]
    if versions:
        title = "version re-score (dp, days: other version minus the logged one; quote line):"
        shown.insert(2, (title, compared[compared["note"].isna()].drop(columns="note")))
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        for title, table in shown:
            print(title)
            print(table.to_string(index=False, float_format="%.4g") if len(table) else "  none")
    for row in summary[summary["note"].notna()].itertuples():
        print(f"excluded {row.cell} {row.version}: {row.note}")
    for row in compared[compared["note"].notna()].itertuples():
        print(f"not re-scored {row.cell} {row.version} under {row.other_version}: {row.note}")
    print(f"wrote {out}")
