"""Prior model files and test sets: what each retrain replaces, kept a season for offline reads.

``meditate`` overwrites ``data/models/{LEAGUE}_{MARKET}.mdl`` and
``data/test_sets/{LEAGUE}_{MARKET}.csv`` in place, so the model behind an older ``Model Version``
in history, and the test set it was graded on, would be gone the moment its retrain finishes.
:func:`keep_prior_model` first copies the outgoing pair to
``data/models/prior/{LEAGUE}_{MARKET}__{version}.mdl`` and
``data/test_sets/prior/{LEAGUE}_{MARKET}__{version}.csv.gz``, which lets an offline pass score
the same legs with each served version and re-run that version's offline reads on its own test
set. The copies are flat files, not per-version folders: one glob per folder prunes them, the
shared name joins a test set to its pickle, and no ``*.csv`` glob matches a ``.csv.gz``. Nothing
serves from either ``prior/``, and ``scripts/sync_to_prod.sh`` sends neither, so the copies stay
on the box that trained them.

A withheld cell stops training, so :func:`retire_test_set` moves its last test set into the same
``prior/`` rather than leave it among the current dumps.
"""

import gzip
import importlib.resources as pkg_resources
import pickle
import shutil
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sportstradamus import data
from sportstradamus.helpers.io import (
    MODELS_DIR,
    VOLUME_STATS,
    market_file_slug,
    model_pickle_path,
    resolve_model_version,
)

# Days a superseded model file and its test set are kept: a season, so each served
# Model Version can be re-scored offline with its own model file and its own test set.
PRIOR_MODEL_RETENTION_DAYS = 400
# Stamp for a test set kept after its model file is gone: the CSV's own modification time in UTC.
# The date leads, as in a model version, so a cell's copies still sort by date; ``mtime`` in the
# middle field says the stamp came from the file, not a model.
MTIME_STAMP_FORMAT = "%Y%m%d.mtime.%H%M%S"


def _gzip_to_prior(test_set_path: Path, stamp: str) -> None:
    prior_dir = test_set_path.parent / "prior"
    prior_dir.mkdir(exist_ok=True)
    kept = prior_dir / f"{test_set_path.stem}__{stamp}.csv.gz"
    with test_set_path.open("rb") as infile, gzip.open(kept, "wb") as outfile:
        shutil.copyfileobj(infile, outfile)


def keep_prior_model(model_path: Path, test_set_path: Path) -> None:
    """Keep the served model and test set a save is about to overwrite as version-stamped copies.

    A no-op unless ``model_path`` already exists directly under the canonical models dir: a
    cell's first save keeps nothing, and neither does a sandbox output (``--artifact-output``,
    a deterministic run), whose destination is some other directory.

    The model copy is named for the outgoing model's own version stamp
    (:func:`~sportstradamus.helpers.io.resolve_model_version`, the ``Model Version`` serving
    wrote on that model's legs), so each ``Model Version`` in history maps to one prior file.
    The test set still on disk was dumped with that model, so it is gzipped to the test sets'
    ``prior/`` under the same stamp; a model whose test set was never dumped is kept alone.
    Keeping a stamp that is already there replaces its files.

    Both copies are freshly written files, so a copy's mtime is the moment its model was
    superseded. A call that keeps a model then deletes every prior model and test set, of any
    cell, whose mtime is more than :data:`PRIOR_MODEL_RETENTION_DAYS` days old.

    Args:
        model_path: Where the caller is about to write the new model pickle.
        test_set_path: Where the caller is about to write the new test-set CSV.
    """
    if model_path.parent != Path(str(MODELS_DIR)) or not model_path.is_file():
        return
    with model_path.open("rb") as infile:
        version = resolve_model_version(str(model_path), pickle.load(infile))
    model_prior_dir = model_path.parent / "prior"
    model_prior_dir.mkdir(exist_ok=True)
    shutil.copyfile(model_path, model_prior_dir / f"{model_path.stem}__{version}.mdl")
    if test_set_path.is_file():
        _gzip_to_prior(test_set_path, version)
    cutoff = time.time() - timedelta(days=PRIOR_MODEL_RETENTION_DAYS).total_seconds()
    test_set_prior_dir = test_set_path.parent / "prior"
    for prior in [*model_prior_dir.glob("*.mdl"), *test_set_prior_dir.glob("*.csv.gz")]:
        if prior.stat().st_mtime < cutoff:
            prior.unlink()


def retire_test_set(league: str, market: str) -> bool:
    """Move a withheld cell's test set out of ``data/test_sets/`` into its ``prior/``, gzipped.

    ``meditate`` no longer trains the cell, so its last test set would sit among the fresh dumps
    and any reader that globs ``test_sets/*.csv`` would mix eras. While the cell's model file is
    still on disk, :func:`keep_prior_model` keeps the pair under the model's version stamp, so
    call this before ``prune_model_pickle`` deletes the model. Once the model is gone, the copy
    is stamped from the CSV's own modification time (:data:`MTIME_STAMP_FORMAT`).

    A volume denominator (``VOLUME_STATS``) keeps its test set: ``prune_model_pickle`` keeps its
    model on disk, and the test set stays that model's pair for ``report`` and the next
    retrain's :func:`keep_prior_model`.

    Returns:
        ``True`` if a test set left ``data/test_sets/``.
    """
    if market in VOLUME_STATS[league]:
        return False
    model_path = model_pickle_path(league, market)
    test_sets_dir = Path(str(pkg_resources.files(data) / "test_sets"))
    test_set_path = test_sets_dir / f"{market_file_slug(league, market)}.csv"
    if model_path.is_file():
        keep_prior_model(model_path, test_set_path)
    elif test_set_path.is_file():
        modified = datetime.fromtimestamp(test_set_path.stat().st_mtime, UTC)
        _gzip_to_prior(test_set_path, modified.strftime(MTIME_STAMP_FORMAT))
    retired = test_set_path.is_file()
    test_set_path.unlink(missing_ok=True)
    return retired
