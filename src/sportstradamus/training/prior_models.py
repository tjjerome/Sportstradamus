"""Prior model files: the served pickle each retrain replaces, kept for the version-age test.

``meditate`` overwrites ``data/models/{LEAGUE}_{MARKET}.mdl`` in place, so the model behind
an older ``Model Version`` in history would be gone the moment its retrain finishes.
:func:`keep_prior_model` copies the outgoing file to
``data/models/prior/{LEAGUE}_{MARKET}__{version}.mdl`` first, which lets an offline pass score
the same legs with both the new and the previous model. Nothing serves from ``prior/``, and
``scripts/sync_to_prod.sh`` leaves it out of the models mirror, so the copies stay on the box
that trained them.
"""

import pickle
import shutil
import time
from datetime import timedelta
from pathlib import Path

from sportstradamus.helpers.io import MODELS_DIR, resolve_model_version

# Days a superseded model file is kept: the version-age test compares a model's first four
# days with the days after, across a weekly retrain cycle.
PRIOR_MODEL_RETENTION_DAYS = 14


def keep_prior_model(model_path: Path) -> None:
    """Keep the served model a save is about to overwrite as a version-stamped copy in ``prior/``.

    A no-op unless ``model_path`` already exists directly under the canonical models dir: a
    cell's first save keeps nothing, and neither does a sandbox output (``--artifact-output``,
    a deterministic run), whose destination is some other directory.

    The copy is named for the outgoing model's own version stamp
    (:func:`~sportstradamus.helpers.io.resolve_model_version`, the ``Model Version`` serving
    wrote on that model's legs), so each ``Model Version`` in history maps to one prior file.
    Keeping a stamp that is already there replaces its file.

    ``copyfile`` carries no metadata over, so a copy's mtime is the moment its model was
    superseded. A call that keeps a file then deletes every prior file, of any cell, whose
    mtime is more than :data:`PRIOR_MODEL_RETENTION_DAYS` days old.

    Args:
        model_path: Where the caller is about to write the new model pickle.
    """
    if model_path.parent != Path(str(MODELS_DIR)) or not model_path.is_file():
        return
    with model_path.open("rb") as infile:
        version = resolve_model_version(str(model_path), pickle.load(infile))
    prior_dir = model_path.parent / "prior"
    prior_dir.mkdir(exist_ok=True)
    shutil.copyfile(model_path, prior_dir / f"{model_path.stem}__{version}.mdl")
    cutoff = time.time() - timedelta(days=PRIOR_MODEL_RETENTION_DAYS).total_seconds()
    for prior in prior_dir.glob("*.mdl"):
        if prior.stat().st_mtime < cutoff:
            prior.unlink()
