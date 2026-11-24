"""Every atomic write in ``helpers/io.py`` stages to a temp file of its own beside the target.

The temp name used to be fixed (``<path>.tmp``), so writers of one target shared it: on
2026-11-23 four parallel MLB re-quotes rewrote ``leagues/MLB/game_starts.parquet`` together,
two died on the rename and the store came out corrupt. A failed write removes its temp file
and leaves the target as it was.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest

from sportstradamus.helpers import io

# Enough writers, and rows per write, that the writes overlap: a shared temp file then
# loses a rename or tears the target.
_WRITERS = 8
_ROWS = 20_000


class _FailsMidWrite:
    """A frame whose parquet write dies after half writing its file."""

    def to_parquet(self, path, **_kwargs):
        Path(path).write_bytes(b"PAR1 half a file")
        raise OSError("disk full")


def test_each_write_stages_to_its_own_temp_beside_the_target(tmp_path):
    target = tmp_path / "game_starts.parquet"

    with io._atomic_tmp(target) as first, io._atomic_tmp(target) as second:
        assert first != second
        assert first.parent == second.parent == tmp_path
        assert first.name.endswith(".tmp")
        assert second.name.endswith(".tmp")
        first.write_text("first")
        second.write_text("second")

    assert target.read_text() == "first"
    assert list(tmp_path.iterdir()) == [target]


def test_concurrent_writers_of_one_target_all_land_a_whole_file(tmp_path):
    target = tmp_path / "game_starts.parquet"
    frames = [pd.DataFrame({"writer": writer, "row": range(_ROWS)}) for writer in range(_WRITERS)]
    start = threading.Barrier(_WRITERS)

    def write(frame):
        start.wait()
        io._atomic_write_parquet(frame, target)

    with ThreadPoolExecutor(_WRITERS) as pool:
        list(pool.map(write, frames))

    landed = pd.read_parquet(target)
    assert landed.equals(frames[landed["writer"].iloc[0]])
    assert list(tmp_path.iterdir()) == [target]


def test_a_failed_write_removes_its_temp_and_keeps_the_target(tmp_path):
    target = tmp_path / "game_starts.parquet"
    stored = pd.DataFrame({"row": [1, 2, 3]})
    io._atomic_write_parquet(stored, target)

    with pytest.raises(OSError, match="disk full"):
        io._atomic_write_parquet(_FailsMidWrite(), target)

    pd.testing.assert_frame_equal(pd.read_parquet(target), stored)
    assert list(tmp_path.iterdir()) == [target]
