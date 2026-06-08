import importlib.util
import io
import json
import pathlib

import pytest

_spec = importlib.util.spec_from_file_location(
    "research_gate", pathlib.Path(__file__).parent.parent / "research-gate.py"
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

GLOBS = ["*/skew_normal.py"]


def test_shipped_flip_is_gated():
    ti = {
        "file_path": "src/sportstradamus/data/config/stat_meta.json",
        "old_string": '"shipped": "withheld"',
        "new_string": '"shipped": "devel"',
    }
    assert mod.is_gated(ti, GLOBS)


def test_stat_meta_non_ship_edit_not_gated():
    ti = {
        "file_path": "src/sportstradamus/data/config/stat_meta.json",
        "old_string": '"dist": "Gamma"',
        "new_string": '"dist": "ZINB"',
    }
    assert not mod.is_gated(ti, GLOBS)


def test_distribution_family_file_is_gated():
    ti = {"file_path": "src/sportstradamus/skew_normal.py", "old_string": "a", "new_string": "b"}
    assert mod.is_gated(ti, GLOBS)


def test_unrelated_edit_not_gated():
    ti = {"file_path": "src/sportstradamus/books.py", "old_string": "a", "new_string": "b"}
    assert not mod.is_gated(ti, GLOBS)


def test_block_path_exits_2(monkeypatch):
    monkeypatch.setattr(mod, "gated_globs", lambda: ["*/skew_normal.py"])
    monkeypatch.setattr(mod, "brief_or_waiver_exists", lambda: False)
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(json.dumps({"tool_input": {"file_path": "x/skew_normal.py"}})),
    )
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 2

