import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "posttask_diff", pathlib.Path(__file__).parent.parent / "posttask-diff.py"
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_no_drift_returns_none():
    snap = {"head": "a" * 40, "branch": "model-research", "status": ""}
    assert mod.drift_message(snap, dict(snap)) is None


def test_head_move_detected():
    before = {"head": "a" * 40, "branch": "feature-x", "status": ""}
    after = {"head": "b" * 40, "branch": "main", "status": ""}
    msg = mod.drift_message(before, after)
    assert msg is not None
    assert "moved HEAD" in msg
    assert "git checkout feature-x" in msg


def test_new_file_detected():
    before = {"head": "a" * 40, "branch": "main", "status": " M kept.py"}
    after = {"head": "a" * 40, "branch": "main", "status": " M kept.py\n M extra.py"}
    msg = mod.drift_message(before, after)
    assert msg is not None
    assert "extra.py" in msg
    assert "kept.py" not in msg.split("revert with")[-1]
