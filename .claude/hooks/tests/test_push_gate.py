import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "push_gate", pathlib.Path(__file__).parent.parent / "push-gate.py"
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_non_push_allows():
    assert mod.decide(False, 100.0, None) == "allow"


def test_push_with_no_code_edits_allows():
    assert mod.decide(True, None, None) == "allow"


def test_push_dirty_newer_than_green_asks():
    assert mod.decide(True, 200.0, 100.0) == "ask"


def test_push_green_missing_asks():
    assert mod.decide(True, 200.0, None) == "ask"


def test_push_green_newer_allows():
    assert mod.decide(True, 100.0, 200.0) == "allow"


def test_is_git_push():
    assert mod.is_git_push("git push origin devel")
    assert mod.is_git_push("git   push")
    assert not mod.is_git_push("git status")
    assert not mod.is_git_push("gh pr create")
