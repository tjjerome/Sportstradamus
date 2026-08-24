"""Golden guards for distribution-aware normalization at the meditate boundary."""

from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from sportstradamus.training import cli as training_cli
from sportstradamus.training.ship_config import TARGET_NORM_NONE, WITHHELD


def _scaffold_meditate(monkeypatch, tmp_path, *, configured_target, cell_meta):
    """Stub every meditate collaborator; returns the ``train_market`` mock."""
    fake_stats = MagicMock()
    fake_stats.trim_gamelog.return_value = None
    train_market = MagicMock()
    package_root = tmp_path / "package"
    (package_root / "config").mkdir(parents=True)

    monkeypatch.setattr(training_cli, "StatsNFL", lambda: fake_stats)
    monkeypatch.setattr(training_cli, "LazyArchive", MagicMock)
    monkeypatch.setattr(training_cli, "book_weights", {})
    monkeypatch.setattr(training_cli.pkg_resources, "files", lambda _package: package_root)
    monkeypatch.setattr(
        training_cli,
        "load_ship_config",
        lambda *, branch: {"NFL": {"passing tds": configured_target}},
    )
    monkeypatch.setattr(
        training_cli,
        "load_stat_meta",
        lambda _path: {"NFL": {"passing tds": cell_meta}},
    )
    monkeypatch.setattr(training_cli, "fit_book_weights", MagicMock(return_value={}))
    monkeypatch.setattr(training_cli, "correlate", MagicMock())
    monkeypatch.setattr(training_cli, "train_market", train_market)
    monkeypatch.setattr(training_cli, "report", MagicMock())
    monkeypatch.setattr(training_cli, "_warn_ship_gate", MagicMock())
    return train_market


@pytest.mark.parametrize(
    (
        "configured_dist",
        "configured_target",
        "bypass_withholding",
        "deterministic",
        "expected_target",
    ),
    [
        ("DPO", WITHHELD, True, False, TARGET_NORM_NONE),
        ("DPO", TARGET_NORM_NONE, False, False, TARGET_NORM_NONE),
        ("DPO", TARGET_NORM_NONE, False, True, TARGET_NORM_NONE),
        (None, TARGET_NORM_NONE, False, True, "ratio_meanyr"),
    ],
    ids=["withheld-bypass", "served-auto", "deterministic-auto", "unmapped-auto"],
)
def test_cli_target_normalization_matches_effective_family_contract(
    configured_dist,
    configured_target,
    bypass_withholding,
    deterministic,
    expected_target,
    monkeypatch,
    tmp_path,
):
    cell_meta = {}
    if configured_dist is not None:
        cell_meta = {
            "dist": configured_dist,
            "shipped": "withheld",
            "target_normalization": TARGET_NORM_NONE,
            "posthoc": "roe_mean",
            "count_dispersion_objective": "pit_ks",
            "blending": "nll",
        }
    train_market = _scaffold_meditate(
        monkeypatch, tmp_path, configured_target=configured_target, cell_meta=cell_meta
    )

    args = ["--league", "NFL", "--market", "passing tds"]
    if bypass_withholding:
        args.append("--bypass-withholding")
    if deterministic:
        args.append("--deterministic")
    result = CliRunner().invoke(training_cli.meditate, args)

    assert result.exit_code == 0, result.output
    train_market.assert_called_once()
    assert train_market.call_args.kwargs["target_normalization"] == expected_target


def test_cli_withheld_cell_refreshes_matrix_prunes_pickle_skips_training(monkeypatch, tmp_path):
    """A withheld cell (no bypass) still builds its matrix — only the fit is skipped."""
    train_market = _scaffold_meditate(
        monkeypatch,
        tmp_path,
        configured_target=WITHHELD,
        cell_meta={"dist": "DPO", "shipped": "withheld"},
    )
    prune = MagicMock()
    monkeypatch.setattr(training_cli, "prune_model_pickle", prune)

    result = CliRunner().invoke(
        training_cli.meditate, ["--league", "NFL", "--market", "passing tds"]
    )

    assert result.exit_code == 0, result.output
    train_market.assert_called_once()
    assert train_market.call_args.kwargs["matrix_only"] is True
    assert "target_normalization" not in train_market.call_args.kwargs
    prune.assert_called_once_with("NFL", "passing tds")
    assert "withheld — matrix refreshed" in result.output
