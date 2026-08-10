"""Golden guards for distribution-aware normalization at the meditate boundary."""

from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from sportstradamus.training import cli as training_cli
from sportstradamus.training.ship_config import TARGET_NORM_NONE, WITHHELD


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
    monkeypatch.setattr(training_cli, "_retry_calibrated_if_g4_only", MagicMock())
    monkeypatch.setattr(training_cli, "_enforce_ship_gate", MagicMock(return_value=0))

    args = ["--league", "NFL", "--market", "passing tds"]
    if bypass_withholding:
        args.append("--bypass-withholding")
    if deterministic:
        args.append("--deterministic")
    result = CliRunner().invoke(training_cli.meditate, args)

    assert result.exit_code == 0, result.output
    train_market.assert_called_once()
    assert train_market.call_args.kwargs["target_normalization"] == expected_target
