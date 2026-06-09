"""Operation Ship 75 combination search — per-market ranking over decodable normalizations.

For each normalization the search trains one deterministic ``meditate`` trial and reads its
honest val-fit→test gate row: the dumped served predictive already carries the pipeline's
validation-fit joint calibration, so the gates are scored the same way production would score
them — the search never re-fits calibration on the test rows (that in-sample re-fit oversold the
first screen; see docs/operation_ship_75.md §5). Markets are ranked by min-gate ship slack; the
top rows are the real-HPO confirm candidates. The deterministic score *ranks* only — nothing
ships off it.
"""

import importlib.resources as pkg_resources
import pathlib
import subprocess

import click
import pandas as pd

from sportstradamus import data as _data_pkg
from sportstradamus.helpers.io import market_file_slug
from sportstradamus.training.pipeline import LOSS_AUTO
from sportstradamus.training.scorecard import (
    apply_thresholds,
    gate_row,
    load_test_set,
    min_gate_slack,
)

# The normalization corners the SkewNormal gate can decode (and therefore score). The EB slug
# `centered_additive_eb_meanyr_k10` decodes off the dumped `GlobalMean` column (the gate re-adds
# the empirical-Bayes prior) — see docs/operation_ship_75.md §5 (combination search).
_DECODABLE_SN_NORMS: tuple[str, ...] = (
    "ratio_meanyr",
    "centered_additive_mean10",
    "centered_additive_eb_meanyr_k10",
)

_SHIP_PRED_COL = "Blended_EV"
_TEST_SETS_ROOT = pathlib.Path(str(pkg_resources.files(_data_pkg) / "test_sets"))
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_DETERMINISTIC_MODEL_ROOT = _REPO_ROOT / "research" / "models" / "deterministic"
# 30-minute ceiling; a deterministic meditate run is fast-HP, but SN cells with large datasets
# can take ~10 min, so 1800 s keeps CI from hanging without cutting off valid runs.
_MEDITATE_TRIAL_TIMEOUT_S = 1800


def _dump_paths(league: str, market: str, norm: str) -> tuple[pathlib.Path, pathlib.Path]:
    """CSV under ``test_sets/deterministic/<norm>/`` and pickle under ``research/models/deterministic/<norm>/``."""
    filename = market_file_slug(league, market)
    csv = _TEST_SETS_ROOT / "deterministic" / norm / f"{filename}.csv"
    mdl = _DETERMINISTIC_MODEL_ROOT / norm / f"{filename}.mdl"
    return csv, mdl


def _run_deterministic_meditate(
    league: str,
    market: str,
    norm: str,
    dist_training_loss: str = LOSS_AUTO,
    blending_loss_fn: str = LOSS_AUTO,
) -> None:
    """Train one deterministic ``(cell, normalization, dist-loss, blend-loss)`` trial via meditate.

    ``--deterministic`` pins RNGs and the fixed fast hyperparameters and dumps to the research
    sandbox (never production); ``--bypass-withholding`` lets a withheld cell train under the
    forced ``--target-normalization``. Non-``auto`` losses are forwarded as ``--dist-training-loss``
    / ``--blending-loss-fn`` so the search can sweep them; ``auto`` leaves that arg off (command
    byte-identical to the normalization-only trial). The trained model is a *ranking* stand-in.
    """
    cmd = [
        "poetry",
        "run",
        "meditate",
        "--league",
        league,
        "--market",
        market,
        "--deterministic",
        "--bypass-withholding",
        "--target-normalization",
        norm,
    ]
    if dist_training_loss != LOSS_AUTO:
        cmd += ["--dist-training-loss", dist_training_loss]
    if blending_loss_fn != LOSS_AUTO:
        cmd += ["--blending-loss-fn", blending_loss_fn]
    subprocess.run(cmd, cwd=_REPO_ROOT, check=True, timeout=_MEDITATE_TRIAL_TIMEOUT_S)


def _score_normalization(league: str, market: str, norm: str) -> dict[str, object]:
    """Score a trained trial's dump; ``dispersion_cal`` / ``skew_cal`` are read from the pickle for context only."""
    csv_path, mdl_path = _dump_paths(league, market, norm)
    df = load_test_set(csv_path, _SHIP_PRED_COL)
    filedict = pd.read_pickle(mdl_path)
    row = apply_thresholds(
        gate_row(
            df, _SHIP_PRED_COL, league=league, market=market, strategy=norm, decode_strategy=norm
        )
    )
    return {
        "normalization": norm,
        "slack": min_gate_slack(row),
        "ships": bool(row.get("ship")),
        # All five gates (value + pass), not just g4 — so a normalization/loss choice's cost on
        # g1 (Brier non-inferiority) or g5 (ECE) is visible on the board, and g2/g3 + g1 brier
        # skill + central-50 coverage show how close the served EV sits to the outcome.
        "g1_pass": bool(row.get("g1_pass")),
        "g1_brier_diff_ci_hi": row.get("g1_brier_diff_ci_hi"),
        "g1_brier_skill": row.get("g1_brier_skill_score"),
        "g2_pass": bool(row.get("g2_pass")),
        "g2_star_z": row.get("g2_star_z"),
        "g3_pass": bool(row.get("g3_pass")),
        "g3_bench_z": row.get("g3_bench_z"),
        "g4_pass": bool(row.get("g4_pass")),
        "g4_pit_ks": row.get("g4_pit_ks"),
        "g4_pit_ks_max": row.get("g4_pit_ks_max"),
        "g5_pass": bool(row.get("g5_pass")),
        "g5_ece_debiased": row.get("g5_ece_debiased"),
        "central50_coverage": row.get("central50_coverage"),
        "dispersion_cal": filedict.get("dispersion_cal", 1.0),
        "skew_cal": filedict.get("skew_cal") or 0.0,
        "n": row.get("n_rows"),
    }


def search_market(
    league: str,
    market: str,
    *,
    normalizations: tuple[str, ...] = _DECODABLE_SN_NORMS,
    retrain: bool = True,
) -> pd.DataFrame:
    """Rank normalization corners by deterministic ship margin.

    When ``retrain`` is False and a dump exists for a normalization, the existing dump is reused.
    Result is sorted by min-gate slack descending; top rows are real-HPO confirm candidates.
    """
    rows = []
    for norm in normalizations:
        if retrain or not all(p.exists() for p in _dump_paths(league, market, norm)):
            _run_deterministic_meditate(league, market, norm)
        rows.append(_score_normalization(league, market, norm))
    return pd.DataFrame(rows).sort_values("slack", ascending=False, ignore_index=True)


@click.command(name="model-strategy-search")
@click.option("--league", required=True, help="League code, e.g. WNBA.")
@click.option("--market", required=True, help="Market stem, e.g. AST.")
@click.option(
    "--retrain/--reuse-dumps",
    default=True,
    help="Retrain each deterministic trial (default) or reuse an existing dump when present.",
)
def main(league: str, market: str, retrain: bool) -> None:
    """Per-market Operation Ship 75 combination search (deterministic ranking)."""
    table = search_market(league, market, retrain=retrain)
    click.echo(table.to_string(index=False))


if __name__ == "__main__":
    main()
