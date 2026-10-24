"""Grade every combo cell priced as a sum of its component models' predictives.

Lane B's sweep driver. For each cell it builds the two row-aligned frames
:func:`training.component_sum.component_sum_frame` returns — the incumbent combo
model restricted to the joined rows (A0) and the same rows priced by the NORTA
component sum (A1) — writes both, and runs the offline ship gates plus the
S1/S2/S3 supersede verdict on them. The frames are round-tripped through
``load_test_set`` before grading, so what this prints is exactly what
``sportstradamus ship scorecard --baseline --candidate`` prints on the same files.

A cell below ``--min-paired-n`` is priced and reported but not judged: Gate 4's KS
threshold is ``1.358/sqrt(n)`` and Gate 1's bootstrap CI spans the whole effect, so
a thin join fails for the wrong reason.

    poetry run python -m sportstradamus.scripts.backtest_component_sum \
        --league MLB --rho both
"""

from importlib import resources as pkg_resources
from pathlib import Path

import click
import pandas as pd

from sportstradamus import data
from sportstradamus.helpers.config import combo_props, stat_meta
from sportstradamus.helpers.io import market_file_slug
from sportstradamus.training.component_sum import component_sum_frame
from sportstradamus.training.scorecard import (
    _decode_strategy_for_frame,
    _gate_headline,
    _supersede_headline,
    apply_thresholds,
    gate_row,
    load_test_set,
    supersede_verdict,
)

_PRED_COL = "Blended_EV"

# Gate 4's KS threshold is 1.358/sqrt(n), so below this a cell passes on thinness
# rather than on calibration. Report it, do not read a verdict off it.
_MIN_PAIRED_N = 300


def _combo_cells(leagues: list[str], markets: tuple[str, ...], root: Path) -> list[tuple[str, str]]:
    """Every (league, market) that is a combo cell and has a dumped test set."""
    cells = []
    for league in leagues:
        for market in stat_meta[league]:
            if markets and market not in markets:
                continue
            if market not in combo_props and "fantasy points" not in market:
                continue
            if (root / f"{market_file_slug(league, market)}.csv").is_file():
                cells.append((league, market))
    return cells


def _grade(league: str, market: str, baseline_path: Path, candidate_path: Path) -> dict:
    """Both gate rows and the supersede verdict, read back off the written CSVs."""
    frames = {
        arm: load_test_set(path, _PRED_COL)
        for arm, path in (("baseline", baseline_path), ("candidate", candidate_path))
    }
    rows = {
        arm: apply_thresholds(
            gate_row(
                df,
                _PRED_COL,
                league=league,
                market=market,
                strategy=arm,
                decode_strategy=_decode_strategy_for_frame(df, league, market),
            )
        )
        for arm, df in frames.items()
    }
    verdict = supersede_verdict(
        frames["baseline"],
        frames["candidate"],
        _PRED_COL,
        league=league,
        market=market,
        strategy="component_sum",
    )
    return {"rows": rows, "verdict": verdict}


@click.command()
@click.option("--league", "leagues", multiple=True, help="Leagues to sweep (default: all).")
@click.option("--markets", multiple=True, help="Restrict to these combo markets.")
@click.option(
    "--rho",
    "rho_sources",
    type=click.Choice(["book", "model", "both"]),
    default="both",
    show_default=True,
    help="Component correlation: the shipped same-player residual table, the components' "
    "own out-of-sample residual correlation, or both as separate arms.",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("/tmp/component_sum"),
    show_default=True,
    help="Where the per-arm baseline/candidate CSVs are written.",
)
@click.option(
    "--min-paired-n",
    default=_MIN_PAIRED_N,
    show_default=True,
    help="Below this the cell is priced and reported but its verdict is not read.",
)
@click.option(
    "--mixture-weight",
    type=float,
    help="Grade a linear pool of the sum and the incumbent at this weight on the sum "
    "instead of the sum alone. 0 reproduces the incumbent, 1 is the sum.",
)
@click.option(
    "--test-sets-dir",
    type=click.Path(exists=True, path_type=Path, file_okay=False),
    help="Override the package test_sets directory.",
)
def main(leagues, markets, rho_sources, out_dir, min_paired_n, mixture_weight, test_sets_dir):
    """Price every combo cell from its component models and grade it against the incumbent."""
    root = test_sets_dir or Path(str(pkg_resources.files(data) / "test_sets"))
    cells = _combo_cells(list(leagues) or sorted(stat_meta), markets, root)
    arms = ["book", "model"] if rho_sources == "both" else [rho_sources]
    out_dir.mkdir(parents=True, exist_ok=True)
    click.echo(f"{len(cells)} combo cells x {len(arms)} rho arm(s) -> {out_dir}\n")

    summary = []
    for league, market in cells:
        for rho_source in arms:
            run_label = f"rho={rho_source}" + (
                f" mix={mixture_weight}" if mixture_weight is not None else ""
            )
            head = f"{league} {market} [{run_label}]"
            try:
                candidate, baseline, diag = component_sum_frame(
                    league,
                    market,
                    test_sets_dir=test_sets_dir,
                    rho_source=rho_source,
                    mixture_weight=mixture_weight,
                )
            except ValueError as exc:
                # A withheld cell's dumped test set can carry a strategy identity that no
                # longer resolves, which `load_test_set` refuses. That is a stale artifact,
                # not a verdict — report the cell and keep the sweep alive.
                click.echo(f"{head}: SKIP — {exc}")
                summary.append(
                    {
                        **_identity(
                            league,
                            market,
                            rho_source,
                            {"combo_rows": 0, "provenance": [], "reason": str(exc)},
                        ),
                        "verdict": "SKIP",
                    }
                )
                continue
            if candidate.empty:
                click.echo(f"{head}: SKIP — {diag['reason']}")
                summary.append({**_identity(league, market, rho_source, diag), "verdict": "SKIP"})
                continue
            stem = (
                f"{market_file_slug(league, market)}_{run_label.replace('=', '').replace(' ', '_')}"
            )
            paths = {}
            for arm, frame in (("baseline", baseline), ("candidate", candidate)):
                paths[arm] = out_dir / f"{stem}_{arm}.csv"
                frame.to_csv(paths[arm], index=False)
            graded = _grade(league, market, paths["baseline"], paths["candidate"])
            n = diag["graded_rows"]
            thin = " [THIN — reported, not judged]" if n < min_paired_n else ""
            click.echo(f"\n{head}  n={n}/{diag['combo_rows']} ({n / diag['combo_rows']:.0%}){thin}")
            click.echo(f"  provenance {', '.join(diag['provenance'])}")
            for arm in ("baseline", "candidate"):
                click.echo(f"  {arm:9s} {_gate_headline(graded['rows'][arm])}")
            click.echo(f"  supersede {_supersede_headline(graded['verdict'])}")
            summary.append(
                {
                    **_identity(league, market, rho_source, diag),
                    "verdict": "SUPERSEDE" if graded["verdict"].get("ship") else "HOLD",
                    "thin": bool(thin),
                    "a0_pit_ks": graded["rows"]["baseline"].get("g4_pit_ks"),
                    "a1_pit_ks": graded["rows"]["candidate"].get("g4_pit_ks"),
                    "a0_ece_db": graded["rows"]["baseline"].get("g5_ece_debiased"),
                    "a1_ece_db": graded["rows"]["candidate"].get("g5_ece_debiased"),
                    "a0_brier_diff": graded["rows"]["baseline"].get("g1_brier_diff_mean"),
                    "a1_brier_diff": graded["rows"]["candidate"].get("g1_brier_diff_mean"),
                    "a0_ship": graded["rows"]["baseline"].get("ship"),
                    "a1_ship": graded["rows"]["candidate"].get("ship"),
                }
            )
    out = out_dir / "summary.csv"
    pd.DataFrame(summary).to_csv(out, index=False)
    click.echo(f"\nsummary -> {out}")


def _identity(league: str, market: str, rho_source: str, diag: dict) -> dict:
    """The columns every summary row carries, priced or skipped."""
    return {
        "league": league,
        "market": market,
        "rho": rho_source,
        "n": diag.get("graded_rows", 0),
        "combo_rows": diag["combo_rows"],
        "provenance": "; ".join(diag["provenance"]),
        "reason": diag["reason"],
    }


if __name__ == "__main__":
    main()
