"""Regenerate training matrices with the §6.3 feature batch, then run the §7.2 deterministic A/B.

Phase ``regen`` rebuilds each cell's matrix via ``get_training_matrix`` (full window,
optional ``--cutoff`` to skip a league's cold-start gamedays) and writes two sidecars:
``M_cand`` (all columns) and ``M_base`` (= ``M_cand`` minus the new batch columns). The
drop-new-cols baseline gives byte-identical rows, so the A/B isolates exactly the new
columns -- no row/method drift (see docs/handoffs/model_improvement_track.md §7.2).

Phase ``ab`` groups cells by their deterministic target-normalization (the only A/B-relevant
training axis: ``none``/ZINB cells fall back to ``ratio_meanyr``; centered cells keep theirs),
swaps each sidecar into the canonical parquet, runs a baseline and a candidate
``meditate --deterministic`` per group, then scorecards every cell with the group's slug as
``--strategy`` so g4 decodes consistently across both arms.

Nothing here is promoted to production: deterministic models land in ``research/`` and sandbox
CSVs in ``data/test_sets/deterministic/``; the canonical training parquet ends as ``M_cand``.
"""
import importlib.resources as pkg_resources
import json
import shutil
import subprocess
from datetime import date
from pathlib import Path

import click
import pandas as pd

from sportstradamus import data as data_pkg
from sportstradamus.helpers.io import market_file_slug
from sportstradamus.stats import StatsNBA, StatsNFL, StatsWNBA
from sportstradamus.training import baselines
from sportstradamus.training.data import trim_matrix
from sportstradamus.training.markets import ALL_MARKETS
from sportstradamus.training.ship_config import STAT_META_PATH

_STATS = {"NBA": StatsNBA, "NFL": StatsNFL, "WNBA": StatsWNBA}
_REPO_ROOT = Path(__file__).resolve().parent.parent
_TRAINING_DATA = Path(str(pkg_resources.files(data_pkg) / "training_data"))
_TEST_SETS = Path(str(pkg_resources.files(data_pkg) / "test_sets"))
_BACKUP = _REPO_ROOT / ".regen_backup"
_WORK = Path("/tmp/regen_ab")
_TRIM_ROWS = 15000
# Default deterministic normalization for cells whose configured slug is not a
# real target-normalization (withheld SkewNormal carrying "none", and the ZINB
# count branch which ignores the slug). Mirrors the meditate --bypass-withholding
# fallback so both A/B arms train under one consistent, decodable normalization.
_DEFAULT_TN = "ratio_meanyr"

# New columns this batch adds to every regenerated matrix. The schema-diff vs the
# pre-regen cache must be a subset of this set; anything else means a profile
# recompute leaked a column into the baseline. The two _n columns are EB sample-count
# inputs -- present in the matrix, never selected as model features.
_KNOWN_NEW_COLS = {
    "Player comps std", "Player comps trend", "Player comps raw", "Player comps z recent",
    "MeanYr_expanding_shifted", "MeanYr_expanding_eb", "MeanYr_expanding_vsopp",
    "MeanYr_expanding_n", "MeanYr_expanding_vsopp_n",
    "PlayerExp_x_DefAvg", "PlayerZ_x_DefPos",
    "Playoff", "SeriesWins", "SeriesLosses", "FacingElimination", "CanClinch",
    "Weekday", "PrimeTime", "RestDiff",
}


def _league_markets(league, markets_arg):
    registry = ALL_MARKETS[league]
    if not markets_arg:
        return registry
    chosen = [m.strip() for m in markets_arg.split(",") if m.strip()]
    unknown = [m for m in chosen if m not in registry]
    if unknown:
        raise click.UsageError(f"markets {unknown!r} not in {league}; valid: {registry!r}")
    return [m for m in registry if m in chosen]


def _tn_group(cell):
    if cell.get("dist") == "SkewNormal":
        tn = cell.get("target_normalization")
        if tn in baselines.TARGET_NORMALIZATION_SLUGS:
            return tn
    return _DEFAULT_TN


def _regen_league(league, cutoff, markets):
    stats = _STATS[league]()
    stats.load()
    base_dir = _WORK / league / "base"
    cand_dir = _WORK / league / "cand"
    base_dir.mkdir(parents=True, exist_ok=True)
    cand_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for market in markets:
        slug = market_file_slug(league, market)
        cand_p = cand_dir / f"{slug}.parquet"
        base_p = base_dir / f"{slug}.parquet"
        if cand_p.exists() and base_p.exists():
            click.echo(f"  {slug}: skip (sidecars exist)")
            continue
        m_cand = trim_matrix(stats.get_training_matrix(market, cutoff), _TRIM_ROWS)
        backup = pd.read_parquet(_BACKUP / f"{slug}.parquet")
        new_cols = set(m_cand.columns) - set(backup.columns)
        gone = sorted(set(backup.columns) - set(m_cand.columns))
        unexpected = new_cols - _KNOWN_NEW_COLS
        if unexpected:
            raise SystemExit(f"{slug}: unexpected new cols {sorted(unexpected)} -- aborting")
        m_cand.drop(columns=sorted(new_cols)).to_parquet(base_p, compression="zstd", index=True)
        m_cand.to_parquet(cand_p, compression="zstd", index=True)
        click.echo(f"  {slug}: rows={len(m_cand)} new={len(new_cols)} gone={gone}")
        rows.append({"slug": slug, "rows": len(m_cand), "n_new": len(new_cols),
                     "gone": ";".join(gone)})
    pd.DataFrame(rows).to_csv(_WORK / league / "regen_summary.csv", index=False)
    click.echo(f"[{league}] regen done: {len(rows)} cells -> {_WORK / league}")


def _run_meditate(league, markets, tn):
    cmd = ["poetry", "run", "meditate", "--deterministic", "--bypass-withholding",
           "--league", league, "--market", ",".join(markets), "--target-normalization", tn]
    click.echo(f"  meditate tn={tn} markets={markets}")
    subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT))


def _scorecard(slug, tn, baseline_csv, candidate_csv):
    cmd = ["poetry", "run", "python", "-m", "sportstradamus.training.scorecard",
           "--baseline", str(baseline_csv), "--candidate", str(candidate_csv),
           "--strategy", tn, "--no-log"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=str(_REPO_ROOT))
    (_WORK / "scorecards").mkdir(parents=True, exist_ok=True)
    (_WORK / "scorecards" / f"{slug}.txt").write_text(out.stdout + "\n--- STDERR ---\n" + out.stderr)
    return out.stdout


def _ab_league(league, markets):
    with open(STAT_META_PATH) as fh:
        meta = json.load(fh)[league]
    groups = {}
    for market in markets:
        groups.setdefault(_tn_group(meta.get(market, {})), []).append(market)

    base_dir = _WORK / league / "base"
    cand_dir = _WORK / league / "cand"
    baseline_stash = _WORK / league / "baseline_csv"
    baseline_stash.mkdir(parents=True, exist_ok=True)

    for tn, mkts in groups.items():
        click.echo(f"[{league}] group tn={tn} ({len(mkts)} cells)")
        _swap_in(base_dir, league, mkts)
        _run_meditate(league, mkts, tn)
        for m in mkts:
            slug = market_file_slug(league, m)
            shutil.copy(_TEST_SETS / "deterministic" / tn / f"{slug}.csv",
                        baseline_stash / f"{slug}.csv")
        _swap_in(cand_dir, league, mkts)
        _run_meditate(league, mkts, tn)

    for tn, mkts in groups.items():
        for m in mkts:
            slug = market_file_slug(league, m)
            cand_csv = _TEST_SETS / "deterministic" / tn / f"{slug}.csv"
            _scorecard(slug, tn, baseline_stash / f"{slug}.csv", cand_csv)
            click.echo(f"  scorecard {slug} -> {_WORK / 'scorecards' / (slug + '.txt')}")
    click.echo(f"[{league}] a/b done; scorecards in {_WORK / 'scorecards'}")


def _swap_in(src_dir, league, markets):
    for m in markets:
        slug = market_file_slug(league, m)
        shutil.copy(src_dir / f"{slug}.parquet", _TRAINING_DATA / f"{slug}.parquet")


@click.command()
@click.option("--league", type=click.Choice(["NFL", "NBA", "WNBA"]), required=True)
@click.option("--cutoff", default=None, help="ISO date lower bound (e.g. 2025-01-01) to skip cold-start gamedays.")
@click.option("--markets", default=None, help="Comma-separated subset of market names; default all league cells.")
@click.option("--phase", type=click.Choice(["regen", "ab", "both"]), default="both")
def main(league, cutoff, markets, phase):
    cutoff_date = date.fromisoformat(cutoff) if cutoff else None
    mkts = _league_markets(league, markets)
    if phase in ("regen", "both"):
        _regen_league(league, cutoff_date, mkts)
    if phase in ("ab", "both"):
        _ab_league(league, mkts)


if __name__ == "__main__":
    main()
