"""Build the shared serving files for a one-league send (``scripts/sync_to_prod.sh --league``).

``book_weights.json``, ``stat_calibration.json`` and ``model_stats.{parquet,csv}`` each hold
every league, so sending them whole carries every league's values. This takes production's
copies (fetched read-only into ``--prod-dir``) and replaces only one league's entries with the
training box's.

Pandas and json only; opens no archive. Writes into ``--out-dir`` and prints the proof: per
league, how many entries differ from production's copy (only the named league may), and that
the named league's entries equal the training box's. Exits 1 when the proof fails, which is
what stops the send.
"""

import json
import sys
from pathlib import Path

import click
import pandas as pd

# model_stats holds one row per (league, market) cell.
CELL_KEY = ["league", "market"]


def _leaves(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _leaves(value, (*path, key))
    else:
        yield path, node


def _merge_json(
    league_key: str, here_path: Path, prod_path: Path, out_path: Path, *, trailing_newline: bool
) -> bool:
    here, prod = json.loads(here_path.read_text()), json.loads(prod_path.read_text())
    merged = {
        league: (here[league_key] if league == league_key else cells)
        for league, cells in prod.items()
    }
    text = json.dumps(merged, indent=4) + ("\n" if trailing_newline else "")
    out_path.write_text(text)
    back = json.loads(out_path.read_text())
    ok = back[league_key] == here[league_key] and list(back) == list(prod)
    print(f"{out_path.name}: leagues {list(back)}")
    for league in back:
        mine, theirs = dict(_leaves(back[league])), dict(_leaves(prod[league]))
        differ = sum(mine.get(path) != theirs.get(path) for path in set(mine) | set(theirs))
        print(f"    {league:5} entries {len(mine):4} | differ from production {differ}")
        ok &= differ == 0 or league == league_key
    return ok


def _same(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    return left.shape == right.shape and bool(
        ((left == right) | (left.isna() & right.isna())).all().all()
    )


def _merge_stats(league_key: str, here_path: Path, prod_path: Path, out_dir: Path) -> bool:
    here, prod = pd.read_parquet(here_path), pd.read_parquet(prod_path)
    if list(here.columns) != list(prod.columns) or not (here.dtypes == prod.dtypes).all():
        print("model_stats: the two boxes' tables differ in columns or types; not merged")
        return False
    first = int((prod["league"] == league_key).to_numpy().argmax())
    merged = pd.concat(
        [
            prod.iloc[:first][prod.iloc[:first]["league"] != league_key],
            here[here["league"] == league_key],
            prod.iloc[first:][prod.iloc[first:]["league"] != league_key],
        ],
        ignore_index=True,
    ).astype(prod.dtypes.to_dict())
    merged.to_parquet(out_dir / "model_stats.parquet", index=False)
    merged.to_csv(out_dir / "model_stats.csv", index=False)
    back = pd.read_parquet(out_dir / "model_stats.parquet")
    ok = (back.dtypes == prod.dtypes).all() and not back.duplicated(CELL_KEY).any()
    print(
        f"model_stats: rows {len(back)} (production {len(prod)}, training box {len(here)})"
        f" | columns {back.shape[1]}"
    )
    for league in back["league"].drop_duplicates():
        mine = back[back["league"] == league].set_index(CELL_KEY).sort_index()
        theirs = prod[prod["league"] == league].set_index(CELL_KEY).sort_index()
        cells = sorted(set(mine.index) | set(theirs.index))
        differ = sum(
            key not in mine.index
            or key not in theirs.index
            or not _same(mine.loc[[key]], theirs.loc[[key]])
            for key in cells
        )
        print(f"    {league:5} cells {len(mine):3} | differ from production {differ}")
        ok &= differ == 0 or league == league_key
    ours = here[here["league"] == league_key].set_index(CELL_KEY).sort_index()
    ok &= _same(back[back["league"] == league_key].set_index(CELL_KEY).sort_index(), ours)
    csv_back = pd.read_csv(out_dir / "model_stats.csv")
    ok &= len(csv_back) == len(back) and list(csv_back.columns) == list(back.columns)
    return bool(ok)


@click.command()
@click.option(
    "--league",
    "league_key",
    required=True,
    help="The one league whose entries are replaced, e.g. NFL.",
)
@click.option(
    "--here-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="The training box's src/sportstradamus/data folder.",
)
@click.option(
    "--prod-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Production's book_weights.json, stat_calibration.json and model_stats.parquet.",
)
@click.option(
    "--out-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Where the merged files are written.",
)
def main(league_key: str, here_dir: Path, prod_dir: Path, out_dir: Path) -> None:
    """Write production's shared serving files with one league's entries replaced."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Each file ends as its own writer leaves it: `meditate` dumps the book weights bare,
    # training/config.py closes the calibration file with a newline.
    checks = [
        _merge_json(
            league_key,
            here_dir / "config/book_weights.json",
            prod_dir / "book_weights.json",
            out_dir / "book_weights.json",
            trailing_newline=False,
        ),
        _merge_json(
            league_key,
            here_dir / "config/stat_calibration.json",
            prod_dir / "stat_calibration.json",
            out_dir / "stat_calibration.json",
            trailing_newline=True,
        ),
        _merge_stats(
            league_key,
            here_dir / "training/model_stats.parquet",
            prod_dir / "model_stats.parquet",
            out_dir,
        ),
    ]
    print(
        "PASS" if all(checks) else "FAIL",
        f"only {league_key} differs from production, and {league_key} equals the training box's",
    )
    sys.exit(0 if all(checks) else 1)


if __name__ == "__main__":
    main()
