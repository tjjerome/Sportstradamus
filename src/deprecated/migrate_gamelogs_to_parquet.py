"""One-shot migration: ``{league}_data.dat`` pickle -> parquet + JSON sidecar.

Run once after deploying the parquet writers. The legacy ``.dat`` files are
left in place as a rollback parachute; delete them in a follow-up commit
after a verified meditate + prophecize round-trip.

Usage:
    poetry run python -m sportstradamus.scripts.migrate_gamelogs_to_parquet
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import click
import pandas as pd

from sportstradamus.helpers.io import _gamelog_paths, read_gamelog, write_gamelog

LEAGUES = ("nba", "wnba", "mlb", "nfl", "nhl")


def _human_size(path: Path) -> str:
    if not path.is_file():
        return "missing"
    size = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def _frames_match(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if left.shape != right.shape:
        return False
    return set(left.columns) == set(right.columns)


def _migrate(league: str) -> bool:
    paths = _gamelog_paths(league)
    legacy = Path(str(paths["legacy_pickle"]))
    if not legacy.is_file():
        click.echo(f"[skip] {league}: {legacy.name} missing")
        return True

    with legacy.open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        gamelog = obj.get("gamelog", pd.DataFrame())
        teamlog = obj.get("teamlog", pd.DataFrame())
        players = obj.get("players", {})
    else:
        # NFL legacy: bare gamelog DataFrame.
        gamelog, teamlog, players = obj, pd.DataFrame(), {}

    click.echo(
        f"[{league}] pickle: gamelog={getattr(gamelog, 'shape', '?')} "
        f"teamlog={getattr(teamlog, 'shape', '?')} "
        f"players={type(players).__name__}"
    )

    write_gamelog(league, gamelog, teamlog, players)
    rt = read_gamelog(league)

    if not _frames_match(rt["gamelog"], gamelog):
        click.echo(
            f"[{league}] gamelog round-trip mismatch: " f"{rt['gamelog'].shape} vs {gamelog.shape}",
            err=True,
        )
        return False
    if isinstance(teamlog, pd.DataFrame) and not teamlog.empty:
        if not _frames_match(rt["teamlog"], teamlog):
            click.echo(
                f"[{league}] teamlog round-trip mismatch: "
                f"{rt['teamlog'].shape} vs {teamlog.shape}",
                err=True,
            )
            return False
    if isinstance(players, dict):
        if set(rt["players"].keys()) != set(players.keys()):
            click.echo(f"[{league}] players keys mismatch", err=True)
            return False
    elif isinstance(players, pd.DataFrame):
        if rt["players"].shape != players.shape:
            click.echo(
                f"[{league}] players DataFrame shape mismatch: "
                f"{rt['players'].shape} vs {players.shape}",
                err=True,
            )
            return False

    sizes = [
        ("legacy", _human_size(legacy)),
        ("gamelog.parquet", _human_size(Path(str(paths["gamelog"])))),
        ("teamlog.parquet", _human_size(Path(str(paths["teamlog"])))),
    ]
    if Path(str(paths["players_parquet"])).is_file():
        sizes.append(("players.parquet", _human_size(Path(str(paths["players_parquet"])))))
    else:
        sizes.append(("players.json", _human_size(Path(str(paths["players_json"])))))
    click.echo(f"[{league}] sizes: " + " ".join(f"{lbl}={sz}" for lbl, sz in sizes))
    return True


@click.command()
@click.option(
    "--league",
    "leagues",
    multiple=True,
    type=click.Choice(LEAGUES, case_sensitive=False),
    help="Migrate only the listed leagues. Default: all.",
)
def main(leagues: tuple[str, ...]) -> None:
    targets = [lg.lower() for lg in leagues] if leagues else list(LEAGUES)
    ok = True
    for league in targets:
        ok &= _migrate(league)
    if not ok:
        sys.exit(1)
    click.echo("Migration OK. Legacy .dat files left in place as backup.")


if __name__ == "__main__":
    main()
