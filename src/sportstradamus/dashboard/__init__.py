"""Sportstradamus Dashboard — Streamlit entry point.

Launched via `poetry run dashboard`. Uses subprocess to start Streamlit
with the main app file.
"""

import subprocess
import sys
from pathlib import Path

import click


@click.command()
def run() -> None:
    """Launch the Sportstradamus Streamlit dashboard.

    Production runs unattended under systemd, so the source-file watcher is
    disabled. The watcher's only effect is the "Source file changed, rerun?"
    popup, which misleads users — clicking "Rerun" only re-executes the
    script, it does not invalidate the data cache, so the displayed data
    looks unchanged. Data freshness is handled by mtime-keyed caches in
    ``dashboard/data.py``.
    """
    app_path = Path(__file__).parent / "app.py"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.fileWatcherType",
            "none",
            "--server.runOnSave",
            "false",
        ],
        check=False,
    )
