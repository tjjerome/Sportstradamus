"""Sportstradamus Dashboard — Streamlit entry point.

Launched via `poetry run reflect`. Uses subprocess to start Streamlit
with the main app file, which auto-discovers pages in the pages/ directory.
"""

import subprocess
import sys
from pathlib import Path

import click


@click.command()
def run():
    """Launch the Sportstradamus Streamlit dashboard."""
    app_path = Path(__file__).parent / "dashboard_app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path)],
        check=False,
    )
