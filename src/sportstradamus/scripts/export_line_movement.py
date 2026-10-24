#!/usr/bin/env python3
"""Refresh ``current_line_movement.parquet`` without rerunning the whole pipeline.

``prophecize`` already writes this snapshot on every run. This command re-derives
it from the archive against the offers already on disk, so the Board's move column
can track a fresh ``confer`` pass at a faster cadence than the prediction pipeline,
or be backfilled after a run that predates the export.

Usage
-----
    sportstradamus export-line-movement
"""

from __future__ import annotations

import click
import pandas as pd

from sportstradamus.helpers.io import CURRENT_LINE_MOVEMENT_PATH, CURRENT_OFFERS_PATH
from sportstradamus.prediction.cli import snapshot_line_movement


@click.command()
def export_line_movement() -> None:
    """Rebuild the per-offer DFS line-movement snapshot from the archive."""
    offers = pd.read_parquet(CURRENT_OFFERS_PATH, engine="pyarrow")
    movement = snapshot_line_movement(offers)
    click.echo(
        f"Wrote {len(movement)} line-movement rows from {len(offers)} offers "
        f"to {CURRENT_LINE_MOVEMENT_PATH}."
    )


if __name__ == "__main__":
    export_line_movement()
