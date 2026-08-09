"""Click options shared across the pipeline CLIs."""

import click

LOG_LEVEL_OPTION = click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Verbosity for the structured JSONL log.",
)
