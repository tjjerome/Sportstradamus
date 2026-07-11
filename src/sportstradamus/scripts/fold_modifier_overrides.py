"""Fold dashboard-captured modifier corrections into the committed configs.

Shell entry point for :func:`helpers.parlay_modifiers.fold_overlay` — invoked
by the nightly ``reflect`` run (no prune), by ``run_job.sh`` after its devel
pull, and by ``sync_from_prod.sh`` on the dev box (both ``--prune``).

Usage
-----
    poetry run python -m sportstradamus.scripts.fold_modifier_overrides [--prune]
"""

import click

from sportstradamus.helpers.parlay_modifiers import fold_overlay


@click.command()
@click.option(
    "--prune",
    is_flag=True,
    help="Drop overlay entries the committed configs already carry (post-pull / dev sync).",
)
def fold_modifier_overrides(prune):
    """Apply data/runtime/modifier_overrides.json onto the committed modifier configs."""
    counts = fold_overlay(prune=prune)
    click.echo(
        f"folded {counts['pairs']} pair modifiers + {counts['rakes']} rakes"
        f" (pruned {counts['pruned']} already-synced)"
    )


if __name__ == "__main__":
    fold_modifier_overrides()
