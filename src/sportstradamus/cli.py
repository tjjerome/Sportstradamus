"""Unified ``sportstradamus`` command-line interface.

Every command is mounted lazily: the target module is imported only when its
subcommand is invoked, so ``sportstradamus confer --help`` never pays the
torch import that ``meditate`` needs. Command objects stay defined in their
home modules — this module only names and arranges them.
"""

from __future__ import annotations

import importlib

import click


class LazyGroup(click.Group):
    """Click group whose subcommands are ``"module:attr"`` strings, imported on use."""

    def __init__(self, *args, lazy_subcommands: dict[str, str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lazy_subcommands = lazy_subcommands or {}

    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted(set(super().list_commands(ctx)) | set(self.lazy_subcommands))

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        spec = self.lazy_subcommands.get(cmd_name)
        if spec is None:
            return super().get_command(ctx, cmd_name)
        modname, attr = spec.split(":")
        cmd = getattr(importlib.import_module(modname), attr)
        cmd.name = cmd_name
        return cmd


@click.group(
    cls=LazyGroup,
    lazy_subcommands={
        "prophecize": "sportstradamus.prediction.cli:main",
        "confer": "sportstradamus.moneylines:confer",
        "meditate": "sportstradamus.training.cli:meditate",
        "reflect": "sportstradamus.nightly:run",
        "dashboard": "sportstradamus.dashboard:run",
    },
)
def cli() -> None:
    """Sports prop modeling: odds collection, training, prediction, betting."""


cli.add_command(
    LazyGroup(
        name="bet",
        help="Bet sizing, pick'em building, and ledger tools.",
        lazy_subcommands={
            "kelly": "sportstradamus.strategies.kelly:kelly",
            "pickem": "sportstradamus.strategies.underdog_pickem:pickem_build",
            "ledger-commit": "sportstradamus.strategies.ledger:ledger_commit",
        },
    )
)

cli.add_command(
    LazyGroup(
        name="fetch",
        help="Third-party stat-site snapshot collectors.",
        lazy_subcommands={
            "fp": "sportstradamus.collectors.fantasypoints.cli:fp_fetch",
            "ctg": "sportstradamus.collectors.cleaningtheglass.cli:ctg_fetch",
            "savant": "sportstradamus.collectors.baseballsavant.cli:savant_fetch",
        },
    )
)

cli.add_command(
    LazyGroup(
        name="ship",
        help="Offline ship gates, graduation, and model-strategy sweeps.",
        lazy_subcommands={
            "scorecard": "sportstradamus.training.scorecard:main",
            "graduation": "sportstradamus.scripts.check_graduation:main",
            "config": "sportstradamus.scripts.generate_ship_config:main",
            "sweep": "sportstradamus.training.model_strategy.sweep:main",
        },
    )
)

cli.add_command(
    LazyGroup(
        name="admin",
        help="Recurring archive and data maintenance.",
        lazy_subcommands={
            "merge-archives": "sportstradamus.scripts.merge_archives:main",
            "fold-overrides": "sportstradamus.scripts.fold_modifier_overrides:fold_modifier_overrides",
            "sweep-runaway-odds": "sportstradamus.scripts.sweep_runaway_odds:main",
            "backfill-live-metrics": "sportstradamus.scripts.backfill_live_metrics:main",
            "backfill-history-eras": "sportstradamus.scripts.backfill_history_eras:backfill_history_eras",
            "count-family-screen": "sportstradamus.scripts.count_family_screen:main",
        },
    )
)
