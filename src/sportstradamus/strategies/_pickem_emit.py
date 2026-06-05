"""Internal helpers for ``strategies/underdog_pickem.py``.

Keeps the orchestrator under the CLAUDE.md 300-line ceiling by hosting the
rank/dedupe pass and the YAML emit. No public API; all symbols private to
the strategies subpackage.
"""

from __future__ import annotations

import datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from sportstradamus.strategies.underdog_pickem import (
        PickemConfig,
        RecommendedEntry,
    )


def rank_and_dedupe(
    entries: list[RecommendedEntry], config: PickemConfig
) -> list[RecommendedEntry]:
    """Sort by EV-per-dollar, drop near-duplicate entries by leg overlap."""
    ranked = sorted(
        entries,
        key=lambda e: (e.ev * e.joint_prob) / float(max(e.recommended_stake, Decimal("0.01"))),
        reverse=True,
    )
    kept: list[RecommendedEntry] = []
    for e in ranked:
        legs_set = set(e.legs)
        if any(len(legs_set & set(k.legs)) > config.max_overlap for k in kept):
            continue
        kept.append(e)
        if len(kept) >= config.top_k:
            break
    return kept


def emit_yaml(
    entries: list[RecommendedEntry],
    date: datetime.date,
    bankroll: Decimal,
    config: PickemConfig,
    out_path: Path,
) -> Path:
    """Serialize ``entries`` to ``out_path`` and return the path."""
    yaml = _lazy_import("yaml")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.datetime.now().isoformat(),
        "date": date.isoformat(),
        "bankroll": str(bankroll),
        "config": {
            "min_model_edge": config.min_model_edge,
            "min_sharp_edge": config.min_sharp_edge,
            "disagreement_threshold": config.disagreement_threshold,
            "min_ev": config.min_ev,
            "entry_sizes": list(config.entry_sizes),
            "contest_variants": list(config.contest_variants),
            "top_k": config.top_k,
            "max_overlap": config.max_overlap,
            "kelly_fraction": config.kelly_fraction,
            "max_stake_pct_bankroll": config.max_stake_pct_bankroll,
        },
        "entries": [
            {
                "id": e.id,
                "contest_variant": e.contest_variant,
                "entry_size": e.entry_size,
                "legs": list(e.legs),
                "joint_prob": float(e.joint_prob),
                "payout_multiplier": float(e.payout_multiplier),
                "ev": float(e.ev),
                "recommended_stake": str(e.recommended_stake),
                "shrinkage": float(e.shrinkage),
                "shrinkage_source": e.shrinkage_source,
                "extras": e.extras,
            }
            for e in entries
        ],
    }
    with open(out_path, "w") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)
    return out_path


_PICKEM_FRAME_COLS = [
    "id",
    "contest_variant",
    "entry_size",
    "legs",
    "joint_prob",
    "payout_multiplier",
    "ev",
    "shrinkage",
    "shrinkage_source",
    "league",
    "game",
]


def entries_to_frame(entries: list[RecommendedEntry]) -> pd.DataFrame:
    """Flatten ranked entries into the dashboard pickem snapshot frame.

    Bankroll-independent fields only: the dashboard sizes stakes live from a
    user-entered bankroll, so ``recommended_stake`` is intentionally dropped.
    """
    if not entries:
        return pd.DataFrame(columns=_PICKEM_FRAME_COLS)
    return pd.DataFrame(
        [
            {
                "id": e.id,
                "contest_variant": e.contest_variant,
                "entry_size": e.entry_size,
                "legs": list(e.legs),
                "joint_prob": float(e.joint_prob),
                "payout_multiplier": float(e.payout_multiplier),
                "ev": float(e.ev),
                "shrinkage": float(e.shrinkage),
                "shrinkage_source": e.shrinkage_source,
                "league": e.extras.get("league", ""),
                "game": e.extras.get("game", ""),
            }
            for e in entries
        ]
    )


def _lazy_import(name: str) -> Any:
    try:
        return __import__(name)
    except ImportError as exc:
        raise ImportError(
            f"`{name}` is required for pickem-build. Install with `poetry install --with strategy`."
        ) from exc
