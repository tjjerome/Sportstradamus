"""Internal helpers for ``strategies/underdog_pickem.py``.

Keeps the orchestrator under the CLAUDE.md 300-line ceiling by hosting the
rank/dedupe pass and the YAML emit. No public API; all symbols private to
the strategies subpackage.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from sportstradamus.strategies.underdog_pickem import (
        PickemConfig,
        RecommendedEntry,
    )


def rank_and_dedupe(
    entries: list[RecommendedEntry], config: PickemConfig
) -> list[RecommendedEntry]:
    """Rank by ``ev`` (the fitted value), highest first; drop near-duplicates by leg overlap."""
    kept: list[RecommendedEntry] = []
    for entry in sorted(entries, key=lambda e: e.ev, reverse=True):
        legs = set(entry.legs)
        if any(len(legs & set(other.legs)) > config.max_overlap for other in kept):
            continue
        kept.append(entry)
        if len(kept) >= config.top_k:
            break
    return kept


def emit_yaml(
    entries: list[RecommendedEntry],
    date: datetime.date,
    config: PickemConfig,
    out_path: Path,
) -> Path:
    """Write the recommendations YAML that ``bet kelly`` re-sizes, and return its path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.datetime.now().isoformat(),
        "date": date.isoformat(),
        "config": {
            "entry_sizes": list(config.entry_sizes),
            "contest_variants": list(config.contest_variants),
            "top_k": config.top_k,
            "max_overlap": config.max_overlap,
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
                "extras": e.extras,
                "platform": e.platform,
            }
            for e in entries
        ],
    }
    with open(out_path, "w") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)
    return out_path
