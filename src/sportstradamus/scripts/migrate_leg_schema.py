"""One-shot migration: Sheets-era leg strings/tuples -> structured schema.

Converts parlay_hist (Leg 1..6 -> legs structs), user_slips (desc JSON ->
legs structs), history (nested Offers tuples -> flat rows, 'nan' platform
normalized, optional Alt Line / closing-prob backfill from the archive),
and the current_* snapshots. Idempotent: each frame is gated on an
unambiguous old-schema marker and skipped when already migrated.

The ``current_offers`` / ``current_game_stories`` conversion is not
reverting a live-writer regression — Task 0.7 already rewired
``prediction/correlation.py`` to emit ``Corr Same`` / ``Corr Opp`` structs
directly, so no current run produces the old ``Team Correlation`` /
``Opp Correlation`` display strings. This step exists only because a
snapshot written by the *pre-Task-0.7* code may still be sitting on disk
(these files are overwritten in place by every ``prophecize`` run, but a
stale copy could persist through a skipped run or a restore).

Usage
-----
    poetry run python -m sportstradamus.scripts.migrate_leg_schema
    poetry run python -m sportstradamus.scripts.migrate_leg_schema --dry-run
    poetry run python -m sportstradamus.scripts.migrate_leg_schema --backfill-close --backfill-alt-lines
"""

from __future__ import annotations

import re
from pathlib import Path

import click
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from sportstradamus import clv
from sportstradamus.analysis import _leg_market_map
from sportstradamus.helpers import Archive, stat_dist, stat_map
from sportstradamus.helpers.io import PARLAY_HIST_PATH, _atomic_write_parquet
from sportstradamus.prediction.cli import (
    _ALT_LINE_TOL_CONTINUOUS,
    _ALT_LINE_TOL_COUNT,
    _COUNT_DISTS,
)
from sportstradamus.prediction.parlay import _resolve_leg_stat

# Sheets-era display-leg encoding: "{Player} {Bet} {Line} {Market} - {Model P}%, {Boost}x".
_LEG_PATTERN = re.compile(r"^(.+?)\s+(Over|Under)\s+([\d.]+)\s+(.+?)\s+-\s+[\d.]+%")

# Old per-game correlation columns (display strings), replaced by Corr Same / Corr Opp
# structs in Task 0.7. A partner clause looks like "Player Over 5.5 PTS (1.23x)".
_CORR_PARTNER_PATTERN = re.compile(r"^(.+?)\s+(Over|Under)\s+([\d.]+)\s+(.+?)\s+\(([\d.]+)x\)$")

# parlay_hist: max legs the Sheets-era schema ever stored (Leg 1..Leg 6).
_MAX_LEGACY_LEGS = 6

# history.parquet: pre-CLV-migration Offers tuples carried only the first six
# fields (Line, Boost, Platform, Bet, Win Prob, Market Prob); the closing trio
# (Close Market Prob, Market CLV, Model CLV) postdates this arity and is
# NaN-padded on read. Frozen from the pre-Task-0.7 ``history_schema.py``
# (``git show 8c4447d:src/sportstradamus/history_schema.py``) — that module no
# longer defines this constant since the live Offers-tuple schema was retired.
_LEGACY_OFFER_ARITY = 6


def _is_nonempty_legs(legs) -> bool:
    """Whether a ``legs`` cell holds at least one leg record.

    A parquet ``list<struct>`` column round-trips through pandas as an
    ``ndarray``, not a ``list`` — plain ``isinstance(x, list)`` silently treats
    every already-migrated row as empty, and a bare ``and legs`` on an ndarray
    raises (ambiguous truth value). Shared by every gate/parser below that
    reads a ``legs`` cell.
    """
    return isinstance(legs, list | np.ndarray) and len(legs) > 0


def _parse_desc(desc, league, game, date, platform):
    """One canonical leg dict from a Sheets-era ``desc`` display string, or None."""
    m = _LEG_PATTERN.match(desc or "")
    if not m:
        return None
    player, bet, line, market = m.groups()
    new_map = _leg_market_map(league, platform, stat_map)
    return {
        "player": player,
        "team": "",
        "market": market.replace("H2H ", ""),
        "stat": _resolve_leg_stat(market, new_map),
        "bet": bet,
        "line": float(line),
        "league": league,
        "game": game,
        "date": date,
        "platform": platform,
        "win_prob": float("nan"),
        "boost": float("nan"),
        "push_prob": float("nan"),
        "kelly": float("nan"),
    }


def _migrate_parlay_hist(path: Path, *, dry_run: bool) -> int:
    """Rename ``Leg 1..6`` display strings to ``legs`` structs. Returns rows touched."""
    name = "parlay_hist.parquet"
    if not path.exists():
        click.echo(f"{name:32s} absent")
        return 0
    if "Leg 1" not in pq.read_schema(path).names:
        click.echo(f"{name:32s} already migrated")
        return 0

    df = pd.read_parquet(path)
    leg_cols = [f"Leg {i}" for i in range(1, _MAX_LEGACY_LEGS + 1) if f"Leg {i}" in df.columns]
    prob_col = "Leg Probs" if "Leg Probs" in df.columns else None

    if dry_run:
        click.echo(f"{name:32s} would convert {leg_cols} -> legs ({len(df)} rows)")
        return len(df)

    df["legs"] = df.apply(lambda r: _row_legacy_legs(r, leg_cols, prob_col), axis=1)
    if "Legs" in df.columns and "Legs Resolved" not in df.columns:
        df = df.rename(columns={"Legs": "Legs Resolved"})
    drop_cols = [*leg_cols, "Fun", "Family", "Desc"]
    if prob_col:
        drop_cols.append(prob_col)
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    _atomic_write_parquet(df, path)
    click.echo(f"{name:32s} converted {len(df)} rows")
    return len(df)


def _row_legacy_legs(row, leg_cols: list[str], prob_col: str | None) -> list[dict]:
    """One row's ``Leg 1..6`` strings -> canonical leg list, win_prob folded in from Leg Probs."""
    probs = row.get(prob_col) if prob_col else None
    probs = list(probs) if isinstance(probs, list | tuple | np.ndarray) else []
    league, game = row.get("League"), row.get("Game")
    date, platform = row.get("Date"), row.get("Platform")
    legs = []
    for i, col in enumerate(leg_cols):
        parsed = _parse_desc(row.get(col), league, game, date, platform)
        if parsed is None:
            continue
        if i < len(probs) and pd.notna(probs[i]):
            parsed["win_prob"] = float(probs[i])
        legs.append(parsed)
    return legs


def _migrate_user_slips(path: Path, *, dry_run: bool) -> int:
    """Convert per-leg ``desc`` strings to canonical leg structs. Returns rows touched."""
    name = "user_slips.parquet"
    if not path.exists():
        click.echo(f"{name:32s} absent")
        return 0

    df = pd.read_parquet(path)
    if df.empty or "legs" not in df.columns:
        click.echo(f"{name:32s} already migrated")
        return 0
    first_legs = next((legs for legs in df["legs"] if _is_nonempty_legs(legs)), None)
    if first_legs is None or "desc" not in first_legs[0]:
        click.echo(f"{name:32s} already migrated")
        return 0

    if dry_run:
        click.echo(f"{name:32s} would convert desc legs ({len(df)} rows)")
        return len(df)

    df["legs"] = df.apply(_row_desc_legs, axis=1)
    _atomic_write_parquet(df, path)
    click.echo(f"{name:32s} converted {len(df)} rows")
    return len(df)


def _row_desc_legs(row) -> list[dict]:
    """One user-slip row's ``desc``-keyed legs -> canonical leg structs."""
    legs = row.get("legs")
    if not _is_nonempty_legs(legs):
        return []
    league, game, platform = row.get("League"), row.get("Game"), row.get("platform")
    date = row.get("saved_at", "")[:10] if isinstance(row.get("saved_at"), str) else ""
    out = []
    for leg in legs:
        parsed = _parse_desc(leg.get("desc"), league, game, date, platform)
        if parsed is not None:
            out.append(parsed)
    return out


# On-disk field order for the pre-Task-0.7 Offers struct (``helpers.io._OFFER_FIELDS``,
# now deleted from that module) — parquet stores each offer as a dict keyed by these
# names, never a raw tuple, so pyarrow can write a homogeneous list<struct> column.
_OFFER_STRUCT_FIELDS = (
    "line",
    "boost",
    "platform",
    "bet",
    "model_p",
    "books_p",
    "close_books_p",
    "market_clv",
    "model_clv",
)


def _explode_offers_legacy(history: pd.DataFrame) -> pd.DataFrame:
    """Frozen copy of the pre-Task-0.7 ``analysis.explode_offers`` tuple-unpack.

    Reads only the historical nested ``Offers`` column shape (removed from the
    live schema by Task 0.7's flat rewrite); adapted from
    ``git show 8c4447d:src/sportstradamus/analysis.py``, with the on-disk
    struct-dict decode folded in from the retired ``helpers.io._offer_dict_to_tuple``.
    """
    if "Offers" not in history.columns:
        return history
    pred_cols = [c for c in history.columns if c != "Offers"]
    rows = []
    for _, pred in history.iterrows():
        offers = pred.get("Offers")
        # A parquet list<struct> column round-trips as an ndarray, not a list.
        if not isinstance(offers, list | np.ndarray) or len(offers) == 0:
            continue
        for offer in offers:
            rows.append(_legacy_offer_row(pred, pred_cols, offer))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _legacy_offer_row(pred, pred_cols: list[str], offer) -> dict:
    padded = _pad_legacy_offer(offer)
    line, boost, platform, bet, model_p, books_p, close_p, market_clv, model_clv = padded
    row = {col: pred[col] for col in pred_cols}
    row.update(
        {
            "Line": line,
            "Boost": boost,
            "Platform": platform,
            "Bet": bet,
            "Win Prob": model_p,
            "Market Prob": books_p,
            "Close Market Prob": close_p,
            "Market CLV": market_clv,
            "Model CLV": model_clv,
        }
    )
    return row


def _pad_legacy_offer(offer):
    """Normalize one on-disk offer (struct-dict or tuple/list) to the 9-field tuple.

    A dict is the real parquet-on-disk shape (see :data:`_OFFER_STRUCT_FIELDS`); a
    bare 6-tuple is the pre-CLV in-memory shape, NaN-padded for the closing trio.
    """
    if isinstance(offer, dict):
        return tuple(offer.get(f, np.nan) for f in _OFFER_STRUCT_FIELDS)
    if isinstance(offer, tuple | list) and len(offer) == _LEGACY_OFFER_ARITY:
        return (*tuple(offer), np.nan, np.nan, np.nan)
    return tuple(offer)


# A close-market probability above this is a corrupted archive read (probabilities
# are bounded in [0, 1]); quarantine rather than silently keep a garbage value.
_CLOSE_PROB_MAX = 1.0


def _migrate_history(
    path: Path, *, dry_run: bool, backfill_close: bool, backfill_alt_lines: bool
) -> int:
    """Explode nested ``Offers`` to the flat schema. Returns rows touched."""
    name = "history.parquet"
    if not path.exists():
        click.echo(f"{name:32s} absent")
        return 0
    if "Offers" not in pq.read_schema(path).names:
        click.echo(f"{name:32s} already migrated")
        return 0

    history = pd.read_parquet(path)
    if dry_run:
        click.echo(f"{name:32s} would explode Offers ({len(history)} predictions)")
        return len(history)

    flat = _explode_offers_legacy(history)
    flat["Platform"] = flat["Platform"].replace("nan", None)

    quarantined = flat["Close Market Prob"] > _CLOSE_PROB_MAX
    n_quarantined = int(quarantined.sum())
    if n_quarantined:
        flat.loc[quarantined, ["Close Market Prob", "Market CLV", "Model CLV"]] = np.nan

    if backfill_close:
        flat = clv.fill_from_archive(flat, Archive())
    if backfill_alt_lines:
        flat = _backfill_alt_lines(flat, Archive())

    _atomic_write_parquet(flat, path)
    click.echo(
        f"{name:32s} exploded to {len(flat)} rows ({n_quarantined} closing-trio quarantined)"
    )
    return len(flat)


def _backfill_alt_lines(flat: pd.DataFrame, archive) -> pd.DataFrame:
    """Stamp ``Alt Line`` from the archived consensus line where recoverable.

    Tolerance routes on distribution family exactly like
    ``prediction.cli._stamp_alt_line`` (count stats move in coarser steps than
    continuous ones) — reusing those live constants directly, not a frozen
    copy, so a backfilled historical flag means the same thing as one stamped
    at write time even if the tolerances are retuned later.
    """
    if "Alt Line" not in flat.columns:
        flat["Alt Line"] = False
    pending = flat.loc[flat["Alt Line"].isna()]
    for key, idx in tqdm(
        pending.groupby(["League", "Market", "Date", "Player"], dropna=False).groups.items(),
        desc="Backfilling Alt Line",
    ):
        league, market, date, player = key
        if not (isinstance(league, str) and isinstance(market, str) and isinstance(player, str)):
            continue
        consensus = archive.get_line(league, market, date, player)
        if not consensus:
            continue
        dist = stat_dist.get(league, {}).get(market)
        tol = _ALT_LINE_TOL_COUNT if dist in _COUNT_DISTS else _ALT_LINE_TOL_CONTINUOUS
        rows = flat.loc[idx]
        is_alt = (rows["Line"] - consensus).abs() > tol
        flat.loc[idx, "Alt Line"] = is_alt
    return flat


def _parse_corr_partners(text: str) -> list[dict]:
    """Comma-joined display-string correlation partners -> structured records."""
    partners = []
    for clause in filter(None, (c.strip() for c in (text or "").split(","))):
        m = _CORR_PARTNER_PATTERN.match(clause)
        if not m:
            continue
        player, bet, line, market, mult = m.groups()
        partners.append(
            {
                "player": player,
                "market": market,
                "bet": bet,
                "line": float(line),
                "mult": float(mult),
            }
        )
    return partners


def _migrate_current_offers(path: Path, *, dry_run: bool) -> int:
    """Rename ``Team Correlation`` / ``Opp Correlation`` strings to struct columns."""
    name = "current_offers.parquet"
    if not path.exists():
        click.echo(f"{name:32s} absent")
        return 0
    names = set(pq.read_schema(path).names)
    if "Team Correlation" not in names:
        click.echo(f"{name:32s} already migrated")
        return 0

    df = pd.read_parquet(path)
    if dry_run:
        click.echo(f"{name:32s} would convert correlation cols ({len(df)} rows)")
        return len(df)

    df["Corr Same"] = df["Team Correlation"].apply(_parse_corr_partners)
    df["Corr Opp"] = df["Opp Correlation"].apply(_parse_corr_partners)
    df = df.drop(columns=["Team Correlation", "Opp Correlation"])
    _atomic_write_parquet(df, path)
    click.echo(f"{name:32s} converted {len(df)} rows")
    return len(df)


def _row_game_story_legs(row) -> list[dict]:
    """One current-game-story row's ``desc``-keyed legs -> canonical leg structs."""
    league, game = row.get("League"), row.get("Game")
    date, platform = row.get("Date"), row.get("Platform")
    out = []
    for leg in row["legs"]:
        parsed = _parse_desc(leg.get("desc"), league, game, date, platform)
        if parsed is not None:
            out.append(parsed)
    return out


def _migrate_current_game_stories(path: Path, *, dry_run: bool) -> int:
    """Rename legacy stringly ``legs`` (Sheets-era ``desc`` shape) to canonical structs."""
    name = "current_game_stories.parquet"
    if not path.exists():
        click.echo(f"{name:32s} absent")
        return 0
    df = pd.read_parquet(path)
    if df.empty or "legs" not in df.columns:
        click.echo(f"{name:32s} already migrated")
        return 0
    first_legs = next((legs for legs in df["legs"] if _is_nonempty_legs(legs)), None)
    if first_legs is None or "desc" not in first_legs[0]:
        click.echo(f"{name:32s} already migrated")
        return 0

    if dry_run:
        click.echo(f"{name:32s} would convert desc legs ({len(df)} rows)")
        return len(df)

    df["legs"] = df.apply(_row_game_story_legs, axis=1)
    _atomic_write_parquet(df, path)
    click.echo(f"{name:32s} converted {len(df)} rows")
    return len(df)


@click.command()
@click.option(
    "--runtime-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Runtime snapshot directory (defaults to the packaged data/runtime).",
)
@click.option("--dry-run", is_flag=True, help="Report planned conversions without writing.")
@click.option(
    "--backfill-close",
    is_flag=True,
    help="Re-derive closing probabilities from the archive for quarantined history rows.",
)
@click.option(
    "--backfill-alt-lines",
    is_flag=True,
    help="Stamp Alt Line on history rows from the archive's consensus line where recoverable.",
)
def main(
    runtime_dir: Path | None, dry_run: bool, backfill_close: bool, backfill_alt_lines: bool
) -> None:
    base = runtime_dir or Path(str(PARLAY_HIST_PATH)).parent
    counts = {
        "parlay_hist.parquet": _migrate_parlay_hist(base / "parlay_hist.parquet", dry_run=dry_run),
        "history.parquet": _migrate_history(
            base / "history.parquet",
            dry_run=dry_run,
            backfill_close=backfill_close,
            backfill_alt_lines=backfill_alt_lines,
        ),
        "user_slips.parquet": _migrate_user_slips(base / "user_slips.parquet", dry_run=dry_run),
        "current_offers.parquet": _migrate_current_offers(
            base / "current_offers.parquet", dry_run=dry_run
        ),
        "current_game_stories.parquet": _migrate_current_game_stories(
            base / "current_game_stories.parquet", dry_run=dry_run
        ),
    }
    click.echo()
    click.echo("Summary:")
    for fname, n in counts.items():
        click.echo(f"  {fname:32s} {n} rows")

    if dry_run:
        return

    partial = [
        fname
        for fname in ("parlay_hist.parquet", "history.parquet", "user_slips.parquet")
        if (base / fname).exists() and _is_partially_migrated(base / fname)
    ]
    if partial:
        raise SystemExit(f"partially migrated after run: {partial}")


def _is_partially_migrated(path: Path) -> bool:
    """Whether an old-schema marker still survives a completed (non-dry-run) migration."""
    names = set(pq.read_schema(path).names)
    if path.name == "parlay_hist.parquet":
        return "Leg 1" in names
    if path.name == "history.parquet":
        return "Offers" in names
    if path.name == "user_slips.parquet":
        df = pd.read_parquet(path, columns=["legs"]) if "legs" in names else pd.DataFrame()
        first_legs = next((legs for legs in df.get("legs", []) if _is_nonempty_legs(legs)), None)
        return first_legs is not None and "desc" in first_legs[0]
    return False


if __name__ == "__main__":
    main()
