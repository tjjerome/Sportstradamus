#!/usr/bin/env python
"""
Migrate archive EVs across the fix-compression merge.

Strategy:
  1. Load the pre-merge `stat_dist.json` / `stat_zi.json` from git revision OLD_REV.
  2. Load the current (post-merge) `stat_dist.json` / `stat_zi.json` from disk.
  3. For every (league, market) whose distribution or gate changed, decode each
     archived EV under the old (dist, gate) and re-encode under the new ones.

SkewNormal encoding uses alpha=0 (symmetric Normal), matching the book-side
assumption.

Does NOT write to the archive. Uncomment the `archive.write()` call at the
bottom when ready.
"""

import json
import subprocess
import numpy as np
import importlib.resources as pkg_resources
from tqdm import tqdm

from sportstradamus import data
from sportstradamus.helpers import Archive, get_ev, get_odds


# Predecessor of e4011bf ("Implement SkewNormal + CRPS for compression
# reduction (Phase 2)"), i.e. the last commit before the fix-compression merge.
OLD_REV = "5d1eca7"

# Team-level markets that aren't encoded via get_ev — skip them unconditionally.
SKIP_MARKETS = {"Moneyline", "Totals", "1st 1 innings"}


def _load_json_at_rev(rev: str, repo_path: str) -> dict:
    """Read a JSON blob from a specific git revision."""
    out = subprocess.run(
        ["git", "show", f"{rev}:{repo_path}"],
        capture_output=True, check=True, text=True,
    )
    return json.loads(out.stdout)


def load_current_json(filename: str) -> dict:
    filepath = pkg_resources.files(data) / filename
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------
# stat_cv.json is unchanged between OLD_REV and HEAD, so load it once.
with open(pkg_resources.files(data) / "stat_cv.json") as f:
    stat_cv = json.load(f)

old_stat_dist = _load_json_at_rev(OLD_REV, "src/sportstradamus/data/stat_dist.json")
old_stat_zi = _load_json_at_rev(OLD_REV, "src/sportstradamus/data/stat_zi.json")
new_stat_dist = load_current_json("stat_dist.json")
new_stat_zi = load_current_json("stat_zi.json")


# ---------------------------------------------------------------------------
# Main migration
# ---------------------------------------------------------------------------
def migrate_archive():
    archive = Archive()

    stats = {
        "converted": 0,
        "deleted": 0,
        "skipped_unchanged_dist": 0,
        "skipped_bad_data": 0,
        "errors": 0,
    }

    for league in tqdm(list(archive.archive.keys()), desc="Leagues", unit="league"):
        for market in tqdm(list(archive[league].keys()), desc=f"{league}", unit="market", leave=False):
            if market in SKIP_MARKETS:
                continue

            old_dist = old_stat_dist.get(league, {}).get(market, "Gamma")
            new_dist = new_stat_dist.get(league, {}).get(market, "Gamma")
            old_gate = old_stat_zi.get(league, {}).get(market, 0)
            new_gate = new_stat_zi.get(league, {}).get(market, 0)

            if old_dist == new_dist and old_gate == new_gate:
                stats["skipped_unchanged_dist"] += 1
                continue

            cv = stat_cv.get(league, {}).get(market, 1)

            for date in list(archive[league][market].keys()):
                for player in list(archive[league][market][date].keys()):
                    entry = archive[league][market][date][player]
                    if not isinstance(entry, dict):
                        continue

                    ev_dict = entry.get("EV", {})
                    if not isinstance(ev_dict, dict) or not ev_dict:
                        continue

                    lines = entry.get("Lines", [])
                    if not lines:
                        stats["skipped_bad_data"] += 1
                        continue

                    # Match Archive.get_line's line convention
                    line = np.floor(2 * np.median(lines)) / 2
                    if np.isnan(line) or line <= 0:
                        stats["skipped_bad_data"] += 1
                        continue

                    new_ev_dict = {}
                    changed = False

                    for book, old_ev in ev_dict.items():
                        if old_ev is None or (isinstance(old_ev, float) and np.isnan(old_ev)) or old_ev <= 0:
                            stats["deleted"] += 1
                            continue

                        try:
                            if float(old_ev) > line * 10 or float(old_ev) < line / 10:
                                stats["deleted"] += 1
                                continue

                            under_prob = get_odds(
                                line, float(old_ev),
                                dist=old_dist, cv=cv,
                                gate=(old_gate or None),
                            )

                            if np.isnan(under_prob) or under_prob <= 1e-6 or under_prob >= 1 - 1e-6:
                                stats["deleted"] += 1
                                continue

                            new_ev = get_ev(
                                line, under_prob,
                                cv=cv, dist=new_dist,
                                gate=(new_gate or None),
                                skew_alpha=(0 if new_dist == "SkewNormal" else None),
                            )

                            if new_ev is None or np.isnan(new_ev) or new_ev <= 0:
                                stats["deleted"] += 1
                            else:
                                new_ev_dict[book] = float(new_ev)
                                changed = True
                                stats["converted"] += 1

                        except Exception:
                            stats["deleted"] += 1

                    if changed:
                        archive[league][market][date][player]["EV"] = new_ev_dict

    print("\n--- Migration Summary ---")
    print(f"  Converted:              {stats['converted']}")
    print(f"  Deleted:                {stats['deleted']}")
    print(f"  Skipped (unchanged):    {stats['skipped_unchanged_dist']}")
    print(f"  Skipped (bad data):     {stats['skipped_bad_data']}")
    print(f"  Errors:                 {stats['errors']}")

    # -----------------------------------------------------------------------
    # Uncomment below to persist changes to disk
    # -----------------------------------------------------------------------
    archive.write(all=True)
    print("Archive migrated and written to disk.")

    return archive


if __name__ == "__main__":
    migrate_archive()
