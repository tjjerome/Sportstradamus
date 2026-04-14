#!/usr/bin/env python
"""
Migrate archive EVs from old distribution encoding (Gamma/NegBin) to SkewNormal (alpha=0) encoding.

For markets that are being switched to SkewNormal + CRPS:
  1. Decode old EV → recover implied under-probability using old distribution
  2. Re-encode → compute new EV using SkewNormal with alpha=0 (Normal distribution)

The assumption is that books are using alpha=0 (symmetric Normal) for the archive,
which simplifies the encoding/decoding.

Does NOT write to the archive. Uncomment the archive.write() call at the bottom when ready.
"""

import json
import os
import numpy as np
import importlib.resources as pkg_resources
from scipy.stats import norm
from tqdm import tqdm

from sportstradamus import data
from sportstradamus.helpers import Archive, get_ev, get_odds


def load_distribution_config():
    """Load distribution configuration from stat_dist.json."""
    filepath = pkg_resources.files(data) / "stat_dist.json"
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def load_zi_config():
    """Load zero-inflation gate configuration from stat_zi.json."""
    filepath = pkg_resources.files(data) / "stat_zi.json"
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# ---------------------------------------------------------------------------
# Load CV / distribution configs
# ---------------------------------------------------------------------------
with open(pkg_resources.files(data) / "stat_cv.json") as f:
    stat_cv = json.load(f)

stat_dist = load_distribution_config()
stat_zi = load_zi_config()

# Markets that will use SkewNormal (based on mean >= 2 rule)
# These are typically high-mean markets (yards, points, etc.)
SKEWNORM_MARKETS = {
    "NFL": ["passing yards", "rushing yards", "receiving yards", "yards", "qb yards",
            "fantasy points prizepicks", "fantasy points underdog"],
    "NBA": ["MIN", "PRA", "PR", "PA", "FGA", "fantasy points prizepicks"],
    "WNBA": ["MIN", "PA", "PR", "PRA", "FGA", "fantasy points prizepicks"],
    "NHL": ["timeOnIce"],
    "MLB": []
}

# Markets stored at the team level — not encoded via get_ev, so skip them
SKIP_MARKETS = {"Moneyline", "Totals", "1st 1 innings"}


# ---------------------------------------------------------------------------
# Main migration
# ---------------------------------------------------------------------------
def migrate_archive():
    archive = Archive()

    stats = {"converted": 0, "skipped_no_change": 0, "skipped_bad_data": 0, "errors": 0}

    for league in tqdm(list(archive.archive.keys()), desc="Leagues", unit="league"):
        markets_to_migrate = SKEWNORM_MARKETS.get(league, [])

        for market in tqdm(list(archive[league].keys()), desc=f"{league}", unit="market", leave=False):
            if market in SKIP_MARKETS or market not in markets_to_migrate:
                continue

            old_cv = stat_cv.get(league, {}).get(market, 1)
            gate = stat_zi.get(league, {}).get(market, 0)

            # New encoding: SkewNormal with alpha=0 (Normal distribution)
            new_dist = "SkewNormal"

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

                    # Use the same line logic as Archive.get_line
                    line = np.floor(2 * np.median(lines)) / 2
                    if np.isnan(line) or line <= 0:
                        stats["skipped_bad_data"] += 1
                        continue

                    new_ev_dict = {}
                    changed = False

                    for book, old_ev in ev_dict.items():
                        if old_ev is None or (isinstance(old_ev, float) and np.isnan(old_ev)) or old_ev <= 0:
                            # Delete problematic EVs (None, NaN, or <= 0)
                            stats["deleted"] = stats.get("deleted", 0) + 1
                            continue

                        try:
                            # Delete if EV is too extreme (likely corrupt archive entries)
                            if float(old_ev) > line * 10 or float(old_ev) < line / 10:
                                stats["deleted"] = stats.get("deleted", 0) + 1
                                continue

                            # Decode using old distribution
                            old_dist = stat_dist.get(league, {}).get(market, "Gamma")
                            old_gate = stat_zi.get(league, {}).get(market, 0)

                            # Use get_odds with the old distribution to invert the EV
                            under_prob = get_odds(line, float(old_ev), old_dist, cv=old_cv, gate=old_gate or None)

                            if np.isnan(under_prob) or under_prob <= 1e-6 or under_prob >= 1 - 1e-6:
                                # Delete EVs that produce extreme probabilities (conversion would fail)
                                stats["deleted"] = stats.get("deleted", 0) + 1
                                continue

                            # Re-encode using SkewNormal (alpha=0, which is Normal)
                            new_ev = get_ev(line, under_prob, cv=old_cv, dist=new_dist, gate=gate or None, skew_alpha=0)

                            if new_ev is None or np.isnan(new_ev) or new_ev <= 0:
                                # Delete EVs that can't be re-encoded
                                stats["deleted"] = stats.get("deleted", 0) + 1
                            else:
                                new_ev_dict[book] = float(new_ev)
                                changed = True
                                stats["converted"] += 1

                        except Exception as exc:
                            # Delete EVs that cause conversion errors
                            stats["deleted"] = stats.get("deleted", 0) + 1

                    if changed:
                        archive[league][market][date][player]["EV"] = new_ev_dict

    print("\n--- Migration Summary ---")
    print(f"  Converted:        {stats['converted']}")
    print(f"  Skipped (no chg): {stats['skipped_no_change']}")
    print(f"  Skipped (bad):    {stats['skipped_bad_data']}")
    print(f"  Errors:           {stats['errors']}")

    # -----------------------------------------------------------------------
    # Uncomment below to persist changes to disk
    # -----------------------------------------------------------------------
    # archive.write(all=True)
    # print("Archive migrated and written to disk.")

    return archive


if __name__ == "__main__":
    migrate_archive()
