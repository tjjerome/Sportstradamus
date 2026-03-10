#!/usr/bin/env python
"""
Remap archive EVs from old stat_cv encoding to new stat_cv + stat_dist encoding.

Old system:
  - cv == 1  → Poisson encoding:  get_ev solved  under = poisson.cdf(ceil(line-1), ev)
  - cv != 1  → Gaussian encoding: get_ev solved  under = norm.cdf(line, ev, ev*cv)

New system:
  - NegBin (cv = 1/r): EV is the NegBin mean matching get_odds's nbinom.cdf
  - SkewNormal / Gaussian (cv = std/mean): same Normal CDF encoding as before

Process for each stored EV:
  1. Decode: recover the implied under‑probability using the OLD cv
  2. Re-encode: compute a new EV using the NEW cv + distribution

Does NOT write to the archive. Uncomment the archive.write() call at the bottom when ready.
"""

import json
import os
import numpy as np
import importlib.resources as pkg_resources
from scipy.stats import poisson, norm, nbinom
from scipy.optimize import brentq
from tqdm import tqdm

from sportstradamus import data
from sportstradamus.helpers import Archive, get_ev

# ---------------------------------------------------------------------------
# Load CV / distribution configs
# ---------------------------------------------------------------------------
with open(pkg_resources.files(data) / "old_stat_cv.json") as f:
    old_stat_cv = json.load(f)

with open(pkg_resources.files(data) / "stat_cv.json") as f:
    new_stat_cv = json.load(f)

with open(pkg_resources.files(data) / "stat_dist.json") as f:
    stat_dist = json.load(f)

# Leagues that borrow another league's CV / distribution settings
LEAGUE_ALIASES = {
    "NCAAB": "NBA",
    "NCAAF": "NFL",
}

# Markets stored at the team level — not encoded via get_ev, so skip them
SKIP_MARKETS = {"Moneyline", "Totals", "1st 1 innings"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_league(league):
    """Return the canonical league used for CV / dist lookups."""
    return LEAGUE_ALIASES.get(league, league)


def decode_old_ev(line, ev, old_cv):
    """
    Invert the old get_ev encoding to recover the implied under‑probability.

    Old get_ev:
      cv == 1  →  Poisson:  adjusts line to ceil(line-1), solves  under = poisson.cdf(adj, ev)
      cv != 1  →  Gaussian: solves  under = norm.cdf(line, ev, ev*cv)
    """
    if ev <= 0 or np.isnan(ev):
        return np.nan

    if old_cv == 1:
        adj_line = np.ceil(float(line) - 1)
        return float(poisson.cdf(adj_line, ev))
    else:
        return float(norm.cdf(float(line), ev, ev * old_cv))


def encode_new_ev_negbin(line, under_prob, r):
    """
    Find the NegBin mean such that:
        nbinom.cdf(line, r, r/(r+mean)) - nbinom.pmf(line, r, r/(r+mean))/2 == under_prob

    This mirrors the NegBin branch of get_odds.
    """
    line = float(line)

    def _negbin_under(mean):
        if mean <= 0:
            return 1.0
        p = r / (r + mean)
        return float(nbinom.cdf(line, r, p) - nbinom.pmf(line, r, p) / 2)

    # Bracket search: mean in (eps, upper)
    lo, hi = 1e-6, max(2 * line / max(1 - under_prob, 0.01), 1.0)
    # Expand upper bracket if needed
    while _negbin_under(hi) > under_prob and hi < 1e6:
        hi *= 2

    if _negbin_under(lo) < under_prob and _negbin_under(hi) > under_prob:
        # Monotonicity goes the wrong way — shouldn't happen, fall back
        return np.nan

    try:
        return float(brentq(lambda m: _negbin_under(m) - under_prob, lo, hi, xtol=1e-8))
    except ValueError:
        return np.nan


def encode_new_ev(line, under_prob, new_cv, new_dist):
    """
    Re-encode an under probability into an EV using the *new* cv / distribution.

    NegBin  → custom solver (matches get_odds NegBin branch)
    Other   → reuse get_ev's Gaussian branch (norm.cdf with std = ev*cv)
    """
    if np.isnan(under_prob) or under_prob <= 0 or under_prob >= 1:
        return np.nan

    if new_dist == "NegBin":
        r = 1.0 / new_cv
        return encode_new_ev_negbin(line, under_prob, r)
    else:
        # SkewNormal / Gaussian — get_ev's Gaussian branch (cv != 1)
        return get_ev(float(line), under_prob, new_cv)


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------
def remap_archive():
    archive = Archive()

    stats = {"converted": 0, "skipped_no_change": 0, "skipped_bad_data": 0, "errors": 0}

    leagues = list(archive.archive.keys())
    for league in tqdm(leagues, desc="Leagues", unit="league"):
        canon = _resolve_league(league)

        old_cvs = old_stat_cv.get(canon, {})
        new_cvs = new_stat_cv.get(canon, {})
        dists = stat_dist.get(canon, {})

        # If this league has no CV/dist data at all, nothing to remap
        if not old_cvs and not new_cvs and not dists:
            continue

        markets = list(archive[league].keys())
        for market in tqdm(markets, desc=f"{league}", unit="market", leave=False):
            if market in SKIP_MARKETS:
                continue

            old_cv = old_cvs.get(market, 1)
            new_cv = new_cvs.get(market, None)
            new_dist = dists.get(market, None)

            # No new config for this market — nothing prescribed, skip
            if new_cv is None and new_dist is None:
                continue

            if new_cv is None:
                new_cv = 1
            if new_dist is None:
                new_dist = "Poisson" if new_cv == 1 else "SkewNormal"

            # Determine whether the encoding actually changes
            old_is_poisson = (old_cv == 1)
            new_is_poisson = (new_dist == "Poisson" and new_cv == 1)
            if old_is_poisson and new_is_poisson:
                stats["skipped_no_change"] += 1
                continue
            if (not old_is_poisson) and (new_dist != "NegBin") and abs(old_cv - new_cv) < 1e-9:
                stats["skipped_no_change"] += 1
                continue

            dates = list(archive[league][market].keys())
            for date in dates:
                players = list(archive[league][market][date].keys())
                for player in players:
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
                            new_ev_dict[book] = old_ev
                            stats["skipped_bad_data"] += 1
                            continue

                        try:
                            under_prob = decode_old_ev(line, float(old_ev), old_cv)

                            if np.isnan(under_prob) or under_prob <= 0 or under_prob >= 1:
                                new_ev_dict[book] = old_ev
                                stats["skipped_bad_data"] += 1
                                continue

                            new_ev = encode_new_ev(line, under_prob, new_cv, new_dist)

                            if new_ev is None or np.isnan(new_ev) or new_ev <= 0:
                                new_ev_dict[book] = old_ev
                                stats["skipped_bad_data"] += 1
                            else:
                                new_ev_dict[book] = float(new_ev)
                                changed = True
                                stats["converted"] += 1

                        except Exception as exc:
                            new_ev_dict[book] = old_ev
                            stats["errors"] += 1

                    if changed:
                        archive[league][market][date][player]["EV"] = new_ev_dict

    print("\n--- Conversion Summary ---")
    print(f"  Converted:        {stats['converted']}")
    print(f"  Skipped (no chg): {stats['skipped_no_change']}")
    print(f"  Skipped (bad):    {stats['skipped_bad_data']}")
    print(f"  Errors:           {stats['errors']}")

    # -----------------------------------------------------------------------
    # Uncomment below to persist changes to disk
    # -----------------------------------------------------------------------
    # archive.write(all=True)
    # print("Archive written to disk.")

    return archive


if __name__ == "__main__":
    remap_archive()
