"""Persistent odds/EV storage backed by klepto HDF archives.

The :class:`Archive` singleton is the only authorized entrypoint for reading
and writing the per-league archive trees on disk (``archive/<LEAGUE>/``).
Consumers share the single instance so that in-memory mutations stay
consistent across the scrape/predict pipeline.

:func:`clean_archive` is the one-off migration helper used when the archive
format or name-normalization rules change — it walks every league/market/date
in the passed-in archive mapping, drops stale dates, canonicalizes player
names through :func:`remove_accents`, and prunes empty branches.
"""

import datetime
import os

import numpy as np
import pandas as pd
from klepto.archives import cache, hdfdir_archive

from sportstradamus.helpers.config import book_weights, stat_cv, stat_dist, stat_zi
from sportstradamus.helpers.distributions import get_ev, no_vig_odds
from sportstradamus.helpers.text import merge_dict, remove_accents


def clean_archive(a, cutoff_date=None):
    """Rewrite ``a`` in place: drop stale dates, canonicalize player names.

    Runs every non-moneyline player name through ``remove_accents`` and folds
    the stale-spelling entries into the canonical-spelling entries via
    ``merge_dict`` so downstream lookups don't have to try multiple keys.
    Dates older than ``cutoff_date`` (default: 4 years ago) are dropped
    outright.
    """
    if cutoff_date is None:
        cutoff_date = (datetime.datetime.today() - datetime.timedelta(days=365 * 4)).date()
    leagues = list(a.keys())
    for league in leagues:
        markets = list(a[league].keys())

        for market in markets:
            for date in list(a[league][market].keys()):
                if date == "" or datetime.datetime.strptime(date, "%Y-%m-%d").date() < cutoff_date:
                    a[league][market].pop(date)
                    continue

                if market not in ["Moneyline", "Totals", "1st 1 innings"]:
                    players = list(a[league][market][date].keys())
                    for player in players:
                        if player not in a[league][market][date]:
                            continue
                        if " + " in player or " vs. " in player:
                            a[league][market][date].pop(player)
                            continue
                        if "Line" in a[league][market][date][player]["EV"]:
                            a[league][market][date][player]["EV"].pop("Line")

                        player_name = remove_accents(player)
                        if player_name != player:
                            a[league][market][date][player_name] = merge_dict(
                                a[league][market][date].get(player_name, {}),
                                a[league][market][date].pop(player),
                            )

                        a[league][market][date][player_name]["Lines"] = [
                            line for line in a[league][market][date][player_name]["Lines"] if line
                        ]

                        if not len(a[league][market][date][player_name]["EV"]) and not len(
                            a[league][market][date][player_name]["Lines"]
                        ):
                            a[league][market][date].pop(player_name)

                if not len(a[league][market][date]):
                    a[league][market].pop(date)

            if not len(a[league][market]):
                a[league].pop(market)

        if not len(a[league]):
            a.pop(league)

    return a


class Archive:
    """Singleton wrapper around per-league klepto HDF archives.

    On construction, scans ``archive/`` for one subdirectory per league and
    opens each as an ``hdfdir_archive`` lazily — data pages load on first
    access via :meth:`_ensure_loaded`. Writes are batched: every mutation
    records the ``(league, market)`` pair in ``_changed_keys`` so
    :meth:`write` can dump only the markets that actually changed.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Discover per-league archives under ``archive/`` and prep caches."""
        if self._initialized:
            return
        self._initialized = True

        self.archive = {}
        self._loaded = set()
        leagues = [f.name for f in os.scandir("archive") if f.is_dir()]
        for league in leagues:
            self.archive[league] = hdfdir_archive(f"archive/{league}", {}, protocol=-1)

        self.default_totals = {
            "MLB": 4.671,
            "NBA": 111.667,
            "WNBA": 81.667,
            "NFL": 22.668,
            "NHL": 2.674,
        }

        self._changed_keys = {}

    def _mark_changed(self, league, market):
        self._changed_keys.setdefault(league, set()).add(market)

    def _ensure_loaded(self, league):
        if league not in self._loaded and league in self.archive:
            if isinstance(self.archive[league], cache):
                self.archive[league].load()
            self._loaded.add(league)

    def __getitem__(self, item):
        """Return the (lazily-loaded) per-league archive for ``item``."""
        self._ensure_loaded(item)
        return self.archive[item]

    def add_dfs(self, offers, platform, key):
        """Add a batch of scraped offers to the archive for one ``platform``.

        ``offers`` is accepted as a list or single dict; duplicates per
        ``(Player, Market)`` are resolved in favor of the offer closest to
        a neutral 1.0 boost. The ``key`` mapping renames sportsbook-native
        market strings into the canonical per-league market names used
        elsewhere in the pipeline.
        """
        if not isinstance(offers, list):
            offers = [offers]

        df = pd.DataFrame(offers)
        if "Boost_Over" not in df.columns:
            df["Boost_Over"] = np.nan
        if "Boost" in df.columns:
            df.loc[df["Boost_Over"].isna(), "Boost_Over"] = df.loc[df["Boost_Over"].isna(), "Boost"]
        df["Boost Factor"] = np.abs(df["Boost_Over"] - 1)
        df = df.loc[~df.sort_values("Boost Factor").duplicated(["Player", "Market"])]
        offers = df.to_dict(orient="records")
        for o in offers:
            if not o["Line"]:
                continue
            self._ensure_loaded(o["League"])
            market = o["Market"].replace("H2H ", "")
            market = key.get(market, market)
            cv = stat_cv.get(o["League"], {}).get(market, 1)
            dist = stat_dist.get(o["League"], {}).get(market, "Gamma")
            gate = stat_zi.get(o["League"], {}).get(market, 0) if dist in ("ZINB", "ZAGamma") else 0
            if o["League"] == "NHL":
                market_swap = {"AST": "assists", "PTS": "points", "BLK": "blocked"}
                market = market_swap.get(market, market)
            if o["League"] == "NBA" or o["League"] == "WNBA":
                market = market.replace("underdog", "prizepicks")

            self._mark_changed(o["League"], market)
            self.archive.setdefault(o["League"], {}).setdefault(market, {}).setdefault(
                o["Date"], {}
            ).setdefault(o["Player"], {"EV": {}, "Lines": []})

            if (
                float(o["Line"])
                not in self.archive[o["League"]][market][o["Date"]][o["Player"]]["Lines"]
            ):
                self.archive[o["League"]][market][o["Date"]][o["Player"]]["Lines"].append(
                    float(o["Line"])
                )

            over = o.get("Boost_Over", 0) if o.get("Boost_Over", 0) > 0 else o.get("Boost", 1)
            odds = no_vig_odds(over, o.get("Boost_Under"))
            self.archive[o["League"]][market][o["Date"]][o["Player"]]["EV"][platform] = get_ev(
                o["Line"], odds[1], cv, dist=dist, gate=gate or None
            )

    def get_moneyline(self, league, date, team):
        """Weighted-average moneyline EV across books for ``team`` on ``date``."""
        self._ensure_loaded(league)
        a = []
        w = []
        arr = self.archive.get(league, {}).get("Moneyline", {}).get(date, {}).get(team, {})
        if not arr:
            return 0.5
        elif type(arr) is not dict:
            self.archive.get(league, {}).get("Moneyline", {}).get(date, {}).pop(team)
            return 0.5
        for book, ev in arr.items():
            a.append(ev)
            w.append(book_weights.get(league, {}).get("Moneyline", {}).get(book, 1))

        return np.average(a, weights=w)

    def get_total(self, league, date, team):
        """Weighted-average game-total EV for ``team`` on ``date``.

        Falls back to the per-league default total when no book has quoted
        the game — callers rely on always getting a numeric back rather
        than a NaN.
        """
        self._ensure_loaded(league)
        a = []
        w = []
        arr = self.archive.get(league, {}).get("Totals", {}).get(date, {}).get(team, {})
        if not arr:
            return self.default_totals.get(league, 1)
        elif type(arr) is not dict:
            self.archive.get(league, {}).get("Totals", {}).get(date, {}).pop(team)
            return self.default_totals.get(league, 1)
        for book, ev in arr.items():
            a.append(ev)
            w.append(book_weights.get(league, {}).get("Totals", {}).get(book, 1))

        return np.average(a, weights=w)

    def get_ev(self, league, market, date, player):
        """Weighted-average player-prop EV across books for one slate entry."""
        self._ensure_loaded(league)
        a = []
        w = []
        arr = (
            self.archive.get(league, {}).get(market, {}).get(date, {}).get(player, {}).get("EV", {})
        )
        if not arr:
            return np.nan
        for book, ev in arr.items():
            a.append(ev)
            w.append(book_weights.get(league, {}).get(market, {}).get(book, 1))

        return np.average(a, weights=w)

    def get_team_market(self, league, market, date, team):
        """Weighted-average team-market EV (non-player, non-moneyline)."""
        self._ensure_loaded(league)
        a = []
        w = []
        arr = self.archive.get(league, {}).get(market, {}).get(date, {}).get(team, {})
        if not arr:
            return np.nan
        for book, ev in arr.items():
            a.append(ev)
            w.append(book_weights.get(league, {}).get(market, {}).get(book, 1))

        return np.average(a, weights=w)

    def get_line(self, league, market, date, player):
        """Consensus line for ``player`` on ``date``: median, floored to ½."""
        self._ensure_loaded(league)
        arr = (
            self.archive.get(league, {})
            .get(market, {})
            .get(date, {})
            .get(player, {})
            .get("Lines", [np.nan])
        )

        line = np.floor(2 * np.median(arr)) / 2

        return 0 if np.isnan(line) else line

    def to_pandas(self, league, market):
        """Flatten one league/market's archive into a wide DataFrame.

        Indexed by ``(date, player)``, one column per book + a ``Line`` column
        carrying the consensus line computed via :meth:`get_line`. Drops
        pre-2023-05-03 rows for non-totals markets (stale format).
        """
        self._ensure_loaded(league)
        records = {}
        if market not in self.archive[league]:
            return pd.DataFrame()
        for date in list(self.archive[league][market].keys()):
            if (
                market not in ["Moneyline", "Total"]
                and datetime.datetime.strptime(date, "%Y-%m-%d").date()
                < datetime.datetime(2023, 5, 3).date()
            ):
                continue
            for player in list(self.archive[league][market][date].keys()):
                if "EV" in self.archive[league][market][date][player]:
                    line = self.get_line(league, market, date, player)
                    record = self.archive[league][market][date][player]["EV"].copy()
                    record["Line"] = line
                    records[(date, player)] = record
                else:
                    records[(date, player)] = self.archive[league][market][date][player]

        return pd.DataFrame.from_dict(records, orient="index")

    def write(self, all=False):
        """Persist pending changes to the on-disk HDF archives.

        ``all=False`` (default) writes only the markets whose
        ``_changed_keys`` entries were recorded since the last write.
        ``all=True`` dumps every league wholesale — use sparingly, the
        incremental path is much cheaper.
        """
        if all:
            for league in list(self.archive.keys()):
                if type(self.archive[league]) is not cache:
                    self.archive[league] = hdfdir_archive(
                        f"archive/{league}", self.archive[league], protocol=-1
                    )
                self.archive[league].dump()
        else:
            for league, markets in self._changed_keys.items():
                if type(self.archive[league]) is not cache:
                    # New league from plain dict — must dump everything
                    self.archive[league] = hdfdir_archive(
                        f"archive/{league}", self.archive[league], protocol=-1
                    )
                    self.archive[league].dump()
                else:
                    for market in markets:
                        self.archive[league].dump(market)

        self._changed_keys.clear()
