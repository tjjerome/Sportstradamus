"""Hardened transport wrapper for the stats.nba.com API.

``stats.nba.com`` (reached through the ``nba_api`` library) is notoriously flaky
from server and cloud hosts: it rejects or stalls requests that lack the right
browser headers, and it hangs connections rather than refusing them. The library
accepts per-request ``headers``, ``timeout``, and ``proxy`` arguments but applies
none of them by default. This module centralizes the known-good header set, a
real read timeout, and exponential-backoff retry logic so that ``StatsNBA`` and
``StatsWNBA`` can make one flat call per endpoint instead of hand-rolling a
silent ``while`` retry loop around every block.
"""

import random
from time import sleep

from sportstradamus.spiderLogger import logger

# Header set documented by the nba_api project as the one stats.nba.com accepts;
# requests without Referer/Origin/x-nba-stats-* are silently dropped or stalled.
NBA_STATS_HEADERS: dict[str, str] = {
    "Host": "stats.nba.com",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}

# Per-request read timeout (seconds). The endpoints are slow but should never
# take a minute; a real timeout turns a hung connection into a fast retry
# instead of blocking the whole update pipeline indefinitely.
NBA_STATS_TIMEOUT: int = 60

# Total tries per endpoint before giving up and raising NBAStatsError.
NBA_MAX_ATTEMPTS: int = 5

# Exponential backoff between retries: NBA_BACKOFF_BASE * 2**(attempt - 1),
# capped at NBA_BACKOFF_CAP, plus jitter. Replaces the old flat 0.1s sleep that
# hammered a rate-limiting server instead of backing off.
NBA_BACKOFF_BASE: float = 1.0
NBA_BACKOFF_CAP: float = 16.0


class NBAStatsError(Exception):
    """Raised when a stats.nba.com endpoint fails every retry attempt."""


def fetch(endpoint_cls: type, **params: object) -> object:
    """Construct an ``nba_api`` endpoint with hardened transport, retrying.

    The HTTP request happens inside the endpoint constructor, so the constructed
    instance is what the caller wants — call ``.get_normalized_dict()`` or
    ``.get_dict()`` on the return value. Headers and a read timeout are injected
    on every attempt; failures are logged and retried with exponential backoff.

    Args:
        endpoint_cls: An ``nba_api.stats.endpoints`` class (e.g.
            ``PlayerGameLogs``). Not an instance — this function instantiates it.
        **params: Endpoint-specific query parameters passed straight through.

    Returns:
        The constructed endpoint instance, with its HTTP request already made.

    Raises:
        NBAStatsError: If every one of ``NBA_MAX_ATTEMPTS`` tries fails.
    """
    name = endpoint_cls.__name__
    for attempt in range(1, NBA_MAX_ATTEMPTS + 1):
        try:
            return endpoint_cls(
                headers=NBA_STATS_HEADERS,
                timeout=NBA_STATS_TIMEOUT,
                **params,
            )
        except Exception as exc:
            logger.warning(
                "stats.nba.com %s failed (attempt %d/%d): %s",
                name,
                attempt,
                NBA_MAX_ATTEMPTS,
                exc,
            )
            if attempt < NBA_MAX_ATTEMPTS:
                backoff = min(NBA_BACKOFF_BASE * 2 ** (attempt - 1), NBA_BACKOFF_CAP)
                sleep(backoff + random.uniform(0, 1))

    raise NBAStatsError(f"{name} failed after {NBA_MAX_ATTEMPTS} attempts")
