"""Reliable transport wrapper for the stats.nba.com API.

stats.nba.com (reached through ``nba_api``) intermittently *hangs* rather than
refusing requests whose browser fingerprint it dislikes — and which fingerprint
it accepts varies by client IP. nba_api's bundled Chrome headers had grown
unreliable from this project's host (every request read-timed-out), while a
live Firefox session against the same endpoint returned data instantly. So this
module sends a real-browser header set captured from a working session, pins an
explicit read timeout so a hung socket aborts fast, and retries each call with
exponential backoff (the old per-call ``while i < 10: ... sleep(0.1)`` loops
just hammered a rate-limiting server). ``StatsNBA`` and ``StatsWNBA`` then make
one flat call per endpoint instead of hand-rolling a silent retry loop.
"""

import random
from time import sleep

from sportstradamus.spiderLogger import logger

# Header set captured from a live, working stats.nba.com browser session
# (Firefox, 2026-05). stats.nba.com fingerprints on the User-Agent + Sec-Fetch-*
# hints and silently hangs requests it dislikes; the bundled nba_api Chrome
# headers had started read-timing-out from this project's host, so we override
# every endpoint with this proven set. Refresh it from the browser dev-tools
# Network tab if calls start hanging again.
NBA_STATS_HEADERS: dict[str, str] = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:150.0) Gecko/20100101 Firefox/150.0",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    # The live browser advertised "gzip, deflate, br, zstd"; trimmed to the
    # codecs the requests/urllib3 stack decodes without extra deps. The server
    # honors the narrower set and returns gzip — a br/zstd body would arrive
    # undecodable and break JSON parsing.
    "Accept-Encoding": "gzip, deflate",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Connection": "keep-alive",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
}

# Per-request read timeout (seconds). Matches nba_api's own default — long
# enough for a slow-but-alive full-season pull, short enough that a genuinely
# hung socket aborts and hands control back to the retry loop instead of
# blocking the update pipeline. Set explicitly so the bound is a named knob.
NBA_STATS_TIMEOUT: int = 30

# Total tries per endpoint before giving up and raising NBAStatsError.
NBA_MAX_ATTEMPTS: int = 5

# Exponential backoff between retries: NBA_BACKOFF_BASE * 2**(attempt - 1),
# capped at NBA_BACKOFF_CAP, plus uniform jitter in [0, NBA_BACKOFF_JITTER].
# Replaces the old flat 0.1s sleep that hammered a rate-limiting server instead
# of backing off.
NBA_BACKOFF_BASE: float = 1.0
NBA_BACKOFF_CAP: float = 16.0
# Max jitter added per retry (seconds). Spreads simultaneous callers so they do
# not all hammer the endpoint at the same backoff boundary.
NBA_BACKOFF_JITTER: float = 1.0


class NBAStatsError(Exception):
    """Raised when a stats.nba.com endpoint fails every retry attempt."""


def fetch(endpoint_cls: type, **params: object) -> object:
    """Construct an ``nba_api`` endpoint with proven headers + timeout, retrying.

    The HTTP request happens inside the endpoint constructor, so the constructed
    instance is what the caller wants — call ``.get_normalized_dict()`` or
    ``.get_dict()`` on the return value. The known-good browser headers and a
    read timeout are injected on every attempt; failures are logged and retried
    with exponential backoff.

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
                sleep(backoff + random.uniform(0, NBA_BACKOFF_JITTER))

    raise NBAStatsError(f"{name} failed after {NBA_MAX_ATTEMPTS} attempts")
