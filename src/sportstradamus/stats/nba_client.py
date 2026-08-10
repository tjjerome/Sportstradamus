"""Reliable transport wrapper for the stats.nba.com / stats.wnba.com APIs.

The NBA stats backend (reached through ``nba_api``) is fussy in three ways this
module defends against:

1. **Fingerprinting.** It *hangs* rather than refusing requests whose browser
   fingerprint it dislikes, and which fingerprint it accepts varies by client
   IP. nba_api's bundled Chrome headers had grown unreliable from this project's
   host (every request read-timed-out); a live Firefox session returned data
   instantly, so we send that captured real-browser header set on every call.
2. **Per-host rate limits.** NBA and WNBA are served by *separate* hosts
   (``stats.nba.com`` / ``stats.wnba.com``) with independent per-IP budgets.
   nba_api hits ``stats.nba.com`` for both, so an All-league update spends NBA's
   and WNBA's calls against one budget — NBA's burst exhausts it and every
   following WNBA call times out ("WNBA completely failed"). Routing each league
   to its own host via :data:`NBA_HOST` / :data:`WNBA_HOST` gives WNBA its own
   budget.
3. **Burst limiting.** Each host throttles bursts; firing a league's ~15 calls
   back to back trips it (observed: NBA's burst stalled its own playoffs call).
   A module-global clock spaces every request by ``NBA_MIN_REQUEST_INTERVAL``.

On top of that each call gets an explicit read timeout so a hung socket aborts
fast and exponential-backoff retries (the old per-call ``while i < 10: ...
sleep(0.1)`` loops just hammered the server). ``StatsNBA`` and ``StatsWNBA``
then make one flat call per endpoint instead of hand-rolling a retry loop.
"""

import random
from time import monotonic, sleep
from typing import NamedTuple

from nba_api.stats.library.http import NBAStatsHTTP

from sportstradamus.spiderLogger import logger

# Real-browser User-Agent captured from a working live session (Firefox,
# 2026-05). The backend fingerprints on it; refresh from the browser dev-tools
# Network tab (on or near the deploy host) if calls start hanging again.
_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:150.0) Gecko/20100101 Firefox/150.0"

# Header set captured from a live, working stats.nba.com browser session. We
# deliberately omit the x-nba-stats-token/origin headers some guides recommend:
# sending them made the backend hang every request from this project's host.
NBA_STATS_HEADERS: dict[str, str] = {
    "Host": "stats.nba.com",
    "User-Agent": _USER_AGENT,
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

# WNBA is the same backend on its own host; only the host-identifying headers
# differ. Captured from a live stats.wnba.com session.
WNBA_STATS_HEADERS: dict[str, str] = NBA_STATS_HEADERS | {
    "Host": "stats.wnba.com",
    "Referer": "https://stats.wnba.com/",
    "Origin": "https://www.wnba.com",
}


class StatsHost(NamedTuple):
    """A stats backend host: the nba_api base-URL template + matching headers."""

    base_url: str
    headers: dict[str, str]


# One host per league. nba_api defaults every endpoint to stats.nba.com; passing
# a host to fetch() swaps the base URL (and headers) for that call so WNBA hits
# its own rate-limit budget. The "{endpoint}" placeholder is nba_api's own.
NBA_HOST = StatsHost("https://stats.nba.com/stats/{endpoint}", NBA_STATS_HEADERS)
WNBA_HOST = StatsHost("https://stats.wnba.com/stats/{endpoint}", WNBA_STATS_HEADERS)

# Per-request read timeout (seconds). Healthy responses return in under ~2s, so
# this is mostly a bound on a hung socket — short enough to abort and hand
# control to the retry loop, long enough not to fail a slow-but-alive pull.
NBA_STATS_TIMEOUT: int = 20

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

# Minimum seconds between successive requests, enforced module-globally across
# every caller (and both hosts). Without it an update fires its calls in a tight
# burst (observed ~7/s) and the backend stalls the offending request until it
# times out. Tunable: raise if stalls persist, lower for speed.
NBA_MIN_REQUEST_INTERVAL: float = 0.75

# Monotonic timestamp of the last request start. Module-global so the throttle
# spans leagues — StatsNBA and StatsWNBA import this one module.
_last_request_ts: float = 0.0


class NBAStatsError(Exception):
    """Raised when a stats endpoint fails every retry attempt."""


def _throttle() -> None:
    """Block until ``NBA_MIN_REQUEST_INTERVAL`` has elapsed since the last call."""
    global _last_request_ts
    wait = NBA_MIN_REQUEST_INTERVAL - (monotonic() - _last_request_ts)
    if wait > 0:
        sleep(wait)
    _last_request_ts = monotonic()


def fetch(endpoint_cls: type, *, host: StatsHost = NBA_HOST, **params: object) -> object:
    """Construct an ``nba_api`` endpoint against ``host``, throttled and retried.

    The HTTP request happens inside the endpoint constructor, so the constructed
    instance is what the caller wants — call ``.get_normalized_dict()`` or
    ``.get_dict()`` on the return value. Requests are spaced by the global
    throttle; the host's browser headers and a read timeout are injected on
    every attempt; failures are logged and retried with exponential backoff.

    Args:
        endpoint_cls: An ``nba_api.stats.endpoints`` class (e.g.
            ``PlayerGameLogs``). Not an instance — this function instantiates it.
        host: Which backend to route to. Defaults to :data:`NBA_HOST`; pass
            :data:`WNBA_HOST` for WNBA so it uses an independent rate-limit
            budget.
        **params: Endpoint-specific query parameters passed straight through.

    Returns:
        The constructed endpoint instance, with its HTTP request already made.

    Raises:
        NBAStatsError: If every one of ``NBA_MAX_ATTEMPTS`` tries fails.
    """
    name = endpoint_cls.__name__
    for attempt in range(1, NBA_MAX_ATTEMPTS + 1):
        _throttle()
        # nba_api reads the target host from this class attribute; swap it for
        # the call and restore it so an interleaved NBA call is unaffected.
        previous_base_url = NBAStatsHTTP.base_url
        NBAStatsHTTP.base_url = host.base_url
        try:
            return endpoint_cls(headers=host.headers, timeout=NBA_STATS_TIMEOUT, **params)
        except Exception as exc:
            logger.warning(
                "%s %s failed (attempt %d/%d): %s",
                host.base_url.split("/")[2],
                name,
                attempt,
                NBA_MAX_ATTEMPTS,
                exc,
            )
            if attempt < NBA_MAX_ATTEMPTS:
                backoff = min(NBA_BACKOFF_BASE * 2 ** (attempt - 1), NBA_BACKOFF_CAP)
                sleep(backoff + random.uniform(0, NBA_BACKOFF_JITTER))
        finally:
            NBAStatsHTTP.base_url = previous_base_url

    raise NBAStatsError(f"{name} failed after {NBA_MAX_ATTEMPTS} attempts")
