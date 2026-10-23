"""HTTP client with ScrapeOps-managed browser-header rotation.

Every sportsbook scraper in :mod:`sportstradamus.books` and
:mod:`sportstradamus.moneylines` shares a single :class:`Scrape` instance so
that a blocked header in one call poisons it for the next caller too — the
weights machinery picks the next-freshest header on retry.
"""

import importlib.resources as pkg_resources
import json
import random
from http import HTTPStatus
from io import StringIO
from time import sleep

import numpy as np
import pandas as pd
import requests
from tqdm.contrib.logging import logging_redirect_tqdm

from sportstradamus import creds
from sportstradamus.spiderLogger import logger

REQUEST_TIMEOUT_S = 60  # bound network hangs; generous because proxy-routed fetches legitimately take tens of seconds

# Real-browser User-Agent captured from a working live Firefox session (2026-05).
# stats.nba.com fingerprints on it (refresh from the browser dev-tools Network tab,
# on or near the deploy host, if NBA calls start hanging again), and moneypuck
# answers the default requests UA with an HTML bot-block page.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:150.0) Gecko/20100101 Firefox/150.0"
)

# Statuses that mean "stop asking" rather than "try again": a rotated header is a
# different browser, not a different client, so retrying either is just knocking harder.
_BLOCKED_STATUSES = (HTTPStatus.FORBIDDEN, HTTPStatus.TOO_MANY_REQUESTS)


class BlockedError(Exception):
    """A host answered 403/429. Back off and stop; never route around it via a proxy."""


class Scrape:
    """HTTP GET client that rotates browser headers on retry.

    Headers are fetched lazily from the ScrapeOps ``browser-headers`` endpoint
    the first time one is needed, then kept in memory for the process lifetime.
    Every failed request decays the active header's weight toward zero while
    bumping every other header's weight, so the next retry picks the
    least-recently-burned header.
    """

    def __init__(self):
        """Load the ScrapeOps key. Headers are fetched on first use."""
        with (pkg_resources.files(creds) / "keys.json").open() as f:
            self._scrapeops_key = json.load(f)["scrapeops"]
        self._headers = None
        self._header = None
        self._weights = None

    def _ensure_headers(self):
        """Fetch the header pool from ScrapeOps on first access.

        A ScrapeOps outage leaves this process a pool of one static browser header,
        so a host that bot-blocks the default requests UA (moneypuck) still answers.
        """
        if self._headers is None:
            url = f"https://headers.scrapeops.io/v1/browser-headers?api_key={self._scrapeops_key}"
            try:
                response = requests.get(url, timeout=REQUEST_TIMEOUT_S)
                response.raise_for_status()
                self._headers = response.json()["result"]
            except requests.RequestException as exc:
                # The exception text carries the URL, and with it the API key.
                logger.warning(
                    f"ScrapeOps headers unavailable ({type(exc).__name__}); "
                    "using the static browser header"
                )
                self._headers = [{"User-Agent": BROWSER_USER_AGENT}]
            self._header = random.choice(self._headers)
            self._weights = np.ones([len(self._headers)])

    @property
    def headers(self):
        """The ScrapeOps header pool, fetched on first access."""
        self._ensure_headers()
        return self._headers

    @property
    def header(self):
        """The currently active browser header."""
        self._ensure_headers()
        return self._header

    @header.setter
    def header(self, value):
        self._header = value

    @property
    def weights(self):
        """Per-header rotation weights; a burned header decays toward zero."""
        self._ensure_headers()
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = value

    def _new_headers(self):
        """Rotate to a new header, weighted against the recently-burned one."""
        self._ensure_headers()
        if len(self._headers) == 1:
            return  # the static fallback: zeroing its only weight would make random.choices raise
        for i, candidate in enumerate(self._headers):
            if candidate == self._header:
                self._weights[i] = 0
            else:
                self._weights[i] += 1

        self._header = random.choices(self._headers, weights=self._weights)[0]

    def get(self, url, max_attempts=3, headers=None, params=None):
        """Perform a GET request, rotating headers on retry.

        Args:
            url: The URL to fetch.
            max_attempts: Maximum number of attempts to make the request.
            headers: Additional headers to include in the request.
            params: Query parameters to include in the request.

        Returns:
            The parsed JSON response on success, or ``{}`` after exhausting
            retries. Callers treat the empty-dict sentinel as "no data".
        """
        if params is None:
            params = {}
        if headers is None:
            headers = {}
        with logging_redirect_tqdm():
            for i in range(1, max_attempts + 1):
                if i > 1:
                    self._new_headers()
                    headers.update(self.header)
                    sleep(random.uniform(1, 3))
                try:
                    response = requests.get(
                        url, headers=headers, params=params, timeout=REQUEST_TIMEOUT_S
                    )
                    if response.status_code == HTTPStatus.OK:
                        return response.json()
                    logger.debug(f"Attempt {i}, Error {response.status_code}")
                except Exception:
                    logger.exception(f"Attempt {i},")

            logger.warning("Max Attempts Reached")
            return {}

    def get_csv(self, url, max_attempts=3):
        """GET a CSV endpoint with header rotation, returning a DataFrame.

        Hosts like moneypuck answer the default requests UA with an HTML
        "data license" bot-block page (HTTP 200) but return the real CSV to a
        rotated browser header, so an HTML content-type is treated as a soft
        block and retried. Returns an empty frame once retries are exhausted;
        callers treat the empty frame as "no data".
        """
        with logging_redirect_tqdm():
            for i in range(1, max_attempts + 1):
                if i > 1:
                    self._new_headers()
                    sleep(random.uniform(1, 3))
                try:
                    response = requests.get(url, headers=self.header, timeout=REQUEST_TIMEOUT_S)
                    content_type = response.headers.get("content-type", "")
                    if response.status_code == HTTPStatus.OK and "html" not in content_type.lower():
                        return pd.read_csv(StringIO(response.text))
                    logger.debug(f"Attempt {i}, CSV block status={response.status_code}")
                except Exception:
                    logger.exception(f"Attempt {i},")

            logger.warning(f"Max Attempts Reached (csv): {url}")
            return pd.DataFrame()

    def get_bytes(self, url, max_attempts=2):
        """GET a binary endpoint with header rotation, returning the raw body.

        The browser header rides the first attempt, unlike :meth:`get` — a CDN that
        rejects the default requests UA would never get to a second one. Redirects are
        not followed either: a host that answers an unknown id with a 3xx to a generic
        placeholder image is saying "no such player", not "look over there".

        Raises:
            BlockedError: The host answered 403/429.

        Returns:
            The response body on success, or ``None`` once retries are exhausted.
            Callers treat the ``None`` sentinel as "no data".
        """
        with logging_redirect_tqdm():
            for i in range(1, max_attempts + 1):
                if i > 1:
                    self._new_headers()
                    sleep(random.uniform(1, 3))
                try:
                    response = requests.get(
                        url,
                        headers=self.header,
                        timeout=REQUEST_TIMEOUT_S,
                        allow_redirects=False,
                    )
                    if response.status_code == HTTPStatus.OK:
                        return response.content
                    if response.status_code in _BLOCKED_STATUSES:
                        raise BlockedError(f"{response.status_code} from {url}")
                    logger.debug(f"Attempt {i}, bytes status={response.status_code}")
                except requests.RequestException:
                    logger.exception(f"Attempt {i},")

            logger.warning(f"Max Attempts Reached (bytes): {url}")
            return None
