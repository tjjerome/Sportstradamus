"""HTTP client with ScrapeOps-managed browser-header rotation.

Every sportsbook scraper in :mod:`sportstradamus.books` and
:mod:`sportstradamus.moneylines` shares a single :class:`Scrape` instance so
that a blocked header in one call poisons it for the next caller too — the
weights machinery picks the next-freshest header on retry.
"""

import importlib.resources as pkg_resources
import json
import random
from time import sleep

import numpy as np
import requests
from tqdm.contrib.logging import logging_redirect_tqdm

from sportstradamus import creds
from sportstradamus.spiderLogger import logger


class Scrape:
    """HTTP GET client that rotates browser headers on retry.

    Headers are fetched lazily from the ScrapeOps ``browser-headers`` endpoint
    the first time one is needed, then kept in memory for the process lifetime.
    Every failed request decays the active header's weight toward zero while
    bumping every other header's weight, so the next retry picks the
    least-recently-burned header.
    """

    def __init__(self):
        """Load API keys. Headers are fetched on first use."""
        with open(pkg_resources.files(creds) / "keys.json") as f:
            _keys = json.load(f)
        self.apikey = _keys["scrapingfish"]
        self._scrapeops_key = _keys["scrapeops"]
        self._headers = None
        self._header = None
        self._weights = None

    def _ensure_headers(self):
        """Fetch the header pool from ScrapeOps on first access."""
        if self._headers is None:
            self._headers = requests.get(
                f"http://headers.scrapeops.io/v1/browser-headers?api_key={self._scrapeops_key}"
            ).json()["result"]
            self._header = random.choice(self._headers)
            self._weights = np.ones([len(self._headers)])

    @property
    def headers(self):
        self._ensure_headers()
        return self._headers

    @property
    def header(self):
        self._ensure_headers()
        return self._header

    @header.setter
    def header(self, value):
        self._header = value

    @property
    def weights(self):
        self._ensure_headers()
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = value

    def _new_headers(self):
        """Rotate to a new header, weighted against the recently-burned one."""
        self._ensure_headers()
        for i in range(len(self._headers)):
            if self._headers[i] == self._header:
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
                    response = requests.get(url, headers=headers, params=params)
                    if response.status_code == 200:
                        return response.json()
                    else:
                        logger.debug("Attempt " + str(i) + ", Error " + str(response.status_code))
                except Exception:
                    logger.exception("Attempt " + str(i) + ",")

            logger.warning("Max Attempts Reached")
            return {}
