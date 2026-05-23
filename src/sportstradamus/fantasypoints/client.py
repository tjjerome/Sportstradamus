"""Cookie-authenticated HTTP client for data.fantasypoints.com.

Reads the session cookie (raw ``Cookie:`` header value copied from a
logged-in browser DevTools session) from ``creds/keys.json``. The
constructor also honours a ``FANTASYPOINTS_COOKIE`` environment
variable so CI / dev runs can stub credentials without writing to the
shared keys file.

Inter-request sleep keeps the catalog walk well under any natural
browsing rate; retries on 429 and 5xx use exponential backoff;
401/403 raise :class:`FantasyPointsAuthError` so the caller can prompt
the user to refresh the cookie.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import json
import os
from time import sleep
from typing import Literal

import requests

from sportstradamus import creds
from sportstradamus.spiderLogger import logger

# Conservative defaults: a realistic Chrome UA so requests look like a
# logged-in browser, and a 2 s pause between catalog endpoints so a
# weekly walk of ~50 tools stays well under any plausible rate ceiling.
_DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_INTER_REQUEST_SLEEP_S = 2.0

# Exponential backoff schedule for retryable failures. Each entry is
# the sleep before the *next* attempt; len(...) gives the retry count.
_RETRY_BACKOFF_S: tuple[float, ...] = (2.0, 5.0, 15.0)

# HTTP statuses that mean the stored session cookie is dead. No retry —
# the human has to log in again and paste a fresh cookie.
_AUTH_STATUSES = (401, 403)


class FantasyPointsAuthError(RuntimeError):
    """FP returned 401/403; the stored session cookie is expired."""


class FantasyPointsClient:
    """Cookie-authenticated HTTP client for the FP Data Suite API."""

    def __init__(
        self,
        *,
        cookie: str | None = None,
        user_agent: str | None = None,
        inter_request_sleep_s: float | None = None,
    ) -> None:
        """Initialise the client.

        Args:
            cookie: Raw ``Cookie:`` header value. When ``None``, read
                from the ``FANTASYPOINTS_COOKIE`` env var, falling back
                to ``fantasypoints_cookie`` in ``creds/keys.json``.
            user_agent: Override the default Chrome UA string.
            inter_request_sleep_s: Pause between successive ``get`` calls.
                Defaults to :data:`_INTER_REQUEST_SLEEP_S`; tests pass 0.
        """
        if cookie is None:
            cookie = os.environ.get("FANTASYPOINTS_COOKIE")
        if cookie is None or user_agent is None:
            keys = self._load_keys()
            if cookie is None:
                cookie = keys.get("fantasypoints_cookie", "")
            if user_agent is None:
                user_agent = keys.get("fantasypoints_user_agent", "") or None
        self._cookie = cookie or ""
        self._user_agent = user_agent or _DEFAULT_UA
        self._sleep_s = (
            inter_request_sleep_s if inter_request_sleep_s is not None else _INTER_REQUEST_SLEEP_S
        )
        self._first_call = True
        if not self._cookie:
            logger.warning(
                "fantasypoints_cookie missing — see docs/fantasypoints.md for "
                "how to capture a fresh session cookie from your browser."
            )

    @staticmethod
    def _load_keys() -> dict[str, str]:
        path = pkg_resources.files(creds) / "keys.json"
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _default_headers(self) -> dict[str, str]:
        return {
            "Cookie": self._cookie,
            "User-Agent": self._user_agent,
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://data.fantasypoints.com/",
        }

    def get(
        self,
        url: str,
        params: dict | None = None,
        headers: dict | None = None,
        accept: Literal["json", "text", "bytes"] = "json",
    ) -> dict | list | str | bytes:
        """GET ``url`` with auth, inter-request pause, and backoff.

        Args:
            url: Absolute URL.
            params: Query parameters.
            headers: Extra headers merged on top of the defaults.
            accept: Decode body as ``json``, ``text``, or raw ``bytes``.

        Returns:
            Decoded body. JSON returns dict/list; text returns str;
            bytes returns ``bytes``.

        Raises:
            FantasyPointsAuthError: When FP returns 401 or 403.
            requests.HTTPError: When retries are exhausted on another
                non-2xx status.
        """
        if not self._first_call:
            sleep(self._sleep_s)
        self._first_call = False
        merged = self._default_headers()
        if headers:
            merged.update(headers)
        attempts = (*_RETRY_BACKOFF_S, None)
        last_response = None
        for backoff in attempts:
            response = requests.get(url, headers=merged, params=params)
            last_response = response
            if response.status_code in _AUTH_STATUSES:
                raise FantasyPointsAuthError(
                    f"FP returned {response.status_code} for {url} — "
                    "session cookie is expired or missing. Refresh it in "
                    "creds/keys.json; see docs/fantasypoints.md."
                )
            if 200 <= response.status_code < 300:
                return self._decode(response, accept)
            is_retryable = response.status_code == 429 or 500 <= response.status_code < 600
            if not is_retryable or backoff is None:
                response.raise_for_status()
            logger.warning(
                "FP %s returned %s, retrying in %.1fs",
                url,
                response.status_code,
                backoff,
            )
            sleep(backoff)
        # All retries exhausted on a retryable status — surface the last response.
        last_response.raise_for_status()
        # raise_for_status() above always raises on non-2xx; this is just for type checkers.
        raise RuntimeError("unreachable")

    @staticmethod
    def _decode(response: requests.Response, accept: str):
        if accept == "json":
            return response.json()
        if accept == "text":
            return response.text
        return response.content
