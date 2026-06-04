"""Bearer-token-authenticated HTTP client for data.fantasypoints.com.

Reads the ``Authorization:`` header value (and optionally a
``Cookie:`` header) from ``creds/keys.json``. Both can be captured
from any logged-in XHR in DevTools and pasted verbatim. The
constructor also honours ``FANTASYPOINTS_AUTHORIZATION`` /
``FANTASYPOINTS_COOKIE`` env vars so CI / dev runs can stub
credentials without writing to the shared keys file.

Both GET and POST are supported. POST is used by the FP Data Suite
v2 endpoints, which carry their per-tool query as a JSON request
body. Inter-request sleep keeps the catalog walk well under any
natural browsing rate; retries on 429 and 5xx use exponential
backoff; 401/403 raise :class:`FantasyPointsAuthError` so the caller
can prompt the user to refresh the token.
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

# Conservative defaults: a realistic Firefox UA so requests look like a
# logged-in browser, and a 2 s pause between catalog endpoints so a
# weekly walk of ~50 tools stays well under any plausible rate ceiling.
_DEFAULT_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:150.0) Gecko/20100101 Firefox/150.0"
_INTER_REQUEST_SLEEP_S = 2.0

# Exponential backoff schedule for retryable failures. Each entry is
# the sleep before the *next* attempt; len(...) gives the retry count.
_RETRY_BACKOFF_S: tuple[float, ...] = (2.0, 5.0, 15.0)

# HTTP statuses that mean the stored token is dead. No retry — the
# human has to log in again and paste a fresh Authorization value.
_AUTH_STATUSES = (401, 403)

Accept = Literal["json", "text", "bytes"]

# Body preview length in decode-failure diagnostics. Long enough to
# identify HTML error pages ("<!DOCTYPE html>...Cloudflare...") or
# compressed binary blobs (leading magic bytes); short enough not to
# spam logs with a 200 KB FP payload that happened to start mid-token.
_DECODE_PREVIEW_CHARS = 300


class FantasyPointsAuthError(RuntimeError):
    """FP returned 401/403; the stored Authorization token is expired."""


class FantasyPointsDecodeError(RuntimeError):
    """FP returned 2xx but the body couldn't be decoded as JSON.

    Most common cause: the catalog entry kept ``Accept-Encoding: br, zstd``
    from the original DevTools curl, so FP responded with brotli/zstd —
    which ``requests`` can't transparently decompress without the
    optional ``brotli`` / ``zstandard`` extras. Re-running
    ``fp-fetch import-curl`` against a fresh curl regenerates the
    catalog entry without the offending header.
    """


class FantasyPointsClient:
    """Bearer-auth HTTP client for the FP Data Suite v2 API."""

    def __init__(
        self,
        *,
        authorization: str | None = None,
        cookie: str | None = None,
        user_agent: str | None = None,
        inter_request_sleep_s: float | None = None,
    ) -> None:
        """Initialise the client.

        Args:
            authorization: Raw ``Authorization:`` header value (e.g.
                ``"Bearer eyJ..."``). When ``None``, read from the
                ``FANTASYPOINTS_AUTHORIZATION`` env var, falling back
                to ``fantasypoints_authorization`` in ``creds/keys.json``.
            cookie: Optional ``Cookie:`` header value. Some endpoints
                still want analytics/session cookies alongside the
                Authorization header.
            user_agent: Override the default Firefox UA string.
            inter_request_sleep_s: Pause between successive requests.
                Defaults to :data:`_INTER_REQUEST_SLEEP_S`; tests pass 0.
        """
        authorization, cookie, user_agent = self._resolve_credentials(
            authorization=authorization,
            cookie=cookie,
            user_agent=user_agent,
        )
        self._authorization = authorization or ""
        self._cookie = cookie or ""
        self._user_agent = user_agent or _DEFAULT_UA
        self._sleep_s = (
            inter_request_sleep_s if inter_request_sleep_s is not None else _INTER_REQUEST_SLEEP_S
        )
        self._first_call = True
        if not self._authorization:
            logger.warning(
                "fantasypoints_authorization missing — see docs/fantasypoints.md "
                "for how to capture the Authorization header from your browser."
            )

    @classmethod
    def _resolve_credentials(
        cls,
        *,
        authorization: str | None,
        cookie: str | None,
        user_agent: str | None,
    ) -> tuple[str | None, str | None, str | None]:
        if authorization is None:
            authorization = os.environ.get("FANTASYPOINTS_AUTHORIZATION")
        if cookie is None:
            cookie = os.environ.get("FANTASYPOINTS_COOKIE")
        if authorization is None or cookie is None or user_agent is None:
            keys = cls._load_keys()
            if authorization is None:
                authorization = keys.get("fantasypoints_authorization", "")
            if cookie is None:
                cookie = keys.get("fantasypoints_cookie", "")
            if user_agent is None:
                user_agent = keys.get("fantasypoints_user_agent", "") or None
        return authorization, cookie, user_agent

    @staticmethod
    def _load_keys() -> dict[str, str]:
        path = pkg_resources.files(creds) / "keys.json"
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _default_headers(self) -> dict[str, str]:
        h = {
            "User-Agent": self._user_agent,
            "Accept": "*/*",
            "Referer": "https://data.fantasypoints.com/",
            "Origin": "https://data.fantasypoints.com",
        }
        if self._authorization:
            h["Authorization"] = self._authorization
        if self._cookie:
            h["Cookie"] = self._cookie
        return h

    def get(
        self,
        url: str,
        params: dict | None = None,
        headers: dict | None = None,
        accept: Accept = "json",
    ) -> dict | list | str | bytes:
        """GET ``url`` with auth, inter-request pause, and backoff."""
        return self._request("GET", url, params=params, headers=headers, accept=accept)

    def refresh_credentials(
        self,
        *,
        authorization: str | None = None,
        cookie: str | None = None,
        user_agent: str | None = None,
    ) -> None:
        """Swap in fresh auth values in-place; subsequent calls use them.

        Used by the CLI's interactive auth-refresh path: when a long-
        running ``fp-fetch run`` hits a 401 mid-batch, the user pastes
        a fresh DevTools curl, the CLI parses out the new headers and
        calls this — no need to recreate the client or restart the run.
        """
        if authorization is not None:
            self._authorization = authorization
        if cookie is not None:
            self._cookie = cookie
        if user_agent is not None:
            self._user_agent = user_agent
        # Reset the inter-request sleep guard so the first post-refresh
        # call doesn't wait unnecessarily — the user already paused
        # while pasting the curl.
        self._first_call = True

    def post(
        self,
        url: str,
        json_body: dict | list | None = None,
        params: dict | None = None,
        headers: dict | None = None,
        accept: Accept = "json",
    ) -> dict | list | str | bytes:
        """POST ``url`` with a JSON body."""
        return self._request(
            "POST",
            url,
            params=params,
            json_body=json_body,
            headers=headers,
            accept=accept,
        )

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict | None = None,
        json_body: dict | list | None = None,
        headers: dict | None = None,
        accept: Accept = "json",
    ) -> dict | list | str | bytes:
        """Common loop: inter-request pause, header merge, backoff, decode.

        Args:
            method: HTTP verb (``"GET"`` or ``"POST"``).
            url: Absolute URL.
            params: Query parameters.
            json_body: Optional JSON body (POST only — passed via
                ``requests``' ``json=`` kwarg so Content-Type and
                serialization are handled automatically).
            headers: Extra headers merged on top of the defaults.
            accept: Decode body as ``json``, ``text``, or raw ``bytes``.

        Returns:
            Decoded body per ``accept``.

        Raises:
            FantasyPointsAuthError: When FP returns 401 or 403.
            requests.HTTPError: When retries are exhausted on another
                non-2xx status.
        """
        if not self._authorization:
            raise FantasyPointsAuthError(
                "fantasypoints_authorization is empty. Paste the Authorization "
                "header value from a logged-in DevTools session into "
                "creds/keys.json (or set the FANTASYPOINTS_AUTHORIZATION env "
                "var). See docs/fantasypoints.md."
            )
        if not self._first_call:
            sleep(self._sleep_s)
        self._first_call = False
        merged = self._build_request_headers(headers, json_body)
        return self._send_with_retries(method, url, merged, params, json_body, accept)

    def _build_request_headers(self, headers: dict | None, json_body) -> dict:
        merged = self._default_headers()
        if headers:
            merged.update(headers)
        if json_body is not None and "Content-Type" not in merged:
            merged["Content-Type"] = "application/json"
        # Defense-in-depth: even if an old catalog entry still carries
        # browser-style Accept-Encoding (gzip, deflate, br, zstd), strip
        # it before the call so the server replies with gzip/identity
        # that ``requests`` can decompress on its own. Without this the
        # body comes back as undecoded brotli/zstd and JSON parsing
        # blows up with "Expecting value: line 1 column 1".
        return {k: v for k, v in merged.items() if k.lower() != "accept-encoding"}

    def _send_with_retries(self, method, url, merged, params, json_body, accept):
        """Issue the request, retrying retryable statuses on the backoff schedule.

        Returns the decoded 2xx body; raises FantasyPointsAuthError on 401/403 and
        requests.HTTPError once retries are exhausted (or on a non-retryable status).
        """
        attempts = (*_RETRY_BACKOFF_S, None)
        last_response = None
        for backoff in attempts:
            response = requests.request(
                method,
                url,
                headers=merged,
                params=params,
                json=json_body,
            )
            last_response = response
            if response.status_code in _AUTH_STATUSES:
                raise FantasyPointsAuthError(
                    f"FP returned {response.status_code} for {method} {url} — "
                    "Authorization token is expired or missing. Refresh "
                    "fantasypoints_authorization in creds/keys.json; see "
                    "docs/fantasypoints.md."
                )
            if 200 <= response.status_code < 300:
                return self._decode(response, accept)
            is_retryable = response.status_code == 429 or 500 <= response.status_code < 600
            if not is_retryable or backoff is None:
                response.raise_for_status()
            logger.warning(
                "FP %s %s returned %s, retrying in %.1fs",
                method,
                url,
                response.status_code,
                backoff,
            )
            sleep(backoff)
        last_response.raise_for_status()
        # raise_for_status() above always raises on non-2xx; this is just for type checkers.
        raise RuntimeError("unreachable")

    @staticmethod
    def _decode(response: requests.Response, accept: Accept) -> dict | list | str | bytes:
        if accept == "json":
            try:
                return response.json()
            except ValueError as e:
                raise FantasyPointsDecodeError(_format_decode_failure(response, e)) from e
        if accept == "text":
            return response.text
        return response.content


def _format_decode_failure(response: requests.Response, error: Exception) -> str:
    r"""Render a single-string diagnostic for a non-JSON 2xx body.

    Includes status, the two response headers most likely to explain a
    decode failure (``Content-Type`` and ``Content-Encoding``), the raw
    body length, and the first ``_DECODE_PREVIEW_CHARS`` of the body
    (escaped to repr so binary garbage shows as ``\xNN`` rather than
    breaking the log line).
    """
    content_type = response.headers.get("Content-Type", "<missing>")
    content_encoding = response.headers.get("Content-Encoding", "<missing>")
    body = response.content or b""
    preview = body[:_DECODE_PREVIEW_CHARS]
    try:
        preview_str = preview.decode("utf-8")
    except UnicodeDecodeError:
        preview_str = repr(preview)
    return (
        f"FP returned {response.status_code} {response.request.method} "
        f"{response.url} but body is not JSON ({error}). "
        f"Content-Type={content_type!r} Content-Encoding={content_encoding!r} "
        f"body_len={len(body)} preview={preview_str!r}. "
        "If Content-Encoding is br/zstd, the catalog entry has a stale "
        "Accept-Encoding header — re-run `fp-fetch import-curl` with a "
        "fresh DevTools curl. See docs/fantasypoints.md."
    )
