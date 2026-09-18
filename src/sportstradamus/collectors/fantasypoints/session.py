"""Mint a Fantasy Points ``ds_session`` cookie from stored credentials.

The rebuilt API authenticates on that cookie alone — it ignores an
``Authorization`` header — and the cookie carries a hard seven-day expiry
with no sliding renewal on use. Left to a human that means a DevTools paste
every week, and a silent ``fp-fetch`` failure whenever the paste is late.

So the collector logs in for itself: ``POST /api/auth/login`` with the
account name and password takes the same values the sign-in form does and
answers with a fresh ``Set-Cookie: ds_session=...``, which is written back
to ``creds/keys.json`` for the next run to reuse. The weekly cron then
renews on its own the first time a 401 comes back.

Credentials live in ``creds/keys.json`` under ``fantasypoints_username`` /
``fantasypoints_password`` (or ``FANTASYPOINTS_USERNAME`` /
``FANTASYPOINTS_PASSWORD`` in the environment). Without them the collector
falls back to whatever cookie is already stored.
"""

from __future__ import annotations

import os

import requests

from sportstradamus.collectors.auth import load_keys, update_keys
from sportstradamus.helpers.scraping import REQUEST_TIMEOUT_S
from sportstradamus.spiderLogger import logger

LOGIN_URL = "https://fantasypointsdata.com/api/auth/login"
SESSION_COOKIE = "ds_session"

# keys.json fields. The cookie shares the slot the human-pasted one used, so
# a renewed session is indistinguishable downstream from a pasted one.
_USERNAME_FIELD = "fantasypoints_username"
_PASSWORD_FIELD = "fantasypoints_password"
_COOKIE_FIELD = "fantasypoints_cookie"


class SessionRenewalError(RuntimeError):
    """Login could not produce a fresh session cookie."""


def renew_session() -> str:
    """Log in, persist the fresh ``ds_session`` to keys.json, and return it.

    Returns:
        The cookie header value, ``ds_session=<token>``.

    Raises:
        SessionRenewalError: Credentials are absent, the login was rejected,
            or the response carried no session cookie.
    """
    keys = load_keys()
    username = os.environ.get("FANTASYPOINTS_USERNAME") or keys.get(_USERNAME_FIELD, "")
    password = os.environ.get("FANTASYPOINTS_PASSWORD") or keys.get(_PASSWORD_FIELD, "")
    if not username or not password:
        raise SessionRenewalError(
            f"No stored credentials. Add {_USERNAME_FIELD!r} and {_PASSWORD_FIELD!r} to "
            "creds/keys.json so fp-fetch can renew its own session."
        )
    response = requests.post(
        LOGIN_URL,
        json={"name": username, "password": password},
        headers={"Accept": "application/json", "Origin": "https://fantasypointsdata.com"},
        timeout=REQUEST_TIMEOUT_S,
    )
    if not response.ok:
        raise SessionRenewalError(
            f"Login rejected ({response.status_code}): {_error_message(response)}"
        )
    token = response.cookies.get(SESSION_COOKIE)
    if not token:
        raise SessionRenewalError(
            f"Login succeeded but set no {SESSION_COOKIE!r} cookie. The login contract "
            "likely changed — re-capture it from the sign-in form."
        )
    cookie = f"{SESSION_COOKIE}={token}"
    update_keys({_COOKIE_FIELD: cookie})
    logger.info("fantasypoints session renewed", extra={"cookie_field": _COOKIE_FIELD})
    return cookie


def _error_message(response: requests.Response) -> str:
    """Pull the API's own ``error`` string out of a failed login response."""
    try:
        return str(response.json().get("error", response.reason))
    except ValueError:
        return response.reason
