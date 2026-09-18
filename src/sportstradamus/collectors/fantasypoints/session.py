"""Mint a Fantasy Points ``ds_session`` cookie from stored credentials.

The rebuilt API authenticates on that cookie alone — it ignores an
``Authorization`` header — and the cookie carries a hard seven-day expiry
with no sliding renewal on use. Left to a human that means a DevTools paste
every week, and a silent ``fp-fetch`` failure whenever the paste is late.

So the collector signs in for itself, making the same two hops the sign-in
form makes: Firebase trades the account email and password for an
``idToken``, then ``POST /api/auth/firebase-login`` trades that for a fresh
``Set-Cookie: ds_session=...``, which is written back to ``creds/keys.json``
for the next run to reuse. The weekly cron then renews on its own the first
time a 401 comes back.

Firebase is the only way in. The sign-in page ships a second form posting an
account name and password to ``/api/auth/login``, but its
``isFirebaseAuthEnabled()`` gate is a literal ``true``, so that form is
unreachable and the route answers 404 for a Firebase-backed account.

Credentials live in ``creds/keys.json`` under ``fantasypoints_email`` /
``fantasypoints_password`` (or ``FANTASYPOINTS_EMAIL`` /
``FANTASYPOINTS_PASSWORD`` in the environment). Without them the collector
falls back to whatever cookie is already stored.
"""

from __future__ import annotations

import os

import requests

from sportstradamus.collectors.auth import load_keys, update_keys
from sportstradamus.helpers.scraping import REQUEST_TIMEOUT_S
from sportstradamus.spiderLogger import logger

# Fantasy Points' own Firebase web config, read out of the sign-in bundle. A
# Firebase web API key names the project rather than authorizing anything —
# it ships inlined in every browser client — so it is not a credential and
# does not belong in keys.json beside the password.
FIREBASE_API_KEY = "AIzaSyAyKSFD2rYt_-_kCYoSM0Xpadeq3dhtaW4"
FIREBASE_SIGNIN_URL = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"
LOGIN_URL = "https://fantasypointsdata.com/api/auth/firebase-login"
SESSION_COOKIE = "ds_session"

# keys.json fields. The cookie shares the slot the human-pasted one used, so
# a renewed session is indistinguishable downstream from a pasted one.
_EMAIL_FIELD = "fantasypoints_email"
_PASSWORD_FIELD = "fantasypoints_password"
_COOKIE_FIELD = "fantasypoints_cookie"


class SessionRenewalError(RuntimeError):
    """Sign-in could not produce a fresh session cookie."""


def renew_session() -> str:
    """Sign in, persist the fresh ``ds_session`` to keys.json, and return it.

    Returns:
        The cookie header value, ``ds_session=<token>``.

    Raises:
        SessionRenewalError: Credentials are absent, either hop was
            rejected, or the response carried no session cookie.
    """
    keys = load_keys()
    email = os.environ.get("FANTASYPOINTS_EMAIL") or keys.get(_EMAIL_FIELD, "")
    password = os.environ.get("FANTASYPOINTS_PASSWORD") or keys.get(_PASSWORD_FIELD, "")
    if not email or not password:
        raise SessionRenewalError(
            f"No stored credentials. Add {_EMAIL_FIELD!r} and {_PASSWORD_FIELD!r} to "
            "creds/keys.json so fp-fetch can renew its own session."
        )
    firebase = requests.post(
        f"{FIREBASE_SIGNIN_URL}?key={FIREBASE_API_KEY}",
        json={"email": email, "password": password, "returnSecureToken": True},
        headers={"Accept": "application/json"},
        timeout=REQUEST_TIMEOUT_S,
    )
    if not firebase.ok:
        raise SessionRenewalError(
            f"Firebase rejected the sign-in ({firebase.status_code}): "
            f"{_error_message(firebase)}. That is the fantasypoints.com account "
            f"email and password, stored under {_EMAIL_FIELD!r}."
        )
    response = requests.post(
        LOGIN_URL,
        json={"idToken": firebase.json()["idToken"]},
        headers={"Accept": "application/json", "Origin": "https://fantasypointsdata.com"},
        timeout=REQUEST_TIMEOUT_S,
    )
    if not response.ok:
        raise SessionRenewalError(
            f"Data Suite rejected the sign-in ({response.status_code}): {_error_message(response)}"
        )
    token = response.cookies.get(SESSION_COOKIE)
    if not token:
        raise SessionRenewalError(
            f"Sign-in succeeded but set no {SESSION_COOKIE!r} cookie. The login "
            "contract likely changed — re-capture it from the sign-in form."
        )
    cookie = f"{SESSION_COOKIE}={token}"
    update_keys({_COOKIE_FIELD: cookie})
    logger.info("fantasypoints session renewed", extra={"cookie_field": _COOKIE_FIELD})
    return cookie


def _error_message(response: requests.Response) -> str:
    """Pull an API's own error text out of a failed sign-in response.

    Both hops report under ``error``, but differently: Firebase nests a
    machine code (``{"error": {"message": "EMAIL_NOT_FOUND"}}``) while the
    Data Suite returns a flat sentence.
    """
    try:
        error = response.json().get("error", response.reason)
    except ValueError:
        return response.reason
    if isinstance(error, dict):
        return str(error.get("message", response.reason))
    return str(error)
