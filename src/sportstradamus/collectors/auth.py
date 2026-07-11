"""Credential resolution and keys.json I/O for collector sources.

Each source names three ``creds/keys.json`` fields — one per auth slot
(authorization / cookie / user-agent) — via :class:`AuthFields`. This
module resolves those slots (env override, then keys.json),
atomically merges updates back into keys.json, and parses the
``Authorization`` / ``Cookie`` / ``User-Agent`` headers out of a
DevTools-copied curl so token rotation is a one-paste operation.

The curl-token parser is shared: both the generic ``refresh-auth``
flow and a source's ``import-curl`` command tokenize the same way.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import json
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

from sportstradamus import creds

# Headers that should never be persisted in a catalog entry —
# Authorization and Cookie come from creds/keys.json, host/connection/
# length are per-request, UA is set by the client, accept-encoding is
# set by requests itself (we can only decompress gzip + deflate without
# extra packages; letting the catalog ask for br/zstd produces undecoded
# binary bodies that fail json parsing).
_STRIPPED_CURL_HEADERS = frozenset(
    {
        "authorization",
        "cookie",
        "user-agent",
        "authority",
        ":authority",
        "host",
        "connection",
        "content-length",
        "alt-used",
        "te",
        "accept-encoding",
    }
)

# Curl option tokens that take a value but whose value we never need.
_CURL_SKIP_VALUE_FLAGS = frozenset({"-b", "--cookie", "-A", "--user-agent", "-e", "--referer"})

# Backslash-newline continuation in shell here-doc style curls.
_LINE_CONTINUATION_RE = re.compile(r"\\\s*\r?\n")

# Number of leading characters to echo when previewing a token value —
# enough to confirm it changed without leaking the full secret.
_TOKEN_PREVIEW_CHARS = 24


@dataclass(frozen=True)
class AuthFields:
    """keys.json field names for a source's three auth slots.

    A ``None`` slot means the source doesn't use it.
    """

    authorization: str | None = None
    cookie: str | None = None
    user_agent: str | None = None


@dataclass(frozen=True)
class ResolvedAuth:
    """Resolved auth values for one source, ready to hand to a client."""

    authorization: str | None
    cookie: str | None
    user_agent: str | None


def load_keys() -> dict:
    """Read ``creds/keys.json`` into a dict; empty dict if the file is absent."""
    path = pkg_resources.files(creds) / "keys.json"
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def update_keys(updates: dict[str, str], path: Path | None = None) -> Path:
    """Merge ``updates`` into ``creds/keys.json`` atomically.

    Preserves every other field; creates the file if absent.

    Args:
        updates: Dict of keys.json fields to overwrite.
        path: Override target path (default: bundled creds/keys.json).

    Returns:
        The path that was written.
    """
    target = path or Path(str(pkg_resources.files(creds) / "keys.json"))
    existing: dict[str, str] = {}
    if target.is_file():
        with target.open() as f:
            existing = json.load(f)
    existing.update(updates)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(existing, f, indent=2)
        f.write("\n")
    tmp.replace(target)
    return target


def read_auth(fields: AuthFields, *, env_prefix: str | None) -> ResolvedAuth:
    """Resolve a source's three auth slots: env override, then keys.json.

    For each slot with a non-``None`` field name, prefer the explicit env
    var (``{ENV_PREFIX}_AUTHORIZATION`` / ``_COOKIE`` / ``_USER_AGENT`` when
    ``env_prefix`` is given), then ``keys.json[field_name]``. A ``None``
    field name resolves to ``None`` for that slot.
    """
    keys = load_keys()
    return ResolvedAuth(
        authorization=_resolve_slot(fields.authorization, "AUTHORIZATION", env_prefix, keys),
        cookie=_resolve_slot(fields.cookie, "COOKIE", env_prefix, keys),
        user_agent=_resolve_slot(fields.user_agent, "USER_AGENT", env_prefix, keys) or None,
    )


def _resolve_slot(
    field_name: str | None, env_suffix: str, env_prefix: str | None, keys: dict
) -> str | None:
    if field_name is None:
        return None
    if env_prefix:
        from_env = os.environ.get(f"{env_prefix}_{env_suffix}")
        if from_env is not None:
            return from_env
    return keys.get(field_name, "")


def refresh_auth_from_curl(
    curl_text: str, fields: AuthFields, *, keys_path: Path | None = None
) -> tuple[Path, dict[str, str]]:
    """Extract auth headers from a curl and write them into keys.json.

    Returns ``(path, updates)`` where ``updates`` is keyed by keys.json
    field name for whichever of the source's slots the curl carried. When
    no matching headers are found, ``updates`` is empty and keys.json is
    still (idempotently) rewritten with no changes.

    Raises:
        ValueError: If the input is not a curl command.
    """
    updates = extract_auth_updates(curl_text, fields)
    path = update_keys(updates, path=keys_path)
    return path, updates


def extract_auth_updates(curl_text: str, fields: AuthFields) -> dict[str, str]:
    """Pull auth-related header values out of a curl, keyed by keys.json field.

    The header→field map is built from ``fields`` (authorization header →
    ``fields.authorization``, cookie → ``fields.cookie``, user-agent →
    ``fields.user_agent``), skipping ``None`` slots.

    Raises:
        ValueError: If the input is not a curl command.
    """
    header_to_field = {
        "authorization": fields.authorization,
        "cookie": fields.cookie,
        "user-agent": fields.user_agent,
    }
    parsed = parse_curl_tokens(tokenize_curl(curl_text))
    out: dict[str, str] = {}
    for key, value in parsed["headers"].items():
        target = header_to_field.get(key.lower())
        if target and value:
            out[target] = value
    return out


def tokenize_curl(curl_text: str) -> list[str]:
    """Split a curl command into shell tokens (minus the leading ``curl``).

    Raises:
        ValueError: If the input does not start with ``curl``.
    """
    cleaned = _LINE_CONTINUATION_RE.sub(" ", curl_text.strip())
    tokens = shlex.split(cleaned)
    if not tokens or tokens[0] != "curl":
        raise ValueError("Input does not look like a curl command.")
    return tokens[1:]


def parse_curl_tokens(tokens: list[str]) -> dict:
    state: dict = {"url": None, "method": None, "headers": {}, "body": None}
    i = 0
    while i < len(tokens):
        i = _apply_curl_token(tokens, i, state)
    return state


def _apply_curl_token(tokens: list[str], i: int, state: dict) -> int:
    """Apply one curl token to ``state``; return the next index to read."""
    tok = tokens[i]
    if tok in ("-H", "--header"):
        i += 1
        if i < len(tokens):
            key, _, value = tokens[i].partition(":")
            state["headers"][key.strip()] = value.strip()
    elif tok in ("-X", "--request"):
        i += 1
        if i < len(tokens):
            state["method"] = tokens[i]
    elif tok in ("-d", "--data", "--data-raw", "--data-binary", "--data-ascii"):
        i += 1
        if i < len(tokens):
            state["body"] = tokens[i]
    elif tok in _CURL_SKIP_VALUE_FLAGS:
        i += 1  # Value irrelevant — UA/cookie come from keys.json.
    elif tok.startswith("-"):
        pass  # Flag without value (--compressed, --location, ...).
    elif state["url"] is None:
        state["url"] = tok
    return i + 1
