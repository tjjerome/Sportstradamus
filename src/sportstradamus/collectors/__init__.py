"""Shared framework for cookie/bearer-authenticated data collectors.

A *collector* snapshots the JSON XHR endpoints behind a logged-in
single-page data app to disk. This package holds the source-neutral
machinery — the endpoint catalog, the HTTP transport, credential I/O, the
fetch/parse/write runner, and a CLI builder — that a concrete source (e.g.
:mod:`sportstradamus.collectors.fantasypoints`) wires together via a
:class:`Source`.
"""

from sportstradamus.collectors.auth import (
    AuthFields,
    ResolvedAuth,
    load_keys,
    read_auth,
    refresh_auth_from_curl,
    update_keys,
)
from sportstradamus.collectors.catalog import (
    EndpointSpec,
    load_catalog,
    save_catalog,
)
from sportstradamus.collectors.cli import Source, build_source_cli
from sportstradamus.collectors.transport import (
    CollectorAuthError,
    CollectorDecodeError,
    CookieClient,
)

__all__ = [
    "AuthFields",
    "CollectorAuthError",
    "CollectorDecodeError",
    "CookieClient",
    "EndpointSpec",
    "ResolvedAuth",
    "Source",
    "build_source_cli",
    "load_catalog",
    "load_keys",
    "read_auth",
    "refresh_auth_from_curl",
    "save_catalog",
    "update_keys",
]
