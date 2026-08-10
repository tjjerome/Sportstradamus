"""Fantasy Points Data Suite collector.

The data suite at ``https://data.fantasypoints.com`` is a single-page app
backed by JSON XHR endpoints. Once a user has logged in via a browser, the
captured ``Authorization`` header (and session cookie) is enough to call
those endpoints directly. The user discovers each endpoint once via
DevTools' "Copy as cURL" action and registers it with ``fp-fetch
import-curl``; the weekly ``fp-fetch run`` cron then walks the catalog and
writes each response to ``data/player_data/NFL/{season}/`` or
``data/team_data/NFL/{season}/`` depending on context prefix.

This package supplies only the FP-specific pieces (auth wiring, v2 body
substitution, NFL parquet routing) on top of the source-neutral framework
in :mod:`sportstradamus.collectors`.

See ``docs/fantasypoints.md`` for the end-user runbook.
"""

from sportstradamus.collectors.catalog import (
    EndpointSpec,
    load_catalog,
    save_catalog,
)
from sportstradamus.collectors.fantasypoints.body_substitute import (
    DEFAULT_MODE,
    Mode,
    substitute_runtime,
)
from sportstradamus.collectors.fantasypoints.cli import fp_fetch
from sportstradamus.collectors.fantasypoints.discover import (
    REGISTRY_BODY,
    REGISTRY_URL,
    expand_registry,
)
from sportstradamus.collectors.fantasypoints.import_curl import parse_curl_to_spec
from sportstradamus.collectors.fantasypoints.source import CATALOG_PATH, FP_SOURCE
from sportstradamus.collectors.fantasypoints.transform import (
    PLAYER_DATA_BASE,
    TEAM_DATA_BASE,
    parquet_path_for_spec,
    parse_table_response,
    write_parquet,
)
from sportstradamus.collectors.fantasypoints.verify import (
    VerificationIssue,
    verify_catalog,
    verify_spec,
)

__all__ = [
    "CATALOG_PATH",
    "DEFAULT_MODE",
    "FP_SOURCE",
    "PLAYER_DATA_BASE",
    "REGISTRY_BODY",
    "REGISTRY_URL",
    "TEAM_DATA_BASE",
    "EndpointSpec",
    "Mode",
    "VerificationIssue",
    "expand_registry",
    "fp_fetch",
    "load_catalog",
    "parquet_path_for_spec",
    "parse_curl_to_spec",
    "parse_table_response",
    "save_catalog",
    "substitute_runtime",
    "verify_catalog",
    "verify_spec",
    "write_parquet",
]
