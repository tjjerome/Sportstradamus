"""Cookie-authenticated client for the paid Fantasy Points Data Suite.

The data suite at ``https://data.fantasypoints.com`` is a single-page
application backed by JSON XHR endpoints. Once a user has logged in via
a browser, the session cookie is enough to call those endpoints
directly. The user discovers each endpoint once via DevTools'
"Copy as cURL" action and registers it with :func:`cli.import_curl`;
the weekly ``fp-fetch run`` cron then walks the catalog and writes
each endpoint's response to ``data/player_data/NFL/{season}/`` or
``data/team_data/NFL/{season}/`` depending on context prefix.

See ``docs/fantasypoints.md`` for the end-user runbook (capturing a
fresh cookie, adding endpoints, refresh schedule).
"""

from sportstradamus.fantasypoints.body_substitute import (
    DEFAULT_MODE,
    Mode,
    substitute_runtime,
)
from sportstradamus.fantasypoints.catalog import (
    CATALOG_PATH,
    EndpointSpec,
    load_catalog,
    save_catalog,
)
from sportstradamus.fantasypoints.client import (
    FantasyPointsAuthError,
    FantasyPointsClient,
    FantasyPointsDecodeError,
)
from sportstradamus.fantasypoints.discover import (
    REGISTRY_BODY,
    REGISTRY_URL,
    expand_registry,
)
from sportstradamus.fantasypoints.transform import (
    PLAYER_DATA_BASE,
    TEAM_DATA_BASE,
    parquet_path_for_spec,
    parse_table_response,
    write_parquet,
)
from sportstradamus.fantasypoints.verify import (
    VerificationIssue,
    verify_catalog,
    verify_spec,
)

__all__ = [
    "CATALOG_PATH",
    "DEFAULT_MODE",
    "PLAYER_DATA_BASE",
    "REGISTRY_BODY",
    "REGISTRY_URL",
    "TEAM_DATA_BASE",
    "EndpointSpec",
    "FantasyPointsAuthError",
    "FantasyPointsClient",
    "FantasyPointsDecodeError",
    "Mode",
    "VerificationIssue",
    "expand_registry",
    "load_catalog",
    "parquet_path_for_spec",
    "parse_table_response",
    "save_catalog",
    "substitute_runtime",
    "verify_catalog",
    "verify_spec",
    "write_parquet",
]
