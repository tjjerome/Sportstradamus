"""Cookie-authenticated client for the paid Fantasy Points Data Suite.

The data suite at ``https://data.fantasypoints.com`` is a single-page
application backed by JSON XHR endpoints. Once a user has logged in via
a browser, the session cookie is enough to call those endpoints
directly. The user discovers each endpoint once via DevTools'
"Copy as cURL" action and registers it with :func:`cli.import_curl`;
the weekly ``fp-fetch run`` cron then walks the catalog and writes
each endpoint's response to ``data/fantasypoints/{season}/week_NN/``.

See ``docs/fantasypoints.md`` for the end-user runbook (capturing a
fresh cookie, adding endpoints, refresh schedule).
"""

from sportstradamus.fantasypoints.catalog import (
    CATALOG_PATH,
    EndpointSpec,
    load_catalog,
    save_catalog,
)
from sportstradamus.fantasypoints.client import (
    FantasyPointsAuthError,
    FantasyPointsClient,
)

__all__ = [
    "CATALOG_PATH",
    "EndpointSpec",
    "FantasyPointsAuthError",
    "FantasyPointsClient",
    "load_catalog",
    "save_catalog",
]
