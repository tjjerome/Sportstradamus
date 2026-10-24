"""Fantasy Points Data Suite collector.

The data suite at ``https://fantasypointsdata.com`` is a single-page app
backed by JSON XHR endpoints. Once a user has logged in via a browser, the
captured ``ds_session`` cookie is enough to call those endpoints directly.
Each endpoint is registered once — from a DevTools "Copy as cURL" via
``fp-fetch import-curl`` — and the weekly ``fp-fetch run`` cron then walks
the catalog and writes each response to ``data/player_data/NFL/{season}/``
or ``data/team_data/NFL/{season}/`` depending on context prefix.

This package supplies only the FP-specific pieces (auth wiring, the
season/week query window, the new-to-legacy column translation, NFL
parquet routing) on top of the source-neutral framework in
:mod:`sportstradamus.collectors`.

See ``docs/fantasypoints.md`` for the end-user runbook and
``docs/fantasypoints_expansion.md`` for what the API offers beyond the
current feature set.
"""

from sportstradamus.collectors.catalog import (
    EndpointSpec,
    load_catalog,
    save_catalog,
)
from sportstradamus.collectors.fantasypoints.cli import fp_fetch
from sportstradamus.collectors.fantasypoints.import_curl import parse_curl_to_spec
from sportstradamus.collectors.fantasypoints.source import (
    CATALOG_PATH,
    DEFAULT_MODE,
    FP_SOURCE,
    Mode,
    period_params,
)
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
    "TEAM_DATA_BASE",
    "EndpointSpec",
    "Mode",
    "VerificationIssue",
    "fp_fetch",
    "load_catalog",
    "parquet_path_for_spec",
    "parse_curl_to_spec",
    "parse_table_response",
    "period_params",
    "save_catalog",
    "verify_catalog",
    "verify_spec",
    "write_parquet",
]
