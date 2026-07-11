"""Unit tests for the fantasypoints package — client, catalog, CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import requests
from click.testing import CliRunner

from sportstradamus.collectors.catalog import (
    EndpointSpec,
    load_catalog,
    save_catalog,
)
from sportstradamus.collectors.fantasypoints.cli import fp_fetch
from sportstradamus.collectors.fantasypoints.import_curl import parse_curl_to_spec
from sportstradamus.collectors.fantasypoints.source import FP_SOURCE
from sportstradamus.collectors.transport import (
    CollectorAuthError,
    CollectorDecodeError,
    CookieClient,
)
from sportstradamus.collectors.fantasypoints.discover import (
    _camel_to_kebab,
    expand_registry,
)
from sportstradamus.collectors.fantasypoints.transform import (
    parquet_path_for_spec,
    parse_table_response,
    write_parquet,
)


class FakeResponse:
    """Minimal stand-in for :class:`requests.Response`."""

    def __init__(
        self,
        status_code,
        body=None,
        text="",
        *,
        content=None,
        headers=None,
        url="https://example/",
        method="GET",
        json_error=None,
    ):
        self.status_code = status_code
        self._body = body
        self.text = text
        if content is not None:
            self.content = content
        else:
            self.content = text.encode() if text else b""
        self.headers = headers or {}
        self.url = url
        self.request = type("FakeRequest", (), {"method": method})()
        self._json_error = json_error

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.HTTPError(f"{self.status_code}")
            err.response = self
            raise err


def _client() -> CookieClient:
    return CookieClient(
        authorization="Bearer test-token",
        cookie="_shopify_y=abc",
        user_agent="UA",
        referer="https://data.fantasypoints.com/",
        origin="https://data.fantasypoints.com",
        inter_request_sleep_s=0.0,
    )


def _patch_request(monkeypatch, handler):
    from sportstradamus.collectors import transport as client_mod

    monkeypatch.setattr(client_mod.requests, "request", handler)


def test_client_raises_on_401(monkeypatch):
    _patch_request(monkeypatch, lambda *a, **k: FakeResponse(401))
    with pytest.raises(CollectorAuthError, match="Authorization token"):
        _client().get("https://example/")


def test_client_raises_on_403(monkeypatch):
    _patch_request(monkeypatch, lambda *a, **k: FakeResponse(403))
    with pytest.raises(CollectorAuthError):
        _client().get("https://example/")


def test_client_retries_on_429_then_succeeds(monkeypatch):
    from sportstradamus.collectors import transport as client_mod

    responses = [FakeResponse(429), FakeResponse(200, body={"ok": True})]
    _patch_request(monkeypatch, lambda *a, **k: responses.pop(0))
    monkeypatch.setattr(client_mod, "_RETRY_BACKOFF_S", (0.0, 0.0, 0.0))
    assert _client().get("https://example/") == {"ok": True}


def test_client_retries_on_500_then_gives_up(monkeypatch):
    from sportstradamus.collectors import transport as client_mod

    _patch_request(monkeypatch, lambda *a, **k: FakeResponse(500))
    monkeypatch.setattr(client_mod, "_RETRY_BACKOFF_S", (0.0, 0.0, 0.0))
    with pytest.raises(requests.HTTPError):
        _client().get("https://example/")


def test_client_returns_text_when_accept_text(monkeypatch):
    _patch_request(
        monkeypatch,
        lambda *a, **k: FakeResponse(200, text="name,value\nfoo,1\n"),
    )
    body = _client().get("https://example/", accept="text")
    assert body.startswith("name,value")


def test_client_env_var_overrides_keys(monkeypatch):
    monkeypatch.setenv("FANTASYPOINTS_AUTHORIZATION", "Bearer from-env")
    monkeypatch.setenv("FANTASYPOINTS_COOKIE", "from_env=1")
    captured = {}

    def capture(method, url, headers=None, params=None, json=None):
        captured["headers"] = headers
        return FakeResponse(200, body={})

    _patch_request(monkeypatch, capture)
    FP_SOURCE.client(inter_request_sleep_s=0.0).get("https://example/")
    assert captured["headers"]["Authorization"] == "Bearer from-env"
    assert captured["headers"]["Cookie"] == "from_env=1"


def test_client_post_sends_json_body(monkeypatch):
    captured = {}

    def capture(method, url, headers=None, params=None, json=None):
        captured["method"] = method
        captured["json"] = json
        return FakeResponse(200, body={"ok": True})

    _patch_request(monkeypatch, capture)
    body = {"context": {"grouping": "team"}, "useCache": True}
    out = _client().post("https://example/", json_body=body)
    assert out == {"ok": True}
    assert captured["method"] == "POST"
    assert captured["json"] == body


def test_client_strips_accept_encoding_from_request_headers(monkeypatch):
    captured = {}

    def capture(method, url, headers=None, params=None, json=None):
        captured["headers"] = headers
        return FakeResponse(200, body={"ok": True})

    _patch_request(monkeypatch, capture)
    _client().get(
        "https://example/",
        headers={"Accept-Encoding": "gzip, deflate, br, zstd"},
    )
    keys = {k.lower() for k in captured["headers"]}
    assert "accept-encoding" not in keys, (
        "Accept-Encoding must be stripped before the request reaches FP; "
        f"got headers={captured['headers']}"
    )


def test_client_raises_decode_error_with_diagnostic_on_non_json_body(monkeypatch):
    def respond_with_garbage(method, url, headers=None, params=None, json=None):
        return FakeResponse(
            200,
            content=b"\x83\xaa\xbf",
            headers={"Content-Type": "application/json", "Content-Encoding": "zstd"},
            url="https://example/v2/ds/nfl/tools/team/line-matchups",
            method="POST",
            json_error=ValueError("Expecting value: line 1 column 1 (char 0)"),
        )

    _patch_request(monkeypatch, respond_with_garbage)
    with pytest.raises(CollectorDecodeError) as exc_info:
        _client().post("https://example/", json_body={"x": 1})
    msg = str(exc_info.value)
    assert "Content-Encoding" in msg and "zstd" in msg
    assert "body_len=" in msg
    assert "re-import" in msg.lower()


def test_client_fails_fast_when_authorization_empty(monkeypatch):
    monkeypatch.delenv("FANTASYPOINTS_AUTHORIZATION", raising=False)
    monkeypatch.delenv("FANTASYPOINTS_COOKIE", raising=False)
    called = {"n": 0}

    def fail_if_called(*a, **k):
        called["n"] += 1
        return FakeResponse(200, body={})

    _patch_request(monkeypatch, fail_if_called)
    client = CookieClient(
        authorization="",
        cookie="",
        user_agent="UA",
        referer="https://data.fantasypoints.com/",
        origin="https://data.fantasypoints.com",
        inter_request_sleep_s=0.0,
    )
    with pytest.raises(CollectorAuthError, match="empty"):
        client.get("https://example/")
    assert called["n"] == 0, "no HTTP request should have been attempted"


def test_parse_curl_get_strips_auth_headers_and_splits_query():
    curl_text = (
        "curl 'https://data.fantasypoints.com/api/nfl/team/line-matchups?week=5&season=2025' "
        "-H 'Authorization: Bearer abc' "
        "-H 'Cookie: _shopify_y=def' "
        "-H 'User-Agent: Mozilla/5.0' "
        "-H 'Accept: application/json' "
        "--compressed"
    )
    spec = parse_curl_to_spec(
        curl_text,
        name="line_matchups",
        output_subdir="team/line_matchups",
    )
    assert spec.method == "GET"
    assert spec.url == "https://data.fantasypoints.com/api/nfl/team/line-matchups"
    assert spec.params == {"week": "5", "season": "2025"}
    assert spec.extra_headers == {"Accept": "application/json"}
    assert spec.json_body is None


def test_parse_curl_post_extracts_method_and_json_body():
    curl_text = (
        "curl 'https://data.fantasypoints.com/v2/ds/nfl/tools/team/line-matchups' "
        "--compressed "
        "-X POST "
        "-H 'Content-Type: application/json' "
        "-H 'Authorization: Bearer abc' "
        "-H 'Cookie: _shopify_y=def' "
        '--data-raw \'{"context":{"grouping":"$team.teamId"},"useCache":true}\''
    )
    spec = parse_curl_to_spec(
        curl_text,
        name="line_matchups",
        output_subdir="team/line_matchups",
    )
    assert spec.method == "POST"
    assert spec.url == "https://data.fantasypoints.com/v2/ds/nfl/tools/team/line-matchups"
    assert spec.params is None
    assert spec.json_body == {
        "context": {"grouping": "$team.teamId"},
        "useCache": True,
    }
    assert spec.extra_headers == {"Content-Type": "application/json"}


def test_parse_curl_handles_multiline_continuations():
    curl_text = (
        "curl 'https://example/' \\\n"
        "  -X POST \\\n"
        "  -H 'Accept: application/json' \\\n"
        "  --data-raw '{\"k\":1}'\n"
    )
    spec = parse_curl_to_spec(curl_text, name="x", output_subdir="x")
    assert spec.method == "POST"
    assert spec.json_body == {"k": 1}
    assert spec.extra_headers == {"Accept": "application/json"}


def test_parse_curl_handles_no_query_string():
    spec = parse_curl_to_spec(
        "curl 'https://data.fantasypoints.com/api/nfl/season-summary' "
        "-H 'Accept: application/json'",
        name="season_summary",
        output_subdir="season_summary",
        weekly=False,
    )
    assert spec.url == "https://data.fantasypoints.com/api/nfl/season-summary"
    assert spec.params is None
    assert spec.weekly is False


def test_parse_curl_rejects_non_curl_input():
    with pytest.raises(ValueError, match="curl command"):
        parse_curl_to_spec("wget https://example/", name="x", output_subdir="x")


def test_parse_curl_consumes_and_ignores_skip_value_flags():
    # -A / -b take a value that must be consumed (UA/cookie come from
    # keys.json), so the following value is not mistaken for the URL.
    spec = parse_curl_to_spec(
        "curl 'https://example/api/x' -A 'Mozilla/5.0' -b 'session=abc' "
        "-H 'Accept: application/json'",
        name="x",
        output_subdir="x",
    )
    assert spec.url == "https://example/api/x"
    assert spec.extra_headers == {"Accept": "application/json"}


def test_parse_curl_invalid_json_body_yields_none():
    spec = parse_curl_to_spec(
        "curl 'https://example/api/x' -X POST --data-raw 'not-json'",
        name="x",
        output_subdir="x",
    )
    assert spec.json_body is None
    assert spec.method == "POST"


def test_parse_curl_defaults_to_post_when_body_present_without_method():
    spec = parse_curl_to_spec(
        "curl 'https://example/api/x' --data-raw '{\"k\":1}'",
        name="x",
        output_subdir="x",
    )
    assert spec.method == "POST"
    assert spec.json_body == {"k": 1}


def test_catalog_round_trip(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    specs = [
        EndpointSpec(
            name="line_matchups",
            url="https://data.fantasypoints.com/v2/ds/nfl/tools/team/line-matchups",
            method="POST",
            json_body={"context": {"grouping": "$team.teamId"}, "useCache": True},
            output_subdir="team/line_matchups",
        ),
    ]
    save_catalog(specs, catalog_path)
    loaded = load_catalog(catalog_path)
    assert len(loaded) == 1
    assert loaded[0].method == "POST"
    assert loaded[0].json_body == {"context": {"grouping": "$team.teamId"}, "useCache": True}


def test_endpoint_spec_renders_template_in_body(tmp_path):
    spec = EndpointSpec(
        name="x",
        url="https://example/",
        method="POST",
        json_body={
            "filters": {"week": "{week}", "season": "{season}"},
            "static": "$team.teamId",
            "nested": [{"week": "{week}"}],
            "useCache": True,
        },
        output_subdir="x",
    )
    rendered = spec.render_json_body(season=2025, week=5)
    assert rendered == {
        "filters": {"week": "5", "season": "2025"},
        "static": "$team.teamId",
        "nested": [{"week": "5"}],
        "useCache": True,
    }


def test_endpoint_spec_renders_params_template(tmp_path):
    spec = EndpointSpec(
        name="line_matchups",
        url="https://example/",
        params={"week": "{week}", "season": "{season}", "static": "abc"},
        output_subdir="team/line_matchups",
        response_format="json",
    )
    assert spec.render_params(season=2025, week=5) == {
        "week": "5",
        "season": "2025",
        "static": "abc",
    }


def _redirect_parquet_dirs(monkeypatch, tmp_path):
    """Point PLAYER_DATA_BASE / TEAM_DATA_BASE at tmp_path so tests don't write to the package."""
    from sportstradamus.collectors.fantasypoints import transform as transform_mod

    monkeypatch.setattr(transform_mod, "PLAYER_DATA_BASE", tmp_path / "player_data")
    monkeypatch.setattr(transform_mod, "TEAM_DATA_BASE", tmp_path / "team_data")


def test_cli_run_dry_run_prints_method_and_url(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    save_catalog(
        [
            EndpointSpec(
                name="team_line_matchups",
                url="https://data.fantasypoints.com/v2/ds/nfl/tools/team/line-matchups",
                method="POST",
                json_body={"useCache": True},
                output_subdir="team/line_matchups",
            ),
        ],
        catalog_path,
    )
    runner = CliRunner()
    result = runner.invoke(
        fp_fetch,
        [
            "run",
            "--season",
            "2025",
            "--week",
            "5",
            "--dry-run",
            "--catalog",
            str(catalog_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "POST" in result.output
    assert "team_line_matchups" in result.output


def test_cli_run_writes_parquet_via_post(monkeypatch, tmp_path):
    from sportstradamus.collectors import transport as client_mod

    _redirect_parquet_dirs(monkeypatch, tmp_path)
    catalog_path = tmp_path / "catalog.json"
    save_catalog(
        [
            EndpointSpec(
                name="team_line_matchups",
                url="https://data.fantasypoints.com/v2/ds/nfl/tools/team/line-matchups",
                method="POST",
                json_body={"context": {"week": "{week}"}, "useCache": True},
                output_subdir="team/line_matchups",
            ),
        ],
        catalog_path,
    )
    monkeypatch.setenv("FANTASYPOINTS_AUTHORIZATION", "Bearer test")
    monkeypatch.setattr(client_mod, "_INTER_REQUEST_SLEEP_S", 0.0)
    captured = {}

    def capture(method, url, headers=None, params=None, json=None):
        captured["method"] = method
        captured["json"] = json
        return FakeResponse(
            200,
            body={
                "content": {
                    "table": {
                        "rows": {
                            "count": 2,
                            "values": [
                                {"teamAbbreviation": "BAL", "gameWeek": 5},
                                {"teamAbbreviation": "DEN", "gameWeek": 5},
                            ],
                        }
                    }
                }
            },
        )

    monkeypatch.setattr(client_mod.requests, "request", capture)
    runner = CliRunner()
    result = runner.invoke(
        fp_fetch,
        [
            "run",
            "--season",
            "2025",
            "--week",
            "5",
            "--catalog",
            str(catalog_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["method"] == "POST"
    assert captured["json"] == {"context": {"week": "5"}, "useCache": True}
    parquet = tmp_path / "team_data" / "NFL" / "2025" / "week_05" / "line_matchups.parquet"
    assert parquet.is_file(), f"expected parquet at {parquet}"
    df = pd.read_parquet(parquet)
    assert len(df) == 2
    assert list(df.columns) == ["teamAbbreviation", "gameWeek"]
    assert set(df["teamAbbreviation"]) == {"BAL", "DEN"}


def test_cli_run_skips_when_nonempty_parquet_already_on_disk(monkeypatch, tmp_path):
    """Default behavior: don't re-fetch a cell that already has rows.

    Lets a half-finished backfill resume cheaply — and a regular `run`
    re-execute after a partial failure without paying for the calls
    that succeeded the first time.
    """
    from sportstradamus.collectors import transport as client_mod

    _redirect_parquet_dirs(monkeypatch, tmp_path)
    catalog_path = tmp_path / "catalog.json"
    save_catalog(
        [
            EndpointSpec(
                name="team_line_matchups",
                url="https://data.fantasypoints.com/v2/ds/nfl/tools/team/line-matchups",
                method="POST",
                json_body={"context": {"week": "{week}"}, "useCache": True},
                output_subdir="team/line_matchups",
            ),
        ],
        catalog_path,
    )
    monkeypatch.setenv("FANTASYPOINTS_AUTHORIZATION", "Bearer test")
    monkeypatch.setattr(client_mod, "_INTER_REQUEST_SLEEP_S", 0.0)
    # Plant a non-empty parquet at the cell's expected path.
    target = tmp_path / "team_data" / "NFL" / "2025" / "week_05" / "line_matchups.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"teamAbbreviation": ["BAL", "DEN"]}).to_parquet(target, index=False)
    call_count = {"n": 0}

    def fail_if_called(*_args, **_kwargs):
        call_count["n"] += 1
        raise AssertionError("HTTP must not be called when a non-empty parquet exists")

    monkeypatch.setattr(client_mod.requests, "request", fail_if_called)
    runner = CliRunner()
    result = runner.invoke(
        fp_fetch,
        ["run", "--season", "2025", "--week", "5", "--catalog", str(catalog_path)],
    )
    assert result.exit_code == 0, result.output
    assert call_count["n"] == 0
    assert "skip" in result.output.lower()
    # Parquet untouched — same two rows we wrote.
    df = pd.read_parquet(target)
    assert set(df["teamAbbreviation"]) == {"BAL", "DEN"}


def test_cli_run_refetch_flag_overrides_skip(monkeypatch, tmp_path):
    """``--refetch`` ignores the on-disk parquet and re-fetches anyway."""
    from sportstradamus.collectors import transport as client_mod

    _redirect_parquet_dirs(monkeypatch, tmp_path)
    catalog_path = tmp_path / "catalog.json"
    save_catalog(
        [
            EndpointSpec(
                name="team_line_matchups",
                url="https://data.fantasypoints.com/v2/ds/nfl/tools/team/line-matchups",
                method="POST",
                json_body={"context": {"week": "{week}"}, "useCache": True},
                output_subdir="team/line_matchups",
            ),
        ],
        catalog_path,
    )
    monkeypatch.setenv("FANTASYPOINTS_AUTHORIZATION", "Bearer test")
    monkeypatch.setattr(client_mod, "_INTER_REQUEST_SLEEP_S", 0.0)
    target = tmp_path / "team_data" / "NFL" / "2025" / "week_05" / "line_matchups.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"teamAbbreviation": ["OLD"]}).to_parquet(target, index=False)

    def respond(*_args, **_kwargs):
        return FakeResponse(
            200,
            body={"content": {"rows": {"values": [{"teamAbbreviation": "NEW", "gameWeek": 5}]}}},
        )

    monkeypatch.setattr(client_mod.requests, "request", respond)
    runner = CliRunner()
    result = runner.invoke(
        fp_fetch,
        [
            "run",
            "--season",
            "2025",
            "--week",
            "5",
            "--refetch",
            "--catalog",
            str(catalog_path),
        ],
    )
    assert result.exit_code == 0, result.output
    df = pd.read_parquet(target)
    assert list(df["teamAbbreviation"]) == ["NEW"]


def test_cli_run_zero_row_parquet_does_not_skip(monkeypatch, tmp_path):
    """A 0-row parquet (previous failed fetch) is treated as not-yet-fetched."""
    from sportstradamus.collectors import transport as client_mod

    _redirect_parquet_dirs(monkeypatch, tmp_path)
    catalog_path = tmp_path / "catalog.json"
    save_catalog(
        [
            EndpointSpec(
                name="team_line_matchups",
                url="https://data.fantasypoints.com/v2/ds/nfl/tools/team/line-matchups",
                method="POST",
                json_body={"context": {}},
                output_subdir="team/line_matchups",
            ),
        ],
        catalog_path,
    )
    monkeypatch.setenv("FANTASYPOINTS_AUTHORIZATION", "Bearer test")
    monkeypatch.setattr(client_mod, "_INTER_REQUEST_SLEEP_S", 0.0)
    target = tmp_path / "team_data" / "NFL" / "2025" / "week_05" / "line_matchups.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"teamAbbreviation": pd.Series(dtype="object")}).to_parquet(target, index=False)
    assert pd.read_parquet(target).empty

    def respond(*_args, **_kwargs):
        return FakeResponse(
            200,
            body={"content": {"rows": {"values": [{"teamAbbreviation": "BAL"}]}}},
        )

    monkeypatch.setattr(client_mod.requests, "request", respond)
    runner = CliRunner()
    result = runner.invoke(
        fp_fetch,
        ["run", "--season", "2025", "--week", "5", "--catalog", str(catalog_path)],
    )
    assert result.exit_code == 0, result.output
    df = pd.read_parquet(target)
    assert list(df["teamAbbreviation"]) == ["BAL"]


def test_cli_run_auth_error_exits_nonzero(monkeypatch, tmp_path):
    from sportstradamus.collectors import transport as client_mod

    _redirect_parquet_dirs(monkeypatch, tmp_path)
    catalog_path = tmp_path / "catalog.json"
    save_catalog(
        [
            EndpointSpec(
                name="team_line_matchups",
                url="https://example/",
                output_subdir="team/line_matchups",
            ),
        ],
        catalog_path,
    )
    monkeypatch.setenv("FANTASYPOINTS_AUTHORIZATION", "Bearer expired")
    monkeypatch.setattr(client_mod, "_INTER_REQUEST_SLEEP_S", 0.0)
    monkeypatch.setattr(client_mod.requests, "request", lambda *a, **k: FakeResponse(401))
    # In a non-TTY context (CliRunner default), the auth-refresh prompt
    # is bypassed and the error propagates as today.
    runner = CliRunner()
    result = runner.invoke(
        fp_fetch,
        [
            "run",
            "--season",
            "2025",
            "--week",
            "5",
            "--catalog",
            str(catalog_path),
        ],
    )
    assert result.exit_code != 0
    assert "authorization" in result.output.lower() or "expired" in result.output.lower()


def test_cli_list_empty_catalog(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text("[]")
    runner = CliRunner()
    result = runner.invoke(fp_fetch, ["list", "--catalog", str(catalog_path)])
    assert result.exit_code == 0
    assert "empty" in result.output.lower()


def test_cli_import_curl_appends_post_spec(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text("[]")
    curl_path = tmp_path / "snippet.curl"
    curl_path.write_text(
        "curl 'https://data.fantasypoints.com/v2/ds/nfl/tools/team/line-matchups' "
        "-X POST "
        "-H 'Authorization: Bearer abc' "
        "-H 'Content-Type: application/json' "
        "--data-raw '{\"useCache\":true}'"
    )
    runner = CliRunner()
    result = runner.invoke(
        fp_fetch,
        [
            "import-curl",
            str(curl_path),
            "--name",
            "line_matchups",
            "--output-subdir",
            "team/line_matchups",
            "--catalog",
            str(catalog_path),
        ],
    )
    assert result.exit_code == 0, result.output
    loaded = load_catalog(catalog_path)
    assert len(loaded) == 1
    assert loaded[0].method == "POST"
    assert loaded[0].json_body == {"useCache": True}


def test_cli_refresh_auth_updates_keys_json(tmp_path):
    keys_path = tmp_path / "keys.json"
    keys_path.write_text(json.dumps({"odds_api": "PRESERVE_ME", "fantasypoints_cookie": "stale"}))
    curl_path = tmp_path / "fresh.curl"
    curl_path.write_text(
        "curl 'https://data.fantasypoints.com/v2/ds/nfl/tools/team/line-matchups' "
        "-X POST "
        "-H 'Authorization: Bearer NEW_TOKEN_VALUE' "
        "-H 'Cookie: _shopify_y=new-cookie-value' "
        "-H 'User-Agent: Mozilla/5.0 (X11; Linux x86_64) Test' "
        "--data-raw '{}'"
    )
    runner = CliRunner()
    result = runner.invoke(
        fp_fetch,
        ["refresh-auth", str(curl_path), "--keys-path", str(keys_path)],
    )
    assert result.exit_code == 0, result.output
    updated = json.loads(keys_path.read_text())
    assert updated["fantasypoints_authorization"] == "Bearer NEW_TOKEN_VALUE"
    assert updated["fantasypoints_cookie"] == "_shopify_y=new-cookie-value"
    assert updated["fantasypoints_user_agent"].startswith("Mozilla/5.0 (X11")
    assert updated["odds_api"] == "PRESERVE_ME", "non-FP keys must be preserved"


def test_cli_refresh_auth_creates_keys_json_when_absent(tmp_path):
    keys_path = tmp_path / "keys.json"
    curl_path = tmp_path / "fresh.curl"
    curl_path.write_text("curl 'https://x/' -H 'Authorization: Bearer ABC' -H 'Cookie: c=1'")
    runner = CliRunner()
    result = runner.invoke(
        fp_fetch,
        ["refresh-auth", str(curl_path), "--keys-path", str(keys_path)],
    )
    assert result.exit_code == 0, result.output
    assert keys_path.is_file()
    assert json.loads(keys_path.read_text())["fantasypoints_authorization"] == "Bearer ABC"


def test_cli_refresh_auth_errors_when_no_auth_headers(tmp_path):
    keys_path = tmp_path / "keys.json"
    keys_path.write_text("{}")
    curl_path = tmp_path / "no_auth.curl"
    curl_path.write_text("curl 'https://x/' -H 'Accept: application/json'")
    runner = CliRunner()
    result = runner.invoke(
        fp_fetch,
        ["refresh-auth", str(curl_path), "--keys-path", str(keys_path)],
    )
    assert result.exit_code != 0
    assert "no authorization" in result.output.lower()


def test_cli_refresh_auth_reads_stdin(tmp_path):
    keys_path = tmp_path / "keys.json"
    keys_path.write_text("{}")
    runner = CliRunner()
    result = runner.invoke(
        fp_fetch,
        ["refresh-auth", "-", "--keys-path", str(keys_path)],
        input="curl 'https://x/' -H 'Authorization: Bearer STDIN_TOKEN'",
    )
    assert result.exit_code == 0, result.output
    assert json.loads(keys_path.read_text())["fantasypoints_authorization"] == "Bearer STDIN_TOKEN"


def test_cli_import_curl_rejects_duplicate_name(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    save_catalog(
        [EndpointSpec(name="line_matchups", url="https://x/", output_subdir="x")],
        catalog_path,
    )
    curl_path = tmp_path / "snippet.curl"
    curl_path.write_text("curl 'https://x/'")
    runner = CliRunner()
    result = runner.invoke(
        fp_fetch,
        [
            "import-curl",
            str(curl_path),
            "--name",
            "line_matchups",
            "--output-subdir",
            "team/dup",
            "--catalog",
            str(catalog_path),
        ],
    )
    assert result.exit_code != 0
    assert "already exists" in result.output
    assert "--replace" in result.output


def test_cli_import_curl_replace_overwrites_body_and_sentinelises_season(tmp_path):
    """``--replace`` swaps the body in place and rewrites the literal season.

    Use case: a discover-generated body returns 0 rows because the SPA
    injects per-tool position + qualifier filters. The user pastes the
    working curl and uses ``--replace`` to overlay the correct body
    onto the existing entry — output_subdir is preserved, the literal
    ``game.season.eq: 2025`` becomes the int sentinel so the entry
    stays usable for any season.
    """
    catalog_path = tmp_path / "catalog.json"
    save_catalog(
        [
            EndpointSpec(
                name="player_passing_advanced",
                url="https://data.fantasypoints.com/v2/ds/nfl/tools/player/passing-advanced/values",
                method="POST",
                json_body={"context": {"filterMatch": {}}, "useCache": True},
                output_subdir="player/passing_advanced",
            ),
        ],
        catalog_path,
    )
    curl_path = tmp_path / "passing_advanced.curl"
    # Real working curl body with position + qualifiers + literal 2025 season.
    curl_path.write_text(
        "curl 'https://data.fantasypoints.com/v2/ds/nfl/tools/player/passing-advanced/values' "
        "-X POST "
        '--data-raw \'{"context":{"tableProperty":"passingAdvanced",'
        '"filterMatch":{"isGamePlayed":{"eq":true},"game.season":{"eq":2025},'
        '"player.position":{"in":["QB"]}},'
        '"filterResult":{"playerStats.passing.dropbacks.total":{"gte":1}}},'
        '"useCache":true}\''
    )
    runner = CliRunner()
    result = runner.invoke(
        fp_fetch,
        [
            "import-curl",
            str(curl_path),
            "--name",
            "player_passing_advanced",
            "--replace",
            "--catalog",
            str(catalog_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Replaced" in result.output
    catalog = load_catalog(catalog_path)
    assert len(catalog) == 1
    spec = catalog[0]
    # output_subdir preserved from the original entry.
    assert spec.output_subdir == "player/passing_advanced"
    # New body landed.
    fm = spec.json_body["context"]["filterMatch"]
    assert fm["player.position"] == {"in": ["QB"]}
    # Season rewritten to the sentinel that body_substitute consumes.
    assert fm["game.season"] == {"eq": "__SEASON_INT__"}
    # Other filterMatch keys preserved untouched.
    assert fm["isGamePlayed"] == {"eq": True}
    # filterResult qualifier carried over.
    assert spec.json_body["context"]["filterResult"] == {
        "playerStats.passing.dropbacks.total": {"gte": 1}
    }


def test_cli_import_curl_replace_errors_on_missing_name(tmp_path):
    """``--replace`` without an existing name and no --output-subdir is an error."""
    catalog_path = tmp_path / "catalog.json"
    save_catalog([], catalog_path)
    curl_path = tmp_path / "snippet.curl"
    curl_path.write_text("curl 'https://x/' -X POST")
    runner = CliRunner()
    result = runner.invoke(
        fp_fetch,
        [
            "import-curl",
            str(curl_path),
            "--name",
            "never_existed",
            "--catalog",
            str(catalog_path),
        ],
    )
    assert result.exit_code != 0
    assert "--output-subdir" in result.output


def _registry_sample() -> dict:
    """Minimal registry payload mirroring FP's tool index response shape."""
    return {
        "content": {
            "tables": {
                "values": [
                    {
                        "isPublished": True,
                        "isPrivate": False,
                        "property": "passingBasic",
                        "context": ["player", "team", "opponent"],
                    },
                    {
                        "isPublished": True,
                        "isPrivate": False,
                        "property": "lineMatchups",
                        "context": ["team", "other"],
                    },
                    {
                        "isPublished": True,
                        "isPrivate": True,
                        "property": "debug",
                        "context": ["player"],
                    },
                    {
                        "isPublished": False,
                        "isPrivate": False,
                        "property": "draftOnly",
                        "context": ["player"],
                    },
                ]
            }
        }
    }


def test_camel_to_kebab_simple():
    assert _camel_to_kebab("passingBasic") == "passing-basic"
    assert _camel_to_kebab("lineMatchups") == "line-matchups"
    assert _camel_to_kebab("receivingSeparationByRoutes") == "receiving-separation-by-routes"
    # No-camel input is returned untouched (still lowercased).
    assert _camel_to_kebab("efficiency") == "efficiency"


def test_expand_registry_filters_unpublished_and_private_by_default():
    specs = expand_registry(_registry_sample())
    names = {s.name for s in specs}
    assert "player_debug" not in names, "isPrivate tables must be skipped by default"
    assert "player_draft_only" not in names, "isPublished=false tables must be skipped"


def test_expand_registry_include_private_keeps_debug_tables():
    specs = expand_registry(_registry_sample(), include_private=True)
    names = {s.name for s in specs}
    assert "player_debug" in names


def test_expand_registry_creates_one_entry_per_context_and_skips_other():
    specs = expand_registry(_registry_sample())
    by_name = {s.name: s for s in specs}
    # passingBasic exposes 3 known contexts:
    assert "player_passing_basic" in by_name
    assert "team_passing_basic" in by_name
    assert "opponent_passing_basic" in by_name
    # lineMatchups exposes team + "other"; "other" must be skipped:
    assert "team_line_matchups" in by_name
    assert "other_line_matchups" not in by_name


def test_expand_registry_skips_names_already_in_catalog():
    specs = expand_registry(
        _registry_sample(),
        existing_names={"team_line_matchups", "player_passing_basic"},
    )
    names = {s.name for s in specs}
    assert "team_line_matchups" not in names
    assert "player_passing_basic" not in names
    # Other contexts of the same tool still come through:
    assert "team_passing_basic" in names
    assert "opponent_passing_basic" in names


def test_expand_registry_generates_valid_url_and_body():
    specs = expand_registry(_registry_sample())
    spec = next(s for s in specs if s.name == "player_passing_basic")
    assert spec.method == "POST"
    # v2 endpoint: ``/values`` suffix on every URL.
    assert spec.url == "https://data.fantasypoints.com/v2/ds/nfl/tools/player/passing-basic/values"
    assert spec.output_subdir == "player/passing_basic"
    ctx = spec.json_body["context"]
    assert ctx["tableProperty"] == "passingBasic"
    assert ctx["routeContext"] == "player"
    assert ctx["routeContextTarget"] == "player"
    assert ctx["modelContext"] == "player"
    assert ctx["grouping"] == "$player.playerId"
    # Sentinel placeholders that body_substitute rewrites at call time.
    assert ctx["weeks"] == {"REG": ["__WEEK_INT__"]}
    # ``isGamePlayed`` + ``game.season`` are both required for FP to
    # return any rows. ``filterPlay.teamRoles`` selects offense vs
    # defense plays — without it FP returns 0 plays.
    assert ctx["filterMatch"] == {
        "isGamePlayed": {"eq": True},
        "game.season": {"eq": "__SEASON_INT__"},
    }
    assert ctx["filterPlay"] == {
        "teamRoles": {"in": ["offense", {"$ifNull": ["$$play.team.roles", []]}]}
    }
    assert spec.json_body["useCache"] is True


def test_expand_registry_team_body_omits_player_only_filters():
    """Team-context tools must NOT carry isGamePlayed / teamRoles.

    Working DevTools curls for team-offense tools (line_matchups,
    run_pass_report) have empty filterMatch (apart from season) and
    empty filterPlay. Sending the player-style filters here makes FP
    return 0 rows.
    """
    specs = expand_registry(_registry_sample())
    spec = next(s for s in specs if s.name == "team_passing_basic")
    ctx = spec.json_body["context"]
    assert ctx["filterMatch"] == {"game.season": {"eq": "__SEASON_INT__"}}
    assert ctx["filterPlay"] == {}


def test_expand_registry_opponent_body_omits_player_only_filters():
    """Opponent-context tools also omit the player-only filterPlay default."""
    specs = expand_registry(_registry_sample())
    spec = next(s for s in specs if s.name == "opponent_passing_basic")
    ctx = spec.json_body["context"]
    assert ctx["filterMatch"] == {"game.season": {"eq": "__SEASON_INT__"}}
    assert ctx["filterPlay"] == {}


def test_expand_registry_routes_team_to_bare_team_url():
    """Team-context URL is ``/team/{slug}/values`` — no ``/offense/`` segment.

    The SPA shows ``/team/{slug}`` for offense views and only inserts
    ``defense`` for the opponent view; ``/team/offense/{slug}`` doesn't
    exist and silently returns empty.
    """
    specs = expand_registry(_registry_sample())
    spec = next(s for s in specs if s.name == "team_passing_basic")
    assert spec.url == "https://data.fantasypoints.com/v2/ds/nfl/tools/team/passing-basic/values"
    ctx = spec.json_body["context"]
    assert ctx["routeContext"] == "team"
    assert ctx["routeContextTarget"] == "offense"
    assert ctx["modelContext"] == "team"


def test_expand_registry_routes_opponent_to_defense_url():
    specs = expand_registry(_registry_sample())
    spec = next(s for s in specs if s.name == "opponent_passing_basic")
    assert (
        spec.url
        == "https://data.fantasypoints.com/v2/ds/nfl/tools/team/defense/passing-basic/values"
    )
    ctx = spec.json_body["context"]
    assert ctx["routeContext"] == "team"
    assert ctx["routeContextTarget"] == "defense"
    assert ctx["modelContext"] == "opponent"


def test_expand_registry_passes_registry_requires_charting_and_roles():
    registry = {
        "content": {
            "tables": {
                "values": [
                    {
                        "isPublished": True,
                        "isPrivate": False,
                        "property": "passingBasic",
                        "context": ["player"],
                        "requiresCharting": True,
                        "requiresPlayByPlay": False,
                        "roles": ["role_uJb30OfZpu4pFfS7VQEq"],
                    },
                ]
            }
        }
    }
    spec = expand_registry(registry)[0]
    ctx = spec.json_body["context"]
    assert ctx["requiresCharting"] is True
    assert ctx["requiresPlayByPlay"] is False
    assert ctx["requiredRoles"] == ["role_uJb30OfZpu4pFfS7VQEq"]


def test_expand_registry_handles_empty_or_malformed_payload():
    assert expand_registry({}) == []
    assert expand_registry({"content": {}}) == []
    assert expand_registry({"content": {"tables": {"values": []}}}) == []


def test_expand_registry_skips_published_table_missing_prop_or_contexts():
    registry = {
        "content": {
            "tables": {
                "values": [
                    {"isPublished": True, "isPrivate": False, "property": "x", "context": []},
                    {"isPublished": True, "isPrivate": False, "context": ["player"]},
                ]
            }
        }
    }
    assert expand_registry(registry) == []


def test_cli_discover_dry_run_lists_new_endpoints(monkeypatch, tmp_path):
    from sportstradamus.collectors import transport as client_mod

    catalog_path = tmp_path / "catalog.json"

    def fake_request(method, url, headers=None, params=None, json=None):
        assert method == "POST"
        return FakeResponse(200, body=_registry_sample())

    monkeypatch.setattr(client_mod.requests, "request", fake_request)
    monkeypatch.setenv("FANTASYPOINTS_AUTHORIZATION", "Bearer test")
    monkeypatch.setenv("FANTASYPOINTS_COOKIE", "c=1")
    runner = CliRunner()
    result = runner.invoke(
        fp_fetch,
        ["discover", "--catalog", str(catalog_path), "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "player_passing_basic" in result.output
    assert "team_line_matchups" in result.output
    assert "Would add" in result.output
    assert not catalog_path.exists(), "dry-run must not write"


def test_cli_discover_writes_catalog_and_skips_existing(monkeypatch, tmp_path):
    from sportstradamus.collectors import transport as client_mod

    catalog_path = tmp_path / "catalog.json"
    save_catalog(
        [
            EndpointSpec(
                name="team_line_matchups",
                url="https://example/preexisting",
                output_subdir="team/line_matchups",
                method="POST",
            )
        ],
        catalog_path,
    )

    def fake_request(method, url, headers=None, params=None, json=None):
        return FakeResponse(200, body=_registry_sample())

    monkeypatch.setattr(client_mod.requests, "request", fake_request)
    monkeypatch.setenv("FANTASYPOINTS_AUTHORIZATION", "Bearer test")
    monkeypatch.setenv("FANTASYPOINTS_COOKIE", "c=1")
    runner = CliRunner()
    result = runner.invoke(
        fp_fetch,
        ["discover", "--catalog", str(catalog_path)],
    )
    assert result.exit_code == 0, result.output
    on_disk = load_catalog(catalog_path)
    names = [s.name for s in on_disk]
    # Pre-existing entry kept with its custom URL:
    line = next(s for s in on_disk if s.name == "team_line_matchups")
    assert line.url == "https://example/preexisting"
    # New entries added:
    assert "player_passing_basic" in names
    assert "team_passing_basic" in names
    assert "opponent_passing_basic" in names
    # Private/unpublished still skipped:
    assert "player_debug" not in names
    assert "player_draft_only" not in names


def test_parse_table_response_extracts_rows_and_columns():
    payload = {
        "content": {
            "table": {
                "rows": {
                    "count": 2,
                    "values": [
                        {"playerFirstName": "A", "gameWeek": 5, "yards": 100},
                        {"playerFirstName": "B", "gameWeek": 5, "yards": 80},
                    ],
                }
            }
        }
    }
    df = parse_table_response(payload)
    assert len(df) == 2
    assert set(df.columns) == {"playerFirstName", "gameWeek", "yards"}
    assert list(df["yards"]) == [100, 80]


def test_parse_table_response_serialises_nested_columns_to_json():
    payload = {
        "content": {
            "table": {
                "rows": {
                    "values": [
                        {
                            "playerId": "X1",
                            "opponentsPlayed": [{"abbr": "CIN"}, {"abbr": "DET"}],
                        },
                    ]
                }
            }
        }
    }
    df = parse_table_response(payload)
    assert df["opponentsPlayed"].iloc[0] == '[{"abbr":"CIN"},{"abbr":"DET"}]'


def test_parse_table_response_empty_on_missing_paths():
    assert parse_table_response({}).empty
    assert parse_table_response({"content": {}}).empty
    assert parse_table_response({"content": {"table": {}}}).empty
    assert parse_table_response({"content": {"table": {"rows": {"values": []}}}}).empty


def test_parse_table_response_handles_v2_values_shape_without_table_wrapper():
    """v2 ``/values`` endpoints return ``content.rows.values`` directly.

    Symptom on regression: parquets land at the right path but every
    file is 1 KB with 0 rows because the parser looks at
    ``content.table.rows`` (the legacy shape).
    """
    payload = {
        "content": {
            "rows": {
                "count": 2,
                "values": [
                    {"playerFirstName": "Patrick", "gameWeek": 5, "passingYards": 312},
                    {"playerFirstName": "Josh", "gameWeek": 5, "passingYards": 287},
                ],
            }
        }
    }
    df = parse_table_response(payload)
    assert len(df) == 2
    assert list(df["passingYards"]) == [312, 287]


def test_parse_table_response_prefers_v2_shape_when_both_present():
    """If both shapes are present, the v2 one wins — that's what FP sends today."""
    payload = {
        "content": {
            "rows": {"values": [{"id": "new"}]},
            "table": {"rows": {"values": [{"id": "old"}]}},
        }
    }
    df = parse_table_response(payload)
    assert list(df["id"]) == ["new"]


def _spec(name="x", url="https://example/", output_subdir="x/x") -> EndpointSpec:
    return EndpointSpec(name=name, url=url, output_subdir=output_subdir)


def test_data_base_resolves_to_filesystem_path_not_multiplexed_repr():
    """Regression: namespace-package data dir must resolve to a real filesystem path.

    `pkg_resources.files(data)` returns a MultiplexedPath whose `str()`
    is `"MultiplexedPath('/real/path')"` (with the wrapper text),
    which silently lands parquets in a literal `MultiplexedPath('...')`
    directory the user can't find. Using `data.__path__[0]` instead
    gives the real path.
    """
    from sportstradamus.collectors.fantasypoints.transform import (
        PLAYER_DATA_BASE,
        TEAM_DATA_BASE,
    )

    assert "MultiplexedPath" not in str(PLAYER_DATA_BASE)
    assert "MultiplexedPath" not in str(TEAM_DATA_BASE)
    assert str(PLAYER_DATA_BASE).endswith("/data/player_data")
    assert str(TEAM_DATA_BASE).endswith("/data/team_data")


def test_parquet_path_routes_player_to_player_data(monkeypatch, tmp_path):
    from sportstradamus.collectors.fantasypoints import transform as tm

    monkeypatch.setattr(tm, "PLAYER_DATA_BASE", tmp_path / "player_data")
    monkeypatch.setattr(tm, "TEAM_DATA_BASE", tmp_path / "team_data")
    p = parquet_path_for_spec(_spec("player_passing_advanced"), season=2025, week=11)
    assert p == tmp_path / "player_data" / "NFL" / "2025" / "week_11" / "passing_advanced.parquet"


def test_parquet_path_routes_team_to_team_data(monkeypatch, tmp_path):
    from sportstradamus.collectors.fantasypoints import transform as tm

    monkeypatch.setattr(tm, "PLAYER_DATA_BASE", tmp_path / "player_data")
    monkeypatch.setattr(tm, "TEAM_DATA_BASE", tmp_path / "team_data")
    p = parquet_path_for_spec(_spec("team_line_matchups"), season=2025, week=5)
    assert p == tmp_path / "team_data" / "NFL" / "2025" / "week_05" / "line_matchups.parquet"


def test_parquet_path_routes_opponent_with_opp_suffix(monkeypatch, tmp_path):
    from sportstradamus.collectors.fantasypoints import transform as tm

    monkeypatch.setattr(tm, "PLAYER_DATA_BASE", tmp_path / "player_data")
    monkeypatch.setattr(tm, "TEAM_DATA_BASE", tmp_path / "team_data")
    p = parquet_path_for_spec(_spec("opponent_passing_advanced"), season=2024, week=18)
    assert p == tmp_path / "team_data" / "NFL" / "2024" / "week_18" / "passing_advanced_opp.parquet"


def test_parquet_path_falls_back_to_url_for_legacy_unprefixed_name(monkeypatch, tmp_path):
    """Hand-imported entries without the discover-prefix still route via URL path."""
    from sportstradamus.collectors.fantasypoints import transform as tm

    monkeypatch.setattr(tm, "PLAYER_DATA_BASE", tmp_path / "player_data")
    monkeypatch.setattr(tm, "TEAM_DATA_BASE", tmp_path / "team_data")
    spec = _spec(
        name="line_matchups",
        url="https://data.fantasypoints.com/v2/ds/nfl/tools/team/line-matchups",
        output_subdir="team/line_matchups",
    )
    p = parquet_path_for_spec(spec, season=2025, week=5)
    assert p == tmp_path / "team_data" / "NFL" / "2025" / "week_05" / "line_matchups.parquet"


def test_parquet_path_falls_back_to_output_subdir_when_url_is_opaque(monkeypatch, tmp_path):
    from sportstradamus.collectors.fantasypoints import transform as tm

    monkeypatch.setattr(tm, "PLAYER_DATA_BASE", tmp_path / "player_data")
    monkeypatch.setattr(tm, "TEAM_DATA_BASE", tmp_path / "team_data")
    spec = _spec(
        name="player_some_thing",
        url="https://opaque-cdn.example/api?id=xyz",
        output_subdir="player/some_thing",
    )
    p = parquet_path_for_spec(spec, season=2025, week=1)
    # Name prefix wins (first routing source).
    assert p == tmp_path / "player_data" / "NFL" / "2025" / "week_01" / "some_thing.parquet"


def test_parquet_path_rejects_unrouted_spec():
    spec = _spec(
        name="misc_thing",
        url="https://opaque-cdn.example/api?id=xyz",
        output_subdir="misc/whatever",
    )
    with pytest.raises(ValueError, match="no recognisable context"):
        parquet_path_for_spec(spec, season=2025, week=1)


def _read_report(result) -> dict:
    """Load the report this CLI run wrote, from its ``Report: <path>`` line.

    Parsing the path out of the run's own output (rather than globbing the
    shared tempdir) keeps the read deterministic under parallel xdist
    workers, which otherwise interleave same-second report filenames.
    """
    for line in result.output.splitlines():
        marker = "Report: "
        if marker in line:
            path = line.split(marker, 1)[1].strip()
            return json.loads(Path(path).read_text())
    raise AssertionError(f"no 'Report:' line in output:\n{result.output}")


def test_cli_run_writes_report_with_per_spec_outcomes(monkeypatch, tmp_path):
    """Run with a mix of ok/empty/failed specs and verify the report captures each."""
    from sportstradamus.collectors import transport as client_mod

    _redirect_parquet_dirs(monkeypatch, tmp_path)
    catalog_path = tmp_path / "catalog.json"
    save_catalog(
        [
            EndpointSpec(
                name="player_passing_basic",
                url="https://fp/v2/ds/nfl/tools/player/passing-basic",
                method="POST",
                json_body={"useCache": True},
                output_subdir="player/passing_basic",
            ),
            EndpointSpec(
                name="team_line_matchups",
                url="https://fp/v2/ds/nfl/tools/team/line-matchups",
                method="POST",
                json_body={"useCache": True},
                output_subdir="team/line_matchups",
            ),
            EndpointSpec(
                name="player_will_fail",
                url="https://fp/v2/ds/nfl/tools/player/will-fail",
                method="POST",
                json_body={"useCache": True},
                output_subdir="player/will_fail",
            ),
        ],
        catalog_path,
    )
    monkeypatch.setenv("FANTASYPOINTS_AUTHORIZATION", "Bearer test")
    monkeypatch.setattr(client_mod, "_INTER_REQUEST_SLEEP_S", 0.0)
    monkeypatch.setattr(client_mod, "_RETRY_BACKOFF_S", (0.0, 0.0, 0.0))

    def fake_request(method, url, headers=None, params=None, json=None):
        if "passing-basic" in url:
            return FakeResponse(
                200,
                body={
                    "content": {
                        "table": {"rows": {"count": 1, "values": [{"playerFirstName": "Q"}]}}
                    }
                },
            )
        if "line-matchups" in url:
            return FakeResponse(200, body={"content": {"table": {"rows": {"values": []}}}})
        # will-fail: HTTP 500 (retried 3x then raises HTTPError).
        return FakeResponse(500, text="upstream is having a moment")

    monkeypatch.setattr(client_mod.requests, "request", fake_request)
    runner = CliRunner()
    result = runner.invoke(
        fp_fetch,
        [
            "run",
            "--season",
            "2025",
            "--week",
            "5",
            "--catalog",
            str(catalog_path),
        ],
    )
    # One spec failed → non-zero exit code, but the other two were
    # still written and the report still got dumped.
    assert result.exit_code != 0
    assert "Report:" in result.output
    report = _read_report(result)
    assert report["command"] == "run"
    assert report["summary"]["total"] == 3
    assert report["summary"]["ok"] == 1
    assert report["summary"]["empty"] == 1
    assert report["summary"]["fetch_failed"] == 1
    by_name = {r["name"]: r for r in report["results"]}
    assert by_name["player_passing_basic"]["status"] == "ok"
    assert by_name["player_passing_basic"]["rows"] == 1
    assert by_name["team_line_matchups"]["status"] == "empty"
    assert by_name["player_will_fail"]["status"] == "fetch_failed"
    # HTTP status surfaced for the failed spec:
    assert by_name["player_will_fail"]["http_status"] == 500
    # Response preview included so I can diagnose the failure mode:
    assert "upstream is having a moment" in (by_name["player_will_fail"]["response_preview"] or "")
    # Request body captured so I can verify what we sent:
    assert by_name["player_passing_basic"]["request_body"] == {"useCache": True}


def test_cli_run_records_routing_failure_in_report_without_aborting_batch(monkeypatch, tmp_path):
    """A spec with an unroutable name fails alone; other specs still write."""
    from sportstradamus.collectors import transport as client_mod

    _redirect_parquet_dirs(monkeypatch, tmp_path)
    catalog_path = tmp_path / "catalog.json"
    save_catalog(
        [
            EndpointSpec(
                name="misc_thing",
                url="https://opaque/api",
                method="POST",
                json_body={"useCache": True},
                output_subdir="misc/thing",
            ),
            EndpointSpec(
                name="player_passing_basic",
                url="https://fp/v2/ds/nfl/tools/player/passing-basic",
                method="POST",
                json_body={"useCache": True},
                output_subdir="player/passing_basic",
            ),
        ],
        catalog_path,
    )
    monkeypatch.setenv("FANTASYPOINTS_AUTHORIZATION", "Bearer test")
    monkeypatch.setattr(client_mod, "_INTER_REQUEST_SLEEP_S", 0.0)

    def fake_request(method, url, headers=None, params=None, json=None):
        return FakeResponse(
            200,
            body={"content": {"table": {"rows": {"count": 1, "values": [{"x": 1}]}}}},
        )

    monkeypatch.setattr(client_mod.requests, "request", fake_request)
    runner = CliRunner()
    result = runner.invoke(
        fp_fetch,
        ["run", "--season", "2025", "--week", "5", "--catalog", str(catalog_path)],
    )
    assert result.exit_code != 0  # one routing failure
    report = _read_report(result)
    by_name = {r["name"]: r for r in report["results"]}
    assert by_name["misc_thing"]["status"] == "routing_failed"
    # The OTHER spec still wrote its parquet:
    assert by_name["player_passing_basic"]["status"] == "ok"
    parquet = tmp_path / "player_data" / "NFL" / "2025" / "week_05" / "passing_basic.parquet"
    assert parquet.is_file()


def test_write_parquet_creates_parent_dirs(tmp_path):
    target = tmp_path / "a" / "b" / "c.parquet"
    write_parquet(pd.DataFrame({"x": [1, 2, 3]}), target)
    assert target.is_file()
    assert list(pd.read_parquet(target)["x"]) == [1, 2, 3]


def test_client_refresh_credentials_updates_in_place(monkeypatch):
    client = _client()
    monkeypatch.setattr(client, "_authorization", "Bearer OLD")
    monkeypatch.setattr(client, "_cookie", "old=1")
    client.refresh_credentials(authorization="Bearer NEW", cookie="new=2")
    assert client._authorization == "Bearer NEW"
    assert client._cookie == "new=2"
    # Sleep guard reset so the post-refresh call doesn't pause.
    assert client._first_call is True


def test_cli_run_interactive_refresh_resumes_after_401(monkeypatch, tmp_path):
    """On 401, the interactive refresh hook runs, updates the client, and the retry succeeds."""
    from sportstradamus.collectors import dispatch as dispatch_mod
    from sportstradamus.collectors import transport as client_mod

    _redirect_parquet_dirs(monkeypatch, tmp_path)
    catalog_path = tmp_path / "catalog.json"
    save_catalog(
        [
            EndpointSpec(
                name="team_line_matchups",
                url="https://example/",
                method="POST",
                json_body={"useCache": True},
                output_subdir="team/line_matchups",
            ),
        ],
        catalog_path,
    )
    monkeypatch.setenv("FANTASYPOINTS_AUTHORIZATION", "Bearer expired")
    monkeypatch.setattr(client_mod, "_INTER_REQUEST_SLEEP_S", 0.0)

    call_count = {"n": 0}
    good_body = {
        "content": {"table": {"rows": {"count": 1, "values": [{"teamAbbreviation": "CIN"}]}}}
    }

    def fake_request(method, url, headers=None, params=None, json=None):
        call_count["n"] += 1
        # First call fails 401, second succeeds (after the refresh).
        if call_count["n"] == 1:
            return FakeResponse(401)
        return FakeResponse(200, body=good_body)

    monkeypatch.setattr(client_mod.requests, "request", fake_request)

    # Stub the prompt helper so CliRunner's faked stdin doesn't interfere.
    # The stub simulates a successful paste-and-update by swapping in a
    # fresh token via refresh_credentials.
    refresh_calls = {"n": 0}

    def fake_prompt(source, client):
        refresh_calls["n"] += 1
        client.refresh_credentials(authorization="Bearer NEW", cookie="new=1")
        return True

    monkeypatch.setattr(dispatch_mod, "_refresh_auth_interactively", fake_prompt)

    runner = CliRunner()
    result = runner.invoke(
        fp_fetch,
        [
            "run",
            "--season",
            "2025",
            "--week",
            "5",
            "--catalog",
            str(catalog_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert refresh_calls["n"] == 1
    assert call_count["n"] == 2
    parquet = tmp_path / "team_data" / "NFL" / "2025" / "week_05" / "line_matchups.parquet"
    assert parquet.is_file()


def test_cli_backfill_dry_run_reports_call_count(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    save_catalog(
        [
            EndpointSpec(
                name="team_line_matchups",
                url="https://example/",
                output_subdir="team/line_matchups",
            ),
            EndpointSpec(
                name="player_passing_basic",
                url="https://example/",
                output_subdir="player/passing_basic",
            ),
        ],
        catalog_path,
    )
    runner = CliRunner()
    result = runner.invoke(
        fp_fetch,
        [
            "backfill",
            "--start-season",
            "2022",
            "--end-season",
            "2024",
            "--start-week",
            "1",
            "--end-week",
            "5",
            "--catalog",
            str(catalog_path),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    # 2 tools x 5 weeks x 3 seasons = 30 calls
    assert "30" in result.output


def _patch_backfill_pacing(monkeypatch):
    """Replace random.uniform + time.sleep in the runner with no-op recorders.

    Returns the lists each gets appended to so tests can assert on
    the pacing ranges and number of sleeps.
    """
    from sportstradamus.collectors import runner as runner_mod

    uniform_calls: list[tuple[float, float]] = []
    sleep_calls: list[float] = []

    def fake_uniform(a, b):
        uniform_calls.append((a, b))
        return (a + b) / 2

    monkeypatch.setattr(runner_mod.random, "uniform", fake_uniform)
    monkeypatch.setattr(runner_mod.time, "sleep", sleep_calls.append)
    return uniform_calls, sleep_calls


def test_cli_backfill_writes_parquets_for_each_week(monkeypatch, tmp_path):
    from sportstradamus.collectors import transport as client_mod

    _redirect_parquet_dirs(monkeypatch, tmp_path)
    _patch_backfill_pacing(monkeypatch)
    catalog_path = tmp_path / "catalog.json"
    save_catalog(
        [
            EndpointSpec(
                name="player_passing_basic",
                url="https://example/",
                method="POST",
                json_body={
                    "filters": {"week": "{week}", "season": "{season}"},
                    "useCache": True,
                },
                output_subdir="player/passing_basic",
            ),
        ],
        catalog_path,
    )
    monkeypatch.setenv("FANTASYPOINTS_AUTHORIZATION", "Bearer test")
    monkeypatch.setattr(client_mod, "_INTER_REQUEST_SLEEP_S", 0.0)

    seen_bodies = []

    def fake_request(method, url, headers=None, params=None, json=None):
        seen_bodies.append(json)
        return FakeResponse(
            200,
            body={
                "content": {
                    "table": {
                        "rows": {"count": 1, "values": [{"playerFirstName": "Q"}]},
                    }
                }
            },
        )

    monkeypatch.setattr(client_mod.requests, "request", fake_request)
    runner = CliRunner()
    result = runner.invoke(
        fp_fetch,
        [
            "backfill",
            "--start-season",
            "2023",
            "--end-season",
            "2023",
            "--start-week",
            "1",
            "--end-week",
            "3",
            "--catalog",
            str(catalog_path),
        ],
    )
    assert result.exit_code == 0, result.output
    # One file per week:
    base = tmp_path / "player_data" / "NFL" / "2023"
    for w in (1, 2, 3):
        assert (base / f"week_{w:02d}" / "passing_basic.parquet").is_file()
    # And each call's body had the substituted week/season:
    sent_weeks = [b["filters"]["week"] for b in seen_bodies]
    assert sent_weeks == ["1", "2", "3"]
    sent_seasons = {b["filters"]["season"] for b in seen_bodies}
    assert sent_seasons == {"2023"}


def test_cli_backfill_uses_week_pause_on_week_transition(monkeypatch, tmp_path):
    """One spec x 3 weeks → 0 same-week pauses, 2 week-transition pauses."""
    from sportstradamus.collectors import transport as client_mod

    _redirect_parquet_dirs(monkeypatch, tmp_path)
    uniform_calls, sleep_calls = _patch_backfill_pacing(monkeypatch)
    catalog_path = tmp_path / "catalog.json"
    save_catalog(
        [
            EndpointSpec(
                name="player_passing_basic",
                url="https://example/",
                method="POST",
                output_subdir="player/passing_basic",
            ),
        ],
        catalog_path,
    )
    monkeypatch.setenv("FANTASYPOINTS_AUTHORIZATION", "Bearer test")
    monkeypatch.setattr(client_mod, "_INTER_REQUEST_SLEEP_S", 0.0)

    def ok(*a, **k):
        return FakeResponse(200, body={"content": {"table": {"rows": {"values": []}}}})

    monkeypatch.setattr(client_mod.requests, "request", ok)
    runner = CliRunner()
    result = runner.invoke(
        fp_fetch,
        [
            "backfill",
            "--start-season",
            "2023",
            "--end-season",
            "2023",
            "--start-week",
            "1",
            "--end-week",
            "3",
            "--catalog",
            str(catalog_path),
        ],
    )
    assert result.exit_code == 0, result.output
    # 3 calls -> 2 pauses (none before the first). Both should be
    # week-transition pauses (each spec change is also a week change
    # since there's only one spec).
    assert uniform_calls == [(8.0, 28.0), (8.0, 28.0)]
    assert len(sleep_calls) == 2


def test_cli_backfill_uses_request_pause_within_a_week(monkeypatch, tmp_path):
    """Two specs x 2 weeks → 1 same-week pause per week + 1 week pause = 3 sleeps."""
    from sportstradamus.collectors import transport as client_mod

    _redirect_parquet_dirs(monkeypatch, tmp_path)
    uniform_calls, sleep_calls = _patch_backfill_pacing(monkeypatch)
    catalog_path = tmp_path / "catalog.json"
    save_catalog(
        [
            EndpointSpec(
                name="player_a",
                url="https://example/a",
                method="POST",
                output_subdir="player/a",
            ),
            EndpointSpec(
                name="player_b",
                url="https://example/b",
                method="POST",
                output_subdir="player/b",
            ),
        ],
        catalog_path,
    )
    monkeypatch.setenv("FANTASYPOINTS_AUTHORIZATION", "Bearer test")
    monkeypatch.setattr(client_mod, "_INTER_REQUEST_SLEEP_S", 0.0)

    def ok(*a, **k):
        return FakeResponse(200, body={"content": {"table": {"rows": {"values": []}}}})

    monkeypatch.setattr(client_mod.requests, "request", ok)
    runner = CliRunner()
    result = runner.invoke(
        fp_fetch,
        [
            "backfill",
            "--start-season",
            "2023",
            "--end-season",
            "2023",
            "--start-week",
            "1",
            "--end-week",
            "2",
            "--request-pause-min",
            "5",
            "--request-pause-max",
            "8",
            "--week-pause-min",
            "60",
            "--week-pause-max",
            "120",
            "--catalog",
            str(catalog_path),
        ],
    )
    assert result.exit_code == 0, result.output
    # 4 calls in order: (s23,w1,a), (s23,w1,b), (s23,w2,a), (s23,w2,b)
    # Pauses between consecutive calls:
    #   before (s23,w1,b): same week -> request range (5,8)
    #   before (s23,w2,a): week transition -> week range (60,120)
    #   before (s23,w2,b): same week -> request range (5,8)
    assert uniform_calls == [(5.0, 8.0), (60.0, 120.0), (5.0, 8.0)]
    assert len(sleep_calls) == 3


# ---------------------------------------------------------------------------
# body_substitute
# ---------------------------------------------------------------------------


def _v2_body():
    """A representative discover-generated catalog body with sentinels."""
    return {
        "context": {
            "tableProperty": "passingBasic",
            "grouping": "$player.playerId",
            "weeks": {"REG": ["__WEEK_INT__"]},
            "filterMatch": {"game.season": {"eq": "__SEASON_INT__"}},
        },
        "useCache": True,
        "flatten": True,
    }


def test_substitute_runtime_replaces_int_sentinels():
    from sportstradamus.collectors.fantasypoints.body_substitute import substitute_runtime

    out = substitute_runtime(_v2_body(), season=2023, week=8, mode="weekly")
    assert out["context"]["weeks"] == {"REG": [8]}
    assert out["context"]["filterMatch"]["game.season"]["eq"] == 2023
    # Values are ints, not strings.
    assert isinstance(out["context"]["weeks"]["REG"][0], int)
    assert isinstance(out["context"]["filterMatch"]["game.season"]["eq"], int)


def test_substitute_runtime_s2d_expands_weeks_to_range():
    from sportstradamus.collectors.fantasypoints.body_substitute import substitute_runtime

    out = substitute_runtime(_v2_body(), season=2023, week=8, mode="season_to_date")
    assert out["context"]["weeks"] == {"REG": [1, 2, 3, 4, 5, 6, 7, 8]}


def test_substitute_runtime_postseason_uses_post_block():
    from sportstradamus.collectors.fantasypoints.body_substitute import substitute_runtime

    out = substitute_runtime(_v2_body(), season=2023, week=3, mode="postseason")
    assert out["context"]["weeks"] == {"POST": [3]}


def test_substitute_runtime_does_not_mutate_input():
    from sportstradamus.collectors.fantasypoints.body_substitute import substitute_runtime

    src = _v2_body()
    snapshot = json.dumps(src, sort_keys=True)
    _ = substitute_runtime(src, season=2023, week=8, mode="weekly")
    assert json.dumps(src, sort_keys=True) == snapshot


def test_substitute_runtime_passes_through_legacy_body_without_sentinels():
    """Legacy hand-imported entries with no v2 sentinels are untouched."""
    from sportstradamus.collectors.fantasypoints.body_substitute import substitute_runtime

    legacy = {"context": {"week": "5"}, "useCache": True}
    out = substitute_runtime(legacy, season=2023, week=8, mode="weekly")
    assert out == legacy


# ---------------------------------------------------------------------------
# Mode-aware parquet filename suffix
# ---------------------------------------------------------------------------


def test_parquet_path_weekly_mode_has_no_suffix(monkeypatch, tmp_path):
    from sportstradamus.collectors.fantasypoints import transform as tm

    monkeypatch.setattr(tm, "PLAYER_DATA_BASE", tmp_path / "player_data")
    monkeypatch.setattr(tm, "TEAM_DATA_BASE", tmp_path / "team_data")
    p = parquet_path_for_spec(_spec("player_passing_basic"), season=2023, week=8, mode="weekly")
    assert p.name == "passing_basic.parquet"


def test_parquet_path_s2d_mode_adds_s2d_suffix(monkeypatch, tmp_path):
    from sportstradamus.collectors.fantasypoints import transform as tm

    monkeypatch.setattr(tm, "PLAYER_DATA_BASE", tmp_path / "player_data")
    monkeypatch.setattr(tm, "TEAM_DATA_BASE", tmp_path / "team_data")
    p = parquet_path_for_spec(
        _spec("player_passing_basic"), season=2023, week=8, mode="season_to_date"
    )
    assert p.name == "passing_basic_s2d.parquet"


def test_parquet_path_postseason_mode_uses_continuation_week_folder(monkeypatch, tmp_path):
    """Postseason rounds 1-4 map to folder weeks 19-22; no filename suffix."""
    from sportstradamus.collectors.fantasypoints import transform as tm

    monkeypatch.setattr(tm, "PLAYER_DATA_BASE", tmp_path / "player_data")
    monkeypatch.setattr(tm, "TEAM_DATA_BASE", tmp_path / "team_data")
    p = parquet_path_for_spec(
        _spec("opponent_coverage_matrix"), season=2023, week=3, mode="postseason"
    )
    assert p.name == "coverage_matrix_opp.parquet"
    assert p.parent.name == "week_21"


def test_parquet_path_postseason_mode_round_one_is_week_19(monkeypatch, tmp_path):
    """Wildcard round (postseason week 1) writes to ``week_19/`` folder."""
    from sportstradamus.collectors.fantasypoints import transform as tm

    monkeypatch.setattr(tm, "PLAYER_DATA_BASE", tmp_path / "player_data")
    monkeypatch.setattr(tm, "TEAM_DATA_BASE", tmp_path / "team_data")
    p = parquet_path_for_spec(_spec("player_passing_basic"), season=2023, week=1, mode="postseason")
    assert p.name == "passing_basic.parquet"
    assert p.parent.name == "week_19"


def test_parquet_path_rejects_unknown_mode(monkeypatch, tmp_path):
    from sportstradamus.collectors.fantasypoints import transform as tm

    monkeypatch.setattr(tm, "PLAYER_DATA_BASE", tmp_path / "player_data")
    monkeypatch.setattr(tm, "TEAM_DATA_BASE", tmp_path / "team_data")
    with pytest.raises(ValueError, match="Unknown mode"):
        parquet_path_for_spec(_spec("player_passing_basic"), season=2023, week=8, mode="bogus")


# ---------------------------------------------------------------------------
# CLI --mode flag end-to-end
# ---------------------------------------------------------------------------


def test_cli_run_mode_s2d_sends_expanded_weeks(monkeypatch, tmp_path):
    """`--mode season_to_date` rewrites context.weeks to [1..N] and writes _s2d filename."""
    from sportstradamus.collectors import transport as client_mod

    _redirect_parquet_dirs(monkeypatch, tmp_path)
    catalog_path = tmp_path / "catalog.json"
    save_catalog(
        [
            EndpointSpec(
                name="player_passing_basic",
                url="https://data.fantasypoints.com/v2/ds/nfl/tools/player/passing-basic/values",
                method="POST",
                json_body={
                    "context": {
                        "tableProperty": "passingBasic",
                        "weeks": {"REG": ["__WEEK_INT__"]},
                        "filterMatch": {"game.season": {"eq": "__SEASON_INT__"}},
                    },
                    "useCache": True,
                },
                output_subdir="player/passing_basic",
            )
        ],
        catalog_path,
    )
    monkeypatch.setenv("FANTASYPOINTS_AUTHORIZATION", "Bearer test")
    monkeypatch.setattr(client_mod, "_INTER_REQUEST_SLEEP_S", 0.0)
    captured = {}

    def capture(method, url, headers=None, params=None, json=None):
        captured["json"] = json
        return FakeResponse(200, body={"content": {"table": {"rows": {"values": [{"a": 1}]}}}})

    monkeypatch.setattr(client_mod.requests, "request", capture)
    runner = CliRunner()
    result = runner.invoke(
        fp_fetch,
        [
            "run",
            "--season",
            "2023",
            "--week",
            "5",
            "--mode",
            "season_to_date",
            "--catalog",
            str(catalog_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["json"]["context"]["weeks"] == {"REG": [1, 2, 3, 4, 5]}
    assert captured["json"]["context"]["filterMatch"]["game.season"]["eq"] == 2023
    parquet = tmp_path / "player_data" / "NFL" / "2023" / "week_05" / "passing_basic_s2d.parquet"
    assert parquet.is_file(), f"expected {parquet} to exist"


def test_cli_discover_replace_flag_discards_existing_catalog(monkeypatch, tmp_path):
    from sportstradamus.collectors import transport as client_mod

    catalog_path = tmp_path / "catalog.json"
    # Seed catalog with a legacy entry that --replace should evict.
    save_catalog(
        [
            EndpointSpec(
                name="legacy_thing",
                url="https://old.example/",
                method="POST",
                output_subdir="legacy/thing",
            )
        ],
        catalog_path,
    )

    def fake_request(method, url, headers=None, params=None, json=None):
        return FakeResponse(200, body=_registry_sample())

    monkeypatch.setattr(client_mod.requests, "request", fake_request)
    monkeypatch.setenv("FANTASYPOINTS_AUTHORIZATION", "Bearer test")
    monkeypatch.setenv("FANTASYPOINTS_COOKIE", "c=1")
    runner = CliRunner()
    result = runner.invoke(fp_fetch, ["discover", "--catalog", str(catalog_path), "--replace"])
    assert result.exit_code == 0, result.output
    names = [s.name for s in load_catalog(catalog_path)]
    assert "legacy_thing" not in names, "legacy entry must be evicted on --replace"
    assert "player_passing_basic" in names


# ---------------------------------------------------------------------------
# verify subcommand + verify_spec / verify_catalog
# ---------------------------------------------------------------------------


def _write_fake_parquet(
    monkeypatch,
    tmp_path,
    *,
    spec_name: str = "player_passing_basic",
    season: int = 2023,
    week: int = 5,
    mode: str = "weekly",
    rows: list[dict] | None = None,
):
    """Write a parquet at the path verify_spec will look at, return (spec, path)."""
    _redirect_parquet_dirs(monkeypatch, tmp_path)
    spec = _spec(
        name=spec_name,
        url=f"https://data.fantasypoints.com/v2/ds/nfl/tools/player/{spec_name}/values",
        output_subdir="player/passing_basic",
    )
    path = parquet_path_for_spec(spec, season=season, week=week, mode=mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows or [{"gameSeason": season, "gameWeek": week, "playerId": 1, "v": 1.0}])
    df.to_parquet(path, index=False)
    return spec, path


def test_verify_spec_passes_when_data_matches(monkeypatch, tmp_path):
    from sportstradamus.collectors.fantasypoints.verify import verify_spec

    spec, _ = _write_fake_parquet(monkeypatch, tmp_path)
    assert verify_spec(spec, season=2023, week=5) == []


def test_verify_spec_flags_missing_file(monkeypatch, tmp_path):
    from sportstradamus.collectors.fantasypoints.verify import verify_spec

    _redirect_parquet_dirs(monkeypatch, tmp_path)
    spec = _spec(name="player_passing_basic", output_subdir="player/passing_basic")
    issues = verify_spec(spec, season=2023, week=5)
    assert len(issues) == 1
    assert issues[0].code == "file_missing"
    assert issues[0].severity == "error"


def test_verify_spec_flags_wrong_week_in_data(monkeypatch, tmp_path):
    """The original bug: parquet at week_05 path but rows are gameWeek=18."""
    from sportstradamus.collectors.fantasypoints.verify import verify_spec

    spec, _ = _write_fake_parquet(
        monkeypatch,
        tmp_path,
        rows=[{"gameSeason": 2023, "gameWeek": 18, "playerId": 1}],
    )
    issues = verify_spec(spec, season=2023, week=5, mode="weekly")
    codes = [i.code for i in issues]
    assert "week_mismatch" in codes
    assert any(i.severity == "error" for i in issues)


def test_verify_spec_s2d_accepts_week_range(monkeypatch, tmp_path):
    from sportstradamus.collectors.fantasypoints.verify import verify_spec

    spec, _ = _write_fake_parquet(
        monkeypatch,
        tmp_path,
        mode="season_to_date",
        rows=[{"gameSeason": 2023, "gameWeek": w, "playerId": 1} for w in (1, 2, 3, 4, 5)],
    )
    assert verify_spec(spec, season=2023, week=5, mode="season_to_date") == []


def test_verify_spec_s2d_flags_week_beyond_requested(monkeypatch, tmp_path):
    from sportstradamus.collectors.fantasypoints.verify import verify_spec

    spec, _ = _write_fake_parquet(
        monkeypatch,
        tmp_path,
        mode="season_to_date",
        rows=[{"gameSeason": 2023, "gameWeek": w, "playerId": 1} for w in (1, 2, 6)],
    )
    issues = verify_spec(spec, season=2023, week=5, mode="season_to_date")
    assert any(i.code == "week_mismatch" for i in issues)


def test_verify_spec_flags_wrong_season(monkeypatch, tmp_path):
    from sportstradamus.collectors.fantasypoints.verify import verify_spec

    spec, _ = _write_fake_parquet(
        monkeypatch,
        tmp_path,
        rows=[{"gameSeason": 2022, "gameWeek": 5, "playerId": 1}],
    )
    issues = verify_spec(spec, season=2023, week=5)
    assert any(i.code == "season_mismatch" for i in issues)


def test_verify_spec_warns_on_empty_parquet(monkeypatch, tmp_path):
    from sportstradamus.collectors.fantasypoints.verify import verify_spec

    _redirect_parquet_dirs(monkeypatch, tmp_path)
    spec = _spec(name="player_passing_basic", output_subdir="player/passing_basic")
    path = parquet_path_for_spec(spec, season=2023, week=5, mode="weekly")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame().to_parquet(path, index=False)
    issues = verify_spec(spec, season=2023, week=5)
    assert len(issues) == 1
    assert issues[0].code == "file_empty"
    assert issues[0].severity == "warn"


def test_verify_spec_flags_postseason_mode_seeing_regular_game_type(monkeypatch, tmp_path):
    from sportstradamus.collectors.fantasypoints.verify import verify_spec

    spec, _ = _write_fake_parquet(
        monkeypatch,
        tmp_path,
        week=1,
        mode="postseason",
        rows=[{"gameSeason": 2023, "gameWeek": 1, "gameType": "REG", "playerId": 1}],
    )
    issues = verify_spec(spec, season=2023, week=1, mode="postseason")
    assert any(i.code == "game_type_not_postseason" for i in issues)


def test_verify_catalog_returns_dict_keyed_by_name(monkeypatch, tmp_path):
    from sportstradamus.collectors.fantasypoints.verify import verify_catalog

    spec_ok, _ = _write_fake_parquet(monkeypatch, tmp_path)
    spec_missing = _spec(name="player_other", output_subdir="player/other")
    results = verify_catalog([spec_ok, spec_missing], season=2023, week=5)
    assert set(results.keys()) == {"player_passing_basic", "player_other"}
    assert results["player_passing_basic"] == []
    assert any(i.code == "file_missing" for i in results["player_other"])


def test_cli_verify_reports_per_spec_status(monkeypatch, tmp_path):
    spec, _ = _write_fake_parquet(monkeypatch, tmp_path)
    catalog_path = tmp_path / "catalog.json"
    save_catalog([spec], catalog_path)
    runner = CliRunner()
    result = runner.invoke(
        fp_fetch,
        ["verify", "--season", "2023", "--week", "5", "--catalog", str(catalog_path)],
    )
    assert result.exit_code == 0, result.output
    assert "OK   player_passing_basic" in result.output
    assert "1 ok" in result.output


def test_cli_verify_exits_nonzero_on_error(monkeypatch, tmp_path):
    spec, _ = _write_fake_parquet(
        monkeypatch,
        tmp_path,
        rows=[{"gameSeason": 2023, "gameWeek": 18, "playerId": 1}],
    )
    catalog_path = tmp_path / "catalog.json"
    save_catalog([spec], catalog_path)
    runner = CliRunner()
    result = runner.invoke(
        fp_fetch,
        ["verify", "--season", "2023", "--week", "5", "--catalog", str(catalog_path)],
    )
    assert result.exit_code != 0
    assert "FAIL" in result.output
    assert "week_mismatch" in result.output
