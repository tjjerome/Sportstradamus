"""Unit tests for the fantasypoints package — client, catalog, CLI."""

from __future__ import annotations

import json

import pytest
import requests
from click.testing import CliRunner

from sportstradamus.fantasypoints.catalog import (
    EndpointSpec,
    load_catalog,
    save_catalog,
)
from sportstradamus.fantasypoints.cli import (
    fp_fetch,
    parse_curl_to_spec,
)
from sportstradamus.fantasypoints.client import (
    FantasyPointsAuthError,
    FantasyPointsClient,
)


class FakeResponse:
    """Minimal stand-in for :class:`requests.Response`."""

    def __init__(self, status_code, body=None, text=""):
        self.status_code = status_code
        self._body = body
        self.text = text
        self.content = text.encode() if text else b""

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.HTTPError(f"{self.status_code}")
            err.response = self
            raise err


def _client() -> FantasyPointsClient:
    return FantasyPointsClient(
        authorization="Bearer test-token",
        cookie="_shopify_y=abc",
        user_agent="UA",
        inter_request_sleep_s=0.0,
    )


def _patch_request(monkeypatch, handler):
    from sportstradamus.fantasypoints import client as client_mod

    monkeypatch.setattr(client_mod.requests, "request", handler)


def test_client_raises_on_401(monkeypatch):
    _patch_request(monkeypatch, lambda *a, **k: FakeResponse(401))
    with pytest.raises(FantasyPointsAuthError, match="Authorization token"):
        _client().get("https://example/")


def test_client_raises_on_403(monkeypatch):
    _patch_request(monkeypatch, lambda *a, **k: FakeResponse(403))
    with pytest.raises(FantasyPointsAuthError):
        _client().get("https://example/")


def test_client_retries_on_429_then_succeeds(monkeypatch):
    from sportstradamus.fantasypoints import client as client_mod

    responses = [FakeResponse(429), FakeResponse(200, body={"ok": True})]
    _patch_request(monkeypatch, lambda *a, **k: responses.pop(0))
    monkeypatch.setattr(client_mod, "_RETRY_BACKOFF_S", (0.0, 0.0, 0.0))
    assert _client().get("https://example/") == {"ok": True}


def test_client_retries_on_500_then_gives_up(monkeypatch):
    from sportstradamus.fantasypoints import client as client_mod

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
    FantasyPointsClient(inter_request_sleep_s=0.0).get("https://example/")
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


def test_client_omits_authorization_when_not_set(monkeypatch):
    monkeypatch.delenv("FANTASYPOINTS_AUTHORIZATION", raising=False)
    monkeypatch.delenv("FANTASYPOINTS_COOKIE", raising=False)
    captured = {}

    def capture(method, url, headers=None, params=None, json=None):
        captured["headers"] = headers
        return FakeResponse(200, body={})

    _patch_request(monkeypatch, capture)
    FantasyPointsClient(
        authorization="",
        cookie="",
        user_agent="UA",
        inter_request_sleep_s=0.0,
    ).get("https://example/")
    assert "Authorization" not in captured["headers"]
    assert "Cookie" not in captured["headers"]


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


def test_endpoint_spec_renders_template_and_path(tmp_path):
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
    path = spec.output_path(base=tmp_path, season=2025, week=5)
    assert path == tmp_path / "2025" / "week_05" / "team" / "line_matchups.json"


def test_endpoint_spec_season_long_path(tmp_path):
    spec = EndpointSpec(
        name="season_summary",
        url="https://example/",
        output_subdir="summary",
        weekly=False,
    )
    path = spec.output_path(base=tmp_path, season=2025, week=5)
    assert path == tmp_path / "2025" / "season" / "summary.json"


def test_cli_run_dry_run_prints_method_and_url(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    save_catalog(
        [
            EndpointSpec(
                name="line_matchups",
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
            "--output",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "POST" in result.output
    assert "line_matchups" in result.output


def test_cli_run_writes_snapshot_via_post(monkeypatch, tmp_path):
    from sportstradamus.fantasypoints import client as client_mod

    catalog_path = tmp_path / "catalog.json"
    save_catalog(
        [
            EndpointSpec(
                name="line_matchups",
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
        return FakeResponse(200, body={"content": {"table": {"rows": []}}, "week": 5})

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
            "--output",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["method"] == "POST"
    assert captured["json"] == {"context": {"week": "5"}, "useCache": True}
    snapshot = tmp_path / "2025" / "week_05" / "team" / "line_matchups.json"
    assert snapshot.is_file()
    assert json.loads(snapshot.read_text())["week"] == 5


def test_cli_run_auth_error_exits_nonzero(monkeypatch, tmp_path):
    from sportstradamus.fantasypoints import client as client_mod

    catalog_path = tmp_path / "catalog.json"
    save_catalog(
        [
            EndpointSpec(
                name="line_matchups",
                url="https://example/",
                output_subdir="team/line_matchups",
            ),
        ],
        catalog_path,
    )
    monkeypatch.setenv("FANTASYPOINTS_AUTHORIZATION", "Bearer expired")
    monkeypatch.setattr(client_mod, "_INTER_REQUEST_SLEEP_S", 0.0)
    monkeypatch.setattr(client_mod.requests, "request", lambda *a, **k: FakeResponse(401))
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
            "--output",
            str(tmp_path),
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
