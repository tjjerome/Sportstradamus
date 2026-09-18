"""Unit tests for the fantasypoints package — client, catalog, CLI."""

from __future__ import annotations

import json
from datetime import date, timedelta
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
from sportstradamus.collectors.fantasypoints import source as fp_source
from sportstradamus.collectors.fantasypoints.cli import fp_fetch
from sportstradamus.collectors.fantasypoints.import_curl import parse_curl_to_spec
from sportstradamus.collectors.fantasypoints.source import FP_SOURCE
from sportstradamus.collectors.fantasypoints.transform import (
    parquet_path_for_spec,
    parse_table_response,
    write_parquet,
)
from sportstradamus.collectors.transport import (
    REQUEST_TIMEOUT_S,
    CollectorAuthError,
    CollectorDecodeError,
    CookieClient,
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
        referer="https://fantasypointsdata.com/",
        origin="https://fantasypointsdata.com",
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
    monkeypatch.setenv("FANTASYPOINTS_COOKIE", "from_env=1")
    captured = {}

    def capture(method, url, headers=None, params=None, json=None, timeout=None):
        captured["headers"] = headers
        return FakeResponse(200, body={})

    _patch_request(monkeypatch, capture)
    FP_SOURCE.client(inter_request_sleep_s=0.0).get("https://example/")
    assert captured["headers"]["Cookie"] == "from_env=1"


def test_client_sends_no_authorization_header(monkeypatch):
    """The session cookie is the only credential the current API accepts."""
    monkeypatch.setenv("FANTASYPOINTS_COOKIE", "ds_session=abc")
    monkeypatch.setenv("FANTASYPOINTS_AUTHORIZATION", "Bearer stale")
    captured = {}

    def capture(method, url, headers=None, params=None, json=None, timeout=None):
        captured["headers"] = headers
        return FakeResponse(200, body={})

    _patch_request(monkeypatch, capture)
    FP_SOURCE.client(inter_request_sleep_s=0.0).get("https://example/")
    assert "Authorization" not in captured["headers"]


def test_client_post_sends_json_body(monkeypatch):
    captured = {}

    def capture(method, url, headers=None, params=None, json=None, timeout=None):
        captured["method"] = method
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse(200, body={"ok": True})

    _patch_request(monkeypatch, capture)
    body = {"context": {"grouping": "team"}, "useCache": True}
    out = _client().post("https://example/", json_body=body)
    assert out == {"ok": True}
    assert captured["method"] == "POST"
    assert captured["json"] == body
    assert captured["timeout"] == REQUEST_TIMEOUT_S


def test_client_strips_accept_encoding_from_request_headers(monkeypatch):
    captured = {}

    def capture(method, url, headers=None, params=None, json=None, timeout=None):
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
    def respond_with_garbage(method, url, headers=None, params=None, json=None, timeout=None):
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
        referer="https://fantasypointsdata.com/",
        origin="https://fantasypointsdata.com",
        inter_request_sleep_s=0.0,
    )
    with pytest.raises(CollectorAuthError, match="empty"):
        client.get("https://example/")
    assert called["n"] == 0, "no HTTP request should have been attempted"


# ---------------------------------------------------------------------------
# session renewal: the cookie is short-lived, so fp-fetch mints its own
# ---------------------------------------------------------------------------


class _FakeLoginResponse:
    def __init__(self, status_code=200, cookies=None, payload=None, reason="OK"):
        self.status_code = status_code
        self.ok = status_code < 400
        self.cookies = cookies or {}
        self.reason = reason
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _patch_login(monkeypatch, response, keys=None):
    """Point session.renew_session at a fake login; capture the keys.json write."""
    from sportstradamus.collectors.fantasypoints import session as session_mod

    sent = {}
    monkeypatch.setattr(
        session_mod,
        "load_keys",
        lambda: (
            keys
            if keys is not None
            else {"fantasypoints_username": "trevor", "fantasypoints_password": "hunter2"}
        ),
    )
    monkeypatch.setattr(session_mod, "update_keys", lambda updates: sent.update(updates))
    monkeypatch.delenv("FANTASYPOINTS_USERNAME", raising=False)
    monkeypatch.delenv("FANTASYPOINTS_PASSWORD", raising=False)

    def fake_post(url, json=None, headers=None, timeout=None):
        sent["url"] = url
        sent["body"] = json
        return response

    monkeypatch.setattr(session_mod.requests, "post", fake_post)
    return sent


def test_renew_session_logs_in_and_persists_the_cookie(monkeypatch):
    from sportstradamus.collectors.fantasypoints.session import renew_session

    sent = _patch_login(monkeypatch, _FakeLoginResponse(cookies={"ds_session": "fresh.token"}))
    assert renew_session() == "ds_session=fresh.token"
    assert sent["url"].endswith("/api/auth/login")
    # The sign-in form posts the account name, not an email field.
    assert sent["body"] == {"name": "trevor", "password": "hunter2"}
    # Written back under the same slot a pasted cookie uses, so nothing
    # downstream can tell a renewed session from a hand-captured one.
    assert sent["fantasypoints_cookie"] == "ds_session=fresh.token"


def test_renew_session_without_stored_credentials_names_the_keys(monkeypatch):
    from sportstradamus.collectors.fantasypoints.session import (
        SessionRenewalError,
        renew_session,
    )

    _patch_login(monkeypatch, _FakeLoginResponse(), keys={})
    with pytest.raises(SessionRenewalError, match="fantasypoints_username"):
        renew_session()


def test_renew_session_surfaces_the_api_error_on_a_rejected_login(monkeypatch):
    from sportstradamus.collectors.fantasypoints.session import (
        SessionRenewalError,
        renew_session,
    )

    _patch_login(
        monkeypatch,
        _FakeLoginResponse(status_code=404, payload={"error": "Not found."}),
    )
    with pytest.raises(SessionRenewalError, match="Not found"):
        renew_session()


def test_renew_session_rejects_a_login_that_sets_no_cookie(monkeypatch):
    """A 200 with no Set-Cookie means the login contract moved, not that we are in."""
    from sportstradamus.collectors.fantasypoints.session import (
        SessionRenewalError,
        renew_session,
    )

    _patch_login(monkeypatch, _FakeLoginResponse(payload={"name": "trevor"}))
    with pytest.raises(SessionRenewalError, match="ds_session"):
        renew_session()


def test_cli_run_renews_the_session_on_401_without_a_tty(monkeypatch, tmp_path):
    """The cron case: a lapsed cookie must recover on its own, mid-batch."""
    import sys as sys_mod

    from sportstradamus.collectors import transport as client_mod
    from sportstradamus.collectors.fantasypoints.source import FP_SOURCE

    _redirect_parquet_dirs(monkeypatch, tmp_path)
    catalog_path = tmp_path / "catalog.json"
    save_catalog(
        [
            EndpointSpec(
                name="team_coverage_matrix",
                url="https://fantasypointsdata.com/api/nfl/coverage-matrix",
                params={"mode": "offense", "seasons": "{season}", "regWeeks": "{week}"},
                output_subdir="team/coverage_matrix",
            ),
        ],
        catalog_path,
    )
    monkeypatch.setenv("FANTASYPOINTS_COOKIE", "ds_session=expired")
    monkeypatch.setattr(client_mod, "_INTER_REQUEST_SLEEP_S", 0.0)
    monkeypatch.setattr(sys_mod.stdin, "isatty", lambda: False)
    renewals = {"n": 0}

    def fake_renew():
        renewals["n"] += 1
        return "ds_session=fresh"

    monkeypatch.setattr(FP_SOURCE, "renew_auth", fake_renew)
    seen_cookies = []

    def respond(method, url, headers=None, params=None, json=None, timeout=None):
        seen_cookies.append(headers.get("Cookie"))
        if headers.get("Cookie") == "ds_session=expired":
            return FakeResponse(401)
        return FakeResponse(200, body=[{"team": "BLT", "games": 1}])

    monkeypatch.setattr(client_mod.requests, "request", respond)
    runner = CliRunner()
    result = runner.invoke(
        fp_fetch,
        ["run", "--season", "2025", "--week", "5", "--catalog", str(catalog_path)],
    )
    assert result.exit_code == 0, result.output
    assert renewals["n"] == 1
    assert seen_cookies == ["ds_session=expired", "ds_session=fresh"]
    parquet = tmp_path / "team_data" / "NFL" / "2025" / "week_05" / "coverage_matrix.parquet"
    assert parquet.is_file()


def test_parse_curl_get_strips_auth_headers_and_splits_query():
    curl_text = (
        "curl 'https://fantasypointsdata.com/api/nfl/coverage-matrix"
        "?mode=defense&seasons=2026&regWeeks=1' "
        "-H 'Cookie: ds_session=def' "
        "-H 'User-Agent: Mozilla/5.0' "
        "-H 'Accept: application/json' "
        "--compressed"
    )
    spec = parse_curl_to_spec(
        curl_text,
        name="opponent_coverage_matrix",
        output_subdir="opponent/coverage_matrix",
    )
    assert spec.method == "GET"
    assert spec.url == "https://fantasypointsdata.com/api/nfl/coverage-matrix"
    # The filter that defines the entry is kept; the captured period is not.
    assert spec.params == {"mode": "defense", "seasons": "{season}", "regWeeks": "{week}"}
    assert spec.extra_headers == {"Accept": "application/json"}
    assert spec.json_body is None


def test_parse_curl_drops_the_browser_noise_the_new_site_sends():
    """A real "Copy as cURL" off the data app drags a dozen useless headers along.

    ``Referer`` is the one that bites: it records whichever page happened to
    be open, so two captures of the same tool would produce two different
    catalog entries — and the client sets its own Referer anyway.
    """
    curl_text = (
        "curl 'https://fantasypointsdata.com/api/nfl/rushing?positions=RB' "
        "-H 'User-Agent: Mozilla/5.0' "
        "-H 'Accept: */*' "
        "-H 'Accept-Language: en-US,en;q=0.9' "
        "-H 'Accept-Encoding: gzip, deflate, br, zstd' "
        "-H 'Referer: https://fantasypointsdata.com/team/playcallers?mode=offense' "
        "-H 'Connection: keep-alive' "
        "-H 'Cookie: ds_session=secret' "
        "-H 'Sec-Fetch-Dest: empty' -H 'Sec-Fetch-Mode: cors' -H 'Sec-Fetch-Site: same-origin' "
        "-H 'DNT: 1' -H 'Sec-GPC: 1' -H 'Priority: u=4' -H 'TE: trailers'"
    )
    spec = parse_curl_to_spec(
        curl_text, name="player_rushing_advanced", output_subdir="player/rushing_advanced"
    )
    assert spec.url == "https://fantasypointsdata.com/api/nfl/rushing"
    assert spec.params == {"positions": "RB", "seasons": "{season}", "regWeeks": "{week}"}
    # Only the one header that says something about the request survives.
    assert spec.extra_headers == {"Accept": "*/*"}


def test_parse_curl_templates_the_period_even_when_the_capture_omits_it():
    """Otherwise the dry-run listing and the run report print a URL missing the week."""
    spec = parse_curl_to_spec(
        "curl 'https://fantasypointsdata.com/api/nfl/pace'",
        name="team_pace",
        output_subdir="team/pace",
    )
    assert spec.params == {"seasons": "{season}", "regWeeks": "{week}"}


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
                name="team_coverage_matrix",
                url="https://fantasypointsdata.com/api/nfl/coverage-matrix",
                params={"mode": "offense", "seasons": "{season}", "regWeeks": "{week}"},
                output_subdir="team/coverage_matrix",
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
    assert "GET" in result.output
    assert "team_coverage_matrix" in result.output
    assert "regWeeks=5" in result.output


def test_cli_run_writes_parquet_via_get(monkeypatch, tmp_path):
    from sportstradamus.collectors import transport as client_mod

    _redirect_parquet_dirs(monkeypatch, tmp_path)
    catalog_path = tmp_path / "catalog.json"
    save_catalog(
        [
            EndpointSpec(
                name="team_coverage_matrix",
                url="https://fantasypointsdata.com/api/nfl/coverage-matrix",
                params={"mode": "offense"},
                output_subdir="team/coverage_matrix",
            ),
        ],
        catalog_path,
    )
    monkeypatch.setenv("FANTASYPOINTS_COOKIE", "ds_session=test")
    monkeypatch.setattr(client_mod, "_INTER_REQUEST_SLEEP_S", 0.0)
    captured = {}

    def capture(method, url, headers=None, params=None, json=None, timeout=None):
        captured["method"] = method
        captured["json"] = json
        captured["params"] = params
        return FakeResponse(200, body=[{"team": "BLT", "games": 1}, {"team": "DEN", "games": 1}])

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
    assert captured["method"] == "GET"
    assert captured["json"] is None
    assert captured["params"] == {"mode": "offense", "seasons": "2025", "regWeeks": "5"}
    parquet = tmp_path / "team_data" / "NFL" / "2025" / "week_05" / "coverage_matrix.parquet"
    assert parquet.is_file(), f"expected parquet at {parquet}"
    df = pd.read_parquet(parquet)
    assert len(df) == 2
    assert set(df["teamAbbreviation"]) == {"BLT", "DEN"}
    assert set(df["gameWeek"]) == {5}


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
                name="team_coverage_matrix",
                url="https://fantasypointsdata.com/api/nfl/coverage-matrix",
                params={"mode": "offense"},
                output_subdir="team/coverage_matrix",
            ),
        ],
        catalog_path,
    )
    monkeypatch.setenv("FANTASYPOINTS_COOKIE", "ds_session=test")
    monkeypatch.setattr(client_mod, "_INTER_REQUEST_SLEEP_S", 0.0)
    # Plant a non-empty parquet at the cell's expected path.
    target = tmp_path / "team_data" / "NFL" / "2025" / "week_05" / "coverage_matrix.parquet"
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
                name="team_coverage_matrix",
                url="https://fantasypointsdata.com/api/nfl/coverage-matrix",
                params={"mode": "offense"},
                output_subdir="team/coverage_matrix",
            ),
        ],
        catalog_path,
    )
    monkeypatch.setenv("FANTASYPOINTS_COOKIE", "ds_session=test")
    monkeypatch.setattr(client_mod, "_INTER_REQUEST_SLEEP_S", 0.0)
    target = tmp_path / "team_data" / "NFL" / "2025" / "week_05" / "coverage_matrix.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"teamAbbreviation": ["OLD"]}).to_parquet(target, index=False)

    def respond(*_args, **_kwargs):
        return FakeResponse(200, body=[{"team": "NEW", "games": 1}])

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
                name="team_coverage_matrix",
                url="https://fantasypointsdata.com/api/nfl/coverage-matrix",
                params={"mode": "offense"},
                output_subdir="team/coverage_matrix",
            ),
        ],
        catalog_path,
    )
    monkeypatch.setenv("FANTASYPOINTS_COOKIE", "ds_session=test")
    monkeypatch.setattr(client_mod, "_INTER_REQUEST_SLEEP_S", 0.0)
    target = tmp_path / "team_data" / "NFL" / "2025" / "week_05" / "coverage_matrix.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"teamAbbreviation": pd.Series(dtype="object")}).to_parquet(target, index=False)
    assert pd.read_parquet(target).empty

    def respond(*_args, **_kwargs):
        return FakeResponse(200, body=[{"team": "BAL", "games": 1}])

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
    monkeypatch.setenv("FANTASYPOINTS_COOKIE", "ds_session=expired")
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
        "curl 'https://fantasypointsdata.com/api/nfl/passing?positions=QB' "
        "-H 'Cookie: ds_session=new-cookie-value' "
        "-H 'User-Agent: Mozilla/5.0 (X11; Linux x86_64) Test'"
    )
    runner = CliRunner()
    result = runner.invoke(
        fp_fetch,
        ["refresh-auth", str(curl_path), "--keys-path", str(keys_path)],
    )
    assert result.exit_code == 0, result.output
    updated = json.loads(keys_path.read_text())
    assert updated["fantasypoints_cookie"] == "ds_session=new-cookie-value"
    assert updated["fantasypoints_user_agent"].startswith("Mozilla/5.0 (X11")
    assert updated["odds_api"] == "PRESERVE_ME", "non-FP keys must be preserved"


def test_cli_refresh_auth_creates_keys_json_when_absent(tmp_path):
    keys_path = tmp_path / "keys.json"
    curl_path = tmp_path / "fresh.curl"
    curl_path.write_text("curl 'https://x/' -H 'Cookie: ds_session=abc'")
    runner = CliRunner()
    result = runner.invoke(
        fp_fetch,
        ["refresh-auth", str(curl_path), "--keys-path", str(keys_path)],
    )
    assert result.exit_code == 0, result.output
    assert keys_path.is_file()
    assert json.loads(keys_path.read_text())["fantasypoints_cookie"] == "ds_session=abc"


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
        input="curl 'https://x/' -H 'Cookie: ds_session=from-stdin'",
    )
    assert result.exit_code == 0, result.output
    assert json.loads(keys_path.read_text())["fantasypoints_cookie"] == "ds_session=from-stdin"


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


def test_parse_table_response_extracts_rows_and_columns():
    payload = [
        {"name": "Patrick Mahomes", "team": "KC", "yards": 312},
        {"name": "Josh Allen", "team": "BUF", "yards": 287},
    ]
    df = parse_table_response(payload)
    assert len(df) == 2
    assert list(df["yards"]) == [312, 287]


def test_parse_table_response_lifts_raw_counts_beside_the_displayed_rates():
    """The ``__raw`` sub-object holds the numerators the recipes pool on."""
    payload = [
        {
            "name": "Patrick Mahomes",
            "team": "KC",
            "dropbacks": 33,
            "__raw": {"coverage_man_dropbacks": 11, "coverage_zone_dropbacks": 21},
        }
    ]
    df = parse_table_response(payload)
    assert df["coverage_man_dropbacks"].iloc[0] == 11
    assert df["dropbacks"].iloc[0] == 33


def test_parse_table_response_displayed_value_wins_over_raw_on_name_collision():
    payload = [{"name": "A B", "team": "KC", "games": 1, "__raw": {"games": 99}}]
    assert parse_table_response(payload)["games"].iloc[0] == 1


def test_parse_table_response_synthesises_identity_columns():
    """The stats layer groups on these; a missing one silently drops every feature."""
    payload = [{"name": "Patrick Mahomes", "team": "KC", "position": "QB", "passer_id": "00-001"}]
    df = parse_table_response(payload, season=2026, week=3)
    row = df.iloc[0]
    assert row["playerPlayerId"] == "00-001"
    assert row["playerFirstName"] == "Patrick"
    assert row["playerLastName"] == "Mahomes"
    assert row["playerPosition"] == "QB"
    assert row["teamTeamId"] == "KC"
    assert row["teamAbbreviation"] == "KC"
    assert row["gameSeason"] == 2026
    assert row["gameWeek"] == 3


def test_parse_table_response_splits_single_word_name_without_raising():
    df = parse_table_response([{"name": "Ogletree", "team": "IND"}])
    assert df["playerFirstName"].iloc[0] == "Ogletree"
    assert df["playerLastName"].iloc[0] == ""


def test_parse_table_response_serialises_nested_columns_to_json():
    payload = [
        {"name": "A B", "team": "CIN", "opponentsPlayed": [{"abbr": "CIN"}, {"abbr": "DET"}]}
    ]
    df = parse_table_response(payload)
    assert df["opponentsPlayed"].iloc[0] == '[{"abbr":"CIN"},{"abbr":"DET"}]'


def test_parse_table_response_empty_on_non_list_payload():
    """An error page or expired session yields some other shape, never a row list."""
    assert parse_table_response([]).empty
    assert parse_table_response({}).empty
    assert parse_table_response({"content": {"rows": {"values": [{"id": 1}]}}}).empty
    assert parse_table_response("<html>login</html>").empty


def _spec(name="x", url="https://example/", output_subdir="x/x") -> EndpointSpec:
    return EndpointSpec(name=name, url=url, output_subdir=output_subdir)


# ---------------------------------------------------------------------------
# column-map translation: new schema -> the legacy vocabulary the recipes read
# ---------------------------------------------------------------------------


def test_column_map_copies_to_the_legacy_name_and_applies_the_scale():
    """Legacy stores rates as fractions and sack yardage as a loss; the API does neither."""
    spec = _spec(name="player_passing_advanced", output_subdir="player/passing_advanced")
    payload = [
        {
            "name": "Patrick Mahomes",
            "team": "KC",
            "passer_id": "00-0033873",
            "yards": 291,
            "cpoe": 4.5,
            "sack_yards": 16,
        }
    ]
    row = parse_table_response(payload, spec=spec).iloc[0]
    assert row["playerStatsPassingYardsTotal"] == 291
    assert row["playerStatsPassingCompletionsOverExpected"] == pytest.approx(0.045)
    assert row["playerStatsPassingSackedYardsLost"] == pytest.approx(-16.0)


def test_column_map_leaves_the_new_schema_columns_in_place():
    """One parquet has to serve both the frozen feature set and any later work."""
    spec = _spec(name="player_passing_advanced", output_subdir="player/passing_advanced")
    df = parse_table_response([{"name": "A B", "team": "KC", "yards": 291}], spec=spec)
    assert df["yards"].iloc[0] == 291


def test_column_map_derives_a_legacy_rate_from_its_raw_pair():
    """Deriving from the counts is exact where the API's displayed rate is pre-rounded."""
    spec = _spec(name="player_wr_coverage_matchup", output_subdir="player/wr_coverage_matchup")
    payload = [{"name": "A B", "team": "KC", "__raw": {"man_routes": 8, "man_rec_yards": 20}}]
    row = parse_table_response(payload, spec=spec).iloc[0]
    assert row["playerStatsCoverageSchemeManReceivingRoutesTotal"] == 8
    assert row["playerStatsCoverageSchemeManReceivingYardsPerRoute"] == pytest.approx(2.5)


def test_column_map_leaves_a_derived_rate_null_on_a_zero_denominator():
    spec = _spec(name="player_wr_coverage_matchup", output_subdir="player/wr_coverage_matchup")
    payload = [{"name": "A B", "team": "KC", "__raw": {"man_routes": 0, "man_rec_yards": 0}}]
    row = parse_table_response(payload, spec=spec).iloc[0]
    assert pd.isna(row["playerStatsCoverageSchemeManReceivingYardsPerRoute"])


def test_column_map_renests_flat_bucket_columns_into_the_legacy_cell():
    """The five bucket-parsing aggregators must read new pulls and archived ones alike."""
    spec = _spec(
        name="player_receiving_separation_by_routes",
        output_subdir="player/receiving_separation_by_routes",
    )
    payload = [
        {
            "name": "A B",
            "team": "KC",
            "__raw": {"go_routes": 4, "go_sep_sum": 10, "flat_routes": 0, "flat_sep_sum": 0},
        }
    ]
    cell = json.loads(parse_table_response(payload, spec=spec)["bucket"].iloc[0])
    assert cell["bucketReceivingSeparationRouteGo"] == {
        "playerStatsReceivingSeparationRoutesTotal": 4,
        "playerStatsReceivingSeparationScorePercentage": 2.5,
    }
    # A route the player never ran contributes no bucket rather than a zero one.
    assert "bucketReceivingSeparationRouteFlat" not in cell


def test_column_map_skips_a_translation_whose_source_column_is_absent():
    """A tool that stops publishing a split loses that column, never gets a wrong one."""
    spec = _spec(name="player_wr_coverage_matchup", output_subdir="player/wr_coverage_matchup")
    df = parse_table_response([{"name": "A B", "team": "KC", "games": 1}], spec=spec)
    assert "playerStatsCoverageSchemeManReceivingYardsPerRoute" not in df.columns


def test_untranslated_when_no_spec_is_given():
    """``import-curl``'s preview wants the raw shape, not the legacy vocabulary."""
    df = parse_table_response([{"name": "A B", "team": "KC", "yards": 291}])
    assert "playerStatsPassingYardsTotal" not in df.columns


def _legacy_columns_by_file_kind() -> dict[str, dict[str, set[str]]]:
    """Map ``{grain: {parquet basename: legacy column names the map produces}}``.

    Walks the shipped catalog rather than the column map directly, so the
    whole chain is covered: a catalog entry that routes somewhere unexpected
    or resolves to no map entry shows up as an empty column set.
    """
    from sportstradamus.collectors.fantasypoints import column_map
    from sportstradamus.collectors.fantasypoints import transform as transform_mod

    by_kind: dict[str, dict[str, set[str]]] = {"player": {}, "team": {}}
    for spec in load_catalog(FP_SOURCE.catalog_path):
        path = parquet_path_for_spec(spec, season=2026, week=1)
        grain = "player" if transform_mod.PLAYER_DATA_BASE in path.parents else "team"
        key = column_map.map_key(*transform_mod._route_spec(spec))
        entry = column_map.load_column_map().get(key, {})
        legacy = set(entry.get("rename", {}).values()) | set(entry.get("derive", {}))
        if entry.get("bucket"):
            legacy.add("bucket")
        by_kind[grain][path.stem] = legacy
    return by_kind


# ``…RushingYardsBeforeContactTotal`` was never in the legacy snapshots either
# — only the per-attempt form was — so these two aggregate columns have been
# empty in production since before the port. The new API does publish the
# total, so a naive mapping would silently revive them and move the frozen
# feature count. Kept dead deliberately.
_DEAD_BEFORE_THE_PORT = frozenset({"rush_ybc_per_att", "def_rush_ybc_allowed_per_att"})


# ``line_matchups`` was retired as a leak (week N's rows carry week N's
# results) and has no catalog entry, but the kind stays in FILE_KINDS because
# archived legacy snapshots still carry the file and the team-abbreviation
# lookup falls back to it for those weeks.
_KINDS_WITH_NO_FETCHER = frozenset({"line_matchups"})


def test_every_file_kind_the_stats_layer_reads_has_a_catalog_entry():
    """A kind with no fetcher is a feature family that is silently always empty."""
    from sportstradamus.collectors.fantasypoints import transform as transform_mod
    from sportstradamus.stats import nfl_fp_team_weekly, nfl_fp_weekly

    fetched = {"player": set(), "team": set()}
    for spec in load_catalog(FP_SOURCE.catalog_path):
        path = parquet_path_for_spec(spec, season=2026, week=1)
        grain = "player" if transform_mod.PLAYER_DATA_BASE in path.parents else "team"
        fetched[grain].add(path.stem)
    unfetched = (set(nfl_fp_weekly.FILE_KINDS) - fetched["player"]) | (
        set(nfl_fp_team_weekly.FILE_KINDS) - fetched["team"]
    )
    assert unfetched == _KINDS_WITH_NO_FETCHER
    # And nothing is fetched that no loader reads.
    assert not fetched["player"] - set(nfl_fp_weekly.FILE_KINDS)
    assert not fetched["team"] - set(nfl_fp_team_weekly.FILE_KINDS)


def test_every_aggregate_output_column_survives_the_new_schema():
    """The pin against the silent-skip chain that ends in a serve-time KeyError.

    A recipe whose source columns are missing is skipped without a word, so
    its output column never appears — and the strict ``expected_columns``
    slice inside every NFL model pickle then raises at serve time. Columns
    carried by two recipes (one per era) need only one to resolve.
    """
    from sportstradamus.stats import nfl_fp_team_weekly_aggregate, nfl_fp_weekly_aggregate

    available = _legacy_columns_by_file_kind()
    unsatisfiable = {}
    for recipes, grain in (
        (nfl_fp_weekly_aggregate._AGGREGATE_RECIPES, "player"),
        (nfl_fp_team_weekly_aggregate._ALL_RECIPES, "team"),
    ):
        by_output = {}
        for recipe in recipes:
            by_output.setdefault(recipe.output_col, []).append(recipe)
        for output_col, alternatives in by_output.items():
            if any(
                all(arg in available[grain].get(r.file_kind, set()) for arg in r.args)
                for r in alternatives
            ):
                continue
            unsatisfiable[output_col] = [(r.file_kind, r.args) for r in alternatives]
    assert set(unsatisfiable) == _DEAD_BEFORE_THE_PORT, unsatisfiable


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


def test_parquet_path_falls_back_to_output_subdir_for_unprefixed_name(monkeypatch, tmp_path):
    """Hand-imported entries without the context prefix route on output_subdir."""
    from sportstradamus.collectors.fantasypoints import transform as tm

    monkeypatch.setattr(tm, "PLAYER_DATA_BASE", tmp_path / "player_data")
    monkeypatch.setattr(tm, "TEAM_DATA_BASE", tmp_path / "team_data")
    spec = _spec(
        name="coverage_matrix",
        url="https://fantasypointsdata.com/api/nfl/coverage-matrix",
        output_subdir="team/coverage_matrix",
    )
    p = parquet_path_for_spec(spec, season=2025, week=5)
    assert p == tmp_path / "team_data" / "NFL" / "2025" / "week_05" / "coverage_matrix.parquet"


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
                url="https://fantasypointsdata.com/api/nfl/passing",
                params={"seasons": "{season}", "regWeeks": "{week}"},
                output_subdir="player/passing_basic",
            ),
            EndpointSpec(
                name="team_coverage_matrix",
                url="https://fantasypointsdata.com/api/nfl/coverage-matrix",
                output_subdir="team/coverage_matrix",
            ),
            EndpointSpec(
                name="player_will_fail",
                url="https://fantasypointsdata.com/api/nfl/will-fail",
                output_subdir="player/will_fail",
            ),
        ],
        catalog_path,
    )
    monkeypatch.setenv("FANTASYPOINTS_COOKIE", "ds_session=test")
    monkeypatch.setattr(client_mod, "_INTER_REQUEST_SLEEP_S", 0.0)
    monkeypatch.setattr(client_mod, "_RETRY_BACKOFF_S", (0.0, 0.0, 0.0))

    def fake_request(method, url, headers=None, params=None, json=None, timeout=None):
        if url.endswith("/passing"):
            return FakeResponse(200, body=[{"name": "Q Player", "games": 1}])
        if url.endswith("/coverage-matrix"):
            return FakeResponse(200, body=[])
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
    assert by_name["team_coverage_matrix"]["status"] == "empty"
    assert by_name["player_will_fail"]["status"] == "fetch_failed"
    # HTTP status surfaced for the failed spec:
    assert by_name["player_will_fail"]["http_status"] == 500
    # Response preview included so I can diagnose the failure mode:
    assert "upstream is having a moment" in (by_name["player_will_fail"]["response_preview"] or "")
    # The request is all in the URL now, so that is what the report has to carry:
    assert by_name["player_passing_basic"]["request_body"] is None
    assert by_name["player_passing_basic"]["url"].endswith("/passing?seasons=2025&regWeeks=5")


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
                output_subdir="misc/thing",
            ),
            EndpointSpec(
                name="player_passing_basic",
                url="https://fantasypointsdata.com/api/nfl/passing",
                output_subdir="player/passing_basic",
            ),
        ],
        catalog_path,
    )
    monkeypatch.setenv("FANTASYPOINTS_COOKIE", "ds_session=test")
    monkeypatch.setattr(client_mod, "_INTER_REQUEST_SLEEP_S", 0.0)

    def fake_request(method, url, headers=None, params=None, json=None, timeout=None):
        return FakeResponse(200, body=[{"name": "Q Player", "games": 1}])

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
    monkeypatch.setenv("FANTASYPOINTS_COOKIE", "ds_session=expired")
    monkeypatch.setattr(client_mod, "_INTER_REQUEST_SLEEP_S", 0.0)

    call_count = {"n": 0}
    good_body = {
        "content": {"table": {"rows": {"count": 1, "values": [{"teamAbbreviation": "CIN"}]}}}
    }

    def fake_request(method, url, headers=None, params=None, json=None, timeout=None):
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
                url="https://fantasypointsdata.com/api/nfl/passing",
                params={"seasons": "{season}", "regWeeks": "{week}"},
                output_subdir="player/passing_basic",
            ),
        ],
        catalog_path,
    )
    monkeypatch.setenv("FANTASYPOINTS_COOKIE", "ds_session=test")
    monkeypatch.setattr(client_mod, "_INTER_REQUEST_SLEEP_S", 0.0)

    seen_params = []

    def fake_request(method, url, headers=None, params=None, json=None, timeout=None):
        seen_params.append(params)
        return FakeResponse(200, body=[{"name": "Q Player", "games": 1}])

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
    # And each call asked for the week it wrote:
    assert [p["regWeeks"] for p in seen_params] == ["1", "2", "3"]
    assert {p["seasons"] for p in seen_params} == {"2023"}


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
    monkeypatch.setenv("FANTASYPOINTS_COOKIE", "ds_session=test")
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
    monkeypatch.setenv("FANTASYPOINTS_COOKIE", "ds_session=test")
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
                url="https://fantasypointsdata.com/api/nfl/passing",
                params={"seasons": "{season}", "regWeeks": "{week}"},
                output_subdir="player/passing_basic",
            )
        ],
        catalog_path,
    )
    monkeypatch.setenv("FANTASYPOINTS_COOKIE", "ds_session=test")
    monkeypatch.setattr(client_mod, "_INTER_REQUEST_SLEEP_S", 0.0)
    captured = {}

    def capture(method, url, headers=None, params=None, json=None, timeout=None):
        captured["params"] = params
        return FakeResponse(200, body=[{"name": "Q Player", "games": 5}])

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
    assert captured["params"] == {"seasons": "2023", "regWeeks": "1,2,3,4,5"}
    parquet = tmp_path / "player_data" / "NFL" / "2023" / "week_05" / "passing_basic_s2d.parquet"
    assert parquet.is_file(), f"expected {parquet} to exist"


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
        url="https://fantasypointsdata.com/api/nfl/passing",
        output_subdir="player/passing_basic",
    )
    path = parquet_path_for_spec(spec, season=season, week=week, mode=mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows or [_PLAYER_ROW])
    df.to_parquet(path, index=False)
    return spec, path


_PLAYER_ROW = {
    "playerPlayerId": "00-0039150",
    "playerFirstName": "Bryce",
    "playerLastName": "Young",
    "games": 1,
    "yards": 146.0,
}


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


def test_verify_spec_flags_unfiltered_window(monkeypatch, tmp_path):
    """The original bug: parquet at week_05 path but rows aggregate the whole season.

    New-API rows carry no week column, so the tell is ``games``: a weekly
    request that returned season totals shows 17 games on one entity.
    """
    from sportstradamus.collectors.fantasypoints.verify import verify_spec

    spec, _ = _write_fake_parquet(monkeypatch, tmp_path, rows=[{**_PLAYER_ROW, "games": 17}])
    issues = verify_spec(spec, season=2023, week=5, mode="weekly")
    assert [i.code for i in issues] == ["window_too_wide"]
    assert issues[0].severity == "error"


def test_verify_spec_s2d_accepts_games_up_to_the_requested_week(monkeypatch, tmp_path):
    from sportstradamus.collectors.fantasypoints.verify import verify_spec

    spec, _ = _write_fake_parquet(
        monkeypatch,
        tmp_path,
        mode="season_to_date",
        rows=[{**_PLAYER_ROW, "games": 5}],
    )
    assert verify_spec(spec, season=2023, week=5, mode="season_to_date") == []


def test_verify_spec_s2d_flags_games_beyond_the_requested_week(monkeypatch, tmp_path):
    from sportstradamus.collectors.fantasypoints.verify import verify_spec

    spec, _ = _write_fake_parquet(
        monkeypatch,
        tmp_path,
        mode="season_to_date",
        rows=[{**_PLAYER_ROW, "games": 6}],
    )
    issues = verify_spec(spec, season=2023, week=5, mode="season_to_date")
    assert any(i.code == "window_too_wide" for i in issues)


def test_verify_spec_warns_when_the_games_column_is_absent(monkeypatch, tmp_path):
    """Without ``games`` the window is unverifiable — say so rather than pass silently."""
    from sportstradamus.collectors.fantasypoints.verify import verify_spec

    row = {k: v for k, v in _PLAYER_ROW.items() if k != "games"}
    spec, _ = _write_fake_parquet(monkeypatch, tmp_path, rows=[row])
    issues = verify_spec(spec, season=2023, week=5)
    assert [i.code for i in issues] == ["missing_games_column"]
    assert issues[0].severity == "warn"


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


def test_verify_spec_flags_missing_player_identity(monkeypatch, tmp_path):
    """A dropped identity column is indistinguishable from an empty week downstream."""
    from sportstradamus.collectors.fantasypoints.verify import verify_spec

    row = {k: v for k, v in _PLAYER_ROW.items() if k != "playerPlayerId"}
    spec, _ = _write_fake_parquet(monkeypatch, tmp_path, rows=[row])
    issues = verify_spec(spec, season=2023, week=5)
    assert [i.code for i in issues] == ["missing_identity_columns"]
    assert issues[0].severity == "error"


def test_verify_spec_requires_team_identity_on_team_specs(monkeypatch, tmp_path):
    from sportstradamus.collectors.fantasypoints.verify import verify_spec

    spec, _ = _write_fake_parquet(
        monkeypatch,
        tmp_path,
        spec_name="team_coverage_matrix",
        rows=[{"teamTeamId": "BLT", "teamAbbreviation": "BLT", "games": 1}],
    )
    assert verify_spec(spec, season=2023, week=5) == []


def test_max_games_for_mode_mirrors_period_params():
    """The window check and the request builder must agree on how wide a window is."""
    from sportstradamus.collectors.fantasypoints.source import period_params
    from sportstradamus.collectors.fantasypoints.verify import max_games_for_mode

    for mode in ("weekly", "season_to_date", "postseason"):
        params = period_params(season=2023, week=5, mode=mode)
        requested = params.get("postWeeks") or params["regWeeks"]
        assert max_games_for_mode(week=5, mode=mode) == len(requested.split(","))


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
    spec, _ = _write_fake_parquet(monkeypatch, tmp_path, rows=[{**_PLAYER_ROW, "games": 17}])
    catalog_path = tmp_path / "catalog.json"
    save_catalog([spec], catalog_path)
    runner = CliRunner()
    result = runner.invoke(
        fp_fetch,
        ["verify", "--season", "2023", "--week", "5", "--catalog", str(catalog_path)],
    )
    assert result.exit_code != 0
    assert "FAIL" in result.output
    assert "window_too_wide" in result.output


def _freeze_today(monkeypatch, today: date) -> None:
    """Pin ``date.today()`` inside the source module without patching datetime globally."""

    class _Frozen(date):
        @classmethod
        def today(cls) -> date:
            return today

    monkeypatch.setattr(fp_source, "date", _Frozen)


@pytest.mark.parametrize(
    ("today", "season", "expected"),
    [
        (date(2026, 9, 16), 2026, 1),
        (date(2026, 9, 17), 2026, 1),
        (date(2026, 9, 23), 2026, 2),
        (date(2025, 9, 10), 2025, 1),
        (date(2026, 9, 9), 2026, 1),
        (date(2027, 2, 1), 2026, 18),
    ],
)
def test_default_week_names_the_last_completed_week(monkeypatch, today, season, expected):
    _freeze_today(monkeypatch, today)
    assert fp_source._default_week(season) == expected


@pytest.mark.parametrize("season", range(2021, 2033))
def test_week_boundary_falls_on_the_tuesday_after_labor_day(monkeypatch, season):
    """Anchoring on Sep 1's first Tuesday instead drifts a week when Sep 1 is a Tuesday."""
    labor_day = date(season, 9, 1)
    while labor_day.weekday() != 0:
        labor_day += timedelta(days=1)
    anchor = labor_day + timedelta(days=1)
    _freeze_today(monkeypatch, anchor + timedelta(days=13))
    assert fp_source._default_week(season) == 1
    _freeze_today(monkeypatch, anchor + timedelta(days=14))
    assert fp_source._default_week(season) == 2


@pytest.mark.parametrize(
    ("today", "expected"),
    [
        (date(2026, 9, 17), 2026),
        (date(2026, 7, 1), 2026),
        (date(2026, 6, 30), 2025),
        (date(2027, 1, 15), 2026),
    ],
)
def test_default_season_flips_in_july(monkeypatch, today, expected):
    _freeze_today(monkeypatch, today)
    assert fp_source._default_season() == expected


def test_shipped_catalog_templates_the_period_on_every_entry():
    """``import-curl --replace`` off a live capture can bake a literal week into params.

    The period params are re-derived per request anyway, but the dry-run
    listing and the run report both print ``render_params`` output — a baked
    literal there is the difference between a URL you can paste into a
    browser and one that silently names last season.
    """
    specs = load_catalog(FP_SOURCE.catalog_path)
    baked = [
        s.name
        for s in specs
        if (s.params or {}).get("seasons") != "{season}"
        or (s.params or {}).get("regWeeks") != "{week}"
    ]
    assert baked == []


def test_shipped_catalog_sends_no_request_bodies():
    """Every tool is a GET; a leftover body would be sent to an endpoint ignoring it."""
    specs = load_catalog(FP_SOURCE.catalog_path)
    assert [s.name for s in specs if s.json_body is not None] == []
    assert {s.method.upper() for s in specs} == {"GET"}
