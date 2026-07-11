"""Runtime fetch-with-auth-retry and CLI echo helpers for a source.

Split out of :mod:`collectors.cli` so the command-group builder stays about
wiring while the per-spec dispatch (with the interactive 401 refresh) and the
dry-run / verify echo formatting live together. Every function here
duck-types ``source`` (a :class:`collectors.cli.Source`) — no import of it, so
``cli`` imports ``dispatch`` and not the reverse.
"""

from __future__ import annotations

import sys

import click

from sportstradamus.collectors.auth import extract_auth_updates, update_keys
from sportstradamus.collectors.catalog import EndpointSpec
from sportstradamus.collectors.report import build_url, exc_to_err_dict, existing_parquet_rows
from sportstradamus.collectors.transport import CollectorAuthError

# One initial attempt plus one retry after an interactive auth-refresh. Cron
# runs never see the second attempt (stdin is not a TTY), so the effective
# retry count in production is 1.
_AUTH_MAX_ATTEMPTS = 2

# Width of the separator bar printed on auth-refresh prompts — purely
# cosmetic, but extracted so it doesn't look like a meaningful threshold.
_AUTH_PROMPT_SEPARATOR_WIDTH = 60


def dispatch_capturing_errors(source, client, spec, *, season, week, mode, use_cache, log):
    """Return ``(body, None)`` or ``(None, err_dict)`` for one spec.

    On 401/403 in a TTY, prompts for a fresh curl and retries once. In
    non-TTY contexts (cron) the auth failure is returned as an error so the
    caller can record it and continue.
    """
    for attempt in range(1, _AUTH_MAX_ATTEMPTS + 1):
        try:
            body = source.dispatch(
                client, spec, season=season, week=week, mode=mode, use_cache=use_cache
            )
            return body, None
        except CollectorAuthError as exc:
            log.warning(
                "auth error", extra={"endpoint": spec.name, "attempt": attempt, "error": str(exc)}
            )
            if attempt == _AUTH_MAX_ATTEMPTS or not _refresh_auth_interactively(source, client):
                return None, exc_to_err_dict(exc)
        except Exception as exc:
            log.error("fetch failed", extra={"endpoint": spec.name, "error": str(exc)})
            return None, exc_to_err_dict(exc)
    return None, {"error_class": "Unknown", "error_message": "retries exhausted"}


def _refresh_auth_interactively(source, client) -> bool:
    """Prompt for a fresh curl on stdin; update keys + client in place.

    Returns ``False`` (without prompting) when stdin is not a TTY, so cron
    runs fail fast instead of hanging on input that will never arrive.
    """
    if not sys.stdin.isatty():
        return False
    click.echo("", err=True)
    click.echo("=" * _AUTH_PROMPT_SEPARATOR_WIDTH, err=True)
    click.echo("Authorization expired.", err=True)
    click.echo("Paste a fresh DevTools curl below, then send EOF (Ctrl+D):", err=True)
    click.echo("=" * _AUTH_PROMPT_SEPARATOR_WIDTH, err=True)
    try:
        curl_text = sys.stdin.read()
    except KeyboardInterrupt:
        click.echo("Aborted.", err=True)
        return False
    if not curl_text.strip():
        click.echo("Empty input. Aborting.", err=True)
        return False
    try:
        updates = extract_auth_updates(curl_text, source.auth_fields)
    except ValueError as exc:
        click.echo(f"Failed to parse curl: {exc}", err=True)
        return False
    if not source.auth_fields.authorization or not updates.get(source.auth_fields.authorization):
        click.echo("No Authorization header found in pasted curl.", err=True)
        return False
    update_keys(updates)
    client.refresh_credentials(
        authorization=updates.get(source.auth_fields.authorization),
        cookie=updates.get(source.auth_fields.cookie) if source.auth_fields.cookie else None,
        user_agent=updates.get(source.auth_fields.user_agent)
        if source.auth_fields.user_agent
        else None,
    )
    click.echo("Auth refreshed. Resuming...", err=True)
    return True


def echo_dry_run(specs, source, *, season, week, mode, refetch) -> None:
    """Print resolved URL + parquet path (and a [SKIP] marker) per spec."""
    for spec in specs:
        url = build_url(spec.url, spec.render_params(season=season, week=week))
        try:
            path = source.path_for(spec, season=season, week=week, mode=mode)
        except ValueError as exc:
            click.echo(f"{spec.name}: {spec.method} {url} -> <unrouted: {exc}>")
            continue
        skip_marker = "  [SKIP]" if not refetch and existing_parquet_rows(path) > 0 else ""
        click.echo(f"{spec.name}: {spec.method} {url} -> {path}{skip_marker}")


def echo_verify(results: dict[str, list], *, season, week, mode) -> None:
    """Print per-spec OK/WARN/FAIL lines; raise if any spec had an error issue."""
    ok = warn = fail = 0
    for name, issues in results.items():
        if not issues:
            ok += 1
            click.echo(f"OK   {name}")
            continue
        has_error = any(i.severity == "error" for i in issues)
        prefix = "FAIL" if has_error else "WARN"
        if has_error:
            fail += 1
        else:
            warn += 1
        click.echo(f"{prefix} {name}")
        for issue in issues:
            click.echo(f"     [{issue.severity}/{issue.code}] {issue.message}")
            click.echo(f"        at {issue.path}")
    click.echo("")
    click.echo(
        f"Summary: {ok} ok, {warn} warn, {fail} fail (season={season}, week={week}, mode={mode})."
    )
    if fail:
        raise click.ClickException(f"{fail} spec(s) failed verification.")


def would_skip(source, spec: EndpointSpec, *, season, week, mode, refetch) -> bool:
    """Pre-check used by backfill to skip the pre-call pause for cached cells."""
    if refetch:
        return False
    try:
        target = source.path_for(spec, season=season, week=week, mode=mode)
    except ValueError:
        return False
    return existing_parquet_rows(target) > 0


def filter_by_name(specs: list[EndpointSpec], names: tuple[str, ...]) -> list[EndpointSpec]:
    wanted = set(names)
    filtered = [s for s in specs if s.name in wanted]
    missing = wanted - {s.name for s in filtered}
    if missing:
        raise click.ClickException(f"Unknown endpoint name(s): {sorted(missing)}")
    return filtered
