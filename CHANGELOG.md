# Changelog

Notable changes, newest first. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [SemVer](https://semver.org/). Detail lives in git history.

## [Unreleased]

### Changed
- Parlay history is now one parquet per game date (`data/runtime/parlay_hist/`);
  prophecize and reflect rewrite only the days they touch instead of the whole
  multi-million-row file (43% of a warm prophecize run, line-profiled). The old
  single-file store self-migrates on the first run; parlays now share the
  history parquet's 365-day retention.

## [4.2.0] - 2026-08-09

First tagged release. Five-league prediction pipeline (NBA, WNBA, MLB, NHL, NFL),
LightGBMLSS distributional models with a six-gate offline ship harness, DuckDB odds
archive, Underdog/Sleeper pick'em strategy tooling, and the Streamlit dashboard.
Git history was rewritten prior to this release to remove credential files committed
in 2023; the affected keys were rotated.

### Security
- Request timeouts on every production HTTP call; ScrapeOps header fetch over TLS.
- Dashboard HTML output escapes API-origin strings (stored-XSS hardening).
- Column-name allowlist ahead of DuckDB identifier interpolation.

### Changed
- **All 19 flat console scripts replaced by one `sportstradamus` umbrella command**
  (`prophecize`/`confer`/`meditate`/`reflect`/`dashboard` top-level; `bet`, `fetch`,
  `ship`, `admin` groups). Cron dispatches via `python -m sportstradamus`; job tokens,
  logs, and healthchecks are unchanged. `meditate --help` now shows only the
  production flags — research axes are hidden (documented in docs/MODEL_LIFECYCLE.md).
- `--legacy-correlation` escape hatch removed (one-cycle window closed).
- `poetry.lock` now committed; `networkx` and `sympy` declared as real dependencies; abandoned `importlib` backport dropped.
- CI installs from the lock (no more on-the-fly pyproject rewrite) and runs the integration suite serially.
- Golden suite pruned of characterization pins (35 files); the rotted root-level
  suite revived and folded into the default run (~60 s wall clock on a dev box).

### Added
- MIT license, security policy, code of conduct, issue/PR templates, Dependabot config.

