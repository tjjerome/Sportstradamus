# Changelog

Notable changes, newest first. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [SemVer](https://semver.org/). Detail lives in git history.

## [Unreleased]

### Security
- Request timeouts on every production HTTP call; ScrapeOps header fetch over TLS.
- Dashboard HTML output escapes API-origin strings (stored-XSS hardening).
- Column-name allowlist ahead of DuckDB identifier interpolation.

### Changed
- `poetry.lock` now committed; `networkx` and `sympy` declared as real dependencies; abandoned `importlib` backport dropped.
- CI installs from the lock (no more on-the-fly pyproject rewrite) and runs the integration suite serially.

### Added
- MIT license, security policy, code of conduct, issue/PR templates, Dependabot config.

## [4.2.0] - 2026-08-06

Baseline for the public changelog: five-league prediction pipeline (NBA, WNBA, MLB, NHL, NFL),
LightGBMLSS distributional models with a six-gate offline ship harness, DuckDB odds archive,
Underdog/Sleeper pick'em strategy tooling, and the Streamlit dashboard.
