# Security Policy

## Reporting a vulnerability

Please report vulnerabilities privately via
[GitHub Security Advisories](https://github.com/tjjerome/Sportstradamus/security/advisories/new).
Do not open a public issue for security problems. You should receive a response
within a week.

## Supported versions

Only the latest state of the `main` branch is supported. There are no security
backports to tagged releases.

## Credential handling

All API keys live in `src/sportstradamus/creds/keys.json`, which is ignored by
git at every level (a nested `.gitignore` plus a root-level `creds/` rule).
Never commit credentials; CI runs against placeholder stubs in
`tests/ci_fixtures/creds/`. Credentials that appeared in early repository
history have been rotated and are no longer valid.

## Model artifacts are trusted input

Trained models are Python pickles under `src/sportstradamus/data/models/`.
Loading a pickle executes arbitrary code by design, so the pipeline only ever
loads artifacts it produced itself on the same machine. Never download and load
a model file from an untrusted source, and treat any request to do so as
hostile.

## SQL construction

All values reaching the DuckDB archive go through `?` parameter binding.
Identifier interpolation is restricted to allowlisted column names
(`helpers/archive.py`); `ATTACH` statements, which DuckDB cannot parameterize,
quote-escape their single path argument (`scripts/merge_archives.py`).

## Dashboard exposure

The Streamlit dashboard has no built-in authentication. Run it on localhost or
behind a private network layer (e.g. a tailnet); never expose it directly to
the public internet.
