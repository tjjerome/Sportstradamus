# Contributing to Sportstradamus

Thanks for your interest in improving Sportstradamus! This guide covers setting
up a development environment, running the test suites, and getting a pull
request merged.

For a map of the codebase — packages, data flow, how to add a league or
market — read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). For install and
runtime usage (API keys, CLI commands, cron), see [README.md](README.md).

Participation in this project is governed by the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Prerequisites

- **Python 3.11** (exact — PyTorch is pinned to a CPU wheel that requires it)
- **Poetry** ≥ 1.7 — [install guide](https://python-poetry.org/docs/#installation)
- **Git**

## Setup

```bash
git clone https://github.com/tjjerome/sportstradamus.git
cd Sportstradamus
poetry install
poetry run pre-commit install   # required once — wires ruff + the smoke test into commits
```

`poetry install` pulls PyTorch CPU-only from a custom source; the first install
downloads ~1–2 GB.

### Credential and config stubs

Live scraping needs real API keys (see README §API Keys and Credentials), but
**running the test suites does not**. A few config files are git-ignored
because they hold secrets or learned outputs, and `helpers/config.py` reads
them at import time — so a fresh clone needs the same stubs CI stages:

```bash
mkdir -p src/sportstradamus/creds
cp tests/ci_fixtures/creds/keys.json src/sportstradamus/creds/keys.json
cp tests/ci_fixtures/data/*.json src/sportstradamus/data/config/
```

The creds stub is just empty strings:

```json
{"odds_api": "", "odds_api_plus": "", "scrapingfish": "", "scrapeops": "",
 "fantasypoints_authorization": "", "fantasypoints_cookie": "",
 "fantasypoints_user_agent": ""}
```

`creds/` is git-ignored — real keys never enter the repo.

## Quality gates

All three must pass before a PR is ready:

```bash
poetry run ruff check src/sportstradamus/   # lint
poetry run pytest tests/golden/             # snapshot + regression suite (parallel)
poetry run pytest -m integration -n0        # end-to-end smoke, fake mode, no network
```

The integration suite runs `confer → meditate → prophecize` against cached
fixtures with all external APIs stubbed. It is not xdist-safe, hence `-n0`.

The test tree, for orientation:

- `tests/` root — behavioral unit tests for model and pipeline code
  (distributions, calibration, feature engineering, schema round-trips, …)
- `tests/golden/` — CLI `--help` snapshots plus regression pins on config,
  imports, and dashboard invariants
- `tests/integration/` — the end-to-end smoke test

If you intentionally change a CLI flag or add a command, regenerate the
affected snapshot and commit the new fixture with your change:

```bash
REGENERATE_SNAPSHOTS=1 poetry run pytest tests/golden/test_cli_help.py
```

## Style

Code conventions live in [docs/STYLE_GUIDE.md](docs/STYLE_GUIDE.md) — read it
once and cite sections by number (`§N`) in reviews. The posture in one line:
**less code, written for a human to maintain** — no wrappers that only forward
a call, no fallbacks for cases that can't happen, no comments that narrate the
code.

Mechanical enforcement (`ruff format` + `ruff check --fix`, line length 100,
Google docstrings) is configured in `pyproject.toml` and runs via the
pre-commit hook, so a hook-clean commit is already format-compliant.

## Dependencies

```bash
poetry add <package>                # runtime dependency
poetry add --group dev <package>    # dev-only
```

PyTorch must stay CPU-only (the `pytorch_cpu` source in `pyproject.toml`). Do
not change the `torch` pin without verifying the new wheel exists in that
source.

## Pull requests

- **Target the `main` branch.**
- Keep PRs small and focused — one fix or feature per PR. Split unrelated
  changes.
- Work on a topic branch; give it a descriptive name.
- The PR checklist mirrors
  [.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md): the
  three quality gates pass, the style guide is followed (no new files over
  ~300 lines), snapshots are regenerated if CLI flags changed, and docs are
  updated where behavior changed (one canonical home per fact).
- For substantial changes — a new league, a new distribution family, a
  training-pipeline rework — open an issue first to discuss the approach.

Model-quality claims need evidence: if your change affects trained-model
behavior, include the relevant before/after rows from
`data/training/model_stats.csv` (or a scorecard A/B run) in the PR
description.

## License

Sportstradamus is [MIT-licensed](LICENSE). By contributing, you agree that
your contributions are licensed under the same terms.
