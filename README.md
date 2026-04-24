# Sportstradamus

Python package that scrapes sportsbook odds and player stats, trains
distributional ML models that predict player performance, and exports value
recommendations to Google Sheets. Supports MLB, NBA, NFL, NHL, and WNBA.

## Developer Setup

```bash
poetry install
poetry run pre-commit install
```

`pre-commit install` wires `ruff` (lint + format) into the local commit hook.
The same checks run in CI on every push. See [docs/STYLE_GUIDE.md](docs/STYLE_GUIDE.md)
for the conventions enforced.

## CLI Entry Points

| Command | Purpose |
|---|---|
| `poetry run prophecize` | Score offers against trained models, export to Google Sheets |
| `poetry run confer` | Fetch current odds/props from sportsbooks |
| `poetry run meditate` | Train or retrain ML models (`--force`, `--league` filters) |
| `poetry run reflect` | Analyze historical parlay performance |
| `poetry run dashboard` | Streamlit dashboard (replaces `reflect` for interactive review) |

See [CLAUDE.md](CLAUDE.md) for the architectural overview.

## Deferred / Archived Code

The following modules were moved to [`src/deprecated/`](src/deprecated/) during
the 2026-04-21 maintainability refactor because they had no caller in any CLI
entry point or `scripts/` module. They are preserved verbatim under
`src/deprecated/` (with a header comment recording the original path and the
last live git SHA) and should be reintroduced if the corresponding feature
returns. See [`src/deprecated/README.md`](src/deprecated/README.md) for the
header protocol and reintroduction process.

- [ ] **TODO: reimplement Kelly bet sizing** (`opt_kelley_bet.py`) — stake
      optimization on +EV picks. Needs integration with `prophecize`'s sheet
      export or a new `kelly` subcommand.
- [ ] **TODO: reimplement parlay search** (`unused_funcs.py::find_bets`,
      `opt_parlay.py`) — combinatorial search over +EV legs. May have been
      replaced by in-line parlay logic in `sportstradamus.py`; decide whether
      to delete or rewire.
- [ ] **TODO: reimplement BettingPros NFL ingest** (`get_lines.py`) —
      redundant with `books.py` scrapers, but a useful fallback.
- [ ] **TODO: reimplement team correlation generator** (`correlation.py`) —
      produces `{LEAGUE}_corr.csv` consumed by `sportstradamus.py`'s parlay
      logic. Those CSVs are currently stale; a script or CLI is needed.
- [ ] **TODO: reimplement LightGBM feature-importance plot**
      (`see_features.py`).
- [ ] **TODO: reimplement testing utilities** (`test.py`) — ad-hoc
      experimentation harness, not pytest tests. Decide whether to convert to
      proper tests or delete.
- [ ] **TODO: decide fate of orphaned `helpers.py` math utilities**
      (`prob_diff`, `prob_sum`, `accel_asc`, `get_active_sports`) — preserved
      in [`src/deprecated/helpers_orphans.py`](src/deprecated/helpers_orphans.py).
- [ ] **TODO: orphan methods** (`Archive.add`, `Archive.clip`, `Archive.merge`,
      `Archive.rename_market`, `Scrape.get_proxy`, `Scrape.post`) — preserved
      in [`src/deprecated/helpers_orphans.py`](src/deprecated/helpers_orphans.py)
      as de-methodized top-level functions. `Archive.add` looks like the
      intended write path for the `confer` pipeline but was never wired in;
      decide whether to wire it or delete.
- [ ] **TODO: deprecated sportsbook scrapers** (`get_dk`, `get_fd`,
      `get_pinnacle`, `get_caesars`, `get_thrive`, `get_pp`, `get_parp`)
      preserved in [`src/deprecated/books_deprecated.py`](src/deprecated/books_deprecated.py).
      Superseded by `moneylines.get_props`. Reintroduce only if direct-book
      scraping becomes preferable to the odds aggregator. The remaining live
      scrapers (`get_ud` for Underdog, `get_sleeper` for Sleeper) stay in
      `books.py`.
