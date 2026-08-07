## What

<!-- One or two sentences: what does this change do, and why? -->

## Checklist

- [ ] `poetry run ruff check src/sportstradamus/` passes
- [ ] `poetry run pytest tests/golden/` passes
- [ ] `poetry run pytest -m integration -n0` passes (fake mode, no network)
- [ ] Follows [docs/STYLE_GUIDE.md](../docs/STYLE_GUIDE.md); no new files over ~300 lines
- [ ] CLI `--help` snapshots regenerated if flags changed (`REGENERATE_SNAPSHOTS=1 poetry run pytest tests/golden/test_cli_help.py`)
- [ ] Docs updated where behavior changed (one canonical home per fact)
