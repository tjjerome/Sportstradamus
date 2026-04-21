# Sportstradamus Style Guide

This guide is the single source of truth for code style in this repository.
Future human developers and Claude instances should read it once and apply
its rules without rederiving them. Keep it short. Update it when conventions
change.

---

## 1. Precedence

When rules collide, higher-numbered instructions win over lower-numbered ones:

1. `CLAUDE.md` — project-specific rules and domain notes.
2. This style guide — code conventions.
3. `ruff` / `mypy` configuration in `pyproject.toml` — mechanical enforcement.
4. The Python defaults (PEP 8, PEP 257, PEP 484, PEP 20).

If this guide disagrees with `CLAUDE.md`, `CLAUDE.md` wins. If it disagrees
with `ruff`, fix whichever is wrong and make them agree.

---

## 2. Standards We Adopt

We deliberately pull from these sources. They are cited rather than copied
wholesale.

- **PEP 8** — layout, whitespace, naming. Enforced via `ruff`.
- **PEP 257** — docstring conventions. Enforced via `ruff` (`D` rules).
- **PEP 484 / 604** — type hints. Advisory `mypy` check; annotations
  required on public APIs (see §7).
- **PEP 20 (Zen of Python)** — readability beats cleverness; flat beats
  nested; explicit beats implicit.
- **Google Python Style Guide** — docstring *format* (Args/Returns/Raises
  blocks). Used because it renders well in both source and Sphinx.
- **Refactoring (Fowler)** — named techniques: Extract Function,
  Extract Module, Introduce Parameter Object, Replace Conditional with
  Polymorphism.
- **Clean Code (Martin)** — selectively: small functions, named constants
  instead of magic numbers, avoid flag arguments. *Not adopted:* the book's
  class-proliferation bias, which is a poor fit for numeric Python.
- **The Elements of Python Style** — prose rules for docstrings and
  comments: short, concrete, no filler.

---

## 3. Formatting

- **Formatter:** `ruff format`. Run it before committing; pre-commit enforces.
- **Line length:** 100 characters.
- **Quotes:** double, except to avoid escaping.
- **Indentation:** 4 spaces. No tabs.
- **Trailing commas:** on multi-line collections and signatures.
- **Blank lines:** 2 between top-level definitions, 1 between methods,
  0 inside functions unless separating logical sections.

---

## 4. Imports

- Three groups, separated by a blank line, each alphabetized:
  stdlib / third-party / first-party (`sportstradamus.*`). `ruff` sorts
  this via the `I` ruleset.
- No wildcard imports.
- Prefer module-qualified access (`import numpy as np` then `np.array(...)`)
  over importing many names from a module, unless the names are used many
  times or the source is a type or constant.
- Relative imports inside a package are fine (`from .base import Stats`).

---

## 5. Naming

- Modules and packages: `snake_case`.
- Classes and type aliases: `PascalCase`.
- Functions, methods, variables: `snake_case`.
- Module-level constants: `UPPER_SNAKE_CASE`.
- Private (module- or class-internal): leading underscore (`_helper`,
  `_CACHE`).
- Never use single-letter names in public APIs. Single letters are only
  acceptable inside a short block when they mirror a math formula (`mu`,
  `sigma`, `x`, `y`) and that formula is cited in a comment.
- Don't use `l`, `O`, or `I` as names (visually confusable with digits).
- Method names describe what the method *does* from the caller's view
  (`get_training_matrix`), not how it works internally.

---

## 6. Docstrings

- **Every module** in `src/sportstradamus/` has a module-level docstring:
  one line describing the purpose, then a paragraph on any non-obvious
  behavior. Scripts in `src/sportstradamus/scripts/` are exempt.
- **Every public function and class** has a docstring. A "public" symbol
  is one not prefixed with `_`.
- **Private helpers** (`_foo`) get a docstring only when the *why* is
  non-obvious. Single-line helpers do not need one.
- **Format:** Google-style sections, in order: one-line summary, optional
  elaboration paragraph, `Args:`, `Returns:`, `Raises:`. Skip any section
  that does not apply.

```python
def fused_loc(model_loc, book_loc, model_weight, dist):
    """Fuse a model-predicted location with a bookmaker-implied location.

    Uses a logarithmic opinion pool for count distributions (NegBin) and a
    precision-weighted blend for continuous distributions (Gamma, SkewNormal).
    See CLAUDE.md "Training Report Diagnostics" for the mathematical
    justification.

    Args:
        model_loc: Location parameter predicted by the LightGBMLSS model.
        book_loc: Location parameter implied by the bookmaker line.
        model_weight: Blend weight in [0.05, 0.9]. 0 = fully bookmaker,
            1 = fully model.
        dist: One of "NegBin", "Gamma", "SkewNormal". Raises ValueError
            otherwise.

    Returns:
        The fused location parameter, same shape as `model_loc`.

    Raises:
        ValueError: If `dist` is not a supported distribution name.
    """
```

- Reference domain terms in docstrings so readers can look them up in the
  glossary (§12) without grepping the codebase.
- Math in docstrings: use plain ASCII or Unicode mathematical operators.
  LaTeX is overkill.

---

## 7. Type Hints

- **Required** on every public function and class method signature,
  including the return type.
- **Required** on any internal function whose argument or return shape
  isn't obvious from a one-line read. If you hesitate to annotate, annotate.
- **Not required** on tiny private helpers (`_clamp(x)`) or on trivially
  typed variables.
- Prefer PEP 604 unions (`str | None`) over `Optional[str]` on 3.11+.
- Where a `dict` or `list` has a stable shape, promote it to `TypedDict`,
  `dataclass`, or `NamedTuple`. Opaque `dict[str, Any]` is a last resort
  and should be called out in a comment.
- `mypy` runs advisory in CI (`follow_imports = silent`); it catches the
  subset of types we annotate without demanding wall-to-wall coverage.

---

## 8. Comments

- Explain **why**, not **what**. Well-named code already says what.
- Good triggers for a comment: a hidden constraint, an invariant, a
  workaround for a specific bug, a non-obvious reason for a value.
- Avoid comments that restate the line below them.
- Do not leave `# TODO` without a name, date, or issue reference. Anonymous
  TODOs decay into permanent lies.
- Do not comment out code. Delete it; `git log` keeps the history.

**Magic numbers.** A numeric literal is "magic" if it encodes a policy
decision (a threshold, a rate, a cap, an API page size). Promote those to
module-level constants with a one-line comment naming the decision:

```python
# Confidence cutoff from CLAUDE.md "Performance Table" — below this,
# filter picks as too uncertain to publish.
MIN_CONFIDENCE: float = 0.54
```

Bare math constants like `0.5`, `1`, or `2 * math.pi` do not need
extraction.

---

## 9. Functions

- **Target length:** ≤ 60 logical lines. Hard suggestion ~120.
- **Deep nesting** (> 4 levels of control flow) is a refactor signal,
  not a feature. Flatten with early returns or extracted helpers.
- **One responsibility per function.** If the docstring reads "loads data,
  fits a model, writes the report", that's three functions.
- **Avoid boolean flag arguments** that switch behavior. Prefer two
  functions, or an enum, or polymorphism.
- Default arguments must be immutable. Never `def f(x=[])`.

---

## 10. Error Handling

- Validate at system boundaries: HTTP responses, file I/O, user input.
  Inside the package, trust your own types.
- Don't catch-and-ignore. If you catch `Exception`, log it and re-raise or
  return a sentinel that the caller will check.
- Let loader failures at module import raise — silent fallbacks hide bugs.
- Raise specific exception types (`ValueError`, `KeyError`,
  `FileNotFoundError`). Custom exceptions only when the caller needs to
  distinguish them.

---

## 11. Dead Code

- Delete unused code in the commit where it becomes unused. `git log` is the
  history.
- If code is temporarily dark but intended to return, move it to
  `src/deprecated/` with the archive-header comment (see `src/deprecated/README.md`)
  and add a TODO to `README.md`. Do not leave it commented out in place.
- Stale imports (a module imports `foo` but never uses it) are dead code.
  `ruff` catches them via the `F401` rule.

---

## 12. Domain Glossary

A paragraph each for the terms that appear everywhere. Skim this before
grepping the codebase for meaning.

- **Offer.** A single row from a sportsbook: player, market, line, odds.
  Produced by `books.py` scrapers and `moneylines.get_props`.
- **Market.** A betting category for a player's performance in a single
  game (e.g., "NBA: points", "NFL: receiving yards"). Each trained model
  corresponds to one market.
- **Line.** The numeric threshold the bookmaker sets for a market. Bettors
  bet over or under this number.
- **Book (or sportsbook).** A source of odds: DraftKings, FanDuel, Pinnacle,
  Caesars, PrizePicks, Underdog, Sleeper, ParlayPlay. Reliability weights
  per book live in `book_weights.json`.
- **Archive.** The `Archive` class in `helpers.py` (moving to
  `helpers/archive.py`). A klepto HDF wrapper that persists odds history to
  disk keyed by `(date, league, market, player)`. Methods: `get_line`,
  `get_ev`, `write`, `clip`.
- **Stats.** The `Stats` ABC in `stats.py` (moving to `stats/base.py`) and
  its league subclasses (`StatsNBA`, `StatsMLB`, `StatsNFL`, `StatsNHL`,
  `StatsWNBA`). Responsible for loading league game logs, computing player
  features, and producing training matrices.
- **Gamelog.** A per-league DataFrame of every game a player has played,
  keyed by `(season, game_id, player_id)`. Feature engineering rolls over
  windows of the gamelog.
- **LightGBMLSS.** The distributional-regression wrapper around LightGBM.
  Predicts a full probability distribution over the outcome, not just a
  point estimate. Distribution type per stat is set in `stat_dist.json`.
- **Comp features.** "Player comparables": a feature set built by finding
  k nearest neighbors in a z-scored profile space and aggregating their
  historical outcomes. Weights are optimized in
  `scripts/optimize_comp_weights.py`.
- **fused_loc.** A helper (currently in `helpers.py`, moving to
  `helpers/distributions.py`) that blends the model's predicted location
  with the bookmaker-implied location using a distribution-specific rule.
  See CLAUDE.md "DIAG — Model Blending & Calibration" for the math.
- **DIAG.** Diagnostic sections in `training_report.txt` written by
  `train.py:report`. Each section name is prefixed with `DIAG —` in the
  report.
- **Meditate / Prophecize / Confer / Reflect / Dashboard.** The five CLI
  entry points, wired in `pyproject.toml`:
  - `meditate` trains models (`train.py`).
  - `prophecize` scores offers and exports to Google Sheets
    (`sportstradamus.py`).
  - `confer` fetches current odds (`moneylines.py`).
  - `reflect` analyzes parlay performance (`nightly.py`).
  - `dashboard` serves the Streamlit UI (`dashboard.py`).

---

## 13. For Claude and Other LLM Contributors

These rules exist because LLM edits are paid for per token. Violating them
makes the refactor expensive.

- **Read this guide once per session.** After that, cite sections by number
  instead of re-reading.
- **Prefer `Edit` over `Write`.** `Write` sends the entire new file;
  `Edit` sends only the diff. Use `Write` only for genuinely new files
  or complete rewrites.
- **Use `ruff format` and `ruff check --fix` before manually editing style.**
  Mechanical fixes cost nothing; hand-editing the same issues costs
  thousands of tokens.
- **Refactor one module per session.** Don't carry full context for more
  than one file at a time. Commit and start fresh.
- **Consult the glossary (§12) before grepping.** The term you're looking
  up is probably there.
- **Preserve public APIs during splits.** When you split a module into a
  package, re-export the old names from `__init__.py`. Callers don't change;
  no cross-codebase grep-and-update needed.
- **Dispatch parallel subagents for independent work.** Per-league stats
  subclasses, per-book scrapers, etc. Each subagent gets one file.
- **Subagent prompts should name this guide by path**
  (`docs/STYLE_GUIDE.md`), not transmit its body. Subagents can read it
  themselves.
- **Do not speculatively abstract.** Three similar lines of code are better
  than a premature abstraction. Extract only after the third concrete reuse.

---

## 14. Enforcement

These tools run in CI and locally via `pre-commit`:

| Tool | Scope | Blocking? |
|---|---|---|
| `ruff check` | `src/sportstradamus/`, `tests/`, `src/sportstradamus/scripts/` | yes |
| `ruff format --check` | same | yes |
| `pytest tests/golden/` | golden CLI snapshot tests | yes |
| `mypy` | `src/sportstradamus/` | advisory — warnings only |

Configuration lives in `pyproject.toml` (`[tool.ruff]`, `[tool.mypy]`) and
`.pre-commit-config.yaml`. To set up locally:

```bash
poetry install
poetry run pre-commit install
```

When `ruff` flags something that feels wrong, fix this guide or the rule
set — don't ignore the rule inline without a `# noqa:` comment that cites a
reason.
