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

## 2. Core Principles

These principles set the default posture for every change. Later sections
spell out the mechanics; this section is the spirit.

1. **Preserve behavior unless the task explicitly requests a behavior change.**
   Refactors that change outputs are not refactors.
2. **Prefer clarity over micro-optimization.** Readability is the
   long-running cost driver in this codebase.
3. **Make data flow and assumptions explicit.** Build named intermediate
   variables for multi-step transformations rather than burying logic in
   deeply nested expressions.
4. **Keep numerically or logically sensitive areas stable and
   well-documented.** Model training, calibration, EV blending, and the
   Archive schema are this category — see §6 docstring rules for the
   extra fields they need.
5. **Add tests before or alongside structural refactors.** Characterization
   tests come first, then the seam, then the next seam. See §14.
6. **Prefer the simplest correct implementation over architectural
   cleverness.** Three similar lines beat a premature abstraction. Extract
   on the third concrete reuse, not the first.
7. **Do not add hidden control logic** (fallbacks, retries, line searches,
   damping, secondary loops) unless explicitly requested. Silent fallbacks
   are how training reports start lying.
8. **Keep top-level orchestrators flat and readable.** The CLI entry
   points (`meditate`, `prophecize`, `confer`, `reflect`) should read like
   a numbered list of workflow steps. Do not hide a core step behind
   extra indirection just to shorten the orchestrator — the orchestrator
   *is* the workflow. Extract a helper when a step has its own internal
   branching, not to flatten the call count.

---

## 3. Standards We Adopt

We deliberately pull from these sources. They are cited rather than copied
wholesale.

- **PEP 8** — layout, whitespace, naming. Enforced via `ruff`.
- **PEP 257** — docstring conventions. Enforced via `ruff` (`D` rules).
- **PEP 484 / 604** — type hints. Advisory `mypy` check; annotations
  required on public APIs (see §8).
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

## 4. Formatting

- **Formatter:** `ruff format`. Run it before committing; pre-commit enforces.
- **Line length:** 100 characters.
- **Quotes:** double, except to avoid escaping.
- **Indentation:** 4 spaces. No tabs.
- **Trailing commas:** on multi-line collections and signatures.
- **Blank lines:** 2 between top-level definitions, 1 between methods,
  0 inside functions unless separating logical sections.
- Follow the existing style in each file touched (indentation, spacing,
  quote style) when it disagrees with the above only for local consistency
  within a tight block.

---

## 5. Imports

- Three groups, separated by a blank line, each alphabetized:
  stdlib / third-party / first-party (`sportstradamus.*`). `ruff` sorts
  this via the `I` ruleset.
- No wildcard imports.
- Prefer module-qualified access (`import numpy as np` then `np.array(...)`)
  over importing many names from a module, unless the names are used many
  times or the source is a type or constant.
- Relative imports inside a package are fine (`from .base import Stats`).
- Remove unused imports in the same commit that orphans them. `ruff`'s
  `F401` rule catches the obvious cases.

---

## 6. Naming

- Modules and packages: `snake_case`.
- Classes and type aliases: `PascalCase`.
- Functions, methods, variables: `snake_case`.
- Module-level constants: `UPPER_SNAKE_CASE`.
- Private (module- or class-internal): leading underscore (`_helper`,
  `_CACHE`).
- Use descriptive, purpose-revealing names. A reader landing in the middle
  of a function should understand what each name means without scrolling.
- Never use single-letter names in public APIs. Single letters are only
  acceptable inside a short block when they mirror a math formula (`mu`,
  `sigma`, `x`, `y`) and that formula is cited in a comment.
- Don't use `l`, `O`, or `I` as names (visually confusable with digits).
- Method names describe what the method *does* from the caller's view
  (`get_training_matrix`), not how it works internally.
- Avoid numbering-based names (`step1`, `phase_2`, `eq4`) in identifiers
  unless they cite a published equation or the user explicitly asks for them.

---

## 7. Docstrings

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

For numerical, statistical, or physics-facing functions (model training,
calibration, EV blending, distribution fusion), also document:

- The state-vector or data convention (column order, key names, shapes).
- Units for inputs and outputs (probabilities, log-odds, dollars, EV cents).
- Any numerical assumption or sensitivity (zero-inflation regime,
  clip ranges, shape ceilings).

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
  glossary (§17) without grepping the codebase.
- Math in docstrings: use plain ASCII or Unicode mathematical operators.
  LaTeX is overkill.

---

## 8. Type Hints

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

## 9. Comments

- Explain **why**, not **what**. Well-named code already says what.
- Good triggers for a comment: a hidden constraint, an invariant, a
  workaround for a specific bug, a non-obvious reason for a value, a
  numerical heuristic, a domain or physical assumption, a sequencing
  dependency between two lines that otherwise look independent.
- Avoid comments that restate the line below them or narrate straightforward
  assignments and simple loops.
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

## 10. Functions

- **Target length:** ≤ 60 logical lines. Hard suggestion ~120.
- **One clear purpose per function.** If the docstring reads "loads data,
  fits a model, writes the report", that's three functions. Extract helpers
  when branching becomes hard to follow.
- **Build explicit intermediate variables** for multi-step transformations.
  A named local that holds the result of one step makes the next step
  reviewable without re-deriving the expression.
- **Use early returns or guards** for parameter validation and impossible-
  state shortcuts.
- **Deep nesting** (> 4 levels of control flow) is a refactor signal,
  not a feature. Flatten with early returns or extracted helpers.
- **Keep top-level orchestrators flat.** A CLI entry point should read as
  a sequence of named steps. Don't hide a workflow step behind an extra
  indirection layer just to shorten the orchestrator — that layer makes
  the workflow harder to follow, not easier. Extract a helper when the
  step has its own logic worth naming; inline it when it does not.
- **Avoid boolean flag arguments** that switch behavior. Prefer two
  functions, or an enum, or polymorphism.
- Default arguments must be immutable. Never `def f(x=[])`.

---

## 11. Error Handling

- Validate at system boundaries: HTTP responses, file I/O, user input.
  Inside the package, trust your own types.
- Don't catch-and-ignore. If you catch `Exception`, log it and re-raise or
  return a sentinel that the caller will check.
- Let loader failures at module import raise — silent fallbacks hide bugs.
- Raise specific exception types (`ValueError`, `KeyError`,
  `FileNotFoundError`). Custom exceptions only when the caller needs to
  distinguish them.
- Do **not** add error handling for scenarios that cannot happen. Trust
  internal invariants. Boundary code validates; interior code does not
  re-validate.

---

## 12. Dead Code

- Delete unused code in the commit where it becomes unused. `git log` is the
  history.
- If code is temporarily dark but intended to return, move it to
  `src/deprecated/` with the archive-header comment (see `src/deprecated/README.md`)
  and add a TODO to `README.md`. Do not leave it commented out in place.
- Stale imports (a module imports `foo` but never uses it) are dead code.
  `ruff` catches them via the `F401` rule.
- Dead helper layers — a function that only forwards to one other function
  with no added clarity — count as dead code. Inline them.

---

## 13. Dependencies

- Don't add a new dependency unless it pulls real weight. A 30-line helper
  is cheaper than a new transitive tree.
- Pin via Poetry (`pyproject.toml` + `poetry.lock`). Loose constraints in
  the lockfile defeat the point of having one.
- PyTorch is CPU-only and pulled from a custom Poetry source (see
  `CLAUDE.md`). Don't replace that source without weighing the install-time
  impact on the production server.

---

## 14. Testing Expectations

- Run the full quality-gate suite from the repository root before claiming
  any task is done:
  - `poetry run ruff check src/sportstradamus/`
  - `poetry run pytest tests/golden/`
  - `poetry run pytest -m integration` (fake-mode, no network)
- Add characterization tests for current behavior before deep structural
  edits, especially around the training pipeline, Archive schema, and
  Sheets export.
- Keep all existing tests passing after changes. A flaky test is a real
  test until proven otherwise.
- Add tests for any new helper that affects outputs, contracts, or schemas.
- Regenerate CLI help snapshots only when a flag change is intentional:
  `REGENERATE_SNAPSHOTS=1 poetry run pytest tests/golden/test_cli_help.py`.

---

## 15. Refactor Workflow

The order matters. Skipping a step is how you ship a "behavior-preserving"
refactor that quietly changes a metric.

1. **Capture baseline behavior.** Run the full test suite and, where
   relevant, record the script outputs you intend to keep stable
   (training report numbers, golden-test snapshots, Sheets payload).
2. **Add or extend tests** around weakly covered behavior touching the
   seam.
3. **Refactor one seam at a time.** One module per session per CLAUDE.md.
4. **Re-run tests after each seam.** If something drifted, fix it before
   moving on.
5. **Review for contract drift** before committing: signatures, key names,
   schemas, output formats, file paths, Sheet column order.

---

## 16. Documentation Updates

Update documentation in the same commit that changes the contract or
workflow it describes. At minimum, consider:

- `README.md` — contributor-facing pointers and quickstart commands.
- `CONTRIBUTING.md` — package map, where to add a league or market.
- `CLAUDE.md` — diagnostic schemas, deployment notes, training report
  fields. The Training Report Diagnostics section in particular must stay
  in sync with `training/report.py`.
- This guide (`docs/STYLE_GUIDE.md`) — when conventions change.
- Tests — when contracts are clarified, freeze the new contract in a test.

---

## 17. Domain Glossary

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
- **Archive.** The `Archive` singleton in `helpers/archive.py`. A DuckDB
  store at `archive/archive.duckdb` with `odds(league, market, game_date,
  entity, book, ev)` and `lines(league, market, game_date, entity, line)`
  tables. Writes are buffered in memory and flushed on `Archive().write()`.
  Read methods: `get_ev`, `get_line`, `get_moneyline`, `get_total`,
  `to_pandas`.
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

## 18. For Claude and Other LLM Contributors

These rules exist because LLM edits are paid for per token, and because
silent behavior changes from an automated contributor are harder to catch
than the same change from a human. Violating them makes the refactor
expensive *and* risky.

**Posture and scope:**

1. **Assume no behavior changes by default.** If your patch alters an
   output, a key name, a Sheet column, or a metric value, that is a
   feature change. Either it was requested or you should stop and ask.
2. **Preserve existing public interfaces, output keys, and file schemas.**
   When you split a module into a package, re-export the old names from
   `__init__.py` until callers are migrated. Don't break the world to
   tidy a name.
3. **Prefer small, reviewable patches.** One module per session, per
   CLAUDE.md. Commit and start fresh.
4. **Do not add features, refactors, or "improvements" beyond what was
   asked.** Speculative cleanup is how scope grows quietly.
5. **Do not add error handling for scenarios that cannot happen.** See §11.
6. **Avoid speculative performance optimizations** unless requested and
   validated against a baseline.
7. **Remove or avoid dead helper layers** that do not improve clarity or
   correctness. A one-line pass-through is not a helper.
8. **Avoid backwards-compatibility bloat.** Trace changes through the
   codebase and fix function signatures where necessary. If a change
   forces regeneration of a large artifact (model pickle, training
   matrix, archive snapshot), call it out explicitly in the patch
   description.
9. **If uncertain about numerical or behavioral impact, stop at structural
   cleanup and call out the risk explicitly.** Don't guess at the training
   pipeline.

**Workflow:**

10. **Read this guide once per session.** After that, cite sections by
    number instead of re-reading.
11. **Prefer `Edit` over `Write`.** `Write` sends the entire new file;
    `Edit` sends only the diff. Use `Write` only for genuinely new files
    or complete rewrites.
12. **Use `ruff format` and `ruff check --fix` before manually editing
    style.** Mechanical fixes cost nothing; hand-editing the same issues
    costs thousands of tokens.
13. **Add docstrings and only meaningful inline comments — do not pad.**
    See §7 and §9.
14. **Run or explicitly request validation** before claiming completion.
    The three quality gates from §14 must pass; if you can't run them in
    your environment, say so and ask the user to.
15. **Consult the glossary (§17) before grepping.** The term you're
    looking up is probably there.
16. **Dispatch parallel subagents for independent work.** Per-league stats
    subclasses, per-book scrapers, etc. Each subagent gets one file.
17. **Subagent prompts should name this guide by path**
    (`docs/STYLE_GUIDE.md`), not transmit its body. Subagents can read it
    themselves.
18. **Do not speculatively abstract.** Three similar lines of code are
    better than a premature abstraction. Extract only after the third
    concrete reuse.

---

## 19. Enforcement

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
