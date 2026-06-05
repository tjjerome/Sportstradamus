# Sportstradamus Style Guide

This guide is the single source of truth for code style in this repository.
Read it once and apply its rules without rederiving them. Cite sections by
number (`§N`) in commits, comments, and review notes. Keep it short; update it
when conventions change.

**The thesis.** This is a solo-maintained numeric Python project, and most new
code now arrives through an AI agent. The dominant failure mode is not wrong
code — the tests catch that — it is *too much* code: wrappers that forward a
call, fallbacks for cases that can't happen, comments that narrate the obvious,
abstractions built for a future that never comes. Every line is a line one
person has to read and maintain later. So the default posture of this guide is
**less code, written for a human to understand months from now.** When a rule
here and a textbook habit disagree, the rule here wins.

---

## 1. Precedence

When rules collide, higher-priority sources win:

1. `CLAUDE.md` — project-specific rules and domain notes.
2. This style guide — code conventions.
3. `ruff` / `mypy` configuration in `pyproject.toml` — mechanical enforcement.
4. The Python defaults (PEP 8, PEP 257, PEP 484, PEP 20).

If this guide disagrees with `CLAUDE.md`, `CLAUDE.md` wins. If it disagrees
with `ruff`, fix whichever is wrong and make them agree — don't paper over the
gap with a blanket `# noqa`.

---

## 2. Core Principles

These set the default posture for every change. Later sections give the
mechanics; this section is the spirit.

1. **Preserve behavior unless the task explicitly requests a behavior change.**
   A refactor that changes an output, a key, or a metric is not a refactor.
2. **Prefer clarity over micro-optimization.** Readability is the long-running
   cost driver in this codebase, not CPU cycles.
3. **Make data flow and assumptions explicit.** For a multi-step
   transformation, name the intermediate that represents a real step so the
   next step is reviewable without re-deriving it. This is *not* license to
   alias every single-use expression to a `result` / `data` temp (§10) — name
   steps, not echoes.
4. **Keep numerically or logically sensitive areas stable and well
   documented.** Model training, calibration, EV blending, distribution
   fusion, and the Archive schema are this category — they get the extra
   docstring fields in §7 and the "stop and flag the risk" treatment in §18.9.
5. **Add tests before or alongside structural refactors.** Characterization
   tests first, then the seam, then the next seam (§15).
6. **Prefer the simplest correct implementation over architectural
   cleverness (YAGNI / KISS).** Build for the task in front of you, not a
   hypothetical future one. Three similar lines beat a premature abstraction;
   extract on the *third* concrete reuse, not the first (rule of three — see
   also §18.18).
7. **Do not add hidden control logic** — fallbacks, retries, line searches,
   damping, secondary loops — unless explicitly requested. Silent fallbacks
   are how training reports start lying.
8. **Keep top-level orchestrators flat and readable.** The CLI entry points
   (`meditate`, `prophecize`, `confer`, `reflect`) should read like a numbered
   list of named workflow steps. The orchestrator *is* the workflow — don't
   hide a core step behind indirection to shorten it, and don't dump the whole
   workflow inline as one wall. Extract a step when it has its own internal
   branching worth naming, not to change the call count (§10).
9. **Reuse before you build.** Before writing a utility, grep `helpers/` and
   the relevant package for one that already exists, and use it. Prefer the
   standard library and our existing stack (`itertools`, `collections`,
   `pathlib`, vectorized `pandas` / `numpy` / `polars`) over hand-rolled or
   reinvented logic (§13). Duplicated logic is the largest measured source of
   drift in AI-assisted code — find it twice, consolidate it; don't copy it a
   third time. **Any parallelism must be justified:** per-league / per-grain
   blocks that encode the *same* knowledge get consolidated (base method, shared
   helper, or a store parameterized by the differing values); a block stays
   parallel only when it is the same shape over genuinely different knowledge
   *and* consolidating would force a banned pure-forwarder (§2.10), in which case
   it carries an explicit `# pylint: disable=duplicate-code` pragma plus a
   one-line rationale. The blocking duplicate-code gate
   (`tests/golden/test_no_duplicate_code.py`, pylint R0801 at
   `min-similarity-lines = 6`) tolerates only pragma-justified clones.
10. **Prefer deep functions and modules over shallow ones.** A function earns
    its existence by hiding real complexity behind a simple interface, not by
    forwarding a call under a new name. A coherent forty-line function that
    does one thing beats six seven-line fragments you must read together to
    follow one thought. Fewer, deeper units; shallower call stacks (§10, §12).

---

## 3. Standards We Adopt

We pull deliberately from these, cited rather than copied:

- **PEP 8** — layout, whitespace, naming. Enforced via `ruff`.
- **PEP 257** — docstring conventions. Enforced via `ruff` (`D` rules).
- **PEP 484 / 604** — type hints. Advisory `mypy`; annotations required on
  public APIs (§8).
- **PEP 20 (Zen of Python)** — readability counts; flat beats nested; explicit
  beats implicit; errors should never pass silently; there should be one
  obvious way to do it.
- **A Philosophy of Software Design (Ousterhout)** — deep modules, complexity
  is incremental, comments capture the intent the code cannot. This is the
  primary design reference for the project. **Where APOSD and Clean Code
  conflict — function size, how aggressively to decompose, the value of
  comments — APOSD wins.**
- **Google Python Style Guide** — docstring *format* only (Args/Returns/Raises
  blocks). Renders well in both source and Sphinx.
- **Refactoring (Fowler)** — named techniques: Extract Function, Extract
  Module, Introduce Parameter Object, Replace Conditional with Polymorphism.
- **Clean Code (Martin)** — selectively: intention-revealing names, named
  constants over magic numbers, no flag arguments. *Not adopted:* its
  class-proliferation bias and its tiny-function dogma, both poor fits for
  numeric Python — prefer Ousterhout's deep modules (§2.10, §10).
- **The Pragmatic Programmer** — DRY as "one authoritative representation of
  each piece of knowledge," tempered by the rule of three (§2.6).

---

## 4. Formatting

- **Formatter:** `ruff format`. Run before committing; the PostToolUse hook and
  pre-commit both enforce it (§19).
- **Line length:** 100 characters.
- **Quotes:** double, except to avoid escaping.
- **Indentation:** 4 spaces. No tabs.
- **Trailing commas:** on multi-line collections and signatures.
- **Blank lines:** 2 between top-level definitions, 1 between methods, 0 inside
  functions except to separate genuine logical sections (and if you find
  yourself separating four of them, that's four helpers — §10).
- **Match the file you are in.** When local convention in a tight block
  disagrees with the above, follow the local convention for that block.
  Internal consistency within a file beats your global preference; mixed style
  within one file is itself a machine tell.

---

## 5. Imports

- Three groups separated by a blank line, each alphabetized: stdlib /
  third-party / first-party (`sportstradamus.*`). `ruff`'s `I` ruleset sorts
  this.
- No wildcard imports.
- Prefer module-qualified access (`import numpy as np`, then `np.array(...)`)
  over importing many names, unless a name is used many times or is a type or
  constant.
- Relative imports inside a package are fine (`from .base import Stats`).
- Imports go at the top of the module. The only exception is a genuine
  lazy-import for an optional dependency (e.g. `cvxpy` / `pyyaml` / `tabulate`
  in `strategies/`, imported inside the function that needs them so core users
  need not install the `strategy` group). Don't scatter late imports to dodge a
  cycle — fix the cycle.
- Remove an unused import in the same commit that orphans it. `ruff`'s `F401`
  catches the obvious cases.

---

## 6. Naming

- Modules and packages: `snake_case`. Classes and type aliases: `PascalCase`.
  Functions, methods, variables: `snake_case`. Module-level constants:
  `UPPER_SNAKE_CASE`. Private (module- or class-internal): leading underscore.
- **Names reveal intent.** A reader landing in the middle of a function should
  understand what each name means without scrolling. The name of a function
  should let a caller skip reading its body.
- **Ban placeholder names that reveal nothing:** `data`, `result`, `info`,
  `val`, `tmp`, `obj`, `item` (outside a tight comprehension), `helper`,
  `process`, `handle`, `do_work`, `manager`, `util`. If the best name you can
  think of is generic, the thing usually does too much or you haven't decided
  what it is — that's the signal to split or rethink, not to ship the vague
  name.
- Single-letter names only inside a short block where they mirror a cited math
  formula (`mu`, `sigma`, `x`, `y`), with the formula named in a comment.
  Never `l`, `O`, or `I` (confusable with digits).
- Method names describe what the method does from the caller's view
  (`get_training_matrix`), not how it works internally.
- Avoid numbering-based names (`step1`, `phase_2`, `eq4`) unless they cite a
  published equation or the user asked for them.

---

## 7. Docstrings

- **Every module** in `src/sportstradamus/` has a module-level docstring: one
  line of purpose, then a paragraph on any non-obvious behavior. Scripts in
  `src/sportstradamus/scripts/` are exempt.
- **Every public function and class** (no leading `_`) has a docstring.
- **Private helpers** (`_foo`) get a docstring only when the *why* is
  non-obvious. A single-line helper does not need one.
- **Never write a signature-echo docstring** — a summary that just restates the
  function name and parameter names and adds nothing the signature already
  says (PEP 257). If that is all you can write, write nothing (on a private
  helper) or write the *why* (on a public one). A redundant docstring is the
  same noise as a redundant comment (§9).
- **Format:** Google-style sections, in order — one-line summary, optional
  elaboration, `Args:`, `Returns:`, `Raises:`. Skip any section that does not
  apply.

For numerical, statistical, or distribution-facing functions (model training,
calibration, EV blending, distribution fusion), also document:

- the state-vector or data convention (column order, key names, shapes);
- units for inputs and outputs (probabilities, log-odds, dollars, EV cents);
- any numerical assumption or sensitivity (zero-inflation regime, clip ranges,
  shape ceilings).

```python
def fused_loc(model_loc, book_loc, model_weight, dist):
    """Fuse a model-predicted location with a bookmaker-implied location.

    Uses a logarithmic opinion pool for count distributions (NegBin) and a
    precision-weighted blend for continuous distributions (Gamma, SkewNormal).
    See the ML Pipeline notes in CLAUDE.md and `helpers/distributions.py` for
    the mathematical justification.

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
  glossary (§17) instead of grepping.
- Math in docstrings: plain ASCII or Unicode operators. LaTeX is overkill.

---

## 8. Type Hints

- **Required** on every public function and method signature, including the
  return type.
- **Required** on any internal function whose argument or return shape isn't
  obvious from a one-line read. When you hesitate, annotate.
- **Not** on tiny private helpers (`_clamp(x)`) or trivially typed locals. An
  annotation on `count: int = 0` is noise the checker already infers — leaving
  it on every local is a machine tell. Annotate boundaries, not the obvious.
- Prefer PEP 604 unions (`str | None`) over `Optional[str]` on 3.11+.
- Where a `dict` or `list` has a stable shape, promote it to a `TypedDict`,
  `dataclass`, or `NamedTuple`. **Avoid `Any`** — it's a hole in the type
  system, not a type. Opaque `dict[str, Any]` is a last resort and must be
  called out in a comment that says why the shape can't be modeled.
- Don't build elaborate generic types where a concrete type reads fine.
- `mypy` runs advisory in CI (`follow_imports = silent`); it checks the subset
  we annotate without demanding wall-to-wall coverage.

---

## 9. Comments

Over-commenting is the single most common tell of machine-written code and the
fastest way to make a file tiring to read. The bar is high.

- **Explain *why*, never *what*.** Well-named code already says what it does.
- **Good triggers for a comment:** a hidden constraint, an invariant, a
  workaround for a specific bug, a non-obvious reason for a value, a numerical
  heuristic, a domain or physical assumption, a sequencing dependency between
  two lines that otherwise look independent.
- **Delete narration.** No comment that restates the line under it
  (`# increment counter` over `i += 1`, `# loop over offers` over the `for`).
- **No decorative banners** or section-divider comments inside a function
  body. A run of them means the function has multiple purposes — promote each
  section to a named helper (§10), don't label it in place.
- **No `# Note:` / `# Important:` spam** on lines that carry no real gotcha,
  and **no emoji** in comments, code, or commit messages.
- **No anonymous `# TODO`.** Every TODO carries a name, date, or issue
  reference; an anonymous one decays into a permanent lie.
- **No commented-out code.** Delete it; `git log` is the history (§12).
- **No stray debug output.** Remove `print()`-debugging before committing; use
  the project logger at the right level for anything that should persist, and
  don't scatter log lines through hot numeric paths.

**Magic numbers.** A numeric literal is "magic" if it encodes a policy decision
— a threshold, rate, cap, page size, or Kelly fraction. Promote those to a
module-level constant with a one-line comment naming the decision:

```python
# Confidence cutoff from CLAUDE.md "Performance Table" — below this,
# filter picks as too uncertain to publish.
MIN_CONFIDENCE: float = 0.54
```

Bare math constants (`0.5`, `1`, `2 * math.pi`) do not need extraction.

---

## 10. Functions

- **Target length:** ≤ 60 logical lines; hard suggestion ~120. An orchestrator
  above ~200 logical lines is a mandatory refactor (§2.8).
- **One clear purpose.** If the docstring summary needs an "and" ("loads data
  *and* fits a model *and* writes the report"), that's three functions. Each
  "and" is a missing helper.
- **But don't over-fragment.** Splitting earns its keep only when it yields a
  cleaner interface — a `_step_*` helper a reader can understand and test on
  its own. Breaking a coherent block into tiny pieces that must be read
  together to follow one thought makes the code *harder* to follow, not easier
  (§2.10). Prefer fewer, deeper functions.
- **A function must add value.** A body that forwards every argument unchanged
  to one other call, or wraps a one-line expression already readable at the
  call site, is a dead wrapper — inline it (§12, §18.7). No wrapper kept "for
  compatibility" (§18.8). This includes the **build-and-invoke thunk** — a
  function whose whole body builds a callable and immediately calls it
  (`_build_cli()()`); a console-script entry point must name the command
  object itself (`module:command`), never a `main()` that builds and runs it.
- **Name intermediates that are steps, not echoes.** `total = sum(...)` then
  using `total` twice is a step; `x = f(a)` then `return x` is an echo —
  `return f(a)`.
- **Use early returns and guard clauses** to validate parameters and shortcut
  impossible states; they flatten nesting better than `else` ladders.
- **Nesting > 4 levels of control flow is a refactor signal,** not a feature.
  Keep cyclomatic complexity in single digits (~10 ceiling); flatten with
  guards, early returns, or a lookup table.
- **No boolean flag arguments** that switch behavior — prefer two functions, an
  enum, or polymorphism.
- **Default arguments are immutable.** Never `def f(x=[])`.
- **Reach for the idiom.** Comprehensions over manual accumulation (but not
  nested to the point of unreadability), `with` for every resource,
  `enumerate` / `zip` over index arithmetic, `pathlib` over `os.path` string
  munging, tuple unpacking over positional indexing. `for i in range(len(x))`
  when you mean `for item in x` is a rewrite.

---

## 11. Error Handling

Python is **EAFP** — easier to ask forgiveness than permission. Try the
operation and handle the exception; don't pre-check with LBYL conditions that
are almost always true. A clean traceback that fails loud beats a swallowed
error that fails quietly.

- **Validate at system boundaries** — HTTP responses, file I/O, user input,
  external API payloads. Inside the package, trust your own types and
  invariants; interior code does not re-validate what a boundary already
  checked.
- **Do not add error handling for scenarios that cannot happen.** No `try`
  around code that won't raise, no `hasattr` / `getattr`-with-default to paper
  over your own code, no redundant `None` checks on values your own call sites
  guarantee. Defensive scaffolding for impossible states is pure machine bloat.
- **Handle only real edge cases** — the inputs the task names and those that
  demonstrably occur in our data. Don't pre-empt speculative inputs; ask first.
- **Catch narrowly.** Catch the specific exception you can actually handle.
  Never a bare `except:`. If you catch `Exception`, log it and either re-raise
  or return a sentinel the caller is documented to check — never catch and
  silently ignore.
- **Raise specific built-in types** (`ValueError`, `KeyError`,
  `FileNotFoundError`). Custom exceptions only when the caller needs to
  distinguish them.
- **Let loader failures at import raise.** A missing model or config should
  stop the run, not degrade silently into a fallback that lies.

---

## 12. Dead Code

- **Delete unused code in the commit that orphans it.** `git log` is the
  history. This includes unused imports (`F401`) and unused locals or
  parameters (`F841`).
- **Don't comment code out** to "keep it around" — that's just dead code with
  extra noise.
- **If code is temporarily dark but genuinely intended to return,** move it to
  `src/deprecated/` with the archive header (see `src/deprecated/README.md`)
  and add a TODO to `README.md`. Don't leave it commented in place.
- **A dead wrapper layer is dead code.** A function that only forwards to one
  other function with no added clarity, validation, or naming improvement gets
  inlined (§10, §18.7).
- **Orphan methods.** After removing a caller, grep for all call sites.
  Zero-caller methods go to `src/deprecated/`, not into the next refactor's
  surprise pile.

---

## 13. Dependencies

- **Don't add a dependency unless it pulls real weight.** A thirty-line helper
  is cheaper to own than a new transitive tree.
- **Don't reinvent it either.** Reimplementing what the stdlib or an installed
  library already does well (`itertools`, `collections`, `pathlib`, pandas /
  numpy / polars vectorization) is the mirror image of dependency bloat — both
  add code to maintain. Use what's already imported (§2.9).
- **Pin via Poetry** (`pyproject.toml` + `poetry.lock`). Optional heavyweight
  deps go in an optional group and are lazy-imported (§5), so core users don't
  carry them.
- PyTorch is CPU-only from a custom Poetry source (see CLAUDE.md). Don't swap
  that source without weighing the install-time impact on the production box.

---

## 14. Testing Expectations

- **Run the full quality-gate suite from the repo root before claiming any task
  is done:**
  - `poetry run ruff check src/sportstradamus/`
  - `poetry run pytest tests/golden/`
  - `poetry run pytest -m integration` (fake-mode, no network)
- Add characterization tests for current behavior before deep structural edits,
  especially around the training pipeline, the Archive schema, and Sheets
  export.
- Keep every existing test passing. A flaky test is a real test until proven
  otherwise.
- Add a test for any new helper that affects an output, contract, or schema.
- **No vacuous tests.** A test that asserts a trivial truth to lift a coverage
  number tests nothing and hides the gap it pretends to fill. Test behavior, or
  don't add the test.
- Regenerate CLI help snapshots only when a flag change is intentional:
  `REGENERATE_SNAPSHOTS=1 poetry run pytest tests/golden/test_cli_help.py`.

---

## 15. Refactor Workflow

The order matters. Skipping a step is how you ship a "behavior-preserving"
refactor that quietly changes a metric.

1. **Capture baseline behavior.** Run the full suite and record the outputs you
   intend to keep stable (training-report numbers, golden snapshots, Sheets
   payload, archive rows).
2. **Add or extend tests** around the weakly covered behavior at the seam.
3. **Refactor one seam at a time.** One module per session; multi-module work
   goes to one subagent per module per CLAUDE.md.
4. **Re-run tests after each seam.** If something drifted, fix it before the
   next seam — don't chain uncertain edits.
5. **Update all call sites in the same change. No half-migrations.** Renaming
   or moving a symbol means fixing every caller now and deleting the old path
   now (§18.2) — not leaving a re-export bridge behind.
6. **Review for contract drift before committing:** signatures, key names,
   schemas, output formats, file paths, Sheet column order.

---

## 16. Documentation Updates

Update docs in the same commit that changes the contract or workflow they
describe. At minimum, consider:

- `README.md` — contributor pointers and quickstart commands.
- `CONTRIBUTING.md` — package map, where to add a league or market.
- `CLAUDE.md` — diagnostic schemas, deployment notes, the per-cell training
  stats columns. The training-stats schema in particular must stay in sync
  with `training/report.py`.
- This guide — when conventions change.
- Tests — when a contract is clarified, freeze it in a test.

---

## 17. Domain Glossary

A paragraph each for the terms that appear everywhere. Skim this before
grepping the codebase for meaning.

- **Offer.** A single row from a sportsbook: player, market, line, odds.
  Produced by `books.py` scrapers and `moneylines.get_props`.
- **Market.** A betting category for a player's performance in one game
  (e.g. "NBA: points", "NFL: receiving yards"). Each trained model corresponds
  to one market.
- **Line.** The numeric threshold the bookmaker sets for a market. Bettors bet
  over or under it.
- **Book (sportsbook).** A source of odds: DraftKings, FanDuel, Pinnacle,
  Caesars, PrizePicks, Underdog, Sleeper, ParlayPlay. Per-book reliability
  weights live in `book_weights.json`.
- **Archive.** The `Archive` singleton in `helpers/archive.py` — a DuckDB store
  at `archive/archive.duckdb` with `odds(league, market, game_date, entity,
  book, ev)` and `lines(league, market, game_date, entity, line)` tables.
  Writes buffer in memory and flush on `Archive().write()`. Read methods:
  `get_ev`, `get_line`, `get_moneyline`, `get_total`, `to_pandas`.
- **Stats.** The `Stats` abstract base class in `stats/base.py` and its league
  subclasses — `StatsNBA` (`stats/nba.py`), `StatsWNBA` (`stats/wnba.py`,
  inherits `StatsNBA`), `StatsMLB` (`stats/mlb.py`), `StatsNFL`
  (`stats/nfl.py`), `StatsNHL` (`stats/nhl.py`). Responsible for loading league
  game logs, computing player features, and producing training matrices.
- **Gamelog.** A per-league DataFrame of every game a player has played, keyed
  by `(season, game_id, player_id)`. Feature engineering rolls over windows of
  the gamelog.
- **LightGBMLSS.** The distributional-regression wrapper around LightGBM.
  Predicts a full probability distribution over the outcome, not a point
  estimate. The distribution family per cell is set in
  `data/config/stat_meta.json`.
- **Comp features.** "Player comparables": a feature set built by finding the
  k nearest neighbors in a z-scored profile space and aggregating their
  historical outcomes. Weights are optimized in
  `scripts/optimize_comp_weights.py`.
- **fused_loc.** The helper in `helpers/distributions.py` that blends the
  model's predicted location with the bookmaker-implied location using a
  distribution-specific rule. See the ML Pipeline notes in CLAUDE.md for the
  math.
- **Model stats.** The per-cell training diagnostics in
  `data/training/model_stats.parquet` (with a `.csv` mirror), written by
  `training/report.py:report()` as one wide row per `(league, market)` cell.
  This is the behavioral regression check for training and the single source of
  truth for diagnostics; the per-column schema is documented in CLAUDE.md.
- **Meditate / Prophecize / Confer / Reflect / Dashboard.** The five CLI entry
  points wired in `pyproject.toml`:
  - `meditate` trains models — the `training/` package, entry `training/cli.py`,
    per-market work in `training/pipeline.py:train_market`.
  - `prophecize` scores offers and exports to Google Sheets — the `prediction/`
    package, entry `prediction/cli.py`, scoring in `prediction/model_prob.py`.
  - `confer` fetches current odds — `moneylines.py`.
  - `reflect` analyzes parlay performance — `nightly.py`.
  - `dashboard` serves the Streamlit UI — `dashboard.py` / `dashboard_app.py`.

---

## 18. For Claude and Other LLM Contributors

These rules exist because automated edits arrive fast and in bulk, and a silent
behavior change from an agent is harder to catch than the same change from a
human. The job is not to produce more code; it is to produce the least code
that correctly does what was asked, in the style of the code already here.

**Posture and scope:**

1. **Assume no behavior changes by default.** If your patch alters an output, a
   key name, a Sheet column, or a metric value, that is a feature change —
   either it was requested or you stop and ask.
2. **Preserve output keys, file schemas, and external contracts — not old
   import paths.** Renaming or moving a symbol means updating every call site
   in the same change (no half-migrations — §15), then deleting the old path.
   Do **not** keep a deleted module or old name alive as a back-compat
   re-export "until callers are migrated" — that is the shim bloat §18.8
   forbids. A package `__init__.py` that re-exports its own submodules is fine:
   that *defines* the package's public API, it is not a bridge to a path that no
   longer exists. What stays stable is behavior the outside world depends on —
   training-report numbers, archive rows, Sheet column order, CLI flags — never
   the internal layout.
3. **Prefer small, reviewable patches, one module at a time.** Single-module
   work stays in the main session; multi-module work defaults to one subagent
   per module (CLAUDE.md is authoritative on the workflow).
4. **Do not add features, refactors, or "improvements" beyond what was asked.**
   A bug fix does not need the surrounding code cleaned up. Speculative cleanup
   is how scope grows quietly. If you spot unrelated work worth doing, say so;
   don't just do it.
5. **Do not add error handling for scenarios that cannot happen (§11).** Trust
   internal invariants; boundary code validates, interior code does not.
6. **Avoid speculative performance optimizations** unless requested and
   validated against a baseline. A faster version that changes a number is a
   behavior change.
7. **Remove or avoid dead helper layers** that don't improve clarity or
   correctness. A one-line pass-through is not a helper (§10, §12).
8. **Avoid backwards-compatibility bloat.** Trace a change through the codebase
   and fix signatures where needed. If a change forces regeneration of a large
   artifact (model pickle, training matrix, archive snapshot), call it out in
   the patch description.
9. **If uncertain about numerical or behavioral impact, stop at structural
   cleanup and flag the risk explicitly.** Don't guess at the training
   pipeline; name what the reviewer should validate before merge.

**Workflow:**

10. **Read this guide once per session.** After that, cite sections by number
    instead of re-reading.
11. **Prefer `Edit` over `Write`.** `Edit` sends a diff; `Write` resends the
    whole file. Use `Write` only for genuinely new files or full rewrites.
12. **Let the tools fix mechanical style.** `ruff format` and
    `ruff check --fix` run automatically on every edit (§19); don't spend
    tokens hand-formatting what they already handle.
13. **Add docstrings and only meaningful comments — do not pad (§7, §9).**
14. **Run or explicitly request validation before claiming completion.** The
    three gates in §14 must pass; if you can't run them, say so and ask.
15. **Consult the glossary (§17) before grepping** — the term is probably there.
16. **Dispatch parallel subagents for independent work** — per-league stats
    subclasses, per-book scrapers. One file per subagent.
17. **Subagent prompts name this guide by path** (`docs/STYLE_GUIDE.md`); they
    read it themselves rather than receiving its body.
18. **Do not speculatively abstract.** Two similar lines stay two lines. Three
    similar lines earn a helper, named for what it does, not for the
    duplication it removed (§2.6).

**Quick reference — machine-code smells and the rule that bans each.** If a
review finds any of these, it fails before substance review:

| Smell | Banned by |
|---|---|
| Wrapper that only forwards a call | §10, §12, §18.7 |
| Build-and-invoke thunk (`f()()` as a function body) | §18.7 |
| Back-compat shim / dual old+new path | §18.2, §18.8 |
| `try`/fallback/validation for impossible cases | §11, §18.5 |
| Comment that narrates the code | §9 |
| Signature-echo or padded docstring | §7, §9 |
| Generic name (`data`, `result`, `helper`, `manager`) | §6 |
| Premature abstraction / class where a function fits | §2.6, §2.10, §18.18 |
| Duplicated logic instead of reuse | §2.9, §13 |
| Reinvented stdlib / library functionality | §2.9, §13 |
| `Any` where the shape is known; annotated obvious locals | §8 |
| Commented-out or orphaned code | §12 |
| Stray `print` debugging / log spam | §9 |
| Speculative perf change without a baseline | §18.6 |
| Vacuous coverage-padding test | §14 |
| Emoji in code or commit messages | §9 |

---

## 19. Enforcement

Style is enforced in three layers so review attention can go to substance, not
nits:

**1. While Claude edits — PostToolUse hooks.** Every `Edit` / `Write` /
`MultiEdit` to a `.py` file runs two hooks (`.claude/settings.json`):

- `ruff check --fix` then `ruff format` on that file — mechanical lint and
  formatting fixed before the agent moves on.
- `complexity-gate.py` (`.claude/hooks/`) — an AST gate that **blocks** the
  edit if a function you changed exceeds the §10 limits (cyclomatic complexity
  > 10, source length > 200, nesting > 4) and is new or got worse than its
  last-committed version. Pre-existing debt in a file never blocks an unrelated
  edit; the gate is a ratchet that only tightens. Override a genuinely
  irreducible case with `# style: allow-complexity` / `-length` / `-nesting`
  inside the function plus a one-line reason — the same documented-suppression
  posture this section applies to `# noqa`.

**2. Before any push, PR, or review — `refactoring-specialist` subagent.**
Mandatory on every Python file touched in the session (CLAUDE.md). It enforces
this guide's structural rules — orchestrator flatness (§2.8, §10), wrapper
elimination (§12), duplicate consolidation (§2.6), magic numbers (§9), comment
de-narration (§9), and the hard rules — citing `§N` for each change.

**3. In CI and local `pre-commit`:**

| Tool | Scope | Blocking? |
|---|---|---|
| `ruff check` | `src/sportstradamus/`, `tests/`, `scripts/` | yes |
| `ruff format --check` | same | yes |
| `pytest tests/golden/` | golden CLI snapshot tests | yes |
| `pytest -m integration` | fake-mode end-to-end smoke | yes (pre-commit) |
| `mypy` | `src/sportstradamus/` | advisory — warnings only |

Configuration lives in `pyproject.toml` (`[tool.ruff]`, `[tool.mypy]`) and
`.pre-commit-config.yaml`. Set up locally:

```bash
poetry install
poetry run pre-commit install
```

The Ruff `ignore` list is a deliberate refactor ratchet — rules are lifted as
each phase fixes the underlying issue, not silenced permanently. When `ruff`
flags something that feels wrong, fix this guide or the rule set; don't bury it
under an inline `# noqa` without a comment citing the reason.