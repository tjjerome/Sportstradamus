# Workstream Brief Template

Skeleton for `docs/handoffs/{slug}.md`. Copy, fill, delete the guidance
comments. A brief is the **single document a fresh session reads after
`CLAUDE.md` + `CONTRIBUTING.md`** to do useful work in its lane — written for a
weaker model than the one that designed the system. Target ≤250 lines excluding
the ledger; under ~80 means under-briefed.

The anti-drift litmus for every sentence: *would this become false after a
normal week of work elsewhere in the repo, without this file being edited?*
If yes, it must be a command to run or a pointer to the canonical home — never
restated prose. Stable facts and executable commands may be restated; cite the
source for any repo rule you restate so conflicts are detectable.

Status line values: `ACTIVE — stage N` · `QUEUED (entry: …)` · `BLOCKED (on: …)`
· `DONE`. The status line and the §10 ledger are the only parts a working
session edits routinely; §1–§9 prose changes only when reality changed
(revise in place, STYLE_GUIDE §16).

---

```markdown
# {Workstream name}

> Status: {ACTIVE — stage N | QUEUED (entry: …) | BLOCKED (on: …) | DONE}

## 1. Mission & money logic

<!-- Two sentences of mission. One paragraph: why this lane earns or protects
money. This anchor keeps a session from drifting into adjacent improvements. -->

## 2. Read first (in order)

<!-- Max ~8 entries, each with a one-line "why". CLAUDE.md + CONTRIBUTING.md
are assumed-read repo law; list only §-anchors of them that this lane leans on,
then home-of-record docs, then implementation files by repo-relative path. -->

## 3. Verify before you trust

<!-- Command block re-deriving every volatile fact this lane depends on.
Rule (state it verbatim in the brief): if command output contradicts brief
prose, the output wins — fix the brief in place (minor) or stop and ask the
owner (material). Typical block:

    git fetch origin && git log --oneline origin/devel -3
    python3 -c "import json,collections; m=json.load(open('src/sportstradamus/data/config/stat_meta.json')); [print(l, dict(collections.Counter(c['shipped'] for c in v.values()))) for l,v in m.items()]"
    ls -la data/training/model_stats.csv     # gate numbers fresh?
    <existence checks for prior-stage artifacts>
-->

### Volatile product assumptions

<!-- The external facts (app payout tables, leg caps, push/void rules, API
shapes) this lane's math depends on, each with a re-verify step. On drift:
stop, re-verify all of stage 0, revise this brief in place, then resume. -->

## 4. Locked decisions

<!-- Owner decisions this lane inherits, each dated. Sessions may not
relitigate; changes are owner-only. Workstream-scoped decisions live HERE
(this is their canonical home); cross-workstream gates live in roadmap v3. -->

## 5. Module footprint & canonical paths

<!-- Exhaustive list of modules this lane may touch, with canonical import
paths (sportstradamus.stats / .training / .prediction / .helpers — per
CONTRIBUTING §Package Map) so deleted shims are not recreated. Editing outside
the footprint is a stop condition (§8). If the lane touches the serving path,
add the inference-path compatibility pointer (model_improvement_track.md §11.3)
— reference it, don't restate it. -->

## 6. Stage plan

<!-- Numbered stages, each 1–4 sessions ending in a verifiable artifact.
Per stage:
- **Goal** (one line)
- **Entry** (what must be true to start)
- **Scope** (modules from §5; one module per subagent for multi-module work)
- **Acceptance** — commands + expected outcomes a session can run itself.
  "Owner is satisfied" is an escalation, never an acceptance criterion.
- **Est. sessions**
- **Kill criteria / if-it-fails branch** — mandatory (model_improvement_track.md
  §7/§9 discipline: scrap the path, take the next; record the verdict in the
  ledger). -->

## 7. Working rules

<!-- Lane-specific rules + the few stable repo rules worth restating WITH
citation (e.g. "dashboard reads parquet snapshots only, never DuckDB —
CLAUDE.md §Hard rules"). State the conflict order once:
command output > CLAUDE.md/CONTRIBUTING.md > home-of-record doc > this brief
> roadmap v3. -->

## 8. Escalation & stop conditions

<!-- Two lists.
STOP and ask the owner when: entry criteria unmet; gates red at session start
through no fault of yours; smoke regression; any change to gate constants or
test tolerances; anything touching credentials, paid APIs, cron, or ToS
surface; two consecutive sessions with no acceptance criterion moving (the
grind detector).
PARK AND PIVOT when blocked externally: append a ledger line with the blocking
reason, set the status line to BLOCKED (on: …), and point the owner at the
roadmap v3 swimlane index for the next lane.
DISPATCH a subagent when: research-analyst (Opus-backed) for this lane's named
research-gated triggers — list them; devel-ship-curator for every devel-bound
PR; prompt-engineer for new briefs / major re-briefs; refactoring-specialist
per the five CLAUDE.md triggers. -->

## 9. Session definition of done

<!-- Fixed checklist — restate verbatim in every brief:
- refactoring-specialist ran on every `.py` touched this session
  (CLAUDE.md five-trigger rule).
- `poetry run ruff check src/sportstradamus/` clean.
- `poetry run pytest tests/golden/` clean.
- `poetry run pytest -m integration -n0` clean, then
  `touch .claude/.state/integration_green`.
- One ledger line appended to §10; status line updated if a stage boundary
  was crossed.
- Never push `devel` directly — devel-ship-curator carves ship PRs.
- Durable non-obvious lesson? Offer a memory capture (CLAUDE.md §Agentic
  workflow conventions). -->

## 10. Ledger (append-only, newest first, cap ~15 — older lines live in git)

<!-- One line per session:
`YYYY-MM-DD · stage N · what landed (SHA) · gates ✓/✓/✓ · next: <one clause>`
Method-failure verdicts also land here with an evidence pointer.
The ONLY append-only zone in repo docs; everything else revises in place. -->
```
