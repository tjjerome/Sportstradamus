# MLB / NHL activation

> Status: QUEUED (entry: stage-0 audits; NO shipping before D1/D2)

## 1. Mission & money logic

Audit the two withheld leagues — MLB (24 cells) and NHL (16 cells), every cell
`shipped: "withheld"` — and produce the decision packets behind owner gates
D1 (MLB activation, ~Jul 2026) and D2 (NHL activation, ~Sep 2026). Post-GO,
grind the cells through the standard ship lifecycle.

Money logic: two entire slates of app-priced markets at marginal cost. The
modeling machinery, the five offline gates, the Gate-2 lifecycle, and the
decision engines (kelly, pickem-build, parlay pricing) are league-agnostic and
already built; what MLB/NHL need is model-quality work + data-freshness repair
plus a small, known CLI-wiring delta (§3 findings) — not new architecture. No
doc gives these leagues a breadth target —
[`model_improvement_track.md`](model_improvement_track.md) §1 covers
NBA/WNBA/NFL only; this lane exists to close that gap. Seasonal urgency: **MLB
is in season now** — the apps price it heavily while NBA/NFL are dark, and the
D1 option decays as the season burns. NHL starts in October; D2 wants its
packet by ~Sep.

## 2. Read first (in order)

1. [`../sportstradamus_roadmap_v3.md`](../sportstradamus_roadmap_v3.md) §5
   (seasonality) + §7 (the D1/D2 rows) — why now, and who decides. Predecessor
   design sketches: [`../archive/sportstradamus_roadmap_v2.md`](../archive/sportstradamus_roadmap_v2.md)
   (non-normative, status claims stale).
2. [`model_improvement_track.md`](model_improvement_track.md) §1 (the
   lens), §3.3 (the failure-mode-census pattern stage 0 copies), §6 operating
   loop + §6 lever stages (the post-GO method, reused wholesale — never
   restated here).
3. [`../ship_gate.md`](../ship_gate.md) — g1–g5 thresholds the packets score
   against.
4. [`CONTRIBUTING.md`](../../CONTRIBUTING.md) §Package Map, §Adding a New
   League, §Adding a New Market, §Shipping to Production — the mechanics this
   lane audits and, post-GO, exercises.
5. [`CLAUDE.md`](../../CLAUDE.md) §Hard rules (DuckDB lock) + §Agentic workflow
   conventions (research-first) — assumed law; the two rules this lane leans
   on hardest.
6. [`model_improvement_track.md`](model_improvement_track.md) §7.5 —
   the model-track footprint and session mechanics stage 1+ mirrors.
7. `src/sportstradamus/stats/mlb.py`, `src/sportstradamus/stats/nhl.py` — the
   two Stats classes under audit (`season_start`, `load`/`update`).
8. `src/sportstradamus/training/markets.py` +
   `src/sportstradamus/data/config/stat_meta.json` — registry vs meta, the
   alignment stage 0 checks.

## 3. Verify before you trust

Rule, verbatim: if command output contradicts brief prose, the output wins —
fix the brief in place (minor) or stop and ask the owner (material).

```bash
git fetch origin && git log --oneline origin/devel -3

# Cell counts + release surface. As of 2026-06-10: MLB 24 / NHL 16, all "withheld"
python3 -c "import json,collections; m=json.load(open('src/sportstradamus/data/config/stat_meta.json')); [print(l, len(v), dict(collections.Counter(c['shipped'] for c in v.values()))) for l,v in m.items()]"

# Registry alignment — stat_meta cells absent from ALL_MARKETS can never train.
# 2026-06-10: MLB orphans '1st inning hits allowed' / '1st inning runs allowed';
# NHL orphan 'fantasy points prizepicks'
python3 - <<'EOF'
import ast, json
meta = json.load(open("src/sportstradamus/data/config/stat_meta.json"))
tree = ast.parse(open("src/sportstradamus/training/markets.py").read())
reg = next(ast.literal_eval(n.value) for n in ast.walk(tree)
           if isinstance(n, ast.AnnAssign) and n.target.id == "ALL_MARKETS")
for lg in ("MLB", "NHL"):
    print(lg, "meta-only:", sorted(set(meta[lg]) - set(reg[lg])))
EOF

# Gamelog freshness — CSVs are gitignored, so a fresh checkout is EMPTY. Sync from
# prod first (scripts/sync_from_prod.sh), then judge: most recent season present?
ls src/sportstradamus/data/player_data/MLB/ src/sportstradamus/data/player_data/NHL/

# Hardcoded season anchors. 2026-06-10: MLB 2024-03-28, NHL 2024-10-04 — two
# seasons stale (contrast NBA 2025-10-21, WNBA 2026-05-15, NFL 2026-09-10)
grep -n "season_start = datetime" src/sportstradamus/stats/*.py

# Which leagues the train/serve CLIs actually instantiate (findings below)
grep -n "from sportstradamus.stats import" src/sportstradamus/training/cli.py \
    src/sportstradamus/prediction/cli.py src/sportstradamus/nightly.py

# Archive odds coverage by league/market — READ-ONLY, short-lived connection.
# Run against a synced dev copy, never beside prod cron (§7 lock rule).
python3 - <<'EOF'
import duckdb
con = duckdb.connect("archive/archive.duckdb", read_only=True)
for t in ("odds", "lines"):
    print(con.execute(f"SELECT league, market, count(*) n, min(game_date) lo, "
        f"max(game_date) hi FROM {t} WHERE league IN ('MLB','NHL') "
        "GROUP BY 1,2 ORDER BY 1, n DESC").fetchdf().to_string())
con.close()
EOF

# App coverage — which MLB/NHL markets Underdog/Sleeper price right now
# (prod snapshot via sync_from_prod.sh; live scrapes are ToS surface → owner, §8)
python3 -c "import pandas as pd; df = pd.read_parquet('src/sportstradamus/data/runtime/current_offers.parquet'); print(df[df.League.isin(['MLB','NHL'])].groupby(['League','Platform','Market']).size())"
```

Findings verified 2026-06-10 (each re-derivable above — re-verify, don't trust):

- **`meditate --league MLB|NHL` currently trains nothing.**
  `training/cli.py` instantiates only `StatsNBA/StatsNFL/StatsWNBA`; the
  league loop hits `stat_structs.get(lg) is None` and silently `continue`s.
  (CONTRIBUTING §Adding a New League's "instantiates each league's Stats class
  dynamically" is stale — trust the grep.) Same gap on the serve path:
  `prediction/cli.py` imports the same three. `nightly.py` carries the full
  five-league map.
- **`confer` never fetches MLB/NHL from the Odds API** —
  `moneylines.py:LEAGUES_OF_INTEREST = ("NBA", "NFL", "WNBA")` (comment: prop
  coverage "thin"; books scrapers were the intended source).
  `scripts/backfill_historical_odds.py` does carry `baseball_mlb` /
  `icehockey_nhl` sport keys. The archive's MLB/NHL book side is therefore
  legacy-era until proven otherwise — the model_improvement_track.md §3.2
  "is the book honest?" check is mandatory before any g1 verdict is believed.
- **All 40 cells are `dist: SkewNormal`** with `target_normalization` /
  `posthoc` `"none"` — including count-shaped stats (home runs, stolen bases,
  goals, assists, blocked). Any re-route is research-gated (§4).
- `StatsMLB` / `StatsNHL` exist and export from `sportstradamus.stats`;
  gamelogs live under `src/sportstradamus/data/player_data/{LEAGUE}/{YEAR}/`
  (gitignored — freshness is only checkable where the data lives).
- `meditate` skips + pickle-prunes withheld cells; audit training needs
  `--bypass-withholding` or a `--deterministic` sandbox run (writes to
  `research/models/deterministic/`, never production). The strategy sweep
  lives on `model-research` only (model_improvement_track.md §6 branch
  asymmetry) — `ls src/sportstradamus/training/model_strategy_sweep.py`
  before planning around it.

### Volatile product assumptions

- **App market coverage per league** — which MLB/NHL markets Underdog/Sleeper
  actually price, and the Fantasy-Points position split `books.py:get_ud`
  assumes. Re-verify via the `current_offers` probe at the start of every
  audit session; on drift, redo the census before anything downstream.
- **Odds API market availability for MLB/NHL props** — the "thin coverage"
  note in `moneylines.py` predates 2026. Re-verify against the Odds API market
  catalog before designing the packet's book side; live calls burn paid
  credits → owner sign-off first (§8).

## 4. Locked decisions

All dated 2026-06-10:

- **D1 / D2 are owner-only gates** (roadmap v3 §7). This lane prepares
  decision packets; it never makes the call and never flips a gate.
- **Stage 0 is audit-only** — no `shipped:` flips, no devel-bound production
  edits. (The research-gate hook fires on any `shipped:` flip in
  `stat_meta.json` regardless; the rule and the hook agree.)
- **Distribution-family re-routing for these cells is research-gated** — any
  SkewNormal → ZINB/NegBin/ZAGamma re-route or dispersion-mechanism change
  requires dispatching the Opus-backed `research-analyst` FIRST and citing its
  `/tmp/researcher_*.md` brief. Hard requirement, not advisory (CLAUDE.md
  §Agentic workflow research-first; `.claude/research_gated.txt`).
- **NO-GO is a valid, successful deliverable** (§6 kill criteria).

## 5. Module footprint & canonical paths

**Stage 0 (now):** read-only on `src/sportstradamus/`; probes in `/tmp`; a
probe worth keeping may land in `src/sportstradamus/scripts/` (one module).
Scratch-branch wiring patches are allowed as evidence-production for the smoke
run, but nothing merges in stage 0.

**Post-GO (stage 1+):** mirrors the model-track footprint —
`sportstradamus.training` (pipeline, scorecard, report),
`data/config/stat_meta.json`, ship-config plumbing — see
[`model_improvement_track.md`](model_improvement_track.md) §7.5 and
§6; not restated here. Plus the activation-specific deltas this audit
names:

| Module | Why |
|---|---|
| `src/sportstradamus/training/cli.py` | instantiate `StatsMLB`/`StatsNHL` (today: NBA/NFL/WNBA only) |
| `src/sportstradamus/prediction/cli.py` | same gap on the serve path |
| `src/sportstradamus/stats/mlb.py`, `stats/nhl.py` | `season_start` anchors, loader repair |
| `src/sportstradamus/training/markets.py` | registry ↔ `stat_meta.json` alignment |
| `src/sportstradamus/data/config/stat_meta.json` | per-cell family/strategy; `shipped` flips post-gate |
| `src/sportstradamus/moneylines.py` | `LEAGUES_OF_INTEREST` — owner-gated (paid API spend) |

Editing outside the footprint is a stop condition (§8). Serving-path changes
carry the inference-path compatibility checklist
([`model_improvement_track.md`](model_improvement_track.md) §7.3) —
reference it, don't restate it.

## 6. Stage plan

### Stage 0a — MLB audit packet

- **Goal:** a D1 decision packet the owner can GO/NO-GO on.
- **Entry:** none — start any time; the ~Jul 2026 D1 window decays weekly.
- **Scope:** §5 stage-0 footprint.
- **The packet** (one committed doc; path recorded in the §10 ledger):
  1. Gamelog freshness verdict + repair-cost estimate — seasons missing; does
     `StatsMLB.load()/update()` still run against today's statsapi;
     `season_start` fix.
  2. Archive odds coverage by market (counts, date range, distinct books) +
     the honesty check: real two-sided prices or a legacy seed
     (model_improvement_track.md §4.2 pattern)? g1 grades against this.
  3. App coverage census: markets Underdog/Sleeper actually price, mapped onto
     the 24 cells; unpriced cells flagged for the owner's denominator call.
  4. Wiring-delta list (§3 findings) with cost estimate.
  5. `meditate --league MLB --bypass-withholding --market <subset>` smoke on a
     scratch branch (or `--deterministic` sandbox) + full scorecard sweep →
     free-passer count (cells already clearing g1–g5) + a failure-mode census
     table à la model_improvement_track.md §3.3.
  6. Recommended GO/NO-GO with reasons + a proposed breadth target for the
     owner to lock at D1.
- **Acceptance:** packet committed + §10 ledger line + owner pointed at it.
  The D1 decision is NOT this lane's to make.
- **Est. sessions:** 1–2.
- **Kill criteria:** data source unfixably stale / league API gone → the
  packet says exactly that, with the repair cost, and recommends NO-GO. A
  NO-GO packet is a successful deliverable, not a failure.

### Stage 0b — NHL audit packet

Same packet shape over the 16 NHL cells, against D2 (~Sep 2026, before puck
drop). Entry: none — run after 0a unless the owner reorders. Off-season
caveat: no live slates until October, so the app census reads last season's
snapshots/archive and the packet dates it as such. Est. 1–2 sessions; same
acceptance and kill criteria.

### Stage 1+ — post-GO per-cell grind (per league)

- **Entry:** D1 (MLB) or D2 (NHL) = GO, owner-committed; breadth target locked.
- **Procedure:** land the packet's wiring deltas + data repair first (one
  module per subagent), then the model_improvement_track.md §6 operating loop
  verbatim — free passers first (§6.0 pattern), then the §6 stage ladder
  cheapest-first. Loop and stages live in
  [`model_improvement_track.md`](model_improvement_track.md) §6;
  this brief does not restate them.
- **Acceptance per cell:** official full-HPO scorecard 5/5
  ([`../ship_gate.md`](../ship_gate.md)) → `shipped: "devel"` via standard
  mechanics (CONTRIBUTING §Shipping; devel-ship-curator carves every PR).
- **Est. sessions:** set by the packet's free-passer count + failure census.
- **Kill criteria:** model_improvement_track.md §8 failure protocol per
  lever/cell. League-level kill is an owner call → status `DONE (no-go)` or
  `BLOCKED (on: next season)`.

## 7. Working rules

- Conflict order: command output > CLAUDE.md/CONTRIBUTING.md > home-of-record
  doc (model_improvement_track, ship_gate) > this brief > roadmap v3.
- **Archive lock discipline.** Probes open `duckdb.connect(...,
  read_only=True)` and close immediately. `Archive()` is a read-write
  singleton whose DuckDB file lock lives for the connection lifetime — never
  hold one casually; CLAUDE.md §Hard rules (dashboard/DuckDB) is the canonical
  statement of the lock physics. `Archive().to_pandas(league, market)` is fine
  inside a short script that exits.
- Production data flows one way here: `scripts/sync_from_prod.sh` (pull).
  `sync_to_prod.sh`, crontab edits, anything on the prod box: owner-only.
- Compute: full-league `meditate` is hours; default to `--market` scoping and
  `--deterministic` sandbox runs for exploration (model_improvement_track.md
  §6); real-HPO only to confirm a ship candidate.
- Packets and this brief revise in place (STYLE_GUIDE §16); the §10 ledger is
  the only append-only zone.

## 8. Escalation & stop conditions

**STOP and ask the owner when:**

- entry criteria unmet (any stage-1 work before an owner-committed GO);
- gates red at session start through no fault of yours;
- integration smoke regression;
- any change to gate constants or test tolerances;
- anything touching credentials, paid APIs (incl. Odds API probes for MLB/NHL
  coverage), cron, or ToS surface (live Underdog/Sleeper scrapes);
- a full-league `meditate` looks necessary — it is real cost; prefer
  `--market` scoping and the deterministic sandbox (model_improvement_track.md
  §6), and ask before any full run;
- cron / production-box changes of any kind — owner-only;
- two consecutive sessions with no acceptance criterion moving (grind detector).

**PARK AND PIVOT when blocked externally:** append a ledger line with the
blocking reason, set the status line to `BLOCKED (on: …)`, and point the owner
at the roadmap v3 §4 swimlane index for the next lane.

**DISPATCH a subagent when:**

- `research-analyst` (Opus-backed) — hard-required before any
  distribution-family re-route or dispersion-mechanism change on these cells
  (§4), and for any packet family recommendation beyond "keep as-is";
- `devel-ship-curator` — every devel-bound PR (post-GO);
- `prompt-engineer` — new briefs / major re-briefs;
- `refactoring-specialist` — per the five CLAUDE.md triggers.

## 9. Session definition of done

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
  workflow conventions).

## 10. Ledger (append-only, newest first, cap ~15 — older lines live in git)

- `2026-06-10 · created · brief drafted from roadmap-v3 migration · next: stage 0a MLB packet (D1 window ~Jul)`
