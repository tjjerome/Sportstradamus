# Model Track — Ship-75 → Ship-90

> Status: ACTIVE — Ship-75 lever work

## 1. Mission & money logic

Get ≥ 75% of each covered league's markets past the five offline ship gates and
into live soak ([`operation_ship_75.md`](../operation_ship_75.md) §North Star:
NBA ≥ 16/21, WNBA ≥ 14/18, NFL ≥ 15/20 — NFL is the binding league), then take
the Ship-90 rung when gate D5 fires. This is the **lead lane**: calibrated full
predictive distributions are the input every decision engine consumes, breadth
is the number of +EV opportunities per slate, and Gate-4 PIT-KS calibration *is*
alt-line pricing accuracy. Nothing downstream out-earns its marginals.

This brief adds only **session mechanics**. The plan itself — lever stack,
per-league path, failure protocol — lives in
[`operation_ship_75.md`](../operation_ship_75.md) and is not restated here.

## 2. Read first (in order)

1. [`operation_ship_75.md`](../operation_ship_75.md) — the home of record.
   Minimum: §Purpose (the lens), §5 lever stack intro + "The operating loop",
   §6 per-league path, §7 failure protocol, §9 verification.
2. [`ship_gate.md`](../ship_gate.md) — gate thresholds g1–g5, Gate-2 live
   graduation, S1–S3 supersede edges. Never restated; never loosened.
3. [`operation_ship_references.md`](../operation_ship_references.md) — research
   verdicts + the inference-path compatibility checklist for any change that
   touches serving.
4. `data/training/model_stats.csv` — per-cell gate numbers, rewritten by every
   `meditate` (column dictionary: CLAUDE.md §Training stats).
5. `src/sportstradamus/training/scorecard.py` — `compute_gates`, the gate
   implementation.

## 3. Verify before you trust

If command output contradicts any prose here or in ship_75's standings prose,
the output wins — fix the doc in place (minor) or stop and ask (material).

```bash
git fetch origin && git log --oneline origin/devel -5
# Shipped counts per league (withheld / devel / main)
python3 -c "import json,collections; m=json.load(open('src/sportstradamus/data/config/stat_meta.json')); [print(l, dict(collections.Counter(c['shipped'] for c in v.values()))) for l,v in m.items()]"
ls -la data/training/model_stats.csv          # stale ⇒ re-run meditate before trusting
poetry run check-graduation                    # lifecycle per (league, market)
```

### Volatile product assumptions

None directly — this lane prices stats, not app products. App-side payout
drift is the decision lanes' problem.

## 4. Locked decisions

- 2026-06-10 — Model track stays the lead lane; other lanes never preempt it
  (owner).
- 2026-06-10 — Gate definitions and thresholds are owner-only; a gate change
  never counts as a lever (ship_75 §5.9).
- Standing (ship_75 §North Star) — ship incrementally: a Gate-1-clearing cell
  goes to `shipped: "devel"` and counts; don't hold ships for batches.

## 5. Module footprint & canonical paths

`sportstradamus.training` (pipeline, scorecard, report, calibration, strategy
driver), `sportstradamus.stats` (feature columns when §5.8 feature work calls
for it), `src/sportstradamus/data/config/{stat_meta.json, ship_config.json}`,
`tests/golden/`. Prediction-side edits only via the inference-path checklist
([`operation_ship_references.md`](../operation_ship_references.md)). Dev-only
diagnostics (`zinb-routing-diagnostics`, `icc-diagnostics`, statsmodels) never
ship to `devel` — the curator's denylist enforces this.

## 6. Stage plan

This lane's stages are **per-cell lever passes**, not a fixed sequence — the
stage plan is ship_75 §5's operating loop, run until targets are met:

1. **Re-score & promote free passers** (ship_75 §5.1) — after any lever or
   data change, re-run the scorecard sweep; flip `shipped` on cells that now
   clear g1–g5. Acceptance: `model_stats.csv` shows `ship == True`; the flip
   is a one-line `stat_meta.json` edit on devel.
2. **Board → candidate → confirm** (ship_75 §5 "The operating loop") —
   `model-strategy-driver --board` ranks candidates per cell (research-branch
   tooling; verify it exists on your branch — `ls
   src/sportstradamus/training/model_strategy_driver.py` — else rank manually
   from `model_stats.csv`); top-K confirmed via real-HPO scorecard A/B
   (`python -m sportstradamus.training.scorecard --baseline … --candidate …`);
   nothing ships on an in-sample floor.
3. **Per-league next step** — read ship_75 §6 for the league you're working;
   NFL is the binding league and its §5.3 blend / §5.7 hierarchical work is
   research-gated.
4. **Ship** — one cell at a time via `devel-ship-curator`; Gate-2 soak does
   the rest (`check-graduation`, monthly `gate-status` cron promotes to main).
5. **D5 transition** — when §North Star targets are met, prepare the Ship-90
   decision packet ([`operation_ship_90.md`](../operation_ship_90.md) §Open
   questions) for the owner; this brief and lane continue under the new rung.

Kill criteria per lever live in ship_75 (§5.x "if-it-fails" branches; §5.9
matrix-exhaustion; §7 failure protocol). Record per-lever verdicts in ship_75 /
its references doc as today; this ledger records session-level outcomes only.

## 7. Working rules

- Conflict order: command output > CLAUDE.md/CONTRIBUTING.md >
  [`operation_ship_75.md`](../operation_ship_75.md) > this brief > roadmap v3.
- Cross-league testing policy: smoke (1–2 markets/league) before full
  verification; smoke regression = hard stop (ship_75 §9).
- Determinism gate before any cross-league A/B:
  `poetry run pytest tests/integration/test_determinism_gate.py -v -m integration`.
- Exploration runs use `meditate --deterministic` (sandboxed writes) and
  `--market` scoping; full-league retrains are expensive — don't run one to
  answer a one-cell question.
- Never train/ship on `main`; production tracks `devel` (CLAUDE.md
  §Production deployment).

## 8. Escalation & stop conditions

**Stop and ask the owner:** gate-constant or test-tolerance changes (always);
smoke regression; gates red at session start through no fault of yours;
anything touching cron, credentials, or paid APIs; two consecutive sessions
with no acceptance criterion moving (grind detector — and a cell that resists
an axis moves to the next axis, never gets re-ground).

**Park and pivot:** if blocked (e.g., NFL Gate-2 soak needs live games that
won't exist until September), append a ledger line with the reason, set
`BLOCKED (on: …)` above, flip the roadmap v3 §4 row, and point the owner at
the swimlane index.

**Dispatch:** `research-analyst` (Opus-backed per its frontmatter) before any
§5.6 family/distribution change, §5.3 blend-structure change, §5.7
hierarchical/TabPFN escalation, or any ship_75 §8 research hole — CLAUDE.md
research-first convention; the research-gate hook enforces the file-level
cases. `devel-ship-curator` for every devel-bound ship PR.
`refactoring-specialist` per the five CLAUDE.md triggers.

## 9. Session definition of done

- refactoring-specialist ran on every `.py` touched this session.
- `poetry run ruff check src/sportstradamus/` clean.
- `poetry run pytest tests/golden/` clean.
- `poetry run pytest -m integration -n0` clean, then
  `touch .claude/.state/integration_green`.
- One ledger line appended below; status line updated on stage boundaries.
- Never push `devel` directly — the curator carves ship PRs.
- Durable non-obvious lesson? Offer a memory capture.

## 10. Ledger (append-only, newest first, cap ~15)

- 2026-06-15 · INBOUND BUG (from dashboard-ux) · `strategies/profit_sim.py` has a payout/Kelly accounting bug that feeds the **S3 paired-Sharpe gate** (`training/scorecard.py`) and **Gate-2 Kelly yield** (`nightly._profit_sim_kelly_yield`): Sleeper `compute_payout` returns the gross boost (median 1.74) but `_settle_day` adds it as NET profit (should be `boost − 1`); the Kelly branch's `if payout <= 1: continue` skips every Underdog bet (payout 0.909) so Kelly mode bets only the overpaid Sleeper legs → MC bankroll explodes (+3.2M%); snapshot also carries `inf` boosts. Owner: fix the canonical engine (no duplicate sim), but it moves gate outputs ⇒ model-track + owner sign-off (§8). Fix payout-net + Kelly net/decimal + inf-guard, AND add optional staking params (flat-off-initial / daily-exposure cap / fractional-Kelly+cap) with **defaults = current behavior**, then revalidate S3 + Gate-2 and curate. Unblocks the dashboard-ux Strategy-simulator rework. Full diagnosis: memory `profit-sim-payout-kelly-bug`.
- 2026-06-10 · created · brief drafted from roadmap-v3 migration · next: free-passer re-score sweep (ship_75 §5.1), then NBA −2 / WNBA −3 calibration cells
