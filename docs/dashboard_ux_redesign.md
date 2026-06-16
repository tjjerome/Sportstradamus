# Dashboard UX redesign — "the Oracle" spec

Canonical home of the owner-approved dashboard design (approved wireframes:
[docs/mockups/site-map.html](mockups/site-map.html),
[docs/mockups/facelift.html](mockups/facelift.html)). This doc says **what** the dashboard is;
the lane brief [handoffs/dashboard-ux.md](handoffs/dashboard-ux.md) says how/when it gets built;
[../DESIGN.md](../DESIGN.md) owns the visual tokens and bans. Don't restate either here.

## 1. Product intent

- **Audience**: owner + friends. Semi-public: self-explanatory views, no auth, no onboarding flow,
  but a skeptic must be able to verify profitability unaided.
- **Agency**: the model recommends; the user decides. Every recommendation is editable, every
  number explainable one click deep, every model claim provable on the Receipts surface.
- **Narrative first**: a parlay is a *story* (one game-script thesis; legs are its consequences).
  The thesis headline sells the slip; the evidence chain backs every leg.
- **Precompute-first**: everything visible is computed by the `prophecize`/`reflect` crons and
  persisted to parquet snapshots. The dashboard reads snapshots and does trivial math only
  (the one exception: slip joint-probability recompute — pure numpy/scipy on ≤6 legs).
- **Free resources only**: no paid APIs. Prose is deterministic template/phrase-bank text
  generated at prophecize time. An optional LLM-rewriter seam may layer on later **only** if a
  free API is available — never a dependency, always falling back to templates.
- **Hard rule inherited**: the dashboard never touches DuckDB (CLAUDE.md §Hard rules).

## 2. Naming map (prophecy voice) & platform taxonomy

Display copy leans mystic; internal field names do not change.

| Concept | Display name |
|---|---|
| Model-built correlated parlay (a `Family` in `current_parlays`) | **Prophecy** |
| Thesis headline for a prophecy | the prophecy's **headline** |
| Per-leg reason string | **the case** |
| Recommendation track record page | **Receipts** |
| Model diagnostics area | **Model Lab** |

Platform taxonomy (binding for all display copy and slip logic):

- Platforms are **Underdog** and **Sleeper**. Nothing else is a "platform".
- **Power / Flex are play types, not platforms or user choices**: 2–3 legs ⇒ Power, 4+ legs ⇒
  Flex. The slip rail shows an informational chip ("Power play" / "Flex play"); there is never a
  Power/Flex selector.
- **Rivals is a leg type** (Underdog head-to-head leg), shown as a chip on the leg — never a
  platform or contest grouping.
- Internal pipeline names (`contest_variant` in pickem emit/parquet) are unchanged; the taxonomy
  governs UI language and slip-engine rules only.
- **Underdog game lines** (Total / Spread / Moneyline) exist as offers but get **no model**
  (sharp lines). They are scarred into the UI (game-line rows on the Game board, team nodes in
  the constellation, slip accepts them) and join the correlation engine in a later stage with
  book-implied probabilities only.

## 3. The six surfaces (`st.navigation`, Material icons, sport switch on every page)

| Surface | Job | Key content |
|---|---|---|
| **Tonight** (home) | "What's on tonight?" | Game cards: matchup, lock countdown, top prophecy headline, top single-leg edge, story count. Click → Game. |
| **Game** | One matchup, fully told | Its prophecies (headline + legs + per-leg case + add/swap), full offer board for the game incl. game-line scar rows, matchup context strip (total, spread/ML, pace), constellation of the game's correlations. |
| **Board** | Cross-game shopping | Every offer; AG Grid themed to tokens; columns: form sparkline, model P, book P, edge, kelly `K`; prophecy chips as filter lenses; "+ slip" per row. |
| **Slips** | Model's pre-built entries | `current_parlays` families + pickem entries, play-type chips (Power/Flex), Rivals legs marked; "Load into rail" → edit as your own. |
| **Receipts** | Prove it | Hero: "if you'd tailed every rec" cumulative units + record. Skeptic checks: record at EV>5%, CLV beat rate, calibration one-liner, worst month (losers shown, never hidden). By league/market/platform grid. Strategy simulator (Profit Sim fold-in). **Your slips**, graded nightly. |
| **Model Lab** | How the sausage is made | Per-market cell health (model_stats + live metrics + lifecycle), calibration/diagnostics/correlation views (old pages 4/5/7), deep-link target from every "market trust" line. |

Global chrome: sport switch (All · WNBA · MLB · NBA · NFL · NHL) filters every surface; the slip
rail is mounted on every page.

## 4. The slip rail (persistent builder)

Lives on every surface (session state, plain non-widget keys). Entry points: "Add to slip" on any
offer row, any prophecy ("add story"), any pre-built entry ("load into rail").

Shows, live, per edit: platform toggle (Underdog | Sleeper) · legs with remove buttons (Rivals
chip where applicable) · auto play-type chip (2–3 → Power, 4+ → Flex) · independent joint
probability (∏p) **and** correlation-adjusted joint probability (Gaussian copula over the
per-game correlation slices; cross-game pairs ρ=0) · payout multiplier (platform + play type +
leg count) · EV · fractional-Kelly stake from a bankroll input. Money is `Decimal`.

Save slip → `data/runtime/user_slips.parquet`; `reflect` grades pending slips nightly; graded
slips appear on Receipts ("your record vs the model's"). Sleeper payouts: until a verified
Sleeper payout table is committed, the rail shows an editable multiplier and suppresses the EV
badge with "payout table unverified" — it never prices off placeholder 1.0× multipliers.

Correlation-block risk (Underdog/Sleeper leg-pairing rejection rules) is a scarred chip on the
rail — placeholder until the pairing-rule model lands.

## 5. Evidence chain ("why this pick")

**Deep-dive dialog v2** (row click anywhere; keeps today's three tabs, adds the case):

- Header: headshot, jersey number, team colors, market + line + platform + edge badge.
- **Projected distribution** — the existing density/PMF chart with line marker, over/under
  shading, P(over) annotation. Unchanged math, restyled.
- **Stat chips** — the inputs that feed the projection (Avg L5, Avg H2H, DVPOA, game total,
  moneyline, minutes trend, comps*), ranked by per-market SHAP importance
  (`data/training/feature_importances.csv`). Clicking a chip flips the chart below to that
  stat's view (last-10 vs line, H2H-only, minutes trend…). *Comps chip is scarred until comp
  outputs persist at prophecize time.
- **The case** — precomputed why-string (template prose: form, matchup, model-vs-book
  disagreement).
- **Market trust** — this market's live 30-day record (`live_metrics_per_market.parquet`
  precision for the bet side) + deep link to its Model Lab cell page.
- **Pairs well with** — top correlated legs with ρ badges; add-to-slip inline.
- Scarred: line-movement timeline tab (needs the archive→parquet export job); comps panel.

**Swap-a-leg dialog** (from any prophecy or the rail): keeps the story context on top
(headline + remaining legs); candidates from the same game ranked by **story fit** = corr with
remaining legs × edge; each row shows the slip-EV delta if swapped in; anti-correlated
candidates are shown but flagged "fights the thesis".

## 6. Asset layer

- **Player assets**: `reflect` writes `data/runtime/player_assets.parquet` (league, player,
  player_id, headshot_url, team, jersey) from the per-league player-ID sidecars; headshot URLs
  are free official-CDN string templates; `dashboard/assets.py` lazily disk-caches images under
  `data/runtime/assets/` and falls back to an initials SVG avatar on any miss.
- **Team assets**: committed `data/config/team_assets.json` (team → logo URL, primary/secondary
  hex), generated/validated once by `scripts/build_team_assets.py`.
- **Ambient imagery**: slot manifest at `data/assets/ambient/ambient_manifest.json`
  (slot → file, opacity, placement, license/attribution). Slots: `page_backdrop`,
  `constellation_backdrop`, `hero_wash`, `countdown_motif`. Empty slots render token-palette
  gradients. Rules (opacity ceiling, contrast floor, never behind tables, no AI art) are FIXED
  in DESIGN.md §3.

### Artist/stock depiction brief (owner asked for suggestions; pick on acquisition)

1. **Sports equipment as constellations** — basketball seams, goalpost, helmet traced in stars
   on a night sky. The strongest single motif: it *is* the brand (use for
   `constellation_backdrop`, league-specific variants).
2. Hourglass with sand pouring into a court/field silhouette (`countdown_motif` — lock
   countdown).
3. Crystal ball reflecting stadium lights (`hero_wash`).
4. Astrolabe / star chart overlaid on a court diagram (`page_backdrop`, very faint).
5. Moon phases as a countdown strip; tarot-frame borders for prophecy cards; nebula behind a
   goalpost silhouette (alternates).

All stock or commissioned; license + attribution recorded per manifest entry; semi-transparent
per DESIGN.md §3 limits.

## 7. Data contracts the UI reads

Owned by the pipeline (canonical detail in code; this table is the UI's reading list):

| Artifact | New in redesign | Consumed by |
|---|---|---|
| `current_offers.parquet` + `K`, `Why`, `Game` | columns | Board, Game, dialogs, rail |
| `current_parlays.parquet` + `Thesis` | column | Tonight, Game, Slips |
| `data/runtime/current_game_corr.parquet` (League, Game, leg_a, leg_b, rho; leg key `Player\|Market\|Bet`) | new file | rail math, constellation, swap dialog |
| `data/runtime/user_slips.parquet` | new file | rail save, Receipts |
| `data/runtime/player_assets.parquet` | new file | assets layer |
| `data/config/team_assets.json` | new file | assets layer |
| `data/training/feature_importances.csv` | existing, newly wired | deep-dive chip ranking |
| `data/runtime/live_metrics_per_market.parquet` | existing, newly wired | market-trust lines, Model Lab |
| `data/runtime/line_movement.parquet` | later stage (archive export cron) | line-movement tab (scarred) |

## 8. Placeholder register (scars — visible, honest, roadmap-backed)

Every scar renders a real panel with "coming" microcopy, feature-detects its data artifact
(flips on when the file/column exists), and is registered in the lane brief's stage plan:

1. Line-movement timeline (per-offer dialog tab) — needs `export-line-movement` cron.
2. Comps panel + comps stat chip — needs comp outputs persisted at prophecize time.
3. Correlation-block risk chip on the rail — needs UD/Sleeper pairing-rule model.
4. Game-line rows on the Game board + team nodes in the constellation — needs game lines in the
   correlation engine (book-implied probs only; **no modeling engine** — locked decision, see
   lane brief §4).
5. Ambient-image slots — need acquired art (manifest fill-in).
6. Optional free-LLM prose rewriter seam — documented only; templates are the contract.

## Changelog

- 2026-06-11 — spec created from owner-approved mockup review (P0 of the dashboard-ux lane).
