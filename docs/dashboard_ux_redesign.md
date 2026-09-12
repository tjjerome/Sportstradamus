# Dashboard UX redesign — "the Oracle" spec

Canonical home of the owner-approved dashboard design (approved wireframes:
[docs/mockups/site-map.html](mockups/site-map.html),
[docs/mockups/facelift.html](mockups/facelift.html)). This doc says **what** the dashboard is;
the closed lane brief [archive/dashboard-ux.md](archive/dashboard-ux.md) records how it was built;
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
- Rivals legs are retired since 2026-09-10 (Underdog dropped the product); the H2H chip is gone.
- Internal pipeline names (`contest_variant` in pickem emit/parquet) are unchanged; the taxonomy
  governs UI language and slip-engine rules only.
- **Game lines are event-contract legs, not pick'em legs.** Underdog "Prediction Picks" and
  Sleeper "Sleeper Markets" (Total / Spread / Moneyline) are CFTC event contracts at market
  prices; on Underdog they combine with player picks only via a **Combo Entry** (contract +
  fantasy picks, compounded payout). Product mechanics + engine live in
  [handoffs/dfs-products.md](handoffs/dfs-products.md) §3/§6 — this doc owns presentation only
  (§5b). Game lines get **no model** (book-implied probabilities — locked, dfs-products §4);
  they are scarred into the UI (game-line rows on the Game board, team nodes in the
  constellation, slip accepts them) and join the correlation engine at dfs-products stage 5.
- **Ladders is a play type** (Underdog: 3–5 picks × 3 rungs, payout keyed to the lowest rung
  reached). Product names stay the DFS apps' own — Ladder, Combo Entry, game lines — in
  every voice; no prophecy-voice renames (locked,
  [handoffs/dfs-products.md](handoffs/dfs-products.md) §4).

## 3. The surfaces (`st.navigation`, Material icons, sport switch on every page)

Four top-level surfaces plus the Model Lab group. The planned Game and Slips pages were folded
into **Games** and the Pick'em tab retired, both owner-approved during the build.

| Surface | Job | Key content |
|---|---|---|
| **Tonight** (home) | "What's on tonight?" | Nebula game cards: matchup, tip-off urgency, the story engine's `game_headline`, best edge vs the DFS payout, model-liked leg count. Click → Games. |
| **Board** | Cross-game shopping | Every offer in the Obsidian AG Grid (phone: card list): `Move` line-movement spark, Win %, Model Edge, Consensus Edge, Kelly; prophecy lenses (Sharp / Longshots / Contrarian / Consensus); "+ slip" per row; detail dialog. |
| **Games** | One matchup, fully told — and the slip editor | Platform + game picker → Total / Spread / Shape hero, the story menu (≤ 5 stories × Bankroll Builder / Shoot the Moon), the constellation (click or tap stars to build; deeper / wider lenses; hover card with last five + movement), satellite and disliked-leg pickers, legs panel, Lock it in. |
| **Receipts** | Prove it | Hero: "if you'd tailed every rec" (ROI / win % / record). Skeptic checks: EV>5% record, CLV beat rate, calibration vs book, worst month (losers shown, never hidden). Record grid by league/market/platform, reliability panel, precomputed strategy-sim grid + live Customize, **your slips** graded nightly. |
| **Model Lab** (Diagnostics · Correlations · Training · Modifiers) | How the sausage is made | Per-cell health (model_stats + live metrics + lifecycle), calibration/diagnostics, correlation heatmap from `corr_market_summary`, gate matrix, the modifier reconciler; deep-link target from every "market trust" line. |

Global chrome: sport switch (All · WNBA · MLB · NBA · NFL · NHL) filters every surface; the slip
rail is mounted on every page.

## 4. The slip rail (persistent builder)

Lives on every surface (session state, plain non-widget keys). Entry points: "Add to slip" on any
offer row, any prophecy ("add story"), any pre-built entry ("load into rail").

Shows, live, per edit: platform toggle (Underdog | Sleeper) · legs with remove buttons ·
auto play-type chip (2–3 → Power, 4+ → Flex) · independent joint
probability (∏p) **and** correlation-adjusted joint probability (Gaussian copula over the
per-game correlation slices; cross-game pairs ρ=0) · payout multiplier (platform + play type +
leg count) · EV · fractional-Kelly stake from a bankroll input. Money is `Decimal`.

Save slip → `data/runtime/user_slips.parquet`; `reflect` grades pending slips nightly; graded
slips appear on Receipts ("your record vs the model's"). Both platforms price off a real pooled
schedule in `prediction/payouts.py` (`payout_curve_for`: Underdog Power/Flex, Sleeper Max/Flex
with the ≤ 2-leg full-refund rule) with the per-leg boost product on top — `slip_engine.py` is
the one sanctioned live calc.

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
  stat's view (last-10 vs line, H2H-only, minutes trend…). *Comps read
  `current_offer_details.parquet` (`comps_vs_opp`, written by `prophecize`); MLB comps are empty
  because pitcher/hitter comps use a different structure, and the panel says so.
- **The case** — precomputed why-string (template prose: form, matchup, model-vs-book
  disagreement).
- **Market trust** — this market's live 30-day record (`live_metrics_per_market.parquet`
  precision for the bet side) + deep link to its Model Lab cell page.
- **Pairs well with** — top correlated legs with ρ badges; add-to-slip inline.
- **Movement** tab — the app's posted line and its price-blended fair line over time
  (`current_line_movement.parquet`).

**Swap-a-leg dialog** (from any prophecy or the rail): keeps the story context on top
(headline + remaining legs); candidates from the same game ranked by **story fit** = corr with
remaining legs × edge; each row shows the slip-EV delta if swapped in; anti-correlated
candidates are shown but flagged "fights the thesis".

## 5b. New bet-type presentation (game-line combos · Ladders · alt lines)

Presentation design for the dfs-products lane's surfaces. Data producers and engine
math live in [handoffs/dfs-products.md](handoffs/dfs-products.md); every element here
is a scar (§8) until its snapshot artifact exists, and builds queue behind the lane
brief's current phase.

**Ladders** (Games board + Receipts):
- Per-pick rung display: the 3 rung lines with model survival probability each; a rung
  selector only if stage-0 capture finds rungs user-selectable (VERIFY — dfs-products §3).
- Slip-level payout-distribution strip: P(lowest rung = r) × payout for r ∈
  {fail, 1, 2, 3} — the discrete outcome vector, not a single EV number.
- Live rung-progress on graded/pending entries: each pick shows the highest rung
  reached; the entry grades at the minimum. Receipts grades ladders per-rung (payout
  tier reached), never binary win/loss.
- Rail: ladder slip mode shows payout preview = f(lowest rung) and the discrete-Kelly
  stake from dfs-products stage 3.

**Game-line / combo legs** (rail + Games board + Receipts):
- Contract chip on an event leg: market price (the probability the exchange charges) +
  our de-vigged consensus beside it; divergence badge when the dfs-products B4 trigger
  fires. Provenance stated honestly: "market price, not model."
- Combo Entry in the rail: fee-split EV breakdown (contract stake vs fantasy
  reservation, compounded payout); correlation-aware combo EV appears only after
  dfs-products stage 5; cash-out is not valued (lane brief §4).
- Evidence chain for an event leg: no projected-distribution tab (no model);
  shows consensus source, and price history once the line-movement snapshot covers event legs
  (today it reads DFS player-prop ladders only).

**Alt-line markers** (Board + rail + Receipts):
- Alt-line flag chip on any offer row whose `Alt Line` is true (flag already stamped
  at scoring; needs the snapshot column — scar). Receipts can then split record by
  standard-vs-alt line using the sanctioned gold identity swatch (DESIGN.md §4a
  exception — identity, never a plotted value).
- Rung-price provenance chip once dfs-products stage 2 lands: whether the book prob at
  an alt line came from a real archived rung or dist-inversion.

**Constellation application** (grammar home stays DESIGN.md §4a; changes owner-only):
- Moneyline/spread star fills with that team's color — grammar-consistent, no change.
- Game-total star has no single team: it fills with a gradient blend of the two teams'
  primary colours, centre-anchored between the team clusters (locked, dfs-products §4).
- Ladder picks render as one star per pick (no new mark grammar); rung detail lives in
  the hover card and rail only.
- Edges to game-line stars appear only when player×game-line ρ exists (dfs-products
  stage 5); until then game-line stars render edge-less.

## 6. Asset layer

The slot catalog (every slot, its placeholder, source, license, priority) is
[docs/art_assets.md](art_assets.md); this section is the contract.

- **Player assets**: not built yet — the [`player-headshots`](handoffs/player-headshots.md)
  lane owns them (per-league CDN fetch into a gitignored disk cache, a `shots` side channel
  into the constellation card, a monthly refresh); the initials disc + team colours are the
  shipped fallback until it lands (§8).
- **Team assets**: committed `data/config/team_assets.json` (team → primary/secondary hex only;
  league marks are an owner IP decision), generated once by `scripts/build_team_assets.py`.
- **Ambient imagery**: slot manifest at `data/assets/ambient/ambient_manifest.json` (slot →
  file, opacity, placement, attribution, source_url); slots `ambient_tonight` (card deck,
  `night_sky.jpg`), `ambient_receipts_hero` and `ambient_games_hero` (both `nebula.jpg`).
  `dashboard/assets.py:ambient_css` renders a slot only when its entry names a file on
  disk (downscaled to 1600 px at import); otherwise the caller's token gradient renders
  byte-identical. `placement` picks the geometry: heroes crop to cover, the Tonight deck
  tiles the image down the column and each card shows the next slice. Licensing is the
  owner's check before a file lands (`attribution` / `source_url` are notes the loader
  never reads). Rules (opacity ceiling, contrast floor, never behind tables, no AI art) are
  FIXED in DESIGN.md §3.
- **Favicon + logo**: hand-authored `dashboard/static/favicon.svg` (`page_icon`); `st.logo`
  renders `dashboard/static/logo_wordmark.png` / `logo_mark.png` only when they exist — the
  commission brief is `docs/art_briefs/logo_guru.md`.

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

All stock or commissioned; the owner clears the license before a file lands and notes
attribution in the manifest entry; semi-transparent per DESIGN.md §3 limits.

## 7. Data contracts the UI reads

Owned by the pipeline (canonical detail in code; this table is the UI's reading list):

| Artifact | New in redesign | Consumed by |
|---|---|---|
| `current_offers.parquet` + `Kelly`, `Why`, `Game` | columns | Board, Games, deep dive, shelf |
| `current_parlays.parquet` + `Thesis` | column | Tonight, Games |
| `data/runtime/current_game_corr.parquet` (League, Game, leg_a, leg_b, rho; leg key `Player\|Market\|Bet`) | new file | shelf math, constellation |
| `data/runtime/current_game_context.parquet` + `current_game_stories.parquet` | new files | Games story menu, Tonight cards |
| `data/runtime/current_offer_details.parquet` (comps, volume, other stats) | new file | deep dive |
| `data/runtime/user_slips.parquet` | new file | shelf save, Receipts |
| `data/runtime/player_assets.parquet` | not built (owner decision, §6) | — |
| `data/config/team_assets.json` (colours only) | new file | constellation, cards |
| `data/assets/ambient/ambient_manifest.json` | new file | `assets.py` ambient slots |
| `data/training/feature_importances.csv` | existing, newly wired | deep-dive chip ranking |
| `data/runtime/live_metrics_per_market.parquet` | existing, newly wired | market-trust lines, Model Lab |
| `data/runtime/current_line_movement.parquet` (per-offer posted + fair line history) | new file (written by `prophecize`) | Board `Move` column, deep-dive Movement tab, Games hover card |
| game-line offer rows (provenance=book-implied) in `current_offers` | later stage (dfs-products stage 4) | Game board rows, rail, §5b chips |
| `Alt Line` flag column in `current_offers`/`current_pickem` | later stage (dfs-products stage 2c; `persist.py` `_OFFER_KEEP_COLS`) | §5b alt markers, Receipts split |
| ladder entry/rung columns (new snapshot artifact) | later stage (dfs-products stage 3) | §5b Ladders views, Receipts per-rung grading |

## 8. Placeholder register (scars — visible, honest, roadmap-backed)

Every scar renders a real panel with "coming" microcopy, feature-detects its data artifact
(flips on when the file/column exists), and is registered in the closed lane brief's follow-ups
(builds queue behind the producing lane). Filled since this spec was written: the comps panel
(`current_offer_details`), the Board sparkline (`Move`, line movement), the card's last five,
and the ambient-image slots (owner-sourced files, [art_assets.md](art_assets.md)).

1. Correlation-block risk chip on the rail — needs UD/Sleeper pairing-rule model.
2. Game-line rows on the Game board + team nodes in the constellation — book-implied probs only
   (**no modeling engine** — locked, [handoffs/dfs-products.md](handoffs/dfs-products.md) §4;
   Combo-Entry mechanics live there §3); joins the correlation engine at dfs-products stage 5.
3. Player headshots — the initials disc stands until the
   [`player-headshots`](handoffs/player-headshots.md) lane lands. Team marks: skipped,
   colours only ([art_assets.md](art_assets.md)).
4. Optional free-LLM prose rewriter seam — documented only; templates are the contract.
5. Ladders views (§5b) — flip on the ladder snapshot artifact (dfs-products stage 3).
6. Alt-line markers + per-rung Receipts grading (§5b) — flip on the `Alt Line` snapshot column
   (dfs-products stage 2c).
7. Combo-EV chips on the rail (§5b) — flip on game-line offer rows (dfs-products stage 4;
   correlation-aware EV at stage 5).

## Changelog

- 2026-09-12 — product names locked to the DFS apps' own and the game-total star fill to a two-team gradient (§2, §5b; dfs-products §4); headshots routed to the `player-headshots` lane (§6, §8).
- 2026-09-11 — lane closed: §3 trued to the shipped nav (Games absorbs Game + Slips, Pick'em retired); Sleeper pricing, comps, `Move` spark and the asset layer trued; §8 renumbered; brief archived.
- 2026-07-10 — §5b new bet-type presentation added (Ladders, game-line combos, alt-line markers); taxonomy + contracts + scars extended; producers = dfs-products lane.
- 2026-06-11 — spec created from owner-approved mockup review (P0 of the dashboard-ux lane).
