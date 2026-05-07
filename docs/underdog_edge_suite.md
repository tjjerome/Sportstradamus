# The Underdog Edge Suite

## A Comprehensive Software Architecture for a Recreational Quant DFS Operation

---

## Reality Check Before You Build

A few honest truths up front, because they shape every architectural decision below:

1. **Most of what's described here is overkill for a $100–$1,000 bankroll.** Building everything will take 6–18 months solo and cost $200–$1,500/yr in infra and data subscriptions. To break even on those costs you'd need to clear ~$500–$2,000 in profit before the platform itself pays back. **You will likely get 80% of the value from 20% of the modules** — specifically Phase 1 below.
2. **Underdog's Terms of Service prohibit automated data scraping and "bots" that interact with the site.** The suite below treats Underdog data as something you observe — manually, via browser, or via narrow informational scrapers others have built — and never as something you place bets through programmatically. Designing this any other way risks account closure and forfeiture of funds.
3. **Sharp sportsbook data is the single largest line item.** Free options exist, but every serious version of this suite needs at minimum **The Odds API** ($30–$120/mo) or **SportsGameOdds** ($99–$499/mo) for prop comparison. OpticOdds and Unabated start at $500+/mo and only make sense at $5K+ bankrolls.
4. **The "AI prop picker" you've seen ads for is mostly marketing.** The suite below skips that path entirely and instead focuses on what actually works: **market arbitrage against sharp books, correlation modeling for parlays, and ADP/advance-equity arbitrage for drafts.**

The architecture below is what a professional version would look like. Treat it as a target and build the minimum viable version that wins.

---

## 1. System Architecture at a Glance

The suite has five logical layers. Each layer is independently runnable; you build out from the data layer.

```
┌──────────────────────────────────────────────────────────────┐
│  5. INTERFACE & MONITORING                                   │
│  Dashboard │ Alerts │ Bet Slip Helper │ Backtesting Reports  │
└────────────────────────▲─────────────────────────────────────┘
                         │
┌──────────────────────────────────────────────────────────────┐
│  4. EXECUTION & RISK                                         │
│  Kelly Sizer │ Exposure Tracker │ Stop-Loss │ Bet Logger     │
└────────────────────────▲─────────────────────────────────────┘
                         │
┌──────────────────────────────────────────────────────────────┐
│  3. DECISION ENGINES                                         │
│  Pick'em EV │ Parlay Builder │ Best Ball Drafter │ BR Stacker│
└────────────────────────▲─────────────────────────────────────┘
                         │
┌──────────────────────────────────────────────────────────────┐
│  2. MODELING                                                 │
│  Player Projections │ Game Sims │ Correlation │ Ensemble     │
└────────────────────────▲─────────────────────────────────────┘
                         │
┌──────────────────────────────────────────────────────────────┐
│  1. DATA INGESTION & STORAGE                                 │
│  Sportsbook Odds │ Underdog Lines │ Player Stats │ News      │
│  PostgreSQL + TimescaleDB │ Parquet │ Redis Cache            │
└──────────────────────────────────────────────────────────────┘
```

### Recommended core tech stack

| Concern | Recommendation | Why |
|---|---|---|
| Language | Python 3.11+ | Sports data ecosystem is overwhelmingly Python; R for some niche modeling |
| Database (warehouse) | PostgreSQL + **TimescaleDB** extension | Time-series-friendly; line snapshots are inherently temporal |
| Database (analytics) | **DuckDB** + Parquet files | In-process columnar; ideal for backtests and ad-hoc queries |
| Cache / queue | Redis | Pub/sub for line-move alerts; cache for hot model outputs |
| Orchestration | **Prefect 2** (or Dagster) | Lighter than Airflow for solo work; modern Python-native |
| ML libraries | XGBoost, LightGBM, scikit-learn, **PyMC** (Bayesian), statsmodels | Standard quant stack; PyMC for hierarchical models |
| Game simulation | NumPy + Numba (JIT) for Monte Carlo speed | A Python sim with Numba runs 100–500× faster than vanilla |
| Backend / API | **FastAPI** | Clean async; easy to wrap models as services |
| Dashboard | **Streamlit** to start, migrate to React + Plotly if needed | Streamlit ships in days; React in months |
| Notifications | Telegram Bot API + Discord webhooks | Both free; phone-native push |
| Containerization | Docker + docker-compose | Required if you want reproducibility |
| Hosting | Single $20/mo Hetzner VPS or Mac mini at home | More than enough for a recreational operation |
| Version control & CI | GitHub + GitHub Actions | Free for private repos at this scale |

You do **not** need Kafka, Spark, Airflow, Snowflake, Kubernetes, or any "big data" tooling. Anyone advising you to use those at this scale is selling consulting.

---

## 2. Layer 1: Data Ingestion & Storage

Data is the entire game. Every other module is an exercise in data quality.

### 2.1 Sportsbook odds (the most important data)

You need a **sharp consensus** to compare Underdog's static prop lines against. Sharpness rank is approximately Pinnacle ≈ Circa > BetMGM > FanDuel ≈ DraftKings > everyone else.

**Provider tradeoffs as of 2026:**

| Provider | Entry price | Sharp books? | Player props? | Realtime? | Best for |
|---|---|---|---|---|---|
| **The Odds API** | $30/mo (500 reqs) – $119/mo | Pinnacle yes (some plans), Circa no | Yes, per-event endpoint | Polling | First build; widely documented |
| **SportsGameOdds** | $99–$499/mo | Pinnacle yes (80+ books) | Yes, alt lines too | Polling | Better value than The Odds API at scale |
| **OddsPapi** | Free tier + paid | 350+ books incl. Pinnacle | Yes | WebSocket on paid | Newer alternative; transparent pricing |
| **OpticOdds** | $500+/mo (sales gated) | All major sharps | Yes | WebSocket push | Real bot territory; not for $1K rolls |
| **Unabated** | $500+/mo | Pinnacle, Circa, exchanges | Yes incl. DFS pick'em lines | WebSocket | Sharp consensus is built-in; nice for verification |
| **Self-scrape Pinnacle** | $0 | Sharp itself | Yes | Polling | Fragile; ToS gray area; not recommended |

**Recommended path for a $100–$1,000 player:** start on The Odds API's lowest paid tier ($30–$59/mo), poll every 60 seconds for player-prop markets on the 1–3 sports you actually play, and store every snapshot.

### 2.2 Underdog line capture

This is the one area where you must tread carefully. Three approaches in increasing aggressiveness:

1. **Manual export.** Underdog has no built-in export. Open-source Chrome extensions like the *Underdog Bet Exporter* exist for capturing your own bet history. **Use this only for capturing your own placed entries**, not for scraping.
2. **Browser-assisted observation.** Open the Underdog app, browse the Pick'em board, and have a tool log what you see. The repo `aidanhall21/underdog-fantasy-pickem-scraper` exists and is one starting point — but treat any automated polling as ToS-adjacent.
3. **Manual workflow.** For a $1,000 bankroll, the most defensible path is: open Underdog → screenshot or transcribe the 10–30 lines you're considering → paste into your suite for analysis. Slow, but completely above-board.

**My recommendation:** rely on third-party aggregators (RotoGrinders, OddsJam, OddsShopper) that already publish DFS pick'em lines with sportsbook comparisons. They've already done the legal-and-engineering work. Your money is better spent on their subscriptions than on building a scraper.

### 2.3 Player stats and play-by-play (free and excellent)

| Sport | Library | What you get |
|---|---|---|
| NFL | `nfl_data_py`, `nflfastR` (R) | Full PBP, snap counts, NextGen Stats, schedules |
| NBA | `nba_api` | PBP, box scores, advanced stats, lineups |
| MLB | `pybaseball`, Baseball Savant | Statcast (exit velo, xwOBA), splits, game logs |
| NHL | `MoneyPuck` (CSV downloads), Natural Stat Trick | xG, shot attempts, on-ice splits |
| WNBA | `wnba_api` (community fork), Her Hoop Stats | Box scores, advanced |
| NCAAF / NCAAB | `cfbfastR`, `hoopR` | PBP, drive-level data |
| PGA | `datagolf` API (paid, ~$10–$30/mo) | Strokes-gained model, course fit |
| Soccer | StatsBomb open data, FBref via `worldfootballR` | xG, shot maps, possession |

These libraries are the foundation. Your projection models are 60% feature engineering on top of them.

### 2.4 News, injuries, weather, lineups

- **Injury news:** Underdog's own news feed (poll the public page), RotoWire News API ($), FantasyLabs Injury Tracker ($).
- **Twitter/X beat reporters:** the X API now costs $200/mo for the Basic tier — **not worth it**. Better: a curated TweetDeck-style follow list, or a community Discord that aggregates breakers. Several exist in ETR/Solver/Stokastic memberships.
- **Weather:** OpenWeatherMap free tier is enough; Pivotal Weather for forecast model maps; ESPN's hourly-forecast embed for quick checks.
- **Lineups:** RotoWire confirmed lineup feed, Lineups.com, MLB starting lineup tweets, NBA's pregame Inactive list (hits ~30 min before tip).
- **Vegas totals/spreads:** included in any odds API; cross-reference Pinnacle and Circa.

### 2.5 Storage schema

Three primary stores:

**A. Time-series odds store (TimescaleDB hypertable)**

```sql
CREATE TABLE prop_lines (
    ts          TIMESTAMPTZ NOT NULL,
    sport       TEXT,
    event_id    TEXT,
    book        TEXT,           -- 'pinnacle', 'fanduel', 'underdog', etc.
    player_id   TEXT,
    market      TEXT,           -- 'player_pass_yds', 'player_points', etc.
    line        NUMERIC,
    over_odds   INT,            -- American odds
    under_odds  INT,
    PRIMARY KEY (ts, book, event_id, player_id, market)
);
SELECT create_hypertable('prop_lines', 'ts');
CREATE INDEX ON prop_lines (player_id, market, ts DESC);
```

This is the workhorse. Every line, every snapshot, forever. A 60-second poll across NFL/NBA/MLB will produce ~500K rows/day — TimescaleDB handles years of this without breaking a sweat.

**B. Player game-log store (Postgres or DuckDB)**

Standard relational schema: one row per (player, game), with all box-score and advanced metrics. Refreshed nightly.

**C. Underdog snapshots and your bets (Postgres)**

```sql
CREATE TABLE underdog_lines  ( ts, player_id, market, line, ... );
CREATE TABLE entries          ( id, ts_placed, contest_type, stake,
                                modeled_edge, modeled_win_prob, status, payout );
CREATE TABLE entry_legs       ( entry_id, leg_idx, player_id, market,
                                line, side, sharp_fair_prob );
```

Tracking your own action with **modeled win probability at time of entry** is critical for measuring whether your edge predictions actually hold up.

### 2.6 Orchestration

Prefect flows:
- `every_60s_pull_props` — sportsbook odds for active games
- `every_5min_pull_underdog_observation` — your manual or extension-assisted Underdog snapshots
- `nightly_refresh_player_stats` — all sports
- `event_driven_news_listener` — injury/lineup webhooks
- `pregame_lock_freeze` — at game lock, snapshot all props as the "closing" line for CLV analysis

---

## 3. Layer 2: The Modeling Engine

This is the technical heart. You're building **distributional forecasts** — not point estimates — because Pick'em is a probability question ("is points > 23.5?"), not a regression question.

### 3.1 The core projection problem

For any (player, game, stat) you need an **estimated probability distribution** of the player's outcome. From that you can answer any line: P(points > 23.5), P(rush_yds > 67.5), etc.

**The standard winning recipe (2024–2026 industry consensus):**

1. **Mean predictor**: gradient-boosted regressor (XGBoost or LightGBM) predicting expected value of the stat.
2. **Distribution predictor**: either
   - **Quantile regression** (LightGBM has native support via `objective='quantile'`) at a grid of quantiles {0.05, 0.1, ..., 0.95}, or
   - **Parametric assumption** (Negative Binomial for counts like points/rebounds, Tweedie or Gamma for yards, Poisson for low-frequency events like home runs/touchdowns), with parameters predicted by the model.
3. **Calibration**: isotonic regression or Platt scaling on the implied over/under probabilities, fit on a holdout set of past lines.

**Example skeleton (NFL passing yards):**

```python
import lightgbm as lgb
import numpy as np

# Features (illustrative — real models have 50–200)
FEATURES = [
    'qb_pass_attempts_l5_avg',
    'qb_pass_yds_per_attempt_l5_avg',
    'qb_dropbacks_per_game_l16_avg',
    'opp_pass_yds_allowed_per_game',
    'opp_pressure_rate',
    'team_implied_total',
    'spread',
    'game_total',
    'is_dome',
    'wind_mph',
    'is_primetime',
    'days_rest',
    'wr1_routes_run_share',
    # ... and dozens more
]

# Mean model
mean_model = lgb.LGBMRegressor(
    n_estimators=2000, learning_rate=0.02, max_depth=6,
    objective='tweedie', tweedie_variance_power=1.5,
)
mean_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
               callbacks=[lgb.early_stopping(100)])

# Quantile models — fit independently for each quantile
quantile_models = {}
for q in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
    m = lgb.LGBMRegressor(
        objective='quantile', alpha=q,
        n_estimators=1500, learning_rate=0.02, max_depth=6,
    )
    m.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    quantile_models[q] = m

def prob_over(features, line):
    """Estimate P(stat > line) from quantile predictions via interpolation."""
    quantile_values = np.array([
        quantile_models[q].predict([features])[0]
        for q in sorted(quantile_models.keys())
    ])
    quantile_levels = np.array(sorted(quantile_models.keys()))
    cdf_at_line = np.interp(line, quantile_values, quantile_levels)
    return 1.0 - cdf_at_line
```

### 3.2 Bayesian hierarchical models (for low-sample players)

XGBoost handles veterans well. It struggles with rookies, post-injury returnees, and players in new roles — exactly the spots where Underdog's lines are most often wrong.

A hierarchical model in PyMC partially pools each player toward a position-level prior:

```python
import pymc as pm

with pm.Model() as nba_points_model:
    # Position-level priors
    pos_mu = pm.Normal('pos_mu', mu=15, sigma=5, shape=n_positions)
    pos_sigma = pm.HalfNormal('pos_sigma', sigma=5, shape=n_positions)

    # Player-level params, partially pooled toward position
    player_effect = pm.Normal(
        'player_effect',
        mu=pos_mu[player_to_pos],
        sigma=pos_sigma[player_to_pos],
        shape=n_players,
    )

    # Game-level effects
    opp_effect = pm.Normal('opp_effect', mu=0, sigma=2, shape=n_teams)
    pace_coef  = pm.Normal('pace_coef',  mu=0, sigma=1)

    mu = (player_effect[player_idx]
          + opp_effect[opp_idx]
          + pace_coef * pace_z)

    # Negative Binomial likelihood for points
    alpha = pm.HalfNormal('alpha', sigma=10)
    pm.NegativeBinomial('y', mu=mu, alpha=alpha, observed=points)

    trace = pm.sample(2000, tune=1000, target_accept=0.9)
```

The benefit: a rookie with five games of data gets shrunk toward the position prior. A 10-year veteran with 600 games barely budges from his observed mean. **This is the single biggest model improvement most amateur builds skip.**

Run the Bayesian model nightly, cache its predictions, and use them as **another input feature** to the GBT — an ensemble of approaches usually beats either alone.

### 3.3 Monte Carlo game simulator

For NFL especially, the cleanest way to handle correlation is to **simulate the game itself**, then read the player stat lines off the simulation. This is what game-script-aware platforms like NumberFire and the high-end DFS optimizers do.

A simple skeleton:

```python
def simulate_nfl_game(home, away, vegas_total, vegas_spread, n_sims=10_000):
    """Returns: dict of player → array of stat samples across sims."""
    # 1. Sample game flow params (Vegas-anchored)
    pace = np.random.normal(home.pace + away.pace, 2, n_sims)
    total_pts = np.random.normal(vegas_total, 8, n_sims)
    margin = np.random.normal(-vegas_spread, 10, n_sims)

    # 2. Translate to drives, pass attempts, rush attempts per team
    home_drives = drives_from_pace(pace, total_pts)
    away_drives = drives_from_pace(pace, total_pts)
    home_pass_att = pass_attempts(home, home_drives, margin)
    away_pass_att = pass_attempts(away, away_drives, -margin)

    # 3. Distribute attempts to receivers via Dirichlet target shares
    home_target_shares = np.random.dirichlet(home.target_share_alphas, n_sims)

    # 4. Sample yards-per-target, catch rates, etc. per player

    # 5. Aggregate to per-player stat samples
    return player_samples
```

The output is `n_sims` joint samples of every player's stat line. From this you get:
- Marginal distributions: `prob_over(player, market, line)`
- **Joint distributions**: `prob_over_AND(p1, m1, l1, p2, m2, l2)` — exactly the input you need for Pick'em parlay correlation.
- Game-script-conditional distributions for free.

Use **Numba** to JIT-compile the hot loop; expect 100K+ NFL games/second on a laptop.

### 3.4 Correlation modeling for parlays

This is the largest under-exploited edge in Underdog Pick'em. Most amateur players treat their 3- and 5-pick entries as if legs are independent. They are not.

Two approaches:

1. **From the game simulator** (preferred when feasible): just count joint outcomes across the simulated games. P(QB passes for >250 AND WR catches for >70) is just the empirical fraction.
2. **Gaussian copula** (when you only have marginals): fit a correlation matrix on historical residuals between player stats, then sample joint outcomes from a multivariate normal and transform via marginal CDFs:

```python
from scipy.stats import norm, multivariate_normal

def sample_joint(marginal_cdfs, corr_matrix, n=10_000):
    """Sample correlated outcomes given marginal CDFs and a correlation matrix."""
    z = multivariate_normal(mean=np.zeros(len(corr_matrix)),
                            cov=corr_matrix).rvs(n)
    u = norm.cdf(z)  # to uniform [0,1]
    samples = np.column_stack([
        cdf.ppf(u[:, i]) for i, cdf in enumerate(marginal_cdfs)
    ])
    return samples
```

Empirically derived correlations to seed:
- QB pass yards ↔ WR1 receiving yards (same team): **+0.55 to +0.70**
- QB pass yards ↔ RB1 rush yards (same team): **−0.20 to −0.35**
- WR1 ↔ WR2 receiving yards (same team): **slightly negative (target competition)**
- Star player points ↔ teammate points (NBA, pace-up game): **+0.15 to +0.30**
- Goalie saves ↔ his team's goal total: strongly **negative**
- Pass yards ↔ opposing team's WR1 receiving yards (shootout bring-back): **+0.30 to +0.45**

### 3.5 Sport-specific model recommendations

| Sport | Primary technique | Why |
|---|---|---|
| **NFL** | Game simulator + GBT for player residuals | Highly correlated; small games per season; strong public PBP |
| **NBA** | GBT + Bayesian hierarchical for usage; light Monte Carlo | High game volume (~1,200/season) → GBT trains well; rotations matter |
| **MLB** | Statcast features + Poisson/NegBin for counts | Huge sample; pitcher–batter matchup is dominant; weather (Coors) matters |
| **NHL** | Shot-attempt + xG model + line combinations | Power play deployment is the hidden variable |
| **WNBA / NCAAB / MLS** | Light GBT with strong regularization | Smaller samples; lean on Vegas totals and pace; bigger market inefficiencies but harder to model |
| **PGA** | Strokes-gained model (DataGolf is the standard) | Course fit dominates; props markets are thin |

### 3.6 Calibration and validation

Models can be predictive but mis-calibrated, which destroys you in betting because you're trading on the probabilities, not the rankings.

For every model in production:

- **Reliability diagram**: bucket predicted probabilities into deciles, plot mean predicted vs. observed. Should be near the diagonal.
- **Brier score** and **log loss** on holdout: track week over week.
- **Closing-line value (CLV)**: compare your bet to the Pinnacle line at game lock. **Long-term CLV is the single best predictor of future profit, much more stable than ROI.** If you're consistently beating the close by 1–3%, you're sharp regardless of what your bankroll did this month.
- **Conformal prediction** (optional, advanced): produces calibrated prediction intervals with finite-sample coverage guarantees. Useful for sizing bets when your model's tail uncertainty matters.

---

## 4. Layer 3: Decision Engines

The modeling layer outputs probabilities. The decision layer turns probabilities into bets.

### 4.1 Pick'em EV engine (single-leg screener)

For every Underdog Pick'em line you observe:

1. **Find the same line at sharp books** — Pinnacle first, Circa second, then DraftKings/FanDuel/BetMGM consensus.
2. **De-vig** the two-sided sharp price to get implied fair probability:
   ```
   p_fair_over = (1 / dec_over) / (1/dec_over + 1/dec_under)
   ```
   This assumes equal proportional vig — for Pinnacle it's close enough.
3. **Compare to Underdog's static implied probability** at –122 odds (54.95%).
4. **Compute leg edge** = p_fair − 0.5495.
5. **Rank by edge** and surface the top opportunities.

Output table:

| Player | Market | UD Line | UD Side | Pinnacle de-vig | Edge | Recommended? |
|---|---|---|---|---|---|---|
| J. Allen | Pass yds | 248.5 | Over | 58.2% | +3.3% | ✅ |
| L. Jackson | Rush yds | 56.5 | Over | 51.1% | −3.9% | ❌ |

**Critical gotcha:** sharp books and Underdog use slightly different stat definitions for some markets. **Always verify** that "rushing yards" both include/exclude kneeldowns identically, that "points" includes/excludes overtime identically, that "fantasy points" use the same scoring weights. Mismatches will silently bleed money.

### 4.2 Multi-leg parlay constructor (with correlation)

The structural edge in Underdog Pick'em is that payouts are fixed regardless of correlation. So:

```
Algorithm: maximize EV of an N-pick entry under constraints
1. Start from the top-K positive-EV legs from §4.1.
2. For each candidate combination of size N (3 or 5), pull the joint
   probability from the game-sim or copula module.
3. Compute entry EV:
     EV = joint_prob_all_hit * payout_multiplier - 1
4. Apply constraints:
   - max one leg per player
   - at most 2 legs from the same game (UD's rule)
   - min joint correlation > +0.10 for ≥2 legs (force at least one
     correlated stack — pure diversification is a trap)
5. Return top entries by EV per dollar.
```

This is the module that distinguishes a serious operator from a recreational one. The math says **a 3-leg correlated stack at modest individual edges (~+2% each) plus correlation +0.30 produces meaningfully higher EV than five independent legs at +3% each.**

### 4.3 Best Ball draft optimizer

Best Ball is a totally different problem. You're not predicting probabilities; you're **building a portfolio of teams to maximize expected prize money** in a tournament with a known payout curve.

The best public framework (used in modified form by The Solver, ETR, and Hayden Winks's spreadsheets):

```
1. Player projection: distribution of weekly fantasy points for every
   player, every week of the season. Use 10K Monte Carlo seasons.
2. League simulation: simulate full 12-person Underdog drafts using
   ADP-derived AI opponents.
3. Scoring sim: roll forward 17 weeks of fantasy points across all
   12 teams, apply Underdog's "best ball" optimal-lineup rule each week.
4. Advance probability: count how often each drafted team advances
   through round 1 (top 2 of 12), round 2 (top 1 of 12 in Week 15),
   round 3 (top 1 of 12 in Week 16), round 4 (top 1 of 672K in Week 17).
5. Expected payout: sum prize × probability across all advance levels.
6. Decision: at every draft pick, choose the player whose addition
   maximizes the team's expected payout, NOT raw projected points.
```

Key engineering details:
- ADP needs to be **stochastic**, not deterministic — opponents pick from a distribution around ADP, not at exact ADP. Underdog publishes pick distributions; Drafters does too.
- **Stack-aware**: when you draft Mahomes, the value of every Chiefs WR/TE rises in your team's expected ceiling weeks.
- **Late-season weighting**: Week 17 contributes ~36% of expected value in BBM-style structure. Players projected to play meaningful Week 17 games are systematically more valuable than ADP suggests.
- **Roster construction constraints**: BBM requires 18 picks: 1+ QB, 2+ RB, 3+ WR, 1+ TE, max 3 QB / 9 WR / etc. (Verify current rules — they shift slightly each year.)

For execution, you can run this **"during the live draft"** as a tablet companion: the optimizer recalculates after every pick and shows you your top 3–5 candidates with EV deltas.

### 4.4 Battle Royale lineup builder

Battle Royale is a single-week 6-player snake draft scored against a giant field. It's closer to DFS GPP optimization than to Best Ball.

The standard recipe:
1. Generate **player projection distributions** for the slate (same models as §3).
2. Generate **field ownership estimates** — how often each player will be drafted across all pods. Underdog ADP is the strongest proxy.
3. Run a **GPP simulator**: simulate 100K complete rosters being drafted by the field, score them, find your roster's percentile rank and prize equity.
4. **Optimize for percentile-weighted prize**, not for median outcome. The Battle Royale payout curve gives ~70% of prize pool to the top 1.5% of entries. Rosters that have a 3% chance of finishing top-1% are worth more than rosters with a 50% chance of finishing top-50%.

**Stacking is mandatory.** No 6-slot roster wins a 50K-entry contest without 2–3 players from a single team.

### 4.5 Champions / peer-to-peer EV engine

In Pick'em Champions, you're playing other users, not the house. Sportsbook line comparisons no longer apply because the "line" is set algorithmically and the prize pool is pari-mutuel.

The decision changes:
- Treat it as a **percentile-rank tournament** like Battle Royale.
- Estimate each leg's probability from your model, not from the implied line.
- Maximize **expected prize equity given the field's likely picks** — which means leveraging contrarian legs.
- Realistic EV ceiling: roughly half of classic Pick'em's, because there's no static-line lag to exploit.

---

## 5. Layer 4: Execution & Risk

This is where most quant amateurs go wrong: they build great models, then size bets emotionally and lose anyway.

### 5.1 Bankroll and Kelly sizer

For every recommended bet, the sizer computes:

```python
def fractional_kelly_stake(bankroll, p_win, decimal_odds, fraction=0.25):
    b = decimal_odds - 1
    q = 1 - p_win
    edge = b * p_win - q
    if edge <= 0:
        return 0.0
    full_kelly = edge / b
    return bankroll * full_kelly * fraction
```

Defaults:
- **Pick'em entries**: 1/4 to 1/2 Kelly, capped at 0.5% of bankroll per entry.
- **Best Ball entries**: flat-stake — Kelly is unstable for tournaments with steep payout curves and uncertain win probabilities. Cap at 5% of bankroll per single Mania-style entry; 0.5–1% per "dog" tournament entry.
- **Hard cap regardless of model edge**: never more than 1% of bankroll on a single Pick'em entry, no matter how juicy the model says.

Why fractional Kelly: **your edge estimate is itself uncertain**. Sizing at full Kelly assuming your 6% edge is real, when it's actually 6% ± 4%, can produce 80% drawdowns even with positive true EV.

### 5.2 Exposure tracker

For Best Ball especially, you need to track:

- **Player exposure**: across all your drafted teams, what % own each player?
- **Stack exposure**: which (QB, WR, ...) combinations dominate?
- **Leverage**: your exposure / field exposure. A player you have at 25% who the field has at 8% is +17% leverage.

The dashboard should flag when any exposure exceeds 35% — that's portfolio concentration risk.

### 5.3 Stop-loss and circuit breakers

Hard rules enforced by the suite:
- **Daily Pick'em limit**: no more than X% of bankroll wagered per day.
- **Drawdown trigger**: if bankroll falls 20% from peak, suite automatically halves recommended stake size and sends an alert.
- **30% drawdown trigger**: stop new entries entirely; review process for two weeks.
- **Tilt detector**: if entries placed in a 60-min window exceed 3× the rolling average, push a "are you sure?" notification.

This sounds paternalistic. It is. **The number-one cause of bankroll death is not bad models — it is doubling up after losses.**

### 5.4 Bet logger and CLV tracker

Every entry, log:
- All inputs at time of placement (lines, sharp consensus, model probability, edge)
- Stake, expected payout, computed EV
- Closing-line snapshot at game lock
- Final result and actual payout

The single most important downstream report is **CLV by entry type**:

| Entry type | Avg CLV | Sample size | Significance |
|---|---|---|---|
| Pick'em 3-leg correlated | +1.8% | 240 | sig at p<0.05 |
| Pick'em 5-leg flex | −0.4% | 88 | not sig |
| Best Ball Puppy | n/a (long settle) | 60 | use process score |

Beating the closing line consistently is the gold standard. ROI is noisy; CLV is signal.

---

## 6. Layer 5: Interface & Monitoring

### 6.1 Dashboard (Streamlit)

The minimum viable dashboard has six tabs:

1. **Today's Slate** — every active game, with weather/lineup status.
2. **Pick'em EV Board** — every Underdog line ranked by edge, with sharp comparison.
3. **Parlay Builder** — interactive: select legs, see joint probability and EV.
4. **Best Ball / Battle Royale** — draft companion view.
5. **Bankroll & Performance** — current bankroll, ROI by contest type, CLV chart, exposure heatmap.
6. **Backtest** — re-run any strategy on historical data, see returns curve.

You can ship a usable version of this in two weekends with Streamlit.

### 6.2 Real-time alerts

Telegram bot (free) with channels for:
- **Stale line alert**: Underdog's line hasn't moved but Pinnacle has — investigate.
- **Injury news**: a player you're considering or have exposure to is downgraded.
- **High-edge appearance**: a new line appears with > 4% modeled edge.
- **Bankroll threshold**: drawdown / win threshold crossed.
- **CLV anomaly**: your last 20 bets are systematically losing CLV — model may be drifting.

### 6.3 Backtesting framework

For every strategy (parlay shape, edge threshold, sport mix), re-run on history:

```python
class BacktestEngine:
    def __init__(self, strategy, start_date, end_date,
                 starting_bankroll=500):
        ...
    def run(self):
        # Walk forward day by day:
        # 1. Snapshot lines as they were at game lock
        # 2. Apply strategy rules to identify bets
        # 3. Settle vs. actual outcomes (already in DB)
        # 4. Update bankroll, log everything
        # 5. Compute summary stats: ROI, max DD, CLV, Sharpe-like ratio
```

Critical to do this correctly:
- **Use the line at game lock**, not at scrape time, to avoid look-ahead bias.
- **Use point-in-time injury status** — if Player X was downgraded at 3pm and you backtest assuming he was always out, you've cheated.
- **Realistic friction**: Underdog occasionally voids legs for inactive players; simulate that.

A good rule: **don't trust any strategy until you've backtested it on at least one full year of out-of-sample data and your CLV is positive.**

---

## 7. Build Roadmap (Phased)

This is the order I'd actually build this in if I were starting today on a $500 bankroll.

### Phase 0 — One Weekend (free)
- Get accounts set up: GitHub, Telegram bot, Postgres locally (or Supabase free tier).
- Read Underdog's ToS and your state's DFS regulations.
- Pick 1 sport to focus on first (NFL is the highest-volume, NBA is daily-volume, MLB has the longest season).
- Establish a manual baseline: spend two weeks playing Underdog at minimum stakes with a paper notebook tracking model vs. result. **This gives you intuition no software replaces.**

### Phase 1 — Weeks 1–4 (~$50/mo): the Pick'em EV Board
The single highest-ROI module to build first.
- Subscribe to **The Odds API** ($30–$59/mo).
- Build the prop ingester (`every_60s_pull_props`).
- Build the de-vig + sharp-comparison logic.
- Manually paste Underdog lines into a CSV; the screener tells you which are +EV.
- Bet only single-leg-equivalent 3-pick correlated stacks; size at 1/4 Kelly.

**Expected outcome:** if you're disciplined, +3–8% ROI on Pick'em immediately. This alone often pays for the entire stack.

### Phase 2 — Weeks 5–8: Bet logger + CLV tracker + simple dashboard
- Streamlit dashboard with Today's Slate + EV Board + Bankroll tabs.
- Telegram alerts for new high-edge lines and injury news.
- Bet log auto-populated from your manual entry of placed bets.

### Phase 3 — Months 3–4: Player projection model (one sport)
- Build the GBT mean + quantile-regression stack for NFL passing/rushing/receiving yards.
- Cross-check against The Odds API's sharp consensus — if your model and Pinnacle disagree, **trust Pinnacle by default** until you've shown CLV > 0 over 200+ bets.
- Use the model only as a **secondary signal** that confirms or downweights what the sharp comparison says.

### Phase 4 — Months 4–6: Game simulator + correlation
- Simple NFL game-script Monte Carlo.
- Joint probability for parlay legs.
- The parlay builder UI.

### Phase 5 — Best Ball season (June–August): Draft optimizer
- ADP-driven mock draft simulator.
- Advance-equity calculator.
- Live draft companion.
- Run before BBM VII drafts open in May; backtest on BBM VI public exposure data.

### Phase 6 — Optional, Year 2: everything else
- Bayesian hierarchical models.
- Conformal prediction.
- Battle Royale optimizer.
- More sports.
- Reactive UI in React.

**At every phase, ask: is the next module's expected EV improvement bigger than the time cost?** For most recreational players, the answer is "no" somewhere around Phase 4.

---

## 8. Total Cost Estimate (Annual)

| Item | Cost | Required? |
|---|---|---|
| The Odds API (Starter) | ~$360/yr | Phase 1+ |
| Establish The Run NFL/NBA bundle | ~$200–$400/yr | Phase 1+ |
| Stokastic Pick'em + Solver Best Ball | ~$300–$500/yr | Optional |
| OpenWeatherMap, RotoWire | ~$100/yr | Phase 2+ |
| VPS or home Mac mini | ~$240/yr | Phase 2+ |
| Domain + monitoring | ~$50/yr | Optional |
| **Total** | **~$900–$1,500/yr** | |

For a $500 bankroll, this is **2–3× the entire bankroll in subscriptions**. Don't do it. Run the manual version of Phase 1 with a free trial of one odds API and a single ETR subscription, total cost <$200/yr, until you've proven you can clear that in profit.

For a $2,000+ bankroll, the full Phase 1–3 stack pays for itself if you average +5% ROI on $5K of yearly action.

---

## 9. Key Anti-Patterns to Avoid

1. **Building a "model" that just trains on past prop hit rates.** The market already prices those. Your model has to predict residuals from the market, not the absolute outcome.
2. **Optimizing for backtest ROI by overfitting parameters.** Use rolling time-series CV with at least 6-month gaps; never use random k-fold on time-series data.
3. **Trusting your model over the sharp market.** If Pinnacle says 56% and your model says 62%, the most likely answer is your model has a bug. Sharp markets are extremely efficient. Treat your model as a **secondary signal**, not the source of truth.
4. **Ignoring transaction costs and friction.** The "10% rake" on Best Ball means you need to project Top-N% in absolute terms, not just beat the median. Embed the rake explicitly in EV calculations.
5. **Building a fully automated bet-placement bot.** Even if it were technically possible (and Underdog's ToS makes it grounds for ban), the marginal value over a manual entry workflow at $100–$1,000 bankroll is essentially zero. Save the engineering time.
6. **Skipping the bet log.** Without a clean log of what you predicted vs. what happened, you cannot improve. Every serious operator has a log. Build it before the model.
7. **Believing claimed +20% ROIs.** Real long-term ROI for sharp recreational Underdog players is +3% to +10%. Anyone selling you a system claiming more is either lying, lucky, or selling the system as the actual edge.

---

## 10. The Honest Bottom Line

A comprehensive software suite for Underdog is a **2-quarter-to-1-year solo engineering project** that, properly built, supports a **+3% to +8% Pick'em ROI and +5% to +15% Best Ball ROI**. Those numbers are real but small relative to the build cost.

For a $100–$1,000 bankroll the rational play is **the minimum viable version of Phase 1**: a single odds-API subscription, a manual workflow, a CSV-backed bet log, and rigorous discipline on staking. That captures 80% of the achievable edge for 5% of the engineering effort.

If you build anyway — and many people will, because building is fun and the "edge" framing makes it feel productive — make sure the system serves the bankroll instead of the other way around. The dashboard's most important number is not how clever the model is. It's **closing-line value per bet**, computed honestly, over a sample large enough to mean something. Get that positive and stable, and the rest is just reps.
