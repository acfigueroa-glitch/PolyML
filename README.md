# PolyML

**A machine-learning trading companion for [Polymarket US](https://docs.polymarket.us).**

PolyML watches Polymarket US markets and **mirrors your own trading activity** —
it never trades on your behalf. It records the market (scores, order books, YES/NO
quotes, odds) and your account (balance, buying power, positions, orders — tracking
when they are placed, removed, and filled). When the market you're trading
**concludes**, it links the decisions you made to the actual outcome, scores them,
and surfaces what you may have missed:

> *Which choice might have been preferable? Which indicators were overlooked,
> contributing to losses?*

The more sessions it sees, the sharper that feedback gets.

> ⚠️ **Observe-only by design.** PolyML exposes no order-placement code path. It
> reads your account and the markets; it does not buy, sell, modify, or cancel.

---

## How it works

```
   Polymarket US API                         PolyML
 ┌───────────────────┐      ┌──────────────────────────────────────────┐
 │ public markets WS │─────▶│ collectors   books / BBO / trades         │
 │ /markets/{slug}/… │      │              balances / positions / orders │
 │                   │      ├──────────────────────────────────────────┤
 │ private WS        │─────▶│ mirror       every order placed/filled/    │
 │ /portfolio/…      │      │              cancelled; positions; balance │
 └───────────────────┘      ├──────────────────────────────────────────┤
                            │ sessions     one per market; concluded     │
                            │              when the market resolves       │
                            ├──────────────────────────────────────────┤
                            │ analysis     link decisions → outcome,      │
                            │              counterfactuals, "overlooked   │
                            │              indicators", + train a model   │
                            └───────────────┬──────────────────────────┘
                                            ▼
                                    SQLite (data/polyml.db)
```

| Layer | Module | Responsibility |
|-------|--------|----------------|
| API | `polyml.api` | Ed25519 request signing, REST client, public/private WebSockets |
| Storage | `polyml.storage` | SQLite schema + persistence for everything observed |
| Collectors | `polyml.collectors` | Periodic snapshots of market & account state |
| Mirror | `polyml.mirror` | Records every action you take (orders/positions/balance) + settled activity |
| Sessions | `polyml.session` | Groups activity per market; detects conclusion/resolution |
| Analysis | `polyml.analysis` | Feature engineering, outcome linkage, the learner |

Everything is stored with both extracted, queryable columns **and** the raw JSON
payload, so no fidelity is lost as the API evolves.

---

## Quickstart

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # or: pip install -e ".[dev]" for tests/tooling
```

Requires Python 3.10+.

### 2. See it work — no credentials needed

Seed a fabricated trading session and read the analysis:

```bash
POLYML_DB_PATH=/tmp/demo.db python scripts/seed_demo.py
POLYML_DB_PATH=/tmp/demo.db polyml report --slug demo-will-it-rain
```

You'll see PolyML flag a premature exit, quantify what holding would have earned,
and list the indicators (sell-heavy book, falling momentum) that were pointing the
other way.

### 3. Go live

Generate an API key at <https://polymarket.us/developer>, then:

```bash
cp .env.example .env        # fill in POLYMARKET_KEY_ID and POLYMARKET_SECRET_KEY
polyml run                  # observe + mirror + learn; Ctrl-C to stop
```

PolyML auto-discovers the markets you hold or have orders in, watches them, opens
a session per market, and analyzes each one when it resolves.

---

## CLI

| Command | What it does |
|---------|--------------|
| `polyml run` | Start the live observe-and-learn loop (needs credentials). |
| `polyml backfill [--pages N]` | Pull your settled history, build & analyze sessions for already-resolved markets, then train. Great first run. |
| `polyml status` | Show how much has been collected (row counts, sessions, latest balance). |
| `polyml discover` | List markets you're involved in and available markets. |
| `polyml analyze [--slug S] [--all]` | Link decisions to outcomes for concluded sessions. |
| `polyml report [--slug S \| --session N]` | Print a session's analysis: decisions, lessons, counterfactuals. |
| `polyml trade [--live] [--market S]` | Run the autonomous one-share scalping bot. **Paper by default.** |
| `polyml train` | Train the learner on all accumulated decisions. |

> **Note on `backfill` vs `run`.** Backfill reconstructs sessions from your
> settled trade history, so it gives you **real labels, PnL, and counterfactuals**
> immediately. But the *predictive features* (order-book imbalance, momentum,
> spread at each decision) can only be captured live — there's no historical
> order-book snapshot for a market that already resolved. So the learner becomes
> genuinely predictive once `polyml run` has been collecting during your sessions.

---

## The learning loop

For each concluded session, the analysis layer:

1. **Reconstructs market state at each decision** (no look-ahead) — best bid/ask,
   spread, top-of-book imbalance, 30s/5m momentum, volatility, your position size,
   time into the session. These are the *indicators*.
2. **Labels each decision** good/bad from realized PnL and the final resolution.
3. **Builds counterfactuals** — e.g. *"exiting at 0.55 left 0.45/share on the
   table; holding to the 1.00 resolution would have been better."*
4. **Flags overlooked indicators** on losing decisions (the book was sell-heavy,
   momentum was falling, the spread was wide…).
5. **Trains a model** (gradient boosting / logistic) across all sessions to learn
   which indicators separate your winning decisions from your losing ones, and
   reports honest hold-out metrics plus feature importances.

With fewer than `learning.min_decisions_to_train` labelled decisions it falls back
to transparent feature/outcome correlations instead of pretending to have a model.

> The model's `score_decision()` is **advisory only** — PolyML surfaces it for your
> insight and never acts on it.

---

## Autonomous trading (one-share scalper)

`polyml trade` runs a bot that buys **one share at a time** and scalps for **any**
net profit — it never chases a bigger one. Per game it:

1. On each order-book update, builds features, asks the learner for a P(good)
   entry score, and runs the scalp strategy.
2. **Enters** (buys 1 share at the ask) only when a profitable exit is actually
   reachable after the dynamic round-trip fee and spread, the book can absorb the
   exit, and the entry passes the buy-pressure / learner gate.
3. **Exits** (sells 1 share at the bid) the moment the round-trip net profit
   clears the hurdle — *any* profit. Optionally cuts losses (stop-loss) and
   flattens before the market resolves rather than riding to a 0/1 settlement.
4. When the **game ends**, it flattens, **self-analyzes its own trades** (good =
   profit, bad = loss), retrains on its accumulated round-trips, and moves to the
   next game a little smarter.

### The fee-aware math

Polymarket's taker fee is dynamic — `fee(p, N) = θ·N·p·(1−p)` — peaking at p = 0.50
and tapering to zero at the tails. Selling N at price p nets
`(1−θ)·N·p + θ·N·p²`, so the **breakeven exit** solves the quadratic

```
θ·p² + (1−θ)·p − C/N = 0        (a = θ ≈ 0.05, b = 1−θ ≈ 0.95, c = −C/N)
```

where C is the entry cost basis (notional + entry fee). The exit only triggers
when **net profit > target hurdle** *and* the book has depth at the exit price.
This is all in `polyml/trading/fees.py` (rigorously unit-tested).

### Safety — live trading is double-opt-in

> ⚠️ Autonomous trading risks real money. The default is **PAPER** mode, which
> simulates fills against the live book and places **no orders**.

Live trading requires **both**:

```bash
# 1) config:  trading.mode: live      (or the --live flag)
# 2) env:     POLYML_ALLOW_LIVE_TRADING=yes
POLYML_ALLOW_LIVE_TRADING=yes polyml trade --live --market <game-slug>
```

If either is missing, the bot refuses and runs in paper mode. Hard caps always
apply: **one share per order**, a max number of concurrent positions, and a
**daily realized-loss kill switch** that halts new entries.

> **Honest caveat on the strategy.** "Take any profit, never chase" collects many
> tiny wins but is exposed on the downside: a stalled position that rides to a 0
> settlement can erase many winners (your own settled history showed exactly this
> pattern). That's why the stop-loss and flatten-before-close guards are **on by
> default** — disabling them (`stop_loss_usd: null`, `flatten_before_close: false`)
> is materially riskier. Validate in paper mode before ever going live.

## Configuration

Defaults live in [`config/default.yaml`](config/default.yaml); copy it to
`config/local.yaml` to override. Key knobs: which markets to watch (or
auto-follow your positions), polling intervals, session-conclusion rules, and the
learner (model type, minimum decisions, hold-out fraction). Credentials come only
from the environment (`.env`) and are never read from YAML.

Environment overrides: `POLYMARKET_KEY_ID`, `POLYMARKET_SECRET_KEY`,
`POLYML_DB_PATH`, `POLYML_REST_BASE_URL`, `POLYML_LOG_LEVEL`, …

---

## Development

```bash
pip install -e ".[dev]"
pytest
```

The test suite covers Ed25519 signing (verified against the public key), payload
parsing & order-book metrics, the session lifecycle, outcome linkage/labelling,
and a full model-training run on synthetic, learnable data — all offline.

---

## Project layout

```
polyml/
  api/         auth.py · rest.py · websocket.py
  storage/     db.py (schema) · models.py (parsing + OrderBook)
  collectors/  market_collector.py · account_collector.py
  mirror/      activity_mirror.py
  session/     manager.py
  analysis/    features.py · outcomes.py · learner.py
  trading/     fees.py · strategy.py · executor.py · engine.py
  runner.py    orchestrates the live observe-only async loop
  cli.py       command-line entrypoint
scripts/
  seed_demo.py end-to-end demo with no credentials
tests/         offline test suite
config/        default.yaml
```

## Status & roadmap

This is the first end-to-end foundation: data collection, the activity mirror,
session/outcome linkage, and a baseline learner that improves as data accumulates.
Natural next steps: richer features (cross-market / news / sports-score signals),
calibrated probability outputs, per-decision attribution dashboards, and a
backtest harness over your collected history.

## Disclaimer

PolyML is a personal research and learning tool. It does not provide financial
advice, and prediction-market trading carries risk of loss. Use at your own risk
and in accordance with Polymarket US's terms of service.

## API reference

Built against the official Polymarket US documentation at
<https://docs.polymarket.us> (REST base `https://api.polymarket.us/v1`, order-book
gateway `https://gateway.polymarket.us/v1`, public stream `…/ws/markets`, private
stream `…/ws/private`, Ed25519 request signing).

Verified against the live API, with a few specifics not obvious from the docs:

- **Every endpoint on the `api.polymarket.us` host requires the signed headers** —
  including market data (`/markets`, `/markets/{slug}/bbo`). Only the gateway
  order book (`gateway.polymarket.us/.../book`) is public.
- The signed message is `"{timestamp}{METHOD}{path}"` with the **path only — no
  query string**.
- Open orders are at `GET /orders/open`; activities want
  `sortOrder=SORT_ORDER_DESCENDING`.
- `positions` is a **map keyed by market slug**; a trade nests its fill under
  `aggressorExecution`/`passiveExecution`; a position resolution settles the long
  outcome to `market.outcomePrices[0]` and reports the final realized PnL in
  `afterPosition.realized`.
- The API is behind Cloudflare rate-limiting (HTTP 429); the client backs off and
  retries, and paging is bounded so a poll doesn't trip it.
