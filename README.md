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
| `polyml status` | Show how much has been collected (row counts, sessions, latest balance). |
| `polyml discover` | List markets you're involved in and available markets. |
| `polyml analyze [--slug S] [--all]` | Link decisions to outcomes for concluded sessions. |
| `polyml report [--slug S \| --session N]` | Print a session's analysis: decisions, lessons, counterfactuals. |
| `polyml train` | Train the learner on all accumulated decisions. |

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
  runner.py    orchestrates the live async loop
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
