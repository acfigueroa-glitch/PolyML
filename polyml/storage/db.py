"""SQLite storage for PolyML.

A single file holds the full observational record: market & book snapshots,
public trades, your balances/positions/orders, your activity (the "mirror"),
trading sessions, derived decisions, market outcomes, and learning runs.

Every table keeps a ``raw`` JSON column alongside extracted, queryable columns
so we never lose fidelity even as the API evolves.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

from polyml.storage.models import now_iso

logger = logging.getLogger(__name__)

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- Point-in-time market metadata (title, state, resolution fields).
CREATE TABLE IF NOT EXISTS market_snapshots (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    market_slug  TEXT NOT NULL,
    title        TEXT,
    state        TEXT,
    captured_at  TEXT NOT NULL,
    raw          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_market_snapshots_slug ON market_snapshots(market_slug, captured_at);

-- Order-book snapshots with extracted top-of-book metrics.
CREATE TABLE IF NOT EXISTS book_snapshots (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    market_slug    TEXT NOT NULL,
    state          TEXT,
    best_bid       REAL,
    best_ask       REAL,
    mid            REAL,
    spread         REAL,
    book_imbalance REAL,
    last_trade_px  REAL,
    open_interest  REAL,
    source         TEXT NOT NULL,          -- 'rest' | 'ws'
    captured_at    TEXT NOT NULL,
    raw            TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_book_snapshots_slug ON book_snapshots(market_slug, captured_at);

-- Public market trades (the tape).
CREATE TABLE IF NOT EXISTS market_trades (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id     TEXT,
    market_slug  TEXT NOT NULL,
    price        REAL,
    qty          REAL,
    side         TEXT,
    traded_at    TEXT,
    captured_at  TEXT NOT NULL,
    raw          TEXT NOT NULL,
    UNIQUE(trade_id, market_slug)
);
CREATE INDEX IF NOT EXISTS ix_market_trades_slug ON market_trades(market_slug, traded_at);

-- Account balance / buying power over time.
CREATE TABLE IF NOT EXISTS balance_snapshots (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    buying_power  REAL,
    total_value   REAL,
    cash          REAL,
    captured_at   TEXT NOT NULL,
    raw           TEXT NOT NULL
);

-- Position snapshots per market.
CREATE TABLE IF NOT EXISTS position_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    market_slug     TEXT,
    net_position    REAL,
    avg_price       REAL,
    unrealized_pnl  REAL,
    captured_at     TEXT NOT NULL,
    raw             TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_position_snapshots_slug ON position_snapshots(market_slug, captured_at);

-- Your order lifecycle: one row per observed state change (placed/modified/
-- filled/cancelled). This is the heart of the "mirror".
CREATE TABLE IF NOT EXISTS order_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id      TEXT,
    market_slug   TEXT,
    side          TEXT,
    order_type    TEXT,
    price         REAL,
    quantity      REAL,
    filled_qty    REAL,
    state         TEXT,                    -- ORDER_STATE_* / EXECUTION_TYPE_*
    event_type    TEXT,                    -- placed | filled | partial_fill | cancelled | rejected | snapshot
    source        TEXT NOT NULL,           -- 'rest' | 'ws'
    event_time    TEXT,
    captured_at   TEXT NOT NULL,
    raw           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_order_events_order ON order_events(order_id, event_time);
CREATE INDEX IF NOT EXISTS ix_order_events_slug ON order_events(market_slug, event_time);

-- Your settled activity from /portfolio/activities (trades, resolutions,
-- deposits/withdrawals). The authoritative record of what actually happened.
CREATE TABLE IF NOT EXISTS activities (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    activity_id    TEXT,
    activity_type  TEXT NOT NULL,
    market_slug    TEXT,
    price          REAL,
    qty            REAL,
    is_aggressor   INTEGER,
    cost_basis     REAL,
    realized_pnl   REAL,
    create_time    TEXT,
    captured_at    TEXT NOT NULL,
    raw            TEXT NOT NULL,
    UNIQUE(activity_id, activity_type)
);
CREATE INDEX IF NOT EXISTS ix_activities_slug ON activities(market_slug, create_time);

-- One trading session per market (opened when we start observing your
-- involvement; concluded when the market resolves).
CREATE TABLE IF NOT EXISTS sessions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    market_slug    TEXT NOT NULL,
    status         TEXT NOT NULL,          -- 'open' | 'concluded' | 'analyzed'
    started_at     TEXT NOT NULL,
    concluded_at   TEXT,
    analyzed_at    TEXT,
    outcome_value  REAL,                   -- resolved price (e.g. 1.0 YES, 0.0 NO)
    realized_pnl   REAL,
    summary        TEXT,
    UNIQUE(market_slug, started_at)
);
CREATE INDEX IF NOT EXISTS ix_sessions_slug ON sessions(market_slug, status);

-- Market resolution / outcome.
CREATE TABLE IF NOT EXISTS outcomes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    market_slug     TEXT NOT NULL UNIQUE,
    resolved_value  REAL,
    resolution_time TEXT,
    captured_at     TEXT NOT NULL,
    raw             TEXT NOT NULL
);

-- Derived, labelled decisions used for learning. Built by the analysis layer
-- from order_events/activities joined against market state at decision time.
CREATE TABLE IF NOT EXISTS decisions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     INTEGER REFERENCES sessions(id),
    market_slug    TEXT NOT NULL,
    decision_type  TEXT NOT NULL,          -- entry | exit | add | reduce | hold
    side           TEXT,
    decided_at     TEXT,
    price          REAL,
    size           REAL,
    features       TEXT NOT NULL,          -- JSON of engineered features
    label_pnl      REAL,                   -- realized PnL attributed to decision
    label_good     INTEGER,                -- 1 if outcome favourable, else 0
    created_at     TEXT NOT NULL,
    UNIQUE(session_id, market_slug, decision_type, decided_at)
);
CREATE INDEX IF NOT EXISTS ix_decisions_session ON decisions(session_id);

-- Each model training run, with metrics + feature importances.
CREATE TABLE IF NOT EXISTS learning_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model           TEXT NOT NULL,
    n_decisions     INTEGER,
    metrics         TEXT,                  -- JSON
    feature_importances TEXT,              -- JSON
    notes           TEXT,
    created_at      TEXT NOT NULL
);
"""


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # The async runner writes from worker threads (asyncio.to_thread) and the
        # WebSocket callbacks, so allow cross-thread use and serialise access with
        # a lock to keep writes safe.
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    # --- low-level helpers -------------------------------------------------------
    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def execute(self, sql: str, params: Iterable[Any] = ()) -> None:
        """Run a write statement under the lock and commit."""
        with self._lock:
            self.conn.execute(sql, tuple(params))
            self.conn.commit()

    def _insert(self, table: str, row: dict[str, Any]) -> int:
        cols = ", ".join(row)
        placeholders = ", ".join("?" for _ in row)
        sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"
        with self._lock:
            cur = self.conn.execute(sql, list(row.values()))
            self.conn.commit()
            return int(cur.lastrowid)

    def _insert_or_ignore(self, table: str, row: dict[str, Any]) -> int | None:
        cols = ", ".join(row)
        placeholders = ", ".join("?" for _ in row)
        sql = f"INSERT OR IGNORE INTO {table} ({cols}) VALUES ({placeholders})"
        with self._lock:
            cur = self.conn.execute(sql, list(row.values()))
            self.conn.commit()
            return int(cur.lastrowid) if cur.rowcount else None

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self.conn.execute(sql, tuple(params)).fetchall())

    def query_one(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            return self.conn.execute(sql, tuple(params)).fetchone()

    # --- typed inserts -----------------------------------------------------------
    @staticmethod
    def _dump(obj: Any) -> str:
        return json.dumps(obj, default=str)

    def insert_market_snapshot(self, slug: str, title: str | None, state: str | None, raw: Any) -> int:
        return self._insert(
            "market_snapshots",
            {"market_slug": slug, "title": title, "state": state,
             "captured_at": now_iso(), "raw": self._dump(raw)},
        )

    def insert_book_snapshot(self, book, source: str, raw: Any) -> int:
        return self._insert(
            "book_snapshots",
            {
                "market_slug": book.market_slug,
                "state": book.state,
                "best_bid": book.best_bid,
                "best_ask": book.best_ask,
                "mid": book.mid,
                "spread": book.spread,
                "book_imbalance": book.book_imbalance,
                "last_trade_px": book.last_trade_px,
                "open_interest": book.open_interest,
                "source": source,
                "captured_at": now_iso(),
                "raw": self._dump(raw),
            },
        )

    def insert_market_trade(self, slug: str, trade_id, price, qty, side, traded_at, raw: Any) -> int | None:
        return self._insert_or_ignore(
            "market_trades",
            {"trade_id": trade_id, "market_slug": slug, "price": price, "qty": qty,
             "side": side, "traded_at": traded_at, "captured_at": now_iso(), "raw": self._dump(raw)},
        )

    def insert_balance(self, buying_power, total_value, cash, raw: Any) -> int:
        return self._insert(
            "balance_snapshots",
            {"buying_power": buying_power, "total_value": total_value, "cash": cash,
             "captured_at": now_iso(), "raw": self._dump(raw)},
        )

    def insert_position(self, slug, net_position, avg_price, unrealized_pnl, raw: Any) -> int:
        return self._insert(
            "position_snapshots",
            {"market_slug": slug, "net_position": net_position, "avg_price": avg_price,
             "unrealized_pnl": unrealized_pnl, "captured_at": now_iso(), "raw": self._dump(raw)},
        )

    def insert_order_event(self, **kw: Any) -> int:
        row = {
            "order_id": kw.get("order_id"),
            "market_slug": kw.get("market_slug"),
            "side": kw.get("side"),
            "order_type": kw.get("order_type"),
            "price": kw.get("price"),
            "quantity": kw.get("quantity"),
            "filled_qty": kw.get("filled_qty"),
            "state": kw.get("state"),
            "event_type": kw.get("event_type"),
            "source": kw.get("source", "ws"),
            "event_time": kw.get("event_time"),
            "captured_at": now_iso(),
            "raw": self._dump(kw.get("raw")),
        }
        return self._insert("order_events", row)

    def insert_activity(self, **kw: Any) -> int | None:
        row = {
            "activity_id": kw.get("activity_id"),
            "activity_type": kw.get("activity_type"),
            "market_slug": kw.get("market_slug"),
            "price": kw.get("price"),
            "qty": kw.get("qty"),
            "is_aggressor": kw.get("is_aggressor"),
            "cost_basis": kw.get("cost_basis"),
            "realized_pnl": kw.get("realized_pnl"),
            "create_time": kw.get("create_time"),
            "captured_at": now_iso(),
            "raw": self._dump(kw.get("raw")),
        }
        return self._insert_or_ignore("activities", row)

    def insert_outcome(self, slug: str, resolved_value, resolution_time, raw: Any) -> None:
        self.execute(
            """INSERT INTO outcomes (market_slug, resolved_value, resolution_time, captured_at, raw)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(market_slug) DO UPDATE SET
                 resolved_value=excluded.resolved_value,
                 resolution_time=excluded.resolution_time,
                 captured_at=excluded.captured_at,
                 raw=excluded.raw""",
            (slug, resolved_value, resolution_time, now_iso(), self._dump(raw)),
        )

    # --- sessions ----------------------------------------------------------------
    def get_or_open_session(self, slug: str) -> int:
        existing = self.query_one(
            "SELECT id FROM sessions WHERE market_slug=? AND status IN ('open','concluded') "
            "ORDER BY id DESC LIMIT 1",
            (slug,),
        )
        if existing:
            return int(existing["id"])
        return self._insert(
            "sessions",
            {"market_slug": slug, "status": "open", "started_at": now_iso()},
        )

    def conclude_session(self, session_id: int, outcome_value, realized_pnl) -> None:
        self.execute(
            "UPDATE sessions SET status='concluded', concluded_at=?, outcome_value=?, realized_pnl=? "
            "WHERE id=? AND status='open'",
            (now_iso(), outcome_value, realized_pnl, session_id),
        )

    def mark_session_analyzed(self, session_id: int, summary: str) -> None:
        self.execute(
            "UPDATE sessions SET status='analyzed', analyzed_at=?, summary=? WHERE id=?",
            (now_iso(), summary, session_id),
        )

    def insert_decision(self, **kw: Any) -> int | None:
        row = {
            "session_id": kw.get("session_id"),
            "market_slug": kw.get("market_slug"),
            "decision_type": kw.get("decision_type"),
            "side": kw.get("side"),
            "decided_at": kw.get("decided_at"),
            "price": kw.get("price"),
            "size": kw.get("size"),
            "features": self._dump(kw.get("features", {})),
            "label_pnl": kw.get("label_pnl"),
            "label_good": kw.get("label_good"),
            "created_at": now_iso(),
        }
        return self._insert_or_ignore("decisions", row)

    def insert_learning_run(self, model, n_decisions, metrics, feature_importances, notes) -> int:
        return self._insert(
            "learning_runs",
            {"model": model, "n_decisions": n_decisions, "metrics": self._dump(metrics),
             "feature_importances": self._dump(feature_importances), "notes": notes,
             "created_at": now_iso()},
        )
