"""Seed a demo session so you can see the full observe -> conclude -> learn flow
without needing live credentials.

It fabricates a market with an evolving order book, a few of "your" trades, and a
resolution, then concludes and analyzes the session.

    python scripts/seed_demo.py            # writes to the configured DB
    POLYML_DB_PATH=/tmp/demo.db python scripts/seed_demo.py
    polyml report --slug demo-will-it-rain

This is illustrative data, not real market history.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Allow running directly (``python scripts/seed_demo.py``) without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polyml.analysis.learner import Learner
from polyml.analysis.outcomes import OutcomeLinker
from polyml.config import load_config
from polyml.session import SessionManager
from polyml.storage.db import Database
from polyml.storage.models import OrderBook

SLUG = "demo-will-it-rain"


def _book(slug: str, bid: float, bid_sz: float, ask: float, ask_sz: float) -> OrderBook:
    return OrderBook.from_payload(
        {
            "marketData": {
                "marketSlug": slug,
                "state": "MARKET_STATE_OPEN",
                "bids": [{"px": {"value": str(bid)}, "qty": str(bid_sz)}],
                "offers": [{"px": {"value": str(ask)}, "qty": str(ask_sz)}],
                "stats": {"lastTradePx": {"value": str((bid + ask) / 2)}, "openInterest": "12000"},
            }
        }
    )


def main() -> None:
    config = load_config()
    db = Database(config.db_path)
    sm = SessionManager(db)
    session_id = sm.open_session(SLUG)

    base = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)

    # An order book that drifts down then recovers — momentum/imbalance vary.
    track = [
        (0.62, 200, 0.64, 150),
        (0.60, 120, 0.62, 260),  # sell pressure building
        (0.55, 90, 0.57, 320),   # falling, sell-heavy book
        (0.58, 140, 0.60, 180),
        (0.70, 300, 0.72, 120),  # recovered, buy-heavy
    ]
    for i, (bid, bsz, ask, asz) in enumerate(track):
        book = _book(SLUG, bid, bsz, ask, asz)
        # Backdate captured_at by editing the row we just inserted.
        rowid = db.insert_book_snapshot(book, source="rest", raw={"demo": True})
        ts = (base + timedelta(minutes=i)).isoformat()
        db.conn.execute("UPDATE book_snapshots SET captured_at=? WHERE id=?", (ts, rowid))
    db.conn.execute(
        "UPDATE sessions SET started_at=? WHERE id=?", (base.isoformat(), session_id)
    )
    db.conn.commit()

    # Your trades: a solid entry, then a panic exit during the dip (the mistake),
    # missing the recovery to a YES resolution.
    db.insert_activity(
        activity_id="demo-buy", activity_type="ACTIVITY_TYPE_TRADE", market_slug=SLUG,
        price=0.62, qty=20.0, is_aggressor=1, cost_basis=12.4, realized_pnl=None,
        create_time=(base + timedelta(seconds=30)).isoformat(),
        raw={"trade": {"intent": "ORDER_INTENT_BUY_LONG"}},
    )
    db.insert_activity(
        activity_id="demo-sell", activity_type="ACTIVITY_TYPE_TRADE", market_slug=SLUG,
        price=0.55, qty=20.0, is_aggressor=1, cost_basis=11.0, realized_pnl=-1.4,
        create_time=(base + timedelta(minutes=2, seconds=10)).isoformat(),
        raw={"trade": {"intent": "ORDER_INTENT_SELL_LONG", "side": "SELL"}},
    )

    # The market resolves YES (1.0) — the exit was premature.
    res_time = (base + timedelta(minutes=10)).isoformat()
    db.insert_outcome(SLUG, resolved_value=1.0, resolution_time=res_time, raw={"demo": True})
    sm.conclude_if_resolved(SLUG)

    report = OutcomeLinker(db).link_session(session_id, SLUG)
    db.mark_session_analyzed(session_id, json.dumps(report, default=str))

    print(f"Seeded session {session_id} for {SLUG}.")
    print(f"  realized_pnl={report['realized_pnl']}  good={report['good_decisions']}  bad={report['bad_decisions']}")
    print("  Lessons:")
    for lesson in report["lessons"]:
        print(f"    • {lesson}")
    print("\n" + Learner(db).train().summary())
    print(f"\nNow try:  polyml report --slug {SLUG}")
    db.close()


if __name__ == "__main__":
    main()
