"""Replay collected order-book history through the paper trader.

For every resolved market that has book snapshots, walk the snapshots in time
order, let the strategy decide at each tick (features reconstructed as-of that
moment), and settle the resulting paper position at the market's outcome. The
result is a fee-aware P&L of "what the model would have done" — the proof of
edge. Requires real book history, which only ``polyml run`` collects; a database
built from ``backfill`` alone has trades but no book, so nothing will trade.
"""

from __future__ import annotations

import logging

from polyml.analysis.features import FeatureBuilder
from polyml.fees import DEFAULT_FEE_RATE
from polyml.storage.db import Database
from polyml.trading.paper import PaperStrategy, PaperTrader, Scorer

logger = logging.getLogger(__name__)

SOURCE = "backtest"


def run_backtest(
    db: Database,
    scorer: Scorer,
    *,
    min_edge: float = 0.02,
    fee_rate: float = DEFAULT_FEE_RATE,
    slugs: list[str] | None = None,
) -> dict:
    """(Re)run the backtest and return a summary. Clears prior backtest rows."""
    db.execute("DELETE FROM paper_positions WHERE source=?", (SOURCE,))

    features = FeatureBuilder(db)
    strategy = PaperStrategy(min_edge=min_edge, fee_rate=fee_rate)
    trader = PaperTrader(db, scorer, strategy=strategy, source=SOURCE)

    if slugs is None:
        slugs = [r["market_slug"] for r in db.query(
            "SELECT market_slug FROM outcomes WHERE resolved_value IS NOT NULL"
        )]

    markets_considered = 0
    for slug in slugs:
        resolved = db.query_one(
            "SELECT resolved_value FROM outcomes WHERE market_slug=?", (slug,)
        )
        if not resolved or resolved["resolved_value"] is None:
            continue
        books = db.query(
            "SELECT captured_at, best_ask FROM book_snapshots "
            "WHERE market_slug=? AND best_ask IS NOT NULL ORDER BY captured_at ASC",
            (slug,),
        )
        if not books:
            continue
        markets_considered += 1
        for book in books:
            when = book["captured_at"]
            feats = features.build(slug, when, decision_price=book["best_ask"])
            order = trader.consider(
                slug, feats, best_ask=book["best_ask"], fee_rate=fee_rate, opened_at=when
            )
            if order is not None:
                break  # one position per market; settle below
        trader.settle(slug, float(resolved["resolved_value"]))

    summary = paper_summary(db, SOURCE)
    summary["markets_with_book"] = markets_considered
    return summary


def paper_summary(db: Database, source: str) -> dict:
    """Aggregate the paper ledger for a given source ('backtest' | 'live')."""
    rows = db.query(
        "SELECT * FROM paper_positions WHERE source=? ORDER BY opened_at ASC", (source,)
    )
    settled = [r for r in rows if r["status"] == "settled" and r["realized_pnl"] is not None]
    wins = [r for r in settled if r["realized_pnl"] > 0]
    gross = sum((r["settled_value"] - r["entry_price"]) * r["qty"] for r in settled)
    fees = sum(r["entry_fee"] for r in settled)
    net = sum(r["realized_pnl"] for r in settled)
    return {
        "source": source,
        "positions": len(rows),
        "open": sum(1 for r in rows if r["status"] == "open"),
        "settled": len(settled),
        "wins": len(wins),
        "win_rate": (len(wins) / len(settled)) if settled else None,
        "gross_pnl": round(gross, 4),
        "fees": round(fees, 4),
        "net_pnl": round(net, 4),
        "avg_edge": round(sum(r["edge"] or 0 for r in rows) / len(rows), 4) if rows else None,
    }
