"""Paper trading: fee-aware decisions and P&L, with no real orders."""

from types import SimpleNamespace

from polyml.storage.db import Database
from polyml.trading.backtest import paper_summary, run_backtest
from polyml.trading.paper import PaperStrategy, PaperTrader


def _strategy(**kw):
    kw.setdefault("min_edge", 0.02)
    kw.setdefault("fee_rate", 0.05)
    return PaperStrategy(**kw)


def test_strategy_buys_only_when_ev_clears_the_fee_and_margin():
    s = _strategy(min_edge=0.02)
    # p=0.30, ask=0.20: fee = 0.05*0.2*0.8 = 0.008; EV = 0.30-0.20-0.008 = 0.092.
    order = s.decide("m", prob=0.30, best_ask=0.20)
    assert order is not None and order.qty == 1
    assert round(order.fee, 4) == 0.008
    assert round(order.edge, 4) == 0.092


def test_strategy_skips_when_edge_below_threshold():
    s = _strategy(min_edge=0.02)
    # p barely above price -> EV < min_edge after fee.
    assert s.decide("m", prob=0.21, best_ask=0.20) is None
    # No edge at all (p below price).
    assert s.decide("m", prob=0.10, best_ask=0.20) is None


def test_strategy_rejects_bad_inputs():
    s = _strategy()
    assert s.decide("m", prob=None, best_ask=0.2) is None
    assert s.decide("m", prob=0.9, best_ask=None) is None
    assert s.decide("m", prob=0.9, best_ask=1.0) is None   # settled price, no trade
    assert s.decide("m", prob=0.9, best_ask=0.0) is None


def test_trader_opens_one_position_and_persists(tmp_path):
    db = Database(tmp_path / "t.db")
    trader = PaperTrader(db, scorer=lambda f: 0.9, strategy=_strategy(), source="live")
    o1 = trader.consider("aec-x", {}, best_ask=0.20)
    assert o1 is not None
    # A second consideration on the same market must NOT open another position.
    o2 = trader.consider("aec-x", {}, best_ask=0.20)
    assert o2 is None
    row = db.query_one("SELECT * FROM paper_positions WHERE market_slug='aec-x'")
    assert row["status"] == "open" and row["qty"] == 1 and row["source"] == "live"
    db.close()


def test_settlement_books_pnl_net_of_fee(tmp_path):
    db = Database(tmp_path / "t.db")
    trader = PaperTrader(db, scorer=lambda f: 0.9, strategy=_strategy(), source="live")
    trader.consider("win", {}, best_ask=0.20)   # fee 0.008
    trader.consider("lose", {}, best_ask=0.20)
    trader.settle("win", 1.0)
    trader.settle("lose", 0.0)
    win = db.query_one("SELECT realized_pnl FROM paper_positions WHERE market_slug='win'")
    lose = db.query_one("SELECT realized_pnl FROM paper_positions WHERE market_slug='lose'")
    assert round(win["realized_pnl"], 3) == round((1.0 - 0.20) - 0.008, 3)   # +0.792
    assert round(lose["realized_pnl"], 3) == round((0.0 - 0.20) - 0.008, 3)  # -0.208
    db.close()


def _book(slug, when, ask):
    return SimpleNamespace(
        market_slug=slug, state="OPEN", best_bid=ask - 0.01, best_ask=ask, mid=ask - 0.005,
        spread=0.01, book_imbalance=0.0, last_trade_px=ask, open_interest=100.0,
    )


def test_backtest_trades_on_edge_and_reports_net_pnl(tmp_path):
    db = Database(tmp_path / "t.db")
    # Two resolved markets, each with book history; a confident scorer.
    for slug, settle, ask in [("aec-win", 1.0, 0.20), ("aec-lose", 0.0, 0.20)]:
        db.insert_outcome(slug, settle, "2026-06-09T00:00:00Z", raw={})
        for i in range(3):
            db.insert_book_snapshot(_book(slug, None, ask), source="rest", raw={})

    summary = run_backtest(db, scorer=lambda f: 0.9, min_edge=0.02, fee_rate=0.05)
    assert summary["markets_with_book"] == 2
    assert summary["settled"] == 2
    assert summary["wins"] == 1
    # win +0.792, lose -0.208 -> net +0.584
    assert round(summary["net_pnl"], 3) == round(0.792 - 0.208, 3)
    assert summary["fees"] > 0
    db.close()


def test_backtest_takes_no_trades_when_model_has_no_edge(tmp_path):
    db = Database(tmp_path / "t.db")
    db.insert_outcome("aec-x", 1.0, "2026-06-09T00:00:00Z", raw={})
    for _ in range(3):
        db.insert_book_snapshot(_book("aec-x", None, 0.50), source="rest", raw={})
    # Scorer returns None (untrained) -> nothing should trade.
    summary = run_backtest(db, scorer=lambda f: None, min_edge=0.02)
    assert summary["settled"] == 0 and summary["positions"] == 0
    assert paper_summary(db, "backtest")["net_pnl"] == 0.0
    db.close()
