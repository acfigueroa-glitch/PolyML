"""Tests for the executor: paper fills, round-trip recording, kill switch, and
the live-trading double opt-in safety guard."""

from polyml.storage.db import Database
from polyml.storage.models import OrderBook
from polyml.trading.executor import LIVE_ENV_FLAG, Executor, ExecutorConfig
from polyml.trading.strategy import ENTER, EXIT, Action


def _book(bid, ask):
    return OrderBook.from_payload(
        {"marketData": {"marketSlug": "m1",
                        "bids": [{"px": {"value": str(bid)}, "qty": "100"}],
                        "offers": [{"px": {"value": str(ask)}, "qty": "100"}]}}
    )


def _ex(tmp_path, **cfg):
    db = Database(tmp_path / "t.db")
    sid = db.get_or_open_session("m1")
    return Executor(rest=None, db=db, config=ExecutorConfig(**cfg), session_for=lambda s: sid), db


def test_paper_round_trip_records_trade(tmp_path):
    ex, db = _ex(tmp_path)
    enter = Action(ENTER, "buy", side="ORDER_INTENT_BUY_LONG", price=0.50, shares=1.0)
    pos = ex.execute(enter, _book(0.49, 0.50), "m1", features={"mid": 0.495})
    assert pos is not None and ex.has_position("m1")

    exit_ = Action(EXIT, "take profit", side="ORDER_INTENT_SELL_LONG", price=0.60, shares=1.0)
    trade = ex.execute(exit_, _book(0.60, 0.61), "m1")
    assert trade is not None and not ex.has_position("m1")
    assert trade.net_pnl > 0

    row = db.query_one("SELECT * FROM bot_trades")
    assert row["label_good"] == 1
    assert row["entry_price"] == 0.50 and row["exit_price"] == 0.60
    assert db.query_one("SELECT COUNT(*) AS n FROM bot_orders")["n"] == 2
    db.close()


def test_only_one_position_at_a_time(tmp_path):
    ex, _ = _ex(tmp_path, max_open_positions=1)
    enter = Action(ENTER, "buy", side="ORDER_INTENT_BUY_LONG", price=0.50, shares=1.0)
    ex.execute(enter, _book(0.49, 0.50), "m1", features={})
    again = ex.execute(enter, _book(0.49, 0.50), "m1", features={})
    assert again is None  # already holding


def test_daily_loss_kill_switch(tmp_path):
    ex, _ = _ex(tmp_path, daily_loss_limit_usd=0.05)
    # Enter at 0.50, exit at 0.40 -> a loss beyond the 0.05 limit.
    ex.execute(Action(ENTER, "buy", side="ORDER_INTENT_BUY_LONG", price=0.50, shares=1.0),
               _book(0.49, 0.50), "m1", features={})
    ex.execute(Action(EXIT, "stop", side="ORDER_INTENT_SELL_LONG", price=0.40, shares=1.0),
               _book(0.40, 0.41), "m1")
    assert ex.halted is True
    # A new entry is now refused.
    blocked = ex.execute(Action(ENTER, "buy", side="ORDER_INTENT_BUY_LONG", price=0.50, shares=1.0),
                         _book(0.49, 0.50), "m1", features={})
    assert blocked is None


def test_live_requested_without_env_falls_back_to_paper(tmp_path, monkeypatch):
    monkeypatch.delenv(LIVE_ENV_FLAG, raising=False)
    ex, _ = _ex(tmp_path, mode="live")
    assert ex.live is False and ex.mode == "paper"
