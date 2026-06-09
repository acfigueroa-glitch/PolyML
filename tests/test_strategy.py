"""Tests for the one-share scalping strategy."""

from polyml.storage.models import OrderBook
from polyml.trading.fees import entry_cost_basis
from polyml.trading.strategy import (
    ENTER,
    EXIT,
    HOLD,
    SKIP,
    BotPosition,
    ScalpStrategy,
    StrategyConfig,
)


def _book(bid, bid_sz, ask, ask_sz):
    return OrderBook.from_payload(
        {"marketData": {"marketSlug": "m1",
                        "bids": [{"px": {"value": str(bid)}, "qty": str(bid_sz)}],
                        "offers": [{"px": {"value": str(ask)}, "qty": str(ask_sz)}]}}
    )


def _position(entry=0.50):
    return BotPosition("m1", "ORDER_INTENT_BUY_LONG", 1.0, entry, entry_cost_basis(entry, 1), "t0")


def test_enters_on_tight_favourable_book():
    # Tight spread, buy pressure, model off.
    s = ScalpStrategy(StrategyConfig(use_model_gate=False, entry_imbalance_min=-1.0))
    a = s.decide(_book(0.49, 100, 0.50, 100), None)
    assert a.kind == ENTER
    assert a.price == 0.50 and a.shares == 1.0
    assert a.target_exit is not None and a.target_exit > 0.50


def test_skips_when_spread_too_wide():
    s = ScalpStrategy(StrategyConfig(use_model_gate=False, max_spread=0.03))
    a = s.decide(_book(0.45, 100, 0.55, 100), None)
    assert a.kind == SKIP and "spread" in a.reason


def test_skips_when_no_bid_depth_for_exit():
    s = ScalpStrategy(StrategyConfig(use_model_gate=False, min_book_depth=5))
    a = s.decide(_book(0.49, 1, 0.50, 100), None)
    assert a.kind == SKIP and "depth" in a.reason


def test_model_gate_blocks_low_scores():
    s = ScalpStrategy(StrategyConfig(use_model_gate=True, model_threshold=0.6,
                                     entry_imbalance_min=-1.0))
    assert s.decide(_book(0.49, 100, 0.50, 100), None, model_score=0.4).kind == SKIP
    assert s.decide(_book(0.49, 100, 0.50, 100), None, model_score=0.9).kind == ENTER


def test_exits_on_any_profit():
    s = ScalpStrategy(StrategyConfig(profit_hurdle_usd=0.01))
    # Bought at 0.50; bid now 0.56 -> comfortably profitable.
    a = s.decide(_book(0.56, 100, 0.57, 100), _position(0.50))
    assert a.kind == EXIT and a.projected_net > 0.01


def test_holds_when_below_hurdle():
    s = ScalpStrategy(StrategyConfig(profit_hurdle_usd=0.01, stop_loss_usd=None))
    # Bid 0.51 barely above entry: net does not clear the fee+hurdle yet.
    a = s.decide(_book(0.51, 100, 0.52, 100), _position(0.50))
    assert a.kind == HOLD


def test_stop_loss_triggers():
    s = ScalpStrategy(StrategyConfig(profit_hurdle_usd=0.01, stop_loss_usd=0.05))
    # Bid collapsed to 0.40 -> net well below -0.05.
    a = s.decide(_book(0.40, 100, 0.41, 100), _position(0.50))
    assert a.kind == EXIT and a.projected_net < 0
    assert "stop loss" in a.reason


def test_flatten_before_close():
    s = ScalpStrategy(StrategyConfig(profit_hurdle_usd=0.01, stop_loss_usd=None,
                                     flatten_before_close=True, close_buffer_minutes=5))
    a = s.decide(_book(0.50, 100, 0.51, 100), _position(0.50), minutes_to_close=2.0)
    assert a.kind == EXIT and "flatten" in a.reason
