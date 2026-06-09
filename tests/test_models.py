"""Tests for payload parsing and order-book metrics."""

from polyml.storage.models import OrderBook, parse_money, parse_time


def test_parse_money_handles_shapes():
    assert parse_money({"value": "0.65", "currency": "USD"}) == 0.65
    assert parse_money("0.42") == 0.42
    assert parse_money(0.5) == 0.5
    assert parse_money(None) is None
    assert parse_money({"currency": "USD"}) is None


def test_parse_time_iso_z():
    dt = parse_time("2024-01-15T10:30:00Z")
    assert dt is not None
    assert dt.year == 2024 and dt.tzinfo is not None


def test_order_book_metrics():
    payload = {
        "marketData": {
            "marketSlug": "will-team-a-win",
            "state": "MARKET_STATE_OPEN",
            "bids": [
                {"px": {"value": "0.64"}, "qty": "300"},
                {"px": {"value": "0.65"}, "qty": "100"},
            ],
            "offers": [
                {"px": {"value": "0.66"}, "qty": "100"},
            ],
            "stats": {"lastTradePx": {"value": "0.65"}, "openInterest": "50000"},
        }
    }
    book = OrderBook.from_payload(payload)
    assert book.market_slug == "will-team-a-win"
    assert book.best_bid == 0.65
    assert book.best_ask == 0.66
    assert book.mid == 0.655
    assert round(book.spread, 4) == 0.01
    # bid size 400, ask size 100 -> imbalance (400-100)/500 = 0.6
    assert round(book.book_imbalance, 3) == 0.6
    assert book.last_trade_px == 0.65
    assert book.open_interest == 50000
