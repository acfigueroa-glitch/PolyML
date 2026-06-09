"""Integration test for the TradeEngine's testable core: enter on a favourable
book, exit on profit, and self-analyze the game at the end."""

import base64

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from polyml.config import load_config
from polyml.storage.models import OrderBook


@pytest.fixture()
def engine(tmp_path, monkeypatch):
    # Provide a valid generated key so the engine constructs without network.
    seed = Ed25519PrivateKey.generate().private_bytes_raw()
    monkeypatch.setenv("POLYMARKET_KEY_ID", "test-key")
    monkeypatch.setenv("POLYMARKET_SECRET_KEY", base64.b64encode(seed).decode())
    monkeypatch.setenv("POLYML_DB_PATH", str(tmp_path / "engine.db"))
    from polyml.trading.engine import TradeEngine
    eng = TradeEngine(load_config())
    yield eng
    eng.db.close()


def _book(bid, bid_sz, ask, ask_sz):
    return OrderBook.from_payload(
        {"marketData": {"marketSlug": "g1",
                        "bids": [{"px": {"value": str(bid)}, "qty": str(bid_sz)}],
                        "offers": [{"px": {"value": str(ask)}, "qty": str(ask_sz)}]}}
    )


def test_engine_enters_then_exits_and_self_analyzes(engine):
    # Favourable entry book: tight spread, strong buy pressure.
    pos = engine.on_book("g1", _book(0.49, 300, 0.50, 80))
    assert pos is not None
    assert engine.executor.has_position("g1")

    # Price rises -> profitable exit.
    trade = engine.on_book("g1", _book(0.60, 300, 0.61, 80))
    assert trade is not None
    assert not engine.executor.has_position("g1")
    assert trade.net_pnl > 0

    # Game ends -> self-analysis report.
    report = engine.finalize_game("g1", book=_book(0.60, 300, 0.61, 80))
    assert report["n_trades"] == 1
    assert report["wins"] == 1 and report["losses"] == 0
    assert report["net_pnl"] > 0
    assert "learning" in report


def test_engine_flattens_open_position_at_game_end(engine):
    engine.on_book("g1", _book(0.49, 300, 0.50, 80))
    assert engine.executor.has_position("g1")
    # Finalize while still holding and underwater -> force-flatten records a trade.
    report = engine.finalize_game("g1", book=_book(0.45, 300, 0.46, 80))
    assert not engine.executor.has_position("g1")
    assert report["n_trades"] == 1
    assert report["losses"] == 1
