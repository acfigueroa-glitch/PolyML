"""The bot's understanding of the Polymarket US fee structure.

Verified against the live API: a taker pays
``fee = contracts * feeCoefficient * price * (1 - price)``; the per-market
``feeCoefficient`` is 0.05 for AEC sports markets, the per-fill amount is
reported as ``commissionNotionalCollected``, rounded to the cent. Makers pay
nothing.
"""

from polyml.fees import (
    fee_difference,
    fee_rate_for,
    fee_rate_from_market,
    protocol_fee,
)
from polyml.mirror import ActivityPoller
from polyml.storage.db import Database


def test_protocol_fee_matches_live_receipt():
    # Live receipt: 10 shares @ 0.20, feeCoefficient 0.05 -> commission $0.0800.
    assert protocol_fee(10, 0.20, fee_rate=0.05) == 0.08
    # And the documented form holds for an arbitrary fill.
    assert protocol_fee(96, 0.19, fee_rate=0.05) == round(96 * 0.05 * 0.19 * 0.81, 5)


def test_fee_peaks_at_fifty_percent_and_is_symmetric():
    assert protocol_fee(100, 0.5) > protocol_fee(100, 0.2)
    assert protocol_fee(100, 0.2) == protocol_fee(100, 0.8)


def test_makers_receive_a_rebate():
    # Makers are paid a small rebate (negative fee), not zero.
    fee = protocol_fee(27, 0.53, is_taker=False)
    assert fee < 0.0
    assert fee == round(-0.0107 * 27 * 0.53 * 0.47, 5)


def test_no_fee_at_or_beyond_the_bounds():
    assert protocol_fee(10, 1.0) == 0.0
    assert protocol_fee(10, 0.0) == 0.0
    assert protocol_fee(0, 0.5) == 0.0
    assert protocol_fee(None, 0.5) == 0.0


def test_tiny_fees_round_to_zero():
    # 1 * 0.05 * 0.0001 * 0.9999 = 5.0e-6 -> 0.00000.
    assert protocol_fee(1, 0.0001) == 0.0


def test_fee_rate_defaults_to_sports():
    assert fee_rate_for("sports") == 0.05
    assert fee_rate_for(None) == 0.05
    assert fee_rate_for("unknown-category") == 0.05


def test_fee_rate_read_from_market_coefficient():
    assert fee_rate_from_market({"feeCoefficient": 0.05}) == 0.05
    assert fee_rate_from_market({"feeCoefficient": "0.07"}) == 0.07
    assert fee_rate_from_market({}) == 0.05          # falls back to default
    assert fee_rate_from_market(None) == 0.05


def test_fee_difference_only_when_actual_known():
    assert fee_difference(0.08, 0.08) == 0.0
    assert fee_difference(None, 0.08) is None


def _trade_activity(trade_id, *, aggressor=True, extra_exec=None, market=None):
    exec_block = {
        "order": {"marketSlug": "aec-wta-x-y-2026-06-09", "intent": "ORDER_INTENT_BUY_LONG"},
        "lastShares": "10.0000",
        "lastPx": {"value": "0.2000", "currency": "USD"},
        "type": "EXECUTION_TYPE_FILL",
        "transactTime": "2026-06-09T19:44:14Z",
    }
    if extra_exec:
        exec_block.update(extra_exec)
    trade = {
        "id": trade_id,
        "market": market if market is not None else {"feeCoefficient": 0.05},
        "aggressorExecution": exec_block if aggressor else None,
        "passiveExecution": None if aggressor else exec_block,
    }
    return {"type": "ACTIVITY_TYPE_TRADE", "trade": trade}


def test_taker_trade_estimates_fee_from_coefficient(tmp_path):
    db = Database(tmp_path / "t.db")
    ActivityPoller(rest=None, db=db)._store(_trade_activity("FEE1"))
    row = db.query_one("SELECT * FROM activities WHERE activity_id='FEE1'")
    assert row["is_aggressor"] == 1
    assert row["est_fee"] == 0.08            # 10 * 0.05 * 0.20 * 0.80
    assert row["actual_fee"] is None         # no commission field in this fixture
    db.close()


def test_maker_trade_estimates_a_rebate(tmp_path):
    db = Database(tmp_path / "t.db")
    ActivityPoller(rest=None, db=db)._store(_trade_activity("FEE2", aggressor=False))
    row = db.query_one("SELECT * FROM activities WHERE activity_id='FEE2'")
    assert row["is_aggressor"] == 0
    assert row["est_fee"] == round(-0.0107 * 10 * 0.20 * 0.80, 5)  # rebate, negative
    db.close()


def test_actual_fee_from_commission_field_reconciles(tmp_path):
    db = Database(tmp_path / "t.db")
    # The execution reports the real per-fill commission.
    ActivityPoller(rest=None, db=db)._store(
        _trade_activity(
            "FEE3",
            extra_exec={"commissionNotionalCollected": {"value": "0.0800", "currency": "USD"}},
        )
    )
    row = db.query_one("SELECT * FROM activities WHERE activity_id='FEE3'")
    assert row["actual_fee"] == 0.08
    assert row["est_fee"] == 0.08
    assert row["fee_diff"] == 0.0
    db.close()


def test_rate_and_basis_points_are_not_mistaken_for_a_fee(tmp_path):
    # feeCoefficient (a rate) and commissionsBasisPoints (legacy, here 0) must
    # NOT be parsed as the actual fee amount.
    db = Database(tmp_path / "t.db")
    ActivityPoller(rest=None, db=db)._store(
        _trade_activity(
            "FEE4",
            market={"feeCoefficient": 0.05},
            extra_exec={
                "order": {
                    "marketSlug": "aec-wta-x-y-2026-06-09",
                    "intent": "ORDER_INTENT_BUY_LONG",
                    "commissionsBasisPoints": "0",
                    "makerCommissionsBasisPoints": "0",
                },
                "commissionSpreadPx": {"value": "0.2000", "currency": "USD"},
            },
        )
    )
    row = db.query_one("SELECT * FROM activities WHERE activity_id='FEE4'")
    # No real commission amount present -> actual_fee stays None (not 0.05/0.20).
    assert row["actual_fee"] is None
    db.close()


def test_migration_adds_fee_columns_to_existing_db(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE activities (id INTEGER PRIMARY KEY AUTOINCREMENT, activity_id TEXT, "
        "activity_type TEXT NOT NULL, market_slug TEXT, price REAL, qty REAL, "
        "is_aggressor INTEGER, cost_basis REAL, realized_pnl REAL, create_time TEXT, "
        "captured_at TEXT NOT NULL, raw TEXT NOT NULL, UNIQUE(activity_id, activity_type))"
    )
    conn.commit()
    conn.close()

    db = Database(path)
    cols = {row["name"] for row in db.query("PRAGMA table_info(activities)")}
    assert {"est_fee", "actual_fee", "fee_diff"} <= cols
    db.close()
