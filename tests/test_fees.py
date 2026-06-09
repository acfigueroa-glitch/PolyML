"""The bot's understanding of the Polymarket US fee structure.

Docs: a taker pays ``fee = contracts * fee_rate * price * (1 - price)`` (sports
rate 0.03), rounded to 5 decimals; makers pay nothing.
"""

from polyml.fees import fee_difference, fee_rate_for, protocol_fee
from polyml.mirror import ActivityPoller
from polyml.storage.db import Database


def test_protocol_fee_matches_docs_example():
    # 10 contracts sold at ~0.131 -> 10 * 0.03 * 0.131 * 0.869 = 0.03415 (5 dp).
    assert protocol_fee(10, 0.131) == round(10 * 0.03 * 0.131 * (1 - 0.131), 5)
    assert protocol_fee(10, 0.131) == 0.03415


def test_fee_peaks_at_fifty_percent_and_is_symmetric():
    # price * (1 - price) is symmetric about 0.5 and maximal there.
    assert protocol_fee(100, 0.5) > protocol_fee(100, 0.2)
    assert protocol_fee(100, 0.2) == protocol_fee(100, 0.8)


def test_makers_pay_nothing():
    assert protocol_fee(27, 0.53, is_taker=False) == 0.0


def test_no_fee_at_or_beyond_the_bounds():
    # A settled instrument (price 0 or 1) and degenerate inputs incur no fee.
    assert protocol_fee(10, 1.0) == 0.0
    assert protocol_fee(10, 0.0) == 0.0
    assert protocol_fee(0, 0.5) == 0.0
    assert protocol_fee(None, 0.5) == 0.0


def test_tiny_fees_round_to_zero():
    # A single share at a very low price rounds below the 5th decimal:
    # 1 * 0.03 * 0.0001 * 0.9999 = 3.0e-6 -> 0.00000.
    assert protocol_fee(1, 0.0001) == 0.0


def test_fee_rate_defaults_to_sports():
    assert fee_rate_for("sports") == 0.03
    assert fee_rate_for("SPORTS") == 0.03
    assert fee_rate_for(None) == 0.03
    assert fee_rate_for("unknown-category") == 0.03


def test_fee_difference_only_when_actual_known():
    assert fee_difference(0.05, 0.03) == 0.02
    assert fee_difference(None, 0.03) is None


def test_taker_trade_stores_estimated_fee(tmp_path):
    db = Database(tmp_path / "t.db")
    activity = {
        "type": "ACTIVITY_TYPE_TRADE",
        "trade": {
            "id": "FEE1",
            "aggressorExecution": {  # aggressor => taker => pays a fee
                "order": {"marketSlug": "aec-atp-x-y-2026-06-09", "intent": "ORDER_INTENT_SELL_LONG"},
                "lastShares": "10.0000",
                "lastPx": {"value": "0.1310", "currency": "USD"},
                "type": "EXECUTION_TYPE_FILL",
                "transactTime": "2026-06-09T13:15:13Z",
            },
            "passiveExecution": None,
        },
    }
    ActivityPoller(rest=None, db=db)._store(activity)
    row = db.query_one("SELECT * FROM activities WHERE activity_id='FEE1'")
    assert row["is_aggressor"] == 1
    assert row["est_fee"] == 0.03415
    assert row["actual_fee"] is None  # fixture omits a fee field
    assert row["fee_diff"] is None
    db.close()


def test_maker_trade_estimates_zero_fee(tmp_path):
    db = Database(tmp_path / "t.db")
    activity = {
        "type": "ACTIVITY_TYPE_TRADE",
        "trade": {
            "id": "FEE2",
            "aggressorExecution": None,
            "passiveExecution": {  # passive => maker => no fee
                "order": {"marketSlug": "aec-atp-x-y-2026-06-09", "intent": "ORDER_INTENT_SELL_LONG"},
                "lastShares": "10.0000",
                "lastPx": {"value": "0.1310", "currency": "USD"},
                "type": "EXECUTION_TYPE_FILL",
                "transactTime": "2026-06-09T13:15:13Z",
            },
        },
    }
    ActivityPoller(rest=None, db=db)._store(activity)
    row = db.query_one("SELECT * FROM activities WHERE activity_id='FEE2'")
    assert row["is_aggressor"] == 0
    assert row["est_fee"] == 0.0
    db.close()


def test_actual_fee_reconciled_when_receipt_reports_it(tmp_path):
    db = Database(tmp_path / "t.db")
    activity = {
        "type": "ACTIVITY_TYPE_TRADE",
        "trade": {
            "id": "FEE3",
            "aggressorExecution": {
                "order": {"marketSlug": "aec-atp-x-y-2026-06-09", "intent": "ORDER_INTENT_SELL_LONG"},
                "lastShares": "10.0000",
                "lastPx": {"value": "0.1310", "currency": "USD"},
                "fee": {"value": "0.0400", "currency": "USD"},
                "type": "EXECUTION_TYPE_FILL",
                "transactTime": "2026-06-09T13:15:13Z",
            },
            "passiveExecution": None,
        },
    }
    ActivityPoller(rest=None, db=db)._store(activity)
    row = db.query_one("SELECT * FROM activities WHERE activity_id='FEE3'")
    assert row["actual_fee"] == 0.04
    assert row["est_fee"] == 0.03415
    assert row["fee_diff"] == round(0.04 - 0.03415, 5)
    db.close()


def test_migration_adds_fee_columns_to_existing_db(tmp_path):
    # Simulate an older database whose activities table lacks the fee columns.
    import sqlite3

    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    # The prior activities schema: same columns minus the fee additions.
    conn.execute(
        "CREATE TABLE activities (id INTEGER PRIMARY KEY AUTOINCREMENT, activity_id TEXT, "
        "activity_type TEXT NOT NULL, market_slug TEXT, price REAL, qty REAL, "
        "is_aggressor INTEGER, cost_basis REAL, realized_pnl REAL, create_time TEXT, "
        "captured_at TEXT NOT NULL, raw TEXT NOT NULL, UNIQUE(activity_id, activity_type))"
    )
    conn.commit()
    conn.close()

    db = Database(path)  # opening it should add the new columns
    cols = {row["name"] for row in db.query("PRAGMA table_info(activities)")}
    assert {"est_fee", "actual_fee", "fee_diff"} <= cols
    db.close()
