"""Tests pinning the *real* Polymarket US payload shapes (verified against the
live API): nested trade executions and position-resolution settlement.
"""

from polyml.mirror import ActivityPoller
from polyml.session import SessionManager
from polyml.storage.db import Database


def _poller(db: Database) -> ActivityPoller:
    # rest is unused by _store; pass None.
    return ActivityPoller(rest=None, db=db)


def test_trade_execution_is_parsed_from_nested_shape(tmp_path):
    db = Database(tmp_path / "t.db")
    activity = {
        "type": "ACTIVITY_TYPE_TRADE",
        "trade": {
            "id": "AHNDQ4FBA25W",
            "aggressorExecution": {
                "id": "AHNDQ4FBC25W",
                "order": {
                    "marketSlug": "aec-atp-talgri-botzan-2026-06-08",
                    "intent": "ORDER_INTENT_BUY_SHORT",
                    "avgPx": {"value": "0.8000", "currency": "USD"},
                },
                "lastShares": "10.0000",
                "lastPx": {"value": "0.8000", "currency": "USD"},
                "type": "EXECUTION_TYPE_FILL",
                "transactTime": "2026-06-09T13:15:13.769981337Z",
            },
            "passiveExecution": None,
        },
    }
    _poller(db)._store(activity)
    row = db.query_one("SELECT * FROM activities WHERE activity_id='AHNDQ4FBA25W'")
    assert row is not None
    assert row["market_slug"] == "aec-atp-talgri-botzan-2026-06-08"
    assert row["price"] == 0.8
    assert row["qty"] == 10.0
    assert row["is_aggressor"] == 1
    assert round(row["cost_basis"], 2) == 8.0
    db.close()


def test_position_resolution_sets_outcome_and_concludes(tmp_path):
    db = Database(tmp_path / "t.db")
    sm = SessionManager(db)
    sid = sm.open_session("aec-atp-piebas-raubra-2026-06-08")

    poller = ActivityPoller(rest=None, db=db, on_resolution=sm.handle_resolution)
    activity = {
        "type": "ACTIVITY_TYPE_POSITION_RESOLUTION",
        "positionResolution": {
            "marketSlug": "aec-atp-piebas-raubra-2026-06-08",
            "beforePosition": {"netPosition": "6"},
            "afterPosition": {
                "netPosition": "0",
                "realized": {"value": "-2.3400", "currency": "USD"},
                "cashValue": {"value": "0.0000", "currency": "USD"},
            },
            "updateTime": "2026-06-08T15:55:45Z",
            # The long instrument (outcomes[0]) settled to 0 -> resolved_value 0.
            "market": {"outcomePrices": ["0", "1"]},
        },
    }
    poller._store(activity)

    outcome = db.query_one(
        "SELECT resolved_value FROM outcomes WHERE market_slug=?",
        ("aec-atp-piebas-raubra-2026-06-08",),
    )
    assert outcome["resolved_value"] == 0.0

    # The resolution should have concluded the session with the authoritative PnL.
    session = db.query_one("SELECT status, realized_pnl FROM sessions WHERE id=?", (sid,))
    assert session["status"] == "concluded"
    assert session["realized_pnl"] == -2.34
    db.close()
