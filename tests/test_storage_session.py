"""Tests for storage, session lifecycle, and outcome linkage."""

from polyml.analysis.outcomes import OutcomeLinker
from polyml.session import SessionManager
from polyml.storage.db import Database
from polyml.storage.models import OrderBook


def _db(tmp_path) -> Database:
    return Database(tmp_path / "test.db")


def test_writes_from_worker_threads(tmp_path):
    """The async runner writes from asyncio.to_thread workers, so the Database
    must tolerate cross-thread access (regression test for a silent failure)."""
    import concurrent.futures

    db = _db(tmp_path)

    def writer(i: int) -> None:
        db.insert_balance(float(i), float(i), float(i), raw={"i": i})

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(writer, range(50)))

    assert db.query_one("SELECT COUNT(*) AS n FROM balance_snapshots")["n"] == 50
    db.close()


def test_insert_and_count(tmp_path):
    db = _db(tmp_path)
    book = OrderBook.from_payload(
        {"marketData": {"marketSlug": "m1", "bids": [{"px": {"value": "0.5"}, "qty": "10"}],
                        "offers": [{"px": {"value": "0.52"}, "qty": "10"}]}}
    )
    db.insert_book_snapshot(book, source="rest", raw={"x": 1})
    db.insert_balance(100.0, 250.0, 100.0, raw={})
    assert db.query_one("SELECT COUNT(*) AS n FROM book_snapshots")["n"] == 1
    assert db.query_one("SELECT COUNT(*) AS n FROM balance_snapshots")["n"] == 1
    db.close()


def test_session_open_is_idempotent(tmp_path):
    db = _db(tmp_path)
    sm = SessionManager(db)
    a = sm.open_session("m1")
    b = sm.open_session("m1")
    assert a == b
    assert db.query_one("SELECT COUNT(*) AS n FROM sessions")["n"] == 1
    db.close()


def test_conclude_on_resolution(tmp_path):
    db = _db(tmp_path)
    concluded = []
    sm = SessionManager(db, on_conclude=lambda sid, slug: concluded.append((sid, slug)))
    sid = sm.open_session("m1")
    # No outcome yet -> no conclusion.
    assert sm.conclude_if_resolved("m1") is None
    db.insert_outcome("m1", resolved_value=1.0, resolution_time="2024-01-15T12:00:00Z", raw={})
    assert sm.conclude_if_resolved("m1") == sid
    assert concluded == [(sid, "m1")]
    row = db.query_one("SELECT status, outcome_value FROM sessions WHERE id=?", (sid,))
    assert row["status"] == "concluded"
    assert row["outcome_value"] == 1.0
    db.close()


def test_outcome_linker_labels_and_counterfactual(tmp_path):
    db = _db(tmp_path)
    sm = SessionManager(db)
    sid = sm.open_session("m1")

    # A book snapshot so feature building has something to read.
    book = OrderBook.from_payload(
        {"marketData": {"marketSlug": "m1", "bids": [{"px": {"value": "0.58"}, "qty": "50"}],
                        "offers": [{"px": {"value": "0.60"}, "qty": "200"}]}}
    )
    db.insert_book_snapshot(book, source="rest", raw={})

    # You bought YES at 0.60, and it resolved to 1.0 -> a good entry.
    db.insert_activity(
        activity_id="t1", activity_type="ACTIVITY_TYPE_TRADE", market_slug="m1",
        price=0.60, qty=10.0, is_aggressor=1, cost_basis=6.0, realized_pnl=None,
        create_time="2024-01-15T11:00:00Z", raw={"trade": {"intent": "ORDER_INTENT_BUY_LONG"}},
    )
    db.insert_outcome("m1", resolved_value=1.0, resolution_time="2024-01-15T12:00:00Z", raw={})
    sm.conclude_if_resolved("m1")

    report = OutcomeLinker(db).link_session(sid, "m1")
    assert report["n_decisions"] == 1
    d = report["decisions"][0]
    assert d["decision_type"] == "entry"
    assert d["label_good"] == 1
    assert d["counterfactual"] and "Good entry" in d["counterfactual"]
    # Decision row persisted.
    assert db.query_one("SELECT COUNT(*) AS n FROM decisions")["n"] == 1
    db.close()
