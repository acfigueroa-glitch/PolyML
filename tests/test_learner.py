"""Tests for the learner, including the small-data heuristic fallback and a full
training run on synthetic, learnable data."""

import json

from polyml.analysis.features import FEATURE_NAMES
from polyml.analysis.learner import Learner
from polyml.storage.db import Database


def _insert_decision(db: Database, features: dict, label: int, idx: int, session_id: int = 1) -> None:
    db.insert_decision(
        session_id=session_id,
        market_slug="m1",
        decision_type="entry",
        side="BUY",
        decided_at=f"2024-01-15T10:{idx:02d}:00Z",
        price=0.5,
        size=1.0,
        features=features,
        label_pnl=1.0 if label else -1.0,
        label_good=label,
    )


def test_heuristic_when_too_few_decisions(tmp_path):
    db = Database(tmp_path / "t.db")
    sid = db.get_or_open_session("m1")
    for i in range(4):
        _insert_decision(db, {n: 0.0 for n in FEATURE_NAMES}, i % 2, i, session_id=sid)
    result = Learner(db, min_decisions=30).train()
    assert result.trained is False
    assert result.model == "heuristic"
    assert db.query_one("SELECT COUNT(*) AS n FROM learning_runs")["n"] == 1
    db.close()


def test_trains_on_learnable_signal(tmp_path):
    db = Database(tmp_path / "t.db")
    sid = db.get_or_open_session("m1")
    # book_imbalance perfectly predicts the label: positive -> good, negative -> bad.
    for i in range(80):
        good = i % 2
        feats = {n: 0.0 for n in FEATURE_NAMES}
        feats["book_imbalance"] = 0.5 if good else -0.5
        feats["momentum_5m"] = 0.01 * (1 if good else -1)
        _insert_decision(db, feats, good, i, session_id=sid)

    result = Learner(db, model="gradient_boosting", min_decisions=30).train()
    assert result.trained is True
    assert result.metrics["accuracy"] >= 0.8
    # The predictive feature should carry meaningful importance.
    assert result.feature_importances.get("book_imbalance", 0) > 0

    # Persisted run should be reloadable.
    row = db.query_one("SELECT feature_importances FROM learning_runs ORDER BY id DESC LIMIT 1")
    assert "book_imbalance" in json.loads(row["feature_importances"])

    # Advisory scoring works once trained.
    score = result and Learner(db).score_decision  # sanity: method exists
    assert callable(score)
    db.close()
