"""The learner.

Aggregates every labelled decision across all concluded sessions and trains a
model to predict whether a decision is *good* (favourable outcome) from the
market indicators present at the time. The point isn't to autotrade — it's to
quantify which signals separate your winning decisions from your losing ones, so
the "indicators overlooked" feedback gets sharper as more sessions accumulate.

With little data it falls back to a transparent heuristic; once enough labelled
decisions exist it trains a gradient-boosting / logistic model and reports honest
holdout metrics plus feature importances.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from polyml.analysis.features import FEATURE_NAMES, FeatureBuilder
from polyml.storage.db import Database

logger = logging.getLogger(__name__)


@dataclass
class LearningResult:
    model: str
    n_decisions: int
    trained: bool
    metrics: dict[str, float] = field(default_factory=dict)
    feature_importances: dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def summary(self) -> str:
        lines = [f"Model: {self.model}  |  decisions: {self.n_decisions}  |  trained: {self.trained}"]
        if self.metrics:
            metric_str = ", ".join(f"{k}={v:.3f}" for k, v in self.metrics.items())
            lines.append(f"  metrics: {metric_str}")
        if self.feature_importances:
            top = sorted(self.feature_importances.items(), key=lambda kv: abs(kv[1]), reverse=True)[:6]
            lines.append("  top indicators: " + ", ".join(f"{k} ({v:+.3f})" for k, v in top))
        if self.notes:
            lines.append(f"  note: {self.notes}")
        return "\n".join(lines)


class Learner:
    def __init__(
        self,
        db: Database,
        *,
        model: str = "gradient_boosting",
        min_decisions: int = 30,
        holdout_fraction: float = 0.25,
        random_state: int = 42,
    ) -> None:
        self.db = db
        self.model_name = model
        self.min_decisions = min_decisions
        self.holdout_fraction = holdout_fraction
        self.random_state = random_state
        self._model: Any = None
        self._feature_builder = FeatureBuilder(db)

    # --- data --------------------------------------------------------------------
    def _load_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        rows = self.db.query(
            "SELECT features, label_good FROM decisions WHERE label_good IS NOT NULL"
        )
        X, y = [], []
        for row in rows:
            feats = json.loads(row["features"])
            X.append([float(feats.get(name, 0.0)) for name in FEATURE_NAMES])
            y.append(int(row["label_good"]))
        return np.array(X, dtype=float), np.array(y, dtype=int)

    # --- training ----------------------------------------------------------------
    def train(self) -> LearningResult:
        X, y = self._load_dataset()
        n = len(y)
        if n < self.min_decisions or len(set(y.tolist())) < 2:
            result = self._heuristic_result(X, y, n)
            self._persist(result)
            return result

        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.model_selection import train_test_split

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=self.holdout_fraction, random_state=self.random_state, stratify=y
        )

        if self.model_name == "logistic":
            model = LogisticRegression(max_iter=1000)
        else:
            model = GradientBoostingClassifier(random_state=self.random_state)
        model.fit(X_tr, y_tr)
        self._model = model

        preds = model.predict(X_te)
        metrics = {"accuracy": float(accuracy_score(y_te, preds)), "holdout_n": float(len(y_te))}
        try:
            proba = model.predict_proba(X_te)[:, 1]
            if len(set(y_te.tolist())) == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_te, proba))
        except Exception:  # noqa: BLE001
            pass

        importances = self._extract_importances(model)
        result = LearningResult(
            model=self.model_name,
            n_decisions=n,
            trained=True,
            metrics=metrics,
            feature_importances=importances,
            notes=f"Trained on {len(y_tr)} decisions, held out {len(y_te)}.",
        )
        self._persist(result)
        return result

    def _extract_importances(self, model: Any) -> dict[str, float]:
        if hasattr(model, "feature_importances_"):
            vals = model.feature_importances_
        elif hasattr(model, "coef_"):
            vals = model.coef_[0]
        else:
            return {}
        return {name: float(v) for name, v in zip(FEATURE_NAMES, vals)}

    def _heuristic_result(self, X: np.ndarray, y: np.ndarray, n: int) -> LearningResult:
        """Correlation of each feature with the good/bad label — a transparent
        stand-in until we have enough data to train a model."""
        importances: dict[str, float] = {}
        if n >= 2 and len(set(y.tolist())) == 2:
            for i, name in enumerate(FEATURE_NAMES):
                col = X[:, i]
                if np.std(col) > 0:
                    importances[name] = float(np.corrcoef(col, y)[0, 1])
        note = (
            f"Only {n} labelled decisions (need {self.min_decisions} to train). "
            "Showing feature/outcome correlations instead."
        )
        return LearningResult(
            model="heuristic",
            n_decisions=n,
            trained=False,
            feature_importances=importances,
            notes=note,
        )

    def _persist(self, result: LearningResult) -> None:
        self.db.insert_learning_run(
            model=result.model,
            n_decisions=result.n_decisions,
            metrics=result.metrics,
            feature_importances=result.feature_importances,
            notes=result.notes,
        )

    # --- inference (advisory only) ----------------------------------------------
    def score_decision(self, features: dict[str, float]) -> float | None:
        """Probability that a decision with these features is 'good'.

        Advisory only — PolyML never acts on this. Returns None if untrained.
        """
        if self._model is None:
            return None
        vec = np.array([[float(features.get(name, 0.0)) for name in FEATURE_NAMES]], dtype=float)
        try:
            return float(self._model.predict_proba(vec)[0, 1])
        except Exception:  # noqa: BLE001
            return None
