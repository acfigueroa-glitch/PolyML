"""Feature engineering.

Given a point in time and a market, reconstruct the market state *as you would
have seen it* and turn it into a numeric feature vector. These are the
"indicators" the learner reasons about — the things that, in hindsight, you may
have over- or under-weighted.

Features are computed only from data observed at-or-before the decision time, so
there is no look-ahead leakage into the labels.
"""

from __future__ import annotations

import logging
from typing import Any

from polyml.storage.db import Database

logger = logging.getLogger(__name__)

# The canonical, ordered feature list. Keep stable so models stay comparable.
FEATURE_NAMES = [
    "best_bid",
    "best_ask",
    "mid",
    "spread",
    "book_imbalance",
    "last_trade_px",
    "open_interest",
    "momentum_30s",
    "momentum_5m",
    "volatility_5m",
    "price_vs_mid",       # decision price minus mid (edge taken / paid)
    "net_position",       # your position size at the time (signed)
    "minutes_since_open", # how far into the session
]


class FeatureBuilder:
    def __init__(self, db: Database) -> None:
        self.db = db

    def _latest_book_before(self, slug: str, when: str) -> Any:
        return self.db.query_one(
            "SELECT * FROM book_snapshots WHERE market_slug=? AND captured_at<=? "
            "ORDER BY captured_at DESC LIMIT 1",
            (slug, when),
        )

    def _book_at_offset(self, slug: str, when: str, seconds_before: int) -> Any:
        """The book snapshot closest to ``seconds_before`` the decision time."""
        return self.db.query_one(
            "SELECT * FROM book_snapshots WHERE market_slug=? "
            "AND captured_at <= datetime(?, ?) ORDER BY captured_at DESC LIMIT 1",
            (slug, when, f"-{seconds_before} seconds"),
        )

    def _recent_books(self, slug: str, when: str, window_seconds: int) -> list[Any]:
        return self.db.query(
            "SELECT mid FROM book_snapshots WHERE market_slug=? "
            "AND captured_at BETWEEN datetime(?, ?) AND ? AND mid IS NOT NULL "
            "ORDER BY captured_at ASC",
            (slug, when, f"-{window_seconds} seconds", when),
        )

    def _net_position_before(self, slug: str, when: str) -> float | None:
        row = self.db.query_one(
            "SELECT net_position FROM position_snapshots WHERE market_slug=? AND captured_at<=? "
            "ORDER BY captured_at DESC LIMIT 1",
            (slug, when),
        )
        return float(row["net_position"]) if row and row["net_position"] is not None else None

    def _session_start(self, slug: str, when: str) -> str | None:
        row = self.db.query_one(
            "SELECT started_at FROM sessions WHERE market_slug=? AND started_at<=? "
            "ORDER BY started_at DESC LIMIT 1",
            (slug, when),
        )
        return row["started_at"] if row else None

    @staticmethod
    def _momentum(books: list[Any]) -> float | None:
        mids = [r["mid"] for r in books if r["mid"] is not None]
        if len(mids) < 2:
            return None
        return mids[-1] - mids[0]

    @staticmethod
    def _volatility(books: list[Any]) -> float | None:
        mids = [r["mid"] for r in books if r["mid"] is not None]
        if len(mids) < 2:
            return None
        mean = sum(mids) / len(mids)
        var = sum((m - mean) ** 2 for m in mids) / len(mids)
        return var ** 0.5

    def build(self, slug: str, when: str, *, decision_price: float | None = None) -> dict[str, float]:
        """Return a feature dict for a decision on ``slug`` at time ``when``."""
        features: dict[str, float] = {name: 0.0 for name in FEATURE_NAMES}

        book = self._latest_book_before(slug, when)
        if book:
            for key in ("best_bid", "best_ask", "mid", "spread", "book_imbalance",
                        "last_trade_px", "open_interest"):
                val = book[key]
                if val is not None:
                    features[key] = float(val)

        mid = features.get("mid") or None
        if decision_price is not None and mid:
            features["price_vs_mid"] = float(decision_price) - mid

        win_30s = self._recent_books(slug, when, 30)
        win_5m = self._recent_books(slug, when, 300)
        m30 = self._momentum(win_30s)
        m5 = self._momentum(win_5m)
        vol = self._volatility(win_5m)
        if m30 is not None:
            features["momentum_30s"] = m30
        if m5 is not None:
            features["momentum_5m"] = m5
        if vol is not None:
            features["volatility_5m"] = vol

        net = self._net_position_before(slug, when)
        if net is not None:
            features["net_position"] = net

        start = self._session_start(slug, when)
        if start:
            delta = self.db.query_one(
                "SELECT (julianday(?) - julianday(?)) * 24 * 60 AS minutes", (when, start)
            )
            if delta and delta["minutes"] is not None:
                features["minutes_since_open"] = max(0.0, float(delta["minutes"]))

        return features

    @staticmethod
    def vectorize(features: dict[str, float]) -> list[float]:
        """Turn a feature dict into an ordered vector matching FEATURE_NAMES."""
        return [float(features.get(name, 0.0)) for name in FEATURE_NAMES]
