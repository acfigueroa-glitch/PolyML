"""Link your decisions to actual outcomes.

When a session concludes, ``OutcomeLinker`` walks the trades you made in that
market, reconstructs the market state at each decision, labels the decision as
good/bad given the realized PnL and the final resolution, and writes a
``decisions`` row. It also produces a human-readable session report:

  * What you did (entries / exits, prices, sizes).
  * What actually happened (resolution, realized PnL).
  * Counterfactuals — "which choice might have been preferable?" (e.g. holding
    to resolution vs exiting early).
  * Which indicators were pointing the other way and may have been overlooked.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from polyml.analysis.features import FEATURE_NAMES, FeatureBuilder
from polyml.storage.db import Database

logger = logging.getLogger(__name__)


def _is_buy(side: str | None) -> bool:
    s = (side or "").upper()
    if "SELL" in s or "SHORT" in s:
        return False
    return "BUY" in s or "LONG" in s


def _side_from_raw(raw: Any) -> str | None:
    """Extract a side/intent string from a stored activity (raw is JSON text)."""
    obj = raw
    if isinstance(raw, str):
        try:
            obj = json.loads(raw)
        except (ValueError, TypeError):
            return None
    if not isinstance(obj, dict):
        return None
    trade = obj.get("trade", obj)
    for key in ("side", "intent", "orderIntent", "aggressorSide"):
        if isinstance(trade, dict) and trade.get(key):
            return str(trade[key])
    return None


def _classify(prev_pos: float, qty_signed: float) -> str:
    """Classify a trade as entry/add/reduce/exit based on position movement."""
    new_pos = prev_pos + qty_signed
    if prev_pos == 0:
        return "entry"
    if abs(new_pos) > abs(prev_pos):
        return "add"
    if abs(new_pos) < abs(prev_pos):
        return "exit" if abs(new_pos) < 1e-9 else "reduce"
    return "hold"


class OutcomeLinker:
    def __init__(self, db: Database) -> None:
        self.db = db
        self.features = FeatureBuilder(db)

    def _trades(self, slug: str) -> list[Any]:
        return self.db.query(
            "SELECT * FROM activities WHERE market_slug=? AND activity_type LIKE '%TRADE%' "
            "AND create_time IS NOT NULL ORDER BY create_time ASC",
            (slug,),
        )

    def _resolved_value(self, slug: str) -> float | None:
        row = self.db.query_one("SELECT resolved_value FROM outcomes WHERE market_slug=?", (slug,))
        return float(row["resolved_value"]) if row and row["resolved_value"] is not None else None

    def link_session(self, session_id: int, slug: str) -> dict[str, Any]:
        """Build decision rows and a report for one concluded session."""
        resolved = self._resolved_value(slug)
        trades = self._trades(slug)
        decisions: list[dict[str, Any]] = []
        position = 0.0
        total_pnl = 0.0

        for trade in trades:
            raw = trade["raw"]
            price = trade["price"]
            qty = trade["qty"] or 0.0
            side = _side_from_raw(raw)
            signed = qty if _is_buy(side) else -qty
            decision_type = _classify(position, signed)
            when = trade["create_time"]

            feats = self.features.build(slug, when, decision_price=price)
            realized = trade["realized_pnl"]
            if realized is not None:
                total_pnl += realized

            label_good = self._label(decision_type, side, price, realized, resolved)

            self.db.insert_decision(
                session_id=session_id,
                market_slug=slug,
                decision_type=decision_type,
                side=side,
                decided_at=when,
                price=price,
                size=qty,
                features=feats,
                label_pnl=realized,
                label_good=label_good,
            )
            decisions.append(
                {
                    "decision_type": decision_type,
                    "side": side,
                    "price": price,
                    "size": qty,
                    "decided_at": when,
                    "realized_pnl": realized,
                    "label_good": label_good,
                    "counterfactual": self._counterfactual(decision_type, side, price, resolved),
                    "overlooked": self._overlooked_indicators(feats, label_good),
                }
            )
            position += signed

        report = self._build_report(slug, resolved, total_pnl, decisions)
        return report

    # --- labelling & counterfactuals --------------------------------------------
    @staticmethod
    def _label(decision_type, side, price, realized, resolved) -> int | None:
        """1 = good decision, 0 = bad, None = undetermined."""
        if realized is not None and decision_type in ("exit", "reduce"):
            return 1 if realized >= 0 else 0
        if resolved is None or price is None:
            return None
        # Fair value at resolution is `resolved` for a YES/long position.
        fair = resolved if _is_buy(side) else (1.0 - resolved)
        # Buying below fair (or selling above) is a good decision.
        if _is_buy(side):
            return 1 if fair >= price else 0
        return 1 if price >= (1.0 - fair) else 0

    @staticmethod
    def _counterfactual(decision_type, side, price, resolved) -> str | None:
        if resolved is None or price is None:
            return None
        if decision_type in ("exit", "reduce") and _is_buy(side) is False:
            # Sold a long position at `price`; holding would have paid `resolved`.
            diff = resolved - price
            if diff > 0.01:
                return (
                    f"Exiting at {price:.2f} left {diff:.2f}/share on the table — "
                    f"the market resolved to {resolved:.2f}. Holding would have been better."
                )
            if diff < -0.01:
                return f"Good exit: you sold at {price:.2f}; it resolved to {resolved:.2f}."
        if decision_type in ("entry", "add") and _is_buy(side):
            edge = resolved - price
            if edge > 0.01:
                return f"Strong entry: bought at {price:.2f}, resolved {resolved:.2f} (+{edge:.2f}/share)."
            if edge < -0.01:
                return (
                    f"Entry at {price:.2f} was above the {resolved:.2f} resolution — "
                    f"the position lost {abs(edge):.2f}/share."
                )
        return None

    def _overlooked_indicators(self, feats: dict[str, float], label_good: int | None) -> list[str]:
        """For a bad decision, surface signals that were pointing the other way."""
        if label_good != 0:
            return []
        flags: list[str] = []
        imb = feats.get("book_imbalance", 0.0)
        if imb <= -0.3:
            flags.append(f"order book was sell-heavy (imbalance {imb:+.2f})")
        elif imb >= 0.3:
            flags.append(f"order book was buy-heavy (imbalance {imb:+.2f})")
        if feats.get("momentum_5m", 0.0) <= -0.03:
            flags.append(f"5m price momentum was falling ({feats['momentum_5m']:+.3f})")
        elif feats.get("momentum_5m", 0.0) >= 0.03:
            flags.append(f"5m price momentum was rising ({feats['momentum_5m']:+.3f})")
        if feats.get("spread", 0.0) >= 0.05:
            flags.append(f"spread was wide ({feats['spread']:.3f}) — thin / uncertain pricing")
        if feats.get("volatility_5m", 0.0) >= 0.03:
            flags.append(f"price was volatile ({feats['volatility_5m']:.3f}) around the decision")
        return flags

    # --- report ------------------------------------------------------------------
    def _build_report(self, slug, resolved, total_pnl, decisions) -> dict[str, Any]:
        good = sum(1 for d in decisions if d["label_good"] == 1)
        bad = sum(1 for d in decisions if d["label_good"] == 0)
        lessons: list[str] = []
        for d in decisions:
            if d["counterfactual"]:
                lessons.append(d["counterfactual"])
            for o in d["overlooked"]:
                lessons.append(f"At your {d['decision_type']} ({d['decided_at']}): {o}")
        return {
            "market_slug": slug,
            "resolved_value": resolved,
            "realized_pnl": round(total_pnl, 4),
            "n_decisions": len(decisions),
            "good_decisions": good,
            "bad_decisions": bad,
            "decisions": decisions,
            "lessons": lessons,
        }
