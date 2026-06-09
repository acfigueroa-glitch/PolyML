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
    """True if the order acquired its instrument (BUY action), regardless of
    whether the instrument is the long or short side."""
    s = (side or "").upper()
    if "BUY" in s:
        return True
    if "SELL" in s:
        return False
    # Bare intents without an explicit action: LONG implies buy-long.
    return "LONG" in s and "SELL" not in s


def _is_short(side: str | None) -> bool:
    return "SHORT" in (side or "").upper()


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
    # Real trades nest the order under aggressorExecution / passiveExecution.
    if isinstance(trade, dict):
        execution = trade.get("aggressorExecution") or trade.get("passiveExecution") or {}
        order = execution.get("order", {}) if isinstance(execution, dict) else {}
        for source in (order, trade):
            for key in ("intent", "side", "action", "orderIntent"):
                if isinstance(source, dict) and source.get(key):
                    return str(source[key])
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
        total_est_fee = 0.0
        total_actual_fee = 0.0
        saw_actual_fee = False

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

            est_fee = trade["est_fee"] if "est_fee" in trade.keys() else None
            actual_fee = trade["actual_fee"] if "actual_fee" in trade.keys() else None
            total_est_fee += est_fee or 0.0
            if actual_fee is not None:
                total_actual_fee += actual_fee
                saw_actual_fee = True

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
                    "est_fee": est_fee,
                    "actual_fee": actual_fee,
                    "label_good": label_good,
                    "counterfactual": self._counterfactual(decision_type, side, price, resolved),
                    "overlooked": self._overlooked_indicators(feats, label_good),
                }
            )
            position += signed

        # Authoritative session PnL = the resolution's final realized figure
        # (set on the session row at conclusion); the per-trade sum omits the
        # settlement of shares held to resolution, so prefer the session value.
        srow = self.db.query_one("SELECT realized_pnl FROM sessions WHERE id=?", (session_id,))
        session_pnl = (
            float(srow["realized_pnl"]) if srow and srow["realized_pnl"] is not None else total_pnl
        )
        fees = {
            "estimated": round(total_est_fee, 5),
            "actual": round(total_actual_fee, 5) if saw_actual_fee else None,
        }
        report = self._build_report(
            slug, resolved, session_pnl, decisions, intra_trade_pnl=total_pnl, fees=fees
        )
        return report

    # --- labelling & counterfactuals --------------------------------------------
    @staticmethod
    def _instrument_settle(side: str | None, resolved: float | None) -> float | None:
        """Settlement (0..1) of the instrument this trade was on. ``resolved`` is
        the long side's settled value; the short instrument settles to 1-that."""
        if resolved is None:
            return None
        return (1.0 - resolved) if _is_short(side) else resolved

    @staticmethod
    def _label(decision_type, side, price, realized, resolved) -> int | None:
        """1 = good decision, 0 = bad, None = undetermined.

        Realized PnL is authoritative when present; otherwise we compare the fill
        price to the instrument's settlement (a buy is good below settlement, a
        sell is good above it).
        """
        if realized is not None and abs(realized) > 1e-9:
            return 1 if realized > 0 else 0
        settle = OutcomeLinker._instrument_settle(side, resolved)
        if settle is None or price is None:
            return None
        if _is_buy(side):
            return 1 if settle >= price else 0
        return 1 if price >= settle else 0

    @staticmethod
    def _counterfactual(decision_type, side, price, resolved) -> str | None:
        settle = OutcomeLinker._instrument_settle(side, resolved)
        if settle is None or price is None:
            return None
        if _is_buy(side):
            edge = settle - price
            if edge > 0.01:
                return f"Good entry: bought at {price:.2f}, instrument settled {settle:.2f} (+{edge:.2f}/share)."
            if edge < -0.01:
                return (
                    f"Entry at {price:.2f} was above the {settle:.2f} settlement — "
                    f"lost {abs(edge):.2f}/share. The position never had fair-value edge."
                )
        else:  # a sell / exit
            diff = settle - price
            if diff > 0.01:
                return (
                    f"Exiting at {price:.2f} left {diff:.2f}/share on the table — "
                    f"the instrument settled to {settle:.2f}. Holding would have been better."
                )
            if diff < -0.01:
                return f"Good exit: sold at {price:.2f}; it settled to {settle:.2f}."
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
    def _build_report(
        self, slug, resolved, session_pnl, decisions, *, intra_trade_pnl=0.0, fees=None
    ) -> dict[str, Any]:
        fees = fees or {"estimated": 0.0, "actual": None}
        good = sum(1 for d in decisions if d["label_good"] == 1)
        bad = sum(1 for d in decisions if d["label_good"] == 0)
        lessons: list[str] = []
        # Cap the lesson list so high-frequency sessions stay readable.
        for d in decisions:
            if d["counterfactual"]:
                lessons.append(d["counterfactual"])
            for o in d["overlooked"]:
                lessons.append(f"At your {d['decision_type']} ({d['decided_at']}): {o}")
        # Fees are pure drag: flag them when they materially erode the result.
        paid_fee = fees["actual"] if fees.get("actual") is not None else fees.get("estimated", 0.0)
        if paid_fee and abs(session_pnl) > 1e-9 and paid_fee >= 0.1 * abs(session_pnl):
            lessons.append(
                f"Fees ({'actual' if fees.get('actual') is not None else 'est.'} ${paid_fee:.2f}) "
                f"were a sizeable share of the ${abs(session_pnl):.2f} result — taker fills are "
                f"costliest near 50% (fee = contracts x rate x price x (1-price))."
            )
        return {
            "market_slug": slug,
            "resolved_value": resolved,
            "realized_pnl": round(session_pnl, 4),
            "intra_session_realized": round(intra_trade_pnl, 4),
            "fees": fees,
            "n_decisions": len(decisions),
            "good_decisions": good,
            "bad_decisions": bad,
            "decisions": decisions,
            "lessons": lessons[:40],
        }
