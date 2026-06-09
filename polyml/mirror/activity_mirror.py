"""Translate private-stream and activity payloads into stored records.

``ActivityMirror`` is the message handler for the private WebSocket. It maps the
documented message shapes into ``order_events``, ``position_snapshots`` and
``balance_snapshots`` rows, and notifies a callback whenever a new market shows
up so the runner can extend the watchlist and open a session.

``ActivityPoller`` periodically sweeps ``/portfolio/activities`` to capture
settled trades and POSITION_RESOLUTION events (which become market outcomes).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from polyml.api.rest import PolymarketAPIError, RestClient
from polyml.fees import fee_difference, fee_rate_from_market, protocol_fee
from polyml.storage.db import Database
from polyml.storage.models import parse_decimal, parse_fee, parse_money

logger = logging.getLogger(__name__)

# Execution type -> our normalised event_type.
_EXEC_EVENT = {
    "EXECUTION_TYPE_PARTIAL_FILL": "partial_fill",
    "EXECUTION_TYPE_FILL": "filled",
    "EXECUTION_TYPE_CANCELED": "cancelled",
    "EXECUTION_TYPE_REJECTED": "rejected",
}


def _resolved_long_value(market: dict, after: dict, before: dict) -> float | None:
    """Settlement value (0..1) of the market's long/first outcome.

    Polymarket settles a binary market's outcomes to 1 (winner) and 0 (loser);
    ``outcomePrices[0]`` is the long instrument's settled price. Falls back to
    cash received per share if the market block is absent.
    """
    prices = market.get("outcomePrices")
    if isinstance(prices, list) and prices:
        try:
            return float(prices[0])
        except (TypeError, ValueError):
            pass
    cash = parse_money(after.get("cashValue"))
    net = parse_decimal(before.get("netPosition"))
    if cash is not None and net not in (None, 0.0):
        return cash / abs(net)
    return None


class ActivityMirror:
    """Async message handler for ``PrivateWebSocket``."""

    def __init__(self, db: Database, on_new_market: Callable[[str], None] | None = None) -> None:
        self.db = db
        self.on_new_market = on_new_market

    async def handle(self, message: dict[str, Any]) -> None:
        """Entry point passed to ``PrivateWebSocket(on_message=...)``."""
        sub_type = message.get("subscriptionType", "")
        # The stream emits an empty ``error`` field as a benign EOF marker on
        # snapshots; only surface non-empty errors.
        if message.get("error"):
            logger.warning("private stream error: %s", message["error"])
        if "orderSubscriptionSnapshot" in message:
            self._handle_order_snapshot(message["orderSubscriptionSnapshot"])
        elif "orderSubscriptionUpdate" in message:
            self._handle_order_update(message["orderSubscriptionUpdate"])
        elif "positionSubscriptionSnapshot" in message:
            for pos in message["positionSubscriptionSnapshot"].get("positions", []):
                self._handle_position(pos)
        elif "positionSubscription" in message:
            self._handle_position(message["positionSubscription"])
        elif "accountBalancesSnapshot" in message:
            self._handle_balance_snapshot(message["accountBalancesSnapshot"])
        elif "accountBalancesUpdate" in message:
            self._handle_balance(message["accountBalancesUpdate"])
        elif not message.get("error"):
            logger.debug("unhandled private message: %s", sub_type or list(message.keys()))

    # --- handlers ----------------------------------------------------------------
    def _note_market(self, slug: str | None) -> None:
        if slug and self.on_new_market:
            self.on_new_market(slug)

    def _handle_order_snapshot(self, snapshot: dict[str, Any]) -> None:
        for order in snapshot.get("orders", []):
            slug = order.get("marketSlug")
            self.db.insert_order_event(
                order_id=order.get("id"),
                market_slug=slug,
                side=order.get("side"),
                order_type=order.get("type"),
                price=parse_money(order.get("price")),
                quantity=parse_decimal(order.get("quantity")),
                state=order.get("state"),
                event_type="snapshot",
                source="ws",
                event_time=order.get("updateTime") or order.get("createTime"),
                raw=order,
            )
            self._note_market(slug)

    def _handle_order_update(self, update: dict[str, Any]) -> None:
        execution = update.get("execution")
        if execution:
            exec_type = execution.get("type", "")
            slug = execution.get("marketSlug")
            self.db.insert_order_event(
                order_id=execution.get("id") or execution.get("orderId"),
                market_slug=slug,
                price=parse_money(execution.get("lastPx")),
                filled_qty=parse_decimal(execution.get("lastShares")),
                state=exec_type,
                event_type=_EXEC_EVENT.get(exec_type, "execution"),
                source="ws",
                event_time=execution.get("updateTime") or execution.get("transactTime"),
                raw=update,
            )
            self._note_market(slug)
            return
        # An order state change without an execution (e.g. accepted/placed).
        order = update.get("order")
        if order:
            slug = order.get("marketSlug")
            self.db.insert_order_event(
                order_id=order.get("id"),
                market_slug=slug,
                side=order.get("side"),
                order_type=order.get("type"),
                price=parse_money(order.get("price")),
                quantity=parse_decimal(order.get("quantity")),
                state=order.get("state"),
                event_type="placed",
                source="ws",
                event_time=order.get("updateTime") or order.get("createTime"),
                raw=update,
            )
            self._note_market(slug)

    def _handle_position(self, payload: dict[str, Any]) -> None:
        slug = payload.get("marketSlug")
        after = payload.get("afterPosition", {}) or {}
        self.db.insert_position(
            slug=slug,
            net_position=parse_decimal(after.get("netPositionDecimal")),
            avg_price=parse_money(after.get("avgPrice")),
            unrealized_pnl=parse_money(after.get("unrealizedPnl")),
            raw=payload,
        )
        self._note_market(slug)

    def _handle_balance(self, payload: dict[str, Any]) -> None:
        change = payload.get("balanceChange", {}) or {}
        after = change.get("afterBalance", {}) or {}
        self.db.insert_balance(
            buying_power=parse_money(after.get("buyingPower")),
            total_value=parse_money(after.get("totalValue")),
            cash=parse_money(after.get("cash") or after.get("currentBalance")),
            raw=payload,
        )

    def _handle_balance_snapshot(self, payload: dict[str, Any]) -> None:
        """Initial balance snapshot: a ``balances`` list (same shape as REST)."""
        balances = payload.get("balances", [])
        data = balances[0] if isinstance(balances, list) and balances else {}
        cash = parse_money(data.get("currentBalance"))
        asset = parse_money(data.get("assetNotional")) or 0.0
        self.db.insert_balance(
            buying_power=parse_money(data.get("buyingPower")),
            total_value=(cash + asset) if cash is not None else None,
            cash=cash,
            raw=payload,
        )


class ActivityPoller:
    """Polls ``/portfolio/activities`` for settled trades and resolutions."""

    def __init__(
        self,
        rest: RestClient,
        db: Database,
        *,
        interval: float = 30.0,
        on_resolution: Callable[[str, float | None, str | None, dict], None] | None = None,
    ) -> None:
        self.rest = rest
        self.db = db
        self.interval = interval
        self.on_resolution = on_resolution
        self._stop = asyncio.Event()

    def collect_once(self, *, max_pages: int = 1) -> None:
        """Fetch recent activities. The periodic poll only needs the newest
        page; ``backfill`` pages deeper for history."""
        try:
            activities = list(self.rest.iter_activities(limit=100, max_pages=max_pages))
        except PolymarketAPIError as exc:
            logger.warning("activities fetch failed: %s", exc)
            return
        for activity in activities:
            self._store(activity)

    def backfill(self, *, max_pages: int = 20) -> int:
        """One-time deeper sweep to capture historical trades and resolutions.

        Returns the number of activities seen. Newly-resolved markets recorded
        here let the analysis layer learn from already-settled sessions.
        """
        count = 0
        try:
            for activity in self.rest.iter_activities(limit=100, max_pages=max_pages, page_pause=0.6):
                self._store(activity)
                count += 1
        except PolymarketAPIError as exc:
            logger.warning("activities backfill stopped: %s", exc)
        return count

    def _store(self, activity: dict[str, Any]) -> None:
        atype = activity.get("type", "")
        trade = activity.get("trade")
        resolution = activity.get("positionResolution")
        balance = activity.get("accountBalanceChange")

        if trade:
            # A trade carries the fill under aggressorExecution (you took
            # liquidity) or passiveExecution (you provided it).
            agg = trade.get("aggressorExecution")
            execution = agg or trade.get("passiveExecution") or {}
            order = execution.get("order", {}) or {}
            price = parse_money(execution.get("lastPx")) or parse_money(order.get("avgPx"))
            qty = parse_decimal(execution.get("lastShares"))
            cost = (price * qty) if (price is not None and qty is not None) else None
            # Only the aggressor (taker) pays a fee. Estimate it from the model
            # (using the market's own feeCoefficient) so the cost is known even
            # before the receipt; reconcile against the actual fee the receipt
            # reports. The execution carries the per-fill commission; pass it
            # before the order so the per-fill amount wins over the order total.
            is_taker = agg is not None
            fee_rate = fee_rate_from_market(trade.get("market"))
            est_fee = protocol_fee(qty, price, is_taker=is_taker, fee_rate=fee_rate)
            actual_fee = parse_fee(execution, order)
            self.db.insert_activity(
                activity_id=trade.get("id"),
                activity_type=atype or "ACTIVITY_TYPE_TRADE",
                market_slug=order.get("marketSlug") or trade.get("marketSlug"),
                price=price,
                qty=qty,
                is_aggressor=1 if agg else 0,
                cost_basis=parse_money(trade.get("costBasis")) or cost,
                realized_pnl=parse_money(trade.get("realizedPnl")),
                est_fee=est_fee,
                actual_fee=actual_fee,
                fee_diff=fee_difference(actual_fee, est_fee),
                create_time=execution.get("transactTime") or order.get("createTime")
                or trade.get("createTime"),
                raw=activity,
            )
        elif resolution:
            slug = resolution.get("marketSlug")
            after = resolution.get("afterPosition", {}) or {}
            before = resolution.get("beforePosition", {}) or {}
            resolved_value = _resolved_long_value(resolution.get("market", {}) or {}, after, before)
            realized = parse_money(after.get("realized"))
            res_time = resolution.get("updateTime") or after.get("updateTime")
            self.db.insert_activity(
                activity_id=f"res-{slug}",
                activity_type=atype or "ACTIVITY_TYPE_POSITION_RESOLUTION",
                market_slug=slug,
                realized_pnl=realized,
                create_time=res_time,
                raw=activity,
            )
            if slug:
                self.db.insert_outcome(slug, resolved_value, res_time, raw=activity)
                if self.on_resolution:
                    self.on_resolution(slug, resolved_value, res_time, activity)
        elif balance:
            self.db.insert_activity(
                activity_id=balance.get("transactionId"),
                activity_type=atype or "ACTIVITY_TYPE_BALANCE_CHANGE",
                price=None,
                qty=parse_money(balance.get("amount")),
                create_time=balance.get("createTime"),
                raw=activity,
            )

    async def run(self) -> None:
        logger.info("ActivityPoller started (interval=%ss)", self.interval)
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self.collect_once)
            except Exception:  # noqa: BLE001
                logger.exception("ActivityPoller sweep failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval)
            except asyncio.TimeoutError:
                pass

    def stop(self) -> None:
        self._stop.set()
