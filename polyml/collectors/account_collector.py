"""Account collector: snapshots balances, positions, and open orders.

These authenticated polls capture your buying power, holdings, and resting
orders on an interval. The private WebSocket streams the same data in real time;
this loop provides a periodic ground-truth snapshot and seeds the watchlist with
markets you're actually involved in (auto-follow).
"""

from __future__ import annotations

import asyncio
import logging

from polyml.api.rest import PolymarketAPIError, RestClient
from polyml.storage.db import Database
from polyml.storage.models import parse_decimal, parse_money

logger = logging.getLogger(__name__)


class AccountCollector:
    def __init__(
        self,
        rest: RestClient,
        db: Database,
        *,
        balance_interval: float = 15.0,
        orders_interval: float = 10.0,
    ) -> None:
        self.rest = rest
        self.db = db
        self.balance_interval = balance_interval
        self.orders_interval = orders_interval
        self._stop = asyncio.Event()
        # Markets discovered from positions/orders, for auto-follow.
        self.involved_slugs: set[str] = set()

    # --- individual sweeps -------------------------------------------------------
    def collect_balances(self) -> None:
        try:
            payload = self.rest.get_balances()
        except PolymarketAPIError as exc:
            logger.warning("balances fetch failed: %s", exc)
            return
        if not payload:
            return
        data = payload.get("balances", payload)
        # Field names vary; probe the common ones.
        buying_power = parse_money(_first(data, "buyingPower", "buying_power", "availableBalance"))
        total_value = parse_money(_first(data, "totalValue", "equity", "portfolioValue"))
        cash = parse_money(_first(data, "cash", "cashBalance", "available"))
        self.db.insert_balance(buying_power, total_value, cash, raw=payload)

    def collect_positions(self) -> None:
        try:
            payload = self.rest.get_positions()
        except PolymarketAPIError as exc:
            logger.warning("positions fetch failed: %s", exc)
            return
        if not payload:
            return
        positions = payload.get("positions", payload if isinstance(payload, list) else [])
        for pos in positions:
            slug = pos.get("marketSlug") or pos.get("market_slug")
            net = parse_decimal(_first(pos, "netPositionDecimal", "netPosition", "quantity"))
            avg = parse_money(_first(pos, "avgPrice", "averagePrice", "costBasis"))
            upnl = parse_money(_first(pos, "unrealizedPnl", "unrealized_pnl"))
            self.db.insert_position(slug, net, avg, upnl, raw=pos)
            if slug and net not in (None, 0.0):
                self.involved_slugs.add(slug)

    def collect_open_orders(self) -> None:
        try:
            payload = self.rest.get_open_orders()
        except PolymarketAPIError as exc:
            logger.warning("open orders fetch failed: %s", exc)
            return
        if not payload:
            return
        orders = payload.get("orders", payload if isinstance(payload, list) else [])
        for order in orders:
            slug = order.get("marketSlug") or order.get("market_slug")
            self.db.insert_order_event(
                order_id=order.get("id") or order.get("orderId"),
                market_slug=slug,
                side=order.get("side"),
                order_type=order.get("type"),
                price=parse_money(order.get("price")),
                quantity=parse_decimal(order.get("quantity")),
                filled_qty=parse_decimal(_first(order, "filledQuantity", "filledQty")),
                state=order.get("state"),
                event_type="snapshot",
                source="rest",
                event_time=order.get("updateTime") or order.get("createTime"),
                raw=order,
            )
            if slug:
                self.involved_slugs.add(slug)

    # --- loops -------------------------------------------------------------------
    async def run(self) -> None:
        logger.info("AccountCollector started")
        await asyncio.gather(self._balance_loop(), self._orders_loop())

    async def _balance_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self.collect_balances)
                await asyncio.to_thread(self.collect_positions)
            except Exception:  # noqa: BLE001
                logger.exception("balance/position sweep failed")
            await self._sleep(self.balance_interval)

    async def _orders_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self.collect_open_orders)
            except Exception:  # noqa: BLE001
                logger.exception("open-orders sweep failed")
            await self._sleep(self.orders_interval)

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    def stop(self) -> None:
        self._stop.set()


def _first(d: dict, *keys: str):
    for key in keys:
        if isinstance(d, dict) and key in d and d[key] is not None:
            return d[key]
    return None
