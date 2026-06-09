"""Market data collector: snapshots markets, order books, and BBO.

Pulls the order book for each watched market on an interval and stores a
normalised snapshot (best bid/ask, mid, spread, book imbalance, last trade,
open interest) plus the raw payload. The public WebSocket gives finer-grained
updates; this loop guarantees a periodic baseline and fills reconnection gaps.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Sequence

from polyml.api.rest import PolymarketAPIError, RestClient
from polyml.storage.db import Database
from polyml.storage.models import OrderBook

logger = logging.getLogger(__name__)


class MarketCollector:
    def __init__(
        self,
        rest: RestClient,
        db: Database,
        *,
        interval: float = 5.0,
    ) -> None:
        self.rest = rest
        self.db = db
        self.interval = interval
        self._slugs: list[str] = []
        self._stop = asyncio.Event()

    def set_watchlist(self, slugs: Sequence[str]) -> None:
        self._slugs = list(dict.fromkeys(slugs))  # de-dup, preserve order

    def collect_market_meta(self, slug: str) -> None:
        try:
            payload = self.rest.get_market(slug)
        except PolymarketAPIError as exc:
            logger.warning("market meta fetch failed for %s: %s", slug, exc)
            return
        if not payload:
            return
        market = payload.get("market", payload)
        self.db.insert_market_snapshot(
            slug=slug,
            title=market.get("title") or market.get("question"),
            state=market.get("state"),
            raw=payload,
        )

    def collect_book(self, slug: str) -> OrderBook | None:
        try:
            payload = self.rest.get_market_book(slug)
        except PolymarketAPIError as exc:
            logger.warning("book fetch failed for %s: %s", slug, exc)
            return None
        if not payload:
            return None
        book = OrderBook.from_payload(payload)
        if not book.market_slug:
            book.market_slug = slug
        self.db.insert_book_snapshot(book, source="rest", raw=payload)
        return book

    def collect_once(self) -> None:
        for slug in self._slugs:
            self.collect_book(slug)

    async def run(self) -> None:
        logger.info("MarketCollector started (interval=%ss, %d markets)", self.interval, len(self._slugs))
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self.collect_once)
            except Exception:  # noqa: BLE001 - keep the loop alive
                logger.exception("MarketCollector sweep failed")
            await self._sleep(self.interval)

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    def stop(self) -> None:
        self._stop.set()
