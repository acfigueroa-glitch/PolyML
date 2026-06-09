"""WebSocket clients for the Polymarket US API.

Two streams (docs.polymarket.us):

  Public markets:  wss://api.polymarket.us/v1/ws/markets
      SUBSCRIPTION_TYPE_MARKET_DATA       full book + stats
      SUBSCRIPTION_TYPE_MARKET_DATA_LITE  price/BBO only
      SUBSCRIPTION_TYPE_TRADE             real-time trades

  Private (your activity):  wss://api.polymarket.us/v1/ws/private
      SUBSCRIPTION_TYPE_ORDER             order lifecycle events
      SUBSCRIPTION_TYPE_ORDER_SNAPSHOT    initial open orders
      SUBSCRIPTION_TYPE_POSITION          position changes
      SUBSCRIPTION_TYPE_ACCOUNT_BALANCE   balance updates

Both endpoints authenticate in the connection handshake using the same Ed25519
headers as REST. Each client auto-reconnects with exponential backoff and
re-applies its subscriptions. Messages are handed to an async callback.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import websockets
from websockets.client import WebSocketClientProtocol

from polyml.api.auth import Ed25519Signer

logger = logging.getLogger(__name__)

MessageHandler = Callable[[dict[str, Any]], Awaitable[None]]

# Max markets per subscription, per the docs.
MAX_MARKETS_PER_SUB = 100


def _handshake_path(url: str) -> str:
    """Host-relative path of a ws(s):// URL, for signing the handshake."""
    after_scheme = url.split("://", 1)[-1]
    slash = after_scheme.find("/")
    return after_scheme[slash:] if slash != -1 else "/"


class _BaseWebSocket:
    def __init__(
        self,
        url: str,
        signer: Ed25519Signer | None,
        on_message: MessageHandler,
        *,
        name: str = "ws",
    ) -> None:
        self.url = url
        self.signer = signer
        self.on_message = on_message
        self.name = name
        self._subscriptions: list[dict[str, Any]] = []
        self._ws: WebSocketClientProtocol | None = None
        self._stop = asyncio.Event()

    def add_subscription(self, message: dict[str, Any]) -> None:
        """Queue a subscribe message; (re)sent on every (re)connect."""
        self._subscriptions.append(message)

    def _auth_headers(self) -> list[tuple[str, str]]:
        if self.signer is None:
            return []
        headers = self.signer.headers("GET", _handshake_path(self.url))
        return list(headers.items())

    async def _send(self, ws: WebSocketClientProtocol, message: dict[str, Any]) -> None:
        await ws.send(json.dumps(message))

    async def run(self) -> None:
        """Connect, (re)subscribe, and pump messages until ``stop()``.

        Reconnects with exponential backoff on any connection error.
        """
        backoff = 1.0
        while not self._stop.is_set():
            try:
                async with websockets.connect(
                    self.url,
                    extra_headers=self._auth_headers(),
                    ping_interval=20,
                    ping_timeout=20,
                    max_size=8 * 1024 * 1024,
                ) as ws:
                    self._ws = ws
                    backoff = 1.0
                    for sub in self._subscriptions:
                        await self._send(ws, sub)
                    logger.info("[%s] connected; %d subscription(s)", self.name, len(self._subscriptions))
                    async for raw in ws:
                        await self._dispatch(raw)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reconnect on anything
                if self._stop.is_set():
                    break
                logger.warning("[%s] disconnected (%s); reconnecting in %.0fs", self.name, exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
            finally:
                self._ws = None

    async def _dispatch(self, raw: str | bytes) -> None:
        try:
            message = json.loads(raw)
        except (ValueError, TypeError):
            logger.debug("[%s] non-JSON frame ignored", self.name)
            return
        try:
            await self.on_message(message)
        except Exception:  # noqa: BLE001 - never let a handler kill the socket
            logger.exception("[%s] message handler raised", self.name)

    def stop(self) -> None:
        self._stop.set()


def _chunk(slugs: Sequence[str], size: int) -> list[list[str]]:
    return [list(slugs[i : i + size]) for i in range(0, len(slugs), size)] or [[]]


class MarketsWebSocket(_BaseWebSocket):
    """Public market-data stream (order books, BBO, trades)."""

    def __init__(self, url: str, signer: Ed25519Signer | None, on_message: MessageHandler) -> None:
        super().__init__(url, signer, on_message, name="markets-ws")

    def subscribe_market_data(self, slugs: Sequence[str], *, lite: bool = False) -> None:
        sub_type = "SUBSCRIPTION_TYPE_MARKET_DATA_LITE" if lite else "SUBSCRIPTION_TYPE_MARKET_DATA"
        for i, chunk in enumerate(_chunk(slugs, MAX_MARKETS_PER_SUB)):
            self.add_subscription(
                {
                    "subscribe": {
                        "requestId": f"md-{i}",
                        "subscriptionType": sub_type,
                        "marketSlugs": chunk,
                    }
                }
            )

    def subscribe_trades(self, slugs: Sequence[str]) -> None:
        for i, chunk in enumerate(_chunk(slugs, MAX_MARKETS_PER_SUB)):
            self.add_subscription(
                {
                    "subscribe": {
                        "requestId": f"trade-{i}",
                        "subscriptionType": "SUBSCRIPTION_TYPE_TRADE",
                        "marketSlugs": chunk,
                    }
                }
            )


class PrivateWebSocket(_BaseWebSocket):
    """Private stream that mirrors *your* orders, positions, and balance."""

    def __init__(self, url: str, signer: Ed25519Signer, on_message: MessageHandler) -> None:
        super().__init__(url, signer, on_message, name="private-ws")

    def subscribe_all(self, slugs: Sequence[str] | None = None) -> None:
        """Subscribe to order snapshot + lifecycle, positions, and balance."""
        market_slugs = list(slugs) if slugs else None
        for sub_type in (
            "SUBSCRIPTION_TYPE_ORDER_SNAPSHOT",
            "SUBSCRIPTION_TYPE_ORDER",
            "SUBSCRIPTION_TYPE_POSITION",
        ):
            payload: dict[str, Any] = {
                "requestId": sub_type.lower(),
                "subscriptionType": sub_type,
            }
            if market_slugs:
                payload["marketSlugs"] = market_slugs
            self.add_subscription({"subscribe": payload})
        # Account balance is not market-scoped.
        self.add_subscription(
            {
                "subscribe": {
                    "requestId": "balance",
                    "subscriptionType": "SUBSCRIPTION_TYPE_ACCOUNT_BALANCE",
                }
            }
        )
