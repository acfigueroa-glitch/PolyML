"""Lightweight parsing helpers and value objects for Polymarket US payloads.

The API expresses money as ``{"value": "0.65", "currency": "USD"}`` and various
quantities as decimal strings. These helpers normalise those into floats and
pull common fields out of the (sometimes nested) response shapes so the rest of
the codebase deals in plain numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def parse_money(obj: Any) -> float | None:
    """Extract a float from a ``{"value": "...", "currency": "..."}`` object,
    a bare string, or a number. Returns None if not parseable."""
    if obj is None:
        return None
    if isinstance(obj, (int, float)):
        return float(obj)
    if isinstance(obj, str):
        try:
            return float(obj)
        except ValueError:
            return None
    if isinstance(obj, dict) and "value" in obj:
        return parse_money(obj["value"])
    return None


def parse_decimal(obj: Any) -> float | None:
    return parse_money(obj)


# Candidate keys the API might use for the trade fee, most-specific first. The
# live fixtures don't yet pin the exact name, so we also fall back to any key
# containing "fee" (ignoring booleans like "feePayer").
_FEE_KEYS = ("fee", "feeAmount", "feePaid", "takerFee", "fees", "commission")


def parse_fee(*payloads: Any) -> float | None:
    """Best-effort actual fee from one or more trade/execution payloads.

    Checks the known fee keys across each payload, then any ``*fee*`` key, and
    returns the first parseable money value. Returns ``None`` if no fee field is
    present (e.g. a maker fill, or an API shape that omits it)."""
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for key in _FEE_KEYS:
            if key in payload:
                value = parse_money(payload[key])
                if value is not None:
                    return value
        for key, raw in payload.items():
            if "fee" in key.lower() and not isinstance(raw, bool):
                value = parse_money(raw)
                if value is not None:
                    return value
    return None


def parse_time(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp (with optional trailing 'Z')."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        # Heuristic: treat large numbers as ms epoch, otherwise seconds.
        seconds = value / 1000.0 if value > 1e11 else float(value)
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def to_iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


@dataclass
class BookLevel:
    price: float
    qty: float


@dataclass
class OrderBook:
    """Normalised top-of-book + depth derived from a market book payload."""

    market_slug: str
    state: str | None
    bids: list[BookLevel] = field(default_factory=list)
    offers: list[BookLevel] = field(default_factory=list)
    last_trade_px: float | None = None
    open_interest: float | None = None
    transact_time: str | None = None

    @property
    def best_bid(self) -> float | None:
        return max((lvl.price for lvl in self.bids), default=None)

    @property
    def best_ask(self) -> float | None:
        return min((lvl.price for lvl in self.offers), default=None)

    @property
    def mid(self) -> float | None:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2.0
        return None

    @property
    def spread(self) -> float | None:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    @property
    def book_imbalance(self) -> float | None:
        """(bid_size - ask_size) / (bid_size + ask_size) at the top of book.
        Ranges -1 (sell pressure) .. +1 (buy pressure)."""
        bid_sz = sum(lvl.qty for lvl in self.bids)
        ask_sz = sum(lvl.qty for lvl in self.offers)
        total = bid_sz + ask_sz
        if total <= 0:
            return None
        return (bid_sz - ask_sz) / total

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "OrderBook":
        data = payload.get("marketData", payload) if isinstance(payload, dict) else {}
        bids = [
            BookLevel(parse_money(b.get("px")) or 0.0, parse_decimal(b.get("qty")) or 0.0)
            for b in data.get("bids", [])
        ]
        offers = [
            BookLevel(parse_money(o.get("px")) or 0.0, parse_decimal(o.get("qty")) or 0.0)
            for o in data.get("offers", [])
        ]
        stats = data.get("stats", {}) or {}
        return cls(
            market_slug=data.get("marketSlug", ""),
            state=data.get("state"),
            bids=bids,
            offers=offers,
            last_trade_px=parse_money(stats.get("lastTradePx")),
            open_interest=parse_decimal(stats.get("openInterest")),
            transact_time=data.get("transactTime"),
        )
