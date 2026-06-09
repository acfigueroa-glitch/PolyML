"""Order execution with safety rails.

Two modes:
  * **paper** (default): simulate fills against the live order book — a taker buy
    fills at the best ask, a taker sell at the best bid. No real orders are sent.
  * **live**: place real IOC orders via the REST client. Requires BOTH
    ``trading.mode: live`` in config AND the environment variable
    ``POLYML_ALLOW_LIVE_TRADING=yes``; otherwise it refuses and falls back to
    paper. This double opt-in makes accidental live trading very hard.

Hard caps regardless of mode: one share per order, a configurable maximum number
of concurrent open positions, and a daily realized-loss kill switch that halts
new entries (open positions may still be closed).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from polyml.api.rest import PolymarketAPIError, RestClient
from polyml.storage.db import Database
from polyml.storage.models import OrderBook
from polyml.trading.fees import THETA, entry_cost_basis, exit_proceeds, taker_fee
from polyml.trading.strategy import ENTER, EXIT, Action, BotPosition

logger = logging.getLogger(__name__)

LIVE_ENV_FLAG = "POLYML_ALLOW_LIVE_TRADING"


@dataclass
class ExecutorConfig:
    mode: str = "paper"                 # 'paper' | 'live'
    theta: float = THETA
    max_open_positions: int = 1
    daily_loss_limit_usd: float = 2.0   # halt new entries past this realized loss
    tif: str = "TIME_IN_FORCE_IMMEDIATE_OR_CANCEL"  # IOC => taker, no resting risk


@dataclass
class _Trade:
    market_slug: str
    side: str
    shares: float
    entry_price: float
    entry_cost: float
    exit_price: float
    exit_proceeds: float
    net_pnl: float
    entry_time: str
    exit_time: str
    exit_reason: str
    features: dict


class Executor:
    def __init__(
        self,
        rest: RestClient | None,
        db: Database,
        config: ExecutorConfig,
        *,
        session_for: Callable[[str], int] | None = None,
    ) -> None:
        self.rest = rest
        self.db = db
        self.config = config
        self.session_for = session_for or (lambda slug: None)
        self.positions: dict[str, BotPosition] = {}
        self._entry_features: dict[str, dict] = {}
        self.realized_today = 0.0
        self.halted = False

        self.live = self._resolve_live_mode(config.mode)
        self.mode = "live" if self.live else "paper"

    def _resolve_live_mode(self, requested: str) -> bool:
        if requested != "live":
            return False
        if os.environ.get(LIVE_ENV_FLAG, "").lower() != "yes":
            logger.warning(
                "LIVE trading requested but %s != 'yes' — refusing; running in PAPER mode.",
                LIVE_ENV_FLAG,
            )
            return False
        if self.rest is None:
            logger.warning("LIVE trading requested but no REST client — running in PAPER mode.")
            return False
        logger.warning("LIVE TRADING ENABLED — the bot will place real one-share orders.")
        return True

    # --- helpers -----------------------------------------------------------------
    @staticmethod
    def _now() -> str:
        return datetime.now(tz=timezone.utc).isoformat()

    def has_position(self, slug: str) -> bool:
        return slug in self.positions

    def position(self, slug: str) -> BotPosition | None:
        return self.positions.get(slug)

    # --- main entry point --------------------------------------------------------
    def execute(self, action: Action, book: OrderBook, slug: str, features: dict | None = None) -> Any:
        if action.kind == ENTER:
            return self._open(action, book, slug, features or {})
        if action.kind == EXIT:
            return self._close(action, book, slug)
        return None  # HOLD / SKIP -> nothing to do

    # --- open --------------------------------------------------------------------
    def _open(self, action: Action, book: OrderBook, slug: str, features: dict) -> BotPosition | None:
        if self.halted:
            return None
        if slug in self.positions or len(self.positions) >= self.config.max_open_positions:
            return None
        price = action.price if action.price is not None else book.best_ask
        if price is None:
            return None

        if self.live and not self._place_live(slug, action.side, price, "entry"):
            return None  # live order didn't fill; stay flat

        cost = entry_cost_basis(price, action.shares, self.config.theta)
        fee = taker_fee(price, action.shares, self.config.theta)
        pos = BotPosition(slug, action.side or "", action.shares, price, cost, self._now())
        self.positions[slug] = pos
        self._entry_features[slug] = features

        self.db.insert_bot_order(
            session_id=self.session_for(slug), market_slug=slug, mode=self.mode,
            action="entry", side=action.side, price=price, shares=action.shares,
            fee=fee, status="filled", reason=action.reason,
        )
        logger.info("[%s] ENTER %s 1@%.2f (cost basis %.4f) — %s",
                    self.mode, slug, price, cost, action.reason)
        return pos

    # --- close -------------------------------------------------------------------
    def _close(self, action: Action, book: OrderBook, slug: str) -> _Trade | None:
        pos = self.positions.get(slug)
        if pos is None:
            return None
        price = action.price if action.price is not None else book.best_bid
        if price is None:
            return None

        if self.live and not self._place_live(slug, action.side, price, "exit"):
            return None  # couldn't exit; keep the position and retry next tick

        proceeds = exit_proceeds(price, pos.shares, self.config.theta)
        fee = taker_fee(price, pos.shares, self.config.theta)
        net = proceeds - pos.cost_basis
        exit_time = self._now()

        self.db.insert_bot_order(
            session_id=self.session_for(slug), market_slug=slug, mode=self.mode,
            action="exit", side=action.side, price=price, shares=pos.shares,
            fee=fee, status="filled", reason=action.reason,
        )
        self.db.insert_bot_trade(
            session_id=self.session_for(slug), market_slug=slug, mode=self.mode,
            side=pos.side, shares=pos.shares, entry_price=pos.entry_price,
            entry_cost=pos.cost_basis, exit_price=price, exit_proceeds=proceeds,
            net_pnl=net, entry_time=pos.entry_time, exit_time=exit_time,
            exit_reason=action.reason, features=self._entry_features.get(slug, {}),
            label_good=1 if net > 0 else 0,
        )

        self.realized_today += net
        logger.info("[%s] EXIT  %s 1@%.2f net %+.4f (day %+.4f) — %s",
                    self.mode, slug, price, net, self.realized_today, action.reason)
        if self.realized_today <= -abs(self.config.daily_loss_limit_usd) and not self.halted:
            self.halted = True
            logger.warning("Daily loss limit hit (%.2f); halting new entries.", self.realized_today)

        del self.positions[slug]
        self._entry_features.pop(slug, None)
        return _Trade(
            market_slug=slug, side=pos.side, shares=pos.shares, entry_price=pos.entry_price,
            entry_cost=pos.cost_basis, exit_price=price, exit_proceeds=proceeds, net_pnl=net,
            entry_time=pos.entry_time, exit_time=exit_time, exit_reason=action.reason,
            features=self._entry_features.get(slug, {}),
        )

    # --- live order placement ----------------------------------------------------
    def _place_live(self, slug: str, intent: str | None, price: float, action: str) -> bool:
        """Place a real one-share IOC order. Returns True if it (likely) filled."""
        order = {
            "marketSlug": slug,
            "intent": intent,
            "type": "ORDER_TYPE_LIMIT",
            "price": {"value": f"{price:.2f}", "currency": "USD"},
            "quantity": 1,
            "tif": self.config.tif,
        }
        try:
            resp = self.rest.create_order(order)  # type: ignore[union-attr]
        except PolymarketAPIError as exc:
            logger.warning("live %s order rejected for %s: %s", action, slug, exc)
            return False
        state = (resp or {}).get("order", resp or {}).get("state") if isinstance(resp, dict) else None
        filled = state in (None, "ORDER_STATE_FILLED", "ORDER_STATE_PARTIALLY_FILLED")
        if not filled:
            logger.info("live %s order for %s did not fill (state=%s)", action, slug, state)
        return filled

    # --- end of session ----------------------------------------------------------
    def force_flatten(self, slug: str, book: OrderBook, reason: str = "session end") -> _Trade | None:
        """Close any open position at the current bid (used when a game ends)."""
        if slug not in self.positions:
            return None
        bid = book.best_bid
        action = Action(EXIT, reason, side=self.positions[slug].side, price=bid, shares=1.0)
        return self._close(action, book, slug)

    def reset_day(self) -> None:
        self.realized_today = 0.0
        self.halted = False
