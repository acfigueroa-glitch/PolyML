"""Paper trading: decide, simulate a fill, settle — never place a real order.

The strategy is deliberately simple and *principled* so its P&L is interpretable
rather than a black box:

* The model gives ``p`` = P(a buy here is a good decision), which for a binary
  contract held to resolution we read as the probability the long instrument
  settles to 1.
* Buying one long contract at price ``price`` then has, before fees, an expected
  value per contract of ``EV = p - price`` (win ``1 - price`` w.p. ``p``, lose
  ``price`` w.p. ``1 - p``). Net of the taker fee, ``EV = p - price - fee``.
* We "buy" exactly when ``EV >= min_edge`` — i.e. only when the model's edge
  clears the fee plus a margin. One contract, one position per market, held to
  resolution.

That EV gate is the whole point: it makes the fee structure a first-class input,
so the harness answers "does the edge survive the fees?" honestly. Nothing here
sends an order; it writes to the ``paper_positions`` ledger only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from polyml.fees import DEFAULT_FEE_RATE, protocol_fee
from polyml.storage.db import Database

logger = logging.getLogger(__name__)

# A scorer maps a feature dict to P(good) in [0, 1], or None if it can't score
# (e.g. the model isn't trained yet).
Scorer = Callable[[dict[str, float]], "float | None"]


@dataclass(frozen=True)
class PaperOrder:
    """A simulated decision to buy one (long) contract — not a real order."""

    market_slug: str
    instrument: str          # 'long'
    qty: float
    price: float             # fill price (best ask for a taker buy)
    fee: float               # taker fee for this fill
    model_prob: float        # p = P(good) the model assigned
    edge: float              # expected value per contract, net of fee


class PaperStrategy:
    """Turns a model score + current book into an optional 1-contract buy."""

    def __init__(
        self,
        *,
        min_edge: float = 0.02,
        max_contracts: int = 1,
        fee_rate: float = DEFAULT_FEE_RATE,
    ) -> None:
        self.min_edge = min_edge
        self.max_contracts = max_contracts
        self.fee_rate = fee_rate

    def decide(
        self,
        market_slug: str,
        *,
        prob: float | None,
        best_ask: float | None,
        fee_rate: float | None = None,
    ) -> PaperOrder | None:
        """Return a buy order when ``EV = prob - best_ask - fee >= min_edge``.

        ``prob`` is the model's P(good); ``best_ask`` is the price we'd pay as a
        taker. Returns None when we can't or shouldn't trade.
        """
        if prob is None or best_ask is None:
            return None
        if not (0.0 < best_ask < 1.0) or not (0.0 <= prob <= 1.0):
            return None
        rate = self.fee_rate if fee_rate is None else fee_rate
        qty = float(self.max_contracts)
        fee = protocol_fee(qty, best_ask, is_taker=True, fee_rate=rate)
        edge = prob - best_ask - fee
        if edge < self.min_edge:
            return None
        return PaperOrder(
            market_slug=market_slug,
            instrument="long",
            qty=qty,
            price=best_ask,
            fee=fee,
            model_prob=prob,
            edge=edge,
        )


class PaperTrader:
    """Drives the strategy against live ticks or a backtest, persisting fills.

    ``scorer`` produces P(good) from features; ``source`` tags the ledger rows
    ('live' or 'backtest') and keeps the two from colliding.
    """

    def __init__(
        self,
        db: Database,
        scorer: Scorer,
        *,
        strategy: PaperStrategy | None = None,
        source: str = "live",
    ) -> None:
        self.db = db
        self.scorer = scorer
        self.strategy = strategy or PaperStrategy()
        self.source = source

    def consider(
        self,
        market_slug: str,
        features: dict[str, float],
        *,
        best_ask: float | None,
        fee_rate: float | None = None,
        opened_at: str | None = None,
    ) -> PaperOrder | None:
        """Score the state and, if it clears the edge gate and we hold no open
        paper position for this market, record a simulated entry."""
        if self.db.has_open_paper_position(market_slug, self.source):
            return None
        prob = self.scorer(features)
        order = self.strategy.decide(
            market_slug, prob=prob, best_ask=best_ask, fee_rate=fee_rate
        )
        if order is None:
            return None
        self.db.open_paper_position(
            market_slug=order.market_slug,
            instrument=order.instrument,
            qty=order.qty,
            entry_price=order.price,
            entry_fee=order.fee,
            model_prob=order.model_prob,
            edge=order.edge,
            source=self.source,
            opened_at=opened_at,
        )
        logger.info(
            "[paper:%s] BUY 1 %s @ %.3f (p=%.2f edge=%+.3f fee=%.4f) — simulated, no real order",
            self.source, market_slug, order.price, order.model_prob, order.edge, order.fee,
        )
        return order

    def settle(self, market_slug: str, resolved_value: float | None) -> None:
        """Settle the open paper position at the long instrument's value (0..1)."""
        if resolved_value is None:
            return
        self.db.settle_paper_position(market_slug, self.source, float(resolved_value))
