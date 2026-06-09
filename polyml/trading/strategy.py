"""The one-share scalping strategy.

Rules (per your spec):
  * Buy exactly ONE share at a time, taking the ask.
  * Exit as soon as the round-trip net profit clears a small USD hurdle — take
    ANY profit, never chase a bigger one.
  * Only enter when a profitable exit is actually reachable after the dynamic
    round-trip fee and the spread, the book can absorb our size, and (optionally)
    the learner's entry score is favourable.
  * Optionally cut losses (stop-loss) and flatten before the market resolves,
    so a stalled position doesn't ride all the way to a 0/1 settlement.

The strategy is pure: it reads an order book + the current bot position and
returns an :class:`Action`. Execution (paper or live) happens elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from polyml.storage.models import OrderBook
from polyml.trading.fees import (
    THETA,
    entry_cost_basis,
    exit_proceeds,
    min_profitable_exit_price,
)

ENTER = "ENTER"
EXIT = "EXIT"
HOLD = "HOLD"
SKIP = "SKIP"


@dataclass
class BotPosition:
    """An open one-share position held by the bot."""

    market_slug: str
    side: str  # e.g. ORDER_INTENT_BUY_LONG (the instrument we're long)
    shares: float
    entry_price: float
    cost_basis: float
    entry_time: str


@dataclass
class Action:
    kind: str  # ENTER | EXIT | HOLD | SKIP
    reason: str
    side: Optional[str] = None
    price: Optional[float] = None
    shares: float = 1.0
    projected_net: Optional[float] = None  # for EXIT: realised net P&L
    target_exit: Optional[float] = None  # for ENTER: min profitable exit price


@dataclass
class StrategyConfig:
    shares_per_trade: float = 1.0
    theta: float = THETA
    profit_hurdle_usd: float = 0.01      # leave with at least this much profit
    max_spread: float = 0.05             # skip entries when the spread is wider
    min_price: float = 0.05              # avoid the extreme tails
    max_price: float = 0.95
    max_target_move: float = 0.08        # required up-move to exit must be small
    min_book_depth: float = 1.0          # shares resting at the bid for our exit
    entry_imbalance_min: float = 0.10    # require some buy pressure to enter
    use_model_gate: bool = True
    model_threshold: float = 0.50        # learner P(good) needed to enter
    stop_loss_usd: Optional[float] = 0.10  # cut a loser; None disables
    flatten_before_close: bool = True
    close_buffer_minutes: float = 5.0    # exit if within this of resolution
    # The instrument we scalp: the market's long/first outcome.
    long_intent: str = "ORDER_INTENT_BUY_LONG"
    long_exit_intent: str = "ORDER_INTENT_SELL_LONG"


def _best_bid_size(book: OrderBook) -> float:
    if not book.bids:
        return 0.0
    best = max(lvl.price for lvl in book.bids)
    return sum(lvl.qty for lvl in book.bids if lvl.price == best)


class ScalpStrategy:
    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or StrategyConfig()

    def decide(
        self,
        book: OrderBook,
        position: BotPosition | None,
        *,
        model_score: float | None = None,
        minutes_to_close: float | None = None,
    ) -> Action:
        if position is None:
            return self._consider_entry(book, model_score)
        return self._consider_exit(book, position, minutes_to_close)

    # --- entry -------------------------------------------------------------------
    def _consider_entry(self, book: OrderBook, model_score: float | None) -> Action:
        c = self.config
        ask, bid = book.best_ask, book.best_bid
        if ask is None or bid is None:
            return Action(SKIP, "no two-sided market")
        if not (c.min_price <= ask <= c.max_price):
            return Action(SKIP, f"ask {ask:.2f} outside [{c.min_price},{c.max_price}]")
        spread = ask - bid
        if spread > c.max_spread:
            return Action(SKIP, f"spread {spread:.3f} > max {c.max_spread}")
        if _best_bid_size(book) < c.min_book_depth:
            return Action(SKIP, "insufficient bid depth to exit our size")
        if book.book_imbalance is not None and book.book_imbalance < c.entry_imbalance_min:
            return Action(SKIP, f"weak buy pressure (imbalance {book.book_imbalance:+.2f})")
        if c.use_model_gate and model_score is not None and model_score < c.model_threshold:
            return Action(SKIP, f"model score {model_score:.2f} < {c.model_threshold}")

        size = c.shares_per_trade
        cost_basis = entry_cost_basis(ask, size, c.theta)
        target = min_profitable_exit_price(cost_basis, size, c.profit_hurdle_usd, c.theta)
        move_needed = target - ask
        if target > c.max_price or move_needed > c.max_target_move:
            return Action(SKIP, f"profitable exit {target:.3f} needs a {move_needed:.3f} move (too far)")

        return Action(
            ENTER,
            reason=f"buy {size:g}@{ask:.2f}; profitable exit at {target:.3f} "
            f"(needs +{move_needed:.3f}), spread {spread:.3f}",
            side=c.long_intent,
            price=ask,
            shares=size,
            target_exit=target,
        )

    # --- exit --------------------------------------------------------------------
    def _consider_exit(
        self, book: OrderBook, position: BotPosition, minutes_to_close: float | None
    ) -> Action:
        c = self.config
        bid = book.best_bid
        if bid is None:
            return Action(HOLD, "no bid to sell into")
        net = exit_proceeds(bid, position.shares, c.theta) - position.cost_basis

        if net > c.profit_hurdle_usd:
            return Action(
                EXIT,
                reason=f"take profit: sell {position.shares:g}@{bid:.2f} for net +{net:.4f}",
                side=c.long_exit_intent,
                price=bid,
                shares=position.shares,
                projected_net=net,
            )
        if c.stop_loss_usd is not None and net <= -abs(c.stop_loss_usd):
            return Action(
                EXIT,
                reason=f"stop loss: sell {position.shares:g}@{bid:.2f} for net {net:.4f}",
                side=c.long_exit_intent,
                price=bid,
                shares=position.shares,
                projected_net=net,
            )
        if (
            c.flatten_before_close
            and minutes_to_close is not None
            and minutes_to_close <= c.close_buffer_minutes
        ):
            return Action(
                EXIT,
                reason=f"flatten before resolution ({minutes_to_close:.1f}m left): "
                f"sell @{bid:.2f} for net {net:.4f}",
                side=c.long_exit_intent,
                price=bid,
                shares=position.shares,
                projected_net=net,
            )
        return Action(HOLD, f"hold: best exit net {net:+.4f} below hurdle {c.profit_hurdle_usd}")
