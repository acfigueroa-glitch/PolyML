"""Autonomous scalping: fee-aware math, strategy, execution, and the trade loop.

PolyML's trading layer buys ONE share at a time and scalps for *any* net profit
after Polymarket US's dynamic round-trip taker fee — it never chases. It runs in
PAPER mode by default (simulated fills against the live book); LIVE order
placement is gated behind an explicit opt-in with hard safety caps.
"""

from polyml.trading.fees import (
    THETA,
    breakeven_exit_price,
    entry_cost_basis,
    exit_proceeds,
    min_profitable_exit_price,
    net_profit,
    taker_fee,
)
from polyml.trading.strategy import Action, ScalpStrategy, StrategyConfig

__all__ = [
    "THETA",
    "taker_fee",
    "entry_cost_basis",
    "exit_proceeds",
    "net_profit",
    "breakeven_exit_price",
    "min_profitable_exit_price",
    "Action",
    "ScalpStrategy",
    "StrategyConfig",
]
