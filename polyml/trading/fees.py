"""Fee-aware scalping math for Polymarket US.

Polymarket charges a *dynamic taker fee* that is a function of the contract's
implied probability ``p``. It peaks at p = 0.50 and tapers to zero at p = 0.00
and p = 1.00. We model it as::

    fee(p, N) = theta * N * p * (1 - p)

with ``theta`` the fee coefficient (~0.05) and ``N`` the number of contracts.

Round-trip economics for a taker who buys N at the ask and later sells N at the
bid:

    entry_cost_basis  C = N * p_in  + fee(p_in,  N)        # cash out the door
    exit_proceeds       = N * p_out - fee(p_out, N)        # cash received
    net_profit          = exit_proceeds - C

Breaking even on the exit means ``exit_proceeds == C``:

    N*p - theta*N*p*(1-p) = C
    => theta*p^2 + (1-theta)*p - C/N = 0          # a=theta, b=1-theta, c=-C/N

which is the quadratic solved by ``breakeven_exit_price``. To clear a target USD
hurdle as well, substitute ``C -> C + hurdle`` (``min_profitable_exit_price``).
"""

from __future__ import annotations

import math

# Default fee coefficient (5%). Polymarket exposes a per-market ``feeCoefficient``
# (we saw 0.05 live); pass it through when available.
THETA = 0.05


def taker_fee(price: float, shares: float, theta: float = THETA) -> float:
    """Dynamic taker fee for ``shares`` contracts at ``price`` (0..1)."""
    p = _clip_price(price)
    return theta * shares * p * (1.0 - p)


def entry_cost_basis(entry_price: float, shares: float, theta: float = THETA) -> float:
    """Total cash spent to acquire ``shares`` at ``entry_price`` (notional + fee)."""
    return shares * entry_price + taker_fee(entry_price, shares, theta)


def exit_proceeds(exit_price: float, shares: float, theta: float = THETA) -> float:
    """Cash received from selling ``shares`` at ``exit_price`` (notional - fee)."""
    return shares * exit_price - taker_fee(exit_price, shares, theta)


def net_profit(entry_price: float, exit_price: float, shares: float, theta: float = THETA) -> float:
    """Round-trip net profit in USD (negative = loss)."""
    return exit_proceeds(exit_price, shares, theta) - entry_cost_basis(entry_price, shares, theta)


def breakeven_exit_price(cost_basis: float, shares: float, theta: float = THETA) -> float:
    """Exit price at which ``exit_proceeds == cost_basis`` (zero net P&L).

    Solves ``theta*p^2 + (1-theta)*p - C/N = 0`` via the quadratic formula and
    returns the economically meaningful root in [0, 1].
    """
    return _solve_exit_price(cost_basis, shares, theta)


def min_profitable_exit_price(
    cost_basis: float, shares: float, hurdle: float = 0.0, theta: float = THETA
) -> float:
    """Lowest exit price whose proceeds clear ``cost_basis + hurdle`` (USD)."""
    return _solve_exit_price(cost_basis + hurdle, shares, theta)


def _solve_exit_price(target_cash: float, shares: float, theta: float) -> float:
    """Smallest p in [0,1] with exit_proceeds(p, shares) == target_cash."""
    if shares <= 0:
        raise ValueError("shares must be positive")
    a = theta
    b = 1.0 - theta
    c = -target_cash / shares
    if a == 0:  # degenerate: linear, no fee
        return _clip_price(-c / b)
    disc = b * b - 4 * a * c
    if disc < 0:
        # Target unreachable even at p=1; clamp.
        return 1.0
    root = (-b + math.sqrt(disc)) / (2 * a)
    return _clip_price(root)


def _clip_price(p: float) -> float:
    return max(0.0, min(1.0, p))
