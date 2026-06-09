"""Tests for the fee-aware scalping math."""

import math

from polyml.trading.fees import (
    breakeven_exit_price,
    entry_cost_basis,
    exit_proceeds,
    min_profitable_exit_price,
    net_profit,
    taker_fee,
)


def test_fee_peaks_at_half_and_zero_at_tails():
    assert taker_fee(0.0, 1) == 0.0
    assert taker_fee(1.0, 1) == 0.0
    assert taker_fee(0.5, 1) == 0.05 * 0.25  # 0.0125
    # Symmetric and maximal at 0.5.
    assert math.isclose(taker_fee(0.3, 1), taker_fee(0.7, 1), abs_tol=1e-12)
    assert taker_fee(0.5, 1) > taker_fee(0.3, 1)


def test_breakeven_matches_worked_example():
    # Buy 1 share at 0.50: entry fee = 0.05*0.5*0.5 = 0.0125, C = 0.5125.
    cost = entry_cost_basis(0.50, 1)
    assert math.isclose(cost, 0.5125, abs_tol=1e-9)
    p = breakeven_exit_price(cost, 1)
    # Selling at the breakeven price yields proceeds == cost basis.
    assert math.isclose(exit_proceeds(p, 1), cost, abs_tol=1e-9)
    # Worked value ~0.525.
    assert math.isclose(p, 0.52497, abs_tol=1e-4)


def test_net_profit_sign():
    cost = entry_cost_basis(0.50, 1)
    be = breakeven_exit_price(cost, 1)
    assert net_profit(0.50, be, 1) == 0.0 or abs(net_profit(0.50, be, 1)) < 1e-9
    assert net_profit(0.50, be + 0.02, 1) > 0
    assert net_profit(0.50, be - 0.02, 1) < 0


def test_min_profitable_exit_clears_hurdle():
    cost = entry_cost_basis(0.40, 1)
    target = min_profitable_exit_price(cost, 1, hurdle=0.02)
    assert math.isclose(exit_proceeds(target, 1) - cost, 0.02, abs_tol=1e-9)
    assert target > breakeven_exit_price(cost, 1)


def test_unreachable_target_clamps_to_one():
    # A tiny share count with an absurd hurdle can't be met below p=1.
    assert min_profitable_exit_price(0.9, 1, hurdle=100.0) == 1.0
