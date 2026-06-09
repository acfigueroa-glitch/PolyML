"""Polymarket US fee model.

Per the Polymarket US docs, a **taker** pays a trading fee of::

    fee = contracts * fee_rate * price * (1 - price)

rounded to 5 decimal places. **Makers pay nothing.** The ``price * (1 - price)``
term peaks at a 50% price and vanishes as the price approaches 0 or 1, so fees
are largest on coin-flip markets and tiny on lopsided ones — which is why a
small fill near a low probability can round all the way down to $0.

``fee_rate`` is the market's ``feeCoefficient`` (0.05 for the AEC sports
markets, confirmed against live receipts: a 10-share fill at 0.20 reports a
``commissionNotionalCollected`` of exactly 10 * 0.05 * 0.20 * 0.80 = $0.08).

The bot uses this in two ways:

* **Simulation** — estimate the fee a (hypothetical) fill *would* incur before
  it happens, so the cost is visible to decision-making.
* **Reconciliation** — compare that estimate against the ``actual`` fee reported
  on the real trade receipt; the difference is a signal that our rate/model is
  off (or that the fill was a maker, which pays nothing).
"""

from __future__ import annotations

# Fallback taker fee rate. The authoritative rate is the per-market
# ``feeCoefficient`` (see ``fee_rate_from_market``); this is only used when a
# payload doesn't carry one. Observed live value for AEC sports markets is 0.05.
DEFAULT_FEE_RATE = 0.05
FEE_RATES: dict[str, float] = {"sports": 0.05}

# Makers are not free: real receipts show them receiving a small *rebate* (a
# negative fee). Fit across 991 live fills gives a maker rate of ~-0.0107 vs the
# +0.05 taker rate (~21% of it). Approximate — refine as more fills accrue.
MAKER_REBATE_RATE = 0.0107

# The protocol rounds fees to 5 decimals; sub-$0.00001 fees become exactly zero.
FEE_DECIMALS = 5


def fee_rate_for(category: str | None = None) -> float:
    """Fallback taker fee rate for a market category (defaults to sports)."""
    if not category:
        return DEFAULT_FEE_RATE
    return FEE_RATES.get(category.lower(), DEFAULT_FEE_RATE)


def fee_rate_from_market(market: object, default: float = DEFAULT_FEE_RATE) -> float:
    """The market's ``feeCoefficient`` if present, else ``default``.

    The live API carries the authoritative rate per market under
    ``trade.market.feeCoefficient`` (e.g. 0.05 for AEC sports markets)."""
    if isinstance(market, dict) and market.get("feeCoefficient") is not None:
        try:
            return float(market["feeCoefficient"])
        except (TypeError, ValueError):
            pass
    return default


def protocol_fee(
    contracts: float | None,
    price: float | None,
    *,
    is_taker: bool = True,
    fee_rate: float = DEFAULT_FEE_RATE,
    maker_rebate_rate: float = MAKER_REBATE_RATE,
) -> float:
    """Estimated fee for a single fill (positive = charged, negative = rebate).

    ``contracts`` is the number of shares (sign is ignored), ``price`` is the
    fill price in dollars (0..1). Takers pay ``+fee_rate * shares * p * (1-p)``;
    makers receive a rebate ``-maker_rebate_rate * shares * p * (1-p)``. A price
    outside the open interval (0, 1) — e.g. a resolved settlement at 0 or 1 —
    incurs neither.
    """
    if contracts is None or price is None:
        return 0.0
    c = abs(float(contracts))
    p = float(price)
    if c <= 0.0 or not (0.0 < p < 1.0):
        return 0.0
    rate = fee_rate if is_taker else -maker_rebate_rate
    return round(c * rate * p * (1.0 - p), FEE_DECIMALS)


def fee_difference(actual_fee: float | None, estimated_fee: float | None) -> float | None:
    """``actual - estimated`` when the real fee is known, else ``None``.

    Positive means we under-estimated (paid more than modelled); near-zero on a
    maker fill where ``actual`` is ~0 and we (correctly) estimated 0 too.
    """
    if actual_fee is None or estimated_fee is None:
        return None
    return round(float(actual_fee) - float(estimated_fee), FEE_DECIMALS)
