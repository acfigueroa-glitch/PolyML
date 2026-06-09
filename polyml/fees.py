"""Polymarket US fee model.

Per the Polymarket US docs, a **taker** pays a trading fee of::

    fee = contracts * fee_rate * price * (1 - price)

rounded to 5 decimal places. **Makers pay nothing.** The ``price * (1 - price)``
term peaks at a 50% price and vanishes as the price approaches 0 or 1, so fees
are largest on coin-flip markets and tiny on lopsided ones — which is why a
small fill near a low probability can round all the way down to $0.

``fee_rate`` depends on the market category (sports is ``0.03``).

The bot uses this in two ways:

* **Simulation** — estimate the fee a (hypothetical) fill *would* incur before
  it happens, so the cost is visible to decision-making.
* **Reconciliation** — compare that estimate against the ``actual`` fee reported
  on the real trade receipt; the difference is a signal that our rate/model is
  off (or that the fill was a maker, which pays nothing).
"""

from __future__ import annotations

# Per-category taker fee rates. Sports is 0.03 per the docs; until a category's
# rate is confirmed we fall back to the sports rate.
DEFAULT_FEE_RATE = 0.03
FEE_RATES: dict[str, float] = {"sports": 0.03}

# The protocol rounds fees to 5 decimals; sub-$0.00001 fees become exactly zero.
FEE_DECIMALS = 5


def fee_rate_for(category: str | None = None) -> float:
    """Taker fee rate for a market category (defaults to the sports rate)."""
    if not category:
        return DEFAULT_FEE_RATE
    return FEE_RATES.get(category.lower(), DEFAULT_FEE_RATE)


def protocol_fee(
    contracts: float | None,
    price: float | None,
    *,
    is_taker: bool = True,
    fee_rate: float = DEFAULT_FEE_RATE,
) -> float:
    """Estimated protocol fee for a single fill.

    ``contracts`` is the number of shares (sign is ignored), ``price`` is the
    fill price in dollars (0..1). Makers (``is_taker=False``) pay nothing, and a
    price outside the open interval (0, 1) — e.g. a resolved settlement at 0 or 1
    — incurs no fee.
    """
    if not is_taker or contracts is None or price is None:
        return 0.0
    c = abs(float(contracts))
    p = float(price)
    if c <= 0.0 or not (0.0 < p < 1.0):
        return 0.0
    return round(c * fee_rate * p * (1.0 - p), FEE_DECIMALS)


def fee_difference(actual_fee: float | None, estimated_fee: float | None) -> float | None:
    """``actual - estimated`` when the real fee is known, else ``None``.

    Positive means we under-estimated (paid more than modelled); near-zero on a
    maker fill where ``actual`` is ~0 and we (correctly) estimated 0 too.
    """
    if actual_fee is None or estimated_fee is None:
        return None
    return round(float(actual_fee) - float(estimated_fee), FEE_DECIMALS)
