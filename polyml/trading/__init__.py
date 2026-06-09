"""Trading layer.

``paper`` simulates the model's decisions with fee-aware fills and records P&L
WITHOUT placing any real orders — the proof-of-edge harness that must show a
positive net result before any live order-placement path is ever enabled.
"""

from polyml.trading.paper import PaperOrder, PaperStrategy, PaperTrader

__all__ = ["PaperOrder", "PaperStrategy", "PaperTrader"]
