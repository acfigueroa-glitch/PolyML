"""PolyML — a machine-learning trading companion for Polymarket US.

PolyML observes Polymarket US markets and *mirrors your own trading activity*
(it never trades on your behalf). When a market resolves it links the decisions
you made to the actual outcome, scores them, and learns which signals you may
have overlooked — so your edge compounds over time.

The package is organised into layers:

    polyml.api         Auth + REST + WebSocket clients for the Polymarket US API.
    polyml.storage     SQLite schema and persistence for everything we observe.
    polyml.collectors  Pollers/streamers that snapshot market & account state.
    polyml.mirror      Records every action you take (orders placed/removed/filled).
    polyml.session     Groups activity into per-market sessions; detects conclusion.
    polyml.analysis    Outcome linkage, feature engineering, and the learner.
"""

__version__ = "0.1.0"
