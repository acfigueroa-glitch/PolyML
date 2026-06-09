"""Session lifecycle management.

A *session* is the arc of your involvement in one market: it opens when we first
see you trading (or holding) it, and concludes when the market resolves. On
conclusion we compute realized PnL for the session and hand it to the analysis
layer, which links your decisions to the outcome and updates the learner.
"""

from __future__ import annotations

import logging
from typing import Callable

from polyml.storage.db import Database

logger = logging.getLogger(__name__)

# Market states that mean "resolved / no longer trading".
TERMINAL_STATES = {
    "MARKET_STATE_TERMINATED",
    "MARKET_STATE_EXPIRED",
    "TERMINATED",
    "EXPIRED",
}


class SessionManager:
    def __init__(
        self,
        db: Database,
        *,
        on_conclude: Callable[[int, str], None] | None = None,
    ) -> None:
        self.db = db
        # Called with (session_id, market_slug) when a session concludes.
        self.on_conclude = on_conclude

    def open_session(self, slug: str) -> int:
        """Idempotently ensure an open session exists for ``slug``."""
        return self.db.get_or_open_session(slug)

    def _session_realized_pnl(self, slug: str) -> float:
        # The resolution activity carries the position's final cumulative
        # realized PnL — authoritative, so prefer it over summing trades.
        res = self.db.query_one(
            "SELECT realized_pnl FROM activities WHERE market_slug=? "
            "AND activity_type LIKE '%RESOLUTION%' AND realized_pnl IS NOT NULL "
            "ORDER BY create_time DESC LIMIT 1",
            (slug,),
        )
        if res and res["realized_pnl"] is not None:
            return float(res["realized_pnl"])
        row = self.db.query_one(
            "SELECT COALESCE(SUM(realized_pnl), 0) AS pnl FROM activities "
            "WHERE market_slug=? AND realized_pnl IS NOT NULL",
            (slug,),
        )
        return float(row["pnl"]) if row and row["pnl"] is not None else 0.0

    def conclude_if_resolved(self, slug: str) -> int | None:
        """If an outcome exists for ``slug`` and a session is open, conclude it.

        Returns the concluded session id, or None if nothing changed.
        """
        outcome = self.db.query_one(
            "SELECT resolved_value FROM outcomes WHERE market_slug=?", (slug,)
        )
        if not outcome:
            return None
        session = self.db.query_one(
            "SELECT id FROM sessions WHERE market_slug=? AND status='open' ORDER BY id DESC LIMIT 1",
            (slug,),
        )
        if not session:
            return None
        session_id = int(session["id"])
        realized = self._session_realized_pnl(slug)
        self.db.conclude_session(session_id, outcome["resolved_value"], realized)
        logger.info("session %d for %s concluded (pnl=%.2f)", session_id, slug, realized)
        if self.on_conclude:
            self.on_conclude(session_id, slug)
        return session_id

    def handle_resolution(self, slug: str, resolved_value, resolution_time, raw) -> None:
        """Callback wired to ``ActivityPoller.on_resolution``."""
        self.conclude_if_resolved(slug)

    def sweep_stale(self, stale_minutes: int = 120) -> list[int]:
        """Conclude sessions whose market is resolved but we missed the event.

        Looks for open sessions whose latest market snapshot is in a terminal
        state, and concludes them. Returns concluded session ids.
        """
        concluded: list[int] = []
        open_sessions = self.db.query(
            "SELECT id, market_slug FROM sessions WHERE status='open'"
        )
        for row in open_sessions:
            slug = row["market_slug"]
            snap = self.db.query_one(
                "SELECT state FROM market_snapshots WHERE market_slug=? "
                "ORDER BY captured_at DESC LIMIT 1",
                (slug,),
            )
            if snap and snap["state"] in TERMINAL_STATES:
                sid = self.conclude_if_resolved(slug)
                if sid is None:
                    # Terminal but no outcome row yet: conclude with unknown value.
                    realized = self._session_realized_pnl(slug)
                    self.db.conclude_session(int(row["id"]), None, realized)
                    sid = int(row["id"])
                    if self.on_conclude:
                        self.on_conclude(sid, slug)
                concluded.append(sid)
        return concluded

    def list_open_sessions(self) -> list[str]:
        rows = self.db.query("SELECT market_slug FROM sessions WHERE status='open'")
        return [r["market_slug"] for r in rows]
