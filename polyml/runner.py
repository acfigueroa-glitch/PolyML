"""Orchestrates the live observe-and-learn loop.

Wires together the REST client, collectors, the mirror (private WebSocket +
activity poller), the public market stream, the session manager, and the
learner. Run it with credentials set and it will:

  1. Discover the markets you're involved in and watch them.
  2. Stream + poll market data and your own activity into SQLite.
  3. Open a session per market; conclude it when the market resolves.
  4. On conclusion, link your decisions to the outcome and retrain the learner.

It never sends orders. Stop with Ctrl-C for a clean shutdown.
"""

from __future__ import annotations

import asyncio
import json
import logging

from polyml.analysis.learner import Learner
from polyml.analysis.outcomes import OutcomeLinker
from polyml.api.auth import Ed25519Signer
from polyml.api.rest import RestClient
from polyml.api.websocket import MarketsWebSocket, PrivateWebSocket
from polyml.collectors import AccountCollector, MarketCollector
from polyml.config import Config
from polyml.mirror import ActivityMirror, ActivityPoller
from polyml.session import SessionManager
from polyml.storage.db import Database

logger = logging.getLogger(__name__)

# How long a graceful shutdown waits for tasks to unwind before giving up and
# abandoning whatever is still stuck (e.g. a worker thread blocked in a
# non-cancellable REST call). Bounded so "stop the bot" can never hang forever.
SHUTDOWN_GRACE_SECONDS = 10.0


class Runner:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.db = Database(config.db_path)

        if not config.credentials.is_complete:
            raise RuntimeError(
                "Live run requires credentials. Set POLYMARKET_KEY_ID and "
                "POLYMARKET_SECRET_KEY (see .env.example)."
            )
        self.signer = Ed25519Signer.from_credentials(
            config.credentials.key_id, config.credentials.secret_key
        )
        self.rest = RestClient(
            config.rest_base_url,
            config.gateway_base_url,
            signer=self.signer,
            timeout=config.get("api.timeout_seconds", 30.0),
        )

        # Watchlist is shared across collectors and is extended at runtime.
        self.watchlist: set[str] = set(config.get("watch.market_slugs", []) or [])

        self.session_manager = SessionManager(self.db, on_conclude=self._on_session_conclude)
        self.linker = OutcomeLinker(self.db)
        self.learner = Learner(
            self.db,
            model=config.get("learning.model", "gradient_boosting"),
            min_decisions=config.get("learning.min_decisions_to_train", 30),
            holdout_fraction=config.get("learning.holdout_fraction", 0.25),
            random_state=config.get("learning.random_state", 42),
        )

        self.market_collector = MarketCollector(
            self.rest, self.db, interval=config.get("collectors.market_book_interval", 5)
        )
        self.account_collector = AccountCollector(
            self.rest,
            self.db,
            balance_interval=config.get("collectors.account_interval", 15),
            orders_interval=config.get("collectors.open_orders_interval", 10),
        )
        self.mirror = ActivityMirror(self.db, on_new_market=self._follow_market)
        self.activity_poller = ActivityPoller(
            self.rest,
            self.db,
            interval=config.get("collectors.activities_interval", 30),
            on_resolution=self.session_manager.handle_resolution,
        )
        self.private_ws = PrivateWebSocket(config.ws_private_url, self.signer, self.mirror.handle)
        self.markets_ws = MarketsWebSocket(config.ws_markets_url, self.signer, self._on_market_message)
        self._tasks: list[asyncio.Task] = []

    # --- watchlist management ----------------------------------------------------
    def _follow_market(self, slug: str) -> None:
        if slug and slug not in self.watchlist:
            self.watchlist.add(slug)
            self.market_collector.set_watchlist(self.watchlist)
            self.session_manager.open_session(slug)
            logger.info("now watching market: %s", slug)

    def _discover_initial_markets(self) -> None:
        if self.config.get("watch.auto_follow_my_positions", True):
            self.account_collector.collect_positions()
            self.account_collector.collect_open_orders()
            for slug in self.account_collector.involved_slugs:
                self._follow_market(slug)
        for slug in list(self.watchlist):
            self.session_manager.open_session(slug)
        self.market_collector.set_watchlist(self.watchlist)

    # --- callbacks ---------------------------------------------------------------
    async def _on_market_message(self, message: dict) -> None:
        """Persist public market-stream updates (book / trades)."""
        from polyml.storage.models import OrderBook, parse_decimal, parse_money

        if "marketData" in message or "bids" in message or "offers" in message:
            book = OrderBook.from_payload(message)
            if book.market_slug:
                await asyncio.to_thread(self.db.insert_book_snapshot, book, "ws", message)
        trade = message.get("trade")
        if trade:
            await asyncio.to_thread(
                self.db.insert_market_trade,
                trade.get("marketSlug", ""),
                trade.get("id"),
                parse_money(trade.get("price")),
                parse_decimal(trade.get("qty")),
                trade.get("side"),
                trade.get("transactTime") or trade.get("createTime"),
                message,
            )

    def _on_session_conclude(self, session_id: int, slug: str) -> None:
        """Link decisions to the outcome and retrain — the learning step."""
        logger.info("analyzing concluded session %d (%s)", session_id, slug)
        report = self.linker.link_session(session_id, slug)
        self.db.mark_session_analyzed(session_id, json.dumps(report, default=str))
        result = self.learner.train()
        logger.info("session %d analysis complete:\n%s", session_id, result.summary())
        for lesson in report.get("lessons", []):
            logger.info("  lesson [%s]: %s", slug, lesson)

    # --- lifecycle ---------------------------------------------------------------
    async def run(self) -> None:
        await asyncio.to_thread(self._discover_initial_markets)

        self.private_ws.subscribe_all()
        if self.config.get("collectors.use_websockets", True) and self.watchlist:
            self.markets_ws.subscribe_market_data(list(self.watchlist))
            self.markets_ws.subscribe_trades(list(self.watchlist))

        self._tasks = [
            asyncio.create_task(self.account_collector.run(), name="account"),
            asyncio.create_task(self.market_collector.run(), name="market"),
            asyncio.create_task(self.activity_poller.run(), name="activities"),
            asyncio.create_task(self.private_ws.run(), name="private-ws"),
            asyncio.create_task(self._stale_sweep_loop(), name="stale-sweep"),
        ]
        if self.config.get("collectors.use_websockets", True):
            self._tasks.append(asyncio.create_task(self.markets_ws.run(), name="markets-ws"))

        logger.info("PolyML running — observing %d market(s). Ctrl-C to stop.", len(self.watchlist))
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            pass

    async def _stale_sweep_loop(self) -> None:
        stale_minutes = self.config.get("sessions.stale_close_minutes", 120)
        while True:
            await asyncio.sleep(60)
            try:
                await asyncio.to_thread(self.session_manager.sweep_stale, stale_minutes)
            except Exception:  # noqa: BLE001
                logger.exception("stale sweep failed")

    async def shutdown(self, *, timeout: float = SHUTDOWN_GRACE_SECONDS) -> int:
        """Stop every loop and release resources.

        Asks each component to stop, cancels the tasks, then waits up to
        ``timeout`` for them to finish. Tasks blocked on uninterruptible work
        in a worker thread (a REST call, a ``time.sleep`` backoff) may not
        unwind in time; rather than hang, we abandon them and return how many
        were still running so the caller can decide to force-exit.
        """
        logger.info("shutting down...")
        for component in (
            self.market_collector,
            self.account_collector,
            self.activity_poller,
            self.private_ws,
            self.markets_ws,
        ):
            component.stop()
        for task in self._tasks:
            task.cancel()

        abandoned = 0
        if self._tasks:
            # asyncio.wait() returns after the timeout WITHOUT re-awaiting the
            # pending tasks, so a wedged task can't drag shutdown out (unlike
            # gather/wait_for, which block until cancellation completes).
            done, pending = await asyncio.wait(self._tasks, timeout=timeout)
            for task in done:
                if not task.cancelled() and task.exception() is not None:
                    logger.debug("task %s ended with: %s", task.get_name(), task.exception())
            if pending:
                abandoned = len(pending)
                logger.warning(
                    "shutdown timed out after %.0fs; abandoning %d stuck task(s): %s",
                    timeout,
                    abandoned,
                    ", ".join(sorted(t.get_name() for t in pending)),
                )
        self._tasks = []
        self.rest.close()
        self.db.close()
        return abandoned
