"""The autonomous trade engine.

Per watched game it:
  1. On each order-book update, builds features, asks the learner for a P(good)
     entry score, runs the one-share scalp strategy, and executes (paper/live).
  2. When the game ends (the market resolves), flattens any open position,
     self-analyzes its OWN trades (good = profit, bad = loss), retrains the
     learner on its accumulated round-trips, and moves on to the next game.

The per-tick logic (`on_book`) and the end-of-game logic (`finalize_game`) are
synchronous and unit-testable; `run` wires them to the live WebSocket + activity
streams.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from polyml.analysis.features import FeatureBuilder
from polyml.analysis.learner import Learner
from polyml.api.auth import Ed25519Signer
from polyml.api.rest import RestClient
from polyml.api.websocket import MarketsWebSocket, PrivateWebSocket
from polyml.config import Config
from polyml.mirror import ActivityMirror, ActivityPoller
from polyml.session import SessionManager
from polyml.session.manager import TERMINAL_STATES
from polyml.storage.db import Database
from polyml.storage.models import OrderBook
from polyml.trading.executor import Executor, ExecutorConfig
from polyml.trading.strategy import ScalpStrategy, StrategyConfig

logger = logging.getLogger(__name__)


class TradeEngine:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.db = Database(config.db_path)
        if not config.credentials.is_complete:
            raise RuntimeError("Trading requires credentials (POLYMARKET_KEY_ID / SECRET_KEY).")
        self.signer = Ed25519Signer.from_credentials(
            config.credentials.key_id, config.credentials.secret_key
        )
        self.rest = RestClient(
            config.rest_base_url, config.gateway_base_url, signer=self.signer,
            timeout=config.get("api.timeout_seconds", 30.0),
        )

        self.strategy = ScalpStrategy(self._strategy_config())
        self.session_manager = SessionManager(self.db, on_conclude=self._on_conclude)
        self.executor = Executor(
            self.rest, self.db, self._executor_config(),
            session_for=lambda slug: self.session_manager.open_session(slug),
        )
        self.features = FeatureBuilder(self.db)
        self.learner = Learner(
            self.db,
            model=config.get("learning.model", "gradient_boosting"),
            min_decisions=config.get("trading.min_trades_to_train", 20),
            holdout_fraction=config.get("learning.holdout_fraction", 0.25),
        )
        self._trained = False
        self._finalized: set[str] = set()
        self.watchlist: set[str] = set(config.get("trading.market_slugs", []) or [])
        self.markets_ws = MarketsWebSocket(config.ws_markets_url, self.signer, self._on_market_message)
        self.mirror = ActivityMirror(self.db)
        self.private_ws = PrivateWebSocket(config.ws_private_url, self.signer, self.mirror.handle)
        self.activity_poller = ActivityPoller(
            self.rest, self.db,
            interval=config.get("collectors.activities_interval", 30),
            on_resolution=self._on_resolution,
        )
        self._tasks: list[asyncio.Task] = []

    # --- config helpers ----------------------------------------------------------
    def _strategy_config(self) -> StrategyConfig:
        t = self.config.get("trading", {}) or {}
        return StrategyConfig(
            profit_hurdle_usd=t.get("profit_hurdle_usd", 0.01),
            max_spread=t.get("max_spread", 0.05),
            min_price=t.get("min_price", 0.05),
            max_price=t.get("max_price", 0.95),
            max_target_move=t.get("max_target_move", 0.08),
            entry_imbalance_min=t.get("entry_imbalance_min", 0.10),
            use_model_gate=t.get("use_model_gate", True),
            model_threshold=t.get("model_threshold", 0.50),
            stop_loss_usd=t.get("stop_loss_usd", 0.10),
            flatten_before_close=t.get("flatten_before_close", True),
            close_buffer_minutes=t.get("close_buffer_minutes", 5.0),
        )

    def _executor_config(self) -> ExecutorConfig:
        t = self.config.get("trading", {}) or {}
        return ExecutorConfig(
            mode=t.get("mode", "paper"),
            max_open_positions=t.get("max_open_positions", 1),
            daily_loss_limit_usd=t.get("daily_loss_limit_usd", 2.0),
        )

    # --- per-tick logic (testable) ----------------------------------------------
    def on_book(self, slug: str, book: OrderBook, minutes_to_close: float | None = None) -> Any:
        """Run the strategy for one book update and execute the resulting action."""
        # If the market has resolved, the game is over: finalize instead of trading.
        if book.state in TERMINAL_STATES:
            self.finalize_game(slug, book=book)
            return None
        if slug in self._finalized:
            return None
        position = self.executor.position(slug)
        features = self.features.build(
            slug, _now_iso(), decision_price=book.best_ask if position is None else book.best_bid
        )
        score = self._model_score(features) if (position is None and self._trained) else None
        action = self.strategy.decide(
            book, position, model_score=score, minutes_to_close=minutes_to_close
        )
        if action.kind in ("HOLD", "SKIP"):
            logger.debug("[%s] %s: %s", slug, action.kind, action.reason)
            return None
        return self.executor.execute(action, book, slug, features=features)

    def _model_score(self, features: dict) -> float | None:
        try:
            return self.learner.score_decision(features)
        except Exception:  # noqa: BLE001
            return None

    # --- end of game (testable) --------------------------------------------------
    def finalize_game(self, slug: str, book: OrderBook | None = None) -> dict[str, Any]:
        """Flatten, self-analyze the bot's trades for this game, learn, continue."""
        if slug in self._finalized:
            return self._self_analysis(slug)
        self._finalized.add(slug)
        if self.executor.has_position(slug):
            # Settle the open position: prefer a live bid, else the resolved value
            # / last known mid (a game-over book often has no bid).
            settle = self._settlement_price(slug, book)
            self.executor.force_flatten(slug, book=book, price=settle, reason="game over — settle")
        report = self._self_analysis(slug)
        # Learn from the bot's own round-trips so the next game starts smarter.
        result = self.learner.train(source="bot")
        self._trained = result.trained or self._trained
        report["learning"] = result.summary()
        sid = self.session_manager.open_session(slug)
        self.db.mark_session_analyzed(sid, json.dumps(report, default=str))
        logger.info("[%s] game finalized: net %+.4f over %d trades (%d win / %d loss)",
                    slug, report["net_pnl"], report["n_trades"], report["wins"], report["losses"])
        for lesson in report["lessons"]:
            logger.info("  lesson: %s", lesson)
        self.watchlist.discard(slug)
        return report

    def _settlement_price(self, slug: str, book: OrderBook | None) -> float | None:
        if book is not None and book.best_bid is not None:
            return book.best_bid
        o = self.db.query_one("SELECT resolved_value FROM outcomes WHERE market_slug=?", (slug,))
        if o and o["resolved_value"] is not None:
            return float(o["resolved_value"])
        if book is not None and book.last_trade_px is not None:
            return book.last_trade_px
        b = self.db.query_one(
            "SELECT mid, last_trade_px FROM book_snapshots WHERE market_slug=? "
            "AND mid IS NOT NULL ORDER BY captured_at DESC LIMIT 1", (slug,)
        )
        if b:
            return b["mid"] if b["mid"] is not None else b["last_trade_px"]
        return None

    def _self_analysis(self, slug: str) -> dict[str, Any]:
        rows = self.db.query(
            "SELECT entry_price, exit_price, net_pnl, exit_reason, label_good "
            "FROM bot_trades WHERE market_slug=? ORDER BY id", (slug,)
        )
        trades = [dict(r) for r in rows]
        wins = [t for t in trades if t["net_pnl"] is not None and t["net_pnl"] > 0]
        losses = [t for t in trades if t["net_pnl"] is not None and t["net_pnl"] <= 0]
        net = sum(t["net_pnl"] or 0.0 for t in trades)
        lessons: list[str] = []
        n_stops = sum(1 for t in trades if "stop loss" in (t["exit_reason"] or ""))
        n_flat = sum(1 for t in trades if "flatten" in (t["exit_reason"] or ""))
        if losses:
            avg_loss = sum(t["net_pnl"] for t in losses) / len(losses)
            lessons.append(
                f"{len(losses)} losing trade(s), avg {avg_loss:+.4f}. "
                + (f"{n_stops} hit the stop loss; " if n_stops else "")
                + (f"{n_flat} were force-flattened at the close." if n_flat else "")
            )
        if wins:
            avg_win = sum(t["net_pnl"] for t in wins) / len(wins)
            lessons.append(f"{len(wins)} winning trade(s), avg {avg_win:+.4f} — scalps worked.")
        if losses and wins:
            if abs(sum(t["net_pnl"] for t in losses)) > sum(t["net_pnl"] for t in wins):
                lessons.append(
                    "Losses outweighed wins: the few losers cost more than the many small "
                    "winners — tighten entries (raise model_threshold) or the stop loss."
                )
        return {
            "market_slug": slug, "n_trades": len(trades), "wins": len(wins),
            "losses": len(losses), "net_pnl": round(net, 4), "lessons": lessons,
            "trades": trades,
        }

    # --- live wiring -------------------------------------------------------------
    async def _on_market_message(self, message: dict) -> None:
        if "marketData" in message or "bids" in message or "offers" in message:
            book = OrderBook.from_payload(message)
            if book.market_slug:
                await asyncio.to_thread(self.db.insert_book_snapshot, book, "ws", message)
                await asyncio.to_thread(self.on_book, book.market_slug, book,
                                        self._minutes_to_close(book.market_slug))

    def _on_resolution(self, slug: str, resolved_value, resolution_time, raw) -> None:
        if slug in self.watchlist or self.executor.has_position(slug):
            self.session_manager.conclude_if_resolved(slug)
            self.finalize_game(slug)

    def _on_conclude(self, session_id: int, slug: str) -> None:
        # SessionManager concluded a session (e.g., via stale sweep).
        self.finalize_game(slug)

    def _minutes_to_close(self, slug: str) -> float | None:
        row = self.db.query_one(
            "SELECT raw FROM market_snapshots WHERE market_slug=? ORDER BY captured_at DESC LIMIT 1",
            (slug,),
        )
        if not row:
            return None
        try:
            data = json.loads(row["raw"])
            market = data.get("market", data)
            end = market.get("endDate") or market.get("gameStartTime")
            if not end:
                return None
            from polyml.storage.models import parse_time
            dt = parse_time(end)
            if dt is None:
                return None
            from datetime import datetime, timezone
            delta = (dt - datetime.now(tz=timezone.utc)).total_seconds() / 60.0
            return delta
        except Exception:  # noqa: BLE001
            return None

    async def run(self) -> None:
        if not self.watchlist:
            await asyncio.to_thread(self._discover_games)
        if not self.watchlist:
            logger.warning("No games to trade. Set trading.market_slugs or hold positions.")
        for slug in self.watchlist:
            self.session_manager.open_session(slug)
        if self.watchlist:
            self.markets_ws.subscribe_market_data(list(self.watchlist))
        self.private_ws.subscribe_all()

        logger.warning("TradeEngine starting in %s mode; watching %d game(s).",
                       self.executor.mode.upper(), len(self.watchlist))
        self._tasks = [
            asyncio.create_task(self.markets_ws.run(), name="markets-ws"),
            asyncio.create_task(self.private_ws.run(), name="private-ws"),
            asyncio.create_task(self.activity_poller.run(), name="activities"),
            asyncio.create_task(self._monitor_loop(), name="monitor"),
        ]
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            pass

    def _discover_games(self) -> None:
        """Pick live games to trade.

        Prefers the markets you're actively in (positions / open orders) — those
        are liquid and certain to resolve — then fills remaining slots with other
        currently-open markets.
        """
        max_games = self.config.get("trading.max_games", 5)

        def _add(slug: str | None) -> bool:
            if not slug or slug in self.watchlist or slug in self._finalized:
                return False
            self.watchlist.add(slug)
            return len(self.watchlist) >= max_games

        # 1) Markets you currently hold or have orders in.
        for slug in self._involved_market_slugs():
            if _add(slug):
                return

        # 2) Other open markets.
        try:
            markets = self.rest.list_markets(
                limit=self.config.get("trading.discover_limit", 20), closed=False
            ) or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("game discovery failed: %s", exc)
            return
        for m in markets.get("markets", []):
            if self._is_resolved(m):
                continue
            if _add(m.get("slug")):
                return

    def _involved_market_slugs(self) -> list[str]:
        slugs: list[str] = []
        try:
            positions = (self.rest.get_positions() or {}).get("positions", {})
            if isinstance(positions, dict):
                slugs.extend(positions.keys())
        except Exception:  # noqa: BLE001
            pass
        try:
            orders = (self.rest.get_open_orders() or {}).get("orders", [])
            slugs.extend(o.get("marketSlug") for o in orders if o.get("marketSlug"))
        except Exception:  # noqa: BLE001
            pass
        # De-dup, preserve order.
        return list(dict.fromkeys(s for s in slugs if s))

    @staticmethod
    def _is_resolved(market: dict) -> bool:
        if market.get("state") in TERMINAL_STATES:
            return True
        if market.get("closed") is True or market.get("archived") is True:
            return True
        return market.get("ep3Status") == "EXPIRED"

    # --- monitor: detect game-over and bring in the next game --------------------
    async def _monitor_loop(self) -> None:
        interval = self.config.get("trading.monitor_interval", 25)
        while True:
            await asyncio.sleep(interval)
            try:
                await asyncio.to_thread(self._poll_market_states)
                await self._top_up_games()
            except Exception:  # noqa: BLE001
                logger.exception("monitor loop error")

    def _poll_market_states(self) -> None:
        for slug in list(self.watchlist):
            try:
                payload = self.rest.get_market(slug)
            except Exception:  # noqa: BLE001
                continue
            if not payload:
                continue
            market = payload.get("market", payload)
            self.db.insert_market_snapshot(
                slug, market.get("title") or market.get("question"), market.get("state"), payload
            )
            if self._is_resolved(market):
                logger.info("[%s] market resolved — finalizing game", slug)
                self.finalize_game(slug)

    async def _top_up_games(self) -> None:
        target = self.config.get("trading.max_games", 5)
        if len(self.watchlist) >= target:
            return
        before = set(self.watchlist)
        await asyncio.to_thread(self._discover_games)
        for slug in self.watchlist - before:
            self.session_manager.open_session(slug)
            await self.markets_ws.send_subscription(
                {"subscribe": {"requestId": f"md-{slug}",
                               "subscriptionType": "SUBSCRIPTION_TYPE_MARKET_DATA",
                               "marketSlugs": [slug]}}
            )
            logger.info("now trading new game: %s", slug)

    async def shutdown(self) -> None:
        logger.info("TradeEngine shutting down...")
        self.markets_ws.stop()
        self.private_ws.stop()
        self.activity_poller.stop()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self.rest.close()
        self.db.close()


def _now_iso() -> str:
    from polyml.storage.models import now_iso
    return now_iso()
