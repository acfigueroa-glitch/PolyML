"""PolyML command-line interface.

    polyml run                Start the live observe-and-learn loop (needs creds).
    polyml status             Show what's been collected so far.
    polyml analyze [--all]    Link decisions to outcomes for concluded sessions.
    polyml report [--slug S]  Print the analysis report for a session.
    polyml train              Train the learner on accumulated decisions.
    polyml discover           List markets you're involved in / available markets.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
from pathlib import Path

from polyml.analysis.learner import Learner
from polyml.analysis.outcomes import OutcomeLinker
from polyml.config import load_config
from polyml.storage.db import Database


def _setup_logging(config) -> None:
    level = getattr(logging, str(config.get("logging.level", "INFO")).upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    log_file = config.get("logging.file")
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=handlers,
    )


# --- commands -------------------------------------------------------------------
def cmd_run(config) -> int:
    from polyml.runner import Runner

    try:
        runner = Runner(config)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    async def _main() -> None:
        loop = asyncio.get_running_loop()
        stop = asyncio.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop.set)
            except NotImplementedError:  # pragma: no cover - Windows
                pass
        run_task = asyncio.create_task(runner.run())
        await stop.wait()
        await runner.shutdown()
        run_task.cancel()

    asyncio.run(_main())
    return 0


def cmd_status(config) -> int:
    db = Database(config.db_path)
    tables = [
        "market_snapshots", "book_snapshots", "market_trades", "balance_snapshots",
        "position_snapshots", "order_events", "activities", "sessions",
        "outcomes", "decisions", "learning_runs",
    ]
    print(f"Database: {config.db_path}")
    for table in tables:
        row = db.query_one(f"SELECT COUNT(*) AS n FROM {table}")
        print(f"  {table:<20} {row['n']:>8}")
    print("\nSessions by status:")
    for row in db.query("SELECT status, COUNT(*) AS n FROM sessions GROUP BY status"):
        print(f"  {row['status']:<12} {row['n']}")
    bal = db.query_one("SELECT * FROM balance_snapshots ORDER BY captured_at DESC LIMIT 1")
    if bal:
        print(f"\nLatest balance: buying_power={bal['buying_power']} total_value={bal['total_value']}")
    db.close()
    return 0


def cmd_analyze(config, args) -> int:
    db = Database(config.db_path)
    linker = OutcomeLinker(db)
    if args.slug:
        sessions = db.query(
            "SELECT id, market_slug FROM sessions WHERE market_slug=? AND status IN ('concluded','analyzed')",
            (args.slug,),
        )
    else:
        status = "concluded" if not args.all else "concluded','analyzed"
        sessions = db.query(
            f"SELECT id, market_slug FROM sessions WHERE status IN ('{status}')"
        )
    if not sessions:
        print("No concluded sessions to analyze. (Sessions conclude when their market resolves.)")
        db.close()
        return 0
    for row in sessions:
        report = linker.link_session(int(row["id"]), row["market_slug"])
        db.mark_session_analyzed(int(row["id"]), json.dumps(report, default=str))
        print(f"Analyzed session {row['id']} ({row['market_slug']}): "
              f"pnl={report['realized_pnl']} good={report['good_decisions']} bad={report['bad_decisions']}")
    # Retrain after analysis.
    learner = _make_learner(config, db)
    print("\n" + learner.train().summary())
    db.close()
    return 0


def cmd_report(config, args) -> int:
    db = Database(config.db_path)
    if args.session:
        row = db.query_one("SELECT * FROM sessions WHERE id=?", (args.session,))
    elif args.slug:
        row = db.query_one(
            "SELECT * FROM sessions WHERE market_slug=? ORDER BY id DESC LIMIT 1", (args.slug,)
        )
    else:
        row = db.query_one("SELECT * FROM sessions WHERE status='analyzed' ORDER BY analyzed_at DESC LIMIT 1")
    if not row:
        print("No analyzed session found. Run `polyml analyze` first.")
        db.close()
        return 1
    print(f"=== Session {row['id']}  market={row['market_slug']}  status={row['status']} ===")
    print(f"started={row['started_at']}  concluded={row['concluded_at']}")
    print(f"outcome={row['outcome_value']}  realized_pnl={row['realized_pnl']}")
    if row["summary"]:
        report = json.loads(row["summary"])
        print(f"\nDecisions: {report['n_decisions']}  good={report['good_decisions']}  bad={report['bad_decisions']}")
        for d in report.get("decisions", []):
            print(f"  - {d['decided_at']} {d['decision_type']} {d['side']} "
                  f"@ {d['price']} x{d['size']}  good={d['label_good']}")
        if report.get("lessons"):
            print("\nLessons:")
            for lesson in report["lessons"]:
                print(f"  • {lesson}")
    db.close()
    return 0


def cmd_backfill(config, args) -> int:
    """Pull settled history, build sessions for resolved markets, analyze, train."""
    from polyml.api.auth import Ed25519Signer
    from polyml.api.rest import RestClient
    from polyml.mirror import ActivityPoller
    from polyml.session import SessionManager

    if not config.credentials.is_complete:
        print("error: backfill needs credentials (POLYMARKET_KEY_ID / POLYMARKET_SECRET_KEY).",
              file=sys.stderr)
        return 2
    signer = Ed25519Signer.from_credentials(config.credentials.key_id, config.credentials.secret_key)
    rest = RestClient(config.rest_base_url, config.gateway_base_url, signer=signer)
    db = Database(config.db_path)

    poller = ActivityPoller(rest, db)
    n = poller.backfill(max_pages=args.pages)
    print(f"Fetched {n} activities across up to {args.pages} pages.")

    sm = SessionManager(db)
    linker = OutcomeLinker(db)
    outcomes = db.query("SELECT market_slug FROM outcomes")
    print(f"Resolved markets with outcomes: {len(outcomes)}")
    for row in outcomes:
        slug = row["market_slug"]
        sm.open_session(slug)
        sm.conclude_if_resolved(slug)
        s = db.query_one("SELECT id FROM sessions WHERE market_slug=? ORDER BY id DESC LIMIT 1", (slug,))
        report = linker.link_session(int(s["id"]), slug)
        db.mark_session_analyzed(int(s["id"]), json.dumps(report, default=str))
        print(f"  {slug}: pnl={report['realized_pnl']:+.2f} "
              f"good={report['good_decisions']} bad={report['bad_decisions']} "
              f"decisions={report['n_decisions']}")

    learner = _make_learner(config, db)
    print("\n" + learner.train().summary())
    rest.close()
    db.close()
    return 0


def cmd_train(config) -> int:
    db = Database(config.db_path)
    learner = _make_learner(config, db)
    print(learner.train().summary())
    db.close()
    return 0


def cmd_discover(config) -> int:
    from polyml.api.auth import Ed25519Signer
    from polyml.api.rest import RestClient

    signer = None
    if config.credentials.is_complete:
        signer = Ed25519Signer.from_credentials(
            config.credentials.key_id, config.credentials.secret_key
        )
    rest = RestClient(config.rest_base_url, config.gateway_base_url, signer=signer)
    if signer:
        print("Markets you're involved in:")
        try:
            positions = rest.get_positions() or {}
            for pos in positions.get("positions", []):
                print(f"  {pos.get('marketSlug')}  net={pos.get('netPositionDecimal')}")
        except Exception as exc:  # noqa: BLE001
            print(f"  (could not fetch positions: {exc})")
    print("\nAvailable markets:")
    try:
        markets = rest.list_markets(limit=config.get("watch.discover_limit", 50)) or {}
        for m in markets.get("markets", [])[:50]:
            print(f"  {m.get('slug')}  {m.get('title') or m.get('question')}  [{m.get('state')}]")
    except Exception as exc:  # noqa: BLE001
        print(f"  (could not fetch markets: {exc})")
    rest.close()
    return 0


def _make_learner(config, db) -> Learner:
    return Learner(
        db,
        model=config.get("learning.model", "gradient_boosting"),
        min_decisions=config.get("learning.min_decisions_to_train", 30),
        holdout_fraction=config.get("learning.holdout_fraction", 0.25),
        random_state=config.get("learning.random_state", 42),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="polyml", description="Observe Polymarket US and learn from your trading.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("run", help="Start the live observe-and-learn loop")
    sub.add_parser("status", help="Show collection statistics")

    p_analyze = sub.add_parser("analyze", help="Link decisions to outcomes for concluded sessions")
    p_analyze.add_argument("--slug", help="Only analyze this market")
    p_analyze.add_argument("--all", action="store_true", help="Re-analyze already-analyzed sessions too")

    p_report = sub.add_parser("report", help="Print a session analysis report")
    p_report.add_argument("--slug", help="Market slug")
    p_report.add_argument("--session", type=int, help="Session id")

    sub.add_parser("train", help="Train the learner on accumulated decisions")
    sub.add_parser("discover", help="List markets you're involved in and available markets")

    p_backfill = sub.add_parser("backfill", help="Pull settled history and analyze resolved markets")
    p_backfill.add_argument("--pages", type=int, default=20, help="Max activity pages to fetch")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config()
    _setup_logging(config)

    if args.command == "run":
        return cmd_run(config)
    if args.command == "status":
        return cmd_status(config)
    if args.command == "analyze":
        return cmd_analyze(config, args)
    if args.command == "report":
        return cmd_report(config, args)
    if args.command == "train":
        return cmd_train(config)
    if args.command == "discover":
        return cmd_discover(config)
    if args.command == "backfill":
        return cmd_backfill(config, args)
    parser.error(f"unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
