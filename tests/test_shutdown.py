"""Shutdown is bounded: stopping the bot never hangs on a wedged task.

These exercise ``Runner.shutdown`` directly (without standing up the full live
stack) to verify that well-behaved tasks are torn down promptly and that a task
stuck in uninterruptible work is abandoned rather than hung on.
"""

from __future__ import annotations

import asyncio
import contextlib
import time

from polyml.runner import Runner


class _Stub:
    """Stands in for a collector / websocket / rest client / db."""

    def __init__(self) -> None:
        self.stopped = False
        self.closed = False

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


def _make_runner(tasks: list[asyncio.Task]) -> tuple[Runner, dict[str, _Stub]]:
    runner = Runner.__new__(Runner)  # bypass __init__ (which needs credentials)
    stubs = {name: _Stub() for name in (
        "market_collector", "account_collector", "activity_poller",
        "private_ws", "markets_ws", "rest", "db",
    )}
    for name, stub in stubs.items():
        setattr(runner, name, stub)
    runner._tasks = tasks
    return runner, stubs


def test_shutdown_cancels_wellbehaved_tasks_promptly() -> None:
    async def _inner() -> int:
        async def loop() -> None:
            while True:
                await asyncio.sleep(3600)

        tasks = [asyncio.create_task(loop(), name=f"t{i}") for i in range(3)]
        await asyncio.sleep(0)  # let them start
        runner, stubs = _make_runner(tasks)

        start = time.monotonic()
        abandoned = await runner.shutdown(timeout=5.0)
        elapsed = time.monotonic() - start

        assert abandoned == 0
        assert elapsed < 1.0  # cancellation is immediate, nowhere near the timeout
        assert all(t.cancelled() for t in tasks)
        assert all(stubs[n].stopped for n in ("market_collector", "private_ws", "markets_ws"))
        assert stubs["rest"].closed and stubs["db"].closed
        return abandoned

    asyncio.run(_inner())


def test_shutdown_abandons_slow_to_cancel_task_without_hanging() -> None:
    async def _inner() -> int:
        async def slow_to_cancel() -> None:
            # Models a task whose teardown (e.g. a websocket close handshake)
            # runs during cancellation and takes longer than the grace period.
            try:
                await asyncio.sleep(3600)
            finally:
                await asyncio.sleep(1.0)

        task = asyncio.create_task(slow_to_cancel(), name="slow")
        await asyncio.sleep(0)  # let it reach the outer await
        runner, _ = _make_runner([task])

        start = time.monotonic()
        abandoned = await runner.shutdown(timeout=0.3)
        elapsed = time.monotonic() - start

        # Returns at the grace deadline, not after the 1s teardown completes.
        assert abandoned == 1
        assert elapsed < 0.9
        # The task is still finishing its teardown; let it complete so the loop
        # can close cleanly (it does honour cancellation, just slowly).
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return abandoned

    assert asyncio.run(_inner()) == 1
