"""Small helpers for running coroutines from Streamlit's synchronous script thread."""
from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict


def run_coroutine(coro: Any) -> Any:
    """Run a coroutine to completion without leaking a half-closed event loop.

    ``asyncio.run`` shuts the loop down properly: pending tasks (for example
    Playwright's transport task started during scraping) are cancelled and
    awaited before the loop closes. A bare ``new_event_loop``/``close`` pair
    leaves those tasks behind and logs "Task was destroyed but it is pending".

    If a loop is already running in the current thread, the coroutine is
    executed in a short-lived worker thread with its own loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    outcome: Dict[str, Any] = {}

    def _runner() -> None:
        try:
            outcome["result"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
            outcome["error"] = exc

    thread = threading.Thread(target=_runner)
    thread.start()
    thread.join()

    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("result")
