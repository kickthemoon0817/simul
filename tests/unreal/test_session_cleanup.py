"""UnrealRuntimeSession closes its aiohttp session, and cleanup() keeps its task.

``cleanup()`` used to schedule ``create_task(close())`` and drop the task:
the event loop holds tasks only weakly, so the close could be collected
before it ran. The CLI never closed its sessions at all, and aiohttp printed
"Unclosed client session" / "Unclosed connector" on stderr at exit.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Iterator, List

import pytest
from aiohttp import web

from simul.adapters import unreal_runtime
from simul.adapters.unreal_runtime import UnrealRuntimeAdapter, UnrealRuntimeSession
from simul.config import Settings


@pytest.fixture(autouse=True)
def _no_leftover_close_tasks() -> Iterator[None]:
    yield
    unreal_runtime._PENDING_CLOSES.clear()


class _HttpSession:
    """Stands in for aiohttp.ClientSession; closing takes a loop iteration."""

    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        await asyncio.sleep(0)
        self.closed = True


def test_cleanup_without_an_http_session_schedules_nothing() -> None:
    session = UnrealRuntimeSession(Settings())

    session.cleanup()  # no running loop, and none is created

    assert session._session is None
    assert not unreal_runtime._PENDING_CLOSES


def test_cleanup_on_a_running_loop_holds_the_close_task_until_it_finishes() -> None:
    async def body() -> None:
        session = UnrealRuntimeSession(Settings())
        http = session._session = _HttpSession()

        session.cleanup()

        # The only strong reference: without it the task could be collected.
        (task,) = unreal_runtime._PENDING_CLOSES
        await task
        assert http.closed is True
        assert session._session is None
        assert not unreal_runtime._PENDING_CLOSES

    asyncio.run(body())


def test_adapter_session_scope_closes_the_http_session() -> None:
    """The MCP server's path: a sync ``with`` around an async tool call."""

    async def body() -> _HttpSession:
        adapter = UnrealRuntimeAdapter(Settings())
        with adapter.create_session() as session:
            http = session._session = _HttpSession()
        await asyncio.gather(*unreal_runtime._PENDING_CLOSES)
        return http

    assert asyncio.run(body()).closed is True


@asynccontextmanager
async def _remote_control() -> AsyncIterator[Settings]:
    async def info(request: web.Request) -> web.Response:
        return web.json_response({"HttpRoutes": []})

    app = web.Application()
    app.router.add_get("/remote/info", info)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    base = Settings()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield base.model_copy(
            update={"unreal": base.unreal.model_copy(update={"host": "127.0.0.1", "port": port})}
        )
    finally:
        await runner.cleanup()


def test_async_with_closes_the_real_aiohttp_session() -> None:
    async def body() -> List[Any]:
        async with _remote_control() as settings:
            async with UnrealRuntimeSession(settings) as session:
                await session._http_get("/remote/info")
                http = session._session
                assert http is not None and not http.closed
            return [http.closed, session._session]

    assert asyncio.run(body()) == [True, None]


def test_close_is_idempotent() -> None:
    async def body() -> None:
        session = UnrealRuntimeSession(Settings())
        session._session = _HttpSession()
        await session.close()
        await session.close()
        assert session._session is None

    asyncio.run(body())
