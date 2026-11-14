"""A TCP server for fake engines that stops even with clients still connected.

From Python 3.12, ``asyncio.Server.wait_closed()`` waits until every client
connection has closed. A fake whose handler is still inside a long "script"
(or never answers, on purpose) kept its connection open, so ``stop()`` hung
there forever on 3.12 and 3.13 while 3.11 returned at once. This server
tracks each client's writer, closes them all on ``stop()``, and bounds the
wait.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Optional, Set

Handler = Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]]


class FakeTcpServer:
    """``asyncio.start_server`` on an ephemeral loopback port, with a ``stop()`` that cannot hang.

    Args:
        handler: The connection handler, as ``asyncio.start_server`` takes it.
    """

    def __init__(self, handler: Handler) -> None:
        self._handler = handler
        self._writers: Set[asyncio.StreamWriter] = set()
        self._server: Optional[asyncio.AbstractServer] = None
        self.port = 0

    async def start(self) -> int:
        """Listen on 127.0.0.1 and return the port."""
        self._server = await asyncio.start_server(self._serve, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self.port

    async def stop(self, timeout: float = 2.0) -> None:
        """Stop listening, close every client connection, and wait at most ``timeout`` seconds."""
        if self._server is None:
            return
        self._server.close()
        for writer in list(self._writers):
            writer.close()
        try:
            await asyncio.wait_for(self._server.wait_closed(), timeout)
        except asyncio.TimeoutError:
            pass

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._writers.add(writer)
        try:
            await self._handler(reader, writer)
        finally:
            self._writers.discard(writer)
