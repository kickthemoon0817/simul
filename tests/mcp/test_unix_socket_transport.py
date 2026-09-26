"""A Unix-socket transport for the bridge, for containerised Isaac Sim.

TCP port forwarding works (#119, #120 made it so), but it carries costs a
shared volume does not: ports to publish and keep from colliding, a docker-proxy
hop, and a network socket that exists at all. The discovery directory is already
bind-mounted between container and host, so a Unix socket beside the discovery
file gives the host a direct path to the bridge with **no published ports**.

The transport is additive: TCP remains the default and keeps working; the
socket appears only when ``socket_path`` is configured. The lessons of #120 are
encoded here as tests before the implementation existed:

* the container runs as root, so the socket file must be connectable
  (``connect(2)`` requires *write* permission on the socket inode) by the
  ordinary user on the host;
* a socket path is only trusted from a discovery file when it sits inside the
  discovery directory — the same boundary that already rejects non-loopback
  hosts.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import stat
import struct
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator

import pytest

from khemoo.simul.lifecycle import BridgeServerLifecycle
from khemoo.simul.protocol import BridgeResponse

from simul.adapters.isaac_socket_client import IsaacSocketClient
from simul.config import Settings
from simul.mcp import backends as backends_module
from simul.mcp import server as server_module
from tests.fakes import FakeFastMCP


@pytest.fixture
def sock_dir() -> Iterator[Path]:
    """A short scratch directory for socket files.

    ``sun_path`` holds 104 bytes on macOS (108 on Linux), and pytest's
    ``tmp_path`` under ``/var/folders/...`` already runs past that, so
    ``bind(2)`` would fail with "AF_UNIX path too long".
    """
    path = Path(tempfile.mkdtemp(prefix="simul-uds-", dir="/tmp")).resolve()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


async def _ping_handler(request: Any) -> BridgeResponse:
    return BridgeResponse.success(request.request_id, {"reachable": True, "via": "uds"})


def _lifecycle(sock_dir: Path) -> BridgeServerLifecycle:
    return BridgeServerLifecycle(
        host="127.0.0.1",
        port=0,
        request_handler=_ping_handler,
        socket_path=str(sock_dir / "bridge.sock"),
    )


async def _uds_round_trip(path: str) -> Dict[str, Any]:
    reader, writer = await asyncio.open_unix_connection(path)
    body = json.dumps(
        {"protocol_version": 1, "request_id": "uds-1", "action": "ping", "payload": {}}
    ).encode()
    writer.write(struct.pack(">I", len(body)) + body)
    await writer.drain()
    header = await reader.readexactly(4)
    payload = await reader.readexactly(struct.unpack(">I", header)[0])
    writer.close()
    await writer.wait_closed()
    return json.loads(payload.decode())


# ---------------------------------------------------------------------------
# Lifecycle: the listener itself
# ---------------------------------------------------------------------------


def test_bridge_answers_over_the_unix_socket(sock_dir: Path) -> None:
    async def _exercise() -> Dict[str, Any]:
        lifecycle = _lifecycle(sock_dir)
        await lifecycle.start()
        try:
            return await _uds_round_trip(str(sock_dir / "bridge.sock"))
        finally:
            await lifecycle.stop()

    response = asyncio.run(_exercise())

    assert response["status"] == "ok"
    assert response["payload"]["reachable"] is True


def test_tcp_still_serves_alongside_the_socket(sock_dir: Path) -> None:
    """Additive means additive: the TCP listener must not disappear."""

    async def _exercise() -> Dict[str, Any]:
        lifecycle = _lifecycle(sock_dir)
        await lifecycle.start()
        try:
            port = lifecycle.actual_port
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            body = json.dumps(
                {
                    "protocol_version": 1,
                    "request_id": "tcp-1",
                    "action": "ping",
                    "payload": {},
                }
            ).encode()
            writer.write(struct.pack(">I", len(body)) + body)
            await writer.drain()
            header = await reader.readexactly(4)
            payload = await reader.readexactly(struct.unpack(">I", header)[0])
            writer.close()
            await writer.wait_closed()
            return json.loads(payload.decode())
        finally:
            await lifecycle.stop()

    assert asyncio.run(_exercise())["status"] == "ok"


def test_socket_file_is_connectable_by_other_users(sock_dir: Path) -> None:
    """connect(2) on a Unix socket needs write permission on the inode.

    In the container the bridge runs as root (#120), so a default-mode socket
    on the shared volume would be one more root-owned file the host user cannot
    use — the same failure as the 0600 discovery file, one layer down.
    """

    async def _exercise() -> int:
        lifecycle = _lifecycle(sock_dir)
        await lifecycle.start()
        try:
            return stat.S_IMODE(os.stat(sock_dir / "bridge.sock").st_mode)
        finally:
            await lifecycle.stop()

    mode = asyncio.run(_exercise())

    assert mode & stat.S_IWOTH, f"socket mode {mode:o}: host user cannot connect"


def test_stale_socket_file_is_replaced_on_start(sock_dir: Path) -> None:
    """A crash leaves the socket file behind; rebinding must not need cleanup."""
    (sock_dir / "bridge.sock").touch()

    async def _exercise() -> Dict[str, Any]:
        lifecycle = _lifecycle(sock_dir)
        await lifecycle.start()
        try:
            return await _uds_round_trip(str(sock_dir / "bridge.sock"))
        finally:
            await lifecycle.stop()

    assert asyncio.run(_exercise())["status"] == "ok"


def test_stop_removes_the_socket_file(sock_dir: Path) -> None:
    async def _exercise() -> None:
        lifecycle = _lifecycle(sock_dir)
        await lifecycle.start()
        await lifecycle.stop()

    asyncio.run(_exercise())

    assert not (sock_dir / "bridge.sock").exists()


def test_discovery_file_advertises_the_socket_path(sock_dir: Path) -> None:
    async def _exercise() -> Dict[str, Any]:
        lifecycle = _lifecycle(sock_dir)
        await lifecycle.start()
        try:
            lifecycle.write_discovery_file(str(sock_dir), pid=7, vscode_port=8226)
            return json.loads((sock_dir / "simul-7.json").read_text())
        finally:
            await lifecycle.stop()

    written = asyncio.run(_exercise())

    assert written["socket_path"] == str(sock_dir / "bridge.sock")
    # TCP details stay, for clients that do not speak UDS.
    assert written["host"] == "127.0.0.1"
    assert written["port"]


# ---------------------------------------------------------------------------
# Client: dialling the socket
# ---------------------------------------------------------------------------


def test_client_bridge_request_uses_the_unix_socket(sock_dir: Path) -> None:
    async def _exercise() -> Dict[str, Any]:
        lifecycle = _lifecycle(sock_dir)
        await lifecycle.start()
        try:
            client = IsaacSocketClient(
                host="127.0.0.1",
                port=1,  # deliberately dead: TCP must not be touched
                bridge_host="127.0.0.1",
                bridge_port=1,
                bridge_socket_path=str(sock_dir / "bridge.sock"),
                prefer_bridge=True,
            )
            return await client.bridge_request("ping", {})
        finally:
            await lifecycle.stop()

    response = asyncio.run(_exercise())

    assert response["status"] == "ok"
    assert response["payload"]["via"] == "uds"


def test_client_ping_works_over_the_socket_alone(sock_dir: Path) -> None:
    """No TCP anywhere: the socket path alone must be enough to ping."""

    async def _exercise() -> bool:
        lifecycle = _lifecycle(sock_dir)
        await lifecycle.start()
        try:
            client = IsaacSocketClient(
                host="127.0.0.1",
                port=1,
                bridge_host="127.0.0.1",
                bridge_port=1,
                bridge_socket_path=str(sock_dir / "bridge.sock"),
                prefer_bridge=True,
                fallback_to_vscode=False,
            )
            return await client.ping()
        finally:
            await lifecycle.stop()

    assert asyncio.run(_exercise()) is True


# ---------------------------------------------------------------------------
# Discovery: building a UDS client from a discovery file
# ---------------------------------------------------------------------------


def _server(monkeypatch: pytest.MonkeyPatch, discovery_dir: Path) -> Any:
    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)
    monkeypatch.setattr(backends_module, "is_headless_available", lambda: False)
    monkeypatch.setattr(backends_module, "is_blender_available", lambda: False)
    monkeypatch.setattr(backends_module, "UnrealRuntimeAdapter", None)
    settings = Settings(
        isaac_sim={
            "discovery_dir": str(discovery_dir),
            "socket_port": 9999,
            "bridge_port": 9998,
        }
    )
    return server_module.SimulMCPServer(settings=settings, backends={"isaac"})


def _write_entry(discovery_dir: Path, socket_path: str, port: int = 8229) -> None:
    (discovery_dir / "simul-7.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),  # alive, so the stale-pid sweep keeps it
                "host": "127.0.0.1",
                "port": port,
                "vscode_port": 8226,
                "socket_path": socket_path,
            }
        )
    )


def test_discovery_builds_a_socket_client(
    monkeypatch: pytest.MonkeyPatch, sock_dir: Path
) -> None:
    async def _exercise() -> Dict[str, Any]:
        lifecycle = BridgeServerLifecycle(
            host="127.0.0.1",
            port=0,
            request_handler=_ping_handler,
            socket_path=str(sock_dir / "bridge.sock"),
        )
        await lifecycle.start()
        try:
            _write_entry(sock_dir, str(sock_dir / "bridge.sock"))
            srv = _server(monkeypatch, sock_dir)
            found = await srv._discover_from_files()
            assert found, "discovery returned nothing"
            client = next(iter(found.values()))
            return await client.bridge_request("ping", {})
        finally:
            await lifecycle.stop()

    response = asyncio.run(_exercise())

    assert response["payload"]["via"] == "uds"


def test_discovery_rejects_a_socket_outside_the_discovery_dir(
    monkeypatch: pytest.MonkeyPatch, sock_dir: Path
) -> None:
    """The discovery dir is the trust boundary, exactly as loopback is for TCP.

    A hostile or corrupted entry must not be able to point the client at an
    arbitrary socket elsewhere on the filesystem.
    """
    inside = sock_dir / "disc"
    inside.mkdir()
    outside = sock_dir / "elsewhere.sock"

    async def _exercise() -> Dict[str, Any]:
        _write_entry(inside, str(outside))
        srv = _server(monkeypatch, inside)
        found = await srv._discover_from_files()
        for client in found.values():
            assert (
                getattr(client, "_bridge_socket_path", None) is None
            ), "discovery accepted a socket path outside the discovery dir"
        return {}

    asyncio.run(_exercise())


def test_discovery_translates_a_container_side_socket_path(
    monkeypatch: pytest.MonkeyPatch, sock_dir: Path
) -> None:
    """The container advertises its own mount point, not the host's.

    The volume is ``$SIMUL_DISCOVERY_DIR:/tmp/simul``, so the file says
    ``/tmp/simul/bridge.sock`` while the host sees the same socket at
    ``$SIMUL_DISCOVERY_DIR/bridge.sock``. When the literal path is not inside
    the local discovery dir, the reader must try the basename inside it —
    which stays within the trust boundary by construction.
    """

    async def _exercise() -> Dict[str, Any]:
        lifecycle = BridgeServerLifecycle(
            host="127.0.0.1",
            port=0,
            request_handler=_ping_handler,
            socket_path=str(sock_dir / "bridge.sock"),
        )
        await lifecycle.start()
        try:
            # Advertise the path as a container would see it.
            _write_entry(sock_dir, "/tmp/simul/bridge.sock")
            srv = _server(monkeypatch, sock_dir)
            found = await srv._discover_from_files()
            assert found, "discovery returned nothing"
            return await next(iter(found.values())).bridge_request("ping", {})
        finally:
            await lifecycle.stop()

    assert asyncio.run(_exercise())["payload"]["via"] == "uds"


def test_discovery_falls_back_to_tcp_when_the_socket_path_is_too_long(
    monkeypatch: pytest.MonkeyPatch, sock_dir: Path
) -> None:
    """A host discovery dir deeper than ``sun_path`` allows must not break the bridge.

    The container advertises its short mount-point path, the reader
    translates it into the host dir, and ``connect(2)`` would then refuse the
    result on every request. Discovery keeps the TCP endpoint instead.
    """
    deep = sock_dir / ("d" * 120)
    deep.mkdir()
    (deep / "bridge.sock").touch()

    async def _exercise() -> Any:
        lifecycle = BridgeServerLifecycle(
            host="127.0.0.1", port=0, request_handler=_ping_handler
        )
        await lifecycle.start()
        try:
            _write_entry(deep, "/tmp/simul/bridge.sock", port=lifecycle.actual_port)
            srv = _server(monkeypatch, deep)
            found = await srv._discover_from_files()
            assert found, "discovery dropped the entry instead of falling back to TCP"
            return next(iter(found.values()))
        finally:
            await lifecycle.stop()

    client = asyncio.run(_exercise())

    assert client._bridge_socket_path is None


# ---------------------------------------------------------------------------
# Two instances sharing one discovery dir must stay distinct (#121 review)
# ---------------------------------------------------------------------------


def _tagged_lifecycle(sock_dir: Path, tag: str) -> BridgeServerLifecycle:
    async def handler(request: Any) -> BridgeResponse:
        return BridgeResponse.success(request.request_id, {"who": tag})

    return BridgeServerLifecycle(
        host="127.0.0.1",
        port=0,
        request_handler=handler,
        socket_path=str(sock_dir / "bridge.sock"),
    )


async def _who(path: str) -> str:
    reader, writer = await asyncio.open_unix_connection(path)
    body = json.dumps(
        {"protocol_version": 1, "request_id": "w", "action": "ping", "payload": {}}
    ).encode()
    writer.write(struct.pack(">I", len(body)) + body)
    await writer.drain()
    header = await reader.readexactly(4)
    payload = json.loads(
        (await reader.readexactly(struct.unpack(">I", header)[0])).decode()
    )
    writer.close()
    await writer.wait_closed()
    return payload["payload"]["who"]


def test_second_bridge_does_not_hijack_a_live_socket(sock_dir: Path) -> None:
    """Both containers ship the same configured name; both must stay reachable.

    Every container is pid 1 in its own namespace, so per-pid names collide
    identically — uniqueness has to come from refusing to steal a live socket
    and binding a generated sibling instead.
    """

    async def _exercise() -> tuple[str, str]:
        a = _tagged_lifecycle(sock_dir, "A")
        b = _tagged_lifecycle(sock_dir, "B")
        await a.start()
        await b.start()
        try:
            assert (
                a.actual_socket_path != b.actual_socket_path
            ), "both bridges claim the same socket path"
            return await _who(a.actual_socket_path), await _who(b.actual_socket_path)
        finally:
            await a.stop()
            await b.stop()

    who_a, who_b = asyncio.run(_exercise())

    assert (who_a, who_b) == ("A", "B")


def test_stopping_one_bridge_leaves_the_other_reachable(sock_dir: Path) -> None:
    """Unlink on stop must only remove a socket the stopper actually owns."""

    async def _exercise() -> str:
        a = _tagged_lifecycle(sock_dir, "A")
        b = _tagged_lifecycle(sock_dir, "B")
        await a.start()
        await b.start()
        try:
            await a.stop()
            return await _who(b.actual_socket_path)
        finally:
            await b.stop()

    assert asyncio.run(_exercise()) == "B"


def test_discovery_file_advertises_the_actual_socket(sock_dir: Path) -> None:
    """A generated sibling name must be what gets advertised."""

    async def _exercise() -> tuple[str, str]:
        a = _tagged_lifecycle(sock_dir, "A")
        b = _tagged_lifecycle(sock_dir, "B")
        await a.start()
        await b.start()
        try:
            b.write_discovery_file(str(sock_dir), pid=99)
            written = json.loads((sock_dir / "simul-99.json").read_text())
            return written["socket_path"], b.actual_socket_path
        finally:
            await a.stop()
            await b.stop()

    advertised, actual = asyncio.run(_exercise())

    assert advertised == actual


def test_two_discovered_instances_resolve_to_distinct_backends(
    monkeypatch: pytest.MonkeyPatch, sock_dir: Path
) -> None:
    """The end-to-end misrouting from the review: tools for instance A must not
    execute inside instance B."""

    async def _exercise() -> tuple[str, str]:
        a = _tagged_lifecycle(sock_dir, "A")
        b = _tagged_lifecycle(sock_dir, "B")
        await a.start()
        await b.start()
        try:
            for i, lc in enumerate((a, b)):
                (sock_dir / f"simul-{100 + i}.json").write_text(
                    json.dumps(
                        {
                            "pid": os.getpid(),
                            "host": "127.0.0.1",
                            "port": 43000 + i,
                            "vscode_port": 42000 + i,
                            "socket_path": lc.actual_socket_path,
                        }
                    )
                )
            srv = _server(monkeypatch, sock_dir)
            found = await srv._discover_from_files()
            assert len(found) == 2, f"expected 2 instances, found {list(found)}"
            answers = []
            for client in found.values():
                response = await client.bridge_request("ping", {})
                answers.append(response["payload"]["who"])
            return tuple(sorted(answers))
        finally:
            await a.stop()
            await b.stop()

    assert asyncio.run(_exercise()) == ("A", "B")


def test_socket_client_failures_name_the_socket_not_the_tcp_pair(
    sock_dir: Path,
) -> None:
    """Diagnostics must point at the endpoint that was dialled.

    A socket-only client reporting "127.0.0.1:8229" — or worse, "None:None" —
    sends the operator to a port that was never touched: the same
    misdirection class #119/#120 were about.
    """
    sock = str(sock_dir / "nothing-here.sock")
    client = IsaacSocketClient(
        host="127.0.0.1",
        port=1,
        bridge_host=None,
        bridge_port=None,
        bridge_socket_path=sock,
        prefer_bridge=True,
        fallback_to_vscode=False,
    )

    assert client.bridge_endpoint == sock
    assert "None" not in client.address

    async def _exercise() -> str:
        try:
            await client.bridge_request("ping", {})
        except ConnectionRefusedError as exc:
            return str(exc)
        raise AssertionError("connect to a missing socket should refuse")

    message = asyncio.run(_exercise())

    assert sock in message
    assert "8229" not in message
