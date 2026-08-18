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
import stat
import struct
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(
    0, str(repo_root / "src" / "simul_mcp" / "bridge_ext" / "khemoo.simul.mcp")
)

from khemoo.simul.mcp.lifecycle import BridgeServerLifecycle  # noqa: E402
from khemoo.simul.mcp.protocol import BridgeResponse  # noqa: E402

from simul_mcp.adapters.isaac_socket_client import IsaacSocketClient  # noqa: E402
from simul_mcp.config import Settings  # noqa: E402
from simul_mcp.mcp import server as server_module  # noqa: E402


async def _ping_handler(request: Any) -> BridgeResponse:
    return BridgeResponse.success(request.request_id, {"reachable": True, "via": "uds"})


def _lifecycle(tmp_path: Path) -> BridgeServerLifecycle:
    return BridgeServerLifecycle(
        host="127.0.0.1",
        port=0,
        request_handler=_ping_handler,
        socket_path=str(tmp_path / "bridge.sock"),
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


def test_bridge_answers_over_the_unix_socket(tmp_path: Path) -> None:
    async def _exercise() -> Dict[str, Any]:
        lifecycle = _lifecycle(tmp_path)
        await lifecycle.start()
        try:
            return await _uds_round_trip(str(tmp_path / "bridge.sock"))
        finally:
            await lifecycle.stop()

    response = asyncio.run(_exercise())

    assert response["status"] == "ok"
    assert response["payload"]["reachable"] is True


def test_tcp_still_serves_alongside_the_socket(tmp_path: Path) -> None:
    """Additive means additive: the TCP listener must not disappear."""

    async def _exercise() -> Dict[str, Any]:
        lifecycle = _lifecycle(tmp_path)
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


def test_socket_file_is_connectable_by_other_users(tmp_path: Path) -> None:
    """connect(2) on a Unix socket needs write permission on the inode.

    In the container the bridge runs as root (#120), so a default-mode socket
    on the shared volume would be one more root-owned file the host user cannot
    use — the same failure as the 0600 discovery file, one layer down.
    """

    async def _exercise() -> int:
        lifecycle = _lifecycle(tmp_path)
        await lifecycle.start()
        try:
            return stat.S_IMODE(os.stat(tmp_path / "bridge.sock").st_mode)
        finally:
            await lifecycle.stop()

    mode = asyncio.run(_exercise())

    assert mode & stat.S_IWOTH, f"socket mode {mode:o}: host user cannot connect"


def test_stale_socket_file_is_replaced_on_start(tmp_path: Path) -> None:
    """A crash leaves the socket file behind; rebinding must not need cleanup."""
    (tmp_path / "bridge.sock").touch()

    async def _exercise() -> Dict[str, Any]:
        lifecycle = _lifecycle(tmp_path)
        await lifecycle.start()
        try:
            return await _uds_round_trip(str(tmp_path / "bridge.sock"))
        finally:
            await lifecycle.stop()

    assert asyncio.run(_exercise())["status"] == "ok"


def test_stop_removes_the_socket_file(tmp_path: Path) -> None:
    async def _exercise() -> None:
        lifecycle = _lifecycle(tmp_path)
        await lifecycle.start()
        await lifecycle.stop()

    asyncio.run(_exercise())

    assert not (tmp_path / "bridge.sock").exists()


def test_discovery_file_advertises_the_socket_path(tmp_path: Path) -> None:
    async def _exercise() -> Dict[str, Any]:
        lifecycle = _lifecycle(tmp_path)
        await lifecycle.start()
        try:
            lifecycle.write_discovery_file(str(tmp_path), pid=7, vscode_port=8226)
            return json.loads((tmp_path / "simul-mcp-7.json").read_text())
        finally:
            await lifecycle.stop()

    written = asyncio.run(_exercise())

    assert written["socket_path"] == str(tmp_path / "bridge.sock")
    # TCP details stay, for clients that do not speak UDS.
    assert written["host"] == "127.0.0.1"
    assert written["port"]


# ---------------------------------------------------------------------------
# Client: dialling the socket
# ---------------------------------------------------------------------------


def test_client_bridge_request_uses_the_unix_socket(tmp_path: Path) -> None:
    async def _exercise() -> Dict[str, Any]:
        lifecycle = _lifecycle(tmp_path)
        await lifecycle.start()
        try:
            client = IsaacSocketClient(
                host="127.0.0.1",
                port=1,  # deliberately dead: TCP must not be touched
                bridge_host="127.0.0.1",
                bridge_port=1,
                bridge_socket_path=str(tmp_path / "bridge.sock"),
                prefer_bridge=True,
            )
            return await client.bridge_request("ping", {})
        finally:
            await lifecycle.stop()

    response = asyncio.run(_exercise())

    assert response["status"] == "ok"
    assert response["payload"]["via"] == "uds"


def test_client_ping_works_over_the_socket_alone(tmp_path: Path) -> None:
    """No TCP anywhere: the socket path alone must be enough to ping."""

    async def _exercise() -> bool:
        lifecycle = _lifecycle(tmp_path)
        await lifecycle.start()
        try:
            client = IsaacSocketClient(
                host="127.0.0.1",
                port=1,
                bridge_host="127.0.0.1",
                bridge_port=1,
                bridge_socket_path=str(tmp_path / "bridge.sock"),
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


class _FakeFastMCP:
    def __init__(self, *args: Any, **kwargs: Any):
        pass

    def tool(self, *args: Any, **kwargs: Any):
        def decorator(func):
            return func

        return decorator

    def resource(self, *args: Any, **kwargs: Any):
        def decorator(func):
            return func

        return decorator

    def add_middleware(self, middleware: Any) -> None:
        return


def _server(monkeypatch: pytest.MonkeyPatch, discovery_dir: Path) -> Any:
    monkeypatch.setattr(server_module, "FastMCP", _FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)
    monkeypatch.setattr(server_module, "is_headless_available", lambda: False)
    monkeypatch.setattr(server_module, "is_blender_available", lambda: False)
    monkeypatch.setattr(server_module, "UnrealRuntimeAdapter", None)
    settings = Settings(
        isaac_sim={
            "discovery_dir": str(discovery_dir),
            "socket_port": 9999,
            "bridge_port": 9998,
        }
    )
    return server_module.SimulMCPServer(settings=settings, backends={"isaac"})


def _write_entry(discovery_dir: Path, socket_path: str) -> None:
    (discovery_dir / "simul-mcp-7.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),  # alive, so the stale-pid sweep keeps it
                "host": "127.0.0.1",
                "port": 8229,
                "vscode_port": 8226,
                "socket_path": socket_path,
            }
        )
    )


def test_discovery_builds_a_socket_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def _exercise() -> Dict[str, Any]:
        lifecycle = BridgeServerLifecycle(
            host="127.0.0.1",
            port=0,
            request_handler=_ping_handler,
            socket_path=str(tmp_path / "bridge.sock"),
        )
        await lifecycle.start()
        try:
            _write_entry(tmp_path, str(tmp_path / "bridge.sock"))
            srv = _server(monkeypatch, tmp_path)
            found = await srv._discover_from_files()
            assert found, "discovery returned nothing"
            client = next(iter(found.values()))
            return await client.bridge_request("ping", {})
        finally:
            await lifecycle.stop()

    response = asyncio.run(_exercise())

    assert response["payload"]["via"] == "uds"


def test_discovery_rejects_a_socket_outside_the_discovery_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The discovery dir is the trust boundary, exactly as loopback is for TCP.

    A hostile or corrupted entry must not be able to point the client at an
    arbitrary socket elsewhere on the filesystem.
    """
    inside = tmp_path / "disc"
    inside.mkdir()
    outside = tmp_path / "elsewhere.sock"

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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The container advertises its own mount point, not the host's.

    The volume is ``$SIMUL_DISCOVERY_DIR:/tmp/simul-mcp``, so the file says
    ``/tmp/simul-mcp/bridge.sock`` while the host sees the same socket at
    ``$SIMUL_DISCOVERY_DIR/bridge.sock``. When the literal path is not inside
    the local discovery dir, the reader must try the basename inside it —
    which stays within the trust boundary by construction.
    """

    async def _exercise() -> Dict[str, Any]:
        lifecycle = BridgeServerLifecycle(
            host="127.0.0.1",
            port=0,
            request_handler=_ping_handler,
            socket_path=str(tmp_path / "bridge.sock"),
        )
        await lifecycle.start()
        try:
            # Advertise the path as a container would see it.
            _write_entry(tmp_path, "/tmp/simul-mcp/bridge.sock")
            srv = _server(monkeypatch, tmp_path)
            found = await srv._discover_from_files()
            assert found, "discovery returned nothing"
            return await next(iter(found.values())).bridge_request("ping", {})
        finally:
            await lifecycle.stop()

    assert asyncio.run(_exercise())["payload"]["via"] == "uds"


# ---------------------------------------------------------------------------
# Two instances sharing one discovery dir must stay distinct (#121 review)
# ---------------------------------------------------------------------------


def _tagged_lifecycle(tmp_path: Path, tag: str) -> BridgeServerLifecycle:
    async def handler(request: Any) -> BridgeResponse:
        return BridgeResponse.success(request.request_id, {"who": tag})

    return BridgeServerLifecycle(
        host="127.0.0.1",
        port=0,
        request_handler=handler,
        socket_path=str(tmp_path / "bridge.sock"),
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


def test_second_bridge_does_not_hijack_a_live_socket(tmp_path: Path) -> None:
    """Both containers ship the same configured name; both must stay reachable.

    Every container is pid 1 in its own namespace, so per-pid names collide
    identically — uniqueness has to come from refusing to steal a live socket
    and binding a generated sibling instead.
    """

    async def _exercise() -> tuple[str, str]:
        a = _tagged_lifecycle(tmp_path, "A")
        b = _tagged_lifecycle(tmp_path, "B")
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


def test_stopping_one_bridge_leaves_the_other_reachable(tmp_path: Path) -> None:
    """Unlink on stop must only remove a socket the stopper actually owns."""

    async def _exercise() -> str:
        a = _tagged_lifecycle(tmp_path, "A")
        b = _tagged_lifecycle(tmp_path, "B")
        await a.start()
        await b.start()
        try:
            await a.stop()
            return await _who(b.actual_socket_path)
        finally:
            await b.stop()

    assert asyncio.run(_exercise()) == "B"


def test_discovery_file_advertises_the_actual_socket(tmp_path: Path) -> None:
    """A generated sibling name must be what gets advertised."""

    async def _exercise() -> tuple[str, str]:
        a = _tagged_lifecycle(tmp_path, "A")
        b = _tagged_lifecycle(tmp_path, "B")
        await a.start()
        await b.start()
        try:
            b.write_discovery_file(str(tmp_path), pid=99)
            written = json.loads((tmp_path / "simul-mcp-99.json").read_text())
            return written["socket_path"], b.actual_socket_path
        finally:
            await a.stop()
            await b.stop()

    advertised, actual = asyncio.run(_exercise())

    assert advertised == actual


def test_two_discovered_instances_resolve_to_distinct_backends(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The end-to-end misrouting from the review: tools for instance A must not
    execute inside instance B."""

    async def _exercise() -> tuple[str, str]:
        a = _tagged_lifecycle(tmp_path, "A")
        b = _tagged_lifecycle(tmp_path, "B")
        await a.start()
        await b.start()
        try:
            for i, lc in enumerate((a, b)):
                (tmp_path / f"simul-mcp-{100 + i}.json").write_text(
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
            srv = _server(monkeypatch, tmp_path)
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
    tmp_path: Path,
) -> None:
    """Diagnostics must point at the endpoint that was dialled.

    A socket-only client reporting "127.0.0.1:8229" — or worse, "None:None" —
    sends the operator to a port that was never touched: the same
    misdirection class #119/#120 were about.
    """
    sock = str(tmp_path / "nothing-here.sock")
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
