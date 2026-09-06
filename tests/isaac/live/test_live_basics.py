"""Live Isaac Sim tier: the launch-free basics against a running instance.

These tests talk to whatever Isaac Sim answers on the configured socket (the
same probe ``simul-mcp isaac ping`` runs) and skip when nothing does, so the
default ``pytest tests/`` run stays clean on a machine without the engine.
They never start or stop Isaac Sim themselves.

To run only this tier::

    pytest tests/isaac/live -m isaac

Each test cleans up what it creates. The bridge toggle test disables and
re-enables the ``khemoo.simul.mcp`` extension through the stock Python
socket, since ``disable_isaac_extension`` rightly refuses to switch off the
transport it speaks through.
"""

from __future__ import annotations

import asyncio
import base64
import socket
import time
import uuid
from typing import Any, Dict, Tuple

import pytest

from simul_mcp.adapters import IsaacRuntimeAdapter
from simul_mcp.config import get_settings
from simul_mcp.mcp.tools.isaac_tools import IsaacTools

pytestmark = pytest.mark.isaac

BRIDGE_EXTENSION = "khemoo.simul.mcp"


def _run(coro: Any) -> Dict[str, Any]:
    return asyncio.run(coro)


def _port_open(address: str, timeout: float = 1.0) -> bool:
    host, _, port = address.rpartition(":")
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_for_port(address: str, expected_open: bool, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _port_open(address) is expected_open:
            return True
        time.sleep(0.5)
    return _port_open(address) is expected_open


@pytest.fixture(scope="module")
def live() -> Tuple[IsaacRuntimeAdapter, IsaacTools]:
    """The adapter and tools for the configured instance; skips when it does not answer."""
    settings = get_settings()
    adapter = IsaacRuntimeAdapter(settings)
    if not _run(adapter.client.ping()):
        pytest.skip(
            f"Isaac Sim is not reachable at {adapter.client.address}; "
            "start it with `simul-mcp isaac launch` to run the live tier"
        )
    return adapter, IsaacTools(adapter.client, settings)


def test_ping_answers(live: Tuple[IsaacRuntimeAdapter, IsaacTools]) -> None:
    adapter, _tools = live
    assert _run(adapter.client.ping()) is True


def test_stage_info_reports_the_stage_conventions(live: Tuple[IsaacRuntimeAdapter, IsaacTools]) -> None:
    _adapter, tools = live
    payload = _run(tools.get_isaac_stage_info())
    assert payload.get("success") is not False, payload
    assert payload.get("up_axis") in ("Y", "Z"), payload
    assert payload.get("meters_per_unit"), payload


def test_create_and_delete_prim_round_trip(live: Tuple[IsaacRuntimeAdapter, IsaacTools]) -> None:
    _adapter, tools = live
    prim_path = f"/World/SimulLiveTest_{uuid.uuid4().hex[:8]}"

    created = _run(tools.create_isaac_prim(prim_path=prim_path, prim_type="Cube"))
    assert created.get("success") is not False, created
    try:
        detail = _run(tools.get_isaac_prim_detail(prim_path=prim_path, aspects=["info"]))
        assert detail.get("success") is not False, detail
        assert detail["info"].get("type") == "Cube", detail
    finally:
        deleted = _run(tools.delete_isaac_prim(prim_path=prim_path))
    assert deleted.get("success") is not False, deleted

    gone = _run(tools.get_isaac_prim_detail(prim_path=prim_path, aspects=["info"]))
    assert gone.get("success") is False or not gone.get("info", {}).get("exists", True), gone


def test_capture_viewport_returns_an_image(live: Tuple[IsaacRuntimeAdapter, IsaacTools]) -> None:
    _adapter, tools = live
    payload = _run(tools.capture_isaac_viewport(width=320, height=180, inline=True))
    assert payload.get("success") is not False, payload
    image = payload.get("image_base64")
    if image:
        assert base64.b64decode(image)[:8] == b"\x89PNG\r\n\x1a\n"
    else:
        assert payload.get("file_path") or payload.get("path"), payload


def test_read_aovs_returns_per_aov_statistics(live: Tuple[IsaacRuntimeAdapter, IsaacTools]) -> None:
    _adapter, tools = live
    payload = _run(tools.read_aovs(aov_names=["HdrColor"], resolution=[64, 64], num_frames=2))
    assert payload.get("success") is not False, payload
    stats = payload.get("aovs") or payload.get("results") or payload
    assert "HdrColor" in stats, payload


def test_bridge_extension_toggle_closes_and_reopens_its_port(
    live: Tuple[IsaacRuntimeAdapter, IsaacTools],
) -> None:
    adapter, _tools = live
    client = adapter.client
    bridge_address = client.bridge_address
    if not client.bridge_enabled or not bridge_address or not _port_open(bridge_address):
        pytest.skip("the bridge extension is not up; nothing to toggle")
    if not client.vscode_address or not _port_open(client.vscode_address):
        pytest.skip("the stock Python socket is not up; the bridge cannot be re-enabled without it")

    def _toggle(enabled: bool) -> None:
        result = _run(
            client.execute_vscode_only(
                "import omni.kit.app\n"
                "manager = omni.kit.app.get_app().get_extension_manager()\n"
                f"manager.set_extension_enabled_immediate({BRIDGE_EXTENSION!r}, {enabled})\n"
            )
        )
        assert result.success, result

    try:
        _toggle(False)
        assert _wait_for_port(bridge_address, expected_open=False, timeout=10.0), "bridge port stayed open"
    finally:
        _toggle(True)
        reopened = _wait_for_port(bridge_address, expected_open=True, timeout=20.0)
    assert reopened, "bridge port did not come back after re-enabling the extension"
    assert _run(client.ping()) is True
