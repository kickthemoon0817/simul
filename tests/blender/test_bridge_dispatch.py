"""Add-on side of the attached bridge, loaded against a stand-in bpy."""

from __future__ import annotations

import importlib.util
import json
import socket
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Dict
from unittest.mock import Mock

import pytest

from simul.blender_bridge.protocol import PROTOCOL_VERSION, BridgeWire


@pytest.fixture
def bridge_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    area = SimpleNamespace(
        type="VIEW_3D", width=800, height=600, regions=[SimpleNamespace(type="WINDOW")]
    )
    window = SimpleNamespace(
        as_pointer=lambda: 4,
        scene=SimpleNamespace(as_pointer=lambda: 5, name="Scene"),
        screen=SimpleNamespace(areas=[area]),
        view_layer=SimpleNamespace(name="ViewLayer"),
    )
    overrides: list = []
    bpy = ModuleType("bpy")
    bpy.context = SimpleNamespace(  # type: ignore[attr-defined]
        mode="OBJECT",
        temp_override=lambda **kw: overrides.append(kw) or nullcontext(),
        window_manager=SimpleNamespace(windows=[window]),
    )
    handlers = ModuleType("bpy.app.handlers")
    handlers.persistent = lambda fn: fn  # type: ignore[attr-defined]
    app = ModuleType("bpy.app")
    app.handlers = handlers  # type: ignore[attr-defined]
    bpy.app = app  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "bpy.app", app)
    monkeypatch.setitem(sys.modules, "bpy.app.handlers", handlers)
    observations = Mock()
    monkeypatch.setitem(
        sys.modules,
        "simul.blender_bridge.agent_cursor",
        SimpleNamespace(
            cursors=Mock(), observations=observations, hide_annotations=nullcontext
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "simul.blender_bridge.agent_control",
        SimpleNamespace(reset_ui=Mock()),
    )
    path = Path(__file__).parents[2] / "src/simul/blender_bridge/bridge.py"
    spec = importlib.util.spec_from_file_location(
        "simul.blender_bridge.bridge_under_test", path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve annotations through sys.modules.
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module.test_observations = observations
    module.test_overrides = overrides
    module.test_window = window
    return module


def _bridge(module: Any, tmp_path: Path) -> Any:
    return module.BlenderBridge(discovery_dir=tmp_path)


def _request(bridge: Any, **extra: Any) -> Dict[str, Any]:
    return {
        "request_id": "r1",
        "protocol": PROTOCOL_VERSION,
        "token": bridge.token,
        "instance_id": bridge.instance_id,
        "deadline": time.time() + 30,
        **extra,
    }


def test_stale_client_protocol_is_named_with_the_fix(
    bridge_module: Any, tmp_path: Path
) -> None:
    bridge = _bridge(bridge_module, tmp_path)
    reply = bridge.dispatch(_request(bridge, protocol=1, method="hello"))
    assert reply["success"] is False
    assert "protocol mismatch" in reply["error"]
    assert "install-bridge" in reply["error"]


@pytest.mark.parametrize("protocol", [1, 2])
def test_stale_addon_endpoint_is_refused_before_connecting(protocol: int) -> None:
    """Add-ons built before agent_id (1) or capture method/script output cap (2).

    An add-on that predates a keyword the server now sends would reject it
    with a bare TypeError; the version check names the fix instead.
    """
    endpoint = {"protocol": protocol, "host": "127.0.0.1", "port": 1, "token": "t"}
    with pytest.raises(ValueError, match="protocol mismatch.*install-bridge"):
        BridgeWire.request(endpoint, {"request_id": "r"}, 0.1)


def test_unencodable_result_still_gets_a_reply(
    bridge_module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A NaN (or oversized) result used to close the socket with no reply."""
    bridge = _bridge(bridge_module, tmp_path)
    monkeypatch.setattr(
        bridge,
        "dispatch",
        lambda request: {"request_id": "r1", "success": True, "result": {"x": float("nan")}},
    )
    server, client = socket.socketpair()
    try:
        server.setblocking(False)
        bridge.listener = Mock(accept=Mock(side_effect=BlockingIOError))
        pending = bridge_module.PendingConnection(server, time.monotonic() + 5)
        bridge.connections.append(pending)
        client.sendall(json.dumps({"request_id": "r1"}).encode() + b"\n")
        for _ in range(5):
            bridge.poll()
            if not bridge.connections:
                break
        client.settimeout(1)
        reply = json.loads(client.recv(65536).split(b"\n", 1)[0])
    finally:
        client.close()
        server.close()
    assert reply["request_id"] == "r1"
    assert reply["success"] is False
    assert reply["error_type"] == "ResponseEncodingError"
    assert "may already have run" in reply["error"]


@pytest.mark.parametrize("label", ["   ", "bad\x00label"])
def test_capture_is_not_lost_to_an_undrawable_label(
    bridge_module: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
) -> None:
    captured = []

    class Session:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def capture_viewport(
            self, width: int = 512, agent_id: str = "agent"
        ) -> Dict[str, Any]:
            captured.append(agent_id)
            return {"image_base64": "abc", "format": "jpeg"}

    monkeypatch.setattr(bridge_module, "BlenderRuntimeSession", Session)
    bridge_module.test_observations.show.side_effect = ValueError("undrawable")
    bridge = _bridge(bridge_module, tmp_path)
    reply = bridge.dispatch(
        _request(
            bridge,
            method="capture_viewport",
            kwargs={"width": 64, "agent_id": label},
            target={"document_id": bridge.document_id, "window_id": "4", "scene_id": "5"},
            path_policy={"enabled": False, "allowed_paths": [], "project_root": str(tmp_path)},
        )
    )
    assert reply["success"] is True, reply
    assert reply["result"]["image_base64"] == "abc"
    assert captured == [label]


def test_capture_targets_and_marks_the_largest_3d_view(
    bridge_module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #228: the capture draws the 3D View the user looks at, and the cue marks it.

    Workspaces such as Animation put a small camera view before the main one.
    """
    small = SimpleNamespace(
        type="VIEW_3D", width=300, height=200, regions=[SimpleNamespace(type="WINDOW")]
    )
    main = SimpleNamespace(
        type="VIEW_3D", width=1600, height=900, regions=[SimpleNamespace(type="WINDOW")]
    )
    timeline = SimpleNamespace(type="DOPESHEET_EDITOR", width=1900, height=1000, regions=[])
    window = bridge_module.test_window
    monkeypatch.setattr(window.screen, "areas", [small, timeline, main])

    class Session:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def capture_viewport(self, agent_id: str = "agent", method: str = "auto") -> Dict[str, Any]:
            return {"image_base64": "abc", "format": "jpeg", "capture_method": "viewport"}

    monkeypatch.setattr(bridge_module, "BlenderRuntimeSession", Session)
    bridge = _bridge(bridge_module, tmp_path)
    reply = bridge.dispatch(
        _request(
            bridge,
            method="capture_viewport",
            kwargs={"agent_id": "viewer", "method": "viewport"},
            target={"document_id": bridge.document_id, "window_id": "4", "scene_id": "5"},
            path_policy={"enabled": False, "allowed_paths": [], "project_root": str(tmp_path)},
        )
    )
    assert reply["success"] is True, reply
    assert bridge_module.test_overrides[-1]["area"] is main
    bridge_module.test_observations.show.assert_called_once_with(window, main, "viewer")


def _session_returning(result: Dict[str, Any]) -> Any:
    class Session:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def get_frame(self) -> Dict[str, Any]:
            return dict(result)

    return Session


def test_operations_pin_the_window_but_never_its_scene_or_view_layer(
    bridge_module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #215: scene/view_layer are derived from the window, never pinned.

    A script that loads a file frees them; a pinned stale view layer made the
    next operator in that script crash Blender 5.0.1 (live-verified).
    """
    monkeypatch.setattr(bridge_module, "BlenderRuntimeSession", _session_returning({"frame": 1}))
    bridge = _bridge(bridge_module, tmp_path)
    reply = bridge.dispatch(
        _request(
            bridge,
            method="get_frame",
            target={"document_id": bridge.document_id, "window_id": "4", "scene_id": "5"},
            path_policy={"enabled": False, "allowed_paths": [], "project_root": str(tmp_path)},
        )
    )
    assert reply["success"] is True, reply
    override = bridge_module.test_overrides[-1]
    window = bridge_module.test_window
    assert override["window"] is window
    assert override["area"].type == "VIEW_3D"
    assert override["region"].type == "WINDOW"
    assert not {"scene", "view_layer", "screen"} & set(override)


def test_document_change_reply_names_the_new_document(
    bridge_module: Any, tmp_path: Path
) -> None:
    """Issue #216: the pin mismatch carries the new identity and the MCP recovery tool."""
    bridge = _bridge(bridge_module, tmp_path)
    reply = bridge.dispatch(
        _request(
            bridge,
            method="get_frame",
            target={"document_id": "old", "window_id": "4", "scene_id": "5"},
            path_policy={"enabled": False, "allowed_paths": [], "project_root": str(tmp_path)},
        )
    )
    assert reply["success"] is False
    assert reply["error_type"] == "AttachmentTargetChanged"
    assert reply["details"] == {"document_id": bridge.document_id, "attached_document_id": "old"}
    assert bridge.document_id in reply["error"]
    assert "attach_blender_window" in reply["error"]
