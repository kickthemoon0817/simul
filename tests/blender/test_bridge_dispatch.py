"""Add-on side of the attached bridge, loaded against a stand-in bpy."""

from __future__ import annotations

import importlib.util
import json
import socket
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, Optional
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
    # Each recorded undo step, with the context override it was pushed in.
    undo_pushes: list = []

    def undo_push(message: str) -> set:
        undo_pushes.append((message, overrides[-1]))
        return {"FINISHED"}

    bpy = ModuleType("bpy")
    bpy.context = SimpleNamespace(  # type: ignore[attr-defined]
        mode="OBJECT",
        temp_override=lambda **kw: overrides.append(kw) or nullcontext(),
        window_manager=SimpleNamespace(windows=[window]),
    )
    bpy.ops = SimpleNamespace(ed=SimpleNamespace(undo_push=undo_push))  # type: ignore[attr-defined]
    handlers = ModuleType("bpy.app.handlers")
    handlers.persistent = lambda fn: fn  # type: ignore[attr-defined]
    handlers.load_post = []  # type: ignore[attr-defined]
    app = ModuleType("bpy.app")
    app.handlers = handlers  # type: ignore[attr-defined]
    app.timers = SimpleNamespace(  # type: ignore[attr-defined]
        register=Mock(), unregister=Mock(), is_registered=Mock(return_value=False)
    )
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
    module.test_undo_pushes = undo_pushes
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


def _framed(bridge: Any, request: Dict[str, Any]) -> bytes:
    """``request`` as the client sends it: protocol and token first, then the rest."""
    first = {"protocol": request.get("protocol", PROTOCOL_VERSION), "token": request.get("token", bridge.token)}
    return BridgeWire.encode({**first, **request})


def test_stale_client_protocol_is_named_with_the_fix(
    bridge_module: Any, tmp_path: Path
) -> None:
    bridge = _bridge(bridge_module, tmp_path)
    reply = bridge.dispatch(_request(bridge, protocol=1, method="hello"))
    assert reply["success"] is False
    assert "protocol mismatch" in reply["error"]
    assert "install-bridge" in reply["error"]


@pytest.mark.parametrize("protocol", [1, 2, 3, 4])
def test_stale_addon_endpoint_is_refused_before_connecting(protocol: int) -> None:
    """Add-ons built before agent_id (1), capture method/script output cap (2),
    focus framing and world-space camera tools (3), or open_blend_file's
    discard_unsaved (4).

    An add-on that predates a keyword the server now sends would reject it
    with a bare TypeError; one built before 4 would read focus_on_object's
    distance_factor as a multiple of the bounding-box diagonal, and one built
    before 5 records agent edits as no undo step, so open_blend_file would
    discard them unasked. The version check names the fix instead.
    """
    from simul.blender_bridge.protocol import BridgeProtocolError
    from simul.mcp.error_types import classify_exception

    endpoint = {"protocol": protocol, "host": "127.0.0.1", "port": 1, "token": "t"}
    with pytest.raises(BridgeProtocolError, match="protocol mismatch.*install-bridge") as caught:
        BridgeWire.request(endpoint, {"request_id": "r"}, 0.1)
    # A ValueError still, but the bridge's failure, not the caller's input.
    assert isinstance(caught.value, ValueError)
    assert classify_exception(caught.value) == "BridgeProtocolError"


def test_unencodable_result_still_gets_a_reply(
    bridge_module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A NaN (or oversized) result used to close the socket with no reply."""
    bridge = _bridge(bridge_module, tmp_path)
    monkeypatch.setattr(
        bridge,
        "dispatch",
        lambda request, on_start=None: {"request_id": "r1", "success": True, "result": {"x": float("nan")}},
    )
    server, client = socket.socketpair()
    try:
        server.setblocking(False)
        bridge.listener = Mock(accept=Mock(side_effect=BlockingIOError))
        pending = bridge_module.PendingConnection(server, time.monotonic() + 5)
        bridge.connections.append(pending)
        client.sendall(_framed(bridge, {"request_id": "r1"}))
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


def test_failed_operation_reply_keeps_the_class_and_the_noted_context(
    bridge_module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bpy failure noted by the session reaches the server with its context."""

    class Session:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def get_frame(self) -> Dict[str, Any]:
            error = TypeError('enum "BOGUS" not found')
            error.add_note("Failed to add rigid body to 'Cube'")
            raise error

    monkeypatch.setattr(bridge_module, "BlenderRuntimeSession", Session)
    bridge = _bridge(bridge_module, tmp_path)
    reply = bridge.dispatch(
        _request(
            bridge,
            method="get_frame",
            target={"document_id": bridge.document_id, "window_id": "4", "scene_id": "5"},
            path_policy={"enabled": False, "allowed_paths": [], "project_root": str(tmp_path)},
        )
    )
    assert reply["success"] is False
    assert reply["error_type"] == "TypeError"
    # R2-BLA-03: the class is the error_type, not a prefix of the message.
    assert reply["error"] == "Failed to add rigid body to 'Cube': enum \"BOGUS\" not found"


# ---------------------------------------------------------------------------
# Agent edits are undo steps (B1-02); nothing an operation raises ends Blender (B1-10)
# ---------------------------------------------------------------------------


def _operation_request(
    bridge: Any, tmp_path: Path, method: str, **extra: Any
) -> Dict[str, Any]:
    return _request(
        bridge,
        method=method,
        target={"document_id": bridge.document_id, "window_id": "4", "scene_id": "5"},
        path_policy={"enabled": False, "allowed_paths": [], "project_root": str(tmp_path)},
        **extra,
    )


class _EditingSession:
    """Stands in for BlenderRuntimeSession; ``calls`` lists the edits that ran."""

    calls: list = []

    def __init__(self, **kwargs: Any) -> None:
        pass

    def create_object(self, object_type: str, name: Optional[str] = None) -> Dict[str, Any]:
        self.calls.append("create_object")
        return {"name": name or object_type.title()}

    def delete_object(self, object_name: str) -> Dict[str, Any]:
        raise ValueError(f"Object not found: {object_name}")

    def get_frame(self) -> Dict[str, Any]:
        return {"frame": 1}

    def capture_viewport(self, agent_id: str = "agent") -> Dict[str, Any]:
        return {"image_base64": "abc", "format": "jpeg"}

    def control_ui(self, agent_control: str, target: Optional[str] = None) -> Dict[str, Any]:
        return {"success": True, "agent_control": agent_control}

    def save_blend_file(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        return {"file_path": file_path}

    def execute_script(
        self, script: str, timeout: Optional[float] = None, max_output_chars: int = 10
    ) -> Dict[str, Any]:
        """Run ``script`` with the bridge in reach; it may set ``result``."""
        namespace: Dict[str, Any] = {"bridge": self.bridge}  # type: ignore[attr-defined]
        exec(script, namespace)  # noqa: S102
        result: Dict[str, Any] = namespace.get("result", {"output": None})
        return result


@pytest.fixture
def editing(bridge_module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A bridge whose operations run on ``_EditingSession``."""
    monkeypatch.setattr(bridge_module, "BlenderRuntimeSession", _EditingSession)
    bridge = _bridge(bridge_module, tmp_path)
    monkeypatch.setattr(_EditingSession, "bridge", bridge, raising=False)
    monkeypatch.setattr(_EditingSession, "calls", [])
    return bridge


def _script(bridge: Any, tmp_path: Path, script: str) -> Dict[str, Any]:
    reply: Dict[str, Any] = bridge.dispatch(
        _operation_request(
            bridge,
            tmp_path,
            "execute_script",
            kwargs={"script": script},
            allow_script_execution=True,
        )
    )
    return reply


def test_a_completed_edit_is_one_undo_step_named_after_its_tool(
    bridge_module: Any, editing: Any, tmp_path: Path
) -> None:
    """B1-02: agent edits recorded no undo step and never marked the file modified.

    One Ctrl+Z then took back every agent edit at once, and quitting Blender
    asked nothing. ed.undo_push records the step and marks the file modified;
    its poll needs a window, which a timer's context lacks.
    """
    reply = editing.dispatch(
        _operation_request(editing, tmp_path, "create_object", kwargs={"object_type": "CUBE"})
    )

    assert reply["success"] is True, reply
    assert bridge_module.test_undo_pushes == [
        ("simul: create_blender_object", {"window": bridge_module.test_window})
    ]


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("get_frame", {}),
        ("capture_viewport", {"agent_id": "viewer"}),
        ("save_blend_file", {"file_path": "/tmp/x.blend"}),
        ("control_ui", {"agent_control": "inspect"}),
        ("control_ui", {"agent_control": "set_tool", "target": "move"}),
    ],
)
def test_reads_saves_and_editor_changes_record_no_undo_step(
    bridge_module: Any,
    editing: Any,
    tmp_path: Path,
    method: str,
    kwargs: Dict[str, Any],
) -> None:
    """A step marks the file modified, which a read or a save must not do."""
    reply = editing.dispatch(_operation_request(editing, tmp_path, method, kwargs=kwargs))

    assert reply["success"] is True, reply
    assert bridge_module.test_undo_pushes == []


@pytest.mark.parametrize("action", ["select_object", "set_property"])
def test_ui_actions_that_change_the_scene_are_undo_steps(
    bridge_module: Any, editing: Any, tmp_path: Path, action: str
) -> None:
    reply = editing.dispatch(
        _operation_request(
            editing,
            tmp_path,
            "control_ui",
            kwargs={"agent_control": action, "target": "Cube"},
        )
    )

    assert reply["success"] is True, reply
    assert [name for name, _ in bridge_module.test_undo_pushes] == [
        f"simul: control_blender_ui ({action})"
    ]


def test_a_refused_edit_records_no_undo_step(
    bridge_module: Any, editing: Any, tmp_path: Path
) -> None:
    reply = editing.dispatch(
        _operation_request(editing, tmp_path, "delete_object", kwargs={"object_name": "Gone"})
    )

    assert reply["success"] is False
    assert bridge_module.test_undo_pushes == []


def test_a_script_is_an_undo_step_even_when_it_reports_an_error(
    bridge_module: Any, editing: Any, tmp_path: Path
) -> None:
    """A script may have changed data before it raised."""
    reply = _script(editing, tmp_path, "result = {'error': 'ValueError: late'}")

    assert reply["success"] is True, reply
    assert reply["result"]["error"] == "ValueError: late"
    assert [name for name, _ in bridge_module.test_undo_pushes] == [
        "simul: execute_blender_script"
    ]


def test_a_script_that_loads_a_file_is_a_step_in_the_new_files_history(
    bridge_module: Any, editing: Any, tmp_path: Path
) -> None:
    """What the script did after the load must still be undoable and count as unsaved.

    Blender marks the loaded file unmodified once its event loop handles the
    load's notifier, after the timer returns (seen live on 5.0.1: a step
    recorded at once left is_dirty false). So the step waits for the next
    poll, in a window of the new file.
    """
    reply = _script(
        editing,
        tmp_path,
        "import bpy\n"
        "from types import SimpleNamespace\n"
        "bridge._on_load(None)\n"
        "bpy.context.window_manager.windows[:] = [SimpleNamespace(as_pointer=lambda: 7)]",
    )

    assert reply["success"] is True, reply
    assert bridge_module.test_undo_pushes == []
    editing.listener = Mock(accept=Mock(side_effect=BlockingIOError))
    editing.poll()
    editing.poll()
    [(name, override)] = bridge_module.test_undo_pushes
    assert name == "simul: execute_blender_script"
    assert override["window"].as_pointer() == 7


def test_the_undo_step_uses_another_window_once_the_attached_one_closed(
    bridge_module: Any, editing: Any, tmp_path: Path
) -> None:
    """The undo history is global, so any open window gives the operator its context."""
    reply = _script(
        editing,
        tmp_path,
        "import bpy\n"
        "from types import SimpleNamespace\n"
        "bpy.context.window_manager.windows[:] = [SimpleNamespace(as_pointer=lambda: 9)]",
    )

    assert reply["success"] is True, reply
    [(name, override)] = bridge_module.test_undo_pushes
    assert name == "simul: execute_blender_script"
    assert override["window"].as_pointer() == 9


def test_an_unrecorded_undo_step_does_not_fail_the_completed_edit(
    bridge_module: Any, editing: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The edit ran; reporting a failure would invite a duplicate retry."""

    def refuse(message: str) -> None:
        raise RuntimeError("Operator bpy.ops.ed.undo_push.poll() failed")

    monkeypatch.setattr(bridge_module.bpy.ops.ed, "undo_push", refuse)
    reply = editing.dispatch(
        _operation_request(editing, tmp_path, "create_object", kwargs={"object_type": "CUBE"})
    )

    assert reply["success"] is True, reply
    assert _EditingSession.calls == ["create_object"]


@pytest.mark.parametrize(
    ("raised", "error_type"),
    [(SystemExit(0), "SystemExit"), (KeyboardInterrupt(), "KeyboardInterrupt")],
)
def test_an_exit_raised_by_an_operation_is_its_error_reply(
    bridge_module: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raised: BaseException,
    error_type: str,
) -> None:
    """B1-10: raised past the bridge, Blender's timer prints SystemExit with
    PyErr_Print, which exits the process, and unregisters itself on any other."""

    class Session:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def get_frame(self) -> Dict[str, Any]:
            raise raised

    monkeypatch.setattr(bridge_module, "BlenderRuntimeSession", Session)
    bridge = _bridge(bridge_module, tmp_path)
    reply = bridge.dispatch(_operation_request(bridge, tmp_path, "get_frame"))

    assert reply["success"] is False
    assert reply["error_type"] == error_type


def test_every_operation_is_classified_for_undo(bridge_module: Any) -> None:
    """A new operation must state whether it changes the document.

    Each undo step is named after the MCP tool that calls its operation, and
    no tool that declares itself read-only records one.
    """
    from simul.adapters.blender_runtime import BlenderRuntimeSession
    from simul.mcp.registration._reg_blender import BLENDER_TOOLS

    operations = {
        name
        for name in dir(BlenderRuntimeSession)
        if not name.startswith("_")
        and name != "cleanup"
        and callable(getattr(BlenderRuntimeSession, name))
    }
    recorded = set(bridge_module.UNDO_STEPS)
    unrecorded = set(bridge_module.NO_UNDO_STEP)
    assert not recorded & unrecorded
    assert recorded | unrecorded == operations
    for spec in BLENDER_TOOLS:
        if spec.operation is None:
            continue
        if spec.hints.read_only:
            assert spec.operation in unrecorded, spec.name
        elif spec.operation in recorded:
            assert bridge_module.UNDO_STEPS[spec.operation] == spec.name
        else:
            # Each one's reason is noted where NO_UNDO_STEP lists it.
            assert spec.operation in {
                "control_ui",
                "export_file",
                # Step through frames and fill caches, but restore the frame.
                "get_object_trajectory",
                "get_trajectories",
                "get_physics_state",
                # Also writes only its frame files.
                "capture_viewport_sequence",
                "open_blend_file",
                "play_animation",
                "save_blend_file",
            }, spec.name


def test_an_unclassified_operation_is_recorded_rather_than_lost(bridge_module: Any) -> None:
    assert bridge_module.undo_step_name("new_operation", {}) == "simul: new_operation"


# ---------------------------------------------------------------------------
# Requests wait their turn instead of being dropped (B1-09, M16)
# ---------------------------------------------------------------------------


class _QueueSession:
    """Records the operations it runs, in order; ``during`` runs inside ``slow``."""

    calls: list = []
    during: Any = None

    def __init__(self, **kwargs: Any) -> None:
        pass

    def get_frame(self) -> Dict[str, Any]:
        self.calls.append("get_frame")
        return {"frame": len(self.calls)}

    def get_file_info(self) -> Dict[str, Any]:
        self.calls.append("get_file_info")
        return {"file_path": ""}

    def get_viewport_info(self) -> Dict[str, Any]:
        """The slow one: whatever ``during`` does happens while it runs."""
        self.calls.append("slow")
        hook = _QueueSession.during
        if hook is not None:
            hook()
        return {"done": True}


class _Wire:
    """A bridge on a real loopback listener, and the clients that talk to it."""

    def __init__(self, module: Any, bridge: Any, tmp_path: Path) -> None:
        self.module = module
        self.bridge = bridge
        self.tmp_path = tmp_path
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(socket.SOMAXCONN)
        self.listener.setblocking(False)
        bridge.listener = self.listener
        self.clients: list = []

    def request(self, method: str, request_id: str, **extra: Any) -> bytes:
        """One whole request, framed as the client frames it."""
        return _framed(
            self.bridge,
            _operation_request(self.bridge, self.tmp_path, method, request_id=request_id, **extra),
        )

    def send(self, method: str, request_id: str, **extra: Any) -> socket.socket:
        """Connect and send one whole request; the bridge reads it when polled."""
        client = self.connect()
        client.sendall(self.request(method, request_id, **extra))
        # Loopback delivery can trail send() a little (seen on macOS).
        time.sleep(0.05)
        return client

    def connect(self) -> socket.socket:
        client = socket.create_connection(self.listener.getsockname(), timeout=5)
        self.clients.append(client)
        return client

    def poll(self, times: int = 1) -> None:
        for _ in range(times):
            assert self.bridge.poll() in (0.01, 0.05)

    def poll_until(self, condition: Any, attempts: int = 200) -> bool:
        """Poll until ``condition()``; loopback delivery is not instantaneous."""
        for _ in range(attempts):
            self.poll()
            if condition():
                return True
            time.sleep(0.005)
        return False

    @staticmethod
    def lines(client: socket.socket, count: int = 1) -> list:
        """Read ``count`` newline-terminated messages from a client."""
        buffer = b""
        while buffer.count(b"\n") < count:
            chunk = client.recv(65536)
            assert chunk, f"connection closed after {buffer!r}"
            buffer += chunk
        return [json.loads(line) for line in buffer.split(b"\n")[:count]]

    def close(self) -> None:
        for client in self.clients:
            client.close()
        self.bridge.connections.clear()
        self.listener.close()


@pytest.fixture
def wire(bridge_module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(bridge_module, "BlenderRuntimeSession", _QueueSession)
    monkeypatch.setattr(_QueueSession, "calls", [])
    monkeypatch.setattr(_QueueSession, "during", None)
    connected = _Wire(bridge_module, _bridge(bridge_module, tmp_path), tmp_path)
    yield connected
    connected.close()


class _Clock:
    """Stands in for the bridge's ``time`` module, with a monotonic clock tests move."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    @staticmethod
    def time() -> float:
        return time.time()


def test_requests_that_arrive_together_all_get_answers_in_order(wire: Any) -> None:
    """B1-09: of six calls sent beside a slow script, five failed with Errno 54.

    The bridge held 8 connections and dropped the rest unread; now it holds
    them all, and runs one operation per poll, oldest first.
    """
    clients = [wire.send("get_frame", f"r{index}") for index in range(12)]
    assert wire.poll_until(lambda: not wire.bridge.connections and len(_QueueSession.calls) == 12)

    for index, client in enumerate(clients):
        [reply] = wire.lines(client)
        assert reply["request_id"] == f"r{index}"
        assert reply["success"] is True, reply
    assert _QueueSession.calls == ["get_frame"] * 12


def test_one_operation_runs_per_poll_so_blender_handles_events_between(
    wire: Any,
) -> None:
    first, second = wire.send("get_frame", "a"), wire.send("get_frame", "b")
    wire.poll()
    assert _QueueSession.calls == ["get_frame"]
    wire.poll()
    assert _QueueSession.calls == ["get_frame", "get_frame"]
    assert wire.lines(first)[0]["result"] == {"frame": 1}
    assert wire.lines(second)[0]["result"] == {"frame": 2}


def test_a_refused_request_does_not_hold_up_the_next(wire: Any) -> None:
    refused = wire.send("get_frame", "bad", token="wrong")
    accepted = wire.send("get_frame", "good")
    wire.poll()
    assert wire.lines(refused)[0]["error_type"] == "AuthenticationError"
    assert wire.lines(accepted)[0]["success"] is True


def test_a_request_behind_a_long_operation_is_still_read(
    wire: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B1-09: each connection had 5 s from accept, then was closed unread.

    Calls sent beside a 7 s script were accepted with it and dropped once it
    finished. A minute spent in another operation does not count against a
    waiting request, or against one still arriving: the bridge could not read
    then.
    """
    clock = _Clock()
    monkeypatch.setattr(wire.module, "time", clock)
    _QueueSession.during = lambda: setattr(clock, "now", clock.now + 60)
    slow = wire.send("get_viewport_info", "slow")
    waiting = wire.send("get_file_info", "waiting")
    arriving = wire.connect()
    request = wire.request("get_frame", "late")
    arriving.sendall(request[:120])
    time.sleep(0.05)
    wire.poll()
    assert wire.lines(slow)[0]["result"] == {"done": True}
    arriving.sendall(request[120:])
    assert wire.poll_until(lambda: len(_QueueSession.calls) == 3)

    assert wire.lines(waiting)[0]["result"] == {"file_path": ""}
    assert wire.lines(arriving)[0]["result"] == {"frame": 3}
    assert _QueueSession.calls == ["slow", "get_file_info", "get_frame"]


def test_a_large_request_is_read_while_its_bytes_keep_coming(
    wire: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M16: a 26.9 MB mesh got Broken pipe; reading had 5 s and 64 KiB per poll.

    A request is dropped only after IDLE_SECONDS without a byte, however long
    the whole takes to arrive (up to REQUEST_SECONDS).
    """
    clock = _Clock()
    monkeypatch.setattr(wire.module, "time", clock)
    client = wire.connect()
    request = wire.request("get_frame", "big")
    # Polled every 0.2 s, too often to count as Blender being busy, for longer
    # in all than IDLE_SECONDS.
    pieces = [request[i : i + 7] for i in range(0, len(request), 7)]
    assert wire.module.IDLE_SECONDS < len(pieces) * 0.2 < wire.module.REQUEST_SECONDS
    for piece in pieces:
        client.sendall(piece)
        time.sleep(0.002)
        wire.poll()
        clock.now += 0.2
    assert wire.poll_until(lambda: _QueueSession.calls == ["get_frame"])

    assert wire.lines(client)[0]["result"] == {"frame": 1}


def _trickle(wire: Any, clock: Any, client: socket.socket, data: bytes, every: float) -> None:
    """Send ``data`` a byte every ``every`` seconds, polling every 0.2 s in between.

    The bridge polls all along, so the waits count against the client, not
    as Blender being busy.
    """
    for byte in data:
        client.sendall(bytes([byte]))
        time.sleep(0.002)
        for _ in range(max(1, round(every / 0.2))):
            wire.poll()
            clock.now += 0.2
        if not wire.bridge.connections or wire.bridge.connections[0].discarding:
            return


def test_an_authenticated_request_that_never_ends_is_refused_at_its_deadline(
    wire: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review: one byte every few seconds renewed the idle timer for ever."""
    clock = _Clock()
    monkeypatch.setattr(wire.module, "time", clock)
    client = wire.connect()
    request = wire.request("get_frame", "slowloris")
    client.sendall(request[: len(wire.bridge._prefix) + 5])
    time.sleep(0.05)
    wire.poll()
    _trickle(wire, clock, client, request[len(wire.bridge._prefix) + 5 : -1], 3.0)
    [reply] = wire.lines(client)

    assert reply["error_type"] == "TimeoutError"
    assert reply["details"] == {"code": "request_timeout", "executed": False}
    assert _QueueSession.calls == []


def test_a_connection_that_does_not_authenticate_in_time_is_closed(
    wire: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _Clock()
    monkeypatch.setattr(wire.module, "time", clock)
    client = wire.connect()
    _trickle(wire, clock, client, wire.bridge._prefix[:-1], 0.2)

    assert wire.bridge.connections == []
    assert client.recv(10) == b""


def test_an_idle_connection_is_dropped(wire: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock()
    monkeypatch.setattr(wire.module, "time", clock)
    client = wire.connect()
    client.sendall(wire.bridge._prefix)
    time.sleep(0.05)
    wire.poll()
    assert wire.bridge.connections and wire.bridge.connections[0].authenticated
    # Polled all along: the client, not Blender, is idle.
    for _ in range(int(wire.module.IDLE_SECONDS / 0.2) + 2):
        clock.now += 0.2
        wire.poll()

    assert wire.bridge.connections == []
    assert client.recv(10) == b""


@pytest.mark.parametrize(
    ("data", "error_type", "code"),
    [
        (b'{"request_id": "r1", "token": "', "BridgeProtocolError", "protocol_mismatch"),
        (b'{"protocol": 3, "token": "', "BridgeProtocolError", "protocol_mismatch"),
        (None, "AuthenticationError", "unauthenticated"),
    ],
    ids=["older-client", "other-protocol", "wrong-token"],
)
def test_an_unauthenticated_request_is_refused_from_its_first_bytes(
    wire: Any, data: Optional[bytes], error_type: str, code: str
) -> None:
    """Review: tokenless 31 MiB lines were read, decoded and held, 21 times their size.

    The protocol and token come first, so the bridge refuses after a few
    dozen bytes and reads the rest only to drop it.
    """
    if data is None:
        data = wire.bridge._prefix[:-10] + b'xxxxxxxxx"'
    client = wire.connect()
    # More than the socket buffers hold, so it is sent while the bridge reads.
    sender = threading.Thread(target=client.sendall, args=(data + b"[]," * 300_000,))
    sender.start()
    try:
        assert wire.poll_until(lambda: bool(wire.bridge.connections) and wire.bridge.connections[0].discarding)
        assert wire.poll_until(lambda: not sender.is_alive())
    finally:
        sender.join(timeout=5)
    [reply] = wire.lines(client)

    assert reply["request_id"] is None
    assert reply["error_type"] == error_type
    assert reply["details"] == {"code": code, "executed": False}
    assert wire.bridge._buffered() == 0
    assert _QueueSession.calls == []


def test_waiting_requests_are_held_undecoded(wire: Any) -> None:
    """Only the request being answered is decoded; the others wait as their bytes."""
    first = wire.send("get_frame", "first")
    second = wire.send("get_frame", "second")
    wire.poll()
    waiting = [p for p in wire.bridge.connections if p.ready]
    assert len(waiting) == 1 and waiting[0].request is None
    assert waiting[0].line.startswith(wire.bridge._prefix)
    wire.poll()
    assert wire.lines(first)[0]["success"] and wire.lines(second)[0]["success"]


def test_request_data_past_the_budget_is_refused_as_busy(
    wire: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review: 64 connections could each hold 32 MiB, decoded, until their turn."""
    monkeypatch.setattr(wire.module, "BUFFER_BUDGET", 4000)
    first = wire.connect()
    first.sendall(wire.request("get_frame", "first")[:-1] + b" " * 3000)
    time.sleep(0.05)
    wire.poll()
    second = wire.connect()
    second.sendall(wire.request("get_frame", "second")[:-1] + b" " * 3000)
    assert wire.poll_until(lambda: any(p.discarding for p in wire.bridge.connections))
    [busy] = wire.lines(second)

    assert busy["request_id"] is None
    assert busy["error_type"] == "InstanceBusy"
    assert busy["details"] == {"code": "bridge_busy", "executed": False}
    # Alone, a request may use the whole budget: the first one completes.
    first.sendall(b"\n")
    assert wire.poll_until(lambda: _QueueSession.calls == ["get_frame"])
    assert wire.lines(first)[0]["success"] is True


def test_a_full_bridge_gives_an_unauthenticated_connection_up_for_a_new_one(
    wire: Any,
) -> None:
    """Review: 64 tokenless connections trickling bytes made a real get_frame fail."""
    idle = [wire.connect() for _ in range(wire.module.MAX_CONNECTIONS)]
    for client in idle:
        client.sendall(b'{"protocol": ')
    time.sleep(0.05)
    wire.poll()
    assert len(wire.bridge.connections) == wire.module.MAX_CONNECTIONS
    client = wire.send("get_frame", "real")
    assert wire.poll_until(lambda: _QueueSession.calls == ["get_frame"])

    assert wire.lines(client)[0]["success"] is True
    assert len(wire.bridge.connections) <= wire.module.MAX_CONNECTIONS
    assert idle[0].recv(10) == b""


def test_a_request_over_the_cap_gets_a_refusal_it_can_read(
    wire: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closing it unread reset the connection under a client still sending."""
    monkeypatch.setattr(wire.module, "MAX_MESSAGE_BYTES", 1000)
    client = wire.connect()
    client.sendall(wire.bridge._prefix)
    for _ in range(2):
        client.sendall(b"x" * 3000)
        time.sleep(0.05)
        wire.poll(3)
    [reply] = wire.lines(client)

    assert reply["request_id"] is None
    assert reply["success"] is False
    assert reply["error_type"] == "PayloadTooLarge"
    assert "not executed" in reply["error"]
    assert reply["details"] == {"code": "request_too_large", "limit_bytes": 1000, "executed": False}
    assert _QueueSession.calls == []
    client.close()
    assert wire.poll_until(lambda: not wire.bridge.connections)


@pytest.mark.parametrize(
    "rest", [b", not json\n", b', "x": ' + b"[" * 100_000 + b"\n"], ids=["garbage", "deeply-nested"]
)
def test_an_unparsable_request_gets_a_refusal_and_the_bridge_keeps_polling(
    wire: Any, rest: bytes
) -> None:
    """A RecursionError from json.loads escaped poll, which in Blender stops its timer."""
    client = wire.connect()
    client.sendall(wire.bridge._prefix + rest)
    time.sleep(0.05)
    wire.poll(2)
    [reply] = wire.lines(client)

    assert reply["error_type"] == "BridgeProtocolError"
    assert reply["details"] == {"code": "invalid_request", "executed": False}
    assert _QueueSession.calls == []


def test_a_failing_annotation_prune_does_not_stop_serving(
    wire: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Blender unregisters a timer that raises; a cosmetic step must not cost requests."""
    monkeypatch.setattr(
        wire.module.cursors, "prune", Mock(side_effect=RuntimeError("overlay broke"))
    )
    client = wire.send("get_frame", "served")
    wire.poll(2)

    assert wire.lines(client)[0]["success"] is True


def test_the_start_is_acknowledged_before_the_operation_runs(wire: Any) -> None:
    """With the start acknowledgement, a client knows whether a request it gave up on ran."""
    seen: list = []
    client = wire.send("get_viewport_info", "acked", ack=True)
    # Waits for the acknowledgement while the operation runs; without one it times out.

    def peek() -> None:
        client.settimeout(2)
        seen.append(client.recv(65536, socket.MSG_PEEK))

    _QueueSession.during = peek
    wire.poll()
    client.settimeout(5)
    ack, reply = wire.lines(client, 2)

    assert ack == {"request_id": "acked", "started": True}
    assert seen == [BridgeWire.encode(ack)]
    assert reply["request_id"] == "acked" and reply["success"] is True


class _PartialSocket:
    """A socket whose send takes only a few bytes; ``sendall`` raises ``failure``."""

    def __init__(self, failure: Optional[BaseException] = None) -> None:
        self.failure = failure
        self.sent = b""
        self.blocking = False

    def recv(self, size: int, flags: int = 0) -> bytes:
        raise BlockingIOError

    def send(self, data: bytes) -> int:
        self.sent += data[:5]
        return 5

    def sendall(self, data: bytes) -> None:
        if self.failure is not None:
            raise self.failure
        self.sent += data

    def settimeout(self, value: float) -> None:
        self.blocking = True

    def setblocking(self, flag: bool) -> None:
        self.blocking = flag


@pytest.mark.parametrize("failure", [None, BrokenPipeError(32, "Broken pipe")])
def test_a_partly_sent_acknowledgement_is_finished_or_the_operation_not_run(
    bridge_module: Any, tmp_path: Path, failure: Optional[BaseException]
) -> None:
    """Review: a partial send of the acknowledgement still ran the operation."""
    bridge = _bridge(bridge_module, tmp_path)
    connection = _PartialSocket(failure)
    pending = bridge_module.PendingConnection(connection, time.monotonic() + 5)
    pending.request = {"request_id": "r", "ack": True}
    ack = BridgeWire.encode({"request_id": "r", "started": True})
    if failure is None:
        bridge._acknowledge_start(pending)
        assert connection.sent == ack
        assert pending.outgoing == b""
    else:
        with pytest.raises(bridge_module.RequestAbandoned):
            bridge._acknowledge_start(pending)
    assert connection.blocking is False


def test_the_discovery_record_advertises_the_start_acknowledgement(
    bridge_module: Any, tmp_path: Path
) -> None:
    """Clients ask for acknowledgements only from an add-on that sends them."""
    from simul.blender_bridge.protocol import START_ACK, BridgeFiles

    bridge = _bridge(bridge_module, tmp_path)
    bridge.start()
    try:
        record = BridgeFiles.read(bridge.discovery_path)
    finally:
        bridge.stop()
    assert record["capabilities"] == [START_ACK]


def test_a_refused_request_is_never_acknowledged(wire: Any) -> None:
    client = wire.send("get_frame", "refused", ack=True, instance_id="other")
    wire.poll()
    [reply] = wire.lines(client)
    assert reply["error_type"] == "ValueError"
    assert reply["request_id"] == "refused"


def test_a_request_whose_client_left_while_it_waited_never_runs(wire: Any) -> None:
    """The client gave up (timed out and closed) while an earlier operation ran."""
    slow = wire.send("get_viewport_info", "slow")
    waiting = wire.send("get_frame", "gone")
    # Closed while the slow one runs; the pause lets the close reach the bridge.
    _QueueSession.during = lambda: (waiting.close(), time.sleep(0.05))
    wire.poll(3)

    assert wire.lines(slow)[0]["success"] is True
    assert _QueueSession.calls == ["slow"]
    assert wire.bridge.connections == []


def test_a_request_whose_deadline_passed_while_it_waited_never_runs(
    wire: Any,
) -> None:
    """Review: the refusal carried no details.executed, so the client could not tell."""
    slow = wire.send("get_viewport_info", "slow")
    late = wire.send("get_frame", "late", deadline=time.time() + 0.05)
    _QueueSession.during = lambda: time.sleep(0.1)
    wire.poll(3)

    assert wire.lines(slow)[0]["success"] is True
    [refused] = wire.lines(late)
    assert refused["error_type"] == "TimeoutError"
    assert "no operation was performed" in refused["error"]
    assert refused["details"] == {"executed": False}
    assert _QueueSession.calls == ["slow"]


def test_stopping_the_bridge_removes_the_playback_handlers(
    bridge_module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """play_blender_animation's handlers are persistent: they outlived the add-on."""
    removed: list = []
    monkeypatch.setattr(
        bridge_module, "stop_watching_playback", lambda: removed.append(True)
    )
    bpy = sys.modules["bpy"]
    bpy.app.timers = SimpleNamespace(is_registered=lambda timer: False)  # type: ignore[attr-defined]
    bpy.app.handlers.load_post = []  # type: ignore[attr-defined]

    _bridge(bridge_module, tmp_path).stop()

    assert removed == [True]


# ---------------------------------------------------------------------------
# A declared error type survives the bridge
# ---------------------------------------------------------------------------


def test_a_missing_object_reaches_the_server_as_not_found(
    bridge_module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The add-on relayed the class name, so a missing object read as ValueError."""
    from simul.adapters.blender_runtime import NotFoundError
    from simul.blender_bridge.protocol import BridgeRemoteError
    from simul.mcp.error_types import exception_fields

    class Session:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def get_frame(self) -> Dict[str, Any]:
            raise NotFoundError("Object not found: Cube", "object_not_found", object_name="Cube")

    monkeypatch.setattr(bridge_module, "BlenderRuntimeSession", Session)
    bridge = _bridge(bridge_module, tmp_path)
    reply = bridge.dispatch(_operation_request(bridge, tmp_path, "get_frame"))

    assert reply["success"] is False
    assert reply["error_type"] == "NotFoundError"
    remote = BridgeRemoteError(reply["error"], reply["error_type"], reply["details"])
    assert exception_fields(remote) == {
        "error_type": "NotFoundError",
        "details": {"code": "object_not_found", "object_name": "Cube"},
    }


def test_a_disabled_script_is_script_execution_disabled_over_the_bridge(
    bridge_module: Any, tmp_path: Path
) -> None:
    bridge = _bridge(bridge_module, tmp_path)
    reply = bridge.dispatch(
        _operation_request(
            bridge,
            tmp_path,
            "execute_script",
            kwargs={"script": "print(1)"},
            allow_script_execution=False,
        )
    )

    assert reply["success"] is False
    assert reply["error_type"] == "ScriptExecutionDisabled"


def test_a_declared_error_type_reaches_the_reply(
    bridge_module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused pattern is a ValidationError in either mode, not its Python class."""
    from simul.utils.regex_guard import RegexRefused

    class Session:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def get_frame(self) -> Dict[str, Any]:
            raise RegexRefused("name_pattern has trouble", "trouble")

    monkeypatch.setattr(bridge_module, "BlenderRuntimeSession", Session)
    bridge = _bridge(bridge_module, tmp_path)
    reply = bridge.dispatch(_operation_request(bridge, tmp_path, "get_frame"))

    assert reply["error_type"] == "ValidationError"
    assert reply["details"] == {"code": "regex_backtracking", "construct": "trouble"}
