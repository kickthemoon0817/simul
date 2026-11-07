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
    assert reply["error"] == "TypeError: Failed to add rigid body to 'Cube': enum \"BOGUS\" not found"


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
                "open_blend_file",
                "play_animation",
                "save_blend_file",
            }, spec.name


def test_an_unclassified_operation_is_recorded_rather_than_lost(bridge_module: Any) -> None:
    assert bridge_module.undo_step_name("new_operation", {}) == "simul: new_operation"
