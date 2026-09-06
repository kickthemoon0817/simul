"""Regression: a position-only camera update must not need a center of interest.

``set_isaac_camera`` reads ``ViewportCameraState.target_world`` after applying a
position. On a camera Kit has never driven through the viewport there is no
``omni:kit:centerOfInterest``, and that read raises instead of returning None, so
the whole call fails even though the position was applied.

The test execs the real generated script against a stubbed Kit surface, so it
reproduces the failure rather than asserting on the script's text.
"""


from __future__ import annotations

import asyncio
import io
import json
import sys
import types
from contextlib import redirect_stdout
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest


from simul_mcp.adapters.isaac_socket_client import ScriptResult
from simul_mcp.mcp.tools.isaac_tools import IsaacTools

# The error Kit raises when target_world transforms a None center-of-interest.
_BOOST_ARGUMENT_ERROR = (
    "Python argument types in\n"
    "    Matrix4d.Transform(Matrix4d, NoneType)\n"
    "did not match C++ signature"
)


# ---------------------------------------------------------------------------
# #90 — position-only camera update on a camera with no center of interest
# ---------------------------------------------------------------------------


class _CameraStateWithoutTarget:
    """Kit's ViewportCameraState for a camera that has no center of interest."""

    def __init__(self, viewport: Any = None) -> None:
        self.applied_positions: List[Any] = []

    def set_position_world(self, value: Any, _absolute: bool) -> None:
        self.applied_positions.append(value)

    @property
    def position_world(self):
        return (1.0, 2.0, 3.0)

    @property
    def target_world(self):
        raise TypeError(_BOOST_ARGUMENT_ERROR)


def _install_kit_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the minimum omni/pxr surface the camera script imports."""

    class _Vec3d:
        def __init__(self, *components: float) -> None:
            self.components = components

    pxr = types.ModuleType("pxr")
    pxr.Gf = types.SimpleNamespace(Vec3d=_Vec3d)  # type: ignore[attr-defined]
    pxr.Usd = types.SimpleNamespace()  # type: ignore[attr-defined]
    pxr.UsdGeom = types.SimpleNamespace()  # type: ignore[attr-defined]

    viewport = types.SimpleNamespace(camera_path="/World/Camera")
    utility = types.ModuleType("omni.kit.viewport.utility")
    utility.get_active_viewport = lambda: viewport  # type: ignore[attr-defined]
    camera_state = types.ModuleType("omni.kit.viewport.utility.camera_state")
    camera_state.ViewportCameraState = _CameraStateWithoutTarget  # type: ignore[attr-defined]

    omni = types.ModuleType("omni")
    omni_kit = types.ModuleType("omni.kit")
    omni_viewport = types.ModuleType("omni.kit.viewport")
    omni.kit = omni_kit  # type: ignore[attr-defined]
    omni_kit.viewport = omni_viewport  # type: ignore[attr-defined]
    omni_viewport.utility = utility  # type: ignore[attr-defined]
    utility.camera_state = camera_state  # type: ignore[attr-defined]

    for name, module in {
        "pxr": pxr,
        "omni": omni,
        "omni.kit": omni_kit,
        "omni.kit.viewport": omni_viewport,
        "omni.kit.viewport.utility": utility,
        "omni.kit.viewport.utility.camera_state": camera_state,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def _capture_camera_script(**kwargs: Any) -> str:
    """Render the script ``set_isaac_camera`` would send to Isaac Sim."""
    captured: List[str] = []

    def _record(code: str) -> ScriptResult:
        captured.append(code)
        return ScriptResult(success=True, output=json.dumps({"ok": True}))

    client = MagicMock()
    client.address = "127.0.0.1:8226"
    client.timeout_seconds = 30.0
    client.bridge_enabled = False
    client.fallback_to_vscode = True
    client.execute = AsyncMock(side_effect=_record)
    client.execute_vscode_only = AsyncMock(side_effect=_record)
    client.execute_bridge_script_only = AsyncMock(side_effect=_record)

    asyncio.run(IsaacTools(client).set_isaac_camera(**kwargs))
    assert captured, "set_isaac_camera generated no script"
    return captured[0]


def test_position_only_update_survives_missing_center_of_interest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A position-only call must succeed when target_world is unreadable."""
    _install_kit_stubs(monkeypatch)
    script = _capture_camera_script(position=[1.0, 2.0, 3.0])

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exec(compile(script, "<camera_script>", "exec"), {"__name__": "__main__"})

    payload: Dict[str, Any] = json.loads(stdout.getvalue().strip())
    assert payload["position"] == [1.0, 2.0, 3.0]
    assert payload["target"] is None
    assert "error" not in payload


