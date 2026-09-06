"""Regressions for per-call work that scales with the whole stage or log.

Three tools did far more work than their arguments implied, all of it on Kit's
main thread, so the cost shows up as a freeze rather than a slow response:

* ``get_isaac_logs`` read the entire newest Kit log to return the last N lines.
  The log tree on a working machine reached 9.3 GB, largest single file 4.34 GB.
* ``get_isaac_prim_info`` called ``attr.Get()`` on every attribute, decompressing
  mesh point/normal/index arrays out of the crate layer, only for the serializer
  to replace anything over 16 elements with ``"[N elements]"``.
* ``list_isaac_prims`` computed depth and then used ``continue``, which skips
  *emitting* a prim but still descends its whole subtree — so ``max_depth=1`` on
  a 50k-prim stage still visited 50k prims.

These tests exec the real generated scripts against stubs, so they measure what
the script does rather than how it is spelled.
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest


from simul_mcp.adapters.isaac_socket_client import ScriptResult
from simul_mcp.config import Settings
from simul_mcp.mcp.tools.isaac_tools import IsaacTools


def _capture_script(method: str, **kwargs: Any) -> str:
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
    client.bridge_request = AsyncMock(return_value=None)

    asyncio.run(getattr(IsaacTools(client, settings=Settings()), method)(**kwargs))
    assert captured, f"{method} generated no script"
    return captured[0]


def _run_script(
    script: str, modules: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    stdout = io.StringIO()
    namespace: Dict[str, Any] = {"__name__": "__main__"}
    with pytest.MonkeyPatch.context() as mp:
        for name, module in (modules or {}).items():
            mp.setitem(sys.modules, name, module)
        with redirect_stdout(stdout):
            exec(compile(script, "<generated>", "exec"), namespace)
    return json.loads(stdout.getvalue().strip())


# ---------------------------------------------------------------------------
# #92 — reading the last N lines must not read the whole log
# ---------------------------------------------------------------------------


def _write_kit_log(directory: Path, lines: int) -> Path:
    log_path = directory / "kit.log"
    with open(log_path, "w") as handle:
        for i in range(lines):
            handle.write(
                f"2026-08-10T12:00:00Z [Error] [omni.test.module] failure number {i}\n"
            )
    return log_path


def test_log_read_scans_only_the_tail_of_a_large_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log_path = _write_kit_log(tmp_path, 60_000)
    size = log_path.stat().st_size
    monkeypatch.setattr("os.path.expanduser", lambda _p: str(tmp_path))

    payload = _run_script(_capture_script("get_isaac_logs", level="error", last_n=50))

    assert payload["returned"] == 50
    assert payload["scanned_bytes"] < size, "whole log was read to return 50 entries"
    assert payload["truncated_scan"] is True
    # The newest lines are the ones that matter.
    assert payload["entries"][-1]["message"].endswith("failure number 59999")


def test_log_read_covers_a_small_file_completely(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A log that fits inside the window must not be reported as truncated."""
    log_path = _write_kit_log(tmp_path, 50)
    monkeypatch.setattr("os.path.expanduser", lambda _p: str(tmp_path))

    payload = _run_script(_capture_script("get_isaac_logs", level="error", last_n=500))

    assert payload["truncated_scan"] is False
    assert payload["scanned_bytes"] == log_path.stat().st_size
    assert payload["returned"] == 50


# ---------------------------------------------------------------------------
# #94 — prim info must not materialise arrays it will not return
# ---------------------------------------------------------------------------


class _RecordingAttribute:
    """A USD attribute that notices when its value is pulled."""

    def __init__(self, name: str, *, is_array: bool, value: Any) -> None:
        self._name = name
        self._value = value
        self._type_name = types.SimpleNamespace(isArray=is_array)
        self.get_calls = 0

    def GetName(self) -> str:
        return self._name

    def GetTypeName(self) -> Any:
        return self._type_name

    def Get(self) -> Any:
        self.get_calls += 1
        return self._value


def _usd_modules(stage: Any) -> Dict[str, Any]:
    omni = types.ModuleType("omni")
    omni_usd = types.ModuleType("omni.usd")
    omni_usd.get_context = lambda: types.SimpleNamespace(get_stage=lambda: stage)
    omni.usd = omni_usd
    pxr = types.ModuleType("pxr")
    pxr.Usd = types.SimpleNamespace(
        Prim=object,
        PrimRange=lambda root: iter(()),
        TimeCode=types.SimpleNamespace(Default=lambda: 0.0),
    )
    pxr.UsdGeom = types.SimpleNamespace(
        Xformable=lambda _p: MagicMock(),
        Imageable=lambda _p: MagicMock(),
    )
    pxr.Gf = types.SimpleNamespace(
        Matrix4d=type("Matrix4d", (), {}),
        Matrix4f=type("Matrix4f", (), {}),
        Matrix3d=type("Matrix3d", (), {}),
        Matrix3f=type("Matrix3f", (), {}),
        Vec3f=lambda *a: (0.0, 0.0, 0.0),
        Vec3d=lambda *a: (0.0, 0.0, 0.0),
        Rotation=lambda *a, **k: MagicMock(),
        Transform=lambda *a, **k: MagicMock(),
    )
    return {"omni": omni, "omni.usd": omni_usd, "pxr": pxr}


def test_prim_info_does_not_pull_array_attribute_values() -> None:
    points = _RecordingAttribute("points", is_array=True, value=list(range(200_000)))
    visibility = _RecordingAttribute("visibility", is_array=False, value="inherited")

    prim = MagicMock()
    prim.IsValid.return_value = True
    prim.GetAttributes.return_value = [points, visibility]
    # Not Xformable: the transform block is irrelevant to what this test claims,
    # and stubbing it would only add fixture surface.
    prim.IsA.return_value = False
    prim.GetRelationships.return_value = []
    stage = MagicMock()
    stage.GetPrimAtPath.return_value = prim

    script = _capture_script("get_isaac_prim_info", prim_path="/World/Mesh")
    try:
        _run_script(script, _usd_modules(stage))
    except TypeError:
        # The script keeps going after the attribute loop into transform and
        # material sections whose MagicMock values are not JSON-serialisable.
        # Stubbing all of USD would add fixture surface without testing more:
        # the claim here is only about which attributes get pulled, and the
        # visibility assertion below proves the loop ran to completion.
        pass

    assert points.get_calls == 0, "array attribute decompressed only to be discarded"
    assert visibility.get_calls == 1, "scalar attributes should still be read"


# ---------------------------------------------------------------------------
# #95 — a depth limit must prune the traversal, not just skip emitting
# ---------------------------------------------------------------------------


class _FakePrim:
    def __init__(self, path: str) -> None:
        self._path = path

    def GetPath(self) -> str:
        return self._path

    def GetName(self) -> str:
        return self._path.rsplit("/", 1)[-1]

    def GetTypeName(self) -> str:
        return "Xform"

    def IsActive(self) -> bool:
        return True

    def GetChildren(self) -> List[Any]:
        return []


class _RecordingPrimRange:
    """Yields a progressively deeper chain and records pruning."""

    def __init__(self, root: Any) -> None:
        self.pruned = 0
        self.visited = 0

    def __iter__(self) -> "_RecordingPrimRange":
        return self

    def __next__(self) -> _FakePrim:
        if self.visited >= 200:
            raise StopIteration
        self.visited += 1
        return _FakePrim("/World" + "/level" * self.visited)

    def PruneChildren(self) -> None:
        self.pruned += 1


def test_depth_limited_listing_prunes_instead_of_skipping() -> None:
    ranges: List[_RecordingPrimRange] = []

    def _make_range(root: Any) -> _RecordingPrimRange:
        created = _RecordingPrimRange(root)
        ranges.append(created)
        return created

    root_prim = MagicMock()
    root_prim.IsValid.return_value = True
    root_prim.GetPath.return_value = "/World"
    stage = MagicMock()
    stage.GetPrimAtPath.return_value = root_prim

    modules = _usd_modules(stage)
    modules["pxr"].Usd = types.SimpleNamespace(
        Prim=object,
        PrimRange=_make_range,
        TimeCode=types.SimpleNamespace(Default=lambda: 0.0),
    )

    script = _capture_script(
        "list_isaac_prims", root_path="/World", max_depth=1, max_results=500
    )
    _run_script(script, modules)

    assert ranges, "script never built a PrimRange"
    assert ranges[0].pruned > 0, "depth limit skipped prims without pruning the subtree"


def test_small_arrays_are_still_returned_in_full() -> None:
    """Skipping every array throws away information the caller needs.

    Before the bulk-array guard, arrays of 16 or fewer elements came back in
    full; only larger ones collapsed to "[N elements]". xformOpOrder is a
    one-element token[] that tells a caller which xform ops exist before it
    calls set_isaac_prim_transform, and primvars:displayColor is a
    one-element color3f[] carrying the object's colour. Both are arrays, and
    neither is bulk.
    """
    op_order = _RecordingAttribute(
        "xformOpOrder", is_array=True, value=["xformOp:translate"]
    )
    points = _RecordingAttribute("points", is_array=True, value=list(range(200_000)))

    prim = MagicMock()
    prim.IsValid.return_value = True
    prim.GetAttributes.return_value = [op_order, points]
    prim.IsA.return_value = False
    prim.GetRelationships.return_value = []
    stage = MagicMock()
    stage.GetPrimAtPath.return_value = prim

    script = _capture_script("get_isaac_prim_info", prim_path="/World/Cube")
    try:
        _run_script(script, _usd_modules(stage))
    except TypeError:
        pass

    assert points.get_calls == 0, "bulk geometry array should still be skipped"
    assert op_order.get_calls == 1, "small non-bulk array was discarded unread"
