"""The stage lifecycle tools through the real MCP stack: save, export, reload, unload, list, isolation.

The session-level behaviour is pinned in
``tests/headless_usd/test_stage_lifecycle.py``. These tests check what only
the server adds: the sandbox on the files a save or export writes, the
envelope's error types, and that each MCP session (an initialize-handshake
connection here) sees only the stages it loaded.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List

import pytest
from fastmcp import Client

pytest.importorskip("pxr", reason="pxr library not available")

from pxr import Sdf, Usd, UsdGeom  # noqa: E402

from simul.config import Settings  # noqa: E402
from simul.mcp.server import SimulMCPServer  # noqa: E402


def _server(sandbox: Path) -> SimulMCPServer:
    base = Settings()
    settings = base.model_copy(
        update={
            "security": base.security.model_copy(
                update={"allowed_paths": [str(sandbox)], "rate_limiting_enabled": False}
            )
        }
    )
    return SimulMCPServer(settings, backends={"usd"})


def _payload(result: Any) -> Dict[str, Any]:
    return json.loads(result.content[-1].text)


async def _call(client: Client, name: str, **arguments: Any) -> Dict[str, Any]:
    return _payload(await client.call_tool(name, arguments, raise_on_error=False))


def _run(server: SimulMCPServer, steps) -> List[Dict[str, Any]]:
    """Run ``steps(client)`` on one connection and return what it collects."""

    async def _main() -> List[Dict[str, Any]]:
        async with Client(server.mcp) as client:
            return await steps(client)

    return asyncio.run(_main())


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    root = tmp_path / "sandbox"
    root.mkdir()
    stage = Usd.Stage.CreateNew(str(root / "robot.usda"))
    robot = UsdGeom.Xform.Define(stage, "/Robot")
    stage.SetDefaultPrim(robot.GetPrim())
    UsdGeom.Cube.Define(stage, "/Robot/link")
    stage.GetRootLayer().Save()
    return root


def test_save_writes_the_file_and_reports_it(sandbox: Path) -> None:
    server = _server(sandbox)
    robot = sandbox / "robot.usda"

    async def steps(client: Client) -> List[Dict[str, Any]]:
        stage_id = (await _call(client, "load_usd_file", file_path=str(robot)))["stage_id"]
        await _call(client, "create_prim", stage_id=stage_id, prim_path="/Robot/Saved", prim_type="Xform")
        listed = await _call(client, "list_usd_stages")
        saved = await _call(client, "save_usd_stage", stage_id=stage_id)
        again = await _call(client, "save_usd_stage", stage_id=stage_id)
        return [listed, saved, again]

    listed, saved, again = _run(server, steps)

    assert listed["count"] == 1 and listed["stages"][0]["has_unsaved_changes"] is True
    assert saved["success"] is True and saved["saved"] is True
    assert saved["files_written"][0]["path"] == str(robot.resolve())
    assert Sdf.Layer.OpenAsAnonymous(str(robot)).GetPrimAtPath("/Robot/Saved")
    assert again["saved"] is False and again["files_written"] == []


def test_export_is_sandboxed_and_refuses_overwrites(sandbox: Path, tmp_path: Path) -> None:
    server = _server(sandbox)
    outside = tmp_path / "outside.usda"
    inside = sandbox / "copy.usdc"

    async def steps(client: Client) -> List[Dict[str, Any]]:
        stage_id = (await _call(client, "load_usd_file", file_path=str(sandbox / "robot.usda")))["stage_id"]
        denied = await _call(client, "export_usd_stage", stage_id=stage_id, file_path=str(outside))
        written = await _call(client, "export_usd_stage", stage_id=stage_id, file_path=str(inside), flatten=True)
        refused = await _call(client, "export_usd_stage", stage_id=stage_id, file_path=str(inside))
        return [denied, written, refused]

    denied, written, refused = _run(server, steps)

    assert denied["error_type"] == "SandboxError"
    assert denied["details"]["access"] == "write"
    assert not outside.exists()
    assert written["success"] is True and written["format"] == "usdc" and written["flatten"] is True
    assert written["files_written"][0]["bytes"] == inside.stat().st_size
    assert refused["error_type"] == "RefusedOperation" and refused["details"]["override"] == "overwrite"


def test_reload_and_unload_need_discard_for_unsaved_edits(sandbox: Path) -> None:
    server = _server(sandbox)

    async def steps(client: Client) -> List[Dict[str, Any]]:
        stage_id = (await _call(client, "load_usd_file", file_path=str(sandbox / "robot.usda")))["stage_id"]
        await _call(client, "delete_prim", stage_id=stage_id, prim_path="/Robot/link")
        return [
            await _call(client, "reload_usd_stage", stage_id=stage_id),
            await _call(client, "reload_usd_stage", stage_id=stage_id, discard_unsaved=True),
            await _call(client, "get_bounding_box", stage_id=stage_id, prim_path="/Robot/link"),
            await _call(client, "delete_prim", stage_id=stage_id, prim_path="/Robot/link"),
            await _call(client, "unload_usd_stage", stage_id=stage_id),
            await _call(client, "unload_usd_stage", stage_id=stage_id, discard_unsaved=True),
            await _call(client, "list_usd_stages"),
        ]

    refused, reloaded, bbox, _, unload_refused, unloaded, listed = _run(server, steps)

    assert refused["error_type"] == "RefusedOperation" and refused["details"]["override"] == "discard_unsaved"
    assert reloaded["success"] is True and reloaded["discarded_unsaved"] is True
    assert bbox["success"] is True  # the deleted prim is back
    assert unload_refused["error_type"] == "RefusedOperation"
    assert unloaded["success"] is True and unloaded["discarded_unsaved"] is True
    assert listed["count"] == 0


def test_each_mcp_session_sees_only_its_own_stages(sandbox: Path) -> None:
    """Over initialize-handshake connections (one session each), as over HTTP."""
    server = _server(sandbox)
    robot = str(sandbox / "robot.usda")

    async def _main() -> List[Dict[str, Any]]:
        async with Client(server.mcp, mode="legacy") as first, Client(server.mcp, mode="legacy") as second:
            mine = (await _call(first, "load_usd_file", file_path=robot))["stage_id"]
            await _call(first, "delete_prim", stage_id=mine, prim_path="/Robot/link")
            peek = await _call(second, "get_bounding_box", stage_id=mine)
            meddle = await _call(
                second, "create_prim", stage_id=mine, prim_path="/Robot/FromSecond", prim_type="Xform"
            )
            theirs = await _call(second, "load_usd_file", file_path=robot)
            listed = await _call(second, "list_usd_stages")
            return [{"mine": mine}, peek, meddle, theirs, listed]

    ids, peek, meddle, theirs, listed = asyncio.run(_main())

    assert peek["error_type"] == "NotFoundError"
    assert meddle["error_type"] == "NotFoundError"
    assert theirs["stage_id"] != ids["mine"]
    assert theirs["total_prims"] == 2  # the first session's unsaved delete is not visible
    assert [stage["stage_id"] for stage in listed["stages"]] == [theirs["stage_id"]]


def test_an_ended_session_releases_its_stages(
    sandbox: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Six handshake sessions each load and edit, then end: nothing is left behind."""
    server = _server(sandbox)
    robot = str(sandbox / "robot.usda")
    loaded: List[str] = []

    async def _main() -> None:
        for index in range(6):
            async with Client(server.mcp, mode="legacy") as client:
                stage_id = (await _call(client, "load_usd_file", file_path=robot))["stage_id"]
                loaded.append(stage_id)
                if index % 2 == 0:
                    await _call(
                        client, "create_prim", stage_id=stage_id, prim_path="/Robot/Extra", prim_type="Xform"
                    )

    with caplog.at_level(logging.WARNING):
        asyncio.run(_main())

    session = server.headless_adapter._session
    assert len(set(loaded)) == 6
    assert session._active_stages == {} and session._stage_records == {}
    assert server._watched_sessions == set()
    warnings = [record.getMessage() for record in caplog.records if "unsaved edits" in record.getMessage()]
    assert len(warnings) == 3
    assert all(loaded[index] in " ".join(warnings) for index in (0, 2, 4))


def test_stage_tools_run_in_the_foreground_even_with_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A background task would run without the request, so it could not name the owning session."""
    from simul.mcp import server as server_module
    from tests.fakes import FakeFastMCP

    tasks = ModuleType("fastmcp_tasks")
    tasks.TasksExtension = type("TasksExtension", (), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastmcp_tasks", tasks)
    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    instance = server_module.SimulMCPServer(Settings(), backends={"usd"})

    usd_tools = set(instance.tools_by_backend()["USD / Headless"])
    assert {"load_usd_file", "summarize_scene"} <= usd_tools
    assert [
        tool.name for tool in instance.mcp.tools if tool.name in usd_tools and tool.kwargs.get("task")
    ] == []


def test_a_load_waits_for_a_running_read(sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A load may reread shared layers in place, so it never runs beside a read."""
    import threading

    from simul.adapters.headless_usd import HeadlessUSDSession

    (sandbox / "other.usda").write_text((sandbox / "robot.usda").read_text())
    server = _server(sandbox)
    armed, entered, release = threading.Event(), threading.Event(), threading.Event()
    original = HeadlessUSDSession.get_stage

    def gated(session: HeadlessUSDSession, stage_id: str) -> Any:
        if armed.is_set():
            armed.clear()
            entered.set()
            release.wait(5)
        return original(session, stage_id)

    monkeypatch.setattr(HeadlessUSDSession, "get_stage", gated)

    async def _main() -> Dict[str, Any]:
        async with Client(server.mcp) as client:
            stage_id = (await _call(client, "load_usd_file", file_path=str(sandbox / "robot.usda")))["stage_id"]
            armed.set()
            read = asyncio.create_task(_call(client, "get_bounding_box", stage_id=stage_id))
            while not entered.is_set():
                await asyncio.sleep(0.01)
            load = asyncio.create_task(_call(client, "load_usd_file", file_path=str(sandbox / "other.usda")))
            await asyncio.sleep(0.3)
            load_finished_beside_the_read = load.done()
            release.set()
            return {"read": await read, "load": await load, "overlapped": load_finished_beside_the_read}

    outcome = asyncio.run(_main())

    assert outcome["overlapped"] is False
    assert outcome["read"]["success"] is True
    assert outcome["load"]["success"] is True


def _legacy_pair(server: SimulMCPServer, steps) -> Any:
    """Run ``steps(first, second)`` over two initialize-handshake sessions."""

    async def _main() -> Any:
        async with Client(server.mcp, mode="legacy") as first, Client(server.mcp, mode="legacy") as second:
            return await steps(first, second)

    return asyncio.run(_main())


def test_concurrent_saves_of_one_file_from_two_sessions_lose_no_write(sandbox: Path) -> None:
    """The audit's race: both saves reported saved and one session's prim was missing."""
    server = _server(sandbox)
    robot = sandbox / "robot.usda"

    async def steps(first: Client, second: Client) -> List[Dict[str, Any]]:
        mine = (await _call(first, "load_usd_file", file_path=str(robot)))["stage_id"]
        theirs = (await _call(second, "load_usd_file", file_path=str(robot)))["stage_id"]
        rounds = []
        for index in range(5):
            await _call(first, "reload_usd_stage", stage_id=mine, discard_unsaved=True)
            await _call(second, "reload_usd_stage", stage_id=theirs, discard_unsaved=True)
            await _call(first, "create_prim", stage_id=mine, prim_path=f"/Robot/A{index}", prim_type="Xform")
            await _call(second, "create_prim", stage_id=theirs, prim_path=f"/Robot/B{index}", prim_type="Xform")
            a, b = await asyncio.gather(
                _call(first, "save_usd_stage", stage_id=mine), _call(second, "save_usd_stage", stage_id=theirs)
            )
            layer = Sdf.Layer.OpenAsAnonymous(str(robot))
            rounds.append(
                {
                    "a": a,
                    "b": b,
                    "has_a": bool(layer.GetPrimAtPath(f"/Robot/A{index}")),
                    "has_b": bool(layer.GetPrimAtPath(f"/Robot/B{index}")),
                }
            )
        return rounds

    for outcome in _legacy_pair(server, steps):
        saved = [key for key in ("a", "b") if outcome[key].get("saved") is True]
        refused = [outcome[key] for key in ("a", "b") if outcome[key].get("success") is False]
        assert len(saved) == 1 and len(refused) == 1, outcome
        assert refused[0]["error_type"] == "RefusedOperation"
        assert refused[0]["details"]["override"] == "overwrite_external_changes"
        assert outcome[f"has_{saved[0]}"] is True


def test_concurrent_exports_to_one_new_path_create_it_once(sandbox: Path) -> None:
    server = _server(sandbox)
    robot = sandbox / "robot.usda"
    (sandbox / "out").mkdir()

    async def steps(first: Client, second: Client) -> List[List[Dict[str, Any]]]:
        mine = (await _call(first, "load_usd_file", file_path=str(robot)))["stage_id"]
        theirs = (await _call(second, "load_usd_file", file_path=str(robot)))["stage_id"]
        rounds = []
        for index in range(5):
            destination = str(sandbox / "out" / f"copy_{index}.usdc")
            rounds.append(
                list(
                    await asyncio.gather(
                        _call(first, "export_usd_stage", stage_id=mine, file_path=destination),
                        _call(second, "export_usd_stage", stage_id=theirs, file_path=destination),
                    )
                )
            )
        return rounds

    for pair in _legacy_pair(server, steps):
        assert sorted(bool(result.get("success")) for result in pair) == [False, True], pair
        refused = next(result for result in pair if not result.get("success"))
        assert refused["error_type"] == "RefusedOperation" and refused["details"]["override"] == "overwrite"
    assert sorted(path.name for path in (sandbox / "out").iterdir()) == [f"copy_{i}.usdc" for i in range(5)]


def test_a_load_that_reuses_a_stage_runs_beside_reads(sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Handing back a clean, current stage reads nothing from disk, so it does not wait for the backend."""
    import threading

    from simul.adapters.headless_usd import HeadlessUSDSession

    (sandbox / "other.usda").write_text((sandbox / "robot.usda").read_text())
    server = _server(sandbox)
    armed, entered, release = threading.Event(), threading.Event(), threading.Event()
    original = HeadlessUSDSession.get_stage

    def gated(session: HeadlessUSDSession, stage_id: str) -> Any:
        if armed.is_set():
            armed.clear()
            entered.set()
            release.wait(5)
        return original(session, stage_id)

    monkeypatch.setattr(HeadlessUSDSession, "get_stage", gated)

    async def _main() -> Dict[str, Any]:
        async with Client(server.mcp) as client:
            robot = (await _call(client, "load_usd_file", file_path=str(sandbox / "robot.usda")))["stage_id"]
            other = (await _call(client, "load_usd_file", file_path=str(sandbox / "other.usda")))["stage_id"]
            armed.set()
            read = asyncio.create_task(_call(client, "get_bounding_box", stage_id=other))
            while not entered.is_set():
                await asyncio.sleep(0.01)
            again = await asyncio.wait_for(
                _call(client, "load_usd_file", file_path=str(sandbox / "robot.usda")), 5
            )
            read_still_running = not read.done()
            release.set()
            await read
            layer = Sdf.Layer.OpenAsAnonymous(str(sandbox / "robot.usda"))
            Sdf.CreatePrimInLayer(layer, "/Robot/Outside").specifier = Sdf.SpecifierDef
            layer.Export(str(sandbox / "robot.usda"))
            fresh = await _call(client, "load_usd_file", file_path=str(sandbox / "robot.usda"))
            return {"robot": robot, "again": again, "overlapped": read_still_running, "fresh": fresh}

    outcome = asyncio.run(_main())

    assert outcome["overlapped"] is True
    assert outcome["again"]["stage_id"] == outcome["robot"]
    assert outcome["again"]["total_prims"] == 2
    assert outcome["fresh"]["stage_id"] != outcome["robot"]
    assert outcome["fresh"]["total_prims"] == 3


def test_a_session_that_ends_during_a_load_leaves_no_stage(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stage used to register after the session-end cleanup had run, with no owner left to reach it."""
    import threading

    from simul.adapters.headless_usd import HeadlessUSDSession
    from simul.usd import stage_files

    server = _server(sandbox)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    compose = stage_files.compose_stage
    load = HeadlessUSDSession.load_stage_or_raise

    def gated_compose(root: Any) -> Any:
        entered.set()
        release.wait(5)
        return compose(root)

    def tracked_load(*args: Any, **kwargs: Any) -> Any:
        try:
            return load(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(stage_files, "compose_stage", gated_compose)
    monkeypatch.setattr(HeadlessUSDSession, "load_stage_or_raise", tracked_load)

    async def _main() -> None:
        client = Client(server.mcp, mode="legacy")
        await client.__aenter__()
        call = asyncio.create_task(
            client.call_tool("load_usd_file", {"file_path": str(sandbox / "robot.usda")}, raise_on_error=False)
        )
        while not entered.is_set():
            await asyncio.sleep(0.01)
        call.cancel()
        try:
            await asyncio.wait_for(client.__aexit__(None, None, None), 5)
        except Exception:
            pass
        while server._watched_sessions:
            await asyncio.sleep(0.01)
        release.set()
        while not finished.is_set():
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.1)

    asyncio.run(_main())

    session = server.headless_adapter._session
    assert session._stage_records == {} and session._active_stages == {}


def test_a_relative_export_path_resolves_against_relative_path_base(tmp_path: Path) -> None:
    """With the sandbox off as with it on, as load_usd_file resolves its path."""
    base = tmp_path / "base"
    base.mkdir()
    (base / "robot.usda").write_text('#usda 1.0\ndef Xform "Robot"\n{\n}\n')
    defaults = Settings()
    settings = defaults.model_copy(
        update={
            "security": defaults.security.model_copy(
                update={
                    "sandbox_enabled": False,
                    "relative_path_base": str(base),
                    "rate_limiting_enabled": False,
                }
            )
        }
    )
    server = SimulMCPServer(settings, backends={"usd"})

    async def steps(client: Client) -> List[Dict[str, Any]]:
        loaded = await _call(client, "load_usd_file", file_path="robot.usda")
        exported = await _call(client, "export_usd_stage", stage_id=loaded["stage_id"], file_path="copy.usda")
        return [loaded, exported]

    loaded, exported = _run(server, steps)

    assert loaded["file_path"] == str((base / "robot.usda").resolve())
    assert exported["file_path"] == str((base / "copy.usda").resolve())
    assert (base / "copy.usda").is_file()


def test_reload_reports_a_sublayer_that_no_longer_parses(sandbox: Path) -> None:
    (sandbox / "sub.usda").write_text('#usda 1.0\ndef Xform "World"\n{\n    def Xform "FromSub"\n    {\n    }\n}\n')
    (sandbox / "scene.usda").write_text("#usda 1.0\n(\n    subLayers = [@./sub.usda@]\n)\n")
    server = _server(sandbox)

    async def steps(client: Client) -> List[Dict[str, Any]]:
        stage_id = (await _call(client, "load_usd_file", file_path=str(sandbox / "scene.usda")))["stage_id"]
        (sandbox / "sub.usda").write_text('#usda 1.0\ndef Xform "World"\n{\n    broken\n')
        return [
            await _call(client, "reload_usd_stage", stage_id=stage_id),
            await _call(client, "list_usd_stages"),
        ]

    reloaded, listed = _run(server, steps)

    sub = str((sandbox / "sub.usda").resolve())
    assert reloaded["success"] is True
    assert sub not in reloaded["reloaded_files"]
    assert [entry["path"] for entry in reloaded["failed_files"]] == [sub]
    assert "cannot parse" in reloaded["message"]
    assert listed["stages"][0]["changed_on_disk"] == [sub]


def test_a_self_referencing_file_loads_with_a_warning(sandbox: Path) -> None:
    (sandbox / "self.usda").write_text(
        '#usda 1.0\ndef Xform "W"\n{\n    def Cube "Proto"\n    {\n        double size = 1\n    }\n'
        '    def "Inst" (\n        references = @./self.usda@</W/Proto>\n    )\n    {\n    }\n}\n'
    )
    server = _server(sandbox)

    async def steps(client: Client) -> List[Dict[str, Any]]:
        return [
            await _call(client, "load_usd_file", file_path=str(sandbox / "self.usda")),
            await _call(client, "load_usd_file", file_path=str(sandbox / "robot.usda")),
        ]

    looped, plain = _run(server, steps)

    assert [warning["code"] for warning in looped["warnings"]] == ["SelfReference"]
    assert looped["warnings"][0]["location"]["layer"] == str((sandbox / "self.usda").resolve())
    assert plain["warnings"] == []
