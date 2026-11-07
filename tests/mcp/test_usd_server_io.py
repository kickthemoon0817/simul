"""Headless USD through the real MCP server: off-loop work, envelopes, URLs, paths, rate limits.

- USD-16: pxr work ran inline on the event loop, so one slow call stalled
  every other request.
- USD-22: successful load, validate, get_prim_info and get_mesh_info
  responses had no ``success``.
- USD-24: ``omniverse://`` URLs passed the sandbox, then failed with a
  generic LoadError.
- USD-25: relative paths resolved against the source checkout instead of
  the client's working directory, and a SandboxError echoed only the raw
  path.
- USD-28: the per-tool rate limit refused 8 of 18 calls while one robot was
  inspected prim by prim.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastmcp import Client

pytest.importorskip("pxr", reason="pxr library not available")

from simul.adapters.headless_usd import HeadlessUSDSession  # noqa: E402
from simul.config import Settings  # noqa: E402
from simul.mcp.server import SimulMCPServer  # noqa: E402
from simul.usd import validation as validation_module  # noqa: E402
from tests import usd_fixtures as fx  # noqa: E402


def _server(root: Path, *, security: Dict[str, Any] | None = None, **server: Any) -> SimulMCPServer:
    settings = Settings()
    update: Dict[str, Any] = {
        "security": settings.security.model_copy(
            update={"allowed_paths": [str(root)], "rate_limiting_enabled": False, **(security or {})}
        )
    }
    if server:
        update["server"] = settings.server.model_copy(update=server)
    return SimulMCPServer(settings.model_copy(update=update), backends={"usd"})


async def _call(client: Client, name: str, **arguments: Any) -> Dict[str, Any]:
    result = await client.call_tool(name, arguments, raise_on_error=False)
    return json.loads(result.content[0].text)


def _call_once(server: SimulMCPServer, name: str, **arguments: Any) -> Dict[str, Any]:
    async def run() -> Dict[str, Any]:
        async with Client(server.mcp) as client:
            return await _call(client, name, **arguments)

    return asyncio.run(run())


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = (tmp_path / "sandbox").resolve()
    path.mkdir()
    return path


class StageGate:
    """Hold the next stage-bound call inside its worker thread until released.

    Every stage-bound USD tool fetches its stage through
    ``HeadlessUSDSession.get_stage`` once the scheduler has admitted it, so
    gating that one method holds a real call mid-work whatever the tool does
    with the stage afterwards.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, hold_seconds: float = 5.0) -> None:
        self.armed = threading.Event()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.threads: List[int] = []
        self._hold = hold_seconds
        original = HeadlessUSDSession.get_stage

        def gated(session: HeadlessUSDSession, stage_id: str) -> Any:
            if self.armed.is_set():
                self.armed.clear()
                self.threads.append(threading.get_ident())
                self.entered.set()
                self.release.wait(self._hold)
            return original(session, stage_id)

        monkeypatch.setattr(HeadlessUSDSession, "get_stage", gated)

    async def wait_entered(self) -> None:
        while not self.entered.is_set():
            await asyncio.sleep(0.01)


class TestOffTheEventLoop:
    def test_a_slow_call_does_not_hold_up_validate(
        self, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        robot = str(fx.write_robot(root / "robot.usda"))
        server = _server(root)
        gate = StageGate(monkeypatch, hold_seconds=1.5)

        async def scenario() -> Dict[str, float]:
            async with Client(server.mcp) as client:
                stage_id = (await _call(client, "load_usd_file", file_path=robot))["stage_id"]
                started = time.monotonic()
                gate.armed.set()
                slow = asyncio.create_task(
                    _call(client, "get_prim_info", stage_id=stage_id, prim_path="/Robot/base")
                )
                await gate.wait_entered()
                fast_started = time.monotonic()
                report = await _call(client, "validate_usd_file", file_path=robot)
                fast = time.monotonic() - fast_started
                info = await _call(client, "get_prim_info", stage_id="other", prim_path="/Robot")
                held = await slow
                assert held["success"] is True
                assert report["is_valid"] is True
                assert info["error_type"] == "NotFoundError"
                return {"fast": fast, "slow": time.monotonic() - started}

        timings = asyncio.run(scenario())

        assert timings["fast"] < 0.8, timings
        assert timings["slow"] >= 1.5
        assert gate.threads and gate.threads[0] != threading.get_ident()

    def test_a_busy_stage_fails_fast_with_stage_busy(
        self, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        robot = str(fx.write_robot(root / "robot.usda"))
        server = _server(root, timeout=1)
        gate = StageGate(monkeypatch)

        async def scenario() -> Dict[str, Any]:
            async with Client(server.mcp) as client:
                stage_id = (await _call(client, "load_usd_file", file_path=robot))["stage_id"]
                gate.armed.set()
                stuck = asyncio.create_task(
                    _call(client, "get_prim_info", stage_id=stage_id, prim_path="/Robot/base")
                )
                await gate.wait_entered()
                busy = await _call(client, "get_bounding_box", stage_id=stage_id)
                gate.release.set()
                first = await stuck
                after = await _call(client, "get_bounding_box", stage_id=stage_id)
                return {"busy": busy, "first": first, "after": after}

        outcome = asyncio.run(scenario())

        assert outcome["busy"]["error_type"] == "StageBusy"
        assert outcome["busy"]["details"]["stage_id"]
        assert outcome["first"]["success"] is True
        assert outcome["after"]["success"] is True


# Arguments for calling every USD tool once against tests/data/simple_scene. A new
# tool with a required argument not listed here fails the test below with its
# name, so it has to be added (and so shown to run through the scheduler).
_TOOL_ARGUMENTS: Dict[str, Any] = {
    "prim_path": "/World/Cube",
    "prim_type": "Xform",
    "attributes": {"custom:scheduled": 1.0},
    "search_type": "by_name",
    "query": "Cube",
}
_TOOL_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "create_prim": {"prim_path": "/World/made_by_test"},
    "delete_prim": {"prim_path": "/World/made_by_test"},
    "reload_usd_stage": {"discard_unsaved": True},
    "unload_usd_stage": {"discard_unsaved": True},
}
# Run last, in this order: the delete undoes the create, then the stage is
# reread and closed.
_LAST = {"delete_prim": 1, "reload_usd_stage": 2, "unload_usd_stage": 3}
# Edits run alone; everything else bound to a stage holds that stage's lock.
# A load or reload may reread a shared sublayer or reference in place, which
# every stage composing it sees, so those run alone too.
_WRITES = {"create_prim", "update_prim_attributes", "delete_prim", "load_usd_file", "reload_usd_stage"}
_UNBOUND = {"validate_usd_file", "list_usd_stages"}


def test_every_usd_tool_runs_through_the_scheduler(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tool that skips the scheduler reads stages on the event loop, outside the write lock."""
    from simul.mcp import usd_scheduler

    scene = root / "simple_scene.usda"
    scene.write_text((Path(__file__).resolve().parents[1] / "data" / "simple_scene.usda").read_text())
    runs: List[Dict[str, Any]] = []
    real_run = usd_scheduler.USDCallScheduler.run

    async def recording_run(self: Any, tool_name: str, work: Any, **kwargs: Any) -> Any:
        loop_thread = threading.get_ident()
        threads: List[int] = []

        def tracked() -> Any:
            threads.append(threading.get_ident())
            return work()

        result = await real_run(self, tool_name, tracked, **kwargs)
        runs.append({"tool": tool_name, "off_loop": threads != [loop_thread], **kwargs})
        return result

    monkeypatch.setattr(usd_scheduler.USDCallScheduler, "run", recording_run)
    server = _server(root)
    usd_tools = sorted(name for name, backend in server._tool_backends.items() if backend == "usd")
    order = sorted(usd_tools, key=lambda name: (name != "load_usd_file", _LAST.get(name, 0), name))
    overrides = {**_TOOL_OVERRIDES, "export_usd_stage": {"file_path": str(root / "exported.usda")}}

    async def scenario() -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        async with Client(server.mcp) as client:
            tools = {tool.name: tool for tool in await client.list_tools()}
            stage_id = None
            for name in order:
                schema = tools[name].input_schema
                arguments: Dict[str, Any] = {}
                for param in schema.get("required", []):
                    if param == "file_path":
                        arguments[param] = str(scene)
                    elif param == "stage_id":
                        arguments[param] = stage_id
                    else:
                        value = overrides.get(name, {}).get(param, _TOOL_ARGUMENTS.get(param))
                        assert value is not None, f"add an argument for {name}.{param}"
                        arguments[param] = value
                for param, value in overrides.get(name, {}).items():
                    if param in schema.get("properties", {}):
                        arguments[param] = value
                results[name] = await _call(client, name, **arguments)
                if name == "load_usd_file":
                    stage_id = results[name]["stage_id"]
        return results

    results = asyncio.run(scenario())

    for name, payload in results.items():
        assert payload.get("success") is not False, (name, payload)
    by_tool = {run["tool"]: run for run in runs}
    assert sorted(by_tool) == usd_tools
    assert len(runs) == len(usd_tools), "each call is scheduled exactly once"
    stage_id = results["load_usd_file"]["stage_id"]
    for name, run in by_tool.items():
        assert run["off_loop"], name
        assert run.get("writes", False) is (name in _WRITES), name
        expected_key = None if name in _UNBOUND or name in _WRITES else stage_id
        assert run.get("stage_key") == expected_key, name


def test_successful_usd_responses_carry_success(root: Path) -> None:
    scene = root / "simple_scene.usda"
    scene.write_text((Path(__file__).resolve().parents[1] / "data" / "simple_scene.usda").read_text())
    server = _server(root)

    async def scenario() -> List[Dict[str, Any]]:
        async with Client(server.mcp) as client:
            loaded = await _call(client, "load_usd_file", file_path=str(scene))
            stage_id = loaded["stage_id"]
            return [
                loaded,
                await _call(client, "validate_usd_file", file_path=str(scene)),
                await _call(client, "get_prim_info", stage_id=stage_id, prim_path="/World"),
                await _call(client, "get_mesh_info", stage_id=stage_id, prim_path="/World/Cube"),
            ]

    for payload in asyncio.run(scenario()):
        assert payload["success"] is True, payload


class TestUrls:
    def test_omniverse_url_is_refused_clearly(self, root: Path) -> None:
        url = "omniverse://localhost/Projects/robot.usd"

        payload = _call_once(_server(root), "load_usd_file", file_path=url)

        assert payload["success"] is False
        assert payload["error_type"] == "LoadError"
        assert payload["details"]["reason"] == "UnsupportedURL"
        assert payload["details"]["file_path"] == url
        assert "no asset resolver for omniverse://" in payload["error"]
        assert "security.allowed_paths" in payload["details"]["hint"]

    def test_validate_reports_the_omniverse_url_as_unreadable(self, root: Path) -> None:
        url = "omniverse://localhost/Projects/robot.usd"

        report = _call_once(_server(root), "validate_usd_file", file_path=url)

        assert (report["is_valid"], report["can_read"]) == (False, False)
        assert report["issues"][0]["code"] == "UnsupportedURL"
        assert report["file_path"] == url

    def test_url_with_a_registered_resolver_reaches_pxr(
        self, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With a resolver for the scheme the URL is not refused; pxr opens it verbatim."""
        monkeypatch.setattr(validation_module, "resolver_uri_schemes", lambda: {"omniverse"})
        url = "omniverse://localhost/Projects/robot.usd"

        payload = _call_once(_server(root), "load_usd_file", file_path=url)

        # No resolver really exists here, so pxr itself fails to open it.
        assert payload["details"]["reason"] == "OpenFailed"
        assert payload["details"]["file_path"] == url
        assert "hint" not in payload["details"]

    def test_scheme_outside_the_allowlist_is_a_sandbox_error(self, root: Path) -> None:
        payload = _call_once(_server(root), "load_usd_file", file_path="https://example.com/a.usd")
        assert payload["error_type"] == "SandboxError"


class TestRelativePaths:
    def test_relative_path_resolves_against_the_working_directory(
        self, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fx.write_robot(root / "robot.usda")
        monkeypatch.chdir(root)
        server = _server(root)

        loaded = _call_once(server, "load_usd_file", file_path="robot.usda")

        assert loaded["success"] is True
        assert loaded["file_path"] == str(root / "robot.usda")

    def test_relative_path_base_setting(
        self, root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fx.write_robot(root / "robot.usda")
        monkeypatch.chdir(tmp_path)
        server = _server(root, security={"relative_path_base": str(root)})

        loaded = _call_once(server, "load_usd_file", file_path="robot.usda")

        assert loaded["file_path"] == str(root / "robot.usda")

    def test_sandbox_error_shows_where_a_relative_path_led(
        self, root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        server = _server(root)

        payload = _call_once(server, "validate_usd_file", file_path="elsewhere/robot.usda")

        assert payload["error_type"] == "SandboxError"
        details = payload["details"]
        assert details["file_path"] == "elsewhere/robot.usda"
        assert details["absolute_path"] == str(tmp_path.resolve() / "elsewhere" / "robot.usda")
        assert details["relative_to"] == str(tmp_path.resolve())
        assert "resolved_path" not in details

    def test_sandbox_error_shows_where_a_link_inside_leads(self, root: Path, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        fx.write_robot(outside / "robot.usda")
        (root / "link.usda").symlink_to(outside / "robot.usda")

        payload = _call_once(_server(root), "load_usd_file", file_path=str(root / "link.usda"))

        assert payload["error_type"] == "SandboxError"
        assert payload["details"]["resolved_path"] == str((outside / "robot.usda").resolve())


class TestPerPrimQueriesAndTheRateLimit:
    def _limited(self, root: Path, **security: Any) -> SimulMCPServer:
        return _server(
            root,
            security={"rate_limiting_enabled": True, "burst_size": 10, "requests_per_minute": 60,
                      **security},
        )

    def test_inspecting_a_robot_prim_by_prim_is_not_refused(self, root: Path) -> None:
        robot = str(fx.write_robot(root / "robot.usda"))
        server = self._limited(root)

        async def scenario() -> List[Dict[str, Any]]:
            async with Client(server.mcp) as client:
                stage_id = (await _call(client, "load_usd_file", file_path=robot))["stage_id"]
                return [
                    await _call(client, "get_prim_info", stage_id=stage_id, prim_path="/Robot/base")
                    for _ in range(18)
                ]

        assert all(payload["success"] is True for payload in asyncio.run(scenario()))

    def test_other_usd_tools_keep_their_per_tool_bucket(self, root: Path) -> None:
        robot = str(fx.write_robot(root / "robot.usda"))
        server = self._limited(root)

        async def scenario() -> List[Dict[str, Any]]:
            async with Client(server.mcp) as client:
                return [await _call(client, "validate_usd_file", file_path=robot) for _ in range(11)]

        outcomes = asyncio.run(scenario())
        assert outcomes[-1]["error_type"] == "RateLimitError"
        assert outcomes[-1]["details"]["scope"] == "tool"

    def test_the_agent_ceiling_still_applies(self, root: Path) -> None:
        robot = str(fx.write_robot(root / "robot.usda"))
        server = self._limited(root, global_requests_per_minute=12)

        async def scenario() -> List[Dict[str, Any]]:
            async with Client(server.mcp) as client:
                stage_id = (await _call(client, "load_usd_file", file_path=robot))["stage_id"]
                return [
                    await _call(client, "get_prim_info", stage_id=stage_id, prim_path="/Robot")
                    for _ in range(12)
                ]

        refused = [p for p in asyncio.run(scenario()) if p.get("error_type") == "RateLimitError"]
        assert refused
        assert refused[0]["details"]["scope"] == "agent"
