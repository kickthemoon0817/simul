"""Live-engine tests for the Unreal adapter.

These mirror the C1–C5 sanity probes documented in
``docs/unreal-e2e-checklist.md``. They run against a real UE editor
reachable at the configured host:port and skip cleanly when no editor
is up — the default ``pytest tests/`` run is clean on a dev box
without UE installed.

To run only this suite::

    pytest tests/ -v -m unreal_live
    # or, by file:
    pytest tests/unreal/test_live.py -v

These tests are intentionally OS-agnostic at the Python level: the
Remote Control API is HTTP, so the same code paths exercise UE on
Windows, Linux, and macOS. Per-OS validation happens by running the
suite on each platform; no test is gated on ``platform.system()``.
"""

from __future__ import annotations

import asyncio
import json as _json

import pytest

from simul.adapters.unreal_connection import discover_instances, probe_instance
from simul.adapters.unreal_runtime import UnrealRuntimeSession
from simul.config import Settings

pytestmark = pytest.mark.unreal_live


def _settings() -> Settings:
    return Settings()


@pytest.fixture(autouse=True)
def _skip_if_unreal_down() -> None:
    """Skip every test in this module if the configured UE port doesn't answer.

    ``probe_instance`` opens and closes its own session, so no session
    outlives the ``asyncio.run`` call — each test owns its own event loop
    and session lifecycle.
    """
    settings = _settings()
    cfg = settings.unreal
    probe = asyncio.run(probe_instance(settings, cfg.host, cfg.port))
    if not probe.get("reachable"):
        pytest.skip(
            f"Unreal Engine not reachable at {cfg.host}:{cfg.port} "
            f"({probe['error']})"
        )


def _run(coro):
    """Run an async coroutine to completion in a fresh event loop."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# C1 — Connectivity & identity
# ---------------------------------------------------------------------------


def test_c1_health_check_reports_connected_with_engine_and_project() -> None:
    async def body() -> None:
        session = UnrealRuntimeSession(_settings())
        try:
            health = await session.health_check()
            assert health.get("connected") is True, health
            engine = health.get("engine_version") or ""
            assert engine.startswith("5."), f"unexpected engine version: {engine!r}"
            assert health.get("project_name"), f"empty project_name: {health!r}"
            assert health.get("is_editor") is True, health
        finally:
            await session.close()

    _run(body())


# ---------------------------------------------------------------------------
# C2 — Multi-instance discovery
# ---------------------------------------------------------------------------


def test_c2_discovery_finds_running_editor_with_matching_project() -> None:
    """Discovery identifies the same project that health_check reports — proves
    discovery is real, not just "anything answered HTTP". The scan and the
    single-endpoint probe agree on the editor's identity."""

    async def body() -> None:
        settings = _settings()
        cfg = settings.unreal
        scanned = await discover_instances(settings)
        (active,) = [item for item in scanned["instances"] if item["active"]]
        assert active["port"] == cfg.port and active["reachable"] is True, active
        assert active["instance_id"] and active["pid"] and active["viewports"], active
        assert active["error"] is None, active
        (single,) = (await discover_instances(settings, scan=False))["instances"]
        assert {k: v for k, v in single.items() if k != "latency_ms"} == {
            k: v for k, v in active.items() if k != "latency_ms"
        }
        session = UnrealRuntimeSession(settings)
        try:
            health = await session.health_check()
            assert active["project_name"] == health.get("project_name"), (
                active,
                health,
            )
            assert active["engine_version"] == health.get("engine_version")
        finally:
            await session.close()

    _run(body())


# ---------------------------------------------------------------------------
# C3 — execute_python contract
# ---------------------------------------------------------------------------


def test_c3a_execute_python_propagates_log_output_for_plain_print() -> None:
    """The adapter should surface UE's LogOutput entries verbatim — the
    MCP-tool layer is what enforces JSON; the adapter must not pre-filter."""

    async def body() -> None:
        session = UnrealRuntimeSession(_settings())
        try:
            result = await session._execute_python(
                "print('hello-from-c3a')", mode="ExecuteFile"
            )
            log = result.get("LogOutput", [])
            assert log, f"no LogOutput from print: {result!r}"
            joined = " ".join(entry.get("Output", "") for entry in log)
            assert "hello-from-c3a" in joined, f"missing print output: {log!r}"
        finally:
            await session.close()

    _run(body())


def test_c3b_execute_python_returns_engine_version_via_json_print() -> None:
    async def body() -> None:
        session = UnrealRuntimeSession(_settings())
        try:
            code = (
                "import unreal, json\n"
                "print(json.dumps({'e': unreal.SystemLibrary.get_engine_version()}))"
            )
            result = await session._execute_python(code, mode="ExecuteFile")
            log = result.get("LogOutput", [])
            payload = next(
                (
                    entry["Output"]
                    for entry in log
                    if entry.get("Output", "").lstrip().startswith("{")
                ),
                None,
            )
            assert payload, f"no JSON line in LogOutput: {log!r}"
            parsed = _json.loads(payload.strip())
            assert parsed.get("e", "").startswith("5."), parsed
        finally:
            await session.close()

    _run(body())


# ---------------------------------------------------------------------------
# C4 — Minimal scene read
# ---------------------------------------------------------------------------


def test_c4_scene_actor_read_returns_count_and_names() -> None:
    async def body() -> None:
        session = UnrealRuntimeSession(_settings())
        try:
            code = (
                "import unreal, json\n"
                "actors = unreal.EditorLevelLibrary.get_all_level_actors()\n"
                "print(json.dumps({"
                "  'count': len(actors),"
                "  'names': [a.get_name() for a in actors[:5]]"
                "}))"
            )
            result = await session._execute_python(code, mode="ExecuteFile")
            log = result.get("LogOutput", [])
            payload = next(
                (
                    entry["Output"]
                    for entry in log
                    if entry.get("Output", "").lstrip().startswith("{")
                ),
                None,
            )
            assert payload, log
            parsed = _json.loads(payload.strip())
            assert parsed.get("count", -1) >= 0
            assert isinstance(parsed.get("names"), list)
            assert len(parsed["names"]) <= 5
        finally:
            await session.close()

    _run(body())


# ---------------------------------------------------------------------------
# C5 — Viewport capture (the killer test for the WorldContextObject /
#       MacEditor / marker-prefix bug)
# ---------------------------------------------------------------------------


def test_c5_capture_viewport_returns_nonempty_image() -> None:
    async def body() -> None:
        session = UnrealRuntimeSession(_settings())
        try:
            result = await session.capture_viewport(
                resolution_x=256, resolution_y=256, format="png", inline=True
            )
            assert result.get("resolution_x") == 256
            assert result.get("resolution_y") == 256
            assert result.get("format") == "png"
            image = result.get("image_base64") or ""
            # 256x256 PNG is at least a few KB; an empty / single-byte
            # string would mean HighResShot didn't fire or the screenshot
            # directory was missed (the original C5 failure mode).
            assert len(image) > 1000, (
                f"image_base64 implausibly short ({len(image)} chars) — "
                "HighResShot likely no-opped or the screenshot dir was missed"
            )
        finally:
            await session.close()

    _run(body())
