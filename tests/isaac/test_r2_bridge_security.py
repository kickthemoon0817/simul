"""Round 2: nothing a non-script tool can do turns bridge auth or stage scripts back on.

* M3: ``set_isaac_carb_settings`` refused only keys strictly inside the
  transport namespaces. Writing the namespace itself, an ancestor (``/exts``,
  ``/``) or a dict that replaces the subtree went through, and the bridge
  re-read its auth settings on every request, so the write reopened 8229 and
  the Unix socket to an empty token. Every path a write reaches is now checked,
  and the bridge latches auth: settings can tighten it, never loosen it.
* M4: the script guard only knew OmniGraph code nodes. Isaac Sim's Python
  Scripting Component (OmniScriptingAPI with ``omni:scripting:scripts``, loaded
  by omni.kit.scripting on 5.1 and omni.behavior.scripting on 6.0) runs the
  files a prim lists, with no prompt once a script-free stage was opened
  (5.0 ``script_manager.py`` ``_open_stage`` / ``_pending_sync_task_handler``).
  Imports, references and duplicates carrying one, writes to its attributes,
  and ``/app/scripting`` settings are refused while script execution is off.
* Minors: the python_server token is redacted from ``get_isaac_logs``; an
  all-lowercase ``...pythonscript`` name is a script word; an import with
  unloaded payloads is refused (the guard cannot see inside); the bridge's
  raw-code path refuses a python_server JSON envelope instead of running it
  as a no-op.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

import pytest
from pxr import Sdf, Usd, UsdGeom

from khemoo.simul import extension as extension_module
from khemoo.simul.extension import (
    BRIDGE_AUTH_TOKEN_SETTING,
    BRIDGE_REQUIRE_AUTH_SETTING,
    PYTHON_SERVER_AUTH_TOKEN_SETTING,
    PYTHON_SERVER_REQUIRE_AUTH_SETTING,
    BridgeAuthLatch,
    IsaacMCPServerExtension,
)
from khemoo.simul.lifecycle import (
    AUTH_ERROR_NAME,
    ENVELOPE_ERROR_NAME,
    BridgeServerLifecycle,
    is_python_server_envelope,
)
from khemoo.simul.protocol import BridgeRequest, BridgeResponse
from simul.adapters.isaac_socket_client import IsaacSocketClient
from simul.mcp.tools.isaac._shared import (
    PROTECTED_CARB_NAMESPACES,
    carb_path_touches,
    carb_write_paths,
)
from simul.mcp.tools.isaac.graph import (
    is_script_attribute,
    is_script_capable_node_type,
    is_script_extension,
    script_guard_source,
)
from tests.isaac.test_materials_import_save_graph import _call, _recording_tools, _settings, _world_stage

# ---------------------------------------------------------------------------
# M3: the carb guard covers every path a write reaches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "settings",
    [
        {"/exts/khemoo.simul/require_auth": False},
        {"/exts/khemoo.simul": {"require_auth": False, "auth_token": ""}},
        {"/exts/khemoo.simul": 0},
        {"/exts/isaacsim.code_editor.python_server": {"require_auth": False}},
        {"/exts/isaacsim.code_editor.python_server": 0},
        {"/exts/isaacsim.code_editor.vscode": {"port": 1}},
        {"/exts": {"khemoo.simul": {"require_auth": False}}},
        {"/exts": 0},
        {"/": {}},
        {"": 0},
        {"exts/khemoo.simul//require_auth": False},
        {"/exts/khemoo.simul/": {"require_auth": False}},
        {"//exts///khemoo.simul": 0},
        {"/rtx": {"../exts/khemoo.simul/require_auth": False}},
        {"/exts/khemoo.simul/./require_auth": False},
        {"/rtx/fog/enabled": True, "/exts": {}},
    ],
)
def test_every_route_into_the_transport_settings_is_refused(settings: Dict[str, Any]) -> None:
    tools, sent = _recording_tools(_settings())

    result = asyncio.run(tools.set_carb_settings(settings))

    assert result["success"] is False
    assert result["error_type"] == "RefusedOperation"
    assert sent == [], "nothing may reach Kit"
    assert "No settings were applied" in result["error"]


@pytest.mark.parametrize(
    "settings",
    [
        {"/rtx/fog/enabled": True},
        {"/rtx/post": {"tonemap": {"op": 1}}},
        {"/extsfoo/x": 1},
        {"/exts/khemoo.simulation/x": 1},
        {"/exts/omni.physx/numThreads": 4},
    ],
)
def test_writes_outside_the_transport_still_go(settings: Dict[str, Any]) -> None:
    tools, sent = _recording_tools(_settings(), outputs=[{"applied": {}, "count": 0}])

    asyncio.run(tools.set_carb_settings(settings))

    assert len(sent) == 1


def test_write_paths_flatten_dict_values_and_normalize() -> None:
    assert carb_write_paths("exts//a/", {"b": {"c": 1}, "d": 2}) == [
        "/exts/a", "/exts/a/b", "/exts/a/b/c", "/exts/a/d",
    ]
    assert carb_write_paths("/a/../b", 1) is None
    assert carb_write_paths("", 1) == ["/"]


@pytest.mark.parametrize("namespace", PROTECTED_CARB_NAMESPACES)
def test_touch_rule_covers_inside_itself_and_ancestors(namespace: str) -> None:
    parent = namespace.rsplit("/", 1)[0]

    assert carb_path_touches(namespace, namespace)
    assert carb_path_touches(namespace + "/require_auth", namespace)
    assert carb_path_touches(parent, namespace)
    assert carb_path_touches("/", namespace)
    assert not carb_path_touches(namespace + "x", namespace)
    assert not carb_path_touches("/rtx", namespace)


def test_persistent_root_needs_consent_like_its_keys() -> None:
    tools, sent = _recording_tools(_settings(), outputs=[{"applied": {}, "count": 1}])

    refused = asyncio.run(tools.set_carb_settings({"/persistent": {"app": {"x": 1}}}))
    assert refused["error_type"] == "RefusedOperation" and sent == []

    asyncio.run(tools.set_carb_settings({"/persistent": {"app": {"x": 1}}}, allow_persistent=True))
    assert len(sent) == 1


@pytest.mark.parametrize(
    "settings",
    [
        {"/app/scripting/ignoreWarningDialog": True},
        {"/app/scripting": {"ignoreWarningDialog": True}},
        {"/app": {"scripting": {"ignoreWarningDialog": True}}},
        {"app/scripting//ignoreWarningDialog": True},
    ],
)
def test_scripting_settings_are_refused_while_scripts_are_off(settings: Dict[str, Any]) -> None:
    off, off_sent = _recording_tools(_settings(scripts=False))
    on, on_sent = _recording_tools(_settings(scripts=True), outputs=[{"applied": {}, "count": 1}])

    refused = asyncio.run(off.set_carb_settings(settings))
    asyncio.run(on.set_carb_settings(settings))

    assert refused["error_type"] == "ScriptExecutionDisabled" and off_sent == []
    assert len(on_sent) == 1


def test_other_app_settings_stay_writable_while_scripts_are_off() -> None:
    tools, sent = _recording_tools(_settings(scripts=False), outputs=[{"applied": {}, "count": 1}])

    asyncio.run(tools.set_carb_settings({"/app/window/width": 1280}))

    assert len(sent) == 1


# ---------------------------------------------------------------------------
# M3: the bridge latches auth; a later settings change cannot loosen it
# ---------------------------------------------------------------------------


def test_latched_auth_survives_the_subtree_being_cleared() -> None:
    live = {BRIDGE_REQUIRE_AUTH_SETTING: True, BRIDGE_AUTH_TOKEN_SETTING: "s3cret"}
    latch = BridgeAuthLatch(lambda key: live.get(key))
    latch.refresh()  # the bridge's start

    live.clear()  # carb set("/exts/khemoo.simul", {...}) replaced the subtree

    assert latch.required is True
    assert latch.check("") is False
    assert latch.check("s3cret") is True


def test_a_token_swapped_in_later_is_not_accepted() -> None:
    live = {PYTHON_SERVER_REQUIRE_AUTH_SETTING: True, PYTHON_SERVER_AUTH_TOKEN_SETTING: "kit-generated"}
    latch = BridgeAuthLatch(lambda key: live.get(key))
    assert latch.check("kit-generated") is True

    live[PYTHON_SERVER_AUTH_TOKEN_SETTING] = "attacker"
    live[BRIDGE_AUTH_TOKEN_SETTING] = "attacker"

    assert latch.check("attacker") is False
    assert latch.check("kit-generated") is True


def test_a_token_python_server_generates_after_the_bridge_starts_is_picked_up() -> None:
    live: Dict[str, Any] = {BRIDGE_REQUIRE_AUTH_SETTING: True}
    latch = BridgeAuthLatch(lambda key: live.get(key))
    latch.refresh()

    assert latch.check("") is False  # fail closed until a token exists
    assert latch.check("anything") is False
    live[PYTHON_SERVER_AUTH_TOKEN_SETTING] = "late-token"
    assert latch.check("late-token") is True


def test_without_auth_settings_the_latch_stays_open_until_auth_appears() -> None:
    live: Dict[str, Any] = {}
    latch = BridgeAuthLatch(lambda key: live.get(key))

    assert latch.check("") is True
    live[BRIDGE_REQUIRE_AUTH_SETTING] = True
    live[BRIDGE_AUTH_TOKEN_SETTING] = "t"
    assert latch.check("") is False  # tightening is allowed
    live.clear()
    assert latch.check("") is False  # loosening is not


def test_unreadable_settings_do_not_loosen_the_latch() -> None:
    state = {"fail": False}
    live = {BRIDGE_AUTH_TOKEN_SETTING: "t"}

    def _get(key: str) -> Any:
        if state["fail"]:
            raise RuntimeError("settings gone")
        return live.get(key)

    latch = BridgeAuthLatch(_get)
    latch.refresh()
    state["fail"] = True

    assert latch.check("") is False
    assert latch.check("t") is True


def _fake_carb(monkeypatch: pytest.MonkeyPatch, live: Dict[str, Any]) -> None:
    settings = types.SimpleNamespace(get=live.get)
    fake = types.SimpleNamespace(settings=types.SimpleNamespace(get_settings=lambda: settings))
    monkeypatch.setattr(extension_module, "carb", fake)


def test_the_extension_authenticates_through_the_latch(monkeypatch: pytest.MonkeyPatch) -> None:
    live = {BRIDGE_REQUIRE_AUTH_SETTING: True, BRIDGE_AUTH_TOKEN_SETTING: "s3cret"}
    _fake_carb(monkeypatch, live)
    extension = IsaacMCPServerExtension()
    extension._auth_latch.refresh()

    live.clear()

    assert extension._auth_required() is True
    assert extension._check_auth_token("") is False
    assert extension._check_auth_token("s3cret") is True


def test_end_to_end_a_cleared_subtree_does_not_open_the_socket() -> None:
    live = {BRIDGE_REQUIRE_AUTH_SETTING: True, BRIDGE_AUTH_TOKEN_SETTING: "s3cret"}
    latch = BridgeAuthLatch(lambda key: live.get(key))
    latch.refresh()

    async def _handler(request: BridgeRequest) -> BridgeResponse:
        return BridgeResponse.success(request.request_id, {"reachable": True})

    async def scenario() -> Dict[str, Any]:
        lifecycle = BridgeServerLifecycle(
            host="127.0.0.1", port=0, request_handler=_handler, authenticate=latch.check
        )
        await lifecycle.start()
        try:
            live.clear()
            client = IsaacSocketClient(
                bridge_host="127.0.0.1", bridge_port=lifecycle.actual_port, prefer_bridge=True, timeout_seconds=5.0
            )
            return await client.bridge_request("ping", {})
        finally:
            await lifecycle.stop()

    reply = asyncio.run(scenario())

    assert reply["status"] == "error" and reply["error"]["name"] == AUTH_ERROR_NAME


# ---------------------------------------------------------------------------
# M4: Python Scripting Components
# ---------------------------------------------------------------------------


def _scripted_asset(path: Path, *, with_api: bool = True, with_attribute: bool = True) -> str:
    layer_stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Xform.Define(layer_stage, "/Asset")
    prim = UsdGeom.Xform.Define(layer_stage, "/Asset/Behaviour").GetPrim()
    if with_api:
        prim.AddAppliedSchema("OmniScriptingAPI")
    if with_attribute:
        prim.CreateAttribute("omni:scripting:scripts", Sdf.ValueTypeNames.AssetArray, False).Set(
            Sdf.AssetPathArray(["/tmp/behaviour.py"])
        )
    layer_stage.SetDefaultPrim(layer_stage.GetPrimAtPath("/Asset"))
    layer_stage.Save()
    return str(path)


@pytest.mark.parametrize(("with_api", "with_attribute"), [(True, True), (True, False), (False, True)])
def test_assets_with_scripting_components_are_not_imported_while_scripts_are_off(
    run_on_stage: Any, tmp_path: Path, with_api: bool, with_attribute: bool
) -> None:
    asset = _scripted_asset(tmp_path / "scripted.usda", with_api=with_api, with_attribute=with_attribute)
    stage = _world_stage()
    holder = stage.DefinePrim("/World/Holder")
    guarded = _recording_tools(_settings(sandbox=False, scripts=False))

    imported = _call(guarded, run_on_stage, stage, "import_isaac_asset", asset_path=asset, target_path="/World/A")
    referenced = _call(guarded, run_on_stage, stage, "add_isaac_reference", prim_path="/World/Holder",
                       reference_path=asset)

    for result in (imported, referenced):
        assert result["success"] is False and result["error_type"] == "ScriptExecutionDisabled"
        assert result["script_capable_prims"][0].endswith("/Behaviour")
        assert "Python Scripting Components" in result["error"]
    assert not stage.GetPrimAtPath("/World/A").IsValid()
    assert not holder.HasAuthoredReferences()

    allowed = _call(
        _recording_tools(_settings(sandbox=False, scripts=True)), run_on_stage, stage, "import_isaac_asset",
        asset_path=asset, target_path="/World/A",
    )
    assert allowed["success"] is True


def test_a_scripted_prim_is_not_duplicated_while_scripts_are_off(run_on_stage: Any) -> None:
    stage = _world_stage()
    prim = UsdGeom.Xform.Define(stage, "/World/Robot").GetPrim()
    prim.AddAppliedSchema("OmniScriptingAPI")
    guarded = _recording_tools(_settings(scripts=False))

    result = _call(guarded, run_on_stage, stage, "duplicate_isaac_prim", prim_path="/World/Robot",
                   new_path="/World/Robot2")

    assert result["success"] is False and result["error_type"] == "ScriptExecutionDisabled"
    assert result["details"]["nodes"] == [{"path": "/World/Robot", "node_type": "OmniScriptingAPI"}]
    assert not stage.GetPrimAtPath("/World/Robot2").IsValid()


@pytest.mark.parametrize("name", ["omni:scripting:scripts", "omni:scripting:anything"])
def test_scripting_attributes_are_code_attributes(name: str) -> None:
    tools, sent = _recording_tools(_settings(scripts=False))

    result = asyncio.run(tools.set_isaac_prim_attribute("/World/Robot", name, ["/tmp/evil.py"]))

    assert is_script_attribute(name)
    assert result["error_type"] == "ScriptExecutionDisabled" and sent == []


def test_the_scripting_api_is_not_a_prim_type_tools_create_while_scripts_are_off() -> None:
    tools, sent = _recording_tools(_settings(scripts=False))

    result = asyncio.run(tools.create_isaac_prim("/World/S", prim_type="OmniScriptingAPI"))

    assert result["error_type"] == "ScriptExecutionDisabled" and sent == []


@pytest.mark.parametrize("extension_id", ["omni.behavior.scripting.core", "omni.behavior.scripting.bundle"])
def test_the_6_0_scripting_extensions_count_as_script_extensions(extension_id: str) -> None:
    assert is_script_extension(extension_id)


def test_ordinary_prims_still_duplicate_and_import(run_on_stage: Any, tmp_path: Path) -> None:
    from tests.isaac.test_materials_import_save_graph import _asset

    stage = _world_stage()
    UsdGeom.Xform.Define(stage, "/World/Box")
    guarded = _recording_tools(_settings(sandbox=False, scripts=False))

    duplicated = _call(guarded, run_on_stage, stage, "duplicate_isaac_prim", prim_path="/World/Box",
                       new_path="/World/Box2")
    imported = _call(guarded, run_on_stage, stage, "import_isaac_asset",
                     asset_path=_asset(tmp_path / "clean.usda"), target_path="/World/Clean")

    assert duplicated["success"] is True and imported["success"] is True


# ---------------------------------------------------------------------------
# Guard leftovers: unloaded payloads, all-lowercase names
# ---------------------------------------------------------------------------


def _payload_asset(path: Path, payload: str) -> str:
    layer_stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(layer_stage, "/Asset").GetPrim()
    child = layer_stage.DefinePrim("/Asset/Heavy")
    child.GetPayloads().AddPayload(payload)
    layer_stage.SetDefaultPrim(root)
    layer_stage.Save()
    return str(path)


def test_an_import_with_unloaded_payloads_is_refused_while_scripts_are_off(run_on_stage: Any, tmp_path: Path) -> None:
    from tests.isaac.test_materials_import_save_graph import _asset

    asset = _payload_asset(tmp_path / "outer.usda", _asset(tmp_path / "inner.usda"))
    stage = Usd.Stage.CreateInMemory(load=Usd.Stage.LoadNone)
    UsdGeom.Xform.Define(stage, "/World")

    refused = _call(_recording_tools(_settings(sandbox=False, scripts=False)), run_on_stage, stage,
                    "import_isaac_asset", asset_path=asset, target_path="/World/A")

    assert refused["error_type"] == "ScriptExecutionDisabled"
    assert refused["unloaded_payload_prims"] == ["/World/A/Heavy"]
    assert not stage.GetPrimAtPath("/World/A").IsValid()


def test_loaded_payloads_are_checked_and_pass(run_on_stage: Any, tmp_path: Path) -> None:
    from tests.isaac.test_materials_import_save_graph import _asset

    asset = _payload_asset(tmp_path / "outer.usda", _asset(tmp_path / "inner.usda"))
    stage = _world_stage()

    result = _call(_recording_tools(_settings(sandbox=False, scripts=False)), run_on_stage, stage,
                   "import_isaac_asset", asset_path=asset, target_path="/World/A")

    assert result["success"] is True


def test_a_scripting_component_inside_a_loaded_payload_is_found(run_on_stage: Any, tmp_path: Path) -> None:
    asset = _payload_asset(tmp_path / "outer.usda", _scripted_asset(tmp_path / "inner.usda"))
    stage = _world_stage()

    result = _call(_recording_tools(_settings(sandbox=False, scripts=False)), run_on_stage, stage,
                   "import_isaac_asset", asset_path=asset, target_path="/World/A")

    assert result["error_type"] == "ScriptExecutionDisabled"
    assert result["script_capable_prims"] == ["/World/A/Heavy/Behaviour"]


@pytest.mark.parametrize(
    ("node_type", "expected"),
    [
        ("acme.nodes.runpythonscript", True),
        ("acme.nodes.pythonscriptnode", True),
        ("acme.nodes.robotdescription", False),
        ("acme.nodes.transcriptreader", False),
        ("acme.nodes.pythonscriptdescription", True),
        ("acme.nodes.transcript", False),
        ("omni.usd.schema.omniscripting.Reader", False),
    ],
)
def test_all_lowercase_script_names_are_script_words(node_type: str, expected: bool) -> None:
    namespace: Dict[str, Any] = {}
    exec(compile(script_guard_source(), "<script-guard>", "exec"), namespace)  # noqa: S102 - our own guard source

    assert is_script_capable_node_type(node_type) is expected
    assert namespace["_script_capable_type"](node_type) is expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [("inputs:pythonscript", True), ("inputs:userscript", True), ("inputs:description", False),
     ("inputs:timecode", False)],
)
def test_all_lowercase_code_attributes(name: str, expected: bool) -> None:
    assert is_script_attribute(name) is expected


# ---------------------------------------------------------------------------
# get_isaac_logs: transport tokens stay out of the result
# ---------------------------------------------------------------------------


def test_the_python_server_token_is_redacted_from_logs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    token = "Kx9-generated-token_abcdefghijklmnop"
    log = tmp_path / "kit.log"
    log.write_text(
        "2026-01-01T00:00:00Z [Info] [isaacsim.code_editor.python_server.extension] "
        f"Python server authentication token: {token}\n"
        f"2026-01-01T00:00:01Z [Warning] [omni.client] retrying with {token} later\n"
        "2026-01-01T00:00:02Z [Warning] [omni.physx] contact offset too small\n"
    )
    store = {"/log/file": str(log), "/exts/isaacsim.code_editor.python_server/auth_token": token}
    carb = types.ModuleType("carb")
    carb_settings = types.ModuleType("carb.settings")
    carb_settings.get_settings = lambda: types.SimpleNamespace(get=store.get)  # type: ignore[attr-defined]
    carb.settings = carb_settings  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "carb", carb)
    monkeypatch.setitem(sys.modules, "carb.settings", carb_settings)
    tools, captured = _recording_tools(_settings())

    def _run(**kwargs: Any) -> Dict[str, Any]:
        asyncio.run(tools.get_isaac_logs(**kwargs))
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(compile(captured[-1], "<logs>", "exec"), {"__name__": "__isaac_script__"})  # noqa: S102
        return json.loads(stdout.getvalue())

    everything = _run(level="info", last_n=10)
    probed = _run(level="info", search="Kx9")

    assert token not in json.dumps(everything)
    assert everything["entries"][0]["message"] == "Python server authentication token: ***"
    assert everything["entries"][1]["message"] == "retrying with *** later"
    assert everything["redacted_lines"] == 2
    assert probed["total_matching"] == 0, "a search must not probe the token"


# ---------------------------------------------------------------------------
# Bridge raw-code path: a python_server envelope is refused, not run as a no-op
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('{"code": "print(1)", "timeout": 5}', True),
        ('{"introspect": "status"}', True),
        ('  {"code": "x", "auth_token": "t"}', True),
        ("print('hi')", False),
        ('{"a": 1}', False),
        ("{not json", False),
    ],
)
def test_envelope_detection(source: str, expected: bool) -> None:
    assert is_python_server_envelope(source) is expected


def _raw_exchange(source: str) -> Dict[str, Any]:
    ran: List[str] = []

    async def _run_code(code: str) -> Dict[str, Any]:
        ran.append(code)
        return {"status": "ok", "output": "ran"}

    async def _handler(request: BridgeRequest) -> BridgeResponse:
        return BridgeResponse.success(request.request_id, {})

    async def scenario() -> Dict[str, Any]:
        lifecycle = BridgeServerLifecycle(
            host="127.0.0.1", port=0, request_handler=_handler, vscode_handler=_run_code
        )
        await lifecycle.start()
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", lifecycle.actual_port)
            writer.write(source.encode())
            await writer.drain()
            writer.write_eof()
            reply = json.loads((await reader.read()).decode())
            writer.close()
            return {"reply": reply, "ran": ran}
        finally:
            await lifecycle.stop()

    return asyncio.run(scenario())


def test_an_envelope_on_the_raw_path_is_refused_and_nothing_runs() -> None:
    outcome = _raw_exchange('{"code": "import os; os.remove(\'x\')", "timeout": 5}')

    assert outcome["ran"] == []
    assert outcome["reply"]["ename"] == ENVELOPE_ERROR_NAME
    assert "8226" in outcome["reply"]["evalue"]


def test_raw_python_on_the_raw_path_still_runs() -> None:
    outcome = _raw_exchange("print('hi')")

    assert outcome["ran"] == ["print('hi')"]
    assert outcome["reply"]["status"] == "ok"


def test_a_misconfigured_client_reads_the_bridge_as_the_raw_flavour() -> None:
    """The flavour probe is an envelope: refused, so auto-detection picks raw source."""
    ran: List[str] = []

    async def _run_code(code: str) -> Dict[str, Any]:
        ran.append(code)
        return {"status": "ok", "output": "pong"}

    async def _handler(request: BridgeRequest) -> BridgeResponse:
        return BridgeResponse.success(request.request_id, {})

    async def scenario() -> Any:
        lifecycle = BridgeServerLifecycle(host="127.0.0.1", port=0, request_handler=_handler, vscode_handler=_run_code)
        await lifecycle.start()
        try:
            client = IsaacSocketClient(host="127.0.0.1", port=lifecycle.actual_port, timeout_seconds=10.0)
            result = await client.execute_vscode_only("print('pong')")
            return client.socket_protocol, result
        finally:
            await lifecycle.stop()

    protocol, result = asyncio.run(scenario())

    assert protocol == "vscode"
    assert result.success is True and ran == ["print('pong')"]


# ---------------------------------------------------------------------------
# Error typing (aligned with simul.mcp.error_types after #329)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("No stage is currently open", "InvalidState"),
        ("Prim not found: /World/X", "NotFoundError"),
        ("Root path not found: /Wrold", "NotFoundError"),
        ("No camera found in scene", "NotFoundError"),
        ("No physics scene on the stage", "NotFoundError"),
        ("Prim is not Xformable: /World/L", "ValidationError"),
        ("Prim does not have MassAPI: /World/B", "ValidationError"),
        ("Not a valid Camera prim: /World/C", "ValidationError"),
        ("Prim already exists at /World/M", "InvalidState"),
        ("No active viewport found", "ViewportUnavailable"),
        ("Viewport capture failed - file not created", "CaptureError"),
        ("PhysX scene query not available. Is physics enabled?", "UnsupportedOperation"),
        ("Failed to open stage: bad file", "EngineError"),
    ],
)
def test_untyped_script_failures_get_a_type_from_the_vocabulary(message: str, expected: str) -> None:
    from simul.mcp import error_types

    tools, _ = _recording_tools(_settings(), outputs=[{"error": message}])

    result = asyncio.run(tools._execute_json_script("print('x')"))

    assert result["success"] is False
    assert result["error_type"] == expected
    assert error_types.is_known(result["error_type"])


def test_a_script_that_types_its_own_failure_keeps_the_type() -> None:
    tools, _ = _recording_tools(
        _settings(), outputs=[{"error": "Prim not found: /x", "error_type": "RefusedOperation"}]
    )

    result = asyncio.run(tools._execute_json_script("print('x')"))

    assert result["error_type"] == "RefusedOperation"


def test_transport_failures_use_the_vocabulary() -> None:
    from simul.adapters.isaac_socket_client import ScriptResult
    from simul.mcp.tools.isaac._base import IsaacScriptBase

    bridge_timeout = ScriptResult(
        success=False, output="", error_name="ScriptInterrupted",
        error_value="Script timed out after 29s and was interrupted", transport="bridge", transport_failure=True,
    )
    too_large = ScriptResult(
        success=False, output="", error_name="ScriptTooLargeForSocket", error_value="too big",
        transport="vscode", transport_failure=True,
    )

    assert IsaacScriptBase._script_failure(bridge_timeout)["error_type"] == "TimeoutError"
    assert IsaacScriptBase._script_failure(too_large)["error_type"] == "PayloadTooLarge"
