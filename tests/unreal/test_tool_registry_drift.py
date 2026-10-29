"""Every Unreal tool is declared once, in ``_unreal_tools.UNREAL_TOOLS``.

``register_unreal_tools`` builds each tool's FastMCP wrapper, request model and
session call from its declaration. A declaration that drifts from the session
method it names would otherwise surface as a ``TypeError`` inside the envelope
on the first call in an agent session; here it fails first. Every declaration
names a session coroutine that accepts the arguments the declaration builds,
every parameter is described, names are unique, the thin surface is a subset of
the declared tools, and each registered wrapper advertises exactly the declared
parameters with the declared hints.

The last section covers what the generated tools do with input their request
model refuses: it is validated inside the envelope, so it is rate limited and
recorded like any other failed call, and a declaration's own check reports its
message as written. (Input that does not fit the schema never reaches a wrapper;
FastMCP refuses it first.)
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from simul.adapters.unreal_runtime import UnrealRuntimeSession
from simul.config import Settings
from simul.mcp import backends as backends_module
from simul.mcp import server as server_module
from simul.mcp.registration._helpers import describe_params
from simul.mcp.registration._reg_unreal import (
    InvalidToolInput,
    request_model_for,
    session_arguments,
)
from simul.mcp.registration._unreal_tools import REQUIRED, UNREAL_TOOLS, UnrealTool
from simul.mcp.usage_tracker import ToolUsageTracker
from simul.tool_surfaces import THIN_UNREAL_TOOLS
from tests.fakes import AvailableAdapter, FakeFastMCP

SPECS: Dict[str, UnrealTool] = {spec.name: spec for spec in UNREAL_TOOLS}
METHOD_SPECS = [spec for spec in UNREAL_TOOLS if spec.method is not None]


def _server(monkeypatch: pytest.MonkeyPatch, **settings: Any) -> server_module.SimulMCPServer:
    """A server with only the Unreal backend registered on the recording double."""
    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)
    monkeypatch.setattr(backends_module, "UnrealRuntimeAdapter", AvailableAdapter)
    monkeypatch.setattr(backends_module, "is_unreal_available", lambda: True)
    return server_module.SimulMCPServer(Settings(**settings), backends={"unreal"})


def _registered(instance: server_module.SimulMCPServer) -> List[str]:
    return [tool.name for tool in instance.mcp.tools if tool.name != "get_tool_usage_stats"]


def _stand_in(spec: UnrealTool) -> SimpleNamespace:
    """A request carrying every parameter's default, or a placeholder for a required one."""
    return SimpleNamespace(**{p.name: "x" if p.default is REQUIRED else p.default for p in spec.params})


# ---------------------------------------------------------------------------
# The declarations
# ---------------------------------------------------------------------------


def test_tool_names_are_unique() -> None:
    names = Counter(spec.name for spec in UNREAL_TOOLS)
    assert [name for name, count in names.items() if count > 1] == []


def test_parameter_names_are_unique_per_tool() -> None:
    for spec in UNREAL_TOOLS:
        names = [param.name for param in spec.params]
        assert len(names) == len(set(names)), spec.name


def test_the_thin_surface_is_declared() -> None:
    assert set(THIN_UNREAL_TOOLS) <= set(SPECS)


def test_each_tool_dispatches_one_way() -> None:
    """A session method or a sessionless call; ``arguments`` replaces the per-parameter mapping."""
    for spec in UNREAL_TOOLS:
        assert (spec.method is None) != (spec.sessionless is None), spec.name
        if spec.sessionless is not None:
            assert spec.arguments is None, spec.name
        if spec.arguments is not None or spec.sessionless is not None:
            assert all(p.convert is None and p.target is None for p in spec.params), (
                f"{spec.name}: convert and target are ignored when the tool builds its own arguments"
            )
        assert spec.request is None or spec.check is None, (
            f"{spec.name}: check is ignored when the tool brings its own request model"
        )


@pytest.mark.parametrize("spec", METHOD_SPECS, ids=lambda spec: spec.name)
def test_declared_method_is_a_session_coroutine(spec: UnrealTool) -> None:
    method = getattr(UnrealRuntimeSession, str(spec.method), None)
    assert method is not None, f"{spec.name}: UnrealRuntimeSession has no {spec.method}"
    assert inspect.iscoroutinefunction(method), spec.name


@pytest.mark.parametrize("spec", METHOD_SPECS, ids=lambda spec: spec.name)
def test_session_arguments_bind_to_the_declared_method(spec: UnrealTool) -> None:
    """Every argument the tool passes is a parameter of the method, and none it requires is missing."""
    if spec.arguments is not None:
        names = list(spec.arguments(_stand_in(spec)))
    else:
        names = [param.target or param.name for param in spec.params]
    signature = inspect.signature(getattr(UnrealRuntimeSession, str(spec.method)))
    signature.bind(None, **{name: object() for name in names})


def test_converted_arguments_reach_the_session_under_their_target() -> None:
    spec = SPECS["set_unreal_material_params"]
    request = SimpleNamespace(
        material_path="/Game/M",
        scalar_params_json='{"Roughness": 0.4}',
        vector_params_json="",
        texture_params_json='{"Albedo": "/Game/T"}',
    )

    assert session_arguments(spec, request) == {
        "material_path": "/Game/M",
        "scalar_params": {"Roughness": 0.4},
        "vector_params": None,
        "texture_params": {"Albedo": "/Game/T"},
    }


@pytest.mark.parametrize("spec", UNREAL_TOOLS, ids=lambda spec: spec.name)
def test_every_parameter_is_described(spec: UnrealTool) -> None:
    documented = describe_params(getattr(UnrealRuntimeSession, spec.method)) if spec.method else {}
    missing = [p.name for p in spec.params if not (p.description or documented.get(p.name))]
    assert missing == [], f"{spec.name}: describe these parameters or document them in the session method"


@pytest.mark.parametrize("spec", UNREAL_TOOLS, ids=lambda spec: spec.name)
def test_request_model_has_one_field_per_parameter(spec: UnrealTool) -> None:
    model = spec.request or request_model_for(spec)
    assert list(model.model_fields) == [param.name for param in spec.params], spec.name
    for param in spec.params:
        field = model.model_fields[param.name]
        if param.default is REQUIRED:
            assert field.is_required(), f"{spec.name}.{param.name}"
        else:
            assert field.default == param.default, f"{spec.name}.{param.name}"


def test_a_check_fails_validation_with_its_own_message() -> None:
    model = request_model_for(SPECS["capture_unreal_viewport"])

    with pytest.raises(InvalidToolInput, match=r"^Invalid format 'jpg'. Must be one of \['png'\]$"):
        model(format="jpg")
    assert model(format="png").format == "png"


# ---------------------------------------------------------------------------
# What registers
# ---------------------------------------------------------------------------


def test_full_surface_registers_every_declaration_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = _server(monkeypatch, unreal={"tool_surface": "full"})
    assert _registered(instance) == [spec.name for spec in UNREAL_TOOLS]


def test_thin_surface_registers_the_thin_set(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = _server(monkeypatch, unreal={"tool_surface": "thin"})
    assert _registered(instance) == [spec.name for spec in UNREAL_TOOLS if spec.name in THIN_UNREAL_TOOLS]


def test_script_tools_register_only_with_script_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = _server(
        monkeypatch, unreal={"tool_surface": "full"}, security={"allow_script_execution": False}
    )
    assert _registered(instance) == [spec.name for spec in UNREAL_TOOLS if not spec.script]


def test_registered_wrappers_mirror_the_declarations(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = _server(monkeypatch, unreal={"tool_surface": "full"})
    registered = {tool.name: tool for tool in instance.mcp.tools}
    for spec in UNREAL_TOOLS:
        tool = registered[spec.name]
        assert tool.kwargs["description"] == spec.description
        hints = tool.kwargs["annotations"].model_dump(by_alias=True, exclude_none=True)
        assert hints["readOnlyHint"] is spec.read_only, spec.name
        assert hints.get("destructiveHint", False) is spec.destructive, spec.name
        assert hints.get("idempotentHint", False) is spec.idempotent, spec.name
        assert hints.get("openWorldHint", True) is True, spec.name
        parameters = inspect.signature(tool.func).parameters
        assert list(parameters) == [param.name for param in spec.params], spec.name
        for param in spec.params:
            assert parameters[param.name].default == param.default, f"{spec.name}.{param.name}"


# ---------------------------------------------------------------------------
# Invalid input is validated inside the envelope
# ---------------------------------------------------------------------------


class _Session:
    def __init__(self) -> None:
        self.calls: List[str] = []

    async def capture_viewport(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append("capture_viewport")
        return {"path": "/shot.png", "size_bytes": 1, "resolution_x": 1, "resolution_y": 1, "format": "png"}


class _Adapter(AvailableAdapter):
    def __init__(self, session: _Session) -> None:
        super().__init__(Settings())
        self.session = session

    def create_session(self) -> Any:
        from contextlib import nullcontext

        return nullcontext(self.session)


def _capture_server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **security: Any) -> Any:
    instance = _server(monkeypatch, unreal={"tool_surface": "full"}, security=security)
    instance.usage_tracker = ToolUsageTracker(log_dir=tmp_path / "usage")
    session = _Session()
    monkeypatch.setattr(instance, "unreal_adapter", _Adapter(session))
    return instance, session


def _call(instance: server_module.SimulMCPServer, tool: str, **arguments: Any) -> Dict[str, Any]:
    result = asyncio.run(instance.mcp.by_name[tool](**arguments))
    return json.loads(result.content[-1].text)


def _records(instance: server_module.SimulMCPServer, tool: str) -> List[Dict[str, Any]]:
    rows = list(reversed(instance.usage_tracker.get_recent(tool_name=tool)))
    return [{key: row[key] for key in ("success", "error") if key in row} for row in rows]


def test_invalid_input_reports_the_check_and_the_input(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    instance, session = _capture_server(monkeypatch, tmp_path)

    payload = _call(instance, "capture_unreal_viewport", format="jpg")

    assert payload == {
        "success": False,
        "error": "Invalid format 'jpg'. Must be one of ['png']",
        "error_type": "ValidationError",
        "details": {"input": {"resolution_x": 1920, "resolution_y": 1080, "format": "jpg", "inline": False}},
    }
    assert session.calls == []
    assert _records(instance, "capture_unreal_viewport") == [
        {"success": False, "error": "Invalid format 'jpg'. Must be one of ['png']"}
    ]


def test_invalid_input_spends_a_rate_limit_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """With one token, the invalid call spends it and the valid one after it is refused."""
    instance, session = _capture_server(monkeypatch, tmp_path, burst_size=1)

    invalid = _call(instance, "capture_unreal_viewport", format="jpg")
    refused = _call(instance, "capture_unreal_viewport")

    assert invalid["error_type"] == "ValidationError"
    assert refused["error_type"] == "RateLimitError"
    assert session.calls == []
    assert _records(instance, "capture_unreal_viewport") == [
        {"success": False, "error": invalid["error"]},
        {"success": False, "error": "rate_limited"},
    ]
