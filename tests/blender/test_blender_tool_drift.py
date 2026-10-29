"""Every Blender tool is one ``BLENDER_TOOLS`` entry, and every entry fits the session it calls.

``register_blender_tools`` builds each tool from its entry and the
``BlenderRuntimeSession`` operation the entry names. An entry naming an
operation the session lacks, overriding a parameter the operation does not
have, or validating with a request model that does not cover the advertised
parameters would register a tool that fails on every call, as
check_blender_object_bounds did while it named ``check_object_bounds``. These
tests fail on that at once, without a Blender.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import typing
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pytest
from pydantic import ValidationError

from simul.adapters.blender_runtime import BlenderRuntimeSession
from simul.config import Settings
from simul.mcp import backends as backends_module
from simul.mcp import server as server_module
from simul.mcp.registration._reg_blender import (
    BLENDER_TOOLS,
    BlenderTool,
    session_operation,
)
from simul.mcp.usage_tracker import ToolUsageTracker
from simul.tool_surfaces import THIN_BLENDER_TOOLS
from tests.fakes import AvailableAdapter, FakeFastMCP

#: The operations a tool may call: what BlenderConnection and the add-on expose.
PUBLIC_OPERATIONS = {
    name
    for name, member in vars(BlenderRuntimeSession).items()
    if not name.startswith("_") and inspect.isfunction(member)
}


def _operation_parameters(spec: BlenderTool) -> Dict[str, inspect.Parameter]:
    """The operation's parameters without ``self``."""
    return dict(list(inspect.signature(session_operation(spec)).parameters.items())[1:])


def _server(tmp_path: Path, **security: Any) -> server_module.SimulMCPServer:
    """A server with the full Blender surface registered on the recording double."""
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(server_module, "FastMCP", FakeFastMCP)
        patcher.setattr(server_module, "TaskConfig", None)
        patcher.setattr(backends_module, "is_blender_available", lambda: True)
        patcher.setattr(backends_module, "BlenderRuntimeAdapter", AvailableAdapter)
        instance = server_module.SimulMCPServer(
            Settings(
                blender={"tool_surface": "full"},
                security={"allow_script_execution": True, **security},
            ),
            backends={"blender"},
        )
    instance.usage_tracker = ToolUsageTracker(log_dir=tmp_path / "usage")
    return instance


@pytest.fixture(scope="module")
def blender_server(
    tmp_path_factory: pytest.TempPathFactory,
) -> server_module.SimulMCPServer:
    return _server(tmp_path_factory.mktemp("drift"))


def _advertised(
    server: server_module.SimulMCPServer, spec: BlenderTool
) -> Dict[str, inspect.Parameter]:
    return dict(inspect.signature(server.mcp.by_name[spec.name]).parameters)


def test_tool_names_are_unique() -> None:
    names = Counter(spec.name for spec in BLENDER_TOOLS)
    assert [name for name, count in names.items() if count > 1] == []


def test_every_operation_is_a_public_session_operation() -> None:
    for spec in BLENDER_TOOLS:
        if spec.operation is None:
            assert (
                spec.build is not None
            ), f"{spec.name} calls nothing and builds nothing"
            continue
        assert spec.operation in PUBLIC_OPERATIONS, spec.name
        # The operation's annotations resolve, so its schema can be derived.
        typing.get_type_hints(session_operation(spec))


def test_overrides_name_parameters_of_the_operation() -> None:
    for spec in BLENDER_TOOLS:
        if spec.build is not None:
            # A hand-written wrapper declares its own parameters and call.
            assert (
                not spec.params and not spec.descriptions and spec.call is None
            ), spec.name
            continue
        parameters = _operation_parameters(spec)
        assert set(spec.params) <= set(parameters), spec.name
        assert set(spec.descriptions) <= set(parameters), spec.name
        if spec.call is not None:
            inspect.signature(spec.call).bind(None, **dict.fromkeys(parameters))


def test_thin_surface_names_declared_tools() -> None:
    assert set(THIN_BLENDER_TOOLS) <= {spec.name for spec in BLENDER_TOOLS}


def test_registration_follows_the_table(
    blender_server: server_module.SimulMCPServer,
) -> None:
    """The same tools in the same order, each with the entry's description and hints."""
    registered = [
        tool for tool in blender_server.mcp.tools if tool.name != "get_tool_usage_stats"
    ]
    assert [tool.name for tool in registered] == [spec.name for spec in BLENDER_TOOLS]
    for tool, spec in zip(registered, BLENDER_TOOLS):
        assert tool.kwargs["description"] == spec.description, spec.name
        hints = tool.kwargs["annotations"].model_dump(exclude_none=True)
        assert hints == {
            "readOnlyHint": spec.hints.read_only,
            **({"destructiveHint": True} if spec.hints.destructive else {}),
            **({"idempotentHint": True} if spec.hints.idempotent else {}),
        }, spec.name
        assert tool.kwargs["output_schema"] is None, spec.name


def test_advertised_parameters_fit_the_operation_and_the_request_model(
    blender_server: server_module.SimulMCPServer,
) -> None:
    """Each advertised parameter is validated by the request model and taken by the operation.

    Every parameter the operation requires is advertised, so no call can miss one.
    """
    for spec in BLENDER_TOOLS:
        advertised = _advertised(blender_server, spec)
        if spec.request is not None:
            assert set(advertised) == set(spec.request.model_fields), spec.name
        if spec.operation is None:
            continue
        operation = _operation_parameters(spec)
        assert set(advertised) <= set(operation), spec.name
        required = {
            name
            for name, parameter in operation.items()
            if parameter.default is parameter.empty
        }
        assert required <= set(advertised), spec.name


def test_generated_wrappers_mirror_the_operation(
    blender_server: server_module.SimulMCPServer,
) -> None:
    """Parameters in the operation's order with its defaults, unless the entry says otherwise."""
    for spec in BLENDER_TOOLS:
        if spec.build is not None:
            continue
        advertised = _advertised(blender_server, spec)
        operation = _operation_parameters(spec)
        assert list(advertised) == list(operation), spec.name
        for name, parameter in operation.items():
            if name not in spec.params:
                assert (
                    advertised[name].default == parameter.default
                ), f"{spec.name}.{name}"
        assert (
            typing.get_type_hints(blender_server.mcp.by_name[spec.name])["return"]
            is server_module.ToolResult
        )


def test_only_get_blender_camera_info_takes_arguments_without_a_request_model(
    blender_server: server_module.SimulMCPServer,
) -> None:
    """A new tool with parameters needs a request model, so its input is validated in the envelope."""
    unvalidated = {
        spec.name
        for spec in BLENDER_TOOLS
        if spec.request is None and _advertised(blender_server, spec)
    }
    assert unvalidated == {"get_blender_camera_info"}


def _invalid_arguments(advertised: Dict[str, inspect.Parameter]) -> Dict[str, Any]:
    """An object for every required parameter, or for the first one when none is required."""
    bogus = {"bogus": 1}
    required = {
        name: bogus
        for name, parameter in advertised.items()
        if parameter.default is parameter.empty
    }
    return required or {next(iter(advertised)): bogus}


def _rows(instance: server_module.SimulMCPServer, tool: str) -> List[Dict[str, Any]]:
    rows = reversed(instance.usage_tracker.get_recent(limit=1000, tool_name=tool))
    return [{"success": row["success"], "error": row.get("error")} for row in rows]


def test_invalid_input_is_rate_limited_and_recorded(tmp_path: Path) -> None:
    """Validation runs inside the envelope: an invalid call spends the tool's one token."""
    instance = _server(tmp_path, burst_size=1, global_requests_per_minute=100_000)
    for spec in BLENDER_TOOLS:
        if spec.request is None:
            continue
        tool = instance.mcp.by_name[spec.name]
        arguments = _invalid_arguments(_advertised(instance, spec))

        invalid = json.loads(asyncio.run(tool(**arguments)).content[-1].text)
        refused = json.loads(asyncio.run(tool(**arguments)).content[-1].text)

        assert invalid["error_type"] == "ValidationError", spec.name
        assert (
            invalid["details"]["input"].keys() == _advertised(instance, spec).keys()
        ), spec.name
        assert refused["error_type"] == "RateLimitError", spec.name
        assert _rows(instance, spec.name) == [
            {"success": False, "error": invalid["error"]},
            {"success": False, "error": "rate_limited"},
        ], spec.name


def test_an_entry_naming_a_missing_operation_fails_to_build() -> None:
    """The defect check_blender_object_bounds shipped with now stops registration."""
    spec = BlenderTool(
        name="check_blender_object_bounds",
        description="x",
        hints=BLENDER_TOOLS[0].hints,
        response=BLENDER_TOOLS[0].response,
        operation="check_object_bounds",
    )
    with pytest.raises(TypeError, match="not a BlenderRuntimeSession operation"):
        session_operation(spec)


def test_advertised_defaults_pass_the_request_model(
    blender_server: server_module.SimulMCPServer,
) -> None:
    """A caller who omits an optional parameter must get a valid call.

    add_blender_force_field advertised ``location=None`` while its request
    model required a list, so every call without a location failed validation.
    """
    for spec in BLENDER_TOOLS:
        # Hand-written wrappers (``build``) may fill a default in before
        # validating, e.g. the agent label; generated ones forward it as is.
        if spec.request is None or spec.build is not None:
            continue
        for name, parameter in _advertised(blender_server, spec).items():
            if parameter.default is parameter.empty:
                continue
            # Validate through the real model; other fields' "missing" errors
            # are expected here, an error on this field is the bug.
            try:
                spec.request.model_validate({name: parameter.default})
            except ValidationError as exc:
                own = [error for error in exc.errors() if error["loc"][:1] == (name,)]
                assert not own, f"{spec.name}.{name}: default {parameter.default!r} is invalid: {own}"


def test_light_color_needs_exactly_three_components() -> None:
    """``color=[]`` used to reach Blender, which then failed assigning an empty tuple."""
    from simul.mcp.schemas.blender import BlenderSetLightParamsRequest

    assert BlenderSetLightParamsRequest(light_name="Light", color=None).color is None
    assert BlenderSetLightParamsRequest(light_name="Light", color=[1, 0.5, 0.25]).color == [1, 0.5, 0.25]
    for bad in ([], [1, 0.5], [1, 0.5, 0.25, 1]):
        with pytest.raises(ValidationError):
            BlenderSetLightParamsRequest(light_name="Light", color=bad)
