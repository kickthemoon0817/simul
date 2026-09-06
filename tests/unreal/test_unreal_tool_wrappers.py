"""Every Unreal tool wrapper calls its session method with arguments the method accepts.

The wrappers restate each ``UnrealRuntimeSession`` call by hand, so a renamed
or removed session parameter used to surface only as a ``TypeError`` inside
the envelope, on the first live call. Here every registered Unreal tool is
invoked against a session double that binds each call to the real method
signature and records the mismatch.
"""

from __future__ import annotations

import asyncio
import inspect
import typing
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, get_args, get_origin

import pytest

from simul_mcp.adapters.unreal_runtime import UnrealRuntimeSession
from simul_mcp.config import Settings
from simul_mcp.mcp import backends as backends_module
from simul_mcp.mcp import server as server_module
from simul_mcp.mcp.registration import register_unreal_tools
from tests.fakes import AvailableAdapter


class _BindingSession:
    """Accepts any session method and checks the call against the real signature."""

    def __init__(self) -> None:
        self.started: List[str] = []  # async methods whose coroutine was created
        self.calls: List[str] = []  # methods whose call was actually made (awaited when async)
        self.failures: List[str] = []

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        method = getattr(UnrealRuntimeSession, name, None)
        if method is None:
            self.failures.append(f"{name}: UnrealRuntimeSession has no such method")
            raise AttributeError(name)
        declared = inspect.getattr_static(UnrealRuntimeSession, name)
        receiver = () if isinstance(declared, (staticmethod, classmethod)) else (self,)

        def _bind(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            try:
                inspect.signature(method).bind(*receiver, *args, **kwargs)
            except TypeError as exc:
                self.failures.append(f"{name}: {exc}")
                raise
            self.calls.append(name)
            return {}

        if not inspect.iscoroutinefunction(method):
            return _bind

        async def _run(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            return _bind(*args, **kwargs)

        def _call(*args: Any, **kwargs: Any) -> Any:
            self.started.append(name)
            return _run(*args, **kwargs)

        return _call


class _Adapter(AvailableAdapter):
    def __init__(self, session: _BindingSession) -> None:
        super().__init__(Settings())
        self.session = session

    @contextmanager
    def create_session(self) -> Iterator[_BindingSession]:
        yield self.session


def _sample(name: str, annotation: Any) -> Any:
    """Build one plausible argument for a required wrapper parameter."""
    origin = get_origin(annotation)
    if origin is typing.Union:
        return _sample(name, next(arg for arg in get_args(annotation) if arg is not type(None)))
    if origin in (list, List):
        return ["/Game/Maps/Test.Test:PersistentLevel.Actor_0"]
    if origin in (dict, Dict):
        return {"key": "value"}
    if annotation is bool:
        return True
    if annotation is int:
        return 1
    if annotation is float:
        return 1.0
    if "operations" in name:
        return "[]"
    if any(token in name for token in ("origin", "normal", "location", "rotation", "scale", "color", "direction")):
        return "0,0,1"
    if "paths" in name:
        return "/Game/Maps/Test.Test:PersistentLevel.A,/Game/Maps/Test.Test:PersistentLevel.B"
    if "path" in name:
        return "/Game/Meshes/SM_Cube"
    return "value"


def _required_arguments(function: Any) -> Dict[str, Any]:
    hints = typing.get_type_hints(function)
    return {
        name: _sample(name, hints.get(name, str))
        for name, parameter in inspect.signature(function).parameters.items()
        if parameter.default is inspect.Parameter.empty
    }


@pytest.fixture(scope="module")
def unreal_tools() -> Dict[str, Any]:
    """The full Unreal tool surface, keyed by name, on a server whose adapter binds calls."""
    from tests.fakes import FakeFastMCP

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
        monkeypatch.setattr(server_module, "TaskConfig", None)
        monkeypatch.setattr(backends_module, "is_headless_available", lambda: False)
        monkeypatch.setattr(backends_module, "is_blender_available", lambda: False)
        monkeypatch.setattr(backends_module, "UnrealRuntimeAdapter", None)
        instance = server_module.SimulMCPServer(settings=Settings(), backends={"isaac"})
    register_unreal_tools(instance, thin=False)
    return {"server": instance, "tools": {name: func for name, func in instance.mcp.by_name.items() if "unreal" in name or "simready" in name}}


def test_the_full_surface_is_under_test(unreal_tools: Dict[str, Any]) -> None:
    assert len(unreal_tools["tools"]) > 50


def test_every_unreal_tool_calls_its_session_method_with_accepted_arguments(
    unreal_tools: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    instance = unreal_tools["server"]
    session = _BindingSession()
    monkeypatch.setattr(instance, "unreal_adapter", _Adapter(session))
    # Sandbox checks inside the session are what the double skips; the
    # wrappers' own path pre-checks must not short-circuit the binding.
    monkeypatch.setattr(instance, "_sandbox_denial", lambda *args, **kwargs: None)

    silent: List[str] = []
    dropped: Dict[str, List[str]] = {}
    for name, function in sorted(unreal_tools["tools"].items()):
        before = len(session.calls) + len(session.failures)
        started_before = len(session.started)
        asyncio.run(function(**_required_arguments(function)))
        if len(session.calls) + len(session.failures) == before:
            silent.append(name)
        started = session.started[started_before:]
        awaited = [call for call in session.calls[before:] if call in started]
        if len(started) != len(awaited):
            dropped[name] = [call for call in started if call not in awaited]

    assert session.failures == []
    # A session coroutine that is created but never awaited does nothing at all.
    assert dropped == {}, f"wrappers that drop a session call without awaiting it: {dropped}"
    # Tools that never open a session (ping, instance listing) are the only ones allowed to stay silent.
    assert set(silent) <= {"ping_unreal", "list_unreal_instances"}, silent
    assert len(session.calls) > 50


@pytest.mark.parametrize(
    ("tool_name", "session_method"),
    [
        ("import_unreal_usd", "import_usd"),
        ("export_unreal_usd", "export_usd"),
        ("convert_to_simready", "convert_to_simready"),
        ("validate_simready_asset", "validate_simready_asset"),
    ],
)
def test_usd_and_simready_wrappers_match_their_session_methods(
    unreal_tools: Dict[str, Any], tool_name: str, session_method: str
) -> None:
    """The four wrappers that used to pass parameters the session never accepted."""
    wrapper_parameters = set(inspect.signature(unreal_tools["tools"][tool_name]).parameters)
    session_parameters = set(inspect.signature(getattr(UnrealRuntimeSession, session_method)).parameters) - {"self"}
    assert wrapper_parameters == session_parameters, (tool_name, wrapper_parameters ^ session_parameters)
