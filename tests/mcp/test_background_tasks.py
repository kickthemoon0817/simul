"""Background task support follows FastMCP's tasks extension.

FastMCP 4 runs background tasks through the ``io.modelcontextprotocol/tasks``
extension from the optional ``fastmcp-tasks`` package (``fastmcp[tasks]``) and
refuses to start a server that has a task-enabled tool but no such extension.
The server registers the extension when the package imports, and only then
gives its tools the optional task config; otherwise they run in the
foreground, as they do in the locked environment.
"""

from __future__ import annotations

import logging
import sys
from types import ModuleType
from typing import Any, Optional

import pytest
from fastmcp.utilities.tasks import TaskConfig

from simul.config import Settings
from simul.mcp import backends as backends_module
from simul.mcp import server as server_module
from tests.fakes import AvailableAdapter, FakeFastMCP


class _TasksExtension:
    """Stands in for ``fastmcp_tasks.TasksExtension``."""


class _BrokenTasksExtension:
    def __init__(self) -> None:
        raise RuntimeError("docket backend unreachable")


def _tasks_package(extension: Optional[type]) -> Optional[ModuleType]:
    """A ``fastmcp_tasks`` module exporting ``extension``; None makes the import fail."""
    if extension is None:
        return None
    module = ModuleType("fastmcp_tasks")
    module.TasksExtension = extension  # type: ignore[attr-defined]
    return module


def _server(
    monkeypatch: pytest.MonkeyPatch, extension: Optional[type]
) -> server_module.SimulMCPServer:
    """A full-surface Unreal server, whose every tool takes the task config."""
    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setitem(sys.modules, "fastmcp_tasks", _tasks_package(extension))
    monkeypatch.setattr(backends_module, "UnrealRuntimeAdapter", AvailableAdapter)
    monkeypatch.setattr(backends_module, "is_unreal_available", lambda: True)
    return server_module.SimulMCPServer(
        Settings(unreal={"tool_surface": "full"}), backends={"unreal"}
    )


def _task_configs(instance: server_module.SimulMCPServer) -> list[Any]:
    unreal = set(instance.tools_by_backend()["Unreal"])
    return [tool.kwargs["task"] for tool in instance.mcp.tools if tool.name in unreal]


def test_without_fastmcp_tasks_every_tool_runs_in_the_foreground(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = _server(monkeypatch, None)

    assert instance.mcp.extensions == []
    assert set(map(type, _task_configs(instance))) == {type(None)}


def test_with_fastmcp_tasks_the_extension_is_registered_and_tools_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = _server(monkeypatch, _TasksExtension)

    assert [type(extension) for extension in instance.mcp.extensions] == [
        _TasksExtension
    ]
    configs = _task_configs(instance)
    assert configs and all(config == TaskConfig(mode="optional") for config in configs)


def test_an_extension_that_fails_to_register_leaves_tasks_off(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        instance = _server(monkeypatch, _BrokenTasksExtension)

    assert instance.mcp.extensions == []
    assert set(map(type, _task_configs(instance))) == {type(None)}
    assert "docket backend unreachable" in caplog.text
