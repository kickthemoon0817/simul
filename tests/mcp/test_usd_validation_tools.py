"""``validate_usd_file`` and ``load_usd_file`` tell the truth through the real MCP stack.

``validate_usd_file`` returned ``is_valid: true`` for files ``load_usd_file``
refused, and every load failure read "Failed to load USD file: <path>". These
drive a real FastMCP server whose sandbox is the test's temporary directory.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict

import pytest
from fastmcp import Client

pytest.importorskip("pxr", reason="pxr library not available")

from simul.config import Settings  # noqa: E402
from simul.mcp.server import SimulMCPServer  # noqa: E402
from tests import usd_fixtures as fx  # noqa: E402


def _server(root: Path, **usd: Any) -> SimulMCPServer:
    settings = Settings()
    security = settings.security.model_copy(
        update={"allowed_paths": [str(root)], "rate_limiting_enabled": False}
    )
    update: Dict[str, Any] = {"security": security}
    if usd:
        update["usd"] = settings.usd.model_copy(update=usd)
    return SimulMCPServer(settings.model_copy(update=update), backends={"usd"})


def _call(server: SimulMCPServer, name: str, **arguments: Any) -> Dict[str, Any]:
    async def run() -> Dict[str, Any]:
        async with Client(server.mcp) as client:
            result = await client.call_tool(name, arguments, raise_on_error=False)
            return json.loads(result.content[0].text)

    return asyncio.run(run())


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path.resolve()


@pytest.mark.parametrize(
    ("builder", "name", "code"),
    [
        (fx.write_corrupt_usda, "corrupt.usda", "ParseError"),
        (fx.write_empty, "empty.usda", "EmptyFile"),
        (fx.write_garbage, "garbage.usdc", "CorruptFile"),
        (fx.write_text_named_usdc, "text.usdc", "CorruptFile"),
    ],
)
def test_validate_and_load_agree_on_unopenable_files(
    root: Path, builder: Any, name: str, code: str
) -> None:
    server = _server(root)
    path = str(builder(root / name))

    report = _call(server, "validate_usd_file", file_path=path)
    assert report["is_valid"] is False
    assert report["can_read"] is False
    assert report["issues"][0]["code"] == code
    assert report["error_count"] == 1

    loaded = _call(server, "load_usd_file", file_path=path)
    assert loaded["success"] is False
    assert loaded["error_type"] == "LoadError"
    assert loaded["details"]["reason"] == code
    assert "Failed to load USD file" not in loaded["error"]


def test_parse_error_location_reaches_the_client(root: Path) -> None:
    server = _server(root)
    path = str(fx.write_corrupt_usda(root / "corrupt.usda"))

    issue = _call(server, "validate_usd_file", file_path=path)["issues"][0]
    assert issue["location"] == {"layer": path, "line": 9, "column": 1}

    loaded = _call(server, "load_usd_file", file_path=path)
    assert "syntax error at line 9, column 1" in loaded["error"]
    assert loaded["details"]["line"] == 9


def test_validate_reports_content_problems(root: Path) -> None:
    server = _server(root)

    broken = _call(server, "validate_usd_file", file_path=str(fx.write_broken_refs(root)))
    assert (broken["is_valid"], broken["can_read"]) == (False, True)
    assert {issue["code"] for issue in broken["issues"]} == {
        "MissingSublayer",
        "MissingReference",
        "MissingPayload",
        "UnresolvedPrimPath",
    }

    physics = _call(
        server, "validate_usd_file", file_path=str(fx.write_bad_physics(root / "bad_physics.usda"))
    )
    assert physics["is_valid"] is False
    assert physics["error_count"] == 5
    assert all(issue["severity"] == "error" for issue in physics["issues"])

    clean = _call(server, "validate_usd_file", file_path=str(fx.write_robot(root / "robot.usda")))
    assert (clean["is_valid"], clean["can_read"], clean["issues"]) == (True, True, [])


def test_load_with_broken_references_succeeds_with_warnings(root: Path) -> None:
    server = _server(root)

    loaded = _call(server, "load_usd_file", file_path=str(fx.write_broken_refs(root)))

    assert loaded.get("error") is None
    assert loaded["stage_id"]
    codes = sorted(warning["code"] for warning in loaded["warnings"])
    assert codes == ["MissingPayload", "MissingReference", "MissingSublayer", "UnresolvedPrimPath"]
    assert {warning["severity"] for warning in loaded["warnings"]} == {"warning"}


def test_validate_uses_the_configured_extensions(root: Path) -> None:
    server = _server(root, allowed_extensions=[".usda"])
    path = str(fx.write_robot(root / "robot.usd"))

    report = _call(server, "validate_usd_file", file_path=path)
    loaded = _call(server, "load_usd_file", file_path=path)

    assert report["issues"][0]["code"] == "UnsupportedExtension"
    assert loaded["details"]["reason"] == "UnsupportedExtension"
