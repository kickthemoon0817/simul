"""Through the real MCP stack, a layer cannot pull an outside file past the sandbox.

The audit's reproduction: ``ref_outside.usda`` inside the sandbox references
``../outside/secret.usda``, and ``get_prim_info /World/Leak`` returned the
outside file's ``custom:api_key``. ``load_usd_file`` and ``validate_usd_file``
now refuse it with a SandboxError that lists the offending path.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest
from fastmcp import Client

pytest.importorskip("pxr", reason="pxr library not available")

from pxr import Sdf, Usd  # noqa: E402

from simul.config import Settings  # noqa: E402
from simul.mcp.server import SimulMCPServer  # noqa: E402

SECRET = "sk-OUTSIDE-SANDBOX-SECRET"


@pytest.fixture
def layout(tmp_path: Path) -> Tuple[Path, Path]:
    inside = (tmp_path / "data").resolve()
    outside = (tmp_path / "outside").resolve()
    inside.mkdir()
    outside.mkdir()
    secret = Usd.Stage.CreateNew(str(outside / "secret.usda"))
    prim = secret.DefinePrim("/Secret", "Xform")
    secret.SetDefaultPrim(prim)
    prim.CreateAttribute("custom:api_key", Sdf.ValueTypeNames.String).Set(SECRET)
    secret.GetRootLayer().Save()

    ref = Usd.Stage.CreateNew(str(inside / "ref_outside.usda"))
    world = ref.DefinePrim("/World", "Xform")
    ref.SetDefaultPrim(world)
    ref.DefinePrim("/World/Leak", "Xform").GetReferences().AddReference("../outside/secret.usda")
    ref.GetRootLayer().Save()

    sub = Sdf.Layer.CreateNew(str(inside / "sublayer_outside.usda"))
    sub.subLayerPaths.append(str(outside / "secret.usda"))
    sub.Save()
    return inside, outside


def _server(root: Path) -> SimulMCPServer:
    settings = Settings()
    security = settings.security.model_copy(
        update={"allowed_paths": [str(root)], "rate_limiting_enabled": False}
    )
    return SimulMCPServer(settings.model_copy(update={"security": security}), backends={"usd"})


def _call(server: SimulMCPServer, name: str, **arguments: Any) -> Dict[str, Any]:
    async def run() -> Dict[str, Any]:
        async with Client(server.mcp) as client:
            result = await client.call_tool(name, arguments, raise_on_error=False)
            return json.loads(result.content[0].text)

    return asyncio.run(run())


@pytest.mark.parametrize("name", ["ref_outside.usda", "sublayer_outside.usda"])
@pytest.mark.parametrize("tool", ["load_usd_file", "validate_usd_file"])
def test_outside_composition_is_a_sandbox_error(
    layout: Tuple[Path, Path], name: str, tool: str
) -> None:
    inside, outside = layout
    server = _server(inside)

    payload = _call(server, tool, file_path=str(inside / name))

    assert payload["success"] is False
    assert payload["error_type"] == "SandboxError"
    assert str(outside / "secret.usda") in payload["error"]
    listed = payload["details"]["outside_sandbox"]
    assert [entry["location"] for entry in listed] == [str(outside / "secret.usda")]
    assert payload["details"]["allowed_roots"] == [str(inside)]
    assert SECRET not in json.dumps(payload)


def test_the_secret_never_reaches_get_prim_info(layout: Tuple[Path, Path]) -> None:
    inside, _ = layout
    server = _server(inside)

    loaded = _call(server, "load_usd_file", file_path=str(inside / "ref_outside.usda"))

    assert "stage_id" not in loaded
    # The id the old code handed out names no stage now.
    from simul.adapters.headless_usd import HeadlessUSDSession

    stage_id = HeadlessUSDSession()._generate_stage_id(str(inside / "ref_outside.usda"))
    info = _call(server, "get_prim_info", stage_id=stage_id, prim_path="/World/Leak")
    assert info["success"] is False
    assert SECRET not in json.dumps(info)


def test_widening_the_sandbox_admits_the_file(layout: Tuple[Path, Path]) -> None:
    inside, _ = layout
    server = _server(inside.parent)

    loaded = _call(server, "load_usd_file", file_path=str(inside / "ref_outside.usda"))
    info = _call(server, "get_prim_info", stage_id=loaded["stage_id"], prim_path="/World/Leak")

    assert info.get("success") is not False
    assert SECRET in json.dumps(info["attributes"]["custom:api_key"])
