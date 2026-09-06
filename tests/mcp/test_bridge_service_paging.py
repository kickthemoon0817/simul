"""The bridge's typed prim listings page the same way the scripts do.

``list_prims`` and ``search_prims`` run inside Kit through the bridge
extension; the fake stage from the script tests stands in for ``omni.usd``
and ``pxr`` so the handlers' paging arithmetic runs here.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

extension_root = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "simul_mcp"
    / "bridge_ext"
    / "khemoo.simul.mcp"
)

from khemoo.simul.mcp.protocol import BridgeRequest  # noqa: E402
from khemoo.simul.mcp.service import BridgeCommandService  # noqa: E402
from tests.isaac.fake_usd import FakePrim, FakeStage, usd_modules  # noqa: E402


def _dispatch(stage: FakeStage, action: str, payload: Dict[str, Any]) -> Any:
    service = BridgeCommandService(executor=MagicMock(), allow_unsafe_execution=False)
    request = BridgeRequest(request_id="req-1", action=action, payload=payload)
    with pytest.MonkeyPatch.context() as patcher:
        for name, module in usd_modules(stage).items():
            patcher.setitem(sys.modules, name, module)
        return asyncio.run(service.dispatch(request))


def _stage(count: int) -> FakeStage:
    prims = [FakePrim("/World", "Xform")]
    prims.extend(FakePrim(f"/World/Mesh{i:03d}", "Mesh") for i in range(count))
    return FakeStage(prims)


def test_list_prims_pages_and_reports_the_applied_limit() -> None:
    response = _dispatch(
        _stage(7),
        "list_prims",
        {"root_path": "/World", "prim_type": "Mesh", "max_results": 3, "offset": 3},
    )
    assert response.status == "ok"
    payload = response.payload
    assert [prim["path"] for prim in payload["prims"]] == [
        "/World/Mesh003",
        "/World/Mesh004",
        "/World/Mesh005",
    ]
    assert payload["offset"] == 3
    assert payload["applied_limit"] == 3
    assert payload["truncated"] is True
    assert payload["next_offset"] == 6


def test_list_prims_honours_the_old_limit_name_and_clamps() -> None:
    response = _dispatch(_stage(2), "list_prims", {"max_items": 50_000})
    assert response.payload["applied_limit"] == 10_000

    response = _dispatch(
        _stage(4),
        "list_prims",
        {"max_items": 2, "root_path": "/World", "prim_type": "Mesh"},
    )
    assert response.payload["count"] == 2
    assert response.payload["truncated"] is True


def test_search_prims_pages_the_matches() -> None:
    response = _dispatch(
        _stage(5),
        "search_prims",
        {"query": "mesh", "search_type": "name", "max_results": 2, "offset": 4},
    )
    assert response.status == "ok"
    payload = response.payload
    assert [match["path"] for match in payload["matches"]] == ["/World/Mesh004"]
    assert payload["applied_limit"] == 2
    assert payload["truncated"] is False
    assert payload["next_offset"] is None


def test_search_prims_requires_a_query() -> None:
    response = _dispatch(_stage(1), "search_prims", {"search_type": "type"})
    assert response.status == "error"
    assert response.error is not None
    assert response.error.name == "InvalidRequest"

    response = _dispatch(_stage(1), "search_prims", {"query": "   "})
    assert response.status == "error"
