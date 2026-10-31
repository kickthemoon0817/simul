"""One script, one size: the script tools report ``code_bytes`` in UTF-8 bytes.

``execute_unreal_script`` logged a script's UTF-8 byte length while
``execute_isaac_script`` logged ``len(code)``, its characters, so a script
with Hangul or an emoji was logged with a different size depending on the
backend that ran it. Isaac's ``MAX_SCRIPT_BYTES`` limit counted characters as
well, despite its name and the "bytes" its refusal names. Both backends now
measure through ``utf8_byte_length``. ``execute_blender_script`` logs no
script size, so Blender has none to compare.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Type

import pytest

from simul.adapters.isaac_socket_client import ScriptResult
from simul.config import Settings
from simul.mcp import server as server_module
from simul.mcp.registration._helpers import utf8_byte_length
from simul.mcp.tools.isaac_tools import MAX_SCRIPT_BYTES
from simul.mcp.usage_tracker import ToolUsageTracker
from tests.fakes import AvailableAdapter, FakeFastMCP

# A Hangul comment (3 bytes per syllable), an emoji (4 bytes) and an accented
# letter (2 bytes): 26 characters, 40 bytes in UTF-8.
SCRIPT = "# 한국어 주석 🙂\nprint('héllo')\n"


@pytest.mark.parametrize(
    ("text", "characters", "size"),
    [
        ("print(1)", 8, 8),
        ("print('안녕하세요')", 14, 24),
        ("print('🙂')", 10, 13),
        (SCRIPT, 26, 40),
    ],
)
def test_utf8_byte_length_counts_encoded_bytes(
    text: str, characters: int, size: int
) -> None:
    assert len(text) == characters
    assert utf8_byte_length(text) == size == len(text.encode("utf-8"))


def test_utf8_byte_length_counts_a_lone_surrogate_instead_of_raising() -> None:
    """Strict UTF-8 cannot encode an unpaired surrogate; it counts as the 3 bytes surrogatepass writes."""
    assert utf8_byte_length("x = '\ud800'") == len("x = ''") + 3


class _IsaacClient:
    """Socket client double that runs every script successfully and keeps it."""

    address = "127.0.0.1:8226"
    bridge_enabled = False
    fallback_to_vscode = True
    timeout_seconds = 30.0

    def __init__(self) -> None:
        self.executed: List[str] = []

    async def execute(self, code: str) -> ScriptResult:
        self.executed.append(code)
        return ScriptResult(success=True, output="ok")


@pytest.fixture
def server(
    fake_fastmcp: Type[FakeFastMCP], tmp_path: Path
) -> server_module.SimulMCPServer:
    """A server with Isaac's and Unreal's script tools and a private usage log.

    Isaac's scripts reach a client double. Unreal's stub adapter fails each
    call when the envelope opens a session, after the size is recorded.
    """
    settings = Settings(
        security={"allow_script_execution": True, "rate_limiting_enabled": False}
    )
    instance = server_module.SimulMCPServer(settings, backends={"isaac", "unreal"})
    instance.usage_tracker = ToolUsageTracker(log_dir=tmp_path)
    client = _IsaacClient()
    instance._isaac_clients = {"default": client}  # type: ignore[dict-item]
    instance.client = client  # type: ignore[assignment]
    instance.unreal_adapter = AvailableAdapter(settings)
    return instance


def _call(
    instance: server_module.SimulMCPServer, tool: str, code: str
) -> Dict[str, Any]:
    """Call a registered script tool and return the JSON block it sends."""
    result = asyncio.run(instance.mcp.by_name[tool](code=code))
    payload: Dict[str, Any] = json.loads(result.content[-1].text)
    return payload


def _records(instance: server_module.SimulMCPServer, tool: str) -> List[Dict[str, Any]]:
    """The tool's usage records, oldest first."""
    return list(reversed(instance.usage_tracker.get_recent(tool_name=tool)))


def test_isaac_and_unreal_log_one_size_for_one_script(
    server: server_module.SimulMCPServer,
) -> None:
    _call(server, "execute_isaac_script", SCRIPT)
    _call(server, "execute_unreal_script", SCRIPT)

    assert server.client.executed == [SCRIPT]  # type: ignore[attr-defined]
    (isaac,) = _records(server, "execute_isaac_script")
    (unreal,) = _records(server, "execute_unreal_script")
    assert isaac["params"] == {"code_bytes": 40}
    assert unreal["params"] == {"code_bytes": 40, "mode": "ExecuteFile"}
    assert isaac["script_sha256"] == hashlib.sha256(SCRIPT.encode("utf-8")).hexdigest()


def test_a_refused_isaac_script_names_the_size_its_usage_record_keeps(
    server: server_module.SimulMCPServer,
) -> None:
    """The limit and the usage record measure in one unit, so the refusal and the log agree."""
    oversize = "#" * (MAX_SCRIPT_BYTES % 3 + 1) + "가" * (MAX_SCRIPT_BYTES // 3)
    size = MAX_SCRIPT_BYTES + 1
    assert len(oversize) < MAX_SCRIPT_BYTES < len(oversize.encode("utf-8")) == size

    payload = _call(server, "execute_isaac_script", oversize)

    assert payload["error_type"] == "PayloadTooLarge"
    assert f"({size} bytes" in payload["error"]
    assert server.client.executed == []  # type: ignore[attr-defined]
    (record,) = _records(server, "execute_isaac_script")
    assert record["params"] == {"code_bytes": size}
