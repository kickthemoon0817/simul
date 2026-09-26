"""Importing the CLI must not load the server or the optional runtimes.

``import simul.cli.main`` took ~720 ms because ``simul.adapters``
eagerly imported pxr/bpy/aiohttp and the CLI imported the MCP server (fastmcp)
at module level — so ``simul --help`` paid for every backend. The adapters and
registration packages now resolve their exports lazily, and the CLI imports the
server only inside the commands that start or inspect it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"

_HEAVY = ("fastmcp", "pxr", "aiohttp", "bpy", "simul.mcp.server")


def _modules_after(statement: str) -> set:
    probe = (
        "import sys, json\n"
        f"sys.path.insert(0, {str(SRC)!r})\n"
        f"{statement}\n"
        "print(json.dumps(sorted(sys.modules)))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout
    return set(json.loads(out.strip().splitlines()[-1]))


def test_cli_import_skips_server_and_runtimes() -> None:
    loaded = _modules_after("import simul.cli.main")
    assert not {name for name in loaded if name.split(".")[0] in _HEAVY or name in _HEAVY}


def test_adapters_package_resolves_exports_on_access() -> None:
    loaded = _modules_after(
        "import simul.adapters as a\n"
        "assert a.IsaacSocketClient.__name__ == 'IsaacSocketClient'\n"
        "assert callable(a.is_unreal_available)"
    )
    assert "simul.adapters.isaac_socket_client" in loaded
    assert "simul.adapters.headless_usd" not in loaded
    assert "simul.adapters.blender_runtime" not in loaded
