"""Regression: a containerised bridge must leave a discovery file the host can read.

The Isaac Sim image keeps /isaac-sim mode 0750 root:root, so the container has
to run as root — the compose file's attempt to run it as uid 1000 fails at
exec with "Permission denied". Root then wrote the discovery file through
``tempfile.mkstemp``, which creates 0600, so the MCP server running as a normal
user on the host shared the volume and still could not read it. Measured against
a live container: ``cat /tmp/simul-mcp/simul-mcp-1.json`` -> Permission denied.

The file holds a pid, a host and two ports — nothing secret. The directory it
sits in stays 0700, so a single-user host is unchanged; the mode only matters
when the directory is deliberately shared, which is exactly the container case.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path


from khemoo.simul.mcp.lifecycle import BridgeServerLifecycle


def _write(tmp_path: Path) -> Path:
    lifecycle = BridgeServerLifecycle(host="0.0.0.0", port=8229, request_handler=None)
    lifecycle._actual_port = 8229
    lifecycle.write_discovery_file(str(tmp_path), pid=1, vscode_port=8226)
    return tmp_path / "simul-mcp-1.json"


def test_discovery_file_is_readable_by_other_users(tmp_path: Path) -> None:
    path = _write(tmp_path)
    mode = stat.S_IMODE(os.stat(path).st_mode)

    assert mode & stat.S_IROTH, (
        f"mode {mode:o}: a root container's file is unreadable by the host user"
    )
    assert not (mode & (stat.S_IWGRP | stat.S_IWOTH)), "must not be group/other writable"


def test_discovery_file_still_parses(tmp_path: Path) -> None:
    written = json.loads(_write(tmp_path).read_text())

    assert written["host"] == "127.0.0.1"
    assert written["port"] == 8229
    assert written["vscode_port"] == 8226
