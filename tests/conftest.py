"""Suite-wide setup: import paths and the shared FastMCP double.

The checkout's ``src`` goes first on ``sys.path`` so the tests exercise the
source they sit next to, whatever ``simul-mcp`` is installed in the
interpreter. The bridge extension root follows, for the tests that import the
Kit extension's modules directly, and the repository root last so test
modules can import ``tests.fakes``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator, Type

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
_BRIDGE_EXT = _SRC / "simul_mcp" / "bridge_ext" / "khemoo.simul.mcp"

for _path in (_REPO, _BRIDGE_EXT, _SRC):
    if str(_path) in sys.path:
        sys.path.remove(str(_path))
    sys.path.insert(0, str(_path))

from tests.fakes import FakeFastMCP  # noqa: E402


@pytest.fixture
def fake_fastmcp(monkeypatch: pytest.MonkeyPatch) -> Iterator[Type[FakeFastMCP]]:
    """Have the server build a recording ``FakeFastMCP`` instead of a real FastMCP.

    Background task support is switched off with it, since the double does
    not run anything.

    Yields:
        The double's class; the instance the server built is ``server.mcp``.
    """
    from simul_mcp.mcp import server as server_module

    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)
    yield FakeFastMCP
