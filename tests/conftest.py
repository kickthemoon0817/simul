"""Suite-wide setup: import paths, the shared FastMCP double and the Isaac platform.

The checkout's ``src`` goes first on ``sys.path`` so the tests exercise the
source they sit next to, whatever ``simul`` is installed in the
interpreter. The bridge extension root follows, for the tests that import the
Kit extension's modules directly, and the repository root last so test
modules can import ``tests.fakes``.

The server registers the Isaac Sim tools by default only on the platforms
Isaac Sim runs on (Linux, Windows). Every test sees such a platform, so the
suite behaves the same on a macOS dev machine as on Linux CI; the tests of
that gate (``tests/mcp/test_isaac_platform_gate.py``) set the platform they
need.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator, Type

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
_BRIDGE_EXT = _SRC / "simul" / "bridge_ext" / "khemoo.simul"

for _path in (_REPO, _BRIDGE_EXT, _SRC):
    if str(_path) in sys.path:
        sys.path.remove(str(_path))
    sys.path.insert(0, str(_path))

from tests.fakes import FakeFastMCP  # noqa: E402


@pytest.fixture(autouse=True, scope="session")
def isaac_platform_supported() -> Iterator[None]:
    """Report an OS Isaac Sim runs on and leave ``isaac_sim.enabled`` unset.

    Patches ``is_isaac_platform_supported`` on ``simul.mcp.backends``, where
    the server reads it, like the runtime probes the tests patch there.
    Session-scoped so module-scoped fixtures that build a server see it too;
    a test's own ``monkeypatch`` still overrides it for that test.
    """
    from simul.mcp import backends

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(
            backends, "is_isaac_platform_supported", lambda platform=None: True
        )
        patcher.delenv("ISAAC_SIM__ENABLED", raising=False)
        yield


@pytest.fixture
def fake_fastmcp(monkeypatch: pytest.MonkeyPatch) -> Iterator[Type[FakeFastMCP]]:
    """Have the server build a recording ``FakeFastMCP`` instead of a real FastMCP.

    Background task support is switched off with it, since the double does
    not run anything.

    Yields:
        The double's class; the instance the server built is ``server.mcp``.
    """
    from simul.mcp import server as server_module

    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)
    yield FakeFastMCP
