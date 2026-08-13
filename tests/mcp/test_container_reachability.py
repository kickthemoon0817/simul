"""Regressions for reaching the bridge inside a Docker container.

Two defects made the containerised path fail in ways that look like the sim is
simply unreachable:

* The compose file bound the bridge to ``127.0.0.1`` *inside* the container.
  Docker's published port forwards to the container's network interface, not to
  its loopback, so the connection is accepted by docker-proxy and then closed
  with no data — a silent hang rather than a refusal. Measured on this machine:
  container bound to ``127.0.0.1`` gives "connected, peer closed with NO DATA";
  bound to ``0.0.0.0`` returns the payload.
* The discovery file records the *bind* address. A wildcard bind therefore
  wrote ``0.0.0.0``, which the reader rejects because it only trusts loopback —
  so a container that was reachable still never appeared in discovery.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "src" / "simul_mcp" / "bridge_ext" / "khemoo.simul.mcp"))

from khemoo.simul.mcp.lifecycle import BridgeServerLifecycle  # noqa: E402

WILDCARDS = ["0.0.0.0", "::", ""]


@pytest.mark.parametrize("bind_host", WILDCARDS)
def test_wildcard_bind_advertises_a_connectable_address(
    bind_host: str, tmp_path: Path
) -> None:
    """A wildcard bind must not be written into the discovery file verbatim.

    "0.0.0.0" means "every interface", which includes loopback — but it is not
    an address a client can connect to, and the reader drops it for not being
    loopback. Advertise the loopback address the wildcard already covers.
    """
    lifecycle = BridgeServerLifecycle(host=bind_host, port=8229, request_handler=None)
    lifecycle._actual_port = 8229

    lifecycle.write_discovery_file(str(tmp_path), pid=4242, vscode_port=8226)

    written = json.loads((tmp_path / "simul-mcp-4242.json").read_text())
    assert written["host"] == "127.0.0.1", (
        f"bind {bind_host!r} advertised as {written['host']!r}, which discovery drops"
    )
    assert written["port"] == 8229
    assert written["vscode_port"] == 8226


def test_explicit_host_is_advertised_unchanged(tmp_path: Path) -> None:
    """A specific bind address is what clients should be told to use."""
    lifecycle = BridgeServerLifecycle(
        host="192.168.1.50", port=8229, request_handler=None
    )
    lifecycle._actual_port = 8229

    lifecycle.write_discovery_file(str(tmp_path), pid=99)

    written = json.loads((tmp_path / "simul-mcp-99.json").read_text())
    assert written["host"] == "192.168.1.50"


def test_compose_binds_the_container_on_all_interfaces() -> None:
    """Binding container-loopback makes the published port a silent hang."""
    compose = (repo_root / "compose.isaac-sim.yml").read_text()

    assert "ISAAC_BIND_HOST:-0.0.0.0" in compose, (
        "the in-container bind defaults to loopback, which Docker cannot forward to"
    )
    # The host side is what must stay loopback-only.
    assert '"127.0.0.1:${ISAAC_BRIDGE_PORT:-8229}' in compose
    assert '"127.0.0.1:${ISAAC_VSCODE_PORT:-8226}' in compose


def test_compose_does_not_force_a_uid_the_image_rejects() -> None:
    """The image keeps /isaac-sim mode 0750 root:root.

    Forcing uid 1000 makes the entrypoint fail with "Permission denied" (exit
    126) before Kit starts — verified against nvcr.io/nvidia/isaac-sim:5.1.0,
    where even the image's own isaac-sim user (1234) cannot traverse it.
    """
    compose = (repo_root / "compose.isaac-sim.yml").read_text()

    assert "ISAAC_SIM_UID:-1000" not in compose, (
        "forcing uid 1000 makes the container exit 126 before Kit starts"
    )
    assert "ISAAC_SIM_USER:-0:0" in compose
