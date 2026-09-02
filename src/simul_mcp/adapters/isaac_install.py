"""
Isaac Sim install inspection: version detection and transport extension names.

Isaac Sim 6.0 moved the port-8226 Python socket server out of
``isaacsim.code_editor.vscode`` into a dedicated
``isaacsim.code_editor.python_server`` extension. The wire format is the same
JSON reply, but the extension name that must be enabled at launch differs, so
anything that launches or configures an Isaac install needs to know which
major version it is talking to.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PYTHON_SERVER_EXTENSION: str = "isaacsim.code_editor.python_server"
VSCODE_EXTENSION: str = "isaacsim.code_editor.vscode"
BRIDGE_EXTENSION: str = "khemoo.simul.mcp"

#: Carb settings namespaces that hold the Python socket server port, newest first.
PYTHON_SOCKET_PORT_SETTINGS: tuple[str, ...] = (
    f"/exts/{PYTHON_SERVER_EXTENSION}/port",
    f"/exts/{VSCODE_EXTENSION}/port",
)

_VERSION_PATTERN = re.compile(r"^\s*(\d+)\.(\d+)\.(\d+)")


@dataclass(frozen=True)
class IsaacVersion:
    """Semantic version of an Isaac Sim install, parsed from its ``VERSION`` file."""

    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    @property
    def python_transport_extension(self) -> str:
        """Extension that serves raw Python on the socket port for this version."""
        return PYTHON_SERVER_EXTENSION if self.major >= 6 else VSCODE_EXTENSION

    @classmethod
    def parse(cls, text: str) -> "IsaacVersion":
        """Parse the leading ``major.minor.patch`` of a VERSION file or version string.

        Args:
            text: Contents of ``<isaac-root>/VERSION``, e.g.
                ``6.0.1-rc.7+release.42383.32955d8d.gl``.

        Returns:
            The parsed version.

        Raises:
            ValueError: If no ``major.minor.patch`` prefix is present.
        """
        match = _VERSION_PATTERN.match(text)
        if match is None:
            raise ValueError(f"Cannot parse Isaac Sim version from {text!r}")
        return cls(int(match.group(1)), int(match.group(2)), int(match.group(3)))


def read_isaac_version(isaac_root: Path) -> Optional[IsaacVersion]:
    """Read the version of an Isaac Sim install from its ``VERSION`` file.

    Args:
        isaac_root: Install root, the directory holding ``isaac-sim.sh``.

    Returns:
        The version, or None when the file is missing or unparseable.
    """
    version_file = isaac_root / "VERSION"
    try:
        text = version_file.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        return IsaacVersion.parse(text)
    except ValueError:
        return None
