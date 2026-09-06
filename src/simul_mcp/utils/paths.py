"""Filesystem sandbox policy, shared by every entry point.

The MCP registration layer, the CLI, and the tools layer all reach the same
filesystem operations. The policy lives here so the check sits *below* all of
them: a rule enforced in one caller is not a policy, it is a convention that the
next caller forgets.

Path resolution deliberately matches what the server did before this module
existed — expand ``$VARS``, expand ``~``, resolve relative paths against the
project root, then require the result to sit under an allowed root.

The project root is the source checkout when simul runs from one. A wheel
install has no such root: relative allowlist entries such as ``examples`` name
nothing there and are dropped, and relative paths passed to tools resolve
against the working directory instead.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional

from ..resources import find_checkout_root

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import Settings

_LOGGER = logging.getLogger(__name__)


class PathPolicy:
    """Decide whether a path may be opened, written, or referenced."""

    def __init__(
        self,
        *,
        enabled: bool,
        allowed_paths: Iterable[str],
        project_root: Optional[Path] = None,
    ) -> None:
        self._enabled = enabled
        self._project_root: Optional[Path] = project_root or find_checkout_root()
        self._allowed_roots: List[Path] = []
        for allowed_path in allowed_paths:
            if self._project_root is None and not Path(os.path.expandvars(allowed_path)).expanduser().is_absolute():
                _LOGGER.info(
                    "Dropping relative sandbox path %r: no source checkout to resolve it against",
                    allowed_path,
                )
                continue
            self._allowed_roots.append(self.resolve(allowed_path))

    @classmethod
    def from_settings(
        cls, settings: "Settings", project_root: Optional[Path] = None
    ) -> "PathPolicy":
        """Build the policy described by a Settings object."""
        return cls(
            enabled=settings.security.sandbox_enabled,
            allowed_paths=settings.security.allowed_paths,
            project_root=project_root,
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def allowed_roots(self) -> List[Path]:
        return list(self._allowed_roots)

    def resolve(self, path_str: str) -> Path:
        """Normalize a path exactly the way the containment test sees it.

        Callers that pass a checked path onward must pass this value, not the
        raw string — otherwise the checked path and the used path can name
        different files (``~``/``$VAR``/relative prefixes resolve here but are
        literal path components to the receiving application).
        """
        expanded = os.path.expandvars(path_str)
        candidate = Path(expanded).expanduser()
        if not candidate.is_absolute():
            candidate = (self._project_root or Path.cwd()) / candidate
        try:
            return candidate.resolve()
        except Exception:
            return candidate.absolute()

    def is_allowed(self, path_str: str) -> bool:
        """Return True when ``path_str`` sits under an allowed root.

        An empty path is refused: it names nothing, so it cannot be shown to be
        inside the sandbox.
        """
        if not self._enabled:
            return True
        if not path_str:
            return False
        candidate = self.resolve(path_str)
        for allowed_root in self._allowed_roots:
            try:
                candidate.relative_to(allowed_root)
                return True
            except ValueError:
                continue
        return False
