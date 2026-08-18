"""Filesystem sandbox policy, shared by every entry point.

The MCP registration layer, the CLI, and the tools layer all reach the same
filesystem operations. The policy lives here so the check sits *below* all of
them: a rule enforced in one caller is not a policy, it is a convention that the
next caller forgets.

Path resolution deliberately matches what the server did before this module
existed — expand ``$VARS``, expand ``~``, resolve relative paths against the
project root, then require the result to sit under an allowed root.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import Settings

# src/simul_mcp/utils/paths.py -> repo root
_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[3]


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
        self._project_root = project_root or _DEFAULT_PROJECT_ROOT
        self._allowed_roots = [self.resolve(p) for p in allowed_paths]

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
            candidate = self._project_root / candidate
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
