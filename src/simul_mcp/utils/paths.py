"""Filesystem sandbox policy, shared by every entry point.

The MCP registration layer, the CLI, and the tools layer all reach the same
filesystem operations. The policy lives here so the check sits *below* all of
them: a rule enforced in one caller is not a policy, it is a convention that the
next caller forgets.

Local paths are normalised before the containment test — expand ``$VARS``,
expand ``~``, resolve relative paths against the project root — and the result
must sit under an allowed root. Strings shaped like ``<scheme>://...`` are URLs,
never project-relative paths: ``file://`` URLs are converted to a local path and
checked like one, every other scheme is admitted only when it is on the scheme
allowlist for the requested access.

The project root is the source checkout when simul runs from one. A wheel
install has no such root: relative allowlist entries such as ``examples`` name
nothing there and are dropped, and relative paths passed to tools resolve
against the working directory instead.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import unquote, urlsplit
from urllib.request import url2pathname

from ..resources import find_checkout_root

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import Settings

_LOGGER = logging.getLogger(__name__)

# RFC 3986 scheme followed by an authority marker. A Windows drive letter
# (``C:\...``) has no ``//`` and so is not mistaken for a URL.
_URL_SCHEME = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*)://")

# Schemes a URL may carry to be read from (open, import, reference) when the
# caller configures nothing else. Nucleus assets live behind ``omniverse://``.
DEFAULT_ALLOWED_URL_SCHEMES: Tuple[str, ...] = ("omniverse",)

# Schemes a URL may carry to be written to. Writes stay local unless a caller
# opts a scheme in explicitly.
DEFAULT_ALLOWED_WRITE_URL_SCHEMES: Tuple[str, ...] = ()

# Sub-directory created under an allowed root to receive viewport captures.
CAPTURE_SUBDIR = "captures"


class SandboxDenied(PermissionError):
    """A path or URL was refused by the sandbox policy.

    Carries the same ``details`` mapping the MCP error envelope reports, so
    every layer that catches it can surface an actionable denial.
    """

    def __init__(self, path_str: str, details: Dict[str, Any]) -> None:
        super().__init__(f"File path is not allowed by sandbox policy: {path_str}")
        self.path_str: str = path_str
        self.details: Dict[str, Any] = details


class PathPolicy:
    """Decide whether a path may be opened, written, or referenced."""

    def __init__(
        self,
        *,
        enabled: bool,
        allowed_paths: Iterable[str],
        project_root: Optional[Path] = None,
        allowed_url_schemes: Iterable[str] = DEFAULT_ALLOWED_URL_SCHEMES,
        allowed_write_url_schemes: Iterable[str] = DEFAULT_ALLOWED_WRITE_URL_SCHEMES,
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
        self._allowed_url_schemes: Tuple[str, ...] = tuple(
            scheme.lower() for scheme in allowed_url_schemes
        )
        self._allowed_write_url_schemes: Tuple[str, ...] = tuple(
            scheme.lower() for scheme in allowed_write_url_schemes
        )

    @classmethod
    def from_settings(
        cls, settings: "Settings", project_root: Optional[Path] = None
    ) -> "PathPolicy":
        """Build the policy described by a Settings object.

        Args:
            settings: Settings whose ``security`` section configures the policy.
            project_root: Root that relative paths resolve against.

        Returns:
            The configured policy.
        """
        return cls(
            enabled=settings.security.sandbox_enabled,
            allowed_paths=settings.security.allowed_paths,
            project_root=project_root,
            allowed_url_schemes=settings.security.allowed_url_schemes,
            allowed_write_url_schemes=settings.security.allowed_write_url_schemes,
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def allowed_roots(self) -> List[Path]:
        return list(self._allowed_roots)

    @property
    def allowed_url_schemes(self) -> Tuple[str, ...]:
        """Schemes a URL may carry for read access."""
        return self._allowed_url_schemes

    @property
    def allowed_write_url_schemes(self) -> Tuple[str, ...]:
        """Schemes a URL may carry for write access."""
        return self._allowed_write_url_schemes

    def authorize(self, path_str: str, *, write: bool = False) -> str:
        """Check ``path_str`` and return the value a caller must pass onward.

        A local path (including a ``file://`` URL) comes back resolved and
        absolute; a permitted remote URL comes back verbatim. Passing the raw
        string onward instead is the bug this method exists to prevent: the
        checked path and the used path can name different files, because
        ``~``/``$VAR``/relative prefixes resolve here but are literal path
        components — or resolve against a different working directory — in the
        receiving application. With the sandbox disabled the raw string passes
        through untouched.

        Args:
            path_str: Path or URL supplied by the caller.
            write: Whether the caller intends to write to the location.

        Returns:
            The resolved local path or the verbatim URL.

        Raises:
            SandboxDenied: If the location is outside the sandbox.
        """
        if not self._enabled:
            return path_str
        if not path_str:
            # An empty path names nothing, so it cannot be shown to be inside
            # the sandbox.
            raise SandboxDenied(path_str, self.denial_details(path_str, write=write))

        scheme = self.url_scheme(path_str)
        if scheme is not None and scheme != "file":
            allowed = self._allowed_write_url_schemes if write else self._allowed_url_schemes
            if scheme in allowed:
                return path_str
            raise SandboxDenied(path_str, self.denial_details(path_str, write=write))

        try:
            candidate = self.resolve(path_str)
        except ValueError:
            # A file:// URL naming a remote host is not a local path.
            raise SandboxDenied(path_str, self.denial_details(path_str, write=write)) from None
        for allowed_root in self._allowed_roots:
            try:
                candidate.relative_to(allowed_root)
                return str(candidate)
            except ValueError:
                continue
        raise SandboxDenied(path_str, self.denial_details(path_str, write=write))

    def is_allowed(self, path_str: str, *, write: bool = False) -> bool:
        """Return True when ``path_str`` may be used for the requested access.

        Args:
            path_str: Path or URL supplied by the caller.
            write: Whether the caller intends to write to the location.

        Returns:
            Whether the policy admits the location.
        """
        try:
            self.authorize(path_str, write=write)
        except SandboxDenied:
            return False
        return True

    def resolve(self, path_str: str) -> Path:
        """Normalize a local path exactly the way the containment test sees it.

        Callers that pass a checked path onward must pass this value, not the
        raw string — otherwise the checked path and the used path can name
        different files (``~``/``$VAR``/relative prefixes resolve here but are
        literal path components to the receiving application). ``file://``
        URLs are converted to the local path they name.

        Args:
            path_str: Local path or ``file://`` URL.

        Returns:
            The absolute, symlink-resolved path.

        Raises:
            ValueError: If ``path_str`` is a URL with a non-``file`` scheme.
        """
        scheme = self.url_scheme(path_str)
        if scheme == "file":
            expanded = self._file_url_to_path(path_str)
        elif scheme is not None:
            raise ValueError(f"Not a filesystem path: {path_str}")
        else:
            expanded = os.path.expandvars(path_str)
        candidate = Path(expanded).expanduser()
        if not candidate.is_absolute():
            candidate = (self._project_root or Path.cwd()) / candidate
        try:
            return candidate.resolve()
        except Exception:
            return candidate.absolute()

    def denial_details(self, path_str: str, *, write: bool = False) -> Dict[str, Any]:
        """Build the ``details`` mapping reported with a sandbox denial.

        Args:
            path_str: The refused path or URL.
            write: Whether the refused access was a write.

        Returns:
            The refused path, the allowed roots and URL schemes, and a hint.
        """
        schemes = self._allowed_write_url_schemes if write else self._allowed_url_schemes
        access = "write" if write else "read"
        if schemes:
            hint = (
                f"Pass a path under one of allowed_roots or a URL whose scheme is in "
                f"allowed_url_schemes ({access} access). Widen the sandbox with "
                f"security.allowed_paths / security.allowed_url_schemes."
            )
        else:
            hint = (
                f"Pass a path under one of allowed_roots; no URL scheme is allowed for "
                f"{access} access. Widen the sandbox with security.allowed_paths / "
                f"security.allowed_write_url_schemes."
            )
        return {
            "file_path": path_str,
            "access": access,
            "allowed_roots": [str(root) for root in self._allowed_roots],
            "allowed_url_schemes": list(schemes),
            "hint": hint,
        }

    def writable_root(self) -> Optional[Path]:
        """Return the first allowed root this process can create files under.

        Roots inside the system temp directory are preferred: scratch output
        belongs in scratch space when the sandbox includes any. A root that does
        not exist yet counts as writable when its nearest existing ancestor is.

        Returns:
            The chosen root, or None when no allowed root is writable.
        """
        temp_root = Path(tempfile.gettempdir()).resolve()
        ordered = sorted(
            self._allowed_roots,
            key=lambda root: 0 if self._is_relative_to(root, temp_root) else 1,
        )
        for root in ordered:
            probe = root
            while not probe.exists() and probe.parent != probe:
                probe = probe.parent
            if probe.is_dir() and os.access(probe, os.W_OK):
                return root
        return None

    def default_capture_dir(self) -> Optional[str]:
        """Return the directory viewport captures land in when none is configured.

        Inside the sandbox this is ``<writable allowed root>/captures``. With
        the sandbox disabled it is ``<system temp>/simul_mcp/captures``.

        Returns:
            The capture directory, or None when the sandbox is enabled and no
            allowed root is writable.
        """
        if not self._enabled:
            return str(Path(tempfile.gettempdir()) / "simul_mcp" / CAPTURE_SUBDIR)
        root = self.writable_root()
        if root is None:
            return None
        return str(root / CAPTURE_SUBDIR)

    @staticmethod
    def url_scheme(path_str: str) -> Optional[str]:
        """Return the lower-cased URL scheme of ``path_str``, or None for a path.

        Args:
            path_str: Path or URL.

        Returns:
            The scheme when the string is shaped like ``<scheme>://...``.
        """
        match = _URL_SCHEME.match(path_str)
        if match is None:
            return None
        return match.group(1).lower()

    @staticmethod
    def _file_url_to_path(url: str) -> str:
        parts = urlsplit(url)
        if parts.netloc not in ("", "localhost"):
            raise ValueError(f"file URL names a remote host: {url}")
        return url2pathname(unquote(parts.path))

    @staticmethod
    def _is_relative_to(path: Path, ancestor: Path) -> bool:
        try:
            path.relative_to(ancestor)
        except ValueError:
            return False
        return True
