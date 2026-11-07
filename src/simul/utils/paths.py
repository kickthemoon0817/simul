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
import stat
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlsplit
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


def shared_directory_problem(path: Path) -> Optional[str]:
    """Explain why ``path`` cannot be trusted as this user's directory, or None.

    Only the components of ``path`` that sit inside a world-writable directory
    (``/tmp``, ``/var/tmp``) are judged, because any local user can create
    entries there: such a component must be owned by this user (or root) and,
    unless it is a symlink, writable by nobody else, and the world-writable
    directory must carry the sticky bit so nobody else can rename it away.
    Components that do not exist yet are not judged. Platforms without POSIX
    ownership (Windows) are never judged.

    Args:
        path: An absolute path, symlinks resolved or not; each component is
            inspected as written (``lstat``).

    Returns:
        A human-readable problem, or None when nothing is wrong.
    """
    getuid = getattr(os, "getuid", None)
    if getuid is None:
        return None
    uid = getuid()
    parent_shared = False
    parent_sticky = True
    for component in [*reversed(path.parents), path]:
        try:
            link = os.lstat(component)
        except FileNotFoundError:
            return None
        except OSError as exc:
            return f"cannot inspect {component}: {exc.strerror or exc}"
        if parent_shared:
            if not parent_sticky:
                return (
                    f"{component.parent} is world-writable without the sticky bit, so any "
                    f"local user can replace {component}"
                )
            if link.st_uid not in (uid, 0):
                return (
                    f"{component} sits in the world-writable {component.parent} and is owned "
                    f"by uid {link.st_uid}, not by this user (uid {uid})"
                )
            if not stat.S_ISLNK(link.st_mode) and link.st_mode & 0o022:
                return (
                    f"{component} sits in the world-writable {component.parent} and is "
                    f"writable by group or others (mode {stat.S_IMODE(link.st_mode):o})"
                )
        try:
            target = os.stat(component)
        except OSError:
            return None  # a dangling symlink: nothing exists there to trust or distrust
        parent_shared = stat.S_ISDIR(target.st_mode) and bool(target.st_mode & stat.S_IWOTH)
        parent_sticky = bool(target.st_mode & stat.S_ISVTX)
    return None


def _inside_world_writable_directory(path: Path) -> bool:
    """Return True when an existing ancestor of ``path`` is world-writable."""
    for ancestor in path.parents:
        try:
            mode = os.stat(ancestor).st_mode
        except OSError:
            continue
        if stat.S_ISDIR(mode) and mode & stat.S_IWOTH:
            return True
    return False


def _make_private_directory(path: Path) -> None:
    """Create ``path`` and any missing parents with mode 0700.

    ``os.makedirs`` applies its mode to the leaf only, so each missing
    component is created on its own. A component another process creates in
    between is left alone; ``shared_directory_problem`` judges it afterwards.
    """
    missing: List[Path] = []
    probe = path
    while not os.path.lexists(probe) and probe.parent != probe:
        missing.append(probe)
        probe = probe.parent
    for component in reversed(missing):
        try:
            os.mkdir(component, 0o700)
        except FileExistsError:
            continue
        except OSError as exc:
            _LOGGER.warning("Cannot create sandbox root %s: %s", component, exc)
            return


SANDBOX_DENIED_MESSAGE = "File path is not allowed by sandbox policy"


class SandboxDenied(PermissionError):
    """A path or URL was refused by the sandbox policy.

    Carries the same ``details`` mapping the MCP error envelope reports, so
    every layer that catches it can surface an actionable denial.
    ``error_message`` is the envelope's ``error``: the generic denial for a
    refused path, or a caller's own sentence when the path itself passed but
    something it draws on did not (a USD layer whose references leave the
    sandbox).
    """

    def __init__(
        self,
        path_str: str,
        details: Dict[str, Any],
        *,
        error_message: str = SANDBOX_DENIED_MESSAGE,
    ) -> None:
        if error_message == SANDBOX_DENIED_MESSAGE:
            super().__init__(f"File path is not allowed by sandbox policy: {path_str}")
        else:
            super().__init__(error_message)
        self.path_str: str = path_str
        self.details: Dict[str, Any] = details
        self.error_message: str = error_message


def sandbox_error(
    details: Dict[str, Any], *, error: str = SANDBOX_DENIED_MESSAGE
) -> Dict[str, Any]:
    """Build the ``SandboxError`` tool envelope every layer reports.

    The dict has the shape of ``ErrorResponse(...).model_dump()``; it is built
    by hand so this module keeps no dependency on the MCP schemas (the Blender
    add-on bundles it without pydantic).

    Args:
        details: The denial details, usually from ``PathPolicy.denial_details``
            or ``SandboxDenied.details``.
        error: The message; the default names a refused path.

    Returns:
        The error envelope with ``error_type`` ``SandboxError``.
    """
    return {
        "success": False,
        "error": error,
        "error_type": "SandboxError",
        "details": details,
    }


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
        # Roots inside a world-writable directory (the default /tmp/simul-work),
        # resolved root -> the path as configured. Another local user can create
        # or plant those, so each use re-checks them (see _root_problem).
        self._shared_roots: Dict[Path, Path] = {}
        for allowed_path in allowed_paths:
            if self._project_root is None and not Path(os.path.expandvars(allowed_path)).expanduser().is_absolute():
                _LOGGER.info(
                    "Dropping relative sandbox path %r: no source checkout to resolve it against",
                    allowed_path,
                )
                continue
            root = self.resolve(allowed_path)
            self._allowed_roots.append(root)
            lexical = self._lexical(allowed_path)
            if _inside_world_writable_directory(lexical) or _inside_world_writable_directory(root):
                self._shared_roots[root] = lexical
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
        authorized, untrusted = self._judge(path_str, write=write, root_problem=self._root_problem)
        if authorized is not None:
            return authorized
        details = self.denial_details(path_str, write=write)
        if untrusted is not None:
            root, problem = untrusted
            details["untrusted_root"] = str(root)
            details["reason"] = problem
            details["hint"] = (
                f"The allowed root {root} sits in a world-writable directory and another "
                f"local user could control it. Remove it (or fix its owner and mode: "
                f"chmod 700) or configure a private directory in security.allowed_paths."
            )
        raise SandboxDenied(path_str, details)

    def _judge(
        self,
        path_str: str,
        *,
        write: bool,
        root_problem: Callable[..., Optional[str]],
        resolve: Optional[Callable[[str], Path]] = None,
    ) -> Tuple[Optional[str], Optional[Tuple[Path, str]]]:
        """The verdict ``authorize`` reports, without building a denial.

        Args:
            path_str: Path or URL supplied by the caller.
            write: Whether the caller intends to write to the location.
            root_problem: Judges an allowed root's trust; ``_root_problem``,
                or a ``ReadCheck``'s memo of it.
            resolve: Turns a local path into the absolute, symlink-resolved
                path the containment test sees; ``resolve`` when None.

        Returns:
            The value to pass onward and None when the location is allowed;
            otherwise None and the untrusted root that held it, if one did.
        """
        if not path_str:
            # An empty path names nothing, so it cannot be shown to be inside
            # the sandbox.
            return None, None
        scheme = self.url_scheme(path_str)
        if scheme is not None and scheme != "file":
            allowed = self._allowed_write_url_schemes if write else self._allowed_url_schemes
            return (path_str if scheme in allowed else None), None
        try:
            candidate = (resolve or self.resolve)(path_str)
        except ValueError:
            # A file:// URL naming a remote host is not a local path.
            return None, None
        untrusted: Optional[Tuple[Path, str]] = None
        for allowed_root in self._allowed_roots:
            if not self._is_relative_to(candidate, allowed_root):
                continue
            problem = root_problem(allowed_root, create=write)
            if problem is None:
                return str(candidate), None
            untrusted = untrusted or (allowed_root, problem)
        return None, untrusted

    def read_check(self) -> "ReadCheck":
        """A read-access check for many paths in one operation, such as a USD layer walk.

        ``is_allowed`` re-resolves every path and re-judges a shared root's
        trust on every call, which is right for one path but costs a USD stack
        that names tens of thousands of asset paths several times its own load
        time. The check this returns judges each distinct path once, resolves
        each directory once, and judges each shared root's trust once, for its
        own life. Build one per operation and drop it after: a long-lived one
        would keep a verdict after the filesystem changed.

        Unlike ``is_allowed`` it takes an absolute path literally, without
        expanding ``~`` or ``$VARS``: such a path comes from a file (a USD
        layer's asset path), and the program that opens it does not expand
        them either, so expanding here would check a different file than the
        one opened.

        Returns:
            The check; call it with a path or URL.
        """
        return ReadCheck(self)

    def _root_problem(self, root: Path, *, create: bool) -> Optional[str]:
        """Explain why an allowed root cannot be used right now, or None.

        Only roots inside a world-writable directory are judged (see
        ``shared_directory_problem``), as configured and as resolved, so a
        symlink planted under the configured name is caught too. A missing one
        is created with mode 0700 first when ``create`` is set, before anyone
        else's process can create it with theirs.

        Args:
            root: One of ``allowed_roots``.
            create: Whether to create the root when it does not exist.

        Returns:
            The problem, or None when the root is usable.
        """
        lexical = self._shared_roots.get(root)
        if lexical is None:
            return None
        if create and not os.path.lexists(lexical):
            _make_private_directory(lexical)
        for candidate in (lexical, root):
            problem = shared_directory_problem(candidate)
            if problem is not None:
                _LOGGER.warning("Refusing sandbox root %s: %s", root, problem)
                return problem
        return None

    def _lexical(self, path_str: str) -> Path:
        """Return ``path_str`` made absolute as ``resolve`` does, without following symlinks."""
        candidate = Path(os.path.expandvars(path_str)).expanduser()
        if not candidate.is_absolute():
            candidate = (self._project_root or Path.cwd()) / candidate
        return Path(os.path.normpath(candidate))

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

    def denial(
        self, path_str: Optional[str], *, write: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Return the SandboxError envelope for ``path_str``, or None when allowed.

        ``None`` also means there is no path to police (an optional target
        that was not supplied).

        Args:
            path_str: Path or URL supplied by the caller, or None.
            write: Whether the caller intends to write to the location.

        Returns:
            The error envelope naming the allowed roots and URL schemes, or None.
        """
        if path_str is None or self.is_allowed(path_str, write=write):
            return None
        return sandbox_error(self.denial_details(path_str, write=write))

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
        A root inside a world-writable directory is created with mode 0700 when
        it is missing and skipped when another user could control it.

        Returns:
            The chosen root, or None when no allowed root is writable.
        """
        temp_root = Path(tempfile.gettempdir()).resolve()
        ordered = sorted(
            self._allowed_roots,
            key=lambda root: 0 if self._is_relative_to(root, temp_root) else 1,
        )
        for root in ordered:
            if self._root_problem(root, create=True) is not None:
                continue
            probe = root
            while not probe.exists() and probe.parent != probe:
                probe = probe.parent
            if probe.is_dir() and os.access(probe, os.W_OK):
                return root
        return None

    def default_capture_dir(self) -> Optional[str]:
        """Return the directory viewport captures land in when none is configured.

        Inside the sandbox this is ``<writable allowed root>/captures``. With
        the sandbox disabled it is ``<system temp>/simul/captures``.

        Returns:
            The capture directory, or None when the sandbox is enabled and no
            allowed root is writable.
        """
        if not self._enabled:
            return str(Path(tempfile.gettempdir()) / "simul" / CAPTURE_SUBDIR)
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
        # url2pathname already percent-decodes; decoding first as well would
        # turn a literal %25XX in the file name into a second escape.
        return url2pathname(parts.path)

    @staticmethod
    def _is_relative_to(path: Path, ancestor: Path) -> bool:
        if os.name == "posix" and path.is_absolute() and ancestor.is_absolute():
            # Both normalised absolute paths (resolve() output): a component
            # prefix test is what relative_to decides, at a fraction of the cost
            # a USD walk over tens of thousands of paths would pay.
            here, root = str(path), str(ancestor)
            return here == root or here.startswith(root if root.endswith("/") else root + "/")
        try:
            path.relative_to(ancestor)
        except ValueError:
            return False
        return True


class ReadCheck:
    """``PathPolicy.is_allowed`` for reads of file-named paths, memoised for one operation.

    Built by ``PathPolicy.read_check``. Verdicts are cached per path string,
    resolved directories per directory, and shared roots' trust per root, so
    a walk that names one texture from a thousand layers resolves it once and
    walks a shared root's ancestry once. An absolute path is taken literally
    (no ``~`` or ``$VARS`` expansion); a relative path or a ``file://`` URL is
    normalised as ``PathPolicy.resolve`` does.
    """

    def __init__(self, policy: PathPolicy) -> None:
        self._policy = policy
        self._verdicts: Dict[str, bool] = {}
        self._root_problems: Dict[Path, Optional[str]] = {}
        self._directories: Dict[str, str] = {}

    def __call__(self, path_str: str) -> bool:
        """Return whether ``path_str`` may be read."""
        if not self._policy.enabled:
            return True
        verdict = self._verdicts.get(path_str)
        if verdict is None:
            authorized, _ = self._policy._judge(
                path_str, write=False, root_problem=self._root_problem, resolve=self._resolve
            )
            verdict = self._verdicts[path_str] = authorized is not None
        return verdict

    def _resolve(self, path_str: str) -> Path:
        """``os.path.realpath`` with each directory resolved once.

        ``realpath(dir/leaf)`` is ``realpath(realpath(dir)/leaf)``, so only the
        leaf is looked at per path: it is followed when it is a symlink and
        appended as it is otherwise, as ``realpath`` does.
        """
        if not os.path.isabs(path_str):
            return self._policy.resolve(path_str)
        if any(part in (".", "..") for part in path_str.split(os.sep)):
            # ``..`` after a symlink means the link's parent: leave it to realpath.
            return Path(os.path.realpath(path_str))
        head, leaf = os.path.split(path_str.rstrip(os.sep) or os.sep)
        if not leaf:
            return Path(os.path.realpath(path_str))
        directory = self._directories.get(head)
        if directory is None:
            directory = self._directories[head] = os.path.realpath(head)
        candidate = os.path.join(directory, leaf)
        if os.path.islink(candidate):
            return Path(os.path.realpath(candidate))
        return Path(candidate)

    def _root_problem(self, root: Path, *, create: bool) -> Optional[str]:
        if root not in self._root_problems:
            self._root_problems[root] = self._policy._root_problem(root, create=create)
        return self._root_problems[root]
