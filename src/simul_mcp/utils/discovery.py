"""
Trust checks and secret handling for the Isaac Sim discovery directory.

The discovery directory (``isaac_sim.discovery_dir``, ``/tmp/simul-mcp`` by
default) is where bridge extensions advertise their ports and where
``simul-mcp isaac launch --generate-auth-token`` leaves the python_server
token. Everything read from it is trusted, so the directory itself has to be
trustworthy: owned by the current user and writable by nobody else.
"""

from __future__ import annotations

import logging
import os
import re
import stat
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

AUTH_TOKEN_FILE_PREFIX = "auth-token-"
AUTH_TOKEN_LOG_PATTERN = re.compile(r"Python server authentication token:\s*(\S+)")


class DiscoveryDir:
    """
    One discovery directory: its trust check plus the auth-token files it holds.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        """Return the directory path."""
        return self._path

    def problem(self) -> Optional[str]:
        """
        Explain why the directory must not be trusted, or return ``None``.

        A directory another local user can write to lets them plant a discovery
        file naming a socket they control, and one owned by someone else was
        never ours to begin with. The owner check is skipped on Windows, where
        ``st_uid`` carries no meaning.

        Returns:
            A human-readable reason, or ``None`` when the directory is missing
            (nothing to trust yet) or passes both checks.
        """
        try:
            info = os.stat(self._path)
        except FileNotFoundError:
            return None
        except OSError as exc:
            return f"cannot stat {self._path}: {exc}"
        if not stat.S_ISDIR(info.st_mode):
            return f"{self._path} is not a directory"
        mode = stat.S_IMODE(info.st_mode)
        if mode & (stat.S_IWGRP | stat.S_IWOTH):
            return f"{self._path} is writable by other users (mode {mode:04o})"
        if sys.platform != "win32" and info.st_uid != os.getuid():
            return f"{self._path} is owned by uid {info.st_uid}, not the current uid {os.getuid()}"
        return None

    def ensure_private(self) -> Optional[str]:
        """
        Create the directory for our exclusive use and report what still stands in the way.

        A directory that already exists 0775 because an older tool created it
        with the default umask is tightened to 0700 when we own it; one owned by
        someone else is left alone and reported.

        Returns:
            The remaining problem, or ``None`` when the directory is ours alone.
        """
        self._path.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            info = os.stat(self._path)
        except OSError as exc:
            return f"cannot stat {self._path}: {exc}"
        owned = sys.platform == "win32" or info.st_uid == os.getuid()
        if owned and stat.S_IMODE(info.st_mode) & (stat.S_IRWXG | stat.S_IRWXO):
            try:
                os.chmod(self._path, 0o700)
            except OSError as exc:
                return f"cannot make {self._path} private: {exc}"
        return self.problem()

    def write_auth_token(self, pid: int, token: str) -> Path:
        """
        Store a generated python_server token for the Isaac process ``pid``.

        Args:
            pid: Process id of the Isaac Sim editor the token belongs to.
            token: The token as printed by ``isaacsim.code_editor.python_server``.

        Returns:
            Path of the 0600 token file.
        """
        self._path.mkdir(mode=0o700, parents=True, exist_ok=True)
        token_file = self._path / f"{AUTH_TOKEN_FILE_PREFIX}{pid}"
        fd = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(token + "\n")
        os.chmod(token_file, 0o600)
        return token_file

    def read_auth_token(self) -> Optional[str]:
        """
        Return the token of the most recently launched, still running Isaac editor.

        Token files whose process has exited are ignored, so a stale file never
        shadows a fresh one. Nothing is read when :meth:`problem` reports the
        directory as untrustworthy.

        Returns:
            The token string, or ``None`` when no usable token file exists.
        """
        if not self._path.is_dir():
            return None
        problem = self.problem()
        if problem is not None:
            logger.warning("Ignoring auth token files: %s", problem)
            return None
        newest: Optional[Path] = None
        newest_mtime = float("-inf")
        for token_file in self._path.glob(f"{AUTH_TOKEN_FILE_PREFIX}*"):
            pid = self._pid_from_name(token_file.name)
            if pid is None or not self._pid_alive(pid):
                continue
            try:
                mtime = token_file.stat().st_mtime
            except OSError:
                continue
            if mtime > newest_mtime:
                newest, newest_mtime = token_file, mtime
        if newest is None:
            return None
        try:
            token = newest.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.warning("Cannot read auth token file %s: %s", newest, exc)
            return None
        return token or None

    def prune_stale_auth_tokens(self) -> int:
        """
        Delete token files whose Isaac process is gone.

        Returns:
            Number of files removed.
        """
        if not self._path.is_dir():
            return 0
        removed = 0
        for token_file in self._path.glob(f"{AUTH_TOKEN_FILE_PREFIX}*"):
            pid = self._pid_from_name(token_file.name)
            if pid is not None and self._pid_alive(pid):
                continue
            try:
                token_file.unlink()
                removed += 1
            except OSError:
                continue
        return removed

    @staticmethod
    def token_from_launch_log(log_path: Path) -> Optional[str]:
        """
        Extract the token ``isaacsim.code_editor.python_server`` prints at startup.

        With ``require_auth=true`` and an empty ``auth_token`` setting the
        extension generates a token and prints
        ``Python server authentication token: <token>`` to stdout
        (``isaacsim/code_editor/python_server/extension.py``,
        ``_get_or_create_auth_token``). The launch command redirects stdout to
        the launch log, so that line is where the token can be read back.

        Args:
            log_path: The editor's launch log.

        Returns:
            The token, or ``None`` when the line has not appeared yet.
        """
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        match = AUTH_TOKEN_LOG_PATTERN.search(text)
        return match.group(1) if match else None

    @staticmethod
    def _pid_from_name(filename: str) -> Optional[int]:
        try:
            return int(filename[len(AUTH_TOKEN_FILE_PREFIX):])
        except ValueError:
            return None

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except OSError:
            return True
        return True
