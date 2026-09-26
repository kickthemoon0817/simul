"""Owner-only JSON records: discovery endpoints and editor attachments.

Imports only the Python standard library, because the Blender add-on bundles
this module and runs it inside Blender's interpreter.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

# Largest record ``BridgeFiles.read`` accepts by default. Matches the Blender
# bridge's message cap so an endpoint record can never exceed what its wire
# carries.
MAX_RECORD_BYTES = 32 * 1024 * 1024


class BridgeFiles:
    """Private discovery and attachment files shared by the editors and the CLI."""

    @staticmethod
    def write(path: Path, payload: dict[str, Any]) -> None:
        """Atomically publish a record readable only by its owner."""
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor, temporary = tempfile.mkstemp(prefix=".simul-", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w") as stream:
                json.dump(payload, stream)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def read(path: Path, max_bytes: int = MAX_RECORD_BYTES) -> dict[str, Any]:
        """Read a regular, owner-only JSON object without following symlinks.

        Args:
            path: The record to read.
            max_bytes: Largest record accepted; a longer file is truncated at
                ``max_bytes + 1`` and fails to parse.

        Returns:
            The decoded object.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            OSError: If ``path`` is a symlink (``O_NOFOLLOW``).
            PermissionError: If another user owns it or its mode is not 0600.
            ValueError: If it is not a regular file or not a JSON object.
        """
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor) as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"Not a regular bridge file: {path}")
            if hasattr(os, "getuid") and (
                metadata.st_uid != os.getuid() or metadata.st_mode & 0o077
            ):
                raise PermissionError(
                    f"Bridge file must be owned by this user and mode 0600: {path}"
                )
            data = json.loads(stream.read(max_bytes + 1))
        if not isinstance(data, dict):
            raise ValueError(f"Invalid bridge file: {path}")
        return data
