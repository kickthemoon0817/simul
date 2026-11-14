"""Whether a process still runs, and which process a pid names, from the standard library alone.

A pid alone does not name a process for long: once a process exits, the
system hands its pid to a later one. A record that keeps a pid (a timeout
hold under ``$SIMUL_HOME/holds``) therefore keeps the process's start time
too, from :func:`process_started`, and a reader that finds another start time
for the pid knows the process it names has exited.

Probes, by platform:

- Linux: ``/proc/<pid>/stat`` (the start time in clock ticks since boot).
- macOS: ``sysctl(KERN_PROC_PID)`` (``p_starttime``).
- Windows: ``OpenProcess`` with ``GetExitCodeProcess`` and
  ``GetProcessTimes``; ``os.kill(pid, 0)`` would end the process there.
- Elsewhere ``os.kill(pid, 0)`` says whether the pid exists, and no start
  time is known.

Every probe is best effort: one that fails reads as "cannot tell", which
callers treat as a process that may still run.
"""

from __future__ import annotations

import os
import struct
import sys
from typing import Any, Optional, Tuple

#: Size of the buffer macOS's ``struct kinfo_proc`` is read into (648 bytes
#: on 64-bit macOS); its first field is ``p_starttime``.
_KINFO_PROC_BUFFER = 1024


def pid_alive(pid: int) -> bool:
    """Whether a process with ``pid`` exists on this host; True when it cannot be told.

    Args:
        pid: The process id.

    Returns:
        False only when no such process exists.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":  # pragma: no cover - exercised on Windows only
        return _windows_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True  # EPERM: another user's process, which exists
    return True


def process_started(pid: int) -> Optional[str]:
    """An opaque token for when process ``pid`` started; None when it cannot be told.

    Two tokens for one pid are equal only when they describe the same
    process. A token is comparable only with another from this host.

    Args:
        pid: The process id.

    Returns:
        The token, or None (no such process, or no probe on this platform).
    """
    if pid <= 0:
        return None
    try:
        if sys.platform.startswith("linux"):
            return _linux_started(pid)
        if sys.platform == "darwin":
            return _darwin_started(pid)
        if sys.platform == "win32":  # pragma: no cover - exercised on Windows only
            return _windows_started(pid)
    except (OSError, ValueError, IndexError, AttributeError, struct.error):
        return None
    return None  # pragma: no cover - another platform


_OWN_START: Optional[str] = None


def own_start() -> Optional[str]:
    """This process's :func:`process_started` token, read once."""
    global _OWN_START
    if _OWN_START is None:
        _OWN_START = process_started(os.getpid())
    return _OWN_START


def _linux_started(pid: int) -> str:
    with open(f"/proc/{pid}/stat", "rb") as stream:
        data = stream.read()
    # Field 2 (the command) is in parentheses and may hold spaces; field 22,
    # the start time, is the 20th after it.
    return data[data.rindex(b")") + 2 :].split()[19].decode()


def _darwin_started(pid: int) -> Optional[str]:
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    # CTL_KERN, KERN_PROC, KERN_PROC_PID, pid
    mib = (ctypes.c_int * 4)(1, 14, 1, pid)
    buffer = ctypes.create_string_buffer(_KINFO_PROC_BUFFER)
    size = ctypes.c_size_t(_KINFO_PROC_BUFFER)
    if libc.sysctl(mib, 4, buffer, ctypes.byref(size), None, ctypes.c_size_t(0)) != 0:
        return None
    if size.value == 0:
        return None  # no such process
    seconds, microseconds = struct.unpack_from("@qi", buffer.raw, 0)
    return f"{seconds}.{microseconds:06d}"


def _windows_handle(pid: int) -> "Tuple[Any, Any]":  # pragma: no cover - Windows only
    import ctypes
    from ctypes import wintypes

    kernel32: Any = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    # PROCESS_QUERY_LIMITED_INFORMATION
    return kernel32, kernel32.OpenProcess(0x1000, False, pid)


def _windows_alive(pid: int) -> bool:  # pragma: no cover - Windows only
    try:
        import ctypes
        from ctypes import wintypes

        kernel32, handle = _windows_handle(pid)
        if not handle:
            # ERROR_INVALID_PARAMETER: no such process. Anything else (access
            # denied) means it exists.
            return bool(getattr(ctypes, "get_last_error")() != 87)
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        return True


def _windows_started(pid: int) -> Optional[str]:  # pragma: no cover - Windows only
    import ctypes
    from ctypes import wintypes

    kernel32, handle = _windows_handle(pid)
    if not handle:
        return None
    try:
        created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
        if not kernel32.GetProcessTimes(
            handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(kernel), ctypes.byref(user)
        ):
            return None
        return str((created.dwHighDateTime << 32) | created.dwLowDateTime)
    finally:
        kernel32.CloseHandle(handle)


__all__ = ["own_start", "pid_alive", "process_started"]
