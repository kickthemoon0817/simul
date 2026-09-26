"""
Unreal Remote Control auto-setup and launcher.

Pure stdlib helpers that:

1. Idempotently patch ``<Project>.uproject`` to enable the ``RemoteControl``
   and ``PythonScriptPlugin`` plugins.
2. Idempotently patch ``<Project>/Config/DefaultRemoteControl.ini`` so the
   Remote Control web server starts automatically and allows Python execution.
3. Resolve a platform-appropriate Unreal Editor launch command
   (macOS ``open -a`` / ``UnrealEditor.app``; Linux ``UnrealEditor``).
4. Spawn the editor detached so the CLI does not block on editor lifetime.

The health-check poll itself lives in the CLI and uses
``UnrealRuntimeSession.health_check()`` so we do not duplicate HTTP logic.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REMOTE_CONTROL_SECTION = "/Script/RemoteControlCommon.RemoteControlSettings"

REQUIRED_PLUGINS: Tuple[str, ...] = ("RemoteControl", "PythonScriptPlugin")

# Identifier we stamp on simul-managed passphrase entries. UE doesn't use
# this for matching (only the hashed Passphrase field is compared) — it's
# purely for human bookkeeping in the project settings UI. simul also uses
# it to find its own entries: rotating --passphrase drops every entry with
# this identifier before writing the new one, so the old secret stops working.
PASSPHRASE_IDENTIFIER = "simul"

# Where a bind cleared by a no-``--bind`` re-run lands. The HTTP key is
# deleted outright (UE's FHttpServerConfig default is ``localhost``); the
# WebSocket key is rewritten to this value because deleting it would fall back
# to URemoteControlSettings' default of ``0.0.0.0``.
LOOPBACK_BIND = "127.0.0.1"
WEBSOCKET_BIND_KEY = "RemoteControlWebsocketServerBindAddress"
HTTP_BIND_KEY = "DefaultBindAddress"


def is_loopback_bind(host: str) -> bool:
    """Return True when ``host`` is a loopback bind address.

    Anything else (including ``0.0.0.0``, ``::`` and UE's ``any`` keyword) is
    treated as network-exposed. Conservative on purpose — better to make the
    user pass --allow-public once than to silently expose remote Python
    execution.
    """
    if not host:
        return True
    h = host.strip().lower()
    return h in {"localhost", "::1"} or h.startswith("127.")


def _required_ini_values(
    port: int,
    *,
    bind: Optional[str] = None,
    websocket_port: Optional[int] = None,
    passphrase_md5: Optional[str] = None,
) -> Dict[str, str]:
    """Build the section-keyed values for ``DefaultRemoteControl.ini``.

    ``bind`` and ``websocket_port`` are written only when the caller supplies
    them — passing ``None`` leaves the corresponding setting untouched so we
    keep UE's default (loopback HTTP host, 30020 WebSocket port).

    When ``passphrase_md5`` is supplied, also pin
    ``bEnforcePassphraseForRemoteClients=True`` so the gate is explicitly
    on (UE's C++ default is True too, but writing it makes the operator's
    intent visible in the ini).
    """
    values = {
        "bAutoStartWebServer": "True",
        "bAutoStartWebSocketServer": "True",
        "RemoteControlHttpServerPort": str(port),
        "bRestrictServerAccess": "True",
        "bEnableRemotePythonExecution": "True",
        "bAllowConsoleCommandRemoteExecution": "True",
    }
    # NOTE: HTTP bind hostname does NOT live on URemoteControlSettings —
    # iter6 traced it to FHttpListenerConfig.BindAddress, populated from
    # GEngineIni's [HTTPServer.Listeners] DefaultBindAddress. The
    # `RemoteControlHttpServerHostname` key earlier patches wrote here was
    # a silent no-op (the field doesn't exist on the URemoteControlSettings
    # CDO). HTTP bind is now patched via `patch_default_engine_ini` below.
    # WebSocket bind, however, IS a real RemoteControlSettings field, so
    # we route the same `bind` value through it here for symmetry — users
    # who pass --bind 127.0.0.1 get loopback on both HTTP and WS.
    if bind is not None:
        values["RemoteControlWebsocketServerBindAddress"] = bind
    if websocket_port is not None:
        values["RemoteControlWebSocketServerPort"] = str(websocket_port)
    if passphrase_md5 is not None:
        values["bEnforcePassphraseForRemoteClients"] = "True"
    return values


HTTP_LISTENERS_SECTION = "HTTPServer.Listeners"


def _insert_ini_entries(lines: List[str], section: str, entries: List[str]) -> None:
    """Insert entries inside the target section, before the next header."""
    if not entries:
        return
    starts = [i for i, line in enumerate(lines) if line.strip() == f"[{section}]"]
    if not starts:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend([f"[{section}]", *entries])
        return
    start = starts[-1] + 1
    end = next(
        (i for i in range(start, len(lines)) if lines[i].strip().startswith("[")),
        len(lines),
    )
    lines[end:end] = entries


def _section_has_entry(lines: List[str], section: str, entry: str) -> bool:
    """Ignore matching entries that belong to an unrelated INI section."""
    inside = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("["):
            inside = stripped == f"[{section}]"
        elif inside and stripped == entry:
            return True
    return False


def _reset_public_bind(
    lines: List[str],
    section: str,
    key: str,
    replacement: Optional[str],
    result: "PatchResult",
) -> List[str]:
    """Drop or rewrite a non-loopback ``key`` inside ``section``.

    A loopback value is kept as is. A non-loopback value is replaced with
    ``replacement`` (recorded in ``result.updated``), or deleted when
    ``replacement`` is None (recorded in ``result.removed``), and a warning
    names the value that was cleared.
    """
    out: List[str] = []
    inside = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            inside = stripped == f"[{section}]"
        elif inside and "=" in stripped and not stripped.startswith(";"):
            name, _, value = stripped.partition("=")
            if name.strip() == key and not is_loopback_bind(value.strip()):
                if replacement is None:
                    result.removed.append(key)
                else:
                    out.append(f"{key}={replacement}")
                    result.updated.append(key)
                result.warnings.append(
                    f"{result.path}: cleared {key}={value.strip()} left by an earlier "
                    "--bind; pass --bind again (with --allow-public) to keep it public."
                )
                continue
        out.append(line)
    return out


def reset_default_engine_ini_bind(project_dir: Path) -> Optional["PatchResult"]:
    """Remove a non-loopback ``DefaultBindAddress`` from ``DefaultEngine.ini``.

    Used when setup runs without ``--bind``: UE's HTTP listener then falls
    back to its ``localhost`` default instead of a public bind an earlier run
    wrote. Returns None when ``Config/DefaultEngine.ini`` does not exist (the
    file is never created here).
    """
    ini_path = Path(project_dir) / "Config" / "DefaultEngine.ini"
    if not ini_path.is_file():
        return None
    result = PatchResult(path=ini_path)
    lines = ini_path.read_text(encoding="utf-8").splitlines()
    out_lines = _reset_public_bind(lines, HTTP_LISTENERS_SECTION, HTTP_BIND_KEY, None, result)
    if result.removed:
        result.changed = True
        ini_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return result


def patch_default_engine_ini(
    project_dir: Path,
    *,
    bind: str,
) -> "PatchResult":
    """Patch ``Config/DefaultEngine.ini`` so UE's HTTP server actually binds
    to ``bind``.

    UE's RemoteControl HTTP server delegates to ``FHttpServerModule``,
    whose listener reads ``[HTTPServer.Listeners] DefaultBindAddress``
    from the engine ini (``GEngineIni`` → ``DefaultEngine.ini``). The
    bind is *not* configurable via ``URemoteControlSettings``; iter6
    traced this directly to UE 5.3 source at
    ``HttpServerConfig.cpp:11-17`` and ``HttpListener.cpp:62-92``.

    Idempotent: only rewrites when the value differs. Touches only the
    ``[HTTPServer.Listeners]`` section; other sections preserved verbatim.
    """
    project_dir = Path(project_dir)
    config_dir = project_dir / "Config"
    config_dir.mkdir(parents=True, exist_ok=True)
    ini_path = config_dir / "DefaultEngine.ini"

    required = {"DefaultBindAddress": bind}
    result = PatchResult(path=ini_path)

    original_lines: List[str] = (
        ini_path.read_text(encoding="utf-8").splitlines() if ini_path.is_file() else []
    )

    out_lines, touched = _rewrite_ini_section(
        original_lines, HTTP_LISTENERS_SECTION, required, result
    )

    missing = [k for k in required if k not in touched]
    _insert_ini_entries(
        out_lines, HTTP_LISTENERS_SECTION, [f"{key}={required[key]}" for key in missing]
    )
    result.added.extend(missing)

    if result.added or result.updated:
        result.changed = True
        ini_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return result


def _passphrase_array_line(passphrase_md5: str) -> str:
    """Build the UE config-array entry for the simul-managed passphrase.

    UE's TArray<FRCPassphrase> Passphrases takes one ``+Passphrases=...``
    line per element. The Identifier is for human bookkeeping; only the
    Passphrase hash is checked at request time
    (WebRemoteControlInternalUtils::CheckPassphrase).
    """
    return (
        f'+Passphrases=(Identifier="{PASSPHRASE_IDENTIFIER}",'
        f'Passphrase="{passphrase_md5}")'
    )


@dataclass
class PatchResult:
    """Describes what a single-file patch did."""

    path: Path
    changed: bool = False
    added: List[str] = field(default_factory=list)
    updated: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    already_ok: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.changed:
            return f"{self.path}: already configured"
        parts: List[str] = []
        if self.added:
            parts.append(f"added {', '.join(self.added)}")
        if self.updated:
            parts.append(f"updated {', '.join(self.updated)}")
        if self.removed:
            parts.append(f"removed {', '.join(self.removed)}")
        return f"{self.path}: {'; '.join(parts) if parts else 'changed'}"


def git_visibility_warning(file_path: Path) -> Optional[str]:
    """Warn when a file holding a secret would travel with the project's git history.

    Args:
        file_path: The file about to receive a secret.

    Returns:
        A warning sentence when the file sits inside a git work tree and is
        either already tracked or not ignored (so ``git add`` would pick it up),
        ``None`` when git is unavailable, the file is outside any repository,
        or ``.gitignore`` already excludes it.
    """
    directory = file_path.parent

    def _git(*args: str) -> Optional[int]:
        try:
            completed = subprocess.run(
                ["git", "-C", str(directory), *args],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return completed.returncode

    if _git("rev-parse", "--is-inside-work-tree") != 0:
        return None
    if _git("ls-files", "--error-unmatch", str(file_path)) == 0:
        return (
            f"{file_path} is tracked by git; the passphrase digest written into it will be "
            "committed with the project. Move the passphrase out of version control or "
            "rotate it before pushing."
        )
    if _git("check-ignore", "-q", str(file_path)) == 1:
        return (
            f"{file_path} is inside a git repository and not ignored; a later `git add` "
            "would commit the passphrase digest. Add it to .gitignore or keep it out of "
            "the commit."
        )
    return None


@dataclass
class SetupResult:
    """Aggregate result of ensure_remote_control_config."""

    uproject: PatchResult
    ini: PatchResult
    # ``engine_ini`` is the PatchResult for [HTTPServer.Listeners]'s
    # DefaultBindAddress: written when --bind is set, checked for a stale
    # public bind otherwise. None when --bind was not supplied and
    # DefaultEngine.ini does not exist (it is never created then).
    engine_ini: Optional[PatchResult] = None

    @property
    def changed(self) -> bool:
        return (
            self.uproject.changed
            or self.ini.changed
            or (self.engine_ini is not None and self.engine_ini.changed)
        )


def patch_uproject(uproject_path: Path, *, agent_overlay: bool = False) -> PatchResult:
    """Enable RemoteControl and PythonScriptPlugin in ``.uproject``.

    Writes the file back only when something actually changed. Preserves
    existing keys and ordering. Plugin entries are matched case-sensitively
    by ``Name`` — matches UE5's own handling.
    """
    uproject_path = Path(uproject_path)
    if not uproject_path.is_file():
        raise FileNotFoundError(f".uproject not found: {uproject_path}")

    text = uproject_path.read_text(encoding="utf-8")
    data = json.loads(text)

    plugins = data.setdefault("Plugins", [])
    if not isinstance(plugins, list):
        raise ValueError(
            f"{uproject_path}: 'Plugins' is not a list (found {type(plugins).__name__})"
        )

    result = PatchResult(path=uproject_path)
    by_name: Dict[str, Dict] = {
        entry.get("Name"): entry
        for entry in plugins
        if isinstance(entry, dict) and entry.get("Name")
    }

    for name in REQUIRED_PLUGINS + (("SimulAgentOverlay",) if agent_overlay else ()):
        existing = by_name.get(name)
        if existing is None:
            plugins.append({"Name": name, "Enabled": True})
            result.added.append(name)
        elif existing.get("Enabled") is not True:
            existing["Enabled"] = True
            result.updated.append(name)
        else:
            result.already_ok.append(name)

    if result.added or result.updated:
        result.changed = True
        serialized = json.dumps(data, indent=2)
        # UE's own writer ends the file with a newline; match it.
        uproject_path.write_text(serialized + "\n", encoding="utf-8")
    return result


def patch_remote_control_ini(
    project_dir: Path,
    port: int = 30010,
    *,
    bind: Optional[str] = None,
    websocket_port: Optional[int] = None,
    passphrase_md5: Optional[str] = None,
    reset_public_bind: bool = False,
) -> PatchResult:
    """Ensure ``Config/DefaultRemoteControl.ini`` has the required settings.

    Idempotent: only rewrites when values are missing or different. Touches
    only keys inside the ``[/Script/RemoteControlCommon.RemoteControlSettings]``
    section; other sections and comments are preserved verbatim.

    ``bind`` writes ``RemoteControlWebsocketServerBindAddress`` (real
    UE field, controls WebSocket bind). The HTTP-side bind is NOT
    configured here — it lives in ``Config/DefaultEngine.ini`` under
    ``[HTTPServer.Listeners] DefaultBindAddress`` and is patched by
    :func:`patch_default_engine_ini`. ``websocket_port`` writes
    ``RemoteControlWebSocketServerPort``. Either left as ``None`` means
    the setting is not touched, preserving UE's default.

    ``reset_public_bind`` (only acts with ``bind=None``) rewrites a
    non-loopback ``RemoteControlWebsocketServerBindAddress`` left by an
    earlier ``--bind`` run to ``127.0.0.1`` and, without ``passphrase_md5``,
    drops simul-identified ``+Passphrases`` entries (UE checks them on
    loopback requests too, so they would lock out the default client).

    ``passphrase_md5`` (when set) writes a single simul-identified
    ``+Passphrases=(...)`` array entry under the same section AND pins
    ``bEnforcePassphraseForRemoteClients=True``. Idempotent — re-running
    with the same hash does not duplicate the line; a different hash
    replaces every earlier simul entry, so a rotated secret stops working.
    Entries with other identifiers are left alone. ENABLING THIS BLOCKS
    EVERY CLIENT (including simul-mcp itself) UNTIL THEY SEND THE
    ``Passphrase: <md5>`` HTTP HEADER on every Remote Control request.
    """
    project_dir = Path(project_dir)
    config_dir = project_dir / "Config"
    config_dir.mkdir(parents=True, exist_ok=True)
    ini_path = config_dir / "DefaultRemoteControl.ini"

    required = _required_ini_values(
        port,
        bind=bind,
        websocket_port=websocket_port,
        passphrase_md5=passphrase_md5,
    )
    result = PatchResult(path=ini_path)

    original_lines: List[str] = (
        ini_path.read_text(encoding="utf-8").splitlines() if ini_path.is_file() else []
    )

    out_lines, touched = _rewrite_ini_section(
        original_lines, REMOTE_CONTROL_SECTION, required, result
    )

    # Track keys we didn't encounter — they need to be appended.
    missing = [k for k in required if k not in touched]
    _insert_ini_entries(
        out_lines, REMOTE_CONTROL_SECTION, [f"{key}={required[key]}" for key in missing]
    )
    result.added.extend(missing)

    if bind is None and reset_public_bind:
        out_lines = _reset_public_bind(
            out_lines, REMOTE_CONTROL_SECTION, WEBSOCKET_BIND_KEY, LOOPBACK_BIND, result
        )
        if passphrase_md5 is None:
            # UE's PassphrasePreprocessor checks every request, loopback
            # included, so a simul passphrase left from an earlier public run
            # would 401 the default client. The CLI never writes a passphrase
            # without a public bind, so it goes with the bind.
            out_lines = _drop_stale_simul_passphrases(out_lines, None, result)

    # +Passphrases is a UE config-array entry, not a key=value setting, so
    # the standard rewriter doesn't touch it. Same hash on re-run is a no-op;
    # a different hash drops the earlier simul entries first — UE accepts any
    # matching entry, so leaving the old one would keep the old secret valid.
    if passphrase_md5 is not None:
        git_warning = git_visibility_warning(ini_path)
        if git_warning is not None:
            result.warnings.append(git_warning)
        passphrase_line = _passphrase_array_line(passphrase_md5)
        out_lines = _drop_stale_simul_passphrases(out_lines, passphrase_line, result)
        if not _section_has_entry(out_lines, REMOTE_CONTROL_SECTION, passphrase_line):
            _insert_ini_entries(out_lines, REMOTE_CONTROL_SECTION, [passphrase_line])
            result.added.append("Passphrases")

    if result.added or result.updated or result.removed:
        result.changed = True
        ini_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return result


def _drop_stale_simul_passphrases(
    lines: List[str], keep: Optional[str], result: PatchResult
) -> List[str]:
    """Remove simul-identified ``+Passphrases`` entries other than ``keep``.

    Only entries inside the Remote Control section whose Identifier is
    ``PASSPHRASE_IDENTIFIER`` are touched; a duplicate of ``keep`` beyond the
    first is dropped too. ``keep=None`` drops every simul entry.
    """
    prefix = f'+Passphrases=(Identifier="{PASSPHRASE_IDENTIFIER}",'
    out: List[str] = []
    inside = False
    kept = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            inside = stripped == f"[{REMOTE_CONTROL_SECTION}]"
        elif inside and stripped.replace(" ", "").startswith(prefix):
            if stripped == keep and not kept:
                kept = True
            else:
                result.removed.append("Passphrases")
                continue
        out.append(line)
    return out


def _rewrite_ini_section(
    lines: List[str],
    section: str,
    required: Dict[str, str],
    result: PatchResult,
) -> Tuple[List[str], set]:
    """Rewrite the target section's keys in-place; return (lines, keys_seen)."""
    out: List[str] = []
    in_target = False
    touched: set = set()
    header = f"[{section}]"

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_target = stripped == header
            out.append(line)
            continue

        if in_target and "=" in stripped and not stripped.startswith(";"):
            key, _, value = stripped.partition("=")
            key = key.strip()
            if key in required:
                desired = required[key]
                if value.strip() != desired:
                    out.append(f"{key}={desired}")
                    result.updated.append(key)
                else:
                    out.append(line)
                    result.already_ok.append(key)
                touched.add(key)
                continue

        out.append(line)

    return out, touched


def ensure_remote_control_config(
    uproject_path: Path,
    port: int = 30010,
    *,
    bind: Optional[str] = None,
    websocket_port: Optional[int] = None,
    passphrase_md5: Optional[str] = None,
    agent_overlay: bool = False,
    keep_public_bind: bool = False,
) -> SetupResult:
    """Run both patches; caller decides what to do with the result.

    See :func:`patch_remote_control_ini` for the meaning of ``bind``,
    ``websocket_port``, and ``passphrase_md5`` — all default to ``None``.

    With ``bind=None`` a non-loopback bind left on disk by an earlier
    ``--bind`` run is cleared (HTTP ``DefaultBindAddress`` removed, WebSocket
    bind reset to ``127.0.0.1``, simul passphrase entries dropped) so the
    editor really gets the loopback default, unless ``keep_public_bind`` is set (the CLI passes
    ``--allow-public`` here).
    """
    uproject_path = Path(uproject_path)
    u = patch_uproject(uproject_path, agent_overlay=agent_overlay)
    i = patch_remote_control_ini(
        uproject_path.parent,
        port=port,
        bind=bind,
        websocket_port=websocket_port,
        passphrase_md5=passphrase_md5,
        reset_public_bind=not keep_public_bind,
    )
    # HTTP bind address lives in DefaultEngine.ini, not RemoteControl —
    # see patch_default_engine_ini's docstring for the UE 5.x source
    # trace. Without --bind the file is only edited to remove a public
    # bind an earlier run wrote; it is never created.
    e: Optional[PatchResult] = None
    if bind is not None:
        e = patch_default_engine_ini(uproject_path.parent, bind=bind)
    elif not keep_public_bind:
        e = reset_default_engine_ini_bind(uproject_path.parent)
    return SetupResult(uproject=u, ini=i, engine_ini=e)


# ---------------------------------------------------------------------------
# Editor launch
# ---------------------------------------------------------------------------


class LauncherNotFound(RuntimeError):
    """Raised when no Unreal Editor executable can be located."""


# Flags that put UE into a no-window-but-still-rendering mode. ``-RenderOffScreen``
# keeps the render pipeline live (so HighResShot, scene capture, replicator etc.
# all work) but creates no visible window — and so no focus dependency, no
# OS-level focus stealing, and no GUI flicker.  The companion flags suppress
# splash, sound, prompts, and route logs to stdout for container/CI capture.
HEADLESS_FLAGS: Tuple[str, ...] = (
    "-RenderOffScreen",
    "-unattended",
    "-nopause",
    "-nosplash",
    "-nosound",
    "-stdout",
    "-FullStdOutLogOutput",
)


def _engine_version_key(entry: Path) -> Tuple[int, ...]:
    """Sort key for ``UE_<major>.<minor>`` install dirs by numeric version.

    A lexical sort puts ``UE_5.9`` above ``UE_5.10``. Names without a
    parseable version sort below every versioned install.
    """
    match = re.match(r"UE_(\d+(?:\.\d+)*)", entry.name)
    if match is None:
        return (-1,)
    return tuple(int(part) for part in match.group(1).split("."))


def _macos_engine_roots() -> Tuple[Path, ...]:
    """Directories the macOS auto-detection scans for ``UE_*`` installs."""
    return (
        Path("/Users/Shared/Epic Games"),  # Epic Launcher default on macOS
        Path("/Applications/Epic Games"),
        Path.home() / "Applications/Epic Games",
    )


def _macos_macos_binary(engine_path: Optional[Path]) -> Optional[Path]:
    """Return the UnrealEditor binary path on macOS, or None if not found."""
    if engine_path is not None:
        binary = (
            engine_path
            / "Engine/Binaries/Mac/UnrealEditor.app/Contents/MacOS/UnrealEditor"
        )
        return binary if binary.is_file() else None
    for root in _macos_engine_roots():
        if not root.is_dir():
            continue
        for entry in sorted(root.glob("UE_*"), key=_engine_version_key, reverse=True):
            binary = (
                entry
                / "Engine/Binaries/Mac/UnrealEditor.app/Contents/MacOS/UnrealEditor"
            )
            if binary.is_file():
                return binary
    return None


def _macos_launch_argv(
    uproject: Path,
    engine_path: Optional[Path],
    headless: bool = False,
) -> List[str]:
    """Return argv for launching UE on macOS.

    For interactive (GUI) launches, prefer LaunchServices ``open -a`` when
    available. For headless launches, always go through the binary directly —
    LaunchServices may apply window-management policy that fights the
    ``-RenderOffScreen`` flag, so we bypass it.
    """
    if engine_path is not None:
        binary = _macos_macos_binary(engine_path)
        if binary is None:
            raise LauncherNotFound(f"UnrealEditor not found under {engine_path}")
        return [str(binary), str(uproject), *(HEADLESS_FLAGS if headless else ())]

    if not headless and shutil.which("open"):
        # open -a is best-effort: if the app isn't registered, the command
        # fails fast and we fall through to the filesystem search below.
        for app_name in ("Unreal Editor", "UnrealEditor"):
            probe = subprocess.run(
                ["open", "-Ra", app_name],
                capture_output=True,
                text=True,
                check=False,
            )
            if probe.returncode == 0:
                return ["open", "-a", app_name, "--args", str(uproject)]

    binary = _macos_macos_binary(None)
    if binary is not None:
        return [str(binary), str(uproject), *(HEADLESS_FLAGS if headless else ())]

    raise LauncherNotFound(
        "Could not find UnrealEditor on macOS. Pass --engine-path pointing at "
        "your UE install (e.g. /Users/Shared/Epic\\ Games/UE_5.7)."
    )


def _linux_launch_argv(
    uproject: Path,
    engine_path: Optional[Path],
    headless: bool = False,
) -> List[str]:
    """Return argv for launching UE on Linux (Ubuntu)."""
    candidates: List[Path] = []
    if engine_path is not None:
        candidates.append(engine_path / "Engine" / "Binaries" / "Linux" / "UnrealEditor")
    env = os.environ.get("UE_ENGINE_PATH") or os.environ.get("UNREAL_ENGINE_PATH")
    if env:
        candidates.append(Path(env) / "Engine" / "Binaries" / "Linux" / "UnrealEditor")

    which = shutil.which("UnrealEditor")
    if which:
        candidates.append(Path(which))

    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return [str(candidate), str(uproject), *(HEADLESS_FLAGS if headless else ())]

    raise LauncherNotFound(
        "Could not find UnrealEditor on Linux. Pass --engine-path pointing at "
        "your UE install root (the directory that contains Engine/)."
    )


def resolve_launch_argv(
    uproject: Path,
    engine_path: Optional[Path] = None,
    headless: bool = False,
) -> List[str]:
    """Return argv to launch Unreal Editor with ``uproject`` on the current OS.

    When ``headless`` is True, append ``-RenderOffScreen`` and friends so UE
    runs without creating a visible window. The render pipeline still runs,
    so viewport capture / scene capture / replicator workflows continue to
    work — they just don't depend on the editor having OS-level focus.
    """
    uproject = Path(uproject)
    if not uproject.is_file():
        raise FileNotFoundError(f".uproject not found: {uproject}")

    system = platform.system()
    if system == "Darwin":
        return _macos_launch_argv(uproject, engine_path, headless=headless)
    if system == "Linux":
        return _linux_launch_argv(uproject, engine_path, headless=headless)
    raise LauncherNotFound(
        f"Unsupported platform for automated launch: {system}. "
        "Start the Unreal Editor manually and re-run `simul unreal setup --no-launch`."
    )


def launch_editor(
    uproject: Path,
    engine_path: Optional[Path] = None,
    headless: bool = False,
) -> subprocess.Popen:
    """Spawn the Unreal Editor detached from the current process group."""
    argv = resolve_launch_argv(uproject, engine_path, headless=headless)
    # start_new_session detaches from the CLI's session so Ctrl-C on the CLI
    # does not kill the editor, and so the editor keeps running after the
    # CLI process exits.
    return subprocess.Popen(
        argv,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
