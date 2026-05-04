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
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REMOTE_CONTROL_SECTION = "/Script/RemoteControlCommon.RemoteControlSettings"

REQUIRED_PLUGINS: Tuple[str, ...] = ("RemoteControl", "PythonScriptPlugin")

# Identifier we stamp on simul-managed passphrase entries. UE doesn't use
# this for matching (only the hashed Passphrase field is compared) — it's
# purely for human bookkeeping in the project settings UI and lets us
# write the same +Passphrases line on re-runs without duplicating.
PASSPHRASE_IDENTIFIER = "simul"


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
        ini_path.read_text(encoding="utf-8").splitlines()
        if ini_path.is_file()
        else []
    )

    out_lines, touched = _rewrite_ini_section(
        original_lines, HTTP_LISTENERS_SECTION, required, result
    )

    missing = [k for k in required if k not in touched]
    if missing:
        if not any(
            line.strip() == f"[{HTTP_LISTENERS_SECTION}]" for line in out_lines
        ):
            if out_lines and out_lines[-1].strip() != "":
                out_lines.append("")
            out_lines.append(f"[{HTTP_LISTENERS_SECTION}]")
        for key in missing:
            out_lines.append(f"{key}={required[key]}")
            result.added.append(key)

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
    already_ok: List[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.changed:
            return f"{self.path}: already configured"
        parts: List[str] = []
        if self.added:
            parts.append(f"added {', '.join(self.added)}")
        if self.updated:
            parts.append(f"updated {', '.join(self.updated)}")
        return f"{self.path}: {'; '.join(parts) if parts else 'changed'}"


@dataclass
class SetupResult:
    """Aggregate result of ensure_remote_control_config."""

    uproject: PatchResult
    ini: PatchResult
    # ``engine_ini`` is None when --bind was not supplied (we don't touch
    # DefaultEngine.ini in that case). When --bind is set, this is the
    # PatchResult for [HTTPServer.Listeners]'s DefaultBindAddress.
    engine_ini: Optional[PatchResult] = None

    @property
    def changed(self) -> bool:
        return (
            self.uproject.changed
            or self.ini.changed
            or (self.engine_ini is not None and self.engine_ini.changed)
        )


def patch_uproject(uproject_path: Path) -> PatchResult:
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

    for name in REQUIRED_PLUGINS:
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

    ``passphrase_md5`` (when set) appends a single ``+Passphrases=(...)``
    array entry under the same section AND pins
    ``bEnforcePassphraseForRemoteClients=True``. Idempotent — re-running
    with the same hash does not duplicate the line. ENABLING THIS BLOCKS
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
        ini_path.read_text(encoding="utf-8").splitlines()
        if ini_path.is_file()
        else []
    )

    out_lines, touched = _rewrite_ini_section(
        original_lines, REMOTE_CONTROL_SECTION, required, result
    )

    # Track keys we didn't encounter — they need to be appended.
    missing = [k for k in required if k not in touched]
    if missing:
        if not any(line.strip() == f"[{REMOTE_CONTROL_SECTION}]" for line in out_lines):
            if out_lines and out_lines[-1].strip() != "":
                out_lines.append("")
            out_lines.append(f"[{REMOTE_CONTROL_SECTION}]")
        for key in missing:
            out_lines.append(f"{key}={required[key]}")
            result.added.append(key)

    # +Passphrases is a UE config-array entry, not a key=value setting, so
    # the standard rewriter doesn't touch it. Append idempotently — same hash
    # on re-run is a no-op; a different hash adds an additional entry (UE
    # accepts any matching passphrase, so this is non-destructive).
    if passphrase_md5 is not None:
        passphrase_line = _passphrase_array_line(passphrase_md5)
        if passphrase_line not in (line.strip() for line in out_lines):
            if not any(line.strip() == f"[{REMOTE_CONTROL_SECTION}]" for line in out_lines):
                if out_lines and out_lines[-1].strip() != "":
                    out_lines.append("")
                out_lines.append(f"[{REMOTE_CONTROL_SECTION}]")
            out_lines.append(passphrase_line)
            result.added.append("Passphrases")

    if result.added or result.updated:
        result.changed = True
        ini_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return result


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
) -> SetupResult:
    """Run both patches; caller decides what to do with the result.

    See :func:`patch_remote_control_ini` for the meaning of ``bind``,
    ``websocket_port``, and ``passphrase_md5`` — all default to ``None``
    (untouched).
    """
    uproject_path = Path(uproject_path)
    u = patch_uproject(uproject_path)
    i = patch_remote_control_ini(
        uproject_path.parent,
        port=port,
        bind=bind,
        websocket_port=websocket_port,
        passphrase_md5=passphrase_md5,
    )
    # HTTP bind address lives in DefaultEngine.ini, not RemoteControl —
    # see patch_default_engine_ini's docstring for the UE 5.x source
    # trace. We only touch this file when --bind is supplied; otherwise
    # UE's default (loopback) stays in effect and Engine.ini is left
    # alone.
    e: Optional[PatchResult] = None
    if bind is not None:
        e = patch_default_engine_ini(uproject_path.parent, bind=bind)
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


def _macos_macos_binary(engine_path: Optional[Path]) -> Optional[Path]:
    """Return the UnrealEditor binary path on macOS, or None if not found."""
    if engine_path is not None:
        binary = (
            engine_path
            / "Engine/Binaries/Mac/UnrealEditor.app/Contents/MacOS/UnrealEditor"
        )
        return binary if binary.is_file() else None
    for root in (
        Path("/Users/Shared/Epic Games"),  # Epic Launcher default on macOS
        Path("/Applications/Epic Games"),
        Path.home() / "Applications/Epic Games",
    ):
        if not root.is_dir():
            continue
        for entry in sorted(root.glob("UE_*"), reverse=True):
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
