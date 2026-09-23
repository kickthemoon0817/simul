"""Build and install the optional native viewport overlay through Unreal setup."""

import hashlib
import json
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .unreal_setup import resolve_launch_argv


def install_agent_overlay(uproject: Path, engine_path: Path | None) -> dict[str, Any]:
    """Build bundled sources for this engine, then install without replacing user plugins."""
    source = Path(__file__).parents[1] / "resources/unreal/SimulAgentOverlay"
    if engine_path is None:
        executable = Path(resolve_launch_argv(uproject)[0]).resolve()
        engine = next((p for p in executable.parents if p.name == "Engine"), None)
        if engine is None:
            raise ValueError(
                "--agent-overlay requires --engine-path for this Unreal installation"
            )
        engine_path = engine.parent
    engine_path = engine_path.expanduser().resolve()
    system = platform.system()
    host = {"Darwin": "Mac", "Linux": "Linux", "Windows": "Win64"}.get(system)
    if host is None:
        raise ValueError(f"Agent overlay builds are unsupported on {system}")
    launcher = (
        engine_path
        / "Engine/Build/BatchFiles"
        / ("RunUAT.bat" if system == "Windows" else "RunUAT.sh")
    )
    if not launcher.is_file():
        raise FileNotFoundError(f"Unreal build tools not found: {launcher}")
    digest = hashlib.sha256(str(engine_path).encode())
    digest.update((engine_path / "Engine/Build/Build.version").read_bytes())
    for path in sorted(source.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(source)).encode())
            digest.update(path.read_bytes())
    fingerprint = digest.hexdigest()
    destination = uproject.parent / "Plugins/SimulAgentOverlay"
    receipt = destination / ".simul-build.json"
    if destination.exists():
        if (
            receipt.is_file()
            and json.loads(receipt.read_text()).get("fingerprint") == fingerprint
        ):
            return {"path": str(destination), "changed": False}
        raise ValueError(
            f"Existing overlay differs from these sources: {destination}. "
            "Close the editor and move that directory aside before rebuilding with --agent-overlay."
        )
    # The installed engine requires its C++ toolchain (Xcode/Visual Studio/clang).
    # Keep build output away from structured CLI stdout, and retain the log on failure.
    log_dir = uproject.parent / "Saved/Logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "SimulAgentOverlay-build.log"
    with tempfile.TemporaryDirectory(prefix="simul-unreal-overlay-") as temporary:
        root = Path(temporary)
        copied = root / "Source/SimulAgentOverlay"
        shutil.copytree(source, copied)
        package = root / "Package"
        command = [
            str(launcher),
            "BuildPlugin",
            f"-Plugin={copied / 'SimulAgentOverlay.uplugin'}",
            f"-Package={package}",
            f"-TargetPlatforms={host}",
            "-Rocket",
        ]
        with log_path.open("w") as log:
            result = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        if result.returncode:
            raise RuntimeError(
                f"Agent overlay build failed; check the C++ toolchain and {log_path}"
            )
        if (
            not (package / "SimulAgentOverlay.uplugin").is_file()
            or not (package / "Binaries").is_dir()
        ):
            raise RuntimeError(f"Unreal produced no overlay binaries; see {log_path}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Rename on the project's filesystem only once the entire package is ready.
        with tempfile.TemporaryDirectory(
            prefix=".simul-overlay-", dir=destination.parent
        ) as staging:
            staged = Path(staging) / "SimulAgentOverlay"
            shutil.copytree(package, staged)
            (staged / ".simul-build.json").write_text(
                json.dumps({"fingerprint": fingerprint})
            )
            staged.rename(destination)
    return {"path": str(destination), "changed": True, "build_log": str(log_path)}
