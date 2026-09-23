"""Connect to an existing Blender window; never launch or replace it implicitly."""

from pathlib import Path
from typing import Optional

import typer

from ..adapters.blender_connection import BlenderAttachments, BlenderConnection
from ..config import get_settings
from .output import emit, emit_error

app = typer.Typer(
    name="blender",
    help="Discover and attach to existing Blender windows",
    add_completion=False,
)


@app.command("install-bridge")
def install_bridge(
    output: Path = typer.Option(
        Path("~/.simul/blender/simul_blender_bridge.zip"),
        "--output",
        help="Add-on ZIP destination",
    ),
) -> None:
    """Build the add-on ZIP to install and enable in an already running Blender."""
    try:
        path = BlenderAttachments.build_addon(output)
        emit(
            {
                "addon_zip": str(path),
                "next_step": (
                    "In the existing Blender window: Preferences > Add-ons > Install from Disk, "
                    "choose this ZIP, then enable Simul Blender Bridge. First enable needs no restart "
                    "or scene reload. When upgrading an already loaded bridge, restart Blender to reload "
                    "its modules. Then run simul blender instances."
                ),
            }
        )
    except (OSError, ValueError) as exc:
        emit_error(str(exc), type(exc).__name__)


@app.command("instances")
def instances() -> None:
    """List bridge-enabled processes, window IDs, files, scenes and unsaved state."""
    emit({"instances": BlenderAttachments(get_settings()).instances()})


@app.command("attach")
def attach(
    instance: Optional[str] = typer.Option(
        None,
        "--instance",
        help="Instance ID; required when several Blender processes exist",
    ),
    window: Optional[str] = typer.Option(
        None,
        "--window",
        help="Window ID; required when the process has several windows",
    ),
) -> None:
    """Attach only when one process/window is unambiguously selected."""
    try:
        emit(BlenderAttachments(get_settings()).attach(instance, window))
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        emit_error(str(exc), type(exc).__name__)


@app.command("status")
def status() -> None:
    """Verify the saved attachment against the running process and window."""
    try:
        emit(BlenderConnection(get_settings()).get_runtime_info())
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        emit_error(str(exc), type(exc).__name__)


@app.command("detach")
def detach() -> None:
    """Forget the selected window, leaving Blender and its scene untouched."""
    try:
        emit(BlenderAttachments(get_settings()).detach())
    except OSError as exc:
        emit_error(str(exc), type(exc).__name__)
