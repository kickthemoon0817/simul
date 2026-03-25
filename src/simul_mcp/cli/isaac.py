"""
Isaac Sim CLI sub-commands.

Provides direct terminal access to a running Isaac Sim instance by
reusing the IsaacTools class -- the same methods that back the MCP tools.
No inline script construction; all Python-over-TCP logic lives in one place.

All commands support ``--json`` (or auto-detect via TTY) for structured
JSON output on stdout, making them consumable by AI agents.
"""

import asyncio
import base64
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from simul_mcp.adapters import IsaacSocketClient
from simul_mcp.cli.output import emit, emit_error, is_json_mode
from simul_mcp.config import get_settings
from simul_mcp.mcp.tools.isaac_tools import IsaacTools

app = typer.Typer(
    name="isaac",
    help="Isaac Sim commands -- interact with a running Isaac Sim instance.",
    add_completion=False,
)
console = Console(stderr=True)


def _tools(host: Optional[str] = None, port: Optional[int] = None, timeout: float = 30.0) -> IsaacTools:
    """Build an IsaacTools instance from settings with optional overrides."""
    settings = get_settings()
    client = IsaacSocketClient(
        host=host or settings.isaac_sim.socket_host,
        port=port or settings.isaac_sim.socket_port,
        timeout_seconds=timeout,
    )
    return IsaacTools(client, settings)


def _run(coro: Any) -> Dict[str, Any]:
    """
    Run an async IsaacTools method and handle errors uniformly.

    In JSON mode: emits the full result dict to stdout.
    In Rich mode: prints errors with Rich formatting.
    Always returns the result dict for further processing.
    """
    result = asyncio.run(coro)
    if result.get("error"):
        if is_json_mode():
            emit_error(
                result["error"],
                result.get("error_type", "Error"),
                result.get("details"),
            )
            return result  # unreachable (emit_error raises), but documents intent
        console.print(f"[red]{result.get('error_type', 'Error')}: {result['error']}[/red]")
        details = result.get("details")
        if details and details.get("traceback"):
            console.print(Panel(details["traceback"], title="Traceback", border_style="red"))
        raise typer.Exit(1)
    return result


# Common host/port/timeout options
_host_opt = typer.Option(None, "--host", "-H", help="Isaac Sim host")
_port_opt = typer.Option(None, "--port", "-p", help="Isaac Sim port")
_timeout_opt = typer.Option(30.0, "--timeout", "-t", help="Timeout in seconds")


# ---------------------------------------------------------------------------
# ping
# ---------------------------------------------------------------------------
@app.command()
def ping(
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
    timeout: float = typer.Option(5.0, "--timeout", "-t", help="Timeout in seconds"),
) -> None:
    """Check connectivity to Isaac Sim."""
    tools = _tools(host, port, timeout)
    reachable = asyncio.run(tools._client.ping())
    if is_json_mode():
        emit({"reachable": reachable, "address": tools._client.address})
        if not reachable:
            raise typer.Exit(1)
        return
    if reachable:
        console.print(f"[green]Pong[/green] from {tools._client.address}")
    else:
        console.print(f"[red]No response[/red] from {tools._client.address}")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------
@app.command()
def status(
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Show Isaac Sim instance status -- stage, prims, simulation state."""
    tools = _tools(host, port)
    stage = _run(tools.get_isaac_stage_info())
    sim = _run(tools.get_isaac_simulation_state())

    if is_json_mode():
        emit({"stage": stage, "simulation": sim})
        return

    sim_state = "playing" if sim.get("is_playing") else ("stopped" if sim.get("is_stopped") else "paused")
    table = Table(title=f"Isaac Sim @ {tools._client.address}")
    table.add_column("Property", style="cyan")
    table.add_column("Value")
    table.add_row("Stage URL", stage.get("stage_url", "(unsaved)") or "(unsaved)")
    table.add_row("Up Axis", stage.get("up_axis", "?"))
    table.add_row("Meters / Unit", str(stage.get("meters_per_unit", "?")))
    table.add_row("Prim Count", f"{stage.get('prim_count', 0):,}")
    table.add_row("Simulation", sim_state)
    table.add_row("Current Time", f"{sim.get('current_time', 0):.3f} s")
    table.add_row("FPS", str(sim.get("fps", "?")))
    console.print(table)


# ---------------------------------------------------------------------------
# scene
# ---------------------------------------------------------------------------
@app.command()
def scene(
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Print a scene overview -- prim types, lights, cameras, physics objects."""
    tools = _tools(host, port)
    summary = _run(tools.get_isaac_scene_summary())
    stats = _run(tools.get_isaac_scene_stats())
    cameras = _run(tools.list_isaac_cameras())
    lights = _run(tools.list_isaac_lights())
    physics = _run(tools.list_isaac_physics_objects())

    if is_json_mode():
        emit({
            "summary": summary,
            "stats": stats,
            "cameras": cameras,
            "lights": lights,
            "physics": physics,
        })
        return

    type_counts = stats.get("type_counts") or summary.get("type_counts", {})
    total = stats.get("total_prims") or summary.get("total_prims", 0)
    type_table = Table(title=f"Scene Overview ({total:,} prims)")
    type_table.add_column("Prim Type", style="cyan")
    type_table.add_column("Count", justify="right")
    for prim_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        if prim_type:
            type_table.add_row(prim_type, str(count))
    console.print(type_table)

    cam_list = cameras.get("cameras", [])
    if cam_list:
        console.print(f"\n[bold]Cameras[/bold] ({len(cam_list)})")
        for cam in cam_list:
            path = cam if isinstance(cam, str) else cam.get("path", str(cam))
            console.print(f"  {path}")

    light_list = lights.get("lights", [])
    if light_list:
        console.print(f"\n[bold]Lights[/bold] ({len(light_list)})")
        for light in light_list:
            if isinstance(light, dict):
                console.print(f"  {light.get('path', '?')}  [dim]({light.get('type', '?')})[/dim]")
            else:
                console.print(f"  {light}")

    phys_list = physics.get("rigid_bodies", []) + physics.get("colliders", [])
    if phys_list:
        console.print(f"\n[bold]Physics Objects[/bold] ({len(phys_list)})")
        for obj in phys_list:
            if isinstance(obj, dict):
                console.print(f"  {obj.get('path', '?')}  [dim]({obj.get('type', '?')})[/dim]")
            else:
                console.print(f"  {obj}")


# ---------------------------------------------------------------------------
# exec
# ---------------------------------------------------------------------------
@app.command("exec")
def exec_script(
    script: Optional[str] = typer.Argument(None, help="Python code string or path to a .py file"),
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
    timeout: float = _timeout_opt,
    raw: bool = typer.Option(False, "--raw", "-r", help="Print raw output without formatting"),
) -> None:
    """Execute Python code inside Isaac Sim. Accepts a code string or .py file path."""
    if script is None:
        if not sys.stdin.isatty():
            code = sys.stdin.read()
        else:
            if is_json_mode():
                emit_error("Provide a script string, .py file path, or pipe code via stdin.", "InputError")
            console.print("[red]Provide a script string, .py file path, or pipe code via stdin.[/red]")
            raise typer.Exit(1)
    elif Path(script).is_file() and script.endswith(".py"):
        code = Path(script).read_text(encoding="utf-8")
    else:
        code = script

    tools = _tools(host, port, timeout)
    try:
        result = asyncio.run(tools._client.execute(code))
    except (ConnectionRefusedError, TimeoutError) as exc:
        if is_json_mode():
            emit_error(str(exc), type(exc).__name__)
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    if is_json_mode() and not raw:
        output_data: Dict[str, Any] = {"success": result.success, "output": result.output}
        if not result.success:
            output_data["error_name"] = result.error_name
            output_data["error_value"] = result.error_value
            output_data["traceback"] = result.traceback
        # Try to parse output as JSON for structured embedding
        try:
            output_data["parsed"] = json.loads(result.output)
        except (json.JSONDecodeError, ValueError):
            pass
        print(json.dumps(output_data))
        if not result.success:
            raise typer.Exit(1)
        return

    if raw:
        print(result.output)
        if not result.success:
            print(result.error_value, file=sys.stderr)
            raise typer.Exit(1)
        return

    if result.success:
        try:
            parsed = json.loads(result.output)
            formatted = json.dumps(parsed, indent=2)
            console.print(Syntax(formatted, "json", theme="monokai"))
        except (json.JSONDecodeError, ValueError):
            console.print(result.output)
    else:
        console.print(f"[red]Error:[/red] {result.error_name}: {result.error_value}")
        if result.traceback:
            console.print(Panel(result.traceback, title="Traceback", border_style="red"))
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------
@app.command()
def capture(
    output: Path = typer.Argument("capture.png", help="Output file path"),
    width: int = typer.Option(1920, "--width", "-W", help="Capture width in pixels"),
    height: int = typer.Option(1080, "--height", help="Capture height in pixels"),
    eye: Optional[str] = typer.Option(None, "--eye", help="Camera eye position as x,y,z"),
    target: Optional[str] = typer.Option(None, "--target", help="Camera target position as x,y,z"),
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Capture the Isaac Sim viewport to an image file."""
    tools = _tools(host, port)

    if eye or target:
        try:
            eye_list = [float(v) for v in (eye or "5,5,3").split(",")]
            target_list = [float(v) for v in (target or "0,0,0").split(",")]
        except ValueError as e:
            if is_json_mode():
                emit_error(f"Invalid numeric value in camera args: {e}", "ValueError")
            console.print(f"[red]Invalid numeric value in camera args: {e}[/red]")
            raise typer.Exit(1)
        _run(tools.set_isaac_camera(position=eye_list, target=target_list))

    data = _run(tools.capture_isaac_viewport(width=width, height=height))

    # Decode base64 image and write to disk
    image_b64 = data.get("image_base64") or data.get("image", "")
    if image_b64:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(base64.b64decode(image_b64))
    else:
        if is_json_mode():
            emit_error("Viewport capture returned no image data", "CaptureError")
        console.print("[red]Viewport capture returned no image data[/red]")
        raise typer.Exit(1)

    if is_json_mode():
        # Build output dict without bulky base64 — the file is on disk
        out = {k: v for k, v in data.items() if k not in ("image_base64", "image")}
        out["file_path"] = str(output.resolve())
        emit(out)
        return
    console.print(f"[green]Captured[/green] {output.resolve()}")


# ---------------------------------------------------------------------------
# Simulation control: start / stop / step / pause / reset
# ---------------------------------------------------------------------------
@app.command()
def start(
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Start (play) the simulation."""
    result = _run(_tools(host, port).start_isaac_simulation())
    if is_json_mode():
        emit(result)
        return
    console.print("[green]play[/green] OK")


@app.command()
def stop(
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Stop the simulation."""
    result = _run(_tools(host, port).stop_isaac_simulation())
    if is_json_mode():
        emit(result)
        return
    console.print("[green]stop[/green] OK")


@app.command()
def pause(
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Pause the simulation."""
    result = _run(_tools(host, port).pause_isaac_simulation())
    if is_json_mode():
        emit(result)
        return
    console.print("[green]pause[/green] OK")


@app.command()
def step(
    count: int = typer.Argument(1, help="Number of physics steps"),
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Step the simulation forward by N physics frames."""
    result = _run(_tools(host, port).step_isaac_simulation(num_steps=count))
    if is_json_mode():
        emit(result)
        return
    console.print(f"[green]stepped[/green] {count} frame(s)")


@app.command()
def reset(
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Stop and reset the simulation to its initial state."""
    result = _run(_tools(host, port).reset_isaac_simulation())
    if is_json_mode():
        emit(result)
        return
    console.print("[green]reset[/green] OK")


# ---------------------------------------------------------------------------
# list-prims
# ---------------------------------------------------------------------------
@app.command("list-prims")
def list_prims(
    root_path: str = typer.Option("/", "--root", "-r", help="Root prim path to start from"),
    depth: int = typer.Option(-1, "--depth", "-d", help="Max traversal depth (-1 = unlimited)"),
    prim_type: Optional[str] = typer.Option(None, "--type", help="Filter by prim type (e.g. Mesh, Xform)"),
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """List prims in the scene with optional filtering."""
    result = _run(_tools(host, port).list_isaac_prims(
        root_path=root_path, max_depth=depth, prim_type=prim_type,
    ))
    if is_json_mode():
        emit(result)
        return
    prims = result.get("prims", [])
    table = Table(title=f"Prims ({len(prims)})")
    table.add_column("Path", style="cyan")
    table.add_column("Type")
    for p in prims:
        if isinstance(p, dict):
            table.add_row(p.get("path", "?"), p.get("type", "?"))
        else:
            table.add_row(str(p), "")
    console.print(table)


# ---------------------------------------------------------------------------
# prim-info
# ---------------------------------------------------------------------------
@app.command("prim-info")
def prim_info(
    prim_path: str = typer.Argument(..., help="USD path of the prim"),
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Get detailed information about a specific prim."""
    result = _run(_tools(host, port).get_isaac_prim_info(prim_path=prim_path))
    if is_json_mode():
        emit(result)
        return
    console.print(Syntax(json.dumps(result, indent=2, default=str), "json", theme="monokai"))


# ---------------------------------------------------------------------------
# create-prim
# ---------------------------------------------------------------------------
@app.command("create-prim")
def create_prim(
    prim_path: str = typer.Argument(..., help="USD path for the new prim"),
    prim_type: str = typer.Option("Xform", "--type", help="USD type name (Xform, Mesh, Cube, Sphere, etc.)"),
    attributes: Optional[str] = typer.Option(None, "--attrs", help='JSON dict of attributes, e.g. \'{"size": 1.0}\''),
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Create a new prim in the stage."""
    attrs_dict = None
    if attributes:
        try:
            attrs_dict = json.loads(attributes)
        except json.JSONDecodeError as e:
            if is_json_mode():
                emit_error(f"Invalid JSON for --attrs: {e}", "JSONDecodeError")
            console.print(f"[red]Invalid JSON for --attrs: {e}[/red]")
            raise typer.Exit(1)

    result = _run(_tools(host, port).create_isaac_prim(
        prim_path=prim_path, prim_type=prim_type, attributes=attrs_dict,
    ))
    if is_json_mode():
        emit(result)
        return
    created_path = result.get('prim_path', prim_path)
    created_type = result.get('prim_type', prim_type)
    console.print(f"[green]Created[/green] {created_path} ({created_type})")


# ---------------------------------------------------------------------------
# delete-prim
# ---------------------------------------------------------------------------
@app.command("delete-prim")
def delete_prim(
    prim_path: str = typer.Argument(..., help="USD path of the prim to delete"),
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Delete a prim from the stage."""
    result = _run(_tools(host, port).delete_isaac_prim(prim_path=prim_path))
    if is_json_mode():
        emit(result)
        return
    console.print(f"[green]Deleted[/green] {prim_path}")


# ---------------------------------------------------------------------------
# set-transform
# ---------------------------------------------------------------------------
@app.command("set-transform")
def set_transform(
    prim_path: str = typer.Argument(..., help="USD path of the prim"),
    translate: Optional[str] = typer.Option(None, "--translate", "--pos", help="Position as x,y,z"),
    rotate: Optional[str] = typer.Option(None, "--rotate", "--rot", help="Euler angles in degrees as x,y,z"),
    scale: Optional[str] = typer.Option(None, "--scale", help="Scale as x,y,z"),
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Set the transform (position, rotation, scale) of a prim."""
    try:
        translate_list = [float(v) for v in translate.split(",")] if translate else None
        rotate_list = [float(v) for v in rotate.split(",")] if rotate else None
        scale_list = [float(v) for v in scale.split(",")] if scale else None
    except ValueError as e:
        if is_json_mode():
            emit_error(f"Invalid numeric value in transform args: {e}", "ValueError")
        console.print(f"[red]Invalid numeric value in transform args: {e}[/red]")
        raise typer.Exit(1)

    result = _run(_tools(host, port).set_isaac_prim_transform(
        prim_path=prim_path,
        translation=translate_list,
        rotation_euler=rotate_list,
        scale=scale_list,
    ))
    if is_json_mode():
        emit(result)
        return
    console.print(f"[green]Transform set[/green] on {prim_path}")


# ---------------------------------------------------------------------------
# search-prims
# ---------------------------------------------------------------------------
@app.command("search-prims")
def search_prims(
    pattern: str = typer.Argument(..., help="Name pattern to search for (supports * and ? wildcards)"),
    prim_type: Optional[str] = typer.Option(None, "--type", help="Filter by prim type"),
    root_path: str = typer.Option("/", "--root", "-r", help="Root prim path"),
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Search for prims by name pattern."""
    result = _run(_tools(host, port).search_isaac_prims(
        search_type="name", query=pattern, root_path=root_path,
    ))
    if is_json_mode():
        emit(result)
        return
    matches = result.get("matches", [])
    table = Table(title=f"Search: '{pattern}' ({len(matches)} matches)")
    table.add_column("Path", style="cyan")
    table.add_column("Type")
    for m in matches:
        if isinstance(m, dict):
            table.add_row(m.get("path", "?"), m.get("type", "?"))
        else:
            table.add_row(str(m), "")
    console.print(table)


# ---------------------------------------------------------------------------
# open-stage
# ---------------------------------------------------------------------------
@app.command("open-stage")
def open_stage(
    file_path: str = typer.Argument(..., help="Path to USD file to open"),
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Open a USD stage file in Isaac Sim."""
    result = _run(_tools(host, port).open_isaac_stage(file_path=file_path))
    if is_json_mode():
        emit(result)
        return
    console.print(f"[green]Opened[/green] {file_path}")


# ---------------------------------------------------------------------------
# save-stage
# ---------------------------------------------------------------------------
@app.command("save-stage")
def save_stage(
    file_path: Optional[str] = typer.Argument(None, help="Path to save USD file (omit for in-place save)"),
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Save the current stage to a USD file."""
    result = _run(_tools(host, port).save_isaac_stage(file_path=file_path))
    if is_json_mode():
        emit(result)
        return
    console.print(f"[green]Saved[/green] {result.get('file_path', file_path or '(current)')}")


# ---------------------------------------------------------------------------
# new-stage
# ---------------------------------------------------------------------------
@app.command("new-stage")
def new_stage(
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Create a new empty stage."""
    result = _run(_tools(host, port).new_isaac_stage())
    if is_json_mode():
        emit(result)
        return
    console.print("[green]New stage created[/green]")


# ---------------------------------------------------------------------------
# create-physics-scene
# ---------------------------------------------------------------------------
@app.command("create-physics-scene")
def create_physics_scene(
    prim_path: str = typer.Option("/World/PhysicsScene", "--path", help="USD path for the physics scene"),
    gravity: float = typer.Option(9.81, "--gravity", "-g", help="Gravity magnitude in m/s^2"),
    gravity_dir: str = typer.Option("0,0,-1", "--gravity-dir", help="Gravity direction as x,y,z"),
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Create a UsdPhysics.Scene with gravity settings."""
    try:
        grav_dir_list = [float(v) for v in gravity_dir.split(",")]
    except ValueError as e:
        if is_json_mode():
            emit_error(f"Invalid numeric value in --gravity-dir: {e}", "ValueError")
        console.print(f"[red]Invalid numeric value in --gravity-dir: {e}[/red]")
        raise typer.Exit(1)
    result = _run(_tools(host, port).create_isaac_physics_scene(
        prim_path=prim_path,
        gravity_direction=grav_dir_list,
        gravity_magnitude=gravity,
    ))
    if is_json_mode():
        emit(result)
        return
    if result.get("already_existed"):
        console.print(f"[yellow]Physics scene already exists[/yellow] at {prim_path}")
    else:
        console.print(f"[green]Physics scene created[/green] at {prim_path} (gravity={gravity} m/s^2)")


# ---------------------------------------------------------------------------
# create-light
# ---------------------------------------------------------------------------
@app.command("create-light")
def create_light(
    prim_path: str = typer.Argument(..., help="USD path for the new light"),
    light_type: str = typer.Option(
        "DomeLight", "--type",
        help="DomeLight, DistantLight, SphereLight, RectLight, DiskLight, CylinderLight",
    ),
    intensity: float = typer.Option(1000.0, "--intensity", "-i", help="Light intensity"),
    color: Optional[str] = typer.Option(None, "--color", help="RGB color as r,g,b (0-1 floats)"),
    temperature: Optional[float] = typer.Option(None, "--temperature", help="Color temperature in Kelvin"),
    angle: Optional[float] = typer.Option(None, "--angle", help="Cone angle for DistantLight (degrees)"),
    texture: Optional[str] = typer.Option(None, "--texture", help="Texture file path for DomeLight (HDRI)"),
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Create a light prim in the scene."""
    try:
        color_list = [float(v) for v in color.split(",")] if color else None
    except ValueError as e:
        if is_json_mode():
            emit_error(f"Invalid numeric value in --color: {e}", "ValueError")
        console.print(f"[red]Invalid numeric value in --color: {e}[/red]")
        raise typer.Exit(1)
    result = _run(_tools(host, port).create_isaac_light(
        prim_path=prim_path,
        light_type=light_type,
        intensity=intensity,
        color=color_list,
        color_temperature=temperature,
        angle=angle,
        texture_file=texture,
    ))
    if is_json_mode():
        emit(result)
        return
    console.print(f"[green]Created[/green] {light_type} at {prim_path} (intensity={intensity})")


# ---------------------------------------------------------------------------
# create-material
# ---------------------------------------------------------------------------
@app.command("create-material")
def create_material(
    material_path: str = typer.Argument(..., help="USD path for the new material (e.g. /World/Looks/Red)"),
    shader_type: str = typer.Option("UsdPreviewSurface", "--shader", help="UsdPreviewSurface or OmniPBR"),
    color: str = typer.Option("0.8,0.8,0.8", "--color", "-c", help="Diffuse RGB color as r,g,b (0-1)"),
    roughness: float = typer.Option(0.5, "--roughness", help="Surface roughness 0-1"),
    metallic: float = typer.Option(0.0, "--metallic", help="Metallic factor 0-1"),
    opacity: float = typer.Option(1.0, "--opacity", help="Opacity 0-1"),
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Create a material with a surface shader."""
    valid_shaders = ("UsdPreviewSurface", "OmniPBR")
    if shader_type not in valid_shaders:
        msg = f"Invalid shader_type '{shader_type}'. Must be one of: {valid_shaders}"
        if is_json_mode():
            emit_error(msg, "ValueError")
        console.print(f"[red]{msg}[/red]")
        raise typer.Exit(1)

    try:
        color_list = [float(v) for v in color.split(",")]
    except ValueError as e:
        if is_json_mode():
            emit_error(f"Invalid numeric value in --color: {e}", "ValueError")
        console.print(f"[red]Invalid numeric value in --color: {e}[/red]")
        raise typer.Exit(1)
    result = _run(_tools(host, port).create_isaac_material(
        material_path=material_path,
        shader_type=shader_type,
        diffuse_color=color_list,
        roughness=roughness,
        metallic=metallic,
        opacity=opacity,
    ))
    if is_json_mode():
        emit(result)
        return
    console.print(f"[green]Created[/green] {shader_type} material at {material_path}")


# ---------------------------------------------------------------------------
# list-extensions
# ---------------------------------------------------------------------------
@app.command("list-extensions")
def list_extensions(
    enabled_only: bool = typer.Option(False, "--enabled", help="Show only enabled extensions"),
    search: Optional[str] = typer.Option(None, "--search", "-s", help="Filter by extension ID substring"),
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """List extensions registered in the running Isaac Sim instance."""
    result = _run(_tools(host, port).list_isaac_extensions(
        enabled_only=enabled_only,
        search=search,
    ))
    if is_json_mode():
        emit(result)
        return
    extensions = result.get("extensions", [])
    table = Table(title=f"Extensions ({result.get('count', len(extensions))})")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Version", style="dim")
    table.add_column("Enabled", justify="center")
    for ext in extensions:
        status = "[green]yes[/green]" if ext.get("enabled") else "[red]no[/red]"
        table.add_row(ext.get("id", "?"), ext.get("version", ""), status)
    console.print(table)


# ---------------------------------------------------------------------------
# enable-extension
# ---------------------------------------------------------------------------
@app.command("enable-extension")
def enable_extension(
    extension_id: str = typer.Argument(..., help="Extension ID to enable (e.g. omni.physx)"),
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Enable an extension by ID in the running Isaac Sim instance."""
    result = _run(_tools(host, port).enable_isaac_extension(extension_id=extension_id))
    if is_json_mode():
        emit(result)
        return
    if result.get("enabled"):
        console.print(f"[green]Enabled[/green] {extension_id} (v{result.get('version', '?')})")
    else:
        console.print(f"[yellow]Warning:[/yellow] {extension_id} may not have been enabled — check with list-extensions")


# ---------------------------------------------------------------------------
# disable-extension
# ---------------------------------------------------------------------------
@app.command("disable-extension")
def disable_extension(
    extension_id: str = typer.Argument(..., help="Extension ID to disable (e.g. omni.physx)"),
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Disable an extension by ID in the running Isaac Sim instance."""
    result = _run(_tools(host, port).disable_isaac_extension(extension_id=extension_id))
    if is_json_mode():
        emit(result)
        return
    if not result.get("enabled"):
        console.print(f"[red]Disabled[/red] {extension_id}")
    else:
        console.print(f"[yellow]Warning:[/yellow] {extension_id} may still be enabled — check with list-extensions")
