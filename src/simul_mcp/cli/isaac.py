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
import textwrap
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


def _tools(
    host: Optional[str] = None,
    port: Optional[int] = None,
    timeout: Optional[float] = None,
) -> IsaacTools:
    """Build an IsaacTools instance from settings with optional overrides."""
    settings = get_settings()
    client = IsaacSocketClient(
        host=host if host is not None else settings.isaac_sim.socket_host,
        port=port if port is not None else settings.isaac_sim.socket_port,
        bridge_host=settings.isaac_sim.bridge_host,
        bridge_port=settings.isaac_sim.bridge_port,
        bridge_timeout_seconds=settings.isaac_sim.bridge_timeout,
        prefer_bridge=settings.isaac_sim.bridge_enabled,
        fallback_to_vscode=settings.isaac_sim.bridge_fallback_to_vscode,
        timeout_seconds=timeout if timeout is not None else settings.isaac_sim.socket_timeout,
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


def _parse_script_result(result: Any) -> Dict[str, Any]:
    """Parse a raw script result, preferring JSON output when available."""
    if not result.success:
        raise RuntimeError(result.error_value or result.error_name or "Script execution failed")
    output = result.output.strip()
    if not output:
        return {}
    try:
        parsed = json.loads(output)
    except (json.JSONDecodeError, ValueError):
        return {"output": result.output}
    if isinstance(parsed, dict):
        return parsed
    return {"parsed": parsed}


# Common host/port/timeout options
_host_opt = typer.Option(None, "--host", "-H", help="Isaac Sim host")
_port_opt = typer.Option(None, "--port", "-p", help="Isaac Sim port")
_timeout_opt = typer.Option(
    None,
    "--timeout",
    "-t",
    help="Timeout in seconds (defaults to isaac_sim.socket_timeout)",
)


# ---------------------------------------------------------------------------
# ping
# ---------------------------------------------------------------------------
@app.command()
def ping(
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
    timeout: Optional[float] = _timeout_opt,
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


@app.command("bridge-capabilities")
def bridge_capabilities(
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
    timeout: Optional[float] = _timeout_opt,
) -> None:
    """Show the current typed bridge capability envelope."""
    tools = _tools(host, port, timeout)
    try:
        response = asyncio.run(tools._client.bridge_request("capabilities", {}))
    except (ConnectionRefusedError, TimeoutError, ValueError) as exc:
        if is_json_mode():
            emit_error(str(exc), type(exc).__name__)
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    if response.get("status") != "ok":
        error = response.get("error", {})
        message = str(error.get("message", "Bridge request failed"))
        error_name = str(error.get("name", "BridgeError"))
        if is_json_mode():
            emit_error(message, error_name)
        console.print(f"[red]{error_name}: {message}[/red]")
        raise typer.Exit(1)

    payload = response.get("payload", {})
    result = {
        "bridge_address": tools._client.bridge_address,
        "transport": payload.get("transport"),
        "protocol_version": response.get("protocol_version"),
        "allow_unsafe_execution": payload.get("allow_unsafe_execution"),
        "actions": payload.get("actions", []),
    }
    if is_json_mode():
        emit(result)
        return

    table = Table(title=f"Bridge Capabilities @ {tools._client.bridge_address}")
    table.add_column("Property", style="cyan")
    table.add_column("Value")
    table.add_row("Transport", str(result.get("transport")))
    table.add_row("Protocol", str(result.get("protocol_version")))
    table.add_row(
        "execute_script",
        "enabled" if result.get("allow_unsafe_execution") else "disabled",
    )
    table.add_row("Actions", ", ".join(result.get("actions", [])))
    console.print(table)


@app.command("bridge-config")
def bridge_config(
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
    timeout: Optional[float] = _timeout_opt,
) -> None:
    """Read the current bridge settings from the running Isaac Sim instance."""
    tools = _tools(host, port, timeout)
    script = textwrap.dedent(
        """\
        import json
        import carb
        import omni.kit.app

        settings = carb.settings.get_settings()
        manager = omni.kit.app.get_app().get_extension_manager()
        print(json.dumps({
            "extension_enabled": manager.is_extension_enabled("khemoo.simul.mcp"),
            "host": settings.get("/exts/khemoo.simul.mcp/host"),
            "port": int(settings.get("/exts/khemoo.simul.mcp/port")),
            "allow_unsafe_execution": bool(settings.get("/exts/khemoo.simul.mcp/allow_unsafe_execution")),
            "max_request_bytes": int(settings.get("/exts/khemoo.simul.mcp/max_request_bytes")),
            "max_response_bytes": int(settings.get("/exts/khemoo.simul.mcp/max_response_bytes")),
        }))
        """
    )
    try:
        result = asyncio.run(tools._client.execute_vscode_only(script))
        parsed = _parse_script_result(result)
    except (ConnectionRefusedError, TimeoutError, RuntimeError, ValueError) as exc:
        if is_json_mode():
            emit_error(str(exc), type(exc).__name__)
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    if is_json_mode():
        emit(parsed)
        return

    table = Table(title="Bridge Config")
    table.add_column("Property", style="cyan")
    table.add_column("Value")
    for key in (
        "extension_enabled",
        "host",
        "port",
        "allow_unsafe_execution",
        "max_request_bytes",
        "max_response_bytes",
    ):
        table.add_row(key, str(parsed.get(key)))
    console.print(table)


@app.command("bridge-set-unsafe")
def bridge_set_unsafe(
    enable: bool = typer.Option(
        True,
        "--enable/--disable",
        help="Enable or disable bridge execute_script permission.",
    ),
    restart: bool = typer.Option(
        True,
        "--restart/--no-restart",
        help="Restart the bridge extension after updating the setting.",
    ),
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
    timeout: Optional[float] = _timeout_opt,
) -> None:
    """Toggle bridge execute_script permission inside the running Isaac Sim instance."""
    tools = _tools(host, port, timeout)
    script = textwrap.dedent(
        f"""\
        import json
        import carb
        import omni.kit.app

        settings = carb.settings.get_settings()
        settings.set("/exts/khemoo.simul.mcp/allow_unsafe_execution", {str(enable)})
        manager = omni.kit.app.get_app().get_extension_manager()
        extension_name = "khemoo.simul.mcp"
        enabled = manager.is_extension_enabled(extension_name)
        restarted = False
        if enabled and {str(restart)}:
            manager.set_extension_enabled_immediate(extension_name, False)
            manager.set_extension_enabled_immediate(extension_name, True)
            restarted = True

        print(json.dumps({{
            "allow_unsafe_execution": {str(enable)},
            "restart_requested": {str(restart)},
            "restarted": restarted,
        }}))
        """
    )
    try:
        result = asyncio.run(tools._client.execute_vscode_only(script))
        parsed = _parse_script_result(result)
    except (ConnectionRefusedError, TimeoutError, RuntimeError, ValueError) as exc:
        if is_json_mode():
            emit_error(str(exc), type(exc).__name__)
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    if is_json_mode():
        emit(parsed)
        return

    state = "enabled" if parsed.get("allow_unsafe_execution") else "disabled"
    console.print(
        f"[green]execute_script[/green] {state}"
        + (" and bridge restarted" if parsed.get("restarted") else "")
    )


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

    async def _fetch() -> tuple:
        return await asyncio.gather(
            tools.get_isaac_stage_info(),
            tools.get_isaac_simulation_state(),
        )

    stage, sim = asyncio.run(_fetch())
    for r in (stage, sim):
        if isinstance(r, dict) and r.get("error"):
            if is_json_mode():
                emit_error(r["error"], r.get("error_type", "Error"), r.get("details"))
            console.print(f"[red]{r.get('error_type', 'Error')}: {r['error']}[/red]")
            raise typer.Exit(1)

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

    async def _fetch() -> tuple:
        return await asyncio.gather(
            tools.get_isaac_scene_summary(),
            tools.get_isaac_scene_stats(),
            tools.list_isaac_cameras(),
            tools.list_isaac_lights(),
            tools.list_isaac_physics_objects(),
        )

    summary, stats, cameras, lights, physics = asyncio.run(_fetch())
    for r in (summary, stats, cameras, lights, physics):
        if isinstance(r, dict) and r.get("error"):
            if is_json_mode():
                emit_error(r["error"], r.get("error_type", "Error"), r.get("details"))
            console.print(f"[red]{r.get('error_type', 'Error')}: {r['error']}[/red]")
            raise typer.Exit(1)

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
    timeout: Optional[float] = _timeout_opt,
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
        if tools._client.bridge_enabled:
            result = asyncio.run(tools._client.execute_vscode_only(code))
        else:
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
        emit(output_data)
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


# ---------------------------------------------------------------------------
# get-carb-settings
# ---------------------------------------------------------------------------
@app.command("get-carb-settings")
def get_carb_settings(
    keys: List[str] = typer.Argument(..., help="Carb setting key paths (e.g. /rtx/fog/enabled)"),
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Read one or more Carbonite settings by key path."""
    result = _run(_tools(host, port).get_carb_settings(keys=keys))
    if is_json_mode():
        emit(result)
        return
    settings = result.get("settings", {})
    table = Table(title="Carb Settings")
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value")
    for key, val in settings.items():
        table.add_row(key, str(val))
    console.print(table)


# ---------------------------------------------------------------------------
# set-carb-settings
# ---------------------------------------------------------------------------
@app.command("set-carb-settings")
def set_carb_settings(
    assignments: List[str] = typer.Argument(..., help="Key=value pairs (e.g. /rtx/fog/enabled=true /rtx/fog/fogEndDist=200.0)"),
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Write one or more Carbonite settings. Pass key=value pairs."""
    parsed: Dict[str, Any] = {}
    for a in assignments:
        if "=" not in a:
            if is_json_mode():
                emit_error(f"Invalid format: {a!r} — expected key=value")
            console.print(f"[red]Invalid format: {a!r} — expected key=value[/red]")
            raise typer.Exit(1)
        key, raw_val = a.split("=", 1)
        val: Any = raw_val
        if raw_val.lower() == "true":
            val = True
        elif raw_val.lower() == "false":
            val = False
        else:
            try:
                val = int(raw_val)
            except ValueError:
                try:
                    val = float(raw_val)
                except ValueError:
                    pass
        parsed[key] = val
    result = _run(_tools(host, port).set_carb_settings(settings=parsed))
    if is_json_mode():
        emit(result)
        return
    applied = result.get("applied", {})
    table = Table(title=f"Applied ({result.get('count', len(applied))} settings)")
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    for key, val in applied.items():
        table.add_row(key, str(val))
    console.print(table)


# ---------------------------------------------------------------------------
# read-aovs
# ---------------------------------------------------------------------------
@app.command("read-aovs")
def read_aovs(
    aov_names: List[str] = typer.Argument(..., help="AOV names to read (e.g. HdrColor DirectDiffuse)"),
    camera: str = typer.Option("/OmniverseKit_Persp", "--camera", "-c", help="Camera prim path"),
    width: int = typer.Option(256, "--width", "-W", help="Render width"),
    height: int = typer.Option(256, "--height", help="Render height"),
    frames: int = typer.Option(5, "--frames", "-f", help="Number of render frames before reading"),
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Read AOV buffers and return per-AOV statistics."""
    result = _run(_tools(host, port).read_aovs(
        aov_names=aov_names,
        camera_path=camera,
        resolution=[width, height],
        num_frames=frames,
    ))
    if is_json_mode():
        emit(result)
        return
    aovs = result.get("aovs", {})
    if not aovs:
        console.print("[dim]No AOV data returned.[/dim]")
        return
    for name, stats in aovs.items():
        if "error" in stats:
            console.print(f"[red]{name}[/red]: {stats['error']}")
            continue
        table = Table(title=f"AOV: {name}")
        table.add_column("Stat", style="cyan")
        table.add_column("Value", justify="right")
        for k, v in stats.items():
            if isinstance(v, list):
                table.add_row(k, ", ".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in v))
            elif isinstance(v, float):
                table.add_row(k, f"{v:.6f}")
            else:
                table.add_row(k, str(v))
        console.print(table)
    attach_errors = result.get("attach_errors", {})
    if attach_errors:
        for name, err in attach_errors.items():
            console.print(f"[yellow]Attach failed:[/yellow] {name}: {err}")


# ---------------------------------------------------------------------------
# list-aovs
# ---------------------------------------------------------------------------
@app.command("list-aovs")
def list_aovs(
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """List all available AOV annotator names."""
    result = _run(_tools(host, port).list_aovs())
    if is_json_mode():
        emit(result)
        return
    annotators = result.get("annotators", [])
    console.print(f"[bold]Available AOVs ({result.get('count', len(annotators))}):[/bold]")
    for name in annotators:
        console.print(f"  {name}")


# ---------------------------------------------------------------------------
# query-typed-prims
# ---------------------------------------------------------------------------
@app.command("query-typed-prims")
def query_typed_prims(
    type_name: str = typer.Argument(..., help="USD schema type (e.g. UsdLux.DistantLight, UsdGeom.Mesh)"),
    attributes: Optional[List[str]] = typer.Option(None, "--attr", "-a", help="Attributes to read"),
    root_path: str = typer.Option("/", "--root", "-r", help="Root path to start traversal"),
    max_prims: int = typer.Option(200, "--max", "-m", help="Max prims to return"),
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Query prims by USD schema type and read attributes."""
    result = _run(_tools(host, port).query_usd_typed_prims(
        type_name=type_name,
        attributes=attributes,
        root_path=root_path,
        max_prims=max_prims,
    ))
    if is_json_mode():
        emit(result)
        return
    prims = result.get("prims", [])
    truncated = result.get("truncated", False)
    title = f"{result.get('type_filter', type_name)} ({result.get('count', len(prims))} found)"
    if truncated:
        title += f" [yellow](truncated at {max_prims})[/yellow]"
    table = Table(title=title)
    table.add_column("Path", style="cyan", no_wrap=True)
    table.add_column("Type", style="dim")
    if attributes:
        for attr in attributes:
            table.add_column(attr)
    for prim in prims:
        row = [prim.get("path", "?"), prim.get("type", "?")]
        if attributes:
            attrs = prim.get("attributes", {})
            for attr in attributes:
                val = attrs.get(attr)
                if isinstance(val, list) and len(val) <= 4:
                    row.append(", ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in val))
                elif isinstance(val, float):
                    row.append(f"{val:.4f}")
                else:
                    row.append(str(val) if val is not None else "[dim]None[/dim]")
        table.add_row(*row)
    console.print(table)


# ---------------------------------------------------------------------------
# viewport-info
# ---------------------------------------------------------------------------
@app.command("viewport-info")
def viewport_info(
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Get active viewport information."""
    result = _run(_tools(host, port).get_viewport_info())
    if is_json_mode():
        emit(result)
        return
    if result.get("error"):
        console.print(f"[red]Error:[/red] {result['error']}")
        return
    table = Table(title="Active Viewport")
    table.add_column("Property", style="cyan")
    table.add_column("Value")
    for key in ("camera_path", "render_product_path", "resolution", "name"):
        val = result.get(key)
        if val is not None:
            table.add_row(key, str(val))
    console.print(table)


# ---------------------------------------------------------------------------
# list-render-vars
# ---------------------------------------------------------------------------
@app.command("list-render-vars")
def list_render_vars(
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """List available render variable names from SyntheticData."""
    result = _run(_tools(host, port).list_render_vars())
    if is_json_mode():
        emit(result)
        return
    if result.get("syntheticdata_error"):
        console.print(f"[red]Error:[/red] {result['syntheticdata_error']}")
        return
    templates = result.get("render_var_templates", [])
    if templates:
        console.print(f"[bold]Render Var Templates ({result.get('render_var_count', len(templates))}):[/bold]")
        for name in templates:
            console.print(f"  {name}")
    sensors = result.get("sensor_types", [])
    if sensors:
        console.print(f"\n[bold]Sensor Types ({result.get('sensor_type_count', len(sensors))}):[/bold]")
        for name in sensors:
            console.print(f"  {name}")


# ---------------------------------------------------------------------------
# runtime-info
# ---------------------------------------------------------------------------
@app.command("runtime-info")
def runtime_info(
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Get consolidated runtime diagnostics from Isaac Sim."""
    result = _run(_tools(host, port).get_runtime_info())
    if is_json_mode():
        emit(result)
        return

    sections = [
        ("App", "app", ["version", "python_version", "update_number"]),
        ("Timeline", "timeline", ["is_playing", "is_stopped", "current_time", "fps"]),
        ("Physics Config", "physics_config", ["gpu_dynamics_enabled", "physics_dt", "solver_type"]),
        ("Renderer", "renderer", ["gpu_name", "raytracing_mode", "hgi_driver"]),
        ("Stage", "stage", ["url", "prim_count"]),
        ("Extensions", "extensions", ["total", "enabled"]),
    ]
    for title, key, fields in sections:
        data = result.get(key, {})
        error = result.get(f"{key}_error")
        if error:
            console.print(f"[red]{title}:[/red] {error}")
            continue
        if not data:
            continue
        table = Table(title=title, show_header=False, box=None, padding=(0, 2))
        table.add_column(style="cyan")
        table.add_column()
        for f in fields:
            val = data.get(f)
            if val is not None:
                table.add_row(f, str(val))
        console.print(table)
        console.print()

    # Physics stats (dynamic body counts)
    physics = result.get("physics", {})
    if physics:
        table = Table(title="Physics Stats", show_header=False, box=None, padding=(0, 2))
        table.add_column(style="cyan")
        table.add_column(justify="right")
        for k, v in sorted(physics.items()):
            table.add_row(k, str(v))
        console.print(table)
        console.print()

    # Viewport
    viewport = result.get("viewport", {})
    if viewport and viewport.get("camera_path"):
        table = Table(title="Viewport", show_header=False, box=None, padding=(0, 2))
        table.add_column(style="cyan")
        table.add_column()
        for k in ("camera_path", "resolution", "fps"):
            val = viewport.get(k)
            if val is not None:
                table.add_row(k, str(val))
        console.print(table)


# ---------------------------------------------------------------------------
# logs
# ---------------------------------------------------------------------------
@app.command("logs")
def logs(
    level: str = typer.Option("warn", "--level", "-l", help="Minimum level: verbose, info, warn, error"),
    last_n: int = typer.Option(50, "--last", "-n", help="Number of recent entries to show"),
    source: Optional[str] = typer.Option(None, "--source", "-s", help="Filter by source module substring"),
    search: Optional[str] = typer.Option(None, "--search", "-q", help="Search within log messages"),
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Read recent log entries from the Isaac Sim console."""
    result = _run(_tools(host, port).get_isaac_logs(
        level=level, last_n=last_n, source_filter=source, search=search,
    ))
    if is_json_mode():
        emit(result)
        return
    if result.get("error"):
        console.print(f"[red]Error:[/red] {result['error']}")
        return

    counts = result.get("counts", {})
    console.print(
        f"[dim]Log:[/dim] {result.get('log_file', '?')}  "
        f"[dim]Totals:[/dim] "
        f"[red]{counts.get('error', 0)} errors[/red], "
        f"[yellow]{counts.get('warn', 0)} warnings[/yellow], "
        f"[blue]{counts.get('info', 0)} info[/blue], "
        f"[dim]{counts.get('verbose', 0)} verbose[/dim]"
    )
    console.print()

    entries = result.get("entries", [])
    if not entries:
        console.print("[dim]No matching log entries.[/dim]")
        return

    level_styles = {"error": "red bold", "warn": "yellow", "info": "blue", "verbose": "dim"}
    for e in entries:
        lvl = e.get("level", "?")
        style = level_styles.get(lvl, "")
        ts = e.get("timestamp", "")
        src = e.get("source", "")
        msg = e.get("message", "")
        ts_str = f"[dim]{ts}[/dim] " if ts else ""
        console.print(f"{ts_str}[{style}]{lvl.upper():>5s}[/{style}] [{src}] {msg}")


# ---------------------------------------------------------------------------
# set-log-level
# ---------------------------------------------------------------------------
@app.command("set-log-level")
def set_log_level_cmd(
    level: str = typer.Argument(..., help="Log level: verbose, info, warn, error"),
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Set the Carbonite logging threshold for Isaac Sim."""
    result = _run(_tools(host, port).set_isaac_log_level(level=level))
    if is_json_mode():
        emit(result)
        return
    if result.get("error"):
        console.print(f"[red]Error:[/red] {result['error']}")
        return
    console.print(
        f"Log level: [dim]{result.get('previous_level', '?')}[/dim] → "
        f"[green]{result.get('current_level', '?')}[/green]"
    )
