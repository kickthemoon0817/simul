"""
Unreal Engine CLI sub-commands.

Provides direct terminal access to a running Unreal Engine instance via
the Remote Control HTTP API. Designed as the primary interface for AI
agents — thin MCP registration keeps context small while this CLI
exposes the full operation set.

All commands support ``--json`` (or auto-detect via TTY) for structured
JSON output on stdout, making them consumable by AI agents via Bash.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from simul_mcp.adapters.unreal_runtime import (
    UnrealRuntimeSession,
    is_unreal_available,
)
from simul_mcp.adapters.unreal_setup import (
    LauncherNotFound,
    ensure_remote_control_config,
    launch_editor,
    resolve_launch_argv,
)
from simul_mcp.cli.output import emit, emit_error, is_json_mode
from simul_mcp.config import get_settings

app = typer.Typer(
    name="unreal",
    help="Unreal Engine commands -- interact with a running UE5 instance via Remote Control API.",
    add_completion=False,
)
console = Console(stderr=True)


def _session(
    host: Optional[str] = None,
    port: Optional[int] = None,
    timeout: Optional[int] = None,
) -> UnrealRuntimeSession:
    """Build an UnrealRuntimeSession with optional overrides."""
    settings = get_settings()
    overrides: Dict[str, Any] = {}
    if host is not None:
        overrides["host"] = host
    if port is not None:
        overrides["port"] = port
    if timeout is not None:
        overrides["timeout"] = timeout
    if overrides:
        unreal_cfg = settings.unreal.model_copy(update=overrides)
        settings = settings.model_copy(update={"unreal": unreal_cfg})
    return UnrealRuntimeSession(settings)


def _run(coro: Any) -> Dict[str, Any]:
    """
    Run an async session method and handle errors uniformly.

    In JSON mode: emits the full result dict to stdout.
    In Rich mode: prints errors with Rich formatting.
    """
    try:
        result = asyncio.run(coro)
    except Exception as e:
        if is_json_mode():
            emit_error(str(e), type(e).__name__)
            return {}  # unreachable (emit_error raises), but documents intent
        console.print(f"[red]{type(e).__name__}: {e}[/red]")
        raise typer.Exit(1)

    if isinstance(result, dict) and result.get("error"):
        if is_json_mode():
            emit_error(
                result["error"],
                result.get("error_type", "Error"),
                result.get("details"),
            )
            return result  # unreachable (emit_error raises), but documents intent
        console.print(f"[red]{result.get('error_type', 'Error')}: {result['error']}[/red]")
        raise typer.Exit(1)
    return result


# Common options
_host_opt = typer.Option(None, "--host", "-H", help="Remote Control API host")
_port_opt = typer.Option(None, "--port", "-p", help="Remote Control API port")
_timeout_opt = typer.Option(None, "--timeout", "-t", help="HTTP timeout in seconds")


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------
@app.command()
def health(
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Check connectivity to Unreal Engine Remote Control API."""
    session = _session(host, port)
    result = _run(session.health_check())
    if is_json_mode():
        emit(result)
        return
    connected = result.get("connected", False)
    if connected:
        console.print(f"[green]Connected[/green] to UE5 @ {session.settings.unreal.host}:{session.settings.unreal.port}")
        if result.get("engine_version"):
            console.print(f"  Engine: {result['engine_version']}")
        if result.get("project_name"):
            console.print(f"  Project: {result['project_name']}")
    else:
        console.print("[red]Not connected[/red]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------
@app.command()
def info(
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Get Unreal Engine runtime information."""
    session = _session(host, port)
    result = _run(session.get_engine_info())
    if is_json_mode():
        emit(result)
        return
    console.print(Syntax(json.dumps(result, indent=2, default=str), "json", theme="monokai"))


# ---------------------------------------------------------------------------
# scene
# ---------------------------------------------------------------------------
@app.command()
def scene(
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Summarize the current Unreal scene for AI consumption."""
    session = _session(host, port)
    result = _run(session.summarize_scene())
    if is_json_mode():
        emit(result)
        return
    console.print(Syntax(json.dumps(result, indent=2, default=str), "json", theme="monokai"))


# ---------------------------------------------------------------------------
# map
# ---------------------------------------------------------------------------
@app.command()
def map(
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Get the currently loaded persistent level path."""
    session = _session(host, port)
    result = _run(session.get_loaded_map())
    if is_json_mode():
        emit(result)
        return
    console.print(f"Map: {result.get('map_path', '?')}")


# ---------------------------------------------------------------------------
# list-actors
# ---------------------------------------------------------------------------
@app.command("list-actors")
def list_actors(
    actor_class: Optional[str] = typer.Option(None, "--class", "-c", help="Filter by actor class"),
    tag: Optional[str] = typer.Option(None, "--tag", help="Filter by actor tag"),
    limit: int = typer.Option(200, "--limit", "-n", help="Max actors to return"),
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """List actors in the current level."""
    session = _session(host, port)
    result = _run(session.list_actors(
        class_filter=actor_class,
        tag_filter=tag,
        max_results=limit,
    ))
    if is_json_mode():
        emit(result)
        return
    actors = result.get("actors", [])
    table = Table(title=f"Actors ({len(actors)})")
    table.add_column("Path", style="cyan")
    table.add_column("Class")
    table.add_column("Label")
    for a in actors:
        if isinstance(a, dict):
            table.add_row(
                a.get("path", "?"),
                a.get("class", "?"),
                a.get("label", ""),
            )
        else:
            table.add_row(str(a), "", "")
    console.print(table)


# ---------------------------------------------------------------------------
# actor-info
# ---------------------------------------------------------------------------
@app.command("actor-info")
def actor_info(
    actor_path: str = typer.Argument(..., help="Actor path in the level"),
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Get detailed information about a specific actor."""
    session = _session(host, port)
    result = _run(session.get_actor_info(actor_path))
    if is_json_mode():
        emit(result)
        return
    console.print(Syntax(json.dumps(result, indent=2, default=str), "json", theme="monokai"))


# ---------------------------------------------------------------------------
# spawn
# ---------------------------------------------------------------------------
@app.command()
def spawn(
    actor_class: str = typer.Argument(..., help="Actor class or asset path to spawn"),
    label: Optional[str] = typer.Option(None, "--label", "-l", help="Actor label"),
    location: Optional[str] = typer.Option(None, "--location", "--pos", help="Location as x,y,z"),
    rotation: Optional[str] = typer.Option(None, "--rotation", "--rot", help="Rotation as pitch,yaw,roll"),
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Spawn an actor from a class or asset path."""
    try:
        loc = tuple(float(v) for v in location.split(",")) if location else (0.0, 0.0, 0.0)
        rot = tuple(float(v) for v in rotation.split(",")) if rotation else (0.0, 0.0, 0.0)
    except (ValueError, IndexError) as e:
        if is_json_mode():
            emit_error(f"Invalid transform value: {e}", "ValueError")
        console.print(f"[red]Invalid transform value: {e}[/red]")
        raise typer.Exit(1)

    session = _session(host, port)
    result = _run(session.spawn_actor(
        asset_path=actor_class,
        location=loc,
        rotation=rot,
        label=label,
    ))
    if is_json_mode():
        emit(result)
        return
    console.print(f"[green]Spawned[/green] {result.get('actor_path', actor_class)}")


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------
@app.command()
def delete(
    actor_path: str = typer.Argument(..., help="Actor path to delete"),
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Delete an actor from the level."""
    session = _session(host, port)
    result = _run(session.delete_actor(actor_path))
    if is_json_mode():
        emit(result)
        return
    console.print(f"[green]Deleted[/green] {actor_path}")


# ---------------------------------------------------------------------------
# set-transform
# ---------------------------------------------------------------------------
@app.command("set-transform")
def set_transform(
    actor_path: str = typer.Argument(..., help="Actor path"),
    location: Optional[str] = typer.Option(None, "--location", "--pos", help="Location as x,y,z"),
    rotation: Optional[str] = typer.Option(None, "--rotation", "--rot", help="Rotation as pitch,yaw,roll"),
    scale: Optional[str] = typer.Option(None, "--scale", help="Scale as x,y,z"),
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Set an actor's location, rotation, and scale."""
    try:
        loc = tuple(float(v) for v in location.split(",")) if location else None
        rot = tuple(float(v) for v in rotation.split(",")) if rotation else None
        sc = tuple(float(v) for v in scale.split(",")) if scale else None
    except ValueError as e:
        if is_json_mode():
            emit_error(f"Invalid numeric value: {e}", "ValueError")
        console.print(f"[red]Invalid numeric value: {e}[/red]")
        raise typer.Exit(1)

    session = _session(host, port)
    result = _run(session.set_actor_transform(
        actor_path=actor_path,
        location=loc,
        rotation=rot,
        scale=sc,
    ))
    if is_json_mode():
        emit(result)
        return
    console.print(f"[green]Transform set[/green] on {actor_path}")


# ---------------------------------------------------------------------------
# set-property
# ---------------------------------------------------------------------------
@app.command("set-property")
def set_property(
    actor_path: str = typer.Argument(..., help="Actor path"),
    property_name: str = typer.Argument(..., help="Property name"),
    value: str = typer.Argument(..., help="Property value (JSON)"),
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Set a property on an Unreal actor by name and JSON value."""
    try:
        parsed_value = json.loads(value)
    except json.JSONDecodeError:
        parsed_value = value  # treat as string

    session = _session(host, port)
    result = _run(session.set_actor_property(
        actor_path=actor_path,
        property_name=property_name,
        property_value=json.dumps(parsed_value) if not isinstance(parsed_value, str) else parsed_value,
    ))
    if is_json_mode():
        emit(result)
        return
    console.print(f"[green]Set[/green] {property_name} on {actor_path}")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------
@app.command()
def search(
    query: str = typer.Argument(..., help="Search query (name or class)"),
    search_class: Optional[str] = typer.Option(None, "--class", "-c", help="Filter by asset class"),
    package_path: Optional[str] = typer.Option(None, "--package", help="Filter by package path"),
    limit: int = typer.Option(50, "--limit", "-n", help="Max results"),
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Search the Unreal Asset Registry."""
    session = _session(host, port)
    result = _run(session.search_assets(
        query=query,
        class_names=[search_class] if search_class else None,
        package_paths=[package_path] if package_path else None,
        max_results=limit,
    ))
    if is_json_mode():
        emit(result)
        return
    assets = result.get("assets", [])
    table = Table(title=f"Assets matching '{query}' ({len(assets)})")
    table.add_column("Path", style="cyan")
    table.add_column("Class")
    for a in assets:
        if isinstance(a, dict):
            table.add_row(a.get("path", "?"), a.get("class", "?"))
        else:
            table.add_row(str(a), "")
    console.print(table)


# ---------------------------------------------------------------------------
# sim
# ---------------------------------------------------------------------------
@app.command()
def sim(
    action: str = typer.Argument(..., help="Simulation action: start, stop, pause, resume, step"),
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Control Play-In-Editor simulation: start, stop, pause, resume, step."""
    valid = {"start", "stop", "pause", "resume", "step"}
    if action not in valid:
        msg = f"Invalid action '{action}'. Must be one of: {', '.join(sorted(valid))}"
        if is_json_mode():
            emit_error(msg, "ValueError")
        console.print(f"[red]{msg}[/red]")
        raise typer.Exit(1)

    session = _session(host, port)
    result = _run(session.control_simulation(action))
    if is_json_mode():
        emit(result)
        return
    console.print(f"[green]{action}[/green] OK")


# ---------------------------------------------------------------------------
# sim-status
# ---------------------------------------------------------------------------
@app.command("sim-status")
def sim_status(
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Get current Play-In-Editor simulation status."""
    session = _session(host, port)
    result = _run(session.get_simulation_status())
    if is_json_mode():
        emit(result)
        return
    state = result.get("state", "unknown")
    console.print(f"Simulation: [bold]{state}[/bold]")


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------
@app.command()
def capture(
    output: Path = typer.Argument("capture.png", help="Output file path"),
    width: int = typer.Option(1920, "--width", "-W", help="Width in pixels"),
    height: int = typer.Option(1080, "--height", help="Height in pixels"),
    format: str = typer.Option("png", "--format", "-f", help="Image format (png, jpeg)"),
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Capture viewport screenshot to a file."""
    import base64

    valid_formats = {"png", "jpeg", "jpg"}
    if format not in valid_formats:
        msg = f"Invalid format '{format}'. Must be one of: {', '.join(sorted(valid_formats))}"
        if is_json_mode():
            emit_error(msg, "ValueError")
        console.print(f"[red]{msg}[/red]")
        raise typer.Exit(1)

    session = _session(host, port)
    result = _run(session.capture_viewport(
        resolution_x=width,
        resolution_y=height,
        format=format,
    ))
    image_b64 = result.get("image_base64") or result.get("image", "")
    if image_b64:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(base64.b64decode(image_b64))
    else:
        if is_json_mode():
            emit_error("Viewport capture returned no image data", "CaptureError")
        console.print("[red]Viewport capture returned no image data[/red]")
        raise typer.Exit(1)

    if is_json_mode():
        out = {k: v for k, v in result.items() if k not in ("image_base64", "image")}
        out["file_path"] = str(output.resolve())
        emit(out)
        return
    console.print(f"[green]Captured[/green] {output.resolve()}")


# ---------------------------------------------------------------------------
# exec
# ---------------------------------------------------------------------------
@app.command("exec")
def exec_script(
    script: Optional[str] = typer.Argument(None, help="Python code string or path to .py file"),
    mode: str = typer.Option("ExecuteFile", "--mode", "-m", help="ExecuteFile, EvaluateStatement, ExecuteStatement"),
    raw: bool = typer.Option(False, "--raw", "-r", help="Print raw output without formatting"),
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Execute Python code inside Unreal Engine."""
    valid_modes = {"ExecuteFile", "EvaluateStatement", "ExecuteStatement"}
    if mode not in valid_modes:
        msg = f"Invalid mode '{mode}'. Must be one of: {', '.join(sorted(valid_modes))}"
        if is_json_mode():
            emit_error(msg, "ValueError")
        console.print(f"[red]{msg}[/red]")
        raise typer.Exit(1)

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

    session = _session(host, port)
    try:
        raw_result = asyncio.run(session._execute_python(code, mode=mode))
    except Exception as e:
        if is_json_mode():
            emit_error(str(e), type(e).__name__)
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    parsed = session._parse_python_json(raw_result)
    success = not parsed.get("error")

    if is_json_mode() and not raw:
        parsed["success"] = success
        parsed["raw"] = raw_result
        print(json.dumps(parsed, default=str))
        if not success:
            raise typer.Exit(1)
        return

    if raw:
        print(json.dumps(raw_result, default=str))
        if not success:
            raise typer.Exit(1)
        return

    if success:
        try:
            formatted = json.dumps(parsed, indent=2, default=str)
            console.print(Syntax(formatted, "json", theme="monokai"))
        except Exception:
            console.print(str(parsed))
    else:
        console.print(f"[red]Error:[/red] {parsed.get('error', 'Unknown')}")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# materials
# ---------------------------------------------------------------------------
@app.command("materials")
def list_materials(
    actor_path: str = typer.Argument(..., help="Actor path to inspect materials"),
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Get material information for an actor."""
    session = _session(host, port)
    result = _run(session.get_material_info(actor_path))
    if is_json_mode():
        emit(result)
        return
    console.print(Syntax(json.dumps(result, indent=2, default=str), "json", theme="monokai"))


# ---------------------------------------------------------------------------
# scene-graph
# ---------------------------------------------------------------------------
@app.command("scene-graph")
def scene_graph(
    root: Optional[str] = typer.Option(None, "--root", "-r", help="Root actor path"),
    depth: int = typer.Option(3, "--depth", "-d", help="Max traversal depth"),
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Query the Unreal scene graph hierarchy."""
    session = _session(host, port)
    result = _run(session.query_scene_graph(
        root_path=root,
        max_depth=depth,
    ))
    if is_json_mode():
        emit(result)
        return
    console.print(Syntax(json.dumps(result, indent=2, default=str), "json", theme="monokai"))


# ---------------------------------------------------------------------------
# set-visibility
# ---------------------------------------------------------------------------
@app.command("set-visibility")
def set_visibility(
    actor_path: str = typer.Argument(..., help="Actor path"),
    visible: bool = typer.Argument(..., help="True or False"),
    host: Optional[str] = _host_opt,
    port: Optional[int] = _port_opt,
) -> None:
    """Set actor visibility."""
    session = _session(host, port)
    result = _run(session.set_actor_visibility(
        actor_path=actor_path,
        visible=visible,
    ))
    if is_json_mode():
        emit(result)
        return
    state = "visible" if visible else "hidden"
    console.print(f"[green]{state}[/green] {actor_path}")


# ---------------------------------------------------------------------------
# setup -- auto-configure Remote Control, optionally launch the editor,
#          then poll until Remote Control accepts connections.
# ---------------------------------------------------------------------------
async def _poll_health(session: UnrealRuntimeSession, timeout: float, interval: float) -> Dict[str, Any]:
    """Call health_check repeatedly until connected or timeout elapses."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    last: Dict[str, Any] = {}
    while loop.time() < deadline:
        last = await session.health_check()
        if last.get("connected"):
            return last
        await asyncio.sleep(interval)
    return last


def _is_loopback_bind(host: str) -> bool:
    """Return True when ``host`` is a loopback or unspecified-loopback bind.

    Anything else is treated as a network-exposed bind by the safety gate.
    Conservative on purpose — better to make the user pass --allow-public
    once than to silently expose remote Python execution.
    """
    if not host:
        return True
    h = host.strip().lower()
    return (
        h in {"localhost", "::1"}
        or h.startswith("127.")
    )


@app.command("setup")
def setup(
    uproject: Path = typer.Argument(..., help="Path to the .uproject file"),
    port: int = typer.Option(30010, "--port", "-p", help="Remote Control HTTP port to configure"),
    engine_path: Optional[Path] = typer.Option(
        None,
        "--engine-path",
        help="Unreal install root (contains Engine/). Usually auto-detected.",
    ),
    bind: Optional[str] = typer.Option(
        None,
        "--bind",
        help=(
            "Override RemoteControlHttpServerHostname. Leave unset to keep "
            "UE's default (typically loopback). Pass an interface IP or "
            "0.0.0.0 to enable cross-host access — combine with --allow-public "
            "for non-loopback binds."
        ),
    ),
    websocket_port: Optional[int] = typer.Option(
        None,
        "--websocket-port",
        help=(
            "Override RemoteControlWebSocketServerPort. Leave unset to keep "
            "UE's default (30020). Use this when running multiple UE editors "
            "on the same host so their WebSocket endpoints don't collide."
        ),
    ),
    allow_public: bool = typer.Option(
        False,
        "--allow-public",
        help=(
            "Required acknowledgment when --bind is non-loopback. UE Remote "
            "Control runs WITHOUT authentication and bEnableRemotePythonExecution "
            "is True, so a public bind exposes arbitrary Python execution to "
            "anything on the network. Pass this flag only after confirming the "
            "trust radius (firewall, trusted LAN, etc.)."
        ),
    ),
    launch: bool = typer.Option(True, "--launch/--no-launch", help="Launch the editor after configuring"),
    headless: bool = typer.Option(
        True,
        "--headless/--no-headless",
        help=(
            "Default: headless. UE launches with -RenderOffScreen so no "
            "window opens. The render pipeline still runs so viewport "
            "capture, scene capture, and replicator workflows work — they "
            "just don't depend on the editor having OS-level focus. Pass "
            "--no-headless when you want the GUI editor open (e.g. to "
            "hand-edit a scene at the same time)."
        ),
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the interactive confirmation prompt",
    ),
    wait_timeout: float = typer.Option(
        90.0,
        "--wait-timeout",
        help="Seconds to wait for Remote Control to accept connections",
    ),
    poll_interval: float = typer.Option(
        2.0,
        "--poll-interval",
        help="Seconds between health-check polls",
    ),
) -> None:
    """Auto-configure Remote Control + Python in a UE project and launch the editor.

    This is the one-call path that replaces the manual checklist of enabling
    ``RemoteControl`` / ``PythonScriptPlugin`` in the .uproject and editing
    ``Config/DefaultRemoteControl.ini`` before every simul session.
    """
    uproject = uproject.expanduser().resolve()
    if not uproject.is_file() or uproject.suffix != ".uproject":
        msg = f"Not a .uproject file: {uproject}"
        if is_json_mode():
            emit_error(msg, "InvalidArgument")
            return
        console.print(f"[red]{msg}[/red]")
        raise typer.Exit(2)

    # Safety gate: a non-loopback bind exposes UE Remote Control to the
    # network. Since we also enable bEnableRemotePythonExecution=True, that
    # means anyone reachable can run arbitrary Python in the editor. Force
    # an explicit acknowledgment via --allow-public.
    if bind is not None and not _is_loopback_bind(bind) and not allow_public:
        msg = (
            f"--bind {bind!r} would expose UE Remote Control (with "
            f"bEnableRemotePythonExecution=True) to the network. Re-run with "
            f"--allow-public to acknowledge the trust-radius implication, or "
            f"pick a loopback bind (omit --bind, or pass 127.0.0.1)."
        )
        if is_json_mode():
            emit_error(msg, "InvalidArgument")
            raise typer.Exit(2)
        console.print(f"[red]{msg}[/red]")
        raise typer.Exit(2)

    # Preview-only launch resolution so we can tell the user what WOULD happen.
    launch_plan: Optional[List[str]] = None
    launch_plan_error: Optional[str] = None
    if launch:
        try:
            launch_plan = resolve_launch_argv(uproject, engine_path, headless=headless)
        except (LauncherNotFound, FileNotFoundError) as exc:
            launch_plan_error = str(exc)

    if not yes and not is_json_mode():
        console.print(Panel.fit(
            f"[bold]simul unreal setup[/bold]\n"
            f"  project:    {uproject}\n"
            f"  port:       {port}\n"
            + (f"  bind:       {bind}\n" if bind is not None else "")
            + (f"  ws port:    {websocket_port}\n" if websocket_port is not None else "")
            + (f"  [yellow]allow-public:[/yellow] yes\n" if allow_public else "")
            + f"  launch:     {'yes' if launch else 'no'}\n"
            + (f"  launch cmd: {' '.join(launch_plan)}\n" if launch_plan else "")
            + (f"  [yellow]launch issue:[/yellow] {launch_plan_error}\n" if launch_plan_error else ""),
            title="Plan",
        ))
        console.print(
            "This will edit the .uproject and DefaultRemoteControl.ini in place. "
            "Re-run with [bold]--yes[/bold] to skip this prompt."
        )
        if not typer.confirm("Proceed?", default=True):
            console.print("[yellow]Aborted by user.[/yellow]")
            raise typer.Exit(1)

    # Patch config files.
    try:
        patch = ensure_remote_control_config(
            uproject,
            port=port,
            bind=bind,
            websocket_port=websocket_port,
        )
    except Exception as exc:
        if is_json_mode():
            emit_error(str(exc), type(exc).__name__)
            return
        console.print(f"[red]config patch failed: {exc}[/red]")
        raise typer.Exit(1)

    if not is_json_mode():
        console.print(patch.uproject.summary())
        console.print(patch.ini.summary())

    # Launch if requested.
    launched = False
    pid: Optional[int] = None
    if launch and launch_plan_error is None:
        try:
            proc = launch_editor(uproject, engine_path, headless=headless)
            launched = True
            pid = proc.pid
            if not is_json_mode():
                console.print(
                    f"[green]Launched[/green] editor (pid {pid}): {' '.join(launch_plan or [])}"
                )
        except (LauncherNotFound, FileNotFoundError) as exc:
            launch_plan_error = str(exc)

    # Poll Remote Control. If we didn't launch, the user is expected to
    # already have the editor running.
    session = _session(port=port)
    if not is_json_mode():
        console.print(f"Waiting for Remote Control @ {session.settings.unreal.host}:{port} ...")
    health = asyncio.run(_poll_health(session, wait_timeout, poll_interval))

    payload: Dict[str, Any] = {
        "uproject": str(uproject),
        "port": port,
        "bind": bind,
        "websocket_port": websocket_port,
        "allow_public": allow_public,
        "patched": {
            "uproject": {
                "changed": patch.uproject.changed,
                "added": patch.uproject.added,
                "updated": patch.uproject.updated,
            },
            "ini": {
                "changed": patch.ini.changed,
                "added": patch.ini.added,
                "updated": patch.ini.updated,
            },
        },
        "launched": launched,
        "launch_pid": pid,
        "launch_error": launch_plan_error,
        "health": health,
        "connected": bool(health.get("connected")),
    }

    if is_json_mode():
        emit(payload)
        if not payload["connected"]:
            raise typer.Exit(1)
        return

    if payload["connected"]:
        console.print(
            f"[green]Remote Control is up[/green] — simul can now talk to this editor."
        )
        return
    console.print(
        "[red]Remote Control did not respond in time.[/red] "
        "Check that the editor finished loading, that plugins are enabled, "
        "and that the port is not already in use."
    )
    raise typer.Exit(1)
