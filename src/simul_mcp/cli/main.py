"""
CLI interface for Simul MCP Server.

Top-level app ``simul`` with backend-specific sub-commands::

    simul server          # start MCP server
    simul info            # show capabilities
    simul version         # show version
    simul validate-config # validate YAML config
    simul commands        # list all commands (JSON manifest)
    simul isaac ping      # check Isaac Sim connectivity
    simul isaac status    # instance status
    simul isaac scene     # scene overview
    simul isaac exec ...  # execute Python script
    simul isaac start     # play simulation
    simul usd info <f>    # analyse USD file
    simul usd validate <f>
    simul usd summary <f>

Global ``--json`` flag forces structured JSON output on stdout for all
commands.  When stdout is not a TTY the CLI auto-detects JSON mode.
"""

import asyncio
import json
import socket
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from simul_mcp.cli.output import emit, emit_error, is_json_mode, set_json_mode
from simul_mcp.config import get_settings, load_settings
from simul_mcp.logging import setup_logging, get_logger
from simul_mcp.mcp.server import SimulMCPServer, start_mcp_server
from simul_mcp.adapters import (
    is_blender_available,
    is_headless_available,
)

# Import sub-apps
from simul_mcp.cli.isaac import app as isaac_app
from simul_mcp.cli.usd_cli import app as usd_app


def _is_isaac_reachable(host: str, port: int, timeout: float = 1.0) -> bool:
    """
    Check if Isaac Sim is reachable on the TCP socket.

    Args:
        host: Isaac Sim host address.
        port: TCP port for the VS Code extension socket.
        timeout: Connection timeout in seconds.

    Returns:
        True if a TCP connection succeeds.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (ConnectionRefusedError, OSError, TimeoutError):
        return False


# ---------------------------------------------------------------------------
# Top-level app with global --json callback
# ---------------------------------------------------------------------------
def _global_callback(
    json_output: bool = typer.Option(
        False, "--json", help="Output structured JSON to stdout (auto-enabled when stdout is not a TTY)",
    ),
) -> None:
    """Global options applied before any sub-command."""
    if json_output:
        set_json_mode(True)


app = typer.Typer(
    name="simul",
    help="Simul -- 3D Simulation & DCC Tools CLI",
    add_completion=False,
    callback=_global_callback,
)
console = Console(stderr=True)

# Register sub-apps
app.add_typer(isaac_app, name="isaac", help="Isaac Sim commands")
app.add_typer(usd_app, name="usd", help="USD file commands (headless)")


# ---------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------
@app.command()
def server(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    transport: str = typer.Option(
        "stdio", "--transport", "-t", help="Transport type (stdio, sse)"
    ),
    log_level: Optional[str] = typer.Option(
        None, "--log-level", "-l", help="Log level (DEBUG, INFO, WARNING, ERROR)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """Start the Simul MCP Server."""
    try:
        if config:
            settings = load_settings(config)
        else:
            settings = get_settings()

        if log_level:
            settings.logging.level = log_level.upper()
        if verbose:
            settings.logging.level = "DEBUG"

        setup_logging(settings)
        logger = get_logger(__name__)

        isaac_host = settings.isaac_sim.socket_host
        isaac_port = settings.isaac_sim.socket_port
        isaac_reachable = _is_isaac_reachable(isaac_host, isaac_port)
        blender_available = is_blender_available()
        usd_available = is_headless_available()

        console.print(
            Panel.fit(
                f"[bold blue]Simul -- 3D Simulation & DCC Tools[/bold blue]\n"
                f"Transport: {transport}\n"
                f"Isaac Sim (TCP :{isaac_port}): {'reachable' if isaac_reachable else 'not reachable (tools will retry at call time)'}\n"
                f"Blender: {'available' if blender_available else 'unavailable'}\n"
                f"USD Headless: {'available' if usd_available else 'unavailable'}\n"
                f"Config: {config or 'default'}\n"
                f"Log Level: {settings.logging.level}",
                title="Starting Server",
            )
        )

        logger.info(f"Starting Simul 3D MCP Server with {transport} transport")
        asyncio.run(start_mcp_server(settings, transport))

    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error starting server: {e}[/red]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------
@app.command()
def info(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
) -> None:
    """Show server information and capabilities."""
    try:
        if config:
            settings = load_settings(config)
        else:
            settings = get_settings()

        isaac_host = settings.isaac_sim.socket_host
        isaac_port = settings.isaac_sim.socket_port
        isaac_reachable = _is_isaac_reachable(isaac_host, isaac_port)
        blender_available = is_blender_available()
        usd_available = is_headless_available()

        # Instantiate server to get actually registered tools
        server_instance = SimulMCPServer(settings)
        tool_names: list[str] = []
        lp = getattr(server_instance.mcp, "local_provider", None)
        if lp is not None:
            components = getattr(lp, "_components", {})
            tool_names = sorted(
                k.split(":")[1].split("@")[0]
                for k in components
                if k.startswith("tool:")
            )

        # Categorise by name prefix
        categories: dict[str, list[str]] = {
            "Isaac Sim": [],
            "Blender": [],
            "Unreal": [],
            "USD / Headless": [],
            "Instance Management": [],
        }
        for name in tool_names:
            if name.startswith(("list_isaac_instances", "set_active_isaac_instance")):
                categories["Instance Management"].append(name)
            elif name.startswith(("isaac_", "ping_isaac", "execute_isaac",
                                  "get_isaac", "set_isaac", "create_isaac",
                                  "delete_isaac", "search_isaac", "list_isaac",
                                  "add_isaac", "duplicate_isaac", "reparent_isaac",
                                  "import_isaac", "new_isaac", "open_isaac",
                                  "save_isaac", "capture_isaac", "start_isaac",
                                  "stop_isaac", "pause_isaac", "reset_isaac",
                                  "step_isaac", "assign_isaac")):
                categories["Isaac Sim"].append(name)
            elif "blender" in name or "simready" in name:
                categories["Blender"].append(name)
            elif "unreal" in name:
                categories["Unreal"].append(name)
            else:
                categories["USD / Headless"].append(name)

        if is_json_mode():
            data = {
                "backends": {
                    "isaac_sim": {"reachable": isaac_reachable, "port": isaac_port},
                    "blender": {"available": blender_available},
                    "usd_headless": {"available": usd_available},
                },
                "tool_count": len(tool_names),
                "categories": {k: v for k, v in categories.items() if v},
            }
            emit(data)
            return

        system_table = Table(title="System Information")
        system_table.add_column("Component", style="cyan")
        system_table.add_column("Status", style="green")
        system_table.add_column("Details")
        system_table.add_row(
            f"Isaac Sim (TCP :{isaac_port})",
            "Reachable" if isaac_reachable else "Not Reachable",
            "Simulation & viewport via TCP socket"
            if isaac_reachable
            else "Isaac Sim tools will retry at call time",
        )
        system_table.add_row(
            "Blender Runtime",
            "Available" if blender_available else "Not Available",
            "Blender scene tools through bpy"
            if blender_available
            else "Install bpy to enable Blender tools",
        )
        system_table.add_row(
            "USD Headless",
            "Available" if usd_available else "Not Available",
            "pxr library for USD file operations" if usd_available else "No USD support",
        )
        console.print(system_table)
        console.print()

        categories_table = Table(title="Tool Categories")
        categories_table.add_column("Category", style="cyan")
        categories_table.add_column("Tools", justify="center")
        for cat_name, cat_tools in categories.items():
            if cat_tools:
                categories_table.add_row(cat_name, str(len(cat_tools)))
        console.print(categories_table)
        console.print()

        tools_table = Table(title="Registered Tools")
        tools_table.add_column("Tool Name", style="cyan")
        tools_table.add_column("Category")
        for cat_name, cat_tools in categories.items():
            for t in cat_tools:
                tools_table.add_row(t, cat_name)
        console.print(tools_table)

        console.print(
            Panel.fit(
                f"[bold]Summary[/bold]\n"
                f"Total Registered Tools: {len(tool_names)}",
                title="Tool Summary",
            )
        )

    except Exception as e:
        if is_json_mode():
            emit_error(str(e), "InfoError")
        console.print(f"[red]Error getting server info: {e}[/red]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# commands  (P3: machine-readable introspection)
# ---------------------------------------------------------------------------
def _collect_commands(
    typer_app: typer.Typer, prefix: str = "",
) -> list[dict[str, object]]:
    """Recursively collect command metadata from a Typer app tree."""
    result: list[dict[str, object]] = []
    click_app = typer.main.get_command(typer_app)

    if hasattr(click_app, "list_commands"):
        ctx = typer.Context(click_app)
        for name in click_app.list_commands(ctx):
            full_name = f"{prefix} {name}".strip() if prefix else name
            cmd = click_app.get_command(ctx, name)
            if cmd is None:
                continue
            if hasattr(cmd, "list_commands"):
                # It's a group — recurse
                sub_typer = None
                for group in getattr(typer_app, "registered_groups", []):
                    if getattr(group, "name", None) == name:
                        sub_typer = getattr(group, "typer_instance", None)
                        break
                if sub_typer is not None:
                    result.extend(_collect_commands(sub_typer, full_name))
                else:
                    # Fallback: list sub-commands from the click group
                    sub_ctx = typer.Context(cmd, parent=ctx)
                    for sub_name in cmd.list_commands(sub_ctx):
                        sub_full = f"{full_name} {sub_name}"
                        sub_cmd = cmd.get_command(sub_ctx, sub_name)
                        if sub_cmd is None:
                            continue
                        params = []
                        for p in sub_cmd.params:
                            params.append({
                                "name": p.name,
                                "type": p.type.name if hasattr(p.type, "name") else str(p.type),
                                "required": p.required,
                                "default": str(p.default) if p.default is not None else None,
                                "help": getattr(p, "help", None),
                            })
                        result.append({
                            "command": sub_full,
                            "description": sub_cmd.help or "",
                            "params": params,
                        })
            else:
                params = []
                for p in cmd.params:
                    params.append({
                        "name": p.name,
                        "type": p.type.name if hasattr(p.type, "name") else str(p.type),
                        "required": p.required,
                        "default": str(p.default) if p.default is not None else None,
                        "help": getattr(p, "help", None),
                    })
                result.append({
                    "command": full_name,
                    "description": cmd.help or "",
                    "params": params,
                })
    return result


@app.command()
def commands() -> None:
    """List all available commands with parameters (always JSON)."""
    cmds = _collect_commands(app)
    print(json.dumps({"commands": cmds, "count": len(cmds)}, indent=2))


# ---------------------------------------------------------------------------
# validate-config
# ---------------------------------------------------------------------------
@app.command()
def validate_config(
    config: Path = typer.Argument(
        ...,
        help="Path to configuration file",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
) -> None:
    """Validate a configuration file."""
    try:
        settings = load_settings(config)

        if is_json_mode():
            emit({
                "valid": True,
                "config_path": str(config),
                "server_name": settings.server.name,
                "log_level": settings.logging.level,
                "usd_cache_enabled": settings.usd.cache_enabled,
                "max_file_size_mb": settings.usd.max_file_size_mb,
                "viewport_max_size": settings.viewport.max_size,
            })
            return

        console.print(f"[cyan]Validating configuration file: {config}[/cyan]")
        console.print("[green]Configuration file is valid[/green]")

        settings_table = Table(title="Configuration Summary")
        settings_table.add_column("Setting", style="cyan")
        settings_table.add_column("Value")
        settings_table.add_row("Server Name", settings.server.name)
        settings_table.add_row("Log Level", settings.logging.level)
        settings_table.add_row("USD Cache Enabled", str(settings.usd.cache_enabled))
        settings_table.add_row("Max File Size (MB)", str(settings.usd.max_file_size_mb))
        settings_table.add_row("Viewport Max Size", str(settings.viewport.max_size))
        console.print(settings_table)

    except Exception as e:
        if is_json_mode():
            emit_error(str(e), "ValidationError")
        console.print(f"[red]Configuration validation failed: {e}[/red]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------
@app.command()
def version() -> None:
    """Show version information."""
    try:
        from simul_mcp import __version__
        version_str = __version__
    except ImportError:
        version_str = "unknown"

    settings = get_settings()
    isaac_port = settings.isaac_sim.socket_port
    isaac_reachable = _is_isaac_reachable(
        settings.isaac_sim.socket_host, isaac_port
    )
    blender_available = is_blender_available()
    usd_available = is_headless_available()

    if is_json_mode():
        emit({
            "version": version_str,
            "isaac_sim": {"reachable": isaac_reachable, "port": isaac_port},
            "blender": {"available": blender_available},
            "usd_headless": {"available": usd_available},
        })
        return

    console.print(
        Panel.fit(
            f"[bold blue]Simul -- 3D Simulation & DCC Tools[/bold blue]\n"
            f"Version: {version_str}\n"
            f"Isaac Sim (TCP :{isaac_port}): {'reachable' if isaac_reachable else 'not reachable'}\n"
            f"Blender: {'available' if blender_available else 'unavailable'}\n"
            f"USD Headless: {'available' if usd_available else 'unavailable'}",
            title="Version Information",
        )
    )


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------
@app.command()
def stats(
    tool_name: Optional[str] = typer.Option(None, "--tool", "-t", help="Filter to a specific tool"),
    recent: int = typer.Option(0, "--recent", "-r", help="Show last N call records"),
    reset: bool = typer.Option(False, "--reset", help="Clear all usage stats"),
) -> None:
    """Show tool usage statistics from the persistent log."""
    from simul_mcp.mcp.usage_tracker import ToolUsageTracker

    if reset:
        tracker = ToolUsageTracker()
        tracker.reset()
        if is_json_mode():
            emit({"success": True, "message": "Stats cleared"})
        else:
            console.print("[green]Usage stats cleared[/green]")
        return

    data = ToolUsageTracker.load_from_log(tool_name=tool_name, limit=recent)

    if is_json_mode():
        emit(data)
        return

    tools = data.get("tools", {})
    if not tools:
        console.print("[dim]No tool usage recorded yet.[/dim]")
        return

    table = Table(title=f"Tool Usage ({data.get('total_calls', 0)} total calls)")
    table.add_column("Tool", style="cyan", no_wrap=True)
    table.add_column("Calls", justify="right")
    table.add_column("OK", justify="right", style="green")
    table.add_column("Fail", justify="right", style="red")
    table.add_column("Avg (ms)", justify="right", style="dim")

    for name, s in tools.items():
        table.add_row(
            name,
            str(s["total_calls"]),
            str(s["successes"]),
            str(s["failures"]),
            f"{s['avg_duration_ms']:.1f}",
        )
    console.print(table)

    recent_records = data.get("recent", [])
    if recent_records:
        console.print()
        log_table = Table(title=f"Recent Calls (last {len(recent_records)})")
        log_table.add_column("Tool", style="cyan")
        log_table.add_column("Duration", justify="right")
        log_table.add_column("Status", justify="center")
        log_table.add_column("Params", style="dim")

        for rec in recent_records:
            status = "[green]OK[/green]" if rec.get("success") else f"[red]{rec.get('error', 'FAIL')}[/red]"
            params_str = json.dumps(rec.get("params", {}), separators=(",", ":")) if rec.get("params") else ""
            log_table.add_row(
                rec.get("tool", "?"),
                f"{rec.get('duration_ms', 0):.1f}ms",
                status,
                params_str[:80],
            )
        console.print(log_table)


if __name__ == "__main__":
    app()
