"""
CLI interface for Simul MCP Server.

This module provides a Typer-based command-line interface for running
the MCP server and utility commands.
"""

import asyncio
import socket
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from simul_mcp.config import get_settings, load_settings
from simul_mcp.logging import setup_logging, get_logger
from simul_mcp.mcp.server import start_mcp_server
from simul_mcp.mcp.tools.registry import get_tool_registry
from simul_mcp.adapters import (
    is_blender_available,
    is_headless_available,
)


def _is_isaac_reachable(host: str = "127.0.0.1", port: int = 8226, timeout: float = 1.0) -> bool:
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

# Initialize CLI
app = typer.Typer(
    name="simul-mcp",
    help="Simul – 3D Simulation & DCC Tools MCP Server",
    add_completion=False,
)
console = Console(stderr=True)


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
):
    """Start the Isaac Sim MCP Server."""
    try:
        # Load settings
        if config:
            settings = load_settings(config)
        else:
            settings = get_settings()

        # Override log level if specified
        if log_level:
            settings.logging.level = log_level.upper()

        if verbose:
            settings.logging.level = "DEBUG"

        # Setup logging
        setup_logging(settings)
        logger = get_logger(__name__)

        # Check runtime availability
        isaac_reachable = _is_isaac_reachable()
        blender_available = is_blender_available()
        usd_available = is_headless_available()

        # Display startup info
        console.print(
            Panel.fit(
                f"[bold blue]Simul – 3D Simulation & DCC Tools[/bold blue]\n"
                f"Transport: {transport}\n"
                f"Isaac Sim (TCP :8226): {'✓ reachable' if isaac_reachable else '✗ not reachable (tools will retry at call time)'}\n"
                f"Blender: {'✓' if blender_available else '✗'}\n"
                f"USD Headless: {'✓' if usd_available else '✗'}\n"
                f"Config: {config or 'default'}\n"
                f"Log Level: {settings.logging.level}",
                title="Starting Server",
            )
        )

        # Start server
        logger.info(f"Starting Simul 3D MCP Server with {transport} transport")
        asyncio.run(start_mcp_server(settings, transport))

    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error starting server: {e}[/red]")
        raise typer.Exit(1)


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
):
    """Show server information and capabilities."""
    try:
        # Load settings
        if config:
            settings = load_settings(config)
        else:
            settings = get_settings()

        # Check runtime availability
        isaac_reachable = _is_isaac_reachable()
        blender_available = is_blender_available()
        usd_available = is_headless_available()

        # Get tool registry
        registry = get_tool_registry(settings)
        capabilities = registry.get_capabilities()

        # Display system info
        system_table = Table(title="System Information")
        system_table.add_column("Component", style="cyan")
        system_table.add_column("Status", style="green")
        system_table.add_column("Details")

        system_table.add_row(
            "Isaac Sim (TCP :8226)",
            "✓ Reachable" if isaac_reachable else "✗ Not Reachable",
            "Simulation & viewport via TCP socket"
            if isaac_reachable
            else "Isaac Sim tools will retry at call time",
        )
        system_table.add_row(
            "Blender Runtime",
            "✓ Available" if blender_available else "✗ Not Available",
            "Blender scene tools through bpy"
            if blender_available
            else "Install bpy to enable Blender tools",
        )
        system_table.add_row(
            "USD Headless",
            "✓ Available" if usd_available else "✗ Not Available",
            "pxr library for USD file operations" if usd_available else "No USD support",
        )

        console.print(system_table)
        console.print()

        # Display tool categories
        categories_table = Table(title="Tool Categories")
        categories_table.add_column("Category", style="cyan")
        categories_table.add_column("Total Tools", justify="center")
        categories_table.add_column("Enabled", justify="center", style="green")
        categories_table.add_column("Disabled", justify="center", style="red")

        for category, info in capabilities["categories"].items():
            categories_table.add_row(
                category.replace("_", " ").title(),
                str(info["total"]),
                str(info["enabled"]),
                str(info["total"] - info["enabled"]),
            )

        console.print(categories_table)
        console.print()

        # Display individual tools
        tools_table = Table(title="Available Tools")
        tools_table.add_column("Tool Name", style="cyan")
        tools_table.add_column("Category")
        tools_table.add_column("Status", justify="center")
        tools_table.add_column("Requirements")

        for tool_name, tool_info in capabilities["tools"].items():
            status = "✓ Enabled" if tool_info["enabled"] else "✗ Disabled"
            status_style = "green" if tool_info["enabled"] else "red"

            requirements = []
            if tool_info.get("requires_blender"):
                requirements.append("Blender")
            if tool_info.get("requires_unreal"):
                requirements.append("Unreal")
            if tool_info.get("requires_usd"):
                requirements.append("USD")

            tools_table.add_row(
                tool_name,
                tool_info["category"].replace("_", " ").title(),
                Text(status, style=status_style),
                ", ".join(requirements) if requirements else "None",
            )

        console.print(tools_table)

        # Summary
        console.print(
            Panel.fit(
                f"[bold]Summary[/bold]\n"
                f"Total Tools: {capabilities['total_tools']}\n"
                f"Enabled: {capabilities['enabled_tools']}\n"
                f"Disabled: {capabilities['total_tools'] - capabilities['enabled_tools']}",
                title="Tool Summary",
            )
        )

    except Exception as e:
        console.print(f"[red]Error getting server info: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def validate_config(
    config: Path = typer.Argument(
        ...,
        help="Path to configuration file",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
):
    """Validate a configuration file."""
    try:
        console.print(f"[cyan]Validating configuration file: {config}[/cyan]")

        # Try to load settings
        settings = load_settings(config)

        # Display validation results
        console.print("[green]✓ Configuration file is valid[/green]")

        # Show key settings
        settings_table = Table(title="Configuration Summary")
        settings_table.add_column("Setting", style="cyan")
        settings_table.add_column("Value")

        settings_table.add_row("Server Name", settings.server.name)
        settings_table.add_row("Server Version", settings.server.version)
        settings_table.add_row("Log Level", settings.logging.level)
        settings_table.add_row("USD Cache Enabled", str(settings.usd.cache_enabled))
        settings_table.add_row("Max File Size (MB)", str(settings.usd.max_file_size_mb))
        settings_table.add_row("Viewport Max Size", str(settings.viewport.max_size))

        console.print(settings_table)

    except Exception as e:
        console.print(f"[red]✗ Configuration validation failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def test_usd(
    file_path: Path = typer.Argument(
        ...,
        help="Path to USD file to test",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
):
    """Test USD file loading and analysis."""
    try:
        # Load settings
        if config:
            settings = load_settings(config)
        else:
            settings = get_settings()

        # Setup minimal logging
        setup_logging(settings)

        console.print(f"[cyan]Testing USD file: {file_path}[/cyan]")

        # Check USD availability
        usd_available = is_headless_available()
        if not usd_available:
            console.print("[red]Error: No USD support available[/red]")
            raise typer.Exit(1)

        # Test file loading
        from simul_mcp.adapters import HeadlessUSDAdapter

        adapter = HeadlessUSDAdapter(settings)
        with adapter.create_session() as session:
            console.print("[yellow]Loading USD file...[/yellow]")
            stage_id = session.load_stage(file_path)

            if not stage_id:
                console.print("[red]✗ Failed to load USD file[/red]")
                raise typer.Exit(1)

            console.print(
                f"[green]✓ Successfully loaded USD file (Stage ID: {stage_id})[/green]"
            )

            # Get stage info
            console.print("[yellow]Analyzing stage...[/yellow]")
            stage_info = session.get_stage_info(stage_id)

            if stage_info:
                # Display stage information
                stage_table = Table(title="Stage Information")
                stage_table.add_column("Property", style="cyan")
                stage_table.add_column("Value")

                stage_table.add_row("Up Axis", stage_info.up_axis)
                stage_table.add_row("Meters per Unit", str(stage_info.meters_per_unit))
                stage_table.add_row(
                    "Time Codes per Second", str(stage_info.time_codes_per_second)
                )
                stage_table.add_row("Start Time", str(stage_info.start_time_code))
                stage_table.add_row("End Time", str(stage_info.end_time_code))
                stage_table.add_row("Frame Rate", str(stage_info.frame_rate))
                stage_table.add_row("Total Prims", str(len(stage_info.all_prims)))
                stage_table.add_row("Root Prims", str(len(stage_info.root_prims)))
                stage_table.add_row("Default Prim", stage_info.default_prim or "None")

                console.print(stage_table)

                # Generate summary
                console.print("[yellow]Generating scene summary...[/yellow]")
                summary = session.summarize_stage(stage_id, include_meshes=True)

                if summary:
                    console.print("[green]✓ Scene analysis complete[/green]")

                    # Show prim type counts
                    if summary.prim_type_counts:
                        prim_table = Table(title="Prim Types")
                        prim_table.add_column("Type", style="cyan")
                        prim_table.add_column("Count", justify="right")

                        for prim_type, count in sorted(
                            summary.prim_type_counts.items(),
                            key=lambda x: x[1],
                            reverse=True,
                        ):
                            prim_table.add_row(prim_type, str(count))

                        console.print(prim_table)

                    # Show mesh statistics
                    if summary.mesh_statistics:
                        mesh_stats = summary.mesh_statistics
                        console.print(
                            Panel.fit(
                                f"[bold]Mesh Statistics[/bold]\n"
                                f"Total Meshes: {mesh_stats.get('total_meshes', 0)}\n"
                                f"Total Vertices: {mesh_stats.get('total_vertices', 0):,}\n"
                                f"Total Faces: {mesh_stats.get('total_faces', 0):,}\n"
                                f"Meshes with Normals: {mesh_stats.get('meshes_with_normals', 0)}\n"
                                f"Meshes with UVs: {mesh_stats.get('meshes_with_uvs', 0)}\n"
                                f"Closed Meshes: {mesh_stats.get('closed_meshes', 0)}",
                                title="Mesh Analysis",
                            )
                        )
                else:
                    console.print("[yellow]⚠ Could not generate scene summary[/yellow]")
            else:
                console.print("[yellow]⚠ Could not get stage information[/yellow]")

        console.print("[green]✓ USD file test completed successfully[/green]")

    except Exception as e:
        console.print(f"[red]✗ USD file test failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def version():
    """Show version information."""
    try:
        from simul_mcp import __version__

        version_str = __version__
    except ImportError:
        version_str = "unknown"

    isaac_reachable = _is_isaac_reachable()
    console.print(
        Panel.fit(
            f"[bold blue]Simul – 3D Simulation & DCC Tools[/bold blue]\n"
            f"Version: {version_str}\n"
            f"Isaac Sim (TCP :8226): {'✓ reachable' if isaac_reachable else '✗ not reachable'}\n"
            f"Blender: {'✓' if is_blender_available() else '✗'}\n"
            f"USD Headless: {'✓' if is_headless_available() else '✗'}",
            title="Version Information",
        )
    )


if __name__ == "__main__":
    app()
