"""
USD headless CLI sub-commands.

Provides terminal access to USD file operations using the headless adapter.
These commands work without a running Isaac Sim instance -- only the pxr
library is required.

All commands support ``--json`` (or auto-detect via TTY) for structured
JSON output on stdout.
"""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from simul_mcp.adapters import is_headless_available
from simul_mcp.cli.output import emit, emit_error, is_json_mode
from simul_mcp.config import get_settings, load_settings

app = typer.Typer(
    name="usd",
    help="USD file commands -- analyse and manipulate USD files locally (no Isaac Sim needed).",
    add_completion=False,
)
console = Console(stderr=True)


def _require_usd() -> None:
    """Exit with an error if headless USD is unavailable."""
    if not is_headless_available():
        if is_json_mode():
            emit_error(
                "pxr library not available -- cannot perform USD operations.",
                "DependencyError",
            )
        console.print("[red]Error: pxr library not available -- cannot perform USD operations.[/red]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# info (replaces the old top-level test-usd command)
# ---------------------------------------------------------------------------
@app.command()
def info(
    file_path: Path = typer.Argument(
        ...,
        help="Path to USD file to analyse",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Configuration file",
        exists=True, file_okay=True, dir_okay=False,
    ),
) -> None:
    """Analyse a USD file -- stage metadata, prim counts, and mesh statistics."""
    _require_usd()
    settings = load_settings(config) if config else get_settings()

    from simul_mcp.adapters import HeadlessUSDAdapter
    from simul_mcp.logging import setup_logging

    setup_logging(settings)
    adapter = HeadlessUSDAdapter(settings)

    with adapter.create_session() as session:
        if not is_json_mode():
            console.print(f"[cyan]Loading[/cyan] {file_path}")
        stage_id = session.load_stage(file_path)
        if not stage_id:
            if is_json_mode():
                emit_error("Failed to load USD file", "LoadError", {"file_path": str(file_path)})
            console.print("[red]Failed to load USD file[/red]")
            raise typer.Exit(1)

        stage_info = session.get_stage_info(stage_id)
        if not stage_info:
            if is_json_mode():
                emit_error("Failed to read stage info", "ReadError")
            console.print("[red]Failed to read stage info[/red]")
            raise typer.Exit(1)

        summary = session.summarize_stage(stage_id, include_meshes=True)

        if is_json_mode():
            data = {
                "file_path": str(file_path),
                "up_axis": stage_info.up_axis,
                "meters_per_unit": stage_info.meters_per_unit,
                "time_codes_per_second": stage_info.time_codes_per_second,
                "start_time_code": stage_info.start_time_code,
                "end_time_code": stage_info.end_time_code,
                "frame_rate": stage_info.frame_rate,
                "total_prims": len(stage_info.all_prims),
                "root_prims": len(stage_info.root_prims),
                "default_prim": stage_info.default_prim,
            }
            if summary:
                data["prim_type_counts"] = summary.prim_type_counts or {}
                data["mesh_statistics"] = summary.mesh_statistics or {}
            emit(data)
            return

        table = Table(title="Stage Information")
        table.add_column("Property", style="cyan")
        table.add_column("Value")
        table.add_row("Up Axis", stage_info.up_axis)
        table.add_row("Meters / Unit", str(stage_info.meters_per_unit))
        table.add_row("Time Codes / s", str(stage_info.time_codes_per_second))
        table.add_row("Start Time", str(stage_info.start_time_code))
        table.add_row("End Time", str(stage_info.end_time_code))
        table.add_row("Frame Rate", str(stage_info.frame_rate))
        table.add_row("Total Prims", str(len(stage_info.all_prims)))
        table.add_row("Root Prims", str(len(stage_info.root_prims)))
        table.add_row("Default Prim", stage_info.default_prim or "None")
        console.print(table)

        if summary and summary.prim_type_counts:
            prim_table = Table(title="Prim Types")
            prim_table.add_column("Type", style="cyan")
            prim_table.add_column("Count", justify="right")
            for prim_type, count in sorted(
                summary.prim_type_counts.items(), key=lambda x: x[1], reverse=True
            ):
                prim_table.add_row(prim_type, str(count))
            console.print(prim_table)

        if summary and summary.mesh_statistics:
            ms = summary.mesh_statistics
            console.print(
                Panel.fit(
                    f"Total Meshes: {ms.get('total_meshes', 0)}\n"
                    f"Total Vertices: {ms.get('total_vertices', 0):,}\n"
                    f"Total Faces: {ms.get('total_faces', 0):,}\n"
                    f"With Normals: {ms.get('meshes_with_normals', 0)}\n"
                    f"With UVs: {ms.get('meshes_with_uvs', 0)}",
                    title="Mesh Statistics",
                )
            )

    if not is_json_mode():
        console.print("[green]Done[/green]")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------
@app.command()
def validate(
    file_path: Path = typer.Argument(
        ...,
        help="Path to USD file to validate",
    ),
) -> None:
    """Validate a USD file without fully loading it."""
    valid_extensions = {".usd", ".usda", ".usdc", ".usdz"}
    path = Path(file_path)

    checks: list[tuple[str, bool, str]] = [
        ("Exists", path.exists(), "File not found"),
        ("Is file", path.is_file() if path.exists() else False, "Not a regular file"),
        ("Extension", path.suffix.lower() in valid_extensions, f"Expected one of {valid_extensions}"),
    ]

    if path.exists() and path.is_file():
        size_mb = path.stat().st_size / (1024 * 1024)
        settings = get_settings()
        max_mb = settings.usd.max_file_size_mb
        checks.append(("Size", size_mb <= max_mb, f"{size_mb:.1f} MB exceeds {max_mb} MB limit"))

    all_passed = all(passed for _, passed, _ in checks)

    if is_json_mode():
        results = []
        for name, passed, detail in checks:
            entry = {"check": name, "passed": passed}
            if not passed:
                entry["detail"] = detail
            results.append(entry)
        emit({
            "file_path": str(path),
            "valid": all_passed,
            "checks": results,
        })
        if not all_passed:
            raise typer.Exit(1)
        return

    table = Table(title=f"Validation: {path.name}")
    table.add_column("Check", style="cyan")
    table.add_column("Result")
    table.add_column("Detail")

    for name, passed, detail in checks:
        symbol = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        table.add_row(name, symbol, "" if passed else detail)
    console.print(table)

    if all_passed:
        console.print("[green]File is valid[/green]")
    else:
        console.print("[red]Validation failed[/red]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
@app.command()
def summary(
    file_path: Path = typer.Argument(
        ...,
        help="Path to USD file",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    format: str = typer.Option("text", "--format", "-f", help="Output format: text or json"),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Configuration file",
        exists=True, file_okay=True, dir_okay=False,
    ),
) -> None:
    """Generate a comprehensive scene summary for a USD file."""
    _require_usd()
    settings = load_settings(config) if config else get_settings()

    from simul_mcp.adapters import HeadlessUSDAdapter
    from simul_mcp.logging import setup_logging

    setup_logging(settings)
    adapter = HeadlessUSDAdapter(settings)

    with adapter.create_session() as session:
        stage_id = session.load_stage(file_path)
        if not stage_id:
            if is_json_mode() or format == "json":
                emit_error("Failed to load USD file", "LoadError", {"file_path": str(file_path)})
            console.print("[red]Failed to load USD file[/red]")
            raise typer.Exit(1)

        result = session.summarize_stage(stage_id, include_meshes=True)
        if not result:
            if is_json_mode() or format == "json":
                emit_error("Failed to generate summary", "SummaryError")
            console.print("[red]Failed to generate summary[/red]")
            raise typer.Exit(1)

        # JSON output: --json flag OR --format json
        if is_json_mode() or format == "json":
            out = {
                "file_path": result.file_path,
                "total_prims": result.total_prims,
                "prim_type_counts": result.prim_type_counts,
                "scene_bbox": result.scene_bbox,
                "scene_center": result.scene_center,
                "scene_size": result.scene_size,
                "hierarchy_depth": result.hierarchy_depth,
                "mesh_statistics": result.mesh_statistics,
            }
            # BUG FIX: write JSON to stdout, not stderr via Rich console
            print(json.dumps(out, indent=2))
            return

        console.print(Panel.fit(
            f"[bold]{file_path.name}[/bold]\n"
            f"Total Prims: {result.total_prims}\n"
            f"Hierarchy Depth: {result.hierarchy_depth}\n"
            f"Scene Center: {result.scene_center}\n"
            f"Scene Size: {result.scene_size}",
            title="Scene Summary",
        ))

        if result.prim_type_counts:
            table = Table(title="Prim Types")
            table.add_column("Type", style="cyan")
            table.add_column("Count", justify="right")
            for prim_type, count in sorted(
                result.prim_type_counts.items(), key=lambda x: x[1], reverse=True
            ):
                table.add_row(prim_type, str(count))
            console.print(table)

        if result.mesh_statistics:
            ms = result.mesh_statistics
            console.print(Panel.fit(
                f"Meshes: {ms.get('total_meshes', 0)}\n"
                f"Vertices: {ms.get('total_vertices', 0):,}\n"
                f"Faces: {ms.get('total_faces', 0):,}",
                title="Mesh Statistics",
            ))

    console.print("[green]Done[/green]")
