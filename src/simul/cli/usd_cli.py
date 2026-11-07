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
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import typer
from rich.markup import escape as rich_escape
from rich.panel import Panel
from rich.table import Table

from simul.cli.output import console, emit, emit_error, fail, is_json_mode
from simul.config import get_settings, load_settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from simul.adapters.headless_usd import HeadlessUSDSession

app = typer.Typer(
    name="usd",
    help="USD file commands -- analyse and manipulate USD files locally (no Isaac Sim needed).",
    add_completion=False,
)


def _require_usd() -> None:
    """Exit with an error if headless USD is unavailable."""
    # Imported here: the probe loads pxr, which `simul --help` must not pay for.
    from simul.adapters import is_headless_available

    if not is_headless_available():
        fail("pxr library not available -- cannot perform USD operations.", "DependencyError")


def _load_or_fail(session: "HeadlessUSDSession", file_path: Path) -> str:
    """Load ``file_path`` in ``session`` or exit 1 with pxr's reason."""
    from simul.usd.validation import USDLoadError

    try:
        stage_id: str = session.load_stage_or_raise(file_path)
    except USDLoadError as exc:
        fail(str(exc), exc.error_type, exc.details)
    return stage_id


def _load_warnings(session: "HeadlessUSDSession", stage_id: str) -> List[Dict[str, Any]]:
    """The composition problems a loaded stage carries (missing references, ...)."""
    from simul.usd.validation import WARNING, composition_issues

    stage = session.get_stage(stage_id)
    return [issue.to_dict() for issue in composition_issues(stage, WARNING)] if stage else []


# ---------------------------------------------------------------------------
# info
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

    from simul.adapters import HeadlessUSDAdapter
    from simul.logging import setup_logging

    setup_logging(settings)
    adapter = HeadlessUSDAdapter(settings)

    with adapter.create_session() as session:
        if not is_json_mode():
            console.print(f"[cyan]Loading[/cyan] {file_path}")
        stage_id = _load_or_fail(session, file_path)

        stage_info = session.get_stage_info(stage_id)
        if not stage_info:
            fail("Failed to read stage info", "ReadError")

        summary = session.summarize_stage(stage_id, include_meshes=True)
        warnings = _load_warnings(session, stage_id)

        if is_json_mode():
            data = {
                "file_path": str(file_path),
                "up_axis": stage_info.up_axis,
                "meters_per_unit": stage_info.meters_per_unit,
                "time_codes_per_second": stage_info.time_codes_per_second,
                "start_time_code": stage_info.start_time_code,
                "end_time_code": stage_info.end_time_code,
                "frame_rate": stage_info.frame_rate,
                "total_prims": stage_info.prim_count,
                "root_prims": len(stage_info.root_prims),
                "default_prim": stage_info.default_prim_path,
            }
            if summary:
                data["prim_type_counts"] = summary.prim_type_counts or {}
                data["mesh_statistics"] = summary.mesh_statistics or {}
            data["warnings"] = warnings
            emit(data)
            return

        for warning in warnings:
            console.print(f"[yellow]Warning[/yellow] {rich_escape(warning['message'])}")

        table = Table(title="Stage Information")
        table.add_column("Property", style="cyan")
        table.add_column("Value")
        table.add_row("Up Axis", stage_info.up_axis)
        table.add_row("Meters / Unit", str(stage_info.meters_per_unit))
        table.add_row("Time Codes / s", str(stage_info.time_codes_per_second))
        table.add_row("Start Time", str(stage_info.start_time_code))
        table.add_row("End Time", str(stage_info.end_time_code))
        table.add_row("Frame Rate", str(stage_info.frame_rate))
        table.add_row("Total Prims", str(stage_info.prim_count))
        table.add_row("Root Prims", str(len(stage_info.root_prims)))
        table.add_row("Default Prim", stage_info.default_prim_path or "None")
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
_SEVERITY_STYLE = {"error": "red", "warning": "yellow", "info": "cyan"}


def _location_text(location: Dict[str, Any]) -> str:
    """Render an issue's location for the human table."""
    where = location.get("property_path") or location.get("prim_path") or ""
    if location.get("line") is not None:
        where = f"line {location['line']}, column {location.get('column', '?')}"
    asset = location.get("asset_path")
    if asset and asset not in where:
        where = f"{where} @{asset}@".strip()
    return where


@app.command()
def validate(
    file_path: Path = typer.Argument(
        ...,
        help="Path to USD file to validate",
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Configuration file",
        exists=True, file_okay=True, dir_okay=False,
    ),
) -> None:
    """Open a USD file with pxr and report what is wrong with it.

    Exits 1 when the file is invalid: pxr cannot open it, or a check reports
    an error (a missing sublayer, reference or payload, a dangling
    defaultPrim, a physics error, a pxr validator error). Warnings alone
    leave the file valid.
    """
    _require_usd()
    settings = load_settings(config) if config else get_settings()

    from simul.usd.validation import validate_usd_file

    resolved = str(file_path.expanduser().resolve())
    report = validate_usd_file(resolved, settings.usd)
    data = report.to_dict()
    errors = data["error_count"]

    if is_json_mode():
        # ``valid`` is the key this command reported before issues existed.
        data["valid"] = report.is_valid
        data["success"] = report.is_valid
        if not report.is_valid:
            data["error"] = (
                f"{resolved} is not valid USD: {errors} error(s); see issues"
                if report.can_read
                else f"{resolved} could not be opened; see issues"
            )
            data["error_type"] = "ValidationFailed"
        print(json.dumps(data))
        if not report.is_valid:
            raise typer.Exit(1)
        return

    table = Table(title=f"Validation: {file_path.name}")
    table.add_column("Severity")
    table.add_column("Code", style="cyan")
    table.add_column("Where")
    table.add_column("Message")
    for issue in data["issues"]:
        style = _SEVERITY_STYLE.get(issue["severity"], "white")
        table.add_row(
            f"[{style}]{issue['severity'].upper()}[/{style}]",
            issue["code"],
            rich_escape(_location_text(issue["location"])),
            rich_escape(issue["message"]),
        )
    if data["issues"]:
        console.print(table)

    counts = f"{errors} error(s), {data['warning_count']} warning(s), {data['info_count']} info"
    if report.is_valid:
        console.print(f"[green]File is valid[/green] ({counts})")
    else:
        console.print(f"[red]Validation failed[/red] ({counts})")
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

    from simul.adapters import HeadlessUSDAdapter
    from simul.logging import setup_logging

    setup_logging(settings)
    adapter = HeadlessUSDAdapter(settings)

    with adapter.create_session() as session:
        from simul.usd.validation import USDLoadError

        try:
            stage_id = session.load_stage_or_raise(file_path)
        except USDLoadError as exc:
            if is_json_mode() or format == "json":
                emit_error(str(exc), exc.error_type, exc.details)
            console.print(f"[red]{rich_escape(str(exc))}[/red]")
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
                "success": True,
                "file_path": result.file_path,
                "total_prims": result.total_prims,
                "prim_type_counts": result.prim_type_counts,
                "scene_bbox": result.scene_bbox,
                "scene_center": result.scene_center,
                "scene_size": result.scene_size,
                "hierarchy_depth": result.hierarchy_depth,
                "mesh_statistics": result.mesh_statistics,
            }
            # stdout, not the Rich console (stderr); --json gets one line like
            # every other command, --format json keeps the indented form.
            print(json.dumps(out) if is_json_mode() else json.dumps(out, indent=2))
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
