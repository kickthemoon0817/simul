#!/usr/bin/env python3
"""
Sample USD Reader Example for Isaac Sim MCP Server.

This example demonstrates how to use the USD reader functionality
to load and analyze USD files without requiring Isaac Sim runtime.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

# Add src to path
script_dir = Path(__file__).resolve().parent
src_path = script_dir.parents[1] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.config import get_settings
from simul_mcp.logging import setup_logging, get_logger
from simul_mcp.adapters import HeadlessUSDAdapter, is_headless_available
from simul_mcp.usd.summarize import generate_scene_digest, format_summary_for_llm


def analyze_usd_file(file_path: str, verbose: bool = False) -> bool:
    """
    Analyze a USD file and print information about it.

    Args:
        file_path: Path to USD file
        verbose: Enable verbose output

    Returns:
        True if analysis was successful
    """
    logger = get_logger(__name__)

    try:
        # Check if file exists
        path = Path(file_path)
        if not path.exists():
            print(f"Error: File not found: {file_path}")
            return False

        if not path.is_file():
            print(f"Error: Path is not a file: {file_path}")
            return False

        # Check USD availability
        if not is_headless_available():
            print(
                "Error: USD support not available. Please install USD Python bindings."
            )
            return False

        print(f"Analyzing USD file: {file_path}")
        print("=" * 60)

        # Create adapter and session
        settings = get_settings()
        adapter = HeadlessUSDAdapter(settings)

        with adapter.create_session() as session:
            # Load the stage
            print("Loading USD stage...")
            stage_id = session.load_stage(file_path)

            if not stage_id:
                print("Error: Failed to load USD file")
                return False

            print(f"✓ Successfully loaded stage (ID: {stage_id})")
            print()

            # Get stage information
            print("Stage Information:")
            print("-" * 20)
            stage_info = session.get_stage_info(stage_id)

            if stage_info:
                print(f"Up Axis: {stage_info.up_axis}")
                print(f"Meters per Unit: {stage_info.meters_per_unit}")
                print(f"Time Codes per Second: {stage_info.time_codes_per_second}")
                print(f"Start Time: {stage_info.start_time_code}")
                print(f"End Time: {stage_info.end_time_code}")
                print(f"Frame Rate: {stage_info.frame_rate}")
                print(f"Total Prims: {len(stage_info.all_prims)}")
                print(f"Root Prims: {len(stage_info.root_prims)}")
                print(f"Layers: {len(stage_info.layers)}")
                print(f"Default Prim: {stage_info.default_prim or 'None'}")

                has_animation = stage_info.start_time_code != stage_info.end_time_code
                print(f"Has Animation: {has_animation}")

                if verbose:
                    print(f"\nRoot Prims: {', '.join(stage_info.root_prims)}")
                    print(f"All Prims: {len(stage_info.all_prims)} total")
                    if len(stage_info.all_prims) <= 20:
                        for prim_path in stage_info.all_prims:
                            print(f"  - {prim_path}")
                    else:
                        print("  (too many to list)")
            else:
                print("Could not get stage information")

            print()

            # Find different prim types
            print("Prim Type Analysis:")
            print("-" * 20)

            prim_types = [
                "Mesh",
                "Xform",
                "Sphere",
                "Cube",
                "Cylinder",
                "Camera",
                "Light",
                "Material",
            ]
            for prim_type in prim_types:
                prims = session.find_prims_by_type(stage_id, prim_type)
                if prims:
                    print(f"{prim_type}: {len(prims)}")
                    if verbose and len(prims) <= 10:
                        for prim_path in prims:
                            print(f"  - {prim_path}")

            print()

            # Get stage bounding box
            print("Scene Bounds:")
            print("-" * 15)
            stage_bbox = session.get_stage_bbox(stage_id)
            if stage_bbox:
                min_pt = stage_bbox["min"]
                max_pt = stage_bbox["max"]
                size = [max_pt[i] - min_pt[i] for i in range(3)]
                center = [(min_pt[i] + max_pt[i]) / 2 for i in range(3)]

                print(f"Min: [{min_pt[0]:.3f}, {min_pt[1]:.3f}, {min_pt[2]:.3f}]")
                print(f"Max: [{max_pt[0]:.3f}, {max_pt[1]:.3f}, {max_pt[2]:.3f}]")
                print(f"Size: [{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}]")
                print(f"Center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
            else:
                print("Could not compute scene bounds")

            print()

            # Generate scene summary
            print("Scene Summary:")
            print("-" * 15)
            summary = session.summarize_stage(stage_id, include_meshes=True)

            if summary:
                # Generate human-readable digest
                digest = generate_scene_digest(summary)
                print(digest)

                # Show mesh statistics if available
                if (
                    summary.mesh_statistics
                    and summary.mesh_statistics.get("total_meshes", 0) > 0
                ):
                    print("\nMesh Statistics:")
                    mesh_stats = summary.mesh_statistics
                    print(f"Total Meshes: {mesh_stats['total_meshes']}")
                    print(f"Total Vertices: {mesh_stats['total_vertices']:,}")
                    print(f"Total Faces: {mesh_stats['total_faces']:,}")
                    print(f"Meshes with Normals: {mesh_stats['meshes_with_normals']}")
                    print(f"Meshes with UVs: {mesh_stats['meshes_with_uvs']}")
                    print(f"Meshes with Colors: {mesh_stats['meshes_with_colors']}")
                    print(f"Closed Meshes: {mesh_stats['closed_meshes']}")
                    print(f"Total Surface Area: {mesh_stats['total_surface_area']:.3f}")
                    print(f"Total Volume: {mesh_stats['total_volume']:.3f}")

                # Show prim type breakdown
                if summary.prim_type_counts:
                    print("\nPrim Type Breakdown:")
                    sorted_types = sorted(
                        summary.prim_type_counts.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    for prim_type, count in sorted_types:
                        print(f"  {prim_type}: {count}")

                if verbose:
                    print("\nLLM-Formatted Summary:")
                    print("-" * 25)
                    llm_summary = format_summary_for_llm(summary)
                    print(llm_summary)
            else:
                print("Could not generate scene summary")

            print()

            # Analyze specific meshes if requested
            if verbose:
                mesh_prims = session.find_prims_by_type(stage_id, "Mesh")
                if mesh_prims:
                    print("Detailed Mesh Analysis:")
                    print("-" * 25)

                    for i, mesh_path in enumerate(mesh_prims[:5]):  # Limit to first 5
                        print(f"\nMesh {i + 1}: {mesh_path}")

                        # Get mesh info
                        mesh_info = session.get_mesh_info(stage_id, mesh_path)
                        if mesh_info:
                            print(f"  Vertices: {mesh_info['vertex_count']:,}")
                            print(f"  Faces: {mesh_info['face_count']:,}")
                            print(f"  Has Normals: {mesh_info['has_normals']}")
                            print(f"  Has UVs: {mesh_info['has_uvs']}")
                            print(f"  Has Colors: {mesh_info['has_colors']}")
                            print(f"  Surface Area: {mesh_info['surface_area']:.3f}")
                            print(f"  Volume: {mesh_info['volume']:.3f}")
                            print(f"  Is Closed: {mesh_info['is_closed']}")
                            print(f"  Topology Valid: {mesh_info['topology_valid']}")

                        # Get bounding box
                        bbox = session.get_prim_bbox(stage_id, mesh_path)
                        if bbox:
                            min_pt = bbox["min"]
                            max_pt = bbox["max"]
                            size = [max_pt[i] - min_pt[i] for i in range(3)]
                            print(
                                f"  Bounding Box Size: [{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}]"
                            )

                    if len(mesh_prims) > 5:
                        print(f"\n... and {len(mesh_prims) - 5} more meshes")

        print("\n" + "=" * 60)
        print("✓ Analysis completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error analyzing USD file: {e}")
        print(f"Error: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Analyze USD files using Isaac Sim MCP Server components",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sample_usd_reader.py scene.usd
  python sample_usd_reader.py scene.usd --verbose
  python sample_usd_reader.py scene.usd --log-level DEBUG
        """,
    )

    parser.add_argument("file_path", help="Path to USD file to analyze")

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output with detailed analysis",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level",
    )

    args = parser.parse_args()

    # Setup logging
    settings = get_settings()
    settings.logging.level = args.log_level
    setup_logging(settings.logging)

    # Analyze the file
    success = analyze_usd_file(args.file_path, args.verbose)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
