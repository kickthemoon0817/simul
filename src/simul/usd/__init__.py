"""
USD (Universal Scene Description) utilities for Isaac Sim MCP Server.

This package provides core USD reading, scene analysis, mesh operations,
and bounding box utilities using the pxr (Pixar USD) library.
"""

from .reader import (
    USDReader,
    USDStageInfo,
    USDPrimInfo,
    USDLayerInfo,
    read_usd_file,
    get_stage_info,
    get_prim_info,
    traverse_stage,
    find_prims_by_type,
    find_prims_by_name,
)

from .mesh_ops import (
    MeshInfo,
    MeshOperations,
    extract_mesh_data,
    get_mesh_statistics,
    decimate_mesh,
    validate_mesh_topology,
    compute_mesh_normals,
    export_mesh_data,
)

from .bbox import (
    BBoxCache,
    compute_world_bbox,
    compute_local_bbox,
    get_prim_bbox,
    get_stage_bbox,
    bbox_to_dict,
    dict_to_bbox,
)

from .summarize import (
    SceneSummarizer,
    SceneSummary,
    PrimSummary,
    MaterialSummary,
    summarize_stage,
    summarize_prim,
    generate_scene_digest,
    format_summary_for_llm,
)

__all__ = [
    # Reader
    "USDReader",
    "USDStageInfo", 
    "USDPrimInfo",
    "USDLayerInfo",
    "read_usd_file",
    "get_stage_info",
    "get_prim_info",
    "traverse_stage",
    "find_prims_by_type",
    "find_prims_by_name",
    
    # Mesh operations
    "MeshInfo",
    "MeshOperations",
    "extract_mesh_data",
    "get_mesh_statistics",
    "decimate_mesh",
    "validate_mesh_topology",
    "compute_mesh_normals",
    "export_mesh_data",
    
    # Bounding box
    "BBoxCache",
    "compute_world_bbox",
    "compute_local_bbox", 
    "get_prim_bbox",
    "get_stage_bbox",
    "bbox_to_dict",
    "dict_to_bbox",
    
    # Summarization
    "SceneSummarizer",
    "SceneSummary",
    "PrimSummary",
    "MaterialSummary", 
    "summarize_stage",
    "summarize_prim",
    "generate_scene_digest",
    "format_summary_for_llm",
]
