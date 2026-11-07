"""
Scene summaries of USD stages for LLM consumption.

One walk of the stage (``search.iter_prims``, instance proxies included) fills
every count, and every mesh is measured by ``mesh_ops.geometry_statistics`` in
batches, never face by face: a stage with one million-vertex mesh, or twenty
thousand small ones, summarises in about a second. An instanced mesh is
measured once per prototype and counted once per instance.

The summary also says what its numbers mean: the scene size in metres as well
as stage units, animation from time samples even when no time range is
authored (with its duration in seconds), and a warning for each layer whose
``upAxis`` or ``metersPerUnit`` disagrees with the stage's, since USD does not
rescale or reorient referenced geometry.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ._pxr import Usd, UsdGeom

from ..logging import get_logger, LoggerMixin
from ..utils.timing import monitor_performance
from ..utils.math import bbox_center, bbox_size, bbox_volume
from .bbox import BBoxCache, bbox_to_dict
from .mesh_ops import (
    MeshArrays,
    MeshOperations,
    bound_material,
    geometry_statistics,
    mesh_attributes,
    read_mesh_arrays,
)
from .reader import USDReader, USDStageInfo
from .search import iter_prims

logger = get_logger(__name__)

# Type-count key for prims with no type.
TYPELESS = "(typeless)"

# Meshes measured per geometry_statistics batch, bounded by face-vertex indices.
_BATCH_INDICES = 4_000_000

# How many root prim paths a summary lists (the count is exact).
_LISTED_PATHS = 50


@dataclass
class MaterialSummary:
    """Summary of a USD material."""
    path: str
    name: str
    shader_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    textures: List[str] = field(default_factory=list)


@dataclass
class PrimSummary:
    """Summary of a USD prim for LLM consumption."""
    path: str
    name: str
    type: str
    purpose: Optional[str]
    visibility: Optional[str]
    bbox: Optional[Dict[str, List[float]]]
    bbox_center: Optional[List[float]]
    bbox_size: Optional[List[float]]
    bbox_volume: Optional[float]
    transform: Optional[Dict[str, Any]]
    mesh_info: Optional[Dict[str, Any]]
    material_bindings: List[str] = field(default_factory=list)
    children_count: int = 0
    children_types: Dict[str, int] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneSummary:
    """Summary of a USD scene for LLM consumption."""
    file_path: str
    stage_info: Dict[str, Any]
    total_prims: int
    prim_type_counts: Dict[str, int]
    scene_bbox: Optional[Dict[str, List[float]]]
    scene_center: Optional[List[float]]
    scene_size: Optional[List[float]]
    scene_volume: Optional[float]
    materials: List[MaterialSummary] = field(default_factory=list)
    root_prims: List[PrimSummary] = field(default_factory=list)
    mesh_statistics: Dict[str, Any] = field(default_factory=dict)
    hierarchy_depth: int = 0
    animation_info: Dict[str, Any] = field(default_factory=dict)
    root_prim_paths: List[str] = field(default_factory=list)
    root_prim_count: int = 0
    scene_size_meters: Optional[List[float]] = None
    warnings: List[Dict[str, Any]] = field(default_factory=list)


def unit_warnings(stage: Usd.Stage) -> List[Dict[str, Any]]:
    """Warn about layers whose authored ``upAxis`` or ``metersPerUnit`` differs from the stage's.

    Only the root layer's metrics apply to a stage. A referenced asset or a
    sublayer authored in other units keeps its numbers, so a robot modelled in
    metres inside a centimetre stage is a hundred times too small, and a
    Z-up asset in a Y-up stage lies on its side.
    """
    stage_up = str(UsdGeom.GetStageUpAxis(stage))
    stage_mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
    session = stage.GetSessionLayer()
    skip = {stage.GetRootLayer().identifier, session.identifier if session else None}
    warnings: List[Dict[str, Any]] = []
    for layer in sorted(stage.GetUsedLayers(), key=lambda used: used.identifier):
        if layer.identifier in skip:
            continue
        info = layer.pseudoRoot
        if info.HasInfo("metersPerUnit"):
            layer_mpu = float(info.GetInfo("metersPerUnit"))
            if not math.isclose(layer_mpu, stage_mpu, rel_tol=1e-9):
                factor = layer_mpu / stage_mpu
                look = f"{factor:g}x smaller" if factor > 1 else f"{1 / factor:g}x larger"
                warnings.append({
                    "code": "MetersPerUnitMismatch",
                    "layer": layer.identifier,
                    "layer_value": layer_mpu,
                    "stage_value": stage_mpu,
                    "message": (
                        f"{layer.GetDisplayName()} authors metersPerUnit {layer_mpu:g} but the stage uses "
                        f"{stage_mpu:g}; USD does not rescale it, so its geometry appears {look} than modelled."
                    ),
                })
        if info.HasInfo("upAxis"):
            layer_up = str(info.GetInfo("upAxis"))
            if layer_up != stage_up:
                warnings.append({
                    "code": "UpAxisMismatch",
                    "layer": layer.identifier,
                    "layer_value": layer_up,
                    "stage_value": stage_up,
                    "message": (
                        f"{layer.GetDisplayName()} is {layer_up}-up but the stage is {stage_up}-up; "
                        "USD does not reorient it."
                    ),
                })
    return warnings


class SceneSummarizer(LoggerMixin):
    """
    USD scene summarizer for generating LLM-friendly scene descriptions.

    Provides functionality to analyze USD scenes and generate concise summaries
    suitable for consumption by language models.
    """

    def __init__(self):
        """Initialize scene summarizer."""
        self.usd_reader = USDReader()
        self.mesh_ops = MeshOperations()
        self.logger.debug("Scene summarizer initialized")

    @monitor_performance("summarizer.summarize_stage")
    def summarize_stage(
        self,
        stage: Usd.Stage,
        file_path: str = "",
        include_meshes: bool = True,
        include_materials: bool = True,
        max_depth: int = 5
    ) -> SceneSummary:
        """
        Generate a summary of a USD stage.

        Args:
            stage: USD Stage to summarize
            file_path: Path to the USD file
            include_meshes: Include mesh statistics
            include_materials: Include the material list
            max_depth: Kept for callers that pass it; the summary no longer
                builds per-root prim summaries, which it used to compute and
                then discard

        Returns:
            SceneSummary object
        """
        scan = self._scan_stage(stage, include_meshes=include_meshes, include_materials=include_materials)
        total_prims = scan["total_prims"]
        stage_info = self.usd_reader.get_stage_info(stage, prim_count=total_prims)

        scene_bbox = BBoxCache(stage).get_stage_bbox()
        scene_bbox_dict = scene_center = scene_size = scene_volume = scene_size_meters = None
        if scene_bbox:
            scene_bbox_dict = bbox_to_dict(scene_bbox)
            scene_center = bbox_center(scene_bbox)
            scene_size = bbox_size(scene_bbox)
            scene_volume = bbox_volume(scene_bbox)
            scene_size_meters = [extent * stage_info.meters_per_unit for extent in scene_size]

        summary = SceneSummary(
            file_path=file_path,
            stage_info={
                'up_axis': stage_info.up_axis,
                'meters_per_unit': stage_info.meters_per_unit,
                'time_codes_per_second': stage_info.time_codes_per_second,
                'start_time': stage_info.start_time_code,
                'end_time': stage_info.end_time_code,
                'frame_rate': stage_info.frame_rate,
                'layer_count': len(stage_info.layers),
            },
            total_prims=total_prims,
            prim_type_counts=scan["prim_type_counts"],
            scene_bbox=scene_bbox_dict,
            scene_center=scene_center,
            scene_size=scene_size,
            scene_volume=scene_volume,
            materials=scan["materials"],
            mesh_statistics=scan["mesh_statistics"],
            hierarchy_depth=scan["hierarchy_depth"],
            animation_info=self._get_animation_info(stage_info),
            root_prim_paths=stage_info.root_prims[:_LISTED_PATHS],
            root_prim_count=len(stage_info.root_prims),
            scene_size_meters=scene_size_meters,
            warnings=unit_warnings(stage),
        )
        self.logger.info(f"Generated scene summary: {total_prims} prims, {summary.root_prim_count} root prims")
        return summary

    @monitor_performance("summarizer.summarize_prim")
    def summarize_prim(
        self,
        prim: Usd.Prim,
        bbox_cache: Optional[BBoxCache] = None,
        include_mesh_info: bool = True,
        max_depth: int = 5,
        current_depth: int = 0
    ) -> PrimSummary:
        """
        Generate a summary of a USD prim.

        Args:
            prim: USD Prim to summarize
            bbox_cache: Bounding box cache (created if None)
            include_mesh_info: Include mesh statistics
            max_depth: Maximum recursion depth
            current_depth: Current recursion depth

        Returns:
            PrimSummary object
        """
        prim_info = self.usd_reader.get_prim_info(prim)
        if bbox_cache is None:
            bbox_cache = BBoxCache(prim.GetStage())
        bbox = bbox_cache.compute_world_bbox(prim)

        mesh_info_dict = None
        if include_mesh_info and prim.IsA(UsdGeom.Mesh):
            mesh_info = self.mesh_ops.get_mesh_statistics(prim)
            mesh_info_dict = {
                'vertex_count': mesh_info.vertex_count,
                'face_count': mesh_info.face_count,
                'has_normals': mesh_info.has_normals,
                'has_uvs': mesh_info.has_uvs,
                'has_colors': mesh_info.has_colors,
                'surface_area': mesh_info.surface_area,
                'volume': mesh_info.volume,
                'is_closed': mesh_info.is_closed,
                'topology_valid': mesh_info.topology_valid
            }

        children_types: Dict[str, int] = {}
        if current_depth < max_depth:
            for child in prim.GetChildren():
                child_type = child.GetTypeName() or TYPELESS
                children_types[child_type] = children_types.get(child_type, 0) + 1

        return PrimSummary(
            path=prim_info.path,
            name=prim_info.name,
            type=prim_info.type_name,
            purpose=prim_info.purpose,
            visibility=prim_info.visibility,
            bbox=bbox_to_dict(bbox) if bbox else None,
            bbox_center=bbox_center(bbox) if bbox else None,
            bbox_size=bbox_size(bbox) if bbox else None,
            bbox_volume=bbox_volume(bbox) if bbox else None,
            transform=self._get_transform_info(prim),
            mesh_info=mesh_info_dict,
            material_bindings=self._get_material_bindings(prim),
            children_count=len(prim_info.children),
            children_types=children_types,
            attributes=self._filter_important_attributes(prim_info.attributes),
            metadata=prim_info.metadata,
        )

    @staticmethod
    def _empty_mesh_statistics() -> Dict[str, Any]:
        return {
            'total_meshes': 0,
            'total_vertices': 0,
            'total_faces': 0,
            'meshes_with_normals': 0,
            'meshes_with_uvs': 0,
            'meshes_with_colors': 0,
            'closed_meshes': 0,
            'invalid_topology_meshes': 0,
            'instanced_meshes': 0,
            'total_surface_area': 0.0,
            'total_volume': 0.0
        }

    def _scan_stage(
        self,
        stage: Usd.Stage,
        include_meshes: bool = True,
        include_materials: bool = True,
    ) -> Dict[str, Any]:
        """Walk the stage once and fill every per-prim accumulator.

        The walk is ``search.iter_prims``: the prims every listing counts,
        instance proxies included, so ``total_prims`` agrees with
        ``load_usd_file`` and a match-everything search. Meshes are collected
        during the walk and measured afterwards in batches; an instanced mesh
        is measured once per prototype and counted once per instance.
        """
        prim_type_counts: Dict[str, int] = {}
        total_prims = 0
        materials: List[MaterialSummary] = []
        hierarchy_depth = 0
        # The prim whose data a mesh is -> (that prim, how often it appears).
        meshes: Dict[str, Tuple[Usd.Prim, int]] = {}
        instanced = 0

        for prim in iter_prims(stage):
            total_prims += 1
            prim_type = prim.GetTypeName() or TYPELESS
            prim_type_counts[prim_type] = prim_type_counts.get(prim_type, 0) + 1

            path_str = prim.GetPath().pathString
            hierarchy_depth = max(hierarchy_depth, path_str.count('/'))

            if include_materials and prim_type == "Material":
                materials.append(MaterialSummary(
                    path=path_str,
                    name=prim.GetName(),
                    shader_type="Unknown",  # Would need shader analysis
                    parameters={},
                    textures=[]
                ))

            if include_meshes and prim.IsA(UsdGeom.Mesh):
                source = prim
                if prim.IsInstanceProxy():
                    source = prim.GetPrimInPrototype()
                    instanced += 1
                key = source.GetPath().pathString
                seen = meshes.get(key)
                meshes[key] = (source, seen[1] + 1 if seen else 1)

        mesh_stats: Dict[str, Any] = {}
        if include_meshes:
            mesh_stats = self._measure_meshes(list(meshes.values()))
            mesh_stats['instanced_meshes'] = instanced

        return {
            'total_prims': total_prims,
            'prim_type_counts': prim_type_counts,
            'materials': materials,
            'mesh_statistics': mesh_stats,
            'hierarchy_depth': hierarchy_depth,
        }

    def _measure_meshes(self, meshes: List[Tuple[Usd.Prim, int]]) -> Dict[str, Any]:
        """Fold every mesh's statistics, times how often it appears, into one total."""
        stats = self._empty_mesh_statistics()
        batch: List[Tuple[Usd.Prim, int, MeshArrays]] = []
        batch_indices = 0

        def flush() -> None:
            measured = geometry_statistics([arrays for _prim, _uses, arrays in batch])
            for (prim, uses, arrays), geometry in zip(batch, measured):
                attributes = mesh_attributes(UsdGeom.Mesh(prim))
                stats['total_meshes'] += uses
                stats['total_vertices'] += uses * len(arrays.points)
                stats['total_faces'] += uses * len(arrays.counts)
                stats['meshes_with_normals'] += uses * (attributes.normals_source is not None)
                stats['meshes_with_uvs'] += uses * bool(attributes.uv_sets)
                stats['meshes_with_colors'] += uses * attributes.has_colors
                stats['closed_meshes'] += uses * geometry.is_closed
                stats['invalid_topology_meshes'] += uses * (not geometry.topology_valid)
                stats['total_surface_area'] += uses * (geometry.surface_area or 0.0)
                stats['total_volume'] += uses * (geometry.volume or 0.0)
            batch.clear()

        for prim, uses in meshes:
            arrays = read_mesh_arrays(UsdGeom.Mesh(prim))
            batch.append((prim, uses, arrays))
            batch_indices += len(arrays.indices)
            if batch_indices >= _BATCH_INDICES:
                flush()
                batch_indices = 0
        flush()
        return stats

    def _get_animation_info(self, stage_info: USDStageInfo) -> Dict[str, Any]:
        """Describe the stage's animation: whether, over which time codes, for how long.

        A stage animates when it authors a time range or when anything in it
        is time-sampled. ``duration_seconds`` is the time-code span divided by
        ``timeCodesPerSecond``: the authored range when there is one, else the
        span of the samples.
        """
        tcps = stage_info.time_codes_per_second
        first: Optional[float] = None
        last: Optional[float] = None
        source: Optional[str] = None
        if stage_info.authored_time_range and stage_info.end_time_code > stage_info.start_time_code:
            first, last, source = stage_info.start_time_code, stage_info.end_time_code, "authored_range"
        elif stage_info.time_sample_range is not None:
            first, last = stage_info.time_sample_range
            source = "time_samples"
        span = (last - first) if first is not None and last is not None else 0.0
        return {
            'has_animation': stage_info.has_animation,
            'source': source,
            'start_time': stage_info.start_time_code,
            'end_time': stage_info.end_time_code,
            'authored_time_range': stage_info.authored_time_range,
            'time_sample_range': stage_info.time_sample_range,
            'duration_time_codes': span,
            'duration_seconds': span / tcps if tcps else None,
            'frame_rate': stage_info.frame_rate,
            'time_codes_per_second': tcps,
        }

    def _get_transform_info(self, prim: Usd.Prim) -> Optional[Dict[str, Any]]:
        """Get the local translation, rotation ([w, x, y, z]) and scale of a prim."""
        if not prim.IsA(UsdGeom.Xformable):
            return None
        transform = UsdGeom.Xformable(prim).GetLocalTransformation()
        translation = transform.ExtractTranslation()
        rotation = transform.ExtractRotationQuat()
        imaginary = rotation.GetImaginary()
        return {
            'translation': [translation[0], translation[1], translation[2]],
            'rotation': [rotation.GetReal(), imaginary[0], imaginary[1], imaginary[2]],
            'scale': [transform.GetRow3(axis).GetLength() for axis in range(3)],
        }

    def _get_material_bindings(self, prim: Usd.Prim) -> List[str]:
        """The material bound to a prim, as ``UsdShade.MaterialBindingAPI`` resolves it."""
        material = bound_material(prim)
        return [material] if material else []

    def _filter_important_attributes(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Keep the attributes that matter in a summary; values are already bounded records."""
        important_names = [
            'points', 'normals', 'primvars:st', 'displayColor',
            'size', 'radius', 'height', 'width', 'depth',
            'focalLength', 'horizontalAperture', 'verticalAperture',
            'intensity', 'color', 'temperature', 'exposure'
        ]
        return {
            name: value
            for name, value in attributes.items()
            if any(important in name for important in important_names)
        }


def generate_scene_digest(summary: SceneSummary) -> str:
    """
    Generate a concise text digest of the scene summary.

    Args:
        summary: SceneSummary object

    Returns:
        Human-readable scene digest
    """
    digest_parts = [
        f"USD Scene: {summary.file_path}",
        f"Total prims: {summary.total_prims}",
    ]
    info = summary.stage_info
    if info:
        digest_parts.append(f"Stage: {info.get('up_axis')}-up, {info.get('meters_per_unit'):g} m per unit")

    if summary.scene_size:
        size = summary.scene_size
        line = f"Scene size: {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} units"
        if summary.scene_size_meters:
            meters = summary.scene_size_meters
            line += f" ({meters[0]:.3g} x {meters[1]:.3g} x {meters[2]:.3g} m)"
        digest_parts.append(line)

    if summary.prim_type_counts:
        top_types = sorted(summary.prim_type_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        type_str = ", ".join([f"{count} {ptype}" for ptype, count in top_types])
        digest_parts.append(f"Main content: {type_str}")

    if summary.mesh_statistics and summary.mesh_statistics.get('total_meshes', 0) > 0:
        mesh_stats = summary.mesh_statistics
        digest_parts.append(f"Meshes: {mesh_stats['total_meshes']} total, "
                            f"{mesh_stats['total_vertices']} vertices, "
                            f"{mesh_stats['total_faces']} faces")

    if summary.materials:
        digest_parts.append(f"Materials: {len(summary.materials)}")

    anim_info = summary.animation_info
    if anim_info.get('has_animation', False):
        seconds = anim_info.get('duration_seconds')
        span = anim_info.get('duration_time_codes', 0.0)
        line = "Animation: " + (f"{seconds:.2f} s" if seconds is not None else f"{span:g} time codes")
        line += (
            f" ({span:g} time codes at {anim_info.get('time_codes_per_second'):g} per second, "
            f"{anim_info.get('frame_rate'):g} FPS"
        )
        if anim_info.get('source') == "time_samples":
            line += "; from time samples, no time range authored"
        digest_parts.append(line + ")")

    for warning in summary.warnings:
        digest_parts.append(f"Warning: {warning['message']}")

    return "\n".join(digest_parts)


def format_summary_for_llm(summary: SceneSummary) -> str:
    """
    Format scene summary for LLM consumption.

    Args:
        summary: SceneSummary object

    Returns:
        JSON string formatted for LLM
    """
    llm_summary = {
        'scene_overview': {
            'file_path': summary.file_path,
            'total_prims': summary.total_prims,
            'prim_types': summary.prim_type_counts,
            'scene_bounds': summary.scene_bbox,
            'scene_center': summary.scene_center,
            'scene_size': summary.scene_size,
            'scene_size_meters': summary.scene_size_meters,
        },
        'content_summary': {
            'root_prims': summary.root_prim_count,
            'materials': len(summary.materials),
            'hierarchy_depth': summary.hierarchy_depth,
            'has_animation': summary.animation_info.get('has_animation', False)
        },
        'mesh_statistics': summary.mesh_statistics,
        'stage_settings': summary.stage_info,
        'warnings': summary.warnings,
    }

    return json.dumps(llm_summary, indent=2)


# Convenience functions
def summarize_stage(stage: Usd.Stage, file_path: str = "") -> SceneSummary:
    """Summarize a USD stage."""
    summarizer = SceneSummarizer()
    return summarizer.summarize_stage(stage, file_path)


def summarize_prim(prim: Usd.Prim) -> PrimSummary:
    """Summarize a USD prim."""
    summarizer = SceneSummarizer()
    return summarizer.summarize_prim(prim)
