"""
Core USD Reader for Isaac Sim MCP Server.

This module provides the main USD reading functionality using the pxr library
for loading USD files, traversing scene graphs, and extracting prim information.
"""

from __future__ import annotations

import os
from pathlib import Path
import fnmatch
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from ._pxr import Gf, Pcp, Sdf, Usd, UsdGeom, UsdShade

from ..logging import get_logger, LoggerMixin
from ..utils.timing import Timer, monitor_performance
from .bbox import describe_bounds
from .search import count_prims, traversal_predicate
from .timecode import resolve_time_code, time_code_value
from .values import DEFAULT_ARRAY_PREVIEW, describe_attribute, to_json

logger = get_logger(__name__)


class PrimType(Enum):
    """Enumeration of common USD prim types."""
    UNKNOWN = "unknown"
    XFORM = "Xform"
    MESH = "Mesh"
    SPHERE = "Sphere"
    CUBE = "Cube"
    CYLINDER = "Cylinder"
    CONE = "Cone"
    PLANE = "Plane"
    CAMERA = "Camera"
    LIGHT = "Light"
    MATERIAL = "Material"
    SHADER = "Shader"
    SCOPE = "Scope"


@dataclass
class USDLayerInfo:
    """Information about a USD layer."""
    identifier: str
    display_name: str
    real_path: str
    file_format: str
    version: str
    comment: str
    documentation: str
    is_anonymous: bool
    is_dirty: bool
    permission: str


@dataclass
class USDPrimInfo:
    """Information about a USD prim."""
    path: str
    name: str
    type_name: str
    prim_type: PrimType
    is_active: bool
    is_loaded: bool
    is_defined: bool
    is_abstract: bool
    is_instance: bool
    is_prototype: bool
    has_authored_references: bool
    has_authored_payloads: bool
    has_authored_inherits: bool
    has_authored_specializes: bool
    kind: Optional[str]
    purpose: Optional[str]
    visibility: Optional[str]
    attributes: Dict[str, Any] = field(default_factory=dict)
    relationships: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List[str] = field(default_factory=list)
    parent: Optional[str] = None
    api_schemas: List[str] = field(default_factory=list)


@dataclass
class USDStageInfo:
    """Information about a USD stage."""
    root_layer_path: str
    session_layer_path: Optional[str]
    layers: List[USDLayerInfo]
    default_prim_path: Optional[str]
    up_axis: str
    meters_per_unit: float
    time_codes_per_second: float
    start_time_code: float
    end_time_code: float
    frame_rate: float
    prim_count: int
    root_prims: List[str]
    stage_variables: Dict[str, Any] = field(default_factory=dict)


_SPECIFIERS = {
    Sdf.SpecifierDef: "def",
    Sdf.SpecifierOver: "over",
    Sdf.SpecifierClass: "class",
}

# Authored metadata reported under its own key, so left out of ``authored``.
_REPORTED_METADATA = frozenset({
    "specifier", "typeName", "kind", "apiSchemas", "references", "payload",
    "inheritPaths", "specializes", "variantSets", "variantSelection",
    "instanceable", "active",
})


def _children(prim: Usd.Prim) -> List[Usd.Prim]:
    """The children every listing counts, instance proxies included."""
    return list(prim.GetFilteredChildren(traversal_predicate()))


def _arc_key(item: Any) -> str:
    return repr(item)


def describe_composition(prim: Usd.Prim) -> Dict[str, List[Dict[str, Any]]]:
    """List the composition arcs authored on a prim, and whether each composed.

    Arcs come from the prim stack (the specs that contribute opinions), so a
    reference whose asset is missing is still reported, with ``resolved``
    False. Arcs a prim only inherits from an ancestor (a prim inside a
    referenced asset) are not repeated here; ``opinion_layers`` in the
    metadata shows where its opinions come from.

    Returns:
        ``references`` and ``payloads`` (``asset_path``, ``prim_path``,
        ``layer`` it was authored in, ``resolved``), ``inherits`` and
        ``specializes`` (``prim_path``, ``layer``). Empty kinds are omitted.
    """
    composed = set()
    for arc in Usd.PrimCompositionQuery(prim).GetCompositionArcs():
        if arc.GetArcType() not in (Pcp.ArcTypeReference, Pcp.ArcTypePayload) or arc.IsAncestral():
            continue
        if arc.HasSpecs():
            composed.add(_arc_key(arc.GetIntroducingListEditor()[1]))

    arcs: Dict[str, List[Dict[str, Any]]] = {}
    for spec in prim.GetPrimStack():
        layer = spec.layer.identifier
        for kind, items in (("references", spec.referenceList), ("payloads", spec.payloadList)):
            for item in items.GetAddedOrExplicitItems():
                arcs.setdefault(kind, []).append({
                    "asset_path": item.assetPath or None,
                    "prim_path": item.primPath.pathString or None,
                    "layer": layer,
                    "resolved": _arc_key(item) in composed,
                })
        for kind, items in (("inherits", spec.inheritPathList), ("specializes", spec.specializesList)):
            for path in items.GetAddedOrExplicitItems():
                arcs.setdefault(kind, []).append({"prim_path": path.pathString, "layer": layer})
    return arcs


def describe_metadata(prim: Usd.Prim) -> Dict[str, Any]:
    """Report a prim's specifier, instancing, composition, variants and authored metadata."""
    variant_sets = prim.GetVariantSets()
    metadata: Dict[str, Any] = {
        "specifier": _SPECIFIERS.get(prim.GetSpecifier(), str(prim.GetSpecifier())),
        "instanceable": prim.IsInstanceable(),
        "composition": describe_composition(prim),
        "variant_sets": {
            name: {
                "selection": variant_sets.GetVariantSet(name).GetVariantSelection() or None,
                "variants": list(variant_sets.GetVariantSet(name).GetVariantNames()),
            }
            for name in variant_sets.GetNames()
        },
        "opinion_layers": list(dict.fromkeys(spec.layer.identifier for spec in prim.GetPrimStack())),
        "authored": {
            key: to_json(value)
            for key, value in prim.GetAllAuthoredMetadata().items()
            if key not in _REPORTED_METADATA
        },
    }
    return metadata


def describe_relationships(prim: Usd.Prim) -> Dict[str, Dict[str, Any]]:
    """Map each relationship to its targets and whether they are authored."""
    return {
        rel.GetName(): {
            "targets": [target.pathString for target in rel.GetTargets()],
            "authored": bool(rel.HasAuthoredTargets()),
        }
        for rel in prim.GetRelationships()
    }


def _local_transform(prim: Usd.Prim, time: Usd.TimeCode) -> Optional[Dict[str, List[float]]]:
    if not prim.IsA(UsdGeom.Xformable):
        return None
    matrix = UsdGeom.Xformable(prim).GetLocalTransformation(time)
    transform = Gf.Transform(matrix)
    translation = transform.GetTranslation()
    rotation = transform.GetRotation().GetQuat()
    imaginary = rotation.GetImaginary()
    scale = transform.GetScale()
    return {
        "translation": [float(c) for c in translation],
        "rotation": [float(rotation.GetReal())] + [float(c) for c in imaginary],
        "scale": [float(c) for c in scale],
    }


def _bound_material(prim: Usd.Prim) -> List[str]:
    material, _relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    return [str(material.GetPath())] if material else []


def describe_prim(
    prim: Usd.Prim,
    *,
    time_code: Optional[Any] = None,
    purposes: Optional[Iterable[str]] = None,
    array_preview: int = DEFAULT_ARRAY_PREVIEW,
    attribute_filter: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Describe one prim as JSON: state, bounds, transform, properties and metadata.

    Args:
        prim: The prim.
        time_code: Time to evaluate values, transforms and bounds at; None
            for the default time.
        purposes: Purposes the bound counts; None for default and render.
        array_preview: Leading elements of each array attribute to include.
        attribute_filter: Glob patterns (``physics:*``); only attributes
            whose name matches one are listed. None lists them all.

    Returns:
        The fields of the ``PrimInfo`` response.
    """
    time = resolve_time_code(time_code)
    patterns = list(attribute_filter) if attribute_filter is not None else None
    attributes = prim.GetAttributes()
    listed = [
        attr
        for attr in attributes
        if patterns is None or any(fnmatch.fnmatchcase(attr.GetName(), pattern) for pattern in patterns)
    ]
    purpose = visibility = computed_purpose = computed_visibility = None
    if prim.IsA(UsdGeom.Imageable):
        imageable = UsdGeom.Imageable(prim)
        purpose = imageable.GetPurposeAttr().Get(time)
        visibility = imageable.GetVisibilityAttr().Get(time)
        computed_purpose = imageable.ComputePurpose()
        computed_visibility = imageable.ComputeVisibility(time)
    children_types: Dict[str, int] = {}
    children = _children(prim)
    for child in children:
        child_type = child.GetTypeName() or "(typeless)"
        children_types[child_type] = children_types.get(child_type, 0) + 1
    bounds = describe_bounds(prim.GetStage(), prim, purposes=purposes, time_code=time)
    info: Dict[str, Any] = {
        "path": prim.GetPath().pathString,
        "name": prim.GetName(),
        "type": prim.GetTypeName(),
        "is_active": prim.IsActive(),
        "is_loaded": prim.IsLoaded(),
        "is_defined": prim.IsDefined(),
        "is_instance": prim.IsInstance(),
        "is_instance_proxy": prim.IsInstanceProxy(),
        "purpose": purpose,
        "computed_purpose": computed_purpose,
        "visibility": visibility,
        "computed_visibility": computed_visibility,
        "kind": Usd.ModelAPI(prim).GetKind() or None,
        "api_schemas": list(prim.GetAppliedSchemas()),
        "bbox": bounds["bbox"],
        "bbox_purposes": bounds["included_purposes"],
        "transform": _local_transform(prim, time),
        "children_count": len(children),
        "children_types": children_types,
        "material_bindings": _bound_material(prim),
        "attribute_count": len(attributes),
        "attributes": {attr.GetName(): describe_attribute(attr, time, array_preview) for attr in listed},
        "relationships": describe_relationships(prim),
        "metadata": describe_metadata(prim),
        "time_code": time_code_value(time),
    }
    if prim.IsInstance():
        info["prototype"] = prim.GetPrototype().GetPath().pathString
    if prim.IsInstanceProxy():
        info["prim_in_prototype"] = prim.GetPrimInPrototype().GetPath().pathString
    return info


class USDReader(LoggerMixin):
    """
    Core USD reader for loading and analyzing USD files.

    This class provides methods to load USD stages, traverse scene graphs,
    and extract detailed information about prims and their properties.
    """

    def __init__(self, enable_caching: bool = True):
        """
        Initialize USD reader.

        Args:
            enable_caching: Enable USD stage caching
        """
        self.enable_caching = enable_caching
        self._stage_cache: Dict[str, Usd.Stage] = {}
        self.logger.info("USD Reader initialized")

    @monitor_performance("usd_reader.open_stage")
    def open_stage(self, file_path: Union[str, Path], use_cache: bool = True) -> Optional[Usd.Stage]:
        """
        Open a USD stage from file.

        Args:
            file_path: Path to USD file
            use_cache: Use cached stage if available

        Returns:
            USD Stage object or None if failed to open
        """
        file_path = str(Path(file_path).resolve())

        # Check cache first
        if use_cache and self.enable_caching and file_path in self._stage_cache:
            self.logger.debug(f"Using cached stage: {file_path}")
            return self._stage_cache[file_path]

        if not os.path.exists(file_path):
            self.logger.error(f"USD file not found: {file_path}")
            return None

        try:
            with Timer(f"Opening USD stage: {file_path}"):
                stage = Usd.Stage.Open(file_path)

                if stage:
                    self.logger.info(f"Successfully opened USD stage: {file_path}")

                    # Cache the stage
                    if self.enable_caching:
                        self._stage_cache[file_path] = stage

                    return stage
                else:
                    self.logger.error(f"Failed to open USD stage: {file_path}")
                    return None

        except Exception as e:
            self.logger.error(f"Error opening USD stage {file_path}: {e}")
            return None

    def close_stage(self, file_path: Union[str, Path]) -> None:
        """
        Close and remove stage from cache.

        Args:
            file_path: Path to USD file
        """
        file_path = str(Path(file_path).resolve())

        if file_path in self._stage_cache:
            del self._stage_cache[file_path]
            self.logger.debug(f"Closed and removed stage from cache: {file_path}")

    def clear_cache(self) -> None:
        """Clear all cached stages."""
        self._stage_cache.clear()
        self.logger.info("Cleared USD stage cache")

    @staticmethod
    def classify_prim_type(type_name: str) -> PrimType:
        """Map a USD type name to the internal PrimType enum."""
        try:
            return PrimType(type_name)
        except ValueError:
            return PrimType.UNKNOWN

    @monitor_performance("usd_reader.get_stage_info")
    def get_stage_info(self, stage: Usd.Stage, *, prim_count: Optional[int] = None) -> USDStageInfo:
        """
        Extract comprehensive information about a USD stage.

        Args:
            stage: USD Stage object
            prim_count: Pre-computed ``search.count_prims`` count. Callers that
                already walked the stage pass it to skip a second traversal.

        Returns:
            USDStageInfo with stage details
        """
        try:
            root_layer = stage.GetRootLayer()

            session_layer = stage.GetSessionLayer()
            session_layer_path = session_layer.identifier if session_layer else None

            # Layer stack, strongest first
            layers = []
            for layer in stage.GetLayerStack():
                layer_info = USDLayerInfo(
                    identifier=layer.identifier,
                    display_name=layer.GetDisplayName(),
                    real_path=layer.realPath,
                    file_format=layer.GetFileFormat().formatId,
                    version=layer.version,
                    comment=layer.comment,
                    documentation=layer.documentation,
                    is_anonymous=layer.anonymous,
                    is_dirty=layer.dirty,
                    permission=str(layer.permissionToEdit)
                )
                layers.append(layer_info)

            default_prim = stage.GetDefaultPrim()
            default_prim_path = str(default_prim.GetPath()) if default_prim else None

            # Stage metadata authored on the root (and session) layer.
            stage_variables = {
                key: to_json(value)
                for key, value in stage.GetPseudoRoot().GetAllAuthoredMetadata().items()
            }

            # The count every listing uses: instance proxies included.
            if prim_count is None:
                prim_count = count_prims(stage)

            root_prims = [str(prim.GetPath()) for prim in stage.GetPseudoRoot().GetChildren()]

            stage_info = USDStageInfo(
                root_layer_path=root_layer.identifier,
                session_layer_path=session_layer_path,
                layers=layers,
                default_prim_path=default_prim_path,
                up_axis=str(UsdGeom.GetStageUpAxis(stage)),
                meters_per_unit=UsdGeom.GetStageMetersPerUnit(stage),
                time_codes_per_second=stage.GetTimeCodesPerSecond(),
                start_time_code=stage.GetStartTimeCode(),
                end_time_code=stage.GetEndTimeCode(),
                frame_rate=stage.GetFramesPerSecond(),
                prim_count=prim_count,
                root_prims=root_prims,
                stage_variables=stage_variables
            )

            self.logger.debug(f"Extracted stage info: {prim_count} prims, {len(layers)} layers")
            return stage_info

        except Exception as e:
            self.logger.error(f"Error extracting stage info: {e}")
            raise

    @monitor_performance("usd_reader.get_prim_info")
    def get_prim_info(
        self,
        prim: Usd.Prim,
        *,
        time_code: Optional[Any] = None,
        array_preview: int = DEFAULT_ARRAY_PREVIEW,
    ) -> USDPrimInfo:
        """
        Extract comprehensive information about a USD prim.

        Args:
            prim: USD Prim object
            time_code: Time to read attribute values at; None for the default time
            array_preview: Leading elements of each array attribute to include

        Returns:
            USDPrimInfo with prim details; ``attributes`` holds one
            ``describe_attribute`` record per attribute
        """
        time = resolve_time_code(time_code)
        type_name = prim.GetTypeName()
        purpose = None
        visibility = None
        if prim.IsA(UsdGeom.Imageable):
            imageable = UsdGeom.Imageable(prim)
            purpose = imageable.GetPurposeAttr().Get(time)
            visibility = imageable.GetVisibilityAttr().Get(time)
        parent = prim.GetParent()
        return USDPrimInfo(
            path=str(prim.GetPath()),
            name=prim.GetName(),
            type_name=type_name,
            prim_type=self.classify_prim_type(type_name),
            is_active=prim.IsActive(),
            is_loaded=prim.IsLoaded(),
            is_defined=prim.IsDefined(),
            is_abstract=prim.IsAbstract(),
            is_instance=prim.IsInstance(),
            is_prototype=prim.IsPrototype(),
            has_authored_references=prim.HasAuthoredReferences(),
            has_authored_payloads=prim.HasAuthoredPayloads(),
            has_authored_inherits=prim.HasAuthoredInherits(),
            has_authored_specializes=prim.HasAuthoredSpecializes(),
            kind=Usd.ModelAPI(prim).GetKind() or None,
            purpose=purpose,
            visibility=visibility,
            attributes={
                attr.GetName(): describe_attribute(attr, time, array_preview)
                for attr in prim.GetAttributes()
            },
            relationships={
                rel.GetName(): [target.pathString for target in rel.GetTargets()]
                for rel in prim.GetRelationships()
            },
            metadata=describe_metadata(prim),
            children=[child.GetName() for child in _children(prim)],
            parent=str(parent.GetPath()) if parent else None,
            api_schemas=list(prim.GetAppliedSchemas()),
        )

    def traverse_stage(
        self,
        stage: Usd.Stage,
        predicate: Optional[callable] = None
    ) -> Iterator[Usd.Prim]:
        """
        Traverse all prims in a stage with optional filtering.

        Args:
            stage: USD Stage to traverse
            predicate: Optional function to filter prims

        Yields:
            USD Prim objects
        """
        for prim in stage.Traverse():
            if predicate is None or predicate(prim):
                yield prim

    def find_prims_by_type(
        self,
        stage: Usd.Stage,
        prim_type: Union[str, PrimType]
    ) -> List[Usd.Prim]:
        """
        Find all prims of a specific type.

        Args:
            stage: USD Stage to search
            prim_type: Prim type to search for

        Returns:
            List of matching prims
        """
        if isinstance(prim_type, PrimType):
            prim_type = prim_type.value

        return [prim for prim in stage.Traverse() if prim.GetTypeName() == prim_type]

    def find_prims_by_name(
        self,
        stage: Usd.Stage,
        name_pattern: str,
        exact_match: bool = False
    ) -> List[Usd.Prim]:
        """
        Find prims by name pattern.

        Args:
            stage: USD Stage to search
            name_pattern: Name pattern to search for
            exact_match: Whether to use exact matching

        Returns:
            List of matching prims
        """
        if exact_match:
            return [prim for prim in stage.Traverse() if prim.GetName() == name_pattern]
        else:
            return [prim for prim in stage.Traverse() if name_pattern in prim.GetName()]


# Convenience functions
def read_usd_file(file_path: Union[str, Path]) -> Optional[Usd.Stage]:
    """
    Convenience function to read a USD file.

    Args:
        file_path: Path to USD file

    Returns:
        USD Stage or None if failed
    """
    reader = USDReader()
    return reader.open_stage(file_path)


def get_stage_info(stage: Usd.Stage) -> USDStageInfo:
    """
    Convenience function to get stage information.

    Args:
        stage: USD Stage

    Returns:
        USDStageInfo object
    """
    reader = USDReader()
    return reader.get_stage_info(stage)


def get_prim_info(prim: Usd.Prim) -> USDPrimInfo:
    """
    Convenience function to get prim information.

    Args:
        prim: USD Prim

    Returns:
        USDPrimInfo object
    """
    reader = USDReader()
    return reader.get_prim_info(prim)


def traverse_stage(stage: Usd.Stage, predicate: Optional[callable] = None) -> Iterator[Usd.Prim]:
    """
    Convenience function to traverse stage.

    Args:
        stage: USD Stage
        predicate: Optional filter function

    Yields:
        USD Prim objects
    """
    reader = USDReader()
    yield from reader.traverse_stage(stage, predicate)


def find_prims_by_type(stage: Usd.Stage, prim_type: Union[str, PrimType]) -> List[Usd.Prim]:
    """
    Convenience function to find prims by type.

    Args:
        stage: USD Stage
        prim_type: Prim type to search for

    Returns:
        List of matching prims
    """
    reader = USDReader()
    return reader.find_prims_by_type(stage, prim_type)


def find_prims_by_name(stage: Usd.Stage, name_pattern: str, exact_match: bool = False) -> List[Usd.Prim]:
    """
    Convenience function to find prims by name.

    Args:
        stage: USD Stage
        name_pattern: Name pattern
        exact_match: Use exact matching

    Returns:
        List of matching prims
    """
    reader = USDReader()
    return reader.find_prims_by_name(stage, name_pattern, exact_match)
