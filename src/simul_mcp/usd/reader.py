"""
Core USD Reader for Isaac Sim MCP Server.

This module provides the main USD reading functionality using the pxr library
for loading USD files, traversing scene graphs, and extracting prim information.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Iterator, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ._pxr import Usd, UsdGeom

from ..logging import get_logger, LoggerMixin
from ..utils.timing import Timer, monitor_performance
from ..utils.math import BBox

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
    def get_stage_info(self, stage: Usd.Stage) -> USDStageInfo:
        """
        Extract comprehensive information about a USD stage.
        
        Args:
            stage: USD Stage object
            
        Returns:
            USDStageInfo with stage details
        """
        try:
            # Get root layer info
            root_layer = stage.GetRootLayer()
            
            # Get session layer info
            session_layer = stage.GetSessionLayer()
            session_layer_path = session_layer.identifier if session_layer else None
            
            # Get all layers
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
            
            # Get stage metadata
            default_prim = stage.GetDefaultPrim()
            default_prim_path = str(default_prim.GetPath()) if default_prim else None
            
            # Get stage variables
            stage_variables = {}
            if hasattr(stage, 'GetMetadata'):
                try:
                    stage_variables = dict(stage.GetMetadata()) or {}
                except:
                    pass
            
            # Count prims
            prim_count = len(list(stage.Traverse()))
            
            # Get root prims
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
    def get_prim_info(self, prim: Usd.Prim) -> USDPrimInfo:
        """
        Extract comprehensive information about a USD prim.
        
        Args:
            prim: USD Prim object
            
        Returns:
            USDPrimInfo with prim details
        """
        try:
            # Basic prim information
            path = str(prim.GetPath())
            name = prim.GetName()
            type_name = prim.GetTypeName()
            
            # Determine prim type enum
            prim_type = self.classify_prim_type(type_name)
            
            # Prim state
            is_active = prim.IsActive()
            is_loaded = prim.IsLoaded()
            is_defined = prim.IsDefined()
            is_abstract = prim.IsAbstract()
            is_instance = prim.IsInstance()
            is_prototype = prim.IsPrototype()
            
            # Composition arcs
            has_authored_references = prim.HasAuthoredReferences()
            has_authored_payloads = prim.HasAuthoredPayloads()
            has_authored_inherits = prim.HasAuthoredInherits()
            has_authored_specializes = prim.HasAuthoredSpecializes()
            
            # Metadata
            kind = None
            if prim.HasAuthoredMetadata('kind'):
                kind = prim.GetMetadata('kind')
            
            # Purpose and visibility (for imageable prims)
            purpose = None
            visibility = None
            if prim.IsA(UsdGeom.Imageable):
                imageable = UsdGeom.Imageable(prim)
                purpose_attr = imageable.GetPurposeAttr()
                if purpose_attr:
                    purpose = purpose_attr.Get()
                
                visibility_attr = imageable.GetVisibilityAttr()
                if visibility_attr:
                    visibility = visibility_attr.Get()
            
            # Attributes
            attributes = {}
            for attr in prim.GetAttributes():
                attr_name = attr.GetName()
                try:
                    attr_value = attr.Get()
                    # Convert USD types to Python types for serialization
                    if hasattr(attr_value, '__iter__') and not isinstance(attr_value, str):
                        attr_value = list(attr_value)
                    attributes[attr_name] = attr_value
                except:
                    attributes[attr_name] = None
            
            # Relationships
            relationships = {}
            for rel in prim.GetRelationships():
                rel_name = rel.GetName()
                try:
                    targets = rel.GetTargets()
                    relationships[rel_name] = [str(target) for target in targets]
                except:
                    relationships[rel_name] = []
            
            # Metadata
            metadata = {}
            try:
                for key in prim.GetMetadataKeys():
                    metadata[key] = prim.GetMetadata(key)
            except:
                pass
            
            # Children and parent
            children = [child.GetName() for child in prim.GetChildren()]
            parent = str(prim.GetParent().GetPath()) if prim.GetParent() else None
            
            prim_info = USDPrimInfo(
                path=path,
                name=name,
                type_name=type_name,
                prim_type=prim_type,
                is_active=is_active,
                is_loaded=is_loaded,
                is_defined=is_defined,
                is_abstract=is_abstract,
                is_instance=is_instance,
                is_prototype=is_prototype,
                has_authored_references=has_authored_references,
                has_authored_payloads=has_authored_payloads,
                has_authored_inherits=has_authored_inherits,
                has_authored_specializes=has_authored_specializes,
                kind=kind,
                purpose=purpose,
                visibility=visibility,
                attributes=attributes,
                relationships=relationships,
                metadata=metadata,
                children=children,
                parent=parent
            )
            
            return prim_info
            
        except Exception as e:
            self.logger.error(f"Error extracting prim info for {prim.GetPath()}: {e}")
            raise
    
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
