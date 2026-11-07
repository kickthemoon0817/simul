"""
Bounding boxes for USD prims and stages.

Every bound comes from ``UsdGeom.BBoxCache``, so it is the bound USD itself
computes: each transform between the prim and the root applies, instances are
followed into their prototypes, and an extent that is not authored is computed
from the geometry. Only prims whose computed purpose is in the included set
contribute. The default set is ``default`` plus ``render``, the geometry a
renderer or simulator sees; ``proxy`` and ``guide`` geometry (preview stand-ins,
collision shapes authored as guides) is left out unless asked for.

A prim with nothing boundable at or below it under those purposes has no
bound: the functions here return ``None`` for it, never a guessed box.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from ._pxr import Usd, UsdGeom

from ..logging import get_logger, LoggerMixin
from ..utils.timing import monitor_performance
from ..utils.math import BBox, bbox_center, bbox_size, bbox_volume
from .timecode import resolve_time_code, time_code_value

logger = get_logger(__name__)

# UsdGeom.Imageable.GetOrderedPurposeTokens(), spelled out so the schema enum
# and the validation below agree without importing pxr at schema build time.
PURPOSES: Tuple[str, ...] = ("default", "render", "proxy", "guide")

# What renderers and simulators see: everything not marked proxy or guide.
DEFAULT_PURPOSES: Tuple[str, ...] = ("default", "render")


def normalize_purposes(purposes: Optional[Iterable[str]] = None) -> Tuple[str, ...]:
    """Validate a purpose selection and return it without duplicates.

    Args:
        purposes: Purpose tokens to include, or None for ``DEFAULT_PURPOSES``.

    Returns:
        The purposes in the order given, each once.

    Raises:
        ValueError: For an unknown token or an empty selection.
    """
    if purposes is None:
        return DEFAULT_PURPOSES
    selected: List[str] = []
    for purpose in purposes:
        if purpose not in PURPOSES:
            raise ValueError(f"Unknown purpose {purpose!r}; expected one of {list(PURPOSES)}")
        if purpose not in selected:
            selected.append(purpose)
    if not selected:
        raise ValueError(f"include_purposes must name at least one of {list(PURPOSES)}")
    return tuple(selected)


def _aligned_range_to_bbox(usd_bbox: Any) -> Optional[BBox]:
    """Convert a ``Gf.BBox3d`` to a ``(min, max)`` pair, or None when empty.

    ``Gf.BBox3d`` has no ``IsEmpty``/``GetMin``/``GetMax``; the axis-aligned
    range (with the box's matrix applied) carries them.
    """
    aligned = usd_bbox.ComputeAlignedRange()
    if aligned.IsEmpty():
        return None
    min_point = aligned.GetMin()
    max_point = aligned.GetMax()
    return (
        [float(min_point[0]), float(min_point[1]), float(min_point[2])],
        [float(max_point[0]), float(max_point[1]), float(max_point[2])],
    )


class BBoxCache(LoggerMixin):
    """
    Bounding boxes of one stage at one time, for one set of purposes.

    Wraps ``UsdGeom.BBoxCache``, which memoises the bounds it computes. The
    cache does not watch the stage: call ``clear_cache`` after an edit.
    """

    def __init__(
        self,
        stage: Usd.Stage,
        time_code: Optional[Any] = None,
        *,
        purposes: Optional[Iterable[str]] = None,
    ):
        """
        Initialize the cache.

        Args:
            stage: USD Stage
            time_code: Time code to evaluate at; None (or ``Usd.TimeCode.Default()``)
                for the default time
            purposes: Purposes whose geometry counts; None for default and render
        """
        self.stage = stage
        self.time_code = resolve_time_code(time_code)
        self.purposes = normalize_purposes(purposes)
        self._usd_bbox_cache = self._new_usd_cache()
        self.logger.debug(
            f"BBox cache initialized at time {self.time_code} for purposes {list(self.purposes)}"
        )

    def _new_usd_cache(self) -> UsdGeom.BBoxCache:
        return UsdGeom.BBoxCache(self.time_code, includedPurposes=list(self.purposes))

    @staticmethod
    def _default_time_code() -> Any:
        """Return the USD default time code."""
        return Usd.TimeCode.Default()

    def clear_cache(self) -> None:
        """Forget every computed bound; the next query recomputes from the stage."""
        self._usd_bbox_cache.Clear()
        self.logger.debug("Cleared bounding box cache")

    def _cache_for(self, use_cache: bool) -> UsdGeom.BBoxCache:
        return self._usd_bbox_cache if use_cache else self._new_usd_cache()

    @monitor_performance("bbox_cache.compute_world_bbox")
    def compute_world_bbox(self, prim: Usd.Prim, use_cache: bool = True) -> Optional[BBox]:
        """
        Compute the world-space bound of a prim and its descendants.

        Args:
            prim: USD Prim (the pseudo-root bounds the whole stage)
            use_cache: Reuse bounds computed earlier; False computes afresh

        Returns:
            ``(min, max)`` in world coordinates, or None when nothing with an
            included purpose is boundable at or below the prim
        """
        return _aligned_range_to_bbox(self._cache_for(use_cache).ComputeWorldBound(prim))

    @monitor_performance("bbox_cache.compute_local_bbox")
    def compute_local_bbox(self, prim: Usd.Prim, use_cache: bool = True) -> Optional[BBox]:
        """
        Compute the bound of a prim and its descendants in the prim's own space.

        The descendants' transforms apply; the prim's own transform and its
        ancestors' do not.

        Args:
            prim: USD Prim
            use_cache: Reuse bounds computed earlier; False computes afresh

        Returns:
            ``(min, max)`` in the prim's object space, or None when nothing
            with an included purpose is boundable at or below the prim
        """
        return _aligned_range_to_bbox(self._cache_for(use_cache).ComputeUntransformedBound(prim))

    def get_stage_bbox(self) -> Optional[BBox]:
        """
        Compute the world-space bound of the whole stage.

        Returns:
            ``(min, max)``, or None when the stage has no boundable geometry
            with an included purpose
        """
        return self.compute_world_bbox(self.stage.GetPseudoRoot())


def bbox_to_dict(bbox: BBox) -> Dict[str, List[float]]:
    """
    Convert bounding box to dictionary format.

    Args:
        bbox: Bounding box tuple

    Returns:
        Dictionary with min and max points
    """
    min_point, max_point = bbox
    return {
        'min': list(min_point),
        'max': list(max_point)
    }


def dict_to_bbox(bbox_dict: Dict[str, List[float]]) -> BBox:
    """
    Convert dictionary to bounding box format.

    Args:
        bbox_dict: Dictionary with min and max points

    Returns:
        Bounding box tuple
    """
    return (bbox_dict['min'], bbox_dict['max'])


def describe_bounds(
    stage: Usd.Stage,
    prim: Optional[Usd.Prim] = None,
    *,
    world_space: bool = True,
    purposes: Optional[Iterable[str]] = None,
    time_code: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Bound a prim (or the stage) and say what the bound covers.

    Args:
        stage: USD Stage
        prim: Prim to bound; None (or the pseudo-root) bounds the whole stage
        world_space: World coordinates when True, else the prim's object space
        purposes: Purposes whose geometry counts; None for default and render
        time_code: Time code to evaluate at; None for the default time

    Returns:
        ``bbox`` (``min``/``max``/``center``/``size``/``volume``, or None),
        ``has_geometry``, ``included_purposes`` and ``time_code``. With no
        bound, ``excluded_purposes_with_geometry`` names the purposes left out
        that do hold geometry there, so the caller knows whether widening the
        selection would help.
    """
    cache = BBoxCache(stage, time_code, purposes=purposes)
    target = stage.GetPseudoRoot() if prim is None else prim
    bbox = cache.compute_world_bbox(target) if world_space else cache.compute_local_bbox(target)
    result: Dict[str, Any] = {
        "bbox": None,
        "has_geometry": bbox is not None,
        "included_purposes": list(cache.purposes),
        "time_code": time_code_value(cache.time_code),
    }
    if bbox is not None:
        result["bbox"] = {
            **bbox_to_dict(bbox),
            "center": bbox_center(bbox),
            "size": bbox_size(bbox),
            "volume": bbox_volume(bbox),
        }
        return result
    excluded = [purpose for purpose in PURPOSES if purpose not in cache.purposes]
    result["excluded_purposes_with_geometry"] = [
        purpose
        for purpose in excluded
        if BBoxCache(stage, cache.time_code, purposes=[purpose]).compute_world_bbox(target) is not None
    ]
    return result


# Convenience functions
def compute_world_bbox(
    prim: Usd.Prim,
    time_code: Optional[Any] = None,
    *,
    purposes: Optional[Iterable[str]] = None,
) -> Optional[BBox]:
    """Compute world bounding box for a prim."""
    cache = BBoxCache(prim.GetStage(), time_code, purposes=purposes)
    return cache.compute_world_bbox(prim)


def compute_local_bbox(
    prim: Usd.Prim,
    time_code: Optional[Any] = None,
    *,
    purposes: Optional[Iterable[str]] = None,
) -> Optional[BBox]:
    """Compute local bounding box for a prim."""
    cache = BBoxCache(prim.GetStage(), time_code, purposes=purposes)
    return cache.compute_local_bbox(prim)


def get_prim_bbox(
    prim: Usd.Prim,
    world_space: bool = True,
    time_code: Optional[Any] = None,
    *,
    purposes: Optional[Iterable[str]] = None,
) -> Optional[BBox]:
    """Get bounding box for a prim in world or local space."""
    cache = BBoxCache(prim.GetStage(), time_code, purposes=purposes)
    if world_space:
        return cache.compute_world_bbox(prim)
    else:
        return cache.compute_local_bbox(prim)


def get_stage_bbox(
    stage: Usd.Stage,
    time_code: Optional[Any] = None,
    *,
    purposes: Optional[Iterable[str]] = None,
) -> Optional[BBox]:
    """Get bounding box for an entire stage."""
    cache = BBoxCache(stage, time_code, purposes=purposes)
    return cache.get_stage_bbox()
