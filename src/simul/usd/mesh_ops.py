"""
Mesh statistics for USD mesh prims.

Every geometric statistic (topology check, watertightness, surface area,
volume, bounds) is computed with numpy over whole arrays, never with a Python
loop per face: a million-vertex mesh used to take 20 s and a few hundred MB of
Python lists, and it now takes a fraction of a second within bounded memory.
``geometry_statistics`` measures many meshes in one pass, which is how
``summarize_scene`` keeps a stage of twenty thousand small meshes fast.

Reads honour ``time_code`` (a deforming mesh has points only as time samples),
normals authored as ``primvars:normals`` count as normals, and materials are
the ones ``UsdShade.MaterialBindingAPI`` resolves, per geometry subset too.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any, Sequence
from dataclasses import dataclass, field
import numpy as np

from ._pxr import Usd, UsdGeom, UsdShade

from ..logging import get_logger, LoggerMixin
from ..utils.timing import monitor_performance
from ..utils.math import BBox
from .timecode import resolve_time_code, time_code_value
from .values import DEFAULT_ARRAY_PREVIEW, describe_array

logger = get_logger(__name__)

# Faces handled per numpy pass; bounds the temporaries of a huge mesh.
_FACE_CHUNK = 1 << 16

_UV_NAMES = frozenset({"st", "st0", "st1", "uv", "uv0", "uv1", "texCoord", "map1"})
_COLOR_NAMES = frozenset({"displayColor", "color", "Cd", "vertexColor"})
_TANGENT_NAMES = frozenset({"tangent", "tangents", "tangentVectors"})


@dataclass
class MeshInfo:
    """Information about a USD mesh."""
    prim_path: str
    vertex_count: int
    face_count: int
    point_count: int
    has_normals: bool
    has_uvs: bool
    has_colors: bool
    has_tangents: bool
    subdivision_scheme: str
    face_varying_linear_interpolation: str
    interpolate_boundary: str
    topology_valid: Optional[bool]
    is_closed: Optional[bool]
    surface_area: Optional[float]
    volume: Optional[float]
    bbox: Optional[BBox]
    materials: List[str] = field(default_factory=list)
    subsets: List[Dict[str, Any]] = field(default_factory=list)
    normals_source: Optional[str] = None
    uv_sets: List[str] = field(default_factory=list)
    physics_material: Optional[str] = None
    points_time_samples: int = 0
    time_code: Optional[float] = None
    measured: bool = True
    sampled_time_range: Optional[List[float]] = None
    note: Optional[str] = None
    consistently_oriented: Optional[bool] = None


@dataclass
class MeshArrays:
    """A mesh's topology and points as numpy arrays.

    ``holes`` lists the faces ``holeIndices`` marks as holes: they are not
    rendered, so they add no area and leave their edges open.
    """
    points: np.ndarray
    counts: np.ndarray
    indices: np.ndarray
    holes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))


@dataclass
class GeometryStats:
    """What ``geometry_statistics`` measures for one mesh.

    ``surface_area`` is None when the topology is invalid (it would index
    points that do not exist), ``volume`` None unless the mesh is closed and
    consistently oriented, and ``bbox`` None when the mesh has no points.
    ``consistently_oriented`` says, for a closed mesh, whether every edge is
    traversed once in each direction by the two faces sharing it (None when
    the mesh is not closed, or too large to tell).
    """
    topology_valid: Optional[bool]
    is_closed: Optional[bool]
    surface_area: Optional[float]
    volume: Optional[float]
    bbox: Optional[BBox]
    consistently_oriented: Optional[bool] = None


# Edge keys carry the edge's direction in their lowest bit while the doubled
# key fits an int64; past this many vertices orientation is not checked.
_DIRECTED_KEY_VERTEX_LIMIT = 1 << 30


def _closed_and_oriented(keys: np.ndarray, directed: bool) -> "tuple[bool, Optional[bool]]":
    """From sorted edge keys: whether every edge is used by exactly two faces, and in opposite directions.

    With ``directed`` keys (``edge * 2 + (a > b)``) a closed, consistently
    wound mesh sorts into pairs ``(2e, 2e + 1)``; a face wound the other way
    from its neighbour leaves a pair with equal keys.
    """
    if len(keys) == 0 or len(keys) % 2:
        return False, None
    first, second = keys[0::2], keys[1::2]
    if not directed:
        closed = bool(np.array_equal(first, second) and np.all(first[1:] != first[:-1]))
        return closed, None
    edges = first >> 1
    closed = bool(np.array_equal(edges, second >> 1) and np.all(edges[1:] != edges[:-1]))
    if not closed:
        return False, None
    return True, bool(np.all(second - first == 1))


def _as_array(value: Any, dtype: Any, columns: int = 0) -> np.ndarray:
    """A numpy view (or copy) of a Vt array or sequence, as ``dtype`` unless it already is a float type."""
    if value is None or len(value) == 0:
        return np.zeros((0, columns) if columns else 0, dtype=dtype)
    array = np.asarray(value)
    if columns:
        # Keep float32/float64 points as they are; geometry math upcasts per chunk.
        if array.dtype not in (np.float32, np.float64):
            array = array.astype(np.float64)
        return array.reshape(-1, columns)
    return array.astype(dtype, copy=False).reshape(-1)


def read_mesh_arrays(mesh: UsdGeom.Mesh, time_code: Optional[Any] = None) -> MeshArrays:
    """Read a mesh's points, face vertex counts, indices and hole faces at ``time_code``."""
    time = resolve_time_code(time_code)
    return MeshArrays(
        points=_as_array(mesh.GetPointsAttr().Get(time), np.float64, 3),
        counts=_as_array(mesh.GetFaceVertexCountsAttr().Get(time), np.int32),
        indices=_as_array(mesh.GetFaceVertexIndicesAttr().Get(time), np.int32),
        holes=_as_array(mesh.GetHoleIndicesAttr().Get(time), np.int64),
    )


TOPOLOGY_ATTRIBUTES = ("points", "faceVertexCounts", "faceVertexIndices")


def unsampled_topology(mesh: UsdGeom.Mesh, time_code: Optional[Any] = None) -> Dict[str, List[float]]:
    """Topology attributes with no value at ``time_code`` that do have time samples.

    A deforming (point-cached) mesh often authors its points only as time
    samples, so at the default time it has none. Measuring it there would
    call its topology invalid; such a mesh has to be read at a sampled time.

    Returns:
        Attribute name to its ``[first, last]`` sample time, for each such
        attribute; empty when the mesh can be measured at ``time_code``.
    """
    time = resolve_time_code(time_code)
    missing: Dict[str, List[float]] = {}
    for name in TOPOLOGY_ATTRIBUTES:
        attr = mesh.GetPrim().GetAttribute(name)
        if attr and attr.Get(time) is None and attr.GetNumTimeSamples():
            samples = attr.GetTimeSamples()
            missing[name] = [float(samples[0]), float(samples[-1])]
    return missing


def _face_pass(
    points: np.ndarray,
    gidx: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    kept: Optional[np.ndarray],
    face_mesh: Optional[np.ndarray],
    centers: np.ndarray,
    mesh_count: int,
    vertex_total: int,
    directed: bool = False,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Walk the faces in chunks: edge keys, area and signed volume per mesh.

    For each face it takes its boundary edges (consecutive face vertices,
    wrapping round). The undirected edges become keys ``lo * vertex_total +
    hi`` (degenerate edges dropped) for the watertightness check; with
    ``directed`` each key is doubled and its lowest bit set when the face runs
    the edge from the higher vertex to the lower, for the winding check. Their cross
    products sum to the face's vector area (Newell's method), whose length is
    the face's area, exact for any planar polygon, concave ones included, and
    whose dot product with the face's centroid gives its divergence-theorem
    volume term. Hole faces (``kept`` False) add neither edges nor area.
    Points are promoted to float64 one chunk at a time and measured from
    their mesh's centre, which keeps a mesh far from the origin precise.

    Returns:
        (edge keys, unsorted; area per mesh; signed volume sum per mesh, to be
        divided by 3).
    """
    keys = np.empty(len(gidx), dtype=np.int64)
    filled = 0
    area = np.zeros(mesh_count)
    raw_volume = np.zeros(mesh_count)
    for first in range(0, len(counts), _FACE_CHUNK):
        face_starts = starts[first:first + _FACE_CHUNK]
        face_counts = counts[first:first + _FACE_CHUNK].astype(np.int64)
        face_count = len(face_counts)
        begin = int(face_starts[0])
        end = int(face_starts[-1] + face_counts[-1])
        segment = gidx[begin:end]
        following = np.arange(1, end - begin + 1, dtype=np.int64)
        following[face_starts + face_counts - 1 - begin] = face_starts - begin
        a = segment
        b = segment[following]
        local_face = np.repeat(np.arange(face_count), face_counts)
        chunk_kept = None if kept is None else kept[first:first + _FACE_CHUNK]

        edge = a != b
        if chunk_kept is not None:
            edge &= chunk_kept[local_face]
        lo = np.minimum(a[edge], b[edge]).astype(np.int64)
        hi = np.maximum(a[edge], b[edge])
        lo *= vertex_total
        lo += hi
        if directed:
            lo *= 2
            lo += a[edge] > b[edge]
        keys[filled:filled + len(lo)] = lo
        filled += len(lo)
        del lo, hi, edge

        chunk_mesh = None if face_mesh is None else face_mesh[first:first + _FACE_CHUNK]
        offset = centers[0] if chunk_mesh is None else centers[chunk_mesh[local_face]]
        pa = points[a].astype(np.float64) - offset
        pb = points[b].astype(np.float64) - offset
        cross = np.cross(pa, pb)
        vector = np.stack(
            [np.bincount(local_face, weights=cross[:, axis], minlength=face_count) for axis in range(3)],
            axis=1,
        ) * 0.5
        centroid = np.stack(
            [np.bincount(local_face, weights=pa[:, axis], minlength=face_count) for axis in range(3)],
            axis=1,
        ) / face_counts[:, None]
        face_area = np.linalg.norm(vector, axis=1)
        face_volume = np.einsum("ij,ij->i", centroid, vector)
        if chunk_kept is not None:
            face_area = face_area * chunk_kept
            face_volume = face_volume * chunk_kept
        if chunk_mesh is None:
            area[0] += face_area.sum()
            raw_volume[0] += face_volume.sum()
        else:
            area += np.bincount(chunk_mesh, weights=face_area, minlength=mesh_count)
            raw_volume += np.bincount(chunk_mesh, weights=face_volume, minlength=mesh_count)
    return keys[:filled], area, raw_volume


def _single_mesh_statistics(mesh: MeshArrays) -> GeometryStats:
    """``geometry_statistics`` for one mesh, without the batch bookkeeping arrays."""
    points, counts, indices, holes = mesh.points, mesh.counts, mesh.indices, mesh.holes
    vertices = len(points)
    faces = len(counts)
    bbox = None
    center = np.zeros((1, 3))
    if vertices:
        low, high = points.min(axis=0).astype(np.float64), points.max(axis=0).astype(np.float64)
        bbox = (low.tolist(), high.tolist())
        center = ((low + high) / 2.0).reshape(1, 3)
    valid = int(counts.sum(dtype=np.int64)) == len(indices)
    if valid and faces:
        valid = int(counts.min()) >= 3
    if valid and len(indices):
        valid = int(indices.min()) >= 0 and int(indices.max()) < vertices
    if valid and len(holes):
        valid = int(holes.min()) >= 0 and int(holes.max()) < faces
    if not valid:
        return GeometryStats(False, False, None, None, bbox)
    if not faces:
        return GeometryStats(True, False, 0.0, None, bbox)
    kept = None
    if len(holes):
        kept = np.ones(faces, dtype=bool)
        kept[holes] = False
    starts = np.cumsum(counts, dtype=np.int64) - counts
    directed = vertices < _DIRECTED_KEY_VERTEX_LIMIT
    keys, area, raw_volume = _face_pass(
        points, indices, starts, counts, kept, None, center, 1, max(1, vertices), directed
    )
    keys.sort()
    closed, oriented = _closed_and_oriented(keys, directed)
    del keys
    return GeometryStats(
        topology_valid=True,
        is_closed=closed,
        surface_area=float(area[0]),
        volume=abs(float(raw_volume[0])) / 3.0 if closed and oriented else None,
        bbox=bbox,
        consistently_oriented=oriented,
    )


def geometry_statistics(meshes: Sequence[MeshArrays]) -> List[GeometryStats]:
    """Measure many meshes in one vectorised pass.

    For each mesh: whether the topology is valid (every index in range, the
    counts summing to the index count, no face under three vertices, every
    hole index naming a face); whether it is closed (every undirected edge of
    its non-hole faces shared by exactly two of them) and, if so, whether it
    is consistently oriented (the two faces run each shared edge in opposite
    directions); its surface area (Newell's method, exact for planar faces,
    concave ones included) and, when closed and consistently oriented, its
    enclosed volume (the divergence theorem over a mesh with flipped faces
    gives a wrong number, so none is reported); and the bounds of its points.

    Args:
        meshes: The meshes' arrays.

    Returns:
        One ``GeometryStats`` per mesh, in order.
    """
    mesh_count = len(meshes)
    if mesh_count == 0:
        return []
    if mesh_count == 1:
        return [_single_mesh_statistics(meshes[0])]
    vertex_counts = np.array([len(m.points) for m in meshes], dtype=np.int64)
    face_counts = np.array([len(m.counts) for m in meshes], dtype=np.int64)
    index_counts = np.array([len(m.indices) for m in meshes], dtype=np.int64)
    hole_counts = np.array([len(m.holes) for m in meshes], dtype=np.int64)
    points = np.concatenate([m.points.astype(np.float64, copy=False) for m in meshes])
    counts = np.concatenate([m.counts for m in meshes]).astype(np.int64, copy=False)
    indices = np.concatenate([m.indices for m in meshes]).astype(np.int64, copy=False)
    holes = np.concatenate([m.holes for m in meshes]).astype(np.int64, copy=False)
    mesh_ids = np.arange(mesh_count)
    mesh_of_face = np.repeat(mesh_ids, face_counts)
    mesh_of_index = np.repeat(mesh_ids, index_counts)
    mesh_of_hole = np.repeat(mesh_ids, hole_counts)

    count_sums = np.bincount(mesh_of_face, weights=counts.astype(np.float64), minlength=mesh_count)
    short_faces = np.bincount(mesh_of_face[counts < 3], minlength=mesh_count)
    out_of_range = np.flatnonzero((indices < 0) | (indices >= vertex_counts[mesh_of_index]))
    bad_indices = np.bincount(mesh_of_index[out_of_range], minlength=mesh_count)
    bad_hole = (holes < 0) | (holes >= face_counts[mesh_of_hole])
    bad_holes = np.bincount(mesh_of_hole[bad_hole], minlength=mesh_count)
    valid = (count_sums == index_counts) & (short_faces == 0) & (bad_indices == 0) & (bad_holes == 0)

    mins = np.zeros((mesh_count, 3))
    maxs = np.zeros((mesh_count, 3))
    present = vertex_counts > 0
    vertex_offsets = np.cumsum(vertex_counts) - vertex_counts
    if present.any():
        mins[present] = np.minimum.reduceat(points, vertex_offsets[present], axis=0)
        maxs[present] = np.maximum.reduceat(points, vertex_offsets[present], axis=0)
    area = np.zeros(mesh_count)
    raw_volume = np.zeros(mesh_count)
    closed = np.zeros(mesh_count, dtype=bool)
    oriented = np.zeros(mesh_count, dtype=bool)
    directed = len(points) < _DIRECTED_KEY_VERTEX_LIMIT

    measured = valid & (face_counts > 0)
    if measured.any():
        kept = None
        if len(holes) and valid[mesh_of_hole].any():
            face_offsets = np.cumsum(face_counts) - face_counts
            usable = valid[mesh_of_hole]
            kept = np.ones(len(counts), dtype=bool)
            kept[holes[usable] + face_offsets[mesh_of_hole[usable]]] = False
        face_mask = measured[mesh_of_face]
        index_mask = measured[mesh_of_index]
        counts_m = counts[face_mask]
        face_mesh = mesh_of_face[face_mask]
        kept_m = None if kept is None else kept[face_mask]
        gidx = indices[index_mask] + vertex_offsets[mesh_of_index[index_mask]]
        starts = np.cumsum(counts_m) - counts_m
        vertex_total = max(1, len(points))
        centers = (mins + maxs) / 2.0

        keys, area, raw_volume = _face_pass(
            points, gidx, starts, counts_m, kept_m, face_mesh, centers, mesh_count, vertex_total, directed
        )
        if len(keys):
            keys.sort()
            edge_keys = keys >> 1 if directed else keys
            change = np.flatnonzero(edge_keys[1:] != edge_keys[:-1]) + 1
            unique_starts = np.concatenate(([0], change))
            uses = np.diff(np.concatenate((unique_starts, [len(keys)])))
            edge_mesh = np.searchsorted(
                vertex_offsets, edge_keys[unique_starts] // vertex_total, side="right"
            ) - 1
            edges = np.bincount(edge_mesh, minlength=mesh_count)
            bad_edges = np.bincount(edge_mesh[uses != 2], minlength=mesh_count)
            closed = (edges > 0) & (bad_edges == 0)
            if directed:
                # Each of a shared edge's two uses runs it the other way: one direction bit set.
                reversed_uses = np.add.reduceat(keys & 1, unique_starts)
                misoriented = np.bincount(edge_mesh[(uses == 2) & (reversed_uses != 1)], minlength=mesh_count)
                oriented = closed & (misoriented == 0)
            del edge_keys
        del keys

    stats = []
    for index in range(mesh_count):
        is_valid = bool(valid[index])
        is_closed = bool(closed[index]) and is_valid
        is_oriented = bool(oriented[index]) if is_closed and directed else None
        stats.append(GeometryStats(
            topology_valid=is_valid,
            is_closed=is_closed,
            surface_area=float(area[index]) if is_valid else None,
            volume=abs(float(raw_volume[index])) / 3.0 if is_closed and is_oriented else None,
            bbox=(mins[index].tolist(), maxs[index].tolist()) if vertex_counts[index] else None,
            consistently_oriented=is_oriented,
        ))
    return stats


def _raw_volume(arrays: MeshArrays) -> float:
    """Signed-volume sum of the fan triangles, whether or not the mesh is closed."""
    if not len(arrays.counts) or not len(arrays.points):
        return 0.0
    counts = arrays.counts
    starts = np.cumsum(counts) - counts
    triangles = np.clip(counts - 2, 0, None)
    tri_face = np.repeat(np.arange(len(counts)), triangles)
    corner = np.arange(int(triangles.sum())) - np.repeat(np.cumsum(triangles) - triangles, triangles) + 1
    base = starts[tri_face]
    p0 = arrays.points[arrays.indices[base]]
    p1 = arrays.points[arrays.indices[base + corner]]
    p2 = arrays.points[arrays.indices[base + corner + 1]]
    return abs(float(np.einsum("ij,ij->i", p0, np.cross(p1, p2)).sum())) / 6.0


@dataclass
class MeshAttributes:
    """The primvar facts a mesh summary reports."""
    normals_source: Optional[str]
    uv_sets: List[str]
    has_colors: bool
    has_tangents: bool


def mesh_attributes(mesh: UsdGeom.Mesh) -> MeshAttributes:
    """Report where a mesh's normals come from and which UV, colour and tangent primvars it has.

    ``primvars:normals`` takes precedence over the ``normals`` attribute, as
    it does for renderers. Only primvars with authored values count.
    """
    normals_source = None
    uv_sets: List[str] = []
    has_colors = False
    has_tangents = False
    for primvar in UsdGeom.PrimvarsAPI(mesh.GetPrim()).GetPrimvarsWithAuthoredValues():
        name = primvar.GetPrimvarName()
        type_name = primvar.GetTypeName()
        if name == "normals":
            normals_source = "primvars:normals"
        elif type_name.role == "TextureCoordinate" or (name in _UV_NAMES and str(type_name).endswith("2[]")):
            uv_sets.append(name)
        elif name in _COLOR_NAMES or (type_name.role == "Color" and type_name.isArray):
            has_colors = True
        elif name in _TANGENT_NAMES:
            has_tangents = True
    if normals_source is None and mesh.GetNormalsAttr().HasAuthoredValue():
        normals_source = "normals"
    return MeshAttributes(normals_source, uv_sets, has_colors, has_tangents)


def bound_material(prim: Usd.Prim, purpose: str = "") -> Optional[str]:
    """Path of the material ``UsdShade.MaterialBindingAPI`` resolves for ``prim``, or None."""
    api = UsdShade.MaterialBindingAPI(prim)
    material, _relationship = (
        api.ComputeBoundMaterial(purpose) if purpose else api.ComputeBoundMaterial()
    )
    return material.GetPath().pathString if material else None


def physics_material(prim: Usd.Prim) -> Optional[str]:
    """The material bound for physics, when it carries ``PhysicsMaterialAPI``.

    A ``physics`` binding falls back to the all-purpose (visual) binding, which
    only counts as a physics material when it has physics properties.
    """
    path = bound_material(prim, "physics")
    if path is None:
        return None
    material = prim.GetStage().GetPrimAtPath(path)
    applied = material.GetAppliedSchemas() if material else []
    return path if "PhysicsMaterialAPI" in applied else None


class MeshOperations(LoggerMixin):
    """
    USD mesh operations and analysis.

    Provides functionality to extract, analyze, and manipulate mesh data
    from USD mesh prims.
    """

    def __init__(self):
        """Initialize mesh operations."""
        self.logger.debug("Mesh operations initialized")

    @staticmethod
    def _default_time_code() -> Any:
        """Return the USD default time code."""
        return Usd.TimeCode.Default()

    @staticmethod
    def _path_to_str(path_value: Any) -> str:
        """Normalise real USD paths and mocked path objects to plain strings."""
        return getattr(path_value, "pathString", str(path_value))

    @monitor_performance("mesh_ops.extract_mesh_data")
    def extract_mesh_data(
        self, mesh_prim: Usd.Prim, time_code: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Extract a mesh's arrays as Python lists (every point, face and index).

        This expands the whole mesh; use ``get_mesh_statistics`` for numbers
        about a large one.

        Args:
            mesh_prim: USD mesh prim
            time_code: Time code for animated data

        Returns:
            Dictionary containing mesh data
        """
        if not mesh_prim.IsA(UsdGeom.Mesh):
            raise ValueError(f"Prim {mesh_prim.GetPath()} is not a mesh")

        mesh = UsdGeom.Mesh(mesh_prim)
        time_code = self._default_time_code() if time_code is None else time_code

        points = mesh.GetPointsAttr().Get(time_code)
        face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get(time_code)
        face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get(time_code)

        points_array = np.array(points) if points else np.array([])
        face_counts_array = np.array(face_vertex_counts) if face_vertex_counts else np.array([])
        face_indices_array = np.array(face_vertex_indices) if face_vertex_indices else np.array([])

        normals = None
        normals_attr = mesh.GetNormalsAttr()
        if normals_attr and normals_attr.Get(time_code):
            normals = np.array(normals_attr.Get(time_code))

        # First UV primvar under a common name wins
        uvs = None
        primvars = UsdGeom.PrimvarsAPI(mesh.GetPrim()).GetPrimvars()
        for primvar in primvars:
            if primvar.GetPrimvarName() in ['st', 'uv', 'texCoord']:
                uv_data = primvar.Get(time_code)
                if uv_data:
                    uvs = np.array(uv_data)
                    break

        colors = None
        for primvar in primvars:
            if primvar.GetPrimvarName() in ['displayColor', 'color']:
                color_data = primvar.Get(time_code)
                if color_data:
                    colors = np.array(color_data)
                    break

        subdivision_scheme = mesh.GetSubdivisionSchemeAttr().Get() or "none"

        mesh_data = {
            'prim_path': self._path_to_str(mesh_prim.GetPath()),
            'points': points_array.tolist() if points_array.size > 0 else [],
            'face_vertex_counts': face_counts_array.tolist() if face_counts_array.size > 0 else [],
            'face_vertex_indices': face_indices_array.tolist() if face_indices_array.size > 0 else [],
            'normals': normals.tolist() if normals is not None else None,
            'uvs': uvs.tolist() if uvs is not None else None,
            'colors': colors.tolist() if colors is not None else None,
            'subdivision_scheme': subdivision_scheme,
            'face_varying_linear_interpolation': mesh.GetFaceVaryingLinearInterpolationAttr().Get(),
            'interpolate_boundary': mesh.GetInterpolateBoundaryAttr().Get(),
            'vertex_count': len(points_array) if points_array.size > 0 else 0,
            'face_count': len(face_counts_array) if face_counts_array.size > 0 else 0,
        }
        return mesh_data

    @monitor_performance("mesh_ops.get_mesh_statistics")
    def get_mesh_statistics(
        self, mesh_prim: Usd.Prim, time_code: Optional[Any] = None
    ) -> MeshInfo:
        """
        Get statistics about a mesh, vectorised over its faces.

        Args:
            mesh_prim: USD mesh prim
            time_code: Time code to read points and topology at; None for the
                default time

        Returns:
            MeshInfo; ``surface_area`` is None for invalid topology, ``volume``
            None unless the mesh is closed and consistently oriented (a
            closed mesh with flipped faces says so in ``note``), ``bbox``
            (the points' bounds in the mesh's own space) None without points. A deforming mesh whose
            points (or topology) exist only as time samples has nothing to
            measure at the default time: it comes back ``measured`` False,
            its topology and geometry None (not invalid), with its
            ``sampled_time_range`` and a ``note`` to pass ``time_code``.
        """
        if not mesh_prim.IsA(UsdGeom.Mesh):
            raise ValueError(f"Prim {mesh_prim.GetPath()} is not a mesh")

        time = resolve_time_code(time_code)
        mesh = UsdGeom.Mesh(mesh_prim)
        arrays = read_mesh_arrays(mesh, time)
        unsampled = unsampled_topology(mesh, time) if not (
            len(arrays.points) and len(arrays.counts) and len(arrays.indices)
        ) else {}
        sampled_range = None
        note = None
        if unsampled:
            sampled_range = [
                min(span[0] for span in unsampled.values()),
                max(span[1] for span in unsampled.values()),
            ]
            note = (
                f"{', '.join(sorted(unsampled))} has no value at "
                f"{'the default time' if time.IsDefault() else f'time {time.GetValue():g}'}; it is "
                f"time-sampled from {sampled_range[0]:g} to {sampled_range[1]:g}. Pass a time_code in "
                "that range to measure the mesh."
            )
            geometry = GeometryStats(None, None, None, None, None)
        else:
            geometry = geometry_statistics([arrays])[0]
            if geometry.is_closed and geometry.consistently_oriented is False:
                note = (
                    "The mesh is closed, but its faces are wound inconsistently: some edge is run in the "
                    "same direction by both faces sharing it, so a face is flipped. The enclosed volume "
                    "computed from that winding would be wrong, so volume is null; reverse the flipped "
                    "faces' vertex order to measure it."
                )
        attributes = mesh_attributes(mesh)
        subsets = self._get_geometry_subsets(mesh_prim)
        own_material = bound_material(mesh_prim)
        materials = [own_material] if own_material else []
        for subset in subsets:
            if subset.get("material") and subset["material"] not in materials:
                materials.append(subset["material"])
        return MeshInfo(
            prim_path=self._path_to_str(mesh_prim.GetPath()),
            vertex_count=len(arrays.points),
            face_count=len(arrays.counts),
            point_count=len(arrays.indices),
            has_normals=attributes.normals_source is not None,
            has_uvs=bool(attributes.uv_sets),
            has_colors=attributes.has_colors,
            has_tangents=attributes.has_tangents,
            subdivision_scheme=mesh.GetSubdivisionSchemeAttr().Get() or "none",
            face_varying_linear_interpolation=mesh.GetFaceVaryingLinearInterpolationAttr().Get(),
            interpolate_boundary=mesh.GetInterpolateBoundaryAttr().Get(),
            topology_valid=geometry.topology_valid,
            is_closed=geometry.is_closed,
            consistently_oriented=geometry.consistently_oriented,
            surface_area=geometry.surface_area,
            volume=geometry.volume,
            bbox=geometry.bbox,
            materials=materials,
            subsets=subsets,
            normals_source=attributes.normals_source,
            uv_sets=attributes.uv_sets,
            physics_material=physics_material(mesh_prim),
            points_time_samples=mesh.GetPointsAttr().GetNumTimeSamples(),
            time_code=time_code_value(time),
            measured=not unsampled,
            sampled_time_range=sampled_range,
            note=note,
        )

    # The helpers below take arrays as the per-face implementation did and run
    # the vectorised code on them.

    @staticmethod
    def _arrays(points: Any, face_counts: Any, face_indices: Any) -> MeshArrays:
        return MeshArrays(
            points=_as_array(points, np.float64, 3),
            counts=_as_array(face_counts, np.int64),
            indices=_as_array(face_indices, np.int64),
        )

    def _validate_mesh_topology(self, face_counts: np.ndarray, face_indices: np.ndarray, vertex_count: int) -> bool:
        """Validate mesh topology."""
        points = np.zeros((int(vertex_count), 3))
        return geometry_statistics([self._arrays(points, face_counts, face_indices)])[0].topology_valid

    def _is_mesh_closed(self, points: np.ndarray, face_counts: np.ndarray, face_indices: np.ndarray) -> bool:
        """Whether every undirected edge is shared by exactly two faces."""
        return geometry_statistics([self._arrays(points, face_counts, face_indices)])[0].is_closed

    def _calculate_surface_area(self, points: np.ndarray, face_counts: np.ndarray, face_indices: np.ndarray) -> float:
        """Surface area of the fan triangulation; 0.0 for an empty or invalid mesh."""
        area = geometry_statistics([self._arrays(points, face_counts, face_indices)])[0].surface_area
        return area or 0.0

    def _calculate_volume(self, points: np.ndarray, face_counts: np.ndarray, face_indices: np.ndarray) -> float:
        """Enclosed volume by the divergence theorem (meaningful for closed meshes)."""
        return _raw_volume(self._arrays(points, face_counts, face_indices))

    def _has_tangents(self, mesh: UsdGeom.Mesh) -> bool:
        """Check if a mesh has tangent data."""
        return mesh_attributes(mesh).has_tangents

    def _get_mesh_materials(self, mesh_prim: Usd.Prim) -> List[str]:
        """The material bound to the mesh, as ``UsdShade.MaterialBindingAPI`` resolves it."""
        material = bound_material(mesh_prim)
        return [material] if material else []

    def _get_geometry_subsets(
        self, mesh_prim: Usd.Prim, index_preview: int = DEFAULT_ARRAY_PREVIEW
    ) -> List[Dict[str, Any]]:
        """Describe the mesh's GeomSubset children, with each material subset's bound material.

        Indices are summarised (``index_count`` and the first ``index_preview``
        of them), so a subset over a million faces stays small.
        """
        subsets = []
        for child in mesh_prim.GetChildren():
            if not child.IsA(UsdGeom.Subset):
                continue
            subset = UsdGeom.Subset(child)
            family = subset.GetFamilyNameAttr().Get()
            indices = describe_array(subset.GetIndicesAttr().Get() or [], index_preview)
            entry: Dict[str, Any] = {
                'name': child.GetName(),
                'path': self._path_to_str(child.GetPath()),
                'element_type': subset.GetElementTypeAttr().Get(),
                'family_name': family,
                'index_count': indices["length"],
                'indices': indices["value"],
            }
            if indices.get("truncated"):
                entry['indices_truncated'] = True
            if family == UsdShade.Tokens.materialBind:
                entry['material'] = bound_material(child)
            subsets.append(entry)
        return subsets


def mesh_info_to_dict(info: MeshInfo) -> Dict[str, Any]:
    """The ``get_mesh_info`` response fields for a ``MeshInfo``."""
    bbox = None
    if info.bbox is not None:
        bbox = {"min": list(info.bbox[0]), "max": list(info.bbox[1])}
    return {
        "prim_path": info.prim_path,
        "vertex_count": info.vertex_count,
        "face_count": info.face_count,
        "point_count": info.point_count,
        "has_normals": info.has_normals,
        "normals_source": info.normals_source,
        "has_uvs": info.has_uvs,
        "uv_sets": info.uv_sets,
        "has_colors": info.has_colors,
        "has_tangents": info.has_tangents,
        "subdivision_scheme": info.subdivision_scheme,
        "topology_valid": info.topology_valid,
        "is_closed": info.is_closed,
        "consistently_oriented": info.consistently_oriented,
        "surface_area": info.surface_area,
        "volume": info.volume,
        "bbox": bbox,
        "materials": info.materials,
        "physics_material": info.physics_material,
        "subsets": info.subsets,
        "points_time_samples": info.points_time_samples,
        "time_code": info.time_code,
        "measured": info.measured,
        "sampled_time_range": info.sampled_time_range,
        "note": info.note,
    }


# Convenience functions
def extract_mesh_data(
    mesh_prim: Usd.Prim, time_code: Optional[Any] = None
) -> Dict[str, Any]:
    """Extract mesh data from a USD mesh prim."""
    ops = MeshOperations()
    return ops.extract_mesh_data(mesh_prim, time_code)


def get_mesh_statistics(
    mesh_prim: Usd.Prim, time_code: Optional[Any] = None
) -> MeshInfo:
    """Get mesh statistics."""
    ops = MeshOperations()
    return ops.get_mesh_statistics(mesh_prim, time_code)


def validate_mesh_topology(mesh_prim: Usd.Prim, time_code: Optional[Any] = None) -> bool:
    """Validate mesh topology."""
    arrays = read_mesh_arrays(UsdGeom.Mesh(mesh_prim), time_code)
    return geometry_statistics([arrays])[0].topology_valid


def compute_mesh_normals(mesh_prim: Usd.Prim) -> bool:
    """Compute mesh normals and store on the prim."""
    try:
        mesh = UsdGeom.Mesh(mesh_prim)
        points = mesh.GetPointsAttr().Get()
        face_counts = mesh.GetFaceVertexCountsAttr().Get()
        face_indices = mesh.GetFaceVertexIndicesAttr().Get()

        if not points or not face_counts or not face_indices:
            logger.error(f"Mesh data missing for {mesh_prim.GetPath()}")
            return False

        points_array = np.array([[p[0], p[1], p[2]] for p in points], dtype=float)
        normals = np.zeros_like(points_array)

        index_offset = 0
        for count in face_counts:
            if count < 3:
                index_offset += count
                continue
            indices = face_indices[index_offset:index_offset + count]
            v0 = points_array[indices[0]]
            v1 = points_array[indices[1]]
            v2 = points_array[indices[2]]
            face_normal = np.cross(v1 - v0, v2 - v0)
            length = np.linalg.norm(face_normal)
            if length > 0:
                face_normal = face_normal / length
                for idx in indices:
                    normals[idx] += face_normal
            index_offset += count

        for i in range(len(normals)):
            length = np.linalg.norm(normals[i])
            if length > 0:
                normals[i] = normals[i] / length

        mesh.GetNormalsAttr().Set([tuple(n) for n in normals.tolist()])
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        logger.info(f"Computed normals for {mesh_prim.GetPath()}")
        return True
    except Exception as e:
        logger.error(f"Failed to compute normals for {mesh_prim.GetPath()}: {e}")
        return False


def decimate_mesh(mesh_prim: Usd.Prim, target_faces: int) -> bool:
    """Naively decimate a mesh by truncating faces."""
    try:
        mesh = UsdGeom.Mesh(mesh_prim)
        face_counts = list(mesh.GetFaceVertexCountsAttr().Get())
        face_indices = list(mesh.GetFaceVertexIndicesAttr().Get())
        total_faces = len(face_counts)

        if target_faces <= 0:
            logger.error(f"Target face count must be positive: {target_faces}")
            return False

        if target_faces >= total_faces:
            logger.info(f"Mesh {mesh_prim.GetPath()} already <= target faces")
            return True

        trimmed_counts = face_counts[:target_faces]
        trimmed_indices = face_indices[:sum(trimmed_counts)]

        mesh.GetFaceVertexCountsAttr().Set(trimmed_counts)
        mesh.GetFaceVertexIndicesAttr().Set(trimmed_indices)
        logger.info(f"Decimated mesh {mesh_prim.GetPath()} to {target_faces} faces")
        return True
    except Exception as e:
        logger.error(f"Failed to decimate mesh {mesh_prim.GetPath()}: {e}")
        return False


def export_mesh_data(mesh_prim: Usd.Prim, output_path: str, format: str = "obj") -> bool:
    """Export mesh data to an OBJ file."""
    if format.lower() != "obj":
        logger.error(f"Unsupported export format: {format}")
        return False

    try:
        mesh = UsdGeom.Mesh(mesh_prim)
        points = mesh.GetPointsAttr().Get()
        face_counts = mesh.GetFaceVertexCountsAttr().Get()
        face_indices = mesh.GetFaceVertexIndicesAttr().Get()

        if not points or not face_counts or not face_indices:
            logger.error(f"Mesh data missing for {mesh_prim.GetPath()}")
            return False

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with output_path_obj.open("w", encoding="utf-8") as handle:
            for point in points:
                handle.write(f"v {point[0]} {point[1]} {point[2]}\n")

            index_offset = 0
            for count in face_counts:
                indices = face_indices[index_offset:index_offset + count]
                if indices:
                    face = " ".join(str(idx + 1) for idx in indices)
                    handle.write(f"f {face}\n")
                index_offset += count

        logger.info(f"Exported mesh {mesh_prim.GetPath()} to {output_path_obj}")
        return True
    except Exception as e:
        logger.error(f"Failed to export mesh {mesh_prim.GetPath()}: {e}")
        return False
