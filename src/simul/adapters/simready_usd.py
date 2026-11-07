"""
Turn the USD layer Blender's exporter wrote into a SimReady asset a simulator reads.

Blender's USD exporter writes geometry, transforms and render materials, and
custom properties only as plain ``userProperties:*`` attributes. A simulator
reads none of those as physics or semantics. ``author_simready_layer`` opens
that layer and:

- makes the asset root the default prim: the one top-level object, or, when
  several objects share the top level, an Xform named after the file. It
  removes Blender's generated ``/root`` container and moves the render
  materials into ``<asset root>/Looks``, so the asset is self-contained;
- sets ``metersPerUnit`` from the scene's unit scale, ``upAxis`` and
  ``kilogramsPerUnit``;
- authors UsdPhysics schemas from each object's ``simready_*`` metadata:
  ``RigidBodyAPI``, ``MassAPI`` (mass, and density converted to the stage's
  units), ``CollisionAPI`` with a ``MeshCollisionAPI`` approximation on the
  object's meshes, and a physics material (``UsdPhysics.MaterialAPI`` on a
  ``UsdShade.Material`` under ``<asset root>/PhysicsMaterials``) bound to those
  colliders with the ``physics`` purpose;
- authors semantics with ``UsdSemantics.LabelsAPI`` (``SemanticsLabelsAPI:<taxonomy>``
  with a ``semantics:labels:<taxonomy>`` token array), or, where this pxr has no
  UsdSemantics, Isaac Sim's older ``SemanticsAPI`` convention
  (``semantic:Semantics_<taxonomy>:params:semanticType`` / ``semanticData``).

The module needs pxr and nothing else: the Blender add-on bundles it next to
``blender_runtime``, where Blender's own pxr runs it, and the server's
usd-core runs it in tests.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

#: Blender writes the object's own name on each object prim (``author_blender_name``).
BLENDER_OBJECT_NAME = "userProperties:blender:object_name"
#: Namespace of every custom property Blender exports.
USER_PROPERTIES = "userProperties:"
#: Scope under the asset root that holds the render materials.
LOOKS_SCOPE = "Looks"
#: Scope under the asset root that holds the physics materials.
PHYSICS_MATERIALS_SCOPE = "PhysicsMaterials"
#: Material binding purpose PhysX and UsdPhysics read physics materials from.
PHYSICS_PURPOSE = "physics"

#: collider_type -> UsdPhysics.MeshCollisionAPI approximation token. The keys
#: are the UsdPhysics tokens themselves plus the earlier SimReady tool tokens.
COLLIDER_APPROXIMATIONS: Dict[str, str] = {
    "convexHull": "convexHull",
    "convexDecomposition": "convexDecomposition",
    "meshSimplification": "meshSimplification",
    "none": "none",
    "boundingCube": "boundingCube",
    "boundingSphere": "boundingSphere",
    "mesh": "none",
    "box": "boundingCube",
    "sphere": "boundingSphere",
    # UsdPhysics has no capsule approximation of a mesh.
    "capsule": "convexHull",
}

#: Semantic metadata key -> semantics taxonomy (the LabelsAPI instance name).
SEMANTIC_TAXONOMIES: Tuple[Tuple[str, str], ...] = (
    ("semantic_class", "class"),
    ("semantic_hierarchy", "hierarchy"),
    ("semantic_qcode", "qcode"),
)

LABELS_API_SCHEMA = "UsdSemantics.LabelsAPI"
LEGACY_SEMANTICS_SCHEMA = "Isaac SemanticsAPI"

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def usd_identifier(name: str, fallback: str = "asset") -> str:
    """A valid USD prim name for ``name``: invalid characters become underscores."""
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not cleaned:
        return fallback
    if cleaned[0].isdigit():
        cleaned = "_" + cleaned
    return cleaned


def labels_api_available() -> bool:
    """Whether this pxr has the UsdSemantics module (Blender 5.0.1 bundles it)."""
    try:
        from pxr import UsdSemantics  # noqa: F401
    except ImportError:
        return False
    return True


def author_simready_layer(
    source_path: str,
    plan: Mapping[str, Any],
    *,
    use_labels_api: Optional[bool] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Read Blender's export at ``source_path`` and author the SimReady asset in memory.

    Args:
        source_path: The layer Blender's USD exporter wrote (.usd, .usda or .usdc).
        plan: What was exported. ``objects``: one entry per exported Blender
            object, ``{"name", "parent", "physics", "semantic"}``, where
            ``parent`` is the Blender parent's name (or None) and ``physics`` /
            ``semantic`` hold the object's ``simready_*`` metadata.
            ``asset_name``: the root prim's name when several objects share the
            top level. ``meters_per_unit``: the scene's unit scale.
            ``embed_metadata``: keep the ``userProperties:*`` attributes.
        use_labels_api: Force UsdSemantics LabelsAPI (True) or Isaac's
            SemanticsAPI convention (False); None picks LabelsAPI when this pxr
            has it.

    Returns:
        The anonymous layer, unsaved (``layer.Export(path)`` writes it), and the
        report: ``default_prim``, ``prims`` (Blender object name -> prim path),
        ``missing_objects`` (planned objects the layer has no prim for), the
        stage metrics, the physics and semantics prims authored, and
        ``warnings``.

    Raises:
        RuntimeError: When the layer cannot be read or holds no Blender object.
    """
    from pxr import Sdf, Usd

    layer = Sdf.Layer.OpenAsAnonymous(source_path)
    if layer is None:
        raise RuntimeError(f"Could not read the exported USD layer {source_path}")
    report: Dict[str, Any] = {"warnings": []}
    objects: List[Mapping[str, Any]] = list(plan.get("objects") or [])

    found = _object_prims(layer) or _object_prims_by_name(
        layer, [str(o["name"]) for o in objects]
    )
    if not found:
        raise RuntimeError(
            "The exported layer holds no Blender object prims "
            f"(no {BLENDER_OBJECT_NAME} attribute); nothing was exported"
        )
    layout = _plan_layout(
        layer, found, usd_identifier(str(plan.get("asset_name") or "asset")), report
    )
    _apply_layout(layer, layout)
    root_path = layout.root
    prims = {name: _remap(path, layout.moves) for name, path in found.items()}
    layer.defaultPrim = root_path.name

    stage = Usd.Stage.Open(layer)
    _set_metrics(stage, float(plan.get("meters_per_unit") or 1.0), report)
    root = stage.GetPrimAtPath(root_path)
    _set_component_kind(root)

    planned = {str(o["name"]): o for o in objects}
    present = {name: path for name, path in prims.items() if name in planned}
    report["missing_objects"] = sorted(name for name in planned if name not in prims)
    report["default_prim"] = str(root_path)
    report["prims"] = {name: str(path) for name, path in present.items()}

    _author_physics(stage, root_path, present, planned, report)
    _author_semantics(
        stage,
        present,
        planned,
        report,
        labels_api_available() if use_labels_api is None else use_labels_api,
    )
    if not plan.get("embed_metadata", True):
        _strip_user_properties(layer)
    return layer, report


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def _prim_specs(spec: Any) -> Iterable[Any]:
    """Every prim spec below ``spec``, depth first."""
    for child in spec.nameChildren:
        yield child
        yield from _prim_specs(child)


def _object_prims(layer: Any) -> Dict[str, Any]:
    """Blender object name -> prim path, from the names Blender authors on object prims."""
    found: Dict[str, Any] = {}
    for spec in _prim_specs(layer.pseudoRoot):
        attribute = spec.attributes.get(BLENDER_OBJECT_NAME)
        if attribute is not None and attribute.default is not None:
            found.setdefault(str(attribute.default), spec.path)
    return found


def _object_prims_by_name(layer: Any, names: Iterable[str]) -> Dict[str, Any]:
    """Object prims matched by prim name, for an exporter that authors no Blender names."""
    wanted = {usd_identifier(name): name for name in names}
    found: Dict[str, Any] = {}
    for spec in _prim_specs(layer.pseudoRoot):
        name = wanted.get(spec.name)
        if name is not None and name not in found and spec.typeName == "Xform":
            found[name] = spec.path
    return found


@dataclass
class _Layout:
    """Where Blender's top level goes.

    Attributes:
        root: The asset root's path, the default prim.
        moves: Prim path -> new path, for every prim that moves.
        removals: Prims removed once the moved ones are copied.
        wrap: A new Xform at ``root`` holds the top-level prims.
    """

    root: Any
    moves: Dict[Any, Any] = field(default_factory=dict)
    removals: List[Any] = field(default_factory=list)
    wrap: bool = False


def _plan_layout(
    layer: Any, found: Mapping[str, Any], asset_name: str, report: Dict[str, Any]
) -> _Layout:
    """Plan the asset root, and where the top-level prims and the materials move."""
    from pxr import Sdf

    pseudo_root = Sdf.Path.absoluteRootPath
    object_paths = set(found.values())
    top = sorted(
        (p for p in object_paths if p.GetParentPath() not in object_paths), key=str
    )
    containers = {p.GetParentPath() for p in top}
    if len(containers) != 1:
        raise RuntimeError(
            "The exported objects do not share one parent prim: "
            + ", ".join(sorted(str(c) for c in containers))
        )
    container_path = containers.pop()
    container = layer.GetPrimAtPath(container_path)
    if container_path != pseudo_root and "xformOpOrder" in container.attributes:
        report["warnings"].append(
            f"Blender's root prim {container_path} carried a transform, which was dropped"
        )

    materials: List[Any] = []
    others: List[Any] = []
    scopes: List[Any] = []
    for child in container.nameChildren:
        if child.path in object_paths:
            continue
        children = list(child.nameChildren)
        if children and all(c.typeName == "Material" for c in children):
            materials.extend(c.path for c in children)
            scopes.append(child.path)
        else:
            others.append(child.path)

    # Names the new top level must not reuse while the old one still exists.
    taken = {c.name for c in layer.pseudoRoot.nameChildren}
    if len(top) == 1 and not others:
        name = top[0].name
        if container_path != pseudo_root and name in taken:
            name = _free_name(name, taken)
        layout = _Layout(root=pseudo_root.AppendChild(name))
        if layout.root != top[0]:
            layout.moves[top[0]] = layout.root
        root_children = {c.name for c in layer.GetPrimAtPath(top[0]).nameChildren}
    else:
        layout = _Layout(
            root=pseudo_root.AppendChild(_free_name(asset_name, taken)), wrap=True
        )
        for path in top + others:
            layout.moves[path] = layout.root.AppendChild(path.name)
        root_children = {path.name for path in top + others}
        report["warnings"].append(
            f"{len(top)} objects share the top level, so they were put under the new "
            f"root {layout.root}; give the asset one root (setup_simready_hierarchy) "
            "to choose its name"
        )

    looks = layout.root.AppendChild(_free_name(LOOKS_SCOPE, root_children))
    used: set = set()
    for material in materials:
        name = _free_name(material.name, used)
        used.add(name)
        layout.moves[material] = looks.AppendChild(name)

    if container_path != pseudo_root:
        layout.removals = [container_path]
    else:
        layout.removals = [p for p in top + others if p in layout.moves] + scopes
    return layout


def _free_name(name: str, taken: Iterable[str]) -> str:
    """``name`` or the first ``name_<n>`` not in ``taken``."""
    used = set(taken)
    candidate, index = name, 1
    while candidate in used:
        candidate = f"{name}_{index}"
        index += 1
    return candidate


def _apply_layout(layer: Any, layout: _Layout) -> None:
    """Copy each moved prim to its new path, remove the old top level, then fix every path."""
    from pxr import Sdf

    if layout.wrap:
        Sdf.PrimSpec(layer.pseudoRoot, layout.root.name, Sdf.SpecifierDef, "Xform")
    # Parents first: the asset root before the Looks scope inside it.
    for source, destination in sorted(
        layout.moves.items(), key=lambda move: move[1].pathElementCount
    ):
        parent_path = destination.GetParentPath()
        if layer.GetPrimAtPath(parent_path) is None:
            Sdf.PrimSpec(
                layer.GetPrimAtPath(parent_path.GetParentPath()),
                parent_path.name,
                Sdf.SpecifierDef,
                "Scope",
            )
        if not Sdf.CopySpec(layer, source, layer, destination):
            raise RuntimeError(f"Could not move {source} to {destination}")
    edit = Sdf.BatchNamespaceEdit()
    for path in layout.removals:
        edit.Add(Sdf.NamespaceEdit.Remove(path))
    if layout.removals and not layer.Apply(edit):
        raise RuntimeError("Could not remove the prims Blender's exporter generated")
    _remap_layer_paths(layer, layout.moves)


def _remap(path: Any, moves: Mapping[Any, Any]) -> Any:
    """``path`` under its moved ancestor's new location (the deepest match wins)."""
    best = None
    for source in moves:
        if path.HasPrefix(source) and (
            best is None or source.pathElementCount > best.pathElementCount
        ):
            best = source
    return path if best is None else path.ReplacePrefix(best, moves[best])


_PATH_LISTS = (
    "explicitItems",
    "addedItems",
    "prependedItems",
    "appendedItems",
    "deletedItems",
    "orderedItems",
)


def _remap_list(proxy: Any, remap: Callable[[Any], Any]) -> None:
    """Rewrite every list of a list-op proxy (targets, connections, inherits) through ``remap``."""
    for name in _PATH_LISTS:
        items = list(getattr(proxy, name))
        updated = [remap(item) for item in items]
        if updated != items:
            getattr(proxy, name)[:] = updated


def _remap_layer_paths(layer: Any, moves: Mapping[Any, Any]) -> None:
    """Point relationships, connections, inherits and internal references at the moved prims."""
    from pxr import Sdf

    def remap_path(path: Any) -> Any:
        if path.IsPropertyPath():
            return _remap(path.GetPrimPath(), moves).AppendProperty(path.name)
        return _remap(path, moves)

    def remap_reference(reference: Any) -> Any:
        if reference.assetPath or reference.primPath.isEmpty:
            return reference
        return Sdf.Reference(
            reference.assetPath,
            remap_path(reference.primPath),
            reference.layerOffset,
            reference.customData,
        )

    for spec in _prim_specs(layer.pseudoRoot):
        for prop in spec.properties:
            if isinstance(prop, Sdf.RelationshipSpec):
                _remap_list(prop.targetPathList, remap_path)
            elif isinstance(prop, Sdf.AttributeSpec):
                _remap_list(prop.connectionPathList, remap_path)
        _remap_list(spec.inheritPathList, remap_path)
        _remap_list(spec.specializesList, remap_path)
        _remap_list(spec.referenceList, remap_reference)


# ---------------------------------------------------------------------------
# Stage metadata
# ---------------------------------------------------------------------------


def _set_metrics(stage: Any, meters_per_unit: float, report: Dict[str, Any]) -> None:
    from pxr import UsdGeom, UsdPhysics

    if not math.isfinite(meters_per_unit) or meters_per_unit <= 0:
        report["warnings"].append(
            f"Scene unit scale {meters_per_unit} is not positive; wrote metersPerUnit 1.0"
        )
        meters_per_unit = 1.0
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    report["meters_per_unit"] = meters_per_unit
    report["up_axis"] = "Z"
    report["kilograms_per_unit"] = 1.0


def _set_component_kind(root: Any) -> None:
    from pxr import Kind, Usd

    Usd.ModelAPI(root).SetKind(Kind.Tokens.component)


def _strip_user_properties(layer: Any) -> None:
    """Remove every ``userProperties:*`` attribute, as Blender does without custom properties."""
    for spec in _prim_specs(layer.pseudoRoot):
        for prop in list(spec.properties):
            if prop.name.startswith(USER_PROPERTIES):
                spec.RemoveProperty(prop)


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------


def _number(
    value: Any, field: str, owner: str, report: Dict[str, Any], *, upper: Optional[float] = None
) -> Optional[float]:
    """``value`` as a finite non-negative float, or None with a warning."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float("nan")
    if isinstance(value, bool) or not math.isfinite(number) or number < 0 or (
        upper is not None and number > upper
    ):
        bound = f" and at most {upper}" if upper is not None else ""
        report["warnings"].append(
            f"{owner}: {field}={value!r} is not a number of at least 0{bound}; not exported"
        )
        return None
    return number


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


_FRICTION_KEYS = ("static_friction", "dynamic_friction", "restitution")


def _nearest(
    name: str,
    planned: Mapping[str, Mapping[str, Any]],
    predicate: Callable[[Mapping[str, Any]], bool],
) -> Optional[Mapping[str, Any]]:
    """The object itself or its nearest exported ancestor that satisfies ``predicate``."""
    seen = set()
    current: Optional[str] = name
    while current is not None and current not in seen and current in planned:
        seen.add(current)
        entry = planned[current]
        if predicate(entry):
            return entry
        parent = entry.get("parent")
        current = None if parent is None else str(parent)
    return None


def _physics(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    physics = entry.get("physics")
    return physics if isinstance(physics, Mapping) else {}


def _author_physics(
    stage: Any,
    root_path: Any,
    prims: Mapping[str, Any],
    planned: Mapping[str, Mapping[str, Any]],
    report: Dict[str, Any],
) -> None:
    """Rigid bodies, mass, colliders and physics materials from the objects' metadata.

    ``is_rigid_body``, ``mass_kg`` and ``density`` apply to the object's own
    prim. ``collider_type`` and the friction/restitution values also apply to
    the meshes of descendants that set none of their own, like an inherited
    material binding, so one call on an asset root makes all its meshes collide.
    """
    from pxr import Sdf, UsdGeom, UsdPhysics, UsdShade

    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    rigid_bodies: List[str] = []
    mass_prims: List[str] = []
    colliders: List[str] = []
    approximations: Dict[str, str] = {}
    materials: Dict[str, Any] = {}
    material_scope = None
    warnings: List[str] = report["warnings"]

    def physics_material(source: Mapping[str, Any]) -> Any:
        nonlocal material_scope
        key = str(source["name"])
        if key in materials:
            return materials[key]
        physics = _physics(source)
        static = _number(physics.get("static_friction"), "static_friction", key, report)
        dynamic = _number(physics.get("dynamic_friction"), "dynamic_friction", key, report)
        restitution = _number(
            physics.get("restitution"), "restitution", key, report, upper=1.0
        )
        if static is None and dynamic is None and restitution is None:
            materials[key] = None
            return None
        if material_scope is None:
            scope_name = _free_name(
                PHYSICS_MATERIALS_SCOPE,
                (c.GetName() for c in stage.GetPrimAtPath(root_path).GetChildren()),
            )
            material_scope = UsdGeom.Scope.Define(stage, root_path.AppendChild(scope_name))
        taken = {c.GetName() for c in material_scope.GetPrim().GetChildren()}
        name = _free_name(usd_identifier(key, "physics_material"), taken)
        material = UsdShade.Material.Define(stage, material_scope.GetPath().AppendChild(name))
        api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        if static is not None:
            api.CreateStaticFrictionAttr(static)
        if dynamic is not None:
            api.CreateDynamicFrictionAttr(dynamic)
        if restitution is not None:
            api.CreateRestitutionAttr(restitution)
        materials[key] = material
        return material

    for name in sorted(prims, key=lambda n: str(prims[n])):
        prim = stage.GetPrimAtPath(prims[name])
        entry = planned[name]
        physics = _physics(entry)

        if _truthy(physics.get("is_rigid_body")):
            UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(True)
            rigid_bodies.append(str(prim.GetPath()))
        mass = _number(physics.get("mass_kg"), "mass_kg", name, report)
        density = _number(physics.get("density"), "density", name, report)
        if mass is not None or density is not None:
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            if mass is not None:
                mass_api.CreateMassAttr(mass)
            if density is not None:
                # density is kg/m^3; UsdPhysics reads it in kg per stage unit cubed.
                mass_api.CreateDensityAttr(density * meters_per_unit**3)
            mass_prims.append(str(prim.GetPath()))

        meshes = [child for child in prim.GetChildren() if child.IsA(UsdGeom.Mesh)]
        collider_source = _nearest(
            name, planned, lambda e: _physics(e).get("collider_type") not in (None, "")
        )
        if collider_source is not None and meshes:
            requested = str(_physics(collider_source)["collider_type"])
            approximation = COLLIDER_APPROXIMATIONS.get(requested)
            if approximation is None:
                warnings.append(
                    f"{collider_source['name']}: collider_type {requested!r} is not one of "
                    f"{sorted(COLLIDER_APPROXIMATIONS)}; no collider exported for {name}"
                )
            else:
                if requested == "capsule":
                    warnings.append(
                        f"{name}: UsdPhysics has no capsule approximation of a mesh; "
                        "exported a convexHull collider"
                    )
                material_source = _nearest(
                    name, planned, lambda e: any(k in _physics(e) for k in _FRICTION_KEYS)
                )
                material = (
                    physics_material(material_source) if material_source is not None else None
                )
                for mesh in meshes:
                    UsdPhysics.CollisionAPI.Apply(mesh)
                    UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr(
                        approximation
                    )
                    colliders.append(str(mesh.GetPath()))
                    approximations[str(mesh.GetPath())] = approximation
                    if material is not None:
                        UsdShade.MaterialBindingAPI.Apply(mesh).Bind(
                            material, UsdShade.Tokens.fallbackStrength, PHYSICS_PURPOSE
                        )
        elif collider_source is not None and collider_source is entry and not _has_mesh_below(
            prim
        ):
            warnings.append(f"{name}: collider_type is set but the object has no mesh")
        if (
            collider_source is None
            and any(k in physics for k in _FRICTION_KEYS)
            and not _has_mesh_below(prim)
        ):
            warnings.append(
                f"{name}: friction/restitution are set but no collider uses them; "
                "set collider_type"
            )

    for body in rigid_bodies:
        path = Sdf.Path(body)
        if not any(Sdf.Path(c).HasPrefix(path) for c in colliders):
            warnings.append(
                f"{body}: rigid body without a collider; set collider_type on it or its meshes"
            )
        for other in rigid_bodies:
            if other != body and path.HasPrefix(Sdf.Path(other)):
                warnings.append(f"{body}: rigid body nested under the rigid body {other}")
        for collider, approximation in approximations.items():
            if approximation == "none" and Sdf.Path(collider).HasPrefix(path):
                warnings.append(
                    f"{collider}: a triangle-mesh collider (collider_type none/mesh) on the "
                    f"dynamic body {body} is not simulated by PhysX; use convexHull or "
                    "convexDecomposition"
                )

    report["rigid_bodies"] = rigid_bodies
    report["mass_prims"] = mass_prims
    report["colliders"] = colliders
    report["physics_materials"] = sorted(
        str(m.GetPath()) for m in materials.values() if m is not None
    )


def _has_mesh_below(prim: Any) -> bool:
    from pxr import Usd, UsdGeom

    return any(p.IsA(UsdGeom.Mesh) for p in Usd.PrimRange(prim))


# ---------------------------------------------------------------------------
# Semantics
# ---------------------------------------------------------------------------


def _labels(entry: Mapping[str, Any], name: str, report: Dict[str, Any]) -> List[Tuple[str, str]]:
    semantic = entry.get("semantic")
    if not isinstance(semantic, Mapping):
        return []
    labels: List[Tuple[str, str]] = []
    for key, taxonomy in SEMANTIC_TAXONOMIES:
        value = semantic.get(key)
        if value not in (None, ""):
            labels.append((taxonomy, str(value)))
    extra = semantic.get("additional_labels")
    if isinstance(extra, Mapping):
        for key, value in extra.items():
            if value in (None, ""):
                continue
            if not _IDENTIFIER.match(str(key)):
                report["warnings"].append(
                    f"{name}: label {key!r} is not a valid USD identifier; not exported"
                )
                continue
            labels.append((str(key), str(value)))
    return labels


def _author_semantics(
    stage: Any,
    prims: Mapping[str, Any],
    planned: Mapping[str, Mapping[str, Any]],
    report: Dict[str, Any],
    use_labels_api: bool,
) -> None:
    from pxr import Sdf, Vt

    labelled: List[str] = []
    for name, path in sorted(prims.items(), key=lambda item: str(item[1])):
        labels = _labels(planned[name], name, report)
        if not labels:
            continue
        prim = stage.GetPrimAtPath(path)
        for taxonomy, value in labels:
            if use_labels_api:
                from pxr import UsdSemantics

                UsdSemantics.LabelsAPI.Apply(prim, taxonomy).CreateLabelsAttr(
                    Vt.TokenArray([value])
                )
            else:
                instance = f"Semantics_{taxonomy}"
                prim.AddAppliedSchema(f"SemanticsAPI:{instance}")
                prefix = f"semantic:{instance}:params:"
                prim.CreateAttribute(
                    prefix + "semanticType", Sdf.ValueTypeNames.String, custom=False
                ).Set(taxonomy)
                prim.CreateAttribute(
                    prefix + "semanticData", Sdf.ValueTypeNames.String, custom=False
                ).Set(value)
        labelled.append(str(path))
    report["semantics_schema"] = LABELS_API_SCHEMA if use_labels_api else LEGACY_SEMANTICS_SCHEMA
    report["labelled_prims"] = labelled
