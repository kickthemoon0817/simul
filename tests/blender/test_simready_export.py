"""export_simready_usd writes USD a simulator reads, checked with pxr.

A stand-in ``bpy`` holds the scene; its ``wm.usd_export`` writes the layer
Blender 5.0.1's exporter writes (checked against real exports): every object
as an Xform under ``/root`` named after it, carrying
``userProperties:blender:object_name`` and its custom properties, its mesh as
a child Mesh bound to ``/root/_materials``, the World as ``/root/env_light``,
non-selected ancestors of selected objects as bare Xforms, and nothing for an
object that is hidden in renders or could not be selected because it is
hidden. The tests then open the written file with the server's pxr, so the
whole export runs in CI without a Blender.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from simul.adapters import blender_runtime
from simul.utils.paths import PathPolicy
from tests.fakes import FakeMatrix

# ---------------------------------------------------------------------------
# A stand-in Blender scene and exporter
# ---------------------------------------------------------------------------


class FakeObject:
    """A Blender object: hierarchy, custom properties, visibility and selection."""

    def __init__(
        self,
        name: str,
        kind: str = "MESH",
        parent: Optional["FakeObject"] = None,
        materials: Optional[List[str]] = None,
        dimensions: tuple = (1.0, 1.0, 1.0),
        data_name: Optional[str] = None,
    ) -> None:
        self.name = name
        self.type = kind
        self.parent = parent
        self.children: List[FakeObject] = []
        if parent is not None:
            parent.children.append(self)
        self.data = (
            SimpleNamespace(name=data_name or name, materials=list(materials or []))
            if kind == "MESH"
            else (None if kind == "EMPTY" else SimpleNamespace(name=data_name or name))
        )
        self.dimensions = dimensions if kind == "MESH" else (0.0, 0.0, 0.0)
        self.location = (0.0, 0.0, 0.0)
        self.rotation_euler = (0.0, 0.0, 0.0)
        self.scale = (1.0, 1.0, 1.0)
        self.props: Dict[str, Any] = {}
        self.hidden = False
        self.hide_viewport = False
        self.hide_render = False
        self.hide_select = False
        self.selected = False
        self.users_collection: List[Any] = []
        # Where Blender places it; the stand-in has no depsgraph, so this is taken as current.
        self.matrix_world = FakeMatrix.Identity(4)
        self.matrix_basis = FakeMatrix.Identity(4)
        self.matrix_parent_inverse = FakeMatrix.Identity(4)
        self.mode = "OBJECT"

    # custom properties
    def __setitem__(self, key: str, value: Any) -> None:
        self.props[key] = value

    def __getitem__(self, key: str) -> Any:
        return self.props[key]

    def keys(self) -> List[str]:
        return list(self.props)

    @property
    def children_recursive(self) -> List["FakeObject"]:
        found: List[FakeObject] = []
        for child in self.children:
            found.append(child)
            found.extend(child.children_recursive)
        return found

    # visibility and selection, as Blender: a hidden object cannot be selected
    def hide_get(self) -> bool:
        return self.hidden

    def hide_set(self, value: bool) -> None:
        self.hidden = value
        if value:
            self.selected = False

    def select_set(self, value: bool) -> None:
        if value and (self.hidden or self.hide_viewport or self.hide_select):
            return
        self.selected = value

    def select_get(self) -> bool:
        return self.selected


class FakeCollection(list):
    """bpy_prop_collection: iterable, with ``get`` by name."""

    def get(self, name: str) -> Optional[FakeObject]:
        return next((o for o in self if o is not None and o.name == name), None)


class LayerObjects(FakeCollection):
    """view_layer.objects: the collection plus the active object."""

    active: Optional[FakeObject] = None


_CUBE = [(-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5),
         (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)]
_CUBE_FACES = [4] * 6
_CUBE_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7]

_EXPORT_OPTIONS = (
    "filepath",
    "selected_objects_only",
    "export_custom_properties",
    "author_blender_name",
    "root_prim_path",
    "export_lights",
    "export_cameras",
    "convert_world_material",
    "convert_scene_units",
)


class FakeBlender:
    """The ``bpy`` module the runtime sees, with a Blender-like USD exporter."""

    def __init__(self, objects: List[FakeObject]) -> None:
        self.objects = FakeCollection(objects)
        self.layer_objects = LayerObjects(objects)
        self.export_calls: List[Dict[str, Any]] = []
        self.mode_calls: List[str] = []
        self.supported = set(_EXPORT_OPTIONS)
        scene = SimpleNamespace(
            objects=self.objects,
            unit_settings=SimpleNamespace(scale_length=1.0),
            world=SimpleNamespace(name="World"),
        )
        self.view_layer = SimpleNamespace(objects=self.layer_objects)
        self.context = SimpleNamespace(
            scene=scene, view_layer=self.view_layer, mode="OBJECT", objects_in_mode=[]
        )
        self.app = SimpleNamespace(version=(5, 0, 1), version_string="5.0.1", background=True)
        self.data = SimpleNamespace(objects=self.objects, filepath="")
        usd_export = self._usd_export
        usd_export_fn: Any = lambda **kw: usd_export(**kw)  # noqa: E731
        usd_export_fn.get_rna_type = lambda: SimpleNamespace(
            properties={name: 0 for name in self.supported}
        )
        self.ops = SimpleNamespace(
            wm=SimpleNamespace(usd_export=usd_export_fn),
            object=SimpleNamespace(mode_set=self._mode_set),
        )

    def _mode_set(self, mode: str) -> set:
        self.mode_calls.append(mode)
        self.context.mode = "OBJECT" if mode == "OBJECT" else f"{mode}_MESH"
        active = self.layer_objects.active
        if active is not None:
            active.mode = mode
        return {"FINISHED"}

    def _usd_export(self, **kwargs: Any) -> set:
        self.export_calls.append(kwargs)
        root = kwargs.get("root_prim_path", "/root")
        custom = kwargs.get("export_custom_properties", True)
        names = kwargs.get("author_blender_name", True)
        selected_only = kwargs.get("selected_objects_only", False)
        allowed = {"MESH", "EMPTY"}
        if kwargs.get("export_lights", True):
            allowed.add("LIGHT")
        if kwargs.get("export_cameras", True):
            allowed.add("CAMERA")

        exported = [
            o
            for o in self.layer_objects
            if o is not None
            and o.type in allowed
            and not o.hide_render
            and (o.selected if selected_only else True)
        ]
        layer = Sdf.Layer.CreateAnonymous("blender.usda")
        stage = Usd.Stage.Open(layer)
        base = Sdf.Path(root) if root else Sdf.Path.absoluteRootPath
        if root:
            UsdGeom.Xform.Define(stage, base)
            stage.GetPrimAtPath(base).SetCustomDataByKey("Blender:generated", True)

        def path_of(obj: FakeObject) -> Sdf.Path:
            chain = []
            current: Optional[FakeObject] = obj
            while current is not None:
                chain.append(current.name.replace(".", "_"))
                current = current.parent
            path = base
            for name in reversed(chain):
                path = path.AppendChild(name)
            return path

        materials: Dict[str, Sdf.Path] = {}
        for obj in exported:
            # Ancestors that were not selected are written as bare transforms.
            ancestor = obj.parent
            while ancestor is not None:
                if not stage.GetPrimAtPath(path_of(ancestor)):
                    UsdGeom.Xform.Define(stage, path_of(ancestor))
                ancestor = ancestor.parent
            xform = UsdGeom.Xform.Define(stage, path_of(obj)).GetPrim()
            local = _local_matrix(obj)
            if not np.allclose(local, np.eye(4)):
                # The transform relative to the parent prim, row vectors as USD's.
                UsdGeom.Xformable(xform).AddTransformOp().Set(Gf.Matrix4d(local.T.tolist()))
            if names:
                xform.CreateAttribute(
                    "userProperties:blender:object_name", Sdf.ValueTypeNames.String
                ).Set(obj.name)
            if custom:
                for key, value in obj.props.items():
                    value_type = (
                        Sdf.ValueTypeNames.String
                        if isinstance(value, str)
                        else Sdf.ValueTypeNames.Double
                    )
                    xform.CreateAttribute(f"userProperties:{key}", value_type).Set(value)
            data_path = xform.GetPath().AppendChild(obj.data.name.replace(".", "_")) if obj.data else None
            if obj.type == "MESH":
                mesh = UsdGeom.Mesh.Define(stage, data_path)
                mesh.CreatePointsAttr(_CUBE)
                mesh.CreateFaceVertexCountsAttr(_CUBE_FACES)
                mesh.CreateFaceVertexIndicesAttr(_CUBE_INDICES)
                for material_name in obj.data.materials:
                    material_path = base.AppendChild("_materials").AppendChild(material_name)
                    if material_name not in materials:
                        material = UsdShade.Material.Define(stage, material_path)
                        shader = UsdShade.Shader.Define(
                            stage, material_path.AppendChild("Principled_BSDF")
                        )
                        shader.CreateIdAttr("UsdPreviewSurface")
                        material.CreateSurfaceOutput().ConnectToSource(
                            shader.ConnectableAPI(), "surface"
                        )
                        materials[material_name] = material_path
                    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
                        UsdShade.Material(stage.GetPrimAtPath(material_path))
                    )
            elif obj.type == "LIGHT":
                stage.DefinePrim(data_path, "SphereLight")
            elif obj.type == "CAMERA":
                UsdGeom.Camera.Define(stage, data_path)
        if kwargs.get("convert_world_material", True):
            stage.DefinePrim(base.AppendChild("env_light"), "DomeLight")
        if root:
            layer.defaultPrim = base.name
        layer.Export(kwargs["filepath"])
        return {"FINISHED"}


def _local_matrix(obj: FakeObject) -> np.ndarray:
    """The object's matrix relative to its parent's world matrix, as Blender's exporter writes it."""
    world = np.asarray(obj.matrix_world.rows)
    if obj.parent is None:
        return world
    return np.linalg.inv(np.asarray(obj.parent.matrix_world.rows)) @ world


def _robot() -> List[FakeObject]:
    """robot (empty) > base_link (mesh) > arm_link (mesh), plus a light and a camera."""
    robot = FakeObject("robot", "EMPTY")
    robot["simready_semantic_class"] = "robot"  # validation requires a class
    base = FakeObject("base_link", parent=robot, materials=["opaque_metal_steel"], data_name="Cube")
    arm = FakeObject("arm_link", parent=base, materials=["opaque_plastic_abs"], data_name="Cylinder")
    light = FakeObject("Light", "LIGHT")
    camera = FakeObject("Camera", "CAMERA")
    return [robot, base, arm, light, camera]


@pytest.fixture
def scene(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Any]:
    fake = FakeBlender(_robot())
    monkeypatch.setattr(blender_runtime, "bpy", fake)
    monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
    policy = PathPolicy(enabled=True, allowed_paths=[str(tmp_path)], project_root=tmp_path)
    session = blender_runtime.BlenderRuntimeSession(path_policy=policy)
    yield SimpleNamespace(fake=fake, session=session, out=tmp_path, obj=fake.objects.get)


def _metadata(session: Any) -> None:
    session.apply_simready_metadata(
        "robot", {"semantic": {"semantic_class": "robot", "semantic_qcode": "Q11012"}}
    )
    session.apply_simready_metadata(
        "base_link",
        {
            "semantic": {"semantic_class": "robot_base", "additional_labels": {"sku": "RA-100"}},
            "physics": {
                "mass_kg": 12.5,
                "collider_type": "convexHull",
                "static_friction": 0.6,
                "dynamic_friction": 0.5,
                "restitution": 0.1,
                "density": 7800.0,
                "is_rigid_body": True,
            },
        },
    )
    session.apply_simready_metadata(
        "arm_link", {"physics": {"mass_kg": 2.0, "collider_type": "convexDecomposition"}}
    )


def _physics_parse(stage: Any) -> Dict[str, List[str]]:
    """UsdPhysics' own parse of the stage: prim paths per object type."""
    loader = getattr(UsdPhysics, "LoadUsdPhysicsFromRange", None)
    if loader is None:  # pragma: no cover - pxr without the parser
        pytest.skip("this pxr has no UsdPhysics parser")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        parsed = loader(stage, ["/"])
    return {str(kind): sorted(str(p) for p in paths) for kind, (paths, _) in parsed.items()}


# ---------------------------------------------------------------------------
# B1: physics and semantics a simulator reads
# ---------------------------------------------------------------------------


def test_export_authors_usd_physics_and_semantics(scene: Any) -> None:
    _metadata(scene.session)
    path = scene.out / "robot.usda"

    result = scene.session.export_simready_usd(str(path), object_names=["robot"])

    stage = Usd.Stage.Open(str(path))
    base = stage.GetPrimAtPath("/robot/base_link")
    cube = stage.GetPrimAtPath("/robot/base_link/Cube")
    cylinder = stage.GetPrimAtPath("/robot/base_link/arm_link/Cylinder")
    assert base.HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdPhysics.MassAPI(base).GetMassAttr().Get() == pytest.approx(12.5)
    assert UsdPhysics.MassAPI(base).GetDensityAttr().Get() == pytest.approx(7800.0)
    assert cube.HasAPI(UsdPhysics.CollisionAPI)
    assert UsdPhysics.MeshCollisionAPI(cube).GetApproximationAttr().Get() == "convexHull"
    assert (
        UsdPhysics.MeshCollisionAPI(cylinder).GetApproximationAttr().Get()
        == "convexDecomposition"
    )
    physics_material = UsdShade.MaterialBindingAPI(cube).ComputeBoundMaterial("physics")[0]
    material_api = UsdPhysics.MaterialAPI(physics_material.GetPrim())
    assert material_api.GetStaticFrictionAttr().Get() == pytest.approx(0.6)
    assert material_api.GetDynamicFrictionAttr().Get() == pytest.approx(0.5)
    assert material_api.GetRestitutionAttr().Get() == pytest.approx(0.1)
    # The arm sets no friction of its own: its collider uses its parent's material.
    assert (
        UsdShade.MaterialBindingAPI(cylinder).ComputeBoundMaterial("physics")[0].GetPath()
        == physics_material.GetPath()
    )
    labels = stage.GetPrimAtPath("/robot").GetAttribute("semantics:labels:class").Get()
    assert list(labels) == ["robot"]
    assert list(base.GetAttribute("semantics:labels:sku").Get()) == ["RA-100"]
    # SimReady's taxonomy for a Wikidata Q-code.
    robot = stage.GetPrimAtPath("/robot")
    assert "SemanticsLabelsAPI:wikidata_qcode" in robot.GetAppliedSchemas()
    assert list(robot.GetAttribute("semantics:labels:wikidata_qcode").Get()) == ["Q11012"]

    parsed = _physics_parse(stage)
    assert parsed["RigidBody"] == ["/robot/base_link"]
    assert parsed["MeshShape"] == [
        "/robot/base_link/Cube",
        "/robot/base_link/arm_link/Cylinder",
    ]
    assert parsed["RigidBodyMaterial"] == [str(physics_material.GetPath())]
    assert result["physics"]["rigid_body_count"] == 1
    assert result["physics"]["collider_count"] == 2
    assert result["semantics"]["labels_schema"] == "UsdSemantics.LabelsAPI"


def test_export_keeps_render_materials_and_their_bindings(scene: Any) -> None:
    path = scene.out / "robot.usda"

    scene.session.export_simready_usd(str(path), object_names=["robot"])

    stage = Usd.Stage.Open(str(path))
    cube = stage.GetPrimAtPath("/robot/base_link/Cube")
    bound = UsdShade.MaterialBindingAPI(cube).ComputeBoundMaterial()[0]
    assert bound.GetPath() == Sdf.Path("/robot/Looks/opaque_metal_steel")
    source = bound.GetSurfaceOutput().GetConnectedSources()[0][0].source
    assert source.GetPrim().GetPath() == Sdf.Path(
        "/robot/Looks/opaque_metal_steel/Principled_BSDF"
    )


# ---------------------------------------------------------------------------
# B7: the scene's unit scale
# ---------------------------------------------------------------------------


def test_export_declares_the_scene_unit_scale_and_converts_density(scene: Any) -> None:
    _metadata(scene.session)
    # Blender stores the scale in single precision.
    scene.fake.context.scene.unit_settings.scale_length = 0.009999999776482582
    path = scene.out / "robot_cm.usda"

    result = scene.session.export_simready_usd(str(path), object_names=["robot"])

    stage = Usd.Stage.Open(str(path))
    assert UsdGeom.GetStageMetersPerUnit(stage) == pytest.approx(0.01)
    assert result["meters_per_unit"] == 0.01
    density = UsdPhysics.MassAPI(stage.GetPrimAtPath("/robot/base_link")).GetDensityAttr()
    assert density.Get() == pytest.approx(7800.0 * 0.01**3)
    assert UsdPhysics.GetStageKilogramsPerUnit(stage) == 1.0
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z


def test_validation_warns_of_a_size_implausible_in_the_declared_units(scene: Any) -> None:
    part = FakeObject("mm_part", materials=["m"], dimensions=(500.0, 500.0, 500.0))
    part.parent = scene.obj("robot")
    part["simready_semantic_class"] = "part"
    scene.fake.objects.append(part)
    scene.fake.layer_objects.append(part)

    in_metres = scene.session.validate_simready_compliance(["mm_part"])
    scene.fake.context.scene.unit_settings.scale_length = 0.001
    in_millimetres = scene.session.validate_simready_compliance(["mm_part"])

    # Large assets exist (a warehouse), so an implausible size is a warning.
    [issue] = [i for i in in_metres["issues"] if i["object_name"] == "mm_part" and i["check"] == "scale"]
    assert issue["severity"] == "warning"
    assert "500 m" in issue["message"]
    assert in_metres["compliant"] is True
    assert not [
        i for i in in_millimetres["issues"]
        if i["object_name"] == "mm_part" and i["check"] == "scale"
    ]
    # A Unit Scale other than 1 is itself a warning [UN.007].
    assert any("Unit Scale is 0.001" in i["message"] for i in in_millimetres["issues"])
    assert in_millimetres["meters_per_unit"] == 0.001


# ---------------------------------------------------------------------------
# B8 and M2: hidden listed objects, and a root's descendants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hide", ["hide_set", "hide_viewport", "hide_render"])
def test_export_writes_a_hidden_listed_object_and_keeps_it_hidden(
    scene: Any, hide: str
) -> None:
    arm = scene.obj("arm_link")
    if hide == "hide_set":
        arm.hide_set(True)
    else:
        setattr(arm, hide, True)
    path = scene.out / "hidden.usda"

    result = scene.session.export_simready_usd(
        str(path), object_names=["robot", "base_link", "arm_link"]
    )

    stage = Usd.Stage.Open(str(path))
    assert stage.GetPrimAtPath("/robot/base_link/arm_link/Cylinder").IsA(UsdGeom.Mesh)
    assert result["object_count"] == 3
    assert {p["object_name"]: p["prim_path"] for p in result["prims"]}["arm_link"] == (
        "/robot/base_link/arm_link"
    )
    flags = (arm.hide_get(), arm.hide_viewport, arm.hide_render)
    assert flags == (hide == "hide_set", hide == "hide_viewport", hide == "hide_render")


def test_export_by_root_name_includes_its_descendants(scene: Any) -> None:
    path = scene.out / "root.usda"

    result = scene.session.export_simready_usd(str(path), object_names=["robot"])

    stage = Usd.Stage.Open(str(path))
    assert stage.GetPrimAtPath("/robot/base_link/Cube").IsA(UsdGeom.Mesh)
    assert stage.GetPrimAtPath("/robot/base_link/arm_link/Cylinder").IsA(UsdGeom.Mesh)
    assert result["object_count"] == 3


def test_export_without_children_writes_only_the_listed_objects(scene: Any) -> None:
    path = scene.out / "alone.usda"

    result = scene.session.export_simready_usd(
        str(path), object_names=["robot"], include_children=False
    )

    stage = Usd.Stage.Open(str(path))
    assert [p["object_name"] for p in result["prims"]] == ["robot"]
    assert not [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]


def test_an_unlisted_descendant_hidden_in_renders_is_skipped_and_reported(scene: Any) -> None:
    scene.obj("arm_link").hide_render = True
    path = scene.out / "skip.usda"

    result = scene.session.export_simready_usd(str(path), object_names=["robot"])

    assert result["skipped_objects"] == [
        {
            "object_name": "arm_link",
            "reason": "hidden in renders; list it in object_names to export it",
        }
    ]
    assert not Usd.Stage.Open(str(path)).GetPrimAtPath("/robot/base_link/arm_link")


# ---------------------------------------------------------------------------
# M3: the asset root, without the scene's lighting
# ---------------------------------------------------------------------------


def test_the_asset_root_is_the_default_prim_and_holds_the_materials(scene: Any) -> None:
    path = scene.out / "robot.usda"

    result = scene.session.export_simready_usd(str(path))

    stage = Usd.Stage.Open(str(path))
    assert stage.GetDefaultPrim().GetPath() == Sdf.Path("/robot")
    assert Usd.ModelAPI(stage.GetDefaultPrim()).GetKind() == "component"
    assert [str(p.GetPath()) for p in stage.GetPseudoRoot().GetChildren()] == ["/robot"]
    assert stage.GetPrimAtPath("/robot/Looks/opaque_plastic_abs").IsA(UsdShade.Material)
    types = {p.GetTypeName() for p in stage.Traverse()}
    assert not types & {"DomeLight", "SphereLight", "Camera"}
    assert result["default_prim"] == "/robot"
    assert {s["object_name"] for s in result["skipped_objects"]} == {"Light", "Camera"}


def test_lights_and_cameras_are_exported_when_asked_under_a_named_root(scene: Any) -> None:
    path = scene.out / "staged scene.usda"

    # Blender's default "Light" and "Camera" names fail validation; the layout is the point.
    result = scene.session.export_simready_usd(
        str(path), include_lights_and_cameras=True, validate_before_export=False
    )

    stage = Usd.Stage.Open(str(path))
    assert stage.GetDefaultPrim().GetPath() == Sdf.Path("/staged_scene")
    assert stage.GetPrimAtPath("/staged_scene/env_light").GetTypeName() == "DomeLight"
    assert stage.GetPrimAtPath("/staged_scene/Light/Light").GetTypeName() == "SphereLight"
    assert stage.GetPrimAtPath("/staged_scene/robot/base_link/Cube")
    assert any("share the top level" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# M1: validation errors stop the export unless forced
# ---------------------------------------------------------------------------


def test_a_failed_validation_writes_nothing_unless_forced(scene: Any) -> None:
    scene.obj("arm_link").data.materials = []
    path = scene.out / "bad.usda"

    refused = scene.session.export_simready_usd(str(path), object_names=["robot"])

    assert refused["success"] is False
    assert refused["error_type"] == "SimReadyValidationFailed"
    assert "arm_link" in refused["error"] and "force=true" in refused["error"]
    assert refused["details"]["file_written"] is False
    assert not path.exists()
    assert not scene.fake.export_calls

    forced = scene.session.export_simready_usd(str(path), object_names=["robot"], force=True)

    assert path.exists()
    assert forced["validation_passed"] is False
    assert forced["forced"] is True


def test_skip_checks_leaves_one_check_out_without_force(scene: Any) -> None:
    scene.obj("arm_link").data.materials = []
    path = scene.out / "skipped.usda"

    result = scene.session.export_simready_usd(
        str(path), object_names=["robot"], skip_checks=["materials"]
    )

    assert path.exists()
    assert result["validation_passed"] is True and result["forced"] is False
    with pytest.raises(ValueError, match="unknown check 'material' .did you mean 'materials'"):
        scene.session.export_simready_usd(str(path), skip_checks=["material"])


def test_a_class_on_an_unexported_ancestor_does_not_count(scene: Any) -> None:
    """Blender writes the unexported robot as a bare Xform: its label is not in the file."""
    path = scene.out / "arm.usda"

    refused = scene.session.export_simready_usd(
        str(path), object_names=["arm_link"], include_children=False
    )

    assert refused["success"] is False
    issue = refused["details"]["issues"][0]
    assert (issue["object_name"], issue["check"]) == ("arm_link", "metadata")
    assert "No semantic label" in issue["message"]


def test_the_root_check_applies_to_the_prim_that_becomes_the_default_prim(scene: Any) -> None:
    # The robot root becomes the default prim even when only its child is exported.
    scene.obj("robot").matrix_world = FakeMatrix.Translation((0.0, 0.0, 2.0))
    scene.obj("arm_link")["simready_semantic_class"] = "arm"

    refused = scene.session.export_simready_usd(
        str(scene.out / "lifted.usda"), object_names=["arm_link"], include_children=False
    )

    assert refused["details"]["issues"][0]["object_name"] == "robot"
    assert "becomes the default prim" in refused["details"]["issues"][0]["message"]


def test_validation_issues_in_a_refusal_are_capped(scene: Any) -> None:
    for index in range(80):
        part = FakeObject(f"Part.{index:03d}", materials=["m"])
        scene.fake.objects.append(part)
        scene.fake.layer_objects.append(part)

    refused = scene.session.export_simready_usd(str(scene.out / "many.usda"))

    assert refused["details"]["issue_count"] >= 80
    assert len(refused["details"]["issues"]) == 50


# ---------------------------------------------------------------------------
# Minor findings
# ---------------------------------------------------------------------------


def test_export_restores_the_selection_and_active_object(scene: Any) -> None:
    light = scene.obj("Light")
    light.select_set(True)
    scene.fake.layer_objects.active = light

    scene.session.export_simready_usd(str(scene.out / "sel.usda"), object_names=["robot"])

    assert [o.name for o in scene.fake.layer_objects if o.select_get()] == ["Light"]
    assert scene.fake.layer_objects.active is light


def test_an_empty_object_list_is_refused(scene: Any) -> None:
    with pytest.raises(ValueError, match="object_names is empty"):
        scene.session.export_simready_usd(str(scene.out / "empty.usda"), object_names=[])


def test_an_extension_that_is_not_usd_is_refused(scene: Any) -> None:
    with pytest.raises(ValueError, match="Unsupported SimReady export extension"):
        scene.session.export_simready_usd(str(scene.out / "asset.txt"))


def test_edit_mode_is_left_for_the_export_and_entered_again(scene: Any) -> None:
    arm = scene.obj("arm_link")
    arm.select_set(True)
    scene.fake.layer_objects.active = arm
    scene.fake.context.mode = "EDIT_MESH"
    arm.mode = "EDIT"
    scene.fake.context.objects_in_mode = [arm]

    scene.session.export_simready_usd(str(scene.out / "edit.usda"), object_names=["robot"])

    assert scene.fake.mode_calls == ["OBJECT", "EDIT"]
    assert scene.fake.context.mode == "EDIT_MESH"
    assert scene.fake.export_calls[0]["filepath"]  # exported while in Object Mode
    assert [o.name for o in scene.fake.layer_objects if o.select_get()] == ["arm_link"]


def test_without_embedded_metadata_no_user_properties_are_written(scene: Any) -> None:
    _metadata(scene.session)
    path = scene.out / "plain.usda"

    scene.session.export_simready_usd(str(path), object_names=["robot"], embed_metadata=False)

    stage = Usd.Stage.Open(str(path))
    assert not [
        a.GetName()
        for prim in stage.Traverse()
        for a in prim.GetAttributes()
        if a.GetName().startswith("userProperties:")
    ]
    # The schemas do not depend on the custom properties being written.
    assert stage.GetPrimAtPath("/robot/base_link").HasAPI(UsdPhysics.RigidBodyAPI)


def test_the_exporter_options_follow_the_request(scene: Any) -> None:
    scene.session.export_simready_usd(str(scene.out / "opts.usda"), object_names=["robot"])

    [call] = scene.fake.export_calls
    assert call["selected_objects_only"] is True
    assert call["export_custom_properties"] is True
    assert call["author_blender_name"] is True
    assert (call["export_lights"], call["export_cameras"], call["convert_world_material"]) == (
        False,
        False,
        False,
    )
    assert call["convert_scene_units"] == "METERS"
    # Blender wrote next to the destination, then the result replaced it: no leftovers.
    assert sorted(p.name for p in scene.out.iterdir()) == ["opts.usda"]


def test_options_this_blender_lacks_are_not_passed(scene: Any) -> None:
    scene.fake.supported = {"filepath", "selected_objects_only", "export_custom_properties"}

    scene.session.export_simready_usd(str(scene.out / "old.usda"), object_names=["robot"])

    assert set(scene.fake.export_calls[0]) == scene.fake.supported


def test_usdz_is_packaged_with_the_authored_layer(scene: Any) -> None:
    _metadata(scene.session)
    path = scene.out / "robot.usdz"

    scene.session.export_simready_usd(str(path), object_names=["robot"])

    stage = Usd.Stage.Open(str(path))
    assert stage.GetDefaultPrim().GetPath() == Sdf.Path("/robot")
    assert stage.GetPrimAtPath("/robot/base_link").HasAPI(UsdPhysics.RigidBodyAPI)
    assert sorted(p.name for p in scene.out.iterdir()) == ["robot.usdz"]


def test_an_object_outside_the_view_layer_is_refused_before_anything_changes(
    scene: Any,
) -> None:
    other = FakeObject("other_scene_part", materials=["m"])
    scene.fake.objects.append(other)  # in bpy.data, not in this view layer

    with pytest.raises(ValueError, match="not in the active view layer"):
        scene.session.export_simready_usd(
            str(scene.out / "x.usda"), object_names=["robot", "other_scene_part"]
        )
    assert not scene.fake.export_calls


def test_the_addon_bundles_the_usd_authoring_module(tmp_path: Path) -> None:
    """blender_runtime imports adapters/simready_usd.py, so the add-on ZIP must carry it."""
    import subprocess
    import sys
    import zipfile

    from simul.adapters.blender_connection import BlenderAttachments

    addon = BlenderAttachments.build_addon(tmp_path / "addon.zip")
    with zipfile.ZipFile(addon) as archive:
        assert "simul_blender_bridge/adapters/simready_usd.py" in archive.namelist()
        archive.extractall(tmp_path / "addons")
    probe = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from simul_blender_bridge.adapters import blender_runtime; "
        "print(blender_runtime.author_simready_layer.__module__)"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(tmp_path / "addons")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == "simul_blender_bridge.adapters.simready_usd"


def test_a_removed_object_left_in_the_view_layer_is_skipped(scene: Any) -> None:
    """Right after bpy.data.objects.remove, Blender 5.0.1 yields None from view_layer.objects."""
    scene.fake.layer_objects.append(None)

    result = scene.session.export_simready_usd(str(scene.out / "stale.usda"), object_names=["robot"])

    assert result["object_count"] == 3


def test_the_file_is_written_exactly_at_file_path_in_a_new_folder(scene: Any) -> None:
    path = scene.out / "assets" / "robot" / "robot.usda"

    result = scene.session.export_simready_usd(str(path), object_names=["robot"])

    assert result["file_path"] == str(path)
    assert result["files_written"] == [str(path)]
    assert sorted(p.name for p in path.parent.iterdir()) == ["robot.usda"]


def test_a_failed_export_removes_the_folder_it_created(scene: Any) -> None:
    folder = scene.out / "new_folder"
    scene.fake.supported.discard("selected_objects_only")

    with pytest.raises(ValueError, match="selected_only is not supported"):
        scene.session.export_simready_usd(str(folder / "robot.usda"), object_names=["robot"])

    assert not folder.exists()


# ---------------------------------------------------------------------------
# Joints: Blender rigid body constraints become UsdPhysics joints
# ---------------------------------------------------------------------------

_LINK = {"is_rigid_body": True, "mass_kg": 1.0, "collider_type": "convexHull"}


def _articulated(scene: Any) -> None:
    """base_link at (0, 0, 0.1) and arm_link (nested under it) at (0, 0, 0.35), rigid bodies."""
    scene.obj("base_link").matrix_world = FakeMatrix.Translation((0.0, 0.0, 0.1))
    scene.obj("arm_link").matrix_world = FakeMatrix.Translation((0.0, 0.0, 0.35))
    for name in ("base_link", "arm_link"):
        scene.session.apply_simready_metadata(name, {"physics": dict(_LINK)})


def _hinge_frame() -> FakeMatrix:
    """At (0, 0, 0.2), its local Z along world +Y (add_rigid_body_constraint's axis=[0, 1, 0])."""
    return FakeMatrix([(1, 0, 0, 0), (0, 0, 1, 0), (0, -1, 0, 0.2), (0, 0, 0, 1)])


def _carrier(scene: Any, kind: str, frame: Optional[FakeMatrix] = None, **settings: Any) -> FakeObject:
    """A constraint empty between base_link and arm_link, top level, as Blender adds one."""
    carrier = FakeObject("RBC_base_link_arm_link", "EMPTY")
    carrier.matrix_world = frame or _hinge_frame()
    carrier.rigid_body_constraint = SimpleNamespace(
        type=kind,
        object1=scene.obj("base_link"),
        object2=scene.obj("arm_link"),
        enabled=True,
        disable_collisions=True,
        use_breaking=False,
        **settings,
    )
    scene.fake.objects.append(carrier)
    scene.fake.layer_objects.append(carrier)
    return carrier


def _joints(stage: Any) -> Dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        parsed = UsdPhysics.LoadUsdPhysicsFromRange(stage, ["/"])
    return {
        str(path): desc
        for kind, (paths, descs) in parsed.items()
        if "Joint" in str(kind)
        for path, desc in zip(paths, descs)
    }


def test_a_hinge_between_nested_links_exports_as_a_revolute_joint(scene: Any) -> None:
    _articulated(scene)
    _carrier(scene, "HINGE", use_limit_ang_z=True, limit_ang_z_lower=0.0, limit_ang_z_upper=0.5)
    path = scene.out / "robot.usda"

    result = scene.session.export_simready_usd(str(path), object_names=["robot"])

    assert result["validation_passed"] is True, result["issues"]
    assert result["physics"]["joint_count"] == 1
    joint_path = "/robot/Joints/RBC_base_link_arm_link"
    assert result["physics"]["joints"] == [joint_path]
    stage = Usd.Stage.Open(str(path))
    joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(joint_path))
    assert joint.GetBody0Rel().GetTargets() == [Sdf.Path("/robot/base_link")]
    assert joint.GetBody1Rel().GetTargets() == [Sdf.Path("/robot/base_link/arm_link")]
    assert joint.GetAxisAttr().Get() == "Z"
    # Blender limits object1 relative to object2 on a hinge; UsdPhysics body1 to body0.
    assert (joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()) == pytest.approx(
        (-28.6479, 0.0), abs=1e-3
    )
    assert not stage.GetPrimAtPath("/robot/RBC_base_link_arm_link")
    desc = _joints(stage)[joint_path]
    cache = UsdGeom.XformCache()
    for body, position in ((desc.body0, desc.localPose0Position), (desc.body1, desc.localPose1Position)):
        world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body))
        assert list(world.Transform(Gf.Vec3d(position))) == pytest.approx([0.0, 0.0, 0.2], abs=1e-6)


def test_a_constraint_empty_is_not_a_prim_and_forces_no_wrapper_root(scene: Any) -> None:
    _articulated(scene)
    _carrier(scene, "FIXED")
    path = scene.out / "scene.usda"

    result = scene.session.export_simready_usd(str(path))

    assert result["default_prim"] == "/robot"
    assert {"object_name": "RBC_base_link_arm_link",
            "reason": "a rigid body constraint: exported as a UsdPhysics joint between its two "
                      "objects, not as a prim"} in result["skipped_objects"]
    stage = Usd.Stage.Open(str(path))
    assert stage.GetPrimAtPath("/robot/Joints/RBC_base_link_arm_link").GetTypeName() == "PhysicsFixedJoint"


_LOCKED = [1.0, -1.0]


@pytest.mark.parametrize(
    ("kind", "settings", "type_name", "expected"),
    [
        ("SLIDER", {"use_limit_lin_x": True, "limit_lin_x_lower": 0.0, "limit_lin_x_upper": 0.3},
         "PhysicsPrismaticJoint", {"axis": "X", "limit": [0.0, 0.3]}),
        ("HINGE", {}, "PhysicsRevoluteJoint", {"axis": "Z", "limit": None}),
        ("POINT", {}, "PhysicsSphericalJoint", {}),
        ("PISTON",
         {"use_limit_lin_x": True, "limit_lin_x_lower": -0.1, "limit_lin_x_upper": 0.2,
          "use_limit_ang_x": True, "limit_ang_x_lower": -0.5, "limit_ang_x_upper": 0.25},
         "PhysicsJoint",
         {"limits": {"transX": [-0.1, 0.2], "rotX": [-28.6479, 14.3239], "transY": _LOCKED,
                     "transZ": _LOCKED, "rotY": _LOCKED, "rotZ": _LOCKED}}),
        ("GENERIC",
         {"use_limit_lin_x": True, "limit_lin_x_lower": 0.0, "limit_lin_x_upper": 0.0,
          "use_limit_lin_z": True, "limit_lin_z_lower": -0.05, "limit_lin_z_upper": 0.1,
          "use_limit_ang_y": True, "limit_ang_y_lower": -0.2, "limit_ang_y_upper": 0.4},
         "PhysicsJoint",
         {"limits": {"transX": _LOCKED, "transZ": [-0.05, 0.1], "rotY": [-22.9183, 11.4592]}}),
    ],
)
def test_each_constraint_type_maps_to_its_joint_and_limits(
    scene: Any, kind: str, settings: Dict[str, Any], type_name: str, expected: Dict[str, Any]
) -> None:
    _articulated(scene)
    _carrier(scene, kind, **settings)
    path = scene.out / f"{kind.lower()}.usda"

    result = scene.session.export_simready_usd(str(path), object_names=["robot"])

    assert result["validation_passed"] is True, result["issues"]
    stage = Usd.Stage.Open(str(path))
    prim = stage.GetPrimAtPath("/robot/Joints/RBC_base_link_arm_link")
    assert prim.GetTypeName() == type_name
    if "axis" in expected:
        assert prim.GetAttribute("physics:axis").Get() == expected["axis"]
        limit = expected["limit"]
        lower = prim.GetAttribute("physics:lowerLimit")
        if limit is None:
            assert not lower.HasAuthoredValue()  # free: Blender leaves an unlimited axis free
        else:
            assert [lower.Get(), prim.GetAttribute("physics:upperLimit").Get()] == pytest.approx(limit)
    limits = {
        schema.split(":", 1)[1]: [
            prim.GetAttribute(f"limit:{schema.split(':', 1)[1]}:physics:low").Get(),
            prim.GetAttribute(f"limit:{schema.split(':', 1)[1]}:physics:high").Get(),
        ]
        for schema in prim.GetAppliedSchemas()
        if schema.startswith("PhysicsLimitAPI:")
    }
    wanted = expected.get("limits", {})
    assert sorted(limits) == sorted(wanted)
    for dof, bounds in wanted.items():
        assert limits[dof] == pytest.approx(bounds, abs=1e-3), dof


def test_a_motor_refuses_the_export_and_a_forced_export_names_the_lost_joint(scene: Any) -> None:
    _articulated(scene)
    _carrier(scene, "MOTOR", use_motor_ang=True)
    path = scene.out / "motor.usda"

    refused = scene.session.export_simready_usd(str(path), object_names=["robot"])

    assert refused["success"] is False and not path.exists()
    errors = [i["message"] for i in refused["details"]["issues"] if i["severity"] == "error"]
    assert any("MOTOR constraint RBC_base_link_arm_link" in m for m in errors)
    # Without a joint, the nested arm is an error too.
    assert any("nested in the rigid body base_link" in m for m in errors)

    forced = scene.session.export_simready_usd(str(path), object_names=["robot"], force=True)

    assert forced["physics"]["joint_count"] == 0
    assert any("MOTOR constraint RBC_base_link_arm_link" in w for w in forced["warnings"])
    assert not Usd.Stage.Open(str(path)).GetPrimAtPath("/robot/Joints")
