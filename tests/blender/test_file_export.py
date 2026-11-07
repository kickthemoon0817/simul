"""export_file, import_file and save_blend_file against stand-ins for Blender's exporters.

Each stand-in operator defines the properties Blender 5.0.1's operator does,
refuses any other keyword as bpy does, and writes files the way that exporter
does: the glTF exporter swaps a path's extension to match ``export_format``
(GLB by default) and creates missing folders, GLTF_SEPARATE writes a ``.bin``
beside the ``.gltf``, the OBJ exporter writes a ``.mtl`` and nothing at all
without geometry, the USD exporter writes ``textures/``, and the FBX, OBJ, STL
and PLY exporters fail on a missing folder. The audit of Blender 5.0.1 found
each of these through the real exporters (B1-03, B1-04, B1-15, B1-19, B1-21).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Optional

import pytest

from simul.adapters import blender_runtime
from simul.config import SecurityConfig, Settings

AXES = ("X", "Y", "Z", "-X", "-Y", "-Z")
CPP_AXES = ("X", "Y", "Z", "NEGATIVE_X", "NEGATIVE_Y", "NEGATIVE_Z")


class FakeProperty:
    """An operator property: its identifier and, for an enum, its items."""

    def __init__(self, items: Optional[tuple] = None) -> None:
        self.enum_items = [SimpleNamespace(identifier=i) for i in items or ()]


class FakeProperties(dict):
    """``get_rna_type().properties``: keyed by identifier, like bpy_prop_collection."""


class FakeOperator:
    """A ``bpy.ops`` operator that accepts only the properties it defines."""

    def __init__(
        self,
        name: str,
        properties: Dict[str, Optional[tuple]],
        run: Callable[[Dict[str, Any]], None],
        calls: List[Any],
    ) -> None:
        self.name = name
        self.properties = FakeProperties(
            {key: FakeProperty(items) for key, items in properties.items()}
        )
        self.run = run
        self.calls = calls

    def get_rna_type(self) -> SimpleNamespace:
        return SimpleNamespace(properties=self.properties)

    def __call__(self, **kwargs: Any) -> set:
        for key in kwargs:
            if key not in self.properties:
                raise TypeError(
                    f'Converting py args to operator properties: keyword "{key}" unrecognized'
                )
        self.calls.append((self.name, dict(kwargs)))
        self.run(kwargs)
        return {"FINISHED"}


class FakeObject:
    """An object: selection, visibility, mode and transform."""

    def __init__(
        self,
        name: str,
        kind: str = "MESH",
        *,
        hidden: bool = False,
        in_view_layer: bool = True,
        parent: Optional["FakeObject"] = None,
        animated: bool = False,
    ) -> None:
        self.name = name
        self.type = kind
        self.hidden = hidden
        self.in_view_layer = in_view_layer
        self.parent = parent
        self.mode = "OBJECT"
        self.selected = False
        self.location = [1.0, 2.0, 3.0]
        self.scale = [1.0, 1.0, 1.0]
        self.animation_data = SimpleNamespace(action="Act") if animated else None

    def select_set(self, value: bool) -> None:
        if not self.in_view_layer:
            raise RuntimeError(
                f"Object '{self.name}' can't be selected because it is not in View Layer 'ViewLayer'!"
            )
        self.selected = bool(value) and not self.hidden

    def select_get(self) -> bool:
        return self.selected


class FakeCollection(dict):
    """``bpy.data.objects`` and friends: iterates the values, ``in`` tests names."""

    def __iter__(self) -> Iterator[Any]:  # type: ignore[override]
        return iter(list(self.values()))


class FakeViewLayerObjects:
    """``view_layer.objects``: the objects in the view layer, plus the active one."""

    def __init__(self, data: FakeCollection) -> None:
        self._data = data
        self.active: Optional[FakeObject] = None

    def __iter__(self) -> Iterator[FakeObject]:
        return iter([o for o in self._data.values() if o.in_view_layer])

    def __contains__(self, name: object) -> bool:
        obj = self._data.get(name)
        return obj is not None and obj.in_view_layer


class FakeScene:
    """A scene whose frame range clamps each end to the other, as Blender's does."""

    def __init__(self) -> None:
        self.name = "Scene"
        self._start, self._end = 1, 250
        self.use_preview_range = True
        self.history: List[tuple] = []

    @property
    def frame_start(self) -> int:
        return self._start

    @frame_start.setter
    def frame_start(self, value: int) -> None:
        self._start = value
        self._end = max(self._end, value)
        self.history.append(("start", value))

    @property
    def frame_end(self) -> int:
        return self._end

    @frame_end.setter
    def frame_end(self, value: int) -> None:
        self._end = value
        self._start = min(self._start, value)
        self.history.append(("end", value))


class FakeBlender:
    """A bpy stand-in whose exporters write files as Blender 5.0.1's do."""

    def __init__(self, version: tuple = (5, 0, 1)) -> None:
        self.calls: List[Any] = []
        self.objects = FakeCollection()
        self.collections = FakeCollection()
        self.scene = FakeScene()
        self.view_layer = SimpleNamespace(
            name="ViewLayer", objects=FakeViewLayerObjects(self.objects)
        )
        self.gltf_preferences = SimpleNamespace(allow_embedded_format=False)
        self.frames_seen: List[tuple] = []
        self.app = SimpleNamespace(version=version, background=True)
        self.data = SimpleNamespace(
            objects=self.objects,
            collections=self.collections,
            filepath="",
            is_dirty=False,
        )
        self.context = SimpleNamespace(
            scene=self.scene,
            view_layer=self.view_layer,
            preferences=SimpleNamespace(
                addons={
                    "io_scene_gltf2": SimpleNamespace(preferences=self.gltf_preferences)
                }
            ),
        )
        self.path = SimpleNamespace(abspath=lambda p: "/blend-dir/" + p[2:])
        self.ops = SimpleNamespace(
            export_scene=SimpleNamespace(
                gltf=self._op(
                    "export_scene.gltf", self._gltf_properties(), self._write_gltf
                ),
                fbx=self._op(
                    "export_scene.fbx", self._fbx_properties(), self._write_plain
                ),
            ),
            import_scene=SimpleNamespace(
                gltf=self._op("import_scene.gltf", {"filepath": None}, self._import),
                fbx=self._op(
                    "import_scene.fbx",
                    {"filepath": None, "global_scale": None, "use_image_search": None},
                    self._import,
                ),
            ),
            wm=SimpleNamespace(
                obj_export=self._op(
                    "wm.obj_export", self._cpp_properties(obj=True), self._write_obj
                ),
                stl_export=self._op(
                    "wm.stl_export", self._cpp_properties(), self._write_plain
                ),
                ply_export=self._op(
                    "wm.ply_export", self._cpp_properties(), self._write_plain
                ),
                usd_export=self._op(
                    "wm.usd_export", self._usd_properties(), self._write_usd
                ),
                obj_import=self._op(
                    "wm.obj_import",
                    {"filepath": None, "global_scale": None},
                    self._import,
                ),
                usd_import=self._op(
                    "wm.usd_import", {"filepath": None, "scale": None}, self._import
                ),
                stl_import=self._op(
                    "wm.stl_import",
                    {"filepath": None, "global_scale": None},
                    self._import,
                ),
                ply_import=self._op(
                    "wm.ply_import",
                    {"filepath": None, "global_scale": None},
                    self._import,
                ),
                save_as_mainfile=self._record("wm.save_as_mainfile"),
                save_mainfile=self._record("wm.save_mainfile"),
            ),
            object=SimpleNamespace(mode_set=self._mode_set),
        )
        self.imported: List[str] = []

    # -- building ----------------------------------------------------------

    def _op(
        self, name: str, properties: Dict[str, Optional[tuple]], run: Any
    ) -> FakeOperator:
        return FakeOperator(name, properties, run, self.calls)

    def _record(self, name: str) -> Callable[..., set]:
        def run(**kwargs: Any) -> set:
            self.calls.append((name, kwargs))
            return {"FINISHED"}

        return run

    @staticmethod
    def _gltf_properties() -> Dict[str, Optional[tuple]]:
        return {
            "filepath": None,
            "check_existing": None,
            "export_format": ("GLB", "GLTF_SEPARATE", "GLTF_EMBEDDED"),
            "use_selection": None,
            "export_apply": None,
            "export_animations": None,
            "export_frame_range": None,
        }

    @staticmethod
    def _fbx_properties() -> Dict[str, Optional[tuple]]:
        return {
            "filepath": None,
            "use_selection": None,
            "use_mesh_modifiers": None,
            "bake_anim": None,
            "global_scale": None,
            "axis_forward": AXES,
            "axis_up": AXES,
        }

    @staticmethod
    def _cpp_properties(obj: bool = False) -> Dict[str, Optional[tuple]]:
        properties: Dict[str, Optional[tuple]] = {
            "filepath": None,
            "export_selected_objects": None,
            "apply_modifiers": None,
            "global_scale": None,
            "forward_axis": CPP_AXES,
            "up_axis": CPP_AXES,
        }
        if obj:
            properties["export_animation"] = None
        return properties

    @staticmethod
    def _usd_properties() -> Dict[str, Optional[tuple]]:
        return {
            "filepath": None,
            "selected_objects_only": None,
            "export_animation": None,
            "export_custom_properties": None,
            "convert_orientation": None,
            "export_global_up_selection": CPP_AXES,
            "export_global_forward_selection": CPP_AXES,
            "convert_scene_units": (
                "METERS",
                "KILOMETERS",
                "CENTIMETERS",
                "MILLIMETERS",
                "INCHES",
                "FEET",
                "YARDS",
                "CUSTOM",
            ),
            "meters_per_unit": None,
            "root_prim_path": None,
        }

    # -- scene ---------------------------------------------------------------

    def add(self, *objects: FakeObject) -> None:
        for obj in objects:
            self.objects[obj.name] = obj

    def exported(self, kwargs: Dict[str, Any], selection_key: str) -> List[str]:
        self.frames_seen.append((self.scene.frame_start, self.scene.frame_end))
        objects = list(self.view_layer.objects)
        if kwargs.get(selection_key):
            objects = [o for o in objects if o.select_get()]
        return [o.name for o in objects]

    def _mode_set(self, mode: str) -> set:
        self.calls.append(("object.mode_set", {"mode": mode}))
        active = self.view_layer.objects.active
        if active is None:
            raise RuntimeError(
                "Operator bpy.ops.object.mode_set.poll() failed, context is incorrect"
            )
        active.mode = mode
        return {"FINISHED"}

    def _leave_edit_mode(self) -> None:
        # Blender's glTF, OBJ and USD exporters switch to Object Mode themselves.
        active = self.view_layer.objects.active
        if active is not None and active.mode != "OBJECT":
            active.mode = "OBJECT"

    # -- exporters -----------------------------------------------------------

    def _write_gltf(self, kwargs: Dict[str, Any]) -> None:
        export_format = kwargs.get("export_format", "GLB")
        if (
            export_format == "GLTF_EMBEDDED"
            and not self.gltf_preferences.allow_embedded_format
        ):
            raise TypeError(
                'Converting py args to operator properties: enum "GLTF_EMBEDDED" not found '
                "in ('GLB', 'GLTF_SEPARATE')"
            )
        self._leave_edit_mode()
        names = self.exported(kwargs, "use_selection")
        root, _ = os.path.splitext(kwargs["filepath"])
        path = Path(root + (".glb" if export_format == "GLB" else ".gltf"))
        path.parent.mkdir(parents=True, exist_ok=True)
        if export_format == "GLTF_SEPARATE":
            path.with_suffix(".bin").write_bytes(b"BIN")
            path.write_text(
                json.dumps(
                    {
                        "buffers": [{"uri": path.with_suffix(".bin").name}],
                        "nodes": names,
                    }
                )
            )
        else:
            path.write_text(json.dumps({"nodes": names}))

    def _write_plain(self, kwargs: Dict[str, Any]) -> None:
        path = Path(kwargs["filepath"])
        if not path.parent.is_dir():
            raise RuntimeError(f"Error: Cannot open file '{path}'")
        key = (
            "use_selection" if "use_selection" in kwargs else "export_selected_objects"
        )
        path.write_text(json.dumps(self.exported(kwargs, key)))

    def _write_obj(self, kwargs: Dict[str, Any]) -> None:
        path = Path(kwargs["filepath"])
        if not path.parent.is_dir():
            raise RuntimeError(f"Error: OBJ Export: Cannot open file '{path}'")
        self._leave_edit_mode()
        names = self.exported(kwargs, "export_selected_objects")
        meshes = [n for n in names if self.objects[n].type == "MESH"]
        if not meshes:
            return  # Blender writes no file without geometry.
        mtl = path.with_suffix(".mtl")
        mtl.write_text("newmtl M\n")
        path.write_text(f"mtllib {mtl.name}\n" + "".join(f"o {n}\n" for n in meshes))

    def _write_usd(self, kwargs: Dict[str, Any]) -> None:
        path = Path(kwargs["filepath"])
        if path.suffix not in (".usd", ".usda", ".usdc", ".usdz"):
            raise RuntimeError(
                f"Error: USD Export: unable to find suitable USD plugin to write {path}"
            )
        self._leave_edit_mode()
        names = self.exported(kwargs, "selected_objects_only")
        (path.parent / "textures").mkdir(parents=True, exist_ok=True)
        (path.parent / "textures" / "color.exr").write_bytes(b"EXR")
        path.write_text(json.dumps(names))

    def _import(self, kwargs: Dict[str, Any]) -> None:
        for obj in self.imported:
            self.objects[obj.name] = obj


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeBlender:
    blender = FakeBlender()
    monkeypatch.setattr(blender_runtime, "bpy", blender)
    monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
    return blender


@pytest.fixture
def session(fake: FakeBlender, tmp_path: Path) -> blender_runtime.BlenderRuntimeSession:
    security = SecurityConfig(allowed_paths=[str(tmp_path)])
    return blender_runtime.BlenderRuntimeSession(settings=Settings(security=security))


def _scene(fake: FakeBlender) -> Dict[str, FakeObject]:
    objects = {
        "Cube": FakeObject("Cube"),
        "Sphere": FakeObject("Sphere"),
        "Light": FakeObject("Light", "LIGHT"),
    }
    fake.add(*objects.values())
    return objects


def _call(fake: FakeBlender, name: str) -> Dict[str, Any]:
    return next(kwargs for op, kwargs in reversed(fake.calls) if op == name)


# -- glTF layout and extensions (B1-03) --------------------------------------


def test_a_gltf_path_writes_that_gltf_and_leaves_a_glb_beside_it_alone(
    fake: FakeBlender, session: Any, tmp_path: Path
) -> None:
    """Exporting model.gltf used to write GLB, which Blender saves as model.glb."""
    _scene(fake)
    sentinel = tmp_path / "model.glb"
    sentinel.write_bytes(b"USER FINAL ASSET")

    result = session.export_file(str(tmp_path / "model.gltf"), "GLTF")

    assert sentinel.read_bytes() == b"USER FINAL ASSET"
    assert result["file_path"] == str(tmp_path / "model.gltf")
    assert result["gltf_format"] == "GLTF_SEPARATE"
    assert result["files_written"] == [
        str(tmp_path / "model.gltf"),
        str(tmp_path / "model.bin"),
    ]
    assert _call(fake, "export_scene.gltf")["export_format"] == "GLTF_SEPARATE"


def test_a_glb_path_writes_a_glb(
    fake: FakeBlender, session: Any, tmp_path: Path
) -> None:
    _scene(fake)

    result = session.export_file(str(tmp_path / "out" / "model.glb"), "gltf")

    assert result["gltf_format"] == "GLB"
    assert result["files_written"] == [str(tmp_path / "out" / "model.glb")]


@pytest.mark.parametrize(
    ("name", "fmt"),
    [
        ("c_noext", "GLTF"),
        ("model.txt", "GLTF"),
        ("h.txt", "USD"),
        ("scene.blend", "FBX"),
        ("mesh", "OBJ"),
        ("part.obj", "STL"),
    ],
)
def test_a_path_without_the_formats_extension_is_refused_before_anything_is_written(
    fake: FakeBlender, session: Any, tmp_path: Path, name: str, fmt: str
) -> None:
    """The glTF exporter appended .glb to c_noext and the reply named c_noext."""
    _scene(fake)

    with pytest.raises(ValueError, match="writes a file ending in"):
        session.export_file(str(tmp_path / name), fmt)

    assert list(tmp_path.iterdir()) == []
    assert fake.calls == []


@pytest.mark.parametrize(
    ("name", "gltf_format"),
    [
        ("model.gltf", "GLB"),
        ("model.glb", "GLTF_SEPARATE"),
        ("model.glb", "GLTF_EMBEDDED"),
    ],
)
def test_a_gltf_format_that_disagrees_with_the_extension_is_refused(
    fake: FakeBlender, session: Any, tmp_path: Path, name: str, gltf_format: str
) -> None:
    _scene(fake)

    with pytest.raises(ValueError, match=f"gltf_format {gltf_format} writes"):
        session.export_file(str(tmp_path / name), "GLTF", gltf_format=gltf_format)

    assert fake.calls == []


def test_gltf_embedded_is_allowed_for_the_export_only(
    fake: FakeBlender, session: Any, tmp_path: Path
) -> None:
    _scene(fake)

    result = session.export_file(
        str(tmp_path / "one.gltf"), "GLTF", gltf_format="GLTF_EMBEDDED"
    )

    assert result["files_written"] == [str(tmp_path / "one.gltf")]
    assert fake.gltf_preferences.allow_embedded_format is False


def test_gltf_format_is_refused_for_other_formats(
    fake: FakeBlender, session: Any, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="gltf_format applies to GLTF"):
        session.export_file(str(tmp_path / "a.fbx"), "FBX", gltf_format="GLB")


# -- companion files ---------------------------------------------------------


def test_an_unrelated_bin_beside_the_gltf_is_not_replaced(
    fake: FakeBlender, session: Any, tmp_path: Path
) -> None:
    _scene(fake)
    (tmp_path / "model.bin").write_bytes(b"SOMEONE ELSE'S BUFFER")

    with pytest.raises(FileExistsError, match="overwrite_companion_files"):
        session.export_file(str(tmp_path / "model.gltf"), "GLTF")

    assert (tmp_path / "model.bin").read_bytes() == b"SOMEONE ELSE'S BUFFER"
    assert not (tmp_path / "model.gltf").exists()


def test_an_unrelated_mtl_is_replaced_only_when_confirmed(
    fake: FakeBlender, session: Any, tmp_path: Path
) -> None:
    _scene(fake)
    (tmp_path / "mesh.mtl").write_text("newmtl Theirs\n")

    with pytest.raises(FileExistsError):
        session.export_file(str(tmp_path / "mesh.obj"), "OBJ")
    result = session.export_file(
        str(tmp_path / "mesh.obj"), "OBJ", overwrite_companion_files=True
    )

    assert result["files_written"] == [
        str(tmp_path / "mesh.obj"),
        str(tmp_path / "mesh.mtl"),
    ]


@pytest.mark.parametrize(("name", "fmt"), [("model.gltf", "GLTF"), ("mesh.obj", "OBJ")])
def test_exporting_again_replaces_the_companion_of_the_file_it_replaces(
    fake: FakeBlender, session: Any, tmp_path: Path, name: str, fmt: str
) -> None:
    _scene(fake)
    session.export_file(str(tmp_path / name), fmt)

    result = session.export_file(str(tmp_path / name), fmt)

    assert len(result["files_written"]) == 2


# -- choosing the objects (B1-04, B1-21) -------------------------------------


def test_gltf_selected_only_uses_blenders_use_selection(
    fake: FakeBlender, session: Any, tmp_path: Path
) -> None:
    """Blender 4.2+ names it use_selection; export_selected raised a TypeError."""
    objects = _scene(fake)
    objects["Cube"].select_set(True)

    result = session.export_file(str(tmp_path / "sel.glb"), "GLTF", selected_only=True)

    assert _call(fake, "export_scene.gltf")["use_selection"] is True
    assert json.loads((tmp_path / "sel.glb").read_text())["nodes"] == ["Cube"]
    assert result["object_count"] == 1


@pytest.mark.parametrize(
    ("name", "fmt"),
    [
        ("a.obj", "OBJ"),
        ("a.fbx", "FBX"),
        ("a.usda", "USD"),
        ("a.stl", "STL"),
        ("a.ply", "PLY"),
        ("a.glb", "GLTF"),
    ],
)
def test_selected_only_with_nothing_selected_is_an_error_and_writes_nothing(
    fake: FakeBlender, session: Any, tmp_path: Path, name: str, fmt: str
) -> None:
    """OBJ reported success with no file, and USD/FBX wrote empty files."""
    _scene(fake)

    with pytest.raises(ValueError, match="no object is selected"):
        session.export_file(str(tmp_path / name), fmt, selected_only=True)

    assert list(tmp_path.iterdir()) == []
    assert fake.calls == []


def test_object_names_export_exactly_those_and_restore_the_selection(
    fake: FakeBlender, session: Any, tmp_path: Path
) -> None:
    objects = _scene(fake)
    objects["Light"].select_set(True)
    fake.view_layer.objects.active = objects["Light"]

    result = session.export_file(
        str(tmp_path / "pair.fbx"), "FBX", object_names=["Sphere", "Cube", "Sphere"]
    )

    assert json.loads((tmp_path / "pair.fbx").read_text()) == ["Cube", "Sphere"]
    assert result["object_count"] == 2
    assert [o.name for o in fake.objects.values() if o.selected] == ["Light"]
    assert fake.view_layer.objects.active is objects["Light"]


def test_a_hidden_object_in_object_names_is_refused_rather_than_left_out(
    fake: FakeBlender, session: Any, tmp_path: Path
) -> None:
    objects = _scene(fake)
    objects["Sphere"].hidden = True
    objects["Light"].select_set(True)

    with pytest.raises(ValueError, match="'Sphere' is hidden"):
        session.export_file(
            str(tmp_path / "pair.fbx"), "FBX", object_names=["Cube", "Sphere"]
        )

    assert [o.name for o in fake.objects.values() if o.selected] == ["Light"]


def test_an_object_outside_the_view_layer_is_refused(
    fake: FakeBlender, session: Any, tmp_path: Path
) -> None:
    _scene(fake)
    fake.add(FakeObject("Elsewhere", in_view_layer=False))

    with pytest.raises(ValueError, match="not in view layer"):
        session.export_file(str(tmp_path / "x.fbx"), "FBX", object_names=["Elsewhere"])


def test_a_missing_object_is_refused(
    fake: FakeBlender, session: Any, tmp_path: Path
) -> None:
    _scene(fake)

    with pytest.raises(ValueError, match="Object not found: Nope"):
        session.export_file(str(tmp_path / "x.fbx"), "FBX", object_names=["Nope"])


def test_collection_name_exports_its_objects_in_the_view_layer(
    fake: FakeBlender, session: Any, tmp_path: Path
) -> None:
    objects = _scene(fake)
    outside = FakeObject("Excluded", in_view_layer=False)
    fake.add(outside)
    fake.collections["Props"] = SimpleNamespace(
        all_objects=[objects["Sphere"], outside, objects["Light"]]
    )

    result = session.export_file(
        str(tmp_path / "props.usda"), "USD", collection_name="Props"
    )

    assert json.loads((tmp_path / "props.usda").read_text()) == ["Sphere", "Light"]
    assert result["object_count"] == 2


def test_an_empty_or_unknown_collection_is_refused(
    fake: FakeBlender, session: Any, tmp_path: Path
) -> None:
    _scene(fake)
    fake.collections["Empty"] = SimpleNamespace(all_objects=[])

    with pytest.raises(ValueError, match="holds no object"):
        session.export_file(str(tmp_path / "x.usda"), "USD", collection_name="Empty")
    with pytest.raises(ValueError, match="Collection not found"):
        session.export_file(str(tmp_path / "x.usda"), "USD", collection_name="Nope")


def test_only_one_way_of_choosing_objects_is_accepted(
    fake: FakeBlender, session: Any, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="one of selected_only"):
        session.export_file(
            str(tmp_path / "x.fbx"), "FBX", selected_only=True, object_names=["Cube"]
        )


# -- modifiers and animation (B1-15) -----------------------------------------


@pytest.mark.parametrize(
    ("name", "fmt", "op", "key"),
    [
        ("m.glb", "GLTF", "export_scene.gltf", "export_apply"),
        ("m.fbx", "FBX", "export_scene.fbx", "use_mesh_modifiers"),
        ("m.obj", "OBJ", "wm.obj_export", "apply_modifiers"),
        ("m.stl", "STL", "wm.stl_export", "apply_modifiers"),
        ("m.ply", "PLY", "wm.ply_export", "apply_modifiers"),
    ],
)
@pytest.mark.parametrize("apply", [True, False])
def test_apply_modifiers_reaches_every_exporter(
    fake: FakeBlender,
    session: Any,
    tmp_path: Path,
    name: str,
    fmt: str,
    op: str,
    key: str,
    apply: bool,
) -> None:
    """The glTF exporter left modifiers off: a Subsurf cube kept its 8 corners."""
    _scene(fake)

    if apply:
        result = session.export_file(str(tmp_path / name), fmt)
    else:
        result = session.export_file(str(tmp_path / name), fmt, apply_modifiers=False)

    assert _call(fake, op)[key] is apply
    assert result["exporter_options"][key] is apply


def test_blender_3_6_exports_obj_through_its_python_exporter(
    fake: FakeBlender, session: Any, tmp_path: Path
) -> None:
    """Before 4.0 OBJ went through export_scene.obj, whose keywords differ."""
    objects = _scene(fake)
    objects["Cube"].select_set(True)
    session._is_blender_4_plus = False
    session._blender_version = (3, 6, 0)
    fake.ops.export_scene.obj = fake._op(
        "export_scene.obj",
        {
            "filepath": None,
            "use_selection": None,
            "use_mesh_modifiers": None,
            "axis_forward": AXES,
            "axis_up": AXES,
        },
        fake._write_plain,
    )

    session.export_file(
        str(tmp_path / "a.obj"),
        "OBJ",
        selected_only=True,
        axis_forward="-Y",
        axis_up="Z",
    )

    assert _call(fake, "export_scene.obj") == {
        "filepath": str(tmp_path / "a.obj"),
        "use_selection": True,
        "use_mesh_modifiers": True,
        "axis_forward": "-Y",
        "axis_up": "Z",
    }


def test_usd_cannot_leave_modifiers_out(
    fake: FakeBlender, session: Any, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="USD export always applies modifiers"):
        session.export_file(str(tmp_path / "m.usda"), "USD", apply_modifiers=False)


@pytest.mark.parametrize(
    ("name", "fmt", "op", "key"),
    [
        ("a.usda", "USD", "wm.usd_export", "export_animation"),
        ("a.glb", "GLTF", "export_scene.gltf", "export_animations"),
        ("a.fbx", "FBX", "export_scene.fbx", "bake_anim"),
    ],
)
def test_animation_is_exported_by_default_and_can_be_left_out(
    fake: FakeBlender,
    session: Any,
    tmp_path: Path,
    name: str,
    fmt: str,
    op: str,
    key: str,
) -> None:
    """Blender's USD exporter leaves animation out unless export_animation is passed."""
    _scene(fake)

    session.export_file(str(tmp_path / name), fmt)
    assert _call(fake, op)[key] is True
    session.export_file(str(tmp_path / name), fmt, export_animation=False)
    assert _call(fake, op)[key] is False


def test_obj_export_never_writes_one_file_per_frame(
    fake: FakeBlender, session: Any, tmp_path: Path
) -> None:
    _scene(fake)

    session.export_file(str(tmp_path / "a.obj"), "OBJ")

    assert "export_animation" not in _call(fake, "wm.obj_export")


def test_a_frame_range_applies_for_the_export_and_is_restored(
    fake: FakeBlender, session: Any, tmp_path: Path
) -> None:
    _scene(fake)

    session.export_file(str(tmp_path / "a.glb"), "GLTF", frame_start=300, frame_end=320)

    assert fake.frames_seen == [(300, 320)]
    assert (fake.scene.frame_start, fake.scene.frame_end) == (1, 250)
    assert fake.scene.use_preview_range is True
    assert _call(fake, "export_scene.gltf")["export_frame_range"] is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"frame_start": 5}, "together"),
        ({"frame_start": 9, "frame_end": 5}, "after frame_end"),
        (
            {"frame_start": 1, "frame_end": 5, "export_animation": False},
            "need export_animation",
        ),
    ],
)
def test_frame_range_misuse_is_refused(
    fake: FakeBlender,
    session: Any,
    tmp_path: Path,
    kwargs: Dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        session.export_file(str(tmp_path / "a.fbx"), "FBX", **kwargs)


def test_a_frame_range_is_refused_for_formats_without_animation(
    fake: FakeBlender, session: Any, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="apply to GLTF, FBX and USD"):
        session.export_file(str(tmp_path / "a.stl"), "STL", frame_start=1, frame_end=2)


# -- format options ----------------------------------------------------------


def test_fbx_axes_and_scale_pass_through_and_cpp_exporters_spell_negative_axes(
    fake: FakeBlender, session: Any, tmp_path: Path
) -> None:
    _scene(fake)

    session.export_file(
        str(tmp_path / "a.fbx"),
        "FBX",
        axis_forward="-Z",
        axis_up="Y",
        global_scale=0.01,
    )
    session.export_file(str(tmp_path / "a.obj"), "OBJ", axis_forward="-Y", axis_up="Z")

    fbx = _call(fake, "export_scene.fbx")
    assert (fbx["axis_forward"], fbx["axis_up"], fbx["global_scale"]) == (
        "-Z",
        "Y",
        0.01,
    )
    obj = _call(fake, "wm.obj_export")
    assert (obj["forward_axis"], obj["up_axis"]) == ("NEGATIVE_Y", "Z")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"axis_forward": "Y"}, "together"),
        ({"axis_forward": "Y", "axis_up": "-Y"}, "different axes"),
    ],
)
def test_axis_misuse_is_refused(
    fake: FakeBlender,
    session: Any,
    tmp_path: Path,
    kwargs: Dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        session.export_file(str(tmp_path / "a.fbx"), "FBX", **kwargs)


@pytest.mark.parametrize(
    ("name", "fmt", "kwargs", "message"),
    [
        (
            "a.glb",
            "GLTF",
            {"axis_forward": "-Z", "axis_up": "Y"},
            "FBX, OBJ, STL and PLY",
        ),
        ("a.usda", "USD", {"global_scale": 2.0}, "global_scale applies"),
        ("a.fbx", "FBX", {"meters_per_unit": 0.01}, "meters_per_unit applies to USD"),
        ("a.fbx", "FBX", {"usd_up_axis": "Y"}, "usd_up_axis applies to USD"),
        (
            "a.glb",
            "GLTF",
            {"root_prim_path": "/World"},
            "root_prim_path applies to USD",
        ),
    ],
)
def test_an_option_of_another_format_is_refused(
    fake: FakeBlender,
    session: Any,
    tmp_path: Path,
    name: str,
    fmt: str,
    kwargs: Dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        session.export_file(str(tmp_path / name), fmt, **kwargs)


def test_usd_up_axis_units_and_root_prim(
    fake: FakeBlender, session: Any, tmp_path: Path
) -> None:
    _scene(fake)

    session.export_file(
        str(tmp_path / "a.usda"),
        "USD",
        usd_up_axis="Y",
        meters_per_unit=0.01,
        root_prim_path="/World",
    )
    usd = _call(fake, "wm.usd_export")
    assert usd["convert_orientation"] is True
    assert usd["export_global_up_selection"] == "Y"
    assert usd["export_global_forward_selection"] == "NEGATIVE_Z"
    assert usd["convert_scene_units"] == "CENTIMETERS"
    assert "meters_per_unit" not in usd
    assert usd["root_prim_path"] == "/World"

    session.export_file(
        str(tmp_path / "b.usda"), "USD", meters_per_unit=0.5, root_prim_path=""
    )
    usd = _call(fake, "wm.usd_export")
    assert (usd["convert_scene_units"], usd["meters_per_unit"]) == ("CUSTOM", 0.5)
    assert usd["root_prim_path"] == ""


def test_an_option_this_blender_lacks_is_an_error_naming_the_version(
    fake: FakeBlender, session: Any, tmp_path: Path
) -> None:
    """Blender 4.2's USD exporter has no metersPerUnit option."""
    for key in ("convert_scene_units", "meters_per_unit"):
        del fake.ops.wm.usd_export.properties[key]

    with pytest.raises(
        ValueError, match="meters_per_unit is not supported by Blender 5.0.1"
    ):
        session.export_file(str(tmp_path / "a.usda"), "USD", meters_per_unit=0.01)


def test_an_invalid_root_prim_path_is_refused(
    fake: FakeBlender, session: Any, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="absolute USD prim path"):
        session.export_file(str(tmp_path / "a.usda"), "USD", root_prim_path="World")


# -- folders, written files, modes (B1-19, B1-21) ----------------------------


@pytest.mark.parametrize(
    ("name", "fmt"),
    [
        ("x.fbx", "FBX"),
        ("x.obj", "OBJ"),
        ("x.stl", "STL"),
        ("x.ply", "PLY"),
        ("x.glb", "GLTF"),
    ],
)
def test_a_missing_folder_is_created_for_every_format(
    fake: FakeBlender, session: Any, tmp_path: Path, name: str, fmt: str
) -> None:
    """FBX failed with a raw traceback and OBJ with "Cannot open file"."""
    _scene(fake)
    target = tmp_path / "new" / "deep" / name

    result = session.export_file(str(target), fmt)

    assert result["files_written"][0] == str(target)


def test_a_failed_export_removes_the_folders_it_created(
    fake: FakeBlender, session: Any, tmp_path: Path
) -> None:
    _scene(fake)

    with pytest.raises(RuntimeError, match="wrote no"):
        session.export_file(
            str(tmp_path / "new" / "lights.obj"), "OBJ", object_names=["Light"]
        )

    assert list(tmp_path.iterdir()) == []


def test_usd_textures_are_reported_as_written(
    fake: FakeBlender, session: Any, tmp_path: Path
) -> None:
    _scene(fake)

    result = session.export_file(str(tmp_path / "scene.usdc"), "USD")

    assert result["files_written"] == [
        str(tmp_path / "scene.usdc"),
        str(tmp_path / "textures" / "color.exr"),
    ]


def test_an_export_that_writes_no_file_is_an_error(
    fake: FakeBlender, session: Any, tmp_path: Path
) -> None:
    """Blender's OBJ exporter reports FINISHED and writes nothing without geometry."""
    _scene(fake)

    with pytest.raises(RuntimeError, match="wrote no lights.obj"):
        session.export_file(str(tmp_path / "lights.obj"), "OBJ", object_names=["Light"])


def test_edit_mode_is_entered_again_after_the_export(
    fake: FakeBlender, session: Any, tmp_path: Path
) -> None:
    """The glTF exporter switched the window to Object Mode and left it there."""
    objects = _scene(fake)
    fake.view_layer.objects.active = objects["Cube"]
    objects["Cube"].mode = "EDIT"

    session.export_file(str(tmp_path / "e.glb"), "GLTF")

    modes = [kwargs["mode"] for op, kwargs in fake.calls if op == "object.mode_set"]
    assert modes == ["OBJECT", "EDIT"]
    assert objects["Cube"].mode == "EDIT"


def test_a_failed_export_still_enters_edit_mode_again(
    fake: FakeBlender, session: Any, tmp_path: Path
) -> None:
    objects = _scene(fake)
    fake.view_layer.objects.active = objects["Cube"]
    objects["Cube"].mode = "EDIT"

    def fail(kwargs: Dict[str, Any]) -> None:
        raise RuntimeError("Error: exporter failed")

    fake.ops.export_scene.gltf.run = fail
    with pytest.raises(RuntimeError, match="exporter failed"):
        session.export_file(str(tmp_path / "new" / "e.glb"), "GLTF")

    assert objects["Cube"].mode == "EDIT"
    assert list(tmp_path.iterdir()) == []


def test_a_blender_relative_path_resolves_against_the_open_file(
    monkeypatch: pytest.MonkeyPatch, fake: FakeBlender
) -> None:
    session = blender_runtime.BlenderRuntimeSession(
        settings=Settings(security={"sandbox_enabled": False})
    )

    assert session._export_target("//out/model.glb") == "/blend-dir/out/model.glb"


# export_simready_usd selects its objects for the export only, through the same
# _run_export: tests/blender/test_simready_export.py covers that against an
# exporter that writes real USD, which its SimReady authoring then reads
# (test_export_restores_the_selection_and_active_object,
# test_the_exporter_options_follow_the_request,
# test_export_without_children_writes_only_the_listed_objects).


# -- save_blend_file (B1-21) ---------------------------------------------------


@pytest.mark.parametrize("name", ["s2.txt", "scene.blend1", "scene"])
def test_save_refuses_a_path_that_is_not_a_blend_file(
    fake: FakeBlender, session: Any, tmp_path: Path, name: str
) -> None:
    """It saved s2.txt and made it the open document."""
    with pytest.raises(ValueError, match="Not a .blend file path"):
        session.save_blend_file(str(tmp_path / name))

    assert fake.calls == []


def test_save_accepts_an_upper_case_blend_extension(
    fake: FakeBlender, session: Any, tmp_path: Path
) -> None:
    result = session.save_blend_file(str(tmp_path / "Scene.BLEND"))

    assert result["file_path"] == str(tmp_path / "Scene.BLEND")


# -- import scale ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "fmt", "op", "key"),
    [
        ("a.obj", "OBJ", "wm.obj_import", "global_scale"),
        ("a.fbx", "FBX", "import_scene.fbx", "global_scale"),
        ("a.stl", "STL", "wm.stl_import", "global_scale"),
        ("a.ply", "PLY", "wm.ply_import", "global_scale"),
    ],
)
def test_import_scale_goes_to_the_importer(
    fake: FakeBlender,
    session: Any,
    fbx_parser: Any,
    tmp_path: Path,
    name: str,
    fmt: str,
    op: str,
    key: str,
) -> None:
    (tmp_path / name).write_text("data")

    result = session.import_file(str(tmp_path / name), fmt, scale=0.01)

    assert _call(fake, op)[key] == 0.01
    assert result["scale"] == 0.01


def test_import_without_scale_passes_no_scale(
    fake: FakeBlender, session: Any, tmp_path: Path
) -> None:
    (tmp_path / "a.obj").write_text("data")

    session.import_file(str(tmp_path / "a.obj"), "OBJ")

    assert _call(fake, "wm.obj_import") == {"filepath": str(tmp_path / "a.obj")}


@pytest.mark.parametrize(
    ("name", "fmt", "op"),
    [("a.glb", "GLTF", "import_scene.gltf"), ("a.usdc", "USD", "wm.usd_import")],
)
def test_gltf_and_usd_import_scale_scales_the_imported_roots(
    fake: FakeBlender, session: Any, tmp_path: Path, name: str, fmt: str, op: str
) -> None:
    """Blender's glTF importer has no scale option, and its USD importer's
    scale did nothing for a USD file Blender 5.0.1 had written."""
    (tmp_path / name).write_bytes(b"data")
    root = FakeObject("Root")
    child = FakeObject("Child", parent=root)
    fake.imported = [root, child]

    result = session.import_file(str(tmp_path / name), fmt, scale=2.0)

    assert (root.location, root.scale) == ([2.0, 4.0, 6.0], [2.0, 2.0, 2.0])
    assert (child.location, child.scale) == ([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
    assert result["imported_objects"] == ["Root", "Child"]
    assert result["warning"] is None
    assert _call(fake, op) == {"filepath": str(tmp_path / name)}


def test_import_scale_warns_about_an_animated_root(
    fake: FakeBlender, session: Any, tmp_path: Path
) -> None:
    (tmp_path / "a.glb").write_bytes(b"glTF")
    cached = FakeObject("Cached")
    cached.constraints = [SimpleNamespace(type="TRANSFORM_CACHE")]
    fake.imported = [FakeObject("Walker", animated=True), cached, FakeObject("Still")]

    result = session.import_file(str(tmp_path / "a.glb"), "GLTF", scale=2.0)

    assert "'Walker', 'Cached' is animated" in result["warning"]


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_import_refuses_a_scale_that_is_not_positive(
    fake: FakeBlender, session: Any, tmp_path: Path, scale: float
) -> None:
    (tmp_path / "a.obj").write_text("data")

    with pytest.raises(ValueError, match="scale must be a positive number"):
        session.import_file(str(tmp_path / "a.obj"), "OBJ", scale=scale)


# -- files an import reads besides the one named (B1-20) ---------------------


@pytest.fixture
def outside(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A folder outside the session's sandbox, holding a secret image."""
    folder = tmp_path_factory.mktemp("outside")
    (folder / "secret.png").write_bytes(b"\x89PNG secret")
    return folder


def _write_gltf(path: Path, uris: List[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "asset": {"version": "2.0"},
                "buffers": [{"uri": "data:application/octet-stream;base64,AAAA"}],
                "images": [{"uri": uri} for uri in uris],
            }
        )
    )


def _assert_refused(raised: pytest.ExceptionInfo, secret: Path, imported: Path) -> None:
    details = raised.value.details
    assert details["file_path"] == str(secret)
    assert details["references_outside"] == [str(secret)]
    assert details["referenced_by"] == str(imported)
    assert "outside the sandbox" in details["hint"]


def test_a_gltf_image_outside_the_sandbox_is_refused_before_importing(
    fake: FakeBlender, session: Any, tmp_path: Path, outside: Path
) -> None:
    """The image a glTF inside the sandbox pointed at outside it was imported."""
    gltf = tmp_path / "textured.gltf"
    _write_gltf(gltf, [str(outside / "secret.png")])

    with pytest.raises(PermissionError) as raised:
        session.import_file(str(gltf), "GLTF")

    _assert_refused(raised, outside / "secret.png", gltf)
    assert fake.calls == []


def test_a_relative_gltf_uri_climbing_out_of_the_sandbox_is_refused(
    fake: FakeBlender, session: Any, tmp_path: Path, outside: Path
) -> None:
    gltf = tmp_path / "deep" / "model.gltf"
    gltf.parent.mkdir()
    relative = os.path.relpath(outside / "secret.png", gltf.parent)
    _write_gltf(gltf, [relative.replace(os.sep, "/").replace(" ", "%20")])

    with pytest.raises(PermissionError):
        session.import_file(str(gltf), "GLTF")


def test_a_glb_json_chunk_is_checked(
    fake: FakeBlender, session: Any, tmp_path: Path, outside: Path
) -> None:
    import struct

    document = json.dumps({"images": [{"uri": str(outside / "secret.png")}]}).encode()
    document += b" " * (-len(document) % 4)
    glb = tmp_path / "model.glb"
    glb.write_bytes(
        b"glTF"
        + struct.pack("<II", 2, 12 + 8 + len(document))
        + struct.pack("<II", len(document), 0x4E4F534A)
        + document
    )

    with pytest.raises(PermissionError):
        session.import_file(str(glb), "GLTF")


def test_gltf_references_inside_the_sandbox_or_missing_import(
    fake: FakeBlender, session: Any, tmp_path: Path, outside: Path
) -> None:
    (tmp_path / "tex.png").write_bytes(b"\x89PNG")
    gltf = tmp_path / "ok.gltf"
    _write_gltf(gltf, ["tex.png", str(outside / "missing.png")])

    session.import_file(str(gltf), "GLTF")

    assert _call(fake, "import_scene.gltf") == {"filepath": str(gltf)}


def test_an_obj_mtl_library_outside_the_sandbox_is_refused(
    fake: FakeBlender, session: Any, tmp_path: Path, outside: Path
) -> None:
    (outside / "evil.mtl").write_text("newmtl M\n")
    obj = tmp_path / "mesh.obj"
    obj.write_text(f"mtllib {os.path.relpath(outside / 'evil.mtl', tmp_path)}\no M\n")

    with pytest.raises(PermissionError) as raised:
        session.import_file(str(obj), "OBJ")

    assert raised.value.details["references_outside"] == [str(outside / "evil.mtl")]


@pytest.mark.parametrize(
    "line", ["map_Kd -s 1 1 1 -o 0 0 {path}", "map_Bump -bm 0.5 {path}", "norm {path}"]
)
def test_an_mtl_texture_outside_the_sandbox_is_refused(
    fake: FakeBlender, session: Any, tmp_path: Path, outside: Path, line: str
) -> None:
    texture = line.format(path=outside / "secret.png")
    (tmp_path / "mesh.mtl").write_text(f"newmtl M\n{texture}\n")
    obj = tmp_path / "mesh.obj"
    obj.write_text("mtllib mesh.mtl\no M\n")

    with pytest.raises(PermissionError) as raised:
        session.import_file(str(obj), "OBJ")

    _assert_refused(raised, outside / "secret.png", obj)


def test_an_mtl_texture_beside_it_imports(
    fake: FakeBlender, session: Any, tmp_path: Path
) -> None:
    (tmp_path / "wood.png").write_bytes(b"\x89PNG")
    (tmp_path / "mesh.mtl").write_text('newmtl M\nmap_Kd -s 2 2 "wood.png"\n')
    obj = tmp_path / "mesh.obj"
    obj.write_text("mtllib mesh.mtl\no M\n")

    session.import_file(str(obj), "OBJ")

    assert _call(fake, "wm.obj_import") == {"filepath": str(obj)}


def test_an_fbx_texture_outside_the_sandbox_is_refused(
    fake: FakeBlender, session: Any, tmp_path: Path, outside: Path, fbx_parser: Any
) -> None:
    fbx = tmp_path / "model.fbx"
    fbx.write_bytes(b"Kaydara FBX Binary")
    fbx_parser.textures.append({b"FileName": str(outside / "secret.png")})

    with pytest.raises(PermissionError) as raised:
        session.import_file(str(fbx), "FBX")

    _assert_refused(raised, outside / "secret.png", fbx)


def test_an_fbx_texture_found_beside_it_imports(
    fake: FakeBlender, session: Any, tmp_path: Path, fbx_parser: Any
) -> None:
    """An absolute path from another machine is looked up by name beside the FBX."""
    (tmp_path / "wood.png").write_bytes(b"\x89PNG")
    fbx = tmp_path / "model.fbx"
    fbx.write_bytes(b"Kaydara FBX Binary")
    fbx_parser.textures.append({b"FileName": "C:/Users/artist/textures/wood.png"})

    session.import_file(str(fbx), "FBX")

    assert _call(fake, "import_scene.fbx") == {
        "filepath": str(fbx),
        "use_image_search": False,
    }


def test_a_usd_texture_or_layer_outside_the_sandbox_is_refused(
    fake: FakeBlender, session: Any, tmp_path: Path, outside: Path
) -> None:
    (outside / "other.usda").write_text("#usda 1.0\n")
    stage = tmp_path / "scene.usda"
    stage.write_text(
        "#usda 1.0\n"
        'def Xform "World" {\n'
        '    def Shader "Tex" {\n'
        f"        asset inputs:file = @{outside / 'secret.png'}@\n"
        "    }\n"
        "}\n"
    )

    with pytest.raises(PermissionError) as raised:
        session.import_file(str(stage), "USD")
    _assert_refused(raised, outside / "secret.png", stage)

    stage.write_text(f'#usda 1.0\n(\n    subLayers = [@{outside / "other.usda"}@]\n)\n')
    with pytest.raises(PermissionError) as raised:
        session.import_file(str(stage), "USD")
    assert raised.value.details["references_outside"] == [str(outside / "other.usda")]


def test_a_usd_layer_inside_the_sandbox_is_followed(
    fake: FakeBlender, session: Any, tmp_path: Path, outside: Path
) -> None:
    secret = outside / "secret.png"
    (tmp_path / "props.usda").write_text(
        f'#usda 1.0\ndef "P" {{\n    asset inputs:file = @{secret}@\n}}\n'
    )
    stage = tmp_path / "scene.usda"
    stage.write_text('#usda 1.0\ndef "World" (\n    references = @./props.usda@\n) {}\n')

    with pytest.raises(PermissionError) as raised:
        session.import_file(str(stage), "USD")

    assert raised.value.details["references_outside"] == [str(secret)]


def test_usd_references_inside_the_sandbox_import(
    fake: FakeBlender, session: Any, tmp_path: Path
) -> None:
    (tmp_path / "tex.png").write_bytes(b"\x89PNG")
    (tmp_path / "props.usda").write_text("#usda 1.0\n")
    stage = tmp_path / "scene.usda"
    stage.write_text(
        "#usda 1.0\n(\n    subLayers = [@./props.usda@]\n)\n"
        'def Shader "Tex" {\n    asset inputs:file = @tex.png@\n}\n'
    )

    session.import_file(str(stage), "USD")

    assert _call(fake, "wm.usd_import") == {"filepath": str(stage)}


def test_a_binary_usd_file_without_pxr_is_refused(
    monkeypatch: pytest.MonkeyPatch, fake: FakeBlender, session: Any, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "pxr", None)
    stage = tmp_path / "scene.usdc"
    stage.write_bytes(b"PXR-USDC binary")

    with pytest.raises(PermissionError) as raised:
        session.import_file(str(stage), "USD")

    assert "Could not check" in raised.value.details["hint"]
    assert fake.calls == []


def test_a_text_usd_file_without_pxr_is_scanned(
    monkeypatch: pytest.MonkeyPatch,
    fake: FakeBlender,
    session: Any,
    tmp_path: Path,
    outside: Path,
) -> None:
    monkeypatch.setitem(sys.modules, "pxr", None)
    secret = outside / "secret.png"
    stage = tmp_path / "scene.usda"
    stage.write_text(f'#usda 1.0\ndef "T" {{\n    asset inputs:file = @{secret}@\n}}\n')

    with pytest.raises(PermissionError):
        session.import_file(str(stage), "USD")


def test_references_are_not_checked_with_the_sandbox_off(
    fake: FakeBlender, tmp_path: Path, outside: Path
) -> None:
    session = blender_runtime.BlenderRuntimeSession(
        settings=Settings(security={"sandbox_enabled": False})
    )
    gltf = tmp_path / "textured.gltf"
    _write_gltf(gltf, [str(outside / "secret.png")])

    session.import_file(str(gltf), "GLTF")

    assert _call(fake, "import_scene.gltf") == {"filepath": str(gltf)}


# -- review of #324: searches and fallbacks the importers make ----------------


def test_fbx_import_switches_blenders_recursive_image_search_off_in_the_sandbox(
    fake: FakeBlender, session: Any, tmp_path: Path, outside: Path, fbx_parser: Any
) -> None:
    """use_image_search walked the texture's folder and found a same-named image
    deeper down, outside the sandbox, which a GLB export then embedded."""
    (outside / "tex" / "deep").mkdir(parents=True)
    (outside / "tex" / "deep" / "missing.png").write_bytes(b"\\x89PNG secret")
    fbx = tmp_path / "model.fbx"
    fbx.write_bytes(b"Kaydara FBX Binary")
    relative = os.path.relpath(outside / "tex" / "missing.png", tmp_path)
    fbx_parser.textures.append({b"RelativeFilename": relative})

    session.import_file(str(fbx), "FBX")

    assert _call(fake, "import_scene.fbx")["use_image_search"] is False


def test_fbx_import_keeps_the_image_search_with_the_sandbox_off(
    fake: FakeBlender, tmp_path: Path, fbx_parser: Any
) -> None:
    session = blender_runtime.BlenderRuntimeSession(
        settings=Settings(security={"sandbox_enabled": False})
    )
    fbx = tmp_path / "model.fbx"
    fbx.write_bytes(b"Kaydara FBX Binary")

    session.import_file(str(fbx), "FBX")

    assert "use_image_search" not in _call(fake, "import_scene.fbx")


def test_an_fbx_texture_named_only_by_its_object_name_is_checked(
    fake: FakeBlender, session: Any, tmp_path: Path, outside: Path, fbx_parser: Any
) -> None:
    """Without a file name the importer loads the texture's own name as a path."""
    fbx = tmp_path / "model.fbx"
    fbx.write_bytes(b"Kaydara FBX Binary")
    fbx_parser.textures.append({b"name": str(outside / "secret.png")})

    with pytest.raises(PermissionError) as raised:
        session.import_file(str(fbx), "FBX")

    _assert_refused(raised, outside / "secret.png", fbx)


def test_an_fbx_texture_spelled_in_another_case_is_checked(
    fake: FakeBlender, session: Any, tmp_path: Path, outside: Path, fbx_parser: Any
) -> None:
    """load_image resolves the case of a path that does not exist (resolve_ncase)."""
    fbx = tmp_path / "model.fbx"
    fbx.write_bytes(b"Kaydara FBX Binary")
    fbx_parser.textures.append({b"FileName": str(outside / "SECRET.PNG")})

    with pytest.raises(PermissionError) as raised:
        session.import_file(str(fbx), "FBX")

    (reported,) = raised.value.details["references_outside"]
    assert reported.lower() == str(outside / "secret.png").lower()


def test_resolve_ncase_matches_blenders_on_a_case_sensitive_file_system(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """bpy.path.resolve_ncase, which load_image tries for each path."""
    tree = {"/r": ["Textures"], "/r/Textures": ["Wood.PNG", "x"]}
    files = {"/r", "/r/Textures", "/r/Textures/Wood.PNG", "/r/Textures/x"}
    monkeypatch.setattr(blender_runtime.os.path, "exists", lambda p: p in files)
    monkeypatch.setattr(blender_runtime.os.path, "isdir", lambda p: p in tree)
    monkeypatch.setattr(blender_runtime.os, "listdir", lambda p: tree[p])

    resolve = blender_runtime._resolve_ncase
    assert resolve("/r/textures/wood.png") == "/r/Textures/Wood.PNG"
    assert resolve("/r/Textures/Wood.PNG") == "/r/Textures/Wood.PNG"
    assert resolve("/r/textures/stone.png") == "/r/textures/stone.png"
    assert resolve("/elsewhere/a.png") == "/elsewhere/a.png"


@pytest.mark.parametrize("fmt", ["GLTF", "FBX", "OBJ"])
def test_references_outside_name_the_resolved_file(
    fake: FakeBlender,
    session: Any,
    tmp_path: Path,
    outside: Path,
    fbx_parser: Any,
    fmt: str,
) -> None:
    """They listed the ../../ path as the reference spelled it."""
    folder = tmp_path / "deep"
    folder.mkdir()
    climb = os.path.relpath(outside / "secret.png", folder)
    if fmt == "GLTF":
        imported = folder / "model.gltf"
        _write_gltf(imported, [climb.replace(os.sep, "/")])
    elif fmt == "FBX":
        imported = folder / "model.fbx"
        imported.write_bytes(b"Kaydara FBX Binary")
        fbx_parser.textures.append({b"RelativeFilename": climb})
    else:
        imported = folder / "mesh.obj"
        (folder / "mesh.mtl").write_text(f"newmtl M\nmap_Kd {climb}\n")
        imported.write_text("mtllib mesh.mtl\no M\n")

    with pytest.raises(PermissionError) as raised:
        session.import_file(str(imported), fmt)

    assert raised.value.details["references_outside"] == [str(outside / "secret.png")]
    assert ".." not in raised.value.details["file_path"]


def test_a_usd_asset_the_resolver_finds_on_its_search_path_is_checked(
    monkeypatch: pytest.MonkeyPatch,
    fake: FakeBlender,
    session: Any,
    tmp_path: Path,
    outside: Path,
) -> None:
    """A search-path asset missing beside the layer resolves through
    PXR_AR_DEFAULT_SEARCH_PATH, which may point outside the sandbox."""
    stage = tmp_path / "scene.usda"
    stage.write_text('#usda 1.0\ndef "T" {\n    asset inputs:file = @textures/secret.png@\n}\n')
    hits = {"textures/secret.png": str(outside / "secret.png")}
    monkeypatch.setattr(
        blender_runtime,
        "_usd_search_path_hit",
        lambda ar, asset, layer, anchored: hits.get(asset),
    )

    with pytest.raises(PermissionError) as raised:
        session.import_file(str(stage), "USD")

    _assert_refused(raised, outside / "secret.png", stage)


def test_usd_search_path_hit_asks_the_resolver_only_for_missing_anchored_paths(
    tmp_path: Path,
) -> None:
    class Resolver:
        def __init__(self) -> None:
            self.asked: List[Any] = []

        def CreateIdentifier(self, asset: str, anchor: Any) -> str:  # noqa: N802
            self.asked.append((asset, anchor))
            return asset

        def Resolve(self, identifier: str) -> str:  # noqa: N802
            return "/search/textures/a.png"

    resolver = Resolver()
    ar = SimpleNamespace(
        GetResolver=lambda: resolver,
        ResolvedPath=lambda path: ("resolved", path),
        SplitPackageRelativePathOuter=lambda path: (path, ""),
    )
    (tmp_path / "here.png").write_bytes(b"x")

    hit = blender_runtime._usd_search_path_hit
    assert hit(ar, "here.png", "/l.usda", str(tmp_path / "here.png")) is None
    assert resolver.asked == []
    assert hit(ar, "textures/a.png", "/l.usda", str(tmp_path / "textures/a.png")) == (
        "/search/textures/a.png"
    )
    assert resolver.asked == [("textures/a.png", ("resolved", "/l.usda"))]
