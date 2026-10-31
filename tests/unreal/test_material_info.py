"""get_unreal_material_info reports a material's parameters, not its object properties.

The tool used to ask Remote Control to describe the asset
(``/remote/object/describe``) and list every property it returned. Observed
on UE 5.7.4, in and out of Play In Editor: for a Material Instance the
"parameters" were the object's properties (ParameterStateId,
PhysMaterialMask, ScalarParameterValues, ...) with no values, and
``parent_path`` was null; given a package path it listed nothing, for a
StaticMesh as for a material. It now reads the asset through editor
scripting: each scalar, vector and texture parameter with its value, whether
an instance overrides it, the parent and base material, the domain and the
blend mode.

A parameter inside a Material Layer was then read by name, which reads the
material's global parameter of that name. Observed on UE 5.7.4, on a material
with two layers and a blend: a name only a layer or the blend had read 0,
black or no texture with ``overridden`` false, whatever the layer's default or
the instance's override for it was, and a name two layers shared was listed
once. Each parameter now has its association (global, layer or blend), its
layer or blend index and that layer's value and override.

The fake editor below answers as UE 5.7.4 did: MaterialEditingLibrary lists
each parameter's name, a layer's with the material's own, and its default
getters (which take only a Material) and the instance getters read by name,
which reads the global parameter; a transient Material Instance Dynamic
parented to a material copies every parameter Unreal lists for it with its
association, index and the value the material renders with (its override,
else its parent's), and a quick copy keeps the names only; an instance keeps
its overrides with their association and index, and can keep overrides for
names that are no longer parameters (the engine's
Widget3DPassThrough_Translucent keeps RefractionDepthBias); and
``EditorAssetLibrary`` loads nothing during PIE while ``unreal.load_asset``
does.
"""

from __future__ import annotations

import ast
import asyncio
import io
import json
import struct
import sys
from contextlib import contextmanager, redirect_stdout
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from unittest.mock import AsyncMock

import pytest
from fastmcp import Client

from simul.adapters.unreal_runtime import UnrealRuntimeSession
from simul.config import Settings
from simul.mcp import server as server_module
from tests.fakes import AvailableAdapter

PARENT = "/Engine/BasicShapes/BasicShapeMaterial"
WIDGET = "/Engine/EngineMaterials/Widget3DPassThrough"
INSTANCE = "/Game/Materials/MI_Chair"
CHILD = "/Game/Materials/MI_Chair_Red"
WIDGET_INSTANCE = "/Game/Materials/MI_Widget"
WHITE = "/Engine/EngineResources/WhiteSquareTexture"
DEFAULT_TEXTURE = "/Engine/EngineResources/DefaultTexture"


def _f32(value: float) -> float:
    """``value`` as the 32-bit float Unreal stores it."""
    return struct.unpack("f", struct.pack("f", value))[0]


def _object_path(package: str) -> str:
    return package + "." + package.rsplit("/", 1)[-1]


# ---------------------------------------------------------------------------
# A fake editor with UE 5.7.4's material answers
# ---------------------------------------------------------------------------


class _Enum:
    def __init__(self, name: str) -> None:
        self.name = name


class Object:
    def __init__(self, package: str) -> None:
        self.package = package

    def get_path_name(self) -> str:
        return _object_path(self.package)

    def get_class(self) -> SimpleNamespace:
        return SimpleNamespace(get_name=lambda: type(self).__name__)


class Texture(Object):
    pass


class Texture2D(Texture):
    pass


class StaticMesh(Object):
    pass


class LinearColor:
    def __init__(
        self, r: float = 0.0, g: float = 0.0, b: float = 0.0, a: float = 0.0
    ) -> None:
        self.r, self.g, self.b, self.a = (_f32(c) for c in (r, g, b, a))


class _Name(str):
    """Unreal hands parameter names back as ``unreal.Name``."""


KINDS = ("scalar", "vector", "texture")

#: A parameter as Unreal identifies it: (kind, name, association, layer or blend index).
Key = Tuple[str, str, str, int]
#: A Material Layer's or blend's parameter and its value: (kind, name, association, index, value).
Layered = Tuple[str, str, str, int, Any]


def _keyed(
    plain: Dict[str, Dict[str, Any]], layered: Iterable[Layered]
) -> Dict[Key, Any]:
    """Global parameters by kind and name, then the layers' and blends' as given."""
    keyed = {
        (kind, name, "GLOBAL_PARAMETER", -1): value
        for kind in KINDS
        for name, value in plain.get(kind, {}).items()
    }
    keyed.update(
        {(kind, name, association, index): value for kind, name, association, index, value in layered}
    )
    return keyed


def _global(keyed: Dict[Key, Any], kind: str, name: str) -> Any:
    """The global parameter of that kind and name (names ignore case), else None."""
    for (k, n, association, _), value in keyed.items():
        if (k, n.lower(), association) == (kind, name.lower(), "GLOBAL_PARAMETER"):
            return value
    return None


class MaterialInterface(Object):
    pass


class Material(MaterialInterface):
    def __init__(
        self,
        package: str,
        domain: str = "MD_SURFACE",
        blend_mode: str = "BLEND_OPAQUE",
        layered: Iterable[Layered] = (),
        **defaults: Dict[str, Any],
    ) -> None:
        super().__init__(package)
        # Every parameter Unreal lists for the material, with its default.
        self.defaults = _keyed(defaults, layered)
        self.properties = {
            "material_domain": _Enum(domain),
            "blend_mode": _Enum(blend_mode),
        }

    def get_editor_property(self, name: str) -> Any:
        return self.properties[name]

    def get_base_material(self) -> "Material":
        return self

    def listed(self) -> List[Key]:
        return list(self.defaults)

    def rendered(self, key: Key) -> Any:
        return self.defaults[key]

    def by_name(self, kind: str, name: str) -> Any:
        """What reading by name answers: the global parameter of that name, if any."""
        return _global(self.defaults, kind, name)


#: What Unreal renders an instance without a parent with (UE 5.7.4).
WORLD_GRID = Material("/Engine/EngineMaterials/WorldGridMaterial")


class _Entry:
    """A ScalarParameterValue / VectorParameterValue / TextureParameterValue."""

    def __init__(self, name: str, value: Any, association: str, index: int) -> None:
        self.parameter_info = SimpleNamespace(
            name=_Name(name), association=_Enum(association), index=index
        )
        self.parameter_value = value


class _BasePropertyOverrides:
    def __init__(self, blend_mode: Optional[str]) -> None:
        self.values = {
            "override_blend_mode": blend_mode is not None,
            "blend_mode": _Enum(blend_mode or "BLEND_OPAQUE"),
        }

    def get_editor_property(self, name: str) -> Any:
        return self.values[name]


class MaterialInstance(MaterialInterface):
    pass


class MaterialInstanceConstant(MaterialInstance):
    def __init__(
        self,
        package: str,
        parent: Optional[MaterialInterface],
        blend_mode: Optional[str] = None,
        layered: Iterable[Layered] = (),
        **overrides: Dict[str, Any],
    ) -> None:
        super().__init__(package)
        self.parent = parent
        # The instance's own overrides, global by kind and name, then a layer's or blend's.
        self.overrides = _keyed(overrides, layered)
        self.base_property_overrides = _BasePropertyOverrides(blend_mode)

    def get_editor_property(self, name: str) -> Any:
        if name == "parent":
            return self.parent
        if name == "base_property_overrides":
            return self.base_property_overrides
        kind = name.replace("_parameter_values", "")
        return [
            _Entry(n, value, association, index)
            for (k, n, association, index), value in self.overrides.items()
            if k == kind
        ]

    def get_base_material(self) -> Material:
        # Unreal falls back to the default surface material without a parent.
        return WORLD_GRID if self.parent is None else self.parent.get_base_material()

    def listed(self) -> List[Key]:
        # An instance has its parent's parameters.
        return (WORLD_GRID if self.parent is None else self.parent).listed()

    def rendered(self, key: Key) -> Any:
        """Its own override (names ignore case), else what its parent renders with."""
        kind, name, association, index = key
        for (k, n, a, i), value in self.overrides.items():
            if (k, n.lower(), a, i) == (kind, name.lower(), association, index):
                return value
        return (WORLD_GRID if self.parent is None else self.parent).rendered(key)

    def by_name(self, kind: str, name: str) -> Any:
        own = _global(self.overrides, kind, name)
        if own is not None:
            return own
        return (WORLD_GRID if self.parent is None else self.parent).by_name(kind, name)

    # The K2 getters read by name: the global parameter's value the instance renders with.
    def get_scalar_parameter_value(self, name: str) -> float:
        value = self.by_name("scalar", name)
        return 0.0 if value is None else _f32(value)

    def get_vector_parameter_value(self, name: str) -> LinearColor:
        return LinearColor(*(self.by_name("vector", name) or [0.0] * 4))

    def get_texture_parameter_value(self, name: str) -> Optional[Texture]:
        return self.by_name("texture", name)


class MaterialInstanceDynamic(MaterialInstance):
    def __init__(self, parent: MaterialInterface) -> None:
        super().__init__("/Engine/Transient/MID_" + parent.package.rsplit("/", 1)[-1])
        self.parent = parent
        self.values: List[Tuple[str, _Entry]] = []

    def copy_material_instance_parameters(
        self, source: MaterialInterface, quick_parameters_only: bool = False
    ) -> None:
        """Every parameter the parent lists, with its association, index and
        the source's value for it, kept as Unreal stores it; the quick copy
        (CopyMaterialUniformParameters) keeps names only."""
        self.values = []
        for key in self.parent.listed():
            kind, name, association, index = key
            value = source.rendered(key)
            if kind == "scalar":
                value = _f32(value)
            elif kind == "vector":
                value = LinearColor(*value)
            if quick_parameters_only:
                association, index = "GLOBAL_PARAMETER", -1
            self.values.append((kind, _Entry(name, value, association, index)))

    def get_editor_property(self, name: str) -> Any:
        kind = name.replace("_parameter_values", "")
        return [entry for k, entry in self.values if k == kind]


def _material_only(material: Any) -> None:
    # UE 5.7.4 refuses an instance: "allowed Class type: 'Material'".
    if not isinstance(material, Material):
        raise TypeError(
            "Cannot nativize '" + type(material).__name__ + "' as 'Material'"
        )


class MaterialEditingLibrary:
    """Lists each parameter's name, a Material Layer's with the material's
    own, and reads by name, which reads the global parameter."""

    @staticmethod
    def _names(material: MaterialInterface, kind: str) -> List[_Name]:
        return [_Name(name) for k, name, _, _ in material.listed() if k == kind]

    @staticmethod
    def get_scalar_parameter_names(material: MaterialInterface) -> List[_Name]:
        return MaterialEditingLibrary._names(material, "scalar")

    @staticmethod
    def get_vector_parameter_names(material: MaterialInterface) -> List[_Name]:
        return MaterialEditingLibrary._names(material, "vector")

    @staticmethod
    def get_texture_parameter_names(material: MaterialInterface) -> List[_Name]:
        return MaterialEditingLibrary._names(material, "texture")

    @staticmethod
    def get_material_default_scalar_parameter_value(material: Any, name: str) -> float:
        _material_only(material)
        return _f32(material.by_name("scalar", name) or 0.0)

    @staticmethod
    def get_material_default_vector_parameter_value(
        material: Any, name: str
    ) -> LinearColor:
        _material_only(material)
        return LinearColor(*(material.by_name("vector", name) or [0.0] * 4))

    @staticmethod
    def get_material_default_texture_parameter_value(
        material: Any, name: str
    ) -> Optional[Texture]:
        _material_only(material)
        return material.by_name("texture", name)


class MIDCreationFlags:
    NONE = _Enum("NONE")
    TRANSIENT = _Enum("TRANSIENT")


class MaterialLibrary:
    @staticmethod
    def create_dynamic_material_instance(
        world_context_object: Any,
        parent: MaterialInterface,
        optional_name: str = "None",
        creation_flags: _Enum = MIDCreationFlags.NONE,
    ) -> MaterialInstanceDynamic:
        return MaterialInstanceDynamic(parent)


class _Editor:
    """The assets and whether PIE runs."""

    def __init__(self) -> None:
        white = Texture2D(WHITE)
        default_texture = Texture2D(DEFAULT_TEXTURE)
        self.parent = Material(
            PARENT,
            scalar={"Roughness": 0.6407},
            vector={"Color": [0.9, 0.9, 0.9, 1.0]},
        )
        self.widget = Material(
            WIDGET,
            blend_mode="BLEND_TRANSLUCENT",
            scalar={"OpacityFromTexture": 1.0},
            vector={"TintColorAndOpacity": [1.0] * 4, "BackColor": [0.0] * 4},
            texture={"SlateUI": white, "Mask": None},
        )
        self.instance = MaterialInstanceConstant(
            INSTANCE, self.parent, scalar={"Roughness": 0.25}
        )
        self.child = MaterialInstanceConstant(
            CHILD, self.instance, vector={"Color": [1.0, 0.0, 0.0, 1.0]}
        )
        self.widget_instance = MaterialInstanceConstant(
            WIDGET_INSTANCE,
            self.widget,
            blend_mode="BLEND_MASKED",
            texture={"SlateUI": default_texture},
        )
        self.assets = {
            asset.package: asset
            for asset in (
                white,
                default_texture,
                self.parent,
                self.widget,
                self.instance,
                self.child,
                self.widget_instance,
                StaticMesh("/Engine/BasicShapes/Cube"),
            )
        }
        self.pie = False

    def load_asset(self, path: str) -> Any:
        package, _, name = path.partition(".")
        asset = self.assets.get(package)
        if asset is None or (name and asset.get_path_name() != path):
            return None
        return asset

    def module(self) -> SimpleNamespace:
        def editor_asset_library_load(path: str) -> Any:
            # "The Editor is currently in a play mode": nothing loads during PIE.
            return None if self.pie else self.load_asset(path)

        return SimpleNamespace(
            Object=Object,
            Material=Material,
            MaterialInterface=MaterialInterface,
            MaterialInstance=MaterialInstance,
            MaterialInstanceConstant=MaterialInstanceConstant,
            MaterialInstanceDynamic=MaterialInstanceDynamic,
            MaterialEditingLibrary=MaterialEditingLibrary,
            MaterialLibrary=MaterialLibrary,
            MIDCreationFlags=MIDCreationFlags,
            Texture=Texture,
            LinearColor=LinearColor,
            load_asset=self.load_asset,
            EditorAssetLibrary=SimpleNamespace(load_asset=editor_asset_library_load),
        )


# What Remote Control's describe answered for an instance on UE 5.7.4.
DESCRIBED = {
    "Name": "MI_Chair",
    "Class": "/Script/Engine.MaterialInstanceConstant",
    "Properties": [
        {"Name": "ParameterStateId", "Type": "FGuid"},
        {"Name": "PhysMaterialMask", "Type": "UPhysicalMaterialMask"},
        {"Name": "Parent", "Type": "UMaterialInterface"},
        {"Name": "ScalarParameterValues", "Type": "TArray<FScalarParameterValue>"},
    ],
}


@pytest.fixture
def editor(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A session whose scripts run in-process against the fake editor.

    Remote Control's describe answers with object properties, as it did.
    """
    fake = _Editor()
    monkeypatch.setitem(sys.modules, "unreal", fake.module())
    session = UnrealRuntimeSession(Settings())
    http_put = AsyncMock(return_value=DESCRIBED)
    monkeypatch.setattr(session, "_http_put", http_put)
    scripts: List[str] = []

    async def execute(code: str, mode: str = "ExecuteFile") -> Dict[str, Any]:
        scripts.append(code)
        out = io.StringIO()
        try:
            with redirect_stdout(out):
                exec(code, {})
        except Exception as exc:  # UE reports a raised script this way
            return {"ReturnValue": False, "CommandResult": repr(exc)}
        return {
            "ReturnValue": True,
            "LogOutput": [
                {"Type": "Info", "Output": line} for line in out.getvalue().splitlines()
            ],
        }

    monkeypatch.setattr(session, "_execute_python", execute)
    return SimpleNamespace(
        session=session, fake=fake, scripts=scripts, http_put=http_put
    )


def _info(editor: SimpleNamespace, path: str) -> Dict[str, Any]:
    return asyncio.run(editor.session.get_material_info(path))


def _param(
    name: str,
    kind: str,
    value: Any,
    overridden: Optional[bool],
    association: str = "global",
    index: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "name": name,
        "param_type": kind,
        "association": association,
        "index": index,
        "value": value,
        "overridden": overridden,
    }


GREY = [_f32(0.9), _f32(0.9), _f32(0.9), 1.0]


# ---------------------------------------------------------------------------
# Adapter: parameters, overrides, parent, domain and blend mode
# ---------------------------------------------------------------------------


def test_an_instance_reports_its_parameters_overrides_and_parent(
    editor: SimpleNamespace,
) -> None:
    """The live regression: object properties with no values, and no parent."""
    result = _info(editor, INSTANCE)

    assert result == {
        "material_path": _object_path(INSTANCE),
        "class_name": "MaterialInstanceConstant",
        "parent_path": _object_path(PARENT),
        "base_material_path": _object_path(PARENT),
        "material_domain": "MD_SURFACE",
        "blend_mode": "BLEND_OPAQUE",
        "parameters": [
            _param("Roughness", "scalar", 0.25, True),
            _param("Color", "vector", GREY, False),
        ],
    }
    assert len(editor.scripts) == 1
    editor.http_put.assert_not_called()


def test_a_material_reports_its_defaults(editor: SimpleNamespace) -> None:
    result = _info(editor, PARENT + ".BasicShapeMaterial")

    assert result == {
        "material_path": _object_path(PARENT),
        "class_name": "Material",
        "parent_path": None,
        "base_material_path": _object_path(PARENT),
        "material_domain": "MD_SURFACE",
        "blend_mode": "BLEND_OPAQUE",
        "parameters": [
            _param("Roughness", "scalar", _f32(0.6407), None),
            _param("Color", "vector", GREY, None),
        ],
    }


def test_an_instance_of_an_instance_inherits_through_its_parent(
    editor: SimpleNamespace,
) -> None:
    result = _info(editor, CHILD)

    assert result["parent_path"] == _object_path(INSTANCE)
    assert result["base_material_path"] == _object_path(PARENT)
    assert result["parameters"] == [
        # The parent's override, inherited.
        _param("Roughness", "scalar", 0.25, False),
        _param("Color", "vector", [1.0, 0.0, 0.0, 1.0], True),
    ]


def test_textures_are_asset_paths_and_the_blend_mode_an_override(
    editor: SimpleNamespace,
) -> None:
    result = _info(editor, WIDGET_INSTANCE)

    assert result["blend_mode"] == "BLEND_MASKED"
    assert result["parameters"] == [
        _param("OpacityFromTexture", "scalar", 1.0, False),
        # By kind, then name ignoring case.
        _param("BackColor", "vector", [0.0] * 4, False),
        _param("TintColorAndOpacity", "vector", [1.0] * 4, False),
        _param("Mask", "texture", None, False),
        _param("SlateUI", "texture", _object_path(DEFAULT_TEXTURE), True),
    ]


def test_an_instance_without_a_blend_override_has_its_parents(
    editor: SimpleNamespace,
) -> None:
    editor.fake.widget_instance.base_property_overrides = _BasePropertyOverrides(None)
    assert _info(editor, WIDGET_INSTANCE)["blend_mode"] == "BLEND_TRANSLUCENT"

    child = MaterialInstanceConstant(
        "/Game/Materials/MI_Widget_Child", editor.fake.widget_instance
    )
    editor.fake.assets[child.package] = child
    editor.fake.widget_instance.base_property_overrides = _BasePropertyOverrides(
        "BLEND_ADDITIVE"
    )
    assert _info(editor, child.package)["blend_mode"] == "BLEND_ADDITIVE"


def test_only_overrides_of_the_materials_own_parameters_count(
    editor: SimpleNamespace,
) -> None:
    """An override for a name that is no longer a parameter, or of a layer
    parameter, is not an override of the material's parameter by that name."""
    editor.fake.instance.overrides = {
        ("scalar", "RefractionDepthBias", "GLOBAL_PARAMETER", -1): 0.0,
        ("scalar", "Roughness", "LAYER_PARAMETER", 0): 0.1,
    }

    result = _info(editor, INSTANCE)

    assert result["parameters"] == [
        _param("Roughness", "scalar", _f32(0.6407), False),
        _param("Color", "vector", GREY, False),
    ]


def test_an_instance_without_a_parent_renders_opaque(editor: SimpleNamespace) -> None:
    """UMaterialInstance::UpdateOverridableBaseProperties sets BLEND_Opaque for
    an instance without a parent before it looks at any override; UE 5.7.4's
    own get_blend_mode() answered BLEND_OPAQUE for one with and without a
    blend override, while the tool reported null."""
    orphan = MaterialInstanceConstant(
        "/Game/Materials/MI_Orphan", None, blend_mode="BLEND_TRANSLUCENT"
    )
    editor.fake.assets[orphan.package] = orphan

    result = _info(editor, orphan.package)

    assert result == {
        "material_path": _object_path(orphan.package),
        "class_name": "MaterialInstanceConstant",
        "parent_path": None,
        "base_material_path": _object_path(WORLD_GRID.package),
        "material_domain": "MD_SURFACE",
        "blend_mode": "BLEND_OPAQUE",
        "parameters": [],
    }


LAYERED = "/Game/LayerTest/M_Layered"
LAYERED_INSTANCE = "/Game/LayerTest/MI_Layered"
LAYERED_CHILD = "/Game/LayerTest/MI_Layered_Child"
RED, GREEN, BLUE = [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]


def _add_layered(editor: SimpleNamespace) -> None:
    """The material built live on UE 5.7.4: two Material Layers and a blend,
    a name the material shares with layer 0 and one both layers have; an
    instance overriding a global parameter, both layers and the blend; and a
    child of that instance with an override of its own."""
    texture = editor.fake.assets[DEFAULT_TEXTURE]
    material = Material(
        LAYERED,
        scalar={"GlobalRough": 0.4, "Shared": 0.1},
        vector={"GlobalTint": [1.0] * 4},
        layered=[
            ("scalar", "LayerRough", "LAYER_PARAMETER", 0, 0.25),
            ("scalar", "Shared", "LAYER_PARAMETER", 0, 0.7),
            ("vector", "LayerTint", "LAYER_PARAMETER", 0, RED),
            ("texture", "LayerTex", "LAYER_PARAMETER", 0, texture),
            ("scalar", "TopOnly", "LAYER_PARAMETER", 1, 0.5),
            ("scalar", "LayerRough", "LAYER_PARAMETER", 1, 0.75),
            ("vector", "LayerTint", "LAYER_PARAMETER", 1, BLUE),
            ("scalar", "BlendAlpha", "BLEND_PARAMETER", 0, 0.5),
        ],
    )
    instance = MaterialInstanceConstant(
        LAYERED_INSTANCE,
        material,
        scalar={"GlobalRough": 0.9},
        layered=[
            ("scalar", "LayerRough", "LAYER_PARAMETER", 0, 0.05),
            ("scalar", "TopOnly", "LAYER_PARAMETER", 1, 0.2),
            ("scalar", "BlendAlpha", "BLEND_PARAMETER", 0, 0.8),
            ("vector", "LayerTint", "LAYER_PARAMETER", 1, GREEN),
        ],
    )
    child = MaterialInstanceConstant(LAYERED_CHILD, instance, scalar={"Shared": 0.6})
    for asset in (material, instance, child):
        editor.fake.assets[asset.package] = asset


def test_layer_and_blend_parameters_are_listed_with_their_layer_and_its_value(
    editor: SimpleNamespace,
) -> None:
    """The live regression: TopOnly, BlendAlpha and LayerTex, which only a
    layer or the blend has, read 0 and no texture, and LayerRough and
    LayerTint, which both layers have, were each listed once, with 0 and
    black, and a warning."""
    _add_layered(editor)

    result = _info(editor, LAYERED)

    assert result["parameters"] == [
        # By kind: the material's own, then layer 0's, layer 1's and the blend's, each by name.
        _param("GlobalRough", "scalar", _f32(0.4), None),
        _param("Shared", "scalar", _f32(0.1), None),
        _param("LayerRough", "scalar", 0.25, None, "layer", 0),
        _param("Shared", "scalar", _f32(0.7), None, "layer", 0),
        _param("LayerRough", "scalar", 0.75, None, "layer", 1),
        _param("TopOnly", "scalar", 0.5, None, "layer", 1),
        _param("BlendAlpha", "scalar", 0.5, None, "blend", 0),
        _param("GlobalTint", "vector", [1.0] * 4, None),
        _param("LayerTint", "vector", RED, None, "layer", 0),
        _param("LayerTint", "vector", BLUE, None, "layer", 1),
        _param("LayerTex", "texture", _object_path(DEFAULT_TEXTURE), None, "layer", 0),
    ]
    assert "warning" not in result


def test_an_instance_reports_each_layers_value_and_override(
    editor: SimpleNamespace,
) -> None:
    """The live regression: the instance's overrides of layer 1 and of the
    blend read 0 with overridden false."""
    _add_layered(editor)

    instance, child = _info(editor, LAYERED_INSTANCE), _info(editor, LAYERED_CHILD)

    assert instance["parameters"] == [
        _param("GlobalRough", "scalar", _f32(0.9), True),
        _param("Shared", "scalar", _f32(0.1), False),
        _param("LayerRough", "scalar", _f32(0.05), True, "layer", 0),
        _param("Shared", "scalar", _f32(0.7), False, "layer", 0),
        _param("LayerRough", "scalar", 0.75, False, "layer", 1),
        _param("TopOnly", "scalar", _f32(0.2), True, "layer", 1),
        _param("BlendAlpha", "scalar", _f32(0.8), True, "blend", 0),
        _param("GlobalTint", "vector", [1.0] * 4, False),
        _param("LayerTint", "vector", RED, False, "layer", 0),
        _param("LayerTint", "vector", GREEN, True, "layer", 1),
        _param("LayerTex", "texture", _object_path(DEFAULT_TEXTURE), False, "layer", 0),
    ]
    # The child renders with its parent's layer overrides and overrides only its own Shared.
    assert child["parameters"][:7] == [
        _param("GlobalRough", "scalar", _f32(0.9), False),
        _param("Shared", "scalar", _f32(0.6), True),
        _param("LayerRough", "scalar", _f32(0.05), False, "layer", 0),
        _param("Shared", "scalar", _f32(0.7), False, "layer", 0),
        _param("LayerRough", "scalar", 0.75, False, "layer", 1),
        _param("TopOnly", "scalar", _f32(0.2), False, "layer", 1),
        _param("BlendAlpha", "scalar", _f32(0.8), False, "blend", 0),
    ]


@pytest.mark.parametrize(
    ("path", "error_type", "error"),
    [
        (
            "/Game/Materials/M_Missing",
            "NotFoundError",
            "No material at '/Game/Materials/M_Missing': pass a Material or Material Instance "
            "asset path such as /Engine/BasicShapes/BasicShapeMaterial",
        ),
        (
            "/Engine/BasicShapes/Cube",
            "InvalidAssetType",
            "/Engine/BasicShapes/Cube is a StaticMesh, not a Material or Material Instance",
        ),
    ],
)
def test_the_path_must_name_a_material(
    editor: SimpleNamespace, path: str, error_type: str, error: str
) -> None:
    result = _info(editor, path)

    assert result == {"success": False, "error_type": error_type, "error": error}


def test_play_in_editor_changes_nothing(editor: SimpleNamespace) -> None:
    """EditorAssetLibrary loads nothing during PIE; the tool does not depend on it."""
    outside = [_info(editor, path) for path in (INSTANCE, PARENT, WIDGET_INSTANCE)]
    editor.fake.pie = True
    during = [_info(editor, path) for path in (INSTANCE, PARENT, WIDGET_INSTANCE)]

    assert during == outside


def test_the_material_path_is_data_not_python() -> None:
    session = UnrealRuntimeSession(Settings())
    session._execute_python = AsyncMock(
        return_value={
            "ReturnValue": True,
            "LogOutput": [{"Type": "Info", "Output": "{}"}],
        }
    )
    malicious = "'; review_injected(); #\\\n"

    asyncio.run(session.get_material_info(malicious))

    tree = ast.parse(session._execute_python.call_args.args[0])
    assert not any(
        isinstance(n, ast.Name) and n.id == "review_injected" for n in ast.walk(tree)
    )


# ---------------------------------------------------------------------------
# MCP: what an agent receives
# ---------------------------------------------------------------------------


class _EditorAdapter(AvailableAdapter):
    def __init__(self, session: UnrealRuntimeSession) -> None:
        super().__init__(Settings())
        self.session = session

    @contextmanager
    def create_session(self) -> Iterator[UnrealRuntimeSession]:
        yield self.session


def _call_tool(
    editor: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, path: str
) -> Dict[str, Any]:
    base = Settings()
    settings = base.model_copy(
        update={"unreal": base.unreal.model_copy(update={"tool_surface": "full"})}
    )
    instance = server_module.SimulMCPServer(settings, backends={"unreal"})
    monkeypatch.setattr(instance, "unreal_adapter", _EditorAdapter(editor.session))

    async def run() -> Any:
        async with Client(instance.mcp) as client:
            result = await client.call_tool(
                "get_unreal_material_info",
                {"material_path": path},
                raise_on_error=False,
            )
        return json.loads(result.content[-1].text)

    return asyncio.run(run())


def test_the_tool_returns_parameters_overrides_and_metadata(
    editor: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The response schema carries every field; undeclared ones would be dropped."""
    payload = _call_tool(editor, monkeypatch, WIDGET_INSTANCE)

    assert payload == {
        "success": True,
        "error": None,
        "material_path": _object_path(WIDGET_INSTANCE),
        "class_name": "MaterialInstanceConstant",
        "parent_path": _object_path(WIDGET),
        "base_material_path": _object_path(WIDGET),
        "material_domain": "MD_SURFACE",
        "blend_mode": "BLEND_MASKED",
        "parameters": [
            _param("OpacityFromTexture", "scalar", 1.0, False),
            _param("BackColor", "vector", [0.0] * 4, False),
            _param("TintColorAndOpacity", "vector", [1.0] * 4, False),
            _param("Mask", "texture", None, False),
            _param("SlateUI", "texture", _object_path(DEFAULT_TEXTURE), True),
        ],
    }


def test_the_tool_returns_each_parameters_layer(
    editor: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The response schema declares association and index; undeclared fields would be dropped."""
    _add_layered(editor)

    payload = _call_tool(editor, monkeypatch, LAYERED_INSTANCE)

    assert payload["success"] is True
    assert "warning" not in payload
    assert [
        (p["name"], p["association"], p["index"], p["value"], p["overridden"])
        for p in payload["parameters"]
        if p["param_type"] == "scalar"
    ] == [
        ("GlobalRough", "global", None, _f32(0.9), True),
        ("Shared", "global", None, _f32(0.1), False),
        ("LayerRough", "layer", 0, _f32(0.05), True),
        ("Shared", "layer", 0, _f32(0.7), False),
        ("LayerRough", "layer", 1, 0.75, False),
        ("TopOnly", "layer", 1, _f32(0.2), True),
        ("BlendAlpha", "blend", 0, _f32(0.8), True),
    ]


def test_the_tool_reports_a_path_that_is_not_a_material(
    editor: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _call_tool(editor, monkeypatch, "/Engine/BasicShapes/Cube")

    assert payload["success"] is False
    assert payload["error_type"] == "InvalidAssetType"
