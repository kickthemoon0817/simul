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

The fake editor below answers as UE 5.7.4 did: the instance getters return
the value an instance renders with (its override, else its parent's), the
MaterialEditingLibrary default getters take only a Material, an instance can
keep overrides for names that are no longer parameters (the engine's
Widget3DPassThrough_Translucent keeps RefractionDepthBias), and
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
from typing import Any, Dict, Iterator, List, Optional, Tuple
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


class MaterialInterface(Object):
    pass


class Material(MaterialInterface):
    def __init__(
        self,
        package: str,
        domain: str = "MD_SURFACE",
        blend_mode: str = "BLEND_OPAQUE",
        layer_names: Optional[Dict[str, List[str]]] = None,
        **defaults: Dict[str, Any],
    ) -> None:
        super().__init__(package)
        self.defaults = {kind: dict(defaults.get(kind, {})) for kind in KINDS}
        # Names Unreal lists again, once per Material Layer that has them;
        # read by name only, a layer-only name has no global value.
        self.layer_names = {
            kind: list((layer_names or {}).get(kind, [])) for kind in KINDS
        }
        self.properties = {
            "material_domain": _Enum(domain),
            "blend_mode": _Enum(blend_mode),
        }

    def get_editor_property(self, name: str) -> Any:
        return self.properties[name]

    def get_base_material(self) -> "Material":
        return self

    def names(self, kind: str) -> List[str]:
        return list(self.defaults[kind]) + self.layer_names[kind]

    def effective(self, kind: str, name: str) -> Any:
        return self.defaults[kind].get(name)


#: What Unreal renders an instance without a parent with (UE 5.7.4).
WORLD_GRID = Material("/Engine/EngineMaterials/WorldGridMaterial")


class _Entry:
    """A ScalarParameterValue / VectorParameterValue / TextureParameterValue."""

    def __init__(self, name: str, value: Any, association: str) -> None:
        self.parameter_info = SimpleNamespace(
            name=_Name(name), association=_Enum(association), index=-1
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
        parent: MaterialInterface,
        blend_mode: Optional[str] = None,
        **overrides: Dict[str, Any],
    ) -> None:
        super().__init__(package)
        self.parent = parent
        # (kind, name, value, association); a GLOBAL_PARAMETER entry is an override.
        self.entries: List[Tuple[str, str, Any, str]] = [
            (kind, name, value, "GLOBAL_PARAMETER")
            for kind in KINDS
            for name, value in overrides.get(kind, {}).items()
        ]
        self.base_property_overrides = _BasePropertyOverrides(blend_mode)

    def get_editor_property(self, name: str) -> Any:
        if name == "parent":
            return self.parent
        if name == "base_property_overrides":
            return self.base_property_overrides
        kind = name.replace("_parameter_values", "")
        return [
            _Entry(n, v, association)
            for k, n, v, association in self.entries
            if k == kind
        ]

    def get_base_material(self) -> Material:
        # Unreal falls back to the default surface material without a parent.
        return WORLD_GRID if self.parent is None else self.parent.get_base_material()

    def names(self, kind: str) -> List[str]:
        return (
            WORLD_GRID.names(kind) if self.parent is None else self.parent.names(kind)
        )

    def effective(self, kind: str, name: str) -> Any:
        for k, n, value, association in self.entries:
            if (k, n, association) == (kind, name, "GLOBAL_PARAMETER"):
                return value
        return None if self.parent is None else self.parent.effective(kind, name)

    # The K2 getters: the value the instance renders with.
    def get_scalar_parameter_value(self, name: str) -> float:
        value = self.effective("scalar", name)
        return 0.0 if value is None else _f32(value)

    def get_vector_parameter_value(self, name: str) -> LinearColor:
        return LinearColor(*(self.effective("vector", name) or [0.0] * 4))

    def get_texture_parameter_value(self, name: str) -> Optional[Texture]:
        return self.effective("texture", name)


def _material_only(material: Any) -> None:
    # UE 5.7.4 refuses an instance: "allowed Class type: 'Material'".
    if not isinstance(material, Material):
        raise TypeError(
            "Cannot nativize '" + type(material).__name__ + "' as 'Material'"
        )


class MaterialEditingLibrary:
    @staticmethod
    def get_scalar_parameter_names(material: MaterialInterface) -> List[_Name]:
        return [_Name(name) for name in material.names("scalar")]

    @staticmethod
    def get_vector_parameter_names(material: MaterialInterface) -> List[_Name]:
        return [_Name(name) for name in material.names("vector")]

    @staticmethod
    def get_texture_parameter_names(material: MaterialInterface) -> List[_Name]:
        return [_Name(name) for name in material.names("texture")]

    @staticmethod
    def get_material_default_scalar_parameter_value(material: Any, name: str) -> float:
        _material_only(material)
        return _f32(material.effective("scalar", name) or 0.0)

    @staticmethod
    def get_material_default_vector_parameter_value(
        material: Any, name: str
    ) -> LinearColor:
        _material_only(material)
        return LinearColor(*(material.effective("vector", name) or [0.0] * 4))

    @staticmethod
    def get_material_default_texture_parameter_value(
        material: Any, name: str
    ) -> Optional[Texture]:
        _material_only(material)
        return material.effective("texture", name)


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
            MaterialEditingLibrary=MaterialEditingLibrary,
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
    name: str, kind: str, value: Any, overridden: Optional[bool]
) -> Dict[str, Any]:
    return {"name": name, "param_type": kind, "value": value, "overridden": overridden}


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
    editor.fake.instance.entries = [
        ("scalar", "RefractionDepthBias", 0.0, "GLOBAL_PARAMETER"),
        ("scalar", "Roughness", 0.1, "LAYER_PARAMETER"),
    ]

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


def test_a_name_unreal_lists_more_than_once_is_listed_once_with_a_warning(
    editor: SimpleNamespace,
) -> None:
    """Unreal lists a parameter name once per Material Layer that has it, and
    its Python API reads parameters by name only."""
    layered = Material(
        "/Game/Materials/M_Layered",
        scalar={"Roughness": 0.5},
        layer_names={
            "scalar": ["Roughness", "Wetness", "Wetness"],
            "vector": ["Tint", "tint"],
        },
    )
    instance = MaterialInstanceConstant(
        "/Game/Materials/MI_Layered", layered, scalar={"Roughness": 0.25}
    )
    for asset in (layered, instance):
        editor.fake.assets[asset.package] = asset

    material, layered_instance = _info(editor, layered.package), _info(
        editor, instance.package
    )

    assert material["parameters"] == [
        _param("Roughness", "scalar", 0.5, None),
        # Only a layer has it: read by name, it has no global value.
        _param("Wetness", "scalar", 0.0, None),
        _param("Tint", "vector", [0.0] * 4, None),
    ]
    assert layered_instance["parameters"] == [
        _param("Roughness", "scalar", 0.25, True),
        _param("Wetness", "scalar", 0.0, False),
        _param("Tint", "vector", [0.0] * 4, False),
    ]
    warning = (
        "Unreal lists Roughness (scalar), Wetness (scalar), Tint (vector) more than once, from "
        "Material Layers. Its Python API reads parameters by name only, so each is listed once, "
        "with its global value, not a layer's"
    )
    assert material["warning"] == layered_instance["warning"] == warning
    # A material without layers has no warning.
    assert "warning" not in _info(editor, PARENT)


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
        "warning": None,
    }


def test_the_tool_returns_the_warning_for_names_listed_more_than_once(
    editor: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The response schema declares warning; an undeclared field would be dropped."""
    layered = Material(
        "/Game/Materials/M_Layered",
        scalar={"Roughness": 0.5},
        layer_names={"scalar": ["Roughness"]},
    )
    editor.fake.assets[layered.package] = layered

    payload = _call_tool(editor, monkeypatch, layered.package)

    assert payload["success"] is True
    assert [p["name"] for p in payload["parameters"]] == ["Roughness"]
    assert payload["warning"].startswith(
        "Unreal lists Roughness (scalar) more than once"
    )


def test_the_tool_reports_a_path_that_is_not_a_material(
    editor: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _call_tool(editor, monkeypatch, "/Engine/BasicShapes/Cube")

    assert payload["success"] is False
    assert payload["error_type"] == "InvalidAssetType"
