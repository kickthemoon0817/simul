"""set_unreal_material_params writes the instance and reports only what Unreal applied.

The tool used to send each parameter as a Remote Control write of the
instance's ``ScalarParameterValues`` / ``VectorParameterValues`` /
``TextureParameterValues`` array and count every request as set. Observed on
UE 5.7.4, in and out of Play In Editor: Remote Control answered OK, the tool
reported ``params_set: 1``, and the instance still had no override, Roughness
reading its 0.6407 default after two calls. It now writes through
MaterialEditingLibrary, the calls the Material Instance editor makes, and
reports the values it reads back. As set_unreal_light_params does, it fails
whenever a value does not apply, so ``success`` alone can be trusted: a
request naming a parameter the material lacks or a texture that does not load
is refused before anything is written, and a value that reads back different
fails the call after the write, keeping what did apply.

The fake editor below answers as UE 5.7.4 did: the MaterialEditingLibrary
setters return False whether or not they applied the value and add an
override for any name, a parameter of the material or not; the instance
getters take only a MaterialInstanceConstant; ``unreal.load_asset`` loads in
and out of PIE while ``EditorAssetLibrary`` loads nothing during PIE.
"""

from __future__ import annotations

import ast
import asyncio
import io
import json
import math
import struct
import sys
from contextlib import contextmanager, redirect_stdout
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional
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
WIDGET_INSTANCE = "/Game/Materials/MI_Widget"
TEXTURE = "/Engine/EngineResources/DefaultTexture"
#: Why a scalar named Bogus is not set on an instance of the basic shape material.
NO_BOGUS = "the material has no scalar parameter named 'Bogus' (its scalar parameters: Roughness)"
#: What the fake instance holds before anything is written to it.
NO_OVERRIDES = {"scalar": {}, "vector": {}, "texture": {}}


def _f32(value: float) -> float:
    """``value`` as the 32-bit float Unreal stores it."""
    return struct.unpack("f", struct.pack("f", value))[0]


# ---------------------------------------------------------------------------
# A fake editor with UE 5.7.4's material answers
# ---------------------------------------------------------------------------


class _Object:
    def __init__(self, package: str) -> None:
        self.package = package

    def get_path_name(self) -> str:
        return self.package + "." + self.package.rsplit("/", 1)[-1]

    def get_class(self) -> SimpleNamespace:
        return SimpleNamespace(get_name=lambda: type(self).__name__)


class Texture(_Object):
    pass


class Texture2D(Texture):
    pass


class StaticMesh(_Object):
    pass


class MaterialInterface(_Object):
    pass


class Material(MaterialInterface):
    def __init__(self, package: str, **parameters: Dict[str, Any]) -> None:
        super().__init__(package)
        self.parameters = {
            kind: dict(parameters.get(kind, {}))
            for kind in ("scalar", "vector", "texture")
        }

    def names(self, kind: str) -> List[str]:
        return list(self.parameters[kind])

    def value(self, kind: str, name: str) -> Any:
        return self.parameters[kind].get(name)


class MaterialInstance(MaterialInterface):
    pass


class MaterialInstanceConstant(MaterialInstance):
    def __init__(
        self,
        package: str,
        parent: MaterialInterface,
        events: Optional[List[str]] = None,
    ) -> None:
        super().__init__(package)
        self.parent = parent
        # What is done to the instance, in order, in the editor's log of
        # transactions: whether the writes fall inside the undoable one.
        self.events: List[str] = [] if events is None else events
        self.overrides: Dict[str, Dict[str, Any]] = {
            "scalar": {},
            "vector": {},
            "texture": {},
        }
        # Parameters whose writes Unreal does not take.
        self.refused: set = set()
        self.modified = 0
        self.updates = 0

    def modify(self) -> bool:
        self.modified += 1
        self.events.append("modify")
        return True

    def names(self, kind: str) -> List[str]:
        return self.parent.names(kind)

    def value(self, kind: str, name: str) -> Any:
        if name in self.overrides[kind]:
            return self.overrides[kind][name]
        return self.parent.value(kind, name)


class LinearColor:
    def __init__(
        self, r: float = 0.0, g: float = 0.0, b: float = 0.0, a: float = 0.0
    ) -> None:
        self.r, self.g, self.b, self.a = (_f32(channel) for channel in (r, g, b, a))


class _Name(str):
    """Unreal hands parameter names back as ``unreal.Name``."""


def _instance_only(instance: Any) -> None:
    # UE 5.7.4: "Cannot nativize 'Material' as 'Instance' (allowed Class type: 'MaterialInstanceConstant')".
    if not isinstance(instance, MaterialInstanceConstant):
        raise TypeError(
            "Cannot nativize '" + type(instance).__name__ + "' as 'Instance'"
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

    # The setters return False whether or not they applied the value, and
    # store an override under any name they are given.
    @staticmethod
    def set_material_instance_scalar_parameter_value(
        instance: Any, name: str, value: float
    ) -> bool:
        _instance_only(instance)
        instance.events.append("set " + name)
        if name not in instance.refused:
            instance.overrides["scalar"][name] = _f32(value)
        return False

    @staticmethod
    def set_material_instance_vector_parameter_value(
        instance: Any, name: str, value: LinearColor
    ) -> bool:
        _instance_only(instance)
        instance.events.append("set " + name)
        if name not in instance.refused:
            instance.overrides["vector"][name] = [value.r, value.g, value.b, value.a]
        return False

    @staticmethod
    def set_material_instance_texture_parameter_value(
        instance: Any, name: str, value: Texture
    ) -> bool:
        _instance_only(instance)
        instance.events.append("set " + name)
        if name not in instance.refused:
            instance.overrides["texture"][name] = value
        return False

    @staticmethod
    def get_material_instance_scalar_parameter_value(instance: Any, name: str) -> float:
        _instance_only(instance)
        value = instance.value("scalar", name)
        return 0.0 if value is None else value

    @staticmethod
    def get_material_instance_vector_parameter_value(
        instance: Any, name: str
    ) -> LinearColor:
        _instance_only(instance)
        value = instance.value("vector", name)
        return LinearColor(*(value or [0.0, 0.0, 0.0, 0.0]))

    @staticmethod
    def get_material_instance_texture_parameter_value(
        instance: Any, name: str
    ) -> Optional[Texture]:
        _instance_only(instance)
        return instance.value("texture", name)

    @staticmethod
    def update_material_instance(instance: Any) -> None:
        _instance_only(instance)
        instance.events.append("update")
        instance.updates += 1


class _Editor:
    """The assets, the transactions opened and whether PIE runs."""

    def __init__(self) -> None:
        texture = Texture2D(TEXTURE)
        white = Texture2D("/Engine/EngineResources/WhiteSquareTexture")
        parent = Material(
            PARENT,
            scalar={"Roughness": _f32(0.6407)},
            vector={"Color": [0.9, 0.9, 0.9, 1.0]},
        )
        widget = Material(
            WIDGET,
            scalar={"OpacityFromTexture": 1.0},
            vector={"BackColor": [0.0] * 4, "TintColorAndOpacity": [1.0] * 4},
            texture={"SlateUI": white},
        )
        self.events: List[str] = []
        self.instance = MaterialInstanceConstant(INSTANCE, parent, self.events)
        self.widget_instance = MaterialInstanceConstant(
            WIDGET_INSTANCE, widget, self.events
        )
        self.assets = {
            asset.package: asset
            for asset in (
                texture,
                white,
                parent,
                widget,
                self.instance,
                self.widget_instance,
                StaticMesh("/Engine/BasicShapes/Cube"),
            )
        }
        self.transactions: List[str] = []
        self.pie = False

    def load_asset(self, path: str) -> Any:
        package, _, name = path.partition(".")
        asset = self.assets.get(package)
        if asset is None or (name and asset.get_path_name() != path):
            return None
        return asset

    def module(self) -> SimpleNamespace:
        @contextmanager
        def scoped_transaction(description: str) -> Iterator[None]:
            self.transactions.append(description)
            self.events.append("begin " + description)
            try:
                yield
            finally:
                self.events.append("end")

        def editor_asset_library_load(path: str) -> Any:
            # "The Editor is currently in a play mode": nothing loads during PIE.
            return None if self.pie else self.load_asset(path)

        return SimpleNamespace(
            Material=Material,
            MaterialInterface=MaterialInterface,
            MaterialInstance=MaterialInstance,
            MaterialInstanceConstant=MaterialInstanceConstant,
            MaterialEditingLibrary=MaterialEditingLibrary,
            Texture=Texture,
            LinearColor=LinearColor,
            ScopedEditorTransaction=scoped_transaction,
            load_asset=self.load_asset,
            EditorAssetLibrary=SimpleNamespace(load_asset=editor_asset_library_load),
        )


@pytest.fixture
def editor(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A session whose scripts run in-process against the fake editor.

    Remote Control property writes answer OK and change nothing, as UE 5.7.4
    answered the instance's parameter arrays written that way.
    """
    fake = _Editor()
    monkeypatch.setitem(sys.modules, "unreal", fake.module())
    session = UnrealRuntimeSession(Settings())
    http_put = AsyncMock(return_value={})
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


def _set(
    editor: SimpleNamespace, path: str = INSTANCE, **params: Any
) -> Dict[str, Any]:
    return asyncio.run(editor.session.set_material_params(path, **params))


# ---------------------------------------------------------------------------
# Adapter: the values are written, read back and reported as read
# ---------------------------------------------------------------------------


def test_a_scalar_is_written_and_reported_as_read_back(editor: SimpleNamespace) -> None:
    """The live regression: params_set 1 while the instance kept no override."""
    first = _set(editor, scalar_params={"Roughness": 0.25})
    second = _set(editor, scalar_params={"Roughness": 0.25})

    assert editor.fake.instance.overrides["scalar"] == {"Roughness": 0.25}
    expected = {
        "material_path": INSTANCE + ".MI_Chair",
        "params_set": 1,
        "applied": {"scalar": {"Roughness": 0.25}},
        "not_applied": {},
    }
    assert first == expected
    assert second == expected  # applied again, not an error
    assert len(editor.scripts) == 2  # write and read-back in one round trip each
    editor.http_put.assert_not_called()


def test_the_writes_are_one_undoable_transaction_followed_by_an_update(
    editor: SimpleNamespace,
) -> None:
    result = _set(
        editor,
        scalar_params={"Roughness": 0.5},
        vector_params={"Color": [1.0, 0.0, 0.0, 1.0]},
    )

    assert result["params_set"] == 2
    assert editor.fake.transactions == ["simul: set material params"]
    assert editor.fake.instance.modified == 1
    assert editor.fake.instance.updates == 1
    # Both writes inside the transaction, so one undo reverts them.
    assert editor.fake.events == [
        "begin simul: set material params",
        "modify",
        "set Roughness",
        "set Color",
        "end",
        "update",
    ]


def test_colors_take_rgb_or_rgba_and_read_back_as_unreal_stores_them(
    editor: SimpleNamespace,
) -> None:
    result = _set(editor, vector_params={"Color": [0.1, 0.2, 0.3]})

    stored = [_f32(0.1), _f32(0.2), _f32(0.3), 1.0]
    assert editor.fake.instance.overrides["vector"] == {"Color": stored}
    assert result["applied"] == {"vector": {"Color": stored}}

    result = _set(editor, vector_params={"Color": [0.5, 0.25, 0.0, 0.5]})
    assert result["applied"] == {"vector": {"Color": [0.5, 0.25, 0.0, 0.5]}}


def test_a_texture_is_set_by_asset_path(editor: SimpleNamespace) -> None:
    result = _set(editor, WIDGET_INSTANCE, texture_params={"SlateUI": TEXTURE})

    assert (
        editor.fake.widget_instance.overrides["texture"]["SlateUI"].package == TEXTURE
    )
    assert result["applied"] == {"texture": {"SlateUI": TEXTURE + ".DefaultTexture"}}
    assert result["params_set"] == 1


def test_a_name_the_material_lacks_refuses_the_request_and_nothing_is_written(
    editor: SimpleNamespace,
) -> None:
    """Unreal's setter would store an override named Bogus; the name is checked
    first, and the valid Color in the same request is not written either."""
    result = _set(
        editor,
        scalar_params={"Bogus": 1.0},
        vector_params={"Color": [0.0, 0.0, 1.0, 1.0]},
    )

    assert result == {
        "success": False,
        "error_type": "ValidationError",
        "error": "Nothing was written to /Game/Materials/MI_Chair.MI_Chair: Bogus (scalar): "
        + NO_BOGUS,
        "material_path": INSTANCE + ".MI_Chair",
        "params_set": 0,
        "applied": {},
        "not_applied": {"scalar": {"Bogus": NO_BOGUS}},
    }
    assert editor.fake.instance.overrides == NO_OVERRIDES
    assert editor.fake.transactions == []
    assert editor.fake.instance.updates == 0


def test_every_refused_parameter_is_named(editor: SimpleNamespace) -> None:
    result = _set(
        editor, scalar_params={"Bogus": 1.0}, texture_params={"SlateUI": TEXTURE}
    )

    assert result["success"] is False
    assert result["error_type"] == "ValidationError"
    assert result["params_set"] == 0
    assert result["applied"] == {}
    assert result["not_applied"] == {
        "scalar": {"Bogus": NO_BOGUS},
        "texture": {"SlateUI": "the material has no texture parameters"},
    }
    assert result["error"] == (
        "Nothing was written to /Game/Materials/MI_Chair.MI_Chair: "
        "Bogus (scalar): the material has no scalar parameter named 'Bogus' (its scalar parameters: Roughness); "
        "SlateUI (texture): the material has no texture parameters"
    )
    assert editor.fake.transactions == []
    assert editor.fake.instance.updates == 0


def test_a_value_unreal_does_not_take_fails_the_call(editor: SimpleNamespace) -> None:
    editor.fake.instance.refused.add("Roughness")

    result = _set(editor, scalar_params={"Roughness": 0.25})

    roughness = f"Unreal read back {json.dumps(_f32(0.6407))} after the write"
    assert result == {
        "success": False,
        "error_type": "MaterialParamsNotApplied",
        "error": "Unreal did not apply every parameter to /Game/Materials/MI_Chair.MI_Chair: "
        "Roughness (scalar): " + roughness,
        "material_path": INSTANCE + ".MI_Chair",
        "params_set": 0,
        "applied": {},
        "not_applied": {"scalar": {"Roughness": roughness}},
    }
    assert editor.fake.transactions == ["simul: set material params"]


def test_a_value_that_reads_back_different_fails_the_call_and_the_rest_stays_set(
    editor: SimpleNamespace,
) -> None:
    """success alone tells whether every value applied; nothing is rolled
    back, as set_unreal_light_params rolls nothing back."""
    editor.fake.instance.refused.add("Roughness")

    result = _set(
        editor,
        scalar_params={"Roughness": 0.25},
        vector_params={"Color": [0.0, 0.0, 1.0, 1.0]},
    )

    roughness = f"Unreal read back {json.dumps(_f32(0.6407))} after the write"
    assert result == {
        "success": False,
        "error_type": "MaterialParamsNotApplied",
        "error": "Unreal did not apply every parameter to /Game/Materials/MI_Chair.MI_Chair: "
        "Roughness (scalar): "
        + roughness
        + ". The parameters in applied were set and stay set",
        "material_path": INSTANCE + ".MI_Chair",
        "params_set": 1,
        "applied": {"vector": {"Color": [0.0, 0.0, 1.0, 1.0]}},
        "not_applied": {"scalar": {"Roughness": roughness}},
    }
    assert editor.fake.instance.overrides["vector"] == {"Color": [0.0, 0.0, 1.0, 1.0]}


def test_names_match_ignoring_case_and_report_the_materials_spelling(
    editor: SimpleNamespace,
) -> None:
    result = _set(editor, scalar_params={"roughness": 0.75})

    assert editor.fake.instance.overrides["scalar"] == {"Roughness": 0.75}
    assert result["applied"] == {"scalar": {"Roughness": 0.75}}


@pytest.mark.parametrize(
    "path", ["/Game/Textures/T_Missing", "/Engine/BasicShapes/Cube", PARENT]
)
def test_a_texture_that_does_not_load_refuses_the_request(
    editor: SimpleNamespace, path: str
) -> None:
    result = _set(
        editor,
        WIDGET_INSTANCE,
        texture_params={"SlateUI": path},
        scalar_params={"OpacityFromTexture": 0.5},
    )

    assert result["success"] is False
    assert result["error_type"] == "ValidationError"
    assert result["applied"] == {}
    assert result["not_applied"] == {"texture": {"SlateUI": f"no texture at {path!r}"}}
    # The valid scalar in the same request is not written either.
    assert editor.fake.widget_instance.overrides == NO_OVERRIDES


@pytest.mark.parametrize(
    ("path", "error_type", "error"),
    [
        (
            "/Game/Materials/MI_Missing",
            "NotFoundError",
            "No material instance at '/Game/Materials/MI_Missing': pass a Material Instance Constant asset "
            "path such as /Game/Materials/MI_Wood",
        ),
        (
            PARENT,
            "InvalidAssetType",
            f"{PARENT} is a Material, not a Material Instance Constant; create an instance of it with "
            "create_unreal_material_instance and set that",
        ),
        (
            "/Engine/BasicShapes/Cube",
            "InvalidAssetType",
            "/Engine/BasicShapes/Cube is a StaticMesh, not a Material Instance Constant",
        ),
    ],
)
def test_the_path_must_be_a_material_instance_constant(
    editor: SimpleNamespace, path: str, error_type: str, error: str
) -> None:
    result = _set(editor, path, scalar_params={"Roughness": 0.25})

    assert result == {"success": False, "error_type": error_type, "error": error}


def test_play_in_editor_changes_nothing(editor: SimpleNamespace) -> None:
    """EditorAssetLibrary loads nothing during PIE; the tool does not depend on it."""
    outside = _set(
        editor,
        scalar_params={"Roughness": 0.25},
        vector_params={"Color": [0.1, 0.2, 0.3, 1.0]},
    )
    editor.fake.pie = True
    during = _set(
        editor,
        scalar_params={"Roughness": 0.25},
        vector_params={"Color": [0.1, 0.2, 0.3, 1.0]},
    )

    assert during == outside
    assert during["params_set"] == 2


def test_a_refused_request_writes_nothing_during_play_in_editor(
    editor: SimpleNamespace,
) -> None:
    editor.fake.pie = True

    result = _set(editor, scalar_params={"Roughness": 0.25, "Bogus": 1.0})

    assert result["error_type"] == "ValidationError"
    assert result["not_applied"] == {"scalar": {"Bogus": NO_BOGUS}}
    assert editor.fake.instance.overrides == NO_OVERRIDES


@pytest.mark.parametrize(
    ("params", "problem"),
    [
        (
            {"scalar_params": {"Roughness": "0.5"}},
            "scalar parameter 'Roughness' must be a finite number",
        ),
        (
            {"scalar_params": {"Roughness": True}},
            "scalar parameter 'Roughness' must be a finite number",
        ),
        (
            {"scalar_params": {"Roughness": math.nan}},
            "scalar parameter 'Roughness' must be a finite number",
        ),
        (
            {"scalar_params": {"Roughness": math.inf}},
            "scalar parameter 'Roughness' must be a finite number",
        ),
        ({"scalar_params": {"Roughness": 1e39}}, "within the 32-bit float range"),
        # A JSON integer has no bound; one too large for a float is refused
        # like any other value out of range, not raised as OverflowError.
        ({"scalar_params": {"Roughness": 10**400}}, "within the 32-bit float range"),
        (
            {"vector_params": {"Color": [1.0, 0.0, 10**400]}},
            "vector parameter 'Color' must be",
        ),
        # Unreal's names ignore case: both would write Roughness, and the
        # first value would come back as not taken.
        (
            {"scalar_params": {"Roughness": 0.1, "roughness": 0.9}},
            "scalar parameter 'roughness' is the same parameter as 'Roughness'",
        ),
        ({"scalar_params": [0.5]}, "scalar parameters must be a JSON object"),
        ({"scalar_params": {"": 0.5}}, "scalar parameter names must be non-empty"),
        (
            {"vector_params": {"Color": [1.0, 0.0]}},
            "vector parameter 'Color' must be [r, g, b] or [r, g, b, a]",
        ),
        (
            {"vector_params": {"Color": [1.0, 0.0, 0.0, 1.0, 0.0]}},
            "vector parameter 'Color' must be",
        ),
        (
            {"vector_params": {"Color": ["1", 0.0, 0.0]}},
            "vector parameter 'Color' must be",
        ),
        ({"vector_params": {"Color": "red"}}, "vector parameter 'Color' must be"),
        (
            {"texture_params": {"SlateUI": ""}},
            "texture parameter 'SlateUI' must be a texture asset path",
        ),
        (
            {"texture_params": {"SlateUI": 5}},
            "texture parameter 'SlateUI' must be a texture asset path",
        ),
        ({}, "No parameters given"),
        (
            {"scalar_params": {}, "vector_params": {}, "texture_params": {}},
            "No parameters given",
        ),
    ],
)
def test_a_malformed_request_is_refused_before_the_editor_is_called(
    params: Dict[str, Any], problem: str
) -> None:
    session = UnrealRuntimeSession(Settings())
    session._execute_python = AsyncMock()

    result = asyncio.run(session.set_material_params(INSTANCE, **params))

    assert result["success"] is False
    assert result["error_type"] == "ValidationError"
    assert problem in result["error"]
    session._execute_python.assert_not_called()


def test_a_refused_entry_is_named_in_not_applied_before_the_editor_is_called() -> None:
    session = UnrealRuntimeSession(Settings())
    session._execute_python = AsyncMock()

    result = asyncio.run(
        session.set_material_params(
            INSTANCE,
            scalar_params={"Roughness": 0.1, "roughness": 0.9, "Metallic": "1"},
            vector_params={"Color": [1.0]},
        )
    )

    assert result["success"] is False
    assert result["error_type"] == "ValidationError"
    assert (result["material_path"], result["params_set"], result["applied"]) == (
        INSTANCE,
        0,
        {},
    )
    assert result["not_applied"] == {
        "scalar": {
            "roughness": "is the same parameter as 'Roughness' (Unreal's parameter names ignore case): "
            "give it once",
            "Metallic": "must be a finite number within the 32-bit float range, got '1'",
        },
        "vector": {
            "Color": "must be [r, g, b] or [r, g, b, a] of finite numbers within the 32-bit float range, "
            "got [1.0]"
        },
    }
    session._execute_python.assert_not_called()


def test_a_bad_entry_refuses_the_whole_request(editor: SimpleNamespace) -> None:
    result = _set(
        editor, scalar_params={"Roughness": 0.25}, vector_params={"Color": [1.0]}
    )

    assert result["error_type"] == "ValidationError"
    assert editor.scripts == []
    assert editor.fake.instance.overrides["scalar"] == {}


def test_the_material_path_and_names_are_data_not_python() -> None:
    session = UnrealRuntimeSession(Settings())
    session._execute_python = AsyncMock(
        return_value={
            "ReturnValue": True,
            "LogOutput": [{"Type": "Info", "Output": "{}"}],
        }
    )
    malicious = "'; review_injected(); #\\\n"

    asyncio.run(session.set_material_params(malicious, scalar_params={malicious: 1.0}))

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
    editor: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, arguments: Dict[str, Any]
) -> Any:
    base = Settings()
    settings = base.model_copy(
        update={"unreal": base.unreal.model_copy(update={"tool_surface": "full"})}
    )
    instance = server_module.SimulMCPServer(settings, backends={"unreal"})
    monkeypatch.setattr(instance, "unreal_adapter", _EditorAdapter(editor.session))

    async def run() -> Any:
        async with Client(instance.mcp) as client:
            result = await client.call_tool(
                "set_unreal_material_params", arguments, raise_on_error=False
            )
        return json.loads(result.content[-1].text)

    return asyncio.run(run())


def test_the_tool_returns_the_values_read_back(
    editor: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The response schema carries applied and not_applied; undeclared fields would be dropped."""
    payload = _call_tool(
        editor,
        monkeypatch,
        {
            "material_path": INSTANCE,
            "scalar_params_json": '{"Roughness": 0.25}',
            "vector_params_json": '{"Color": [0.5, 0.5, 0.5]}',
        },
    )

    assert payload == {
        "success": True,
        "error": None,
        "material_path": INSTANCE + ".MI_Chair",
        "params_set": 2,
        "applied": {
            "scalar": {"Roughness": 0.25},
            "vector": {"Color": [0.5, 0.5, 0.5, 1.0]},
        },
        "not_applied": {},
    }


def test_the_tool_refuses_a_parameter_the_material_lacks_writing_nothing(
    editor: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _call_tool(
        editor,
        monkeypatch,
        {
            "material_path": INSTANCE,
            "scalar_params_json": '{"Roughness": 0.25, "Bogus": 1}',
        },
    )

    assert payload["success"] is False
    assert payload["error_type"] == "ValidationError"
    assert payload["params_set"] == 0
    assert payload["applied"] == {}
    assert payload["not_applied"] == {"scalar": {"Bogus": NO_BOGUS}}
    assert editor.fake.instance.overrides == NO_OVERRIDES


def test_the_tool_fails_when_a_value_does_not_read_back(
    editor: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The error envelope keeps applied and not_applied, so a caller sees what stayed set."""
    editor.fake.instance.refused.add("Roughness")

    payload = _call_tool(
        editor,
        monkeypatch,
        {
            "material_path": INSTANCE,
            "scalar_params_json": '{"Roughness": 0.25}',
            "vector_params_json": '{"Color": [0.5, 0.5, 0.5]}',
        },
    )

    assert payload["success"] is False
    assert payload["error_type"] == "MaterialParamsNotApplied"
    assert payload["params_set"] == 1
    assert payload["applied"] == {"vector": {"Color": [0.5, 0.5, 0.5, 1.0]}}
    assert list(payload["not_applied"]["scalar"]) == ["Roughness"]
