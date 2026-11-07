"""``set_unreal_light_params`` against a fake editor with UE 5.7.4's light classes.

The audit on UE 5.7.4 found that intensity -5 turned the light to 0 while the
reply said it was not applied (U2-09), that 100 K and 1e9 K temperatures were
stored, a SkyLight was refused with "exactly one LightComponent" (it is a
LightComponentBase, not a LightComponent), the intensity unit and color space
were not stated, and a missing actor with no parameters returned success
(U2-13). The session now checks ranges before writing (tested in
test_unreal_runtime.py); this runs the editor script itself.
"""

from __future__ import annotations

import asyncio
import io
import sys
import traceback
from contextlib import redirect_stdout
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from simul.adapters import unreal_runtime
from simul.config import Settings

LEVEL = "/Game/Maps/Lab.Lab:PersistentLevel."


class Color:
    def __init__(self, r: int = 255, g: int = 255, b: int = 255, a: int = 255) -> None:
        self.r, self.g, self.b, self.a = r, g, b, a


class LightComponentBase:
    PROPERTIES = ("intensity", "light_color", "cast_shadows")

    def __init__(self, owner: "Actor", **values: Any) -> None:
        self.owner = owner
        self.values: Dict[str, Any] = {
            "intensity": 3.14,
            "light_color": Color(),
            "cast_shadows": True,
        }
        self.values.update(values)
        # How Unreal changes a value as it is written, as it turned a negative intensity into 0.
        self.clamps: Dict[str, Any] = {}
        self.writes: List[Any] = []

    def get_editor_property(self, name: str) -> Any:
        if name not in self.values:
            raise Exception(f"{type(self).__name__}: Failed to find property '{name}'")
        return self.values[name]

    def set_editor_property(self, name: str, value: Any) -> None:
        self.writes.append((name, value, self.owner.editor.open))
        self.values[name] = self.clamps[name](value) if name in self.clamps else value

    def get_path_name(self) -> str:
        return self.owner.get_path_name() + ".LightComponent0"

    def get_class(self) -> SimpleNamespace:
        return SimpleNamespace(get_name=lambda: type(self).__name__)


class LightComponent(LightComponentBase):
    def __init__(self, owner: "Actor", **values: Any) -> None:
        super().__init__(owner, temperature=6500.0, use_temperature=False, **values)


class DirectionalLightComponent(LightComponent):
    pass


class LocalLightComponent(LightComponent):
    def __init__(self, owner: "Actor", **values: Any) -> None:
        super().__init__(
            owner,
            attenuation_radius=1000.0,
            intensity_units=SimpleNamespace(name="CANDELAS"),
            **values,
        )


class PointLightComponent(LocalLightComponent):
    pass


class SkyLightComponent(LightComponentBase):
    pass


class Actor:
    def __init__(self, editor: "Editor", name: str, component_class: type) -> None:
        self.editor, self.name = editor, name
        self.root_component = component_class(self)

    def get_path_name(self) -> str:
        return LEVEL + self.name

    def get_components_by_class(self, cls: type) -> List[Any]:
        return [self.root_component] if isinstance(self.root_component, cls) else []

    def get_world(self) -> Any:
        return self.editor.world


class Editor:
    def __init__(self) -> None:
        self.actors: Dict[str, Actor] = {}
        self.open: Optional[str] = None
        self.pie = False
        package = SimpleNamespace(get_name=lambda: "/Game/Maps/Lab")
        self.world = SimpleNamespace(get_outermost=lambda: package, get_path_name=lambda: "/Game/Maps/Lab.Lab")
        self.play_world = SimpleNamespace(get_path_name=lambda: "/Game/Maps/UEDPIE_0_Lab.Lab")

    def light(self, name: str, component_class: type) -> Actor:
        actor = Actor(self, name, component_class)
        self.actors[actor.get_path_name()] = actor
        return actor

    def module(self) -> SimpleNamespace:
        editor = self

        class Transaction:
            def __init__(self, description: str) -> None:
                self.description = description

            def __enter__(self) -> "Transaction":
                editor.open = self.description
                return self

            def __exit__(self, *exc_info: Any) -> None:
                editor.open = None

        return SimpleNamespace(
            Actor=Actor,
            LightComponentBase=LightComponentBase,
            LightComponent=LightComponent,
            DirectionalLightComponent=DirectionalLightComponent,
            SkyLightComponent=SkyLightComponent,
            Color=Color,
            ScopedEditorTransaction=Transaction,
            LevelEditorSubsystem="LevelEditorSubsystem",
            UnrealEditorSubsystem="UnrealEditorSubsystem",
            LayersSubsystem="LayersSubsystem",
            SystemLibrary=SimpleNamespace(is_valid=lambda obj: obj is not None),
            get_editor_subsystem={
                "LevelEditorSubsystem": SimpleNamespace(is_in_play_in_editor=lambda: editor.pie),
                # The editor world, as actor_at finds it in and out of PIE.
                "UnrealEditorSubsystem": SimpleNamespace(
                    get_editor_world=lambda: None if editor.pie else editor.world,
                    get_game_world=lambda: editor.play_world if editor.pie else None,
                ),
                "LayersSubsystem": SimpleNamespace(get_world=lambda: editor.world),
            }.__getitem__,
            load_object=lambda outer, path: editor.actors.get(path),
            find_object=lambda outer, path: editor.actors.get(path),
        )

    def run_python(self, code: str) -> Dict[str, Any]:
        output = io.StringIO()
        try:
            with redirect_stdout(output):
                exec(code, {})
        except Exception:
            return {
                "ReturnValue": False,
                "CommandResult": traceback.format_exc(),
                "LogOutput": [],
            }
        return {
            "ReturnValue": True,
            "LogOutput": [
                {"Type": "Info", "Output": line}
                for line in output.getvalue().splitlines()
            ],
        }


def _session(
    monkeypatch: pytest.MonkeyPatch, editor: Editor
) -> unreal_runtime.UnrealRuntimeSession:
    monkeypatch.setitem(sys.modules, "unreal", editor.module())
    monkeypatch.setattr(unreal_runtime, "UNREAL_AVAILABLE", True)
    monkeypatch.setattr(unreal_runtime, "AIOHTTP_AVAILABLE", True)
    session = unreal_runtime.UnrealRuntimeSession(Settings())

    async def execute(code: str, mode: str = "ExecuteFile") -> Dict[str, Any]:
        return editor.run_python(code)

    monkeypatch.setattr(session, "_execute_python", execute)
    return session


def run(coroutine: Any) -> Dict[str, Any]:
    return asyncio.run(coroutine)


def test_a_point_light_is_written_in_one_transaction_and_reports_its_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = Editor()
    light = editor.light("PointLight_0", PointLightComponent)
    session = _session(monkeypatch, editor)

    result = run(
        session.set_light_params(
            light.get_path_name(), intensity=5000.0, color=[0.2, 0.4, 1.0]
        )
    )

    assert "error" not in result, result
    assert (result["light_class"], result["intensity_unit"], result["params_set"]) == (
        "PointLightComponent",
        "candelas",
        2,
    )
    assert result["applied"] == {
        "intensity": 5000.0,
        "light_color": [51, 102, 255, 255],
    }
    assert result["current"]["attenuation_radius"] == 1000.0
    assert all(
        transaction == "simul: set light params"
        for _, _, transaction in light.root_component.writes
    )


def test_a_directional_light_is_in_lux_and_has_no_attenuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = Editor()
    sun = editor.light("DirectionalLight_0", DirectionalLightComponent)
    session = _session(monkeypatch, editor)

    read = run(session.set_light_params(sun.get_path_name()))
    refused = run(
        session.set_light_params(
            sun.get_path_name(), intensity=9.0, attenuation_radius=500.0
        )
    )

    assert (read["intensity_unit"], read["params_set"], read["applied"]) == (
        "lux",
        0,
        {},
    )
    assert read["current"]["intensity"] == 3.14
    assert refused["error_type"] == "ValidationError"
    assert "DirectionalLightComponent has no attenuation_radius" in refused["error"]
    assert sun.root_component.writes == []  # the valid intensity was not written either


def test_a_sky_light_takes_intensity_color_and_shadows_but_not_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U2-13: a SkyLight was refused with "exactly one LightComponent"."""
    editor = Editor()
    sky = editor.light("SkyLight_0", SkyLightComponent)
    session = _session(monkeypatch, editor)

    result = run(
        session.set_light_params(sky.get_path_name(), intensity=2.0, cast_shadows=False)
    )
    refused = run(session.set_light_params(sky.get_path_name(), temperature=4000.0))

    assert (result["light_class"], result["intensity_unit"], result["params_set"]) == (
        "SkyLightComponent",
        "scale",
        2,
    )
    assert sky.root_component.values["intensity"] == 2.0
    assert (
        refused["error_type"] == "ValidationError" and "temperature" in refused["error"]
    )


def test_a_value_unreal_stores_differently_is_put_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U2-09: the light went from 3.14 to 0 while the reply said it was not applied."""
    editor = Editor()
    light = editor.light("PointLight_0", PointLightComponent)
    light.root_component.clamps["intensity"] = lambda value: min(value, 5.0)
    session = _session(monkeypatch, editor)

    result = run(
        session.set_light_params(
            light.get_path_name(), intensity=7.0, cast_shadows=False
        )
    )

    assert result["success"] is False
    assert result["error_type"] == "LightParamsNotApplied"
    assert (
        "intensity (it stored 5.0)" in result["error"] and "put back" in result["error"]
    )
    assert light.root_component.values["intensity"] == 3.14
    assert result["applied"] == {"cast_shadows": False} and result["params_set"] == 1
    assert result["current"]["intensity"] == 3.14


def test_an_edit_during_pie_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    editor = Editor()
    editor.pie = True
    light = editor.light("PointLight_0", PointLightComponent)
    session = _session(monkeypatch, editor)

    result = run(session.set_light_params(light.get_path_name(), intensity=10.0))

    assert "does not see" in result["warning"]
