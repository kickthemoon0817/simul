"""set_unreal_actor_transform: optional vectors, and an omitted one keeps its value.

The tool used to take nine scalars defaulting to 0 and 1, so setting only
``location_x`` moved the actor to (x, 0, 0) and reset its rotation and scale
(reproduced live on UE 5.7). It now takes optional ``location``, ``rotation``
and ``scale`` vectors and writes only the ones given.

The adapter tests run the real editor script against a stateful stand-in
for the ``unreal`` module; the MCP tests go through a real FastMCP client so
the advertised schema and its validation are what an agent gets.
"""

from __future__ import annotations

import ast
import asyncio
import io
import json
import math
import sys
from contextlib import contextmanager, redirect_stdout
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List
from unittest.mock import AsyncMock

import pytest
from fastmcp import Client

from simul.adapters.unreal_runtime import (
    UnrealRuntimeSession,
    _transform_component_applied,
)
from simul.config import Settings
from simul.mcp import server as server_module
from tests.fakes import AvailableAdapter

ACTOR = "/Game/Map.Map:PersistentLevel.PointLight_0"


# ---------------------------------------------------------------------------
# A stateful stand-in for the parts of ``unreal`` the transform script uses
# ---------------------------------------------------------------------------


class _Vector:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x, self.y, self.z = x, y, z


class _Rotator:
    # Unreal's positional order is (roll, pitch, yaw); the script must use keywords.
    def __init__(self, roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0) -> None:
        self.roll, self.pitch, self.yaw = roll, pitch, yaw


class _Actor:
    def __init__(self, path: str, has_root: bool = True) -> None:
        self.path = path
        self.root_component = object() if has_root else None
        self.location = _Vector(10.0, 20.0, 30.0)
        self.rotation = _Rotator(pitch=10.0, yaw=20.0, roll=30.0)
        self.scale = _Vector(2.0, 3.0, 4.0)
        self.modified = 0

    def modify(self) -> bool:
        self.modified += 1
        return True

    # Without a root component Unreal has nowhere to store a transform and
    # the setters do nothing.
    def set_actor_location(self, location: _Vector, sweep: bool, teleport: bool) -> None:
        if self.root_component is not None:
            self.location = location

    def set_actor_rotation(self, rotation: _Rotator, teleport: bool) -> None:
        if self.root_component is not None:
            # Unreal reads rotations back normalised to (-180, 180].
            def wrap(angle: float) -> float:
                return angle - 360.0 * math.ceil((angle - 180.0) / 360.0)

            self.rotation = _Rotator(
                roll=wrap(rotation.roll), pitch=wrap(rotation.pitch), yaw=wrap(rotation.yaw)
            )

    def set_actor_scale3d(self, scale: _Vector) -> None:
        if self.root_component is not None:
            self.scale = scale

    # Unreal's getters read the root component; without one they return the
    # identity transform (AActor::GetActorLocation/Rotation/Scale3D), not a
    # transform the actor held before.
    def get_actor_location(self) -> _Vector:
        return self.location if self.root_component is not None else _Vector()

    def get_actor_rotation(self) -> _Rotator:
        return self.rotation if self.root_component is not None else _Rotator()

    def get_actor_scale3d(self) -> _Vector:
        return self.scale if self.root_component is not None else _Vector(1.0, 1.0, 1.0)

    def get_path_name(self) -> str:
        return self.path


def _fake_unreal(actor: _Actor, transactions: List[str]) -> SimpleNamespace:
    @contextmanager
    def scoped_transaction(description: str) -> Iterator[None]:
        transactions.append(description)
        yield

    return SimpleNamespace(
        Actor=_Actor,
        Vector=_Vector,
        Rotator=_Rotator,
        ScopedEditorTransaction=scoped_transaction,
        load_object=lambda outer, path: actor if path == actor.path else None,
    )


@pytest.fixture
def editor(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A session whose scripts run in-process against a fake editor."""
    actor = _Actor(ACTOR)
    transactions: List[str] = []
    monkeypatch.setitem(sys.modules, "unreal", _fake_unreal(actor, transactions))
    session = UnrealRuntimeSession(Settings())
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
            "LogOutput": [{"Type": "Info", "Output": line} for line in out.getvalue().splitlines()],
        }

    monkeypatch.setattr(session, "_execute_python", execute)
    return SimpleNamespace(session=session, actor=actor, transactions=transactions, scripts=scripts)


def _transform(actor: _Actor) -> Dict[str, List[float]]:
    return {
        "location": [actor.location.x, actor.location.y, actor.location.z],
        "rotation": [actor.rotation.pitch, actor.rotation.yaw, actor.rotation.roll],
        "scale": [actor.scale.x, actor.scale.y, actor.scale.z],
    }


# ---------------------------------------------------------------------------
# Adapter: only the given components are written
# ---------------------------------------------------------------------------


def test_location_alone_keeps_rotation_and_scale(editor: SimpleNamespace) -> None:
    """The live regression: location_x alone used to zero Y/Z, rotation and scale."""
    result = asyncio.run(editor.session.set_actor_transform(ACTOR, location=[500.0, 20.0, 30.0]))

    assert _transform(editor.actor) == {
        "location": [500.0, 20.0, 30.0],
        "rotation": [10.0, 20.0, 30.0],
        "scale": [2.0, 3.0, 4.0],
    }
    assert result == {"actor_path": ACTOR, **_transform(editor.actor)}
    assert len(editor.scripts) == 1  # write and read-back in one round trip


@pytest.mark.parametrize(
    ("given", "changed"),
    [
        ({"rotation": [0.0, 90.0, 0.0]}, "rotation"),
        ({"scale": [1.0, 1.0, 1.0]}, "scale"),
    ],
)
def test_each_component_alone_leaves_the_others(
    editor: SimpleNamespace, given: Dict[str, List[float]], changed: str
) -> None:
    before = _transform(editor.actor)

    result = asyncio.run(editor.session.set_actor_transform(ACTOR, **given))

    after = _transform(editor.actor)
    assert after[changed] == given[changed]
    assert {k: v for k, v in after.items() if k != changed} == {
        k: v for k, v in before.items() if k != changed
    }
    assert "error" not in result


def test_all_components_are_written_in_one_undoable_transaction(editor: SimpleNamespace) -> None:
    result = asyncio.run(
        editor.session.set_actor_transform(
            ACTOR, location=(1, 2, 3), rotation=(4, 5, 6), scale=(7, 8, 9)
        )
    )

    assert _transform(editor.actor) == {
        "location": [1.0, 2.0, 3.0],
        "rotation": [4.0, 5.0, 6.0],
        "scale": [7.0, 8.0, 9.0],
    }
    assert result["location"] == [1.0, 2.0, 3.0]
    assert editor.transactions == ["simul: set actor transform"]
    assert editor.actor.modified == 1


def test_nothing_given_reads_the_transform_back_without_writing(editor: SimpleNamespace) -> None:
    result = asyncio.run(editor.session.set_actor_transform(ACTOR))

    assert result == {"actor_path": ACTOR, **_transform(editor.actor)}
    assert editor.transactions == []
    assert editor.actor.modified == 0


def test_a_rotation_read_back_normalised_is_not_a_failure(editor: SimpleNamespace) -> None:
    result = asyncio.run(editor.session.set_actor_transform(ACTOR, rotation=[0.0, 270.0, 0.0]))

    assert result["rotation"] == [0.0, -90.0, 0.0]
    assert "error" not in result


def test_an_actor_without_a_root_component_reports_what_was_not_applied(
    editor: SimpleNamespace,
) -> None:
    editor.actor.root_component = None

    result = asyncio.run(
        editor.session.set_actor_transform(ACTOR, location=[500.0, 0.0, 0.0], scale=[2.0, 3.0, 4.0])
    )

    assert result["success"] is False
    assert result["error_type"] == "TransformNotApplied"
    assert result["error"] == "Unreal did not apply: location, scale"
    # The read-back is Unreal's identity transform for a root-less actor (seen
    # live on UE 5.7), reported with the error.
    assert (result["location"], result["rotation"], result["scale"]) == (
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    )


def test_a_root_less_actor_reads_back_identity_so_an_identity_request_matches(
    editor: SimpleNamespace,
) -> None:
    """Only components that read back different are named: a scale of 1 reads
    back as requested even though nothing stored it."""
    editor.actor.root_component = None

    result = asyncio.run(
        editor.session.set_actor_transform(ACTOR, location=[500.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0])
    )

    assert result["error"] == "Unreal did not apply: location"


def test_a_missing_actor_is_a_script_error(editor: SimpleNamespace) -> None:
    result = asyncio.run(
        editor.session.set_actor_transform("/Game/Map.Map:PersistentLevel.Gone", location=[0, 0, 0])
    )

    assert result["success"] is False
    assert result["error_type"] == "ScriptError"
    assert "Actor not found" in result["error"]


@pytest.mark.parametrize(
    "bad",
    [
        [1.0, 2.0],
        [1.0, 2.0, 3.0, 4.0],
        [math.nan, 0.0, 0.0],
        [math.inf, 0.0, 0.0],
        ["1", "2", "3"],
        [True, 0.0, 0.0],
        [1.0, None, 3.0],
        "1,2,3",
    ],
)
@pytest.mark.parametrize("component", ["location", "rotation", "scale"])
def test_a_malformed_vector_is_refused_before_anything_is_sent(component: str, bad: Any) -> None:
    session = UnrealRuntimeSession(Settings())
    session._execute_python = AsyncMock()

    with pytest.raises(ValueError, match=f"{component} must be three finite numbers"):
        asyncio.run(session.set_actor_transform(ACTOR, **{component: bad}))

    session._execute_python.assert_not_called()


def test_the_actor_path_is_data_not_python() -> None:
    session = UnrealRuntimeSession(Settings())
    session._execute_python = AsyncMock(
        return_value={"ReturnValue": True, "LogOutput": [{"Type": "Info", "Output": "{}"}]}
    )
    malicious = "'; review_injected(); #\\\n"

    asyncio.run(session.set_actor_transform(malicious))

    tree = ast.parse(session._execute_python.call_args.args[0])
    assert not any(isinstance(n, ast.Name) and n.id == "review_injected" for n in ast.walk(tree))


@pytest.mark.parametrize(
    ("name", "wanted", "got", "applied"),
    [
        # Unreal returns rotations normalised and through a quaternion.
        ("rotation", [0, 270, 0], [0, -90, 0], True),
        ("rotation", [120, 0, 0], [60, 180, 180], True),
        ("rotation", [10, 20, 30], [9.999999999999998, 19.999999999999986, 29.999999999999996], True),
        # Within ~0.08 degrees of the pole FQuat::Rotator snaps the pitch to 90
        # (and re-derives yaw and roll); the actor did turn.
        ("rotation", [89.95, 0, 0], [90.0, 0.0, 0.0], True),
        ("rotation", [89.93, 45, 10], [90.0, 90.0, 55.049476], True),
        # FRotator::Quaternion drops whole turns before converting.
        ("rotation", [0, 1e17, 0], [0, -80, 0], True),
        ("rotation", [0, 0, 0], [0, 1, 0], False),
        ("rotation", [0, 0, 0], [0, 0.5, 0], False),
        ("location", [1, 2, 3], [1, 2, 3.0000001], True),
        ("location", [500, 0, 0], [0, 0, 0], False),
        ("scale", [2, 3, 4], [1, 1, 1], False),
    ],
)
def test_read_back_comparison(name: str, wanted: list, got: list, applied: bool) -> None:
    assert _transform_component_applied(name, wanted, got) is applied


# ---------------------------------------------------------------------------
# MCP: the advertised schema, its validation, and what reaches the session
# ---------------------------------------------------------------------------


class _RecordingSession:
    def __init__(self) -> None:
        self.calls: List[tuple] = []

    async def set_actor_transform(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(("set_actor_transform", kwargs))
        return {
            "actor_path": kwargs["actor_path"],
            "location": [1.0, 2.0, 3.0],
            "rotation": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
        }

    async def set_light_params(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(("set_light_params", kwargs))
        return {"actor_path": kwargs["actor_path"], "params_set": 1}

    async def set_collision(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(("set_collision", kwargs))
        return {
            "actor_path": kwargs["actor_path"],
            "collision_preset": "OverlapAll",
            "collision_enabled": True,
        }


class _RecordingAdapter(AvailableAdapter):
    def __init__(self, session: _RecordingSession) -> None:
        super().__init__(Settings())
        self.session = session

    @contextmanager
    def create_session(self) -> Iterator[_RecordingSession]:
        yield self.session


@pytest.fixture
def mcp(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    base = Settings()
    settings = base.model_copy(
        update={"unreal": base.unreal.model_copy(update={"tool_surface": "full"})}
    )
    instance = server_module.SimulMCPServer(settings, backends={"unreal"})
    session = _RecordingSession()
    monkeypatch.setattr(instance, "unreal_adapter", _RecordingAdapter(session))

    def call(name: str, arguments: Dict[str, Any]) -> Any:
        async def run() -> Any:
            async with Client(instance.mcp) as client:
                return await client.call_tool(name, arguments, raise_on_error=False)

        return asyncio.run(run())

    def schema(name: str) -> Dict[str, Any]:
        async def run() -> Any:
            async with Client(instance.mcp) as client:
                return {tool.name: tool.inputSchema for tool in await client.list_tools()}

        return asyncio.run(run())[name]

    return SimpleNamespace(call=call, schema=schema, session=session)


def _array_branch(prop: Dict[str, Any]) -> Dict[str, Any]:
    return next(branch for branch in prop["anyOf"] if branch.get("type") == "array")


def test_transform_schema_takes_three_optional_vectors(mcp: SimpleNamespace) -> None:
    schema = mcp.schema("set_unreal_actor_transform")

    assert set(schema["properties"]) == {"actor_path", "location", "rotation", "scale"}
    assert schema["required"] == ["actor_path"]
    for name in ("location", "rotation", "scale"):
        prop = schema["properties"][name]
        assert prop.get("default") is None
        branch = _array_branch(prop)
        assert (branch["minItems"], branch["maxItems"]) == (3, 3)
        assert branch["items"]["type"] == "number"
        assert "keep" in prop["description"].lower()


def test_transform_passes_only_the_given_vector(mcp: SimpleNamespace) -> None:
    result = mcp.call(
        "set_unreal_actor_transform", {"actor_path": ACTOR, "location": [500, 20, 30]}
    )

    assert not result.is_error
    assert mcp.session.calls == [
        (
            "set_actor_transform",
            {"actor_path": ACTOR, "location": [500.0, 20.0, 30.0], "rotation": None, "scale": None},
        )
    ]


def test_transform_accepts_a_vector_sent_as_json_text(mcp: SimpleNamespace) -> None:
    result = mcp.call("set_unreal_actor_transform", {"actor_path": ACTOR, "scale": "[2, 2, 2]"})

    assert not result.is_error
    assert mcp.session.calls[0][1]["scale"] == [2.0, 2.0, 2.0]


@pytest.mark.parametrize("bad", [[1, 2], [1, 2, 3, 4], ["x", 0, 0], 5])
def test_transform_rejects_a_vector_that_is_not_three_numbers(
    mcp: SimpleNamespace, bad: Any
) -> None:
    result = mcp.call("set_unreal_actor_transform", {"actor_path": ACTOR, "rotation": bad})

    assert result.is_error
    assert mcp.session.calls == []


def test_the_old_scalar_parameters_are_gone(mcp: SimpleNamespace) -> None:
    result = mcp.call("set_unreal_actor_transform", {"actor_path": ACTOR, "location_x": 500})

    assert result.is_error
    assert mcp.session.calls == []


def test_light_color_is_one_optional_vector(mcp: SimpleNamespace) -> None:
    schema = mcp.schema("set_unreal_light_params")
    assert not {"color_r", "color_g", "color_b"} & set(schema["properties"])
    branch = _array_branch(schema["properties"]["color"])
    assert (branch["minItems"], branch["maxItems"]) == (3, 3)
    assert (branch["items"]["minimum"], branch["items"]["maximum"]) == (0.0, 1.0)

    assert mcp.call("set_unreal_light_params", {"actor_path": ACTOR, "color": [1.5, 0, 0]}).is_error
    assert not mcp.call("set_unreal_light_params", {"actor_path": ACTOR, "intensity": 5.0}).is_error

    assert mcp.session.calls == [
        (
            "set_light_params",
            {
                "actor_path": ACTOR,
                "intensity": 5.0,
                "color": None,
                "temperature": None,
                "use_temperature": None,
                "attenuation_radius": None,
                "cast_shadows": None,
            },
        )
    ]


def test_collision_preset_alone_leaves_collision_enabled_unset(mcp: SimpleNamespace) -> None:
    result = mcp.call("set_unreal_collision", {"actor_path": ACTOR, "collision_preset": "OverlapAll"})

    assert not result.is_error
    assert mcp.session.calls == [
        (
            "set_collision",
            {"actor_path": ACTOR, "collision_preset": "OverlapAll", "collision_enabled": None},
        )
    ]
    assert json.loads(result.content[0].text)["collision_preset"] == "OverlapAll"
