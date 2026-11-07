"""Actor lookup, parenting and components against a UE-shaped editor.

Observed on UE 5.7.4 through the MCP server:

- After ``delete_unreal_actor`` reported ``deleted: true``,
  ``get_unreal_actor_info`` on the same path succeeded with an empty name, a
  zero scale and mobility "None", and focus succeeded with a false "no
  bounds" warning: ``actor_at`` loaded the path with ``load_object``, which
  still finds a destroyed actor until garbage collection.
- With MapB open, info, transform and focus calls on a path into MapA loaded
  MapA from disk and reported success; the edits left MapA dirty, so a later
  Save All wrote changes nobody saw.
- ``set_unreal_actor_parent`` reported success for the actor itself as its
  parent, for a cycle and for a parent that does not exist:
  ``K2_AttachToActor`` returned ``ReturnValue: false`` and the reply ignored
  it. The attachment was not undoable, and detaching was an empty parent.
- ``add_unreal_component`` failed for every input with ``400, message=''``:
  Remote Control refuses ``AddComponentByClass`` ("deprecated or unavailable
  remotely"), ``component_name`` was unused and the class was always
  ``/Script/Engine.<name>``.

Now ``actor_at`` finds the actor among loaded objects (``find_object``, which
never loads a map) and accepts only a live actor of the editor world;
parenting refuses those cases with their reason, attaches in one undoable
transaction and reads the parent back; and components are added through
``SubobjectDataSubsystem`` the way the editor's Add Component button adds
them. The fake editor below answers the scripts the way UE 5.7.4 did.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

import pytest

from tests.unreal.actor_editor import (
    LEVEL,
    MAP_A,
    MAP_B,
    Editor,
    Mobility,
    PointLightComponent,
    SpotLightComponent,
    _call,
    _not_found,
    _run,
    _session,
    install_editor,
)


@pytest.fixture
def editor(monkeypatch: pytest.MonkeyPatch) -> Editor:
    return install_editor(monkeypatch)

# ---------------------------------------------------------------------------
# actor_at: only a live actor of the editor world
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("get_actor_info", {}),
        ("focus_on_actor", {}),
        ("set_actor_transform", {"location": [1.0, 2.0, 3.0]}),
        ("add_component", {"component_class": "PointLightComponent"}),
    ],
)
def test_a_deleted_actor_is_not_found(
    editor: Editor, monkeypatch: pytest.MonkeyPatch, method: str, kwargs: Dict[str, Any]
) -> None:
    """It read back as a live actor with an empty name and a zero scale."""
    crate = editor.spawn("Crate")
    editor.destroy(crate)
    session = _session(editor, monkeypatch)

    result = asyncio.run(getattr(session, method)(crate.path, **kwargs))

    assert _not_found(result, crate.path), result
    assert "no live actor at that path in the level open in the editor, " + MAP_B in result["error"]
    # Nothing was written, and focus moved no camera.
    assert editor.events == [] and session.calls == []


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("get_actor_info", {}),
        ("focus_on_actor", {}),
        ("set_actor_transform", {"location": [999.0, 0.0, 0.0]}),
        ("add_component", {"component_class": "PointLightComponent"}),
    ],
)
def test_a_path_into_a_map_that_is_not_open_is_refused_without_loading_it(
    editor: Editor, monkeypatch: pytest.MonkeyPatch, method: str, kwargs: Dict[str, Any]
) -> None:
    """With MapB open, a MapA path loaded MapA from disk and edited it unseen."""
    path = MAP_A + ":PersistentLevel.Crate"
    session = _session(editor, monkeypatch)

    result = asyncio.run(getattr(session, method)(path, **kwargs))

    assert _not_found(result, path), result
    assert editor.loads == [] and "/Game/Maps/MapA" in editor.disk
    assert editor.events == [] and session.calls == []


def test_an_actor_of_a_map_loaded_but_not_open_is_refused(editor: Editor, monkeypatch: pytest.MonkeyPatch) -> None:
    """A map's actors can stay loaded after it closes; they are still not the open level's."""
    stale = editor.spawn("Crate", editor.map_a)

    result = _run(editor, monkeypatch, "set_actor_transform", stale.path, location=[999.0, 0.0, 0.0])

    assert _not_found(result, stale.path), result
    assert "it belongs to " + MAP_A + ", not to the level open in the editor, " + MAP_B in result["error"]
    assert editor.events == []


def test_the_editor_world_is_found_during_pie(editor: Editor, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unreal hands out no editor world during PIE; editor_world() finds it all the same."""
    crate = editor.spawn("Crate")
    editor.pie = True

    added = _run(editor, monkeypatch, "add_component", crate.path, "PointLightComponent")

    assert added["component_class"] == "PointLightComponent", added


# ---------------------------------------------------------------------------
# set_actor_parent
# ---------------------------------------------------------------------------


def test_attaching_keeps_the_world_transform_in_one_undoable_transaction(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    child, parent, old = editor.spawn("Child"), editor.spawn("Parent"), editor.spawn("Old")
    child.parent = old

    result = _run(editor, monkeypatch, "set_actor_parent", child.path, parent.path)

    assert result == {
        "actor_path": child.path,
        "parent_path": parent.path,
        "previous_parent_path": old.path,
        "changed": True,
    }
    assert child.parent is parent
    assert editor.events == [
        ("begin", "simul: attach actor"),
        ("modify", "Child"),
        ("modify", "Parent"),
        ("modify", "Old"),
        ("attach", "Child", "Parent", ("KeepWorld", "KeepWorld", "KeepWorld")),
        ("end", "simul: attach actor"),
    ]


def test_attaching_to_the_current_parent_changes_nothing(editor: Editor, monkeypatch: pytest.MonkeyPatch) -> None:
    child, parent = editor.spawn("Child"), editor.spawn("Parent")
    child.parent = parent

    result = _run(editor, monkeypatch, "set_actor_parent", child.path, parent.path)

    assert (result["parent_path"], result["previous_parent_path"], result["changed"]) == (
        parent.path,
        parent.path,
        False,
    )
    assert editor.events == []


def test_an_actor_cannot_be_its_own_parent(editor: Editor, monkeypatch: pytest.MonkeyPatch) -> None:
    """It used to report success while K2_AttachToActor returned false."""
    crate = editor.spawn("Crate")

    result = _run(editor, monkeypatch, "set_actor_parent", crate.path, crate.path)

    assert result["success"] is False and result["error_type"] == "ValidationError"
    assert result["error"] == "An actor cannot be attached to itself: " + crate.path
    assert editor.events == [] and crate.parent is None


def test_a_cycle_is_refused_with_its_reason(editor: Editor, monkeypatch: pytest.MonkeyPatch) -> None:
    top, middle, bottom = editor.spawn("Top"), editor.spawn("Middle"), editor.spawn("Bottom")
    middle.parent, bottom.parent = top, middle

    result = _run(editor, monkeypatch, "set_actor_parent", top.path, bottom.path)

    assert result["success"] is False and result["error_type"] == "ValidationError"
    assert result["error"] == (
        f"Attaching {top.path} to {bottom.path} would make a cycle: {bottom.path} is attached under "
        f"{top.path}; detach {bottom.path} first"
    )
    assert editor.events == [] and top.parent is None


def test_a_missing_parent_is_not_found_and_the_child_keeps_its_parent(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    child, parent = editor.spawn("Child"), editor.spawn("Parent")
    child.parent = parent
    missing = LEVEL + "Nope"

    result = _run(editor, monkeypatch, "set_actor_parent", child.path, missing)

    assert _not_found(result, missing), result
    assert child.parent is parent and editor.events == []


def test_a_static_actor_is_not_attached_under_a_movable_one(editor: Editor, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unreal refuses it and returns false without saying why; the reply says why."""
    child, parent = editor.spawn("Wall"), editor.spawn("Cart", mobility=Mobility.MOVABLE)

    result = _run(editor, monkeypatch, "set_actor_parent", child.path, parent.path)

    assert result["success"] is False and result["error_type"] == "InvalidState"
    assert f"{child.path} is Static and {parent.path} is Movable" in result["error"]
    assert "Movable first" in result["error"]
    assert editor.events == [] and child.parent is None


def test_an_attachment_unreal_refuses_is_an_error_and_its_transaction_is_cancelled(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    child, parent = editor.spawn("Child"), editor.spawn("Parent")
    editor.refuse_attach = True

    result = _run(editor, monkeypatch, "set_actor_parent", child.path, parent.path)

    assert result["success"] is False and result["error_type"] == "EngineError"
    assert result["error"] == f"Unreal did not attach {child.path} to {parent.path}; the editor log says why"
    assert ("cancel", "simul: attach actor") in editor.events and child.parent is None


def test_detaching_is_explicit_and_undoable(editor: Editor, monkeypatch: pytest.MonkeyPatch) -> None:
    child, parent = editor.spawn("Child"), editor.spawn("Parent")
    child.parent = parent

    detached = _run(editor, monkeypatch, "set_actor_parent", child.path, detach=True)
    again = _run(editor, monkeypatch, "set_actor_parent", child.path, detach=True)

    assert detached == {
        "actor_path": child.path,
        "parent_path": None,
        "previous_parent_path": parent.path,
        "changed": True,
    }
    assert again == {"actor_path": child.path, "parent_path": None, "previous_parent_path": None, "changed": False}
    assert editor.events == [
        ("begin", "simul: detach actor"),
        ("modify", "Child"),
        ("modify", "Parent"),
        ("detach", "Child", ("KeepWorld", "KeepWorld", "KeepWorld")),
        ("end", "simul: detach actor"),
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "Pass parent_path to attach the actor, or detach=True to detach it from its parent"),
        (
            {"parent_path": LEVEL + "Parent", "detach": True},
            "Pass parent_path to attach the actor, or detach=True to detach it from its parent, not both",
        ),
    ],
)
def test_the_session_needs_a_parent_or_detach(
    editor: Editor, monkeypatch: pytest.MonkeyPatch, kwargs: Dict[str, Any], message: str
) -> None:
    """An omitted parent used to detach silently."""
    child = editor.spawn("Child")
    session = _session(editor, monkeypatch)

    result = asyncio.run(session.set_actor_parent(child.path, **kwargs))

    assert result == {"success": False, "error_type": "ValidationError", "error": message}
    assert session.scripts == []


# ---------------------------------------------------------------------------
# add_component
# ---------------------------------------------------------------------------


def test_a_component_is_added_under_the_root_named_and_read_back(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every input used to fail: Remote Control refuses AddComponentByClass."""
    crate = editor.spawn("Crate")

    result = _run(editor, monkeypatch, "add_component", crate.path, "PointLightComponent", "Lamp")

    assert result == {
        "actor_path": crate.path,
        "component_path": crate.path + ".Lamp",
        "component_name": "Lamp",
        "component_class": "PointLightComponent",
        "parent_component": "StaticMeshComponent0",
        "is_root": False,
    }
    assert editor.events == [
        ("begin", "simul: add component"),
        ("modify", "Crate"),
        ("add", "PointLightComponent"),
        ("rename", "Lamp"),
        ("end", "simul: add component"),
    ]


def test_a_class_path_and_a_parent_component_are_honoured(editor: Editor, monkeypatch: pytest.MonkeyPatch) -> None:
    crate = editor.spawn("Crate")
    lamp = PointLightComponent("Lamp", crate)
    lamp.attach_parent = crate.root_component
    crate.components.append(lamp)
    editor.memory["/Script/Engine.SpotLightComponent"] = SpotLightComponent.CLASS  # type: ignore[assignment]

    result = _run(
        editor,
        monkeypatch,
        "add_component",
        crate.path,
        "/Script/Engine.SpotLightComponent",
        parent_component="Lamp",
    )

    assert (result["component_name"], result["component_class"], result["parent_component"]) == (
        "SpotLight",
        "SpotLightComponent",
        "Lamp",
    )


def test_a_component_that_is_not_a_scene_component_attaches_to_nothing(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    crate = editor.spawn("Crate")

    added = _run(editor, monkeypatch, "add_component", crate.path, "ProjectileMovementComponent", "Mover")
    refused = _run(
        editor, monkeypatch, "add_component", crate.path, "ProjectileMovementComponent", parent_component="Mover"
    )

    assert (added["parent_component"], added["is_root"]) == (None, False)
    assert refused["success"] is False and refused["error_type"] == "ValidationError"
    assert "not a scene component" in refused["error"]


@pytest.mark.parametrize(
    ("component_class", "error_type", "message"),
    [
        ("PrimitiveComponent", "ValidationError", "PrimitiveComponent is abstract; pass a concrete component class"),
        (
            "StaticMeshActor",
            "ValidationError",
            "StaticMeshActor is the class StaticMeshActor, not a component class (an ActorComponent)",
        ),
        ("NoSuchComponent", "NotFoundError", "No class named 'NoSuchComponent'"),
        ("/Game/Maps/MapA.MapA", "NotFoundError", "/Game/Maps/MapA.MapA is in a map package, not a class"),
        (MAP_A + ":PersistentLevel.Crate", "NotFoundError", "names an object inside a package, not a class"),
    ],
)
def test_a_class_that_cannot_be_added_is_refused_before_anything_is_added(
    editor: Editor, monkeypatch: pytest.MonkeyPatch, component_class: str, error_type: str, message: str
) -> None:
    """An abstract class added through the subsystem fires an engine ensure on UE 5.7.4."""
    crate = editor.spawn("Crate")

    result = _run(editor, monkeypatch, "add_component", crate.path, component_class)

    assert result["success"] is False and result["error_type"] == error_type, result
    assert message in result["error"]
    assert editor.events == [] and editor.loads == []


def test_a_name_already_used_in_the_actor_is_refused_before_adding(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    crate = editor.spawn("Crate")

    result = _run(editor, monkeypatch, "add_component", crate.path, "PointLightComponent", "StaticMeshComponent0")

    assert result["success"] is False and result["error_type"] == "ValidationError"
    assert "already has a component or subobject named 'StaticMeshComponent0'" in result["error"]
    assert editor.events == []


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("", "component_name must not be empty"),
        ("Desk Lamp", "component_name 'Desk Lamp' has characters Unreal refuses in an object name: ' '"),
        ("a.b", "has characters Unreal refuses in an object name: '.'"),
        ("None", "Unreal reserves for no name"),
        ("x" * 1024, "component_name is 1024 characters long; Unreal keeps at most 1023"),
    ],
)
def test_a_name_unreal_cannot_take_is_refused_before_the_editor_is_asked(
    editor: Editor, monkeypatch: pytest.MonkeyPatch, name: str, message: str
) -> None:
    crate = editor.spawn("Crate")
    session = _session(editor, monkeypatch)

    result = asyncio.run(session.add_component(crate.path, "PointLightComponent", name))

    assert result["success"] is False and result["error_type"] == "ValidationError"
    assert message in result["error"]
    assert session.scripts == []


def test_a_missing_parent_component_is_not_found(editor: Editor, monkeypatch: pytest.MonkeyPatch) -> None:
    crate = editor.spawn("Crate")

    result = _run(editor, monkeypatch, "add_component", crate.path, "PointLightComponent", parent_component="Nope")

    assert result["success"] is False and result["error_type"] == "NotFoundError"
    assert result["error"].startswith("Component not found: Nope")
    assert editor.events == []


def test_a_component_unreal_does_not_name_is_removed_and_its_transaction_cancelled(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    crate = editor.spawn("Crate")
    editor.rename_fails = True

    result = _run(editor, monkeypatch, "add_component", crate.path, "PointLightComponent", "Lamp")

    assert result["success"] is False and result["error_type"] == "EngineError"
    assert "did not name the new PointLightComponent 'Lamp'" in result["error"]
    assert ("delete", "PointLight") in editor.events and ("cancel", "simul: add component") in editor.events
    assert [c.get_name() for c in crate.components] == ["StaticMeshComponent0"]


def test_a_component_unreal_does_not_add_is_an_error(editor: Editor, monkeypatch: pytest.MonkeyPatch) -> None:
    crate = editor.spawn("Crate")
    editor.add_fails = "Cannot add a subobject of class 'Point Light Component'"

    result = _run(editor, monkeypatch, "add_component", crate.path, "PointLightComponent")

    assert result["success"] is False and result["error_type"] == "EngineError"
    assert result["error"].endswith(": Cannot add a subobject of class 'Point Light Component'")
    assert ("cancel", "simul: add component") in editor.events


# ---------------------------------------------------------------------------
# Through the MCP tools
# ---------------------------------------------------------------------------


def test_the_parent_tool_detaches_only_when_asked(
    editor: Editor, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    child, parent = editor.spawn("Child"), editor.spawn("Parent")
    child.parent = parent

    unasked = _call(
        editor, monkeypatch, tmp_path, "set_unreal_actor_parent", {"actor_path": child.path, "parent_path": ""}
    )
    asked = _call(
        editor,
        monkeypatch,
        tmp_path,
        "set_unreal_actor_parent",
        {"actor_path": child.path, "parent_path": "", "detach": True},
    )

    assert unasked["success"] is False and unasked["error_type"] == "ValidationError"
    assert "detach=true" in unasked["error"]
    assert asked["success"] is True
    assert (asked["parent_path"], asked["previous_parent_path"], asked["changed"]) == (None, parent.path, True)
    assert child.parent is None


def test_the_component_tool_returns_what_was_read_back(
    editor: Editor, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    crate = editor.spawn("Crate")

    result = _call(
        editor,
        monkeypatch,
        tmp_path,
        "add_unreal_component",
        {"actor_path": crate.path, "component_class": "PointLightComponent", "component_name": "Lamp"},
    )

    assert result["success"] is True
    assert (result["component_path"], result["component_name"], result["parent_component"], result["is_root"]) == (
        crate.path + ".Lamp",
        "Lamp",
        "StaticMeshComponent0",
        False,
    )
