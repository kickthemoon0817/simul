"""Actor edits that are undoable and say what they did, against a UE-shaped editor.

Observed on UE 5.7.4 through the MCP server:

- Visibility, parenting and ``call_unreal_actor_function`` created no undo
  transaction: after spawning a cylinder and hiding another actor, one undo
  deleted the cylinder and left the actor hidden.
- ``set_unreal_actor_visibility`` set hidden-in-game alone, so the actor still
  showed in the level viewport and in captures; ``propagate`` did nothing (it
  is not a parameter of ``SetActorHiddenInGame``), and nothing was undoable.
- ``delete_unreal_actor`` on a missing path or a label returned ``success:
  true, deleted: false``, and it deleted the level's builder brush.
- ``set_unreal_camera_view`` echoed the request: a pitch of 95 read back as
  89.9999 and an X of 1e300 as 4.4e12.

Now each of them is one transaction with every actor modified first, the
visibility tool sets the editor's hide, the game's or both, on the actor and
the actors attached under it, delete refuses what is not there and what the
editor does not let you delete, and the camera is range-checked and read back.
The fake editor is ``tests/unreal/actor_editor.py``.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any, Dict, List, Tuple

import pytest

from simul.adapters import unreal_runtime
from simul.adapters._unreal_scripts import pie_refusal
from simul.config import Settings
from tests.unreal.actor_editor import (
    LEVEL,
    Actor,
    Brush,
    Editor,
    WorldSettings,
    _call,
    _not_found,
    _run,
    _session,
    install_editor,
)
from tests.unreal.editor_world import World


@pytest.fixture
def editor(monkeypatch: pytest.MonkeyPatch) -> Editor:
    return install_editor(monkeypatch)


def _family(editor: Editor) -> Tuple[Actor, Actor, Actor]:
    """A parent, its child and the child's child."""
    parent, child, grandchild = editor.spawn("Parent"), editor.spawn("Child"), editor.spawn("Grandchild")
    child.parent, grandchild.parent = parent, child
    return parent, child, grandchild


def _flags(actor: Actor) -> Dict[str, Any]:
    return {
        "actor_path": actor.path,
        "hidden_in_editor": actor.hidden_in_editor,
        "hidden_in_editor_at_startup": actor.hidden_at_startup,
        "hidden_in_game": actor.hidden_in_game,
    }


#: Events that change the level, as the fake editor records them.
CHANGES = {
    "attach", "detach", "add", "rename", "delete", "destroy", "hide in editor", "hide in game", "hide at startup",
}


def _assert_undoable(events: List[Tuple[Any, ...]]) -> None:
    """Every change sits inside one transaction, after the actor it changes was modified."""
    depth, modified = 0, set()
    for event in events:
        kind = event[0]
        if kind == "begin":
            depth += 1
        elif kind == "end":
            depth -= 1
        elif kind == "modify":
            modified.add(event[1])
        elif kind in CHANGES:
            assert depth == 1, f"{event} is outside a transaction"
            if kind.startswith("hide") or kind in ("attach", "detach"):
                assert event[1] in modified, f"{event[1]} changed before it was modified"
    assert depth == 0
    assert sum(1 for event in events if event[0] == "begin") == 1, "one undo step"


# ---------------------------------------------------------------------------
# set_actor_visibility
# ---------------------------------------------------------------------------


def test_hiding_hides_the_actor_and_everything_attached_under_it_in_the_editor_and_the_game(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It set hidden-in-game on the actor alone, which the viewport and captures ignore."""
    parent, child, grandchild = _family(editor)

    result = _run(editor, monkeypatch, "set_actor_visibility", parent.path, visible=False)

    for actor in (parent, child, grandchild):
        assert (actor.hidden_in_editor, actor.hidden_in_game, actor.hidden_at_startup) == (True, True, False)
    assert result == {
        "actor_path": parent.path,
        "visible": False,
        "scope": "all",
        "propagate": True,
        "persistent": False,
        "actors": [_flags(parent), _flags(child), _flags(grandchild)],
    }
    _assert_undoable(editor.events)


@pytest.mark.parametrize(
    ("scope", "editor_hidden", "game_hidden"),
    [("editor", True, False), ("game", False, True), ("all", True, True)],
)
def test_the_scope_names_the_visibility_that_changes(
    editor: Editor, monkeypatch: pytest.MonkeyPatch, scope: str, editor_hidden: bool, game_hidden: bool
) -> None:
    crate = editor.spawn("Crate")

    result = _run(editor, monkeypatch, "set_actor_visibility", crate.path, visible=False, scope=scope)

    assert (crate.hidden_in_editor, crate.hidden_in_game) == (editor_hidden, game_hidden)
    assert result["actors"] == [_flags(crate)] and result["scope"] == scope


def test_without_propagate_only_the_actor_changes(editor: Editor, monkeypatch: pytest.MonkeyPatch) -> None:
    parent, child, _ = _family(editor)

    result = _run(editor, monkeypatch, "set_actor_visibility", parent.path, visible=False, propagate=False)

    assert parent.hidden_in_editor and not child.hidden_in_editor
    assert [state["actor_path"] for state in result["actors"]] == [parent.path]


def test_persistent_sets_hidden_at_startup_through_the_selection_and_restores_it(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Python and Remote Control cannot write bHiddenEd; the editor's startup
    commands act on the selection, which selects no hidden actor."""
    parent, child, _ = _family(editor)
    other = editor.spawn("Other")
    editor.selection = [other]
    child.hidden_in_editor = True  # hidden earlier, so not selectable as it is

    result = _run(
        editor, monkeypatch, "set_actor_visibility", parent.path, visible=False, scope="editor", persistent=True
    )

    assert result.get("success", True) is True, result
    assert all(state["hidden_in_editor_at_startup"] and state["hidden_in_editor"] for state in result["actors"])
    assert not any(state["hidden_in_game"] for state in result["actors"])
    assert editor.selection == [other]
    assert ("select", ["Parent", "Child", "Grandchild"]) in editor.events
    assert ("hide at startup", "Child", True) in editor.events
    _assert_undoable(editor.events)


def test_showing_with_persistent_clears_hidden_at_startup(editor: Editor, monkeypatch: pytest.MonkeyPatch) -> None:
    crate = editor.spawn("Crate")
    crate.hidden_in_editor = crate.hidden_at_startup = crate.hidden_in_game = True

    result = _run(editor, monkeypatch, "set_actor_visibility", crate.path, visible=True, persistent=True)

    assert result["actors"] == [_flags(crate)]
    assert (crate.hidden_in_editor, crate.hidden_at_startup, crate.hidden_in_game) == (False, False, False)


def test_without_persistent_hidden_at_startup_is_left_alone_and_reported(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    crate = editor.spawn("Crate")
    crate.hidden_at_startup = True

    result = _run(editor, monkeypatch, "set_actor_visibility", crate.path, visible=True)

    assert crate.hidden_at_startup and result["actors"][0]["hidden_in_editor_at_startup"] is True
    assert result.get("success", True) is True


def test_persistent_is_refused_during_pie(editor: Editor, monkeypatch: pytest.MonkeyPatch) -> None:
    """The editor selects nothing during PIE, so the startup flag cannot be set."""
    crate = editor.spawn("Crate")
    editor.pie = True

    result = _run(editor, monkeypatch, "set_actor_visibility", crate.path, visible=False, persistent=True)

    assert result == pie_refusal("change what is hidden at editor startup")
    assert editor.events == []


def test_the_editor_and_game_flags_change_during_pie(editor: Editor, monkeypatch: pytest.MonkeyPatch) -> None:
    crate = editor.spawn("Crate")
    editor.pie = True

    result = _run(editor, monkeypatch, "set_actor_visibility", crate.path, visible=False)

    assert result["actors"] == [_flags(crate)] and crate.hidden_in_editor and crate.hidden_in_game


def test_a_flag_that_does_not_read_back_is_an_error_naming_the_actor(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, child, _ = _family(editor)
    monkeypatch.setattr(child, "set_actor_hidden_in_game", lambda hidden: None)

    result = _run(editor, monkeypatch, "set_actor_visibility", parent.path, visible=False)

    assert result["success"] is False and result["error_type"] == "EngineError"
    assert result["error"] == "Unreal did not apply the visibility to: " + child.path
    assert result["actors"][1] == _flags(child)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"scope": "viewport"}, "scope must be one of ['all', 'editor', 'game'], not 'viewport'"),
        ({"scope": "game", "persistent": True}, "persistent applies to the editor's hiding"),
    ],
)
def test_a_scope_the_tool_does_not_know_is_refused_before_the_editor_is_asked(
    editor: Editor, monkeypatch: pytest.MonkeyPatch, kwargs: Dict[str, Any], message: str
) -> None:
    crate = editor.spawn("Crate")
    session = _session(editor, monkeypatch)

    result = asyncio.run(session.set_actor_visibility(crate.path, visible=False, **kwargs))

    assert result["success"] is False and result["error_type"] == "ValidationError"
    assert result["error"].startswith(message)
    assert session.scripts == []


def test_the_running_games_copy_of_an_actor_is_not_found_and_the_error_names_the_editors(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PIE copies the level into a UEDPIE_<n>_<map> world; edits go to the editor's actor."""
    editor.pie = True
    editor.spawn("Crate")
    copy = editor.spawn("Crate", World("/Game/Maps/UEDPIE_0_MapB.MapB"))

    result = _run(editor, monkeypatch, "set_actor_visibility", copy.path, visible=False)

    assert _not_found(result, copy.path)
    assert result["error"].endswith(
        "Play In Editor's copy of an actor; pass the editor's, " + LEVEL + "Crate)"
    )


def test_visibility_of_a_deleted_actor_is_not_found(editor: Editor, monkeypatch: pytest.MonkeyPatch) -> None:
    crate = editor.spawn("Crate")
    editor.destroy(crate)

    result = _run(editor, monkeypatch, "set_actor_visibility", crate.path, visible=False)

    assert _not_found(result, crate.path) and editor.events == []


def test_the_visibility_tool_returns_every_flag_read_back(
    editor: Editor, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    parent, child, _ = _family(editor)

    result = _call(
        editor,
        monkeypatch,
        tmp_path,
        "set_unreal_actor_visibility",
        {"actor_path": parent.path, "visible": False, "scope": "editor"},
    )
    refused = _call(
        editor, monkeypatch, tmp_path, "set_unreal_actor_visibility", {"actor_path": parent.path, "scope": "both"}
    )

    assert result["success"] is True and len(result["actors"]) == 3
    assert result["actors"][1] == _flags(child)
    assert refused["success"] is False and refused["error_type"] == "ValidationError"


# ---------------------------------------------------------------------------
# One undo step each: the hazard the audit hit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "build"),
    [
        ("set_actor_visibility", lambda e, a, b: ((a.path,), {"visible": False})),
        ("set_actor_parent", lambda e, a, b: ((e.spawn("Third").path, a.path), {})),
        ("set_actor_parent", lambda e, a, b: ((b.path,), {"detach": True})),
        ("add_component", lambda e, a, b: ((a.path, "PointLightComponent", "Lamp"), {})),
        ("delete_actor", lambda e, a, b: ((a.path,), {})),
    ],
)
def test_each_actor_edit_is_one_undo_step_with_its_actors_modified_first(
    editor: Editor, monkeypatch: pytest.MonkeyPatch, method: str, build: Any
) -> None:
    """An edit outside a transaction left the undo history alone, so the next
    undo reverted the edit before it: a spawned cylinder, not the hide."""
    first, second = editor.spawn("First"), editor.spawn("Second")
    second.parent = first
    args, kwargs = build(editor, first, second)

    result = _run(editor, monkeypatch, method, *args, **kwargs)

    assert result.get("success", True) is True, result
    _assert_undoable(editor.events)


# ---------------------------------------------------------------------------
# delete_actor
# ---------------------------------------------------------------------------


def test_delete_deletes_and_reads_the_deletion_back(editor: Editor, monkeypatch: pytest.MonkeyPatch) -> None:
    crate = editor.spawn("Crate")

    result = _run(editor, monkeypatch, "delete_actor", crate.path)

    assert result == {"actor_path": crate.path, "deleted": True}
    assert crate.valid is False
    assert editor.events == [("begin", "simul: delete actor"), ("destroy", "Crate"), ("end", "simul: delete actor")]


@pytest.mark.parametrize(
    ("path", "hint"),
    [
        (LEVEL + "Missing", False),
        ("Crate", True),
    ],
)
def test_a_missing_path_or_a_label_is_not_found(
    editor: Editor, monkeypatch: pytest.MonkeyPatch, path: str, hint: bool
) -> None:
    """It used to be success with deleted false."""
    editor.spawn("Crate")

    result = _run(editor, monkeypatch, "delete_actor", path)

    assert result["success"] is False and result["error_type"] == "NotFoundError"
    assert _not_found(result, path)
    assert ("not its label" in result["error"]) is hint
    assert editor.events == []


def test_deleting_an_actor_twice_is_not_found_the_second_time(editor: Editor, monkeypatch: pytest.MonkeyPatch) -> None:
    crate = editor.spawn("Crate")

    first = _run(editor, monkeypatch, "delete_actor", crate.path)
    second = _run(editor, monkeypatch, "delete_actor", crate.path)

    assert first["deleted"] is True
    assert second["error_type"] == "NotFoundError"


@pytest.mark.parametrize(
    ("cls", "name", "kind"),
    [
        (WorldSettings, "WorldSettings", "the level's WorldSettings"),
        (Brush, "Brush_0", "the level's builder brush"),
        (Actor, "DefaultPhysicsVolume_0", "an actor the editor keeps out of the Outliner"),
    ],
)
def test_an_actor_the_editor_does_not_let_you_delete_is_refused(
    editor: Editor, monkeypatch: pytest.MonkeyPatch, cls: Any, name: str, kind: str
) -> None:
    """destroy_actor deleted the builder brush; the level listing leaves it out."""
    actor = editor.spawn(name, cls=cls)
    actor.listed = False

    result = _run(editor, monkeypatch, "delete_actor", actor.path)

    assert result["success"] is False and result["error_type"] == "ValidationError"
    assert result["error"] == f"{actor.path} is {kind}, which the editor does not let you delete"
    assert actor.valid and editor.events == []


def test_a_delete_unreal_refuses_is_an_error_and_its_transaction_cancelled(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    crate = editor.spawn("Crate")
    editor.refuse_destroy = True

    result = _run(editor, monkeypatch, "delete_actor", crate.path)

    assert result == {"success": False, "error_type": "EngineError", "error": "Unreal did not delete " + crate.path}
    assert ("cancel", "simul: delete actor") in editor.events and crate.valid


# ---------------------------------------------------------------------------
# call_actor_function
# ---------------------------------------------------------------------------


def _scripting_session(editor: Editor, monkeypatch: pytest.MonkeyPatch) -> Any:
    session = _session(editor, monkeypatch, Settings(security={"allow_script_execution": True}))
    session.bodies = []  # type: ignore[attr-defined]

    async def put(path: str, body: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        session.bodies.append(body)  # type: ignore[attr-defined]
        return {"ReturnValue": True}

    monkeypatch.setattr(session, "_http_put", put)
    return session


def test_a_function_call_is_one_undo_step(editor: Editor, monkeypatch: pytest.MonkeyPatch) -> None:
    """Remote Control's generateTransaction records the actor before the function runs."""
    crate = editor.spawn("Crate")
    session = _scripting_session(editor, monkeypatch)

    result = asyncio.run(session.call_actor_function(crate.path, "SetActorHiddenInGame", '{"bNewHidden": true}'))

    assert result["function_name"] == "SetActorHiddenInGame"
    assert session.bodies == [
        {
            "objectPath": crate.path,
            "functionName": "SetActorHiddenInGame",
            "parameters": {"bNewHidden": True},
            "generateTransaction": True,
        }
    ]


def test_a_function_call_on_a_deleted_actor_is_refused_before_remote_control(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    crate = editor.spawn("Crate")
    editor.destroy(crate)
    session = _scripting_session(editor, monkeypatch)

    result = asyncio.run(session.call_actor_function(crate.path, "K2_GetActorLocation"))

    assert _not_found(result, crate.path)
    assert session.bodies == []


# ---------------------------------------------------------------------------
# set_camera_view
# ---------------------------------------------------------------------------


class Viewport:
    """The level viewport camera over Remote Control; ``keep_roll`` False drops roll."""

    def __init__(self) -> None:
        self.camera: Dict[str, Any] = {
            "CameraLocation": {"X": 0.0, "Y": 0.0, "Z": 1000.0},
            "CameraRotation": {"Pitch": 0.0, "Yaw": 0.0, "Roll": 0.0},
        }
        self.calls: List[str] = []
        self.keep_roll = True

    async def call_function(self, object_path: str, function_name: str, parameters: Any = None) -> Dict[str, Any]:
        self.calls.append(function_name)
        if function_name == "SetLevelViewportCameraInfo":
            self.camera = {key: dict(value) for key, value in parameters.items()}
            if not self.keep_roll:
                self.camera["CameraRotation"]["Roll"] = 0.0
            return {}
        return {**self.camera, "ReturnValue": True}


def _camera_session(monkeypatch: pytest.MonkeyPatch) -> Tuple[Any, Viewport]:
    session = unreal_runtime.UnrealRuntimeSession(Settings())
    viewport = Viewport()
    monkeypatch.setattr(session, "_call_function", viewport.call_function)
    return session, viewport


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rotation": (95.0, 0.0, 0.0)}, "pitch must be from -90 to 90 degrees"),
        ({"rotation": (-90.5, 0.0, 0.0)}, "pitch must be from -90 to 90 degrees"),
        ({"location": (1e300, 0.0, 0.0)}, "location components must be within 4398046511104 cm"),
        ({"location": (0.0, -4.5e12, 0.0)}, "location components must be within 4398046511104 cm"),
        ({"location": (math.nan, 0.0, 0.0)}, "location must be three finite numbers"),
        ({"rotation": (0.0, math.inf, 0.0)}, "rotation must be three finite numbers"),
    ],
)
def test_a_camera_unreal_would_clamp_is_refused_before_it_moves(
    monkeypatch: pytest.MonkeyPatch, kwargs: Dict[str, Any], message: str
) -> None:
    """A pitch of 95 read back as 89.9999 and an X of 1e300 as 4.4e12, while the reply echoed the request."""
    session, viewport = _camera_session(monkeypatch)

    result = asyncio.run(session.set_camera_view(**kwargs))

    assert result["success"] is False and result["error_type"] == "ValidationError"
    assert result["error"].startswith(message)
    assert viewport.calls == []


def test_the_reply_is_the_camera_read_back(monkeypatch: pytest.MonkeyPatch) -> None:
    session, viewport = _camera_session(monkeypatch)

    result = asyncio.run(session.set_camera_view(location=(4398046511104.0, 0.0, -5.0), rotation=(90.0, 400.0, 10.0)))

    assert result == {"location": (4398046511104.0, 0.0, -5.0), "rotation": (90.0, 400.0, 10.0)}
    assert viewport.calls == ["SetLevelViewportCameraInfo", "GetLevelViewportCameraInfo"]


def test_a_camera_that_reads_back_otherwise_is_an_error_with_what_it_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    session, viewport = _camera_session(monkeypatch)
    viewport.keep_roll = False

    result = asyncio.run(session.set_camera_view(location=(1.0, 2.0, 3.0), rotation=(-10.0, 20.0, 30.0)))

    assert result["success"] is False and result["error_type"] == "EngineError"
    assert result["error"] == "The viewport camera reads back other than what was set: rotation"
    assert result["rotation"] == (-10.0, 20.0, 0.0) and result["location"] == (1.0, 2.0, 3.0)
