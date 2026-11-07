"""No simul code or test saves more than its own packages in an Unreal editor.

Unreal's broad saves (EditorLoadingAndSavingUtils.save_dirty_packages and its
variants, EditorAssetLibrary.save_directory over a root) write every dirty
package the editor holds, engine and plugin content included: a live review
test that ran Save All after a glTF import had dirtied
/InterchangeAssets/gltf/M_Default re-saved
Engine/Plugins/Interchange/Assets/Content/gltf/M_Default.uasset into the shared
UE 5.7 install. Save the packages you made instead: a test's own map with
LevelEditorSubsystem.save_current_level, or EditorLoadingAndSavingUtils.
save_packages / EditorAssetLibrary.save_loaded_asset on the packages you name.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]

# save_dirty_packages, save_dirty_packages_with_dialog, save_dirty_map_packages...
_SAVE_DIRTY = re.compile(r"\bsave_dirty_\w*packages\w*\s*\(")
# save_directory on "/", an engine or plugin root, or the whole /Game.
_SAVE_ROOT = re.compile(r"\bsave_directory\s*\(\s*f?['\"]/(?!Game/[^'\"]+)")


def broad_saves(text: str) -> List[str]:
    """The broad editor saves in one source file, as ``line: text``."""
    found = []
    for number, line in enumerate(text.splitlines(), 1):
        if _SAVE_DIRTY.search(line) or _SAVE_ROOT.search(line):
            found.append(f"{number}: {line.strip()}")
    return found


def test_no_source_or_test_saves_every_dirty_package() -> None:
    this_file = Path(__file__).resolve()  # its examples below are the broad saves themselves
    offenders = {
        str(path.relative_to(ROOT)): saves
        for folder in ("src", "tests")
        for path in sorted((ROOT / folder).rglob("*.py"))
        if path.resolve() != this_file and (saves := broad_saves(path.read_text(encoding="utf-8")))
    }

    assert offenders == {}, (
        "Broad Unreal saves write engine and plugin packages into the shared install; "
        f"save only the packages you made: {offenders}"
    )


def test_the_guard_catches_the_broad_saves_and_allows_scoped_ones() -> None:
    caught = [
        "ok = unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True, True)",
        "unreal.EditorLoadingAndSavingUtils.save_dirty_packages_with_dialog(True, True)",
        "unreal.EditorAssetLibrary.save_directory('/Engine/')",
        'unreal.EditorAssetLibrary.save_directory("/Game")',
        "unreal.EditorAssetLibrary.save_directory('/')",
    ]
    allowed = [
        "ok = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).save_current_level()",
        "unreal.EditorLoadingAndSavingUtils.save_packages([package], False)",
        "unreal.EditorAssetLibrary.save_loaded_asset(instance)",
        "unreal.EditorAssetLibrary.save_directory('/Game/SimulReview_1234/')",
    ]

    assert [line for line in caught if not broad_saves(line)] == []
    assert [line for line in allowed if broad_saves(line)] == []
