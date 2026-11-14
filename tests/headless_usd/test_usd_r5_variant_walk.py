"""USD round 5 m1: a load without the sandbox reads only the variant it composes.

Before: with the sandbox off (and always in the CLI), the walk that stamps
each file before pxr reads it (#369) opened the layers of every variant, so
an asset with a "lod" variant set of 8 variants, each referencing its own
50k-prim file, loaded 2.9 times slower (265 ms to 765 ms), and
``simul usd info`` went from 0.93 s to 1.43 s.

Now a sandbox-off load opens only the arcs authored outside variant sets;
an arc inside one is stamped but not read, so if it is the selected one,
composition reads it after its stamp. The CLI, which never reuses, reloads
or saves a stage, walks nothing at all.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator

import pytest

pytest.importorskip("pxr", reason="pxr library not available")

from pxr import Sdf  # noqa: E402

from simul.adapters import headless_usd  # noqa: E402
from simul.adapters.headless_usd import HeadlessUSDSession  # noqa: E402
from simul.usd import sandbox as sandbox_module, stage_files  # noqa: E402

_VARIANTS = ("v0", "v1", "v2", "v3")


def _asset(directory: Path) -> Dict[str, Path]:
    """asset.usda: a sublayer, a reference outside variants, and a "lod" set whose variants each reference a file."""
    (directory / "sub.usda").write_text('#usda 1.0\ndef Xform "FromSub"\n{\n}\n')
    (directory / "base.usda").write_text('#usda 1.0\ndef Xform "Base"\n{\n}\n')
    for name in _VARIANTS:
        (directory / f"{name}.usda").write_text(
            f'#usda 1.0\ndef Xform "Mesh"\n{{\n    custom string lod = "{name}"\n}}\n'
        )
    variants = "\n".join(
        f'        "{name}" (references = @./{name}.usda@</Mesh>) {{\n        }}' for name in _VARIANTS
    )
    (directory / "asset.usda").write_text(
        "#usda 1.0\n(\n    subLayers = [@./sub.usda@]\n)\n"
        'def Xform "Asset" (\n    references = @./base.usda@</Base>\n'
        '    variants = { string lod = "v1" }\n    prepend variantSets = "lod"\n)\n{\n'
        f'    variantSet "lod" = {{\n{variants}\n    }}\n}}\n'
    )
    return {path.stem: path for path in directory.glob("*.usda")}


def _age(path: Path, seconds: float = 3600.0) -> None:
    then = time.time() - seconds
    os.utime(path, (then, then))


def _held(path: Path) -> bool:
    return bool(Sdf.Layer.Find(str(path)))


def _lod(session: HeadlessUSDSession, stage_id: str) -> Any:
    return session.get_stage(stage_id).GetPrimAtPath("/Asset").GetAttribute("lod").Get()


@pytest.fixture
def session() -> Iterator[HeadlessUSDSession]:
    usd_session = HeadlessUSDSession()
    usd_session.set_owner_resolver(lambda: "session-a")
    yield usd_session
    usd_session.cleanup()


def test_the_walk_stamps_an_arc_inside_a_variant_without_opening_it(tmp_path: Path) -> None:
    paths = _asset(tmp_path)
    root = Sdf.Layer.FindOrOpen(str(paths["asset"]))
    reads: Dict[str, Any] = {}

    opened = sandbox_module.open_arc_layers(root, reads, variants=False)

    assert sorted(Path(layer.realPath).name for layer in opened) == ["base.usda", "sub.usda"]
    assert not any(_held(paths[name]) for name in _VARIANTS), "no variant's file was read"
    stamped = sorted(Path(path).name for path in reads)
    assert stamped == ["base.usda", "sub.usda"] + [f"{name}.usda" for name in _VARIANTS]
    assert reads[str(paths["v2"].resolve())] == stage_files.stamp(str(paths["v2"].resolve()))


def test_past_the_scan_budget_every_arc_is_opened_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _asset(tmp_path)
    monkeypatch.setattr(sandbox_module, "VARIANT_SCAN_BUDGET", 0)
    root = Sdf.Layer.FindOrOpen(str(paths["asset"]))

    opened = sandbox_module.open_arc_layers(root, {}, variants=False)

    assert len(opened) == 2 + len(_VARIANTS)


def test_a_load_without_the_sandbox_reads_only_the_selected_variant(
    session: HeadlessUSDSession, tmp_path: Path
) -> None:
    paths = _asset(tmp_path)

    stage_id = session.load_stage_or_raise(paths["asset"])

    assert _lod(session, stage_id) == "v1"
    assert _held(paths["v1"]) and _held(paths["base"])
    assert not any(_held(paths[name]) for name in _VARIANTS if name != "v1")
    assert session.stage_status(stage_id)["changed_on_disk"] == []
    assert session.load_stage_or_raise(paths["asset"]) == stage_id, "the stage is reused: its stamps hold"


def test_the_selected_variant_s_file_is_stamped_before_composition_reads_it(
    session: HeadlessUSDSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A write that lands after composition read the file is caught, even one that keeps an old mtime.

    Read unrecorded, a file last modified long ago is trusted as read; its
    stamp from before the read tells the write apart (size, inode), so the
    load rereads it before it returns, as #369 has it for every file.
    """
    paths = _asset(tmp_path)
    for path in paths.values():
        _age(path)
    compose = stage_files.compose_stage

    def composed_then_written(root: Any) -> Any:
        stage = compose(root)
        staged = paths["v1"].with_name("v1.partial")
        staged.write_text(
            '#usda 1.0\ndef Xform "Mesh"\n{\n    custom string lod = "rewritten by another program"\n}\n'
        )
        os.replace(staged, paths["v1"])
        _age(paths["v1"])
        return stage

    monkeypatch.setattr(stage_files, "compose_stage", composed_then_written)
    stage_id = session.load_stage_or_raise(paths["asset"])

    assert _lod(session, stage_id) == "rewritten by another program", "reread: its stamp predated the read"
    assert session.stage_status(stage_id)["changed_on_disk"] == []


def test_a_load_that_tracks_no_changes_walks_nothing(
    session: HeadlessUSDSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _asset(tmp_path)

    def no_walk(root: Any, reads: Any) -> Any:
        raise AssertionError("the CLI's load walked the arcs")

    monkeypatch.setattr(headless_usd, "open_arc_layers", no_walk)

    stage_id = session.load_stage_or_raise(paths["asset"], track_changes=False)

    assert _lod(session, stage_id) == "v1"
    assert not any(_held(paths[name]) for name in _VARIANTS if name != "v1")
