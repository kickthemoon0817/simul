"""The result budget pages a paged tool's result as a page, and nothing else.

Each paged tool declares the lists its ``offset`` pages
(``simul.mcp.paged_results.PAGED_RESULTS``). These tests pin what that buys:

- following ``next_offset`` through budget-cut pages reaches every entry once
  and always moves forward. It used to stall at ``next_offset == offset`` when
  other content or one huge entry filled the budget (Isaac round-2 audit M1);
- the three lists ``list_isaac_physics_objects`` pages with one offset are cut
  to the same index, not one of them to zero;
- a list the tool does not page (Blender's trajectory targets, whose offset
  counts samples) is cut without moving ``next_offset``, and says so;
- a cut page's hint names the tool's own paging parameters.
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pytest

from simul.mcp.paged_results import PAGED_RESULTS, PagedList, tool_result
from simul.mcp.result_budget import (
    DEFAULT_RESULT_BUDGET_BYTES,
    apply_result_budget,
    encode_result_budget,
)

PRIMS = PagedList(("prims",))
PHYSICS = PagedList(("rigid_bodies", "colliders", "joints"))
TRAJECTORIES = PAGED_RESULTS["get_blender_trajectories"]


def _entries(count: int, start: int = 0, pad: int = 40) -> List[Dict[str, Any]]:
    return [
        {"path": f"/World/envs/env_{i:04d}/FrankaPandaWithRobotiq2F85Gripper_v2/link_{'x' * pad}", "type": "Xform"}
        for i in range(start, start + count)
    ]


def _size(payload: Any) -> int:
    return len(json.dumps(payload, default=str))


def _page(entries: List[Dict[str, Any]], offset: int, limit: int, **extra: Any) -> Dict[str, Any]:
    """An Isaac-style single-list page."""
    page = entries[offset:offset + limit]
    truncated = offset + len(page) < len(entries)
    return {
        "count": len(page),
        "total": len(entries),
        "offset": offset,
        "applied_limit": limit,
        "truncated": truncated,
        "next_offset": offset + len(page) if truncated else None,
        "prims": page,
        **extra,
    }


def _walk(
    make_page, paging: Optional[PagedList], fields: Iterable[str], limit_pages: int = 500
) -> Dict[str, List[str]]:
    """Follow next_offset from 0 like an agent; collect each field's paths; assert progress."""
    seen: Dict[str, List[str]] = {field: [] for field in fields}
    offset: Optional[int] = 0
    for _ in range(limit_pages):
        sent = json.loads(encode_result_budget(make_page(offset), paging=paging))
        assert _size(sent) <= DEFAULT_RESULT_BUDGET_BYTES + 4096, _size(sent)
        for field in seen:
            seen[field].extend(entry["path"] for entry in sent.get(field) or [])
        following = sent.get("next_offset")
        if following is None:
            return seen
        assert following > offset, f"no progress: next_offset {following} at offset {offset}"
        offset = following
    raise AssertionError("paging did not finish")


# ---------------------------------------------------------------------------
# One declared list
# ---------------------------------------------------------------------------


def test_cut_page_continues_right_after_what_it_kept() -> None:
    page = apply_result_budget(_page(_entries(3000), offset=1000, limit=1000), paging=PRIMS)

    kept = len(page["prims"])
    assert 0 < kept < 1000
    assert page["next_offset"] == 1000 + kept
    assert page["count"] == kept
    assert page["total"] == 3000
    assert page["truncated"] is True
    assert page["truncation"]["fields"] == ["prims"]
    assert page["truncation"]["next_offset"] == 1000 + kept


def test_following_cut_pages_reaches_every_entry_once() -> None:
    entries = _entries(2500)

    seen = _walk(lambda offset: _page(entries, offset, 1000), PRIMS, ["prims"])

    assert seen["prims"] == [entry["path"] for entry in entries]


def test_last_page_cut_by_the_budget_gains_a_next_offset() -> None:
    page = apply_result_budget(_page(_entries(1000), offset=0, limit=1000), paging=PRIMS)

    assert page["next_offset"] == len(page["prims"]) < 1000


def test_grand_total_count_is_not_rewritten() -> None:
    payload = {"count": 9000, "offset": 0, "next_offset": 5000, "prims": _entries(5000)}

    page = apply_result_budget(payload, paging=PRIMS)

    assert page["count"] == 9000
    assert page["next_offset"] == len(page["prims"])


def test_hint_names_the_tools_own_parameters() -> None:
    extensions = PagedList(("extensions",), limit_param="limit")
    payload = {"offset": 0, "next_offset": None, "extensions": _entries(2000)}

    page = apply_result_budget(payload, paging=extensions)

    hint = page["truncation"]["hint"]
    assert f"offset={page['next_offset']}" in hint
    assert "smaller limit" in hint
    assert "max_prims" not in hint and "root_path" not in hint


def test_a_stale_hint_with_the_old_offset_is_restated() -> None:
    """Unreal's paged reads say "Call again with offset=N"; N must follow the cut."""
    payload = _page(_entries(3000), offset=0, limit=1000)
    payload["hint"] = "1000 of 3000 actors from offset 0. Call again with offset=1000 for the next page."

    page = apply_result_budget(payload, paging=PRIMS)

    assert f"offset={page['next_offset']}" in page["hint"]
    assert "offset=1000" not in page["hint"]


# ---------------------------------------------------------------------------
# Progress: never next_offset == offset
# ---------------------------------------------------------------------------


def test_other_content_filling_the_budget_still_moves_the_page_forward() -> None:
    """Non-paged content alone over budget used to leave the page empty at next_offset == offset."""
    entries = _entries(400)

    def make(offset: int) -> Dict[str, Any]:
        return _page(entries, offset, 200, warnings=[f"warning {i} " + "w" * 200 for i in range(250)])

    seen = _walk(make, PRIMS, ["prims"])

    assert seen["prims"] == [entry["path"] for entry in entries]


def test_an_entry_larger_than_the_budget_is_sent_compacted() -> None:
    huge = {"path": "/World/Huge", "type": "Mesh", "attributes": {"points": list(range(30_000))}}
    entries = [huge] + _entries(10)
    payload = _page(entries, offset=0, limit=100)

    page = apply_result_budget(payload, paging=PRIMS)

    first = page["prims"][0]
    assert first["path"] == "/World/Huge"
    assert first["truncated_item"] is True
    assert first["dropped_keys"] == ["attributes"]
    assert first["original_bytes"] > DEFAULT_RESULT_BUDGET_BYTES
    assert page["next_offset"] == 1
    assert page["truncation"]["compacted_entries"] == ["prims[0]"]


@pytest.mark.parametrize("seed", range(12))
def test_random_pages_always_progress_and_deliver_everything(seed: int) -> None:
    rng = random.Random(seed)
    entries = [
        {"path": f"/p{i}", "blob": "b" * rng.choice([10, 200, 2000, 60_000])}
        for i in range(rng.randint(20, 400))
    ]
    limit = rng.choice([10, 100, 1000])
    noise = ["n" * rng.choice([0, 100, 50_000])]

    seen = _walk(lambda offset: _page(entries, offset, limit, notes=noise), PRIMS, ["prims"])

    assert seen["prims"] == [entry["path"] for entry in entries]


# ---------------------------------------------------------------------------
# Several lists under one offset
# ---------------------------------------------------------------------------


def _physics_page(bodies, colliders, joints, offset: int, limit: int) -> Dict[str, Any]:
    """list_isaac_physics_objects' shape: three lists sliced at one offset."""
    longest = max(len(bodies), len(colliders), len(joints))
    truncated = offset + limit < longest
    return {
        "root_path": "/",
        "rigid_body_count": len(bodies),
        "collider_count": len(colliders),
        "joint_count": len(joints),
        "offset": offset,
        "applied_limit": limit,
        "truncated": truncated,
        "next_offset": offset + limit if truncated else None,
        "rigid_bodies": bodies[offset:offset + limit],
        "colliders": colliders[offset:offset + limit],
        "joints": joints[offset:offset + limit],
    }


def test_parallel_lists_are_cut_to_the_same_index() -> None:
    bodies, colliders, joints = _entries(240), _entries(240, 1000), _entries(180, 2000)

    page = apply_result_budget(_physics_page(bodies, colliders, joints, 0, 200), paging=PHYSICS)

    kept = page["truncation"]["returned"]
    assert 0 < kept < 200
    assert len(page["rigid_bodies"]) == len(page["colliders"]) == len(page["joints"]) == kept
    assert page["next_offset"] == kept


def test_following_parallel_pages_reaches_every_list_whole() -> None:
    """60 cloned envs at the default page size used to send colliders [] and next_offset 0."""
    bodies, colliders, joints = _entries(240), _entries(240, 1000), _entries(180, 2000)

    seen = _walk(
        lambda offset: _physics_page(bodies, colliders, joints, offset, 200),
        PHYSICS,
        ["rigid_bodies", "colliders", "joints"],
    )

    assert seen["rigid_bodies"] == [entry["path"] for entry in bodies]
    assert seen["colliders"] == [entry["path"] for entry in colliders]
    assert seen["joints"] == [entry["path"] for entry in joints]


def test_short_parallel_lists_are_left_whole() -> None:
    bodies, colliders, joints = _entries(5), _entries(1000, 1000), []

    page = apply_result_budget(_physics_page(bodies, colliders, joints, 0, 1000), paging=PHYSICS)

    assert len(page["rigid_bodies"]) == 5
    assert len(page["colliders"]) == page["next_offset"] < 1000


# ---------------------------------------------------------------------------
# A nested list the offset does not page
# ---------------------------------------------------------------------------


def _trajectory_page(targets: int, points: int, offset: int, limit: int) -> Dict[str, Any]:
    """Blender's trajectory page (#331): targets with points; the offset counts points."""
    name = "é" * 127
    end = min(points, offset + limit)
    return {
        "success": True,
        "offset": offset,
        "point_count": end - offset,
        "next_offset": end if end < points else None,
        "trajectories": [
            {
                "object_name": name,
                "bone_name": "é" * 29 + f"{t:02d}",
                "points": [{"frame": f + 1, "location": [0.1 * f, 0.2, 0.3]} for f in range(offset, end)],
            }
            for t in range(targets)
        ],
    }


def test_undeclared_list_cut_leaves_next_offset_alone_and_says_so() -> None:
    """The budget cut 32 targets to 31 and rewrote next_offset to 31, skipping frames 2 to 31."""
    payload = _trajectory_page(32, 40, 0, 40)

    page = apply_result_budget(payload, paging=None)

    assert len(page["trajectories"]) < 32
    assert page["next_offset"] is None  # as the tool sent it
    assert page["truncation"]["field"] == "trajectories"
    assert page["truncation"]["next_offset"] == "unchanged"
    assert "will not appear on later pages" in page["truncation"]["hint"]


def test_declared_nested_lists_are_paged_by_their_own_offset() -> None:
    targets, points = 32, 40

    def make(offset: int) -> Dict[str, Any]:
        return _trajectory_page(targets, points, offset, 40)

    frames: Dict[str, List[int]] = {}
    offset: Optional[int] = 0
    while offset is not None:
        sent = json.loads(encode_result_budget(make(offset), paging=TRAJECTORIES))
        assert len(sent["trajectories"]) == targets
        lengths = {len(target["points"]) for target in sent["trajectories"]}
        assert len(lengths) == 1, "every target cut to the same index"
        assert sent["point_count"] == lengths.pop()
        for target in sent["trajectories"]:
            frames.setdefault(target["bone_name"], []).extend(p["frame"] for p in target["points"])
        following = sent["next_offset"]
        assert following is None or following > offset
        offset = following

    assert len(frames) == targets
    assert all(seen == list(range(1, points + 1)) for seen in frames.values())


def test_a_cut_page_clears_cursors_into_its_last_entry() -> None:
    """Blender keyframe pages: next_key_offset points into the page's last channel."""
    channels = [{"data_path": f"pose.bones[\"b{i}\"].location", "keys": list(range(300))} for i in range(60)]
    payload = {"offset": 0, "channels": channels, "next_offset": None, "next_key_offset": 120}

    page = apply_result_budget(payload, paging=PAGED_RESULTS["get_blender_keyframes"])

    assert 0 < len(page["channels"]) < 60
    assert page["next_offset"] == len(page["channels"])
    assert page["next_key_offset"] is None
    assert "smaller limit" in page["truncation"]["hint"]


def test_non_paged_list_of_a_declared_tool_is_reported_separately() -> None:
    payload = _page(_entries(10), 0, 10, warnings=["w" * 300 for _ in range(300)])

    page = apply_result_budget(payload, paging=PRIMS)

    assert len(page["prims"]) == 10
    cut = page["truncated_fields"][0]
    assert cut["field"] == "warnings"
    assert cut["next_offset"] == "unchanged"
    assert page["next_offset"] is None


# ---------------------------------------------------------------------------
# Where the declaration comes from
# ---------------------------------------------------------------------------


def test_the_envelopes_tool_name_selects_the_declaration() -> None:
    bodies, colliders, joints = _entries(240), _entries(240, 1000), _entries(180, 2000)
    payload = _physics_page(bodies, colliders, joints, 0, 200)

    with tool_result("list_isaac_physics_objects"):
        declared = json.loads(encode_result_budget(payload))
    by_name = json.loads(encode_result_budget(payload, tool_name="list_isaac_physics_objects"))
    undeclared = json.loads(encode_result_budget(payload))

    assert declared == by_name
    assert len(declared["colliders"]) == len(declared["rigid_bodies"]) == declared["next_offset"]
    assert undeclared["next_offset"] == 200
    assert undeclared["truncation"]["next_offset"] == "unchanged"


def test_default_hint_names_no_tools_parameters() -> None:
    page = apply_result_budget({"success": True, "items": _entries(3000)})

    assert "max_prims" not in page["truncation"]["hint"]
    assert "root_path" not in page["truncation"]["hint"]


def _exec_isaac_page(monkeypatch: pytest.MonkeyPatch) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    import asyncio

    from simul.config import Settings
    from simul.mcp import backends as backends_module
    from simul.mcp import server as server_module
    from tests.fakes import FakeFastMCP

    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)
    monkeypatch.setattr(backends_module, "is_headless_available", lambda: False)
    monkeypatch.setattr(backends_module, "is_blender_available", lambda: False)
    monkeypatch.setattr(backends_module, "UnrealRuntimeAdapter", None)
    instance = server_module.SimulMCPServer(settings=Settings())
    raw = _physics_page(_entries(240), _entries(240, 1000), _entries(180, 2000), 0, 200)

    async def _call() -> Dict[str, Any]:
        return dict(raw)

    result = asyncio.run(instance._exec_isaac("list_isaac_physics_objects", _call()))
    return raw, json.loads(result.content[-1].text)


def test_the_server_envelope_pages_by_the_tools_declaration(monkeypatch: pytest.MonkeyPatch) -> None:
    raw, sent = _exec_isaac_page(monkeypatch)

    kept = sent["truncation"]["returned"]
    assert len(sent["rigid_bodies"]) == len(sent["colliders"]) == len(sent["joints"]) == kept
    assert sent["next_offset"] == kept < raw["next_offset"]
