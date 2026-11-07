"""The result budget reaches nested fields and emits valid JSON.

Two ways a result used to slip past the budget (paging is in
``test_result_budget_paging.py``):

- the hard limit only replaced top-level strings, so a prim's million-point
  attribute (nested under ``attributes``) crossed the wire whole, 51 MB;
- ``json.dumps`` writes ``inf`` as the bare token ``Infinity``, which strict
  JSON parsers reject (``physics:breakForce`` defaults to it).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from simul.mcp.result_budget import (
    HARD_RESULT_LIMIT_BYTES,
    apply_result_budget,
    encode_result_budget,
)


def _prims(count: int) -> List[Dict[str, Any]]:
    return [
        {
            "path": f"/World/Environment/Building_{i}/Floor/Mesh_{i}",
            "name": f"Mesh_{i}",
            "type": "Mesh",
            "depth": 4,
            "child_count": 3,
        }
        for i in range(count)
    ]


def _encoded_size(payload: Any) -> int:
    return len(json.dumps(payload, default=str))


def _strict_loads(text: str) -> Any:
    def _reject(name: str) -> Any:
        raise ValueError(f"non-standard JSON constant {name}")

    return json.loads(text, parse_constant=_reject)


# ---------------------------------------------------------------------------
# The hard limit reaches nested fields
# ---------------------------------------------------------------------------


def test_nested_list_above_the_hard_limit_is_cut_where_it_sits() -> None:
    points = [[float(i), 0.5, -0.25] for i in range(200_000)]
    payload = {
        "success": True,
        "path": "/World/Terrain",
        "attributes": {
            "points": {"type": "point3f[]", "value": points},
            "doubleSided": {"type": "bool", "value": False},
        },
    }

    text = encode_result_budget(payload)
    result = _strict_loads(text)

    assert len(text) <= HARD_RESULT_LIMIT_BYTES
    kept = result["attributes"]["points"]["value"]
    assert 0 < len(kept) < len(points)
    assert kept == points[: len(kept)]
    assert result["attributes"]["doubleSided"] == {"type": "bool", "value": False}
    assert result["truncated"] is True
    assert result["truncated_fields"] == [
        {
            "field": "attributes.points.value",
            "returned": len(kept),
            "total": len(points),
            "hint": result["truncated_fields"][0]["hint"],
        }
    ]
    assert payload["attributes"]["points"]["value"] is points, "the caller's payload was mutated"
    assert len(payload["attributes"]["points"]["value"]) == 200_000


def test_nested_string_above_the_hard_limit_is_replaced_by_a_marker() -> None:
    payload = {"success": True, "result": {"stdout": "x" * 900_000, "code": 0}}

    result = apply_result_budget(payload)

    assert result["result"]["stdout"]["truncated_field"] == "result.stdout"
    assert result["result"]["stdout"]["original_bytes"] == 900_000
    assert result["result"]["code"] == 0
    assert _encoded_size(result) <= HARD_RESULT_LIMIT_BYTES


def test_mapping_of_many_small_entries_keeps_a_prefix() -> None:
    attributes = {f"attr_{i:05d}": {"type": "float", "value": i} for i in range(20_000)}
    payload = {"success": True, "attributes": attributes}

    result = apply_result_budget(payload)

    kept = result["attributes"]
    assert _encoded_size(result) <= HARD_RESULT_LIMIT_BYTES
    assert 0 < len(kept) < len(attributes)
    assert list(kept) == list(attributes)[: len(kept)]
    assert result["truncated_fields"][0]["field"] == "attributes"


def test_list_of_records_is_cut_by_record() -> None:
    payload = {"success": True, "data": {"results": _prims(20_000)}}

    result = apply_result_budget(payload)

    kept = result["data"]["results"]
    assert _encoded_size(result) <= HARD_RESULT_LIMIT_BYTES
    assert kept == payload["data"]["results"][: len(kept)]
    assert result["truncated_fields"][0]["field"] == "data.results"


def test_payload_under_the_hard_limit_keeps_its_nested_fields() -> None:
    """Between the budget and the hard limit a nested bulk is flagged, not cut."""
    payload = {"success": True, "info": {"values": list(range(30_000))}}

    result = apply_result_budget(payload)

    assert result["info"]["values"] == payload["info"]["values"]
    assert result["oversized_bytes"] == _encoded_size(payload)
    assert "truncated_fields" not in result


def test_encoding_matches_the_budgeted_payload_after_a_nested_cut() -> None:
    payload = {"success": True, "attributes": {"points": {"value": list(range(300_000))}}}

    assert encode_result_budget(payload) == json.dumps(apply_result_budget(payload), default=str)


# ---------------------------------------------------------------------------
# Valid JSON: non-finite floats
# ---------------------------------------------------------------------------


def test_non_finite_floats_become_strings() -> None:
    payload = {
        "success": True,
        "attributes": {"physics:breakForce": {"value": float("inf")}},
        "limits": [float("-inf"), 1.5, float("nan")],
    }

    text = encode_result_budget(payload)
    result = _strict_loads(text)

    assert result["attributes"]["physics:breakForce"]["value"] == "Infinity"
    assert result["limits"] == ["-Infinity", 1.5, "NaN"]
    assert text == json.dumps(apply_result_budget(payload), default=str)


def test_finite_payload_encoding_is_unchanged() -> None:
    payload = {"success": True, "values": [1.5, -2.0, 0.0], "name": "ok"}

    assert encode_result_budget(payload) == json.dumps(payload, default=str)
