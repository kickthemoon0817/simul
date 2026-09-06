"""Direct tests for the success-from-error helper.

iter16 introduced ``simul_mcp.mcp.registration._helpers.apply_success_from_error``
to replace 56 unconditional ``payload['success'] = True`` lines in
``_reg_unreal.py``. The helper's contract is small but load-bearing
— every Unreal tool now passes its adapter response through it
before serializing to the MCP wire format. A regression here
silently turns ``health_check`` / ``ping`` / etc. back into the
misleading-success bug iter8 fixed for Isaac.
"""

from __future__ import annotations

from simul_mcp.mcp.registration._helpers import apply_success_from_error


def test_payload_with_no_error_marks_success_true() -> None:
    """The happy path: adapter returned cleanly with no error key."""
    payload = {"reachable": True, "latency_ms": 12.3}
    apply_success_from_error(payload)
    assert payload["success"] is True


def test_payload_with_error_key_marks_success_false() -> None:
    """The bug iter16 fixes: adapter returned a payload that includes
    ``error`` (e.g. ``ping`` against an unreachable port). Must
    surface as ``success=False`` so the MCP client doesn't see a
    falsely-green response."""
    payload = {
        "reachable": False,
        "address": "127.0.0.1:30010",
        "latency_ms": None,
        "error": "Cannot connect to 127.0.0.1:30010",
    }
    apply_success_from_error(payload)
    assert payload["success"] is False
    assert payload["error"] == "Cannot connect to 127.0.0.1:30010"


def test_payload_with_error_none_marks_success_true() -> None:
    """Edge case: payload sets ``error: None`` explicitly. Treated
    as no-error per ``payload.get('error') is None``."""
    payload = {"connected": True, "engine_version": "5.3.2", "error": None}
    apply_success_from_error(payload)
    assert payload["success"] is True


def test_pre_existing_success_field_is_not_overwritten() -> None:
    """Some adapters set ``success`` themselves (e.g. via a domain
    rule the wrapper can't see). The helper must not clobber that."""
    payload = {"success": False, "reason": "Domain-specific failure"}
    apply_success_from_error(payload)
    assert payload["success"] is False

    payload2 = {"success": True, "data": "ok-from-adapter"}
    apply_success_from_error(payload2)
    assert payload2["success"] is True


def test_returns_same_dict_for_chaining() -> None:
    """Mutation is in place; the returned dict is the same object so
    callers can chain ``result = apply_success_from_error(payload)``."""
    payload = {"foo": 1}
    returned = apply_success_from_error(payload)
    assert returned is payload


def test_unreal_health_check_failure_shape_marks_false() -> None:
    """Concrete shape: the actual return shape from
    ``UnrealRuntimeSession.health_check()`` on connection failure
    (per src/simul_mcp/adapters/unreal_runtime.py:573-578).
    """
    payload = {
        "connected": False,
        "error": "Cannot connect to host 127.0.0.1:30010 ssl:default [Connect call failed]",
    }
    apply_success_from_error(payload)
    assert payload["success"] is False
    assert payload["connected"] is False


def test_unreal_ping_success_shape_marks_true() -> None:
    """Concrete shape: ``UnrealRuntimeSession.ping()`` on success
    (per src/simul_mcp/adapters/unreal_runtime.py:350-354)."""
    payload = {
        "reachable": True,
        "address": "127.0.0.1:30010",
        "latency_ms": 4.2,
    }
    apply_success_from_error(payload)
    assert payload["success"] is True


def test_parse_python_json_error_only_shape_marks_false() -> None:
    """Concrete shape: ``UnrealRuntimeSession._parse_python_json``
    on Python execution failure returns just ``{"error": "..."}``
    with no other keys (per unreal_runtime.py:534, 545). Nine
    adapter methods (``generate_mesh_primitive``,
    ``edit_mesh_topology``, ``convert_mesh_format``, etc.) feed
    through this path, so the minimal-shape edge case is worth
    pinning explicitly."""
    payload = {"error": "No JSON output from Python execution"}
    apply_success_from_error(payload)
    assert payload["success"] is False
    assert payload["error"] == "No JSON output from Python execution"


def test_suffixed_error_key_marks_success_false() -> None:
    """A script that reports a partial failure under ``<section>_error``
    while returning the sections that worked is not a success."""
    payload = {"render_var_templates": ["LdrColor"], "syntheticdata_error": "boom"}
    apply_success_from_error(payload)
    assert payload["success"] is False


def test_suffixed_error_key_set_to_none_marks_success_true() -> None:
    payload = {"timeline": {"is_playing": False}, "timeline_error": None}
    apply_success_from_error(payload)
    assert payload["success"] is True


def test_plural_errors_key_does_not_demote() -> None:
    """``attach_errors`` is a per-item report, not a failure of the call."""
    payload = {"aovs": {"HdrColor": {}}, "attach_errors": {"Bogus": "unknown annotator"}}
    apply_success_from_error(payload)
    assert payload["success"] is True
