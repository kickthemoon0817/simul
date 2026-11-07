"""SimReady metadata: a real input schema, refused unknown fields, clearing, tolerant reads.

The runtime keeps its own copy of the metadata rules (it ships inside the
Blender add-on and cannot import the MCP schemas); the last tests keep that
copy and the request models in step.
"""

from __future__ import annotations

import asyncio
import typing
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest
from fastmcp import Client
from pydantic import ValidationError

from simul.adapters import blender_runtime
from simul.config import Settings
from simul.mcp import server as server_module
from simul.mcp.schemas import simready as schemas
from simul.utils.paths import PathPolicy


class Obj:
    def __init__(self, name: str, kind: str = "MESH") -> None:
        self.name = name
        self.type = kind
        self.props: Dict[str, Any] = {}
        self.library: Optional[str] = None

    def __setitem__(self, key: str, value: Any) -> None:
        self.props[key] = value

    def __getitem__(self, key: str) -> Any:
        return self.props[key]

    def __delitem__(self, key: str) -> None:
        del self.props[key]

    def keys(self) -> List[str]:
        return list(self.props)


class Objects(list):
    def get(self, name: str) -> Optional[Obj]:
        return next((o for o in self if o.name == name), None)


@pytest.fixture
def box(monkeypatch: pytest.MonkeyPatch) -> Any:
    obj = Obj("box")
    objects = Objects([obj])
    fake = SimpleNamespace(
        data=SimpleNamespace(objects=objects),
        context=SimpleNamespace(scene=SimpleNamespace(objects=objects), mode="OBJECT"),
        app=SimpleNamespace(version=(5, 0, 1), version_string="5.0.1", background=True),
    )
    monkeypatch.setattr(blender_runtime, "bpy", fake)
    monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
    return SimpleNamespace(obj=obj, session=blender_runtime.BlenderRuntimeSession(), fake=fake)


# ---------------------------------------------------------------------------
# M7: the fields are advertised, unknown ones refused, values cleared
# ---------------------------------------------------------------------------


def _listed_schema(tmp_path: Path, tool: str) -> Dict[str, Any]:
    settings = Settings(
        blender={
            "enabled": True,
            "mode": "attached",
            "tool_surface": "full",
            "discovery_dir": str(tmp_path / "blender"),
            "attachment_path": str(tmp_path / "blender" / "attachment.json"),
        },
        security={"rate_limiting_enabled": False},
    )
    instance = server_module.SimulMCPServer(settings, backends={"blender"})

    async def run() -> List[Any]:
        async with Client(instance.mcp) as client:
            return await client.list_tools()

    [listed] = [t for t in asyncio.run(run()) if t.name == tool]
    return listed.model_dump(mode="json", by_alias=True)["inputSchema"]


def _object_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """The object branch of an Optional[...] schema."""
    for branch in schema.get("anyOf", [schema]):
        if branch.get("type") == "object" or "properties" in branch:
            return branch
    raise AssertionError(schema)


def test_the_metadata_input_schema_lists_the_real_fields(tmp_path: Path) -> None:
    schema = _listed_schema(tmp_path, "apply_simready_metadata")

    metadata = _object_schema(schema["properties"]["metadata"])
    assert metadata["additionalProperties"] is False
    physics = _object_schema(metadata["properties"]["physics"])
    assert physics["additionalProperties"] is False
    assert {"mass_kg", "collider_type", "is_rigid_body"} <= set(physics["properties"])
    collider = physics["properties"]["collider_type"]["anyOf"][0]
    assert collider["enum"] == list(schemas.COLLIDER_TYPES)
    assert "convexDecomposition" in collider["enum"]
    clear = schema["properties"]["clear"]["anyOf"][0]["items"]["enum"]
    assert {"mass_kg", "physics", "all"} <= set(clear)


def test_the_hierarchy_semantic_input_schema_lists_the_real_fields(tmp_path: Path) -> None:
    schema = _listed_schema(tmp_path, "setup_simready_hierarchy")

    semantic = _object_schema(schema["properties"]["semantic"])
    assert semantic["additionalProperties"] is False
    assert set(semantic["properties"]) == {
        "semantic_class",
        "semantic_hierarchy",
        "semantic_qcode",
        "additional_labels",
    }


def test_an_unknown_field_is_refused_with_the_closest_name() -> None:
    with pytest.raises(ValidationError, match="unknown field 'mass' \\(did you mean 'mass_kg'\\?\\)"):
        schemas.SimReadyApplyMetadataRequest(
            object_name="box", metadata={"physics": {"mass": 5.0}}
        )
    with pytest.raises(ValidationError, match="did you mean 'physics'"):
        schemas.SimReadyApplyMetadataRequest(object_name="box", metadata={"phisics": {}})


def test_the_runtime_refuses_unknown_fields_too_and_writes_nothing(box: Any) -> None:
    with pytest.raises(ValueError, match="unknown physics field 'mass' \\(did you mean 'mass_kg'\\?\\)"):
        box.session.apply_simready_metadata(
            "box", {"physics": {"mass": 5.0, "collider_type": "convexHull"}}
        )
    assert box.obj.props == {}


def test_an_invalid_value_refuses_the_whole_call(box: Any) -> None:
    with pytest.raises(ValueError, match="physics.restitution=1.5 must be at most 1"):
        box.session.apply_simready_metadata(
            "box", {"physics": {"restitution": 1.5, "mass_kg": 2.0}}
        )
    assert box.obj.props == {}


def test_is_rigid_body_false_clears_an_earlier_true(box: Any) -> None:
    box.session.apply_simready_metadata("box", {"physics": {"is_rigid_body": True}})
    box.session.apply_simready_metadata("box", {"physics": {"mass_kg": 3.0}})
    assert box.session.get_simready_metadata("box")["metadata"]["physics"]["is_rigid_body"] is True

    box.session.apply_simready_metadata("box", {"physics": {"is_rigid_body": False}})

    physics = box.session.get_simready_metadata("box")["metadata"]["physics"]
    assert physics == {"is_rigid_body": False, "mass_kg": 3.0}


def test_clear_removes_fields_groups_and_labels(box: Any) -> None:
    box.session.apply_simready_metadata(
        "box",
        {
            "semantic": {"semantic_class": "crate", "additional_labels": {"sku": "C-1"}},
            "physics": {"mass_kg": 3.0, "density": 500.0},
        },
    )

    result = box.session.apply_simready_metadata("box", clear=["mass_kg", "additional_labels"])

    assert sorted(result["cleared_properties"]) == ["simready_label_sku", "simready_mass_kg"]
    assert sorted(box.obj.props) == ["simready_density", "simready_semantic_class"]
    box.session.apply_simready_metadata(
        "box", {"physics": {"collider_type": "convexHull"}}, clear=["all"]
    )
    assert box.obj.props == {"simready_collider_type": "convexHull"}


def test_an_unknown_clear_name_and_an_empty_call_are_refused(box: Any) -> None:
    with pytest.raises(ValueError, match="did you mean 'mass_kg'"):
        box.session.apply_simready_metadata("box", clear=["mass"])
    with pytest.raises(ValueError, match="Nothing to apply"):
        box.session.apply_simready_metadata("box", {})


def test_a_label_key_must_be_an_identifier(box: Any) -> None:
    with pytest.raises(ValueError, match="label key 'product id' must be a USD identifier"):
        box.session.apply_simready_metadata(
            "box", {"semantic": {"additional_labels": {"product id": "X"}}}
        )
    with pytest.raises(ValidationError):
        schemas.SimReadySemanticLabels(additional_labels={"product id": "X"})


# ---------------------------------------------------------------------------
# M8: stored values are read tolerantly
# ---------------------------------------------------------------------------


def test_stored_values_that_break_the_rules_are_reported_not_fatal(box: Any) -> None:
    box.obj.props.update(
        {
            "simready_collider_type": "convexDecomposition",
            "simready_restitution": 1.5,
            "simready_mass_kg": "heavy",
            "simready_semantic_qcode": "Q1",
            "simready_mass": 5.0,
            "simready_label_bad key": "x",
        }
    )

    payload = box.session.get_simready_metadata("box")

    assert payload["metadata"] == {
        "semantic": {"semantic_qcode": "Q1"},
        "physics": {"collider_type": "convexDecomposition"},
    }
    problems = {i["property"]: i["problem"] for i in payload["invalid_values"]}
    assert problems["simready_restitution"] == "must be at most 1"
    assert problems["simready_mass_kg"] == "must be a number"
    assert "did you mean 'mass_kg'" in problems["simready_mass"]
    assert "identifier" in problems["simready_label_bad key"]
    # The server's response model accepts it: no validation error with nothing returned.
    response = schemas.SimReadyGetMetadataResponse(success=True, **payload)
    assert response.metadata is not None
    assert response.metadata.physics is not None
    assert response.metadata.physics.collider_type == "convexDecomposition"


def test_an_object_without_metadata_reads_as_empty(box: Any) -> None:
    payload = box.session.get_simready_metadata("box")

    assert payload == {
        "object_name": "box",
        "metadata": None,
        "has_simready_data": False,
        "invalid_values": [],
    }


# ---------------------------------------------------------------------------
# The runtime's rules and the request models stay in step
# ---------------------------------------------------------------------------


def test_the_runtime_fields_match_the_request_models() -> None:
    fields = blender_runtime.BlenderRuntimeSession._SIMREADY_FIELDS
    assert set(fields["semantic"]) == set(schemas.SimReadySemanticLabels.model_fields) - {
        "additional_labels"
    }
    assert set(fields["physics"]) == set(schemas.SimReadyPhysicsProperties.model_fields)
    assert set(fields["material"]) == set(schemas.SimReadyMaterialProperties.model_fields)
    assert set(schemas.SimReadyMetadata.model_fields) == set(fields)


def test_the_collider_types_match_everywhere() -> None:
    from simul.adapters.simready_usd import COLLIDER_APPROXIMATIONS

    assert schemas.COLLIDER_TYPES == blender_runtime.SIMREADY_COLLIDER_TYPES
    assert schemas.COLLIDER_TYPES == typing.get_args(schemas.ColliderType)
    assert set(schemas.COLLIDER_TYPES) == set(COLLIDER_APPROXIMATIONS)


def test_the_clear_names_match_the_runtime() -> None:
    session = blender_runtime.BlenderRuntimeSession
    expected = (
        session._simready_field_names()
        + list(session._SIMREADY_FIELDS)
        + ["additional_labels", "all"]
    )
    assert list(typing.get_args(schemas.SimReadyClearName)) == expected


@pytest.mark.parametrize(
    ("field", "value", "accepted"),
    [
        ("mass_kg", -1.0, False),
        ("restitution", 1.0, True),
        ("restitution", 1.01, False),
        ("texel_density", 0.0, False),
        ("collider_type", "cylinder", False),
        ("semantic_qcode", "Q0", False),
        ("semantic_qcode", "Q42", True),
    ],
)
def test_the_runtime_and_the_models_agree_on_values(
    box: Any, field: str, value: Any, accepted: bool
) -> None:
    group = next(
        g for g, names in blender_runtime.BlenderRuntimeSession._SIMREADY_FIELDS.items()
        if field in names
    )
    model = {
        "semantic": schemas.SimReadySemanticLabels,
        "physics": schemas.SimReadyPhysicsProperties,
        "material": schemas.SimReadyMaterialProperties,
    }[group]

    try:
        model(**{field: value})
        model_ok = True
    except ValidationError:
        model_ok = False
    try:
        box.session.apply_simready_metadata("box", {group: {field: value}})
        runtime_ok = True
    except ValueError:
        runtime_ok = False

    assert (model_ok, runtime_ok) == (accepted, accepted)


# ---------------------------------------------------------------------------
# The #325 review: taxonomy-named labels, Edit Mode, linked data, error codes
# ---------------------------------------------------------------------------


def test_a_label_named_after_a_fields_taxonomy_sets_that_field(box: Any) -> None:
    """wikidata_qcode is SimReady's Q-code taxonomy: as a label it is the Q-code."""
    box.session.apply_simready_metadata(
        "box", {"semantic": {"additional_labels": {"wikidata_qcode": "Q2413", "sku": "M-1"}}}
    )

    assert box.obj.props == {"simready_semantic_qcode": "Q2413", "simready_label_sku": "M-1"}


def test_a_label_that_would_overwrite_its_field_is_refused(box: Any) -> None:
    """An additional label `class` overwrote semantic_class in the USD."""
    with pytest.raises(blender_runtime.ValidationError) as excinfo:
        box.session.apply_simready_metadata(
            "box",
            {"semantic": {"semantic_class": "crate", "additional_labels": {"class": "vehicle"}}},
        )

    assert excinfo.value.details["code"] == "InvalidMetadata"
    assert "sets semantic.semantic_class, which is also given as 'crate'" in str(excinfo.value)
    assert box.obj.props == {}
    with pytest.raises(ValidationError, match="sets semantic_class"):
        schemas.SimReadySemanticLabels(semantic_class="crate", additional_labels={"class": "vehicle"})
    with pytest.raises(ValidationError, match="Wikidata Q-code"):
        schemas.SimReadySemanticLabels(additional_labels={"wikidata_qcode": "x"})


def test_a_stored_label_named_after_a_taxonomy_is_reported(box: Any) -> None:
    box.obj["simready_label_class"] = "vehicle"

    payload = box.session.get_simready_metadata("box")

    assert payload["metadata"] is None
    assert payload["invalid_values"][0]["property"] == "simready_label_class"


def test_edit_mode_and_linked_objects_are_refused(box: Any) -> None:
    box.fake.context.mode = "EDIT_MESH"
    with pytest.raises(blender_runtime.ValidationError) as excinfo:
        box.session.apply_simready_metadata("box", {"physics": {"mass_kg": 1.0}})
    assert excinfo.value.details["code"] == "ObjectModeRequired"

    box.fake.context.mode = "OBJECT"
    box.obj.library = "//parts.blend"
    with pytest.raises(blender_runtime.ValidationError) as excinfo:
        box.session.apply_simready_metadata("box", {"physics": {"mass_kg": 1.0}})
    assert excinfo.value.details["code"] == "LinkedData"
    assert box.obj.props == {}


@pytest.mark.parametrize(
    ("metadata", "clear", "code"),
    [
        ({"physics": {"mass": 1}}, None, "InvalidMetadata"),
        ({}, None, "NothingToApply"),
        (None, ["mass"], "InvalidClear"),
    ],
)
def test_refusals_carry_a_code(box: Any, metadata: Any, clear: Any, code: str) -> None:
    with pytest.raises(blender_runtime.ValidationError) as excinfo:
        box.session.apply_simready_metadata("box", metadata, clear=clear)
    assert excinfo.value.details["code"] == code


@pytest.mark.parametrize(
    "call",
    [
        lambda s, _: s.apply_simready_metadata("ghost", {"physics": {"mass_kg": 1.0}}),
        lambda s, _: s.get_simready_metadata("ghost"),
        lambda s, _: s.validate_simready_compliance(["box", "ghost"]),
        lambda s, out: s.export_simready_usd(str(out / "ghost.usda"), object_names=["ghost"]),
        lambda s, _: s.setup_simready_hierarchy("asset", ["ghost"]),
    ],
    ids=["apply", "get", "validate", "export", "setup"],
)
def test_every_simready_tool_reports_an_unknown_object_as_not_found(
    box: Any, tmp_path: Path, call: Any
) -> None:
    box.fake.context.view_layer = SimpleNamespace(objects=box.fake.data.objects)
    policy = PathPolicy(enabled=True, allowed_paths=[str(tmp_path)], project_root=tmp_path)
    session = blender_runtime.BlenderRuntimeSession(path_policy=policy)

    with pytest.raises(blender_runtime.NotFoundError) as excinfo:
        call(session, tmp_path)

    # The code every Blender tool reports for an unknown object.
    assert excinfo.value.details["code"] == "object_not_found"
    assert excinfo.value.details["object_name"] == "ghost"
    assert excinfo.value.details.get("objects", ["ghost"]) == ["ghost"]
    assert "Object not found: ghost" in str(excinfo.value)
    assert box.obj.props == {}
