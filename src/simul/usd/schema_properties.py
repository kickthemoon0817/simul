"""What the USD schema registry declares a property name to be, whatever schemas a prim has.

A prim's own definition (its type and applied API schemas) knows only its own
properties. A name such as ``physics:mass`` or ``skel:skeleton`` belongs to an
API schema the prim may not have yet, and ``physics:body0`` to a prim type it
is not. The edit tools look the name up here so that a value for such a
property gets the schema's type, not one guessed from JSON, and a
relationship is never written as an attribute.

Multiple-apply schemas declare templates (``drive:__INSTANCE_NAME__:physics:stiffness``);
a name matches one with any instance name the registry allows, and the schema
is reported with that instance (``PhysicsDriveAPI:angular``).
``MaterialBindingAPI`` declares no properties, though every name in the
``material:binding`` namespace is one of its relationships.

API schemas are what a prim gains by having one applied, so their
declarations always count, and win over a prim type's. A prim type's
declaration counts only for a namespaced name (``physics:body0``,
``physics:axis``): plain names such as ``axis``, ``size``, ``type`` or
``camera`` are declared by some prim type or other, yet on any other prim they
are ordinary custom attributes. ``inputs:`` and ``outputs:`` names are shader
and light parameters, typed by the node that reads them, so prim types do not
type them either.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ._pxr import Sdf, Tf, Usd

_INSTANCE = "__INSTANCE_NAME__"
_MATERIAL_BINDING = "material:binding"
_MATERIAL_BINDING_API = "MaterialBindingAPI"
# Connectable namespaces whose types come from the shader or light node, not a prim type.
_NODE_NAMESPACES = ("inputs:", "outputs:")


@dataclass(frozen=True)
class Declaration:
    """One schema's declaration of a property.

    Attributes:
        schema: The schema name; a multiple-apply schema with its instance
            (``PhysicsDriveAPI:angular``).
        api: The schema is an applied API schema (it must be applied to a
            prim), not a prim type.
        relationship: Declared as a relationship.
        type_name: The attribute's value type; None for a relationship.
        variability: ``Sdf.VariabilityVarying`` or ``Sdf.VariabilityUniform``.
        allowed_tokens: The attribute's allowed tokens, if it lists any.
    """

    schema: str
    api: bool
    relationship: bool
    type_name: Optional[Sdf.ValueTypeName] = None
    variability: Any = Sdf.VariabilityVarying
    allowed_tokens: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SchemaProperty:
    """Every registered schema's declaration of one property name."""

    name: str
    declarations: Tuple[Declaration, ...]

    @property
    def relationship(self) -> bool:
        """Some schema declares it a relationship."""
        return any(declaration.relationship for declaration in self.declarations)

    @property
    def type_names(self) -> Tuple[Sdf.ValueTypeName, ...]:
        """The distinct attribute value types the schemas declare, in registry order."""
        types: List[Sdf.ValueTypeName] = []
        for declaration in self.declarations:
            if declaration.type_name is not None and declaration.type_name not in types:
                types.append(declaration.type_name)
        return tuple(types)

    @property
    def api_schemas(self) -> Tuple[str, ...]:
        """The API schemas that declare it: what a prim must have applied for USD to read it."""
        return tuple(dict.fromkeys(d.schema for d in self.declarations if d.api))

    @property
    def prim_types(self) -> Tuple[str, ...]:
        """The prim types that declare it."""
        return tuple(dict.fromkeys(d.schema for d in self.declarations if not d.api))

    def declaration_for(self, type_name: Sdf.ValueTypeName) -> Declaration:
        """The first declaration of ``type_name`` (its variability and allowed tokens)."""
        return next(d for d in self.declarations if d.type_name == type_name)

    def declared_by(self, limit: int = 3) -> str:
        """Name the declaring schemas for a message, API schemas first."""
        names = list(self.api_schemas) + list(self.prim_types)
        shown = ", ".join(names[:limit])
        return shown + (f" and {len(names) - limit} more" if len(names) > limit else "")


def _definition(type_: Tf.Type, name: str) -> Optional[Usd.PrimDefinition]:
    registry = Usd.SchemaRegistry()
    if Usd.SchemaRegistry.IsConcrete(type_):
        return registry.FindConcretePrimDefinition(name)
    if Usd.SchemaRegistry.IsAppliedAPISchema(type_):
        return registry.FindAppliedAPIPrimDefinition(name)
    if Usd.SchemaRegistry.IsTyped(type_):
        return registry.FindAbstractPrimDefinition(name)
    return None


def _declaration(schema: str, api: bool, spec: Any) -> Declaration:
    if isinstance(spec, Sdf.RelationshipSpec):
        return Declaration(schema, api, True)
    tokens = spec.GetInfo("allowedTokens") if spec.HasInfo("allowedTokens") else ()
    return Declaration(
        schema, api, False, spec.typeName, spec.variability, tuple(str(token) for token in tokens or ())
    )


@lru_cache(maxsize=1)
def _index() -> Tuple[Dict[str, Tuple[Declaration, ...]], Tuple[Tuple[str, str, str, Declaration], ...]]:
    """Every registered schema's properties: plain names, and multiple-apply templates.

    Returns:
        ``(names, templates)``: property name to its declarations, and one
        ``(prefix, suffix, schema, declaration)`` per template property.
    """
    names: Dict[str, List[Declaration]] = {}
    templates: List[Tuple[str, str, str, Declaration]] = []
    for type_ in Tf.Type.FindByName("UsdSchemaBase").GetAllDerivedTypes():
        schema = Usd.SchemaRegistry.GetSchemaTypeName(type_)
        definition = _definition(type_, schema) if schema else None
        if definition is None:
            continue
        api = bool(Usd.SchemaRegistry.IsAppliedAPISchema(type_))
        for property_name in definition.GetPropertyNames():
            spec = definition.GetSchemaPropertySpec(property_name)
            if spec is None:
                continue
            declaration = _declaration(schema, api, spec)
            if _INSTANCE in property_name:
                prefix, _, suffix = property_name.partition(_INSTANCE)
                templates.append((prefix, suffix, schema, declaration))
            else:
                names.setdefault(property_name, []).append(declaration)
    return (
        {name: tuple(declarations) for name, declarations in names.items()},
        tuple(templates),
    )


def _template_matches(name: str) -> Sequence[Declaration]:
    """Declarations of the multiple-apply templates ``name`` instantiates, the most specific only.

    ``collection:foo:includes`` fits both ``collection:__INSTANCE_NAME__:includes``
    (instance ``foo``) and ``collection:__INSTANCE_NAME__`` (instance
    ``foo:includes``); the shortest instance name is the one meant.
    """
    matches: List[Tuple[int, Declaration]] = []
    for prefix, suffix, schema, declaration in _index()[1]:
        if len(name) <= len(prefix) + len(suffix) or not name.startswith(prefix) or not name.endswith(suffix):
            continue
        instance = name[len(prefix):len(name) - len(suffix)]
        if instance.startswith(":") or instance.endswith(":"):
            continue
        if not Usd.SchemaRegistry.IsAllowedAPISchemaInstanceName(schema, instance):
            continue
        matches.append(
            (
                len(instance),
                Declaration(
                    f"{schema}:{instance}",
                    declaration.api,
                    declaration.relationship,
                    declaration.type_name,
                    declaration.variability,
                    declaration.allowed_tokens,
                ),
            )
        )
    if not matches:
        return ()
    shortest = min(length for length, _ in matches)
    return [declaration for length, declaration in matches if length == shortest]


def is_material_binding_name(name: str) -> bool:
    """Whether ``name`` is in UsdShade's ``material:binding`` namespace, where every property is a relationship."""
    return name == _MATERIAL_BINDING or name.startswith(_MATERIAL_BINDING + ":")


def lookup(name: str) -> Optional[SchemaProperty]:
    """Return what the registered schemas declare ``name`` to be, or None when none counts.

    API schema declarations win; a prim type's counts only for a namespaced
    name outside ``inputs:`` and ``outputs:`` (see the module docstring).

    Args:
        name: A property name, such as ``physics:mass`` or
            ``drive:angular:physics:stiffness``.
    """
    if is_material_binding_name(name):
        return SchemaProperty(name, (Declaration(_MATERIAL_BINDING_API, True, True),))
    declarations = tuple(_index()[0].get(name) or _template_matches(name))
    apis = tuple(declaration for declaration in declarations if declaration.api)
    if apis:
        declarations = apis
    elif ":" not in name or name.startswith(_NODE_NAMESPACES):
        return None
    return SchemaProperty(name, declarations) if declarations else None


def restricted_to(declared: SchemaProperty, type_name: Sdf.ValueTypeName) -> SchemaProperty:
    """``declared`` with only the declarations of ``type_name``."""
    chosen = tuple(declaration for declaration in declared.declarations if declaration.type_name == type_name)
    return SchemaProperty(declared.name, chosen)


__all__ = ["Declaration", "SchemaProperty", "is_material_binding_name", "lookup", "restricted_to"]
