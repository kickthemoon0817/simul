"""Prim Manipulation tools for Isaac Sim."""

import json
import textwrap
from typing import Any, Dict, Optional

from ...schemas.common import ErrorResponse
from ._shared import (
    COERCE_ATTRIBUTE_VALUE_CORE,
    DEFAULT_WORLD_PATH,
    DEFINE_PRIM_CORE,
    JSON_VALUE_CORE,
    SET_PRIM_TRANSFORM_CORE,
    STAGE_ROOT_PATH,
    XFORM_READ_CORE,
    FloatList,
    _compose_script,
    _pyval,
)
from .._meta import tool_meta

DELETE_MODES = ("auto", "remove", "deactivate")

# Attribute-name prefix of an OmniGraph script node's code (inputs:script) and
# script file (inputs:scriptPath). Writing either runs arbitrary Python inside
# Kit, which is what security.allow_script_execution=false exists to stop.
SCRIPT_ATTRIBUTE_PREFIX = "inputs:script"


def _is_script_capable_node_type(node_type: Any) -> bool:
    """Whether an OmniGraph ``node:type`` value names a node that runs Python.

    omni.graph.scriptnode.ScriptNode executes the code in its inputs:script,
    and Isaac's base .kit enables omni.graph.scriptnode on 5.1 and 6.0. Local
    stand-in for the graph lane's ``is_script_capable_node_type``; unify the
    two once that lands.
    """
    return isinstance(node_type, str) and "script" in node_type.lower()


# Where a prim's opinions live, for the tools that remove, copy or move it.
# Editing the root layer alone ignored the stage's edit target and every
# opinion that arrives through a reference: deleting a referenced robot link
# removed only the user's local override and reported success with the link
# still there. Needs ``Sdf``.
PRIM_LAYERING_CORE = """\
    def _ancestral_arc(prim):
        # A reference, payload, inherit, specialize or variant authored on an
        # ancestor brings this prim in. Its definition lives in that arc's
        # layers, which no edit on this stage removes or moves.
        try:
            pending = list(prim.GetPrimIndex().rootNode.children)
        except Exception:
            return None
        while pending:
            node = pending.pop()
            if node.IsDueToAncestor() and node.hasSpecs:
                return {
                    "arc": str(node.arcType).replace("Pcp.ArcType", "").lower(),
                    "introduced_at": str(node.GetIntroPath()),
                    "layer": node.layerStack.identifier.rootLayer.identifier,
                }
            pending.extend(node.children)
        return None

    def _spec_defines(spec):
        # An over without arcs only adjusts a prim; a def or class, or any
        # composition arc, keeps it in existence on its own.
        if spec.specifier != Sdf.SpecifierOver:
            return True
        return bool(
            spec.hasReferences or spec.hasPayloads or spec.hasInheritPaths or spec.hasSpecializes
            or spec.variantSetNameList.GetAddedOrExplicitItems()
        )

    def _local_specs(stage, prim):
        local = set(layer.identifier for layer in stage.GetLayerStack(True))
        return [
            spec for spec in prim.GetPrimStack()
            if spec.layer.identifier in local and spec.path == prim.GetPath()
        ]

    def _edit_blockers(stage, prim):
        # Why the edit target's opinions are not all there is to the prim:
        # removing or moving them would leave it defined where it was.
        path = prim.GetPath()
        blockers = []
        arc = _ancestral_arc(prim)
        if arc is not None:
            blockers.append(str(path) + " comes from a " + arc["arc"] + " authored on " + arc["introduced_at"]
                            + " (" + arc["layer"] + "); its definition lives there, not in this stage's layers")
        target = stage.GetEditTarget()
        target_layer = target.GetLayer()
        target_path = target.MapToSpecPath(path)
        for spec in _local_specs(stage, prim):
            if spec.layer == target_layer and spec.path == target_path:
                continue
            if _spec_defines(spec):
                blockers.append(str(path) + " is also defined in " + spec.layer.identifier
                                + ", outside the edit target (" + target_layer.identifier + ")")
        return blockers

    def _instance_refusal(prim, action):
        if prim.IsInstanceProxy():
            instance = prim.GetParent()
            while instance.IsInstanceProxy():
                instance = instance.GetParent()
            return {
                "error": (
                    str(prim.GetPath()) + " is inside the instance " + str(instance.GetPath())
                    + ", whose contents are shared and read-only here. " + action + " the instance "
                    "prim itself, or make it non-instanceable first."
                ),
                "instance": str(instance.GetPath()),
            }
        if prim.IsInPrototype():
            return {"error": str(prim.GetPath()) + " is an instancing prototype, which can't be edited."}
        return None

    def _kit_command(name, **kwargs):
        # A Kit command lands on Kit's undo stack; a raw pxr edit does not.
        try:
            import omni.kit.commands
        except Exception:
            return False
        try:
            omni.kit.commands.execute(name, **kwargs)
        except Exception:
            return False
        return True
"""

DELETE_PRIM_CORE = """\
    def _delete_prim(stage, prim_path, mode):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return {"error": "Prim not found: " + prim_path}
        refusal = _instance_refusal(prim, "Delete or deactivate")
        if refusal is not None:
            return refusal
        edit_layer = stage.GetEditTarget().GetLayer()
        result = {"prim_path": prim_path, "mode": mode, "edit_target": edit_layer.identifier}
        blockers = _edit_blockers(stage, prim)
        if mode == "remove" and blockers:
            return dict(result, blockers=blockers, error=(
                "Can't remove " + prim_path + ": " + "; ".join(blockers) + ". Nothing was changed. "
                "Pass mode='deactivate' (or 'auto') to deactivate it instead: it and its subtree leave "
                "traversal, rendering and physics, and its local overrides are kept."
            ))
        if mode == "deactivate" or blockers:
            was_active = prim.IsActive()
            prim.SetActive(False)
            after = stage.GetPrimAtPath(prim_path)
            if after.IsValid() and after.IsActive():
                return dict(result, error="Failed to deactivate " + prim_path)
            result.update({
                "action": "deactivated",
                "deleted": False,
                "deactivated": True,
                "was_active": was_active,
                "method": "pxr Usd.Prim.SetActive(False) at the edit target",
                "undoable": False,
                "note": (
                    "The prim stays on the stage with active=false (list it with include_inactive=true); "
                    "re-activate it with stage.GetPrimAtPath(path).SetActive(True) in execute_isaac_script."
                ),
            })
            if blockers:
                result["reason"] = "; ".join(blockers) + ", so it was deactivated rather than removed."
            return result

        def _still_defined():
            current = stage.GetPrimAtPath(prim_path)
            return current.IsValid() and current.IsDefined()

        method = "pxr Usd.Stage.RemovePrim at the edit target"
        undoable = False
        # Kit's DeletePrims is undoable; when the edit target holds every
        # local opinion it removes exactly what RemovePrim would.
        if all(spec.layer == edit_layer for spec in _local_specs(stage, prim)):
            if _kit_command("DeletePrims", paths=[prim_path]) and not _still_defined():
                method = "omni.kit.commands DeletePrims"
                undoable = True
        if _still_defined():
            stage.RemovePrim(prim_path)
        if _still_defined():
            return dict(result, error="Removing " + prim_path + " at the edit target left it defined")
        result.update({
            "action": "removed",
            "deleted": True,
            "deactivated": False,
            "method": method,
            "undoable": undoable,
        })
        leftover = stage.GetPrimAtPath(prim_path)
        if leftover.IsValid():
            result["note"] = (
                "An override (over) for the path remains in "
                + ", ".join(spec.layer.identifier for spec in leftover.GetPrimStack())
                + "; it defines nothing, so the prim is gone from traversal and rendering."
            )
        return result
"""

DUPLICATE_PRIM_CORE = """\
    def _arc_items(prim, dest_layer):
        # The references and payloads the prim's own opinions author (the
        # local ones and those from the ancestral arcs that bring it in), with
        # asset paths made absolute so they resolve from dest_layer. Arcs a
        # reference itself pulls in further down are left to it.
        references = []
        payloads = []
        pending = [prim.GetPrimIndex().rootNode]
        while pending:
            node = pending.pop()
            if not (node.IsRootNode() or node.IsDueToAncestor()):
                continue
            for node_layer in node.layerStack.layers:
                spec = node_layer.GetPrimAtPath(node.path)
                if not spec:
                    continue
                for items, cls, out in (
                    (spec.referenceList.GetAddedOrExplicitItems(), Sdf.Reference, references),
                    (spec.payloadList.GetAddedOrExplicitItems(), Sdf.Payload, payloads),
                ):
                    for item in items:
                        asset = item.assetPath
                        if asset:
                            asset = node_layer.ComputeAbsolutePath(asset)
                        elif node_layer != dest_layer:
                            asset = node_layer.identifier
                        out.append(cls(asset, item.primPath, item.layerOffset))
            pending.extend(node.children)
        return references, payloads

    def _restore_instances(stage, src_path, layer, dst_spec_path):
        # Stage.Flatten() points every instance at a /Flattened_Prototype_N
        # prim it writes beside the subtree. Only the subtree is copied, so
        # aim each copied instance back at the arcs its source composes from.
        restored = []
        pending = [layer.GetPrimAtPath(dst_spec_path)]
        while pending:
            spec = pending.pop()
            if not spec:
                continue
            flattened = [
                item for item in spec.referenceList.GetAddedOrExplicitItems()
                if not item.assetPath and str(item.primPath).startswith("/Flattened_Prototype")
            ]
            if flattened:
                source = stage.GetPrimAtPath(src_path.AppendPath(spec.path.MakeRelativePath(dst_spec_path)))
                references, payloads = _arc_items(source, layer)
                spec.referenceList.ClearEditsAndMakeExplicit()
                spec.referenceList.explicitItems = references
                if payloads:
                    spec.payloadList.ClearEditsAndMakeExplicit()
                    spec.payloadList.explicitItems = payloads
                restored.append(str(spec.path))
                continue
            pending.extend(spec.nameChildren)
        return restored

    def _duplicate_prim(stage, prim_path, new_path):
        src = stage.GetPrimAtPath(prim_path)
        if not src.IsValid():
            return {"error": "Source prim not found: " + prim_path}
        refusal = _instance_refusal(src, "Duplicate")
        if refusal is not None:
            return refusal
        valid = Sdf.Path.IsValidPathString(new_path)
        # Returns True, or a (False, reason) tuple, which is truthy.
        if isinstance(valid, tuple):
            valid = valid[0]
        if not valid or not Sdf.Path(new_path).IsAbsolutePath():
            return {"error": "Invalid destination path: " + new_path}
        dst_path = Sdf.Path(new_path)
        if not dst_path.IsPrimPath():
            return {"error": "Destination is not a prim path: " + new_path}
        if stage.GetPrimAtPath(dst_path).IsValid():
            return {"error": "Destination already exists: " + new_path}
        if dst_path.HasPrefix(src.GetPath()):
            return {"error": "Can't duplicate " + prim_path + " into its own subtree: " + new_path}
        parent = stage.GetPrimAtPath(dst_path.GetParentPath())
        if not parent.IsValid():
            return {"error": "Parent of the destination not found: " + str(dst_path.GetParentPath())}
        refusal = _instance_refusal(parent, "Duplicate into")
        if refusal is not None:
            return refusal
        target = stage.GetEditTarget()
        layer = target.GetLayer()
        dst_spec_path = target.MapToSpecPath(dst_path)
        if not dst_spec_path.GetParentPath().IsAbsoluteRootPath():
            # CopySpec needs the destination's parent spec in the layer; an
            # over adds none of its own opinions.
            Sdf.CreatePrimInLayer(layer, dst_spec_path.GetParentPath())
        blockers = _edit_blockers(stage, src)
        result = {"source_path": prim_path, "new_path": new_path, "edit_target": layer.identifier}
        if not blockers:
            # The edit target holds everything that defines the prim: copying
            # its spec copies the arcs too, so a referenced asset stays a
            # reference and an instance stays an instance.
            copied = Sdf.CopySpec(layer, target.MapToSpecPath(src.GetPath()), layer, dst_spec_path)
            result["method"] = "spec_copy"
        else:
            # Part of the prim is defined elsewhere (a referenced asset, another
            # layer): copying the edit target's spec alone gave an undefined
            # over nothing could see. Copy the composed result instead.
            masked = Usd.Stage.OpenMasked(
                stage.GetRootLayer(), stage.GetSessionLayer(), stage.GetPathResolverContext(),
                Usd.StagePopulationMask([src.GetPath()]), Usd.Stage.LoadAll,
            )
            flat = masked.Flatten(False)
            copied = Sdf.CopySpec(flat, src.GetPath(), layer, dst_spec_path)
            restored = _restore_instances(stage, src.GetPath(), layer, dst_spec_path) if copied else []
            result["method"] = "flattened_copy"
            result["reason"] = "; ".join(blockers)
            result["note"] = (
                "The copy is the source's composed result written into the edit target, so it no "
                "longer follows the arcs that define the source: later edits to that asset don't "
                "reach it. Instanceable prims inside keep their own references."
            )
            if restored:
                result["instances_restored"] = restored
        new_prim = stage.GetPrimAtPath(dst_path)
        if not copied or not new_prim.IsValid() or not new_prim.IsDefined():
            if layer.GetPrimAtPath(dst_spec_path):
                cleanup = Sdf.BatchNamespaceEdit()
                cleanup.Add(Sdf.NamespaceEdit.Remove(dst_spec_path))
                layer.Apply(cleanup)
            return dict(result, error="Duplicating " + prim_path + " produced no defined prim at " + new_path)
        result.update({
            "type": new_prim.GetTypeName(),
            "duplicated": True,
            "undoable": False,
        })
        return result
"""

REPARENT_PRIM_CORE = """\
    def _simple_xform_stack(xformable):
        Op = UsdGeom.XformOp
        rotate_types = (
            Op.TypeRotateXYZ, Op.TypeRotateXZY, Op.TypeRotateYXZ, Op.TypeRotateYZX,
            Op.TypeRotateZXY, Op.TypeRotateZYX, Op.TypeRotateX, Op.TypeRotateY,
            Op.TypeRotateZ, Op.TypeOrient,
        )
        for op in xformable.GetOrderedXformOps():
            if op.GetNumTimeSamples() > 0:
                return "it has animated xform ops (" + op.GetOpName() + ")"
            if op.GetOpType() == Op.TypeTransform:
                continue
            if op.IsInverseOp() or len(op.GetOpName().split(":")) != 2:
                return "its op stack has pivot or suffixed ops (" + op.GetOpName() + ")"
            if op.GetOpType() not in rotate_types + (Op.TypeTranslate, Op.TypeScale):
                return "its op stack has a " + op.GetOpName() + " op"
        return None

    def _retarget(stage, old_path, new_path):
        # A namespace edit moves specs, not the paths that point at them: a
        # moved material left its cube unbound, a moved link left its joint's
        # body0 on the old path. Rewrite every target and connection under the
        # old path, the moved subtree's own included.
        changed = []
        for prim in stage.TraverseAll():
            for rel in prim.GetAuthoredRelationships():
                targets = rel.GetTargets()
                if any(target.HasPrefix(old_path) for target in targets):
                    rel.SetTargets([
                        t.ReplacePrefix(old_path, new_path) if t.HasPrefix(old_path) else t for t in targets
                    ])
                    changed.append(str(rel.GetPath()))
            for attr in prim.GetAuthoredAttributes():
                if not attr.HasAuthoredConnections():
                    continue
                sources = attr.GetConnections()
                if any(source.HasPrefix(old_path) for source in sources):
                    attr.SetConnections([
                        s.ReplacePrefix(old_path, new_path) if s.HasPrefix(old_path) else s for s in sources
                    ])
                    changed.append(str(attr.GetPath()))
        return changed

    def _reparent_prim(stage, prim_path, new_parent_path, keep_world_transform):
        default_time = Usd.TimeCode.Default()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return {"error": "Prim not found: " + prim_path}
        if prim.IsPseudoRoot():
            return {"error": "The pseudo-root can't be reparented"}
        refusal = _instance_refusal(prim, "Move")
        if refusal is not None:
            return refusal
        parent = stage.GetPrimAtPath(new_parent_path)
        if not parent.IsValid():
            return {"error": "Parent not found: " + new_parent_path}
        refusal = _instance_refusal(parent, "Move into")
        if refusal is not None:
            return refusal
        old_path = prim.GetPath()
        # Read before the move: the Usd.Prim handle expires with its path.
        name = prim.GetName()
        # Sdf.Path joins "/" and "Box" as "/Box"; string concatenation made "//Box".
        new_path = parent.GetPath().AppendChild(name)
        result = {"old_path": str(old_path), "new_path": str(new_path), "parent": str(parent.GetPath())}
        if parent.GetPath() == old_path.GetParentPath():
            return dict(result, reparented=False, unchanged=True, note="Already a child of " + str(parent.GetPath()))
        if parent.GetPath().HasPrefix(old_path):
            return dict(result, error="Can't move " + prim_path + " under itself (" + new_parent_path + ")")
        if stage.GetPrimAtPath(new_path).IsValid():
            return dict(result, error="Destination already exists: " + str(new_path))
        blockers = _edit_blockers(stage, prim)
        if blockers:
            return dict(result, blockers=blockers, error=(
                "Can't move " + prim_path + ": " + "; ".join(blockers) + ". Nothing was changed. "
                "Duplicate it with duplicate_isaac_prim (a flattened copy) and deactivate the original "
                "with delete_isaac_prim, or edit the layer it comes from."
            ))

        # World pose to keep, and the local transform that keeps it under the
        # new parent, worked out before anything moves.
        xformable = UsdGeom.Xformable(prim) if prim.IsA(UsdGeom.Xformable) else None
        world_before = None
        keep_parts = None
        if keep_world_transform and xformable is not None and not xformable.GetResetXformStack():
            cache = UsdGeom.XformCache(default_time)
            world_before = cache.GetLocalToWorldTransform(prim)
            parent_world = Gf.Matrix4d(1) if parent.IsPseudoRoot() else cache.GetLocalToWorldTransform(parent)
            old_parent_world = cache.GetParentToWorldTransform(prim)
            if not Gf.IsClose(parent_world, old_parent_world, 1e-9):
                local = world_before * parent_world.GetInverse()
                parts = Gf.Transform(local)
                rebuilt = Gf.Transform()
                rebuilt.SetScale(parts.GetScale())
                rebuilt.SetRotation(parts.GetRotation())
                rebuilt.SetTranslation(parts.GetTranslation())
                problem = _simple_xform_stack(xformable)
                if problem is None and not Gf.IsClose(rebuilt.GetMatrix(), local, 1e-6):
                    problem = "keeping it under the new parent needs a sheared transform"
                if problem is not None:
                    return dict(result, error=(
                        "Can't keep the world transform of " + prim_path + " under " + str(parent.GetPath())
                        + ": " + problem + ". Nothing was changed. Pass keep_world_transform=false to move "
                        "it with its local transform unchanged."
                    ))
                z_y_x = parts.GetRotation().Decompose(Gf.Vec3d(0, 0, 1), Gf.Vec3d(0, 1, 0), Gf.Vec3d(1, 0, 0))
                keep_parts = (list(parts.GetTranslation()), list(reversed(z_y_x)), list(parts.GetScale()))

        def _moved():
            return stage.GetPrimAtPath(new_path).IsValid() and not stage.GetPrimAtPath(old_path).IsValid()

        method = None
        if _kit_command("MovePrim", path_from=str(old_path), path_to=str(new_path),
                        keep_world_transform=bool(keep_world_transform)) and _moved():
            method = "omni.kit.commands MovePrim"
        elif not _moved() and _kit_command("MovePrim", path_from=str(old_path), path_to=str(new_path)) and _moved():
            # A Kit too old for keep_world_transform; the pose is fixed below.
            method = "omni.kit.commands MovePrim"
        if not _moved():
            target = stage.GetEditTarget()
            layer = target.GetLayer()
            parent_spec_path = target.MapToSpecPath(parent.GetPath())
            if not parent_spec_path.IsAbsoluteRootPath():
                Sdf.CreatePrimInLayer(layer, parent_spec_path)
            edit = Sdf.BatchNamespaceEdit()
            edit.Add(Sdf.NamespaceEdit.ReparentAndRename(
                target.MapToSpecPath(old_path), parent_spec_path, name, -1
            ))
            if not layer.CanApply(edit) or not layer.Apply(edit):
                return dict(result, error="The namespace edit moving " + prim_path + " failed on " + layer.identifier)
            method = "pxr Sdf namespace edit on the edit target"
        if not _moved():
            return dict(result, error="Moving " + prim_path + " left it at " + prim_path)

        fixups = []
        retargeted = _retarget(stage, old_path, new_path)
        if retargeted:
            fixups.append("retargeted relationships/connections")
        kept = None
        if world_before is not None:
            moved_xf = UsdGeom.Xformable(stage.GetPrimAtPath(new_path))
            if not Gf.IsClose(moved_xf.ComputeLocalToWorldTransform(default_time), world_before, 1e-6) and keep_parts:
                outcome = _set_prim_transform(stage, str(new_path), keep_parts[0], keep_parts[1], keep_parts[2])
                if "error" in outcome:
                    result["world_transform_error"] = outcome["error"]
                fixups.append("adjusted the local transform")
            kept = Gf.IsClose(moved_xf.ComputeLocalToWorldTransform(default_time), world_before, 1e-5)
        result.update({
            "reparented": True,
            "method": method,
            "retargeted": retargeted[:50],
            "retargeted_count": len(retargeted),
            "world_transform_kept": kept,
            "undoable": method.startswith("omni.kit") and not fixups,
        })
        if method.startswith("omni.kit") and fixups:
            result["undo_note"] = (
                "The move is on Kit's undo stack; the " + " and ".join(fixups) + " step is not, so undoing "
                "the move leaves it behind."
            )
        default_prim = stage.GetRootLayer().defaultPrim
        if old_path.GetParentPath().IsAbsoluteRootPath() and default_prim == name:
            result["note"] = "It was the stage's default prim; defaultPrim still names " + default_prim + "."
        return result
"""


class PrimEditMixin:
    # ------------------------------------------------------------------
    # Phase 3: Prim Manipulation
    # ------------------------------------------------------------------

    def _script_write_refusal(
        self, attribute_name: str, value: Any
    ) -> Optional[Dict[str, Any]]:
        """Refuse an attribute write that would run Python while scripts are off.

        OmniGraph nodes are USD prims, so with
        ``security.allow_script_execution`` false a script could still be
        planted through the prim tools: as a script node's ``inputs:script``
        (or ``inputs:scriptPath``), or by setting ``node:type`` to a
        script-capable node type.

        Args:
            attribute_name: Attribute about to be written.
            value: Its new value.

        Returns:
            The ScriptExecutionDisabled payload, or None when the write may go.
        """
        if self.settings.security.allow_script_execution:
            return None
        if attribute_name.startswith(SCRIPT_ATTRIBUTE_PREFIX):
            reason = f"{attribute_name} holds code an OmniGraph script node runs"
        elif attribute_name == "node:type" and _is_script_capable_node_type(value):
            reason = f"node:type {value!r} is a script-capable OmniGraph node"
        else:
            return None
        return ErrorResponse(
            error=(
                f"Refusing the write: {reason}, and arbitrary script execution is "
                "disabled by security.allow_script_execution. Set "
                "SECURITY__ALLOW_SCRIPT_EXECUTION=true to allow it."
            ),
            error_type="ScriptExecutionDisabled",
        ).model_dump()

    @tool_meta(
        name="create_isaac_prim",
        description=(
            "Create a new USD prim at the given path with optional type (Xform, Scope, Mesh, "
            "Sphere, Cube, Cylinder, Cone, Capsule, Plane, Camera, ...). The type must be a "
            "registered concrete schema (a mistyped 'cube' is refused with a suggestion), "
            "and every attribute must be one the type's schema defines; otherwise nothing "
            "is created."
        ),
        read_only=False,
    )
    async def create_isaac_prim(
        self,
        prim_path: str,
        prim_type: str = "Xform",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new prim in the current Isaac Sim stage.

        Args:
            prim_path: USD path for the new prim (e.g. "/World/MyObject").
            prim_type: Concrete USD schema type name (e.g. "Xform", "Mesh",
                "Cube", "Sphere"); "" makes a typeless prim.
            attributes: Optional dict of attribute name→value to set on
                creation. Each must be defined by the type's schema; values
                convert like set_isaac_prim_attribute's.

        Returns:
            Dict confirming the created prim path and type.
        """
        for name, value in (attributes or {}).items():
            refusal = self._script_write_refusal(str(name), value)
            if refusal is not None:
                return refusal
        _prim_path = _pyval(prim_path)
        _prim_type = _pyval(prim_type)
        _attributes = _pyval(dict(attributes or {}))
        script = _compose_script(
            """\
            import json
            import omni.usd
            from pxr import Gf, Tf, Usd
            """,
            COERCE_ATTRIBUTE_VALUE_CORE,
            DEFINE_PRIM_CORE,
            f"""\
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                print(json.dumps(_define_prim(stage, {_prim_path}, {_prim_type}, {_attributes})))
            """,
        )
        return await self._execute_json_script(script)

    @tool_meta(
        name="delete_isaac_prim",
        description=(
            "Delete a prim and all its children at the stage's edit target. A prim that "
            "comes from a referenced asset (every robot link) or is also defined in "
            "another layer can't be removed from this stage: mode='auto' (default) "
            "deactivates it instead and says so, mode='remove' refuses, "
            "mode='deactivate' always deactivates (local overrides are kept). The reply's "
            "action is 'removed' or 'deactivated'. Refuses the pseudo-root '/'. Refuses "
            "'/World' unless allow_root_delete=true, since that removes the whole scene."
        ),
        read_only=False,
        destructive=True,
        idempotent=True,
    )
    async def delete_isaac_prim(
        self, prim_path: str, allow_root_delete: bool = False, mode: str = "auto"
    ) -> Dict[str, Any]:
        """
        Delete a prim and its children from the current stage.

        The pseudo-root ``/`` is never deleted. ``/World`` is refused unless
        ``allow_root_delete`` is set, because removing it empties the scene.

        Args:
            prim_path: USD path of the prim to delete.
            allow_root_delete: Permit deleting ``/World``.
            mode: "auto" removes the prim when the edit target holds
                everything that defines it and deactivates it otherwise;
                "remove" refuses rather than deactivate; "deactivate" always
                deactivates (the prim and its subtree leave traversal,
                rendering and physics, local overrides kept).

        Returns:
            Dict with ``action`` ("removed" or "deactivated"), ``deleted``,
            ``deactivated``, ``reason`` when the prim could not be removed,
            ``method`` and ``undoable``; or an error, or a RefusedOperation.
        """
        if mode not in DELETE_MODES:
            return ErrorResponse(
                error=f"Invalid mode {mode!r}. Must be one of: {', '.join(DELETE_MODES)}",
                error_type="ValueError",
            ).model_dump()
        normalized = prim_path.rstrip("/") or STAGE_ROOT_PATH
        if normalized == STAGE_ROOT_PATH:
            return self._refusal(
                "Refusing to delete the stage pseudo-root. Delete a child prim, or use "
                "new_isaac_stage to start over.",
                prim_path=prim_path,
            )
        if normalized == DEFAULT_WORLD_PATH and not allow_root_delete:
            return self._refusal(
                f"Refusing to delete {DEFAULT_WORLD_PATH}: it holds the whole scene. "
                "Pass allow_root_delete=true to delete it anyway.",
                prim_path=prim_path,
                override="allow_root_delete",
            )
        _prim_path = _pyval(prim_path)
        _mode = _pyval(mode)
        script = _compose_script(
            """\
            import json
            import omni.usd
            from pxr import Sdf, Usd
            """,
            PRIM_LAYERING_CORE,
            DELETE_PRIM_CORE,
            f"""\
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                print(json.dumps(_delete_prim(stage, {_prim_path}, {_mode})))
            """,
        )
        return await self._execute_json_script(script)

    @tool_meta(
        name="set_isaac_prim_transform",
        description=(
            "Set a prim's local transform (relative to its parent): translate, rotate (euler "
            "XYZ degrees), and/or scale. A matrix op (xformOp:transform) is converted to the "
            "equivalent translate/orient/scale stack first. An animated (time-sampled) op is "
            "refused unless time_code says which sample to write. The reply reads the result "
            "back: top-level values are local (space='local'), 'world' holds the world pose."
        ),
        read_only=False,
        destructive=True,
        idempotent=True,
    )
    async def set_isaac_prim_transform(
        self,
        prim_path: str,
        translation: Optional[FloatList] = None,
        rotation_euler: Optional[FloatList] = None,
        scale: Optional[FloatList] = None,
        time_code: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Set the local transform of a prim (translation, rotation, scale).

        Args:
            prim_path: USD path of the prim.
            translation: Local position as [x, y, z] in stage units (metres
                by default; Isaac Sim stages are Z-up), relative to the
                parent prim.
            rotation_euler: Local rotation in degrees as [x, y, z], applied
                in XYZ Euler order. Written to an existing orient or
                rotateXYZ op, otherwise to a new rotateXYZ op.
            scale: Local scale factors as [x, y, z]; 1.0 is unscaled.
            time_code: Time code (frame) to write the values at, for a prim
                whose transform ops are animated. Omit to write the default
                (static) value; an op that already has time samples is then
                refused, since the samples would hide the new value.

        Returns:
            Dict read back from the stage after the edit: ``space`` is
            ``"local"`` and the top-level ``translation``,
            ``rotation_quat`` ([w, x, y, z]), ``rotation_euler_xyz`` and
            ``scale`` are the local transform; ``world`` holds the same keys
            in world space. ``*_set`` echo the request and ``xform_op_order``
            the resulting op stack.
        """
        _prim_path = _pyval(prim_path)
        _translation = _pyval(list(translation) if translation is not None else None)
        _rotation = _pyval(list(rotation_euler) if rotation_euler is not None else None)
        _scale = _pyval(list(scale) if scale is not None else None)
        _time_code = _pyval(None if time_code is None else float(time_code))
        script = _compose_script(
            """\
            import json
            import omni.usd
            from pxr import Usd, UsdGeom, Gf
            """,
            XFORM_READ_CORE,
            SET_PRIM_TRANSFORM_CORE,
            f"""\
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                print(json.dumps(_set_prim_transform(
                    stage, {_prim_path}, {_translation}, {_rotation}, {_scale}, {_time_code}
                )))
            """,
        )
        return await self._execute_json_script(script)

    @tool_meta(
        name="set_isaac_prim_visibility",
        description="Show or hide a prim in the viewport.",
        read_only=False,
        destructive=True,
        idempotent=True,
    )
    async def set_isaac_prim_visibility(
        self, prim_path: str, visible: bool
    ) -> Dict[str, Any]:
        """
        Set the visibility of a prim.

        Args:
            prim_path: USD path of the prim.
            visible: True to make visible, False to hide.

        Returns:
            Dict confirming the visibility state.
        """
        _prim_path = _pyval(prim_path)
        _vis_token = _pyval("inherited" if visible else "invisible")
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import UsdGeom

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath({_prim_path})
                if not prim.IsValid():
                    print(json.dumps({{"error": "Prim not found: " + {_prim_path}}}))
                elif not prim.IsA(UsdGeom.Imageable):
                    print(json.dumps({{"error": "Prim is not Imageable: " + {_prim_path}}}))
                else:
                    img = UsdGeom.Imageable(prim)
                    img.GetVisibilityAttr().Set({_vis_token})
                    print(json.dumps({{
                        "prim_path": {_prim_path},
                        "visibility": {_vis_token},
                        "effective_visibility": img.ComputeVisibility(),
                    }}))
        """)
        return await self._execute_json_script(script)

    @tool_meta(
        name="set_isaac_prim_attribute",
        description=(
            "Set an arbitrary attribute value on a prim, overwriting its current value. The "
            "attribute must already exist. The JSON value is converted to the attribute's "
            "USD type: quaternions (xformOp:orient, physics:localRot0) as [w, x, y, z] with "
            "the real part first, matrices (xformOp:transform) as a list of rows or a flat "
            "row-major list, vectors as [x, y, z], arrays as lists of those. Writes the "
            "default (static) value; the reply warns when time samples hide it."
        ),
        read_only=False,
        destructive=True,
        idempotent=True,
    )
    async def set_isaac_prim_attribute(
        self,
        prim_path: str,
        attribute_name: str,
        value: Any,
    ) -> Dict[str, Any]:
        """
        Set a single attribute value on a prim.

        Args:
            prim_path: USD path of the prim.
            attribute_name: Name of the attribute to set.
            value: Value to set, as JSON, converted to the attribute's type:
                a quaternion is [w, x, y, z] (real part first), a matrix a
                list of rows or a flat row-major list (16 numbers for 4x4),
                a vector a list of its components.

        Returns:
            Dict with the attribute's ``type``, ``value_set`` as given and
            ``value_read`` read back from the stage as JSON (an array over
            16 elements as a length summary), plus ``warning`` when time
            samples override the default value written. With script
            execution disabled, a script node's ``inputs:script*`` and a
            script-capable ``node:type`` are refused.
        """
        refusal = self._script_write_refusal(attribute_name, value)
        if refusal is not None:
            return refusal
        _prim_path = _pyval(prim_path)
        _attr_name = _pyval(attribute_name)
        _val_str = _pyval(json.dumps(value))
        script = _compose_script(
            """\
            import json
            import omni.usd
            from pxr import Gf, Sdf
            """,
            JSON_VALUE_CORE,
            COERCE_ATTRIBUTE_VALUE_CORE,
            f"""\
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath({_prim_path})
                if not prim.IsValid():
                    print(json.dumps({{"error": "Prim not found: " + {_prim_path}}}))
                else:
                    attr = prim.GetAttribute({_attr_name})
                    if not attr.IsValid():
                        print(json.dumps({{"error": "Attribute not found: " + {_attr_name} + " on " + {_prim_path}}}))
                    else:
                        val = json.loads({_val_str})
                        type_label = str(attr.GetTypeName())
                        failure = None
                        try:
                            if not attr.Set(_coerce_attribute_value(attr, val)):
                                failure = "USD refused the value"
                        except Exception as e:
                            # A Tf error reads "Error in '<function>' at line N
                            # in file <path> : '<message>'"; keep the message.
                            text = str(e).strip()
                            failure = text.splitlines()[-1].split(" : ", 1)[-1].strip(" '") if text else ""
                            failure = failure or type(e).__name__
                        if failure is not None:
                            print(json.dumps({{
                                "error": "Failed to set " + {_attr_name} + " (" + type_label + "): " + failure,
                                "type": type_label,
                            }}))
                        else:
                            read_back = attr.Get()
                            if attr.GetTypeName().isArray and read_back is not None and len(read_back) > 16:
                                value_read = {{"length": len(read_back), "elements_omitted": True}}
                            else:
                                value_read = _json_value(read_back)
                            result = {{
                                "prim_path": {_prim_path},
                                "attribute": {_attr_name},
                                "type": type_label,
                                "value_set": val,
                                "value_read": value_read,
                            }}
                            samples = attr.GetNumTimeSamples()
                            if samples:
                                result["warning"] = (
                                    "The attribute has " + str(samples) + " time samples, which "
                                    "override the default value written here at every time."
                                )
                            print(json.dumps(result))
            """,
        )
        return await self._execute_json_script(script)

    @tool_meta(
        name="duplicate_isaac_prim",
        description=(
            "Duplicate a prim (deep copy) to a new path at the stage's edit target. A prim "
            "defined entirely there is copied with its composition arcs (a referenced "
            "asset stays a reference); one that comes from a referenced asset or another "
            "layer is copied as its flattened composed result (method='flattened_copy'), "
            "with instanceable prims inside kept instanced. Instance proxies are refused."
        ),
        read_only=False,
    )
    async def duplicate_isaac_prim(
        self, prim_path: str, new_path: str
    ) -> Dict[str, Any]:
        """
        Duplicate a prim to a new path.

        Args:
            prim_path: Source prim path.
            new_path: Destination prim path; its parent must exist.

        Returns:
            Dict with ``method`` ("spec_copy" or "flattened_copy"), the new
            prim's ``type`` and ``duplicated``; a flattened copy also gives
            ``reason`` and ``note``.
        """
        _prim_path = _pyval(prim_path)
        _new_path = _pyval(new_path)
        script = _compose_script(
            """\
            import json
            import omni.usd
            from pxr import Sdf, Usd
            """,
            PRIM_LAYERING_CORE,
            DUPLICATE_PRIM_CORE,
            f"""\
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                print(json.dumps(_duplicate_prim(stage, {_prim_path}, {_new_path})))
            """,
        )
        return await self._execute_json_script(script)

    @tool_meta(
        name="reparent_isaac_prim",
        description=(
            "Move a prim under a new parent in the hierarchy, keeping its world transform "
            "(keep_world_transform=true by default). Relationships and connections that "
            "point at the prim or into its subtree (material bindings, joint bodies) are "
            "retargeted to the new path. Uses Kit's undoable MovePrim when available, "
            "otherwise a namespace edit at the stage's edit target. A prim that comes from "
            "a referenced asset or is also defined in another layer can't be moved and is "
            "refused unchanged."
        ),
        read_only=False,
        destructive=True,
        idempotent=True,
    )
    async def reparent_isaac_prim(
        self, prim_path: str, new_parent_path: str, keep_world_transform: bool = True
    ) -> Dict[str, Any]:
        """
        Move a prim under a new parent.

        Args:
            prim_path: USD path of the prim to move.
            new_parent_path: USD path of the new parent prim ("/" for the
                stage root).
            keep_world_transform: Rewrite the prim's local transform so it
                stays where it is in the world. False keeps the local
                transform, so the prim moves with its new parent.

        Returns:
            Dict with the ``new_path``, ``method``, ``retargeted`` property
            paths, ``world_transform_kept`` and ``undoable``.
        """
        _prim_path = _pyval(prim_path)
        _new_parent = _pyval(new_parent_path)
        _keep = _pyval(bool(keep_world_transform))
        script = _compose_script(
            """\
            import json
            import omni.usd
            from pxr import Gf, Sdf, Usd, UsdGeom
            """,
            XFORM_READ_CORE,
            SET_PRIM_TRANSFORM_CORE,
            PRIM_LAYERING_CORE,
            REPARENT_PRIM_CORE,
            f"""\
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                print(json.dumps(_reparent_prim(stage, {_prim_path}, {_new_parent}, {_keep})))
            """,
        )
        return await self._execute_json_script(script)
