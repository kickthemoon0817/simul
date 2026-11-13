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

# OmniGraph nodes are USD prims, so with security.allow_script_execution off
# the prim tools are another way to plant code Kit runs. graph.py holds the
# one definition of "script-capable" (names server side, prims through
# script_guard_source() Kit side); the prim tools ask the same predicates.
from .graph import (
    _script_execution_refusal,
    is_script_attribute,
    is_script_capable_node_type,
    script_guard_source,
)

DELETE_MODES = ("auto", "remove", "deactivate")

# The refusal envelope a generated script prints when the Kit-side guard
# (script_guard_source) finds a code-running node. A ScriptNode's every input
# steers its code (inputs:usePath switches it to a file), so any write to such
# a node is refused, not only the code attribute.
SCRIPT_GUARD_REFUSAL_CORE = """\
    def _script_guard_error(message, **details):
        return {
            "error": (
                message + " Script execution is disabled by security.allow_script_execution; "
                "set SECURITY__ALLOW_SCRIPT_EXECUTION=true to allow it."
            ),
            "error_type": "ScriptExecutionDisabled",
            "details": details,
        }

    def _node_type_of(prim):
        attr = prim.GetAttribute("node:type")
        return str(attr.Get()) if attr and attr.IsValid() and attr.Get() is not None else None
"""

# Where a prim's opinions live, for the tools that remove, copy or move it.
# Editing the root layer alone ignored the stage's edit target and every
# opinion that arrives through a reference: deleting a referenced robot link
# removed only the user's local override and reported success with the link
# still there. Needs ``Sdf``.
PRIM_LAYERING_CORE = """\
    def _fail(error_type, code, message, **details):
        # Typed failure: error_type is the envelope's, details.code says which.
        details["code"] = code
        return {"error": message, "error_type": error_type, "details": details}

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

    def _session_layers(stage):
        return set(layer.identifier for layer in stage.GetLayerStack(True)) - set(
            layer.identifier for layer in stage.GetLayerStack(False)
        )

    def _session_only(stage, prim):
        # Kit keeps its own prims (the OmniverseKit_* viewport cameras) in the
        # session layer. They aren't part of the scene the user saves, and an
        # override written into the user's layer to hide one would be.
        session = _session_layers(stage)
        if stage.GetEditTarget().GetLayer().identifier in session:
            return False
        defining = [spec for spec in _local_specs(stage, prim) if _spec_defines(spec)]
        return bool(defining) and all(spec.layer.identifier in session for spec in defining)

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
            return _fail(
                "InvalidState", "INSTANCE_PROXY",
                str(prim.GetPath()) + " is inside the instance " + str(instance.GetPath())
                + ", whose contents are shared and read-only here. " + action + " the instance "
                "prim itself, or make it non-instanceable first.",
                instance=str(instance.GetPath()),
            )
        if prim.IsInPrototype():
            return _fail(
                "InvalidState", "PROTOTYPE",
                str(prim.GetPath()) + " is an instancing prototype, which can't be edited.",
            )
        return None

    def _kit_attempt(name, kwargs, optional, done):
        # Kit's commands land on its undo stack; a raw pxr edit doesn't.
        # "unavailable": no omni.kit.commands, so pxr does the edit.
        # "done": Kit did it. "declined": Kit ran and changed nothing, as it
        # does for a prim marked no_delete or one it treats as referenced
        # content; that refusal is respected, not overridden. "raised": Kit
        # failed. Only a TypeError naming one of the optional keywords (an
        # older Kit without it) earns another call without that keyword: a
        # call that returned and changed nothing is Kit declining, and
        # calling again with Kit's defaults (destructive=True edits every
        # local layer) would override it.
        try:
            import omni.kit.commands
        except Exception:
            return "unavailable", None
        call = dict(kwargs)
        dropped = []
        while True:
            try:
                omni.kit.commands.execute(name, **call)
            except TypeError as exc:
                unknown = [key for key in optional if key in call and key in str(exc)]
                if not unknown:
                    return "raised", "TypeError: " + str(exc).strip()
                dropped.extend(unknown)
                call = {key: value for key, value in call.items() if key not in unknown}
                continue
            except Exception as exc:
                return "raised", type(exc).__name__ + ": " + str(exc).strip()
            break
        if not done():
            return "declined", None
        if dropped:
            return "done", "this Kit's " + name + " takes no " + ", ".join(dropped) + "; called without"
        return "done", None
"""

DELETE_PRIM_CORE = """\
    def _delete_prim(stage, prim_path, mode):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return _fail("NotFoundError", "PRIM_NOT_FOUND", "Prim not found: " + prim_path)
        refusal = _instance_refusal(prim, "Delete or deactivate")
        if refusal is not None:
            return refusal
        if prim.GetMetadata("no_delete") or prim.GetCustomDataByKey("no_delete"):
            return _fail("InvalidState", "NO_DELETE", prim_path + " is marked no_delete; nothing was changed.")
        if _session_only(stage, prim):
            return _fail(
                "InvalidState", "SESSION_LAYER_PRIM",
                prim_path + " is defined only in the session layer (Kit's own prims, such as the "
                "OmniverseKit_* viewport cameras, live there). It isn't part of the saved scene, and "
                "deactivating it would write an override into the scene's layer. Nothing was changed.",
            )
        edit_layer = stage.GetEditTarget().GetLayer()
        result = {"prim_path": prim_path, "mode": mode, "edit_target": edit_layer.identifier}
        blockers = _edit_blockers(stage, prim)
        if mode == "remove" and blockers:
            return dict(result, **_fail(
                "InvalidState", "DEFINED_ELSEWHERE",
                "Can't remove " + prim_path + ": " + "; ".join(blockers) + ". Nothing was changed. "
                "Pass mode='deactivate' (or 'auto') to deactivate it instead: it and its subtree leave "
                "traversal, rendering and physics, and its local overrides are kept.",
                blockers=blockers,
            ))
        if mode == "deactivate" or blockers:
            was_active = prim.IsActive()
            prim.SetActive(False)
            after = stage.GetPrimAtPath(prim_path)
            if after.IsValid() and after.IsActive():
                return dict(result, **_fail("InvalidState", "DEACTIVATE_FAILED", "Failed to deactivate " + prim_path))
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
        # Kit's DeletePrims is undoable. Called only when the edit target
        # holds every local opinion, where it removes what RemovePrim would;
        # destructive=False keeps it to the edit target as well.
        if all(spec.layer == edit_layer for spec in _local_specs(stage, prim)):
            status, kit_note = _kit_attempt(
                "DeletePrims", {"paths": [prim_path], "destructive": False}, ("destructive",),
                lambda: not _still_defined(),
            )
            if status == "declined":
                return dict(result, **_fail(
                    "InvalidState", "KIT_DECLINED",
                    "Kit's DeletePrims declined to delete " + prim_path + " (Kit refuses prims it "
                    "protects, such as ones marked no_delete); nothing was changed.",
                ))
            if status == "raised":
                return dict(result, **_fail(
                    "InvalidState", "KIT_ERROR", "Kit's DeletePrims failed on " + prim_path + ": " + kit_note
                ))
            if status == "done":
                method = "omni.kit.commands DeletePrims"
                undoable = True
                if kit_note:
                    result["kit_note"] = kit_note
        if _still_defined():
            stage.RemovePrim(prim_path)
        if _still_defined():
            return dict(result, **_fail(
                "InvalidState", "REMOVE_FAILED", "Removing " + prim_path + " at the edit target left it defined"
            ))
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

    def _duplicate_prim(stage, prim_path, new_path, script_guard):
        src = stage.GetPrimAtPath(prim_path)
        if not src.IsValid():
            return _fail("NotFoundError", "PRIM_NOT_FOUND", "Source prim not found: " + prim_path)
        refusal = _instance_refusal(src, "Duplicate")
        if refusal is not None:
            return refusal
        if script_guard:
            nodes = [
                {"path": path, "node_type": _node_type_of(stage.GetPrimAtPath(path)) or ""}
                for path in _script_capable_prims_under(src)
            ]
            if nodes:
                return _script_guard_error(
                    "Refusing to duplicate " + prim_path + ": it holds OmniGraph nodes that run code ("
                    + ", ".join(node["path"] + " " + node["node_type"] for node in nodes) + ").",
                    nodes=nodes,
                )
        valid = Sdf.Path.IsValidPathString(new_path)
        # Returns True, or a (False, reason) tuple, which is truthy.
        if isinstance(valid, tuple):
            valid = valid[0]
        if not valid or not Sdf.Path(new_path).IsAbsolutePath():
            return _fail("ValidationError", "INVALID_PATH", "Invalid destination path: " + new_path)
        dst_path = Sdf.Path(new_path)
        if not dst_path.IsPrimPath():
            return _fail("ValidationError", "INVALID_PATH", "Destination is not a prim path: " + new_path)
        if stage.GetPrimAtPath(dst_path).IsValid():
            return _fail("ValidationError", "DESTINATION_EXISTS", "Destination already exists: " + new_path)
        if dst_path.HasPrefix(src.GetPath()):
            return _fail(
                "ValidationError", "INVALID_DESTINATION",
                "Can't duplicate " + prim_path + " into its own subtree: " + new_path,
            )
        parent = stage.GetPrimAtPath(dst_path.GetParentPath())
        if not parent.IsValid():
            return _fail(
                "NotFoundError", "PARENT_NOT_FOUND",
                "Parent of the destination not found: " + str(dst_path.GetParentPath()),
            )
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
            return dict(result, **_fail(
                "InvalidState", "COPY_FAILED",
                "Duplicating " + prim_path + " produced no defined prim at " + new_path,
            ))
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

    def _retarget_arcs(layer, old_path, new_path):
        # Internal references and payloads, inherits and specializes in the
        # edit target that name the old path or something under it.
        changed = []
        prim_paths = []
        layer.Traverse(Sdf.Path.absoluteRootPath, lambda path: prim_paths.append(path) if path.IsPrimPath() else None)
        for spec_path in prim_paths:
            spec = layer.GetPrimAtPath(spec_path)
            if not spec:
                continue
            for proxy, kind in ((spec.inheritPathList, "inherits"), (spec.specializesList, "specializes")):
                for item in list(proxy.GetAddedOrExplicitItems()):
                    if item.HasPrefix(old_path):
                        proxy.ReplaceItemEdits(item, item.ReplacePrefix(old_path, new_path))
                        changed.append(str(spec_path) + " " + kind)
            for proxy, cls, kind in (
                (spec.referenceList, Sdf.Reference, "references"),
                (spec.payloadList, Sdf.Payload, "payload"),
            ):
                for item in list(proxy.GetAddedOrExplicitItems()):
                    if not item.assetPath and not item.primPath.isEmpty and item.primPath.HasPrefix(old_path):
                        moved = item.primPath.ReplacePrefix(old_path, new_path)
                        if cls is Sdf.Reference:
                            replacement = Sdf.Reference("", moved, item.layerOffset, item.customData)
                        else:
                            replacement = Sdf.Payload("", moved, item.layerOffset)
                        proxy.ReplaceItemEdits(item, replacement)
                        changed.append(str(spec_path) + " " + kind)
        return changed

    def _reparent_prim(stage, prim_path, new_parent_path, keep_world_transform):
        default_time = Usd.TimeCode.Default()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return _fail("NotFoundError", "PRIM_NOT_FOUND", "Prim not found: " + prim_path)
        if prim.IsPseudoRoot():
            return _fail("ValidationError", "INVALID_SOURCE", "The pseudo-root can't be reparented")
        refusal = _instance_refusal(prim, "Move")
        if refusal is not None:
            return refusal
        parent = stage.GetPrimAtPath(new_parent_path)
        if not parent.IsValid():
            return _fail("NotFoundError", "PARENT_NOT_FOUND", "Parent not found: " + new_parent_path)
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
            return dict(result, **_fail(
                "ValidationError", "INVALID_DESTINATION",
                "Can't move " + prim_path + " under itself (" + new_parent_path + ")",
            ))
        if stage.GetPrimAtPath(new_path).IsValid():
            return dict(result, **_fail(
                "ValidationError", "DESTINATION_EXISTS", "Destination already exists: " + str(new_path)
            ))
        blockers = _edit_blockers(stage, prim)
        if blockers:
            return dict(result, **_fail(
                "InvalidState", "DEFINED_ELSEWHERE",
                "Can't move " + prim_path + ": " + "; ".join(blockers) + ". Nothing was changed. "
                "Duplicate it with duplicate_isaac_prim (a flattened copy) and deactivate the original "
                "with delete_isaac_prim, or edit the layer it comes from.",
                blockers=blockers,
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
                    return dict(result, **_fail(
                        "InvalidState", "WORLD_TRANSFORM_NOT_KEEPABLE",
                        "Can't keep the world transform of " + prim_path + " under " + str(parent.GetPath())
                        + ": " + problem + ". Nothing was changed. Pass keep_world_transform=false to move "
                        "it with its local transform unchanged.",
                    ))
                z_y_x = parts.GetRotation().Decompose(Gf.Vec3d(0, 0, 1), Gf.Vec3d(0, 1, 0), Gf.Vec3d(1, 0, 0))
                keep_parts = (list(parts.GetTranslation()), list(reversed(z_y_x)), list(parts.GetScale()))

        def _moved():
            # An over left at the old path in another layer defines nothing.
            moved_to = stage.GetPrimAtPath(new_path)
            left = stage.GetPrimAtPath(old_path)
            return moved_to.IsValid() and moved_to.IsDefined() and not (left.IsValid() and left.IsDefined())

        # Kit's MovePrim is undoable. Called only when the edit target holds
        # every opinion that defines the prim (checked above);
        # destructive=False keeps it to the edit target as well.
        status, kit_note = _kit_attempt(
            "MovePrim",
            {
                "path_from": str(old_path), "path_to": str(new_path),
                "keep_world_transform": bool(keep_world_transform), "destructive": False,
            },
            ("keep_world_transform", "destructive"),
            _moved,
        )
        if status == "declined":
            return dict(result, **_fail(
                "InvalidState", "KIT_DECLINED",
                "Kit's MovePrim declined to move " + prim_path + "; nothing was changed.",
            ))
        if status == "raised":
            return dict(result, **_fail(
                "InvalidState", "KIT_ERROR", "Kit's MovePrim failed on " + prim_path + ": " + kit_note
            ))
        if status == "done":
            method = "omni.kit.commands MovePrim"
            if kit_note:
                result["kit_note"] = kit_note
        else:
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
                return dict(result, **_fail(
                    "InvalidState", "NAMESPACE_EDIT_FAILED",
                    "The namespace edit moving " + prim_path + " failed on " + layer.identifier,
                ))
            method = "pxr Sdf namespace edit on the edit target"
        if not _moved():
            return dict(result, **_fail("InvalidState", "MOVE_FAILED", "Moving " + prim_path + " left it in place"))

        fixups = []
        retargeted = _retarget(stage, old_path, new_path)
        if not method.startswith("omni.kit"):
            # Kit's MovePrim rewrites the arcs that name a moved prim; a bare
            # namespace edit leaves them on the old path.
            retargeted += _retarget_arcs(stage.GetEditTarget().GetLayer(), old_path, new_path)
        if retargeted:
            fixups.append("retargeted relationships/connections")
        kept = None
        if world_before is not None:
            moved_xf = UsdGeom.Xformable(stage.GetPrimAtPath(new_path))
            if Gf.IsClose(moved_xf.ComputeLocalToWorldTransform(default_time), world_before, 1e-6):
                kept = True
            elif keep_parts:
                # Written by a second script: the transform core would take
                # this one past the size Isaac Sim 5.1's socket accepts.
                result["pending_world_fix"] = {
                    "translation": keep_parts[0],
                    "rotation_euler": keep_parts[1],
                    "scale": keep_parts[2],
                    "world": [world_before[i][j] for i in range(4) for j in range(4)],
                }
            else:
                kept = False
        result.update({
            "reparented": True,
            "method": method,
            "retargeted": retargeted[:50],
            "retargeted_count": len(retargeted),
            "world_transform_kept": kept,
            "fixups": fixups,
        })
        default_prim = stage.GetRootLayer().defaultPrim
        if old_path.GetParentPath().IsAbsoluteRootPath() and default_prim == name:
            result["note"] = "It was the stage's default prim; defaultPrim still names " + default_prim + "."
        return result
"""


class PrimEditMixin:
    # ------------------------------------------------------------------
    # Phase 3: Prim Manipulation
    # ------------------------------------------------------------------

    @property
    def _script_guard(self) -> bool:
        """Whether writes that would plant code must be refused."""
        return not self.settings.security.allow_script_execution

    def _script_write_refusal(
        self, attribute_name: str, value: Any
    ) -> Optional[Dict[str, Any]]:
        """Refuse an attribute write that would plant code while scripts are off.

        Covers what is decidable without the stage: a code-carrying attribute
        name, or ``node:type`` set to a node type that runs code. The scripts
        add the check that needs the stage (the target prim's own
        ``node:type``).

        Args:
            attribute_name: Attribute about to be written.
            value: Its new value.

        Returns:
            The ScriptExecutionDisabled payload, or None when the write may go.
        """
        if not self._script_guard:
            return None
        if is_script_attribute(attribute_name):
            return _script_execution_refusal(
                f"Refusing to write {attribute_name}: it carries code or a path to code "
                "an OmniGraph node runs.",
                attribute=attribute_name,
            )
        if attribute_name == "node:type" and is_script_capable_node_type(value):
            return _script_execution_refusal(
                f"Refusing to set node:type to {value!r}: that node runs code.",
                attribute=attribute_name,
                node_type=value,
            )
        return None

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
            Dict confirming the created prim path and type, and any missing
            ancestors it created (``ancestors_created``).
        """
        if self._script_guard and is_script_capable_node_type(prim_type):
            return _script_execution_refusal(
                f"Refusing to create a {prim_type!r} prim: that type runs code.", prim_type=prim_type
            )
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
            from pxr import Gf, Sdf, Tf, Usd
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
            "action is 'removed' or 'deactivated'. Removal goes through Kit's undoable "
            "DeletePrims (destructive=false) when available, and a refusal from Kit is "
            "reported, not overridden. Refuses prims marked no_delete, prims defined only in "
            "the session layer (Kit's own viewport cameras), the pseudo-root '/', and '/World' "
            "unless allow_root_delete=true, since that removes the whole scene."
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
                error_type="ValidationError",
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
        # A placement made while paused after a step is reverted on stop.
        return await self._execute_json_script(script, annotations=("paused_edit",))

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
            "default (static) value; the reply warns when time samples hide it. With "
            "script execution disabled, writes that would plant code are refused: "
            "code-carrying attributes (inputs:script*, inputs:codeStr, ...), a node:type "
            "that runs code, and any attribute of such a node."
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
            execution disabled, a code-carrying attribute, a node:type that
            runs code and any attribute of such a node are refused
            (``ScriptExecutionDisabled``).
        """
        refusal = self._script_write_refusal(attribute_name, value)
        if refusal is not None:
            return refusal
        _prim_path = _pyval(prim_path)
        _attr_name = _pyval(attribute_name)
        _val_str = _pyval(json.dumps(value))
        _guard = _pyval(self._script_guard)
        script = _compose_script(
            """\
            import json
            import omni.usd
            from pxr import Gf, Sdf, Usd
            """,
            JSON_VALUE_CORE,
            COERCE_ATTRIBUTE_VALUE_CORE,
            script_guard_source(),
            SCRIPT_GUARD_REFUSAL_CORE,
            f"""\
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath({_prim_path})
                guarded_type = None
                if {_guard} and prim.IsValid() and _script_capable_prim(prim):
                    guarded_type = _node_type_of(prim) or "code-running"
                if not prim.IsValid():
                    print(json.dumps({{
                        "error": "Prim not found: " + {_prim_path},
                        "error_type": "NotFoundError",
                        "details": {{"code": "PRIM_NOT_FOUND"}},
                    }}))
                elif guarded_type:
                    # Every input of a node that runs code steers that code.
                    print(json.dumps(_script_guard_error(
                        "Refusing to write " + {_attr_name} + " on " + {_prim_path} + ": it is a "
                        + guarded_type + " node, which runs code.",
                        node_type=guarded_type,
                    )))
                else:
                    attr = prim.GetAttribute({_attr_name})
                    if not attr.IsValid():
                        print(json.dumps({{
                            "error": "Attribute not found: " + {_attr_name} + " on " + {_prim_path},
                            "error_type": "NotFoundError",
                            "details": {{"code": "ATTRIBUTE_NOT_FOUND"}},
                        }}))
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
                                "error_type": "ValidationError",
                                "details": {{"code": "VALUE_REJECTED", "type": type_label}},
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
        # An edit made while paused after a step is reverted on stop.
        return await self._execute_json_script(script, annotations=("paused_edit",))

    @tool_meta(
        name="duplicate_isaac_prim",
        description=(
            "Duplicate a prim (deep copy) to a new path at the stage's edit target. A prim "
            "defined entirely there is copied with its composition arcs (a referenced "
            "asset stays a reference); one that comes from a referenced asset or another "
            "layer is copied as its flattened composed result (method='flattened_copy'), "
            "with instanceable prims inside kept instanced. Instance proxies are refused. "
            "With script execution disabled, a subtree holding OmniGraph nodes that run "
            "code is refused."
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
        _guard = _pyval(self._script_guard)
        script = _compose_script(
            """\
            import json
            import omni.usd
            from pxr import Sdf, Usd
            """,
            script_guard_source(),
            SCRIPT_GUARD_REFUSAL_CORE,
            PRIM_LAYERING_CORE,
            DUPLICATE_PRIM_CORE,
            f"""\
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                print(json.dumps(_duplicate_prim(stage, {_prim_path}, {_new_path}, {_guard})))
            """,
        )
        return await self._execute_json_script(script)

    @tool_meta(
        name="reparent_isaac_prim",
        description=(
            "Move a prim under a new parent in the hierarchy, keeping its world transform "
            "(keep_world_transform=true by default). Relationships and connections that "
            "point at the prim or into its subtree (material bindings, joint bodies) are "
            "retargeted to the new path. Uses Kit's undoable MovePrim (destructive=false) "
            "when available, and reports a refusal from Kit rather than overriding it; "
            "without Kit, a namespace edit at the stage's edit target. A prim that comes "
            "from a referenced asset or is also defined in another layer can't be moved "
            "and is refused unchanged."
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
        result = await self._execute_json_script(script)
        fixups = list(result.pop("fixups", None) or [])
        pending = result.pop("pending_world_fix", None)
        if pending is not None:
            # The move left the world pose behind (no Kit to keep it): write
            # the local transform that restores it, in a script of its own.
            fix = await self._execute_json_script(
                _compose_script(
                    """\
                    import json
                    import omni.usd
                    from pxr import Gf, Usd, UsdGeom
                    """,
                    XFORM_READ_CORE,
                    SET_PRIM_TRANSFORM_CORE,
                    f"""\
                    stage = omni.usd.get_context().get_stage()
                    path = {_pyval(result.get("new_path"))}
                    outcome = _set_prim_transform(
                        stage, path, {_pyval(pending["translation"])},
                        {_pyval(pending["rotation_euler"])}, {_pyval(pending["scale"])}
                    )
                    expected = Gf.Matrix4d(*{_pyval(pending["world"])})
                    world = UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(
                        Usd.TimeCode.Default()
                    )
                    print(json.dumps({{
                        "kept": "error" not in outcome and Gf.IsClose(world, expected, 1e-5),
                        "transform_error": outcome.get("error"),
                    }}))
                    """,
                )
            )
            fixups.append("adjusted the local transform")
            result["world_transform_kept"] = bool(fix.get("kept"))
            failure = fix.get("transform_error") or (fix.get("error") if fix.get("success") is False else None)
            if failure or not fix.get("kept"):
                result["world_transform_error"] = failure or "The world pose did not come back after the move"
                result["success"] = False
        if result.get("reparented"):
            method = str(result.get("method") or "")
            result["undoable"] = method.startswith("omni.kit") and not fixups
            if method.startswith("omni.kit") and fixups:
                result["undo_note"] = (
                    "The move is on Kit's undo stack; the " + " and ".join(fixups)
                    + " step is not, so undoing the move leaves it behind."
                )
        return result
