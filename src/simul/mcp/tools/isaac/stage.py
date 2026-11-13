"""Asset & Stage Operations tools for Isaac Sim."""

import textwrap
from typing import Any, Dict, List, Optional

from ...schemas.common import ErrorResponse
from ._shared import (
    OPEN_ASSET_CORE,
    _compose_script,
    _pyval,
)
from ._physics_units import physics_units_source
from .graph import script_guard_source
from .._meta import SandboxedPath, tool_meta

# What the in-place save is about to write, read before anything is written so
# the sandbox can police it: the stage's own file and every other dirty layer
# a save writes along with it, plus the timeline state.
SAVE_PROBE_SCRIPT = textwrap.dedent("""\
    import json
    import omni.usd

    ctx = omni.usd.get_context()
    stage = ctx.get_stage()
    if stage is None:
        print(json.dumps({"error": "No stage is currently open"}))
    else:
        try:
            import omni.timeline
            timeline_stopped = bool(omni.timeline.get_timeline_interface().is_stopped())
        except Exception:
            timeline_stopped = None
        session = stage.GetSessionLayer()
        dirty = sorted({
            layer.identifier for layer in stage.GetUsedLayers()
            if layer.dirty and not layer.anonymous and layer != session
        })
        print(json.dumps({
            "stage_url": ctx.get_stage_url() or "",
            "dirty_layers": dirty,
            "timeline_stopped": timeline_stopped,
        }))
""")

# Shared by the save scripts: refuse while the timeline is not stopped, since a
# playing or paused simulation has written simulated poses into the stage that
# stopping it would reset. Defines _timeline_refusal(allow).
TIMELINE_GUARD_CORE = """\
    def _timeline_refusal(allow):
        try:
            import omni.timeline
            stopped = omni.timeline.get_timeline_interface().is_stopped()
        except Exception:
            return None
        if stopped or allow:
            return None
        return {
            "error": (
                "The timeline is playing or paused, so the stage holds simulated poses that "
                "stopping it would reset. Stop the simulation first, or pass "
                "allow_while_playing=true to save the current state."
            ),
            "error_type": "RefusedOperation",
            "override": "allow_while_playing",
        }
"""

# File types Kit does not open as USD layers; named in the error so the caller
# knows to convert first rather than retry the path.
CONVERT_FIRST_SUFFIXES: tuple[str, ...] = (".obj", ".fbx", ".gltf", ".glb", ".stl", ".dae", ".3ds", ".ply")

# Resolves which prim of an asset a reference targets, adds the reference,
# checks that it composed something and holds no code-running node, and
# corrects or reports a metersPerUnit / upAxis mismatch. Every failure undoes
# what the call created or changed. Defines _import_asset(...) and
# _reference_on_prim(...); needs script_guard_source() for the node check.
REFERENCE_ASSET_CORE = """\
    import math

    def _unique_path(stage, path):
        if not stage.GetPrimAtPath(path).IsValid():
            return path
        index = 1
        while stage.GetPrimAtPath(path + "_" + str(index)).IsValid():
            index += 1
        return path + "_" + str(index)

    def _first_missing(stage, path):
        # The top-most prim on the way to path that does not exist yet:
        # removing it undoes every ancestor DefinePrim creates along with path.
        current = Sdf.Path(path)
        top = None
        while current != Sdf.Path.absoluteRootPath and not stage.GetPrimAtPath(current).IsValid():
            top = current
            current = current.GetParentPath()
        return top

    def _failure(message, error_type, code=None, **extra):
        result = {"error": message, "error_type": error_type}
        if code is not None:
            result["details"] = {"code": code}
        result.update(extra)
        return result

    def _asset_root(layer, asset_prim_path):
        # (prim path to reference or None for the defaultPrim, its name, warning, failure)
        if asset_prim_path is not None:
            if layer.GetPrimAtPath(asset_prim_path) is None:
                return None, None, None, _failure(
                    "No prim " + asset_prim_path + " in " + layer.identifier + "; nothing was changed",
                    "NotFoundError",
                )
            return asset_prim_path, Sdf.Path(asset_prim_path).name, None, None
        if layer.defaultPrim:
            return None, str(layer.defaultPrim).strip("/").split("/")[-1], None, None
        for spec in layer.rootPrims:
            if spec.specifier == Sdf.SpecifierDef:
                return str(spec.path), spec.name, (
                    "The asset has no defaultPrim; referenced its first root prim "
                    + str(spec.path) + ". Pass asset_prim_path to choose another."
                ), None
        return None, None, None, _failure(
            "The asset has no defaultPrim and no root prim to reference: " + layer.identifier
            + "; nothing was changed",
            "ValidationError", "ASSET_HAS_NO_PRIMS",
        )

    def _metrics(stage, layer):
        root = layer.pseudoRoot
        asset_meters = float(root.GetInfo("metersPerUnit")) if root.HasInfo("metersPerUnit") else None
        asset_up = str(root.GetInfo("upAxis")) if root.HasInfo("upAxis") else None
        stage_meters = float(UsdGeom.GetStageMetersPerUnit(stage)) or 1.0
        stage_up = str(UsdGeom.GetStageUpAxis(stage))
        scale = None
        if asset_meters is not None and not math.isclose(asset_meters, stage_meters, rel_tol=1e-9):
            scale = asset_meters / stage_meters
        rotate_x = None
        if asset_up is not None and asset_up != stage_up:
            rotate_x = 90.0 if (asset_up, stage_up) == ("Y", "Z") else -90.0
        # kilogramsPerUnit has a defined fallback (1.0, kilograms), unlike
        # metersPerUnit, so an asset that does not author it is in kilograms.
        asset_kilograms = float(root.GetInfo("kilogramsPerUnit")) if root.HasInfo("kilogramsPerUnit") else 1.0
        stage_kilograms = float(UsdPhysics.GetStageKilogramsPerUnit(stage)) or 1.0
        notes = []
        if asset_meters is None:
            notes.append("the asset does not author metersPerUnit (USD's fallback is centimetres); no scale applied")
        if asset_up is None:
            notes.append("the asset does not author upAxis; no rotation applied")
        return {
            "asset_meters_per_unit": asset_meters,
            "asset_up_axis": asset_up,
            "asset_kilograms_per_unit": asset_kilograms,
            "stage_meters_per_unit": stage_meters,
            "stage_up_axis": stage_up,
            "stage_kilograms_per_unit": stage_kilograms,
            "scale_needed": scale,
            "rotate_x_needed": rotate_x,
            "mass_ratio": asset_kilograms / stage_kilograms,
            "applied": False,
            "notes": notes,
        }

    def _physics_units(root, units, apply):
        # PhysX reads physics values in stage units whatever the transform
        # does. The tool's second script, which carries the units table,
        # rescales (or, when not applying, only counts) the values in asset
        # units; this records what it needs.
        length_ratio = units["scale_needed"] if units["scale_needed"] is not None else 1.0
        mass_ratio = units["mass_ratio"]
        if math.isclose(length_ratio, 1.0, rel_tol=1e-12) and math.isclose(mass_ratio, 1.0, rel_tol=1e-12):
            return
        units["physics"] = {
            "pending": str(root.GetPath()),
            "applied": bool(apply),
            "length_ratio": length_ratio,
            "mass_ratio": mass_ratio,
        }

    def _needs_correction(units):
        return units["scale_needed"] is not None or units["rotate_x_needed"] is not None

    def _add_correction_ops(prim, units):
        # Suffixed ops, so set_isaac_prim_transform leaves them alone, at the
        # end of the op order: innermost, converting before any transform the
        # caller sets on this prim.
        xformable = UsdGeom.Xformable(prim)
        if units["rotate_x_needed"] is not None:
            attr = prim.GetAttribute("xformOp:rotateX:unitsResolve")
            op = UsdGeom.XformOp(attr) if attr.IsValid() else xformable.AddRotateXOp(
                UsdGeom.XformOp.PrecisionFloat, "unitsResolve")
            op.Set(units["rotate_x_needed"])
        if units["scale_needed"] is not None:
            scale = units["scale_needed"]
            attr = prim.GetAttribute("xformOp:scale:unitsResolve")
            op = UsdGeom.XformOp(attr) if attr.IsValid() else xformable.AddScaleOp(
                UsdGeom.XformOp.PrecisionFloat, "unitsResolve")
            op.Set(Gf.Vec3f(scale, scale, scale))
        units["applied"] = True
        units["correction_prim"] = str(prim.GetPath())
        units["xform_op_order"] = [op.GetOpName() for op in xformable.GetOrderedXformOps()]

    def _references_undo(stage, prim):
        # Restores the edit target's references list op on prim as it is now.
        target = stage.GetEditTarget()
        layer = target.GetLayer()
        spec_path = target.MapToSpecPath(prim.GetPath())
        spec = layer.GetPrimAtPath(spec_path)
        had = spec is not None and spec.HasInfo("references")
        previous = spec.GetInfo("references") if had else None

        def _undo():
            current = layer.GetPrimAtPath(spec_path)
            if current is None:
                return
            if had:
                current.SetInfo("references", previous)
            else:
                current.ClearInfo("references")

        return _undo

    def _add_reference(prim, asset_path, root_path):
        # "already", "nothing" (and undone) or "added"
        reference = Sdf.Reference(asset_path, root_path) if root_path else Sdf.Reference(asset_path)
        existing = prim.GetMetadata("references")
        for item in (existing.GetAppliedItems() if existing else []):
            if item.assetPath == reference.assetPath and item.primPath == reference.primPath:
                return "already"
        before = len(prim.GetPrimStack())
        prim.GetReferences().AddReference(reference)
        if len(prim.GetPrimStack()) <= before:
            return "nothing"
        return "added"

    def _composed_nothing(asset_path, root_path, root_name):
        return _failure(
            "The reference to " + asset_path + " composed nothing"
            + (" (prim " + root_path + ")" if root_path else " (its defaultPrim " + str(root_name) + ")")
            + "; nothing was changed",
            "ValidationError", "REFERENCE_COMPOSED_NOTHING",
        )

    def _code_nodes_refusal(found):
        return _failure(
            "The asset contains prim(s) that run Python: OmniGraph nodes that run code given "
            "as data, or Python Scripting Components (OmniScriptingAPI / omni:scripting:scripts, "
            "which Kit loads without asking once scripts are allowed) ("
            + ", ".join(found[:5]) + "). Script execution is disabled by "
            "security.allow_script_execution; nothing was changed.",
            "ScriptExecutionDisabled", None, script_capable_prims=found,
        )

    def _unloaded_payloads_refusal(found):
        return _failure(
            "The asset contains payload(s) that are not loaded (" + ", ".join(found[:5])
            + "), so what they would bring in, code included, cannot be checked. Script "
            "execution is disabled by security.allow_script_execution; nothing was changed.",
            "ScriptExecutionDisabled", None, unloaded_payload_prims=found,
        )

    def _guard_refusal(prim, guard):
        # None when the referenced content may stay.
        if not guard:
            return None
        found = _script_capable_prims_under(prim)
        if found:
            return _code_nodes_refusal(found)
        unloaded = _unloaded_payload_prims_under(prim)
        if unloaded:
            return _unloaded_payloads_refusal(unloaded)
        return None

    def _import_asset(stage, asset_path, requested, asset_prim_path, correct_units, guard):
        layer = Sdf.Layer.FindOrOpen(asset_path)
        root_path, root_name, warning, failure = _asset_root(layer, asset_prim_path)
        if failure is not None:
            return failure
        units = _metrics(stage, layer)
        target = _unique_path(stage, requested)
        created = _first_missing(stage, target)
        wrapped = correct_units and _needs_correction(units)
        if wrapped:
            # The correction sits on a wrapper Xform with the reference on its
            # child, so the asset root's own transform (in asset units) is
            # converted too, and a transform the caller later sets on the
            # wrapper cannot drop it.
            wrapper = UsdGeom.Xform.Define(stage, target).GetPrim()
            _add_correction_ops(wrapper, units)
            holder = stage.DefinePrim(Sdf.Path(target).AppendChild(root_name))
        else:
            holder = stage.DefinePrim(target)
            if _needs_correction(units):
                units["notes"].append("mismatch reported, not corrected (correct_units=false)")
        if _add_reference(holder, asset_path, root_path) != "added":
            stage.RemovePrim(created)
            return _composed_nothing(asset_path, root_path, root_name)
        refusal = _guard_refusal(holder, guard)
        if refusal is not None:
            stage.RemovePrim(created)
            return refusal
        _physics_units(holder, units, correct_units)
        result = {
            "referenced_prim": root_path or "/" + root_name,
            "used_default_prim": root_path is None,
            "asset_root_path": str(holder.GetPath()),
            "prim_type": holder.GetTypeName(),
            "child_count": len(holder.GetChildren()),
            "units": units,
        }
        if warning is not None:
            result["warning"] = warning
        return result

    def _reference_on_prim(stage, prim_path, asset_path, asset_prim_path, correct_units, guard):
        layer = Sdf.Layer.FindOrOpen(asset_path)
        root_path, root_name, warning, failure = _asset_root(layer, asset_prim_path)
        if failure is not None:
            return failure
        prim = stage.GetPrimAtPath(prim_path)
        created = None
        if not prim.IsValid():
            created = _first_missing(stage, prim_path)
            prim = stage.DefinePrim(prim_path)
        if prim.IsInstanceProxy():
            return _failure("Prim is inside an instance and cannot be edited here: " + prim_path, "ValidationError",
                            "INSTANCE_PROXY")
        undo = _references_undo(stage, prim)
        prior = prim.GetMetadata("references")
        had_references = bool(prior and list(prior.GetAppliedItems()))
        # What the prim held before this reference is in the stage's units;
        # a correction here would convert it along with the asset.
        prior_children = [child.GetName() for child in prim.GetAllChildren()]
        prior_values = [attr.GetName() for attr in prim.GetAttributes() if attr.HasAuthoredValue()]

        def _roll_back():
            if created is not None:
                stage.RemovePrim(created)
            else:
                undo()

        status = _add_reference(prim, asset_path, root_path)
        if status == "already":
            return {"already_referenced": True, "referenced_prim": root_path or "/" + root_name}
        if status == "nothing":
            _roll_back()
            return _composed_nothing(asset_path, root_path, root_name)
        refusal = _guard_refusal(prim, guard)
        if refusal is not None:
            _roll_back()
            return refusal
        units = _metrics(stage, layer)
        declined = None
        mismatch = _needs_correction(units) or units["mass_ratio"] != 1.0
        # By namespace: the units table travels in the second script only.
        prior_physics = [
            name for name in prior_values
            if name.split(":")[0].startswith(("physics", "physx")) or name.split(":")[0] in ("drive", "limit", "state")
        ]
        if correct_units and mismatch and (prior_children or prior_physics):
            held = []
            if prior_children:
                held.append("children (" + ", ".join(prior_children[:5])
                            + (", ..." if len(prior_children) > 5 else "") + ")")
            if prior_physics:
                held.append("physics values (" + ", ".join(prior_physics[:5])
                            + (", ..." if len(prior_physics) > 5 else "") + ")")
            declined = (
                "correct_units was not applied: the prim already had " + " and ".join(held)
                + " in the stage's units, which a correction here would convert too. "
                "import_isaac_asset wraps the reference in a corrected Xform instead."
            )
        elif correct_units and had_references and mismatch:
            # Values from the prim's other references are in their own units;
            # rescaling the subtree here would convert them too.
            declined = (
                "correct_units was not applied: the prim also references other assets, whose "
                "values a correction here would convert too. import_isaac_asset wraps the "
                "reference in a corrected Xform instead."
            )
        elif _needs_correction(units) and correct_units:
            own_ops = [
                op.GetOpName() for op in UsdGeom.Xformable(prim).GetOrderedXformOps()
                if not op.GetOpName().endswith(":unitsResolve")
            ] if prim.IsA(UsdGeom.Xformable) else None
            if own_ops is None:
                declined = (
                    "correct_units was not applied: the prim is a " + prim.GetTypeName()
                    + ", not Xformable."
                )
            elif own_ops:
                # Ops from the asset root are in asset units; corrected in
                # place they would apply unconverted.
                declined = (
                    "correct_units was not applied: the prim already has transform ops ("
                    + ", ".join(own_ops) + "), which a correction here would leave unconverted. "
                    "import_isaac_asset wraps the reference in a corrected Xform instead."
                )
            else:
                _add_correction_ops(prim, units)
        elif _needs_correction(units):
            units["notes"].append("mismatch reported, not corrected (correct_units=false)")
        if declined is not None:
            units["notes"].append(declined)
        _physics_units(prim, units, correct_units and declined is None)
        result = {
            "already_referenced": False,
            "referenced_prim": root_path or "/" + root_name,
            "used_default_prim": root_path is None,
            "prim_type": prim.GetTypeName(),
            "child_count": len(prim.GetChildren()),
            "units": units,
        }
        # The caller asked for the correction, so a refusal to apply it is
        # surfaced at the top level, not only inside units.notes.
        warnings = [text for text in (warning, declined) if text is not None]
        if warnings:
            result["warning"] = " ".join(warnings)
        return result
"""


def _convert_first_hint(path: str) -> str:
    """Return the conversion hint for a non-USD asset path, or an empty string.

    Args:
        path: The asset path or URL the caller passed.

    Returns:
        A sentence naming the converter for OBJ/FBX/glTF-style files, or "".
    """
    if path.lower().endswith(CONVERT_FIRST_SUFFIXES):
        return (
            " Only USD layers (.usd, .usda, .usdc, .usdz) are referenced: convert OBJ, FBX, glTF "
            "and similar files to USD first (omni.kit.asset_converter, through "
            "execute_isaac_script)."
        )
    return ""


class StageAssetMixin:
    # ------------------------------------------------------------------
    # Phase 8: Asset & Stage Operations
    # ------------------------------------------------------------------

    @tool_meta(
        name="open_isaac_stage",
        description=(
            "Open a USD stage file in Isaac Sim. Refused while the current stage has "
            "unsaved edits unless discard_unsaved=true. "
            "Paths: sandbox (security.allowed_paths); URLs need an allowed scheme."
        ),
        read_only=False,
        destructive=True,
        idempotent=True,
        sandboxed_paths=(SandboxedPath("file_path"),),
    )
    async def open_isaac_stage(
        self, file_path: str, discard_unsaved: bool = False
    ) -> Dict[str, Any]:
        """
        Open a USD stage file in Isaac Sim.

        Args:
            file_path: Path or URL to the USD file.
            discard_unsaved: Replace the current stage even when it has
                unsaved edits. Off by default so an open cannot silently
                throw away work.

        Returns:
            Dict confirming the stage was opened, or a RefusedOperation error
            when unsaved edits would be lost.
        """
        denial = self._sandbox_denial(file_path)
        if denial is not None:
            return denial
        _file_path = _pyval(self._path_policy.authorize(file_path))
        _discard = _pyval(discard_unsaved)
        script = textwrap.dedent(f"""\
            import json
            import omni.usd

            ctx = omni.usd.get_context()
            if ctx.has_pending_edit() and not {_discard}:
                print(json.dumps({{
                    "error": "The current stage has unsaved edits. Save it with save_isaac_stage, "
                             "or pass discard_unsaved=true to open " + {_file_path} + " anyway.",
                    "error_type": "RefusedOperation",
                    "current_stage": ctx.get_stage_url(),
                    "override": "discard_unsaved",
                }}))
            else:
                result, error = await ctx.open_stage_async({_file_path})
                if result:
                    stage = ctx.get_stage()
                    total = sum(1 for _ in stage.Traverse()) if stage else 0
                    print(json.dumps({{
                        "file_path": {_file_path},
                        "opened": True,
                        "total_prims": total,
                    }}))
                else:
                    print(json.dumps({{"error": f"Failed to open stage: {{error}}"}}))
        """)
        # A large stage takes longer to load than the default budget, and an
        # interrupted open still finishes loading inside Kit.
        return await self._execute_json_script(
            script, timeout_seconds=self._heavy_script_timeout()
        )

    @tool_meta(
        name="save_isaac_stage",
        description=(
            "Save the current stage. In place (no file_path), the stage's file and every other "
            "dirty layer the save writes must be inside the write sandbox. With file_path, "
            "save-as there; an existing local file is refused unless overwrite=true, and a URL "
            "target always needs overwrite=true because its existence cannot be checked. "
            "Refused while the timeline is playing or paused (simulated poses would be saved) "
            "unless allow_while_playing=true. "
            "Paths: sandbox (security.allowed_paths); URLs need an allowed write scheme."
        ),
        read_only=False,
        destructive=True,
        idempotent=True,
        sandboxed_paths=(SandboxedPath("file_path", write=True),),
    )
    async def save_isaac_stage(
        self,
        file_path: Optional[str] = None,
        overwrite: bool = False,
        allow_while_playing: bool = False,
    ) -> Dict[str, Any]:
        """
        Save the current stage, in place or to file_path.

        An in-place save first asks Isaac Sim which files it would write (the
        stage's own file and any other dirty layer) and checks each against
        the write sandbox. A save-as target that already exists is refused
        unless ``overwrite`` is set; the check runs inside Isaac Sim because
        the file system is Isaac's, and a URL target, whose existence cannot
        be checked from there, needs ``overwrite`` in every case.

        Args:
            file_path: Optional path to save to. None saves to current location.
            overwrite: Replace an existing file at ``file_path``; required for
                any URL target.
            allow_while_playing: Save even while the timeline is playing or
                paused. Off by default: the stage then holds simulated poses
                that stopping the simulation would reset.

        Returns:
            Dict confirming the stage was saved, a SandboxError naming the
            layer outside the sandbox, or a RefusedOperation error naming the
            override that would let the save through.
        """
        denial = self._sandbox_denial(file_path, write=True)
        if denial is not None:
            return denial
        _allow = _pyval(bool(allow_while_playing))
        if file_path:
            target = self._path_policy.authorize(file_path, write=True)
            if self._path_policy.url_scheme(target) not in (None, "file") and not overwrite:
                return self._refusal(
                    f"Refusing to save to {target} without overwrite=true: whether a file "
                    "already exists at a URL cannot be checked before writing it.",
                    file_path=target,
                    override="overwrite",
                )
            _file_path = _pyval(target)
            _overwrite = _pyval(overwrite)
            script = _compose_script(
                """\
                import json
                import os
                import omni.usd
                """,
                TIMELINE_GUARD_CORE,
                f"""\
                ctx = omni.usd.get_context()
                target = {_file_path}
                current = ctx.get_stage_url() or ""
                refusal = _timeline_refusal({_allow})
                same_file = os.path.exists(target) and os.path.exists(current) and os.path.samefile(target, current)
                if refusal is not None:
                    print(json.dumps(refusal))
                elif os.path.exists(target) and not same_file and not {_overwrite}:
                    print(json.dumps({{
                        "error": "Refusing to overwrite existing file " + target + ". "
                                 "Pass overwrite=true to replace it.",
                        "error_type": "RefusedOperation",
                        "file_path": target,
                        "override": "overwrite",
                    }}))
                else:
                    result, error, saved_paths = await ctx.save_as_stage_async(target)
                    if result:
                        print(json.dumps({{"file_path": target, "saved": True}}))
                    else:
                        print(json.dumps({{"error": f"Failed to save stage: {{error}}"}}))
                """,
            )
            # Writing a large stage outlasts the default budget; a timeout
            # would report failure for a save Kit then finishes.
            return await self._execute_json_script(script, timeout_seconds=self._heavy_script_timeout())

        probe = await self._execute_json_script(SAVE_PROBE_SCRIPT)
        if not probe.get("success"):
            return probe
        url = str(probe.get("stage_url") or "")
        if not url or url.startswith("anon:"):
            return ErrorResponse(
                error="The stage has never been saved, so it has no file to save in place. Pass file_path.",
                error_type="ValidationError",
            ).model_dump()
        layers: List[str] = [url] + [layer for layer in probe.get("dirty_layers") or [] if layer != url]
        for layer in layers:
            layer_denial = self._sandbox_denial(layer, write=True)
            if layer_denial is not None:
                layer_denial["details"]["layer"] = layer
                layer_denial["details"]["note"] = (
                    "An in-place save writes the stage's file and every dirty layer; "
                    "this one is outside the write sandbox, so nothing was saved."
                )
                return layer_denial
        if probe.get("timeline_stopped") is False and not allow_while_playing:
            return self._refusal(
                "The timeline is playing or paused, so the stage holds simulated poses that "
                "stopping it would reset. Stop the simulation first, or pass "
                "allow_while_playing=true to save the current state.",
                file_path=url,
                override="allow_while_playing",
            )
        script = _compose_script(
            """\
            import json
            import omni.usd
            """,
            TIMELINE_GUARD_CORE,
            f"""\
            ctx = omni.usd.get_context()
            expected = {_pyval(url)}
            refusal = _timeline_refusal({_allow})
            if (ctx.get_stage_url() or "") != expected:
                print(json.dumps({{
                    "error": "The open stage changed since it was checked (now "
                             + str(ctx.get_stage_url()) + "); nothing was saved",
                }}))
            elif refusal is not None:
                print(json.dumps(refusal))
            else:
                result, error, saved_paths = await ctx.save_stage_async()
                if result:
                    print(json.dumps({{
                        "file_path": expected,
                        "saved": True,
                        "layers_checked": {_pyval(layers)},
                    }}))
                else:
                    print(json.dumps({{"error": f"Failed to save stage: {{error}}"}}))
            """,
        )
        # Writing a large stage outlasts the default budget; a timeout
        # would report failure for a save Kit then finishes.
        return await self._execute_json_script(script, timeout_seconds=self._heavy_script_timeout())

    @tool_meta(
        name="new_isaac_stage",
        description=(
            "Create a new empty stage in Isaac Sim. Refused while the current stage has "
            "unsaved edits unless discard_unsaved=true."
        ),
        read_only=False,
        destructive=True,
        idempotent=True,
    )
    async def new_isaac_stage(self, discard_unsaved: bool = False) -> Dict[str, Any]:
        """
        Create a new empty stage in Isaac Sim.

        Args:
            discard_unsaved: Replace the current stage even when it has
                unsaved edits. Off by default so a reset cannot silently
                throw away work.

        Returns:
            Dict confirming the new stage was created, or a RefusedOperation
            error when unsaved edits would be lost.
        """
        _discard = _pyval(discard_unsaved)
        script = textwrap.dedent(f"""\
            import json
            import omni.usd

            ctx = omni.usd.get_context()
            if ctx.has_pending_edit() and not {_discard}:
                print(json.dumps({{
                    "error": "The current stage has unsaved edits. Save it with save_isaac_stage, "
                             "or pass discard_unsaved=true to replace it with an empty stage.",
                    "error_type": "RefusedOperation",
                    "current_stage": ctx.get_stage_url(),
                    "override": "discard_unsaved",
                }}))
            else:
                result, error = await ctx.new_stage_async()
                if result:
                    print(json.dumps({{"created": True, "new_stage": True}}))
                else:
                    print(json.dumps({{"error": f"Failed to create new stage: {{error}}"}}))
        """)
        # Closing a large stage first is what takes the time.
        return await self._execute_json_script(
            script, timeout_seconds=self._heavy_script_timeout()
        )

    @tool_meta(
        name="import_isaac_asset",
        description=(
            "Import a USD asset (.usd, .usda, .usdc, .usdz) into the current stage as a "
            "reference on a new prim. An existing prim at target_path is never merged into: "
            "the import goes to the first free target_path_N, reported as target_path. "
            "Without a defaultPrim the asset's first root prim is referenced (reported), or "
            "pass asset_prim_path; a reference that composes nothing is undone and fails. "
            "A metersPerUnit/upAxis mismatch the asset authors is corrected: target_path "
            "becomes an Xform carrying xformOp:scale:unitsResolve / "
            "xformOp:rotateX:unitsResolve and the asset is referenced on its child "
            "<target_path>/<asset root name>, reported as asset_root_path "
            "(correct_units=false only reports it). With security.allow_script_execution "
            "off, an asset containing code-running OmniGraph nodes or Python Scripting "
            "Components (OmniScriptingAPI), or payloads that are not loaded, is refused. "
            "OBJ/FBX/glTF need converting to USD first. "
            "Paths: sandbox (security.allowed_paths); URLs need an allowed scheme."
        ),
        read_only=False,
        sandboxed_paths=(SandboxedPath("asset_path"),),
    )
    async def import_isaac_asset(
        self,
        asset_path: str,
        target_path: str = "/World/ImportedAsset",
        asset_prim_path: Optional[str] = None,
        correct_units: bool = True,
    ) -> Dict[str, Any]:
        """
        Import a USD asset into the current stage as a reference on a new prim.

        Args:
            asset_path: File path or URL to the USD asset.
            target_path: USD path for the new prim. When a prim already exists
                there the import goes to the first free ``<target_path>_N``.
            asset_prim_path: Prim inside the asset to reference. Defaults to
                the asset's defaultPrim, or its first root prim when it has
                none.
            correct_units: When the asset authors a metersPerUnit or upAxis
                that differs from the stage's, make ``target_path`` an Xform
                with a uniform ``xformOp:scale:unitsResolve`` (asset / stage
                metersPerUnit) and an ``xformOp:rotateX:unitsResolve``
                (+-90 degrees), and reference the asset on its child named
                after the asset root, so the root's own transform is
                converted too. With false the mismatch is only reported.
                Metrics the asset does not author are not guessed at.

        Returns:
            Dict with the ``target_path`` used (and ``requested_target_path``),
            ``asset_root_path`` (where the asset's root prim composes:
            ``target_path``, or its child when a correction was applied), the
            ``referenced_prim``, the composed ``prim_type`` and
            ``child_count``, and ``units`` (both sides' metrics, the correction
            needed and whether it was applied).
        """
        problem = _asset_prim_path_problem(asset_prim_path)
        if problem is not None:
            return problem
        denial = self._sandbox_denial(asset_path)
        if denial is not None:
            return denial
        _asset_path = _pyval(self._path_policy.authorize(asset_path))
        script = _compose_script(
            """\
            import json
            import omni.usd
            from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
            """,
            OPEN_ASSET_CORE,
            script_guard_source(),
            REFERENCE_ASSET_CORE,
            f"""\
            asset_path = {_asset_path}
            requested = {_pyval(target_path)}
            asset_prim_path = {_pyval(asset_prim_path)}
            correct_units = {_pyval(bool(correct_units))}
            guard = {_pyval(not self.settings.security.allow_script_execution)}
            convert_hint = {_pyval(_convert_first_hint(asset_path))}
            """,
            """\
            stage = omni.usd.get_context().get_stage()
            asset_error = _open_asset(asset_path) if stage is not None else None
            if stage is None:
                print(json.dumps({"error": "No stage is currently open"}))
            elif asset_error is not None:
                print(json.dumps({"error": asset_error + convert_hint, "error_type": "NotFoundError"}))
            elif not Sdf.Path.IsValidPathString(requested) or not Sdf.Path(requested).IsAbsolutePath():
                print(json.dumps({
                    "error": "target_path is not an absolute prim path: " + requested,
                    "error_type": "ValidationError",
                }))
            else:
                target = _unique_path(stage, requested)
                result = _import_asset(stage, asset_path, requested, asset_prim_path, correct_units, guard)
                if "error" not in result:
                    result.update({
                        "asset_path": asset_path,
                        "target_path": target,
                        "requested_target_path": requested,
                        "renamed": target != requested,
                        "imported": True,
                    })
                print(json.dumps(result))
            """,
        )
        return await self._rescale_physics_values(await self._execute_json_script(script))

    @tool_meta(
        name="add_isaac_reference",
        description=(
            "Add a USD reference to a prim (created when missing) so it composes in content "
            "from another USD file. Without a defaultPrim the asset's first root prim is "
            "referenced (reported), or pass asset_prim_path; a reference that composes nothing "
            "is undone and fails, and the same reference twice is reported, not duplicated. "
            "A metersPerUnit/upAxis mismatch is reported; correct_units=true adds unitsResolve "
            "ops and rescales the asset's physics values only when the prim had no children, "
            "physics values, other references or transform ops of its own (otherwise use "
            "import_isaac_asset, which wraps the reference). With "
            "security.allow_script_execution off, an asset containing code-running OmniGraph "
            "nodes or Python Scripting Components (OmniScriptingAPI), or payloads that are not "
            "loaded, is refused. "
            "Paths: sandbox (security.allowed_paths); URLs need an allowed scheme."
        ),
        read_only=False,
        sandboxed_paths=(SandboxedPath("reference_path"),),
    )
    async def add_isaac_reference(
        self,
        prim_path: str,
        reference_path: str,
        asset_prim_path: Optional[str] = None,
        correct_units: bool = False,
    ) -> Dict[str, Any]:
        """
        Add a USD reference to a prim.

        Args:
            prim_path: USD path of the prim to add the reference to; created
                as a typeless prim when it does not exist.
            reference_path: File path or URL to the referenced USD asset.
            asset_prim_path: Prim inside the asset to reference. Defaults to
                the asset's defaultPrim, or its first root prim when it has
                none.
            correct_units: Correct a metersPerUnit/upAxis mismatch the asset
                authors with ``unitsResolve`` scale and rotateX ops on the
                prim, and rescale the asset's physics values. Applied only
                when the prim then has no other transform ops (ops the asset
                root brings are in asset units and would stay unconverted)
                and had no children, physics values or other references
                before (those are in stage units and would be converted
                too). A decline is reported in ``units.notes`` and
                ``warning``; use import_isaac_asset, which wraps the
                reference. Off by default here.

        Returns:
            Dict with ``added`` (false when the same reference was already
            there), ``referenced_prim``, ``total_references`` and ``units``.
        """
        problem = _asset_prim_path_problem(asset_prim_path)
        if problem is not None:
            return problem
        denial = self._sandbox_denial(reference_path)
        if denial is not None:
            return denial
        _ref_path = _pyval(self._path_policy.authorize(reference_path))
        script = _compose_script(
            """\
            import json
            import omni.usd
            from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
            """,
            OPEN_ASSET_CORE,
            script_guard_source(),
            REFERENCE_ASSET_CORE,
            f"""\
            prim_path = {_pyval(prim_path)}
            asset_path = {_ref_path}
            asset_prim_path = {_pyval(asset_prim_path)}
            correct_units = {_pyval(bool(correct_units))}
            guard = {_pyval(not self.settings.security.allow_script_execution)}
            convert_hint = {_pyval(_convert_first_hint(reference_path))}
            """,
            """\
            stage = omni.usd.get_context().get_stage()
            asset_error = _open_asset(asset_path) if stage is not None else None
            if stage is None:
                print(json.dumps({"error": "No stage is currently open"}))
            elif asset_error is not None:
                print(json.dumps({"error": asset_error + convert_hint, "error_type": "NotFoundError"}))
            elif not Sdf.Path.IsValidPathString(prim_path) or not Sdf.Path(prim_path).IsAbsolutePath():
                print(json.dumps({
                    "error": "prim_path is not an absolute prim path: " + prim_path,
                    "error_type": "ValidationError",
                }))
            else:
                result = _reference_on_prim(stage, prim_path, asset_path, asset_prim_path, correct_units, guard)
                if "error" not in result:
                    prim = stage.GetPrimAtPath(prim_path)
                    references = prim.GetMetadata("references")
                    result.update({
                        "prim_path": prim_path,
                        "reference_path": asset_path,
                        "added": not result.pop("already_referenced"),
                        "total_references": len(references.GetAppliedItems()) if references else 0,
                    })
                print(json.dumps(result))
            """,
        )
        return await self._rescale_physics_values(await self._execute_json_script(script))

    async def _rescale_physics_values(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Run the physics-units pass an import or reference asked for.

        The units table is a third of a script, so it travels in a second
        script, sent only when the asset's units differ from the stage's.

        Args:
            result: The import or reference script's result. Its
                ``units.physics`` holds ``pending`` (the prim to walk) when a
                pass is due.

        Returns:
            ``result`` with ``units.physics`` replaced by the pass's report,
            and a warning for values the pass could not rescale.
        """
        units = result.get("units") if result.get("success") else None
        pending = units.get("physics") if isinstance(units, dict) else None
        if not isinstance(pending, dict) or "pending" not in pending:
            return result
        root_path = pending.pop("pending")
        apply = bool(pending["applied"])
        script = _compose_script(
            """\
            import json
            import omni.usd
            from pxr import Usd, UsdPhysics
            """,
            physics_units_source(),
            f"""\
            root_path = {_pyval(root_path)}
            length_ratio = {_pyval(float(pending["length_ratio"]))}
            mass_ratio = {_pyval(float(pending["mass_ratio"]))}
            apply = {_pyval(apply)}
            """,
            """\
            stage = omni.usd.get_context().get_stage()
            root = stage.GetPrimAtPath(root_path) if stage is not None else None
            if root is None or not root.IsValid():
                print(json.dumps({"error": "The referenced prim is gone: " + root_path}))
            else:
                print(json.dumps(_physics_unit_correction(stage, root, length_ratio, mass_ratio, apply)))
            """,
        )
        report = await self._execute_json_script(script, timeout_seconds=self._heavy_script_timeout())
        if not report.get("success"):
            units["physics"] = dict(pending, applied=False, failure=report.get("error"))
            warning = (
                "The asset was referenced, but its physics values were not "
                + ("rescaled" if apply else "counted") + " (" + str(report.get("error"))
                + "); PhysX will read them in stage units."
            )
        else:
            report.pop("success", None)
            units["physics"] = report
            warning = None
            if not apply and report.get("attribute_count"):
                units["notes"].append(
                    f"{report['attribute_count']} physics value(s) in asset units were not rescaled"
                )
            if report.get("uncorrected_count"):
                warning = (
                    f"{report['uncorrected_count']} physics value(s) in asset units could not be "
                    "rescaled (see units.physics.uncorrected); PhysX will read them in stage units."
                )
        if warning is not None:
            result["warning"] = f"{result['warning']} {warning}" if result.get("warning") else warning
        return result


def _asset_prim_path_problem(asset_prim_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Return a ValueError envelope when ``asset_prim_path`` is not a prim path.

    Args:
        asset_prim_path: The prim inside the asset the caller named, or None.

    Returns:
        The error envelope, or None when the value is usable or omitted.
    """
    if asset_prim_path is None or asset_prim_path.startswith("/"):
        return None
    return ErrorResponse(
        error=f"asset_prim_path must be an absolute prim path such as /Robot, got {asset_prim_path!r}",
        error_type="ValidationError",
    ).model_dump()
