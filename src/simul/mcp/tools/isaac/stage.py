"""Asset & Stage Operations tools for Isaac Sim."""

import textwrap
from typing import Any, Dict, List, Optional

from ...schemas.common import ErrorResponse
from ._shared import (
    OPEN_ASSET_CORE,
    _compose_script,
    _pyval,
)
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
# checks that it composed something, and corrects or reports a metersPerUnit /
# upAxis mismatch. Defines _reference_asset(...) and _unique_path(stage, path).
REFERENCE_ASSET_CORE = """\
    import math

    def _unique_path(stage, path):
        if not stage.GetPrimAtPath(path).IsValid():
            return path
        index = 1
        while stage.GetPrimAtPath(path + "_" + str(index)).IsValid():
            index += 1
        return path + "_" + str(index)

    def _asset_root(layer, asset_prim_path):
        # (prim path to reference or None for the defaultPrim, warning, error)
        if asset_prim_path is not None:
            if layer.GetPrimAtPath(asset_prim_path) is None:
                return None, None, "No prim " + asset_prim_path + " in " + layer.identifier
            return asset_prim_path, None, None
        if layer.defaultPrim:
            return None, None, None
        for spec in layer.rootPrims:
            if spec.specifier == Sdf.SpecifierDef:
                return str(spec.path), (
                    "The asset has no defaultPrim; referenced its first root prim "
                    + str(spec.path) + ". Pass asset_prim_path to choose another."
                ), None
        return None, None, "The asset has no defaultPrim and no root prim to reference: " + layer.identifier

    def _layer_metrics(layer):
        root = layer.pseudoRoot
        meters = float(root.GetInfo("metersPerUnit")) if root.HasInfo("metersPerUnit") else None
        up_axis = str(root.GetInfo("upAxis")) if root.HasInfo("upAxis") else None
        return meters, up_axis

    def _units(stage, prim, layer, correct):
        asset_meters, asset_up = _layer_metrics(layer)
        stage_meters = float(UsdGeom.GetStageMetersPerUnit(stage)) or 1.0
        stage_up = str(UsdGeom.GetStageUpAxis(stage))
        scale = None
        if asset_meters is not None and not math.isclose(asset_meters, stage_meters, rel_tol=1e-9):
            scale = asset_meters / stage_meters
        rotate_x = None
        if asset_up is not None and asset_up != stage_up:
            rotate_x = 90.0 if (asset_up, stage_up) == ("Y", "Z") else -90.0
        report = {
            "asset_meters_per_unit": asset_meters,
            "asset_up_axis": asset_up,
            "stage_meters_per_unit": stage_meters,
            "stage_up_axis": stage_up,
            "scale_needed": scale,
            "rotate_x_needed": rotate_x,
            "applied": False,
        }
        notes = []
        if asset_meters is None:
            notes.append("the asset does not author metersPerUnit (USD's fallback is centimetres); no scale applied")
        if asset_up is None:
            notes.append("the asset does not author upAxis; no rotation applied")
        if (scale is not None or rotate_x is not None) and correct:
            if not prim.IsA(UsdGeom.Xformable):
                notes.append("the referenced prim is a " + prim.GetTypeName() + ", not Xformable; nothing applied")
            else:
                # Suffixed ops at the end of the op order: innermost, so the
                # asset is converted before any transform the caller sets, and
                # set_isaac_prim_transform leaves them alone.
                xformable = UsdGeom.Xformable(prim)
                if rotate_x is not None:
                    attr = prim.GetAttribute("xformOp:rotateX:unitsResolve")
                    op = UsdGeom.XformOp(attr) if attr.IsValid() else xformable.AddRotateXOp(
                        UsdGeom.XformOp.PrecisionFloat, "unitsResolve")
                    op.Set(rotate_x)
                if scale is not None:
                    attr = prim.GetAttribute("xformOp:scale:unitsResolve")
                    op = UsdGeom.XformOp(attr) if attr.IsValid() else xformable.AddScaleOp(
                        UsdGeom.XformOp.PrecisionFloat, "unitsResolve")
                    op.Set(Gf.Vec3f(scale, scale, scale))
                report["applied"] = True
                report["xform_op_order"] = [op.GetOpName() for op in xformable.GetOrderedXformOps()]
        elif scale is not None or rotate_x is not None:
            notes.append("mismatch reported, not corrected (correct_units=false)")
        if notes:
            report["notes"] = notes
        return report

    def _edit_spec(stage, prim):
        target = stage.GetEditTarget()
        return target.GetLayer(), target.MapToSpecPath(prim.GetPath())

    def _reference_asset(stage, prim, asset_path, asset_prim_path, correct_units, created):
        layer = Sdf.Layer.FindOrOpen(asset_path)
        root_path, warning, error = _asset_root(layer, asset_prim_path)
        if error is not None:
            if created:
                stage.RemovePrim(prim.GetPath())
            return {"error": error + "; nothing was changed"}
        reference = Sdf.Reference(asset_path, root_path) if root_path else Sdf.Reference(asset_path)
        existing = prim.GetMetadata("references")
        for item in (existing.GetAppliedItems() if existing else []):
            if item.assetPath == reference.assetPath and item.primPath == reference.primPath:
                return {"already_referenced": True, "referenced_prim": root_path or layer.defaultPrim}
        edit_layer, spec_path = _edit_spec(stage, prim)
        spec = edit_layer.GetPrimAtPath(spec_path)
        had = spec is not None and spec.HasInfo("references")
        previous = spec.GetInfo("references") if had else None
        before = len(prim.GetPrimStack())
        prim.GetReferences().AddReference(reference)
        if len(prim.GetPrimStack()) <= before:
            # Nothing composed: undo the reference rather than leave an empty
            # prim behind that reads as an import.
            if created:
                stage.RemovePrim(prim.GetPath())
            else:
                spec = edit_layer.GetPrimAtPath(spec_path)
                if had:
                    spec.SetInfo("references", previous)
                else:
                    spec.ClearInfo("references")
            return {"error": (
                "The reference to " + asset_path + " composed nothing"
                + (" (prim " + root_path + ")" if root_path else " (its defaultPrim " + layer.defaultPrim + ")")
                + "; nothing was changed"
            )}
        result = {
            "already_referenced": False,
            "referenced_prim": root_path or layer.defaultPrim,
            "used_default_prim": root_path is None,
            "prim_type": prim.GetTypeName(),
            "child_count": len(prim.GetChildren()),
            "units": _units(stage, prim, layer, correct_units),
        }
        if warning is not None:
            result["warning"] = warning
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
            return await self._execute_json_script(script)

        probe = await self._execute_json_script(SAVE_PROBE_SCRIPT)
        if not probe.get("success"):
            return probe
        url = str(probe.get("stage_url") or "")
        if not url or url.startswith("anon:"):
            return ErrorResponse(
                error="The stage has never been saved, so it has no file to save in place. Pass file_path.",
                error_type="ValueError",
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
        return await self._execute_json_script(script)

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
            "A metersPerUnit/upAxis mismatch the asset authors is corrected with "
            "xformOp:scale:unitsResolve / xformOp:rotateX:unitsResolve on the new prim "
            "(correct_units=false only reports it). OBJ/FBX/glTF need converting to USD "
            "first. "
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
                that differs from the stage's, add a uniform
                ``xformOp:scale:unitsResolve`` (asset / stage metersPerUnit)
                and an ``xformOp:rotateX:unitsResolve`` (+-90 degrees) at the
                end of the new prim's op order. With false the mismatch is
                only reported. Metrics the asset does not author are not
                guessed at.

        Returns:
            Dict with the ``target_path`` used (and ``requested_target_path``),
            the ``referenced_prim``, the composed ``prim_type`` and
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
            from pxr import Gf, Sdf, UsdGeom
            """,
            OPEN_ASSET_CORE,
            REFERENCE_ASSET_CORE,
            f"""\
            asset_path = {_asset_path}
            requested = {_pyval(target_path)}
            asset_prim_path = {_pyval(asset_prim_path)}
            correct_units = {_pyval(bool(correct_units))}
            convert_hint = {_pyval(_convert_first_hint(asset_path))}
            """,
            """\
            stage = omni.usd.get_context().get_stage()
            asset_error = _open_asset(asset_path) if stage is not None else None
            if stage is None:
                print(json.dumps({"error": "No stage is currently open"}))
            elif asset_error is not None:
                print(json.dumps({"error": asset_error + convert_hint}))
            elif not Sdf.Path.IsValidPathString(requested) or not Sdf.Path(requested).IsAbsolutePath():
                print(json.dumps({"error": "target_path is not an absolute prim path: " + requested}))
            else:
                target = _unique_path(stage, requested)
                prim = stage.DefinePrim(target)
                result = _reference_asset(stage, prim, asset_path, asset_prim_path, correct_units, True)
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
        return await self._execute_json_script(script)

    @tool_meta(
        name="add_isaac_reference",
        description=(
            "Add a USD reference to a prim (created when missing) so it composes in content "
            "from another USD file. Without a defaultPrim the asset's first root prim is "
            "referenced (reported), or pass asset_prim_path; a reference that composes nothing "
            "is undone and fails, and the same reference twice is reported, not duplicated. "
            "A metersPerUnit/upAxis mismatch is reported; correct_units=true corrects it as "
            "import_isaac_asset does. "
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
                authors with ``unitsResolve`` scale and rotateX ops, as
                import_isaac_asset does. Off by default here: the prim may
                carry other references and its own transform.

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
            from pxr import Gf, Sdf, UsdGeom
            """,
            OPEN_ASSET_CORE,
            REFERENCE_ASSET_CORE,
            f"""\
            prim_path = {_pyval(prim_path)}
            asset_path = {_ref_path}
            asset_prim_path = {_pyval(asset_prim_path)}
            correct_units = {_pyval(bool(correct_units))}
            convert_hint = {_pyval(_convert_first_hint(reference_path))}
            """,
            """\
            stage = omni.usd.get_context().get_stage()
            asset_error = _open_asset(asset_path) if stage is not None else None
            if stage is None:
                print(json.dumps({"error": "No stage is currently open"}))
            elif asset_error is not None:
                print(json.dumps({"error": asset_error + convert_hint}))
            elif not Sdf.Path.IsValidPathString(prim_path) or not Sdf.Path(prim_path).IsAbsolutePath():
                print(json.dumps({"error": "prim_path is not an absolute prim path: " + prim_path}))
            else:
                prim = stage.GetPrimAtPath(prim_path)
                created = not prim.IsValid()
                if created:
                    prim = stage.DefinePrim(prim_path)
                if prim.IsInstanceProxy():
                    result = {"error": "Prim is inside an instance and cannot be edited here: " + prim_path}
                else:
                    result = _reference_asset(stage, prim, asset_path, asset_prim_path, correct_units, created)
                if "error" not in result:
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
        return await self._execute_json_script(script)


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
        error_type="ValueError",
    ).model_dump()
