"""Asset & Stage Operations tools for Isaac Sim."""

import json
import textwrap
from typing import Any, Callable, Dict, List, Optional

from ....adapters import IsaacSocketClient, ScriptResult
from ...schemas.common import ErrorResponse
from ._shared import (
    BULK_GEOMETRY_ATTRIBUTES,
    LOG_SCAN_WINDOW_BYTES,
    MAX_CAPTURE_DIMENSION,
    MAX_INLINE_CAPTURE_BYTES,
    MAX_RETAINED_CAPTURES,
    MAX_SCRIPT_BYTES,
    PRIM_DETAIL_ASPECTS,
    FloatList,
    _pyval,
    logger,
)
from .._meta import SandboxedPath, tool_meta


class StageAssetMixin:
    # ------------------------------------------------------------------
    # Phase 8: Asset & Stage Operations
    # ------------------------------------------------------------------

    @tool_meta(
        name="open_isaac_stage",
        description=(
            "Open a USD stage file in Isaac Sim. Paths must be inside the configured "
            "sandbox (security.allowed_paths); omniverse:// URLs are allowed when their "
            "scheme is in security.allowed_url_schemes. Refused while the current stage has "
            "unsaved edits unless discard_unsaved=true."
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
        return await self._execute_json_script(script)

    @tool_meta(
        name="save_isaac_stage",
        description=(
            "Save the current stage, overwriting the file on disk. Optionally provide a "
            "file path to save-as to a new location; an existing file there is refused "
            "unless overwrite=true. Paths must be inside the configured sandbox "
            "(security.allowed_paths); writes to URLs are refused unless the scheme is in "
            "security.allowed_write_url_schemes."
        ),
        read_only=False,
        destructive=True,
        idempotent=True,
        sandboxed_paths=(SandboxedPath("file_path", write=True),),
    )
    async def save_isaac_stage(
        self, file_path: Optional[str] = None, overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        Save the current stage. If file_path is given, saves to that path.

        Saving in place always writes the stage's own file. A save-as target
        that already exists is refused unless ``overwrite`` is set; the check
        runs inside Isaac Sim because the file system is Isaac's.

        Args:
            file_path: Optional path to save to. None saves to current location.
            overwrite: Replace an existing file at ``file_path``.

        Returns:
            Dict confirming the stage was saved, or a RefusedOperation error
            naming the file that would have been overwritten.
        """
        denial = self._sandbox_denial(file_path, write=True)
        if denial is not None:
            return denial
        if file_path:
            _file_path = _pyval(self._path_policy.authorize(file_path, write=True))
            _overwrite = _pyval(overwrite)
            script = textwrap.dedent(f"""\
                import json
                import os
                import omni.usd

                ctx = omni.usd.get_context()
                target = {_file_path}
                current = ctx.get_stage_url() or ""
                same_file = os.path.exists(target) and os.path.exists(current) and os.path.samefile(target, current)
                if os.path.exists(target) and not same_file and not {_overwrite}:
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
            """)
        else:
            script = textwrap.dedent("""\
                import json
                import omni.usd

                ctx = omni.usd.get_context()
                stage = ctx.get_stage()
                if stage is None:
                    print(json.dumps({"error": "No stage is currently open"}))
                else:
                    url = ctx.get_stage_url()
                    if not url:
                        print(json.dumps({"error": "Stage has no file path. Use file_path parameter."}))
                    else:
                        result, error, saved_paths = await ctx.save_stage_async()
                        if result:
                            print(json.dumps({"file_path": url, "saved": True}))
                        else:
                            print(json.dumps({"error": f"Failed to save stage: {error}"}))
            """)
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
        return await self._execute_json_script(script)

    @tool_meta(
        name="import_isaac_asset",
        description=(
            "Import an external asset (USD, USDZ, OBJ, FBX) into the current stage at a "
            "target path. Paths must be inside the configured sandbox "
            "(security.allowed_paths); omniverse:// URLs are allowed when their scheme is "
            "in security.allowed_url_schemes."
        ),
        read_only=False,
        sandboxed_paths=(SandboxedPath("asset_path"),),
    )
    async def import_isaac_asset(
        self,
        asset_path: str,
        target_path: str = "/World/ImportedAsset",
    ) -> Dict[str, Any]:
        """
        Import a USD asset into the current stage as a reference.

        Args:
            asset_path: File path or URL to the USD asset.
            target_path: USD path where the asset should be placed.

        Returns:
            Dict confirming the asset was imported.
        """
        denial = self._sandbox_denial(asset_path)
        if denial is not None:
            return denial
        _asset_path = _pyval(self._path_policy.authorize(asset_path))
        _target_path = _pyval(target_path)
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import Sdf, UsdGeom

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.DefinePrim({_target_path})
                if not prim.IsValid():
                    print(json.dumps({{"error": "Failed to create prim at: " + {_target_path}}}))
                else:
                    prim.GetReferences().AddReference({_asset_path})
                    print(json.dumps({{
                        "asset_path": {_asset_path},
                        "target_path": {_target_path},
                        "imported": True,
                    }}))
        """)
        return await self._execute_json_script(script)

    @tool_meta(
        name="add_isaac_reference",
        description=(
            "Add a USD reference to a prim so it composes in content from another USD file. "
            "Paths must be inside the configured sandbox (security.allowed_paths); "
            "omniverse:// URLs are allowed when their scheme is in "
            "security.allowed_url_schemes."
        ),
        read_only=False,
        sandboxed_paths=(SandboxedPath("reference_path"),),
    )
    async def add_isaac_reference(
        self,
        prim_path: str,
        reference_path: str,
    ) -> Dict[str, Any]:
        """
        Add a USD reference to an existing prim.

        Args:
            prim_path: USD path of the prim to add the reference to.
            reference_path: File path or URL to the referenced USD asset.

        Returns:
            Dict confirming the reference was added.
        """
        denial = self._sandbox_denial(reference_path)
        if denial is not None:
            return denial
        _prim_path = _pyval(prim_path)
        _ref_path = _pyval(self._path_policy.authorize(reference_path))
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import Sdf

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath({_prim_path})
                if not prim.IsValid():
                    prim = stage.DefinePrim({_prim_path})
                if not prim.IsValid():
                    print(json.dumps({{"error": "Cannot create prim: " + {_prim_path}}}))
                else:
                    prim.GetReferences().AddReference({_ref_path})
                    refs = prim.GetMetadata("references")
                    ref_count = len(refs.GetAddedOrExplicitItems()) if refs else 0
                    print(json.dumps({{
                        "prim_path": {_prim_path},
                        "reference_path": {_ref_path},
                        "added": True,
                        "total_references": ref_count,
                    }}))
        """)
        return await self._execute_json_script(script)

