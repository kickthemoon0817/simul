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


class StageAssetMixin:
    # ------------------------------------------------------------------
    # Phase 8: Asset & Stage Operations
    # ------------------------------------------------------------------

    async def open_isaac_stage(self, file_path: str) -> Dict[str, Any]:
        """
        Open a USD stage file in Isaac Sim.

        Args:
            file_path: Path or URL to the USD file.

        Returns:
            Dict confirming the stage was opened.
        """
        denial = self._sandbox_denial(file_path)
        if denial is not None:
            return denial
        _file_path = _pyval(self._path_policy.authorize(file_path))
        script = textwrap.dedent(f"""\
            import json
            import omni.usd

            ctx = omni.usd.get_context()
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

    async def save_isaac_stage(
        self, file_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Save the current stage. If file_path is given, saves to that path.

        Args:
            file_path: Optional path to save to. None saves to current location.

        Returns:
            Dict confirming the stage was saved.
        """
        denial = self._sandbox_denial(file_path, write=True)
        if denial is not None:
            return denial
        if file_path:
            _file_path = _pyval(self._path_policy.authorize(file_path, write=True))
            script = textwrap.dedent(f"""\
                import json
                import omni.usd

                ctx = omni.usd.get_context()
                result, error, saved_paths = await ctx.save_as_stage_async({_file_path})
                if result:
                    print(json.dumps({{"file_path": {_file_path}, "saved": True}}))
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

    async def new_isaac_stage(self) -> Dict[str, Any]:
        """
        Create a new empty stage in Isaac Sim.

        Returns:
            Dict confirming the new stage was created.
        """
        script = textwrap.dedent("""\
            import json
            import omni.usd

            ctx = omni.usd.get_context()
            result, error = await ctx.new_stage_async()
            if result:
                print(json.dumps({"created": True, "new_stage": True}))
            else:
                print(json.dumps({"error": f"Failed to create new stage: {error}"}))
        """)
        return await self._execute_json_script(script)

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

