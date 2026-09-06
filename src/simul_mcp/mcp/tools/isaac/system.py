"""Extension management tools for Isaac Sim."""

import json
import textwrap
from typing import Any, Callable, Dict, List, Optional

from ....adapters import IsaacSocketClient, ScriptResult
from ...schemas.common import ErrorResponse
from ._shared import (
    PROTECTED_CARB_SETTING_PREFIXES,
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
from .._meta import tool_meta


class SystemMixin:
    # ------------------------------------------------------------------
    # Extension management
    # ------------------------------------------------------------------

    @tool_meta(
        name="list_isaac_extensions",
        description=(
            "List extensions registered in the running Isaac Sim instance. Returns each "
            "extension's ID (version-suffixed, e.g. 'omni.physx-107.3.7'), version, and "
            "enabled status. Lists enabled extensions by default; pass enabled_only=false "
            "with search='<substring>' to scope a wider query."
        ),
        read_only=True,
        idempotent=True,
    )
    async def list_isaac_extensions(
        self,
        enabled_only: bool = True,
        search: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        List extensions registered in the running Isaac Sim instance.

        Args:
            enabled_only: If True, return only enabled extensions.
            search: Optional case-insensitive substring filter on the
                extension ID.
            limit: Maximum number of extensions to return per page. Clamped
                to [1, 1000]; the effective cap is reported as applied_limit.
            offset: Number of matching extensions to skip before the page
                starts; pass the previous page's next_offset to continue.

        Returns:
            Dict with a page of extensions (id, enabled, version string),
            sorted by id, plus the total number of matches.
        """
        _enabled_only = repr(enabled_only)
        _search = repr(search)
        limit = max(1, min(limit, 1000))
        offset = max(0, offset)
        script = textwrap.dedent(f"""\
            import json
            import omni.kit.app

            def _version_string(version):
                # Kit reports versions as [major, minor, patch, prerelease, build];
                # render them the way the id suffix spells them.
                if isinstance(version, (list, tuple)):
                    parts = list(version)
                    text = ".".join(str(part) for part in parts[:3])
                    if len(parts) > 3 and parts[3]:
                        text += "-" + str(parts[3])
                    if len(parts) > 4 and parts[4]:
                        text += "+" + str(parts[4])
                    return text
                return str(version) if version is not None else ""

            ext_manager = omni.kit.app.get_app().get_extension_manager()
            raw = ext_manager.get_extensions()
            extensions = []
            for ext in raw:
                ext_id = ext.get("id", "") or ext.get("name", "")
                enabled = ext.get("enabled", False)

                if {_enabled_only} and not enabled:
                    continue
                search_term = {_search}
                if search_term and search_term.lower() not in ext_id.lower():
                    continue

                extensions.append({{
                    "id": ext_id,
                    "enabled": enabled,
                    "version": _version_string(ext.get("version", "")),
                }})

            extensions.sort(key=lambda e: e["id"])
            offset = {offset}
            limit = {limit}
            page = extensions[offset:offset + limit]
            truncated = offset + len(page) < len(extensions)
            print(json.dumps({{
                "count": len(page),
                "total": len(extensions),
                "offset": offset,
                "applied_limit": limit,
                "truncated": truncated,
                "next_offset": offset + len(page) if truncated else None,
                "extensions": page,
            }}))
        """)
        return await self._execute_json_script(script)

    @tool_meta(
        name="enable_isaac_extension",
        description=(
            "Enable an extension immediately in the running Isaac Sim instance. Accepts "
            "EITHER the bare canonical Kit extension name (e.g. 'omni.physx', "
            "'isaacsim.replicator.behavior') OR the version-suffixed ID returned by "
            "list_isaac_extensions (e.g. 'omni.physx-107.3.7'). The bare name is preferred "
            "— it is what the underlying Kit set_extension_enabled_immediate API takes and "
            "keeps callers decoupled from the installed version."
        ),
        read_only=False,
        idempotent=True,
    )
    async def enable_isaac_extension(self, extension_id: str) -> Dict[str, Any]:
        """
        Enable an extension by its ID in the running Isaac Sim instance.

        Accepts either the bare canonical extension name (e.g. "worv.env.sun")
        — which is what ``omni.kit.app.IExtensionManager.set_extension_enabled_immediate``
        natively takes — or the fully version-suffixed ID returned by
        ``list_isaac_extensions`` (e.g. "worv.env.sun-0.3.0").

        Args:
            extension_id: The extension name or fully qualified ID
                (e.g. "isaacsim.core.utils", "omni.physx", "worv.env.sun-0.3.0").

        Returns:
            Dict with success status and extension info after enabling.
        """
        _ext_id = repr(extension_id)
        script = textwrap.dedent(f"""\
            import json
            import omni.kit.app

            ext_id = {_ext_id}
            ext_manager = omni.kit.app.get_app().get_extension_manager()
            ext_manager.set_extension_enabled_immediate(ext_id, True)

            # Verify by matching on bare name or fully qualified ID — both are
            # accepted by the Kit extension manager, but get_extensions() always
            # returns the version-suffixed form in 'id' and the bare form in 'name'.
            bare_query = ext_id.rsplit("-", 1)[0] if "-" in ext_id else ext_id
            found = False
            for ext in ext_manager.get_extensions():
                eid = ext.get("id", "")
                ename = ext.get("name", "")
                if ext_id in (eid, ename) or bare_query == ename:
                    found = True
                    print(json.dumps({{
                        "extension_id": eid or ename,
                        "name": ename,
                        "enabled": ext.get("enabled", False),
                        "version": ext.get("version", ""),
                    }}))
                    break

            if not found:
                print(json.dumps({{
                    "success": False,
                    "error": "Extension not found: " + ext_id,
                }}))
        """)
        return await self._execute_json_script(script)

    # ------------------------------------------------------------------
    # Carb settings
    # ------------------------------------------------------------------

    @tool_meta(
        name="get_isaac_carb_settings",
        description=(
            "Read one or more Carbonite (carb) settings by key path. Settings control RTX "
            "renderer options, fog, exposure, tone mapping, and other runtime parameters. "
            "Key paths look like /rtx/fog/enabled, /rtx/post/tonemap/op, "
            "/rtx/raytracing/showLights. Returns the current value for each requested key."
        ),
        read_only=True,
        idempotent=True,
    )
    async def get_carb_settings(self, keys: List[str]) -> Dict[str, Any]:
        """
        Read one or more Carbonite settings by key path.

        Args:
            keys: List of setting key paths (e.g. ["/rtx/fog/enabled"]).

        Returns:
            Dict mapping each key to its current value.
        """
        _keys = _pyval(keys)
        script = textwrap.dedent(f"""\
            import json
            import carb.settings

            settings = carb.settings.get_settings()
            keys = {_keys}
            result = {{}}
            for key in keys:
                val = settings.get(key)
                try:
                    if hasattr(val, '__iter__') and not isinstance(val, (str, list, tuple, dict)):
                        val = list(val)
                except Exception:
                    val = str(val)
                result[key] = val
            print(json.dumps({{"settings": result}}))
        """)
        return await self._execute_json_script(script)

    @tool_meta(
        name="set_isaac_carb_settings",
        description=(
            "Write one or more Carbonite (carb) settings by key path. Takes a dict of "
            'key-value pairs (e.g. {"/rtx/fog/enabled": true, "/rtx/fog/fogEndDist": '
            "200.0}). Returns the verified new values after applying. Changes take effect "
            "immediately in the renderer. Keys under /exts/khemoo.simul.mcp/ and "
            "/exts/isaacsim.code_editor.python_server/ configure the MCP transport and are "
            "refused."
        ),
        read_only=False,
        destructive=True,
        idempotent=True,
    )
    async def set_carb_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Write one or more Carbonite settings by key path.

        Args:
            settings: Dict mapping key paths to values
                      (e.g. {"/rtx/fog/enabled": true, "/rtx/fog/fogEndDist": 200.0}).

        Returns:
            Dict with applied settings and their verified new values, or a
            RefusedOperation error when any key configures the MCP transport.
        """
        protected_keys = sorted(
            key
            for key in settings
            if ("/" + key.lstrip("/")).startswith(PROTECTED_CARB_SETTING_PREFIXES)
        )
        if protected_keys:
            return self._refusal(
                "Refusing to write transport settings: "
                + ", ".join(protected_keys)
                + ". Keys under "
                + " and ".join(PROTECTED_CARB_SETTING_PREFIXES)
                + " configure the socket simul-mcp is speaking through. No settings were applied.",
                refused_keys=protected_keys,
                protected_prefixes=list(PROTECTED_CARB_SETTING_PREFIXES),
            )
        _settings = _pyval(settings)
        script = textwrap.dedent(f"""\
            import json
            import carb.settings

            cs = carb.settings.get_settings()
            to_set = {_settings}
            applied = {{}}
            for key, val in to_set.items():
                cs.set(key, val)
                new_val = cs.get(key)
                try:
                    if hasattr(new_val, '__iter__') and not isinstance(new_val, (str, list, tuple, dict)):
                        new_val = list(new_val)
                except Exception:
                    new_val = str(new_val)
                applied[key] = new_val
            print(json.dumps({{"applied": applied, "count": len(applied)}}))
        """)
        return await self._execute_json_script(script)

