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


class SystemMixin:
    # ------------------------------------------------------------------
    # Extension management
    # ------------------------------------------------------------------

    async def list_isaac_extensions(
        self,
        enabled_only: bool = True,
        search: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        List extensions registered in the running Isaac Sim instance.

        Args:
            enabled_only: If True, return only enabled extensions.
            search: Optional substring filter on extension ID.

        Returns:
            Dict with extensions list, each containing id and enabled status.
        """
        _enabled_only = repr(enabled_only)
        _search = repr(search)
        script = textwrap.dedent(f"""\
            import json
            import omni.kit.app

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
                    "version": ext.get("version", ""),
                }})

            extensions.sort(key=lambda e: e["id"])
            print(json.dumps({{
                "count": len(extensions),
                "extensions": extensions,
            }}))
        """)
        return await self._execute_json_script(script)

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

