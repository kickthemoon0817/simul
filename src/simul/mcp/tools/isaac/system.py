"""Extension management tools for Isaac Sim."""

import textwrap
from typing import Any, Dict, List, Optional, Tuple

from ._shared import (
    PERSISTENT_CARB_NAMESPACE,
    PROTECTED_CARB_NAMESPACES,
    PROTECTED_CARB_SETTING_PREFIXES,
    SCRIPTING_CARB_NAMESPACES,
    _pyval,
    carb_path_touches,
    carb_write_paths,
    names_file_location,
)
from .graph import _script_execution_refusal, is_script_extension
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
        _enabled_only = _pyval(enabled_only)
        _search = _pyval(search)
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
            "Enable an extension immediately in the running Isaac Sim instance. With "
            "security.allow_script_execution off, the extensions that run Python attached to "
            "the stage (omni.graph.scriptnode, omni.kit.scripting, omni.warp, and any with "
            "a word starting with 'script' in its name) are refused."
        ),
        read_only=False,
        idempotent=True,
    )
    async def enable_isaac_extension(self, extension_id: str) -> Dict[str, Any]:
        """
        Enable an extension by its ID in the running Isaac Sim instance.

        Args:
            extension_id: Bare Kit extension name (preferred, e.g. "omni.physx")
                or the version-suffixed ID from list_isaac_extensions
                (e.g. "omni.physx-107.3.7").

        Returns:
            Dict with success status and extension info after enabling, or a
            ScriptExecutionDisabled error for a script extension while
            script execution is off.
        """
        if not self.settings.security.allow_script_execution and is_script_extension(extension_id):
            return _script_execution_refusal(
                f"Refusing to enable {extension_id!r}: it runs Python attached to the stage.",
                extension_id=extension_id,
            )
        _ext_id = _pyval(extension_id)
        script = textwrap.dedent(f"""\
            import json
            import omni.kit.app

            ext_id = {_ext_id}
            ext_manager = omni.kit.app.get_app().get_extension_manager()
            ext_manager.set_extension_enabled_immediate(ext_id, True)

            # Verify by matching on bare name or fully qualified ID — both are
            # accepted by the Kit extension manager, but get_extensions() always
            # returns the version-suffixed form in 'id' and the bare form in 'name'.
            bare_query = ext_id.split("-", 1)[0]
            found = False
            for ext in ext_manager.get_extensions():
                eid = ext.get("id", "")
                ename = ext.get("name", "")
                if ext_id in (eid, ename) or bare_query == ename:
                    found = True
                    enabled = bool(ext.get("enabled", False))
                    payload = {{
                        "extension_id": eid or ename,
                        "name": ename,
                        "enabled": enabled,
                        "version": ext.get("version", ""),
                    }}
                    if not enabled:
                        # The manager refuses silently (unresolved dependency,
                        # startup exception); without an explicit failure the
                        # envelope would report success for a no-op.
                        payload["success"] = False
                        payload["error"] = (
                            "Extension did not enable: " + (eid or ename)
                            + " (check the Kit log for dependency or startup errors)"
                        )
                    print(json.dumps(payload))
                    break

            if not found:
                print(json.dumps({{
                    "success": False,
                    "error": "Extension not found: " + ext_id,
                }}))
        """)
        # A large extension (and its dependency chain) can take longer to
        # start than the default budget; an interrupt then lands inside some
        # extension's startup code.
        return await self._execute_json_script(
            script, timeout_seconds=self._heavy_script_timeout()
        )

    # ------------------------------------------------------------------
    # Carb settings
    # ------------------------------------------------------------------

    @tool_meta(
        name="get_isaac_carb_settings",
        description=(
            "Read one or more Carbonite (carb) settings by key path. Settings control RTX "
            "renderer options, fog, exposure, tone mapping, and other runtime parameters. "
            "Key paths look like /rtx/fog/enabled, /rtx/post/tonemap/op, "
            "/rtx/raytracing/showLights. Returns the current value for each requested key; "
            "transport auth tokens (any auth_token key) read back as ***."
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

            # Transport tokens stay out of tool results (and so out of agent
            # transcripts), whether asked for directly or inside a subtree.
            def _redact(name, value):
                if name == "auth_token":
                    return "***" if value else value
                if isinstance(value, dict):
                    return {{k: _redact(k, v) for k, v in value.items()}}
                return value

            result = {{}}
            for key in keys:
                val = settings.get(key)
                try:
                    if hasattr(val, '__iter__') and not isinstance(val, (str, list, tuple, dict)):
                        val = list(val)
                except Exception:
                    val = str(val)
                result[key] = _redact(key.rstrip("/").rsplit("/", 1)[-1], val)
            print(json.dumps({{"settings": result}}))
        """)
        return await self._execute_json_script(script)

    def _sandboxed_carb_value(self, key: str, value: Any) -> Tuple[Any, Optional[Dict[str, Any]]]:
        """Pass the file locations in a Carb write through the write sandbox.

        A dict value writes a subtree, so each of its keys is judged by its
        own full path.

        Args:
            key: The Carb path written.
            value: Its value.

        Returns:
            ``(value to write, None)``, or ``(value, SandboxError payload)``.
        """
        if isinstance(value, dict):
            resolved: Dict[str, Any] = {}
            for child_key, child_value in value.items():
                resolved[child_key], denial = self._sandboxed_carb_value(
                    key.rstrip("/") + "/" + str(child_key), child_value
                )
                if denial is not None:
                    return value, denial
            return resolved, None
        if not names_file_location(key):
            return value, None
        resolved_value, denial = self._sandboxed_location(key, value)
        if denial is not None:
            denial["details"]["setting"] = key
        return resolved_value, denial

    @tool_meta(
        name="set_isaac_carb_settings",
        description=(
            "Write one or more Carbonite (carb) settings by key path. Takes a dict of "
            'key-value pairs (e.g. {"/rtx/fog/enabled": true, "/rtx/fog/fogEndDist": '
            "200.0}). Returns the verified new values after applying. Changes take effect "
            "immediately in the renderer. A write that touches the namespaces configuring "
            "the MCP transport (/exts/khemoo.simul/, /exts/isaacsim.code_editor.python_server/, "
            "/exts/isaacsim.code_editor.vscode/) is refused: a key inside one, the namespace "
            "itself, or an ancestor such as /exts or / (a dict value replaces the whole "
            "subtree). With script execution disabled, /app/scripting (the Python Scripting "
            "Component's settings) is refused the same way. Keys under /persistent/ are saved "
            "to the user's Kit configuration and outlive this session, so they are refused "
            "unless allow_persistent=true. A string for a key named like a file location "
            "(/log/file, a ...Dir, ...Folder, ...Path or ...Url key; not a prim or camera path) "
            "must be inside the file sandbox (security.allowed_paths), judged as a write, or "
            "nothing is applied."
        ),
        read_only=False,
        destructive=True,
        idempotent=True,
    )
    async def set_carb_settings(
        self, settings: Dict[str, Any], allow_persistent: bool = False
    ) -> Dict[str, Any]:
        """
        Write one or more Carbonite settings by key path.

        Carb's ``set(path, dict)`` replaces the subtree at ``path``, so every
        path a write reaches is checked, a dict value's keys included: one
        that is a protected namespace, lies inside it or is an ancestor of it
        is refused. Keys with ``.`` or ``..`` segments are refused as
        ambiguous; empty segments (``a//b``) are collapsed before checking.

        Args:
            settings: Dict mapping key paths to values
                      (e.g. {"/rtx/fog/enabled": true, "/rtx/fog/fogEndDist": 200.0}).
            allow_persistent: Also write keys under /persistent/, which Kit
                saves to the user's configuration (Isaac Sim's .kit files set
                settings.persistent = true), so the change applies to every
                later session too. Refused without it.

        Returns:
            Dict with applied settings and their verified new values (plus a
            warning naming any persistent keys written), or a
            RefusedOperation error when any key touches the MCP transport's
            settings, is malformed or is persistent without allow_persistent,
            or a ScriptExecutionDisabled error for /app/scripting while
            script execution is off, or a SandboxError when a key named like
            a file location gets a path outside the file sandbox.
        """
        written: Dict[str, List[str]] = {}
        malformed = []
        for key, value in settings.items():
            paths = carb_write_paths(key, value)
            if paths is None:
                malformed.append(key)
            else:
                written[key] = paths
        if malformed:
            return self._refusal(
                "Refusing to write settings with '.' or '..' path segments: "
                + ", ".join(sorted(malformed))
                + ". No settings were applied.",
                refused_keys=sorted(malformed),
            )

        def _touching(namespaces: Tuple[str, ...]) -> List[str]:
            return sorted(
                key
                for key, paths in written.items()
                if any(carb_path_touches(path, namespace) for path in paths for namespace in namespaces)
            )

        protected_keys = _touching(PROTECTED_CARB_NAMESPACES)
        if protected_keys:
            return self._refusal(
                "Refusing to write transport settings: "
                + ", ".join(protected_keys)
                + ". Keys in, or above, "
                + ", ".join(PROTECTED_CARB_NAMESPACES)
                + " configure the socket simul is speaking through (a write to an ancestor "
                "replaces the whole subtree). No settings were applied.",
                refused_keys=protected_keys,
                protected_prefixes=list(PROTECTED_CARB_SETTING_PREFIXES),
                protected_namespaces=list(PROTECTED_CARB_NAMESPACES),
            )
        if not self.settings.security.allow_script_execution:
            scripting_keys = _touching(SCRIPTING_CARB_NAMESPACES)
            if scripting_keys:
                return _script_execution_refusal(
                    "Refusing to write " + ", ".join(scripting_keys)
                    + ": /app/scripting decides whether Kit runs the Python scripts attached "
                    "to a stage (ignoreWarningDialog runs them without asking). No settings "
                    "were applied.",
                    refused_keys=scripting_keys,
                    protected_namespaces=list(SCRIPTING_CARB_NAMESPACES),
                )
        persistent_keys = _touching((PERSISTENT_CARB_NAMESPACE,))
        if persistent_keys and not allow_persistent:
            return self._refusal(
                "Refusing to write persistent settings without allow_persistent=true: "
                + ", ".join(persistent_keys)
                + ". Keys under /persistent/ are saved to the user's Kit configuration and "
                "apply to every later session. No settings were applied.",
                refused_keys=persistent_keys,
                override="allow_persistent",
            )
        # A key named like a file location (/log/file, a dump or output
        # directory, /app/exts/folders) makes Kit write, read or load from
        # that path.
        sandboxed: Dict[str, Any] = {}
        for key, value in settings.items():
            sandboxed[key], denial = self._sandboxed_carb_value(key, value)
            if denial is not None:
                return denial
        _settings = _pyval(sandboxed)
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
        result = await self._execute_json_script(script)
        if persistent_keys and result.get("success"):
            result["persistent_keys"] = persistent_keys
            result["warning"] = (
                "Wrote " + ", ".join(persistent_keys) + " under /persistent/: Kit saves these "
                "to the user's configuration, so they also apply to later sessions."
            )
        return result
