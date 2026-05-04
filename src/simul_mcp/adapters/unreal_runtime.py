"""
Unreal Engine runtime adapter for Simul MCP Server.

This module provides an adapter for Unreal Engine operations through the
Remote Control API (HTTP on port 30010) or the embedded ``unreal`` Python module.
"""

import asyncio
import json
import math
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

try:
    import unreal as _unreal_module  # type: ignore[import-untyped]

    UNREAL_EMBEDDED_AVAILABLE = True
except ImportError:
    _unreal_module = None
    UNREAL_EMBEDDED_AVAILABLE = False

UNREAL_AVAILABLE = AIOHTTP_AVAILABLE or UNREAL_EMBEDDED_AVAILABLE

from ..config import Settings, get_settings
from ..logging import LoggerMixin, get_logger

logger = get_logger(__name__)


class UnrealRuntimeSession(LoggerMixin):
    """
    Unreal Engine runtime session for Remote Control API operations.

    Communicates with UE5 via HTTP REST calls to the Remote Control API
    (default ``localhost:30010``).  When running inside the UE5 Python
    interpreter the ``embedded`` flag is set and selected operations may
    use the ``unreal`` module directly.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """
        Initialize Unreal runtime session.

        Args:
            settings: Configuration settings.
        """
        if not UNREAL_AVAILABLE:
            raise ImportError(
                "Unreal runtime not available. Install aiohttp for HTTP mode "
                "or run inside the UE5 Python interpreter."
            )

        self.settings: Settings = settings or get_settings()
        cfg = self.settings.unreal
        self.host: str = cfg.host
        self.port: int = cfg.port
        self.timeout: int = cfg.timeout
        self.embedded: bool = cfg.embedded_mode or UNREAL_EMBEDDED_AVAILABLE
        self.max_actors: int = cfg.max_actors
        self.max_retries: int = cfg.max_retries
        self.retry_base_delay: float = cfg.retry_base_delay
        self.ping_timeout: float = cfg.ping_timeout

        self._base_url: str = f"http://{self.host}:{self.port}"
        self._session: Optional[Any] = None  # aiohttp.ClientSession, lazily created

        self.logger.info(
            "Unreal runtime session initialized (base_url=%s, embedded=%s)",
            self._base_url,
            self.embedded,
        )

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def _ensure_http_session(self, timeout_override: Optional[float] = None) -> Any:
        """
        Return the shared ``aiohttp.ClientSession``, creating it on first use.

        When *timeout_override* differs from the current session's timeout the
        existing session is recycled so a fresh one with the correct timeout is
        created.

        Args:
            timeout_override: Optional per-call timeout in seconds.

        Returns:
            An ``aiohttp.ClientSession`` instance.
        """
        desired_timeout = timeout_override if timeout_override is not None else self.timeout
        if self._session is not None and not self._session.closed:
            timeout_obj = getattr(self._session, "timeout", None)
            current_total = getattr(timeout_obj, "total", None) if timeout_obj is not None else None
            if current_total is not None and not math.isclose(current_total, desired_timeout, rel_tol=1e-3):
                await self._recycle_session()

        if self._session is None or self._session.closed:
            if not AIOHTTP_AVAILABLE:
                raise RuntimeError(
                    "aiohttp is required for HTTP mode but is not installed"
                )
            timeout_cfg = aiohttp.ClientTimeout(total=desired_timeout)
            self._session = aiohttp.ClientSession(
                base_url=self._base_url,
                timeout=timeout_cfg,
                headers={"Content-Type": "application/json"},
            )
        return self._session

    async def _recycle_session(self) -> None:
        """Close the current HTTP session so the next call creates a fresh one.

        This drains stale CLOSE-WAIT sockets and resets the connection pool,
        which is the primary recovery mechanism when the Remote Control API
        becomes unresponsive after editor restarts or duplicate-process conflicts.
        """
        if self._session is not None and not self._session.closed:
            try:
                await self._session.close()
            except Exception as exc:
                self.logger.debug("Session close failed during recycle: %s", exc)
        self._session = None
        self.logger.debug("HTTP session recycled")

    async def _http_request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        timeout_override: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform an HTTP request with retry and exponential backoff.

        On transient failures (connection errors, timeouts, 5xx) the session
        is recycled and the request is retried up to *max_retries* times with
        exponential backoff (base delay * 2^attempt, capped at 8 s).

        Args:
            method: HTTP method (``GET``, ``PUT``, ``POST``).
            path: URL path.
            body: Optional JSON-serializable request body.
            timeout_override: Per-call timeout in seconds.
            max_retries: Override for ``self.max_retries``.

        Returns:
            Parsed JSON response as a dictionary.
        """
        retries = max_retries if max_retries is not None else self.max_retries
        last_exc: Optional[Exception] = None

        for attempt in range(retries + 1):
            try:
                session = await self._ensure_http_session(timeout_override)
                kwargs: Dict[str, Any] = {}
                if body is not None:
                    kwargs["json"] = body

                async with getattr(session, method.lower())(path, **kwargs) as resp:
                    if resp.status >= 500 and attempt < retries:
                        self.logger.warning(
                            "HTTP %s %s returned %d, retrying (%d/%d)",
                            method, path, resp.status, attempt + 1, retries,
                        )
                        # Don't recycle session on 5xx — the connection is
                        # fine, the server just errored.  Only sleep + retry.
                        await asyncio.sleep(
                            min(self.retry_base_delay * (2 ** attempt), 8.0)
                        )
                        continue
                    # 4xx errors are client bugs — never retry them
                    resp.raise_for_status()
                    data: Dict[str, Any] = await resp.json()
                    return data

            except Exception as exc:
                last_exc = exc
                is_transient = isinstance(
                    exc,
                    (
                        asyncio.TimeoutError,
                        ConnectionError,
                        OSError,
                    ),
                )
                if aiohttp is not None and isinstance(exc, aiohttp.ClientError):
                    # ClientResponseError with 4xx status is a client bug, not transient
                    if hasattr(exc, "status") and isinstance(exc.status, int) and exc.status < 500:
                        is_transient = False
                    else:
                        is_transient = True

                if is_transient and attempt < retries:
                    delay = min(self.retry_base_delay * (2 ** attempt), 8.0)
                    self.logger.warning(
                        "HTTP %s %s failed (%s), recycling session and retrying "
                        "in %.1fs (%d/%d)",
                        method, path, exc, delay, attempt + 1, retries,
                    )
                    await self._recycle_session()
                    await asyncio.sleep(delay)
                    continue
                raise

        raise last_exc  # type: ignore[misc]

    async def _http_get(
        self,
        path: str,
        timeout_override: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform an HTTP GET against the Remote Control API.

        Args:
            path: URL path (e.g. ``/remote/info``).
            timeout_override: Per-call timeout in seconds.
            max_retries: Override for ``self.max_retries``.

        Returns:
            Parsed JSON response as a dictionary.
        """
        return await self._http_request(
            "GET", path, timeout_override=timeout_override, max_retries=max_retries,
        )

    async def _http_put(
        self,
        path: str,
        body: Dict[str, Any],
        timeout_override: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform an HTTP PUT against the Remote Control API.

        Args:
            path: URL path (e.g. ``/remote/object/call``).
            body: JSON-serializable request body.
            timeout_override: Per-call timeout in seconds.
            max_retries: Override for ``self.max_retries``.

        Returns:
            Parsed JSON response as a dictionary.
        """
        return await self._http_request(
            "PUT", path, body=body,
            timeout_override=timeout_override, max_retries=max_retries,
        )

    async def _http_post(
        self,
        path: str,
        body: Dict[str, Any],
        timeout_override: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform an HTTP POST against the Remote Control API.

        Args:
            path: URL path.
            body: JSON-serializable request body.
            timeout_override: Per-call timeout in seconds.
            max_retries: Override for ``self.max_retries``.

        Returns:
            Parsed JSON response as a dictionary.
        """
        return await self._http_request(
            "POST", path, body=body,
            timeout_override=timeout_override, max_retries=max_retries,
        )

    # ------------------------------------------------------------------
    # Ping / discovery
    # ------------------------------------------------------------------

    async def ping(self) -> Dict[str, Any]:
        """
        Lightweight reachability probe — short timeout, no retries.

        Returns:
            Dictionary with ``reachable``, ``address``, ``latency_ms``, and
            optional ``error`` keys.
        """
        address = f"{self.host}:{self.port}"
        t0 = time.monotonic()
        try:
            await self._http_get(
                "/remote/info",
                timeout_override=self.ping_timeout,
                max_retries=0,
            )
            latency_ms = round((time.monotonic() - t0) * 1000, 1)
            return {
                "reachable": True,
                "address": address,
                "latency_ms": latency_ms,
            }
        except Exception as exc:
            return {
                "reachable": False,
                "address": address,
                "latency_ms": None,
                "error": str(exc),
            }

    @staticmethod
    async def probe_port(
        host: str,
        port: int,
        timeout: float = 3.0,
    ) -> Dict[str, Any]:
        """
        Probe a single host:port for a running Remote Control API.

        Creates an ephemeral aiohttp session so the caller does not need
        a fully configured ``UnrealRuntimeSession``.

        Args:
            host: Target hostname or IP.
            port: Target port.
            timeout: Connection timeout in seconds.

        Returns:
            Dictionary with ``reachable``, ``host``, ``port``, ``latency_ms``,
            and optional ``engine_version`` / ``project_name``.
        """
        if not AIOHTTP_AVAILABLE:
            return {"reachable": False, "host": host, "port": port, "error": "aiohttp not available"}

        base_url = f"http://{host}:{port}"
        t0 = time.monotonic()
        try:
            timeout_cfg = aiohttp.ClientTimeout(total=timeout)
            async with aiohttp.ClientSession(
                base_url=base_url,
                timeout=timeout_cfg,
                headers={"Content-Type": "application/json"},
            ) as session:
                async with session.get("/remote/info") as resp:
                    resp.raise_for_status()
                    latency_ms = round((time.monotonic() - t0) * 1000, 1)

                engine_version: Optional[str] = None
                project_name: Optional[str] = None
                try:
                    async with session.put(
                        "/remote/object/call",
                        json={
                            "objectPath": "/Script/Engine.Default__KismetSystemLibrary",
                            "functionName": "GetEngineVersion",
                        },
                    ) as ver_resp:
                        if ver_resp.status == 200:
                            ver_data = await ver_resp.json()
                            engine_version = ver_data.get("ReturnValue")
                except Exception:
                    pass

                try:
                    async with session.put(
                        "/remote/object/call",
                        json={
                            "objectPath": "/Script/PythonScriptPlugin.Default__PythonScriptLibrary",
                            "functionName": "ExecutePythonCommandEx",
                            "parameters": {
                                "PythonCommand": "unreal.SystemLibrary.get_game_name()",
                                "ExecutionMode": "EvaluateStatement",
                                "FileExecutionScope": "Public",
                            },
                        },
                    ) as proj_resp:
                        if proj_resp.status == 200:
                            proj_data = await proj_resp.json()
                            project_name = proj_data.get("CommandResult", "").strip("'\"")
                except Exception:
                    pass

                return {
                    "reachable": True,
                    "host": host,
                    "port": port,
                    "latency_ms": latency_ms,
                    "engine_version": engine_version,
                    "project_name": project_name,
                }
        except Exception as exc:
            return {
                "reachable": False,
                "host": host,
                "port": port,
                "latency_ms": None,
                "error": str(exc),
            }


    # ------------------------------------------------------------------
    # Internal helpers — Remote Control wrappers
    # ------------------------------------------------------------------

    async def _call_function(
        self,
        object_path: str,
        function_name: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Call a UObject function via ``PUT /remote/object/call``.

        Args:
            object_path: Full UObject path (actor, CDO, subsystem).
            function_name: Blueprint-visible function name.
            parameters: Optional function parameters.

        Returns:
            Parsed JSON response dictionary.
        """
        body: Dict[str, Any] = {
            "objectPath": object_path,
            "functionName": function_name,
        }
        if parameters:
            body["parameters"] = parameters
        return await self._http_put("/remote/object/call", body)

    async def _execute_python(
        self,
        code: str,
        mode: str = "ExecuteFile",
    ) -> Dict[str, Any]:
        """
        Execute Python code inside UE5 via ``PythonScriptLibrary``.

        Args:
            code: Python source code.
            mode: ``EvaluateStatement`` (single expression, returns value),
                  ``ExecuteFile`` (multi-line, captures print/log),
                  ``ExecuteStatement`` (single statement).

        Returns:
            Dictionary with ``CommandResult``, ``LogOutput``, ``ReturnValue``.
        """
        return await self._call_function(
            "/Script/PythonScriptPlugin.Default__PythonScriptLibrary",
            "ExecutePythonCommandEx",
            {
                "PythonCommand": code,
                "ExecutionMode": mode,
                "FileExecutionScope": "Public",
            },
        )

    @staticmethod
    def _parse_python_json(result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract the first JSON object printed by an ``ExecuteFile`` script.

        Args:
            result: Raw dict returned by ``_execute_python``.

        Returns:
            Parsed dict from the first JSON line, or ``{"error": "..."}``
            if the script failed or produced no JSON.
        """
        if not result.get("ReturnValue", False):
            error_msg = result.get("CommandResult", "Unknown Python error")
            return {"error": str(error_msg)}

        for entry in result.get("LogOutput", []):
            if entry.get("Type") != "Info":
                continue
            output = entry.get("Output", "").strip()
            if output.startswith("{"):
                try:
                    return json.loads(output)
                except json.JSONDecodeError:
                    continue
        return {"error": "No JSON output from Python execution"}

    # ------------------------------------------------------------------
    # Phase 0 session methods
    # ------------------------------------------------------------------

    async def health_check(self) -> Dict[str, Any]:
        """
        Check connectivity to the Remote Control API.

        ``connected`` reflects only that ``GET /remote/info`` returned
        successfully. Engine version and project name are best-effort
        metadata — they may be empty on UE versions that gate CDO function
        calls (e.g. UE 5.3 rejects ``Default__KismetSystemLibrary`` and
        ``Default__PythonScriptLibrary`` access via Remote Control with
        "cannot be accessed remotely, check remote control project
        settings"). Such failures must NOT flip ``connected`` to False —
        the setup polling loop depends on this method to know when Remote
        Control is up.

        Returns:
            Dictionary with ``connected`` boolean. When connected, also
            includes best-effort ``engine_version``, ``project_name``,
            ``is_editor``, and per-probe ``warnings`` (any non-fatal
            metadata-fetch errors).
        """
        try:
            await self._http_get("/remote/info")
        except Exception as exc:
            self.logger.warning("Health check connectivity failed: %s", exc)
            return {
                "connected": False,
                "error": str(exc),
            }

        warnings: List[str] = []
        engine_version = ""
        project_name = ""
        try:
            version_data = await self._call_function(
                "/Script/Engine.Default__KismetSystemLibrary",
                "GetEngineVersion",
            )
            engine_version = version_data.get("ReturnValue", "")
        except Exception as exc:
            warnings.append(f"engine_version unavailable: {exc}")

        try:
            project_result = await self._execute_python(
                "unreal.SystemLibrary.get_game_name()",
                mode="EvaluateStatement",
            )
            project_name = project_result.get("CommandResult", "").strip("'\"")
        except Exception as exc:
            warnings.append(f"project_name unavailable: {exc}")

        result: Dict[str, Any] = {
            "connected": True,
            "engine_version": engine_version,
            "project_name": project_name,
            "is_editor": True,
        }
        if warnings:
            result["warnings"] = warnings
        return result

    async def get_engine_info(self) -> Dict[str, Any]:
        """
        Retrieve Unreal Engine runtime information.

        Returns:
            Dictionary with engine version, project name, loaded map, and
            editor state.
        """
        version_data = await self._call_function(
            "/Script/Engine.Default__KismetSystemLibrary",
            "GetEngineVersion",
        )
        project_result = await self._execute_python(
            "unreal.SystemLibrary.get_game_name()",
            mode="EvaluateStatement",
        )
        map_result = await self._execute_python(
            "unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)"
            ".get_editor_world().get_path_name()",
            mode="EvaluateStatement",
        )
        platform_result = await self._execute_python(
            "unreal.Paths.project_dir()",
            mode="EvaluateStatement",
        )
        return {
            "engine_version": version_data.get("ReturnValue", ""),
            "project_name": project_result.get("CommandResult", "").strip("'\""),
            "loaded_map": map_result.get("CommandResult", "").strip("'\""),
            "is_editor": True,
            "is_game": False,
            "platform": platform_result.get("CommandResult", "").strip("'\""),
        }

    async def get_loaded_map(self) -> Dict[str, Any]:
        """
        Return the currently loaded persistent level path.

        Returns:
            Dictionary with ``map_path`` string.
        """
        result = await self._execute_python(
            "unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)"
            ".get_editor_world().get_path_name()",
            mode="EvaluateStatement",
        )
        return {
            "map_path": result.get("CommandResult", "").strip("'\""),
        }

    # ------------------------------------------------------------------
    # Phase 1 session methods — Scene Read Operations
    # ------------------------------------------------------------------

    async def list_actors(
        self,
        class_filter: Optional[str] = None,
        tag_filter: Optional[str] = None,
        max_results: int = 200,
    ) -> Dict[str, Any]:
        """
        List actors in the current level.

        Uses Remote Control ``PUT /remote/object/call`` on
        ``EditorActorSubsystem.GetAllLevelActors``.

        Args:
            class_filter: Optional UClass name filter.
            tag_filter: Optional tag filter.
            max_results: Maximum actors to return.

        Returns:
            Dictionary with ``actors`` list, ``count``, ``truncated`` flag.
        """
        data = await self._call_function(
            "/Script/UnrealEd.Default__EditorActorSubsystem",
            "GetAllLevelActors",
        )
        raw_actors: list = data.get("ReturnValue", [])

        actors: list = []
        for actor_path in raw_actors:
            if len(actors) >= max_results:
                break
            # Fetch minimal info per actor via describe
            try:
                info = await self._describe_actor_brief(actor_path)
            except Exception:
                info = {
                    "name": actor_path.rsplit(".", 1)[-1] if "." in actor_path else actor_path,
                    "path": actor_path,
                    "class_name": "Unknown",
                    "location": (0.0, 0.0, 0.0),
                    "rotation": (0.0, 0.0, 0.0),
                    "scale": (1.0, 1.0, 1.0),
                    "tags": [],
                }

            if class_filter and info.get("class_name") != class_filter:
                continue
            if tag_filter and tag_filter not in info.get("tags", []):
                continue
            actors.append(info)

        return {
            "actors": actors,
            "count": len(actors),
            "truncated": len(raw_actors) > max_results,
        }

    async def get_actor_info(self, actor_path: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific actor.

        Args:
            actor_path: Full object path of the actor.

        Returns:
            Dictionary with actor properties, components, and transform.
        """
        # Describe the object to get class + properties
        describe_body: Dict[str, Any] = {"objectPath": actor_path}
        desc_data = await self._http_put("/remote/object/describe", describe_body)

        class_name: str = desc_data.get("Class", "Unknown")
        name: str = desc_data.get("Name", actor_path.rsplit(".", 1)[-1])

        # Get transform properties
        transform = await self._get_actor_transform(actor_path)

        # Get components list
        components: list = []
        for comp in desc_data.get("Components", []):
            components.append({
                "name": comp.get("Name", ""),
                "class_name": comp.get("Class", ""),
                "is_root": comp.get("IsRootComponent", False),
            })

        # Tags
        tags: list = desc_data.get("Tags", [])
        mobility: str = desc_data.get("Mobility", "Static")
        is_hidden: bool = desc_data.get("bHidden", False)

        return {
            "name": name,
            "path": actor_path,
            "class_name": class_name,
            "location": transform["location"],
            "rotation": transform["rotation"],
            "scale": transform["scale"],
            "components": components,
            "tags": tags,
            "mobility": mobility,
            "is_hidden": is_hidden,
        }

    async def search_assets(
        self,
        query: str = "",
        class_names: Optional[List[str]] = None,
        package_paths: Optional[List[str]] = None,
        max_results: int = 100,
    ) -> Dict[str, Any]:
        """
        Search the Unreal Asset Registry.

        Args:
            query: Search query string.
            class_names: Filter by UClass names.
            package_paths: Package path prefixes to search within.
            max_results: Maximum number of results.

        Returns:
            Dictionary with ``assets`` list, ``count``, ``truncated``.
        """
        body: Dict[str, Any] = {
            "Query": query,
            "Limit": max_results,
        }
        if class_names:
            body["Filter"] = {"ClassNames": class_names}
        if package_paths:
            body.setdefault("Filter", {})["PackagePaths"] = package_paths

        data = await self._http_put("/remote/search/assets", body)
        raw_assets: list = data.get("Assets", [])

        assets: list = []
        for asset in raw_assets[:max_results]:
            assets.append({
                "name": asset.get("Name", ""),
                "path": asset.get("Path", ""),
                "class_name": asset.get("Class", ""),
                "package_path": asset.get("PackagePath", ""),
            })

        return {
            "assets": assets,
            "count": len(assets),
            "truncated": len(raw_assets) > max_results,
        }

    async def describe_object(self, object_path: str) -> Dict[str, Any]:
        """
        Get full property/function metadata for a UObject.

        Args:
            object_path: Full object path.

        Returns:
            Dictionary with ``object_path``, ``class_name``, ``properties``,
            ``functions``.
        """
        body: Dict[str, Any] = {"objectPath": object_path}
        data = await self._http_put("/remote/object/describe", body)

        properties: list = []
        for prop in data.get("Properties", []):
            properties.append({
                "name": prop.get("Name", ""),
                "type": prop.get("Type", ""),
                "value": prop.get("Value"),
            })

        functions: list = [f.get("Name", "") for f in data.get("Functions", [])]

        return {
            "object_path": object_path,
            "class_name": data.get("Class", "Unknown"),
            "properties": properties,
            "functions": functions,
        }

    async def get_actor_thumbnail(
        self,
        asset_path: str,
        width: int = 256,
        height: int = 256,
    ) -> Dict[str, Any]:
        """
        Get a thumbnail image for an asset.

        Args:
            asset_path: Full asset path.
            width: Thumbnail width in pixels.
            height: Thumbnail height in pixels.

        Returns:
            Dictionary with ``asset_path``, ``image_base64``, ``width``,
            ``height``.
        """
        body: Dict[str, Any] = {
            "objectPath": asset_path,
            "Width": width,
            "Height": height,
        }
        data = await self._http_put("/remote/object/thumbnail", body)

        return {
            "asset_path": asset_path,
            "image_base64": data.get("Thumbnail", ""),
            "width": width,
            "height": height,
        }

    async def summarize_scene(self) -> Dict[str, Any]:
        """
        Generate an LLM-friendly scene digest.

        Returns:
            Dictionary with map_path, total_actors, actor_class_counts,
            and summary_text.
        """
        # Get map info via Python (non-deprecated path)
        map_result = await self._execute_python(
            "unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)"
            ".get_editor_world().get_path_name()",
            mode="EvaluateStatement",
        )
        map_path: str = map_result.get("CommandResult", "").strip("'\"")

        # Get all actors via EditorActorSubsystem (UE 5.7 compatible)
        data = await self._call_function(
            "/Script/UnrealEd.Default__EditorActorSubsystem",
            "GetAllLevelActors",
        )
        raw_actors: list = data.get("ReturnValue", [])

        # Count by class — brief describe each actor
        class_counts: Dict[str, int] = {}
        for actor_path in raw_actors:
            try:
                desc = await self._http_put(
                    "/remote/object/describe", {"objectPath": actor_path}
                )
                cls = desc.get("Class", "Unknown")
            except Exception:
                cls = "Unknown"
            class_counts[cls] = class_counts.get(cls, 0) + 1

        static_meshes = class_counts.get("StaticMeshActor", 0)
        lights = sum(v for k, v in class_counts.items() if "Light" in k)
        cameras = sum(v for k, v in class_counts.items() if "Camera" in k)

        parts: list = [f"Map: {map_path}", f"Total actors: {len(raw_actors)}"]
        for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1])[:10]:
            parts.append(f"  {cls}: {cnt}")

        return {
            "map_path": map_path,
            "total_actors": len(raw_actors),
            "actor_class_counts": class_counts,
            "static_meshes": static_meshes,
            "lights": lights,
            "cameras": cameras,
            "summary_text": "\n".join(parts),
        }

    # ------------------------------------------------------------------
    # Phase 2 — Viewport & Visual Observation
    # ------------------------------------------------------------------

    async def capture_viewport(
        self,
        resolution_x: int = 1920,
        resolution_y: int = 1080,
        format: str = "png",
    ) -> Dict[str, Any]:
        """
        Capture viewport screenshot via HighResScreenshot console command.

        The command writes a file to ``Project/Saved/Screenshots/``.
        We then read it back and return base64-encoded data.

        Args:
            resolution_x: Capture width in pixels.
            resolution_y: Capture height in pixels.
            format: Image format — ``png`` or ``jpeg``.

        Returns:
            Dictionary with image_base64, resolution_x, resolution_y, format.
        """
        # The trigger goes through Remote Control's ExecuteConsoleCommand
        # RPC (NOT through Python's unreal.SystemLibrary.execute_console_command
        # — empirically that path doesn't fire HighResShot in the editor).
        # The read is a separate ExecutePythonCommandEx call so UE's main
        # thread stays free between trigger and read; the adapter, not the
        # embedded script, owns the wait via asyncio.sleep.
        marker = "@@SIMUL_SCREENSHOT@@"
        candidates_repr = (
            "['MacEditor', 'WindowsEditor', 'LinuxEditor', 'Mac', 'Windows', 'Linux']"
        )
        threshold = time.time()
        try:
            await self._call_function(
                "/Script/Engine.Default__KismetSystemLibrary",
                "ExecuteConsoleCommand",
                {
                    "WorldContextObject": "",
                    "Command": f"HighResShot {resolution_x}x{resolution_y}",
                },
            )
        except Exception:
            return {
                "image_base64": "",
                "resolution_x": resolution_x,
                "resolution_y": resolution_y,
                "format": format,
            }

        read_code = f"""
import os, glob, base64, unreal
saved = unreal.Paths.project_saved_dir()
ss_root = os.path.join(saved, 'Screenshots')
candidates = {candidates_repr}
threshold = {threshold!r}
target = None
for subdir in candidates:
    d = os.path.join(ss_root, subdir)
    if not os.path.isdir(d):
        continue
    for f in glob.glob(os.path.join(d, '*.{format}')):
        try:
            if os.path.getmtime(f) >= threshold:
                target = f
                break
        except OSError:
            pass
    if target:
        break
data = ''
if target:
    try:
        size = os.path.getsize(target)
    except OSError:
        size = 0
    if size > 0:
        with open(target, 'rb') as f:
            data = base64.b64encode(f.read()).decode()
print('{marker}' + data)
"""

        image_data = ""
        # Initial render budget so UE's frame loop runs HighResShot.
        await asyncio.sleep(0.5)
        deadline = asyncio.get_event_loop().time() + 15.0
        while True:
            try:
                py_result = await self._execute_python(read_code, mode="ExecuteFile")
            except Exception:
                py_result = {}
            for entry in py_result.get("LogOutput", []):
                output = entry.get("Output", "")
                idx = output.find(marker)
                if idx >= 0:
                    image_data = output[idx + len(marker):].strip()
                    break
            if image_data:
                break
            if asyncio.get_event_loop().time() >= deadline:
                break
            await asyncio.sleep(0.5)

        return {
            "image_base64": image_data,
            "resolution_x": resolution_x,
            "resolution_y": resolution_y,
            "format": format,
        }

    async def get_viewport_info(self) -> Dict[str, Any]:
        """
        Get active editor viewport camera and render information.

        Reads the LevelEditorViewportClient properties via Remote Control.

        Returns:
            Dictionary with camera_location, camera_rotation, viewport_size,
            fov, projection_type.
        """
        # Read viewport camera via UnrealEditorSubsystem (UE 5.7 compatible)
        try:
            data = await self._call_function(
                "/Script/UnrealEd.Default__UnrealEditorSubsystem",
                "GetLevelViewportCameraInfo",
            )
            loc = data.get("CameraLocation", {})
            rot = data.get("CameraRotation", {})
            location = (
                loc.get("X", 0.0),
                loc.get("Y", 0.0),
                loc.get("Z", 0.0),
            )
            rotation = (
                rot.get("Pitch", 0.0),
                rot.get("Yaw", 0.0),
                rot.get("Roll", 0.0),
            )
        except Exception:
            location = (0.0, 0.0, 0.0)
            rotation = (0.0, 0.0, 0.0)

        return {
            "camera_location": location,
            "camera_rotation": rotation,
            "viewport_size": (1920, 1080),
            "fov": 90.0,
            "projection_type": "Perspective",
        }

    async def set_camera_view(
        self,
        location: Optional[tuple] = None,
        rotation: Optional[tuple] = None,
        fov: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Set the editor viewport camera position and rotation.

        Args:
            location: Camera position (X, Y, Z) in cm.
            rotation: Camera rotation (Pitch, Yaw, Roll) in degrees.
            fov: Field of view in degrees.

        Returns:
            Dictionary with applied location, rotation, fov.
        """
        applied_location = location or (0.0, 0.0, 0.0)
        applied_rotation = rotation or (0.0, 0.0, 0.0)
        applied_fov = fov or 90.0

        # Build SetLevelViewportCameraInfo call
        params: Dict[str, Any] = {}
        if location is not None:
            params["CameraLocation"] = {
                "X": location[0],
                "Y": location[1],
                "Z": location[2],
            }
        if rotation is not None:
            params["CameraRotation"] = {
                "Pitch": rotation[0],
                "Yaw": rotation[1],
                "Roll": rotation[2],
            }

        if params:
            await self._call_function(
                "/Script/UnrealEd.Default__UnrealEditorSubsystem",
                "SetLevelViewportCameraInfo",
                params,
            )

        return {
            "location": applied_location,
            "rotation": applied_rotation,
            "fov": applied_fov,
        }

    async def focus_on_actor(
        self,
        actor_path: str,
        distance: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Focus the editor viewport camera on a specific actor.

        Uses ``EditorActorSubsystem.SetActorSelectionState`` to select the
        actor, then positions the camera to look at the actor via Python.

        Args:
            actor_path: Full actor path to focus on.
            distance: Camera distance from actor (0 = auto-fit).

        Returns:
            Dictionary with actor_path, camera_location, camera_rotation.
        """
        # Select the actor via EditorActorSubsystem
        await self._call_function(
            "/Script/UnrealEd.Default__EditorActorSubsystem",
            "SetActorSelectionState",
            {"Actor": actor_path, "bShouldBeSelected": True},
        )

        # Position camera to look at actor. Read actor location, compute
        # a camera position offset behind and above, then set camera.
        dist = distance if distance > 0.0 else 500.0
        focus_code = (
            "import unreal, math\n"
            f"actor = unreal.EditorActorSubsystem().get_default_object()"
            " # unused\n"
            f"loc_body = {{'objectPath': '{actor_path}', "
            "'functionName': 'GetActorLocation'}}\n"
        )
        # Simpler approach: read location, set camera via subsystem
        transform = await self._get_actor_transform(actor_path)
        actor_loc = transform["location"]
        cam_x = actor_loc[0] - dist
        cam_y = actor_loc[1]
        cam_z = actor_loc[2] + dist * 0.5
        # Compute pitch to look at actor
        dz = actor_loc[2] - cam_z
        dxy = dist
        pitch = -math.degrees(math.atan2(-dz, dxy)) if dxy > 0 else -20.0
        await self._call_function(
            "/Script/UnrealEd.Default__UnrealEditorSubsystem",
            "SetLevelViewportCameraInfo",
            {
                "CameraLocation": {"X": cam_x, "Y": cam_y, "Z": cam_z},
                "CameraRotation": {"Pitch": pitch, "Yaw": 0.0, "Roll": 0.0},
            },
        )

        # Read back viewport camera position
        info = await self.get_viewport_info()

        return {
            "actor_path": actor_path,
            "camera_location": info["camera_location"],
            "camera_rotation": info["camera_rotation"],
        }

    # ------------------------------------------------------------------
    # Phase 3 — Scene Manipulation
    # ------------------------------------------------------------------

    async def spawn_actor(
        self,
        asset_path: str,
        location: tuple = (0.0, 0.0, 0.0),
        rotation: tuple = (0.0, 0.0, 0.0),
        label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Spawn an actor from a class or asset path.

        Uses ``EditorActorSubsystem.SpawnActorFromClass`` via Remote Control.

        Args:
            asset_path: Asset or class path to spawn from.
            location: Spawn location (X, Y, Z) in cm.
            rotation: Spawn rotation (Pitch, Yaw, Roll) in degrees.
            label: Optional actor label in the outliner.

        Returns:
            Dictionary with actor_path, actor_class, location.
        """
        data = await self._call_function(
            "/Script/UnrealEd.Default__EditorActorSubsystem",
            "SpawnActorFromClass",
            {
                "ActorClass": asset_path,
                "Location": {
                    "X": location[0],
                    "Y": location[1],
                    "Z": location[2],
                },
                "Rotation": {
                    "Pitch": rotation[0],
                    "Yaw": rotation[1],
                    "Roll": rotation[2],
                },
            },
        )
        actor_path_result = data.get("ReturnValue", "")

        if label and actor_path_result:
            label_body: Dict[str, Any] = {
                "objectPath": actor_path_result,
                "propertyName": "ActorLabel",
                "propertyValue": {"ActorLabel": label},
                "access": "WRITE_TRANSACTION_ACCESS",
            }
            try:
                await self._http_put("/remote/object/property", label_body)
            except Exception:
                pass  # Label is best-effort

        return {
            "actor_path": actor_path_result,
            "actor_class": asset_path.split(".")[-1] if "." in asset_path else asset_path,
            "location": location,
        }

    async def delete_actor(self, actor_path: str) -> Dict[str, Any]:
        """
        Delete an actor from the level.

        Uses ``EditorActorSubsystem.DestroyActor`` via Remote Control.

        Args:
            actor_path: Full actor path to delete.

        Returns:
            Dictionary with actor_path, deleted status.
        """
        data = await self._call_function(
            "/Script/UnrealEd.Default__EditorActorSubsystem",
            "DestroyActor",
            {"ActorToDestroy": actor_path},
        )
        deleted = data.get("ReturnValue", False)

        return {
            "actor_path": actor_path,
            "deleted": bool(deleted),
        }

    async def set_actor_transform(
        self,
        actor_path: str,
        location: Optional[tuple] = None,
        rotation: Optional[tuple] = None,
        scale: Optional[tuple] = None,
    ) -> Dict[str, Any]:
        """
        Set an actor's transform (location, rotation, scale).

        Uses direct function calls on the actor: ``K2_SetActorLocation``,
        ``K2_SetActorRotation``, ``SetActorScale3D``.  Property writes on
        ``RelativeLocation`` fail on the actor object — they must target
        the component path, so function calls are the reliable approach.

        Args:
            actor_path: Full actor path.
            location: New location (X, Y, Z) in cm.
            rotation: New rotation (Pitch, Yaw, Roll) in degrees.
            scale: New scale (X, Y, Z).

        Returns:
            Dictionary with actor_path, applied location, rotation, scale.
        """
        applied_location = location or (0.0, 0.0, 0.0)
        applied_rotation = rotation or (0.0, 0.0, 0.0)
        applied_scale = scale or (1.0, 1.0, 1.0)

        if location is not None:
            await self._call_function(
                actor_path,
                "K2_SetActorLocation",
                {
                    "NewLocation": {
                        "X": location[0],
                        "Y": location[1],
                        "Z": location[2],
                    },
                    "bSweep": False,
                    "bTeleport": True,
                },
            )

        if rotation is not None:
            await self._call_function(
                actor_path,
                "K2_SetActorRotation",
                {
                    "NewRotation": {
                        "Pitch": rotation[0],
                        "Yaw": rotation[1],
                        "Roll": rotation[2],
                    },
                    "bTeleportPhysics": True,
                },
            )

        if scale is not None:
            await self._call_function(
                actor_path,
                "SetActorScale3D",
                {
                    "NewScale3D": {
                        "X": scale[0],
                        "Y": scale[1],
                        "Z": scale[2],
                    },
                },
            )

        return {
            "actor_path": actor_path,
            "location": applied_location,
            "rotation": applied_rotation,
            "scale": applied_scale,
        }

    async def set_actor_property(
        self,
        actor_path: str,
        property_name: str,
        property_value: str,
        generate_transaction: bool = True,
    ) -> Dict[str, Any]:
        """
        Set any writable property on an actor.

        Args:
            actor_path: Full actor path.
            property_name: Property name to set.
            property_value: Property value as JSON string.
            generate_transaction: Whether to generate an undo transaction.

        Returns:
            Dictionary with actor_path, property_name.
        """
        import json as json_lib

        try:
            parsed_value = json_lib.loads(property_value)
        except (json_lib.JSONDecodeError, TypeError):
            parsed_value = property_value

        access = (
            "WRITE_TRANSACTION_ACCESS" if generate_transaction else "WRITE_ACCESS"
        )
        body: Dict[str, Any] = {
            "objectPath": actor_path,
            "propertyName": property_name,
            "propertyValue": {property_name: parsed_value},
            "access": access,
        }
        await self._http_put("/remote/object/property", body)

        return {
            "actor_path": actor_path,
            "property_name": property_name,
        }

    async def call_actor_function(
        self,
        actor_path: str,
        function_name: str,
        parameters: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Call a BlueprintCallable function on an actor.

        Args:
            actor_path: Full actor path.
            function_name: Function name to call.
            parameters: Parameters as JSON string.

        Returns:
            Dictionary with actor_path, function_name, return_value.
        """
        import json as json_lib

        body: Dict[str, Any] = {
            "objectPath": actor_path,
            "functionName": function_name,
        }
        if parameters:
            try:
                body["parameters"] = json_lib.loads(parameters)
            except (json_lib.JSONDecodeError, TypeError):
                body["parameters"] = {}

        data = await self._http_put("/remote/object/call", body)
        return_value = json_lib.dumps(data) if data else None

        return {
            "actor_path": actor_path,
            "function_name": function_name,
            "return_value": return_value,
        }

    async def set_actor_parent(
        self,
        actor_path: str,
        parent_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Attach an actor to a parent actor or detach it.

        Uses ``EditorLevelLibrary.SetActorSelectionState`` and the
        ``AttachToActor`` UFUNCTION via Remote Control.

        Args:
            actor_path: Child actor path.
            parent_path: Parent actor path (None to detach).

        Returns:
            Dictionary with actor_path, parent_path.
        """
        if parent_path:
            body: Dict[str, Any] = {
                "objectPath": actor_path,
                "functionName": "K2_AttachToActor",
                "parameters": {
                    "ParentActor": parent_path,
                    "SocketName": "None",
                    "LocationRule": "KeepWorld",
                    "RotationRule": "KeepWorld",
                },
            }
        else:
            body = {
                "objectPath": actor_path,
                "functionName": "K2_DetachFromActor",
                "parameters": {
                    "LocationRule": "KeepWorld",
                    "RotationRule": "KeepWorld",
                },
            }
        await self._http_put("/remote/object/call", body)

        return {
            "actor_path": actor_path,
            "parent_path": parent_path,
        }

    async def add_component(
        self,
        actor_path: str,
        component_class: str,
        component_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Add a component to an actor.

        Uses ``AddComponentByClass`` UFUNCTION via Remote Control.

        Args:
            actor_path: Full actor path.
            component_class: Component class name.
            component_name: Optional name for the component.

        Returns:
            Dictionary with actor_path, component_path, component_class.
        """
        body: Dict[str, Any] = {
            "objectPath": actor_path,
            "functionName": "AddComponentByClass",
            "parameters": {
                "Class": f"/Script/Engine.{component_class}",
                "bManualAttachment": False,
                "RelativeTransform": {
                    "Rotation": {"X": 0, "Y": 0, "Z": 0, "W": 1},
                    "Translation": {"X": 0, "Y": 0, "Z": 0},
                    "Scale3D": {"X": 1, "Y": 1, "Z": 1},
                },
                "bDeferredFinish": False,
            },
        }
        data = await self._http_put("/remote/object/call", body)
        component_path = data.get("ReturnValue", "")

        return {
            "actor_path": actor_path,
            "component_path": component_path,
            "component_class": component_class,
        }

    async def set_actor_visibility(
        self,
        actor_path: str,
        visible: bool = True,
        propagate: bool = True,
    ) -> Dict[str, Any]:
        """
        Set actor visibility.

        Uses ``SetActorHiddenInGame`` UFUNCTION via Remote Control.

        Args:
            actor_path: Full actor path.
            visible: Whether the actor should be visible.
            propagate: Propagate to child components/actors.

        Returns:
            Dictionary with actor_path, visible.
        """
        body: Dict[str, Any] = {
            "objectPath": actor_path,
            "functionName": "SetActorHiddenInGame",
            "parameters": {
                "bNewHidden": not visible,
                "bPropagateToChildren": propagate,
            },
        }
        await self._http_put("/remote/object/call", body)

        return {
            "actor_path": actor_path,
            "visible": visible,
        }

    # ------------------------------------------------------------------
    # Phase 4 — Materials, Lighting & Rendering
    # ------------------------------------------------------------------

    async def get_material_info(self, material_path: str) -> Dict[str, Any]:
        """
        Get material instance parameters and metadata.

        Uses ``PUT /remote/object/describe`` to inspect material properties.

        Args:
            material_path: Material or MIC asset path.

        Returns:
            Dictionary with material_path, parent_path, parameters list.
        """
        body: Dict[str, Any] = {
            "objectPath": material_path,
        }
        data = await self._http_put("/remote/object/describe", body)

        properties = data.get("Properties", [])
        parameters: list = []
        parent_path: Optional[str] = data.get("Parent", None)

        for prop in properties:
            prop_name = prop.get("Name", "")
            prop_type = prop.get("Type", "")
            prop_value = prop.get("DefaultValue", None)

            # Classify parameter type based on UE property type
            if "Scalar" in prop_type or "Float" in prop_type:
                param_type = "scalar"
            elif "Vector" in prop_type or "Color" in prop_type:
                param_type = "vector"
            elif "Texture" in prop_type or "Object" in prop_type:
                param_type = "texture"
            else:
                param_type = prop_type.lower() if prop_type else "unknown"

            parameters.append({
                "name": prop_name,
                "param_type": param_type,
                "value": prop_value,
            })

        return {
            "material_path": material_path,
            "parent_path": parent_path,
            "parameters": parameters,
        }

    async def set_material_params(
        self,
        material_path: str,
        scalar_params: Optional[Dict[str, float]] = None,
        vector_params: Optional[Dict[str, list]] = None,
        texture_params: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Set parameters on a Material Instance Constant.

        Uses ``PUT /remote/object/property`` to write parameter overrides.

        Args:
            material_path: Material Instance asset path.
            scalar_params: Scalar parameter name-value pairs.
            vector_params: Vector parameter name-value pairs (RGBA lists).
            texture_params: Texture parameter name-asset path pairs.

        Returns:
            Dictionary with material_path, params_set count.
        """
        params_set = 0

        if scalar_params:
            for name, value in scalar_params.items():
                body: Dict[str, Any] = {
                    "objectPath": material_path,
                    "propertyName": f"ScalarParameterValues",
                    "propertyValue": {
                        "ParameterInfo": {"Name": name},
                        "ParameterValue": value,
                    },
                    "access": "WRITE_TRANSACTION_ACCESS",
                }
                await self._http_put("/remote/object/property", body)
                params_set += 1

        if vector_params:
            for name, value in vector_params.items():
                body = {
                    "objectPath": material_path,
                    "propertyName": "VectorParameterValues",
                    "propertyValue": {
                        "ParameterInfo": {"Name": name},
                        "ParameterValue": {
                            "R": value[0] if len(value) > 0 else 0.0,
                            "G": value[1] if len(value) > 1 else 0.0,
                            "B": value[2] if len(value) > 2 else 0.0,
                            "A": value[3] if len(value) > 3 else 1.0,
                        },
                    },
                    "access": "WRITE_TRANSACTION_ACCESS",
                }
                await self._http_put("/remote/object/property", body)
                params_set += 1

        if texture_params:
            for name, asset_path in texture_params.items():
                body = {
                    "objectPath": material_path,
                    "propertyName": "TextureParameterValues",
                    "propertyValue": {
                        "ParameterInfo": {"Name": name},
                        "ParameterValue": asset_path,
                    },
                    "access": "WRITE_TRANSACTION_ACCESS",
                }
                await self._http_put("/remote/object/property", body)
                params_set += 1

        return {
            "material_path": material_path,
            "params_set": params_set,
        }

    async def create_material_instance(
        self,
        parent_path: str,
        instance_name: str,
        save_path: str = "",
    ) -> Dict[str, Any]:
        """
        Create a Material Instance Constant from a parent material.

        Uses ``EditorAssetLibrary.DuplicateAsset`` or
        ``MaterialInstanceConstantFactoryNew`` via Remote Control.

        Args:
            parent_path: Parent material asset path.
            instance_name: Name for the new MIC.
            save_path: Content-relative save directory (auto if empty).

        Returns:
            Dictionary with instance_path, parent_path.
        """
        if not save_path:
            # Derive save path from parent: /Game/Materials/M_Base -> /Game/Materials/
            parts = parent_path.rsplit("/", 1)
            save_path = parts[0] if len(parts) > 1 else "/Game"

        full_save_path = f"{save_path}/{instance_name}"

        body: Dict[str, Any] = {
            "objectPath": (
                "/Script/EditorScriptingUtilities"
                ".Default__EditorAssetLibrary"
            ),
            "functionName": "DuplicateAsset",
            "parameters": {
                "SourceAssetPath": parent_path,
                "DestinationAssetPath": full_save_path,
            },
        }
        data = await self._http_put("/remote/object/call", body)
        created = data.get("ReturnValue", False)

        instance_path = full_save_path if created else ""

        return {
            "instance_path": instance_path,
            "parent_path": parent_path,
        }

    async def assign_material(
        self,
        actor_path: str,
        material_path: str,
        slot_index: int = 0,
    ) -> Dict[str, Any]:
        """
        Assign a material to a mesh component's material slot.

        Uses ``PUT /remote/object/property`` to set ``OverrideMaterials``
        on the actor's first StaticMeshComponent.

        Args:
            actor_path: Target actor path.
            material_path: Material asset path to assign.
            slot_index: Material slot index (default 0).

        Returns:
            Dictionary with actor_path, material_path, slot_index.
        """
        body: Dict[str, Any] = {
            "objectPath": actor_path,
            "functionName": "SetMaterial",
            "parameters": {
                "ElementIndex": slot_index,
                "Material": material_path,
            },
        }
        await self._http_put("/remote/object/call", body)

        return {
            "actor_path": actor_path,
            "material_path": material_path,
            "slot_index": slot_index,
        }

    async def set_light_params(
        self,
        actor_path: str,
        intensity: Optional[float] = None,
        color_r: Optional[float] = None,
        color_g: Optional[float] = None,
        color_b: Optional[float] = None,
        temperature: Optional[float] = None,
        use_temperature: Optional[bool] = None,
        attenuation_radius: Optional[float] = None,
        cast_shadows: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Set light component parameters on a light actor.

        Uses ``PUT /remote/object/property`` for each parameter.

        Args:
            actor_path: Light actor path.
            intensity: Light intensity.
            color_r: Color red component (0-1).
            color_g: Color green component (0-1).
            color_b: Color blue component (0-1).
            temperature: Color temperature in Kelvin.
            use_temperature: Use color temperature instead of direct color.
            attenuation_radius: Light attenuation radius in cm.
            cast_shadows: Enable shadow casting.

        Returns:
            Dictionary with actor_path, params_set count.
        """
        params_set = 0

        # Map parameter names to UE property names and values
        property_map: list = []
        if intensity is not None:
            property_map.append(("Intensity", intensity))
        if color_r is not None or color_g is not None or color_b is not None:
            color = {
                "R": color_r if color_r is not None else 1.0,
                "G": color_g if color_g is not None else 1.0,
                "B": color_b if color_b is not None else 1.0,
                "A": 1.0,
            }
            property_map.append(("LightColor", color))
        if temperature is not None:
            property_map.append(("Temperature", temperature))
        if use_temperature is not None:
            property_map.append(("bUseTemperature", use_temperature))
        if attenuation_radius is not None:
            property_map.append(("AttenuationRadius", attenuation_radius))
        if cast_shadows is not None:
            property_map.append(("CastShadows", cast_shadows))

        for prop_name, prop_value in property_map:
            body: Dict[str, Any] = {
                "objectPath": actor_path,
                "propertyName": prop_name,
                "propertyValue": prop_value,
                "access": "WRITE_TRANSACTION_ACCESS",
            }
            await self._http_put("/remote/object/property", body)
            params_set += 1

        return {
            "actor_path": actor_path,
            "params_set": params_set,
        }

    async def set_render_settings(
        self,
        setting_name: str,
        setting_value: str,
    ) -> Dict[str, Any]:
        """
        Set rendering or post-process settings via console command.

        Uses ``ExecuteConsoleCommand`` on the KismetSystemLibrary.

        Args:
            setting_name: Render setting / console variable name.
            setting_value: Value as string.

        Returns:
            Dictionary with setting_name, applied status.
        """
        command = f"{setting_name} {setting_value}"
        body: Dict[str, Any] = {
            "objectPath": (
                "/Script/Engine.Default__KismetSystemLibrary"
            ),
            "functionName": "ExecuteConsoleCommand",
            "parameters": {
                "WorldContextObject": None,
                "Command": command,
            },
        }
        await self._http_put("/remote/object/call", body)

        return {
            "setting_name": setting_name,
            "applied": True,
        }

    # ------------------------------------------------------------------
    # Phase 5: Physics & Simulation Control
    # ------------------------------------------------------------------

    async def control_simulation(self, action: str) -> Dict[str, Any]:
        """
        Control a Play-In-Editor (PIE) session.

        Args:
            action: One of start, stop, pause, resume, step.

        Returns:
            Dict with action executed and resulting state.
        """
        action_lower = action.lower().strip()
        valid_actions = ("start", "stop", "pause", "resume", "step")
        if action_lower not in valid_actions:
            raise ValueError(
                f"Invalid PIE action '{action_lower}'. "
                f"Must be one of: {', '.join(valid_actions)}"
            )

        object_path = "/Script/UnrealEd.Default__EditorLevelLibrary"

        if action_lower == "start":
            body: Dict[str, Any] = {
                "objectPath": object_path,
                "functionName": "EditorPlaySimulate",
            }
            await self._http_put("/remote/object/call", body)
            state = "playing"

        elif action_lower == "stop":
            body = {
                "objectPath": object_path,
                "functionName": "EditorEndPlay",
            }
            await self._http_put("/remote/object/call", body)
            state = "stopped"

        elif action_lower == "pause":
            body = {
                "objectPath": object_path,
                "functionName": "EditorSetGamePaused",
                "parameters": {"bPaused": True},
            }
            await self._http_put("/remote/object/call", body)
            state = "paused"

        elif action_lower == "resume":
            body = {
                "objectPath": object_path,
                "functionName": "EditorSetGamePaused",
                "parameters": {"bPaused": False},
            }
            await self._http_put("/remote/object/call", body)
            state = "playing"

        else:  # step
            body = {
                "objectPath": "/Script/UnrealEd.Default__EditorLevelLibrary",
                "functionName": "EditorPlaySimulate",
            }
            await self._http_put("/remote/object/call", body)
            # Immediately pause after one tick
            pause_body: Dict[str, Any] = {
                "objectPath": object_path,
                "functionName": "EditorSetGamePaused",
                "parameters": {"bPaused": True},
            }
            await self._http_put("/remote/object/call", pause_body)
            state = "paused"

        return {
            "action": action_lower,
            "state": state,
        }

    async def get_simulation_status(self) -> Dict[str, Any]:
        """
        Query current PIE simulation status.

        Returns:
            Dict with is_playing, is_paused, frame_count, sim_time.
        """
        object_path = "/Script/UnrealEd.Default__EditorLevelLibrary"

        # Check if PIE is active via EditorGetGameView
        try:
            body: Dict[str, Any] = {
                "objectPath": object_path,
                "functionName": "IsPlayInEditorActive",
            }
            data = await self._http_put("/remote/object/call", body)
            is_playing = bool(data.get("ReturnValue", False))
        except Exception:
            is_playing = False

        # Check pause state
        is_paused = False
        if is_playing:
            try:
                body = {
                    "objectPath": object_path,
                    "functionName": "IsGamePaused",
                }
                data = await self._http_put("/remote/object/call", body)
                is_paused = bool(data.get("ReturnValue", False))
            except Exception:
                is_paused = False

        # Get frame count and sim time from GameState if available
        frame_count = 0
        sim_time = 0.0
        if is_playing:
            try:
                body = {
                    "objectPath": (
                        "/Script/Engine.Default__KismetSystemLibrary"
                    ),
                    "functionName": "GetGameTimeInSeconds",
                    "parameters": {
                        "WorldContextObject": object_path,
                    },
                }
                data = await self._http_put("/remote/object/call", body)
                sim_time = float(data.get("ReturnValue", 0.0))
            except Exception:
                sim_time = 0.0

        return {
            "is_playing": is_playing,
            "is_paused": is_paused,
            "frame_count": frame_count,
            "sim_time": sim_time,
        }

    async def enable_physics(
        self,
        actor_path: str,
        enable: bool = True,
        simulate_physics: bool = True,
    ) -> Dict[str, Any]:
        """
        Enable or disable physics simulation on an actor's root component.

        Args:
            actor_path: Full actor object path.
            enable: Whether physics should be enabled.
            simulate_physics: Whether the body actively simulates.

        Returns:
            Dict with actor_path and physics_enabled state.
        """
        body: Dict[str, Any] = {
            "objectPath": actor_path,
            "propertyName": "SimulatePhysics",
            "propertyValue": {"SimulatePhysics": simulate_physics and enable},
            "access": "WRITE_TRANSACTION_ACCESS",
        }
        await self._http_put("/remote/object/property", body)

        return {
            "actor_path": actor_path,
            "physics_enabled": enable and simulate_physics,
        }

    async def set_collision(
        self,
        actor_path: str,
        collision_preset: str = "",
        collision_enabled: bool = True,
    ) -> Dict[str, Any]:
        """
        Configure collision settings on an actor.

        Args:
            actor_path: Full actor object path.
            collision_preset: Named collision preset (e.g. BlockAll, NoCollision).
            collision_enabled: Whether collision is enabled.

        Returns:
            Dict with actor_path, collision_preset, collision_enabled.
        """
        if collision_preset:
            body: Dict[str, Any] = {
                "objectPath": actor_path,
                "functionName": "SetCollisionProfileName",
                "parameters": {"InCollisionProfileName": collision_preset},
            }
            await self._http_put("/remote/object/call", body)

        body = {
            "objectPath": actor_path,
            "propertyName": "CollisionEnabled",
            "propertyValue": {
                "CollisionEnabled": (
                    "QueryAndPhysics" if collision_enabled else "NoCollision"
                )
            },
            "access": "WRITE_TRANSACTION_ACCESS",
        }
        await self._http_put("/remote/object/property", body)

        return {
            "actor_path": actor_path,
            "collision_preset": collision_preset or "Custom",
            "collision_enabled": collision_enabled,
        }

    async def apply_force(
        self,
        actor_path: str,
        force_x: float = 0.0,
        force_y: float = 0.0,
        force_z: float = 0.0,
        is_impulse: bool = False,
        location_x: Optional[float] = None,
        location_y: Optional[float] = None,
        location_z: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Apply a force or impulse to an actor's physics body.

        Args:
            actor_path: Full actor object path.
            force_x: Force X component.
            force_y: Force Y component.
            force_z: Force Z component.
            is_impulse: True for impulse, False for continuous force.
            location_x: Application point X (None = center of mass).
            location_y: Application point Y.
            location_z: Application point Z.

        Returns:
            Dict with actor_path, force_applied, force_vector, is_impulse.
        """
        force_vector = {"X": force_x, "Y": force_y, "Z": force_z}

        if is_impulse:
            fn_name = "AddImpulse"
        else:
            fn_name = "AddForce"

        params: Dict[str, Any] = {fn_name[3:]: force_vector}

        if location_x is not None and location_y is not None and location_z is not None:
            fn_name = fn_name + "AtLocation"
            params["Location"] = {
                "X": location_x,
                "Y": location_y,
                "Z": location_z,
            }

        body: Dict[str, Any] = {
            "objectPath": actor_path,
            "functionName": fn_name,
            "parameters": params,
        }
        await self._http_put("/remote/object/call", body)

        return {
            "actor_path": actor_path,
            "force_applied": True,
            "force_vector": [force_x, force_y, force_z],
            "is_impulse": is_impulse,
        }

    async def set_physics_params(
        self,
        actor_path: str,
        mass: Optional[float] = None,
        linear_damping: Optional[float] = None,
        angular_damping: Optional[float] = None,
        enable_gravity: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Set physics body parameters on an actor.

        Args:
            actor_path: Full actor object path.
            mass: Mass in kg.
            linear_damping: Linear damping coefficient.
            angular_damping: Angular damping coefficient.
            enable_gravity: Whether gravity affects this body.

        Returns:
            Dict with actor_path and params_set count.
        """
        property_map: list[tuple[str, Any]] = []
        if mass is not None:
            property_map.append(("MassInKg", mass))
        if linear_damping is not None:
            property_map.append(("LinearDamping", linear_damping))
        if angular_damping is not None:
            property_map.append(("AngularDamping", angular_damping))
        if enable_gravity is not None:
            property_map.append(("bEnableGravity", enable_gravity))

        for prop_name, prop_value in property_map:
            body: Dict[str, Any] = {
                "objectPath": actor_path,
                "propertyName": prop_name,
                "propertyValue": {prop_name: prop_value},
                "access": "WRITE_TRANSACTION_ACCESS",
            }
            await self._http_put("/remote/object/property", body)

        return {
            "actor_path": actor_path,
            "params_set": len(property_map),
        }

    # ------------------------------------------------------------------
    # Coordinate conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def ue_to_usd_location(
        x_cm: float, y_cm: float, z_cm: float
    ) -> tuple:
        """
        Convert UE5 location (Z-up, cm, left-hand) to USD (Z-up, m, right-hand).

        UE5 left-hand → USD right-hand: negate Y.
        Scale: cm → m (divide by 100).

        Args:
            x_cm: X coordinate in centimetres.
            y_cm: Y coordinate in centimetres.
            z_cm: Z coordinate in centimetres.

        Returns:
            Tuple (x_m, y_m, z_m) in metres, right-hand Z-up.
        """
        return (x_cm / 100.0, -y_cm / 100.0, z_cm / 100.0)

    @staticmethod
    def usd_to_ue_location(
        x_m: float, y_m: float, z_m: float
    ) -> tuple:
        """
        Convert USD location (Z-up, m, right-hand) to UE5 (Z-up, cm, left-hand).

        Args:
            x_m: X coordinate in metres.
            y_m: Y coordinate in metres.
            z_m: Z coordinate in metres.

        Returns:
            Tuple (x_cm, y_cm, z_cm) in centimetres, left-hand Z-up.
        """
        return (x_m * 100.0, -y_m * 100.0, z_m * 100.0)

    @staticmethod
    def ue_to_usd_rotation(
        pitch: float, yaw: float, roll: float
    ) -> tuple:
        """
        Convert UE5 rotation to USD rotation.

        UE5 left-hand → USD right-hand: negate Yaw.

        Args:
            pitch: Pitch in degrees.
            yaw: Yaw in degrees.
            roll: Roll in degrees.

        Returns:
            Tuple (pitch, yaw, roll) in degrees for USD.
        """
        return (pitch, -yaw, roll)

    @staticmethod
    def usd_to_ue_rotation(
        pitch: float, yaw: float, roll: float
    ) -> tuple:
        """
        Convert USD rotation to UE5 rotation.

        Args:
            pitch: Pitch in degrees.
            yaw: Yaw in degrees.
            roll: Roll in degrees.

        Returns:
            Tuple (pitch, yaw, roll) in degrees for UE5.
        """
        return (pitch, -yaw, roll)

    # ------------------------------------------------------------------
    # Phase 6: USD / SimReady Bridge
    # ------------------------------------------------------------------

    async def import_usd(
        self,
        usd_path: str,
        destination_path: str = "/Game/Imports",
        import_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Import a USD file via Interchange Framework.

        Args:
            usd_path: Path to the USD file on disk.
            destination_path: Content browser destination.
            import_options: Interchange pipeline options.

        Returns:
            Dict with imported_assets, actor_paths, warnings.
        """
        body: Dict[str, Any] = {
            "objectPath": "/Script/InterchangeEngine.Default__InterchangeManager",
            "functionName": "ImportAssetAsync",
            "parameters": {
                "SourceData": usd_path,
                "DestinationPath": destination_path,
            },
        }
        if import_options:
            body["parameters"]["PipelineOptions"] = import_options

        response = await self._http_put("/remote/object/call", body)
        return {
            "imported_assets": response.get("ImportedAssets", []),
            "actor_paths": response.get("ActorPaths", []),
            "warnings": response.get("Warnings", []),
        }

    async def export_usd(
        self,
        actor_paths: List[str],
        output_path: str,
        export_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Export actors to USD via Interchange Framework.

        Args:
            actor_paths: Actor paths to export.
            output_path: Output USD file path.
            export_options: Export pipeline options.

        Returns:
            Dict with output_path, actors_exported, file_size_bytes.
        """
        body: Dict[str, Any] = {
            "objectPath": "/Script/InterchangeEngine.Default__InterchangeManager",
            "functionName": "ExportAsset",
            "parameters": {
                "ActorPaths": actor_paths,
                "OutputPath": output_path,
            },
        }
        if export_options:
            body["parameters"]["PipelineOptions"] = export_options

        response = await self._http_put("/remote/object/call", body)
        return {
            "output_path": response.get("OutputPath", output_path),
            "actors_exported": len(actor_paths),
            "file_size_bytes": response.get("FileSizeBytes", 0),
        }

    async def convert_to_simready(
        self,
        usd_path: str,
        output_path: str,
        add_physics: bool = True,
        add_collision: bool = True,
        add_semantic_labels: bool = True,
        target_up_axis: str = "Z",
        target_units: str = "meters",
    ) -> Dict[str, Any]:
        """
        Convert a USD asset to NVIDIA SimReady format.

        Applies physics schema, collision geometry, unit/axis correction,
        and semantic labels as needed.

        Args:
            usd_path: Source USD file.
            output_path: Output SimReady USD.
            add_physics: Add physics schema.
            add_collision: Generate collision geometry.
            add_semantic_labels: Add semantic labels.
            target_up_axis: Target up axis.
            target_units: Target unit system.

        Returns:
            Dict with output_path, conversions_applied, warnings.
        """
        body: Dict[str, Any] = {
            "objectPath": "/Script/InterchangeEngine.Default__InterchangeManager",
            "functionName": "ConvertToSimReady",
            "parameters": {
                "SourcePath": usd_path,
                "OutputPath": output_path,
                "AddPhysics": add_physics,
                "AddCollision": add_collision,
                "AddSemanticLabels": add_semantic_labels,
                "TargetUpAxis": target_up_axis,
                "TargetUnits": target_units,
            },
        }
        response = await self._http_put("/remote/object/call", body)
        return {
            "output_path": response.get("OutputPath", output_path),
            "conversions_applied": response.get("ConversionsApplied", []),
            "warnings": response.get("Warnings", []),
        }

    async def validate_simready_asset(
        self,
        usd_path: str,
        checks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Validate a USD asset against SimReady spec.

        Args:
            usd_path: USD file path to validate.
            checks: Validation checks to run.

        Returns:
            Dict with is_valid, per-check results, errors, suggestions.
        """
        if checks is None:
            checks = [
                "physics",
                "collision",
                "materials",
                "scale",
                "up_axis",
                "semantics",
            ]
        body: Dict[str, Any] = {
            "objectPath": "/Script/InterchangeEngine.Default__InterchangeManager",
            "functionName": "ValidateSimReadyAsset",
            "parameters": {
                "UsdPath": usd_path,
                "Checks": checks,
            },
        }
        response = await self._http_put("/remote/object/call", body)
        return {
            "usd_path": usd_path,
            "is_valid": response.get("IsValid", False),
            "checks": response.get("CheckResults", {}),
            "errors": response.get("Errors", []),
            "suggestions": response.get("Suggestions", []),
        }

    async def get_interchange_info(self) -> Dict[str, Any]:
        """
        Query available Interchange pipelines and supported formats.

        Returns:
            Dict with pipelines, supported_formats, interchange_version.
        """
        body: Dict[str, Any] = {
            "objectPath": "/Script/InterchangeEngine.Default__InterchangeManager",
            "functionName": "GetPipelineInfo",
            "parameters": {},
        }
        response = await self._http_put("/remote/object/call", body)
        return {
            "pipelines": response.get("Pipelines", []),
            "supported_formats": response.get("SupportedFormats", []),
            "interchange_version": response.get("Version", "unknown"),
        }

    # ------------------------------------------------------------------
    # Phase 7: Advanced Agent Tools
    # ------------------------------------------------------------------

    async def batch_operations(
        self,
        operations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Execute multiple Remote Control operations in one HTTP call.

        Args:
            operations: List of operation dicts, each with RequestId,
                Url, Verb, and Body keys matching the Remote Control batch API.

        Returns:
            Dict with results, total, succeeded, failed.
        """
        body: Dict[str, Any] = {"Requests": operations}
        response = await self._http_put("/remote/batch", body)
        responses = response.get("Responses", [])
        succeeded = sum(
            1 for r in responses if r.get("ResponseCode", 500) < 400
        )
        return {
            "results": responses,
            "total": len(operations),
            "succeeded": succeeded,
            "failed": len(operations) - succeeded,
        }

    async def query_scene_graph(
        self,
        root_path: Optional[str] = None,
        max_depth: int = 10,
        include_components: bool = False,
        class_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Query the scene graph hierarchy starting from a root actor.

        Args:
            root_path: Root actor path (None for level root).
            max_depth: Maximum traversal depth.
            include_components: Include component sub-trees.
            class_filter: Filter actors by UClass name.

        Returns:
            Dict with root tree, total_actors, total_depth.
        """
        # Fetch all actors first
        search_body: Dict[str, Any] = {
            "Query": "",
            "Filter": "Actor",
        }
        if class_filter:
            search_body["Filter"] = class_filter

        response = await self._http_put("/remote/search/assets", search_body)
        actors = response.get("Assets", [])

        # Build tree
        tree: Dict[str, Any] = {"name": "Root", "path": root_path or "/", "children": []}
        depth = 0
        for actor in actors[:max_depth * 50]:
            path = actor.get("Path", "")
            if root_path and not path.startswith(root_path):
                continue
            node = {
                "name": actor.get("Name", ""),
                "path": path,
                "class": actor.get("Class", ""),
                "children": [],
            }
            tree["children"].append(node)

        return {
            "root": tree,
            "total_actors": len(tree["children"]),
            "total_depth": min(depth + 1, max_depth),
        }

    async def analyze_scene_for_robotics(
        self,
        analysis_types: Optional[List[str]] = None,
        actor_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze the scene for robotics use-cases.

        Args:
            analysis_types: Analyses to run (traversability, graspability,
                collision_complexity).
            actor_filter: Filter to specific actor subtree.

        Returns:
            Dict with traversable_surfaces, graspable_objects,
            collision_summary, total_actors_analyzed.
        """
        if analysis_types is None:
            analysis_types = ["traversability", "graspability", "collision_complexity"]

        body: Dict[str, Any] = {
            "objectPath": "/Script/UnrealEd.Default__EditorLevelLibrary",
            "functionName": "GetAllLevelActors",
            "parameters": {},
        }
        response = await self._http_put("/remote/object/call", body)
        all_actors = response.get("ReturnValue", [])

        traversable: List[Dict[str, Any]] = []
        graspable: List[Dict[str, Any]] = []
        collision_info: Dict[str, Any] = {
            "total_actors": len(all_actors),
            "with_collision": 0,
            "complex_collision": 0,
        }

        for actor_path in all_actors:
            if actor_filter and not str(actor_path).startswith(actor_filter):
                continue
            try:
                desc_body: Dict[str, Any] = {"objectPath": str(actor_path)}
                desc = await self._http_put("/remote/object/describe", desc_body)
                class_name = desc.get("Class", "")

                if "StaticMesh" in class_name or "Floor" in str(actor_path):
                    traversable.append({
                        "path": str(actor_path),
                        "class": class_name,
                    })
                if "Physics" in str(desc.get("Properties", {})):
                    collision_info["with_collision"] += 1
                    graspable.append({
                        "path": str(actor_path),
                        "class": class_name,
                    })
            except Exception:
                continue

        return {
            "traversable_surfaces": traversable,
            "graspable_objects": graspable,
            "collision_summary": collision_info,
            "total_actors_analyzed": len(all_actors),
        }

    async def generate_procedural_scene(
        self,
        scene_type: str,
        parameters: Optional[Dict[str, Any]] = None,
        bounds_min: Optional[List[float]] = None,
        bounds_max: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a procedural scene via PCG or scripted spawning.

        Args:
            scene_type: Scene type (warehouse, outdoor, room, corridor).
            parameters: Generation params (size, density, seed, etc.).
            bounds_min: Min bounds [x, y, z] in cm.
            bounds_max: Max bounds [x, y, z] in cm.

        Returns:
            Dict with actors_spawned, total_spawned, scene_type, seed.
        """
        if parameters is None:
            parameters = {}
        if bounds_min is None:
            bounds_min = [0.0, 0.0, 0.0]
        if bounds_max is None:
            bounds_max = [1000.0, 1000.0, 500.0]

        seed = parameters.get("seed", 42)
        body: Dict[str, Any] = {
            "objectPath": "/Script/UnrealEd.Default__EditorLevelLibrary",
            "functionName": "GenerateProceduralScene",
            "parameters": {
                "SceneType": scene_type,
                "Parameters": parameters,
                "BoundsMin": {
                    "X": bounds_min[0],
                    "Y": bounds_min[1],
                    "Z": bounds_min[2],
                },
                "BoundsMax": {
                    "X": bounds_max[0],
                    "Y": bounds_max[1],
                    "Z": bounds_max[2],
                },
                "Seed": seed,
            },
        }
        response = await self._http_put("/remote/object/call", body)
        return {
            "actors_spawned": response.get("ActorsSpawned", []),
            "total_spawned": response.get("TotalSpawned", 0),
            "scene_type": scene_type,
            "seed": seed,
        }

    async def get_actor_by_semantic_label(
        self,
        label: str,
        match_mode: str = "exact",
        max_results: int = 100,
    ) -> Dict[str, Any]:
        """
        Find actors by semantic tag or label.

        Args:
            label: Semantic label to search for.
            match_mode: Match mode (exact, contains, regex).
            max_results: Maximum results.

        Returns:
            Dict with actors, total_matches, label_searched.
        """
        body: Dict[str, Any] = {
            "objectPath": "/Script/UnrealEd.Default__EditorLevelLibrary",
            "functionName": "GetAllLevelActors",
            "parameters": {},
        }
        response = await self._http_put("/remote/object/call", body)
        all_actors = response.get("ReturnValue", [])

        matches: List[Dict[str, Any]] = []
        for actor_path in all_actors:
            if len(matches) >= max_results:
                break
            try:
                tag_body: Dict[str, Any] = {
                    "objectPath": str(actor_path),
                    "propertyName": "Tags",
                }
                tag_resp = await self._http_put(
                    "/remote/object/property", tag_body
                )
                tags = tag_resp.get("Tags", [])
                tag_strings = [str(t) for t in tags]

                matched = False
                if match_mode == "exact":
                    matched = label in tag_strings
                elif match_mode == "contains":
                    matched = any(label in t for t in tag_strings)

                if matched:
                    matches.append({
                        "path": str(actor_path),
                        "tags": tag_strings,
                        "label": label,
                    })
            except Exception:
                continue

        return {
            "actors": matches,
            "total_matches": len(matches),
            "label_searched": label,
        }

    # ------------------------------------------------------------------
    # Phase 8: Geometry & Modeling (GeometryScript)
    # ------------------------------------------------------------------

    async def generate_mesh_primitive(
        self,
        primitive_type: str,
        dimensions: Optional[Dict[str, float]] = None,
        segments: int = 32,
        location: Optional[List[float]] = None,
        actor_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a parametric mesh primitive via GeometryScript.

        Args:
            primitive_type: box, sphere, cylinder, cone, torus, capsule.
            dimensions: Type-specific dimensions.
            segments: Tessellation segments.
            location: Spawn location [x, y, z] in cm.
            actor_label: Optional label for the actor.

        Returns:
            Dict with actor_path, primitive_type, triangle/vertex counts.
        """
        if dimensions is None:
            dimensions = {}
        if location is None:
            location = [0.0, 0.0, 0.0]
        label = actor_label or f"DynMesh_{primitive_type}"
        seg = max(segments, 4)
        d = dimensions

        # Build the append call depending on primitive_type
        ptype = primitive_type.lower()
        if ptype == "box":
            w = d.get("width", 100.0)
            h = d.get("height", 100.0)
            dp = d.get("depth", 100.0)
            append_line = (
                f"gs_prim.append_box(mesh, opts, xf, {w}, {h}, {dp},"
                f" 1, 1, 1, origin)"
            )
        elif ptype == "sphere":
            r = d.get("radius", 50.0)
            append_line = (
                f"gs_prim.append_sphere_lat_long(mesh, opts, xf,"
                f" {r}, {seg}, {seg}, origin)"
            )
        elif ptype == "cylinder":
            r = d.get("radius", 50.0)
            h = d.get("height", 100.0)
            append_line = (
                f"gs_prim.append_cylinder(mesh, opts, xf,"
                f" {r}, {h}, {seg}, 1, True, origin)"
            )
        elif ptype == "cone":
            r = d.get("base_radius", 50.0)
            h = d.get("height", 100.0)
            append_line = (
                f"gs_prim.append_cone(mesh, opts, xf,"
                f" {r}, 0.0, {h}, {seg}, 1, True, origin)"
            )
        elif ptype == "capsule":
            r = d.get("radius", 50.0)
            h = d.get("hemisphere_height", 50.0)
            append_line = (
                f"gs_prim.append_capsule(mesh, opts, xf,"
                f" {r}, {h}, {h}, {seg}, {max(seg // 2, 4)}, origin)"
            )
        elif ptype == "torus":
            inner = d.get("inner_radius", 20.0)
            outer = d.get("outer_radius", 50.0)
            append_line = (
                f"gs_prim.append_torus(mesh, opts, xf,"
                f" {inner}, {outer}, {seg}, {seg}, origin)"
            )
        else:
            return {"error": f"Unsupported primitive type: {primitive_type}"}

        lx, ly, lz = location[0], location[1], location[2]
        code = (
            "import unreal, json\n"
            "gs_prim = unreal.GeometryScript_Primitives\n"
            "gs_q = unreal.GeometryScript_MeshQueries\n"
            "mesh = unreal.DynamicMesh()\n"
            "opts = unreal.GeometryScriptPrimitiveOptions()\n"
            "opts.polygroup_mode = "
            "unreal.GeometryScriptPrimitivePolygroupMode.PER_FACE\n"
            "xf = unreal.Transform()\n"
            "origin = unreal.GeometryScriptPrimitiveOriginMode.CENTER\n"
            f"{append_line}\n"
            "subsys = unreal.get_editor_subsystem("
            "unreal.EditorActorSubsystem)\n"
            f"actor = subsys.spawn_actor_from_class("
            f"unreal.DynamicMeshActor, "
            f"unreal.Vector({lx}, {ly}, {lz}))\n"
            f"actor.set_actor_label('{label}')\n"
            "comp = actor.dynamic_mesh_component\n"
            "comp.set_dynamic_mesh(mesh)\n"
            "tris = gs_q.get_num_triangle_i_ds(mesh)\n"
            "verts = gs_q.get_vertex_count(mesh)\n"
            "print(json.dumps({"
            "'actor_path': actor.get_path_name(),"
            f"'primitive_type': '{ptype}',"
            "'triangle_count': tris,"
            "'vertex_count': verts"
            "}))\n"
        )
        result = await self._execute_python(code)
        return self._parse_python_json(result)

    async def apply_mesh_boolean(
        self,
        target_mesh_path: str,
        tool_mesh_path: str,
        operation: str,
    ) -> Dict[str, Any]:
        """
        Apply boolean operation between two meshes via GeometryScript.

        Args:
            target_mesh_path: Target mesh actor (modified in-place).
            tool_mesh_path: Tool mesh actor (operand).
            operation: union, subtract, or intersect.

        Returns:
            Dict with target_mesh_path, operation, result triangle/vertex counts.
        """
        op_map = {
            "union": "UNION",
            "subtract": "SUBTRACT",
            "intersect": "INTERSECTION",
        }
        ue_op = op_map.get(operation.lower(), "UNION")
        code = (
            "import unreal, json\n"
            "gs_bool = unreal.GeometryScript_MeshBooleans\n"
            "gs_q = unreal.GeometryScript_MeshQueries\n"
            "subsys = unreal.get_editor_subsystem("
            "unreal.EditorActorSubsystem)\n"
            "target_actor = None\n"
            "tool_actor = None\n"
            "for a in subsys.get_all_level_actors():\n"
            "    p = a.get_path_name()\n"
            f"    if p == '{target_mesh_path}':\n"
            "        target_actor = a\n"
            f"    elif p == '{tool_mesh_path}':\n"
            "        tool_actor = a\n"
            "if target_actor is None or tool_actor is None:\n"
            "    print(json.dumps({'error': 'Actor(s) not found'}))\n"
            "else:\n"
            "    t_mesh = target_actor.dynamic_mesh_component"
            ".get_dynamic_mesh()\n"
            "    tool_mesh = tool_actor.dynamic_mesh_component"
            ".get_dynamic_mesh()\n"
            "    bool_opts = unreal.GeometryScriptMeshBooleanOptions()\n"
            "    gs_bool.apply_mesh_boolean(\n"
            "        t_mesh, unreal.Transform(),\n"
            "        tool_mesh, unreal.Transform(),\n"
            f"        unreal.GeometryScriptBooleanOperation.{ue_op},"
            " bool_opts)\n"
            "    target_actor.dynamic_mesh_component"
            ".notify_mesh_modified()\n"
            "    tris = gs_q.get_num_triangle_i_ds(t_mesh)\n"
            "    verts = gs_q.get_vertex_count(t_mesh)\n"
            "    print(json.dumps({"
            f"'target_mesh_path': '{target_mesh_path}',"
            f"'operation': '{operation}',"
            "'result_triangle_count': tris,"
            "'result_vertex_count': verts"
            "}))\n"
        )
        result = await self._execute_python(code)
        return self._parse_python_json(result)

    async def compute_convex_hull(
        self,
        mesh_path: str,
    ) -> Dict[str, Any]:
        """
        Compute convex hull envelope of a mesh.

        Args:
            mesh_path: Source mesh actor path.

        Returns:
            Dict with hull actor path, vertex/triangle counts.
        """
        code = (
            "import unreal, json\n"
            "gs_c = unreal.GeometryScript_Containment\n"
            "gs_q = unreal.GeometryScript_MeshQueries\n"
            "subsys = unreal.get_editor_subsystem("
            "unreal.EditorActorSubsystem)\n"
            "src = None\n"
            "for a in subsys.get_all_level_actors():\n"
            f"    if a.get_path_name() == '{mesh_path}':\n"
            "        src = a\n"
            "        break\n"
            "if src is None:\n"
            "    print(json.dumps({'error': "
            f"'Actor not found: {mesh_path}'""}))\n"
            "else:\n"
            "    mesh = src.dynamic_mesh_component.get_dynamic_mesh()\n"
            "    hull_mesh = unreal.DynamicMesh()\n"
            "    sel = unreal.GeometryScriptMeshSelection()\n"
            "    opts = unreal.GeometryScriptConvexHullOptions()\n"
            "    gs_c.compute_mesh_convex_hull("
            "mesh, hull_mesh, sel, opts)\n"
            "    hull_actor = subsys.spawn_actor_from_class("
            "unreal.DynamicMeshActor, src.get_actor_location())\n"
            "    hull_actor.set_actor_label("
            "src.get_actor_label() + '_Hull')\n"
            "    hull_actor.dynamic_mesh_component"
            ".set_dynamic_mesh(hull_mesh)\n"
            "    h_tris = gs_q.get_num_triangle_i_ds(hull_mesh)\n"
            "    h_verts = gs_q.get_vertex_count(hull_mesh)\n"
            "    s_tris = gs_q.get_num_triangle_i_ds(mesh)\n"
            "    print(json.dumps({"
            f"'mesh_path': '{mesh_path}',"
            "'hull_actor_path': hull_actor.get_path_name(),"
            "'hull_vertex_count': h_verts,"
            "'hull_triangle_count': h_tris,"
            "'source_triangle_count': s_tris"
            "}))\n"
        )
        result = await self._execute_python(code)
        return self._parse_python_json(result)

    async def decompose_convex_hull(
        self,
        mesh_path: str,
        max_hulls: int = 16,
        max_vertices_per_hull: int = 32,
        min_cluster_size: int = 256,
        resolution: int = 100000,
    ) -> Dict[str, Any]:
        """
        V-HACD convex decomposition for collision geometry.

        Args:
            mesh_path: Source mesh actor path.
            max_hulls: Maximum convex pieces.
            max_vertices_per_hull: Max vertices per hull.
            min_cluster_size: Minimum cluster size for V-HACD.
            resolution: Voxelization resolution.

        Returns:
            Dict with hull_count, triangle/vertex counts for combined hull.
        """
        # GeometryScriptConvexDecompositionOptions has:
        # num_hulls, error_tolerance, search_factor,
        # simplify_to_face_count, min_part_thickness
        code = (
            "import unreal, json\n"
            "gs_c = unreal.GeometryScript_Containment\n"
            "gs_q = unreal.GeometryScript_MeshQueries\n"
            "subsys = unreal.get_editor_subsystem("
            "unreal.EditorActorSubsystem)\n"
            "src = None\n"
            "for a in subsys.get_all_level_actors():\n"
            f"    if a.get_path_name() == '{mesh_path}':\n"
            "        src = a\n"
            "        break\n"
            "if src is None:\n"
            "    print(json.dumps({'error': "
            f"'Actor not found: {mesh_path}'""}))\n"
            "else:\n"
            "    mesh = src.dynamic_mesh_component.get_dynamic_mesh()\n"
            "    decomp_mesh = unreal.DynamicMesh()\n"
            "    opts = unreal.GeometryScriptConvexDecompositionOptions()\n"
            f"    opts.num_hulls = {max_hulls}\n"
            f"    opts.simplify_to_face_count = {max_vertices_per_hull}\n"
            "    gs_c.compute_mesh_convex_decomposition("
            "mesh, decomp_mesh, opts)\n"
            "    d_actor = subsys.spawn_actor_from_class("
            "unreal.DynamicMeshActor, src.get_actor_location())\n"
            "    d_actor.set_actor_label("
            "src.get_actor_label() + '_Decomp')\n"
            "    d_actor.dynamic_mesh_component"
            ".set_dynamic_mesh(decomp_mesh)\n"
            "    d_tris = gs_q.get_num_triangle_i_ds(decomp_mesh)\n"
            "    d_verts = gs_q.get_vertex_count(decomp_mesh)\n"
            "    n_comp = gs_q.get_num_connected_components("
            "decomp_mesh)\n"
            "    print(json.dumps({"
            f"'mesh_path': '{mesh_path}',"
            "'hull_count': n_comp,"
            "'decomp_actor_path': d_actor.get_path_name(),"
            "'total_triangles': d_tris,"
            "'total_vertices': d_verts"
            "}))\n"
        )
        result = await self._execute_python(code)
        return self._parse_python_json(result)

    async def edit_mesh_topology(
        self,
        mesh_path: str,
        operation: str,
        face_selection: Optional[str] = None,
        edge_selection: Optional[str] = None,
        distance: Optional[float] = None,
        offset: Optional[float] = None,
        scale: Optional[List[float]] = None,
        count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Edit mesh topology (extrude, bevel, inset, loop cut, scale_faces).

        Args:
            mesh_path: Mesh actor path.
            operation: extrude_faces, bevel_edges, inset_faces,
                loop_cut, scale_faces.
            face_selection: Face selection filter (unused in current impl).
            edge_selection: Edge selection filter (unused in current impl).
            distance: Extrude/bevel distance.
            offset: Inset offset.
            scale: Scale factors [x, y, z] (unused in current impl).
            count: Subdivision count for bevel.

        Returns:
            Dict with mesh_path, operation, result triangle/vertex counts.
        """
        dist = distance if distance is not None else 20.0
        ofs = offset if offset is not None else 5.0
        subdivs = count if count is not None else 1
        op = operation.lower()

        # Build the operation-specific Python block
        if op == "extrude_faces":
            op_block = (
                "    ext_opts = unreal"
                ".GeometryScriptMeshLinearExtrudeOptions()\n"
                f"    ext_opts.distance = {dist}\n"
                "    sel = unreal.GeometryScriptMeshSelection()\n"
                "    gs_mod.apply_mesh_linear_extrude_faces("
                "mesh, ext_opts, sel)\n"
            )
        elif op == "bevel_edges":
            op_block = (
                "    bev_opts = unreal"
                ".GeometryScriptMeshBevelSelectionOptions()\n"
                f"    bev_opts.bevel_distance = {dist}\n"
                f"    bev_opts.subdivisions = {subdivs}\n"
                "    sel = unreal.GeometryScriptMeshSelection()\n"
                "    gs_mod.apply_mesh_bevel_selection(mesh, sel,\n"
                "        unreal.GeometryScriptMeshBevelSelectionMode"
                ".TRIANGLE_AREA, bev_opts)\n"
            )
        elif op == "inset_faces":
            op_block = (
                "    ins_opts = unreal"
                ".GeometryScriptMeshInsetOutsetFacesOptions()\n"
                f"    ins_opts.distance = {ofs}\n"
                "    sel = unreal.GeometryScriptMeshSelection()\n"
                "    gs_mod.apply_mesh_inset_outset_faces("
                "mesh, ins_opts, sel)\n"
            )
        else:
            return {"error": f"Unsupported topology operation: {operation}"}

        code = (
            "import unreal, json\n"
            "gs_mod = unreal.GeometryScript_MeshModeling\n"
            "gs_q = unreal.GeometryScript_MeshQueries\n"
            "subsys = unreal.get_editor_subsystem("
            "unreal.EditorActorSubsystem)\n"
            "actor = None\n"
            "for a in subsys.get_all_level_actors():\n"
            f"    if a.get_path_name() == '{mesh_path}':\n"
            "        actor = a\n"
            "        break\n"
            "if actor is None:\n"
            "    print(json.dumps({'error': "
            f"'Actor not found: {mesh_path}'""}))\n"
            "else:\n"
            "    mesh = actor.dynamic_mesh_component"
            ".get_dynamic_mesh()\n"
            "    before_tris = gs_q.get_num_triangle_i_ds(mesh)\n"
            + op_block
            + "    actor.dynamic_mesh_component"
            ".notify_mesh_modified()\n"
            "    tris = gs_q.get_num_triangle_i_ds(mesh)\n"
            "    verts = gs_q.get_vertex_count(mesh)\n"
            "    print(json.dumps({"
            f"'mesh_path': '{mesh_path}',"
            f"'operation': '{operation}',"
            "'result_triangle_count': tris,"
            "'result_vertex_count': verts,"
            "'previous_triangle_count': before_tris"
            "}))\n"
        )
        result = await self._execute_python(code)
        return self._parse_python_json(result)

    async def subdivide_mesh(
        self,
        mesh_path: str,
        level: int = 2,
        scheme: str = "catmull_clark",
    ) -> Dict[str, Any]:
        """
        Catmull-Clark or uniform tessellation subdivision.

        Args:
            mesh_path: Mesh actor path.
            level: Subdivision level (1-4).
            scheme: catmull_clark, loop, or uniform.

        Returns:
            Dict with mesh_path, level, scheme, result triangle/vertex counts.
        """
        lvl = max(1, min(level, 4))
        sch = scheme.lower()

        if sch == "catmull_clark":
            subdiv_block = (
                "    gl = unreal.GeometryScriptGroupLayer()\n"
                "    unreal.GeometryScript_OpenSubdiv"
                f".apply_polygroup_catmull_clark_sub_d(mesh, {lvl}, gl)\n"
            )
        else:
            # uniform tessellation for loop / uniform / bilinear
            subdiv_block = (
                "    unreal.GeometryScript_MeshSubdivide"
                f".apply_uniform_tessellation(mesh, {lvl})\n"
            )

        code = (
            "import unreal, json\n"
            "gs_q = unreal.GeometryScript_MeshQueries\n"
            "subsys = unreal.get_editor_subsystem("
            "unreal.EditorActorSubsystem)\n"
            "actor = None\n"
            "for a in subsys.get_all_level_actors():\n"
            f"    if a.get_path_name() == '{mesh_path}':\n"
            "        actor = a\n"
            "        break\n"
            "if actor is None:\n"
            "    print(json.dumps({'error': "
            f"'Actor not found: {mesh_path}'""}))\n"
            "else:\n"
            "    mesh = actor.dynamic_mesh_component"
            ".get_dynamic_mesh()\n"
            "    before_tris = gs_q.get_num_triangle_i_ds(mesh)\n"
            + subdiv_block
            + "    actor.dynamic_mesh_component"
            ".notify_mesh_modified()\n"
            "    tris = gs_q.get_num_triangle_i_ds(mesh)\n"
            "    verts = gs_q.get_vertex_count(mesh)\n"
            "    print(json.dumps({"
            f"'mesh_path': '{mesh_path}',"
            f"'level': {lvl},"
            f"'scheme': '{scheme}',"
            "'result_triangle_count': tris,"
            "'result_vertex_count': verts,"
            "'previous_triangle_count': before_tris"
            "}))\n"
        )
        result = await self._execute_python(code)
        return self._parse_python_json(result)

    async def simplify_mesh(
        self,
        mesh_path: str,
        target_triangle_count: Optional[int] = None,
        target_percentage: Optional[float] = None,
        max_error: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Simplify/decimate a mesh.

        Args:
            mesh_path: Mesh actor path.
            target_triangle_count: Target triangle count.
            target_percentage: Target percentage (0.0-1.0).
            max_error: Max geometric error tolerance.

        Returns:
            Dict with original/result triangles, reduction ratio.
        """
        # Determine target count: explicit, or derived from percentage
        target_line = ""
        if target_triangle_count is not None:
            target_line = f"    target_count = {target_triangle_count}\n"
        elif target_percentage is not None:
            target_line = (
                f"    target_count = max(1, int(before_tris"
                f" * {target_percentage}))\n"
            )
        else:
            target_line = "    target_count = max(1, before_tris // 2)\n"

        code = (
            "import unreal, json\n"
            "gs_s = unreal.GeometryScript_MeshSimplification\n"
            "gs_q = unreal.GeometryScript_MeshQueries\n"
            "subsys = unreal.get_editor_subsystem("
            "unreal.EditorActorSubsystem)\n"
            "actor = None\n"
            "for a in subsys.get_all_level_actors():\n"
            f"    if a.get_path_name() == '{mesh_path}':\n"
            "        actor = a\n"
            "        break\n"
            "if actor is None:\n"
            "    print(json.dumps({'error': "
            f"'Actor not found: {mesh_path}'""}))\n"
            "else:\n"
            "    mesh = actor.dynamic_mesh_component"
            ".get_dynamic_mesh()\n"
            "    before_tris = gs_q.get_num_triangle_i_ds(mesh)\n"
            + target_line
            + "    opts = unreal.GeometryScriptSimplifyMeshOptions()\n"
            "    gs_s.apply_simplify_to_triangle_count("
            "mesh, target_count, opts)\n"
            "    actor.dynamic_mesh_component"
            ".notify_mesh_modified()\n"
            "    tris = gs_q.get_num_triangle_i_ds(mesh)\n"
            "    verts = gs_q.get_vertex_count(mesh)\n"
            "    ratio = round(1.0 - tris / before_tris, 4)"
            " if before_tris > 0 else 0.0\n"
            "    print(json.dumps({"
            f"'mesh_path': '{mesh_path}',"
            "'original_triangles': before_tris,"
            "'result_triangles': tris,"
            "'result_vertex_count': verts,"
            "'reduction_ratio': ratio"
            "}))\n"
        )
        result = await self._execute_python(code)
        return self._parse_python_json(result)

    async def cut_mesh_plane(
        self,
        mesh_path: str,
        plane_origin: List[float],
        plane_normal: List[float],
        fill_holes: bool = True,
        keep_both_sides: bool = False,
    ) -> Dict[str, Any]:
        """
        Cut/slice a mesh along an arbitrary plane.

        Args:
            mesh_path: Mesh actor path.
            plane_origin: Plane origin [x, y, z] in cm.
            plane_normal: Plane normal [x, y, z].
            fill_holes: Fill cut holes with faces.
            keep_both_sides: Keep both halves (unused — UE plane cut
                keeps one side and optionally fills).

        Returns:
            Dict with mesh_path, result triangle/vertex counts.
        """
        ox, oy, oz = plane_origin[0], plane_origin[1], plane_origin[2]
        nx, ny, nz = plane_normal[0], plane_normal[1], plane_normal[2]
        fill_py = "True" if fill_holes else "False"
        code = (
            "import unreal, json\n"
            "gs_bool = unreal.GeometryScript_MeshBooleans\n"
            "gs_q = unreal.GeometryScript_MeshQueries\n"
            "subsys = unreal.get_editor_subsystem("
            "unreal.EditorActorSubsystem)\n"
            "actor = None\n"
            "for a in subsys.get_all_level_actors():\n"
            f"    if a.get_path_name() == '{mesh_path}':\n"
            "        actor = a\n"
            "        break\n"
            "if actor is None:\n"
            "    print(json.dumps({'error': "
            f"'Actor not found: {mesh_path}'""}))\n"
            "else:\n"
            "    mesh = actor.dynamic_mesh_component"
            ".get_dynamic_mesh()\n"
            "    before_tris = gs_q.get_num_triangle_i_ds(mesh)\n"
            "    cut_frame = unreal.Transform()\n"
            f"    cut_frame.translation = "
            f"unreal.Vector({ox}, {oy}, {oz})\n"
            # Build rotation from normal — use MathLibrary.make_rot_from_z
            f"    normal = unreal.Vector({nx}, {ny}, {nz})\n"
            "    rot = unreal.MathLibrary.make_rot_from_z(normal)\n"
            "    cut_frame.rotation = rot.quaternion()\n"
            "    cut_opts = unreal"
            ".GeometryScriptMeshPlaneCutOptions()\n"
            f"    cut_opts.fill_holes = {fill_py}\n"
            "    gs_bool.apply_mesh_plane_cut("
            "mesh, cut_frame, cut_opts)\n"
            "    actor.dynamic_mesh_component"
            ".notify_mesh_modified()\n"
            "    tris = gs_q.get_num_triangle_i_ds(mesh)\n"
            "    verts = gs_q.get_vertex_count(mesh)\n"
            "    print(json.dumps({"
            f"'mesh_path': '{mesh_path}',"
            "'result_triangle_count': tris,"
            "'result_vertex_count': verts,"
            "'previous_triangle_count': before_tris"
            "}))\n"
        )
        result = await self._execute_python(code)
        return self._parse_python_json(result)

    async def validate_mesh(
        self,
        mesh_path: str,
        checks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Validate mesh geometry using GeometryScript query functions.

        Args:
            mesh_path: Mesh actor path.
            checks: Checks to run (currently all checks always run).

        Returns:
            Dict with is_valid, triangle/vertex counts, border/component
            info.
        """
        code = (
            "import unreal, json\n"
            "gs_q = unreal.GeometryScript_MeshQueries\n"
            "subsys = unreal.get_editor_subsystem("
            "unreal.EditorActorSubsystem)\n"
            "actor = None\n"
            "for a in subsys.get_all_level_actors():\n"
            f"    if a.get_path_name() == '{mesh_path}':\n"
            "        actor = a\n"
            "        break\n"
            "if actor is None:\n"
            "    print(json.dumps({'error': "
            f"'Actor not found: {mesh_path}'""}))\n"
            "else:\n"
            "    mesh = actor.dynamic_mesh_component"
            ".get_dynamic_mesh()\n"
            "    tris = gs_q.get_num_triangle_i_ds(mesh)\n"
            "    verts = gs_q.get_vertex_count(mesh)\n"
            "    open_edges = gs_q.get_num_open_border_edges(mesh)\n"
            "    open_loops = gs_q.get_num_open_border_loops(mesh)\n"
            "    components = gs_q.get_num_connected_components("
            "mesh)\n"
            "    has_normals = gs_q.get_has_triangle_normals(mesh)\n"
            "    has_gaps = gs_q.get_has_triangle_id_gaps(mesh)\n"
            "    is_watertight = (open_edges == 0)\n"
            "    is_valid = is_watertight and not has_gaps\n"
            "    issues = []\n"
            "    if open_edges > 0:\n"
            "        issues.append("
            "f'Non-watertight: {open_edges} open border edges')\n"
            "    if open_loops > 0:\n"
            "        issues.append("
            "f'{open_loops} open border loops')\n"
            "    if has_gaps:\n"
            "        issues.append('Triangle ID gaps detected')\n"
            "    if not has_normals:\n"
            "        issues.append('Missing triangle normals')\n"
            "    print(json.dumps({"
            f"'mesh_path': '{mesh_path}',"
            "'is_valid': is_valid,"
            "'triangle_count': tris,"
            "'vertex_count': verts,"
            "'open_border_edges': open_edges,"
            "'open_border_loops': open_loops,"
            "'connected_components': components,"
            "'has_normals': has_normals,"
            "'issues': issues"
            "}))\n"
        )
        result = await self._execute_python(code)
        return self._parse_python_json(result)

    async def convert_mesh_format(
        self,
        mesh_path: str,
        target_format: str,
        tessellation_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Convert between StaticMesh and DynamicMesh via GeometryScript.

        Args:
            mesh_path: Source mesh actor/asset path.
            target_format: 'static_mesh' or 'dynamic_mesh'.
            tessellation_options: Unused (reserved for future CAD support).

        Returns:
            Dict with source/result paths, formats, triangle count.
        """
        fmt = target_format.lower()
        if fmt == "static_mesh":
            # DynamicMeshActor → new StaticMesh asset
            code = (
                "import unreal, json\n"
                "gs_a = unreal.GeometryScript_AssetUtils\n"
                "gs_q = unreal.GeometryScript_MeshQueries\n"
                "subsys = unreal.get_editor_subsystem("
                "unreal.EditorActorSubsystem)\n"
                "actor = None\n"
                "for a in subsys.get_all_level_actors():\n"
                f"    if a.get_path_name() == '{mesh_path}':\n"
                "        actor = a\n"
                "        break\n"
                "if actor is None:\n"
                "    print(json.dumps({'error': "
                f"'Actor not found: {mesh_path}'"'}))\n'
                "else:\n"
                "    mesh = actor.dynamic_mesh_component"
                ".get_dynamic_mesh()\n"
                "    tris = gs_q.get_num_triangle_i_ds(mesh)\n"
                "    # Create a transient StaticMesh asset\n"
                "    pkg_name = '/Game/Generated/' + "
                "actor.get_actor_label()\n"
                "    sm = unreal.EditorAssetLibrary"
                ".does_asset_exist(pkg_name)\n"
                "    # For now report conversion info\n"
                "    print(json.dumps({"
                f"'source_path': '{mesh_path}',"
                "'target_format': 'static_mesh',"
                "'triangle_count': tris,"
                "'note': 'copy_mesh_to_static_mesh requires "
                "an existing StaticMesh asset target'"
                "}))\n"
            )
        elif fmt == "dynamic_mesh":
            # StaticMesh asset → DynamicMeshActor in the level
            code = (
                "import unreal, json\n"
                "gs_a = unreal.GeometryScript_AssetUtils\n"
                "gs_q = unreal.GeometryScript_MeshQueries\n"
                "subsys = unreal.get_editor_subsystem("
                "unreal.EditorActorSubsystem)\n"
                f"sm_asset = unreal.EditorAssetLibrary"
                f".load_asset('{mesh_path}')\n"
                "if sm_asset is None:\n"
                "    print(json.dumps({'error': "
                f"'Asset not found: {mesh_path}'"'}))\n'
                "else:\n"
                "    dyn_mesh = unreal.DynamicMesh()\n"
                "    asset_opts = unreal"
                ".GeometryScriptCopyMeshFromAssetOptions()\n"
                "    gs_a.copy_mesh_from_static_mesh("
                "sm_asset, dyn_mesh, asset_opts)\n"
                "    actor = subsys.spawn_actor_from_class("
                "unreal.DynamicMeshActor, unreal.Vector())\n"
                "    actor.set_actor_label("
                "sm_asset.get_name() + '_Dynamic')\n"
                "    actor.dynamic_mesh_component"
                ".set_dynamic_mesh(dyn_mesh)\n"
                "    tris = gs_q.get_num_triangle_i_ds(dyn_mesh)\n"
                "    verts = gs_q.get_vertex_count(dyn_mesh)\n"
                "    print(json.dumps({"
                f"'source_path': '{mesh_path}',"
                "'result_path': actor.get_path_name(),"
                "'target_format': 'dynamic_mesh',"
                "'triangle_count': tris,"
                "'vertex_count': verts"
                "}))\n"
            )
        else:
            return {"error": f"Unsupported target format: {target_format}"}

        result = await self._execute_python(code)
        return self._parse_python_json(result)

    async def remesh_mesh(
        self,
        mesh_path: str,
        mode: str = "uniform",
        target_edge_length: Optional[float] = None,
        target_triangle_count: Optional[int] = None,
        smoothing_iterations: int = 3,
    ) -> Dict[str, Any]:
        """
        Remesh for clean topology using GeometryScript_Remeshing.

        Args:
            mesh_path: Mesh actor path.
            mode: 'uniform' (only mode currently supported).
            target_edge_length: Target edge length in cm.
            target_triangle_count: Target triangle count.
            smoothing_iterations: Smoothing passes (remesh_iterations).

        Returns:
            Dict with original/result triangles, vertex count.
        """
        edge_len = target_edge_length if target_edge_length else 5.0
        tri_count = target_triangle_count if target_triangle_count else 0
        iters = max(1, smoothing_iterations)

        # Build target_type — use GeometryScriptUniformRemeshTargetType enum
        if target_triangle_count is not None:
            target_block = (
                f"    uni_opts.target_triangle_count = {tri_count}\n"
                "    uni_opts.target_type = unreal"
                ".GeometryScriptUniformRemeshTargetType"
                ".TRIANGLE_COUNT\n"
            )
        else:
            target_block = (
                f"    uni_opts.target_edge_length = {edge_len}\n"
                "    uni_opts.target_type = unreal"
                ".GeometryScriptUniformRemeshTargetType"
                ".TARGET_EDGE_LENGTH\n"
            )

        code = (
            "import unreal, json\n"
            "gs_r = unreal.GeometryScript_Remeshing\n"
            "gs_q = unreal.GeometryScript_MeshQueries\n"
            "subsys = unreal.get_editor_subsystem("
            "unreal.EditorActorSubsystem)\n"
            "actor = None\n"
            "for a in subsys.get_all_level_actors():\n"
            f"    if a.get_path_name() == '{mesh_path}':\n"
            "        actor = a\n"
            "        break\n"
            "if actor is None:\n"
            "    print(json.dumps({'error': "
            f"'Actor not found: {mesh_path}'"'}))\n'
            "else:\n"
            "    mesh = actor.dynamic_mesh_component"
            ".get_dynamic_mesh()\n"
            "    before_tris = gs_q.get_num_triangle_i_ds(mesh)\n"
            "    remesh_opts = unreal"
            ".GeometryScriptRemeshOptions()\n"
            f"    remesh_opts.remesh_iterations = {iters}\n"
            "    uni_opts = unreal"
            ".GeometryScriptUniformRemeshOptions()\n"
            + target_block
            + "    gs_r.apply_uniform_remesh("
            "mesh, remesh_opts, uni_opts)\n"
            "    actor.dynamic_mesh_component"
            ".notify_mesh_modified()\n"
            "    tris = gs_q.get_num_triangle_i_ds(mesh)\n"
            "    verts = gs_q.get_vertex_count(mesh)\n"
            "    print(json.dumps({"
            f"'mesh_path': '{mesh_path}',"
            f"'mode': '{mode}',"
            "'original_triangles': before_tris,"
            "'result_triangles': tris,"
            "'result_vertex_count': verts"
            "}))\n"
        )
        result = await self._execute_python(code)
        return self._parse_python_json(result)

    async def compute_mesh_uv(
        self,
        mesh_path: str,
        method: str = "auto_uv",
        uv_channel: int = 0,
        island_padding: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Generate UV coordinates using GeometryScript_UVs.

        Args:
            mesh_path: Mesh actor path.
            method: 'xatlas' or 'patch_builder' (default xatlas).
            uv_channel: UV set index.
            island_padding: Unused (reserved for future option support).

        Returns:
            Dict with method, uv_channel, triangle/vertex counts.
        """
        m = method.lower()
        if m in ("patch_builder", "patchbuilder"):
            uv_block = (
                "    uv_opts = unreal"
                ".GeometryScriptPatchBuilderOptions()\n"
                f"    gs_uv.auto_generate_patch_builder_mesh_u_vs("
                f"mesh, {uv_channel}, uv_opts)\n"
            )
            method_label = "patch_builder"
        else:
            # Default to XAtlas for auto_uv, xatlas, atlas_pack, etc.
            uv_block = (
                "    uv_opts = unreal"
                ".GeometryScriptXAtlasOptions()\n"
                f"    gs_uv.auto_generate_x_atlas_mesh_u_vs("
                f"mesh, {uv_channel}, uv_opts)\n"
            )
            method_label = "xatlas"

        code = (
            "import unreal, json\n"
            "gs_uv = unreal.GeometryScript_UVs\n"
            "gs_q = unreal.GeometryScript_MeshQueries\n"
            "subsys = unreal.get_editor_subsystem("
            "unreal.EditorActorSubsystem)\n"
            "actor = None\n"
            "for a in subsys.get_all_level_actors():\n"
            f"    if a.get_path_name() == '{mesh_path}':\n"
            "        actor = a\n"
            "        break\n"
            "if actor is None:\n"
            "    print(json.dumps({'error': "
            f"'Actor not found: {mesh_path}'"'}))\n'
            "else:\n"
            "    mesh = actor.dynamic_mesh_component"
            ".get_dynamic_mesh()\n"
            + uv_block
            + "    actor.dynamic_mesh_component"
            ".notify_mesh_modified()\n"
            "    tris = gs_q.get_num_triangle_i_ds(mesh)\n"
            "    verts = gs_q.get_vertex_count(mesh)\n"
            "    print(json.dumps({"
            f"'mesh_path': '{mesh_path}',"
            f"'method': '{method_label}',"
            f"'uv_channel': {uv_channel},"
            "'triangle_count': tris,"
            "'vertex_count': verts"
            "}))\n"
        )
        result = await self._execute_python(code)
        return self._parse_python_json(result)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _describe_actor_brief(self, actor_path: str) -> Dict[str, Any]:
        """
        Minimal actor description for listing purposes.

        Args:
            actor_path: Full object path.

        Returns:
            Dictionary with name, path, class_name, transform, tags.
        """
        body: Dict[str, Any] = {"objectPath": actor_path}
        desc = await self._http_put("/remote/object/describe", body)

        transform = await self._get_actor_transform(actor_path)
        name_part = actor_path.rsplit(".", 1)[-1] if "." in actor_path else actor_path

        return {
            "name": desc.get("Name", name_part),
            "path": actor_path,
            "class_name": desc.get("Class", "Unknown"),
            "location": transform["location"],
            "rotation": transform["rotation"],
            "scale": transform["scale"],
            "tags": desc.get("Tags", []),
        }

    async def _get_actor_transform(self, actor_path: str) -> Dict[str, Any]:
        """
        Read actor transform (location, rotation, scale) via property access.

        Args:
            actor_path: Full object path.

        Returns:
            Dictionary with location, rotation, scale tuples.
        """
        result: Dict[str, Any] = {
            "location": (0.0, 0.0, 0.0),
            "rotation": (0.0, 0.0, 0.0),
            "scale": (1.0, 1.0, 1.0),
        }

        for prop_name, key, default in (
            ("RootComponent.RelativeLocation", "location", (0.0, 0.0, 0.0)),
            ("RootComponent.RelativeRotation", "rotation", (0.0, 0.0, 0.0)),
            ("RootComponent.RelativeScale3D", "scale", (1.0, 1.0, 1.0)),
        ):
            try:
                body: Dict[str, Any] = {
                    "objectPath": actor_path,
                    "propertyName": prop_name,
                    "access": "READ_ACCESS",
                }
                prop_data = await self._http_put("/remote/object/property", body)
                val = prop_data.get(prop_name, {})
                if isinstance(val, dict):
                    result[key] = (
                        val.get("X", default[0]),
                        val.get("Y", default[1]),
                        val.get("Z", default[2]),
                    )
                    if key == "rotation":
                        result[key] = (
                            val.get("Pitch", default[0]),
                            val.get("Yaw", default[1]),
                            val.get("Roll", default[2]),
                        )
            except Exception as e:
                self.logger.warning(
                    "Failed to fetch %s property: %s", key, e
                )
                result[key] = default

        return result

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying HTTP session if open."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
        self.logger.debug("Unreal runtime session closed")

    def cleanup(self) -> None:
        """
        Synchronous cleanup hook for context-manager teardown.

        Schedules the async ``close()`` on the running event loop when
        possible, otherwise creates a new loop to run it.
        """
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.close())
        except RuntimeError:
            asyncio.run(self.close())


class UnrealRuntimeAdapter(LoggerMixin):
    """Adapter for Unreal Engine runtime operations."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """
        Initialize Unreal runtime adapter.

        Args:
            settings: Configuration settings.
        """
        self.settings: Settings = settings or get_settings()
        self.logger.info("Unreal runtime adapter initialized")

    @contextmanager
    def create_session(self) -> Any:
        """
        Create an Unreal runtime session context manager.

        Yields:
            UnrealRuntimeSession instance.
        """
        session = UnrealRuntimeSession(self.settings)
        try:
            yield session
        finally:
            session.cleanup()

    def is_available(self) -> bool:
        """
        Check whether the Unreal runtime is reachable.

        Returns:
            True when aiohttp or the embedded unreal module is available
            and the adapter is enabled in settings.
        """
        return UNREAL_AVAILABLE and self.settings.unreal.enabled

    def get_capabilities(self) -> List[str]:
        """
        Get list of Unreal runtime capabilities.

        Returns:
            Capability list for the Unreal runtime adapter.
        """
        if not self.is_available():
            return []

        return [
            "unreal_health_check",
            "get_unreal_engine_info",
            "get_unreal_loaded_map",
            "list_unreal_actors",
            "get_unreal_actor_info",
            "search_unreal_assets",
            "describe_unreal_object",
            "get_unreal_actor_thumbnail",
            "summarize_unreal_scene",
            "capture_unreal_viewport",
            "get_unreal_viewport_info",
            "set_unreal_camera_view",
            "focus_unreal_on_actor",
            "spawn_unreal_actor",
            "delete_unreal_actor",
            "set_unreal_actor_transform",
            "set_unreal_actor_property",
            "call_unreal_actor_function",
            "set_unreal_actor_parent",
            "add_unreal_component",
            "set_unreal_actor_visibility",
            "get_unreal_material_info",
            "set_unreal_material_params",
            "create_unreal_material_instance",
            "assign_unreal_material",
            "set_unreal_light_params",
            "set_unreal_render_settings",
            "control_unreal_simulation",
            "get_unreal_simulation_status",
            "enable_unreal_physics",
            "set_unreal_collision",
            "apply_unreal_force",
            "set_unreal_physics_params",
            "import_unreal_usd",
            "export_unreal_usd",
            "get_unreal_interchange_info",
            "batch_unreal_operations",
            "query_unreal_scene_graph",
            "analyze_unreal_scene_for_robotics",
            "generate_unreal_procedural_scene",
            "get_unreal_actor_by_semantic_label",
            "generate_unreal_mesh_primitive",
            "apply_unreal_mesh_boolean",
            "compute_unreal_convex_hull",
            "decompose_unreal_convex_hull",
            "edit_unreal_mesh_topology",
            "subdivide_unreal_mesh",
            "simplify_unreal_mesh",
            "cut_unreal_mesh_plane",
            "validate_unreal_mesh",
            "convert_unreal_mesh_format",
            "remesh_unreal_mesh",
            "compute_unreal_mesh_uv",
        ]


def create_unreal_session(
    settings: Optional[Settings] = None,
) -> UnrealRuntimeSession:
    """
    Create an Unreal runtime session.

    Args:
        settings: Configuration settings.

    Returns:
        UnrealRuntimeSession instance.
    """
    return UnrealRuntimeSession(settings)


def is_unreal_available() -> bool:
    """Check whether the Unreal runtime is available."""
    return UNREAL_AVAILABLE