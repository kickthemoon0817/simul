"""
Isaac Sim TCP socket tool registration for Simul MCP Server.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from fastmcp.tools.tool import ToolResult


if TYPE_CHECKING:
    from ..server import SimulMCPServer


def register_isaac_tools(server: "SimulMCPServer") -> None:
    """
    Register Isaac Sim tools.

    Two lightweight tools that send Python code over TCP to a running
    Isaac Sim instance via the stock isaacsim.code_editor.vscode extension.
    """

    @server._script_tool(
        name="execute_isaac_script",
        description=(
            "Execute arbitrary Python code inside a running Isaac Sim application. "
            "The code runs in Kit's Python scope with full access to omni.*, pxr.*, "
            "and isaacsim.* APIs. stdout is captured and returned. For structured "
            "results, print JSON via json.dumps(). Each execution is independent — "
            "always import modules at the top of every script. "
            "Call ping_isaac first to verify connectivity before sending scripts. "
            "Read the 'simul://isaac-sim/skills' resource for scripting patterns, "
            "API quick reference, and Isaac Sim 5.1 / 6.0 namespace migration notes."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=False, open_world=True, destructive=True
        ),
    )
    async def execute_isaac_script(code: str) -> ToolResult:
        """
        Execute Python code inside the running Isaac Sim process.

        The code is sent over the configured raw-script transport. When the
        typed bridge is enabled, raw script execution still uses the VS Code
        socket so the bridge can stay on the typed control path.

        Args:
            code: Python source code to execute in Isaac Sim.

        Returns:
            Dict with success, output, and optional error info.
        """
        return await server._exec_isaac(
            "execute_isaac_script",
            server._isaac_tools.execute_script(code),
            params={"code_bytes": len(code)},
            script_sha256=hashlib.sha256(code.encode("utf-8")).hexdigest(),
        )

    @server.mcp.tool(
        name="ping_isaac",
        description=(
            "Pre-flight check: verify that a running Isaac Sim instance is "
            "reachable on the configured TCP socket. Call this before "
            "execute_isaac_script to confirm connectivity and get the target address."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
    )
    async def ping_isaac() -> ToolResult:
        """
        Ping Isaac Sim to verify connectivity.

        Returns:
            Dict with reachable status, address, and timeout. ``success``
            tracks ``reachable``: an unreachable instance is a failed ping.
        """
        rate_error = server._check_rate_limit("ping_isaac")
        if rate_error:
            return server._as_text_result(rate_error)
        client = server._get_request_isaac_client()
        reachable = await client.ping()
        return server._as_text_result(
            {
                "success": reachable,
                "reachable": reachable,
                "address": client.address,
                "bridge_address": client.bridge_address,
                "bridge_circuit_open": client.bridge_circuit_open,
                "vscode_address": client.vscode_address,
                "timeout_seconds": client.timeout_seconds,
            }
        )

    @server.mcp.tool(
        name="interrupt_isaac_script",
        description=(
            "Stop the script the Isaac Sim bridge is currently running, so a "
            "runaway execute_isaac_script does not hold the instance for every "
            "agent. Reaches a coroutine script or one suspended at an await; a "
            "synchronous script that never yields also blocks the bridge's "
            "event loop, so that one is stopped by the per-request timeout sent "
            "with every script instead. Requires the bridge extension; the "
            "stock Python socket has no interrupt path. Check "
            "get_isaac_runtime_info's bridge section (busy, busy_since, "
            "current_action) first to tell a busy instance from a hung one."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=True, open_world=True
        ),
    )
    async def interrupt_isaac_script() -> ToolResult:
        """
        Interrupt the script running on the active Isaac Sim instance.

        Bypasses the per-instance lock on purpose: the call being interrupted
        is usually the one holding it.

        Returns:
            Dict with interrupted, was_busy, phase and current_action.
        """
        rate_error = server._check_rate_limit("interrupt_isaac_script")
        if rate_error:
            return server._as_text_result(rate_error)
        return server._as_text_result(await server._isaac_tools.interrupt_script())

    # -- Scene Inspection tools -------------------------------------------

    @server.mcp.tool(
        name="get_isaac_stage_info",
        description=(
            "Get current stage metadata including root layer, up-axis, "
            "meters-per-unit, total prim count, and default prim."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
    )
    async def get_isaac_stage_info() -> ToolResult:
        return await server._exec_isaac(
            "get_isaac_stage_info",
            server._isaac_tools.get_isaac_stage_info(),
        )

    @server.mcp.tool(
        name="list_isaac_prims",
        description=(
            "List prims in the current Isaac Sim stage under a given root path. "
            "Supports depth control and prim type filtering."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
    )
    async def list_isaac_prims(
        root_path: str = "/",
        prim_type: Optional[str] = None,
        max_depth: int = -1,
        max_items: int = 100,
    ) -> ToolResult:
        return await server._exec_isaac(
            "list_isaac_prims",
            server._isaac_tools.list_isaac_prims(
                root_path=root_path,
                prim_type=prim_type,
                max_depth=max_depth,
                max_items=max_items,
            ),
        )

    @server.mcp.tool(
        name="get_isaac_prim_detail",
        description=(
            "Read one or more aspects of a prim in a single call. Aspects: "
            "info, transform, ancestors, relationships, variants, bounding_box, "
            "mesh, light, material, rigid_body, collision, joint, mass, "
            "animation. Defaults to info."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
        output_schema=None,
    )
    async def get_isaac_prim_detail(
        prim_path: str,
        aspects: Optional[List[str]] = None,
    ) -> ToolResult:
        return await server._exec_isaac(
            "get_isaac_prim_detail",
            server._isaac_tools.get_isaac_prim_detail(
                prim_path=prim_path, aspects=aspects
            ),
        )

    @server.mcp.tool(
        name="search_isaac_prims",
        description=(
            "Search prims by type name or name pattern. "
            "search_type can be 'type' or 'name'."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
    )
    async def search_isaac_prims(
        search_type: str = "type",
        query: str = "Mesh",
        root_path: str = "/",
        max_results: int = 100,
    ) -> ToolResult:
        return await server._exec_isaac(
            "search_isaac_prims",
            server._isaac_tools.search_isaac_prims(
                search_type=search_type,
                query=query,
                root_path=root_path,
                max_results=max_results,
            ),
        )

    @server.mcp.tool(
        name="get_isaac_scene_summary",
        description=(
            "Get a high-level summary of the current scene: prim type counts, "
            "total prims, physics objects, cameras, lights, and materials."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
    )
    async def get_isaac_scene_summary() -> ToolResult:
        return await server._exec_isaac(
            "get_isaac_scene_summary",
            server._isaac_tools.get_isaac_scene_summary(),
        )

    # -- Viewport & Camera tools ------------------------------------------

    @server.mcp.tool(
        name="get_isaac_camera_info",
        description=(
            "Get camera properties (focal length, clipping range, resolution) "
            "for the active or specified camera."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
    )
    async def get_isaac_camera_info(
        camera_path: Optional[str] = None,
    ) -> ToolResult:
        return await server._exec_isaac(
            "get_isaac_camera_info",
            server._isaac_tools.get_isaac_camera_info(camera_path=camera_path),
        )

    @server.mcp.tool(
        name="list_isaac_cameras",
        description="List all camera prims in the current Isaac Sim stage.",
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
    )
    async def list_isaac_cameras() -> ToolResult:
        return await server._exec_isaac(
            "list_isaac_cameras",
            server._isaac_tools.list_isaac_cameras(),
        )

    @server.mcp.tool(
        name="set_isaac_camera",
        description=(
            "Set the active viewport camera position, target (look-at), "
            "or switch to a specific camera prim."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=True, open_world=True
        ),
    )
    async def set_isaac_camera(
        position: Optional[List[float]] = None,
        target: Optional[List[float]] = None,
        camera_path: Optional[str] = None,
    ) -> ToolResult:
        return await server._exec_isaac(
            "set_isaac_camera",
            server._isaac_tools.set_isaac_camera(
                position=position,
                target=target,
                camera_path=camera_path,
            ),
        )

    @server.mcp.tool(
        name="capture_isaac_viewport",
        description=(
            "Capture the current viewport to a PNG on the Isaac Sim host and "
            "return its path. Writes the PNG under the capture directory "
            "(viewport.capture_dir, default <allowed root>/captures) and reclaims "
            "the oldest captures there; the directory must be inside the "
            "configured sandbox (security.allowed_paths). Pass inline=true to "
            "also receive the image itself as an image content block; captures "
            "above the inline size cap return the path alone."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=True, open_world=True, destructive=True
        ),
    )
    async def capture_isaac_viewport(
        width: int = 1280,
        height: int = 720,
        inline: bool = False,
    ) -> ToolResult:
        return await server._exec_isaac(
            "capture_isaac_viewport",
            server._isaac_tools.capture_isaac_viewport(
                width=width, height=height, inline=inline
            ),
        )

    # -- Prim Manipulation tools ------------------------------------------

    @server.mcp.tool(
        name="create_isaac_prim",
        description=(
            "Create a new USD prim at the given path with optional type "
            "(Xform, Mesh, Sphere, Cube, Cylinder, Cone, Capsule, Plane)."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=False, open_world=True
        ),
    )
    async def create_isaac_prim(
        prim_path: str,
        prim_type: str = "Xform",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        return await server._exec_isaac(
            "create_isaac_prim",
            server._isaac_tools.create_isaac_prim(
                prim_path=prim_path,
                prim_type=prim_type,
                attributes=attributes,
            ),
        )

    @server.mcp.tool(
        name="delete_isaac_prim",
        description=(
            "Delete a prim and all its children from the stage. Refuses the "
            "pseudo-root '/'. Refuses '/World' unless allow_root_delete=true, "
            "since that removes the whole scene."
        ),
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
            destructive=True,
        ),
    )
    async def delete_isaac_prim(
        prim_path: str, allow_root_delete: bool = False
    ) -> ToolResult:
        return await server._exec_isaac(
            "delete_isaac_prim",
            server._isaac_tools.delete_isaac_prim(
                prim_path=prim_path, allow_root_delete=allow_root_delete
            ),
        )

    @server.mcp.tool(
        name="set_isaac_prim_transform",
        description=(
            "Set a prim's local transform: translate, rotate (euler XYZ "
            "degrees), and/or scale."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=True, open_world=True, destructive=True
        ),
    )
    async def set_isaac_prim_transform(
        prim_path: str,
        translation: Optional[List[float]] = None,
        rotation_euler: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
    ) -> ToolResult:
        return await server._exec_isaac(
            "set_isaac_prim_transform",
            server._isaac_tools.set_isaac_prim_transform(
                prim_path=prim_path,
                translation=translation,
                rotation_euler=rotation_euler,
                scale=scale,
            ),
        )

    @server.mcp.tool(
        name="set_isaac_prim_visibility",
        description="Show or hide a prim in the viewport.",
        annotations=server._tool_annotations(
            read_only=False, idempotent=True, open_world=True, destructive=True
        ),
    )
    async def set_isaac_prim_visibility(prim_path: str, visible: bool) -> ToolResult:
        return await server._exec_isaac(
            "set_isaac_prim_visibility",
            server._isaac_tools.set_isaac_prim_visibility(
                prim_path=prim_path, visible=visible
            ),
        )

    @server.mcp.tool(
        name="set_isaac_prim_attribute",
        description=(
            "Set an arbitrary attribute value on a prim, overwriting its "
            "current value. The attribute must already exist."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=True, open_world=True, destructive=True
        ),
    )
    async def set_isaac_prim_attribute(
        prim_path: str,
        attribute_name: str,
        value: Any,
    ) -> ToolResult:
        return await server._exec_isaac(
            "set_isaac_prim_attribute",
            server._isaac_tools.set_isaac_prim_attribute(
                prim_path=prim_path,
                attribute_name=attribute_name,
                value=value,
            ),
        )

    @server.mcp.tool(
        name="duplicate_isaac_prim",
        description="Duplicate a prim (deep copy) to a new path.",
        annotations=server._tool_annotations(
            read_only=False, idempotent=False, open_world=True
        ),
    )
    async def duplicate_isaac_prim(prim_path: str, new_path: str) -> ToolResult:
        return await server._exec_isaac(
            "duplicate_isaac_prim",
            server._isaac_tools.duplicate_isaac_prim(
                prim_path=prim_path, new_path=new_path
            ),
        )

    @server.mcp.tool(
        name="reparent_isaac_prim",
        description="Move a prim under a new parent in the hierarchy.",
        annotations=server._tool_annotations(
            read_only=False, idempotent=True, open_world=True, destructive=True
        ),
    )
    async def reparent_isaac_prim(prim_path: str, new_parent_path: str) -> ToolResult:
        return await server._exec_isaac(
            "reparent_isaac_prim",
            server._isaac_tools.reparent_isaac_prim(
                prim_path=prim_path,
                new_parent_path=new_parent_path,
            ),
        )

    # -- Physics Inspection tools -----------------------------------------

    @server.mcp.tool(
        name="get_isaac_physics_scene",
        description=(
            "Get physics scene configuration: gravity, solver settings, "
            "and physics scene prim paths."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
    )
    async def get_isaac_physics_scene() -> ToolResult:
        return await server._exec_isaac(
            "get_isaac_physics_scene",
            server._isaac_tools.get_isaac_physics_scene(),
        )

    @server.mcp.tool(
        name="list_isaac_physics_objects",
        description=(
            "List all prims with physics APIs: rigid bodies, colliders, "
            "and joints under a root path."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
    )
    async def list_isaac_physics_objects(
        root_path: str = "/",
    ) -> ToolResult:
        return await server._exec_isaac(
            "list_isaac_physics_objects",
            server._isaac_tools.list_isaac_physics_objects(root_path=root_path),
        )

    # -- Physics Configuration tools --------------------------------------

    @server.mcp.tool(
        name="add_isaac_rigid_body",
        description=("Apply RigidBodyAPI to a prim, optionally making it kinematic."),
        annotations=server._tool_annotations(
            read_only=False, idempotent=False, open_world=True
        ),
    )
    async def add_isaac_rigid_body(
        prim_path: str, kinematic: bool = False
    ) -> ToolResult:
        return await server._exec_isaac(
            "add_isaac_rigid_body",
            server._isaac_tools.add_isaac_rigid_body(
                prim_path=prim_path, kinematic=kinematic
            ),
        )

    @server.mcp.tool(
        name="add_isaac_collision",
        description=(
            "Apply CollisionAPI to a prim with optional mesh approximation "
            "(none, convexHull, convexDecomposition, boundingSphere, boundingCube)."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=False, open_world=True
        ),
    )
    async def add_isaac_collision(
        prim_path: str, approximation: str = "none"
    ) -> ToolResult:
        return await server._exec_isaac(
            "add_isaac_collision",
            server._isaac_tools.add_isaac_collision(
                prim_path=prim_path, approximation=approximation
            ),
        )

    @server.mcp.tool(
        name="set_isaac_mass_properties",
        description=(
            "Set mass, density, and/or center of mass on a prim. "
            "Applies MassAPI if not already present."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=True, open_world=True, destructive=True
        ),
    )
    async def set_isaac_mass_properties(
        prim_path: str,
        mass: Optional[float] = None,
        density: Optional[float] = None,
        center_of_mass: Optional[List[float]] = None,
    ) -> ToolResult:
        return await server._exec_isaac(
            "set_isaac_mass_properties",
            server._isaac_tools.set_isaac_mass_properties(
                prim_path=prim_path,
                mass=mass,
                density=density,
                center_of_mass=center_of_mass,
            ),
        )

    @server.mcp.tool(
        name="set_isaac_physics_material",
        description=(
            "Create or update a physics material with friction and "
            "restitution coefficients."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=True, open_world=True, destructive=True
        ),
    )
    async def set_isaac_physics_material(
        prim_path: str,
        static_friction: float = 0.5,
        dynamic_friction: float = 0.5,
        restitution: float = 0.0,
    ) -> ToolResult:
        return await server._exec_isaac(
            "set_isaac_physics_material",
            server._isaac_tools.set_isaac_physics_material(
                prim_path=prim_path,
                static_friction=static_friction,
                dynamic_friction=dynamic_friction,
                restitution=restitution,
            ),
        )

    @server.mcp.tool(
        name="create_isaac_physics_scene",
        description=(
            "Create a UsdPhysics.Scene prim with configurable gravity. "
            "This is the prerequisite for any physics simulation — rigid bodies, "
            "colliders, and joints require a physics scene to function."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=False, open_world=True
        ),
    )
    async def create_isaac_physics_scene(
        prim_path: str = "/World/PhysicsScene",
        gravity_direction: Optional[List[float]] = None,
        gravity_magnitude: float = 9.81,
    ) -> ToolResult:
        return await server._exec_isaac(
            "create_isaac_physics_scene",
            server._isaac_tools.create_isaac_physics_scene(
                prim_path=prim_path,
                gravity_direction=gravity_direction,
                gravity_magnitude=gravity_magnitude,
            ),
        )

    # -- Simulation Control tools -----------------------------------------

    @server.mcp.tool(
        name="get_isaac_simulation_state",
        description=(
            "Get the current simulation state: playing, paused, or stopped, "
            "along with current time and time-codes-per-second."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
    )
    async def get_isaac_simulation_state() -> ToolResult:
        return await server._exec_isaac(
            "get_isaac_simulation_state",
            server._isaac_tools.get_isaac_simulation_state(),
        )

    @server.mcp.tool(
        name="start_isaac_simulation",
        description="Start (play) the simulation timeline.",
        annotations=server._tool_annotations(
            read_only=False, idempotent=True, open_world=True
        ),
    )
    async def start_isaac_simulation() -> ToolResult:
        return await server._exec_isaac(
            "start_isaac_simulation",
            server._isaac_tools.start_isaac_simulation(),
        )

    @server.mcp.tool(
        name="stop_isaac_simulation",
        description="Stop the simulation and reset to initial state.",
        annotations=server._tool_annotations(
            read_only=False, idempotent=True, open_world=True
        ),
    )
    async def stop_isaac_simulation() -> ToolResult:
        return await server._exec_isaac(
            "stop_isaac_simulation",
            server._isaac_tools.stop_isaac_simulation(),
        )

    @server.mcp.tool(
        name="pause_isaac_simulation",
        description="Pause the currently running simulation.",
        annotations=server._tool_annotations(
            read_only=False, idempotent=True, open_world=True
        ),
    )
    async def pause_isaac_simulation() -> ToolResult:
        return await server._exec_isaac(
            "pause_isaac_simulation",
            server._isaac_tools.pause_isaac_simulation(),
        )

    @server.mcp.tool(
        name="step_isaac_simulation",
        description=("Step the simulation forward by N physics steps."),
        annotations=server._tool_annotations(
            read_only=False, idempotent=False, open_world=True
        ),
    )
    async def step_isaac_simulation(
        num_steps: int = 1,
    ) -> ToolResult:
        return await server._exec_isaac(
            "step_isaac_simulation",
            server._isaac_tools.step_isaac_simulation(num_steps=num_steps),
        )

    @server.mcp.tool(
        name="reset_isaac_simulation",
        description="Reset the simulation to initial state and time 0.",
        annotations=server._tool_annotations(
            read_only=False, idempotent=True, open_world=True
        ),
    )
    async def reset_isaac_simulation() -> ToolResult:
        return await server._exec_isaac(
            "reset_isaac_simulation",
            server._isaac_tools.reset_isaac_simulation(),
        )

    @server.mcp.tool(
        name="get_isaac_simulation_time",
        description=(
            "Get current simulation time, start/end times, "
            "and time-codes-per-second."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
    )
    async def get_isaac_simulation_time() -> ToolResult:
        return await server._exec_isaac(
            "get_isaac_simulation_time",
            server._isaac_tools.get_isaac_simulation_time(),
        )

    # -- Materials & Appearance tools -------------------------------------

    @server.mcp.tool(
        name="list_isaac_materials",
        description="List all material prims in the current stage.",
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
    )
    async def list_isaac_materials() -> ToolResult:
        return await server._exec_isaac(
            "list_isaac_materials",
            server._isaac_tools.list_isaac_materials(),
        )

    @server.mcp.tool(
        name="assign_isaac_material",
        description=(
            "Bind a material to a prim so the prim renders "
            "with that material's appearance."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=True, open_world=True, destructive=True
        ),
    )
    async def assign_isaac_material(prim_path: str, material_path: str) -> ToolResult:
        return await server._exec_isaac(
            "assign_isaac_material",
            server._isaac_tools.assign_isaac_material(
                prim_path=prim_path,
                material_path=material_path,
            ),
        )

    @server.mcp.tool(
        name="set_isaac_material_property",
        description=(
            "Set an input property on a material's surface shader "
            "(e.g. diffuse_color_constant, metallic_constant, "
            "roughness_constant)."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=True, open_world=True, destructive=True
        ),
    )
    async def set_isaac_material_property(
        material_path: str,
        property_name: str,
        value: Any,
    ) -> ToolResult:
        return await server._exec_isaac(
            "set_isaac_material_property",
            server._isaac_tools.set_isaac_material_property(
                material_path=material_path,
                property_name=property_name,
                value=value,
            ),
        )

    @server.mcp.tool(
        name="create_isaac_material",
        description=(
            "Create a new material with a surface shader. Supports "
            "UsdPreviewSurface (portable) and OmniPBR (NVIDIA MDL). "
            "Set diffuse color, roughness, metallic, and opacity. "
            "After creation, use assign_isaac_material to bind it to prims."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=False, open_world=True
        ),
    )
    async def create_isaac_material(
        material_path: str,
        shader_type: str = "UsdPreviewSurface",
        diffuse_color: Optional[List[float]] = None,
        roughness: float = 0.5,
        metallic: float = 0.0,
        opacity: float = 1.0,
    ) -> ToolResult:
        return await server._exec_isaac(
            "create_isaac_material",
            server._isaac_tools.create_isaac_material(
                material_path=material_path,
                shader_type=shader_type,
                diffuse_color=diffuse_color,
                roughness=roughness,
                metallic=metallic,
                opacity=opacity,
            ),
        )

    # -- Asset & Stage Operations tools -----------------------------------

    @server.mcp.tool(
        name="open_isaac_stage",
        description=(
            "Open a USD stage file in Isaac Sim. Paths must be inside the "
            "configured sandbox (security.allowed_paths); omniverse:// URLs are "
            "allowed when their scheme is in security.allowed_url_schemes. "
            "Refused while the current stage has unsaved edits unless "
            "discard_unsaved=true."
        ),
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
            destructive=True,
        ),
    )
    async def open_isaac_stage(
        file_path: str, discard_unsaved: bool = False
    ) -> ToolResult:
        denial = server._sandbox_denial(file_path)
        if denial is not None:
            return server._as_text_result(denial)
        return await server._exec_isaac(
            "open_isaac_stage",
            server._isaac_tools.open_isaac_stage(
                file_path=file_path, discard_unsaved=discard_unsaved
            ),
        )

    @server.mcp.tool(
        name="save_isaac_stage",
        description=(
            "Save the current stage, overwriting the file on disk. Optionally "
            "provide a file path to save-as to a new location; an existing file "
            "there is refused unless overwrite=true. Paths must be "
            "inside the configured sandbox (security.allowed_paths); writes to "
            "URLs are refused unless the scheme is in "
            "security.allowed_write_url_schemes."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=True, open_world=True, destructive=True
        ),
    )
    async def save_isaac_stage(
        file_path: Optional[str] = None,
        overwrite: bool = False,
    ) -> ToolResult:
        denial = server._sandbox_denial(file_path, write=True)
        if denial is not None:
            return server._as_text_result(denial)
        return await server._exec_isaac(
            "save_isaac_stage",
            server._isaac_tools.save_isaac_stage(
                file_path=file_path, overwrite=overwrite
            ),
        )

    @server.mcp.tool(
        name="new_isaac_stage",
        description=(
            "Create a new empty stage in Isaac Sim. Refused while the current "
            "stage has unsaved edits unless discard_unsaved=true."
        ),
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
            destructive=True,
        ),
    )
    async def new_isaac_stage(discard_unsaved: bool = False) -> ToolResult:
        return await server._exec_isaac(
            "new_isaac_stage",
            server._isaac_tools.new_isaac_stage(discard_unsaved=discard_unsaved),
        )

    @server.mcp.tool(
        name="import_isaac_asset",
        description=(
            "Import an external asset (USD, USDZ, OBJ, FBX) into "
            "the current stage at a target path. Paths must be inside the "
            "configured sandbox (security.allowed_paths); omniverse:// URLs are "
            "allowed when their scheme is in security.allowed_url_schemes."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=False, open_world=True
        ),
    )
    async def import_isaac_asset(
        asset_path: str,
        target_path: str = "/World/ImportedAsset",
    ) -> ToolResult:
        denial = server._sandbox_denial(asset_path)
        if denial is not None:
            return server._as_text_result(denial)
        return await server._exec_isaac(
            "import_isaac_asset",
            server._isaac_tools.import_isaac_asset(
                asset_path=asset_path, target_path=target_path
            ),
        )

    @server.mcp.tool(
        name="add_isaac_reference",
        description=(
            "Add a USD reference to a prim so it composes in "
            "content from another USD file. Paths must be inside the "
            "configured sandbox (security.allowed_paths); omniverse:// URLs are "
            "allowed when their scheme is in security.allowed_url_schemes."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=False, open_world=True
        ),
    )
    async def add_isaac_reference(prim_path: str, reference_path: str) -> ToolResult:
        denial = server._sandbox_denial(reference_path)
        if denial is not None:
            return server._as_text_result(denial)
        return await server._exec_isaac(
            "add_isaac_reference",
            server._isaac_tools.add_isaac_reference(
                prim_path=prim_path,
                reference_path=reference_path,
            ),
        )

    # -- Scene Inspection & Exploration tools -----------------------------

    @server.mcp.tool(
        name="list_isaac_lights",
        description=(
            "List all light prims in the scene with type, "
            "intensity, and color information."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
    )
    async def list_isaac_lights(root_path: str = "/") -> ToolResult:
        return await server._exec_isaac(
            "list_isaac_lights",
            server._isaac_tools.list_isaac_lights(root_path=root_path),
        )

    @server.mcp.tool(
        name="create_isaac_light",
        description=(
            "Create a light prim in the scene. Supports DomeLight (environment/HDRI), "
            "DistantLight (sun), SphereLight (point), RectLight (area), "
            "DiskLight, and CylinderLight. Configure intensity, color, "
            "color temperature, and texture."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=False, open_world=True
        ),
    )
    async def create_isaac_light(
        prim_path: str,
        light_type: str = "DomeLight",
        intensity: float = 1000.0,
        color: Optional[List[float]] = None,
        color_temperature: Optional[float] = None,
        angle: Optional[float] = None,
        texture_file: Optional[str] = None,
    ) -> ToolResult:
        return await server._exec_isaac(
            "create_isaac_light",
            server._isaac_tools.create_isaac_light(
                prim_path=prim_path,
                light_type=light_type,
                intensity=intensity,
                color=color,
                color_temperature=color_temperature,
                angle=angle,
                texture_file=texture_file,
            ),
        )

    @server.mcp.tool(
        name="get_isaac_subtree",
        description=(
            "Get a full subtree as a flat list with depth info, "
            "type, and child counts for each prim."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
    )
    async def get_isaac_subtree(
        root_path: str = "/",
        max_depth: int = 5,
        max_prims: int = 150,
    ) -> ToolResult:
        return await server._exec_isaac(
            "get_isaac_subtree",
            server._isaac_tools.get_isaac_subtree(
                root_path=root_path,
                max_depth=max_depth,
                max_prims=max_prims,
            ),
        )

    @server.mcp.tool(
        name="get_isaac_layer_info",
        description=(
            "Get USD layer stack information: root layer, sublayers, "
            "session layer, and layer dirty states."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
    )
    async def get_isaac_layer_info() -> ToolResult:
        return await server._exec_isaac(
            "get_isaac_layer_info",
            server._isaac_tools.get_isaac_layer_info(),
        )

    @server.mcp.tool(
        name="get_isaac_scene_stats",
        description=(
            "Get aggregate scene statistics: total vertices, faces, "
            "meshes, lights, cameras, materials, and prim type counts."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
    )
    async def get_isaac_scene_stats(root_path: str = "/") -> ToolResult:
        return await server._exec_isaac(
            "get_isaac_scene_stats",
            server._isaac_tools.get_isaac_scene_stats(root_path=root_path),
        )

    @server.mcp.tool(
        name="get_isaac_texture_dependencies",
        description=(
            "List all external texture files referenced by scene materials "
            "with their referencing material paths."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
    )
    async def get_isaac_texture_dependencies(root_path: str = "/") -> ToolResult:
        return await server._exec_isaac(
            "get_isaac_texture_dependencies",
            server._isaac_tools.get_isaac_texture_dependencies(root_path=root_path),
        )

    @server.mcp.tool(
        name="focus_isaac_viewport",
        description=(
            "Frame the viewport camera to focus on a specific prim, "
            "zooming to fit it in view."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=True, open_world=True
        ),
    )
    async def focus_isaac_viewport(prim_path: str) -> ToolResult:
        return await server._exec_isaac(
            "focus_isaac_viewport",
            server._isaac_tools.focus_isaac_viewport(prim_path=prim_path),
        )

    @server.mcp.tool(
        name="get_isaac_selection",
        description=(
            "Get the currently selected prims in the Isaac Sim viewport "
            "with their paths, names, and types."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
    )
    async def get_isaac_selection() -> ToolResult:
        return await server._exec_isaac(
            "get_isaac_selection",
            server._isaac_tools.get_isaac_selection(),
        )

    @server.mcp.tool(
        name="get_isaac_ui_state",
        description=(
            "Get a consolidated snapshot of the Isaac Sim GUI and app "
            "state: omni.ui windows (visibility, focus, docking), active "
            "viewport, selection, timeline, and open-stage status. "
            "Degrades gracefully on headless runs."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
    )
    async def get_isaac_ui_state() -> ToolResult:
        return await server._exec_isaac(
            "get_isaac_ui_state",
            server._isaac_tools.get_isaac_ui_state(),
        )

    @server.mcp.tool(
        name="get_isaac_ui_window",
        description=(
            "Inspect one omni.ui window as a widget tree: widget types, "
            "identifiers, label text, state flags, and model values, "
            "depth- and count-limited. Window titles come from "
            "get_isaac_ui_state."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
    )
    async def get_isaac_ui_window(
        window_title: str,
        max_depth: int = 4,
        max_widgets: int = 400,
    ) -> ToolResult:
        return await server._exec_isaac(
            "get_isaac_ui_window",
            server._isaac_tools.get_isaac_ui_window(
                window_title=window_title,
                max_depth=max_depth,
                max_widgets=max_widgets,
            ),
        )

    @server.mcp.tool(
        name="raycast_isaac_scene",
        description=(
            "Cast a ray into the physics scene and return the closest "
            "hit prim, position, normal, and distance."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
    )
    async def raycast_isaac_scene(
        origin: List[float],
        direction: List[float],
        max_distance: float = 10000.0,
    ) -> ToolResult:
        return await server._exec_isaac(
            "raycast_isaac_scene",
            server._isaac_tools.raycast_isaac_scene(
                origin=origin,
                direction=direction,
                max_distance=max_distance,
            ),
        )

    @server.mcp.tool(
        name="find_isaac_prims_in_area",
        description=(
            "Find prims whose bounding box center is within a given "
            "radius of a point. Results sorted by distance."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
    )
    async def find_isaac_prims_in_area(
        center: List[float],
        radius: float,
        prim_type: Optional[str] = None,
        root_path: str = "/",
        max_results: int = 100,
    ) -> ToolResult:
        return await server._exec_isaac(
            "find_isaac_prims_in_area",
            server._isaac_tools.find_isaac_prims_in_area(
                center=center,
                radius=radius,
                prim_type=prim_type,
                root_path=root_path,
                max_results=max_results,
            ),
        )

    # ------------------------------------------------------------------
    # Extension management
    # ------------------------------------------------------------------

    @server.mcp.tool(
        name="list_isaac_extensions",
        description=(
            "List extensions registered in the running Isaac Sim instance. "
            "Returns each extension's ID (version-suffixed, e.g. "
            "'omni.physx-107.3.7'), version, and enabled status. Lists enabled "
            "extensions by default; pass enabled_only=false with "
            "search='<substring>' to scope a wider query."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
    )
    async def list_isaac_extensions(
        enabled_only: bool = True,
        search: Optional[str] = None,
    ) -> ToolResult:
        return await server._exec_isaac(
            "list_isaac_extensions",
            server._isaac_tools.list_isaac_extensions(
                enabled_only=enabled_only,
                search=search,
            ),
        )

    @server.mcp.tool(
        name="enable_isaac_extension",
        description=(
            "Enable an extension immediately in the running Isaac Sim instance. "
            "Accepts EITHER the bare canonical Kit extension name (e.g. "
            "'omni.physx', 'isaacsim.replicator.behavior') OR the "
            "version-suffixed ID returned by list_isaac_extensions (e.g. "
            "'omni.physx-107.3.7'). The bare "
            "name is preferred — it is what the underlying Kit "
            "set_extension_enabled_immediate API takes and keeps callers "
            "decoupled from the installed version."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=True, open_world=True
        ),
    )
    async def enable_isaac_extension(extension_id: str) -> ToolResult:
        return await server._exec_isaac(
            "enable_isaac_extension",
            server._isaac_tools.enable_isaac_extension(
                extension_id=extension_id,
            ),
        )

    @server.mcp.tool(
        name="disable_isaac_extension",
        description=(
            "Disable an extension immediately in the running Isaac Sim instance. "
            "Accepts EITHER the bare canonical Kit extension name (e.g. "
            "'omni.physx', 'isaacsim.replicator.behavior') OR the "
            "version-suffixed ID returned by list_isaac_extensions (e.g. "
            "'omni.physx-107.3.7'). The bare "
            "name is preferred — it keeps callers decoupled from the installed "
            "version. Refuses the transport extensions simul-mcp speaks through "
            "(khemoo.simul.mcp, isaacsim.code_editor.python_server, "
            "isaacsim.code_editor.vscode)."
        ),
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
            destructive=True,
        ),
    )
    async def disable_isaac_extension(extension_id: str) -> ToolResult:
        return await server._exec_isaac(
            "disable_isaac_extension",
            server._isaac_tools.disable_isaac_extension(
                extension_id=extension_id,
            ),
        )

    # ------------------------------------------------------------------
    # Carb settings
    # ------------------------------------------------------------------

    @server.mcp.tool(
        name="get_isaac_carb_settings",
        description=(
            "Read one or more Carbonite (carb) settings by key path. "
            "Settings control RTX renderer options, fog, exposure, tone mapping, "
            "and other runtime parameters. Key paths look like /rtx/fog/enabled, "
            "/rtx/post/tonemap/op, /rtx/raytracing/showLights. "
            "Returns the current value for each requested key."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
    )
    async def get_isaac_carb_settings(keys: List[str]) -> ToolResult:
        return await server._exec_isaac(
            "get_isaac_carb_settings",
            server._isaac_tools.get_carb_settings(keys=keys),
        )

    @server.mcp.tool(
        name="set_isaac_carb_settings",
        description=(
            "Write one or more Carbonite (carb) settings by key path. "
            'Takes a dict of key-value pairs (e.g. {"/rtx/fog/enabled": true, '
            '"/rtx/fog/fogEndDist": 200.0}). Returns the verified new values '
            "after applying. Changes take effect immediately in the renderer. "
            "Keys under /exts/khemoo.simul.mcp/ and "
            "/exts/isaacsim.code_editor.python_server/ configure the MCP "
            "transport and are refused."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=True, open_world=True, destructive=True
        ),
    )
    async def set_isaac_carb_settings(settings: Dict[str, Any]) -> ToolResult:
        return await server._exec_isaac(
            "set_isaac_carb_settings",
            server._isaac_tools.set_carb_settings(settings=settings),
        )

    # ------------------------------------------------------------------
    # AOV / Replicator
    # ------------------------------------------------------------------

    @server.mcp.tool(
        name="read_isaac_aovs",
        description=(
            "Read one or more AOV (render pass) buffers and return per-AOV "
            "statistics. Handles the full replicator pipeline in a single call: "
            "creates a render product, attaches annotators, renders frames, "
            "reads numpy data, computes stats (shape, min, max, mean, "
            "rgb_max, rgb_mean, nonzero_pixels for color AOVs), and cleans up. "
            "Common AOVs: HdrColor, DirectDiffuse, DirectSpecular, "
            "IndirectDiffuse, Reflections, AmbientOcclusion, Depth, "
            "SmoothNormal, normals. Use list_isaac_aovs for the registered set."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=False, open_world=True
        ),
    )
    async def read_isaac_aovs(
        aov_names: List[str],
        camera_path: str = "/OmniverseKit_Persp",
        resolution: Optional[List[int]] = None,
        num_frames: int = 5,
    ) -> ToolResult:
        return await server._exec_isaac(
            "read_isaac_aovs",
            server._isaac_tools.read_aovs(
                aov_names=aov_names,
                camera_path=camera_path,
                resolution=resolution,
                num_frames=num_frames,
            ),
        )

    @server.mcp.tool(
        name="list_isaac_aovs",
        description=(
            "List all available AOV annotator names registered in the current "
            "Isaac Sim session. Use this to discover which AOV names can be "
            "passed to read_isaac_aovs."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
    )
    async def list_isaac_aovs() -> ToolResult:
        return await server._exec_isaac(
            "list_isaac_aovs",
            server._isaac_tools.list_aovs(),
        )

    # ------------------------------------------------------------------
    # USD schema queries
    # ------------------------------------------------------------------

    @server.mcp.tool(
        name="query_isaac_typed_prims",
        description=(
            "Query prims by USD schema type and optionally read their "
            "attributes. Traverses the stage from root_path, finds all prims "
            "matching the schema type, and reads the requested attributes. "
            "Supported types: UsdLux.DistantLight, UsdLux.SphereLight, "
            "UsdLux.DomeLight, UsdGeom.Mesh, UsdGeom.PointInstancer, "
            "UsdGeom.Xform, UsdShade.Material, or any USD type name. "
            "Attributes are read via schema API first (e.g. 'intensity' -> "
            "GetIntensityAttr), then fallback to generic and inputs: prefix."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
    )
    async def query_isaac_typed_prims(
        type_name: str,
        attributes: Optional[List[str]] = None,
        root_path: str = "/",
        max_prims: int = 200,
    ) -> ToolResult:
        return await server._exec_isaac(
            "query_isaac_typed_prims",
            server._isaac_tools.query_usd_typed_prims(
                type_name=type_name,
                attributes=attributes,
                root_path=root_path,
                max_prims=max_prims,
            ),
        )

    # ------------------------------------------------------------------
    # Viewport / render info
    # ------------------------------------------------------------------

    @server.mcp.tool(
        name="get_isaac_viewport_info",
        description=(
            "Get detailed information about the active viewport including "
            "camera path, render product path, resolution, and viewport name."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
    )
    async def get_isaac_viewport_info() -> ToolResult:
        return await server._exec_isaac(
            "get_isaac_viewport_info",
            server._isaac_tools.get_viewport_info(),
        )

    @server.mcp.tool(
        name="list_isaac_render_vars",
        description=(
            "List available render variable (AOV) names from SyntheticData "
            "and sensor types. Use this to discover what render passes and "
            "sensor data are available in the current session."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
    )
    async def list_isaac_render_vars() -> ToolResult:
        return await server._exec_isaac(
            "list_isaac_render_vars",
            server._isaac_tools.list_render_vars(),
        )

    # ------------------------------------------------------------------
    # OmniGraph
    # ------------------------------------------------------------------

    @server.mcp.tool(
        name="list_isaac_graphs",
        description=(
            "List all OmniGraph graphs in the current Isaac Sim session. "
            "Returns each graph's path, evaluator type, pipeline stage, "
            "node count, and backing type. Use this to discover available "
            "graphs before querying or modifying their nodes."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
    )
    async def list_isaac_graphs() -> ToolResult:
        return await server._exec_isaac(
            "list_isaac_graphs",
            server._isaac_tools.list_isaac_graphs(),
        )

    @server.mcp.tool(
        name="get_isaac_graph_nodes",
        description=(
            "List nodes in an OmniGraph graph with their types, input/output "
            "attributes, and connections. Use this to inspect graph structure, "
            "find node types, and trace data flow between nodes."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
    )
    async def get_isaac_graph_nodes(
        graph_path: str,
        max_nodes: int = 200,
    ) -> ToolResult:
        return await server._exec_isaac(
            "get_isaac_graph_nodes",
            server._isaac_tools.get_isaac_graph_nodes(
                graph_path=graph_path,
                max_nodes=max_nodes,
            ),
        )

    @server.mcp.tool(
        name="create_isaac_graph_node",
        description=(
            "Create a node in an OmniGraph graph. Specify the graph path, "
            "desired node path, and node type ID. Use list_isaac_graph_node_types "
            "to discover available node types. Common types include "
            "omni.graph.nodes.OnPlaybackTick, omni.graph.nodes.ReadVariable, "
            "isaacsim.core.nodes.IsaacArticulationController."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=False, open_world=True
        ),
    )
    async def create_isaac_graph_node(
        graph_path: str,
        node_path: str,
        node_type: str,
    ) -> ToolResult:
        return await server._exec_isaac(
            "create_isaac_graph_node",
            server._isaac_tools.create_isaac_graph_node(
                graph_path=graph_path,
                node_path=node_path,
                node_type=node_type,
            ),
        )

    @server.mcp.tool(
        name="connect_isaac_graph_nodes",
        description=(
            "Connect two OmniGraph node attributes by their full paths. "
            "Source is typically an outputs: attribute, target is an inputs: "
            "attribute. Example: connect "
            "'/World/Graph/OnTick.outputs:tick' to "
            "'/World/Graph/Controller.inputs:execIn'."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=True, open_world=True
        ),
    )
    async def connect_isaac_graph_nodes(
        source_attr_path: str,
        target_attr_path: str,
    ) -> ToolResult:
        return await server._exec_isaac(
            "connect_isaac_graph_nodes",
            server._isaac_tools.connect_isaac_graph_nodes(
                source_attr_path=source_attr_path,
                target_attr_path=target_attr_path,
            ),
        )

    @server.mcp.tool(
        name="set_isaac_graph_node_values",
        description=(
            "Set attribute values on an OmniGraph node. Pass a dict of "
            "attribute names to values. Attribute names should include the "
            "namespace prefix (e.g. 'inputs:velocity', 'inputs:enabled'). "
            "Use get_isaac_graph_nodes to discover attribute names first."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=True, open_world=True, destructive=True
        ),
    )
    async def set_isaac_graph_node_values(
        node_path: str,
        values: Dict[str, Any],
    ) -> ToolResult:
        return await server._exec_isaac(
            "set_isaac_graph_node_values",
            server._isaac_tools.set_isaac_graph_node_values(
                node_path=node_path,
                values=values,
            ),
        )

    @server.mcp.tool(
        name="list_isaac_graph_node_types",
        description=(
            "List available OmniGraph node types that can be used with "
            "create_isaac_graph_node. Use search to filter by substring "
            "(e.g. 'Isaac', 'OnPlayback', 'Articulation'). Returns "
            "registered node type IDs."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
    )
    async def list_isaac_graph_node_types(
        search: Optional[str] = None,
        max_types: int = 200,
    ) -> ToolResult:
        return await server._exec_isaac(
            "list_isaac_graph_node_types",
            server._isaac_tools.list_isaac_graph_node_types(
                search=search,
                max_types=max_types,
            ),
        )

    @server.mcp.tool(
        name="delete_isaac_graph_node",
        description=(
            "Delete a node from an OmniGraph graph. Removes the node "
            "and all its connections."
        ),
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
            destructive=True,
        ),
    )
    async def delete_isaac_graph_node(
        graph_path: str,
        node_path: str,
    ) -> ToolResult:
        return await server._exec_isaac(
            "delete_isaac_graph_node",
            server._isaac_tools.delete_isaac_graph_node(
                graph_path=graph_path,
                node_path=node_path,
            ),
        )

    # ------------------------------------------------------------------
    # Runtime diagnostics
    # ------------------------------------------------------------------

    @server.mcp.tool(
        name="get_isaac_logs",
        description=(
            "Read recent log entries from the running Isaac Sim console. "
            "Returns entries filtered by minimum level (verbose/info/warn/error), "
            "optional source module filter, and optional text search. "
            "Includes per-level counts and log file path. Use this to diagnose "
            "errors, warnings, or check what happened during a simulation."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
    )
    async def get_isaac_logs(
        level: str = "warn",
        last_n: int = 50,
        source_filter: Optional[str] = None,
        search: Optional[str] = None,
    ) -> ToolResult:
        return await server._exec_isaac(
            "get_isaac_logs",
            server._isaac_tools.get_isaac_logs(
                level=level,
                last_n=last_n,
                source_filter=source_filter,
                search=search,
            ),
        )

    @server.mcp.tool(
        name="set_isaac_log_level",
        description=(
            "Set the Carbonite logging threshold for the running Isaac Sim. "
            "Levels: verbose, info, warn, error. Lower levels include all "
            "higher levels. Use 'verbose' for maximum detail when debugging."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=True, open_world=True, destructive=True
        ),
    )
    async def set_isaac_log_level(level: str) -> ToolResult:
        return await server._exec_isaac(
            "set_isaac_log_level",
            server._isaac_tools.set_isaac_log_level(level=level),
        )

    @server.mcp.tool(
        name="get_isaac_runtime_info",
        description=(
            "Get consolidated runtime diagnostics from the running Isaac Sim "
            "instance: Kit app version, timeline state, physics stats "
            "(rigid body/articulation/shape counts, CUDA availability), "
            "GPU/renderer info, viewport state, stage summary, and extension "
            "counts. Use this for a quick health check or to understand the "
            "current state of the simulation environment."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
    )
    async def get_isaac_runtime_info() -> ToolResult:
        return await server._exec_isaac(
            "get_isaac_runtime_info",
            server._isaac_tools.get_runtime_info(),
        )
