"""OmniGraph tools for Isaac Sim."""

import textwrap
from typing import Any, Dict, Optional

from ...schemas.common import ErrorResponse
from ._shared import (
    DEFAULT_MAX_RESULTS,
    _pyval,
)
from .._meta import tool_meta

# A node type or attribute whose name contains this runs Python supplied as
# data: omni.graph.scriptnode.ScriptNode executes its inputs:script (or the
# file named by inputs:scriptPath) every time the graph evaluates. With
# security.allow_script_execution off, such a node would be execute_isaac_script
# by another name, so the graph tools refuse to create one or write to one.
# The match is deliberately broad: refusing a harmless "script"-named node
# costs one call, admitting a script node costs the setting.
SCRIPT_CAPABLE_NODE_HINT = "script"

# Extensions whose only purpose is running Python attached to the stage.
# omni.graph.scriptnode is already enabled in Isaac Sim's own .kit, so the
# node guard above is the one that matters; this keeps the refusal coherent.
SCRIPT_EXTENSIONS: frozenset[str] = frozenset({"omni.graph.scriptnode", "omni.kit.scripting"})


def is_script_capable_node_type(node_type: str) -> bool:
    """Return True for an OmniGraph node type that can run Python given as data.

    Args:
        node_type: Node type ID, e.g. ``omni.graph.scriptnode.ScriptNode``.

    Returns:
        Whether creating or editing a node of this type amounts to running
        agent-authored code.
    """
    return SCRIPT_CAPABLE_NODE_HINT in node_type.lower()


def is_script_attribute(attribute_name: str) -> bool:
    """Return True for a node attribute that carries code or a path to code.

    Args:
        attribute_name: Attribute name, e.g. ``inputs:script``.

    Returns:
        Whether writing this attribute can change the Python a node runs.
    """
    return SCRIPT_CAPABLE_NODE_HINT in attribute_name.lower()


def _script_execution_refusal(message: str, **details: Any) -> Dict[str, Any]:
    """Return the envelope for an operation that would run agent-authored Python.

    Args:
        message: What was refused.
        **details: Structured context for the caller.

    Returns:
        An ErrorResponse dict with ``error_type`` ``ScriptExecutionDisabled``.
    """
    return ErrorResponse(
        error=(
            message + " Script execution is disabled by security.allow_script_execution; "
            "set SECURITY__ALLOW_SCRIPT_EXECUTION=true to allow it."
        ),
        error_type="ScriptExecutionDisabled",
        details=details or None,
    ).model_dump()


class OmniGraphMixin:
    # ------------------------------------------------------------------
    # OmniGraph
    # ------------------------------------------------------------------

    @tool_meta(
        name="list_isaac_graphs",
        description=(
            "List all OmniGraph graphs in the current Isaac Sim session. Returns each "
            "graph's path, evaluator type, pipeline stage, node count, and backing type. "
            "Use this to discover available graphs before querying or modifying their "
            "nodes."
        ),
        read_only=True,
        idempotent=True,
    )
    async def list_isaac_graphs(self) -> Dict[str, Any]:
        """
        List all OmniGraph graphs in the current Isaac Sim session.

        Returns:
            Dict with list of graphs, each containing path, evaluator,
            pipeline stage, node count, and backing type.
        """
        script = textwrap.dedent("""\
            import json
            import omni.graph.core as og

            graphs = og.get_all_graphs()
            result = []
            for g in graphs:
                try:
                    nodes = g.get_nodes()
                    result.append({
                        "path": g.get_path_to_graph(),
                        "evaluator": g.get_evaluator_name(),
                        "pipeline_stage": str(g.get_pipeline_stage()),
                        "node_count": len(nodes),
                        "backing_type": str(g.get_graph_backing_type()),
                    })
                except Exception:
                    pass
            print(json.dumps({"count": len(result), "graphs": result}))
        """)
        return await self._execute_json_script(script)

    @tool_meta(
        name="get_isaac_graph_nodes",
        description=(
            "List nodes in an OmniGraph graph with their types, input/output attributes, "
            "and connections. Use this to inspect graph structure, find node types, and "
            "trace data flow between nodes."
        ),
        read_only=True,
        idempotent=True,
    )
    async def get_isaac_graph_nodes(
        self,
        graph_path: str,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> Dict[str, Any]:
        """
        List nodes in an OmniGraph graph with their types and connections.

        Args:
            graph_path: USD path to the graph (e.g. "/World/ActionGraph").
            max_results: Maximum number of nodes to return. Clamped to [1, 1000].

        Returns:
            Dict with node list, each containing path, type, and
            input/output attribute names with connections.
        """
        max_results = max(1, min(max_results, 1000))
        _graph_path = _pyval(graph_path)
        script = textwrap.dedent(f"""\
            import json
            import omni.graph.core as og

            graph = og.get_graph_by_path({_graph_path})
            if not graph or not graph.is_valid():
                print(json.dumps({{"error": "Graph not found: " + {_graph_path}}}))
            else:
                nodes = graph.get_nodes()
                max_results = {max_results}
                truncated = len(nodes) > max_results
                result = []
                for node in nodes[:max_results]:
                    try:
                        attrs = node.get_attributes()
                        inputs = []
                        outputs = []
                        for attr in attrs:
                            name = attr.get_name()
                            connected = []
                            try:
                                for conn in attr.get_upstream_connections():
                                    connected.append(conn.get_path())
                            except Exception:
                                pass
                            entry = {{"name": name}}
                            if connected:
                                entry["connected_from"] = connected
                            if name.startswith("inputs:"):
                                inputs.append(entry)
                            elif name.startswith("outputs:"):
                                outputs.append(entry)
                        result.append({{
                            "path": node.get_prim_path(),
                            "type": node.get_type_name(),
                            "inputs": inputs,
                            "outputs": outputs,
                        }})
                    except Exception:
                        result.append({{
                            "path": node.get_prim_path(),
                            "type": node.get_type_name(),
                            "error": "failed to read attributes",
                        }})
                print(json.dumps({{
                    "graph_path": {_graph_path},
                    "count": len(result),
                    "total": len(nodes),
                    "truncated": truncated,
                    "nodes": result,
                }}))
        """)
        return await self._execute_json_script(script)

    @tool_meta(
        name="create_isaac_graph_node",
        description=(
            "Create a node in an OmniGraph graph. Specify the graph path, desired node "
            "path, and node type ID. Use list_isaac_graph_node_types to discover available "
            "node types. Common types include omni.graph.nodes.OnPlaybackTick, "
            "omni.graph.nodes.ReadVariable, "
            "isaacsim.core.nodes.IsaacArticulationController. With "
            "security.allow_script_execution off, node types that run Python given as data "
            "(any type with 'script' in its name, such as omni.graph.scriptnode.ScriptNode) "
            "are refused."
        ),
        read_only=False,
    )
    async def create_isaac_graph_node(
        self,
        graph_path: str,
        node_path: str,
        node_type: str,
    ) -> Dict[str, Any]:
        """
        Create a node in an OmniGraph graph.

        Args:
            graph_path: USD path to the target graph.
            node_path: Path for the new node within the graph.
            node_type: OmniGraph node type ID
                       (e.g. "omni.graph.nodes.OnPlaybackTick").

        Returns:
            Dict with created node path and type, or a
            ScriptExecutionDisabled error for a script-capable node type
            while script execution is off.
        """
        if not self.settings.security.allow_script_execution and is_script_capable_node_type(node_type):
            return _script_execution_refusal(
                f"Refusing to create {node_type!r}: a script node runs the Python in its inputs.",
                node_type=node_type,
            )
        _graph_path = _pyval(graph_path)
        _node_path = _pyval(node_path)
        _node_type = _pyval(node_type)
        script = textwrap.dedent(f"""\
            import json
            import omni.graph.core as og

            graph = og.get_graph_by_path({_graph_path})
            if not graph or not graph.is_valid():
                print(json.dumps({{"error": "Graph not found: " + {_graph_path}}}))
            else:
                try:
                    node = graph.create_node({_node_path}, {_node_type}, True)
                    if node and node.is_valid():
                        print(json.dumps({{
                            "created": True,
                            "path": node.get_prim_path(),
                            "type": node.get_type_name(),
                        }}))
                    else:
                        print(json.dumps({{"error": "Failed to create node"}}))
                except Exception as e:
                    print(json.dumps({{"error": str(e)}}))
        """)
        return await self._execute_json_script(script)

    @tool_meta(
        name="connect_isaac_graph_nodes",
        description=(
            "Connect two OmniGraph node attributes by their full paths. Source is typically "
            "an outputs: attribute, target is an inputs: attribute. Example: connect "
            "'/World/Graph/OnTick.outputs:tick' to '/World/Graph/Controller.inputs:execIn'."
        ),
        read_only=False,
        idempotent=True,
    )
    async def connect_isaac_graph_nodes(
        self,
        source_attr_path: str,
        target_attr_path: str,
    ) -> Dict[str, Any]:
        """
        Connect two OmniGraph node attributes.

        Args:
            source_attr_path: Full path to the source attribute
                              (e.g. "/World/Graph/OnTick.outputs:tick").
            target_attr_path: Full path to the target attribute
                              (e.g. "/World/Graph/MyNode.inputs:execIn").

        Returns:
            Dict with connection confirmation.
        """
        _source = _pyval(source_attr_path)
        _target = _pyval(target_attr_path)
        script = textwrap.dedent(f"""\
            import json
            import omni.graph.core as og

            try:
                og.Controller.connect({_source}, {_target})
                print(json.dumps({{
                    "connected": True,
                    "source": {_source},
                    "target": {_target},
                }}))
            except Exception as e:
                print(json.dumps({{"error": str(e)}}))
        """)
        return await self._execute_json_script(script)

    @tool_meta(
        name="set_isaac_graph_node_values",
        description=(
            "Set attribute values on an OmniGraph node. Pass a dict of attribute names to "
            "values. Attribute names should include the namespace prefix (e.g. "
            "'inputs:velocity', 'inputs:enabled'). Use get_isaac_graph_nodes to discover "
            "attribute names first. Every attribute is attempted; the call fails "
            "(success=false) when any of them failed, with the ones that were set under "
            "'applied' and the failures under 'errors'. With security.allow_script_execution "
            "off, script attributes and script-capable nodes are refused."
        ),
        read_only=False,
        destructive=True,
        idempotent=True,
    )
    async def set_isaac_graph_node_values(
        self,
        node_path: str,
        values: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Set attribute values on an OmniGraph node.

        Args:
            node_path: USD path to the node.
            values: Dict mapping attribute names to values
                    (e.g. {"inputs:velocity": 1.5, "inputs:enabled": true}).

        Returns:
            Dict with ``applied`` and ``count``. When any attribute failed,
            ``success`` is false, ``error`` names the failed attributes and
            ``errors`` holds each one's message; the attributes in
            ``applied`` were still set.
        """
        if not values:
            return ErrorResponse(error="values is empty; nothing to set", error_type="ValueError").model_dump()
        guard = not self.settings.security.allow_script_execution
        script_attributes = sorted(name for name in values if is_script_attribute(name))
        if guard and script_attributes:
            return _script_execution_refusal(
                "Refusing to write " + ", ".join(script_attributes)
                + ": a script attribute holds Python the node runs. Nothing was set.",
                attributes=script_attributes,
            )
        _node_path = _pyval(node_path)
        _values = _pyval(values)
        script = textwrap.dedent(f"""\
            import json
            import omni.graph.core as og

            node_path = {_node_path}
            values = {_values}
            guard = {_pyval(guard)}
            applied = {{}}
            errors = {{}}

            try:
                node = og.Controller.node(node_path)
            except Exception:
                node = None
            if node is None or not node.is_valid():
                print(json.dumps({{"error": "Node not found: " + node_path}}))
            elif guard and {_pyval(SCRIPT_CAPABLE_NODE_HINT)} in node.get_type_name().lower():
                print(json.dumps({{
                    "error": "Refusing to edit " + node_path + ", a " + node.get_type_name()
                             + " node, which runs Python given as data. Script execution is disabled "
                             "by security.allow_script_execution; nothing was set.",
                    "error_type": "ScriptExecutionDisabled",
                    "node_type": node.get_type_name(),
                }}))
            else:
                for attr_name, val in values.items():
                    try:
                        og.Controller.set(og.Controller.attribute(attr_name, node_path), val)
                        applied[attr_name] = val
                    except Exception as e:
                        errors[attr_name] = str(e)
                result = {{"node_path": node_path, "applied": applied, "count": len(applied)}}
                if errors:
                    # The envelope rule reads only error / *_error keys, so a
                    # dict of per-attribute failures alone would read as success.
                    result["success"] = False
                    result["error"] = (
                        "Failed to set " + str(len(errors)) + " of " + str(len(values))
                        + " attribute(s): " + ", ".join(sorted(errors))
                    )
                    result["errors"] = errors
                print(json.dumps(result))
        """)
        return await self._execute_json_script(script)

    @tool_meta(
        name="list_isaac_graph_node_types",
        description=(
            "List available OmniGraph node types that can be used with "
            "create_isaac_graph_node. Use search to filter by substring (e.g. 'Isaac', "
            "'OnPlayback', 'Articulation'). Returns registered node type IDs."
        ),
        read_only=True,
        idempotent=True,
    )
    async def list_isaac_graph_node_types(
        self,
        search: Optional[str] = None,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> Dict[str, Any]:
        """
        List available OmniGraph node types.

        Args:
            search: Optional substring filter on node type name.
            max_results: Maximum number of types to return. Clamped to [1, 2000].

        Returns:
            Dict with list of registered node type names.
        """
        max_results = max(1, min(max_results, 2000))
        _search = _pyval(search)
        script = textwrap.dedent(f"""\
            import json
            import omni.graph.core as og

            registered = og.get_registered_nodes()
            all_types = sorted(registered)
            search = {_search}
            if search:
                all_types = [t for t in all_types if search.lower() in t.lower()]
            max_results = {max_results}
            truncated = len(all_types) > max_results
            result = all_types[:max_results]
            print(json.dumps({{
                "count": len(result),
                "total": len(all_types),
                "truncated": truncated,
                "total_registered": len(registered),
                "node_types": result,
            }}))
        """)
        return await self._execute_json_script(script)

    @tool_meta(
        name="delete_isaac_graph_node",
        description=(
            "Delete a node from an OmniGraph graph. Removes the node and all its "
            "connections."
        ),
        read_only=False,
        destructive=True,
        idempotent=True,
    )
    async def delete_isaac_graph_node(
        self,
        graph_path: str,
        node_path: str,
    ) -> Dict[str, Any]:
        """
        Delete a node from an OmniGraph graph.

        Args:
            graph_path: USD path to the graph.
            node_path: Path of the node to delete.

        Returns:
            Dict with deletion confirmation.
        """
        _graph_path = _pyval(graph_path)
        _node_path = _pyval(node_path)
        script = textwrap.dedent(f"""\
            import json
            import omni.graph.core as og

            graph = og.get_graph_by_path({_graph_path})
            if not graph or not graph.is_valid():
                print(json.dumps({{"error": "Graph not found: " + {_graph_path}}}))
            else:
                try:
                    success = graph.destroy_node({_node_path}, True)
                    print(json.dumps({{"deleted": success, "node_path": {_node_path}}}))
                except Exception as e:
                    print(json.dumps({{"error": str(e)}}))
        """)
        return await self._execute_json_script(script)
