"""OmniGraph tools for Isaac Sim."""

import json
import textwrap
from typing import Any, Callable, Dict, List, Optional

from ....adapters import IsaacSocketClient, ScriptResult
from ...schemas.common import ErrorResponse
from ._shared import (
    BULK_GEOMETRY_ATTRIBUTES,
    DEFAULT_MAX_RESULTS,
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


class OmniGraphMixin:
    # ------------------------------------------------------------------
    # OmniGraph
    # ------------------------------------------------------------------

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
            Dict with created node path and type.
        """
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
                    (e.g. {{"inputs:velocity": 1.5, "inputs:enabled": true}}).

        Returns:
            Dict with applied values.
        """
        _node_path = _pyval(node_path)
        _values = _pyval(values)
        script = textwrap.dedent(f"""\
            import json
            import omni.graph.core as og

            node_path = {_node_path}
            values = {_values}
            applied = {{}}
            errors = {{}}

            try:
                for attr_name, val in values.items():
                    try:
                        og.Controller.set(
                            og.Controller.attribute(attr_name, node_path),
                            val
                        )
                        applied[attr_name] = val
                    except Exception as e:
                        errors[attr_name] = str(e)
            except Exception as e:
                print(json.dumps({{"error": str(e)}}))
            else:
                result = {{"applied": applied, "count": len(applied)}}
                if errors:
                    result["errors"] = errors
                print(json.dumps(result))
        """)
        return await self._execute_json_script(script)

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

