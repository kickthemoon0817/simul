"""OmniGraph tools for Isaac Sim."""

import inspect
import re
import textwrap
from typing import Any, Dict, Optional

from ...schemas.common import ErrorResponse
from ._shared import (
    DEFAULT_MAX_RESULTS,
    _compose_script,
    _pyval,
)
from .._meta import tool_meta

# ---------------------------------------------------------------------------
# Script-execution guard
#
# With security.allow_script_execution off, a node that runs Python supplied
# as data is execute_isaac_script by another name: ScriptNode runs its
# inputs:script (or the file in inputs:scriptPath) on every evaluation,
# omni.warp.WarpKernel writes inputs:codeStr to a .py file and imports it, and
# Replicator's augmentation and condition nodes exec their *Script inputs.
# omni.warp and omni.graph.scriptnode are enabled in Isaac Sim's own .kit, so
# the guard is on the nodes and their code attributes, not only on enabling
# extensions. This is the one definition of "script-capable": the graph,
# stage and prim tools all ask these predicates (server side) or embed
# script_guard_source() (Kit side), which carries the same functions.
#
# Names are matched by word, not by substring: a name splits at separators
# (. : _ -) and camelCase boundaries, and a word matches a hint when it starts
# with it. So codeStr (code|str), augmentationScript and scriptnode match,
# while usdTimecode (usd|timecode), robot_description (robot|description)
# and omniscripting do not. Over Isaac Sim's shipped node types this flags
# exactly the five code-running types in the deny-list and their code inputs
# (tests/isaac/data/shipped_node_types.json pins it).
# ---------------------------------------------------------------------------

# Kept for callers that import the single hint.
SCRIPT_CAPABLE_NODE_HINT = "script"

# Word prefixes that mark a node type or extension name as code-running.
SCRIPT_CAPABLE_NODE_HINTS: tuple[str, ...] = ("script",)

# Code-running node types, compared without case as the full type ID or its
# last dotted segment, whatever their names say.
SCRIPT_CAPABLE_NODE_TYPES: frozenset[str] = frozenset(
    {
        "omni.graph.scriptnode.ScriptNode",
        "omni.warp.WarpKernel",
        "omni.replicator.core.OgnAugment",
        "omni.replicator.core.OgnAugmentCPU",
        "omni.replicator.core.OgnConditionScript",
    }
)

# Word prefixes that mark an attribute as carrying code: inputs:script /
# scriptPath, *Script (Replicator's augmentation and condition scripts),
# WarpKernel's inputs:codeStr / codeFile / codeProvider.
CODE_ATTRIBUTE_HINTS: tuple[str, ...] = ("script", "code")

# Extensions that run Python attached to the stage, besides any whose name has
# a word starting with one of SCRIPT_CAPABLE_NODE_HINTS. omni.warp ships
# WarpKernel; its library, omni.warp.core, does not run stage code.
SCRIPT_EXTENSIONS: frozenset[str] = frozenset(
    {"omni.graph.scriptnode", "omni.kit.scripting", "omni.warp"}
)

# The names the shared functions read, as they are also named inside a
# generated script (see script_guard_source).
SCRIPT_GUARD_TYPES: tuple[str, ...] = tuple(sorted(name.lower() for name in SCRIPT_CAPABLE_NODE_TYPES))
SCRIPT_GUARD_SEGMENTS: tuple[str, ...] = tuple(sorted({name.rsplit(".", 1)[-1] for name in SCRIPT_GUARD_TYPES}))
SCRIPT_GUARD_HINTS = SCRIPT_CAPABLE_NODE_HINTS
_GUARD_WORD_PATTERN = r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+"
_GUARD_WORD = re.compile(_GUARD_WORD_PATTERN)


# The three functions below run both here and inside Kit: script_guard_source()
# embeds their source, so they may use only re, _GUARD_WORD and the
# SCRIPT_GUARD_* constants.
def _guard_words(text: Any) -> list[str]:
    """Split a name into lower-case words at separator and camelCase boundaries."""
    return [word.lower() for word in _GUARD_WORD.findall(str(text or ""))]


def _guard_has_word(text: Any, prefixes: Any) -> bool:
    """Return True when a word of ``text`` starts with one of ``prefixes``."""
    return any(word.startswith(prefix) for word in _guard_words(text) for prefix in prefixes)


def _script_capable_type(type_name: Any) -> bool:
    """Return True for a code-running node type (deny-list or a script word)."""
    lowered = str(type_name or "").strip().lower()
    return (
        lowered in SCRIPT_GUARD_TYPES
        or lowered.rsplit(".", 1)[-1] in SCRIPT_GUARD_SEGMENTS
        or _guard_has_word(type_name, SCRIPT_GUARD_HINTS)
    )


def is_script_capable_node_type(type_name: str) -> bool:
    """Return True for an OmniGraph node type that can run Python given as data.

    Args:
        type_name: Node type ID, e.g. ``omni.graph.scriptnode.ScriptNode`` or
            ``omni.warp.WarpKernel``.

    Returns:
        Whether creating or editing a node of this type amounts to running
        agent-authored code: a type in ``SCRIPT_CAPABLE_NODE_TYPES`` (by full
        ID or last segment) or one with a word starting with "script".
    """
    return _script_capable_type(type_name)


def is_script_attribute(name: str) -> bool:
    """Return True for a node attribute that carries code or a path to code.

    Args:
        name: Attribute name, e.g. ``inputs:script``, ``inputs:codeStr`` or
            ``inputs:augmentationScript``. A full attribute path
            (``/World/G/node.inputs:script``) works too; only the part after
            the prim path is matched.

    Returns:
        Whether a word of the name starts with "script" or "code"
        (``inputs:usdTimecode`` does not).
    """
    return _guard_has_word(str(name or "").rsplit(".", 1)[-1], CODE_ATTRIBUTE_HINTS)


def is_script_extension(extension_id: str) -> bool:
    """Return True for an extension that runs Python attached to the stage.

    Args:
        extension_id: Bare name or version-suffixed ID
            (``omni.graph.scriptnode-1.50.0+107.3.0``).

    Returns:
        Whether enabling it is refused while script execution is off: one of
        ``SCRIPT_EXTENSIONS`` or a name with a word starting with "script"
        (``omni.isaac.robot_description_editor`` is not).
    """
    bare = extension_bare_name(extension_id)
    return bare.lower() in SCRIPT_EXTENSIONS or _guard_has_word(bare, SCRIPT_CAPABLE_NODE_HINTS)


def extension_bare_name(extension_id: str) -> str:
    """Return the extension name without its version suffix.

    Kit IDs are ``<name>-<version>``; names contain dots, never hyphens,
    while versions may (``1.0.0-rc1``), so the name ends at the first hyphen.

    Args:
        extension_id: Bare name or version-suffixed ID.

    Returns:
        The bare extension name.
    """
    return (extension_id or "").strip().split("-", 1)[0]


# Kit-side prim checks, for generated scripts that see prims rather than
# names. Defines _script_capable_prim(prim) and
# _script_capable_prims_under(prim, limit) on top of the shared functions.
# Inactive prims and instance proxies are included: activating one later must
# not slip a code node past the check.
SCRIPT_GUARD_CORE = """\
    def _script_capable_prim(prim):
        attr = prim.GetAttribute("node:type")
        return bool(attr and attr.IsValid() and _script_capable_type(attr.Get()))

    def _script_capable_prims_under(prim, limit=20):
        from pxr import Usd
        found = []
        predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
        for candidate in Usd.PrimRange(prim, predicate):
            if _script_capable_prim(candidate):
                found.append(str(candidate.GetPath()))
                if len(found) >= limit:
                    break
        return found
"""


def script_guard_source() -> str:
    """Return the Kit-side guard, with the constants and rules it reads, as source.

    Embed it in a generated script (``_compose_script`` fragment) to check
    prims inside Kit with the same functions ``is_script_capable_node_type``
    runs here. It defines ``_script_capable_type(type_name)``,
    ``_script_capable_prim(prim)`` and ``_script_capable_prims_under(prim,
    limit=20)``.

    Returns:
        Python source defining the guard functions.
    """
    shared = "\n\n".join(
        inspect.getsource(function)
        for function in (_guard_words, _guard_has_word, _script_capable_type)
    )
    return (
        "import re\n"
        "from typing import Any\n"
        f"SCRIPT_GUARD_TYPES = {_pyval(SCRIPT_GUARD_TYPES)}\n"
        f"SCRIPT_GUARD_SEGMENTS = {_pyval(SCRIPT_GUARD_SEGMENTS)}\n"
        f"SCRIPT_GUARD_HINTS = {_pyval(SCRIPT_GUARD_HINTS)}\n"
        f"_GUARD_WORD = re.compile({_pyval(_GUARD_WORD_PATTERN)})\n\n"
        + shared
        + "\n\n"
        + textwrap.dedent(SCRIPT_GUARD_CORE)
    )


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
            "node types. Common types include omni.graph.action.OnPlaybackTick, "
            "omni.graph.core.ReadVariable, "
            "isaacsim.core.nodes.IsaacArticulationController. With "
            "security.allow_script_execution off, node types that run Python given as data "
            "(ScriptNode, omni.warp.WarpKernel, Replicator's augmentation and condition nodes, "
            "and any type with a word starting with 'script') are refused."
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
                       (e.g. "omni.graph.action.OnPlaybackTick").

        Returns:
            Dict with created node path and type, or a
            ScriptExecutionDisabled error for a script-capable node type
            while script execution is off.
        """
        if not self.settings.security.allow_script_execution and is_script_capable_node_type(node_type):
            return _script_execution_refusal(
                f"Refusing to create {node_type!r}: a node of this type runs Python given as data.",
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
            "'/World/Graph/OnTick.outputs:tick' to '/World/Graph/Controller.inputs:execIn'. "
            "With security.allow_script_execution off, a connection into a code attribute "
            "(a name with a word starting with script or code) or into a script-capable node is refused: "
            "it would rewrite the code that node runs."
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
            Dict with connection confirmation, or a ScriptExecutionDisabled
            error for a target that carries code while script execution is
            off.
        """
        guard = not self.settings.security.allow_script_execution
        if guard and is_script_attribute(target_attr_path):
            return _script_execution_refusal(
                f"Refusing to connect into {target_attr_path!r}: a code attribute holds Python "
                "the node runs, and a connection rewrites it.",
                target=target_attr_path,
            )
        script = _compose_script(
            """\
            import json
            import omni.graph.core as og
            """,
            script_guard_source(),
            f"""\
            source = {_pyval(source_attr_path)}
            target = {_pyval(target_attr_path)}
            guard = {_pyval(guard)}
            """,
            """\
            node_type = None
            if guard:
                try:
                    node_type = og.Controller.node(target.rsplit(".", 1)[0]).get_type_name()
                except Exception:
                    node_type = None
            if guard and _script_capable_type(node_type):
                print(json.dumps({
                    "error": "Refusing to connect into " + target + ", an input of a " + str(node_type)
                             + " node, which runs Python given as data. Script execution is disabled "
                             "by security.allow_script_execution; nothing was connected.",
                    "error_type": "ScriptExecutionDisabled",
                    "node_type": node_type,
                }))
            else:
                try:
                    og.Controller.connect(source, target)
                    print(json.dumps({"connected": True, "source": source, "target": target}))
                except Exception as e:
                    print(json.dumps({"error": str(e)}))
            """,
        )
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
            "off, code attributes (a name with a word starting with script or code) and any write to a "
            "script-capable node (ScriptNode, omni.warp.WarpKernel, ...) are refused."
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
            return ErrorResponse(error="values is empty; nothing to set", error_type="ValidationError").model_dump()
        guard = not self.settings.security.allow_script_execution
        script_attributes = sorted(name for name in values if is_script_attribute(name))
        if guard and script_attributes:
            return _script_execution_refusal(
                "Refusing to write " + ", ".join(script_attributes)
                + ": a code attribute holds Python the node runs. Nothing was set.",
                attributes=script_attributes,
            )
        script = _compose_script(
            """\
            import json
            import omni.graph.core as og
            """,
            script_guard_source(),
            f"""\
            node_path = {_pyval(node_path)}
            values = {_pyval(values)}
            guard = {_pyval(guard)}
            """,
            """\
            applied = {}
            errors = {}
            try:
                node = og.Controller.node(node_path)
            except Exception:
                node = None
            if node is None or not node.is_valid():
                print(json.dumps({"error": "Node not found: " + node_path, "error_type": "NotFoundError"}))
            elif guard and _script_capable_type(node.get_type_name()):
                print(json.dumps({
                    "error": "Refusing to edit " + node_path + ", a " + node.get_type_name()
                             + " node, which runs Python given as data. Script execution is disabled "
                             "by security.allow_script_execution; nothing was set.",
                    "error_type": "ScriptExecutionDisabled",
                    "node_type": node.get_type_name(),
                }))
            else:
                for attr_name, val in values.items():
                    try:
                        og.Controller.set(og.Controller.attribute(attr_name, node_path), val)
                        applied[attr_name] = val
                    except Exception as e:
                        errors[attr_name] = str(e)
                result = {"node_path": node_path, "applied": applied, "count": len(applied)}
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
            """,
        )
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
