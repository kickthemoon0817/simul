"""Search and list the prims of a composed stage, paged and counted consistently.

Every walk here uses one predicate: active, defined, loaded, non-abstract prims,
plus (by default) the instance proxies under instanceable prims. The default
``Stage.Traverse()`` skips instance proxies, so a stage with three instanced
robots reported 65 prims where 116 compose, an instance looked childless, and a
search for joints found only the non-instanced copies. ``load_usd_file``'s
``total_prims``, ``summarize_scene`` and a match-everything search now count the
same prims.

Results are pages: ``offset`` and ``limit`` select a slice of the matches in
traversal order, ``total_matches`` counts them all and ``next_offset`` says
where the next page starts.
"""

from __future__ import annotations

import fnmatch
import re
from typing import Any, Callable, Dict, Iterator, List, Optional

from ._pxr import Usd
from .pxr_errors import relationship_targets
from .values import describe_attribute

SEARCH_TYPES = ("by_name", "by_path", "by_type", "by_api_schema")
MATCH_MODES = ("auto", "exact", "substring", "glob", "regex")

DEFAULT_SEARCH_LIMIT = 200
MAX_SEARCH_LIMIT = 5000
DEFAULT_SUBTREE_LIMIT = 150
MAX_SUBTREE_LIMIT = 1000


class SearchQueryError(ValueError):
    """A query the chosen match mode cannot use (an invalid regular expression)."""

    error_type = "ValidationError"


def traversal_predicate(include_instance_proxies: bool = True) -> Any:
    """Return the prim predicate every listing and count uses."""
    if include_instance_proxies:
        return Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    return Usd.PrimDefaultPredicate


def iter_prims(
    stage: Usd.Stage,
    root: Optional[Usd.Prim] = None,
    include_instance_proxies: bool = True,
) -> Iterator[Usd.Prim]:
    """Yield the prims under ``root`` (itself included), or every prim of the stage.

    The pseudo-root is never yielded.
    """
    start = stage.GetPseudoRoot() if root is None else root
    for prim in Usd.PrimRange(start, traversal_predicate(include_instance_proxies)):
        if not prim.IsPseudoRoot():
            yield prim


def count_prims(stage: Usd.Stage, include_instance_proxies: bool = True) -> int:
    """Count the stage's prims as every listing does."""
    return sum(1 for _ in iter_prims(stage, None, include_instance_proxies))


def _string_matcher(query: str, mode: str, case_sensitive: bool) -> Callable[[str], bool]:
    """Build a predicate over strings for one of the string match modes."""
    if mode == "regex":
        try:
            pattern = re.compile(query, 0 if case_sensitive else re.IGNORECASE)
        except re.error as exc:
            raise SearchQueryError(f"Invalid regular expression {query!r}: {exc}") from exc
        return lambda text: pattern.search(text) is not None
    needle = query if case_sensitive else query.casefold()

    def fold(text: str) -> str:
        return text if case_sensitive else text.casefold()

    if mode == "exact":
        return lambda text: fold(text) == needle
    if mode == "glob":
        return lambda text: fnmatch.fnmatchcase(fold(text), needle)
    return lambda text: needle in fold(text)


def _prim_matcher(
    search_type: str, query: str, match: str, case_sensitive: bool, notes: List[str]
) -> Callable[[Usd.Prim], bool]:
    """Build the prim predicate a search applies."""
    if search_type in ("by_name", "by_path"):
        text_of = (lambda prim: prim.GetName()) if search_type == "by_name" else (
            lambda prim: prim.GetPath().pathString
        )
        matches = _string_matcher(query, "substring" if match == "auto" else match, case_sensitive)
        return lambda prim: matches(text_of(prim))

    if search_type == "by_type":
        if match == "auto":
            schema = Usd.SchemaRegistry.GetTypeFromName(query)
            if schema and Usd.SchemaRegistry.IsTyped(schema):
                return lambda prim: prim.IsA(schema)
            notes.append(
                f"{query!r} is not a registered typed schema, so prim type names were "
                "compared to it exactly. Registered types match their whole family "
                "(PhysicsJoint finds every joint, Gprim every geometric primitive)."
            )
            match = "exact"
        matches = _string_matcher(query, match, case_sensitive)
        return lambda prim: matches(prim.GetTypeName())

    # by_api_schema: applied schema names; a multiple-apply schema is applied
    # as "Name:instance", which its bare name matches in auto mode.
    if match == "auto":
        needle = query if case_sensitive else query.casefold()

        def applied(prim: Usd.Prim) -> bool:
            for name in prim.GetAppliedSchemas():
                folded = name if case_sensitive else name.casefold()
                if folded == needle or folded.startswith(needle + ":"):
                    return True
            return False

        return applied
    matches = _string_matcher(query, match, case_sensitive)
    return lambda prim: any(matches(name) for name in prim.GetAppliedSchemas())


def search_prims(
    stage: Usd.Stage,
    search_type: str,
    query: str,
    *,
    match: str = "auto",
    case_sensitive: bool = False,
    root: Optional[Usd.Prim] = None,
    include_instance_proxies: bool = True,
    offset: int = 0,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> Dict[str, Any]:
    """Find prims by name, path, type (schema family) or applied API schema.

    Args:
        stage: Stage to search.
        search_type: One of ``SEARCH_TYPES``.
        query: Name, path, type or API schema to look for. An empty query
            with ``by_name``/``by_path`` substring matching matches every prim.
        match: ``auto`` (substring for names and paths, the schema family for
            types, the schema and its multiple-apply instances for API
            schemas), ``exact``, ``substring``, ``glob`` or ``regex``.
        case_sensitive: Compare case-sensitively; schema family matching in
            ``auto`` mode is always exact about the schema name.
        root: Search only this prim and its descendants; None for the stage.
        include_instance_proxies: Look inside instances.
        offset: Matches to skip, in traversal order.
        limit: Matches to return.

    Returns:
        ``results`` (paths), ``count`` (in this page), ``total_matches``,
        ``searched`` (prims visited), ``offset``, ``limit``, ``truncated``
        (more matches follow), ``next_offset`` and ``notes``.

    Raises:
        ValueError: For an unknown search type or match mode.
        SearchQueryError: For an invalid regular expression.
    """
    if search_type not in SEARCH_TYPES:
        raise ValueError(f"search_type must be one of {list(SEARCH_TYPES)}")
    if match not in MATCH_MODES:
        raise ValueError(f"match must be one of {list(MATCH_MODES)}")
    offset = max(0, int(offset))
    limit = max(1, min(int(limit), MAX_SEARCH_LIMIT))
    notes: List[str] = []
    matches = _prim_matcher(search_type, query, match, case_sensitive, notes)
    results: List[str] = []
    total = 0
    searched = 0
    for prim in iter_prims(stage, root, include_instance_proxies):
        searched += 1
        if not matches(prim):
            continue
        if offset <= total < offset + limit:
            results.append(prim.GetPath().pathString)
        total += 1
    truncated = offset + len(results) < total
    return {
        "results": results,
        "count": len(results),
        "total_matches": total,
        "searched": searched,
        "offset": offset,
        "limit": limit,
        "truncated": truncated,
        "next_offset": offset + len(results) if truncated else None,
        "notes": notes,
    }


def _authored_properties(prim: Usd.Prim, array_preview: int, patterns: Optional[List[str]]) -> Dict[str, Any]:
    def wanted(name: str) -> bool:
        return patterns is None or any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)

    attributes = {
        attr.GetName(): describe_attribute(attr, None, array_preview)
        for attr in prim.GetAuthoredAttributes()
        if wanted(attr.GetName())
    }
    problems: List[Dict[str, Any]] = []
    relationships = {}
    for rel in prim.GetAuthoredRelationships():
        if wanted(rel.GetName()):
            targets = relationship_targets(rel, problems)
            if targets is not None:
                relationships[rel.GetName()] = targets
    properties: Dict[str, Any] = {"attributes": attributes, "relationships": relationships}
    if problems:
        properties["warnings"] = problems
    return properties


def list_subtree(
    stage: Usd.Stage,
    root: Usd.Prim,
    *,
    max_depth: int = 5,
    include_instance_proxies: bool = True,
    include_properties: bool = False,
    attribute_filter: Optional[List[str]] = None,
    array_preview: int = 4,
    offset: int = 0,
    limit: int = DEFAULT_SUBTREE_LIMIT,
) -> Dict[str, Any]:
    """List a prim and its descendants, depth-first, one record per prim.

    Args:
        stage: Stage the root belongs to.
        root: Subtree root (the pseudo-root lists the stage).
        max_depth: Deepest level below the root to list (the root is 0).
        include_instance_proxies: Descend into instances.
        include_properties: Add each prim's authored attributes (as
            ``{type, value, authored, ...}`` records) and authored
            relationship targets.
        attribute_filter: Glob patterns limiting the properties added.
        array_preview: Leading elements of array attributes to include.
        offset: Prims to skip, in traversal order.
        limit: Prims to return.

    Returns:
        ``prims`` (``path``, ``name``, ``type``, ``depth``, ``child_count``,
        ``api_schemas``, plus ``kind``/``is_instance``/``is_instance_proxy``
        when set), ``count``, ``total``, ``offset``, ``limit``, ``truncated``
        and ``next_offset``.
    """
    offset = max(0, int(offset))
    limit = max(1, min(int(limit), MAX_SUBTREE_LIMIT))
    max_depth = max(0, int(max_depth))
    predicate = traversal_predicate(include_instance_proxies)
    root_depth = root.GetPath().pathElementCount
    prims: List[Dict[str, Any]] = []
    total = 0
    walk = iter(Usd.PrimRange(root, predicate))
    for prim in walk:
        depth = prim.GetPath().pathElementCount - root_depth
        if depth >= max_depth:
            walk.PruneChildren()
        if prim.IsPseudoRoot():
            continue
        if offset <= total < offset + limit:
            record: Dict[str, Any] = {
                "path": prim.GetPath().pathString,
                "name": prim.GetName(),
                "type": prim.GetTypeName(),
                "depth": depth,
                "child_count": len(prim.GetFilteredChildren(predicate)),
                "api_schemas": list(prim.GetAppliedSchemas()),
            }
            kind = Usd.ModelAPI(prim).GetKind()
            if kind:
                record["kind"] = kind
            if prim.IsInstance():
                record["is_instance"] = True
            if prim.IsInstanceProxy():
                record["is_instance_proxy"] = True
            if include_properties:
                record.update(_authored_properties(prim, array_preview, attribute_filter))
            prims.append(record)
        total += 1
    truncated = offset + len(prims) < total
    return {
        "root_path": root.GetPath().pathString,
        "max_depth": max_depth,
        "prims": prims,
        "count": len(prims),
        "total": total,
        "offset": offset,
        "limit": limit,
        "truncated": truncated,
        "next_offset": offset + len(prims) if truncated else None,
    }


__all__ = [
    "DEFAULT_SEARCH_LIMIT",
    "DEFAULT_SUBTREE_LIMIT",
    "MATCH_MODES",
    "MAX_SEARCH_LIMIT",
    "MAX_SUBTREE_LIMIT",
    "SEARCH_TYPES",
    "SearchQueryError",
    "count_prims",
    "iter_prims",
    "list_subtree",
    "search_prims",
    "traversal_predicate",
]
