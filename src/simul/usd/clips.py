"""Find the value clip sets a layer authors.

Both the composition sandbox (to tell a clip path from another asset value)
and the animation range (to find clipped prims) need every prim spec of a
layer that authors ``clips``, in every variant. Walking spec objects
(``rootPrims``, ``nameChildren``, ``variantSets``) costs about 21 µs per
spec in Python, 4 s for a 200,000-prim layer. ``Sdf.Layer.Traverse`` walks
the paths in C++ and hands each to a callback, and only prim and variant
paths are looked up, so the same layer takes about 0.2 s.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ._pxr import Sdf


def clip_spec_paths(layer: "Sdf.Layer") -> List["Sdf.Path"]:
    """The paths of the prim specs in ``layer`` that author ``clips``, variant specs included.

    A path inside a variant keeps its variant selection
    (``/Model{look=a}``); ``StripAllVariantSelections`` gives the prim.
    """
    found: List[Sdf.Path] = []

    def visit(path: "Sdf.Path") -> None:
        if path.IsPrimPath() or path.IsPrimVariantSelectionPath():
            # A variant set's own path (/Model{look=}) has no prim spec.
            spec = layer.GetPrimAtPath(path)
            if spec and spec.HasInfo("clips"):
                found.append(path)

    layer.Traverse(Sdf.Path.absoluteRootPath, visit)
    return found


def clip_sets(layer: "Sdf.Layer") -> List[Tuple["Sdf.Path", Dict[str, Any]]]:
    """Each prim spec authoring ``clips`` in ``layer``, with its clip sets by name."""
    return [(path, dict(layer.GetPrimAtPath(path).GetInfo("clips"))) for path in clip_spec_paths(layer)]


__all__ = ["clip_sets", "clip_spec_paths"]
