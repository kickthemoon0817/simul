"""Opening, saving and exporting the headless session's stages.

Each loaded stage gets its own instance of its root layer. Sdf keeps one
layer object per identifier for the whole process, so two stages opened
from the same file (or a stage and another stage that references that file)
would otherwise share the layer, and an unsaved edit made through one would
show up in the other. :func:`open_private_root` opens the root layer with a
file format argument unique to the load, which Sdf keys as a separate layer
while it still reads from and saves to the real file. Edits only ever go to
that private root layer (the stage's edit target), so the shared layers a
stage composes (sublayers, references, payloads) are never dirtied and can
be reloaded from disk when the files change.

A changed file's new content is read into a throwaway layer
(:func:`open_candidate`) and checked against the composition sandbox before
the shared layer is reloaded in place, since a reload recomposes every
stage that uses the layer.
"""

from __future__ import annotations

import os
import re
import secrets
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pxr import UsdUtils

from ._pxr import Sdf, Usd

# The file format argument that makes a load's root layer its own instance.
STAGE_ARG = "simul_stage"
# The one that keys a throwaway copy of a file read to check its new content.
CHECK_ARG = "simul_check"
EXPORT_FORMATS = (".usda", ".usdc", ".usd", ".usdz")
# A scheme such as omniverse:// or https:// ; a Windows drive letter is not one.
_URL = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]+:")
# This module's own file format arguments, as Sdf writes them into an identifier.
_OWN_ARG = rf"(?:{STAGE_ARG}|{CHECK_ARG})=[^&:@'\"\s]*"
_OWN_ARG_LATER = re.compile(rf"&{_OWN_ARG}")
_OWN_ARG_FIRST = re.compile(rf"(?<=:SDF_FORMAT_ARGS:){_OWN_ARG}&")
_OWN_ARG_ONLY = re.compile(rf":SDF_FORMAT_ARGS:{_OWN_ARG}")


def public_identifier(text: str) -> str:
    """``text`` with this module's file format arguments taken out of every layer identifier in it.

    A private root layer's identifier is the file path plus
    ``:SDF_FORMAT_ARGS:simul_stage=<hex>`` (a throwaway check copy's,
    ``simul_check=<hex>``). Those are the session's bookkeeping, not part of
    the file a caller named, so whatever a caller sees (a layer in a result,
    a pxr message quoting one) names the file. Other format arguments stay.
    """
    if "SDF_FORMAT_ARGS" not in text:
        return text
    text = _OWN_ARG_LATER.sub("", text)
    text = _OWN_ARG_FIRST.sub("", text)
    return _OWN_ARG_ONLY.sub("", text)


def open_private_root(identifier: str) -> Sdf.Layer:
    """Open ``identifier`` (a resolved path or a URL) as a root layer no other stage shares.

    Nothing is composed yet, so the layer can be held to the composition
    sandbox before any layer it names is read.

    Raises:
        Tf.ErrorException: pxr could not parse the file.
        RuntimeError: pxr could not open it and gave no reason.
    """
    root = Sdf.Layer.FindOrOpen(identifier, args={STAGE_ARG: secrets.token_hex(8)})
    if root is None:
        raise RuntimeError(f"pxr could not open {identifier}")
    return root


def compose_stage(root: Sdf.Layer) -> Usd.Stage:
    """Compose a stage on ``root``, with ``root`` as its edit target.

    Raises:
        RuntimeError: pxr could not compose a stage.
    """
    stage = Usd.Stage.Open(root)
    if stage is None:
        raise RuntimeError(f"pxr could not compose a stage from {root.realPath or root.identifier}")
    stage.SetEditTarget(Usd.EditTarget(root))
    return stage


def open_private_stage(file_path: str) -> Tuple[Usd.Stage, str]:
    """Open ``file_path`` as a stage whose root layer no other stage shares (no sandbox check).

    Returns:
        The stage and its root layer's identifier.
    """
    stage = compose_stage(open_private_root(file_path))
    return stage, stage.GetRootLayer().identifier


def open_candidate(layer: Sdf.Layer) -> Sdf.Layer:
    """Read ``layer``'s file again into a separate, throwaway layer.

    The layer in use keeps its content; the candidate holds what the file
    says now, so it can be checked before ``layer`` is reloaded in place.

    Raises:
        Tf.ErrorException: pxr could not parse the file's new content.
        RuntimeError: pxr could not open it and gave no reason.
    """
    path, args = Sdf.Layer.SplitIdentifier(layer.identifier)
    arguments = {key: value for key, value in dict(args).items() if key != STAGE_ARG}
    arguments[CHECK_ARG] = secrets.token_hex(8)
    candidate = Sdf.Layer.FindOrOpen(path, args=arguments)
    if candidate is None:
        raise RuntimeError(f"pxr could not open {path}")
    return candidate


def mtime(path: str) -> Optional[int]:
    """A file's modification time in nanoseconds, or None when it cannot be read."""
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return None


def file_layers(stage: Usd.Stage) -> List[Sdf.Layer]:
    """The file-backed layers a stage composes, root first."""
    root = stage.GetRootLayer()
    layers = [root] + [layer for layer in stage.GetUsedLayers() if layer != root]
    return [layer for layer in layers if not layer.anonymous and layer.realPath]


def file_mtimes(stage: Usd.Stage) -> Dict[str, Optional[int]]:
    """Modification time of each file the stage composes, by real path."""
    return {layer.realPath: mtime(layer.realPath) for layer in file_layers(stage)}


def changed_files(stage: Usd.Stage, recorded: Dict[str, Optional[int]]) -> List[str]:
    """The files whose modification time differs from ``recorded``."""
    return [path for path, when in recorded.items() if mtime(path) != when]


def _reanchor(layer: Sdf.Layer, source: Sdf.Layer, destination_dir: Optional[str]) -> int:
    """Rewrite ``layer``'s relative asset paths to resolve from ``destination_dir``.

    ``layer`` holds a copy of ``source``'s content. Relative asset paths in
    it (sublayers, references, payloads, asset-valued attributes) were
    authored relative to ``source``; written to another directory they would
    point at nothing. Each one is anchored to ``source`` and, when it names
    an existing file, made relative to ``destination_dir`` (or absolute when
    ``destination_dir`` is None, for packaging).

    Returns:
        The number of asset paths rewritten.
    """
    rewritten = [0]

    def modify(asset_path: str) -> str:
        if not asset_path or os.path.isabs(asset_path) or _URL.match(asset_path):
            return asset_path
        absolute = str(source.ComputeAbsolutePath(asset_path) or "")
        if not absolute or not os.path.isabs(absolute) or not os.path.exists(absolute):
            return asset_path
        if destination_dir is None:
            new_path = absolute
        else:
            new_path = os.path.relpath(absolute, destination_dir).replace(os.sep, "/")
            if not new_path.startswith("."):
                new_path = "./" + new_path
        if new_path != asset_path:
            rewritten[0] += 1
        return new_path

    UsdUtils.ModifyAssetPaths(layer, modify)
    return rewritten[0]


def export_stage(stage: Usd.Stage, destination: str, *, flatten: bool) -> Dict[str, object]:
    """Write ``stage`` to ``destination``, a .usda, .usdc, .usd or .usdz file.

    Without ``flatten`` the root layer is written as it is (its sublayers,
    references and payloads stay composition arcs), with relative asset
    paths re-anchored to the new location. With ``flatten`` the composed
    stage is written as one layer. A .usdz package is built with
    ``UsdUtils.CreateNewUsdzPackage`` from an intermediate .usdc in a
    temporary directory beside ``destination`` (removed afterwards), so it
    carries the files the layer depends on.

    Returns:
        ``rewritten_asset_paths``: how many asset paths were re-anchored.

    Raises:
        RuntimeError: pxr failed to write the file.
    """
    root = stage.GetRootLayer()
    suffix = Path(destination).suffix.lower()
    packaging = suffix == ".usdz"
    rewritten = 0
    if flatten:
        layer = stage.Flatten()
    else:
        layer = Sdf.Layer.CreateAnonymous(".usdc" if packaging else suffix)
        layer.TransferContent(root)
        rewritten = _reanchor(layer, root, None if packaging else os.path.dirname(destination))
    if not packaging:
        if not layer.Export(destination):
            raise RuntimeError(f"pxr could not write {destination}")
        return {"rewritten_asset_paths": rewritten}
    # The intermediate goes next to the package, inside the directory the
    # sandbox authorized for the write, never into the system temp dir.
    with tempfile.TemporaryDirectory(prefix=".simul-usdz-", dir=os.path.dirname(destination)) as scratch:
        intermediate = os.path.join(scratch, f"{Path(destination).stem or 'stage'}.usdc")
        if not layer.Export(intermediate):
            raise RuntimeError(f"pxr could not write the intermediate layer for {destination}")
        if not UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(intermediate), destination):
            raise RuntimeError(f"pxr could not package {destination}")
    return {"rewritten_asset_paths": rewritten}


__all__ = [
    "CHECK_ARG",
    "EXPORT_FORMATS",
    "STAGE_ARG",
    "changed_files",
    "compose_stage",
    "export_stage",
    "file_layers",
    "file_mtimes",
    "mtime",
    "open_candidate",
    "open_private_root",
    "open_private_stage",
]
