"""Opening, saving and exporting the headless session's stages.

Each loaded stage gets its own instance of its root layer. Sdf keeps one
layer object per identifier for the whole process, so two stages opened
from the same file (or a stage and another stage that references that file)
would otherwise share the layer, and an unsaved edit made through one would
show up in the other. :func:`open_private_stage` opens the root layer with a
file format argument unique to the load, which Sdf keys as a separate layer
while it still reads from and saves to the real file. Edits only ever go to
that private root layer (the stage's edit target), so the shared layers a
stage composes (sublayers, references, payloads) are never dirtied and can
be reloaded from disk when the files change.
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
EXPORT_FORMATS = (".usda", ".usdc", ".usd", ".usdz")
# A scheme such as omniverse:// or https:// ; a Windows drive letter is not one.
_URL = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]+:")


def open_private_stage(file_path: str) -> Tuple[Usd.Stage, str]:
    """Open ``file_path`` as a stage whose root layer no other stage shares.

    Args:
        file_path: Resolved path of a USD file.

    Returns:
        The stage and the token that keys its root layer.

    Raises:
        RuntimeError: pxr could not open the file.
    """
    token = secrets.token_hex(8)
    root = Sdf.Layer.FindOrOpen(file_path, args={STAGE_ARG: token})
    if root is None:
        raise RuntimeError(f"pxr could not open {file_path}")
    stage = Usd.Stage.Open(root)
    if stage is None:
        raise RuntimeError(f"pxr could not compose a stage from {file_path}")
    stage.SetEditTarget(Usd.EditTarget(root))
    return stage, token


def _mtime(path: str) -> Optional[int]:
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
    return {layer.realPath: _mtime(layer.realPath) for layer in file_layers(stage)}


def changed_files(stage: Usd.Stage, recorded: Dict[str, Optional[int]]) -> List[str]:
    """The files whose modification time differs from ``recorded``."""
    return [path for path, mtime in recorded.items() if _mtime(path) != mtime]


def refresh_shared_layers(stage: Usd.Stage) -> List[str]:
    """Reload the stage's non-root layers whose files changed on disk.

    Sdf hands every stage the same layer object for a sublayer or a
    referenced file and never rereads it on its own, so an edit made to the
    file outside this process stays invisible. ``Reload()`` without
    ``force`` rereads a layer only when its file's timestamp changed; dirty
    layers are skipped, since reloading would discard their edits (the
    session never edits them, so none should be dirty).

    Returns:
        The real paths of the layers that were reloaded.
    """
    root = stage.GetRootLayer()
    reloaded = []
    for layer in file_layers(stage):
        if layer == root or layer.dirty:
            continue
        if layer.Reload():
            reloaded.append(layer.realPath)
    return reloaded


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
    "EXPORT_FORMATS",
    "STAGE_ARG",
    "changed_files",
    "export_stage",
    "file_layers",
    "file_mtimes",
    "open_private_stage",
    "refresh_shared_layers",
]
