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

Changes on disk are told by each file's :func:`stamp` (modification time,
size and inode), so a write in the same clock tick as the read still shows.
A file is stamped just before pxr reads it (:func:`open_layer` records the
stamps of the layers a load opens), never after: a write that lands while
pxr reads the file then leaves an older stamp, and the file shows as changed
instead of being vouched for with the content it had before.
An export is written to a temporary file beside the destination and then
moved into place in one step (:func:`export_stage`): a reader never sees a
partial file, and without ``overwrite`` the destination is claimed
exclusively, so two writers cannot both create it.
"""

from __future__ import annotations

import os
import re
import secrets
import shutil
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from pxr import Ar, UsdUtils

from ._pxr import Sdf, Tf, Usd

# The file format argument that makes a load's root layer its own instance.
STAGE_ARG = "simul_stage"
# The one that keys a throwaway copy of a file read to check its new content.
CHECK_ARG = "simul_check"
EXPORT_FORMATS = (".usda", ".usdc", ".usd", ".usdz")
# What a file looked like when it was read: modification time (ns), size and
# inode; None when it cannot be stat'ed.
Stamp = Optional[Tuple[int, int, int]]
# A scheme such as omniverse:// or https:// ; a Windows drive letter is not one.
_URL = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]+:")
# This module's own file format arguments, as Sdf writes them into an identifier.
_OWN_ARG = rf"(?:{STAGE_ARG}|{CHECK_ARG})=[^&:@'\"\s]*"
_OWN_ARG_LATER = re.compile(rf"&{_OWN_ARG}")
_OWN_ARG_FIRST = re.compile(rf"(?<=:SDF_FORMAT_ARGS:){_OWN_ARG}&")
_OWN_ARG_ONLY = re.compile(rf":SDF_FORMAT_ARGS:{_OWN_ARG}")
# Listing pxr's layers fails while another thread releases one (see
# layers_in_memory); the list is asked for this many times, a little later each time.
_LIST_ATTEMPTS = 4
_LIST_BACKOFF_SECONDS = 0.01


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


class LoadedLayersUnavailable(RuntimeError):
    """pxr could not list the layers it holds, each time :func:`layers_in_memory` asked."""


def layers_in_memory() -> Set[str]:
    """Identifiers of the file layers pxr holds in this process now.

    pxr hands a stage that opens one of these files the copy in memory,
    which may be older than the file, instead of reading the file.

    ``Sdf.Layer.GetLoadedLayers`` raises ``Tf.ErrorException`` ("Multiple
    Python objects for C++ object ...") when another thread is releasing a
    layer it lists: the layer's Python object is gone while its C++ object is
    still being destroyed. The MCP server unloads an ended session's stages
    on a worker thread beside reads such as ``validate_usd_file``, so this
    happens. The list is a snapshot, so it is asked for again, a little
    later each time.

    Raises:
        LoadedLayersUnavailable: Every attempt failed; nothing says which
            layers pxr holds.
    """
    failure: Optional[BaseException] = None
    for attempt in range(_LIST_ATTEMPTS):
        if attempt:
            time.sleep(_LIST_BACKOFF_SECONDS * attempt)
        try:
            loaded = Sdf.Layer.GetLoadedLayers()
        except Tf.ErrorException as exc:
            failure = exc
            continue
        identifiers: Set[str] = set()
        for layer in loaded:
            try:
                if layer and not layer.anonymous:
                    identifiers.add(layer.identifier)
            except Exception:
                continue  # released while the list was read
        return identifiers
    raise LoadedLayersUnavailable(
        f"pxr could not list the layers it holds ({_LIST_ATTEMPTS} attempts): {failure}"
    ) from failure


def open_layer(identifier: str, reads: Optional[Dict[str, Stamp]] = None) -> Optional[Sdf.Layer]:
    """Open ``identifier`` as ``Sdf.Layer.FindOrOpen`` does, stamping the file first when pxr reads it.

    pxr reads the file only for a layer it does not hold yet. With ``reads``,
    the file's :func:`stamp` is taken just before that read and recorded
    under the layer's real path, so a write that lands while pxr reads it
    leaves the recorded stamp older than the file. A layer pxr already holds
    is returned as it is and not recorded: this call read nothing.

    Args:
        identifier: An anchored layer identifier, as composition opens it.
        reads: Where to record the stamps; None records nothing.

    Returns:
        The layer, or None when pxr cannot open it (composition reports why).
    """
    try:
        if reads is None or Sdf.Layer.Find(identifier):
            return Sdf.Layer.FindOrOpen(identifier)
        path = Sdf.Layer.SplitIdentifier(identifier)[0]
        resolved = str(Ar.GetResolver().Resolve(path)) if path else ""
        before = stamp(resolved) if resolved else None
        layer = Sdf.Layer.FindOrOpen(identifier)
    except Exception:
        return None
    if layer is not None and layer.realPath:
        # A stamp of another file than the one read (an unexpected resolve)
        # only makes the layer look changed, so it is reread.
        reads.setdefault(layer.realPath, before)
    return layer


def stamp_unread(identifier: str, reads: Dict[str, Stamp]) -> None:
    """Stamp the file ``identifier`` names before composition reads it, without reading it here.

    For a layer a load may not use (one an unselected variant names): its
    stamp is recorded under its resolved path as ``open_layer`` records it,
    so if composition does read it later the stamp still predates the read.
    Nothing is recorded for a layer pxr already holds, which composition
    does not read.

    Args:
        identifier: An anchored layer identifier, as composition opens it.
        reads: Where the stamp is recorded.
    """
    try:
        if Sdf.Layer.Find(identifier):
            return
        path = Sdf.Layer.SplitIdentifier(identifier)[0]
        resolved = str(Ar.GetResolver().Resolve(path)) if path else ""
    except Exception:
        return
    if resolved:
        reads.setdefault(resolved, stamp(resolved))


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


def stamp(path: str) -> Stamp:
    """What ``path`` looks like on disk now, to compare with a later look; None when it cannot be read.

    The modification time alone misses a write in the same clock tick as the
    read (a coarse filesystem clock); the size and the inode (a save that
    replaces the file gets a new one) catch most of those. A layer inside a
    package (``/x/pkg.usdz[inner.usda]``) changes with its package, so it is
    stamped by the package file.
    """
    if Ar.IsPackageRelativePath(path):
        path = Ar.SplitPackageRelativePathOuter(path)[0]
    try:
        info = os.stat(path)
    except OSError:
        return None
    return (info.st_mtime_ns, info.st_size, info.st_ino)


def file_identity(path: str) -> str:
    """One key for every spelling of the file ``path`` names, for locking writers of that file.

    Keyed by the directory's device and inode and the file name in Unicode
    NFC, case-folded: on APFS or NTFS ``Case.usda`` and ``case.usda``, or a
    name in NFC and in NFD, are one file, and the key must not change when a
    save replaces the file (a new inode) or for a destination not created
    yet. On a case-sensitive filesystem two names that differ only in case
    share a key, which only makes their writers take turns. Falls back to
    the normalised path when the directory cannot be stat'ed.
    """
    directory, name = os.path.split(os.path.abspath(path))
    folded = unicodedata.normalize("NFC", name).casefold()
    try:
        info = os.stat(directory)
    except OSError:
        return unicodedata.normalize("NFC", os.path.join(directory, name)).casefold()
    return f"{info.st_dev}:{info.st_ino}:{folded}"


def same_file(first: str, second: str) -> bool:
    """Whether two paths name one file, however each is spelled."""
    try:
        if os.path.samefile(first, second):
            return True
    except OSError:
        pass
    return file_identity(first) == file_identity(second)


def file_layers(stage: Usd.Stage) -> List[Sdf.Layer]:
    """The file-backed layers a stage composes, root first."""
    root = stage.GetRootLayer()
    layers = [root] + [layer for layer in stage.GetUsedLayers() if layer != root]
    return [layer for layer in layers if not layer.anonymous and layer.realPath]


def file_stamps(stage: Usd.Stage) -> Dict[str, Stamp]:
    """The :func:`stamp` of each file the stage composes, by real path."""
    return {layer.realPath: stamp(layer.realPath) for layer in file_layers(stage)}


def changed_files(stage: Usd.Stage, recorded: Dict[str, Stamp]) -> List[str]:
    """The files whose :func:`stamp` differs from ``recorded``."""
    return [path for path, when in recorded.items() if stamp(path) != when]


def composes_own_file(stage: Usd.Stage) -> bool:
    """Whether the stage reaches its own root file again through an arc (``@./self.usda@</Proto>``).

    Such an arc opens the file's shared layer, not the stage's private root,
    so it shows the file as it is on disk, without the stage's unsaved edits.
    """
    root = stage.GetRootLayer()
    path = root.realPath
    return bool(path) and any(layer != root and layer.realPath == path for layer in stage.GetUsedLayers())


def _resolves(asset_path: str) -> bool:
    """Whether pxr's resolver finds ``asset_path`` (a file inside a package, for one)."""
    try:
        return bool(str(Ar.GetResolver().Resolve(asset_path)))
    except Exception:
        return False


def _reanchor(layer: Sdf.Layer, source: Sdf.Layer, destination_dir: Optional[str]) -> int:
    """Rewrite ``layer``'s relative asset paths to resolve from ``destination_dir``.

    ``layer`` holds a copy of ``source``'s content. Relative asset paths in
    it (sublayers, references, payloads, asset-valued attributes) were
    authored relative to ``source``; written to another directory they would
    point at nothing. Each one is anchored to ``source`` and, when it names
    an existing file, made relative to ``destination_dir`` (or absolute when
    ``destination_dir`` is None, for packaging).

    A path that anchors inside a package (the files beside the root layer of
    a .usdz, or ``./other.usdz[inner.usda]``) is kept pointing into that
    package: ``../pkg.usdz[inner.usda]`` from the new location, or the
    package's absolute path for packaging, which pxr then packs as a nested
    package.

    Returns:
        The number of asset paths rewritten.
    """
    rewritten = [0]

    def relative(path: str) -> str:
        new_path = os.path.relpath(path, destination_dir).replace(os.sep, "/")
        return new_path if new_path.startswith(".") else "./" + new_path

    def modify(asset_path: str) -> str:
        if not asset_path or os.path.isabs(asset_path) or _URL.match(asset_path):
            return asset_path
        absolute = str(source.ComputeAbsolutePath(asset_path) or "")
        if Ar.IsPackageRelativePath(absolute):
            outer, inner = Ar.SplitPackageRelativePathOuter(absolute)
            if not os.path.isabs(outer) or not os.path.exists(outer) or not _resolves(absolute):
                return asset_path
            new_path = (
                absolute if destination_dir is None else Ar.JoinPackageRelativePath(relative(outer), inner)
            )
        elif not absolute or not os.path.isabs(absolute) or not os.path.exists(absolute):
            return asset_path
        else:
            new_path = absolute if destination_dir is None else relative(absolute)
        if new_path != asset_path:
            rewritten[0] += 1
        return new_path

    UsdUtils.ModifyAssetPaths(layer, modify)
    return rewritten[0]


class DestinationExists(Exception):
    """``export_stage`` without ``overwrite`` found its destination already there."""


def _temporary_beside(destination: str) -> str:
    """A path in ``destination``'s directory, with its extension, that names no file yet."""
    directory, name = os.path.split(destination)
    stem, suffix = os.path.splitext(name)
    return os.path.join(directory, f".{stem}.simul-{secrets.token_hex(6)}{suffix}")


def _commit(written: str, destination: str, *, overwrite: bool) -> None:
    """Move the finished file ``written`` to ``destination`` in one step.

    With ``overwrite`` it replaces whatever is there, keeping that file's
    permissions. Without it the destination is claimed exclusively: a hard
    link fails when the name exists, so of two writers only one creates it.
    On a filesystem without hard links an empty placeholder is created with
    ``O_EXCL`` first and then replaced.

    Raises:
        DestinationExists: ``overwrite`` is False and the destination exists.
    """
    if overwrite:
        if os.path.exists(destination):
            try:
                shutil.copymode(destination, written)
            except OSError:
                pass
        os.replace(written, destination)
        return
    try:
        os.link(written, destination)
    except FileExistsError:
        raise DestinationExists(destination) from None
    except OSError:
        try:
            os.close(os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666))
        except FileExistsError:
            raise DestinationExists(destination) from None
        try:
            os.replace(written, destination)
        except BaseException:
            os.unlink(destination)  # the empty placeholder this call created
            raise
        return
    os.unlink(written)


def export_stage(
    stage: Usd.Stage, destination: str, *, flatten: bool, overwrite: bool = False
) -> Dict[str, object]:
    """Write ``stage`` to ``destination``, a .usda, .usdc, .usd or .usdz file.

    Without ``flatten`` the root layer is written as it is (its sublayers,
    references and payloads stay composition arcs), with relative asset
    paths re-anchored to the new location. With ``flatten`` the composed
    stage is written as one layer. A .usdz package is built with
    ``UsdUtils.CreateNewUsdzPackage`` from an intermediate .usdc in a
    temporary directory beside ``destination`` (removed afterwards), so it
    carries the files the layer depends on.

    The file is written under a temporary name in ``destination``'s
    directory and moved into place once complete (``_commit``), so a failed
    write leaves nothing behind and a reader never sees half a file.

    Returns:
        ``rewritten_asset_paths``: how many asset paths were re-anchored.

    Raises:
        DestinationExists: ``overwrite`` is False and ``destination`` exists
            when the file is moved into place.
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
    written = _temporary_beside(destination)
    try:
        if not packaging:
            if not layer.Export(written):
                raise RuntimeError(f"pxr could not write {destination}")
        else:
            # The intermediate goes next to the package, inside the directory
            # the sandbox authorized for the write, never into the system temp dir.
            with tempfile.TemporaryDirectory(prefix=".simul-usdz-", dir=os.path.dirname(destination)) as scratch:
                intermediate = os.path.join(scratch, f"{Path(destination).stem or 'stage'}.usdc")
                if not layer.Export(intermediate):
                    raise RuntimeError(f"pxr could not write the intermediate layer for {destination}")
                if not UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(intermediate), written):
                    raise RuntimeError(f"pxr could not package {destination}")
        _commit(written, destination, overwrite=overwrite)
    finally:
        if os.path.lexists(written):
            os.unlink(written)
    return {"rewritten_asset_paths": rewritten}


__all__ = [
    "CHECK_ARG",
    "EXPORT_FORMATS",
    "STAGE_ARG",
    "DestinationExists",
    "LoadedLayersUnavailable",
    "Stamp",
    "changed_files",
    "compose_stage",
    "composes_own_file",
    "export_stage",
    "file_identity",
    "file_layers",
    "file_stamps",
    "layers_in_memory",
    "open_candidate",
    "open_layer",
    "open_private_root",
    "open_private_stage",
    "same_file",
    "stamp",
]
