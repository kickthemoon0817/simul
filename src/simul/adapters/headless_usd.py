"""
Headless USD adapter for Isaac Sim MCP Server.

This module provides a headless USD adapter that works with pure pxr library
without requiring Isaac Sim runtime. Suitable for USD file analysis and
processing without GUI or simulation capabilities.
"""

from __future__ import annotations

import os
import secrets
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple, Union

from ..usd._pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from ..logging import get_logger, LoggerMixin
from ..utils.timing import monitor_performance
from ..config import Settings, get_settings
from ..usd import USDReader, BBoxCache, MeshOperations, SceneSummarizer
from ..usd import editing, stage_files
from ..usd.editing import USDEditError
from ..usd.reader import USDStageInfo, USDPrimInfo
from ..usd.summarize import SceneSummary
from ..usd.sandbox import CompositionSandbox, OutsidePath
from ..usd.validation import (
    USDLoadError,
    check_usd_file,
    check_usd_url,
    open_root_layer,
    url_scheme,
)
from ..utils.paths import SandboxDenied

logger = get_logger(__name__)

_UNSET = object()


def _checked(check: Callable[[], Any]) -> Any:
    """Run a composition sandbox check, naming files without the session's layer arguments.

    A private root layer's identifier carries a ``simul_stage`` file format
    argument and a throwaway copy a ``simul_check`` one; a denial reports the
    plain file path the caller knows.
    """
    try:
        return check()
    except SandboxDenied as exc:
        for entry in exc.details.get("outside_sandbox", []):
            if isinstance(entry, dict) and entry.get("layer"):
                entry["layer"] = Sdf.Layer.SplitIdentifier(entry["layer"])[0]
        raise


# A session that keeps loading without unloading is refused past this many stages.
MAX_STAGES_PER_OWNER = 32
SAVE_FAILED = editing.SAVE_FAILED
RELOAD_FAILED = editing.RELOAD_FAILED
INVALID_FILE_PATH = editing.INVALID_FILE_PATH


@dataclass
class _StageRecord:
    """What the session knows about one loaded stage besides the stage itself.

    Attributes:
        file_path: Resolved path of the root file.
        owner: The MCP session that loaded the stage (None outside MCP, and
            for anonymous sessionless HTTP requests).
        mtimes: Modification time of each file the stage composes, as of
            the load, the last reload or the last save.
        unreal_content: The Unreal content paths the composition sandbox
            let through as editor objects rather than files, as of the load
            or the last reload.
    """

    file_path: str
    owner: Optional[str]
    mtimes: Dict[str, Optional[int]] = field(default_factory=dict)
    unreal_content: List[OutsidePath] = field(default_factory=list)


class HeadlessUSDSession(LoggerMixin):
    """
    A headless USD session for file operations without Isaac Sim runtime.

    Provides USD file loading, analysis, and manipulation capabilities
    using only the pxr library.

    Stage model: every load opens the file into its own stage with a root
    layer no other stage shares (``stage_files.open_private_root``), and
    edits go only to that root layer, so unsaved edits never leak into
    another stage. A stage belongs to the MCP session that loaded it and is
    invisible to every other session. Loading a file the session already
    holds a clean, current stage for returns that stage; otherwise (unsaved
    edits, or the file changed on disk) a fresh stage is opened from disk.

    A changed sublayer or referenced file is reloaded in place, which every
    stage composing it sees; its new content is first held to the
    composition sandbox (``set_sandbox_factory``), and a file that now
    reaches outside it is not reloaded.

    Calls may run on worker threads (the MCP server's USD scheduler); the
    stage registry is guarded by a lock, and the calling MCP session is
    named per thread with ``acting_for``.
    """

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize headless USD session.

        Args:
            settings: Configuration settings
        """
        self.settings = settings or get_settings()
        self.usd_reader = USDReader(enable_caching=self.settings.usd.cache_enabled)
        self.mesh_ops = MeshOperations()
        self.summarizer = SceneSummarizer()

        self._active_stages: Dict[str, Usd.Stage] = {}
        self._bbox_caches: Dict[str, BBoxCache] = {}
        self._stage_records: Dict[str, _StageRecord] = {}
        self._registry = threading.RLock()
        # Modification time of each shared (non-root) layer's file when the
        # session last read or checked it, to tell which ones changed since.
        self._shared_mtimes: Dict[str, Optional[int]] = {}
        # Builds a fresh CompositionSandbox for the reload checks; None (the
        # CLI, direct use) checks nothing beyond what a load is given.
        self._sandbox_factory: Optional[Callable[[], CompositionSandbox]] = None
        # Names the MCP session a call belongs to: acting_for sets it for the
        # calling thread; the resolver is the fallback. Outside MCP every
        # caller is the same owner.
        self._acting = threading.local()
        self._owner_resolver: Callable[[], Optional[str]] = lambda: None

        self.logger.info("Headless USD session initialized")

    def set_owner_resolver(self, resolver: Callable[[], Optional[str]]) -> None:
        """
        Install the function that names the MCP session of the current call.

        Used when no ``acting_for`` block names it for the calling thread.

        Args:
            resolver: Returns the calling session's id, or None
        """
        self._owner_resolver = resolver

    def set_sandbox_factory(self, factory: Optional[Callable[[], CompositionSandbox]]) -> None:
        """
        Install what builds a fresh composition sandbox for reload checks.

        Args:
            factory: Returns a new CompositionSandbox for the server's path
                policy; None turns the checks off
        """
        self._sandbox_factory = factory

    @contextmanager
    def acting_for(self, owner: Optional[str]) -> Iterator[None]:
        """
        Run the calls in this block, on this thread, as MCP session ``owner``.

        The server reads the session from the request on its event loop and
        runs the pxr work on a worker thread, where the request is not
        visible; this carries it across.

        Args:
            owner: The calling MCP session's id, or None
        """
        previous = getattr(self._acting, "owner", _UNSET)
        self._acting.owner = owner
        try:
            yield
        finally:
            if previous is _UNSET:
                del self._acting.owner
            else:
                self._acting.owner = previous

    def _owner(self) -> Optional[str]:
        owner = getattr(self._acting, "owner", _UNSET)
        if owner is not _UNSET:
            return owner
        try:
            return self._owner_resolver()
        except Exception:
            return None

    def _owned_ids(self) -> List[str]:
        owner = self._owner()
        with self._registry:
            return [stage_id for stage_id, record in self._stage_records.items() if record.owner == owner]

    def load_stage(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Load a USD stage from file.

        Returns the caller's existing stage for the file when it has no
        unsaved edits and no file it composes changed on disk; otherwise
        opens a new stage from disk, rereading any changed sublayer or
        referenced file first.

        Args:
            file_path: Path to USD file

        Returns:
            Stage identifier or None if failed; ``load_stage_or_raise``
            reports why.

        Raises:
            USDEditError: The caller already holds ``MAX_STAGES_PER_OWNER``
                stages.
        """
        try:
            return self.load_stage_or_raise(file_path)
        except USDLoadError as e:
            self.logger.error(str(e))
            return None
        except USDEditError:
            raise
        except Exception as e:
            self.logger.error(f"Error loading USD stage {file_path}: {e}")
            return None

    @monitor_performance("headless_session.load_stage")
    def load_stage_or_raise(
        self, file_path: Union[str, Path], sandbox: Optional[CompositionSandbox] = None
    ) -> str:
        """
        Load a USD stage from file, saying why when it cannot.

        Returns the caller's existing stage for the file when it has no
        unsaved edits and no file it composes changed on disk. Otherwise it
        opens the file into a private root layer, holds that layer and
        everything it names to ``sandbox`` before composing, composes the
        stage, rereads any shared layer whose file changed (once its new
        content passed the sandbox too), and checks the composed stage.

        Args:
            file_path: Path to USD file
            sandbox: Holds every layer and asset the stage draws on to the
                file sandbox; None (the CLI) checks nothing beyond the file.

        Returns:
            Stage identifier

        Raises:
            USDLoadError: The file failed the access checks shared with
                validation (missing, not a file, extension outside
                ``usd.allowed_extensions``, unreadable, empty, over
                ``usd.max_file_size_mb``) or pxr could not parse it; the
                error carries pxr's own reason.
            SandboxDenied: The file's composition reaches outside the
                sandbox, or a changed shared file now does (that file is not
                reloaded; stages using it keep its old content).
            USDEditError: The caller already holds ``MAX_STAGES_PER_OWNER``
                stages.
        """
        remote = url_scheme(str(file_path)) is not None
        head = b""
        if remote:
            # Only with an asset resolver registered for the scheme.
            file_path = str(file_path)
            check_usd_url(file_path)
        else:
            file_path = str(Path(file_path).resolve())
            head = check_usd_file(file_path, self.settings.usd).head

        owner = self._owner()
        with self._registry:
            for stage_id, record in self._stage_records.items():
                stage = self._active_stages[stage_id]
                if (
                    record.owner == owner
                    and record.file_path == file_path
                    and not stage.GetRootLayer().dirty
                    and not stage_files.changed_files(stage, record.mtimes)
                ):
                    self.logger.info(f"Reusing clean USD stage {stage_id} for {file_path}")
                    return stage_id
        owned = self._owned_ids()
        if len(owned) >= MAX_STAGES_PER_OWNER:
            raise USDEditError(
                editing.REFUSED,
                f"This session already holds {len(owned)} loaded stages, the limit. Unload stages it "
                "no longer needs with unload_usd_stage (list_usd_stages shows them) and load again.",
                limit=MAX_STAGES_PER_OWNER,
                loaded_stages=owned,
            )

        try:
            root = stage_files.open_private_root(file_path)
        except Exception as exc:
            # The file is parsed once, into the private root layer. Only when
            # that fails is it opened again the way validation opens it, so
            # the error names pxr's reason (ParseError with line and column,
            # CorruptFile, ...) as validate_usd_file reports it.
            open_root_layer(file_path, head)
            raise USDLoadError(
                file_path, "OpenFailed", f"pxr could not open it: {editing.tf_error_reason(exc)}"
            ) from exc
        # Held while the stage is composed, so composition finds these layers
        # in pxr's registry instead of parsing the files again.
        walked = _checked(lambda: sandbox.check_layers(root)) if sandbox is not None else []
        try:
            stage = stage_files.compose_stage(root)
        except Exception as exc:
            raise USDLoadError(
                file_path,
                "OpenFailed",
                f"pxr parsed the {root.GetFileFormat().formatId} layer but could not compose a "
                f"stage from it: {editing.tf_error_reason(exc)}",
            ) from exc
        reloaded = self._refresh_shared(stage)
        if reloaded:
            self.logger.info(f"Reread changed files for {file_path}: {reloaded}")
        if sandbox is not None:
            # Judged afresh; a stage that fails is never registered.
            _checked(lambda: sandbox.check_stage(stage))
        del walked

        stage_id = self._generate_stage_id(file_path)
        with self._registry:
            self._active_stages[stage_id] = stage
            self._bbox_caches[stage_id] = BBoxCache(stage)
            self._stage_records[stage_id] = _StageRecord(
                file_path=file_path,
                owner=owner,
                mtimes=stage_files.file_mtimes(stage),
                unreal_content=sandbox.content_paths if sandbox is not None else [],
            )
        self.logger.info(f"Loaded USD stage: {file_path} -> {stage_id}")
        return stage_id

    def unreal_content_paths(self, stage_id: str) -> List[OutsidePath]:
        """
        The Unreal content paths the sandbox let through for a loaded stage.

        Each is an asset value shaped like ``/Engine/.../Name.Name`` that
        names an editor object, not a file (see
        ``simul.usd.sandbox.is_unreal_content_path``), as Unreal's USD
        exporter writes one. Empty when the load ran without a sandbox.

        Args:
            stage_id: Stage identifier

        Returns:
            The paths, with the layer authoring each; empty for an unknown stage.
        """
        with self._registry:
            record = self._stage_records.get(stage_id)
            return list(record.unreal_content) if record is not None else []

    def _refresh_shared(self, stage: Usd.Stage) -> List[str]:
        """
        Reload the shared layers of ``stage`` whose files changed since the session read them.

        Sdf hands every stage the same layer object for a sublayer or a
        referenced file and never rereads it on its own, so an edit made to
        the file outside this process would stay invisible. A reload
        recomposes every stage using the layer, other sessions' included, so
        each changed file's new content is first read into a throwaway layer
        and held to the composition sandbox; only when every one passes are
        the layers reloaded. Afterwards each other stage using a reloaded
        layer is checked again, and one that now reaches outside the sandbox
        (a file swapped between the check and the reload) is dropped.

        Returns:
            The real paths of the layers reloaded.

        Raises:
            SandboxDenied: A changed file now reaches outside the sandbox;
                nothing was reloaded.
        """
        root = stage.GetRootLayer()
        changed: List[Any] = []
        with self._registry:
            for layer in stage_files.file_layers(stage):
                if layer == root or layer.dirty:
                    continue
                path = layer.realPath
                current = stage_files.mtime(path)
                if path not in self._shared_mtimes:
                    # First seen: this load just read it.
                    self._shared_mtimes[path] = current
                elif self._shared_mtimes[path] != current:
                    changed.append(layer)
        if not changed:
            return []
        factory = self._sandbox_factory
        refreshable = []
        for layer in changed:
            try:
                candidate = stage_files.open_candidate(layer)
            except Exception as exc:
                # Unreadable now: keep the content in use; status still shows
                # the file as changed on disk.
                self.logger.warning(f"Not rereading {layer.realPath}, which pxr cannot parse now: {exc}")
                continue
            if factory is not None:
                _checked(lambda: factory().check_layers(candidate))
            refreshable.append(layer)
        reloaded = []
        for layer in refreshable:
            if layer.Reload():
                reloaded.append(layer.realPath)
            with self._registry:
                self._shared_mtimes[layer.realPath] = stage_files.mtime(layer.realPath)
        if reloaded and factory is not None:
            self._recheck_stages_using(reloaded, factory, skip=stage)
        return reloaded

    def _recheck_stages_using(
        self, paths: List[str], factory: Callable[[], CompositionSandbox], skip: Optional[Usd.Stage] = None
    ) -> None:
        """Drop each loaded stage using one of ``paths`` that now reaches outside the sandbox."""
        reloaded = set(paths)
        with self._registry:
            candidates = [
                (stage_id, self._active_stages[stage_id])
                for stage_id, record in self._stage_records.items()
                if reloaded & set(record.mtimes) and self._active_stages[stage_id] is not skip
            ]
        for stage_id, stage in candidates:
            try:
                factory().check_stage(stage)
            except SandboxDenied as exc:
                self._drop(stage_id)
                self.logger.warning(
                    f"Dropped USD stage {stage_id}: after its files were reread it draws on files "
                    f"outside the sandbox ({exc})"
                )

    def _owned_stage(self, stage_id: str) -> Tuple[Usd.Stage, _StageRecord]:
        stage = self.require_stage(stage_id)
        with self._registry:
            return stage, self._stage_records[stage_id]

    def unload_stage(self, stage_id: str, discard_unsaved: bool = False) -> bool:
        """
        Unload a USD stage.

        Args:
            stage_id: Stage identifier
            discard_unsaved: Unload even though the stage has unsaved edits

        Returns:
            True if unloaded, False when the caller holds no such stage

        Raises:
            USDEditError: The stage has unsaved edits and ``discard_unsaved``
                is False.
        """
        stage = self.get_stage(stage_id)
        if stage is None:
            self.logger.warning(f"Stage not found: {stage_id}")
            return False
        if stage.GetRootLayer().dirty and not discard_unsaved:
            raise USDEditError(
                editing.REFUSED,
                f"Stage {stage_id} has unsaved edits. Save them with save_usd_stage or "
                "export_usd_stage, or pass discard_unsaved=true to drop them.",
                stage_id=stage_id,
                override="discard_unsaved",
            )
        self._drop(stage_id)
        self.logger.info(f"Unloaded USD stage: {stage_id}")
        return True

    def _drop(self, stage_id: str) -> None:
        with self._registry:
            self._active_stages.pop(stage_id, None)
            self._bbox_caches.pop(stage_id, None)
            self._stage_records.pop(stage_id, None)
            # Forget the shared files no remaining stage composes, so a later
            # load that reads one fresh does not take it for changed.
            in_use = {path for record in self._stage_records.values() for path in record.mtimes}
            for path in [path for path in self._shared_mtimes if path not in in_use]:
                del self._shared_mtimes[path]

    def release_owner(self, owner: Optional[str]) -> List[str]:
        """
        Unload every stage an MCP session loaded, once that session has ended.

        Nothing could reach those stages afterwards (their ids are random and
        their owner is gone), so they would otherwise stay in memory until the
        server exits. Unsaved edits are dropped with them; a warning names the
        stages that had any.

        Args:
            owner: The ended session's id

        Returns:
            The ids of the stages released
        """
        with self._registry:
            released = [stage_id for stage_id, record in self._stage_records.items() if record.owner == owner]
            dirty = [
                f"{stage_id} ({self._stage_records[stage_id].file_path})"
                for stage_id in released
                if self._active_stages[stage_id].GetRootLayer().dirty
            ]
            for stage_id in released:
                self._drop(stage_id)
        if dirty:
            self.logger.warning(
                f"MCP session {owner} ended with unsaved edits in {len(dirty)} USD stage(s), which "
                f"are dropped: {', '.join(dirty)}"
            )
        if released:
            self.logger.info(f"Released {len(released)} USD stage(s) of ended MCP session {owner}")
        return released

    def get_stage(self, stage_id: str) -> Optional[Usd.Stage]:
        """
        Get a loaded USD stage.

        Args:
            stage_id: Stage identifier

        Returns:
            USD Stage, or None if not found or loaded by another MCP session
        """
        with self._registry:
            stage = self._active_stages.get(stage_id)
            record = self._stage_records.get(stage_id)
        if stage is None:
            return None
        if record is not None and record.owner != self._owner():
            return None
        return stage

    def require_stage(self, stage_id: str) -> Usd.Stage:
        """
        Get a loaded USD stage, or raise the NotFoundError an edit reports.

        Args:
            stage_id: Stage identifier

        Returns:
            The USD Stage

        Raises:
            USDEditError: No stage is loaded under ``stage_id``.
        """
        stage = self.get_stage(stage_id)
        if stage is None:
            raise USDEditError(
                editing.NOT_FOUND,
                f"Stage not loaded: {stage_id!r}. Load the file with load_usd_file and pass the "
                "stage_id it returns.",
                stage_id=stage_id,
            )
        return stage

    def list_stages(self) -> List[str]:
        """
        List the stages the calling MCP session has loaded.

        Returns:
            List of stage identifiers
        """
        return self._owned_ids()

    def stage_file_path(self, stage_id: str) -> str:
        """
        The resolved root file of a loaded stage.

        Raises:
            USDEditError: No such stage for the caller.
        """
        return self._owned_stage(stage_id)[1].file_path

    def stage_status(self, stage_id: str) -> Dict[str, Any]:
        """
        Report a stage's file, unsaved state and the files changed on disk since it was read.

        Args:
            stage_id: Stage identifier

        Returns:
            ``stage_id``, ``file_path``, ``has_unsaved_changes`` and
            ``changed_on_disk`` (real paths)

        Raises:
            USDEditError: No such stage for the caller.
        """
        stage, record = self._owned_stage(stage_id)
        return {
            "stage_id": stage_id,
            "file_path": record.file_path,
            "has_unsaved_changes": bool(stage.GetRootLayer().dirty),
            "changed_on_disk": stage_files.changed_files(stage, record.mtimes),
        }

    def save_stage(self, stage_id: str, overwrite_external_changes: bool = False) -> Dict[str, Any]:
        """
        Save the stage's edit target (its root layer) to its own file.

        Args:
            stage_id: Stage identifier
            overwrite_external_changes: Save even though the file changed
                on disk after the stage read it

        Returns:
            ``stage_id``, ``file_path``, ``saved`` and ``files_written``
            (``path`` and ``bytes`` per file)

        Raises:
            USDEditError: The save was refused or failed.
        """
        stage, record = self._owned_stage(stage_id)
        root = stage.GetRootLayer()
        path = record.file_path
        if Path(path).suffix.lower() == ".usdz":
            raise USDEditError(
                editing.UNSUPPORTED,
                f"{path} is a .usdz package, which cannot be saved in place. Write the edits to a "
                "new file with export_usd_stage instead.",
                stage_id=stage_id,
                file_path=path,
            )
        if not root.dirty:
            return {"stage_id": stage_id, "file_path": path, "saved": False, "files_written": []}
        if path in stage_files.changed_files(stage, {path: record.mtimes.get(path)}) and not (
            overwrite_external_changes
        ):
            raise USDEditError(
                editing.REFUSED,
                f"{path} changed on disk after this stage read it; saving would overwrite those "
                "changes. Export the edits to a new file with export_usd_stage, reload the stage with "
                "reload_usd_stage (discarding the edits), or pass overwrite_external_changes=true.",
                stage_id=stage_id,
                file_path=path,
                override="overwrite_external_changes",
            )
        try:
            saved = root.Save()
        except Exception as exc:
            raise USDEditError(
                SAVE_FAILED, f"Saving {path} failed: {editing.tf_error_reason(exc)}", file_path=path
            ) from exc
        if not saved:
            raise USDEditError(SAVE_FAILED, f"Saving {path} failed.", file_path=path)
        record.mtimes = stage_files.file_mtimes(stage)
        self.logger.info(f"Saved USD stage {stage_id} to {path}")
        return {
            "stage_id": stage_id,
            "file_path": path,
            "saved": True,
            "files_written": [{"path": path, "bytes": os.path.getsize(path)}],
        }

    def export_stage(
        self, stage_id: str, file_path: str, flatten: bool = False, overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        Write the stage to a new file, as its root layer or flattened.

        The stage stays bound to its own file; its unsaved state does not
        change.

        Args:
            stage_id: Stage identifier
            file_path: Destination; the extension picks the format (.usda,
                .usdc, .usd or .usdz)
            flatten: Write the composed stage as one layer
            overwrite: Replace an existing file

        Returns:
            ``stage_id``, ``file_path``, ``format``, ``flatten``,
            ``files_written`` and ``rewritten_asset_paths``

        Raises:
            USDEditError: The export was refused or failed.
        """
        stage, record = self._owned_stage(stage_id)
        destination = str(Path(file_path).resolve())
        suffix = Path(destination).suffix.lower()
        if suffix not in stage_files.EXPORT_FORMATS:
            raise USDEditError(
                INVALID_FILE_PATH,
                f"Cannot export to {destination}: the extension must be one of "
                f"{', '.join(stage_files.EXPORT_FORMATS)}.",
                file_path=destination,
            )
        if destination == record.file_path:
            raise USDEditError(
                editing.REFUSED,
                f"{destination} is the stage's own file; save it with save_usd_stage instead.",
                file_path=destination,
            )
        if not os.path.isdir(os.path.dirname(destination)):
            raise USDEditError(
                INVALID_FILE_PATH,
                f"Cannot export to {destination}: the directory {os.path.dirname(destination)} does not exist.",
                file_path=destination,
            )
        if os.path.exists(destination) and not overwrite:
            raise USDEditError(
                editing.REFUSED,
                f"Refusing to overwrite existing file {destination}. Pass overwrite=true to replace it.",
                file_path=destination,
                override="overwrite",
            )
        try:
            result = stage_files.export_stage(stage, destination, flatten=flatten)
        except Exception as exc:
            raise USDEditError(
                SAVE_FAILED,
                f"Exporting {stage_id} to {destination} failed: {editing.tf_error_reason(exc)}",
                file_path=destination,
            ) from exc
        self.logger.info(f"Exported USD stage {stage_id} to {destination} (flatten={flatten})")
        return {
            "stage_id": stage_id,
            "file_path": destination,
            "format": suffix.lstrip("."),
            "flatten": flatten,
            "files_written": [{"path": destination, "bytes": os.path.getsize(destination)}],
            "rewritten_asset_paths": result["rewritten_asset_paths"],
        }

    def reload_stage(self, stage_id: str, discard_unsaved: bool = False) -> Dict[str, Any]:
        """
        Reread the stage from disk: its root file and every composed file that changed.

        Args:
            stage_id: Stage identifier
            discard_unsaved: Reload even though the stage has unsaved edits,
                which are lost

        Returns:
            ``stage_id``, ``file_path``, ``discarded_unsaved`` and
            ``reloaded_files``

        A changed root file is read into a throwaway layer and held to the
        composition sandbox before the stage's root is reloaded, and changed
        shared files are checked the same way before they are reloaded in
        place (see ``_refresh_shared``). A file that now reaches outside the
        sandbox is not reloaded, so the stage keeps its content. After the
        reload the stage is checked again as a whole; if it reaches outside
        the sandbox anyway (a file swapped between the check and the reload),
        it is unloaded.

        Raises:
            USDEditError: The stage has unsaved edits and ``discard_unsaved``
                is False, or the reload failed.
            SandboxDenied: A changed file now reaches outside the sandbox.
        """
        stage, record = self._owned_stage(stage_id)
        root = stage.GetRootLayer()
        dirty = bool(root.dirty)
        if dirty and not discard_unsaved:
            raise USDEditError(
                editing.REFUSED,
                f"Stage {stage_id} has unsaved edits, which a reload would discard. Save or export "
                "them first, or pass discard_unsaved=true.",
                stage_id=stage_id,
                override="discard_unsaved",
            )
        changed = stage_files.changed_files(stage, record.mtimes)
        factory = self._sandbox_factory
        if factory is not None and record.file_path in changed:
            try:
                candidate = stage_files.open_candidate(root)
            except Exception as exc:
                raise USDEditError(
                    RELOAD_FAILED,
                    f"Reloading {record.file_path} failed: {editing.tf_error_reason(exc)}",
                    file_path=record.file_path,
                ) from exc
            _checked(lambda: factory().check_layers(candidate))
        try:
            root.Reload(True)
            reloaded = self._refresh_shared(stage)
        except SandboxDenied:
            raise
        except Exception as exc:
            raise USDEditError(
                RELOAD_FAILED,
                f"Reloading {record.file_path} failed: {editing.tf_error_reason(exc)}",
                file_path=record.file_path,
            ) from exc
        self._invalidate_bbox_cache(stage_id)
        if factory is not None:
            checker = factory()
            try:
                _checked(lambda: checker.check_stage(stage))
            except SandboxDenied:
                self._drop(stage_id)
                self.logger.warning(
                    f"Unloaded USD stage {stage_id}: after the reload it draws on files outside the sandbox"
                )
                raise
        with self._registry:
            record.mtimes = stage_files.file_mtimes(stage)
            if factory is not None:
                record.unreal_content = checker.content_paths
        reloaded_files = sorted(set(changed) | set(reloaded) | {record.file_path})
        self.logger.info(f"Reloaded USD stage {stage_id} (discarded unsaved edits: {dirty})")
        return {
            "stage_id": stage_id,
            "file_path": record.file_path,
            "discarded_unsaved": dirty,
            "reloaded_files": reloaded_files,
        }

    @monitor_performance("headless_session.get_stage_info")
    def get_stage_info(self, stage_id: str) -> Optional[USDStageInfo]:
        """
        Get information about a loaded stage.

        Args:
            stage_id: Stage identifier

        Returns:
            USDStageInfo or None if stage not found
        """
        stage = self.get_stage(stage_id)
        if stage:
            return self.usd_reader.get_stage_info(stage)
        return None

    @monitor_performance("headless_session.get_prim_info")
    def get_prim_info(self, stage_id: str, prim_path: str) -> Optional[USDPrimInfo]:
        """
        Get information about a prim in a stage.

        Args:
            stage_id: Stage identifier
            prim_path: Path to the prim

        Returns:
            USDPrimInfo or None if not found
        """
        stage = self.get_stage(stage_id)
        if stage:
            prim = stage.GetPrimAtPath(prim_path)
            if prim and prim.IsValid():
                return self.usd_reader.get_prim_info(prim)
        return None

    def get_prim_transform(self, stage_id: str, prim_path: str) -> Optional[Dict[str, List[float]]]:
        """Get local transform for a prim."""
        stage = self.get_stage(stage_id)
        if not stage:
            return None

        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Xformable):
            return None

        xformable = UsdGeom.Xformable(prim)
        matrix = xformable.GetLocalTransformation()
        transform = Gf.Transform(matrix)

        translation = transform.GetTranslation()
        rotation = transform.GetRotation().GetQuat()
        imaginary = rotation.GetImaginary()
        scale = transform.GetScale()

        return {
            "translation": [translation[0], translation[1], translation[2]],
            "rotation": [rotation.GetReal(), imaginary[0], imaginary[1], imaginary[2]],
            "scale": [scale[0], scale[1], scale[2]],
        }

    def get_children_type_counts(self, stage_id: str, prim_path: str) -> Dict[str, int]:
        """Get counts of child prim types."""
        stage = self.get_stage(stage_id)
        if not stage:
            return {}

        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return {}

        counts: Dict[str, int] = {}
        for child in prim.GetChildren():
            type_name = child.GetTypeName() or "Unknown"
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts

    def get_material_bindings(self, stage_id: str, prim_path: str) -> List[str]:
        """Get bound material paths for a prim."""
        stage = self.get_stage(stage_id)
        if not stage:
            return []

        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return []

        binding_api = UsdShade.MaterialBindingAPI(prim)
        material, _ = binding_api.ComputeBoundMaterial()
        if material:
            return [str(material.GetPath())]
        return []

    def _invalidate_bbox_cache(self, stage_id: str) -> None:
        """Drop cached bounds for a stage after its scene description changed."""
        bbox_cache = self._bbox_caches.get(stage_id)
        if bbox_cache is not None:
            bbox_cache.clear_cache()

    def create_prim(
        self,
        stage_id: str,
        prim_path: str,
        prim_type: str,
        attributes: Optional[Mapping[str, Any]] = None,
        redefine_existing: bool = False,
        attribute_types: Optional[Mapping[str, str]] = None,
        time_code: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Define a prim and set its initial attributes, all or nothing.

        Args:
            stage_id: Stage identifier
            prim_path: Absolute path of the new prim
            prim_type: A concrete prim type the USD schema registry knows
            attributes: Initial attribute values, relationship targets and
                the active / instanceable / kind metadata
            redefine_existing: Allow re-typing a prim that already exists
            attribute_types: USD value type per new attribute name
            time_code: Write the attribute values as samples at this time

        Returns:
            Summary of the change (see ``editing.create_prim``)

        Raises:
            USDEditError: The edit was refused or failed; nothing changed.
        """
        stage = self.require_stage(stage_id)
        try:
            return editing.create_prim(
                stage,
                prim_path,
                prim_type,
                attributes,
                redefine_existing=redefine_existing,
                attribute_types=attribute_types,
                time_code=time_code,
            )
        finally:
            self._invalidate_bbox_cache(stage_id)

    def update_prim_attributes(
        self,
        stage_id: str,
        prim_path: str,
        attributes: Mapping[str, Any],
        attribute_types: Optional[Mapping[str, str]] = None,
        time_code: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Set attributes, relationship targets and metadata on a prim, all or nothing.

        Args:
            stage_id: Stage identifier
            prim_path: Absolute path of an existing prim
            attributes: Name to JSON value
            attribute_types: USD value type per new attribute name
            time_code: Write the attribute values as samples at this time;
                without it, animated attributes are refused

        Returns:
            Summary of the change (see ``editing.update_prim``)

        Raises:
            USDEditError: The edit was refused or failed; nothing changed.
        """
        stage = self.require_stage(stage_id)
        try:
            return editing.update_prim(
                stage,
                prim_path,
                attributes,
                attribute_types=attribute_types,
                time_code=time_code,
            )
        finally:
            self._invalidate_bbox_cache(stage_id)

    def delete_prim(
        self, stage_id: str, prim_path: str, allow_root_delete: bool = False
    ) -> Dict[str, Any]:
        """
        Delete a prim and its subtree from the stage's edit target.

        Args:
            stage_id: Stage identifier
            prim_path: Absolute path of an existing prim
            allow_root_delete: Permit deleting the stage's default prim

        Returns:
            Summary of the change (see ``editing.delete_prim``)

        Raises:
            USDEditError: The delete was refused or failed; nothing changed.
        """
        stage = self.require_stage(stage_id)
        try:
            return editing.delete_prim(stage, prim_path, allow_root_delete=allow_root_delete)
        finally:
            self._invalidate_bbox_cache(stage_id)

    @monitor_performance("headless_session.summarize_stage")
    def summarize_stage(self, stage_id: str, include_meshes: bool = True) -> Optional[SceneSummary]:
        """
        Generate a summary of a loaded stage.

        Args:
            stage_id: Stage identifier
            include_meshes: Include detailed mesh information

        Returns:
            SceneSummary or None if stage not found
        """
        stage = self.get_stage(stage_id)
        if stage:
            # The file the stage was loaded from, not its private root layer's
            # identifier (which carries the session's simul_stage argument).
            with self._registry:
                record = self._stage_records.get(stage_id)
            file_path = record.file_path if record is not None else stage_files.public_identifier(
                stage.GetRootLayer().identifier
            )

            return self.summarizer.summarize_stage(
                stage,
                file_path=file_path,
                include_meshes=include_meshes
            )
        return None

    def find_prims_by_type(self, stage_id: str, prim_type: str) -> List[str]:
        """
        Find prims of a specific type in a stage.

        Args:
            stage_id: Stage identifier
            prim_type: Prim type to search for

        Returns:
            List of prim paths
        """
        stage = self.get_stage(stage_id)
        if stage:
            prims = self.usd_reader.find_prims_by_type(stage, prim_type)
            return [str(prim.GetPath()) for prim in prims]
        return []

    def find_prims_by_name(self, stage_id: str, name_pattern: str, exact_match: bool = False) -> List[str]:
        """
        Find prims by name pattern in a stage.

        Args:
            stage_id: Stage identifier
            name_pattern: Name pattern to search for
            exact_match: Use exact matching

        Returns:
            List of prim paths
        """
        stage = self.get_stage(stage_id)
        if stage:
            prims = self.usd_reader.find_prims_by_name(stage, name_pattern, exact_match)
            return [str(prim.GetPath()) for prim in prims]
        return []

    def get_prim_bbox(
        self, stage_id: str, prim_path: str, world_space: bool = True
    ) -> Optional[Dict[str, List[float]]]:
        """
        Get bounding box for a prim.

        Args:
            stage_id: Stage identifier
            prim_path: Path to the prim
            world_space: Return world-space bbox

        Returns:
            Bounding box dictionary or None
        """
        stage = self.get_stage(stage_id)
        bbox_cache = self._bbox_caches.get(stage_id)

        if stage and bbox_cache:
            prim = stage.GetPrimAtPath(prim_path)
            if prim and prim.IsValid():
                if world_space:
                    bbox = bbox_cache.compute_world_bbox(prim)
                else:
                    bbox = bbox_cache.compute_local_bbox(prim)

                if bbox:
                    from ..usd.bbox import bbox_to_dict
                    return bbox_to_dict(bbox)

        return None

    def get_stage_bbox(self, stage_id: str) -> Optional[Dict[str, List[float]]]:
        """
        Get bounding box for entire stage.

        Args:
            stage_id: Stage identifier

        Returns:
            Bounding box dictionary or None
        """
        bbox_cache = self._bbox_caches.get(stage_id) if self.get_stage(stage_id) else None
        if bbox_cache:
            bbox = bbox_cache.get_stage_bbox()
            if bbox:
                from ..usd.bbox import bbox_to_dict
                return bbox_to_dict(bbox)
        return None

    def get_mesh_info(self, stage_id: str, prim_path: str) -> Optional[Dict[str, Any]]:
        """
        Get mesh information for a mesh prim.

        Args:
            stage_id: Stage identifier
            prim_path: Path to mesh prim

        Returns:
            Mesh information dictionary or None
        """
        stage = self.get_stage(stage_id)
        if stage:
            prim = stage.GetPrimAtPath(prim_path)
            if prim and prim.IsValid() and prim.IsA(UsdGeom.Mesh):
                try:
                    mesh_info = self.mesh_ops.get_mesh_statistics(prim)
                    return {
                        "prim_path": mesh_info.prim_path,
                        "vertex_count": mesh_info.vertex_count,
                        "face_count": mesh_info.face_count,
                        "point_count": mesh_info.point_count,
                        "has_normals": mesh_info.has_normals,
                        "has_uvs": mesh_info.has_uvs,
                        "has_colors": mesh_info.has_colors,
                        "has_tangents": mesh_info.has_tangents,
                        "subdivision_scheme": mesh_info.subdivision_scheme,
                        "topology_valid": mesh_info.topology_valid,
                        "is_closed": mesh_info.is_closed,
                        "surface_area": mesh_info.surface_area,
                        "volume": mesh_info.volume,
                        "materials": mesh_info.materials,
                        "subsets": mesh_info.subsets,
                        # None for a mesh without points, which has no bounds.
                        "bbox": None if mesh_info.bbox is None else {
                            "min": list(mesh_info.bbox[0]),
                            "max": list(mesh_info.bbox[1]),
                        },
                        "consistently_oriented": mesh_info.consistently_oriented,
                    }
                except Exception as e:
                    self.logger.error(f"Error getting mesh info for {prim_path}: {e}")
        return None

    def _generate_stage_id(self, file_path: str) -> str:
        """Generate a stage identifier: the file's stem and a random, unguessable suffix."""
        return f"{Path(file_path).stem}_{secrets.token_hex(6)}"

    def cleanup(self) -> None:
        """Clean up all resources."""
        try:
            with self._registry:
                self._active_stages.clear()
                self._bbox_caches.clear()
                self._stage_records.clear()
                self._shared_mtimes.clear()
            self.usd_reader.clear_cache()
            self.logger.info("Headless USD session cleaned up")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


class HeadlessUSDAdapter(LoggerMixin):
    """
    Adapter for headless USD operations.

    Provides a high-level interface for USD operations without Isaac Sim runtime.
    One session lives for the adapter's whole lifetime so that a ``stage_id``
    handed out by ``load_stage`` stays valid for every later call; ``close``
    releases the loaded stages.
    """

    name: str = "usd"

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize headless USD adapter.

        Args:
            settings: Configuration settings
        """
        self.settings = settings or get_settings()
        self._session = HeadlessUSDSession(self.settings)
        self.logger.info("Headless USD adapter initialized")

    @contextmanager
    def create_session(self) -> Iterator[HeadlessUSDSession]:
        """
        Yield the adapter's shared headless USD session.

        Yields:
            The long-lived HeadlessUSDSession; stages loaded in one call
            remain available to the next.
        """
        yield self._session

    def close(self) -> None:
        """Release every loaded stage and cached reader state."""
        self._session.cleanup()

    def set_owner_resolver(self, resolver: Callable[[], Optional[str]]) -> None:
        """Scope stages to the MCP session ``resolver`` names for each call."""
        self._session.set_owner_resolver(resolver)

    def release_session(self, session_id: str) -> None:
        """Unload the stages of an MCP session that has ended."""
        self._session.release_owner(session_id)

    def set_sandbox_factory(self, factory: Optional[Callable[[], CompositionSandbox]]) -> None:
        """Hold the files a reload rereads to the composition sandbox ``factory`` builds."""
        self._session.set_sandbox_factory(factory)

    def is_available(self) -> bool:
        """
        Check if headless USD operations are available.

        Returns:
            True when the adapter module is importable
        """
        return True

    def get_capabilities(self) -> List[str]:
        """
        Get list of supported capabilities.

        Returns:
            List of capability names
        """
        capabilities = [
            "load_usd_files",
            "analyze_scene_structure",
            "extract_mesh_data",
            "compute_bounding_boxes",
            "generate_scene_summaries",
            "search_prims",
            "validate_topology"
        ]

        return capabilities


# Convenience functions
def create_headless_session(settings: Optional[Settings] = None) -> HeadlessUSDSession:
    """
    Create a headless USD session.

    Args:
        settings: Configuration settings

    Returns:
        HeadlessUSDSession instance
    """
    return HeadlessUSDSession(settings)


def is_headless_available() -> bool:
    """Check if headless USD operations are available."""
    return True
