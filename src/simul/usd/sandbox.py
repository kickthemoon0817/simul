"""Keep a USD stage's composition inside the file sandbox.

The sandbox used to check only the file a tool was given. A layer inside it
could reference ``../outside/secret.usda`` or sublayer an absolute path, and
pxr's default resolver opened the target, so ``get_prim_info`` returned
attributes from a file the policy would have refused by name.

``CompositionSandbox`` checks everything a stage draws on against the same
``PathPolicy`` as the root file:

- Before the stage is composed, ``check_layers`` walks the layers from the
  root: for each one, every asset path it authors (sublayers, references,
  payloads, value clips and clip manifests, and asset-valued attributes and
  metadata such as textures), in every variant, whether or not that variant
  is selected, with expression variables evaluated. pxr's
  ``UsdUtils.ExtractExternalReferences`` lists them. Each path is anchored to
  its layer the way composition anchors it and resolved under the resolver
  context ``Usd.Stage.Open`` binds for the root, so a search-path reference
  that pxr would find in the working directory is checked where pxr would
  find it. If the anchored or the resolved location sits outside the allowed
  roots, or the path is a URL whose scheme is not allowed, the load is
  refused with a ``SandboxDenied`` that lists every offending path. Only
  layers that pass are opened and walked further, so a refused file is never
  read.
- After composition, ``check_stage`` checks every layer the stage uses by
  then (``GetUsedLayers``) and walks any the first pass did not see. It is the
  backstop for whatever the first pass could not predict (a resolver that
  finds a layer somewhere its anchored path did not say, a file swapped on
  disk between the passes), and judges every location afresh rather than
  from the first pass's cache. A file read this way is never reported: the
  stage is discarded with the refusal. Value clips are the exception: when a
  clip set authors a ``manifestAssetPath``, pxr opens its clip layers only on
  the first read of a time-varying value, after this check. Clips are
  therefore covered by the first pass alone, which checks every clip and
  manifest path before anything opens.

Asset paths that resolve nowhere are checked by their anchored location too.
Otherwise the difference between "refused" and "loaded with a missing
reference" would tell the caller whether a file outside the sandbox exists.
A search-path asset path that neither anchors nor resolves (``OmniPBR.mdl``)
names no location and passes. Textures and other non-layer assets are held
to the sandbox as well. The headless tools never open them, but pxr stats
every one it resolves (reading an asset-valued attribute, validation's
unresolved-dependency check), so an outside texture path would again tell
the caller which outside files exist; and one rule for every path the file
names is the one a user can predict. A file whose textures live outside the
sandbox needs ``security.allowed_paths`` widened to cover them.

Each load runs one ``PathPolicy.read_check``: a location named from many
layers (a texture every sublayer shares) is resolved and judged once, and a
shared root's trust (``/tmp/simul-work``) is walked once, not once per path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Set, Tuple

from pxr import Ar, Sdf, UsdUtils

from ..utils.paths import SandboxDenied

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pxr import Usd

    from ..utils.paths import PathPolicy

# A denial lists at most this many paths; ``outside_count`` has the total.
MAX_REPORTED_PATHS = 25

_HINT = (
    "The file is inside the sandbox, but its composition draws on files outside it. "
    "Move them under one of allowed_roots, make the asset paths relative to the file, "
    "or widen the sandbox with security.allowed_paths."
)


@dataclass(frozen=True)
class OutsidePath:
    """An asset path a layer names that sits outside the sandbox."""

    kind: str
    asset_path: str
    location: str
    layer: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "kind": self.kind,
            "asset_path": self.asset_path,
            "location": self.location,
            "layer": self.layer,
        }


def _outer_path(identifier: str) -> str:
    """Strip file format arguments and package-relative parts from a layer identifier."""
    path = Sdf.Layer.SplitIdentifier(identifier)[0] if identifier else identifier
    if path and Ar.IsPackageRelativePath(path):
        path = Ar.SplitPackageRelativePathOuter(path)[0]
    return path


def _is_layer_path(path: str) -> bool:
    """Whether pxr would open ``path`` as a layer (it has a layer file format)."""
    return _is_layer_extension(os.path.splitext(_outer_path(path))[1])


@lru_cache(maxsize=256)
def _is_layer_extension(extension: str) -> bool:
    try:
        return bool(Sdf.FileFormat.FindByExtension(extension))
    except Exception:
        return False


def _composition_arcs(layer: "Sdf.Layer") -> Set[str]:
    """The sublayer, reference and payload asset paths ``layer`` authors.

    An expression-valued one (`` `"${DIR}/a.usd"` ``) is evaluated with the
    layer's expression variables, as ``ExtractExternalReferences`` reports it.
    """
    arcs = set(layer.GetCompositionAssetDependencies())
    expression = getattr(Sdf, "VariableExpression", None)
    if expression is None:
        return arcs
    variables = dict(layer.expressionVariables or {})
    for arc in list(arcs):
        if expression.IsExpression(arc):
            result = expression(arc).Evaluate(variables)
            if isinstance(result.value, str) and not result.errors:
                arcs.add(result.value)
    return arcs


class CompositionSandbox:
    """Check the layers and assets a USD stage draws on against a path policy.

    Use one instance per load: ``check_layers`` records the layers it walked
    so ``check_stage`` walks only the ones composition added.

    Args:
        policy: The policy the root file passed. With the sandbox disabled
            every check passes.
    """

    def __init__(self, policy: "PathPolicy") -> None:
        self._policy = policy
        self._visited: Set[str] = set()
        # One memoised read check per pass: within a pass each distinct
        # location is judged once and each shared root's trust walked once.
        # check_stage starts a fresh one, so the filesystem is judged anew
        # after composition rather than remembered from the first pass.
        self._allowed_check = policy.read_check()
        self._resolved: Dict[str, str] = {}

    @property
    def enabled(self) -> bool:
        return self._policy.enabled

    @property
    def policy(self) -> "PathPolicy":
        """The path policy every location is judged by."""
        return self._policy

    def options(self) -> Dict[str, Any]:
        """The constructor arguments, to build the same sandbox in another process.

        Validation rebuilds the sandbox from these in a fresh process
        (``simul.usd._validate_fresh``); an option added to ``__init__`` must
        be added here too, or that process would not honour it.
        """
        return {"policy": self._policy}

    def check_layers(self, root_layer: "Sdf.Layer") -> List["Sdf.Layer"]:
        """Walk the layers ``root_layer`` draws on before the stage is composed.

        Args:
            root_layer: The opened root layer; its own path already passed the
                sandbox.

        Returns:
            The layers opened on the way. The caller holds them while it opens
            the stage, so composition finds them in pxr's registry rather than
            parsing them again.

        Raises:
            SandboxDenied: An asset path any of the layers authors sits outside
                the sandbox.
        """
        if not self.enabled:
            return []
        # The context Usd.Stage.Open binds for a root layer, so a path resolves
        # here exactly where composition will find it.
        context = Ar.GetResolver().CreateDefaultContextForAsset(root_layer.identifier)
        with Ar.ResolverContextBinder(context):
            held, outside = self._walk([root_layer])
        self._raise_if_outside(root_layer.identifier, outside)
        return held

    def check_stage(self, stage: "Usd.Stage") -> None:
        """Check the layers a composed stage uses by now.

        Every location is judged afresh, not from ``check_layers``'s cache,
        so a file or directory swapped on disk between the two passes is
        caught. Value clip layers that pxr opens lazily (a clip set with a
        manifest) are not open yet; ``check_layers`` already held their paths
        to the sandbox.

        Raises:
            SandboxDenied: A used layer, or an asset path one of them authors,
                sits outside the sandbox.
        """
        if not self.enabled:
            return
        # Nothing from the first pass is reused: a directory swapped for a
        # symlink out of the sandbox since then must be judged as it is now.
        self._allowed_check = self._policy.read_check()
        self._resolved = {}
        outside: List[OutsidePath] = []
        unseen: List["Sdf.Layer"] = []
        root_id = stage.GetRootLayer().identifier
        for layer in stage.GetUsedLayers(includeClipLayers=True):
            if layer.anonymous:
                continue
            location = layer.realPath or layer.identifier
            if not self._allowed(location):
                outside.append(OutsidePath("layer", layer.identifier, location, root_id))
            elif layer.identifier not in self._visited:
                unseen.append(layer)
        with Ar.ResolverContextBinder(stage.GetPathResolverContext()):
            _, found = self._walk(unseen)
        outside.extend(found)
        self._raise_if_outside(root_id, outside)

    def _walk(self, start: List["Sdf.Layer"]) -> Tuple[List["Sdf.Layer"], List[OutsidePath]]:
        """Breadth-first over ``start`` and every allowed layer they reach."""
        held: List["Sdf.Layer"] = []
        outside: List[OutsidePath] = []
        queue: List["Sdf.Layer"] = list(start)
        while queue:
            layer = queue.pop(0)
            if layer.identifier in self._visited:
                continue
            self._visited.add(layer.identifier)
            for kind, authored in self._dependencies(layer):
                verdict = self._check(layer, kind, authored)
                if isinstance(verdict, OutsidePath):
                    outside.append(verdict)
                    continue
                if verdict is None or not _is_layer_path(verdict):
                    continue
                child = self._open(verdict)
                if child is not None and child.identifier not in self._visited:
                    held.append(child)
                    queue.append(child)
        return held, outside

    @staticmethod
    def _dependencies(layer: "Sdf.Layer") -> Iterator[Tuple[str, str]]:
        """The asset paths ``layer`` authors, with the kind of each.

        Kinds: ``sublayer``, ``reference`` and ``payload`` for composition
        arcs; ``clip`` for another layer file the layer names, which is a
        value clip or clip manifest; ``asset`` for anything else, such as a
        texture.
        """
        arcs = _composition_arcs(layer)
        try:
            # Evaluates expression-variable asset paths, and covers every
            # variant, value clips and asset-valued attributes and metadata.
            sublayers, references, payloads = UsdUtils.ExtractExternalReferences(layer.identifier)
        except Exception:
            # Without pxr's full listing, still hold every composition arc
            # (sublayers, references, payloads) to the sandbox.
            sublayers = list(layer.subLayerPaths)
            references = sorted(arcs)
            payloads = []
        for path in sublayers:
            yield "sublayer", path
        for path in payloads:
            yield "payload", path
        for path in references:
            if path in arcs:
                yield "reference", path
            else:
                yield ("clip" if _is_layer_path(path) else "asset"), path

    def _check(self, layer: "Sdf.Layer", kind: str, authored: str) -> Any:
        """Return an ``OutsidePath``, the identifier to follow, or None for nothing to check."""
        if not authored:
            return None
        scheme = self._policy.url_scheme(authored)
        if scheme is not None and scheme != "file":
            if self._allowed_check(authored):
                return None
            return OutsidePath(kind, authored, authored, layer.identifier)
        try:
            identifier = layer.ComputeAbsolutePath(authored)
        except Exception:
            identifier = authored
        resolved = self._resolve(identifier)
        anchored = _outer_path(identifier)
        locations = [_outer_path(resolved)] if resolved else []
        if anchored and os.path.isabs(anchored) and anchored not in locations:
            locations.append(anchored)
        for location in locations:
            if not self._allowed(location):
                return OutsidePath(kind, authored, location, layer.identifier)
        return identifier if (resolved or os.path.isabs(anchored)) else None

    def _resolve(self, identifier: str) -> str:
        """``Ar.Resolve`` once per identifier for the load (many layers name one texture)."""
        resolved = self._resolved.get(identifier)
        if resolved is None:
            try:
                resolved = str(Ar.GetResolver().Resolve(identifier))
            except Exception:
                resolved = ""
            self._resolved[identifier] = resolved
        return resolved

    def _allowed(self, location: str) -> bool:
        return self._allowed_check(_outer_path(location))

    @staticmethod
    def _open(identifier: str) -> Optional["Sdf.Layer"]:
        """Open an allowed layer to walk it; a broken one is left to composition to report."""
        try:
            return Sdf.Layer.FindOrOpen(identifier)
        except Exception:
            return None

    def _raise_if_outside(self, root: str, outside: List[OutsidePath]) -> None:
        if not outside:
            return
        unique: Dict[Tuple[str, str], OutsidePath] = {}
        for entry in outside:
            unique.setdefault((entry.location, entry.asset_path), entry)
        entries = list(unique.values())
        root_path = _outer_path(root)
        details = self._policy.denial_details(root_path)
        details["outside_sandbox"] = [entry.to_dict() for entry in entries[:MAX_REPORTED_PATHS]]
        details["outside_count"] = len(entries)
        details["hint"] = _HINT
        shown = ", ".join(entry.location for entry in entries[:3])
        more = f" and {len(entries) - 3} more" if len(entries) > 3 else ""
        raise SandboxDenied(
            root_path,
            details,
            error_message=(
                f"USD file {root_path} draws on {len(entries)} file(s) outside the "
                f"sandbox: {shown}{more}"
            ),
        )


__all__ = ["CompositionSandbox", "MAX_REPORTED_PATHS", "OutsidePath"]
