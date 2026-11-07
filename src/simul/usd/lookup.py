"""Find the prim a tool request names, or say precisely why there is none.

The read tools take a ``prim_path`` string from the caller. A relative path, a
trailing slash or a property path used to fall through to
``Stage.GetPrimAtPath`` and come back as a generic failure; these helpers tell
a malformed path (``ValidationError``) apart from a prim that does not exist
(``NotFoundError``). The exceptions carry the ``error_type`` the tool envelope
reports.
"""

from __future__ import annotations

from ._pxr import Sdf, Usd


class USDLookupError(LookupError):
    """A stage or prim named by a request does not exist."""

    error_type = "NotFoundError"


class USDPathError(ValueError):
    """A prim path that is not an absolute USD prim path."""

    error_type = "ValidationError"


def parse_prim_path(prim_path: str) -> Sdf.Path:
    """Parse ``prim_path`` as an absolute prim path.

    Args:
        prim_path: Path such as ``/World/Robot``; ``/`` is the pseudo-root.

    Returns:
        The parsed path.

    Raises:
        USDPathError: The path is malformed, relative, or names a property or
            variant selection.
    """
    checked = Sdf.Path.IsValidPathString(prim_path or "")
    # pxr returns True for a valid string and (False, reason) for an invalid one.
    valid, reason = (checked, "") if isinstance(checked, bool) else (checked[0], checked[1])
    if not valid:
        raise USDPathError(f"Invalid prim path {prim_path!r}: {reason}")
    path = Sdf.Path(prim_path)
    if not path.IsAbsolutePath() or not (path.IsPrimPath() or path.IsAbsoluteRootPath()):
        raise USDPathError(
            f"Invalid prim path {prim_path!r}: expected an absolute prim path such as /World/Robot"
        )
    return path


def lookup_prim(stage: Usd.Stage, prim_path: str) -> Usd.Prim:
    """Return the prim at ``prim_path``.

    Instance proxies (prims under an instanceable prim) are found too.

    Args:
        stage: Stage to look in.
        prim_path: Absolute prim path; ``/`` is the pseudo-root.

    Returns:
        The valid prim.

    Raises:
        USDPathError: The path is not an absolute prim path.
        USDLookupError: No prim exists at the path.
    """
    prim = stage.GetPrimAtPath(parse_prim_path(prim_path))
    if not prim or not prim.IsValid():
        raise USDLookupError(f"Prim not found: {prim_path}")
    return prim


__all__ = ["USDLookupError", "USDPathError", "lookup_prim", "parse_prim_path"]
