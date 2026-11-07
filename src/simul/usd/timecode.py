"""Time codes for headless USD reads.

Tools take ``time_code`` as an optional number; ``None`` means USD's default
time, the value authored without a time sample. These helpers turn that into
a ``Usd.TimeCode`` and back, so every reader resolves it the same way.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from ._pxr import Usd


def resolve_time_code(time_code: Optional[Any] = None) -> Usd.TimeCode:
    """Return the ``Usd.TimeCode`` a read should use.

    Args:
        time_code: ``None`` for the default time, a number for that time
            code, or a ``Usd.TimeCode`` (returned as is).

    Returns:
        The time code.

    Raises:
        ValueError: For a NaN or infinite number.
    """
    if time_code is None:
        return Usd.TimeCode.Default()
    if isinstance(time_code, Usd.TimeCode):
        return time_code
    value = float(time_code)
    if not math.isfinite(value):
        raise ValueError(f"time_code must be a finite number, got {time_code!r}")
    return Usd.TimeCode(value)


def time_code_value(time_code: Usd.TimeCode) -> Optional[float]:
    """Return ``time_code`` as the number a response reports, None for the default time."""
    if time_code.IsDefault():
        return None
    return float(time_code.GetValue())


__all__ = ["resolve_time_code", "time_code_value"]
