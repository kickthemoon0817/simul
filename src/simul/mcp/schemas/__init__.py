"""MCP schema package grouped by backend/domain."""

from . import blender as _blender
from . import common as _common
from . import instance as _instance
from . import simready as _simready
from . import unreal as _unreal
from . import usd as _usd

from .blender import *
from .common import *
from .instance import *
from .simready import *
from .unreal import *
from .usd import *

__all__ = [
    *_common.__all__,
    *_usd.__all__,
    *_instance.__all__,
    *_simready.__all__,
    *_blender.__all__,
    *_unreal.__all__,
]
