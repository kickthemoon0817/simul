"""Isaac Sim tools, assembled from per-domain mixins.

The public class and every name below are re-exported by
``simul_mcp.mcp.tools.isaac_tools`` so existing imports keep working.
"""

from ._base import IsaacScriptBase
from ._shared import (
    BULK_GEOMETRY_ATTRIBUTES,
    LOG_SCAN_WINDOW_BYTES,
    MAX_CAPTURE_DIMENSION,
    MAX_INLINE_CAPTURE_BYTES,
    MAX_RETAINED_CAPTURES,
    MAX_SCRIPT_BYTES,
    PRIM_DETAIL_ASPECTS,
    FloatList,
    _pyval,
)
from .camera import CameraMixin
from .core import CoreToolsMixin
from .diagnostics import DiagnosticsMixin
from .exploration import ExplorationMixin
from .graph import OmniGraphMixin
from .materials import MaterialsMixin
from .physics import PhysicsMixin
from .prims import PrimEditMixin
from .render import RenderMixin
from .scene import SceneInspectionMixin
from .simulation import SimulationMixin
from .stage import StageAssetMixin
from .system import SystemMixin


class IsaacTools(
    CoreToolsMixin,
    SceneInspectionMixin,
    CameraMixin,
    PrimEditMixin,
    PhysicsMixin,
    SimulationMixin,
    MaterialsMixin,
    StageAssetMixin,
    ExplorationMixin,
    SystemMixin,
    RenderMixin,
    OmniGraphMixin,
    DiagnosticsMixin,
    IsaacScriptBase,
):
    """
    Tools for Isaac Sim runtime operations via TCP socket.

    Each public method prefers a typed bridge action when one exists. If the
    action is unavailable, the method can fall back to raw script execution
    according to its routing policy.
    """
