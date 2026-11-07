"""
Unreal Engine runtime adapter for Simul.

This module provides an adapter for Unreal Engine operations through the
Remote Control API (HTTP on port 30010) or the embedded ``unreal`` Python module.
"""

import asyncio
import base64
import errno
import hashlib
import importlib
import json
import logging
import math
import re
import socket
import tempfile
import textwrap
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ..config import Settings, get_settings
from ..logging import LoggerMixin, get_logger
from ..utils.paths import PathPolicy
from ..utils.regex_guard import (
    exponential_regex_construct as _exponential_regex_construct,
)
from ._unreal_errors import (
    CONNECTION_ERROR,
    NOT_FOUND_ERROR,
    TIMEOUT_ERROR,
    UnrealConnectionError,
    UnrealError,
    UnrealTimeoutError,
    remote_control_error,
    remote_control_error_type,
    remote_control_message,
    script_not_found,
)
from ._unreal_procedural import new_generation, plan_scene
from ._unreal_scripts import (
    ACTOR_HELPERS,
    ADD_COMPONENT,
    ASSIGN_MATERIAL,
    CREATE_MATERIAL_INSTANCE,
    EDITOR_WORLD,
    ENABLE_PHYSICS,
    ENGINE_METADATA,
    FIND_ACTORS_BY_TAG,
    FOCUS_ACTOR,
    INCOMPATIBLE_BODY,
    INTERCHANGE_INFO,
    LOADED_MAP,
    MATERIAL_INFO,
    PIE_ACTIVE,
    PLACE_BASIC_SHAPES,
    ROBOTICS_SCENE_ANALYSIS,
    SCENE_SUMMARY,
    SET_ACTOR_PARENT,
    SET_ACTOR_TRANSFORM,
    SET_COLLISION,
    SET_LIGHT_PARAMS,
    SET_MATERIAL_PARAMS,
    SET_PHYSICS_PARAMS,
    SIMULATION_STATUS,
    SPAWN_ACTOR,
    USD_EXPORT,
    USD_IMPORT,
    pie_refusal,
)

#: ``ExecutionMode`` values UE's ``PythonScriptLibrary.ExecutePythonCommandEx``
#: accepts. Shared by the ``execute_unreal_script`` tool and ``simul unreal exec``.
UNREAL_EXEC_MODES = frozenset({"ExecuteFile", "EvaluateStatement", "ExecuteStatement"})

#: Analyses ``analyze_scene_for_robotics`` runs, all of them by default.
ROBOTICS_ANALYSES: Tuple[str, ...] = ("traversability", "graspability", "collision_complexity")

#: What ``analyze_scene_for_robotics`` classifies by, applied to the bounding
#: box of an actor's collision-enabled components. A traversable surface is at
#: least ``surface_min_area_m2`` in plan, no thicker than
#: ``surface_max_thickness_ratio`` of its shorter side and tilted at most
#: ``surface_max_tilt_degrees``; a graspable object fits in a cube
#: ``grasp_max_size_cm`` on a side.
ROBOTICS_CRITERIA: Dict[str, float] = {
    "surface_min_area_m2": 1.0,
    "surface_max_thickness_ratio": 0.25,
    "surface_max_tilt_degrees": 15.0,
    "grasp_max_size_cm": 60.0,
}

#: Most actor paths the collision summary lists; its counts cover every actor.
ROBOTICS_LISTED_LIMIT = 50

#: ``set_physics_params`` mass range, in kg. Unreal clamps a lighter override
#: to 0.001 kg (FBodyInstance::MassInKgOverride's ClampMin); a body heavier than
#: the maximum swamps the solver against kg-scale bodies, and 1e40 kg overflowed
#: Unreal's 32-bit float to inf.
PHYSICS_MIN_MASS_KG = 0.001
PHYSICS_MAX_MASS_KG = 1e9

#: Most linear or angular damping ``set_physics_params`` takes. Unreal scales a
#: velocity by 1 / (1 + dt * damping) each step, so a body this damped stops
#: within one 120 Hz step, as it would with any larger damping.
PHYSICS_MAX_DAMPING = 1e6

#: Most assets ``search_assets`` returns per call. Remote Control's asset
#: search sizes its result array to the limit it is sent, so a negative limit
#: crashed the editor ("Trying to resize TArray to an invalid size of
#: 4294967295", UE 5.7.4).
MAX_ASSET_SEARCH_RESULTS = 1000

#: The content root of the project. Assets elsewhere (``/Engine``, a plugin's
#: mount point) are shared by every project using that engine or plugin, so the
#: material tools write there only when told to (``allow_engine_content``).
PROJECT_CONTENT_ROOT = "/Game"

#: Where ``create_material_instance`` files an instance of a parent outside
#: ``/Game`` when no ``save_path`` is given.
DEFAULT_MATERIAL_FOLDER = "/Game/Materials"

# Characters Unreal does not allow in a folder name (INVALID_LONGPACKAGE_CHARACTERS,
# UE 5.7); an object name's are _INVALID_OBJECT_NAME_CHARACTERS, below.
_INVALID_FOLDER_CHARACTERS = frozenset("\\:*?\"<>|' ,.&!~\n\r\t@#")

# UE's FMD5 hex output is lowercase. Accept either case from configured
# input and normalize. Anything that isn't a clean 32-char hex digest is
# treated as plaintext and MD5-hashed (UE's FMD5::HashAnsiString).
_HEX32_RE = re.compile(r"^[0-9a-fA-F]{32}$")


def _passphrase_to_md5(value: Optional[str]) -> Optional[str]:
    """Return the lowercase MD5 hex UE expects in the Passphrase header.

    Pass-through for already-hashed inputs (case-folded), MD5-hash for
    plaintext. None in → None out (no header injection). Non-ASCII
    plaintext raises UnicodeEncodeError to mirror UE's
    FMD5::HashAnsiString narrowing — silently mismatching hashes are a
    worse failure mode than a clean error.
    """
    if value is None:
        return None
    if _HEX32_RE.match(value):
        return value.lower()
    return hashlib.md5(value.encode("ascii", errors="strict")).hexdigest()


try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

try:
    importlib.import_module("unreal")

    UNREAL_EMBEDDED_AVAILABLE = True
except ImportError:
    UNREAL_EMBEDDED_AVAILABLE = False

UNREAL_AVAILABLE = AIOHTTP_AVAILABLE or UNREAL_EMBEDDED_AVAILABLE


logger = get_logger(__name__)

# Largest capture returned as inline base64. A 1080p screenshot is typically
# several megabytes once encoded, which overruns a client's per-result budget;
# above this the caller gets the path on the editor host instead. Kept local to
# the adapter so this layer does not depend on the MCP layer.
MAX_INLINE_CAPTURE_BYTES = 262_144
# Editor-relative screenshot directories HighResShot writes into, per platform.
_SCREENSHOT_SUBDIRS = ("MacEditor", "WindowsEditor", "LinuxEditor", "Mac", "Windows", "Linux")
# Interval between screenshot polls; each poll is one cheap Remote Control call.
_CAPTURE_POLL_SECONDS = 0.1
# Bytes per capture_to_file transfer call. Measured on UE 5.7: Remote Control
# returned 16 MiB script output intact; 1-2 MiB chunks had the best
# throughput (~115 ms per MiB vs ~255 ms per MiB at 128 KiB).
_CAPTURE_CHUNK_BYTES = 2 * 1024 * 1024


#: Prints whether Play In Editor runs, as ``{"pie_active": bool}``.
PIE_STATE = EDITOR_WORLD + "print(json.dumps({'pie_active': pie_active()}))\n"

#: ``focus_on_actor``'s warning when Play In Editor left the actor unselected.
FOCUS_PIE_WARNING = (
    "Play In Editor is running, and Unreal's editor cannot select actors until it stops, so "
    "the actor was not selected; the camera was moved. To select it, stop PIE with "
    "control_unreal_simulation, action 'stop' (CLI: simul unreal sim stop), and focus again."
)

#: ``focus_on_actor``'s warning when Unreal did not select the actor outside PIE.
#: The causes are UE 5.7's: ``UUnrealEdEngine::SelectActor`` skips an actor
#: with a root selection parent (a ChildActorComponent's child actor, an actor
#: attached inside a level instance), and the level editor's
#: ``CanSelectActorElement`` refuses hidden actors, hidden or locked levels,
#: non-selectable actors, a selection lock and selections the editor mode
#: disallows.
FOCUS_NOT_SELECTED_WARNING = (
    "Unreal did not select the actor. It does not select an actor hidden in the editor or in a "
    "hidden or locked level, a child actor or other actor it selects only through a parent (an "
    "actor inside a level instance, for example), or any actor while the active editor mode or "
    "a selection lock prevents selection"
)

# focus_on_actor's auto-fit (distance 0) fits the actor in a view of a given
# field of view and shape. Editor Python cannot read the level viewport's field
# of view: UE 5.7 keeps each level viewport's FOVAngle in
# ULevelEditorViewportSettings.PerInstanceSettings, a config-only property the
# Python API refuses to read (the class is not even exported to Python), and no
# Blueprint-callable function returns a viewport's field of view or size. So
# the caller passes them, and the defaults are UE's default field of view, 90
# degrees across (EditorViewportDefs::DefaultPerspectiveFOVAngle, held across
# the width by the default AspectRatio_MaintainXFOV), and the 16:9 of the
# default capture, 1920 x 1080, whose vertical field of view (about 58.7
# degrees) is then the narrower one. A capture renders the viewport's view at
# its own size, keeping the field of view across the width, so the capture's
# shape counts, not the on-screen viewport's (about 1305 x 1265 on the headless
# editor, UE 5.7.4).
#
# The editor's own Focus (the F key, the CAMERA ALIGN ACTIVEVIEWPORTONLY exec
# command, which editor Python can run) reads the viewport's field of view and
# size but fits the on-screen viewport alone: on the headless editor it stands
# about 1.03 bounding radii from the actor's center, and a 16:9 capture of a
# 1 m cube then shows none of its edges. It also frames whatever is selected
# and eases the camera there over a quarter of a second.
FOCUS_FOV_DEGREES = 90.0
FOCUS_ASPECT_RATIO = 16.0 / 9.0

#: Narrowest field of view the auto-fit takes, in degrees: Unreal renders a
#: narrower one this wide (FEditorViewportClient::CalcSceneView clamps the
#: view's field of view to at least 0.001 degrees, UE 5.7).
FOCUS_MIN_FOV_DEGREES = 0.001

#: Widest aspect ratio the auto-fit takes, whose inverse is the narrowest: a
#: capture 16384 pixels by 1, the largest capture_viewport takes. Within these
#: and FOCUS_MIN_FOV_DEGREES the fitted distance is finite for any bounds.
FOCUS_MAX_ASPECT_RATIO = 16384.0

#: The auto-fit camera stands this many times as far from the actor's bounding
#: sphere as it would to have the sphere touch the top and bottom of the view.
FOCUS_FIT_MARGIN = 1.1

#: Smallest bounding-sphere radius the auto-fit frames, in cm: the minimum the
#: editor's own Focus uses (FEditorViewportClient::MinimumFocusRadius).
FOCUS_MIN_RADIUS = 10.0

#: Unreal's near clip plane, in cm (the engine's default NearClipPlane): the
#: viewport draws nothing nearer the camera. The auto-fit camera keeps the
#: actor's bounding sphere, FOCUS_FIT_MARGIN times as large, beyond it. In UE's
#: default view FOCUS_MIN_RADIUS alone keeps it that far back, but a wider
#: field of view, or a view as tall as it is wide, fits a small actor from so
#: close that the plane would cut it.
FOCUS_NEAR_CLIP = 10.0

#: ``focus_on_actor``'s distance, in cm, for an actor without bounds to fit.
FOCUS_DEFAULT_DISTANCE = 500.0

#: ``focus_on_actor``'s ``fit_warning`` when distance 0 found no bounds to fit.
FOCUS_NO_BOUNDS_WARNING = (
    "The actor has no bounds to fit (zero extent: no primitive component with a size, such as "
    "a StaticMeshActor without a mesh), so the camera was placed at the default distance of "
    "500 cm from the actor's location"
)


def focus_view_error(fov: float, aspect_ratio: float) -> Optional[str]:
    """Why ``focus_on_actor`` cannot fit an actor in a view of this field of view and shape.

    Args:
        fov: The view's field of view across its width, in degrees.
        aspect_ratio: The view's width over its height.

    Returns:
        The message for a field of view under FOCUS_MIN_FOV_DEGREES or not
        under 180 degrees, or an aspect ratio outside 1 / FOCUS_MAX_ASPECT_RATIO
        to FOCUS_MAX_ASPECT_RATIO, or None when both are usable.
    """
    if not FOCUS_MIN_FOV_DEGREES <= fov < 180.0:
        return f"fov must be at least {FOCUS_MIN_FOV_DEGREES:g} and less than 180 degrees, not {fov}"
    if not 1.0 / FOCUS_MAX_ASPECT_RATIO <= aspect_ratio <= FOCUS_MAX_ASPECT_RATIO:
        return (
            f"aspect_ratio must be a width / height from 1/{FOCUS_MAX_ASPECT_RATIO:g} to "
            f"{FOCUS_MAX_ASPECT_RATIO:g}, not {aspect_ratio}"
        )
    return None


def _focus_fit_distance(
    extent: Sequence[float],
    fov: float = FOCUS_FOV_DEGREES,
    aspect_ratio: float = FOCUS_ASPECT_RATIO,
) -> float:
    """
    Return the ``focus_on_actor`` distance that fits a bounding box in the view.

    The camera looks at the box's center from behind and above, as
    ``focus_on_actor`` places it, and stands FOCUS_FIT_MARGIN times as far
    from the box's bounding sphere as the sphere needs to fill the narrower
    side of the view. The view keeps ``fov`` across its width, as Unreal's
    viewports do, so a view wider than tall is narrower up and down. A sphere
    smaller than FOCUS_MIN_RADIUS counts as that size, as in the editor's own
    Focus. The camera also stands back far enough for the sphere, made
    FOCUS_FIT_MARGIN times as large, to stay beyond the FOCUS_NEAR_CLIP
    plane, which only a small actor in a wide or square view needs.

    Args:
        extent: The box's half-size along X, Y and Z, in cm.
        fov: The view's field of view across its width, in degrees.
        aspect_ratio: The view's width over its height.

    Returns:
        How far behind the box's center the camera stands, in cm. It also
        stands half that above the center, so its distance from the center is
        sqrt(1.25) times this.
    """
    radius = max(math.hypot(*extent), FOCUS_MIN_RADIUS)
    half_width = math.radians(fov) / 2
    half_angle = min(half_width, math.atan(math.tan(half_width) / aspect_ratio))
    from_center = max(
        radius * FOCUS_FIT_MARGIN / math.sin(half_angle),
        radius * FOCUS_FIT_MARGIN + FOCUS_NEAR_CLIP,
    )
    return from_center / math.hypot(1.0, 0.5)


def _outside_pie(script: str, action: str) -> str:
    """``script``, run only while Play In Editor is stopped.

    While PIE runs the wrapped script prints the PIEActive refusal for
    ``action`` instead (``_unreal_scripts.pie_refusal``). For scripts whose
    editor calls Unreal refuses during PIE, such as EditorActorSubsystem
    spawns, which return None then. ``script`` holds no multi-line string
    literal, which the indentation would change.

    Args:
        script: Editor Python, run at the top level.
        action: What PIE prevents, completing "cannot ...": "spawn actors".

    Returns:
        The wrapped script.
    """
    return (
        EDITOR_WORLD
        + "if pie_active():\n"
        + f"    print(json.dumps(pie_refusal({action!r})))\n"
        + "else:\n"
        + textwrap.indent(script, "    ")
    )


#: What ``ping``, ``health_check`` and discovery report when Remote Control
#: answers HTTP 401: the editor enforces a passphrase and the request carried
#: none or a wrong one (UE 5.7 checks loopback clients too). Remote Control
#: answered, so the editor is reachable; this is an authentication error.
PASSPHRASE_REFUSED_ERROR = (
    "Remote Control refused the request (HTTP 401): set UNREAL__PASSPHRASE "
    "to the editor's passphrase"
)

#: ``ping`` ``error_type`` for a request that never left this machine. Its
#: endpoint was not tried, so the failure says nothing about an editor there.
PROBE_NOT_SENT = "ProbeNotSent"

# Socket errors this machine raises before a request leaves it: no free file
# descriptor, buffer or local port, or local policy refused the connection.
_LOCAL_SOCKET_ERRNOS = frozenset(
    {
        errno.EMFILE,
        errno.ENFILE,
        errno.ENOBUFS,
        errno.ENOMEM,
        errno.EADDRNOTAVAIL,
        errno.EACCES,
        errno.EPERM,
    }
)


def _http_status(exc: BaseException) -> Optional[int]:
    """The HTTP status Remote Control answered a failed request with, or None."""
    if isinstance(exc, UnrealError):
        return exc.status
    if aiohttp is not None and isinstance(exc, aiohttp.ClientResponseError):
        return exc.status
    return None


def _refusal(exc: BaseException) -> Optional[Dict[str, Any]]:
    """The authentication-error fields when Remote Control answered 401, else None."""
    if _http_status(exc) == 401:
        return {
            "status": 401,
            "error": PASSPHRASE_REFUSED_ERROR,
            "error_type": "AuthenticationError",
        }
    return None


#: The Passphrase header a batch carries when no passphrase is configured
#: (``UnrealRuntimeSession._request_headers``): any non-empty value that is not
#: an MD5 digest, so it never matches a configured passphrase.
BATCH_PASSPHRASE_PLACEHOLDER = "simul-no-passphrase"


async def _response_text(resp: Any) -> str:
    """An error response's body as text; empty when it cannot be read."""
    try:
        reader = getattr(resp, "text", None)
        if reader is not None:
            return str(await reader())
        # A response double that only decodes JSON.
        return json.dumps(await resp.json())
    except Exception:
        return ""


#: Sets one console variable and reads it back; expects ``args`` = {"name",
#: "value"}. UE's GetConsoleVariableStringValue returns "" for a name it has
#: no variable for, so the command is sent only for a variable that exists:
#: an r.* console command is never run. (A string variable holding "" reads as
#: missing; the tool sets numbers only.) Bools read back as true or false,
#: ints with %d and floats with %g (FConsoleVariableConversionHelper, UE 5.7).
SET_CONSOLE_VARIABLE = """
import json, unreal
previous = unreal.SystemLibrary.get_console_variable_string_value(args["name"])
if previous == "":
    print(json.dumps({"known": False}))
else:
    unreal.SystemLibrary.execute_console_command(None, args["name"] + " " + args["value"])
    print(json.dumps({"known": True, "previous_value": previous,
                      "value": unreal.SystemLibrary.get_console_variable_string_value(args["name"])}))
"""


def _console_value_matches(read_back: str, requested: str) -> bool:
    """Whether a console variable read back holds the requested number.

    Floats read back with six significant digits (%g), so they match within
    that precision; ``true`` and ``false`` are 1 and 0.
    """
    words = {"true": 1.0, "false": 0.0}
    try:
        got = words.get(read_back.strip().lower())
        got = float(read_back) if got is None else got
        wanted = float(requested)
    except ValueError:
        return False
    return math.isclose(got, wanted, rel_tol=1e-5, abs_tol=1e-6)


#: The keys of one batch operation, as Remote Control names them.
_BATCH_KEYS = ("RequestId", "URL", "Verb", "Body")

#: The HTTP verbs Remote Control routes a batched request by (UE 5.7
#: RemotePayloadSerializer's ParseHttpVerb, less OPTIONS, which no simul use
#: needs).
BATCH_VERBS = ("GET", "PUT", "POST", "PATCH", "DELETE")

#: Most operations one ``batch_operations`` call sends. The editor runs them
#: back to back on its game thread, inside one HTTP request.
MAX_BATCH_OPERATIONS = 100


def _batch_requests(operations: Any) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Shape ``batch_operations``' operations as Remote Control reads them.

    A malformed batch is refused rather than forwarded: UE 5.7's batch route
    is one HTTP request the editor parses and runs on its game thread, and a
    request it was not written for is where its assertions fire.

    Args:
        operations: The caller's operations.

    Returns:
        Each operation with its keys in ``_BATCH_KEYS`` order, its Verb in
        upper case and a unique RequestId, and None; or no operations and why
        the request is malformed.
    """
    if not isinstance(operations, list) or not operations:
        return [], "operations must be a non-empty JSON array of operation objects"
    if len(operations) > MAX_BATCH_OPERATIONS:
        return [], f"a batch takes at most {MAX_BATCH_OPERATIONS} operations, got {len(operations)}"
    shaped: List[Dict[str, Any]] = []
    for index, operation in enumerate(operations):
        if not isinstance(operation, dict):
            return [], f"operation {index} must be an object with URL, Verb and an optional Body, got {operation!r}"
        fields: Dict[str, Any] = {}
        for key, value in operation.items():
            canonical = next((name for name in _BATCH_KEYS if name.lower() == str(key).lower()), None)
            if canonical is None:
                return [], f"operation {index} has an unknown key {key!r}; use RequestId, URL, Verb and Body"
            if canonical in fields:
                return [], f"operation {index} gives {canonical} twice"
            fields[canonical] = value
        request_id = fields.get("RequestId")
        if request_id is not None and (not isinstance(request_id, int) or isinstance(request_id, bool)):
            return [], f"operation {index}'s RequestId must be an integer, got {request_id!r}"
        url = fields.get("URL")
        if not isinstance(url, str) or not url.startswith("/remote/"):
            return [], f"operation {index} needs URL, a Remote Control route such as /remote/object/call"
        # Unreal's router may match routes ignoring case, so compare that way.
        if url.split("?", 1)[0].rstrip("/").lower() == "/remote/batch":
            return [], f"operation {index} is a batch: a batch cannot contain another"
        verb = fields.get("Verb")
        if not isinstance(verb, str) or verb.upper() not in BATCH_VERBS:
            return [], f"operation {index}'s Verb must be one of {', '.join(BATCH_VERBS)}, got {verb!r}"
        fields["Verb"] = verb.upper()
        if "Body" in fields and not isinstance(fields["Body"], dict):
            return [], f"operation {index}'s Body must be a JSON object"
        shaped.append(fields)
    given = [fields["RequestId"] for fields in shaped if fields.get("RequestId") is not None]
    if len(set(given)) != len(given):
        return [], "RequestId values must be unique: Remote Control matches each Body to its RequestId"
    taken = set(given)
    next_id = 1
    requests: List[Dict[str, Any]] = []
    for fields in shaped:
        request_id = fields.get("RequestId")
        if request_id is None:
            while next_id in taken:
                next_id += 1
            request_id = next_id
            taken.add(request_id)
        requests.append({"RequestId": request_id, **{key: fields[key] for key in _BATCH_KEYS[1:] if key in fields}})
    return requests, None


def _is_python_call(path: str, body: Optional[Dict[str, Any]]) -> bool:
    """Whether a request runs Python in the editor (``ExecutePythonCommandEx``)."""
    return (
        path == "/remote/object/call"
        and body is not None
        and body.get("functionName") == "ExecutePythonCommandEx"
    )


def _not_sent(exc: BaseException) -> Optional[str]:
    """Why a request failed on this machine before reaching its endpoint, or None.

    aiohttp keeps the socket error as ``ClientConnectorError.os_error``. A host
    name that does not resolve, or a socket error from this machine's limits
    or policy, means nothing was sent.
    """
    cause = getattr(exc, "os_error", None) or exc
    if isinstance(cause, socket.gaierror) or (
        isinstance(cause, OSError) and cause.errno in _LOCAL_SOCKET_ERRNOS
    ):
        return f"Probe not sent ({cause})"
    return None


# Close tasks scheduled by UnrealRuntimeSession.cleanup(). The event loop keeps
# only weak references to tasks, so a task nothing else holds can be collected
# before it finishes; each one is held here until it is done.
_PENDING_CLOSES: Set["asyncio.Task[None]"] = set()

# A transform component reads back as applied within these tolerances. Unreal
# stores transforms as doubles, so the location and scale slack only absorbs
# conversion noise. A rotation needs more: past |pitch| ~89.918 degrees
# FQuat::Rotator crosses SINGULARITY_THRESHOLD and snaps the pitch to +-90,
# which moves the orientation read back by up to ~0.115 degrees.
_TRANSFORM_RELATIVE_TOLERANCE = 1e-3
_ROTATION_TOLERANCE_DEGREES = 0.2


def _vector3(name: str, value: Any, axes: str) -> List[float]:
    """Return ``value`` as a list of three finite floats.

    Args:
        name: Parameter name, used in the error message.
        value: The caller's list or tuple.
        axes: What the three numbers are, e.g. ``"x, y, z"``.

    Raises:
        ValueError: ``value`` is not exactly three finite numbers.
    """
    if (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in value
        )
    ):
        return [float(v) for v in value]
    raise ValueError(f"{name} must be three finite numbers [{axes}], got {value!r}")


#: Characters Unreal refuses in an object name (INVALID_OBJECTNAME_CHARACTERS, UE 5.7).
_INVALID_OBJECT_NAME_CHARACTERS = "\"' ,/.:|&!~\n\r\t@#(){}[]=;^%$`"

#: Longest object name Unreal keeps: an FName holds fewer than NAME_SIZE (1024) characters.
_MAX_OBJECT_NAME_LENGTH = 1023


def _object_name_error(name: str, value: str) -> Optional[str]:
    """Why ``value`` cannot name an Unreal object, such as a component, or None.

    Args:
        name: Parameter name, used in the message.
        value: The proposed object name.
    """
    if not value:
        return f"{name} must not be empty"
    if len(value) > _MAX_OBJECT_NAME_LENGTH:
        return f"{name} is {len(value)} characters long; Unreal keeps at most {_MAX_OBJECT_NAME_LENGTH}"
    invalid = "".join(sorted({c for c in value if c in _INVALID_OBJECT_NAME_CHARACTERS}))
    if invalid:
        return f"{name} {value!r} has characters Unreal refuses in an object name: {invalid!r}"
    if value.lower() == "none":
        return f"{name} cannot be 'None', which Unreal reserves for no name"
    return None


def _rotator_quaternion(rotation: Sequence[float]) -> Tuple[float, float, float, float]:
    """Quaternion (x, y, z, w) of a [pitch, yaw, roll] rotator in degrees.

    The same composition as ``FRotator::Quaternion``, which drops whole turns
    first: converting a huge angle to radians directly loses its remainder.
    """
    pitch, yaw, roll = (math.radians(math.fmod(angle, 360.0)) / 2 for angle in rotation)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    sr, cr = math.sin(roll), math.cos(roll)
    return (
        cr * sp * sy - sr * cp * cy,
        -cr * sp * cy - sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _transform_component_applied(
    name: str, wanted: Sequence[float], got: Sequence[float]
) -> bool:
    """Whether a read-back transform component matches the value written.

    Rotations are compared as orientations: Unreal reads a rotator back
    normalised from a quaternion, so a yaw of 270 returns as -90, a pitch
    past 90 comes back with yaw and roll flipped, and a pitch within ~0.08
    degrees of +-90 comes back snapped to +-90.
    """
    if name == "rotation":
        dot = sum(a * b for a, b in zip(_rotator_quaternion(wanted), _rotator_quaternion(got)))
        return abs(dot) >= math.cos(math.radians(_ROTATION_TOLERANCE_DEGREES) / 2)
    return all(
        abs(w - g) <= _TRANSFORM_RELATIVE_TOLERANCE * max(1.0, abs(w))
        for w, g in zip(wanted, got)
    )


# Unreal stores material scalars and colors as 32-bit floats, so a value of
# larger magnitude cannot be set.
_FLOAT32_MAX = 3.4028234663852886e38

# What each kind of material parameter takes, for the request's error message.
_MATERIAL_VALUE_FORMS = {
    "scalar": "a finite number within the 32-bit float range",
    "vector": "[r, g, b] or [r, g, b, a] of finite numbers within the 32-bit float range",
    "texture": "a texture asset path",
}


def _material_parameter_values(
    scalar_params: Any, vector_params: Any, texture_params: Any
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, str]], List[str]]:
    """Check requested material parameters and shape them for the editor script.

    Args:
        scalar_params: Parameter name to number, or None.
        vector_params: Parameter name to [r, g, b] or [r, g, b, a], or None.
        texture_params: Parameter name to texture asset path, or None.

    Returns:
        The values by kind ("scalar", "vector", "texture"; a kind with nothing
        requested is left out), each color with four channels (alpha 1 when
        three are given); by kind and name, why a parameter is refused (a
        malformed value, or a name given twice in different case); and every
        problem with the request as a message, empty when there is none. A
        request with no parameter at all is wrong too.
    """

    def number(value: Any) -> bool:
        # The range check also rules out NaN and infinity, and unlike
        # math.isfinite it does not overflow on an integer too large for a
        # float (a JSON integer has no bound).
        return isinstance(value, (int, float)) and not isinstance(value, bool) and abs(value) <= _FLOAT32_MAX

    values: Dict[str, Dict[str, Any]] = {}
    refused: Dict[str, Dict[str, str]] = {}
    problems: List[str] = []

    def refuse(kind: str, name: str, reason: str) -> None:
        refused.setdefault(kind, {})[name] = reason
        problems.append(f"{kind} parameter {name!r} {reason}")

    for kind, given in (("scalar", scalar_params), ("vector", vector_params), ("texture", texture_params)):
        if given is None:
            continue
        if not isinstance(given, dict):
            problems.append(f"{kind} parameters must be a JSON object of parameter name to value, got {given!r}")
            continue
        # The editor script matches names ignoring case, as Unreal does, so
        # two spellings of one name would write one parameter twice and
        # report the first value as not taken.
        spellings: Dict[str, str] = {}
        for name, value in given.items():
            if not isinstance(name, str) or not name.strip():
                problems.append(f"{kind} parameter names must be non-empty, got {name!r}")
                continue
            first = spellings.setdefault(name.lower(), name)
            if first != name:
                refuse(
                    kind,
                    name,
                    f"is the same parameter as {first!r} (Unreal's parameter names ignore case): give it once",
                )
            elif kind == "scalar" and number(value):
                values.setdefault(kind, {})[name] = float(value)
            elif (
                kind == "vector"
                and isinstance(value, (list, tuple))
                and len(value) in (3, 4)
                and all(number(channel) for channel in value)
            ):
                values.setdefault(kind, {})[name] = [float(channel) for channel in value] + [1.0] * (4 - len(value))
            elif kind == "texture" and isinstance(value, str) and value.strip():
                values.setdefault(kind, {})[name] = value.strip()
            else:
                refuse(kind, name, f"must be {_MATERIAL_VALUE_FORMS[kind]}, got {value!r}")
    if not values and not problems:
        problems.append("No parameters given: pass at least one scalar, vector or texture parameter")
    return values, refused, problems


def _content_folder_problem(folder: str) -> Optional[str]:
    """Why ``folder`` is not a content folder path Unreal can create, or None.

    A folder is ``/<root>/<name>/...``: absolute, each name non-empty and free
    of the characters Unreal refuses in a package path. ``.`` and ``..`` are
    refused rather than resolved: Unreal does not resolve them either, and
    sanitized ``/Game/../Engine`` into a folder named ``__`` under ``/Game``.

    Args:
        folder: The folder path, without a trailing slash.

    Returns:
        The problem, or None for a usable folder.
    """
    if not folder.startswith("/"):
        return f"{folder!r} is not a content folder path such as /Game/Materials"
    names = folder[1:].split("/")
    for name in names:
        if name in ("", ".", ".."):
            return f"{folder!r} has an empty, '.' or '..' folder name; Unreal does not resolve them"
        bad = sorted(set(name) & _INVALID_FOLDER_CHARACTERS)
        if bad:
            return f"{folder!r} has characters Unreal does not allow in a folder name: {''.join(bad)!r}"
    return None


def _in_project_content(path: str) -> bool:
    """Whether a content path is under ``/Game``, the project's own content."""
    return path == PROJECT_CONTENT_ROOT or path.startswith(PROJECT_CONTENT_ROOT + "/")


def _outside_project_refusal(path: str, action: str, hint: str) -> Dict[str, Any]:
    """The SandboxError envelope of a write to content outside ``/Game``."""
    return {
        "success": False,
        "error_type": "SandboxError",
        "error": (
            f"Refused to {action} {path}: it is engine or plugin content, outside the project's "
            f"{PROJECT_CONTENT_ROOT} folder, and every project using that engine or plugin shares it"
        ),
        "details": {"content_path": path, "allowed_root": PROJECT_CONTENT_ROOT, "hint": hint},
    }


def _sanitized_object_name(name: str) -> str:
    """``name`` as Unreal names an asset after it (ObjectTools::SanitizeObjectName)."""
    return "".join("_" if char in _INVALID_OBJECT_NAME_CHARACTERS else char for char in name)


class UnrealRuntimeSession(LoggerMixin):
    """
    Unreal Engine runtime session for Remote Control API operations.

    Communicates with UE5 via HTTP REST calls to the Remote Control API
    (default ``localhost:30010``).  When running inside the UE5 Python
    interpreter the ``embedded`` flag is set and selected operations may
    use the ``unreal`` module directly.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """
        Initialize Unreal runtime session.

        Args:
            settings: Configuration settings.
        """
        if not UNREAL_AVAILABLE:
            raise ImportError(
                "Unreal runtime not available. Install aiohttp for HTTP mode "
                "or run inside the UE5 Python interpreter."
            )

        self.settings: Settings = settings or get_settings()
        cfg = self.settings.unreal
        self.host: str = cfg.host
        self.port: int = cfg.port
        self.timeout: int = cfg.timeout
        self.embedded: bool = cfg.embedded_mode or UNREAL_EMBEDDED_AVAILABLE
        self.max_actors: int = cfg.max_actors
        self.max_retries: int = cfg.max_retries
        self.retry_base_delay: float = cfg.retry_base_delay
        self.ping_timeout: float = cfg.ping_timeout

        # Hash the configured passphrase once at construction. None means no
        # header is injected — preserves the no-auth status quo for users
        # who haven't enabled passphrase enforcement on the editor side.
        self._passphrase_md5: Optional[str] = _passphrase_to_md5(cfg.passphrase)

        self._base_url: str = f"http://{self.host}:{self.port}"
        self._session: Optional[Any] = None  # aiohttp.ClientSession, lazily created
        self._attachment: Optional[Dict[str, Any]] = None
        # Enforced in the session layer so MCP registration, the CLI and tests
        # all sit above the same check.
        self._path_policy: PathPolicy = PathPolicy.from_settings(self.settings)

        self.logger.info(
            "Unreal runtime session initialized (base_url=%s, embedded=%s)",
            self._base_url,
            self.embedded,
        )

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _default_headers(self) -> Dict[str, str]:
        """Build the static headers attached to every Remote Control request.

        Always includes the JSON content-type. When a passphrase is
        configured, also injects ``Passphrase: <md5>`` — UE 5.x's
        ``WebRemoteControlInternalUtils::CheckPassphrase`` compares this
        header verbatim against the hashes in
        ``URemoteControlSettings::Passphrases``.
        """
        headers = {"Content-Type": "application/json"}
        if self._passphrase_md5:
            headers["Passphrase"] = self._passphrase_md5
        return headers

    async def _ensure_http_session(self, timeout_override: Optional[float] = None) -> Any:
        """
        Return the shared ``aiohttp.ClientSession``, creating it on first use.

        When *timeout_override* differs from the current session's timeout the
        existing session is recycled so a fresh one with the correct timeout is
        created.

        Args:
            timeout_override: Optional per-call timeout in seconds.

        Returns:
            An ``aiohttp.ClientSession`` instance.
        """
        desired_timeout = timeout_override if timeout_override is not None else self.timeout
        if self._session is not None and not self._session.closed:
            timeout_obj = getattr(self._session, "timeout", None)
            current_total = getattr(timeout_obj, "total", None) if timeout_obj is not None else None
            if current_total is not None and not math.isclose(current_total, desired_timeout, rel_tol=1e-3):
                await self._recycle_session()

        if self._session is None or self._session.closed:
            if not AIOHTTP_AVAILABLE:
                raise RuntimeError(
                    "aiohttp is required for HTTP mode but is not installed"
                )
            timeout_cfg = aiohttp.ClientTimeout(total=desired_timeout)
            self._session = aiohttp.ClientSession(
                base_url=self._base_url,
                timeout=timeout_cfg,
                headers=self._default_headers(),
            )
        return self._session

    async def _recycle_session(self) -> None:
        """Close the current HTTP session so the next call creates a fresh one.

        This drains stale CLOSE-WAIT sockets and resets the connection pool,
        which is the primary recovery mechanism when the Remote Control API
        becomes unresponsive after editor restarts or duplicate-process conflicts.
        """
        if self._session is not None and not self._session.closed:
            try:
                await self._session.close()
            except Exception as exc:
                self.logger.debug("Session close failed during recycle: %s", exc)
        self._session = None
        self.logger.debug("HTTP session recycled")

    async def _http_request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        timeout_override: Optional[float] = None,
        max_retries: Optional[int] = None,
        response_kind: str = "json",
        _skip_attachment: bool = False,
    ) -> Dict[str, Any]:
        """
        Perform an HTTP request with retry and exponential backoff.

        Read-only requests retry transient failures (connection errors, timeouts,
        5xx) with exponential backoff. Mutations are sent once because a failed
        response does not establish whether the editor applied the operation.

        Args:
            method: HTTP method (``GET``, ``PUT``, ``POST``).
            path: URL path.
            body: Optional JSON-serializable request body.
            timeout_override: Per-call timeout in seconds.
            max_retries: Override for ``self.max_retries``.

        Returns:
            Parsed JSON response as a dictionary.
        """
        if self.settings.unreal.mode == "attached" and not _skip_attachment:
            body = await self._guard_attachment(method, path, body)
        # Remote Control uses PUT even for arbitrary Python and actor mutations.
        # Once sent, a timeout cannot establish that the operation did not run.
        retry_safe = (
            method == "GET"
            or path
            in {
                "/remote/object/describe",
                "/remote/search/assets",
                "/remote/object/thumbnail",
            }
            or (
                path == "/remote/object/property"
                and (body or {}).get("access") == "READ_ACCESS"
            )
        )
        retries = (
            (max_retries if max_retries is not None else self.max_retries)
            if retry_safe
            else 0
        )
        last_exc: Optional[Exception] = None
        headers = self._request_headers(path)

        for attempt in range(retries + 1):
            try:
                session = await self._ensure_http_session(timeout_override)
                kwargs: Dict[str, Any] = {}
                if body is not None:
                    kwargs["json"] = body
                if headers:
                    kwargs["headers"] = headers

                async with getattr(session, method.lower())(path, **kwargs) as resp:
                    if resp.status >= 500 and attempt < retries:
                        self.logger.warning(
                            "HTTP %s %s returned %d, retrying (%d/%d)",
                            method,
                            path,
                            resp.status,
                            attempt + 1,
                            retries,
                        )
                        # Don't recycle session on 5xx — the connection is
                        # fine, the server just errored.  Only sleep + retry.
                        await asyncio.sleep(
                            min(self.retry_base_delay * (2**attempt), 8.0)
                        )
                        continue
                    # 4xx errors are client bugs — never retry them. Remote
                    # Control says why in the body, which raise_for_status
                    # would drop.
                    if resp.status >= 400:
                        raise remote_control_error(
                            resp.status, await _response_text(resp), method, path
                        )
                    if response_kind == "image":
                        from io import BytesIO

                        from PIL import Image

                        image_bytes = await resp.read()
                        with Image.open(BytesIO(image_bytes)) as image:
                            image_format = (image.format or "").lower()
                            if image_format not in {"png", "jpeg"}:
                                raise ValueError(
                                    f"Unsupported thumbnail format: {image_format}"
                                )
                            return {
                                "image_base64": base64.b64encode(image_bytes).decode(
                                    "ascii"
                                ),
                                "format": image_format,
                                "width": image.width,
                                "height": image.height,
                            }
                    data: Dict[str, Any] = await resp.json()
                    return data

            except Exception as exc:
                last_exc = exc
                if isinstance(exc, UnrealError):
                    # Remote Control answered; a 5xx was already retried above.
                    raise
                is_transient = isinstance(
                    exc,
                    (
                        asyncio.TimeoutError,
                        ConnectionError,
                        OSError,
                    ),
                )
                if aiohttp is not None and isinstance(exc, aiohttp.ClientError):
                    # ClientResponseError with 4xx status is a client bug, not transient
                    if (
                        hasattr(exc, "status")
                        and isinstance(exc.status, int)
                        and exc.status < 500
                    ):
                        is_transient = False
                    else:
                        is_transient = True

                if is_transient and attempt < retries:
                    delay = min(self.retry_base_delay * (2**attempt), 8.0)
                    self.logger.warning(
                        "HTTP %s %s failed (%s), recycling session and retrying "
                        "in %.1fs (%d/%d)",
                        method,
                        path,
                        exc,
                        delay,
                        attempt + 1,
                        retries,
                    )
                    await self._recycle_session()
                    await asyncio.sleep(delay)
                    continue
                raise self._transport_error(
                    exc, method, path, body, timeout_override, retry_safe
                ) from exc

        raise last_exc  # type: ignore[misc]

    def _request_headers(self, path: str) -> Dict[str, str]:
        """Headers one request adds to the session's defaults.

        UE 5.7's batch route reads the Passphrase header without checking it
        is there (``Request.Headers[PassphraseHeader].Last()``,
        FWebRemoteControlModule::HandleBatchRequest), which aborts the editor
        on a failed assertion when it is missing. Every other route looks it
        up first. Without a configured passphrase a batch therefore carries a
        placeholder: Remote Control accepts any value while passphrases are
        not enforced (WebRemoteControlInternalUtils::CheckPassphrase), and
        an editor that enforces them refuses this client with HTTP 401 before
        the batch route runs, with or without the header. The placeholder must
        not be empty, because UE's HTTP server drops a header without a value.

        Args:
            path: The request's route.

        Returns:
            ``Passphrase: <placeholder>`` for a batch sent without a configured
            passphrase (the session's default headers carry a configured one),
            otherwise nothing.
        """
        if path == "/remote/batch" and not self._passphrase_md5:
            return {"Passphrase": BATCH_PASSPHRASE_PLACEHOLDER}
        return {}

    def _transport_error(
        self,
        exc: Exception,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]],
        timeout_override: Optional[float],
        retry_safe: bool,
    ) -> Exception:
        """The error to report for a request Remote Control never answered.

        Args:
            exc: What the request raised.
            method: The request's HTTP method.
            path: The request's route.
            body: The request's body.
            timeout_override: The request's timeout, when it had its own.
            retry_safe: Whether the request only reads.

        Returns:
            An ``UnrealTimeoutError`` for a timeout, which for a request that
            changes something says the editor may still be running it; an
            ``UnrealConnectionError`` for a connection that failed or closed;
            otherwise ``exc`` itself.
        """
        details: Dict[str, Any] = {"route": path}
        if isinstance(exc, asyncio.TimeoutError):
            timeout = timeout_override if timeout_override is not None else self.timeout
            details["timeout_s"] = timeout
            if _is_python_call(path, body):
                message = (
                    f"The editor did not finish the script within {timeout}s. It may still be running "
                    "in the editor, and Remote Control cannot cancel it: check the editor's state "
                    "before running it again (UNREAL__TIMEOUT sets the limit)"
                )
            elif retry_safe:
                message = f"Remote Control did not answer {method} {path} within {timeout}s"
            else:
                message = (
                    f"Remote Control did not answer {method} {path} within {timeout}s. The editor "
                    "may still apply the request: check its state before retrying"
                )
            return UnrealTimeoutError(message, TIMEOUT_ERROR, details=details)
        connection_failed = isinstance(exc, ConnectionError) or (
            aiohttp is not None and isinstance(exc, aiohttp.ClientConnectionError)
        )
        if connection_failed:
            address = f"{self.host}:{self.port}"
            if aiohttp is not None and isinstance(exc, aiohttp.ServerDisconnectedError):
                message = (
                    f"The editor at {address} closed the connection without answering {method} {path}: "
                    "it may have crashed or be shutting down"
                )
            else:
                message = f"Cannot reach Remote Control at {address} ({exc})"
            error = UnrealConnectionError(message, CONNECTION_ERROR, details=details)
            # Kept for _not_sent, which tells a request that never left this
            # machine from an endpoint that refused it.
            error.os_error = getattr(exc, "os_error", None) or (exc if isinstance(exc, OSError) else None)
            return error
        return exc

    def _load_attachment(self) -> Dict[str, Any]:
        """Pin one attachment for this tool session; never fall back to host/port."""
        if self.settings.unreal.mode != "attached":
            raise RuntimeError(
                "Named editor control requires UNREAL__MODE=attached and simul unreal attach"
            )
        if self._attachment is None:
            from .unreal_connection import read_attachment

            self._attachment = read_attachment(
                Path(self.settings.unreal.attachment_path).expanduser()
            )
            self.host, self.port = self._attachment["host"], self._attachment["port"]
            self._base_url = f"http://{self.host}:{self.port}"
        return self._attachment["target"]

    async def _guard_attachment(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Guard scripts in their execution call; preflight legacy HTTP operations."""
        from ._unreal_attach_scripts import EDITOR_STATE

        target = self._load_attachment()
        guard = EDITOR_STATE + f"_simul_editor.verify({target!r})\n"
        is_python = (
            path == "/remote/object/call"
            and body is not None
            and body.get("objectPath")
            == "/Script/PythonScriptPlugin.Default__PythonScriptLibrary"
            and body.get("functionName") == "ExecutePythonCommandEx"
        )
        if is_python:
            parameters = dict(body.get("parameters", {}))
            code = parameters.get("PythonCommand", "")
            mode = parameters.get("ExecutionMode", "ExecuteFile")
            if mode == "EvaluateStatement":
                parameters["PythonCommand"] = f"(exec({guard!r}), eval({code!r}))[1]"
            elif mode in {"ExecuteFile", "ExecuteStatement"}:
                parameters["PythonCommand"] = f"exec({(guard + code)!r})"
            else:
                raise ValueError("Unsupported Unreal Python execution mode")
            return {**body, "parameters": parameters}
        if body and (
            body.get("functionName")
            in {"GetLevelViewportCameraInfo", "SetLevelViewportCameraInfo"}
            or (
                body.get("functionName") == "ExecuteConsoleCommand"
                and str(body.get("parameters", {}).get("Command", "")).startswith(
                    "HighResShot"
                )
            )
        ):
            guard += (
                "if str(unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).get_active_viewport_config_key()) "
                f"!= {target['viewport']!r}:\n"
                "    raise RuntimeError('This operation uses the active viewport; "
                "activate the attached viewport first')\n"
            )
        checked = await self._http_request(
            "PUT",
            "/remote/object/call",
            {
                "objectPath": "/Script/PythonScriptPlugin.Default__PythonScriptLibrary",
                "functionName": "ExecutePythonCommandEx",
                "parameters": {
                    "PythonCommand": guard + "print(json.dumps({'verified': True}))",
                    "ExecutionMode": "ExecuteFile",
                    "FileExecutionScope": "Public",
                },
            },
            _skip_attachment=True,
        )
        result = self._parse_python_json(checked)
        if result.get("error") or result.get("verified") is not True:
            raise RuntimeError(
                result.get("error", "Unreal attachment verification failed")
            )
        return body

    async def attachment_status(self) -> Dict[str, Any]:
        """Verify and inspect the saved editor selection."""
        from ._unreal_attach_scripts import EDITOR_STATE

        target = self._load_attachment()
        result = await self._execute_json_script(
            EDITOR_STATE + "print(json.dumps(_simul_editor.identity()))"
        )
        return {
            **result,
            "mode": "attached",
            "host": self.host,
            "port": self.port,
            "viewport": target["viewport"],
        }

    async def editor_identity(self) -> Dict[str, Any]:
        """Read the editor's process, project, map and level-viewport identity.

        Discovery and attachment use this. The first call installs a small
        in-memory module and map-change listener in the editor; it changes no
        scene data.

        Returns:
            ``instance_id``, ``document_id``, ``pid``, ``engine_version``,
            ``project_name``, ``project_path``, ``map_path``, ``viewports``,
            ``active_viewport`` and ``dirty_map_packages``, or a
            ``ScriptError`` envelope when the editor's Python raised.
        """
        from ._unreal_attach_scripts import EDITOR_IDENTITY

        # An editor without identity is normal while it starts up, and
        # discovery reports the reason with the instance: not an error here.
        return await self._execute_json_script(EDITOR_IDENTITY, failure_log_level=logging.DEBUG)

    async def control_ui(
        self,
        agent_control: str,
        target: Optional[str] = None,
        property_name: Optional[str] = None,
        value: Optional[List[float]] = None,
        enabled: Optional[bool] = None,
        agent_id: str = "agent",
        position: Optional[List[float]] = None,
        activity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform a bounded action in the attached map and level viewport."""
        from ._unreal_attach_scripts import CONTROL_UI, EDITOR_STATE

        actions = (
            "inspect",
            "select_actor",
            "clear_selection",
            "set_property",
            "pilot_actor",
            "eject_actor",
            "set_game_view",
            "move_cursor",
            "clear_cursor",
        )
        if (
            not isinstance(agent_id, str)
            or not 1 <= len(agent_id) <= 64
            or not agent_id.isprintable()
            or not agent_id.strip()
        ):
            raise ValueError("agent_id must be a printable label of 1 to 64 characters")
        if position is not None and (
            agent_control != "move_cursor"
            or len(position) != 2
            or not all(math.isfinite(n) and 0 <= n <= 1 for n in position)
        ):
            raise ValueError(
                "position is only valid for move_cursor: two finite coordinates in [0, 1], from bottom-left"
            )
        if activity is not None and (
            agent_control != "move_cursor"
            or not activity.isprintable()
            or len(activity) > 256
        ):
            raise ValueError(
                "activity is only valid for move_cursor: a printable label of up to 256 characters"
            )
        if agent_control not in actions:
            raise ValueError(f"Unsupported agent_control; choose from {actions}")
        if target is not None and agent_control not in {
            "select_actor",
            "set_property",
            "pilot_actor",
        }:
            raise ValueError("target is only valid for actor actions")
        if agent_control == "set_property":
            if property_name not in {"location", "rotation", "scale"}:
                raise ValueError("property_name must be location, rotation or scale")
            if (
                value is None
                or len(value) != 3
                or not all(math.isfinite(v) for v in value)
            ):
                raise ValueError(
                    "value must contain three finite numbers; rotation uses degrees"
                )
        elif property_name is not None or value is not None:
            raise ValueError("property_name and value are only valid for set_property")
        if agent_control == "set_game_view":
            if type(enabled) is not bool:
                raise ValueError("set_game_view requires enabled=true or false")
        elif enabled is not None:
            raise ValueError("enabled is only valid for set_game_view")
        attached = self._load_attachment()
        args = {
            "action": agent_control,
            "target": target,
            "property_name": property_name,
            "value": value,
            "enabled": enabled,
            "viewport": attached["viewport"],
            "actions": actions,
            "agent_id": agent_id,
            "position": position,
            "activity": activity,
        }
        return await self._execute_json_script(
            EDITOR_STATE + f"args = {args!r}\n" + CONTROL_UI
        )

    async def _http_get(
        self,
        path: str,
        timeout_override: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform an HTTP GET against the Remote Control API.

        Args:
            path: URL path (e.g. ``/remote/info``).
            timeout_override: Per-call timeout in seconds.
            max_retries: Override for ``self.max_retries``.

        Returns:
            Parsed JSON response as a dictionary.
        """
        return await self._http_request(
            "GET", path, timeout_override=timeout_override, max_retries=max_retries,
        )

    async def _http_put(
        self,
        path: str,
        body: Dict[str, Any],
        timeout_override: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform an HTTP PUT against the Remote Control API.

        Args:
            path: URL path (e.g. ``/remote/object/call``).
            body: JSON-serializable request body.
            timeout_override: Per-call timeout in seconds.
            max_retries: Override for ``self.max_retries``.

        Returns:
            Parsed JSON response as a dictionary.
        """
        return await self._http_request(
            "PUT", path, body=body,
            timeout_override=timeout_override, max_retries=max_retries,
        )

    # ------------------------------------------------------------------
    # Ping (multi-editor discovery: unreal_connection.discover_instances)
    # ------------------------------------------------------------------

    async def ping(self) -> Dict[str, Any]:
        """
        Lightweight reachability probe — short timeout, no retries.

        ``reachable`` means Remote Control answered ``GET /remote/info``. A
        refusal is an answer: on HTTP 401 the editor is reachable and the
        payload reports an authentication error. Discovery
        (``unreal_connection.probe_instance``) takes reachability from here.

        Returns:
            ``reachable``, ``address`` and ``latency_ms``, plus ``error``
            when the probe failed or was refused; ``status`` when Remote
            Control answered with an HTTP error; and ``error_type``
            ``AuthenticationError`` (HTTP 401: ``UNREAL__PASSPHRASE`` is
            missing or wrong) or ``ProbeNotSent`` (the request never left
            this machine: out of file descriptors or local ports, or the host
            name did not resolve).
        """
        address = f"{self.host}:{self.port}"
        t0 = time.monotonic()
        try:
            if self.settings.unreal.mode == "attached":
                self._load_attachment()
                address = f"{self.host}:{self.port}"
            await self._http_get(
                "/remote/info",
                timeout_override=self.ping_timeout,
                max_retries=0,
            )
        except Exception as exc:
            refusal = _refusal(exc)
            if refusal is not None:
                return {
                    "reachable": True,
                    "address": address,
                    "latency_ms": round((time.monotonic() - t0) * 1000, 1),
                    **refusal,
                }
            failure: Dict[str, Any] = {
                "reachable": False,
                "address": address,
                "latency_ms": None,
                # A timeout's text is empty.
                "error": str(exc) or f"No answer within {self.ping_timeout}s",
            }
            status = _http_status(exc)
            if status is not None:
                failure["status"] = status
            not_sent = _not_sent(exc)
            if not_sent is not None:
                failure["error"] = not_sent
                failure["error_type"] = PROBE_NOT_SENT
            return failure
        return {
            "reachable": True,
            "address": address,
            "latency_ms": round((time.monotonic() - t0) * 1000, 1),
        }

    # ------------------------------------------------------------------
    # Internal helpers — Remote Control wrappers
    # ------------------------------------------------------------------

    async def _call_function(
        self,
        object_path: str,
        function_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Call a UObject function via ``PUT /remote/object/call``.

        Args:
            object_path: Full UObject path (actor, CDO, subsystem).
            function_name: Blueprint-visible function name.
            parameters: Optional function parameters.
            timeout: Seconds to wait for the answer; None waits ``unreal.timeout``.

        Returns:
            Parsed JSON response dictionary.
        """
        body: Dict[str, Any] = {
            "objectPath": object_path,
            "functionName": function_name,
        }
        if parameters:
            body["parameters"] = parameters
        return await self._http_put("/remote/object/call", body, timeout_override=timeout)

    async def _execute_python(
        self,
        code: str,
        mode: str = "ExecuteFile",
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute Python code inside UE5 via ``PythonScriptLibrary``.

        Args:
            code: Python source code.
            mode: ``EvaluateStatement`` (single expression, returns value),
                  ``ExecuteFile`` (multi-line, captures print/log),
                  ``ExecuteStatement`` (single statement).
            timeout: Seconds to wait for the script; None waits ``unreal.timeout``.

        Returns:
            Dictionary with ``CommandResult``, ``LogOutput``, ``ReturnValue``.
        """
        parameters = {"PythonCommand": code, "ExecutionMode": mode, "FileExecutionScope": "Public"}
        library = "/Script/PythonScriptPlugin.Default__PythonScriptLibrary"
        if timeout is None:
            return await self._call_function(library, "ExecutePythonCommandEx", parameters)
        return await self._call_function(library, "ExecutePythonCommandEx", parameters, timeout=timeout)

    async def _execute_json_script(
        self,
        code: str,
        *,
        failure_log_level: int = logging.ERROR,
        expected_errors: Sequence[str] = (),
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run a fixed tool script and preserve failures as error envelopes.

        A failure whose ``error_type`` is PIEActive or one of
        ``expected_errors`` is the editor refusing the request as asked (Play
        In Editor running, a path that names nothing it can use): the
        caller's to fix, not a fault, so it is logged at INFO. So is a script
        that raised ``SimulNotFound`` (``_unreal_errors.script_not_found``),
        which is reported as ``NotFoundError`` with its message rather than
        as a ``ScriptError`` traceback.

        Args:
            code: The script.
            failure_log_level: Level any other failure is logged at; lower it
                where a failure is expected. The envelope carries the reason
                either way.
            expected_errors: Further ``error_type`` values the script reports
                for a request it refused as asked.
            timeout: Seconds to wait for the script; None waits ``unreal.timeout``.
        """
        raw = await (self._execute_python(code) if timeout is None else self._execute_python(code, timeout=timeout))
        result = self._parse_python_json(raw)
        not_found = script_not_found(str(result.get("error") or ""))
        if not_found is not None:
            self.logger.info("Unreal refused the request (%s): %s", NOT_FOUND_ERROR, not_found)
            return {"success": False, "error_type": NOT_FOUND_ERROR, "error": not_found}
        if result.get("error"):
            error_type = result.get("error_type")
            if error_type == PIE_ACTIVE or error_type in expected_errors:
                self.logger.info("Unreal refused the request (%s): %s", error_type, result["error"])
            else:
                self.logger.log(failure_log_level, "Unreal tool script failed: %s", result["error"])
            return {"success": False, "error_type": "ScriptError", **result}
        return result

    async def _pie_active(self) -> bool:
        """Whether Play In Editor runs; False when the editor cannot say.

        A probe that explains a refusal after the fact: when it fails, the
        caller reports the refusal as it was rather than this failure.
        """
        try:
            state = await self._execute_json_script(PIE_STATE, failure_log_level=logging.DEBUG)
        except Exception as exc:
            self.logger.debug("Could not read the Play In Editor state: %s", exc)
            return False
        return state.get("pie_active") is True

    def _refused_during_pie(self, action: str) -> Dict[str, Any]:
        """The PIEActive envelope for ``action``, logged as a refusal."""
        refusal = pie_refusal(action)
        self.logger.info("Unreal refused the request (%s): %s", PIE_ACTIVE, refusal["error"])
        return refusal

    async def _component_operation(
        self,
        actor_path: str,
        code: str,
        payload: Dict[str, Any],
        *,
        mesh: bool = False,
    ) -> Dict[str, Any]:
        """Resolve an unambiguous compatible component before a fixed operation."""
        cls = "MeshComponent" if mesh else "PrimitiveComponent"
        script = (
            ACTOR_HELPERS + f"\nactor = actor_at({actor_path!r})\n"
            f"component = actor_component(actor, unreal.{cls})\n"
            + code
            + f"\nprint(json.dumps({payload!r}))"
        )
        return await self._execute_json_script(script)

    async def _run_mesh_script(
        self, body: str, *, pie_action: Optional[str] = None, **actors: str
    ) -> Dict[str, Any]:
        """Run a GeometryScript body against actors resolved by path.

        Each keyword binds a script variable to ``actor_at(path)`` from
        ``ACTOR_HELPERS`` -- a lookup among loaded objects instead of a scan
        over every level actor, which never loads a map. ``body`` is the mesh
        op's block, indented one level; it is dedented here. A path that names
        no live actor of the editor world raises SimulNotFound inside the
        editor and comes back as an error envelope. With ``pie_action`` (a body that
        spawns actors), the body runs only while Play In Editor is stopped
        and the call is otherwise refused as PIEActive (``_outside_pie``).
        """
        lookups = "".join(f"{name} = actor_at({path!r})\n" for name, path in actors.items())
        operation = (
            "subsys = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)\n"
            + lookups
            + textwrap.dedent(body)
        )
        if pie_action is not None:
            operation = _outside_pie(operation, pie_action)
        return await self._execute_json_script(ACTOR_HELPERS + operation)

    @staticmethod
    def _script_execution_denied() -> Dict[str, Any]:
        return {
            "success": False,
            "error": "Arbitrary Unreal calls are disabled by SECURITY__ALLOW_SCRIPT_EXECUTION.",
            "error_type": "ScriptExecutionDisabled",
        }

    @staticmethod
    def _parse_python_json(result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract the first JSON object printed by an ``ExecuteFile`` script.

        Args:
            result: Raw dict returned by ``_execute_python``.

        Returns:
            Parsed dict from the first JSON line, or ``{"error": "..."}``
            if the script failed or produced no JSON.
        """
        if not result.get("ReturnValue", False):
            error_msg = result.get("CommandResult", "Unknown Python error")
            return {"error": str(error_msg)}
        parsed = UnrealRuntimeSession._first_json_object(result)
        if parsed is None:
            return {"error": "No JSON output from Python execution"}
        return parsed

    @staticmethod
    def _first_json_object(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return the first JSON object printed on an Info log line, or None."""
        for entry in result.get("LogOutput") or []:
            if not isinstance(entry, dict) or entry.get("Type") != "Info":
                continue
            output = str(entry.get("Output", "")).strip()
            if output.startswith("{"):
                try:
                    parsed = json.loads(output)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    return parsed
        return None

    @staticmethod
    def _python_failure(result: Dict[str, Any]) -> Optional[str]:
        """Return the error text of a failed ``ExecutePythonCommandEx`` call, else None.

        ``ReturnValue`` is False when the Python raised; UE then puts the
        traceback in ``CommandResult``, which is never a value to report.
        """
        if result.get("ReturnValue", False):
            return None
        message = str(result.get("CommandResult") or "").strip()
        if not message:
            message = "".join(
                str(entry.get("Output", ""))
                for entry in result.get("LogOutput") or []
                if isinstance(entry, dict) and entry.get("Type") == "Error"
            ).strip()
        return message or "Unknown Python error"

    @staticmethod
    def _script_error(message: str) -> Dict[str, Any]:
        return {"success": False, "error": message, "error_type": "ScriptError"}

    async def execute_script(self, code: str, mode: str = "ExecuteFile") -> Dict[str, Any]:
        """Run caller-supplied Python and report what it produced.

        ``EvaluateStatement`` returns the expression's value (UE's
        ``CommandResult``) as ``result``. The file and statement modes return
        the first JSON object the script printed; a script that ran but
        printed no JSON returns its printed ``output`` instead of failing.

        Args:
            code: Python source code.
            mode: ``ExecuteFile``, ``EvaluateStatement`` or ``ExecuteStatement``.

        Returns:
            The script's JSON object, ``result``/``output`` fields, or a
            ``ScriptError`` envelope when the Python raised.
        """
        raw = await self._execute_python(code, mode=mode)
        failure = self._python_failure(raw)
        if failure is not None:
            return self._script_error(failure)
        info = [
            str(entry.get("Output", ""))
            for entry in raw.get("LogOutput") or []
            if isinstance(entry, dict) and entry.get("Type") == "Info"
        ]
        output = "".join(info)
        if mode == "EvaluateStatement":
            return {"success": True, "result": raw.get("CommandResult"), "output": output}
        parsed = self._first_json_object(raw)
        if parsed is None:
            return {"success": True, "output": output}
        if parsed.get("error"):
            return {"success": False, "error_type": "ScriptError", **parsed}
        return parsed

    @staticmethod
    def _parse_marker_json(result: Dict[str, Any], marker: str) -> Dict[str, Any]:
        """
        Extract the JSON object printed after ``marker`` by a script.

        Returns ``{"error": ...}`` when the script failed or never printed
        the marker, and ``{}`` when the marker carried no parseable object.
        """
        if not result.get("ReturnValue", False):
            return {"error": str(result.get("CommandResult", "Unknown Python error"))}
        for entry in result.get("LogOutput", []):
            output = entry.get("Output", "")
            idx = output.find(marker)
            if idx < 0:
                continue
            try:
                parsed = json.loads(output[idx + len(marker):].strip())
            except ValueError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {"error": "Script printed no " + marker + " line"}

    # ------------------------------------------------------------------
    # Phase 0 session methods
    # ------------------------------------------------------------------

    async def health_check(self) -> Dict[str, Any]:
        """
        Check connectivity to the Remote Control API.

        ``connected`` reflects only that ``GET /remote/info`` returned
        successfully. Engine version and project name are best-effort
        metadata — they may be empty on UE versions that gate CDO function
        calls (e.g. UE 5.3 rejects ``Default__KismetSystemLibrary`` and
        ``Default__PythonScriptLibrary`` access via Remote Control with
        "cannot be accessed remotely, check remote control project
        settings"). Such failures must NOT flip ``connected`` to False —
        the setup polling loop depends on this method to know when Remote
        Control is up.

        Returns:
            Dictionary with a ``reachable`` boolean — the field every backend
            ping shares, decided as in ``ping`` — and ``connected``, its older
            name, kept for one release. When reachable, also includes
            best-effort ``engine_version``, ``project_name``, ``is_editor``,
            and per-probe ``warnings`` (any non-fatal metadata-fetch errors).
            When Remote Control refuses the request (HTTP 401) it is
            ``reachable`` but not ``connected``, with ``ping``'s
            authentication error.
        """
        try:
            await self._http_get("/remote/info")
        except Exception as exc:
            refusal = _refusal(exc)
            if refusal is not None:
                self.logger.warning("Health check: %s", refusal["error"])
                # Remote Control answered, so the editor is reachable; callers
                # still reading ``connected`` must not take it as usable.
                return {"reachable": True, "connected": False, **refusal}
            # A timeout's text is empty; say what happened, as ping does.
            error = str(exc) or f"No answer within {self.timeout}s"
            self.logger.warning("Health check connectivity failed: %s", error)
            return {
                "reachable": False,
                "connected": False,
                "error": error,
            }

        # One script for all metadata: a single Remote Control round trip
        # instead of one per field. A failed or gated script leaves the
        # fields empty and records the failure as a warning -- never as
        # error text inside a field.
        warnings: List[str] = []
        engine_version = ""
        project_name = ""
        try:
            info = self._parse_python_json(
                await self._execute_python(ENGINE_METADATA)
            )
        except Exception as exc:
            info = {"error": str(exc)}
        if info.get("error"):
            warnings.append(f"engine_version and project_name unavailable: {info['error']}")
        else:
            engine_version = str(info.get("engine_version", ""))
            project_name = str(info.get("project_name", ""))

        result: Dict[str, Any] = {
            "reachable": True,
            "connected": True,
            "engine_version": engine_version,
            "project_name": project_name,
            "is_editor": True,
        }
        if warnings:
            result["warnings"] = warnings
        return result

    async def get_engine_info(self) -> Dict[str, Any]:
        """
        Retrieve Unreal Engine runtime information.

        Returns:
            Dictionary with engine version, project name, absolute project
            directory, loaded map (the editor world's, also while Play In
            Editor runs), the platform the editor runs on
            (``UGameplayStatics::GetPlatformName``: Mac, Windows, Linux),
            editor state and ``pie_active``, or a ``ScriptError`` envelope
            when a probe raised.
        """
        info = await self._execute_json_script(ENGINE_METADATA)
        if info.get("error"):
            return info
        return {
            "engine_version": info["engine_version"],
            "project_name": info["project_name"],
            "project_dir": info["project_dir"],
            "loaded_map": info["loaded_map"],
            "is_editor": True,
            "is_game": False,
            "platform": info["platform"],
            "pie_active": bool(info.get("pie_active")),
        }

    async def get_loaded_map(self) -> Dict[str, Any]:
        """
        Return the currently loaded persistent level path.

        Returns:
            Dictionary with ``map_path``, the editor world's persistent level
            (also while Play In Editor runs), and ``pie_active``; or a
            ``ScriptError`` envelope when the editor world could not be read.
        """
        return await self._execute_json_script(LOADED_MAP)

    # ------------------------------------------------------------------
    # Phase 1 session methods — Scene Read Operations
    # ------------------------------------------------------------------

    async def list_actors(
        self,
        class_filter: Optional[str] = None,
        tag_filter: Optional[str] = None,
        max_results: int = 200,
    ) -> Dict[str, Any]:
        """List actors with actual world transforms, class names and actor tags.

        While Play In Editor runs, Unreal's editor lists no level actors, so
        the editor world is read directly (``_unreal_scripts.EDITOR_WORLD``'s
        ``level_actors``) and ``pie_active`` is True: the same actors, as
        they are in the editor rather than in the running game, and possibly
        the builder brush and helper actors the Outliner hides.
        """
        if not 1 <= max_results <= 5000:
            raise ValueError("max_results must be between 1 and 5000")
        code = ACTOR_HELPERS + EDITOR_WORLD + f"""
class_filter = {class_filter!r}
tag_filter = {tag_filter!r}
actors = []
truncated = False
for actor in level_actors():
    cls = actor.get_class()
    if class_filter and class_filter not in (cls.get_name(), cls.get_path_name()):
        continue
    if tag_filter and not actor.actor_has_tag(tag_filter):
        continue
    if len(actors) == {max_results!r}:
        truncated = True
        break
    actors.append(actor_info(actor))
print(json.dumps({{"actors": actors, "count": len(actors), "truncated": truncated,
                  "pie_active": pie_active()}}))
"""
        return await self._execute_json_script(code)

    async def get_actor_info(self, actor_path: str) -> Dict[str, Any]:
        """Read an actor's world transform, components, tags and visibility, and ``pie_active``.

        Returns:
            The actor's info. A not-found error naming the path when it names
            no live actor of the level open in the editor: nothing there, an
            actor already deleted, or one of a map that is not open, which
            is never loaded (``_unreal_scripts.ACTOR_HELPERS``' ``actor_at``).
        """
        return await self._execute_json_script(
            ACTOR_HELPERS
            + f"\nprint(json.dumps(dict(actor_info(actor_at({actor_path!r})), pie_active=pie_active())))"
        )

    async def search_assets(
        self,
        query: str = "",
        class_names: Optional[List[str]] = None,
        package_paths: Optional[List[str]] = None,
        max_results: int = 100,
    ) -> Dict[str, Any]:
        """
        Search the Unreal Asset Registry.

        Args:
            query: Search query string.
            class_names: Filter by UClass names.
            package_paths: Package path prefixes to search within.
            max_results: Maximum number of results, from 1 to
                ``MAX_ASSET_SEARCH_RESULTS`` (1000).

        Returns:
            Dictionary with ``assets`` list, ``count``, ``truncated``, or a
            ``ValidationError`` envelope, sent to no editor, when
            ``max_results`` is out of range: Remote Control sizes its result
            array to the limit, and -1 crashed the editor.
        """
        if (
            isinstance(max_results, bool)
            or not isinstance(max_results, int)
            or not 1 <= max_results <= MAX_ASSET_SEARCH_RESULTS
        ):
            return {
                "success": False,
                "error_type": "ValidationError",
                "error": f"max_results must be an integer from 1 to {MAX_ASSET_SEARCH_RESULTS}, got {max_results!r}",
            }
        body: Dict[str, Any] = {
            "Query": query,
            "Limit": max_results,
        }
        if class_names:
            body["Filter"] = {"ClassNames": class_names}
        if package_paths:
            body.setdefault("Filter", {})["PackagePaths"] = package_paths

        data = await self._http_put("/remote/search/assets", body)
        raw_assets: list = data.get("Assets", [])

        assets: list = []
        for asset in raw_assets[:max_results]:
            assets.append({
                "name": asset.get("Name", ""),
                "path": asset.get("Path", ""),
                "class_name": asset.get("Class", ""),
                "package_path": asset.get("PackagePath", ""),
            })

        return {
            "assets": assets,
            "count": len(assets),
            "truncated": len(raw_assets) > max_results,
        }

    async def describe_object(self, object_path: str) -> Dict[str, Any]:
        """
        Get full property/function metadata for a UObject.

        Args:
            object_path: Full object path.

        Returns:
            Dictionary with ``object_path``, ``class_name``, ``properties``,
            ``functions``.
        """
        body: Dict[str, Any] = {"objectPath": object_path}
        data = await self._http_put("/remote/object/describe", body)

        properties: list = []
        for prop in data.get("Properties", []):
            properties.append({
                "name": prop.get("Name", ""),
                "type": prop.get("Type", ""),
                "value": prop.get("Value"),
            })

        functions: list = [f.get("Name", "") for f in data.get("Functions", [])]

        return {
            "object_path": object_path,
            "class_name": data.get("Class", "Unknown"),
            "properties": properties,
            "functions": functions,
        }

    async def get_actor_thumbnail(
        self,
        asset_path: str,
        width: int = 256,
        height: int = 256,
    ) -> Dict[str, Any]:
        """
        Get a thumbnail image for an asset.

        Args:
            asset_path: Full asset path.
            width: Thumbnail width in pixels.
            height: Thumbnail height in pixels.

        Returns:
            Dictionary with ``asset_path``, ``image_base64``, ``width``,
            ``height``.
        """
        body: Dict[str, Any] = {
            "objectPath": asset_path,
            "Width": width,
            "Height": height,
        }
        data = await self._http_request(
            "PUT", "/remote/object/thumbnail", body, response_kind="image"
        )
        return {"asset_path": asset_path, **data}

    async def summarize_scene(self) -> Dict[str, Any]:
        """
        Generate an LLM-friendly scene digest.

        Returns:
            Dictionary with map_path, total_actors, actor_class_counts,
            summary_text and pie_active (while Play In Editor runs, the
            editor world is read directly, as in ``list_actors``).
        """
        # Count classes inside the editor in one script rather than one
        # ``/remote/object/describe`` round trip per actor.
        scene = await self._execute_json_script(SCENE_SUMMARY)
        if scene.get("error"):
            return scene
        map_path: str = scene["map_path"]
        total_actors: int = scene["total_actors"]
        class_counts: Dict[str, int] = scene["actor_class_counts"]

        # Keys are class path names (``/Script/Engine.StaticMeshActor``);
        # compare the short name so the count isn't always zero.
        static_meshes = sum(
            v for k, v in class_counts.items()
            if k.rsplit(".", 1)[-1] == "StaticMeshActor"
        )
        lights = sum(v for k, v in class_counts.items() if "Light" in k)
        cameras = sum(v for k, v in class_counts.items() if "Camera" in k)

        pie_active = bool(scene.get("pie_active"))
        parts: list = [f"Map: {map_path}", f"Total actors: {total_actors}"]
        if pie_active:
            parts.append("Play In Editor is running; these are the editor world's actors")
        for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1])[:10]:
            parts.append(f"  {cls}: {cnt}")

        return {
            "map_path": map_path,
            "total_actors": total_actors,
            "actor_class_counts": class_counts,
            "static_meshes": static_meshes,
            "lights": lights,
            "cameras": cameras,
            "summary_text": "\n".join(parts),
            "pie_active": pie_active,
        }

    # ------------------------------------------------------------------
    # Phase 2 — Viewport & Visual Observation
    # ------------------------------------------------------------------

    async def capture_viewport(
        self,
        resolution_x: int = 1920,
        resolution_y: int = 1080,
        format: str = "png",
        inline: bool = False,
    ) -> Dict[str, Any]:
        """
        Capture viewport screenshot via HighResScreenshot console command.

        The command writes a file to ``Project/Saved/Screenshots/``.
        We then read it back and return base64-encoded data.

        Args:
            resolution_x: Capture width in pixels.
            resolution_y: Capture height in pixels.
            format: Image format — ``png``. The CLI can convert to JPEG.

        Returns:
            Dictionary with image_base64, resolution_x, resolution_y, format.
        """
        if format != "png":
            return {
                "success": False,
                "error_type": "UnsupportedFormat",
                "error": "Unreal HighResShot captures PNG. Use CLI capture --format jpeg to convert locally.",
            }
        if not 1 <= resolution_x <= 16384 or not 1 <= resolution_y <= 16384:
            raise ValueError("Capture dimensions must be between 1 and 16384")
        # The trigger goes through Remote Control's ExecuteConsoleCommand
        # RPC (NOT through Python's unreal.SystemLibrary.execute_console_command
        # — empirically that path doesn't fire HighResShot in the editor).
        # The read is a separate ExecutePythonCommandEx call so UE's main
        # thread stays free between trigger and read; the adapter, not the
        # embedded script, owns the wait via asyncio.sleep.
        #
        # New files are recognised by name: the editor's screenshot
        # directories are listed once before the trigger, and each poll
        # considers only names absent from that snapshot. That needs no
        # clock agreement between this process and the editor host, and a
        # coarse filesystem mtime can't hide the new file.
        marker = "@@SIMUL_SCREENSHOT@@"
        prelude = f"""
import os, base64, json, unreal
ss_root = os.path.join(os.path.abspath(unreal.Paths.project_saved_dir()), 'Screenshots')
def shots():
    found = []
    for subdir in {_SCREENSHOT_SUBDIRS!r}:
        try:
            with os.scandir(os.path.join(ss_root, subdir)) as entries:
                found.extend(e for e in entries if e.name.endswith('.{format}') and e.is_file())
        except OSError:
            pass
    return found
"""
        try:
            snapshot = self._parse_marker_json(
                await self._execute_python(
                    prelude
                    + f"print({marker!r} + json.dumps({{'existing': [e.path for e in shots()]}}))\n",
                    mode="ExecuteFile",
                ),
                marker,
            )
        except Exception as exc:
            snapshot = {"error": str(exc)}
        if "existing" not in snapshot:
            return {
                "success": False,
                "error_type": "CaptureError",
                "error": "Could not list the editor screenshot directory before capture: "
                + str(snapshot.get("error", "no listing returned")),
            }
        existing: List[str] = snapshot["existing"]
        try:
            await self._call_function(
                "/Script/Engine.Default__KismetSystemLibrary",
                "ExecuteConsoleCommand",
                {
                    "WorldContextObject": "",
                    "Command": f"HighResShot {resolution_x}x{resolution_y}",
                },
            )
        except Exception as exc:
            return {"success": False, "error_type": "CaptureError", "error": str(exc)}

        max_inline = MAX_INLINE_CAPTURE_BYTES
        # A PNG is complete once its IEND chunk is on disk; checking the tail
        # keeps a poll from returning a file the editor is still writing.
        read_code = prelude + f"""
existing = set({existing!r})
target = None
target_mtime = None
for entry in shots():
    if entry.path in existing:
        continue
    try:
        mtime = entry.stat().st_mtime
    except OSError:
        continue
    if target is None or mtime > target_mtime:
        target, target_mtime = entry.path, mtime
payload = {{'path': '', 'size_bytes': 0}}
if target:
    try:
        size = os.path.getsize(target)
        with open(target, 'rb') as f:
            f.seek(max(0, size - 12))
            complete = size > 0 and b'IEND' in f.read()
    except OSError:
        complete = False
    if complete:
        payload = {{'path': target, 'size_bytes': size}}
        if {inline!r} and size <= {max_inline}:
            with open(target, 'rb') as f:
                payload['image_base64'] = base64.b64encode(f.read()).decode()
        elif {inline!r}:
            payload['inline_skipped'] = (
                'Capture is %d bytes, above the %d byte inline cap. '
                'Read the file at path, or lower the resolution.'
                % (size, {max_inline})
            )
print({marker!r} + json.dumps(payload))
"""

        capture_payload: Dict[str, Any] = {}
        # Short first wait so UE's frame loop runs HighResShot, then poll.
        await asyncio.sleep(_CAPTURE_POLL_SECONDS)
        deadline = asyncio.get_event_loop().time() + 15.0
        while True:
            try:
                capture_payload = self._parse_marker_json(
                    await self._execute_python(read_code, mode="ExecuteFile"),
                    marker,
                )
            except Exception:
                capture_payload = {}
            if capture_payload.get("path"):
                break
            if asyncio.get_event_loop().time() >= deadline:
                break
            await asyncio.sleep(_CAPTURE_POLL_SECONDS)

        result: Dict[str, Any] = {
            "path": capture_payload.get("path", ""),
            "size_bytes": capture_payload.get("size_bytes", 0),
            "resolution_x": resolution_x,
            "resolution_y": resolution_y,
            "format": format,
        }
        if not result["path"] or not result["size_bytes"]:
            return {
                "success": False,
                "error_type": "CaptureError",
                "error": "Unreal did not produce a nonempty screenshot before the capture deadline.",
            }
        if "image_base64" in capture_payload:
            result["image_base64"] = capture_payload["image_base64"]
            result["encoding"] = "base64"
        if "inline_skipped" in capture_payload:
            result["inline_skipped"] = capture_payload["inline_skipped"]
        return result

    async def capture_to_file(
        self,
        output: Path,
        resolution_x: int = 1920,
        resolution_y: int = 1080,
        format: str = "png",
    ) -> Dict[str, Any]:
        """Download a capture in bounded chunks, independently of the MCP inline cap.

        Always transfer from the editor host; a coincident local pathname is
        not proof that the editor is local. Atomically replace output only after
        the download and image validation succeed.
        """
        if format not in {"png", "jpeg", "jpg"}:
            raise ValueError("format must be png, jpeg or jpg")
        result = await self.capture_viewport(resolution_x, resolution_y, inline=False)
        if result.get("error"):
            return result
        output = Path(output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        capture_path = result["path"]
        size = result["size_bytes"]
        temporary: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as stream:
                temporary = Path(stream.name)
                for offset in range(0, size, _CAPTURE_CHUNK_BYTES):
                    length = min(_CAPTURE_CHUNK_BYTES, size - offset)
                    chunk = await self._execute_json_script(f"""
import os, json, base64, unreal
root = os.path.realpath(os.path.join(unreal.Paths.project_saved_dir(), 'Screenshots'))
path = os.path.realpath({capture_path!r})
if os.path.commonpath([root, path]) != root:
    raise ValueError('Capture path is outside the editor screenshot directory')
with open(path, 'rb') as stream:
    stream.seek({offset!r})
    data = stream.read({length!r})
print(json.dumps({{'data': base64.b64encode(data).decode('ascii')}}))
""")
                    if chunk.get("error"):
                        return chunk
                    data = base64.b64decode(chunk["data"], validate=True)
                    if len(data) != length:
                        raise IOError(
                            "Screenshot changed or was truncated during transfer"
                        )
                    stream.write(data)
            from PIL import Image

            with Image.open(temporary) as captured:
                captured.load()
                if format in {"jpeg", "jpg"}:
                    captured.convert("RGB").save(temporary, format="JPEG")
            temporary.replace(output)
            return {
                **result,
                "file_path": str(output),
                "format": format,
                "size_bytes": output.stat().st_size,
            }
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    async def get_viewport_info(self) -> Dict[str, Any]:
        """
        Get active editor viewport camera and render information.

        Reads the camera through ``UnrealEditorSubsystem.GetLevelViewportCameraInfo``.
        Remote Control exposes no viewport size, field of view or projection
        type, so those are reported as None rather than guessed. A failed
        read is an error, never a default camera.

        Returns:
            Dictionary with camera_location, camera_rotation, viewport_size,
            fov, projection_type.
        """
        data = await self._call_function(
            "/Script/UnrealEd.Default__UnrealEditorSubsystem",
            "GetLevelViewportCameraInfo",
        )
        loc = data.get("CameraLocation")
        rot = data.get("CameraRotation")
        if data.get("ReturnValue") is False or not isinstance(loc, dict) or not isinstance(rot, dict):
            return {
                "success": False,
                "error": "No active level editor viewport camera to read",
                "error_type": "ViewportUnavailable",
            }
        return {
            "camera_location": (loc.get("X", 0.0), loc.get("Y", 0.0), loc.get("Z", 0.0)),
            "camera_rotation": (rot.get("Pitch", 0.0), rot.get("Yaw", 0.0), rot.get("Roll", 0.0)),
            "viewport_size": None,
            "fov": None,
            "projection_type": None,
        }

    async def set_camera_view(
        self,
        location: Optional[tuple] = None,
        rotation: Optional[tuple] = None,
    ) -> Dict[str, Any]:
        """
        Set the editor viewport camera position and rotation.

        Field of view is not settable: Remote Control exposes no setter for
        the level viewport's FOV.

        Args:
            location: Camera position (X, Y, Z) in cm; None leaves it unchanged.
            rotation: Camera rotation (Pitch, Yaw, Roll) in degrees; None
                leaves it unchanged.

        Returns:
            Dictionary with the applied location and rotation; a field that
            was not set is None.
        """
        # SetLevelViewportCameraInfo takes both location and rotation, and
        # Remote Control zero-fills an omitted one; resend the current value
        # for whichever the caller left unset.
        params: Dict[str, Any] = {}
        if location is not None:
            params["CameraLocation"] = {
                "X": location[0],
                "Y": location[1],
                "Z": location[2],
            }
        if rotation is not None:
            params["CameraRotation"] = {
                "Pitch": rotation[0],
                "Yaw": rotation[1],
                "Roll": rotation[2],
            }
        if len(params) == 1:
            current = await self._call_function(
                "/Script/UnrealEd.Default__UnrealEditorSubsystem",
                "GetLevelViewportCameraInfo",
            )
            key = "CameraRotation" if location is not None else "CameraLocation"
            if current.get("ReturnValue") is False or not isinstance(current.get(key), dict):
                return {
                    "success": False,
                    "error": "Could not read the current viewport camera to keep "
                             + ("rotation" if location is not None else "location"),
                    "error_type": "ViewportUnavailable",
                }
            params[key] = current[key]

        if params:
            await self._call_function(
                "/Script/UnrealEd.Default__UnrealEditorSubsystem",
                "SetLevelViewportCameraInfo",
                params,
            )

        return {
            "location": tuple(location) if location is not None else None,
            "rotation": tuple(rotation) if rotation is not None else None,
        }

    async def focus_on_actor(
        self,
        actor_path: str,
        distance: float = 0.0,
        fov: float = FOCUS_FOV_DEGREES,
        aspect_ratio: float = FOCUS_ASPECT_RATIO,
    ) -> Dict[str, Any]:
        """
        Focus the editor viewport camera on a specific actor and select it.

        One editor script adds the actor to the editor selection
        (``EditorActorSubsystem.set_actor_selection_state``), reads the
        selection back and reads the actor's location and bounds
        (``get_actor_bounds``: every primitive component, child actors'
        included). The editor-only components the editor's own Focus leaves
        out are left out of the bounds too, since theirs can reach far beyond
        the actor: the frustum the editor draws for a camera or a scene
        capture (10 m on every side of it), a CineCamera's debug focus plane
        (100 m across at the focus distance, 1 km by default, and still there,
        hidden, once turned off) and a spare icon attached to nothing, which a
        SkyLight leaves at the world origin. A shape (box, sphere or capsule
        component) whose collision is turned off is left out too, as the
        editor's Focus leaves out every shape: it collides with nothing, like
        a SphereReflectionCapture's influence sphere (30 m in radius by
        default) or a BoxReflectionCapture's boxes. Any other component that
        is not editor-only counts, such as a trigger box's collision shape,
        and an actor whose ``get_actor_bounds`` is not the box around its
        components (a level instance, a NavLinkProxy) keeps it. The camera is
        then placed ``distance`` behind the point it looks at (toward -X) and
        half as far above it, pitched down at it, and read back.

        A positive ``distance`` looks at the actor's location (its pivot).
        Distance 0 or less fits the actor in the view, as the editor's own Focus does:
        the camera looks at the center of the bounds and stands back until
        their bounding sphere fills the narrower side of the view with 10% to
        spare, and at least until the sphere, with that margin, is beyond
        Unreal's 10 cm near clip plane. The view is ``fov`` degrees across and
        ``aspect_ratio`` wide per unit of height, keeping ``fov`` across its
        width as Unreal's viewports and captures do. Editor Python cannot
        read the level viewport's field of view, so the defaults are UE's
        default, 90 degrees across, on the 16:9 of the default capture (1920
        x 1080); a narrower field of view, or a view wider than about 2:1,
        can crop the actor unless they are passed. An actor whose bounds have
        no size (zero extent, such as a StaticMeshActor without a mesh) gets
        the default 500 cm from its location instead, and ``fit_warning``
        says so. In the editor an actor without a mesh, such as an empty
        actor or a light, usually has bounds: those of its editor icon.

        While Play In Editor (or Simulate) runs the actor is not selected:
        Unreal's editor reports no selection then and clears it when PIE
        stops (UE 5.7.4). The camera still moves.

        Args:
            actor_path: Full actor path to focus on.
            distance: How far behind the actor the camera stands, in cm; it
                also rises half as far, looking down at the actor. 0 or less
                fits the actor's bounds in the view.
            fov: Field of view across the view's width, in degrees, that
                auto-fit frames the actor in: the level viewport's, which
                editor Python cannot read (UE's default is 90). At least
                0.001, the narrowest Unreal renders, and less than 180. Used
                only when distance is 0 or less.
            aspect_ratio: Width over height of that view: a capture's
                resolution_x / resolution_y (the default capture is 16:9),
                from 1/16384 to 16384. Used only when distance is 0 or less.

        Returns:
            Dictionary with actor_path, camera_location and camera_rotation
            (read back), distance (the one used: the distance passed, the
            fitted one, or 500 for an actor without bounds), selected (the
            selection read back), pie_active, fit_warning when auto-fit
            found no bounds to fit, and a warning when the actor is not
            selected: Play In Editor running, or an actor Unreal does not
            select (hidden in the editor or in a hidden or locked level, a
            child actor or an actor inside a level instance, or any actor
            while the editor mode or a selection lock prevents selection). A
            not-found error, before the camera moves, when the path names no
            live actor of the level open in the editor (an actor deleted, or
            one of a map that is not open, which is never loaded).

        Raises:
            ValueError: ``fov`` is under 0.001 degrees or not under 180, or
                ``aspect_ratio`` is outside 1/16384 to 16384, before the
                actor is selected or the camera moves.
        """
        problem = focus_view_error(fov, aspect_ratio)
        if problem is not None:
            raise ValueError(problem)
        args = {"actor_path": actor_path}
        focus = await self._execute_json_script(f"args = {args!r}\n" + FOCUS_ACTOR)
        if focus.get("error"):
            return focus

        fit_warning: Optional[str] = None
        if distance > 0.0:
            target, dist = focus["location"], distance
        elif any(focus["bounds_extent"]):
            target = focus["bounds_origin"]
            dist = _focus_fit_distance(focus["bounds_extent"], fov, aspect_ratio)
        else:
            # No primitive component with a size. The bounds' origin is then
            # the world origin, or where a component without a mesh sits, so
            # the camera looks at the actor's location instead.
            target, dist = focus["location"], FOCUS_DEFAULT_DISTANCE
            fit_warning = FOCUS_NO_BOUNDS_WARNING

        # Position camera to look at the target: offset behind and above it.
        cam_x = target[0] - dist
        cam_y = target[1]
        cam_z = target[2] + dist * 0.5
        # Compute pitch to look at the target
        dz = target[2] - cam_z
        pitch = -math.degrees(math.atan2(-dz, dist))
        await self._call_function(
            "/Script/UnrealEd.Default__UnrealEditorSubsystem",
            "SetLevelViewportCameraInfo",
            {
                "CameraLocation": {"X": cam_x, "Y": cam_y, "Z": cam_z},
                "CameraRotation": {"Pitch": pitch, "Yaw": 0.0, "Roll": 0.0},
            },
        )

        # Read back viewport camera position
        info = await self.get_viewport_info()
        if info.get("success") is False:
            return info

        selected = bool(focus["selected"])
        pie_active = bool(focus["pie_active"])
        result: Dict[str, Any] = {
            "actor_path": actor_path,
            "camera_location": info["camera_location"],
            "camera_rotation": info["camera_rotation"],
            "distance": dist,
            "selected": selected,
            "pie_active": pie_active,
        }
        if fit_warning is not None:
            result["fit_warning"] = fit_warning
        if not selected:
            result["warning"] = FOCUS_PIE_WARNING if pie_active else FOCUS_NOT_SELECTED_WARNING
        return result

    # ------------------------------------------------------------------
    # Phase 3 — Scene Manipulation
    # ------------------------------------------------------------------

    async def spawn_actor(
        self,
        asset_path: str,
        location: Sequence[float] = (0.0, 0.0, 0.0),
        rotation: Sequence[float] = (0.0, 0.0, 0.0),
        label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Spawn an actor from an Actor class or from an asset.

        A class path (``/Script/Engine.PointLight``, a Blueprint's
        ``/Game/BP_Door.BP_Door_C``) spawns that class. Any other asset path
        is placed the way the editor places an asset dropped into the level:
        ``/Engine/BasicShapes/Cube.Cube`` becomes a StaticMeshActor showing
        the cube, a Blueprint asset its actor. Spawn and label are one
        undoable editor transaction, run as one editor script.

        Args:
            asset_path: Actor class path or asset path to spawn from.
            location: Spawn location (X, Y, Z) in cm.
            rotation: Spawn rotation (Pitch, Yaw, Roll) in degrees.
            label: Optional actor label in the outliner.

        Returns:
            Dictionary with actor_path, actor_class (the spawned actor's
            class), label, and the location and rotation read back from the
            new actor. An error envelope, never an empty actor_path, when the
            path loads nothing (``NotFoundError``) or names something Unreal
            cannot place: a class that is not an Actor, an actor already in a
            level, an asset no actor factory accepts (``SpawnError``); and
            ``PIEActive`` while Play In Editor runs, when Unreal's editor
            spawns nothing. Those are logged as refusals, not failures.

        Raises:
            ValueError: location or rotation is not three finite numbers.
        """
        args = {
            "path": asset_path,
            "location": _vector3("location", location, "x, y, z"),
            "rotation": _vector3("rotation", rotation, "pitch, yaw, roll"),
            "label": label or None,
        }
        result = await self._execute_json_script(
            f"args = {args!r}\n" + SPAWN_ACTOR,
            expected_errors=("NotFoundError", "SpawnError"),
        )
        if not result.get("error") and not result.get("actor_path"):
            return self._script_error(
                f"Unreal reported no spawned actor for {asset_path}"
            )
        return result

    async def delete_actor(self, actor_path: str) -> Dict[str, Any]:
        """
        Delete an actor from the level.

        Uses ``EditorActorSubsystem.DestroyActor`` via Remote Control.

        Args:
            actor_path: Full actor path to delete.

        Returns:
            Dictionary with actor_path, deleted status. When Unreal destroyed
            nothing because Play In Editor runs (DestroyActor refuses then), a
            ``PIEActive`` error envelope instead.
        """
        data = await self._call_function(
            "/Script/UnrealEd.Default__EditorActorSubsystem",
            "DestroyActor",
            {"ActorToDestroy": actor_path},
        )
        deleted = bool(data.get("ReturnValue", False))
        if not deleted and await self._pie_active():
            return self._refused_during_pie("delete actors")

        return {
            "actor_path": actor_path,
            "deleted": deleted,
        }

    async def set_actor_transform(
        self,
        actor_path: str,
        location: Optional[Sequence[float]] = None,
        rotation: Optional[Sequence[float]] = None,
        scale: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """
        Set an actor's world location, rotation and/or scale.

        Only the components given are written, in one undoable editor
        transaction; a component left as None keeps its current value. The
        whole transform is read back afterwards, in the same script.

        Args:
            actor_path: Full actor path.
            location: World position as [x, y, z] in cm; None keeps the
                current location.
            rotation: World rotation as [pitch, yaw, roll] in degrees; None
                keeps the current rotation.
            scale: Scale factors as [x, y, z]; None keeps the current scale.

        Returns:
            Dictionary with actor_path and the location, rotation and scale
            read back after the write. A ``ScriptError`` envelope when the
            actor cannot be found, and a ``TransformNotApplied`` error, with
            the read-back values attached, naming each given component that
            reads back different from what was written (an actor without a
            root component has no transform to set and reads back the
            identity transform).

        Raises:
            ValueError: A given component is not three finite numbers.
        """
        values = {
            name: _vector3(name, value, axes)
            for name, value, axes in (
                ("location", location, "x, y, z"),
                ("rotation", rotation, "pitch, yaw, roll"),
                ("scale", scale, "x, y, z"),
            )
            if value is not None
        }
        args = {"actor_path": actor_path, "values": values}
        result = await self._execute_json_script(
            f"args = {args!r}\n" + SET_ACTOR_TRANSFORM
        )
        if result.get("error"):
            return result
        rejected = [
            name
            for name, wanted in values.items()
            if not _transform_component_applied(name, wanted, result[name])
        ]
        if rejected:
            return {
                **result,
                "success": False,
                "error": "Unreal did not apply: " + ", ".join(rejected),
                "error_type": "TransformNotApplied",
            }
        return result

    async def set_actor_property(
        self,
        actor_path: str,
        property_name: str,
        property_value: str,
        generate_transaction: bool = True,
    ) -> Dict[str, Any]:
        """
        Set any writable property on an actor.

        Args:
            actor_path: Full actor path.
            property_name: Property name to set.
            property_value: Property value as JSON string.
            generate_transaction: Whether to generate an undo transaction.

        Returns:
            Dictionary with actor_path, property_name.
        """
        import json as json_lib

        try:
            parsed_value = json_lib.loads(property_value)
        except (json_lib.JSONDecodeError, TypeError):
            parsed_value = property_value

        access = (
            "WRITE_TRANSACTION_ACCESS" if generate_transaction else "WRITE_ACCESS"
        )
        body: Dict[str, Any] = {
            "objectPath": actor_path,
            "propertyName": property_name,
            "propertyValue": {property_name: parsed_value},
            "access": access,
        }
        await self._http_put("/remote/object/property", body)

        return {
            "actor_path": actor_path,
            "property_name": property_name,
        }

    async def call_actor_function(
        self,
        actor_path: str,
        function_name: str,
        parameters: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Call a BlueprintCallable function on an actor.

        Args:
            actor_path: Full actor path.
            function_name: Function name to call.
            parameters: Parameters as JSON string.

        Returns:
            Dictionary with actor_path, function_name, return_value.
        """
        if not self.settings.security.allow_script_execution:
            return self._script_execution_denied()

        import json as json_lib

        body: Dict[str, Any] = {
            "objectPath": actor_path,
            "functionName": function_name,
        }
        if parameters:
            try:
                decoded = json_lib.loads(parameters)
            except (json_lib.JSONDecodeError, TypeError) as exc:
                return {
                    "success": False,
                    "error": f"parameters is not valid JSON: {exc}",
                    "error_type": "ValidationError",
                }
            if not isinstance(decoded, dict):
                return {
                    "success": False,
                    "error": "parameters must be a JSON object mapping parameter names to values",
                    "error_type": "ValidationError",
                }
            body["parameters"] = decoded

        data = await self._http_put("/remote/object/call", body)
        return_value = json_lib.dumps(data) if data else None

        return {
            "actor_path": actor_path,
            "function_name": function_name,
            "return_value": return_value,
        }

    async def set_actor_parent(
        self,
        actor_path: str,
        parent_path: Optional[str] = None,
        detach: bool = False,
    ) -> Dict[str, Any]:
        """
        Attach an actor to a parent actor, or detach it from its parent.

        One editor script (``_unreal_scripts.SET_ACTOR_PARENT``) attaches with
        ``attach_to_actor`` (``K2_AttachToActor``) or detaches with
        ``detach_from_actor``, keeping the world transform, in one undoable
        transaction, and reads the parent back. Exactly one of
        ``parent_path`` and ``detach`` is given. Unreal refuses to attach an
        actor to itself, to an actor attached under it, or a Static actor
        under a parent that is not Static, and returns False without a
        reason; those are refused first, each with its reason.

        Args:
            actor_path: Full path of the actor to attach or detach.
            parent_path: Full path of the actor to attach it to.
            detach: Detach the actor from its parent, keeping its world
                transform. An actor with no parent is left as it is.

        Returns:
            Dictionary with actor_path, parent_path (the parent read back;
            None once detached), previous_parent_path and changed (whether
            the parent changed). A ``ValidationError`` envelope when neither or
            both of parent_path and detach are given, for the actor itself as
            its parent and for a cycle; ``InvalidState`` for an actor without
            a root component or a Static child under a parent that is not
            Static; a not-found error for an actor or parent path that names
            no live actor of the level open in the editor; ``EngineError``
            when Unreal still did not attach or detach it.
        """
        if bool(parent_path) == bool(detach):
            return {
                "success": False,
                "error_type": "ValidationError",
                "error": (
                    "Pass parent_path to attach the actor, or detach=True to detach it from its parent"
                    + (", not both" if detach else "")
                ),
            }
        args = {"actor_path": actor_path, "parent_path": None if detach else parent_path}
        return await self._execute_json_script(
            f"args = {args!r}\n" + SET_ACTOR_PARENT,
            expected_errors=("NotFoundError", "ValidationError", "InvalidState"),
        )

    async def add_component(
        self,
        actor_path: str,
        component_class: str,
        component_name: Optional[str] = None,
        parent_component: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Add an instance component to an actor in the level.

        One editor script (``_unreal_scripts.ADD_COMPONENT``) adds it the way
        the editor's Add Component button does, through
        ``SubobjectDataSubsystem.add_new_subobject`` (Remote Control refuses
        ``AddComponentByClass`` on UE 5.7), names it, in one undoable
        transaction, and reads it back from the actor. A scene component is
        attached under ``parent_component`` or the root component, and
        becomes the root of an actor without one.

        Args:
            actor_path: Full path of the actor.
            component_class: A native component class name
                (``PointLightComponent``) or a class path
                (``/Script/Engine.PointLightComponent``, a Blueprint
                component's ``/Game/BP_Lamp.BP_Lamp_C`` or its
                ``/Game/BP_Lamp.BP_Lamp``). Abstract classes are refused.
            component_name: Name of the new component, unique inside the
                actor; None lets Unreal name it after its class.
            parent_component: Name of the actor's scene component to attach
                the new one under; None attaches it under the root component.
                Only for a scene component.

        Returns:
            Dictionary with actor_path, component_path, component_name,
            component_class (the class read back), parent_component (the
            component it is attached under, None for a component that is not
            a scene component or became the root) and is_root. A
            ``ValidationError`` envelope for an invalid or taken name, a
            class that is abstract or not a component class, or a
            parent_component for a component that takes none; a not-found
            error for a class, parent component or actor that is not there;
            ``EngineError`` when Unreal adds or names nothing.
        """
        problem = None
        if not component_class.strip():
            problem = "component_class must name a component class"
        elif component_name is not None:
            problem = _object_name_error("component_name", component_name)
        if problem is None and parent_component is not None:
            problem = _object_name_error("parent_component", parent_component)
        if problem is not None:
            return {"success": False, "error_type": "ValidationError", "error": problem}
        args = {
            "actor_path": actor_path,
            "component_class": component_class.strip(),
            "component_name": component_name,
            "parent_component": parent_component,
        }
        return await self._execute_json_script(
            f"args = {args!r}\n" + ADD_COMPONENT,
            expected_errors=("NotFoundError", "ValidationError"),
        )

    async def set_actor_visibility(
        self,
        actor_path: str,
        visible: bool = True,
        propagate: bool = True,
    ) -> Dict[str, Any]:
        """
        Set actor visibility.

        Uses ``SetActorHiddenInGame`` UFUNCTION via Remote Control.

        Args:
            actor_path: Full actor path.
            visible: Whether the actor should be visible.
            propagate: Propagate to child components/actors.

        Returns:
            Dictionary with actor_path, visible.
        """
        body: Dict[str, Any] = {
            "objectPath": actor_path,
            "functionName": "SetActorHiddenInGame",
            "parameters": {
                "bNewHidden": not visible,
                "bPropagateToChildren": propagate,
            },
        }
        await self._http_put("/remote/object/call", body)

        return {
            "actor_path": actor_path,
            "visible": visible,
        }

    # ------------------------------------------------------------------
    # Phase 4 — Materials, Lighting & Rendering
    # ------------------------------------------------------------------

    async def get_material_info(self, material_path: str) -> Dict[str, Any]:
        """
        Get the parameters and metadata of a Material or Material Instance.

        Reads the asset through editor scripting, in or out of Play In
        Editor. Remote Control's ``/remote/object/describe`` answered with the
        asset's object properties instead, and no parent (observed on UE
        5.7.4).

        Args:
            material_path: Material or Material Instance asset path, e.g.
                /Game/Materials/MI_Wood.

        Returns:
            Dictionary with material_path (the asset's object path),
            class_name, parent_path (an instance's parent; None for a
            Material), base_material_path (the Material at the root of the
            parent chain), material_domain and blend_mode (Unreal's enum
            names, such as MD_SURFACE and BLEND_OPAQUE; an instance's blend
            mode is its override, else its parent's, and BLEND_OPAQUE without
            a parent), and parameters: one entry per scalar, vector and
            texture parameter with its name, param_type, association
            (``global`` for the material's own parameter, ``layer`` or
            ``blend`` for one inside its Material Layers), index (the layer or
            blend index; None for a global parameter), value (a float,
            [r, g, b, a] or a texture asset path; a Material's default, an
            instance's override or inherited value, for that layer) and
            overridden (whether the instance overrides it; None for a
            Material). Unreal's Python API lists and reads parameters by name
            only, so they are read through a transient Material Instance
            Dynamic that copies the material's parameters with their layers.
            An error envelope when the path loads nothing (``NotFoundError``)
            or is not a material (``InvalidAssetType``).
        """
        return await self._execute_json_script(
            f"args = {{'path': {material_path!r}}}\n" + MATERIAL_INFO,
            expected_errors=("NotFoundError", "InvalidAssetType"),
        )

    async def set_material_params(
        self,
        material_path: str,
        scalar_params: Optional[Dict[str, float]] = None,
        vector_params: Optional[Dict[str, list]] = None,
        texture_params: Optional[Dict[str, str]] = None,
        allow_engine_content: bool = False,
    ) -> Dict[str, Any]:
        """
        Set parameters on a Material Instance Constant and read them back.

        Checks the request, then writes through Unreal's MaterialEditingLibrary
        in one undoable editor transaction and reads every value back in the
        same script, in or out of Play In Editor. A Remote Control write of the
        instance's parameter arrays answered OK and changed nothing (observed
        on UE 5.7.4). ``success`` alone tells whether every value applied: a
        request with any problem is refused before anything is written, and a
        value that reads back different fails the call. Names are matched
        against the material's global parameters ignoring case, as Unreal
        does. A name only the material's Material Layers have is refused:
        Unreal's Python API writes parameters by name, and the global override
        it would add reads back as set while the layer stays as it was. The
        instance is left unsaved. An instance outside ``/Game`` is refused
        unless ``allow_engine_content`` is set: engine and plugin content is
        shared by every project using that engine or plugin, and an edit to
        /Engine/BasicShapes/BasicShapeMaterial_Inst left it dirty.

        Args:
            material_path: Material Instance asset path.
            scalar_params: Scalar parameter name to value.
            vector_params: Vector parameter name to [r, g, b] or [r, g, b, a];
                alpha defaults to 1.
            texture_params: Texture parameter name to texture asset path.
            allow_engine_content: Also edit an instance outside the project's
                /Game folder (engine or plugin content).

        Returns:
            Dictionary with material_path (the instance's object path),
            params_set (how many values read back as requested), applied
            (those values as read back, by kind and parameter name) and
            not_applied (empty). An error envelope with the same fields:
            ``ValidationError`` when the request is refused and nothing is
            written (not_applied gives, by kind and name, each parameter the
            material has no global parameter of that kind for, naming the
            layers and blends that have it when only its Material Layers do,
            each texture that does not load, each malformed value and each
            name given twice in different case; a request without parameters
            or not shaped as objects is refused too, before the editor is
            called), and ``MaterialParamsNotApplied`` when a value read back
            different after the write (not_applied names each; the values in
            applied were written and stay written, as for lights). An error
            envelope without them when the path loads nothing
            (``NotFoundError``), is not a Material Instance Constant
            (``InvalidAssetType``) or is outside /Game without
            ``allow_engine_content`` (``SandboxError``).
        """
        values, refused, problems = _material_parameter_values(scalar_params, vector_params, texture_params)
        if problems:
            return {
                "success": False,
                "error_type": "ValidationError",
                "error": "; ".join(problems),
                "material_path": material_path,
                "params_set": 0,
                "applied": {},
                "not_applied": refused,
            }
        args = {"path": material_path, "values": values, "allow_engine_content": bool(allow_engine_content)}
        return await self._execute_json_script(
            f"args = {args!r}\n" + SET_MATERIAL_PARAMS,
            expected_errors=(
                "NotFoundError",
                "InvalidAssetType",
                "ValidationError",
                "MaterialParamsNotApplied",
                "SandboxError",
            ),
        )

    async def create_material_instance(
        self,
        parent_path: str,
        instance_name: str,
        save_path: str = "",
        allow_engine_content: bool = False,
    ) -> Dict[str, Any]:
        """
        Create a Material Instance Constant from a parent material.

        Uses ``MaterialInstanceConstantFactoryNew`` through ``AssetTools`` and
        sets the parent, then saves the new asset. It is written under the
        project's /Game folder: without ``save_path``, next to a parent in
        /Game, else in /Game/Materials. A folder outside /Game is refused
        unless ``allow_engine_content`` is set: an instance of an /Engine
        parent used to be saved into the shared engine install
        (Engine/Content/BasicShapes).

        Args:
            parent_path: Parent material (or material instance) asset path.
            instance_name: Name for the new MIC.
            save_path: Content folder to save it in, such as /Game/Materials;
                empty picks the parent's folder when that is under /Game, else
                /Game/Materials.
            allow_engine_content: Also write to a folder outside /Game (engine
                or plugin content, shared by every project using it).

        Returns:
            Dictionary with instance_path, parent_path and class_name, or a
            ``ScriptError`` envelope when the parent is missing, the
            destination exists or the asset could not be created, a
            ``PIEActive`` envelope while Play In Editor runs, when Unreal's
            EditorAssetLibrary loads, finds and saves no assets, a
            ``ValidationError`` envelope for a folder or name Unreal cannot
            use and a ``SandboxError`` envelope for a folder outside /Game.
        """
        if not save_path:
            # Next to a parent in /Game: /Game/Materials/M_Base -> /Game/Materials.
            parent_folder = parent_path.rsplit("/", 1)[0] if "/" in parent_path else ""
            save_path = parent_folder if _in_project_content(parent_folder) else DEFAULT_MATERIAL_FOLDER
        save_path = save_path.rstrip("/")
        problem = _content_folder_problem(save_path) or _object_name_error("instance_name", instance_name)
        if problem is not None:
            return {"success": False, "error_type": "ValidationError", "error": problem}
        if not allow_engine_content and not _in_project_content(save_path):
            return _outside_project_refusal(
                save_path,
                "create a material instance in",
                "Pass a save_path under /Game, such as /Game/Materials, or allow_engine_content=true "
                "to write to the shared folder",
            )
        args = {"parent": parent_path, "name": instance_name, "folder": save_path}
        code = f"args = {args!r}\n" + CREATE_MATERIAL_INSTANCE
        return await self._execute_json_script(code)

    async def assign_material(
        self,
        actor_path: str,
        material_path: str,
        slot_index: int = 0,
    ) -> Dict[str, Any]:
        """
        Assign a material to a mesh component's material slot.

        Runs in one undoable editor transaction that records the component
        first, which also marks its level (or, in a World Partition level, the
        actor's package) dirty, so saving keeps the assignment, and reads the
        slot back. ``MeshComponent.set_material`` alone, which the tool used
        to call, recorded and dirtied nothing: a save kept nothing, a reload
        reverted it and Undo did not undo it (observed on UE 5.7.4). The actor
        must be in the level being edited; a path in another map is refused
        without loading that map.

        Args:
            actor_path: Target actor path.
            material_path: Material asset path to assign.
            slot_index: Material slot index (default 0).

        Returns:
            Dictionary with actor_path, component_path, material_path (the
            assigned material's object path), slot_index and
            previous_material_path (what the slot rendered with before). An
            error envelope when the actor is not in the level being edited
            (``Actor not found``), the material path names no material
            (``NotFoundError``), the slot is out of range
            (``ValidationError``) or the slot reads back a different material
            (``MaterialNotApplied``).
        """
        if isinstance(slot_index, bool) or not isinstance(slot_index, int):
            return {
                "success": False,
                "error_type": "ValidationError",
                "error": f"slot_index must be an integer, got {slot_index!r}",
            }
        args = {"actor_path": actor_path, "material_path": material_path, "slot_index": slot_index}
        return await self._execute_json_script(
            f"args = {args!r}\n" + ASSIGN_MATERIAL,
            expected_errors=("NotFoundError", "ValidationError"),
        )

    async def set_light_params(
        self,
        actor_path: str,
        intensity: Optional[float] = None,
        color: Optional[Sequence[float]] = None,
        temperature: Optional[float] = None,
        use_temperature: Optional[bool] = None,
        attenuation_radius: Optional[float] = None,
        cast_shadows: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Set light component parameters on a light actor.

        Writes the actor's light component in one editor transaction and
        reads every value back. A parameter left as None keeps its current
        value.

        Args:
            actor_path: Light actor path.
            intensity: Light intensity.
            color: Light color as [r, g, b], each channel 0-1; None keeps the
                current color.
            temperature: Color temperature in Kelvin.
            use_temperature: Use color temperature instead of direct color.
            attenuation_radius: Light attenuation radius in cm (point and
                spot lights only).
            cast_shadows: Enable shadow casting.

        Returns:
            Dictionary with actor_path, component_path, params_set and the
            read-back ``applied`` values, or a ``ScriptError`` envelope when
            the actor has no light component, a property does not exist on
            it, or Unreal did not apply a value.

        Raises:
            ValueError: ``color`` is not three numbers between 0 and 1.
        """
        # Python editor-property names on ULightComponent and subclasses.
        values: Dict[str, Any] = {}
        if intensity is not None:
            values["intensity"] = intensity
        if color is not None:
            rgb = _vector3("color", color, "r, g, b")
            if not all(0.0 <= channel <= 1.0 for channel in rgb):
                raise ValueError(f"color channels must be between 0 and 1, got {list(color)!r}")
            # LightColor is an FColor: 0-255 channels.
            values["light_color"] = [round(channel * 255) for channel in rgb] + [255]
        if temperature is not None:
            values["temperature"] = temperature
        if use_temperature is not None:
            values["use_temperature"] = use_temperature
        if attenuation_radius is not None:
            values["attenuation_radius"] = attenuation_radius
        if cast_shadows is not None:
            values["cast_shadows"] = cast_shadows

        if not values:
            return {"actor_path": actor_path, "params_set": 0}

        args = {"actor_path": actor_path, "values": values}
        result = await self._execute_json_script(f"args = {args!r}\n" + SET_LIGHT_PARAMS)
        return {"actor_path": actor_path, **result}

    async def set_render_settings(
        self,
        setting_name: str,
        setting_value: str,
    ) -> Dict[str, Any]:
        """
        Set one numeric rendering console variable and read it back.

        Only existing ``r.*`` and ``sg.*`` console variables are set, through
        the editor's console. Post-process settings live on PostProcessVolume
        actors and cameras, not in console variables, so this tool does not
        change them.

        Args:
            setting_name: Console variable name, r.* or sg.* (r.ScreenPercentage).
            setting_value: A single number, as text (75, 0.5, 1e3).

        Returns:
            ``setting_name``, ``applied`` (Unreal holds the requested value,
            read back), ``value`` (read back, as Unreal formats it) and
            ``previous_value``. A variable Unreal does not have fails with
            ``NotFoundError`` and changes nothing; a value Unreal did not take
            fails with ``RenderSettingNotApplied`` and the values read back.
        """
        # This is a fixed rendering tool, not another arbitrary console/code
        # dispatcher when SECURITY__ALLOW_SCRIPT_EXECUTION is false.
        if not re.fullmatch(r"(?:r|sg)\.[A-Za-z0-9_.]+", setting_name):
            raise ValueError(
                "setting_name must be an r.* or sg.* rendering console variable"
            )
        if not re.fullmatch(
            r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", setting_value
        ):
            raise ValueError("setting_value must be a single numeric value")
        args = {"name": setting_name, "value": setting_value}
        state = await self._execute_json_script(f"args = {args!r}\n" + SET_CONSOLE_VARIABLE)
        if state.get("error"):
            return {"setting_name": setting_name, **state}
        result: Dict[str, Any] = {"setting_name": setting_name, "requested_value": setting_value}
        if not state.get("known"):
            self.logger.info("Unreal has no console variable %s", setting_name)
            return {
                **result,
                "success": False,
                "applied": False,
                "error": f"Unreal has no console variable named {setting_name}; nothing was changed",
                "error_type": NOT_FOUND_ERROR,
            }
        previous, value = str(state["previous_value"]), str(state["value"])
        result.update(previous_value=previous, value=value)
        if _console_value_matches(value, setting_value):
            return {**result, "applied": True}
        if value == previous:
            reason = (
                f"Unreal kept {setting_name} at {value}: the variable is read-only while the editor "
                "runs (many r.* variables take effect only at startup, from Config/DefaultEngine.ini), "
                "or a higher-priority source set it; the editor log says which"
            )
        else:
            reason = f"Unreal set {setting_name} to {value}, not {setting_value}: it limits or rounds this variable"
        return {
            **result,
            "success": False,
            "applied": False,
            "error": reason,
            "error_type": "RenderSettingNotApplied",
        }

    # ------------------------------------------------------------------
    # Phase 5: Physics & Simulation Control
    # ------------------------------------------------------------------

    async def control_simulation(self, action: str) -> Dict[str, Any]:
        """
        Control a Play-In-Editor (PIE) session.

        Starting an already paused session resumes it. Commands targeting the
        current state leave the session unchanged.

        Args:
            action: One of start, stop, pause, resume, step.

        Returns:
            Dict with action executed and resulting state.
        """

        action = action.lower().strip()
        if action not in {"start", "stop", "pause", "resume", "step"}:
            raise ValueError(f"Invalid PIE action: {action}")
        if action == "step":
            return {
                "success": False,
                "error_type": "UnsupportedOperation",
                "error": "Exact single-frame PIE stepping is not exposed by the supported Unreal Python API.",
            }
        status = await self.get_simulation_status()
        if status.get("error"):
            return status
        if action in {"pause", "resume"} and not status["is_playing"]:
            return {
                "success": False,
                "error_type": "InvalidState",
                "error": "PIE is not running",
            }
        if action == "start" and not status["is_playing"]:
            command = "unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).editor_play_simulate()"
        elif action == "stop" and status["is_playing"]:
            command = "unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).editor_request_end_play()"
        elif action in {"start", "pause", "resume"} and status["is_paused"] != (
            action == "pause"
        ):
            command = (
                "world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_game_world()\n"
                f"if not unreal.GameplayStatics.set_game_paused(world, {action == 'pause'!r}):\n"
                "    raise RuntimeError('Unreal refused the pause state change')"
            )
        else:
            command = "pass"
        requested = await self._execute_json_script(
            "import unreal, json\n"
            + command
            + "\nprint(json.dumps({'requested': True}))"
        )
        if requested.get("error"):
            return requested
        # Editor lifecycle requests are deferred until after this HTTP request's frame.
        deadline = asyncio.get_running_loop().time() + self.timeout
        while True:
            status = await self.get_simulation_status()
            if status.get("error"):
                return status
            state = (
                "stopped"
                if not status["is_playing"]
                else ("paused" if status["is_paused"] else "playing")
            )
            target = {
                "start": "playing",
                "stop": "stopped",
                "pause": "paused",
                "resume": "playing",
            }[action]
            if state == target:
                return {"action": action, "state": state}
            if asyncio.get_running_loop().time() >= deadline:
                return {
                    "success": False,
                    "error_type": "TimeoutError",
                    "error": f"PIE did not reach {target}; current state: {state}",
                }
            await asyncio.sleep(0.1)

    async def get_simulation_status(self) -> Dict[str, Any]:
        """
        Query current PIE simulation status.

        Returns:
            Dict with is_playing, is_paused, frame_count, sim_time.
        """

        return await self._execute_json_script(SIMULATION_STATUS)

    async def enable_physics(
        self,
        actor_path: str,
        enable: bool = True,
        kinematic: bool = False,
        make_movable: bool = True,
        allow_detach: bool = False,
    ) -> Dict[str, Any]:
        """
        Make an actor's body simulate physics, make it kinematic, or turn physics off.

        Edits the actor's primitive component (its root, else its only one)
        in one undoable editor transaction that marks the map dirty, so Save
        All writes it and one Undo reverts it. A simulated body falls under
        gravity and takes forces; a kinematic body keeps its physics collision,
        so simulated bodies hit it, but moves only when its transform is set;
        turning physics off stops the body simulating and keeps its collision
        and mobility. Both physical modes need a Movable body with physics
        collision (QUERY_AND_PHYSICS or PHYSICS_ONLY) and, for a static mesh,
        a mesh with simple collision; a simulated mesh may not use its render
        triangles as simple collision. Unreal accepts the simulate flag on any
        body and reads it back as simulating even when the body never moves,
        so a body that cannot take the mode is refused with nothing changed
        (``IncompatibleBody``, each reason in ``details.problems``).

        Unreal detaches a body from its parent when it starts simulating,
        keeping its world transform; turning physics off does not reattach
        it. So an attached body is refused unless ``allow_detach`` is true;
        then it is detached in the same transaction (Undo reattaches it) and
        the reply names the parent in ``detached_from``.

        A path in a running simulation (``UEDPIE_<n>_<map>``) edits the play
        copy, which is neither undoable nor saved. An editor path while Play
        In Editor runs edits the editor world, which the running simulation
        does not see; the reply's ``warning`` says so.

        Args:
            actor_path: Full actor object path.
            enable: True makes the body physical (simulated, or kinematic with
                ``kinematic``); False stops it simulating physics.
            kinematic: With ``enable``, make the body kinematic instead of
                simulated: physics collision on, not simulated.
            make_movable: Make a Static or Stationary body Movable, which both
                physical modes need; False refuses such a body instead.
            allow_detach: Let a simulated body attached to a parent be
                detached from it, as Unreal does; False refuses it.

        Returns:
            Dict with actor_path, world ("editor" or "simulation"),
            physics_enabled (the body simulates), physics_mode ("simulated",
            "kinematic", "static" or "none" without physics collision) and
            the body read back: component_path, simulate_physics, mobility,
            collision_mode, collision_preset, mass_kg, mass_override_kg,
            gravity_enabled, linear_damping, angular_damping, attach_parent;
            mobility_changed_from and detached_from when this call changed
            them; warning. Or an IncompatibleBody envelope, or
            PhysicsNotApplied when the body did not read back as asked (every
            value restored).
        """
        mode = "off" if not enable else ("kinematic" if kinematic else "simulate")
        args = {
            "actor_path": actor_path,
            "mode": mode,
            "make_movable": bool(make_movable),
            "allow_detach": bool(allow_detach),
            "transaction": "simul: enable physics" if enable else "simul: disable physics",
        }
        return await self._execute_json_script(
            f"args = {args!r}\n" + ENABLE_PHYSICS,
            expected_errors=(INCOMPATIBLE_BODY, "PhysicsNotApplied"),
        )

    async def set_collision(
        self,
        actor_path: str,
        collision_preset: str = "",
        collision_enabled: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Configure collision settings on an actor.

        Writes the actor's primitive component in one undoable editor
        transaction that marks the map dirty (a path in a running simulation
        edits the play copy, which is neither undoable nor saved). Setting
        ``collision_enabled`` on a static mesh clears its Use Default
        Collision flag, as a preset does: with the flag set, Unreal put the
        mesh's own collision back in the running simulation and after the map
        reloaded.

        Args:
            actor_path: Full actor object path.
            collision_preset: Named collision preset (e.g. BlockAll,
                PhysicsActor, OverlapAll, NoCollision); empty keeps the
                current preset. An unknown preset changes nothing.
            collision_enabled: True for query and physics collision, False
                for none. None keeps the collision mode the preset sets, or
                the current one when no preset is given. Overriding a
                preset's mode turns the preset into Custom.

        Returns:
            Dict with actor_path, world, the collision_preset,
            collision_enabled and collision_mode read back afterwards,
            use_default_collision (null for a component that is not a static
            mesh), the body's physics_mode and other values as
            ``enable_physics`` reads them, and warning (a simulated body left
            with query-only collision falls through everything, one with no
            collision does not move at all); or a
            ValidationError envelope for an unknown preset, or a
            ``ScriptError`` envelope for an actor without one unambiguous
            primitive component.
        """
        args = {
            "actor_path": actor_path,
            "preset": collision_preset or None,
            "enabled": collision_enabled,
            "transaction": "simul: set collision",
        }
        return await self._execute_json_script(
            f"args = {args!r}\n" + SET_COLLISION, expected_errors=("ValidationError",)
        )

    async def apply_force(
        self,
        actor_path: str,
        force_x: float = 0.0,
        force_y: float = 0.0,
        force_z: float = 0.0,
        is_impulse: bool = False,
        location_x: Optional[float] = None,
        location_y: Optional[float] = None,
        location_z: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Apply a force or impulse to an actor's physics body.

        Args:
            actor_path: Full actor object path.
            force_x: Force X component.
            force_y: Force Y component.
            force_z: Force Z component.
            is_impulse: True for impulse, False for continuous force.
            location_x: Application point X (None = center of mass).
            location_y: Application point Y.
            location_z: Application point Z.

        Returns:
            Dict with actor_path, force_applied, force_vector, is_impulse.
        """

        location = (location_x, location_y, location_z)
        if any(v is not None for v in location) and not all(
            v is not None for v in location
        ):
            raise ValueError("Provide all three force application coordinates or none")
        name = "add_impulse" if is_impulse else "add_force"
        args = (
            f"unreal.Vector({float(force_x)!r}, {float(force_y)!r}, {float(force_z)!r})"
        )
        if location_x is not None and location_y is not None and location_z is not None:
            name += "_at_location"
            args += f", unreal.Vector({float(location_x)!r}, {float(location_y)!r}, {float(location_z)!r})"
        code = (
            "if not component.is_simulating_physics():\n"
            "    raise RuntimeError('Component is not simulating physics')\n"
            f"component.{name}({args})\n"
        )
        return await self._component_operation(
            actor_path,
            code,
            {
                "actor_path": actor_path,
                "force_applied": True,
                "force_vector": (force_x, force_y, force_z),
                "is_impulse": is_impulse,
            },
        )

    async def set_physics_params(
        self,
        actor_path: str,
        mass: Optional[float] = None,
        linear_damping: Optional[float] = None,
        angular_damping: Optional[float] = None,
        enable_gravity: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Set physics body parameters on an actor and read them back.

        Writes the actor's primitive component in one undoable editor
        transaction that marks the map dirty (a path in a running simulation
        edits the play copy, which is neither undoable nor saved), then reads
        every value back. A Static or Stationary body keeps the values, but
        Unreal gives it no mass and never simulates it until it is Movable,
        which the reply's warning says. With no parameter given it only reads
        the body.

        Args:
            actor_path: Full actor object path.
            mass: Mass override in kg, from 0.001 (Unreal's least) to 1e9.
            linear_damping: Linear damping, 0 to 1e6.
            angular_damping: Angular damping, 0 to 1e6.
            enable_gravity: Whether gravity affects this body.

        Returns:
            Dict with actor_path, world, applied (each value given, read back:
            mass as the override stored, as the 32-bit float Unreal keeps),
            params_set (how many read back as given), the body as
            ``enable_physics`` reads it (mass_kg is the mass Unreal computes,
            0 for a Static body) and warning; or a PhysicsParamsNotApplied
            envelope naming a value that read back different (the others stay
            written).

        Raises:
            ValueError: A value is not finite or out of its range.
        """
        values: Dict[str, Any] = {}
        if mass is not None:
            if not (math.isfinite(mass) and PHYSICS_MIN_MASS_KG <= mass <= PHYSICS_MAX_MASS_KG):
                raise ValueError(
                    f"mass must be from {PHYSICS_MIN_MASS_KG:g} to {PHYSICS_MAX_MASS_KG:g} kg, got {mass!r}"
                )
            values["mass"] = float(mass)
        for name, value in (("linear_damping", linear_damping), ("angular_damping", angular_damping)):
            if value is not None:
                if not (math.isfinite(value) and 0.0 <= value <= PHYSICS_MAX_DAMPING):
                    raise ValueError(f"{name} must be from 0 to {PHYSICS_MAX_DAMPING:g}, got {value!r}")
                values[name] = float(value)
        if enable_gravity is not None:
            values["enable_gravity"] = bool(enable_gravity)
        args = {"actor_path": actor_path, "values": values, "transaction": "simul: set physics params"}
        return await self._execute_json_script(
            f"args = {args!r}\n" + SET_PHYSICS_PARAMS, expected_errors=("PhysicsParamsNotApplied",)
        )

    # ------------------------------------------------------------------
    # Coordinate conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def ue_to_usd_location(
        x_cm: float, y_cm: float, z_cm: float
    ) -> tuple:
        """
        Convert UE5 location (Z-up, cm, left-hand) to USD (Z-up, m, right-hand).

        UE5 left-hand → USD right-hand: negate Y.
        Scale: cm → m (divide by 100).

        Args:
            x_cm: X coordinate in centimetres.
            y_cm: Y coordinate in centimetres.
            z_cm: Z coordinate in centimetres.

        Returns:
            Tuple (x_m, y_m, z_m) in metres, right-hand Z-up.
        """
        return (x_cm / 100.0, -y_cm / 100.0, z_cm / 100.0)

    @staticmethod
    def usd_to_ue_location(
        x_m: float, y_m: float, z_m: float
    ) -> tuple:
        """
        Convert USD location (Z-up, m, right-hand) to UE5 (Z-up, cm, left-hand).

        Args:
            x_m: X coordinate in metres.
            y_m: Y coordinate in metres.
            z_m: Z coordinate in metres.

        Returns:
            Tuple (x_cm, y_cm, z_cm) in centimetres, left-hand Z-up.
        """
        return (x_m * 100.0, -y_m * 100.0, z_m * 100.0)

    @staticmethod
    def ue_to_usd_rotation(
        pitch: float, yaw: float, roll: float
    ) -> tuple:
        """
        Convert UE5 rotation to USD rotation.

        UE5 left-hand → USD right-hand: negate Yaw.

        Args:
            pitch: Pitch in degrees.
            yaw: Yaw in degrees.
            roll: Roll in degrees.

        Returns:
            Tuple (pitch, yaw, roll) in degrees for USD.
        """
        return (pitch, -yaw, roll)

    @staticmethod
    def usd_to_ue_rotation(
        pitch: float, yaw: float, roll: float
    ) -> tuple:
        """
        Convert USD rotation to UE5 rotation.

        Args:
            pitch: Pitch in degrees.
            yaw: Yaw in degrees.
            roll: Roll in degrees.

        Returns:
            Tuple (pitch, yaw, roll) in degrees for UE5.
        """
        return (pitch, -yaw, roll)

    # ------------------------------------------------------------------
    # Phase 6: USD / SimReady Bridge
    # ------------------------------------------------------------------

    def _import_export_timeout(self) -> float:
        """Seconds the import and export tools wait: ``unreal.import_export_timeout``, at least ``timeout``."""
        return float(max(self.timeout, self.settings.unreal.import_export_timeout))

    async def _run_import_export(
        self, code: str, work: str, expected_errors: Sequence[str] = ()
    ) -> Dict[str, Any]:
        """Run an import or export script with ``unreal.import_export_timeout``.

        Remote Control cannot cancel a script it stopped waiting for, so a
        timeout is reported as work that may still finish: the first USD
        export of an editor session took 26-40 s against the 30 s default and
        came back as a bare TimeoutError while its files were still written
        (observed on UE 5.7.4).

        Args:
            code: The script.
            work: What the script does, completing "... may still complete":
                "the export to /tmp/x.usda".
            expected_errors: ``error_type`` values the script reports for a
                request it refused as asked.

        Returns:
            The script's result, or a ``TimeoutError`` envelope.
        """
        timeout = self._import_export_timeout()
        try:
            return await self._execute_json_script(code, expected_errors=expected_errors, timeout=timeout)
        except asyncio.TimeoutError:
            self.logger.warning("Unreal did not answer within %gs; %s may still complete", timeout, work)
            return {
                "success": False,
                "error_type": "TimeoutError",
                "error": (
                    f"Unreal did not answer within {timeout:g} s, but the editor keeps running the script, "
                    f"so {work} may still complete. Check the result before retrying: a retry runs it again."
                ),
                "details": {
                    "timeout_seconds": timeout,
                    "hint": "Raise UNREAL__IMPORT_EXPORT_TIMEOUT (unreal.import_export_timeout) for large files "
                    "or levels.",
                },
            }

    async def import_usd(
        self,
        usd_path: str,
        destination_path: str = "/Game/Imports",
        import_options: Optional[Dict[str, Any]] = None,
        replace_existing: bool = False,
    ) -> Dict[str, Any]:
        """
        Import a USD file with the optional USDImporter plugin.

        The assets go to ``<destination_path>/<file name>`` and, with
        import_actors, the actors to the level being edited. Without
        ``replace_existing`` the import is refused when that folder already
        holds assets, and existing actors are kept (new ones are added beside
        them): Unreal's USD importer otherwise replaces both, an earlier
        import's actors and customized assets included. The options are set
        without saving them to the project's config, where the editor's own
        import dialog would read them. The call waits
        ``unreal.import_export_timeout``.

        Args:
            usd_path: Path to the USD file on disk.
            destination_path: Content browser destination.
            import_options: Interchange pipeline options.
            replace_existing: Replace the assets in the destination folder and
                the level's actors from an earlier import of this file.

        Returns:
            Dict with imported_assets, actor_paths (actors added to the level),
            removed_actor_paths (actors the import replaced) and warnings, or
            an error envelope: ``InvalidState`` when the folder already holds
            assets and ``replace_existing`` is false (nothing is imported),
            ``NotFoundError`` for a file the editor host does not have,
            ``EngineError`` when the importer produced nothing, ``PIEActive``,
            ``PluginUnavailable`` and ``TimeoutError``.

        Raises:
            SandboxDenied: If ``usd_path`` is outside the sandbox.
        """

        usd_path = self._path_policy.authorize(usd_path)
        if not destination_path.startswith("/Game/") or ".." in destination_path.split(
            "/"
        ):
            raise ValueError("destination_path must be a /Game/ content directory")
        if type(replace_existing) is not bool:
            raise ValueError("replace_existing must be true or false")
        options = import_options or {}
        allowed = {
            "import_actors",
            "import_geometry",
            "import_materials",
            "import_lights",
            "import_cameras",
        }
        if set(options) - allowed or any(type(v) is not bool for v in options.values()):
            raise ValueError(
                f"import_options accepts boolean values for: {sorted(allowed)}"
            )
        args = {
            "path": usd_path,
            "destination": destination_path.rstrip("/"),
            "name": _sanitized_object_name(Path(usd_path).stem),
            "options": options,
            "replace": replace_existing,
        }
        return await self._run_import_export(
            f"args = {args!r}\n" + USD_IMPORT,
            f"the import of {usd_path}",
            expected_errors=("NotFoundError", "InvalidState", "PluginUnavailable"),
        )

    async def export_usd(
        self,
        actor_paths: List[str],
        output_path: str,
        export_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Export selected actors with the optional USDImporter plugin.

        Every actor must be in the level being edited; a path in another map
        is refused without loading that map. The actors exported are read
        back from the written stage: Unreal's exporter leaves out what it does
        not export (editor helpers such as WorldSettings or PlayerStart)
        without an error, and an export that wrote none of them fails. The
        call waits ``unreal.import_export_timeout``, since the first USD
        export of an editor session loads the USD Python modules (25-40 s).

        Args:
            actor_paths: Actor paths to export.
            output_path: Output USD file path.
            export_options: Export pipeline options.

        Returns:
            Dict with output_path, actors_exported (the actors the stage has a
            prim for), exported_prims (each one's actor_path and prim_path),
            skipped_actors (the actors it has none for) and file_size_bytes,
            or an error envelope: ``EngineError`` when Unreal exported nothing
            or failed, ``PIEActive``, ``PluginUnavailable`` and
            ``TimeoutError``.

        Raises:
            SandboxDenied: If ``output_path`` is outside the sandbox.
        """

        output_path = self._path_policy.authorize(output_path, write=True)
        if not actor_paths or any(not path for path in actor_paths):
            raise ValueError("At least one actor path is required")
        if Path(output_path).suffix.lower() not in {".usd", ".usda", ".usdc"}:
            raise ValueError("Output must have a .usd, .usda or .usdc extension")
        options = export_options or {}
        allowed = {"export_actor_folders", "export_sublayers"}
        if set(options) - allowed or any(type(v) is not bool for v in options.values()):
            raise ValueError(
                f"export_options accepts boolean values for: {sorted(allowed)}"
            )
        args = {"path": output_path, "actors": actor_paths, "options": options}
        return await self._run_import_export(
            f"args = {args!r}\n" + USD_EXPORT,
            f"the export to {output_path}",
            expected_errors=("PluginUnavailable",),
        )

    async def convert_to_simready(
        self,
        usd_path: str,
        output_path: str,
        add_physics: bool = True,
        add_collision: bool = True,
        add_semantic_labels: bool = True,
        target_up_axis: str = "Z",
        target_units: str = "meters",
    ) -> Dict[str, Any]:
        """
        Compatibility entry point for unavailable SimReady conversion.

        Returns UnsupportedOperation; no file is read or written.

        Args:
            usd_path: Source USD file.
            output_path: Output SimReady USD.
            add_physics: Add physics schema.
            add_collision: Generate collision geometry.
            add_semantic_labels: Add semantic labels.
            target_up_axis: Target up axis.
            target_units: Target unit system.

        Returns:
            Dict with output_path, conversions_applied, warnings.

        Raises:
            SandboxDenied: If either path is outside the sandbox.
        """

        self._path_policy.authorize(usd_path)
        self._path_policy.authorize(output_path, write=True)
        return {
            "success": False,
            "error_type": "UnsupportedOperation",
            "error": "SimReady conversion is not implemented. Unreal Interchange has no ConvertToSimReady API.",
        }

    async def validate_simready_asset(
        self,
        usd_path: str,
        checks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compatibility entry point for unavailable SimReady validation.

        Args:
            usd_path: USD file path to validate.
            checks: Validation checks to run.

        Returns:
            Dict with is_valid, per-check results, errors, suggestions.
        """

        self._path_policy.authorize(usd_path)
        return {
            "success": False,
            "error_type": "UnsupportedOperation",
            "error": "SimReady validation is not implemented. Unreal Interchange has no ValidateSimReadyAsset API.",
        }

    async def get_interchange_info(self) -> Dict[str, Any]:
        """
        Query supported Interchange formats and USD plugin availability.

        Returns:
            Dict with pipelines, supported_formats, interchange_version.
        """

        return await self._execute_json_script(INTERCHANGE_INFO)

    # ------------------------------------------------------------------
    # Phase 7: Advanced Agent Tools
    # ------------------------------------------------------------------

    async def batch_operations(
        self,
        operations: Any,
    ) -> Dict[str, Any]:
        """
        Execute multiple Remote Control operations in one HTTP call.

        Each operation is sent as ``{"RequestId", "URL", "Verb", "Body"}`` in
        that order, with a unique RequestId: Remote Control hands each Body to
        the RequestId it read last before it (UE 5.7
        GetBatchRequestStructDelimiters), so an operation without one, a
        repeated one or one written after its Body would run with another
        operation's body. Operations without a RequestId get the next free
        ones, in order.

        Args:
            operations: List of operation objects, or that list as JSON text.
                Each has ``URL`` (a Remote Control route such as
                ``/remote/object/call``), ``Verb`` (``GET``, ``PUT``, ...), an
                optional ``Body`` object and an optional integer
                ``RequestId``; key names ignore case, as Remote Control's do.

        Returns:
            Dict with results (Remote Control's per-operation answers, each
            failed one also carrying ``error`` and ``error_type``), total,
            succeeded and failed; or a ValidationError envelope when an
            operation is malformed, before anything is sent.
        """
        if not self.settings.security.allow_script_execution:
            return self._script_execution_denied()
        if isinstance(operations, str):
            try:
                operations = json.loads(operations)
            except ValueError as exc:
                return {
                    "success": False,
                    "error": f"operations is not valid JSON: {exc}",
                    "error_type": "ValidationError",
                }
        requests, problem = _batch_requests(operations)
        if problem is not None:
            return {"success": False, "error": problem, "error_type": "ValidationError"}
        response = await self._http_put("/remote/batch", {"Requests": requests})
        responses = response.get("Responses", [])
        results: List[Any] = []
        for answer in responses:
            code = answer.get("ResponseCode", 500) if isinstance(answer, dict) else 500
            if isinstance(answer, dict) and isinstance(code, int) and code >= 400:
                raw = answer.get("ResponseBody")
                message = remote_control_message(raw if isinstance(raw, str) else json.dumps(raw))
                answer = {
                    **answer,
                    "error": message or f"HTTP {code}",
                    "error_type": remote_control_error_type(code, message),
                }
            results.append(answer)
        succeeded = sum(
            1
            for r in results
            if isinstance(r, dict) and isinstance(r.get("ResponseCode"), int) and r["ResponseCode"] < 400
        )
        return {
            "results": results,
            "total": len(requests),
            "succeeded": succeeded,
            "failed": len(requests) - succeeded,
        }

    async def query_scene_graph(
        self,
        root_path: Optional[str] = None,
        max_depth: int = 10,
        include_components: bool = False,
        class_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Query the scene graph hierarchy starting from a root actor.

        Args:
            root_path: Root actor path (None for level root).
            max_depth: Maximum traversal depth.
            include_components: Include component sub-trees.
            class_filter: Filter actors by UClass name.

        Returns:
            Dict with root tree, total_actors, total_depth.
        """
        # Fetch all actors first
        search_body: Dict[str, Any] = {
            "Query": "",
            "Filter": "Actor",
        }
        if class_filter:
            search_body["Filter"] = class_filter

        response = await self._http_put("/remote/search/assets", search_body)
        actors = response.get("Assets", [])

        # Build tree
        tree: Dict[str, Any] = {"name": "Root", "path": root_path or "/", "children": []}
        depth = 0
        for actor in actors[:max_depth * 50]:
            path = actor.get("Path", "")
            if root_path and not path.startswith(root_path):
                continue
            node = {
                "name": actor.get("Name", ""),
                "path": path,
                "class": actor.get("Class", ""),
                "children": [],
            }
            tree["children"].append(node)

        return {
            "root": tree,
            "total_actors": len(tree["children"]),
            "total_depth": min(depth + 1, max_depth),
        }

    async def analyze_scene_for_robotics(
        self,
        analysis_types: Optional[List[str]] = None,
        actor_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Classify level actors for robotics from their collision setup.

        One editor script reads every actor loaded in the editor world. Only
        primitive components with collision enabled count, an actor's size is
        the bounding box of those components, and the thresholds are
        ``ROBOTICS_CRITERIA``, returned as ``criteria``:

        * traversability: floor-like surfaces a robot can stand on. Physics
          collision, at least 1 m^2 in plan, no thicker than a quarter of the
          shorter side and tilted at most 15 degrees. The bounding box decides,
          so a landscape section counts whole, hills included.
        * graspability: Movable actors with physics collision that fit in a
          60 cm cube, with whether they simulate physics and their mass (null
          while Unreal reports none).
        * collision_complexity: actors with collision and with physics
          collision, collision-enabled components by profile, landscape
          heightfields, and for static meshes the simple shapes and convex
          hulls of the mesh asset, how many use the render triangles as
          simple collision (``complex_collision``) and how many have physics
          collision but no simple shape (``missing_simple_collision``; physics
          bodies pass through them). The first 50 actors with that last
          problem are listed.

        Args:
            analysis_types: Any of traversability, graspability and
                collision_complexity; None or empty runs all three.
            actor_filter: Only analyze actors whose label or object path
                contains this text, ignoring case; None or empty analyzes
                every actor.

        Returns:
            Dict with traversable_surfaces and graspable_objects (each entry
            name, path, class_name and size [x, y, z] in cm; surfaces add
            top_z, area_m2 and tilt_degrees), collision_summary (empty unless
            collision_complexity ran), total_actors_analyzed, analysis_types
            (those that ran), criteria and pie_active (while Play In Editor
            runs, the editor world is read, as in ``list_actors``, and the
            static mesh collision counts simple_shapes, convex_hulls,
            complex_collision and missing_simple_collision are null, as is
            missing_simple_collision_actors: Unreal's StaticMeshEditorSubsystem
            reports none then); or a ScriptError envelope when the editor
            script failed.

        Raises:
            ValueError: An analysis type is not one of the three.
        """
        requested = [name.strip().lower() for name in analysis_types or [] if name.strip()]
        unknown = [name for name in requested if name not in ROBOTICS_ANALYSES]
        if unknown:
            raise ValueError(
                f"analysis_types: unknown {', '.join(map(repr, unknown))}; use "
                f"{', '.join(ROBOTICS_ANALYSES)}"
            )
        analyses = [name for name in ROBOTICS_ANALYSES if name in requested or not requested]
        args = {
            "analyses": analyses,
            "actor_filter": actor_filter or None,
            "criteria": ROBOTICS_CRITERIA,
            "listed_limit": ROBOTICS_LISTED_LIMIT,
        }
        result = await self._execute_json_script(f"args = {args!r}\n" + ROBOTICS_SCENE_ANALYSIS)
        if result.get("error"):
            return result
        return {**result, "analysis_types": analyses, "criteria": dict(ROBOTICS_CRITERIA)}

    async def generate_procedural_scene(
        self,
        scene_type: str,
        parameters: Optional[Dict[str, Any]] = None,
        bounds_min: Optional[List[float]] = None,
        bounds_max: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Build a test scene from Engine basic shapes inside an axis-aligned box.

        The layout is planned here (``_unreal_procedural.plan_scene``, which
        documents every scene type and parameter), so a layout that does not
        fit fails before anything is spawned and the same arguments always
        give the same scene. One editor script then spawns each piece as a
        StaticMeshActor inside one undoable transaction, labelled
        ``<Scene>_<Kind>_<n>``, tagged ``simul_procedural`` and with this
        call's own tag, ``simul_<scene_type>_<seed>_<generation>``, and filed
        in this call's own outliner folder, ``Simul/<Scene>_<seed>_<generation>``,
        where ``<generation>`` is fresh for every call
        (``_unreal_procedural.new_generation``): generating a scene twice
        gives two groups that never mix. When a spawn fails, rollback removes
        every actor spawned and cancels the transaction, and no folder is
        made (actors are filed only once all have spawned).

        Args:
            scene_type: room, corridor, warehouse or outdoor.
            parameters: The scene type's parameters, plus ``seed`` (an
                integer from 0 to 2**32 - 1, default 42) for the random sizes
                and positions.
            bounds_min: Minimum corner [x, y, z] in cm; default [0, 0, 0].
            bounds_max: Maximum corner [x, y, z] in cm; default
                [1000, 1000, 500].

        Returns:
            Dict with actors_spawned (actor paths in spawn order),
            total_spawned, scene_type, seed, tag (this call's own tag) and
            folder; a SpawnError envelope, with nothing left in the level, when
            a spawn failed (``details.actors_left`` lists any actor rollback
            could not remove); or a PIEActive envelope while Play In Editor
            runs, when Unreal's editor spawns nothing.

        Raises:
            ValueError: An unknown scene type or parameter, a value out of
                range, or a layout that does not fit the bounds.
        """
        plan = plan_scene(scene_type, parameters, bounds_min, bounds_max)
        generation = new_generation()
        tag, folder = plan.tag(generation), plan.folder(generation)
        args = {
            "pieces": [piece.spawn_args() for piece in plan.pieces],
            "folder": folder,
            "tags": ["simul_procedural", tag],
            "transaction": f"simul: generate {plan.scene_type} scene",
        }
        result = await self._execute_json_script(f"args = {args!r}\n" + PLACE_BASIC_SHAPES)
        if result.get("error"):
            return result
        spawned = result.get("actors_spawned")
        if not isinstance(spawned, list) or len(spawned) != len(plan.pieces):
            return self._script_error(
                f"Unreal reported {spawned!r} for {len(plan.pieces)} planned actors"
            )
        return {
            "actors_spawned": spawned,
            "total_spawned": len(spawned),
            "scene_type": plan.scene_type,
            "seed": plan.seed,
            "tag": tag,
            "folder": folder,
        }

    async def get_actor_by_semantic_label(
        self,
        label: str,
        match_mode: str = "exact",
        max_results: int = 100,
    ) -> Dict[str, Any]:
        """
        Find level actors whose actor tags match a semantic label.

        Semantic labels are actor tags (Actor > Tags in the Details panel).
        Matching ignores case, as Unreal's own tag comparison does. One editor
        script reads every actor's tags.

        Args:
            label: Semantic label (actor tag) to search for.
            match_mode: exact (a tag equals label), contains (a tag contains
                label) or regex (label is a Python regular expression searched
                for in each tag; a repeated group that holds a quantifier or
                alternatives, as in (\\w+)+, is refused because it can run for
                hours); every mode ignores case.
            max_results: Maximum number of actors to return, 1 to 5000.

        Returns:
            Dict with ``actors`` (each match as a ``list_unreal_actors``
            entry, tags included), ``total_matches`` (every matching actor,
            those past max_results too), ``truncated``, ``label_searched`` and
            ``pie_active`` (while Play In Editor runs, the editor world is
            searched, as in ``list_actors``), or a ``ScriptError`` envelope
            when the editor script failed.

        Raises:
            ValueError: match_mode is not exact, contains or regex; in regex
                mode, label is not a valid regular expression or repeats a
                group that holds a quantifier or alternatives (see
                ``simul.utils.regex_guard``); or max_results is out of range.
        """
        modes = ("exact", "contains", "regex")
        if match_mode not in modes:
            raise ValueError(
                f"match_mode must be one of {', '.join(modes)}, got {match_mode!r}"
            )
        if not 1 <= max_results <= 5000:
            raise ValueError("max_results must be between 1 and 5000")
        if match_mode == "regex":
            try:
                re.compile(label)
            except re.error as exc:
                raise ValueError(
                    f"label is not a valid regular expression: {exc}"
                ) from exc
            construct = _exponential_regex_construct(label)
            if construct is not None:
                raise ValueError(
                    f"label has {construct}: a failing search for it can freeze "
                    "the editor for hours, since every way of splitting a tag "
                    "between the repetitions is tried. Repeat a plain character "
                    "class instead (\\w+ rather than (\\w+)+), or use contains mode"
                )
        args = {"label": label, "match_mode": match_mode, "max_results": max_results}
        return await self._execute_json_script(
            f"args = {args!r}\n" + FIND_ACTORS_BY_TAG
        )

    # ------------------------------------------------------------------
    # Phase 8: Geometry & Modeling (GeometryScript)
    # ------------------------------------------------------------------

    async def generate_mesh_primitive(
        self,
        primitive_type: str,
        dimensions: Optional[Dict[str, float]] = None,
        segments: int = 32,
        location: Optional[List[float]] = None,
        actor_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a parametric mesh primitive via GeometryScript.

        Args:
            primitive_type: box, sphere, cylinder, cone, torus, capsule.
            dimensions: Type-specific dimensions.
            segments: Tessellation segments.
            location: Spawn location [x, y, z] in cm.
            actor_label: Optional label for the actor.

        Returns:
            Dict with actor_path, primitive_type, triangle/vertex counts.
        """
        if dimensions is None:
            dimensions = {}
        if location is None:
            location = [0.0, 0.0, 0.0]
        label = actor_label or f"DynMesh_{primitive_type}"
        seg = max(segments, 4)
        d = {key: float(value) for key, value in dimensions.items()}

        # Build the append call depending on primitive_type
        ptype = primitive_type.lower()
        if ptype == "box":
            w = d.get("width", 100.0)
            h = d.get("height", 100.0)
            dp = d.get("depth", 100.0)
            append_line = (
                f"gs_prim.append_box(mesh, opts, xf, {w}, {h}, {dp},"
                f" 1, 1, 1, origin)"
            )
        elif ptype == "sphere":
            r = d.get("radius", 50.0)
            append_line = (
                f"gs_prim.append_sphere_lat_long(mesh, opts, xf,"
                f" {r}, {seg}, {seg}, origin)"
            )
        elif ptype == "cylinder":
            r = d.get("radius", 50.0)
            h = d.get("height", 100.0)
            append_line = (
                f"gs_prim.append_cylinder(mesh, opts, xf,"
                f" {r}, {h}, {seg}, 1, True, origin)"
            )
        elif ptype == "cone":
            r = d.get("base_radius", 50.0)
            h = d.get("height", 100.0)
            append_line = (
                f"gs_prim.append_cone(mesh, opts, xf,"
                f" {r}, 0.0, {h}, {seg}, 1, True, origin)"
            )
        elif ptype == "capsule":
            r = d.get("radius", 50.0)
            h = d.get("hemisphere_height", 50.0)
            append_line = (
                f"gs_prim.append_capsule(mesh, opts, xf,"
                f" {r}, {h}, {h}, {seg}, {max(seg // 2, 4)}, origin)"
            )
        elif ptype == "torus":
            inner = d.get("inner_radius", 20.0)
            outer = d.get("outer_radius", 50.0)
            append_line = (
                f"gs_prim.append_torus(mesh, opts, xf,"
                f" {inner}, {outer}, {seg}, {seg}, origin)"
            )
        else:
            return {
                "success": False,
                "error": f"Unsupported primitive type: {primitive_type}",
            }

        lx, ly, lz = (float(value) for value in location)
        code = (
            "import unreal, json\n"
            "gs_prim = unreal.GeometryScript_Primitives\n"
            "gs_q = unreal.GeometryScript_MeshQueries\n"
            "mesh = unreal.DynamicMesh()\n"
            "opts = unreal.GeometryScriptPrimitiveOptions()\n"
            "opts.polygroup_mode = "
            "unreal.GeometryScriptPrimitivePolygroupMode.PER_FACE\n"
            "xf = unreal.Transform()\n"
            "origin = unreal.GeometryScriptPrimitiveOriginMode.CENTER\n"
            f"{append_line}\n"
            "subsys = unreal.get_editor_subsystem("
            "unreal.EditorActorSubsystem)\n"
            f"actor = subsys.spawn_actor_from_class("
            f"unreal.DynamicMeshActor, "
            f"unreal.Vector({lx}, {ly}, {lz}))\n"
            f"actor.set_actor_label({label!r})\n"
            "comp = actor.dynamic_mesh_component\n"
            "comp.set_dynamic_mesh(mesh)\n"
            "tris = gs_q.get_num_triangle_i_ds(mesh)\n"
            "verts = gs_q.get_vertex_count(mesh)\n"
            "print(json.dumps({"
            "'actor_path': actor.get_path_name(),"
            f"'primitive_type': {ptype!r},"
            "'triangle_count': tris,"
            "'vertex_count': verts"
            "}))\n"
        )
        result = await self._execute_python(_outside_pie(code, "spawn actors"))
        return self._parse_python_json(result)

    async def apply_mesh_boolean(
        self,
        target_mesh_path: str,
        tool_mesh_path: str,
        operation: str,
    ) -> Dict[str, Any]:
        """
        Apply boolean operation between two meshes via GeometryScript.

        Args:
            target_mesh_path: Target mesh actor (modified in-place).
            tool_mesh_path: Tool mesh actor (operand).
            operation: union, subtract, or intersect.

        Returns:
            Dict with target_mesh_path, operation, result triangle/vertex counts.
        """
        op_map = {
            "union": "UNION",
            "subtract": "SUBTRACT",
            "intersect": "INTERSECTION",
        }
        ue_op = op_map.get(operation.lower(), "UNION")
        code = (
            "    gs_bool = unreal.GeometryScript_MeshBooleans\n"
            "    gs_q = unreal.GeometryScript_MeshQueries\n"
            "    t_mesh = target_actor.dynamic_mesh_component"
            ".get_dynamic_mesh()\n"
            "    tool_mesh = tool_actor.dynamic_mesh_component"
            ".get_dynamic_mesh()\n"
            "    bool_opts = unreal.GeometryScriptMeshBooleanOptions()\n"
            "    gs_bool.apply_mesh_boolean(\n"
            "        t_mesh, unreal.Transform(),\n"
            "        tool_mesh, unreal.Transform(),\n"
            f"        unreal.GeometryScriptBooleanOperation.{ue_op},"
            " bool_opts)\n"
            "    target_actor.dynamic_mesh_component"
            ".notify_mesh_modified()\n"
            "    tris = gs_q.get_num_triangle_i_ds(t_mesh)\n"
            "    verts = gs_q.get_vertex_count(t_mesh)\n"
            "    print(json.dumps({"
            f"'target_mesh_path': {target_mesh_path!r},"
            f"'operation': {operation!r},"
            "'result_triangle_count': tris,"
            "'result_vertex_count': verts"
            "}))\n"
        )
        return await self._run_mesh_script(code, target_actor=target_mesh_path, tool_actor=tool_mesh_path)

    async def compute_convex_hull(
        self,
        mesh_path: str,
    ) -> Dict[str, Any]:
        """
        Compute convex hull envelope of a mesh.

        Args:
            mesh_path: Source mesh actor path.

        Returns:
            Dict with hull actor path, vertex/triangle counts.
        """
        code = (
            "    gs_c = unreal.GeometryScript_Containment\n"
            "    gs_q = unreal.GeometryScript_MeshQueries\n"
            "    mesh = src.dynamic_mesh_component.get_dynamic_mesh()\n"
            "    hull_mesh = unreal.DynamicMesh()\n"
            "    sel = unreal.GeometryScriptMeshSelection()\n"
            "    opts = unreal.GeometryScriptConvexHullOptions()\n"
            "    gs_c.compute_mesh_convex_hull("
            "mesh, hull_mesh, sel, opts)\n"
            "    hull_actor = subsys.spawn_actor_from_class("
            "unreal.DynamicMeshActor, src.get_actor_location())\n"
            "    hull_actor.set_actor_label("
            "src.get_actor_label() + '_Hull')\n"
            "    hull_actor.dynamic_mesh_component"
            ".set_dynamic_mesh(hull_mesh)\n"
            "    h_tris = gs_q.get_num_triangle_i_ds(hull_mesh)\n"
            "    h_verts = gs_q.get_vertex_count(hull_mesh)\n"
            "    s_tris = gs_q.get_num_triangle_i_ds(mesh)\n"
            "    print(json.dumps({"
            f"'mesh_path': {mesh_path!r},"
            "'hull_actor_path': hull_actor.get_path_name(),"
            "'hull_vertex_count': h_verts,"
            "'hull_triangle_count': h_tris,"
            "'source_triangle_count': s_tris"
            "}))\n"
        )
        return await self._run_mesh_script(code, pie_action="spawn actors", src=mesh_path)

    async def decompose_convex_hull(
        self,
        mesh_path: str,
        max_hulls: int = 16,
        max_vertices_per_hull: int = 32,
        min_cluster_size: int = 256,
        resolution: int = 100000,
    ) -> Dict[str, Any]:
        """
        V-HACD convex decomposition for collision geometry.

        Args:
            mesh_path: Source mesh actor path.
            max_hulls: Maximum convex pieces.
            max_vertices_per_hull: Max vertices per hull.
            min_cluster_size: Minimum cluster size for V-HACD.
            resolution: Voxelization resolution.

        Returns:
            Dict with hull_count, triangle/vertex counts for combined hull.
        """
        # GeometryScriptConvexDecompositionOptions has:
        # num_hulls, error_tolerance, search_factor,
        # simplify_to_face_count, min_part_thickness
        code = (
            "    gs_c = unreal.GeometryScript_Containment\n"
            "    gs_q = unreal.GeometryScript_MeshQueries\n"
            "    mesh = src.dynamic_mesh_component.get_dynamic_mesh()\n"
            "    decomp_mesh = unreal.DynamicMesh()\n"
            "    opts = unreal.GeometryScriptConvexDecompositionOptions()\n"
            f"    opts.num_hulls = {max_hulls}\n"
            f"    opts.simplify_to_face_count = {max_vertices_per_hull}\n"
            "    gs_c.compute_mesh_convex_decomposition("
            "mesh, decomp_mesh, opts)\n"
            "    d_actor = subsys.spawn_actor_from_class("
            "unreal.DynamicMeshActor, src.get_actor_location())\n"
            "    d_actor.set_actor_label("
            "src.get_actor_label() + '_Decomp')\n"
            "    d_actor.dynamic_mesh_component"
            ".set_dynamic_mesh(decomp_mesh)\n"
            "    d_tris = gs_q.get_num_triangle_i_ds(decomp_mesh)\n"
            "    d_verts = gs_q.get_vertex_count(decomp_mesh)\n"
            "    n_comp = gs_q.get_num_connected_components("
            "decomp_mesh)\n"
            "    print(json.dumps({"
            f"'mesh_path': {mesh_path!r},"
            "'hull_count': n_comp,"
            "'decomp_actor_path': d_actor.get_path_name(),"
            "'total_triangles': d_tris,"
            "'total_vertices': d_verts"
            "}))\n"
        )
        return await self._run_mesh_script(code, pie_action="spawn actors", src=mesh_path)

    async def edit_mesh_topology(
        self,
        mesh_path: str,
        operation: str,
        face_selection: Optional[str] = None,
        edge_selection: Optional[str] = None,
        distance: Optional[float] = None,
        offset: Optional[float] = None,
        scale: Optional[List[float]] = None,
        count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Edit mesh topology (extrude, bevel, inset, loop cut, scale_faces).

        Args:
            mesh_path: Mesh actor path.
            operation: extrude_faces, bevel_edges, inset_faces,
                loop_cut, scale_faces.
            face_selection: Face selection filter (unused in current impl).
            edge_selection: Edge selection filter (unused in current impl).
            distance: Extrude/bevel distance.
            offset: Inset offset.
            scale: Scale factors [x, y, z] (unused in current impl).
            count: Subdivision count for bevel.

        Returns:
            Dict with mesh_path, operation, result triangle/vertex counts.
        """
        dist = distance if distance is not None else 20.0
        ofs = offset if offset is not None else 5.0
        subdivs = count if count is not None else 1
        op = operation.lower()

        # Build the operation-specific Python block
        if op == "extrude_faces":
            op_block = (
                "    ext_opts = unreal"
                ".GeometryScriptMeshLinearExtrudeOptions()\n"
                f"    ext_opts.distance = {dist}\n"
                "    sel = unreal.GeometryScriptMeshSelection()\n"
                "    gs_mod.apply_mesh_linear_extrude_faces("
                "mesh, ext_opts, sel)\n"
            )
        elif op == "bevel_edges":
            op_block = (
                "    bev_opts = unreal"
                ".GeometryScriptMeshBevelSelectionOptions()\n"
                f"    bev_opts.bevel_distance = {dist}\n"
                f"    bev_opts.subdivisions = {subdivs}\n"
                "    sel = unreal.GeometryScriptMeshSelection()\n"
                "    gs_mod.apply_mesh_bevel_selection(mesh, sel,\n"
                "        unreal.GeometryScriptMeshBevelSelectionMode"
                ".TRIANGLE_AREA, bev_opts)\n"
            )
        elif op == "inset_faces":
            op_block = (
                "    ins_opts = unreal"
                ".GeometryScriptMeshInsetOutsetFacesOptions()\n"
                f"    ins_opts.distance = {ofs}\n"
                "    sel = unreal.GeometryScriptMeshSelection()\n"
                "    gs_mod.apply_mesh_inset_outset_faces("
                "mesh, ins_opts, sel)\n"
            )
        else:
            return {
                "success": False,
                "error": f"Unsupported topology operation: {operation}",
            }

        code = (
            "    gs_mod = unreal.GeometryScript_MeshModeling\n"
            "    gs_q = unreal.GeometryScript_MeshQueries\n"
            "    mesh = actor.dynamic_mesh_component"
            ".get_dynamic_mesh()\n"
            "    before_tris = gs_q.get_num_triangle_i_ds(mesh)\n"
            + op_block
            + "    actor.dynamic_mesh_component"
            ".notify_mesh_modified()\n"
            "    tris = gs_q.get_num_triangle_i_ds(mesh)\n"
            "    verts = gs_q.get_vertex_count(mesh)\n"
            "    print(json.dumps({"
            f"'mesh_path': {mesh_path!r},"
            f"'operation': {operation!r},"
            "'result_triangle_count': tris,"
            "'result_vertex_count': verts,"
            "'previous_triangle_count': before_tris"
            "}))\n"
        )
        return await self._run_mesh_script(code, actor=mesh_path)

    async def subdivide_mesh(
        self,
        mesh_path: str,
        level: int = 2,
        scheme: str = "catmull_clark",
    ) -> Dict[str, Any]:
        """
        Catmull-Clark or uniform tessellation subdivision.

        Args:
            mesh_path: Mesh actor path.
            level: Subdivision level (1-4).
            scheme: catmull_clark, loop, or uniform.

        Returns:
            Dict with mesh_path, level, scheme, result triangle/vertex counts.
        """
        lvl = max(1, min(level, 4))
        sch = scheme.lower()

        if sch == "catmull_clark":
            subdiv_block = (
                "    gl = unreal.GeometryScriptGroupLayer()\n"
                "    unreal.GeometryScript_OpenSubdiv"
                f".apply_polygroup_catmull_clark_sub_d(mesh, {lvl}, gl)\n"
            )
        else:
            # uniform tessellation for loop / uniform / bilinear
            subdiv_block = (
                "    unreal.GeometryScript_MeshSubdivide"
                f".apply_uniform_tessellation(mesh, {lvl})\n"
            )

        code = (
            "    gs_q = unreal.GeometryScript_MeshQueries\n"
            "    mesh = actor.dynamic_mesh_component"
            ".get_dynamic_mesh()\n"
            "    before_tris = gs_q.get_num_triangle_i_ds(mesh)\n"
            + subdiv_block
            + "    actor.dynamic_mesh_component"
            ".notify_mesh_modified()\n"
            "    tris = gs_q.get_num_triangle_i_ds(mesh)\n"
            "    verts = gs_q.get_vertex_count(mesh)\n"
            "    print(json.dumps({"
            f"'mesh_path': {mesh_path!r},"
            f"'level': {lvl},"
            f"'scheme': {scheme!r},"
            "'result_triangle_count': tris,"
            "'result_vertex_count': verts,"
            "'previous_triangle_count': before_tris"
            "}))\n"
        )
        return await self._run_mesh_script(code, actor=mesh_path)

    async def simplify_mesh(
        self,
        mesh_path: str,
        target_triangle_count: Optional[int] = None,
        target_percentage: Optional[float] = None,
        max_error: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Simplify/decimate a mesh.

        Args:
            mesh_path: Mesh actor path.
            target_triangle_count: Target triangle count.
            target_percentage: Target percentage (0.0-1.0).
            max_error: Max geometric error tolerance.

        Returns:
            Dict with original/result triangles, reduction ratio.
        """
        # Determine target count: explicit, or derived from percentage
        target_line = ""
        if target_triangle_count is not None:
            target_line = f"    target_count = {target_triangle_count}\n"
        elif target_percentage is not None:
            target_line = (
                f"    target_count = max(1, int(before_tris"
                f" * {target_percentage}))\n"
            )
        else:
            target_line = "    target_count = max(1, before_tris // 2)\n"

        code = (
            "    gs_s = unreal.GeometryScript_MeshSimplification\n"
            "    gs_q = unreal.GeometryScript_MeshQueries\n"
            "    mesh = actor.dynamic_mesh_component"
            ".get_dynamic_mesh()\n"
            "    before_tris = gs_q.get_num_triangle_i_ds(mesh)\n"
            + target_line
            + "    opts = unreal.GeometryScriptSimplifyMeshOptions()\n"
            "    gs_s.apply_simplify_to_triangle_count("
            "mesh, target_count, opts)\n"
            "    actor.dynamic_mesh_component"
            ".notify_mesh_modified()\n"
            "    tris = gs_q.get_num_triangle_i_ds(mesh)\n"
            "    verts = gs_q.get_vertex_count(mesh)\n"
            "    ratio = round(1.0 - tris / before_tris, 4)"
            " if before_tris > 0 else 0.0\n"
            "    print(json.dumps({"
            f"'mesh_path': {mesh_path!r},"
            "'original_triangles': before_tris,"
            "'result_triangles': tris,"
            "'result_vertex_count': verts,"
            "'reduction_ratio': ratio"
            "}))\n"
        )
        return await self._run_mesh_script(code, actor=mesh_path)

    async def cut_mesh_plane(
        self,
        mesh_path: str,
        plane_origin: List[float],
        plane_normal: List[float],
        fill_holes: bool = True,
        keep_both_sides: bool = False,
    ) -> Dict[str, Any]:
        """
        Cut/slice a mesh along an arbitrary plane.

        Args:
            mesh_path: Mesh actor path.
            plane_origin: Plane origin [x, y, z] in cm.
            plane_normal: Plane normal [x, y, z].
            fill_holes: Fill cut holes with faces.
            keep_both_sides: Keep both halves (unused — UE plane cut
                keeps one side and optionally fills).

        Returns:
            Dict with mesh_path, result triangle/vertex counts.
        """
        ox, oy, oz = plane_origin[0], plane_origin[1], plane_origin[2]
        nx, ny, nz = plane_normal[0], plane_normal[1], plane_normal[2]
        fill_py = "True" if fill_holes else "False"
        code = (
            "    gs_bool = unreal.GeometryScript_MeshBooleans\n"
            "    gs_q = unreal.GeometryScript_MeshQueries\n"
            "    mesh = actor.dynamic_mesh_component"
            ".get_dynamic_mesh()\n"
            "    before_tris = gs_q.get_num_triangle_i_ds(mesh)\n"
            "    cut_frame = unreal.Transform()\n"
            f"    cut_frame.translation = "
            f"unreal.Vector({ox}, {oy}, {oz})\n"
            # Build rotation from normal — use MathLibrary.make_rot_from_z
            f"    normal = unreal.Vector({nx}, {ny}, {nz})\n"
            "    rot = unreal.MathLibrary.make_rot_from_z(normal)\n"
            "    cut_frame.rotation = rot.quaternion()\n"
            "    cut_opts = unreal"
            ".GeometryScriptMeshPlaneCutOptions()\n"
            f"    cut_opts.fill_holes = {fill_py}\n"
            "    gs_bool.apply_mesh_plane_cut("
            "mesh, cut_frame, cut_opts)\n"
            "    actor.dynamic_mesh_component"
            ".notify_mesh_modified()\n"
            "    tris = gs_q.get_num_triangle_i_ds(mesh)\n"
            "    verts = gs_q.get_vertex_count(mesh)\n"
            "    print(json.dumps({"
            f"'mesh_path': {mesh_path!r},"
            "'result_triangle_count': tris,"
            "'result_vertex_count': verts,"
            "'previous_triangle_count': before_tris"
            "}))\n"
        )
        return await self._run_mesh_script(code, actor=mesh_path)

    async def validate_mesh(
        self,
        mesh_path: str,
        checks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Validate mesh geometry using GeometryScript query functions.

        Args:
            mesh_path: Mesh actor path.
            checks: Checks to run (currently all checks always run).

        Returns:
            Dict with is_valid, triangle/vertex counts, border/component
            info.
        """
        code = (
            "    gs_q = unreal.GeometryScript_MeshQueries\n"
            "    mesh = actor.dynamic_mesh_component"
            ".get_dynamic_mesh()\n"
            # UFUNCTIONs with out-params (e.g. GetNumOpenBorderLoops'
            # bAmbiguousTopologyFound) come back as (count, *outs) tuples.
            "    def _count(r):\n"
            "        return r[0] if isinstance(r, tuple) else r\n"
            "    tris = _count(gs_q.get_num_triangle_i_ds(mesh))\n"
            "    verts = _count(gs_q.get_vertex_count(mesh))\n"
            "    open_edges = _count("
            "gs_q.get_num_open_border_edges(mesh))\n"
            "    open_loops = _count("
            "gs_q.get_num_open_border_loops(mesh))\n"
            "    components = _count("
            "gs_q.get_num_connected_components(mesh))\n"
            "    has_normals = gs_q.get_has_triangle_normals(mesh)\n"
            "    has_gaps = gs_q.get_has_triangle_id_gaps(mesh)\n"
            "    is_watertight = (open_edges == 0)\n"
            "    is_valid = is_watertight and not has_gaps\n"
            "    issues = []\n"
            "    if open_edges > 0:\n"
            "        issues.append("
            "f'Non-watertight: {open_edges} open border edges')\n"
            "    if open_loops > 0:\n"
            "        issues.append("
            "f'{open_loops} open border loops')\n"
            "    if has_gaps:\n"
            "        issues.append('Triangle ID gaps detected')\n"
            "    if not has_normals:\n"
            "        issues.append('Missing triangle normals')\n"
            "    print(json.dumps({"
            f"'mesh_path': {mesh_path!r},"
            "'is_valid': is_valid,"
            "'triangle_count': tris,"
            "'vertex_count': verts,"
            "'open_border_edges': open_edges,"
            "'open_border_loops': open_loops,"
            "'connected_components': components,"
            "'has_normals': has_normals,"
            "'issues': issues"
            "}))\n"
        )
        return await self._run_mesh_script(code, actor=mesh_path)

    async def convert_mesh_format(
        self,
        mesh_path: str,
        target_format: str,
        tessellation_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Convert between StaticMesh and DynamicMesh via GeometryScript.

        Args:
            mesh_path: Source mesh actor/asset path.
            target_format: 'static_mesh' or 'dynamic_mesh'.
            tessellation_options: Unused (reserved for future CAD support).

        Returns:
            Dict with source/result paths, formats, triangle count.
        """
        fmt = target_format.lower()
        if fmt == "static_mesh":
            # DynamicMeshActor → new StaticMesh asset
            code = (
                "    gs_a = unreal.GeometryScript_AssetUtils\n"
                "    gs_q = unreal.GeometryScript_MeshQueries\n"
                "    mesh = actor.dynamic_mesh_component"
                ".get_dynamic_mesh()\n"
                "    tris = gs_q.get_num_triangle_i_ds(mesh)\n"
                "    # Create a transient StaticMesh asset\n"
                "    pkg_name = '/Game/Generated/' + "
                "actor.get_actor_label()\n"
                "    sm = unreal.EditorAssetLibrary"
                ".does_asset_exist(pkg_name)\n"
                "    # For now report conversion info\n"
                "    print(json.dumps({"
                f"'source_path': {mesh_path!r},"
                "'target_format': 'static_mesh',"
                "'triangle_count': tris,"
                "'note': 'copy_mesh_to_static_mesh requires "
                "an existing StaticMesh asset target'"
                "}))\n"
            )
            return await self._run_mesh_script(code, actor=mesh_path)
        elif fmt == "dynamic_mesh":
            # StaticMesh asset → DynamicMeshActor in the level
            code = (
                "import unreal, json\n"
                "gs_a = unreal.GeometryScript_AssetUtils\n"
                "gs_q = unreal.GeometryScript_MeshQueries\n"
                "subsys = unreal.get_editor_subsystem("
                "unreal.EditorActorSubsystem)\n"
                f"sm_asset = unreal.EditorAssetLibrary"
                f".load_asset({mesh_path!r})\n"
                "if sm_asset is None:\n"
                "    print(json.dumps({'error': "
                f"{('Asset not found: ' + mesh_path)!r}"
                "}))\n"
                "else:\n"
                "    dyn_mesh = unreal.DynamicMesh()\n"
                "    asset_opts = unreal"
                ".GeometryScriptCopyMeshFromAssetOptions()\n"
                "    gs_a.copy_mesh_from_static_mesh("
                "sm_asset, dyn_mesh, asset_opts)\n"
                "    actor = subsys.spawn_actor_from_class("
                "unreal.DynamicMeshActor, unreal.Vector())\n"
                "    actor.set_actor_label("
                "sm_asset.get_name() + '_Dynamic')\n"
                "    actor.dynamic_mesh_component"
                ".set_dynamic_mesh(dyn_mesh)\n"
                "    tris = gs_q.get_num_triangle_i_ds(dyn_mesh)\n"
                "    verts = gs_q.get_vertex_count(dyn_mesh)\n"
                "    print(json.dumps({"
                f"'source_path': {mesh_path!r},"
                "'result_path': actor.get_path_name(),"
                "'target_format': 'dynamic_mesh',"
                "'triangle_count': tris,"
                "'vertex_count': verts"
                "}))\n"
            )
        else:
            return {
                "success": False,
                "error": f"Unsupported target format: {target_format}",
            }

        # Only dynamic_mesh gets here; it spawns the DynamicMeshActor.
        result = await self._execute_python(_outside_pie(code, "spawn actors"))
        return self._parse_python_json(result)

    async def remesh_mesh(
        self,
        mesh_path: str,
        mode: str = "uniform",
        target_edge_length: Optional[float] = None,
        target_triangle_count: Optional[int] = None,
        smoothing_iterations: int = 3,
    ) -> Dict[str, Any]:
        """
        Remesh for clean topology using GeometryScript_Remeshing.

        Args:
            mesh_path: Mesh actor path.
            mode: 'uniform' (only mode currently supported).
            target_edge_length: Target edge length in cm.
            target_triangle_count: Target triangle count.
            smoothing_iterations: Smoothing passes (remesh_iterations).

        Returns:
            Dict with original/result triangles, vertex count.
        """
        edge_len = target_edge_length if target_edge_length else 5.0
        tri_count = target_triangle_count if target_triangle_count else 0
        iters = max(1, smoothing_iterations)

        # Build target_type — use GeometryScriptUniformRemeshTargetType enum
        if target_triangle_count is not None:
            target_block = (
                f"    uni_opts.target_triangle_count = {tri_count}\n"
                "    uni_opts.target_type = unreal"
                ".GeometryScriptUniformRemeshTargetType"
                ".TRIANGLE_COUNT\n"
            )
        else:
            target_block = (
                f"    uni_opts.target_edge_length = {edge_len}\n"
                "    uni_opts.target_type = unreal"
                ".GeometryScriptUniformRemeshTargetType"
                ".TARGET_EDGE_LENGTH\n"
            )

        code = (
            "    gs_r = unreal.GeometryScript_Remeshing\n"
            "    gs_q = unreal.GeometryScript_MeshQueries\n"
            "    mesh = actor.dynamic_mesh_component"
            ".get_dynamic_mesh()\n"
            "    before_tris = gs_q.get_num_triangle_i_ds(mesh)\n"
            "    remesh_opts = unreal"
            ".GeometryScriptRemeshOptions()\n"
            f"    remesh_opts.remesh_iterations = {iters}\n"
            "    uni_opts = unreal"
            ".GeometryScriptUniformRemeshOptions()\n"
            + target_block
            + "    gs_r.apply_uniform_remesh("
            "mesh, remesh_opts, uni_opts)\n"
            "    actor.dynamic_mesh_component"
            ".notify_mesh_modified()\n"
            "    tris = gs_q.get_num_triangle_i_ds(mesh)\n"
            "    verts = gs_q.get_vertex_count(mesh)\n"
            "    print(json.dumps({"
            f"'mesh_path': {mesh_path!r},"
            f"'mode': {mode!r},"
            "'original_triangles': before_tris,"
            "'result_triangles': tris,"
            "'result_vertex_count': verts"
            "}))\n"
        )
        return await self._run_mesh_script(code, actor=mesh_path)

    async def compute_mesh_uv(
        self,
        mesh_path: str,
        method: str = "auto_uv",
        uv_channel: int = 0,
        island_padding: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Generate UV coordinates using GeometryScript_UVs.

        Args:
            mesh_path: Mesh actor path.
            method: 'xatlas' or 'patch_builder' (default xatlas).
            uv_channel: UV set index.
            island_padding: Unused (reserved for future option support).

        Returns:
            Dict with method, uv_channel, triangle/vertex counts.
        """
        m = method.lower()
        if m in ("patch_builder", "patchbuilder"):
            uv_block = (
                "    uv_opts = unreal"
                ".GeometryScriptPatchBuilderOptions()\n"
                f"    gs_uv.auto_generate_patch_builder_mesh_u_vs("
                f"mesh, {uv_channel}, uv_opts)\n"
            )
            method_label = "patch_builder"
        else:
            # Default to XAtlas for auto_uv, xatlas, atlas_pack, etc.
            uv_block = (
                "    uv_opts = unreal"
                ".GeometryScriptXAtlasOptions()\n"
                f"    gs_uv.auto_generate_x_atlas_mesh_u_vs("
                f"mesh, {uv_channel}, uv_opts)\n"
            )
            method_label = "xatlas"

        code = (
            "    gs_uv = unreal.GeometryScript_UVs\n"
            "    gs_q = unreal.GeometryScript_MeshQueries\n"
            "    mesh = actor.dynamic_mesh_component"
            ".get_dynamic_mesh()\n" + uv_block + "    actor.dynamic_mesh_component"
            ".notify_mesh_modified()\n"
            "    tris = gs_q.get_num_triangle_i_ds(mesh)\n"
            "    verts = gs_q.get_vertex_count(mesh)\n"
            "    print(json.dumps({"
            f"'mesh_path': {mesh_path!r},"
            f"'method': {method_label!r},"
            f"'uv_channel': {uv_channel},"
            "'triangle_count': tris,"
            "'vertex_count': verts"
            "}))\n"
        )
        return await self._run_mesh_script(code, actor=mesh_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying HTTP session if open.

        Await this on the event loop that made the session's requests,
        before that loop ends: aiohttp cannot close the connection pool
        afterwards and reports an unclosed client session and connector
        when the session is garbage-collected.
        """
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None
        self.logger.debug("Unreal runtime session closed")

    async def __aenter__(self) -> "UnrealRuntimeSession":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()

    def cleanup(self) -> None:
        """
        Synchronous cleanup hook for context-manager teardown.

        With an event loop running, schedules ``close()`` on it and holds the
        task until it finishes. Without one, any HTTP session was opened on
        a loop that has already ended; ``close()`` runs on a new loop, where
        aiohttp can still mark the pool closed but not reach the old loop's
        sockets, so async callers should ``await close()`` (or use
        ``async with``) instead.
        """
        if self._session is None or self._session.closed:
            self._session = None
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            try:
                asyncio.run(self.close())
            except Exception as exc:
                self.logger.debug("Closing the HTTP session off its loop failed: %s", exc)
            return
        task = loop.create_task(self.close())
        _PENDING_CLOSES.add(task)
        task.add_done_callback(_PENDING_CLOSES.discard)


class UnrealRuntimeAdapter(LoggerMixin):
    """Adapter for Unreal Engine runtime operations."""

    name: str = "unreal"

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """
        Initialize Unreal runtime adapter.

        Args:
            settings: Configuration settings.
        """
        self.settings: Settings = settings or get_settings()
        self.logger.info("Unreal runtime adapter initialized")

    @contextmanager
    def create_session(self) -> Any:
        """
        Create an Unreal runtime session context manager.

        Yields:
            UnrealRuntimeSession instance.
        """
        session = UnrealRuntimeSession(self.settings)
        try:
            yield session
        finally:
            session.cleanup()

    def is_available(self) -> bool:
        """
        Check whether the Unreal runtime is reachable.

        Returns:
            True when aiohttp or the embedded unreal module is available
            and the adapter is enabled in settings.
        """
        return UNREAL_AVAILABLE and self.settings.unreal.enabled

    def close(self) -> None:
        """Nothing is held between sessions; each one cleans up after itself."""

    def get_capabilities(self) -> List[str]:
        """
        Name the Unreal tools a server with this adapter's settings registers.

        Read from the tool declarations (``_unreal_tools.registered_tool_names``):
        the thin or full surface ``unreal.tool_surface`` selects, without the
        script tools while ``security.allow_script_execution`` is off, so a
        newly declared tool is reported without a second list to update.

        Returns:
            The tool names in registration order, or an empty list when the
            adapter is not available.
        """
        if not self.is_available():
            return []
        # The declarations live in the MCP registration layer, which imports
        # this module lazily as well; neither needs the other at import time.
        from ..mcp.registration._unreal_tools import registered_tool_names

        return list(registered_tool_names(self.settings))


def create_unreal_session(
    settings: Optional[Settings] = None,
) -> UnrealRuntimeSession:
    """
    Create an Unreal runtime session.

    Args:
        settings: Configuration settings.

    Returns:
        UnrealRuntimeSession instance.
    """
    return UnrealRuntimeSession(settings)


def is_unreal_available() -> bool:
    """Check whether the Unreal runtime is available."""
    return UNREAL_AVAILABLE
