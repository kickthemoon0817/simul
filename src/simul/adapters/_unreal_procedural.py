"""Layouts for ``generate_unreal_procedural_scene``, planned before the editor is asked.

``plan_scene`` turns a scene type, its parameters and an axis-aligned box into
pieces: Engine basic shapes (100 cm cubes, cylinders and spheres with the pivot
at the centre), each with a size, a centre and a mobility. The editor script
only spawns what the plan lists, so a layout that does not fit is a ValueError
before anything is spawned, and the same arguments always give the same scene.
Units are Unreal's: centimetres, Z up.

The seed is an integer from 0 to ``MAX_SEED``, and each seed gives its own
layout. (``random.Random`` seeds with the absolute value, so a negative seed
would repeat the layout of its absolute value under another name.) The layout
is the plan's; the names an actual scene is filed under are its generation's:
every generation gets a fresh id (``new_generation``), so generating the same
plan twice gives two scenes with their own tag and outliner folder.
"""

from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

CUBE = "/Engine/BasicShapes/Cube.Cube"
CYLINDER = "/Engine/BasicShapes/Cylinder.Cylinder"
SPHERE = "/Engine/BasicShapes/Sphere.Sphere"

#: Edge of the Engine basic shapes in cm; a piece's scale is its size over this.
SHAPE_SIZE = 100.0

SCENE_TYPES: Tuple[str, ...] = ("room", "corridor", "warehouse", "outdoor")
DEFAULT_SEED = 42
#: Largest seed; seeds run from 0 to this.
MAX_SEED = 2**32 - 1
DEFAULT_BOUNDS_MIN: Tuple[float, float, float] = (0.0, 0.0, 0.0)
DEFAULT_BOUNDS_MAX: Tuple[float, float, float] = (1000.0, 1000.0, 500.0)

#: Most actors one scene may have.
MAX_PIECES = 500
#: Most props, trees, rocks or shelf rows one scene may ask for.
MAX_COUNT = 200

FLOOR_THICKNESS = 10.0
#: Smallest x and y extent of the bounds, in cm.
MIN_SIDE = 100.0
#: Smallest height between the top of the floor and the top of the bounds, in cm.
MIN_HEADROOM = 50.0
#: Smallest free width or depth inside the walls, in cm.
MIN_INTERIOR = 50.0
#: Free space kept around walls, shelves, props, trees and rocks, in cm.
CLEARANCE = 20.0
#: Side range of a prop box, in cm.
PROP_SIZE: Tuple[float, float] = (30.0, 80.0)
#: Width a corridor keeps free beside its props, in cm.
PASSAGE = 60.0
#: Depth kept free in front of a room's doorway, in cm.
DOORWAY_DEPTH = 100.0
#: Height range of a tree, in cm; its crown is 0.4 of its height across.
TREE_HEIGHT: Tuple[float, float] = (250.0, 500.0)
#: Smallest height a tree is given, in cm.
MIN_TREE_HEIGHT = 100.0
#: Side range of a rock in plan, in cm.
ROCK_SIZE: Tuple[float, float] = (40.0, 120.0)
#: Floor area per prop in a room or a warehouse, ground area per tree and per
#: rock outdoors, in cm^2, and corridor length per prop, in cm: the default
#: counts, so the defaults fit any bounds.
PROP_AREA = 16e4
TREE_AREA = 20e4
ROCK_AREA = 10e4
CORRIDOR_PROP_LENGTH = 300.0
#: Most props, trees or rocks a default count asks for, so that a default
#: scene stays within MAX_PIECES.
MAX_DEFAULT_COUNT = 150
_ATTEMPTS = 100

#: Parameters each scene type takes; every type takes ``seed``.
SCENE_PARAMETERS: Dict[str, Tuple[str, ...]] = {
    "room": ("seed", "wall_thickness", "wall_height", "door_width", "props"),
    "corridor": ("seed", "wall_thickness", "wall_height", "props"),
    "warehouse": (
        "seed",
        "wall_thickness",
        "wall_height",
        "rows",
        "shelf_depth",
        "shelf_height",
        "aisle_width",
        "props",
    ),
    "outdoor": ("seed", "trees", "rocks"),
}


@dataclass(frozen=True)
class Piece:
    """One actor of a planned scene: a basic shape scaled to ``size`` at ``center``.

    Attributes:
        label: Outliner label, ``<Scene>_<Kind>_<n>``.
        mesh: Basic shape asset path.
        center: World position of the shape's centre, in cm.
        size: Extent along x, y and z, in cm.
        movable: Movable mobility (props); everything else is Static.
    """

    label: str
    mesh: str
    center: Tuple[float, float, float]
    size: Tuple[float, float, float]
    movable: bool = False

    def spawn_args(self) -> Dict[str, Any]:
        """What the editor script needs to spawn this piece."""
        return {
            "label": self.label,
            "mesh": self.mesh,
            "location": [round(value, 3) for value in self.center],
            "scale": [round(value / SHAPE_SIZE, 6) for value in self.size],
            "movable": self.movable,
        }


@dataclass(frozen=True)
class ScenePlan:
    """A planned scene: what to spawn, and the names each generation of it is filed under."""

    scene_type: str
    seed: int
    pieces: Tuple[Piece, ...]

    def tag(self, generation: str) -> str:
        """Actor tag on every piece one generation of this plan spawns.

        Args:
            generation: The generation's id (``new_generation``).
        """
        return f"simul_{self.scene_type}_{self.seed}_{generation}"

    def folder(self, generation: str) -> str:
        """Outliner folder one generation of this plan files its pieces under.

        Args:
            generation: The generation's id (``new_generation``).
        """
        return f"Simul/{self.scene_type.title()}_{self.seed}_{generation}"


def new_generation() -> str:
    """A fresh id for one generation of a scene: 8 random hex digits."""
    return uuid.uuid4().hex[:8]


@dataclass(frozen=True)
class _Area:
    """An axis-aligned rectangle in plan (x, y), in cm."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def depth(self) -> float:
        return self.y1 - self.y0

    def clear_of(self, others: Sequence["_Area"], margin: float) -> bool:
        """Whether this rectangle is at least ``margin`` away from every other one."""
        return all(
            self.x1 + margin <= other.x0
            or other.x1 + margin <= self.x0
            or self.y1 + margin <= other.y0
            or other.y1 + margin <= self.y0
            for other in others
        )


def _centred(x: float, y: float, width: float, depth: float) -> _Area:
    return _Area(x - width / 2, y - depth / 2, x + width / 2, y + depth / 2)


class _Plan:
    """Collects the pieces of one scene and the plan areas they keep clear."""

    def __init__(self, scene_type: str, seed: int, lo: Sequence[float], hi: Sequence[float]) -> None:
        self.scene = scene_type.title()
        self.rng = random.Random(seed)
        self.bounds = _Area(lo[0], lo[1], hi[0], hi[1])
        self.ground = lo[2] + FLOOR_THICKNESS
        self.headroom = hi[2] - self.ground
        self.pieces: List[Piece] = []
        self.keep_clear: List[_Area] = []
        self._counts: Dict[str, int] = {}

    def add(
        self,
        kind: str,
        mesh: str,
        footprint: _Area,
        z0: float,
        height: float,
        *,
        movable: bool = False,
        keep_clear: bool = True,
    ) -> None:
        """Place a shape filling ``footprint`` in plan from ``z0`` up to ``z0 + height``."""
        index = self._counts.get(kind, 0)
        self._counts[kind] = index + 1
        center = ((footprint.x0 + footprint.x1) / 2, (footprint.y0 + footprint.y1) / 2, z0 + height / 2)
        size = (footprint.width, footprint.depth, height)
        self.pieces.append(Piece(f"{self.scene}_{kind}_{index}", mesh, center, size, movable))
        if keep_clear:
            self.keep_clear.append(footprint)

    def floor(self, kind: str = "Floor") -> None:
        self.add(kind, CUBE, self.bounds, self.ground - FLOOR_THICKNESS, FLOOR_THICKNESS, keep_clear=False)

    def find_spot(self, area: _Area, width: float, depth: float) -> Optional[_Area]:
        """A random ``width`` x ``depth`` rectangle inside ``area`` clear of what is placed, or None."""
        if width > area.width or depth > area.depth:
            return None
        x = self.rng.uniform(area.x0 + width / 2, area.x1 - width / 2)
        y = self.rng.uniform(area.y0 + depth / 2, area.y1 - depth / 2)
        spot = _centred(x, y, width, depth)
        return spot if spot.clear_of(self.keep_clear, CLEARANCE) else None

    def prop_height(self) -> float:
        return self.rng.uniform(min(PROP_SIZE[0], self.headroom), min(PROP_SIZE[1], self.headroom))


def plan_scene(
    scene_type: str,
    parameters: Optional[Mapping[str, Any]] = None,
    bounds_min: Optional[Sequence[float]] = None,
    bounds_max: Optional[Sequence[float]] = None,
) -> ScenePlan:
    """Plan a scene of Engine basic shapes inside the bounds.

    Every scene has a 10 cm floor slab filling the bounds in plan, at the
    bottom of the bounds. Props are Movable boxes 30 to 80 cm on a side
    (smaller on a floor too narrow for 80 cm), set on the floor; every other
    piece is Static. Everything placed keeps 20 cm clear of everything else.

    * room: four walls on the edges of the bounds, a doorway centred in the
      wall at minimum y (``door_width``, 100 cm or the width between the side
      walls if less, 0 for none) kept clear for 1 m inside, and ``props``
      (one per 16 m^2 inside the walls; none when the room is under 70 cm
      across inside its walls).
    * corridor: walls along the two long sides, and ``props`` (one per 3 m of
      corridor) beside either wall, never two at the same point along the
      corridor, so a 60 cm passage stays free.
    * warehouse: four walls, ``rows`` of solid shelves (as many as fit) along
      the longer side with ``aisle_width`` (250 cm) aisles around them, and
      ``props`` in the aisles (one per 16 m^2 of free floor; smaller in narrow
      aisles, none when aisles are under 90 cm).
    * outdoor: ground, ``trees`` (one per 20 m^2; a cylinder trunk under a
      sphere crown 0.4 of the tree's height across, 2.5 to 5 m tall and
      capped by the bounds, so that the crown fits their narrower side) and
      ``rocks`` (one per 10 m^2; flattened spheres).

    Walls are ``wall_thickness`` (20 cm) thick and ``wall_height`` tall (up
    to the top of the bounds). Shelves are ``shelf_depth`` (100 cm) deep and
    ``shelf_height`` (300 cm, capped by the bounds) tall. Props, trees and
    rocks left out follow the densities above, at most 150 of each, so they
    fit whatever the bounds; a count given that does not fit is a ValueError,
    and so is a warehouse with no room for one row of shelves unless ``rows``
    is 0.

    Args:
        scene_type: room, corridor, warehouse or outdoor, in any case.
        parameters: The scene type's parameters (``SCENE_PARAMETERS``), and
            ``seed`` (an integer from 0 to ``MAX_SEED``, default 42) for the
            random sizes and positions.
        bounds_min: Minimum corner [x, y, z] in cm; default [0, 0, 0].
        bounds_max: Maximum corner [x, y, z] in cm; default [1000, 1000, 500].

    Returns:
        The plan.

    Raises:
        ValueError: An unknown scene type or parameter, a value out of range,
            or a layout that does not fit the bounds.
    """
    kind = scene_type.strip().lower() if isinstance(scene_type, str) else scene_type
    if kind not in SCENE_TYPES:
        raise ValueError(f"scene_type must be one of {', '.join(SCENE_TYPES)}, got {scene_type!r}")
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, Mapping):
        raise ValueError(f"parameters must be a JSON object, got {type(parameters).__name__}")
    allowed = SCENE_PARAMETERS[kind]
    unknown = sorted(set(parameters) - set(allowed))
    if unknown:
        raise ValueError(
            f"{kind} does not take {', '.join(map(repr, unknown))}; its parameters are {', '.join(allowed)}"
        )
    seed = parameters.get("seed", DEFAULT_SEED)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be an integer from 0 to {MAX_SEED}, got {seed!r}")
    lo = _corner("bounds_min", DEFAULT_BOUNDS_MIN if bounds_min is None else bounds_min)
    hi = _corner("bounds_max", DEFAULT_BOUNDS_MAX if bounds_max is None else bounds_max)
    if hi[0] - lo[0] < MIN_SIDE or hi[1] - lo[1] < MIN_SIDE or hi[2] - lo[2] < FLOOR_THICKNESS + MIN_HEADROOM:
        raise ValueError(
            f"bounds_max must exceed bounds_min by at least {MIN_SIDE:g} cm in x and y and "
            f"{FLOOR_THICKNESS + MIN_HEADROOM:g} cm in z, got {list(lo)} to {list(hi)}"
        )
    plan = _Plan(kind, seed, lo, hi)
    _BUILDERS[kind](plan, parameters)
    return ScenePlan(kind, seed, tuple(plan.pieces))


def _corner(name: str, value: Any) -> Tuple[float, float, float]:
    if (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(_is_number(item) and math.isfinite(item) for item in value)
    ):
        return (float(value[0]), float(value[1]), float(value[2]))
    raise ValueError(f"{name} must be three finite numbers [x, y, z] in cm, got {value!r}")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _length(parameters: Mapping[str, Any], name: str, default: float) -> float:
    """A length parameter in cm: a finite number above 0."""
    value = parameters.get(name, default)
    if not _is_number(value) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a number of centimetres above 0, got {value!r}")
    return float(value)


def _count(parameters: Mapping[str, Any], name: str, default: int) -> int:
    value = parameters.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_COUNT:
        raise ValueError(f"{name} must be an integer from 0 to {MAX_COUNT}, got {value!r}")
    return value


def _per(amount: float, rate: float) -> int:
    """A default count: one per ``rate`` of ``amount``, at most MAX_DEFAULT_COUNT."""
    return min(MAX_DEFAULT_COUNT, int(amount // rate))


def _check_total(walls: int, *counts: int) -> None:
    total = 1 + walls + sum(counts)
    if total > MAX_PIECES:
        raise ValueError(f"The scene would have {total} actors; at most {MAX_PIECES} are spawned per call")


def _wall_height(plan: _Plan, parameters: Mapping[str, Any]) -> float:
    height = _length(parameters, "wall_height", plan.headroom)
    if height > plan.headroom:
        raise ValueError(
            f"wall_height {height:g} cm exceeds the {plan.headroom:g} cm between the floor and bounds_max z"
        )
    return height


def _interior(plan: _Plan, thickness: float, x_walls: bool, y_walls: bool) -> _Area:
    """The plan area inside the walls; walls at the x and/or y edges of the bounds."""
    b = plan.bounds
    inner = _Area(
        b.x0 + (thickness if x_walls else 0.0),
        b.y0 + (thickness if y_walls else 0.0),
        b.x1 - (thickness if x_walls else 0.0),
        b.y1 - (thickness if y_walls else 0.0),
    )
    if inner.width < MIN_INTERIOR or inner.depth < MIN_INTERIOR:
        raise ValueError(
            f"wall_thickness {thickness:g} cm leaves less than {MIN_INTERIOR:g} cm inside the walls"
        )
    return inner


def _walls(plan: _Plan, thickness: float, height: float, door_width: float = 0.0) -> None:
    """Walls on the four edges of the bounds; a doorway centred in the wall at minimum y."""
    b, ground = plan.bounds, plan.ground
    if door_width > 0:
        middle = (b.x0 + b.x1) / 2
        plan.add("Wall", CUBE, _Area(b.x0, b.y0, middle - door_width / 2, b.y0 + thickness), ground, height)
        plan.add("Wall", CUBE, _Area(middle + door_width / 2, b.y0, b.x1, b.y0 + thickness), ground, height)
        plan.keep_clear.append(
            _Area(middle - door_width / 2, b.y0 + thickness, middle + door_width / 2, b.y0 + thickness + DOORWAY_DEPTH)
        )
    else:
        plan.add("Wall", CUBE, _Area(b.x0, b.y0, b.x1, b.y0 + thickness), ground, height)
    plan.add("Wall", CUBE, _Area(b.x0, b.y1 - thickness, b.x1, b.y1), ground, height)
    plan.add("Wall", CUBE, _Area(b.x0, b.y0 + thickness, b.x0 + thickness, b.y1 - thickness), ground, height)
    plan.add("Wall", CUBE, _Area(b.x1 - thickness, b.y0 + thickness, b.x1, b.y1 - thickness), ground, height)


def _prop_floor(inner: _Area, largest: float = PROP_SIZE[1]) -> Tuple[_Area, float]:
    """Where props stand inside the walls, and the largest prop side that fits there.

    Props keep CLEARANCE from the walls, so they are placed in ``inner`` moved
    in by CLEARANCE on every side; sampling only there keeps a narrow floor
    from failing by chance. A largest side under PROP_SIZE[0] means no prop
    fits.
    """
    floor = _Area(inner.x0 + CLEARANCE, inner.y0 + CLEARANCE, inner.x1 - CLEARANCE, inner.y1 - CLEARANCE)
    return floor, min(largest, floor.width, floor.depth)


def _props(plan: _Plan, count: int, area: _Area, largest: float = PROP_SIZE[1]) -> None:
    """``count`` Movable boxes up to ``largest`` cm on a side standing in ``area``, clear of everything placed."""
    for index in range(count):
        for _ in range(_ATTEMPTS):
            width, depth = plan.rng.uniform(PROP_SIZE[0], largest), plan.rng.uniform(PROP_SIZE[0], largest)
            spot = plan.find_spot(area, width, depth)
            if spot is not None:
                plan.add("Box", CUBE, spot, plan.ground, plan.prop_height(), movable=True)
                break
        else:
            raise ValueError(
                f"props: only {index} of {count} props fit on the free floor; lower props or enlarge the bounds"
            )


def _room(plan: _Plan, parameters: Mapping[str, Any]) -> None:
    thickness = _length(parameters, "wall_thickness", 20.0)
    height = _wall_height(plan, parameters)
    door_width = parameters.get("door_width", 100.0)
    if not _is_number(door_width) or not math.isfinite(door_width) or door_width < 0:
        raise ValueError(f"door_width must be a number of centimetres, 0 for no doorway, got {door_width!r}")
    inner = _interior(plan, thickness, True, True)
    if "door_width" not in parameters:
        # The default doorway narrows to fit a narrow room.
        door_width = min(door_width, inner.width)
    if door_width > inner.width:
        raise ValueError(f"door_width {door_width:g} cm does not fit in the {inner.width:g} cm between the side walls")
    floor, largest = _prop_floor(inner)
    roomy = largest >= PROP_SIZE[0]
    props = _count(parameters, "props", _per(inner.width * inner.depth, PROP_AREA) if roomy else 0)
    if props and not roomy:
        raise ValueError(
            f"props: the room is {min(inner.width, inner.depth):g} cm across inside its walls; props need "
            f"{PROP_SIZE[0] + 2 * CLEARANCE:g} cm"
        )
    _check_total(5 if door_width else 4, props)
    plan.floor()
    _walls(plan, thickness, height, float(door_width))
    _props(plan, props, floor, largest)


def _corridor(plan: _Plan, parameters: Mapping[str, Any]) -> None:
    thickness = _length(parameters, "wall_thickness", 20.0)
    height = _wall_height(plan, parameters)
    b = plan.bounds
    along_x = b.width >= b.depth
    inner = _interior(plan, thickness, not along_x, along_x)
    if along_x:
        a0, a1, c0, c1 = inner.x0, inner.x1, inner.y0, inner.y1
    else:
        a0, a1, c0, c1 = inner.y0, inner.y1, inner.x0, inner.x1
    # A prop stands CLEARANCE off a wall and leaves PASSAGE free beside it.
    deepest = min(PROP_SIZE[1], (c1 - c0) - CLEARANCE - PASSAGE)
    roomy = deepest >= PROP_SIZE[0]
    props = _count(parameters, "props", _per(a1 - a0, CORRIDOR_PROP_LENGTH) if roomy else 0)
    if props and not roomy:
        raise ValueError(
            f"props: the corridor is {c1 - c0:g} cm wide inside its walls; props need "
            f"{PROP_SIZE[0] + CLEARANCE + PASSAGE:g} cm to leave a {PASSAGE:g} cm passage"
        )
    _check_total(2, props)
    plan.floor()
    if along_x:
        plan.add("Wall", CUBE, _Area(b.x0, b.y0, b.x1, inner.y0), plan.ground, height)
        plan.add("Wall", CUBE, _Area(b.x0, inner.y1, b.x1, b.y1), plan.ground, height)
    else:
        plan.add("Wall", CUBE, _Area(b.x0, b.y0, inner.x0, b.y1), plan.ground, height)
        plan.add("Wall", CUBE, _Area(inner.x1, b.y0, b.x1, b.y1), plan.ground, height)
    taken: List[Tuple[float, float]] = []
    for index in range(props):
        for _ in range(_ATTEMPTS):
            length = plan.rng.uniform(*PROP_SIZE)
            depth = plan.rng.uniform(PROP_SIZE[0], deepest)
            if length > a1 - a0:
                continue
            middle = plan.rng.uniform(a0 + length / 2, a1 - length / 2)
            start, end = middle - length / 2, middle + length / 2
            if any(start < last + CLEARANCE and first < end + CLEARANCE for first, last in taken):
                continue
            if plan.rng.random() < 0.5:
                across = (c0 + CLEARANCE, c0 + CLEARANCE + depth)
            else:
                across = (c1 - CLEARANCE - depth, c1 - CLEARANCE)
            spot = _Area(start, across[0], end, across[1]) if along_x else _Area(across[0], start, across[1], end)
            plan.add("Box", CUBE, spot, plan.ground, plan.prop_height(), movable=True)
            taken.append((start, end))
            break
        else:
            raise ValueError(
                f"props: only {index} of {props} props fit along the corridor; lower props or lengthen the bounds"
            )


def _warehouse(plan: _Plan, parameters: Mapping[str, Any]) -> None:
    thickness = _length(parameters, "wall_thickness", 20.0)
    height = _wall_height(plan, parameters)
    depth = _length(parameters, "shelf_depth", 100.0)
    aisle = _length(parameters, "aisle_width", 250.0)
    shelf_height = _length(parameters, "shelf_height", min(300.0, plan.headroom))
    if shelf_height > plan.headroom:
        raise ValueError(
            f"shelf_height {shelf_height:g} cm exceeds the {plan.headroom:g} cm between the floor and bounds_max z"
        )
    inner = _interior(plan, thickness, True, True)
    along_x = inner.width >= inner.depth
    a0, a1, c0, c1 = (
        (inner.x0, inner.x1, inner.y0, inner.y1) if along_x else (inner.y0, inner.y1, inner.x0, inner.x1)
    )
    across = c1 - c0
    fit = max(0, math.floor((across - aisle) / (depth + aisle)))
    if "rows" in parameters:
        rows = _count(parameters, "rows", 0)
        if rows > fit:
            raise ValueError(
                f"rows: {rows} rows do not fit; {fit} rows of {depth:g} cm shelves with {aisle:g} cm "
                f"aisles fit across {across:g} cm"
            )
    elif fit:
        rows = min(fit, MAX_COUNT)
    else:
        raise ValueError(
            f"No shelf row of {depth:g} cm with {aisle:g} cm aisles fits across {across:g} cm; enlarge the "
            "bounds, lower shelf_depth or aisle_width, or pass rows 0"
        )
    shelf_length = a1 - a0 - 2 * aisle
    if rows and shelf_length < MIN_SIDE:
        raise ValueError(
            f"Shelves need {MIN_SIDE + 2 * aisle:g} cm along the warehouse for {aisle:g} cm aisles at both "
            f"ends; it is {a1 - a0:g} cm inside the walls"
        )
    # Props stand in the aisles with clearance on both sides and room to move.
    floor, largest = _prop_floor(inner, min(PROP_SIZE[1], aisle - 3 * CLEARANCE))
    roomy = largest >= PROP_SIZE[0]
    free_floor = inner.width * inner.depth - rows * shelf_length * depth
    props = _count(parameters, "props", _per(free_floor, PROP_AREA) if roomy else 0)
    if props and not roomy:
        if aisle - 3 * CLEARANCE < PROP_SIZE[0]:
            raise ValueError(
                f"props: {aisle:g} cm aisles leave no room for props; props need aisles of "
                f"{PROP_SIZE[0] + 3 * CLEARANCE:g} cm"
            )
        raise ValueError(
            f"props: the warehouse is {min(inner.width, inner.depth):g} cm across inside its walls; props "
            f"need {PROP_SIZE[0] + 2 * CLEARANCE:g} cm"
        )
    _check_total(4, rows, props)
    plan.floor()
    _walls(plan, thickness, height)
    first = c0 + (across - (rows * depth + (rows + 1) * aisle)) / 2 + aisle
    for row in range(rows):
        lo = first + row * (depth + aisle)
        if along_x:
            shelf = _Area(a0 + aisle, lo, a1 - aisle, lo + depth)
        else:
            shelf = _Area(lo, a0 + aisle, lo + depth, a1 - aisle)
        plan.add("Shelf", CUBE, shelf, plan.ground, shelf_height)
    _props(plan, props, floor, largest)


def _outdoor(plan: _Plan, parameters: Mapping[str, Any]) -> None:
    ground = plan.bounds.width * plan.bounds.depth
    tall_enough = plan.headroom >= MIN_TREE_HEIGHT
    trees = _count(parameters, "trees", _per(ground, TREE_AREA) if tall_enough else 0)
    rocks = _count(parameters, "rocks", _per(ground, ROCK_AREA))
    _check_total(0, 2 * trees, rocks)
    if trees and not tall_enough:
        raise ValueError(
            f"trees: need {MIN_TREE_HEIGHT:g} cm between the ground and bounds_max z, there are {plan.headroom:g} cm"
        )
    plan.floor("Ground")
    # The crown, 0.4 of the tree's height across, has to fit the narrower side of the bounds.
    narrow = min(plan.bounds.width, plan.bounds.depth)
    for index in range(trees):
        for _ in range(_ATTEMPTS):
            tree = min(plan.rng.uniform(*TREE_HEIGHT), plan.headroom, narrow / 0.4)
            crown = min(0.4 * tree, narrow)
            spot = plan.find_spot(plan.bounds, crown, crown)
            if spot is None:
                continue
            x, y = (spot.x0 + spot.x1) / 2, (spot.y0 + spot.y1) / 2
            trunk = 0.08 * tree
            # The trunk reaches into the crown, which tops the tree.
            plan.add("Trunk", CYLINDER, _centred(x, y, trunk, trunk), plan.ground, 0.65 * tree, keep_clear=False)
            plan.add("Crown", SPHERE, spot, plan.ground + 0.6 * tree, crown)
            break
        else:
            raise ValueError(f"trees: only {index} of {trees} trees fit; lower trees or enlarge the bounds")
    for index in range(rocks):
        for _ in range(_ATTEMPTS):
            width, depth = plan.rng.uniform(*ROCK_SIZE), plan.rng.uniform(*ROCK_SIZE)
            spot = plan.find_spot(plan.bounds, width, depth)
            if spot is not None:
                height = min(plan.rng.uniform(0.4, 0.8) * min(width, depth), plan.headroom)
                plan.add("Rock", SPHERE, spot, plan.ground, height)
                break
        else:
            raise ValueError(f"rocks: only {index} of {rocks} rocks fit; lower rocks or enlarge the bounds")


_BUILDERS = {"room": _room, "corridor": _corridor, "warehouse": _warehouse, "outdoor": _outdoor}
