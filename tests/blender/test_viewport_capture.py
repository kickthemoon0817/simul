"""capture_viewport captures the 3D View the user sees (issue #228).

Attached mode used to try a GPU path that could never work inside Blender (no
Pillow, a removed GPUOffScreen.read_color) and silently fell back to a render
through the scene camera, failing with "Cannot render, no camera" in a scene
without one. These tests pin the replacement against a stand-in bpy: the
viewport is redrawn offscreen with the region's own matrices, an external
engine's Rendered view is read from the screen, and a render only happens when
asked for, through a temporary camera when the scene has none.
"""

from __future__ import annotations

import base64
import math
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Tuple
from unittest.mock import Mock

import numpy as np
import pytest

from simul.adapters import blender_runtime
from simul.utils.paths import PathPolicy

JPEG_PREFIX = b"\xff\xd8"


class FakePixels:
    def __init__(self, values: Optional[List[float]] = None) -> None:
        self.values = values

    def foreach_set(self, values: Any) -> None:
        self.values = [float(v) for v in values]

    def foreach_get(self, out: Any) -> None:
        out[:] = self.values


class FakeImage:
    def __init__(self, name: str, width: int, height: int, pixels: Any = None) -> None:
        self.name = name
        self.size = (width, height)
        self.file_format = "PNG"
        self.pixels = FakePixels(pixels)
        self.saved: Optional[Tuple[str, int]] = None

    def scale(self, width: int, height: int) -> None:
        self.size = (width, height)

    def save(self, filepath: str, quality: int) -> None:
        self.saved = (self.file_format, quality)
        Path(filepath).write_bytes(JPEG_PREFIX + b"viewport")

    def save_render(self, filepath: str, scene: Any) -> None:
        Path(filepath).write_bytes(JPEG_PREFIX + b"render")


class FakeImages:
    def __init__(self) -> None:
        self.live: List[FakeImage] = []
        self.render_result: Optional[FakeImage] = None
        self.screenshot: Optional[Tuple[int, int, List[float]]] = None

    def new(self, name: str, width: int, height: int, alpha: bool = False) -> FakeImage:
        image = FakeImage(name, width, height)
        self.live.append(image)
        return image

    def load(self, filepath: str, check_existing: bool = False) -> FakeImage:
        assert Path(filepath).is_file()
        assert self.screenshot is not None
        width, height, pixels = self.screenshot
        image = FakeImage("screenshot", width, height, pixels)
        self.live.append(image)
        return image

    def remove(self, image: FakeImage) -> None:
        self.live.remove(image)

    def get(self, name: str) -> Optional[FakeImage]:
        return self.render_result if name == "Render Result" else None


class FakeBuffer(list):
    """gpu.types.Buffer stand-in: a flat value list whose shape can be set."""

    dimensions: Any = None


class FakeOffScreen:
    def __init__(self, gpu: "FakeGPU", width: int, height: int) -> None:
        self.gpu = gpu
        self.size = (width, height)
        self.drawn: Optional[Tuple[Any, ...]] = None
        self.freed = False
        gpu.offscreens.append(self)

    def draw_view3d(self, *args: Any, do_color_management: bool = False) -> None:
        self.drawn = (*args, do_color_management)

    @contextmanager
    def bind(self) -> Iterator[None]:
        self.gpu.bound = self
        yield
        self.gpu.bound = None

    def free(self) -> None:
        self.freed = True


class FakeGPU(ModuleType):
    def __init__(self) -> None:
        super().__init__("gpu")
        self.offscreens: List[FakeOffScreen] = []
        self.bound: Optional[FakeOffScreen] = None
        self.reads: List[Tuple[Any, ...]] = []
        self.types = SimpleNamespace(
            GPUOffScreen=lambda w, h: FakeOffScreen(self, w, h)
        )
        self.state = SimpleNamespace(active_framebuffer_get=self._framebuffer)

    def _framebuffer(self) -> Any:
        assert self.bound is not None, "pixels are read from the bound offscreen"

        def read_color(
            x: int, y: int, w: int, h: int, channels: int, slot: int, fmt: str
        ) -> FakeBuffer:
            self.reads.append((x, y, w, h, channels, slot, fmt))
            return FakeBuffer([0.5] * (w * h * channels))

        return SimpleNamespace(read_color=read_color)


class FakeImageSettings(SimpleNamespace):
    """Blender 5.0's output format rules that a capture has to work around."""

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "file_format" and self.media_type == "VIDEO" and value != "FFMPEG":
            raise TypeError(f"enum \"{value}\" not found in ('FFMPEG')")
        super().__setattr__(name, value)
        if name == "file_format" and value == "JPEG":
            # JPEG stores neither alpha nor 16 bits; Blender downgrades both.
            super().__setattr__("color_mode", "RGB")
            super().__setattr__("color_depth", "8")


def _image_settings() -> FakeImageSettings:
    """A user's video output, which a capture must leave as it found it."""
    return FakeImageSettings(
        media_type="VIDEO",
        file_format="FFMPEG",
        color_mode="RGBA",
        color_depth="16",
        quality=90,
    )


class FakeBlender:
    """Just enough of bpy for the capture paths, recording what they did."""

    def __init__(
        self,
        tmp_path: Path,
        *,
        with_view: bool = True,
        shading: str = "SOLID",
        engine: str = "BLENDER_EEVEE",
        camera: Any = None,
        objects: Optional[List[Any]] = None,
    ) -> None:
        self.images = FakeImages()
        self.view = SimpleNamespace(
            view_matrix=SimpleNamespace(inverted=lambda: self.view_to_world),
            window_matrix=[
                [1.5, 0, 0, 0],
                [0, 3.0, 0, 0],
                [0, 0, -1, -0.02],
                [0, 0, -1, 0],
            ],
            is_perspective=True,
            update=Mock(),
        )
        self.view_to_world = SimpleNamespace(
            translation=(1.0, 2.0, 3.0),
            col=[(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (1, 2, 3, 1)],
        )
        self.space = SimpleNamespace(
            shading=SimpleNamespace(type=shading),
            region_3d=self.view,
            clip_start=0.01,
            clip_end=1000.0,
        )
        self.header = SimpleNamespace(
            type="HEADER", width=2000, height=100, x=0, y=1000, data=None
        )
        self.region = SimpleNamespace(
            type="WINDOW", width=2000, height=1000, x=0, y=0, data=self.view
        )
        self.area = SimpleNamespace(
            type="VIEW_3D",
            width=2000,
            height=1100,
            x=0,
            y=0,
            regions=[self.header, self.region],
            spaces=SimpleNamespace(active=self.space),
        )
        self.window = SimpleNamespace(screen=SimpleNamespace(areas=[self.area]))
        self.settings = _image_settings()
        self.render = SimpleNamespace(
            engine=engine,
            resolution_x=1920,
            resolution_y=1080,
            resolution_percentage=50,
            image_settings=self.settings,
        )
        self.linked: List[Any] = []
        self.scene = SimpleNamespace(
            camera=camera,
            render=self.render,
            objects=list(objects or []),
            collection=SimpleNamespace(
                objects=SimpleNamespace(link=self.linked.append)
            ),
        )
        self.renders: List[Dict[str, Any]] = []
        self.removed: List[Any] = []
        self.cameras_removed: List[Any] = []
        self.render_error: Optional[Exception] = None
        self.screenshots: List[Dict[str, Any]] = []
        self.overrides: List[Dict[str, Any]] = []
        self.bpy = SimpleNamespace(
            app=SimpleNamespace(version=(5, 0, 1), background=False),
            context=SimpleNamespace(
                scene=self.scene,
                view_layer=SimpleNamespace(name="ViewLayer"),
                area=self.area if with_view else None,
                window=self.window if with_view else None,
                temp_override=self._temp_override,
            ),
            data=SimpleNamespace(
                images=self.images,
                cameras=SimpleNamespace(
                    new=self._new_camera_data, remove=self.cameras_removed.append
                ),
                objects=SimpleNamespace(
                    new=self._new_object, remove=self._remove_object
                ),
            ),
            ops=SimpleNamespace(
                render=SimpleNamespace(render=self._render),
                screen=SimpleNamespace(screenshot_area=self._screenshot_area),
            ),
            types=SimpleNamespace(RenderEngine=type("RenderEngine", (), {})),
        )

    def _temp_override(self, **context: Any) -> Any:
        self.overrides.append(context)
        return nullcontext()

    @staticmethod
    def _new_camera_data(name: str) -> SimpleNamespace:
        # A new camera: 50 mm lens on a 36 mm sensor.
        return SimpleNamespace(
            name=name,
            angle=2 * math.atan(18 / 50),
            sensor_fit="AUTO",
            type="PERSP",
            clip_start=0.1,
            clip_end=100.0,
            ortho_scale=6.0,
        )

    @staticmethod
    def _new_object(name: str, data: Any) -> SimpleNamespace:
        return SimpleNamespace(
            name=name,
            data=data,
            type="CAMERA",
            matrix_world=None,
            location=None,
            rotation_euler=None,
        )

    def _remove_object(self, obj: Any, do_unlink: bool = False) -> None:
        assert do_unlink
        self.removed.append(obj)

    def _render(self, write_still: bool = False) -> None:
        if self.render_error is not None:
            raise self.render_error
        self.renders.append(
            {
                "camera": self.scene.camera,
                "size": (self.render.resolution_x, self.render.resolution_y),
                "percentage": self.render.resolution_percentage,
                "format": (
                    self.settings.media_type,
                    self.settings.file_format,
                    self.settings.quality,
                ),
            }
        )
        self.images.render_result = FakeImage("Render Result", 1, 1)

    def _screenshot_area(self, filepath: str) -> set:
        self.screenshots.append(
            {
                "format": (self.settings.media_type, self.settings.file_format),
                "context": self.overrides[-1],
            }
        )
        Path(filepath).write_bytes(b"png")
        return {"FINISHED"}


@pytest.fixture
def gpu(monkeypatch: pytest.MonkeyPatch) -> FakeGPU:
    module = FakeGPU()
    monkeypatch.setitem(sys.modules, "gpu", module)
    return module


def _session(
    monkeypatch: pytest.MonkeyPatch,
    fake: FakeBlender,
    tmp_path: Path,
    attached: bool = True,
) -> blender_runtime.BlenderRuntimeSession:
    monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
    monkeypatch.setattr(blender_runtime, "bpy", fake.bpy)
    policy = PathPolicy(
        enabled=True, allowed_paths=[str(tmp_path)], project_root=tmp_path
    )
    return blender_runtime.BlenderRuntimeSession(
        path_policy=policy, scene_scope=attached
    )


def _scene_camera() -> SimpleNamespace:
    return SimpleNamespace(name="Camera", type="CAMERA")


# -- method selection -------------------------------------------------------


@pytest.mark.parametrize(
    ("attached", "method", "use_render_fallback", "expected"),
    [
        (True, "auto", False, "viewport"),
        (False, "auto", False, "render"),
        (True, "render", False, "render"),
        (False, "viewport", False, "viewport"),
        (True, "auto", True, "render"),
        (True, "render", True, "render"),
    ],
)
def test_capture_method_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    attached: bool,
    method: str,
    use_render_fallback: bool,
    expected: str,
) -> None:
    session = _session(monkeypatch, FakeBlender(tmp_path), tmp_path, attached=attached)
    assert session._resolve_capture_method(method, use_render_fallback) == expected


def test_contradictory_or_unknown_methods_are_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _session(monkeypatch, FakeBlender(tmp_path), tmp_path)
    with pytest.raises(ValueError, match="contradicts"):
        session.capture_viewport(method="viewport", use_render_fallback=True)
    with pytest.raises(ValueError, match="method must be one of"):
        session.capture_viewport(method="gpu_offscreen")


def test_a_failed_viewport_capture_does_not_fall_back_to_a_render(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A render through some camera is not what the user sees; fail instead."""
    fake = FakeBlender(tmp_path, with_view=False, camera=_scene_camera())
    session = _session(monkeypatch, fake, tmp_path, attached=True)
    with pytest.raises(RuntimeError, match="No 3D View.*method='render'"):
        session.capture_viewport()
    assert fake.renders == []


def test_background_blender_has_no_3d_view_despite_its_startup_windows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """blender -b loads the startup file's windows and areas but never draws them.

    Live-verified on 5.0.1: the area is there, its region has no RegionView3D.
    """
    fake = FakeBlender(tmp_path, camera=None)
    fake.bpy.app.background = True
    fake.region.data = None
    session = _session(monkeypatch, fake, tmp_path, attached=False)

    with pytest.raises(RuntimeError, match="No 3D View"):
        session.capture_viewport(method="viewport")
    result = session.capture_viewport(width=200, height=100)

    assert result["capture_method"] == "render_temp_camera_framing"
    assert (result["width"], result["height"]) == (200, 100)


def test_an_undrawn_region_is_not_captured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeBlender(tmp_path)
    fake.region.data = None
    fake.space.region_3d = object()
    session = _session(monkeypatch, fake, tmp_path)
    with pytest.raises(RuntimeError, match="no drawable region"):
        session.capture_viewport()


# -- viewport ---------------------------------------------------------------


def test_viewport_capture_redraws_the_3d_view_offscreen_without_a_camera(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, gpu: FakeGPU
) -> None:
    fake = FakeBlender(tmp_path, camera=None)
    session = _session(monkeypatch, fake, tmp_path)

    result = session.capture_viewport(width=512, height=512, jpeg_quality=70)

    assert result["capture_method"] == "viewport"
    # The 2000 x 1000 region keeps its aspect ratio inside 512 x 512.
    assert (result["width"], result["height"]) == (512, 256)
    assert base64.b64decode(result["image_base64"]).startswith(JPEG_PREFIX)
    assert result["format"] == "jpeg" and result["engine"] == "BLENDER_EEVEE"
    (offscreen,) = gpu.offscreens
    assert offscreen.size == (512, 256) and offscreen.freed
    scene, view_layer, space, region, view_matrix, projection, managed = offscreen.drawn
    assert (scene, space, region) == (fake.scene, fake.space, fake.region)
    assert view_matrix is fake.view.view_matrix
    assert projection is fake.view.window_matrix
    assert managed is True
    # Matrices are refreshed first: a view changed since the last redraw would be stale.
    fake.view.update.assert_called_once_with()
    assert gpu.reads == [(0, 0, 512, 256, 4, 0, "FLOAT")]
    assert fake.renders == [] and fake.scene.camera is None
    assert fake.images.live == [], "the temporary image is removed"
    assert (
        list(tmp_path.joinpath("captures").iterdir()) == []
    ), "scratch files are removed"


def test_viewport_capture_prefers_the_largest_3d_view_without_a_context_area(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, gpu: FakeGPU
) -> None:
    fake = FakeBlender(tmp_path)
    small_view = SimpleNamespace(**{**vars(fake.view), "update": Mock()})
    small_region = SimpleNamespace(
        type="WINDOW", width=300, height=200, x=0, y=0, data=small_view
    )
    small = SimpleNamespace(
        type="VIEW_3D",
        width=300,
        height=200,
        x=0,
        y=0,
        regions=[small_region],
        spaces=SimpleNamespace(
            active=SimpleNamespace(**{**vars(fake.space), "region_3d": small_view})
        ),
    )
    fake.window.screen.areas = [small, fake.area]
    fake.bpy.context.area = None
    session = _session(monkeypatch, fake, tmp_path)

    session.capture_viewport(width=512, height=512)

    assert gpu.offscreens[0].drawn[3] is fake.region


def test_quad_view_captures_the_region_of_region_3d(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, gpu: FakeGPU
) -> None:
    fake = FakeBlender(tmp_path)
    locked = [
        SimpleNamespace(type="WINDOW", width=1000, height=500, x=0, y=0, data=object())
        for _ in range(3)
    ]
    fake.region.width, fake.region.height = 1000, 500
    fake.area.regions = [*locked, fake.region]
    session = _session(monkeypatch, fake, tmp_path)

    session.capture_viewport()

    assert gpu.offscreens[0].drawn[3] is fake.region


@pytest.mark.parametrize(
    ("shading", "engine", "bl_use_eevee_viewport", "external"),
    [
        ("SOLID", "CYCLES", True, False),
        ("WIREFRAME", "CYCLES", True, False),
        ("RENDERED", "BLENDER_EEVEE", True, False),
        ("RENDERED", "BLENDER_WORKBENCH", True, False),
        ("RENDERED", "CYCLES", True, True),
        ("MATERIAL", "CYCLES", True, False),
        ("MATERIAL", "THIRD_PARTY", False, True),
    ],
)
def test_only_an_external_engines_view_is_read_from_the_screen(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    shading: str,
    engine: str,
    bl_use_eevee_viewport: bool,
    external: bool,
) -> None:
    fake = FakeBlender(tmp_path, shading=shading, engine=engine)
    # A registered engine is a RenderEngine subclass; keep it alive for __subclasses__().
    registered = type(
        engine,
        (fake.bpy.types.RenderEngine,),
        {"bl_idname": engine, "bl_use_eevee_viewport": bl_use_eevee_viewport},
    )
    _session(monkeypatch, fake, tmp_path)
    drawn_externally = blender_runtime.BlenderRuntimeSession._drawn_by_external_engine
    assert drawn_externally(fake.space, engine) is external
    assert registered.bl_idname == engine


def test_cycles_rendered_view_is_cropped_from_the_screen(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, gpu: FakeGPU
) -> None:
    """Offscreen, Cycles draws nothing: the image would hold only overlays."""
    fake = FakeBlender(tmp_path, shading="RENDERED", engine="CYCLES")
    fake.region.width, fake.region.height = 200, 100
    fake.area.width, fake.area.height = 200, 110
    # The screenshot covers the whole area at twice its coordinates (HiDPI);
    # rows run bottom-up: 200 rows of main region (0.25), then 20 of header (1.0).
    shot = np.concatenate(
        [
            np.full(400 * 200 * 4, 0.25, np.float32),
            np.full(400 * 20 * 4, 1.0, np.float32),
        ]
    )
    fake.images.screenshot = (400, 220, shot)
    created: List[FakeImage] = []
    original_new = fake.images.new

    def _track(name: str, width: int, height: int, alpha: bool = False) -> FakeImage:
        image = original_new(name, width, height, alpha)
        created.append(image)
        return image

    monkeypatch.setattr(fake.images, "new", _track)
    session = _session(monkeypatch, fake, tmp_path)

    result = session.capture_viewport(width=100, height=100, jpeg_quality=80)

    assert result["capture_method"] == "viewport_screen"
    assert (result["width"], result["height"]) == (100, 50)
    assert gpu.offscreens == []
    (screenshot,) = fake.screenshots
    # A PNG intermediate, written as an image although the scene outputs video.
    assert screenshot["format"] == ("IMAGE", "PNG")
    assert screenshot["context"] == {
        "window": fake.window,
        "area": fake.area,
        "region": fake.region,
    }
    (cropped,) = created
    assert len(cropped.pixels.values) == 400 * 200 * 4
    assert set(cropped.pixels.values) == {0.25}, "the header rows are cropped away"
    assert cropped.size == (100, 50)
    assert cropped.saved == ("JPEG", 80)
    assert fake.images.live == []
    assert vars(fake.settings) == vars(_image_settings())


# -- render -----------------------------------------------------------------


def test_render_uses_the_scene_camera_and_restores_every_setting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    camera = _scene_camera()
    fake = FakeBlender(tmp_path, camera=camera)
    session = _session(monkeypatch, fake, tmp_path)

    result = session.capture_viewport(
        width=320, height=200, jpeg_quality=60, method="render"
    )

    assert result["capture_method"] == "render"
    assert (result["width"], result["height"]) == (320, 200)
    assert base64.b64decode(result["image_base64"]).startswith(JPEG_PREFIX)
    (render,) = fake.renders
    assert render == {
        "camera": camera,
        "size": (320, 200),
        "percentage": 100,
        "format": ("IMAGE", "JPEG", 60),
    }
    assert (
        fake.render.resolution_x,
        fake.render.resolution_y,
        fake.render.resolution_percentage,
    ) == (
        1920,
        1080,
        50,
    )
    # JPEG downgrades color mode and depth; the user's output settings come back whole.
    assert vars(fake.settings) == vars(_image_settings())
    assert fake.linked == [] and fake.scene.camera is camera


def test_render_without_a_camera_aligns_a_temporary_one_to_the_3d_view(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeBlender(tmp_path, camera=None)
    session = _session(monkeypatch, fake, tmp_path)

    result = session.capture_viewport(width=512, height=512, method="render")

    assert result["capture_method"] == "render_temp_camera_view"
    assert (result["width"], result["height"]) == (512, 256)
    (render,) = fake.renders
    camera = render["camera"]
    assert camera.name == "simul_temp_camera" and fake.linked == [camera]
    assert render["size"] == (512, 256)
    assert camera.matrix_world is fake.view_to_world
    # Wider than tall: the field spans the width, from the projection's [0][0].
    assert camera.data.angle == pytest.approx(2 * math.atan(1 / 1.5))
    assert (camera.data.clip_start, camera.data.clip_end) == (0.01, 1000.0)
    assert fake.removed == [camera] and fake.cameras_removed == [camera.data]
    assert fake.scene.camera is None


def test_orthographic_view_gets_an_orthographic_camera_backed_off_its_clip_range(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeBlender(tmp_path, camera=None)
    fake.view.is_perspective = False
    fake.view.window_matrix = [
        [0.2, 0, 0, 0],
        [0, 0.4, 0, 0],
        [0, 0, -0.002, 0],
        [0, 0, 0, 1],
    ]
    fake.region.width, fake.region.height = 500, 1000
    session = _session(monkeypatch, fake, tmp_path)

    result = session.capture_viewport(width=512, height=512, method="render")

    camera = fake.renders[0]["camera"]
    assert (result["width"], result["height"]) == (256, 512)
    assert camera.data.type == "ORTHO"
    # Taller than wide: the field spans the height, from [1][1].
    assert camera.data.ortho_scale == pytest.approx(2 / 0.4)
    assert camera.location == pytest.approx((1.0, 2.0, 3.0 + 500.0))
    assert camera.data.clip_end == 1000.0


def test_embedded_render_without_a_camera_frames_the_visible_objects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _object(
        kind: str, offset: Tuple[float, float, float], visible: bool = True
    ) -> SimpleNamespace:
        matrix = [
            [1, 0, 0, offset[0]],
            [0, 1, 0, offset[1]],
            [0, 0, 1, offset[2]],
            [0, 0, 0, 1],
        ]
        corners = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        return SimpleNamespace(
            type=kind,
            hide_render=False,
            matrix_world=matrix,
            bound_box=corners,
            visible_get=lambda: visible,
        )

    objects = [
        _object("MESH", (4.0, 0.0, 0.0)),
        _object("MESH", (6.0, 0.0, 0.0)),
        _object("LIGHT", (100.0, 100.0, 100.0)),
        _object("MESH", (-50.0, 0.0, 0.0), visible=False),
    ]
    fake = FakeBlender(tmp_path, with_view=False, camera=None, objects=objects)
    session = _session(monkeypatch, fake, tmp_path, attached=False)

    result = session.capture_viewport(width=400, height=300)

    assert result["capture_method"] == "render_temp_camera_framing"
    assert (result["width"], result["height"]) == (400, 300)
    camera = fake.renders[0]["camera"]
    # Euler XYZ (x, 0, z) turns the camera's -Z into this direction.
    rx, _ry, rz = camera.rotation_euler
    look = (-math.sin(rz) * math.sin(rx), math.cos(rz) * math.sin(rx), -math.cos(rx))
    center = (
        5.0,
        0.0,
        0.0,
    )  # the two visible meshes; the light and hidden mesh are ignored
    to_center = [c - p for c, p in zip(center, camera.location)]
    distance = math.sqrt(sum(v * v for v in to_center))
    assert look == pytest.approx([v / distance for v in to_center])
    # The bounding sphere of the 4 x 2 x 2 box fits the narrower (vertical) field.
    radius = math.dist((3, -1, -1), (7, 1, 1)) / 2
    half_field = math.atan(18 / 50 * 300 / 400)
    assert distance * math.sin(half_field) > radius
    assert (
        camera.data.clip_start
        < distance - radius
        < distance + radius
        < camera.data.clip_end
    )
    assert fake.scene.camera is None and fake.removed == [camera]


def test_attached_render_without_a_3d_view_frames_the_visible_objects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A window showing no 3D View has no view to align to; the camera frames instead."""
    fake = FakeBlender(tmp_path, with_view=False, camera=None)
    session = _session(monkeypatch, fake, tmp_path, attached=True)

    result = session.capture_viewport(width=300, height=200, method="render")

    assert result["capture_method"] == "render_temp_camera_framing"
    assert (result["width"], result["height"]) == (300, 200)
    (render,) = fake.renders
    assert render["camera"].name == "simul_temp_camera"
    assert fake.scene.camera is None and fake.removed == [render["camera"]]


def test_temporary_camera_is_removed_when_placing_it_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeBlender(tmp_path, camera=None)
    fake.view.update.side_effect = RuntimeError("view gone")
    session = _session(monkeypatch, fake, tmp_path)

    with pytest.raises(RuntimeError, match="view gone"):
        session.capture_viewport(method="render")

    (camera,) = fake.linked
    assert fake.removed == [camera] and fake.cameras_removed == [camera.data]
    assert fake.scene.camera is None and fake.renders == []


def test_a_failed_screenshot_is_an_error_and_restores_the_output_format(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, gpu: FakeGPU
) -> None:
    fake = FakeBlender(tmp_path, shading="RENDERED", engine="CYCLES")
    monkeypatch.setattr(
        fake.bpy.ops.screen, "screenshot_area", lambda filepath: {"CANCELLED"}
    )
    session = _session(monkeypatch, fake, tmp_path)

    with pytest.raises(RuntimeError, match="could not read the 3D View"):
        session.capture_viewport()

    assert vars(fake.settings) == vars(_image_settings())
    assert fake.images.live == [] and gpu.offscreens == []


def test_temporary_camera_is_removed_when_the_render_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeBlender(tmp_path, camera=None)
    fake.render_error = RuntimeError("GPU lost")
    session = _session(monkeypatch, fake, tmp_path)

    with pytest.raises(RuntimeError, match="GPU lost"):
        session.capture_viewport(method="render")

    assert fake.scene.camera is None
    assert len(fake.removed) == 1 and fake.removed[0].name == "simul_temp_camera"
    assert vars(fake.settings) == vars(_image_settings())
    assert fake.render.resolution_percentage == 50


def test_no_writable_capture_directory_fails_before_touching_the_scene(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeBlender(tmp_path, camera=None)
    monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
    monkeypatch.setattr(blender_runtime, "bpy", fake.bpy)
    session = blender_runtime.BlenderRuntimeSession(
        path_policy=PathPolicy(enabled=True, allowed_paths=[], project_root=tmp_path),
        scene_scope=True,
    )
    with pytest.raises(PermissionError, match="capture directory"):
        session.capture_viewport(method="render")
    assert fake.linked == [] and fake.renders == []


# -- sequences --------------------------------------------------------------


def test_sequence_captures_each_frame_with_the_resolved_method(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeBlender(tmp_path)
    fake.scene.frame_current = 7
    fake.scene.frame_set = lambda frame: setattr(fake.scene, "frame_current", frame)
    session = _session(monkeypatch, fake, tmp_path)
    calls: List[Dict[str, Any]] = []

    def _capture(**kwargs: Any) -> Dict[str, Any]:
        calls.append({**kwargs, "frame": fake.scene.frame_current})
        return {"image_base64": "X", "capture_method": "viewport"}

    monkeypatch.setattr(session, "capture_viewport", _capture)

    result = session.capture_viewport_sequence(1, 3, width=64, height=64)

    assert [c["frame"] for c in calls] == [1, 2, 3]
    assert {c["method"] for c in calls} == {"viewport"}
    assert result["capture_method"] == "viewport" and fake.scene.frame_current == 7
    empty = session.capture_viewport_sequence(5, 1, method="render")
    assert empty == {"frames": [], "frame_count": 0, "capture_method": "render"}
