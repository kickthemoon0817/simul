"""Transient, labelled agent pointers drawn in Blender, never OS input events."""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import bpy


class AgentCursors:
    """One latest-action marker per agent/window, scoped to live editor identities."""

    lifetime = 120.0
    limit = 64

    def __init__(self) -> None:
        self.markers: dict[tuple[str, str], dict[str, Any]] = {}
        self.handlers: list[tuple[Any, Any]] = []
        self.suspended = False

    def update(
        self, window: Any, area: Any, agent_id: str, position: list[float], label: str
    ) -> dict[str, Any]:
        if not self.handlers:
            for space in (bpy.types.SpaceView3D, bpy.types.SpaceProperties):
                self.handlers.append(
                    (
                        space,
                        space.draw_handler_add(self.draw, (), "WINDOW", "POST_PIXEL"),
                    )
                )
        self.prune()
        key = (str(window.as_pointer()), agent_id)
        if key not in self.markers and len(self.markers) >= self.limit:
            oldest = min(self.markers, key=lambda k: self.markers[k]["updated_at"])
            del self.markers[oldest]
        palette = (
            (0.15, 0.85, 1.0, 1.0),
            (1.0, 0.55, 0.2, 1.0),
            (0.8, 0.4, 1.0, 1.0),
            (0.3, 1.0, 0.5, 1.0),
        )
        marker = {
            "agent_id": agent_id,
            "window_id": key[0],
            "area_id": str(area.as_pointer()),
            "area_type": area.type,
            "scene_id": str(window.scene.as_pointer()),
            "workspace_id": str(window.workspace.as_pointer()),
            "position": list(position),
            "label": label,
            "updated_at": time.time(),
            "color": list(
                palette[hashlib.sha256(agent_id.encode()).digest()[0] % len(palette)]
            ),
        }
        self.markers[key] = marker
        self.redraw()
        return dict(marker)

    def inspect(self, window: Any) -> list[dict[str, Any]]:
        self.prune()
        return [
            dict(m)
            for m in self.markers.values()
            if m["window_id"] == str(window.as_pointer())
        ]

    def clear(self, window: Any, agent_id: str) -> bool:
        removed = (
            self.markers.pop((str(window.as_pointer()), agent_id), None) is not None
        )
        self.redraw()
        return removed

    def prune(self) -> None:
        windows = {str(w.as_pointer()): w for w in bpy.context.window_manager.windows}
        stale = []
        for key, marker in self.markers.items():
            window = windows.get(marker["window_id"])
            if (
                window is None
                or time.time() - marker["updated_at"] >= self.lifetime
                or str(window.scene.as_pointer()) != marker["scene_id"]
                or str(window.workspace.as_pointer()) != marker["workspace_id"]
                or not any(
                    str(a.as_pointer()) == marker["area_id"]
                    and a.type == marker["area_type"]
                    for a in window.screen.areas
                )
            ):
                stale.append(key)
        for key in stale:
            del self.markers[key]
        if stale:
            self.redraw()

    @staticmethod
    def redraw() -> None:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type in {"VIEW_3D", "PROPERTIES"}:
                    area.tag_redraw()

    def stop(self) -> None:
        self.markers.clear()
        for space, handler in self.handlers:
            space.draw_handler_remove(handler, "WINDOW")
        self.handlers.clear()
        self.redraw()

    def visible(self) -> list[dict[str, Any]]:
        # Blender supplies the actual drawing window/area, not the last request's context.
        window, area, region = bpy.context.window, bpy.context.area, bpy.context.region
        if self.suspended or window is None or area is None or region is None:
            return []
        return [
            m
            for m in self.markers.values()
            if (
                m["window_id"] == str(window.as_pointer())
                and m["area_id"] == str(area.as_pointer())
                and m["scene_id"] == str(window.scene.as_pointer())
                and m["workspace_id"] == str(window.workspace.as_pointer())
                and time.time() - m["updated_at"] < self.lifetime
            )
        ]

    def draw(self) -> None:
        visible = self.visible()
        if not visible:
            return
        area, region = bpy.context.area, bpy.context.region
        import blf  # type: ignore[import-not-found]
        import gpu  # type: ignore[import-not-found]
        from gpu_extras.batch import batch_for_shader  # type: ignore[import-not-found]

        scale = bpy.context.preferences.system.ui_scale
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        previous_blend = gpu.state.blend_get()
        gpu.state.blend_set("ALPHA")
        try:
            for index, marker in enumerate(
                sorted(visible, key=lambda m: m["agent_id"])
            ):
                x = marker["position"][0] * (region.width - 1)
                y = marker["position"][1] * (region.height - 1)
                size = 13 * scale
                shader.bind()
                shader.uniform_float("color", marker["color"])
                batch_for_shader(
                    shader,
                    "TRIS",
                    {
                        "pos": [
                            (x, y),
                            (x + size, y - size),
                            (x + size * 0.3, y - size * 1.4),
                        ]
                    },
                ).draw(shader)
                blf.size(0, 13 * scale)
                blf.color(0, *marker["color"])
                blf.position(
                    0,
                    min(x + size, max(0, region.width - 160 * scale)),
                    max(4, y + 4 * scale),
                    0,
                )
                blf.draw(0, marker["agent_id"])
                # A stable status list keeps coincident pointers distinguishable.
                label = f"{marker['agent_id']}: {marker['label']}"
                panel_x = 80 * scale if area.type == "VIEW_3D" else 16 * scale
                panel_y = max(
                    4,
                    region.height
                    - ((110 if area.type == "VIEW_3D" else 40) + 22 * index) * scale,
                )
                while (
                    len(label) > 4
                    and blf.dimensions(0, label)[0]
                    > region.width - panel_x - 16 * scale
                ):
                    label = label[:-4] + "..."
                width = blf.dimensions(0, label)[0] + 12 * scale
                left, right = panel_x - 6 * scale, panel_x + width
                bottom, top = panel_y - 4 * scale, panel_y + 16 * scale
                shader.uniform_float("color", (0.025, 0.025, 0.025, 0.85))
                batch_for_shader(
                    shader,
                    "TRIS",
                    {
                        "pos": [
                            (left, bottom),
                            (right, bottom),
                            (right, top),
                            (left, bottom),
                            (right, top),
                            (left, top),
                        ]
                    },
                ).draw(shader)
                blf.position(0, panel_x, panel_y, 0)
                blf.draw(0, label)
        finally:
            gpu.state.blend_set(previous_blend)


class AgentObservations(AgentCursors):
    """A short eye badge and fading border after an agent views a capture."""

    lifetime = 2.4

    def show(self, window: Any, area: Any, agent_id: str) -> dict[str, Any]:
        if (
            not isinstance(agent_id, str)
            or not 1 <= len(agent_id) <= 64
            or not agent_id.isprintable()
            or not agent_id.strip()
        ):
            raise ValueError("agent_id must be a printable label of 1 to 64 characters")
        return self.update(window, area, agent_id, [0.5, 0.5], "Viewing scene")

    def prune(self) -> None:
        super().prune()
        if self.markers:
            # The bridge's existing timer drives the fade, including the last erase.
            self.redraw()

    def draw(self) -> None:
        visible = self.visible()
        if not visible:
            return
        import blf  # type: ignore[import-not-found]
        import gpu  # type: ignore[import-not-found]
        from gpu_extras.batch import batch_for_shader  # type: ignore[import-not-found]

        region = bpy.context.region
        scale = bpy.context.preferences.system.ui_scale
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        previous_blend = gpu.state.blend_get()
        gpu.state.blend_set("ALPHA")

        def rectangle(x: float, y: float, width: float, height: float) -> None:
            batch_for_shader(
                shader,
                "TRIS",
                {
                    "pos": [
                        (x, y),
                        (x + width, y),
                        (x + width, y + height),
                        (x, y),
                        (x + width, y + height),
                        (x, y + height),
                    ]
                },
            ).draw(shader)

        try:
            shader.bind()
            for index, marker in enumerate(
                sorted(visible, key=lambda m: m["agent_id"])
            ):
                age = max(0.0, time.time() - marker["updated_at"])
                fade = min(1.0, (self.lifetime - age) / 0.8)
                alpha = fade * (0.5 + 0.25 * math.cos(age * math.tau / self.lifetime))
                color = (*marker["color"][:3], alpha)
                shader.uniform_float("color", color)
                inset = (3 + index * 4) * scale
                thickness = 2 * scale
                width, height = region.width - 2 * inset, region.height - 2 * inset
                if width > 0 and height > 0:
                    rectangle(inset, inset, width, thickness)
                    rectangle(
                        inset, region.height - inset - thickness, width, thickness
                    )
                    rectangle(inset, inset, thickness, height)
                    rectangle(
                        region.width - inset - thickness, inset, thickness, height
                    )
                # Bottom-left avoids the existing latest-action labels and the gizmo.
                x, y = 32 * scale, (40 + index * 28) * scale
                label = f"{marker['agent_id']} · Viewing scene"
                blf.size(0, 13 * scale)
                while (
                    len(label) > 4
                    and blf.dimensions(0, label)[0] > region.width - 100 * scale
                ):
                    label = label[:-4] + "..."
                shader.uniform_float("color", (0.025, 0.035, 0.045, 0.85 * fade))
                rectangle(
                    x - 18 * scale,
                    y - 7 * scale,
                    blf.dimensions(0, label)[0] + 54 * scale,
                    26 * scale,
                )
                shader.uniform_float("color", (*marker["color"][:3], fade))
                # An eye outline and pupil, built from triangles for portable thickness.
                eye = []
                for step in range(40):
                    a, b = step * math.tau / 40, (step + 1) * math.tau / 40
                    outer_a = (
                        x + 10 * scale * math.cos(a),
                        y + 5 * scale + 6 * scale * math.sin(a),
                    )
                    outer_b = (
                        x + 10 * scale * math.cos(b),
                        y + 5 * scale + 6 * scale * math.sin(b),
                    )
                    inner_a = (
                        x + 8 * scale * math.cos(a),
                        y + 5 * scale + 4 * scale * math.sin(a),
                    )
                    inner_b = (
                        x + 8 * scale * math.cos(b),
                        y + 5 * scale + 4 * scale * math.sin(b),
                    )
                    eye.extend(
                        [
                            outer_a,
                            outer_b,
                            inner_b,
                            outer_a,
                            inner_b,
                            inner_a,
                            (x, y + 5 * scale),
                            (
                                x + 2.5 * scale * math.cos(a),
                                y + 5 * scale + 2.5 * scale * math.sin(a),
                            ),
                            (
                                x + 2.5 * scale * math.cos(b),
                                y + 5 * scale + 2.5 * scale * math.sin(b),
                            ),
                        ]
                    )
                batch_for_shader(shader, "TRIS", {"pos": eye}).draw(shader)
                blf.color(0, *marker["color"][:3], fade)
                blf.position(0, x + 20 * scale, y, 0)
                blf.draw(0, label)
        finally:
            gpu.state.blend_set(previous_blend)


cursors = AgentCursors()
observations = AgentObservations()


@contextmanager
def hide_annotations() -> Iterator[None]:
    """Keep agent UI annotations out of captured images; restore even on failure."""
    previous = cursors.suspended, observations.suspended
    cursors.suspended = observations.suspended = True
    try:
        yield
    finally:
        cursors.suspended, observations.suspended = previous
