# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Small, scene-independent orientation cube for the local workspace."""

from __future__ import annotations

from itertools import pairwise
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from imgui_bundle import imgui
    from numpy.typing import ArrayLike

    from newton._src.viewer.gui.viewport import ViewportControls


class OrientationGizmo:
    """Present camera orientation and handle view snapping, orbiting, and corner placement.

    ViewportControls owns this presenter; it changes the existing camera rather
    than maintaining a second camera or scene representation.
    """

    _half_extent = 19.5

    def __init__(self, controls: ViewportControls) -> None:
        """Initialize corner placement, hover fading, drag state, and optional camera animation."""
        self.controls = controls
        self.corner = "bottom_left"
        self._press = None
        self._moving = None
        self._animation = None
        self._faces = []
        self._opacity = 0.65
        self._fade_time = perf_counter()
        self.rect = (0, 0, 0, 0)

    def snap(self, direction: ArrayLike) -> None:
        """Schedule an animated orbit toward a view direction while preserving the current pivot."""
        camera = self.controls.workspace.viewer.camera
        start = camera.yaw, camera.pitch
        position, pivot = camera.pos, camera.pivot
        self.controls.camera_view(direction)
        target = camera.yaw, camera.pitch
        camera.pos, camera.pivot = position, pivot
        camera.yaw, camera.pitch = start
        self._animation = perf_counter(), start, target

    def _animate(self) -> None:
        """Advance the 0.18-second view transition along the shortest yaw arc and clear it on completion."""
        if self._animation is None:
            return
        start_time, start, target = self._animation
        t = min(1.0, (perf_counter() - start_time) / 0.18)
        blend = t * t * (3 - 2 * t)
        yaw = start[0] + ((target[0] - start[0] + 180) % 360 - 180) * blend
        pitch = start[1] + (target[1] - start[1]) * blend
        camera = self.controls.workspace.viewer.camera
        camera.orbit(yaw - camera.yaw, pitch - camera.pitch)
        if t == 1.0:
            self._animation = None

    def draw(self) -> None:
        """Draw the orientation cube and axes, handle orbit or placement drags, and expose camera context settings."""
        self._animate()
        workspace = self.controls.workspace
        ui, camera, s = workspace.gui.ui.imgui, workspace.viewer.camera, workspace.widgets.scale
        vx, vy, vw, vh = workspace.rect
        width, height = 118 * s, 122 * s
        if vw < width + 20 * s or vh < height + 55 * s:
            return
        x = vx + vw - width - 10 * s if self.corner.endswith("right") else vx + 10 * s
        y = vy + 49 * s if self.corner.startswith("top") else vy + vh - height - 10 * s
        if self._moving is not None:
            x, y = self._moving
        self.rect = x, y, width, height
        ui.set_next_window_pos(ui.ImVec2(x, y))
        ui.set_next_window_size(ui.ImVec2(width, height))
        ui.push_style_var(ui.StyleVar_.window_padding, ui.ImVec2(0, 0))
        ui.push_style_var(ui.StyleVar_.window_border_size, 0.0)
        flags = (
            ui.WindowFlags_.no_decoration
            | ui.WindowFlags_.no_move
            | ui.WindowFlags_.no_saved_settings
            | ui.WindowFlags_.no_background
            | ui.WindowFlags_.no_scrollbar
            | ui.WindowFlags_.no_scroll_with_mouse
        )
        ui.begin("##orientation_gizmo", flags=flags)
        ui.invisible_button("##orientation_cube", ui.ImVec2(width, height))
        hovered, active = ui.is_item_hovered(), ui.is_item_active()
        io = ui.get_io()
        mouse = np.array((io.mouse_pos.x, io.mouse_pos.y))
        if active and self._press is not None:
            origin, direction, kind, dragged, previous = self._press
            dragged = dragged or np.linalg.norm(mouse - origin) > 3 * s
            if dragged:
                delta = mouse - previous
                if kind == "move":
                    self._moving = (
                        float(np.clip(x + delta[0], vx, vx + vw - width)),
                        float(np.clip(y + delta[1], vy, vy + vh - height)),
                    )
                else:
                    # Drag the cube's surface with the pointer. ImGui's Y axis
                    # points down, and Camera's yaw convention reverses for Y-up.
                    # Relate arc length to the drawn cube, independent of DPI.
                    angles = np.rad2deg(delta / (self._half_extent * np.sqrt(2) * s))
                    yaw_sign = 1 if camera.up_axis == 1 else -1
                    camera.orbit(yaw_sign * angles[0], -angles[1] if kind == "orbit" else 0)
                previous = mouse.copy()
            self._press = origin, direction, kind, dragged, previous
        draw = ui.get_window_draw_list()
        emphasized = hovered or active or ui.is_popup_open("##orientation_settings")
        target_opacity = 1.0 if emphasized else 0.65
        now = perf_counter()
        self._opacity += (target_opacity - self._opacity) * -np.expm1(-(now - self._fade_time) / 0.10)
        self._fade_time = now

        def color(r: float, g: float, b: float, a: float = 1) -> int:
            return ui.get_color_u32(ui.ImVec4(r, g, b, a * self._opacity))

        def point(xy: ArrayLike) -> imgui.ImVec2:
            return ui.ImVec2(float(xy[0]), float(xy[1]))

        center = np.array((x + width * 0.5, y + 56 * s))
        right, up, toward = map(np.asarray, (camera.get_right(), camera.get_up(), -camera.get_front()))
        basis = np.array((right, -up))

        def project(vertices: ArrayLike) -> np.ndarray:
            return center + np.asarray(vertices) @ basis.T * (self._half_extent * s)

        def fill(polygon: ArrayLike, shade: float) -> None:
            signed_area = sum(
                a[0] * b[1] - b[0] * a[1] for a, b in zip(polygon, np.roll(polygon, -1, axis=0), strict=True)
            )
            ordered = polygon if signed_area >= 0 else polygon[::-1]
            draw.add_convex_poly_filled([point(p) for p in ordered], shade)

        faces = []
        labels = {(camera.up_axis, 1): "TOP", (camera.up_axis, -1): "BOTTOM"}
        front = np.zeros(3)
        front[2 if camera.up_axis == 1 else 1] = 1 if camera.up_axis == 1 else -1
        world_up = np.eye(3)[camera.up_axis]
        world_right = np.cross(-front, world_up)
        for vector, positive, negative in ((front, "FRONT", "BACK"), (world_right, "RIGHT", "LEFT")):
            axis = int(np.argmax(np.abs(vector)))
            labels[axis, int(vector[axis])] = positive
            labels[axis, -int(vector[axis])] = negative
        for axis in range(3):
            others = [i for i in range(3) if i != axis]
            for sign in (-1, 1):
                normal = np.eye(3)[axis] * sign
                facing = float(normal @ toward)
                if facing < 0.001:
                    continue
                vertices = np.zeros((4, 3))
                vertices[:, axis] = sign
                vertices[:, others] = ((-1, -1), (1, -1), (1, 1), (-1, 1))
                faces.append((facing, axis, sign, others, vertices, project(vertices)))
        faces.sort(key=lambda face: face[0])
        self._faces = faces
        # Resolve the target first, then shade its region on every adjacent
        # visible face. A corner is one 3D target, even across three polygons.
        hit = None
        if hovered:
            for _, axis, sign, others, _, polygon in faces:
                uv = np.linalg.solve(
                    np.column_stack((polygon[1] - polygon[0], polygon[3] - polygon[0])), mouse - polygon[0]
                )
                if (uv >= 0).all() and (uv <= 1).all():
                    hit = np.zeros(3)
                    hit[axis] = sign
                    hit[others] = np.where(uv < 0.23, -1, np.where(uv > 0.77, 1, 0))
            nearest = 5 * s
            for _, _, _, _, vertices, polygon in faces:
                for vertex, position in zip(vertices, polygon, strict=True):
                    distance = np.linalg.norm(mouse - position)
                    if distance < nearest:
                        hit, nearest = vertex.copy(), distance

        # Project fixed world-space endpoints just like the cube. Screen-space
        # normalization and overlap avoidance introduce jumps during orbiting.
        axis_colors = ((0.94, 0.43, 0.40), (0.49, 0.84, 0.55), (0.43, 0.69, 1.0))
        positions = center + basis.T * (44 * s)
        self._axis_heads = positions
        axis_hover = None
        for axis in np.argsort(toward):
            distance = np.linalg.norm(mouse - positions[axis])
            if hovered and (hit is None or np.count_nonzero(hit) == 1 or distance <= 5 * s) and distance <= 8 * s:
                axis_hover = int(axis)
                hit = np.eye(3)[axis] * (-1 if io.key_shift else 1)
        for axis in np.argsort(toward):
            rgb = axis_colors[axis]
            draw.add_line(point(center), point(positions[axis]), color(*rgb, 0.90), 2 * s)
        self._hover_direction = hit
        self._highlight_faces = []
        for facing, axis, sign, others, _vertices, polygon in faces:
            shade = 0.17 + 0.17 * facing
            fill(polygon, color(shade, shade, shade, 0.98))
            if hit is not None and hit[axis] == sign:
                edges = (0, 0.23, 0.77, 1)
                if np.count_nonzero(hit) == 1:
                    region = polygon
                else:
                    cells = [int(hit[i]) + 1 for i in others]
                    u0, u1 = edges[cells[0]], edges[cells[0] + 1]
                    v0, v1 = edges[cells[1]], edges[cells[1] + 1]
                    region = np.array(
                        [
                            polygon[0] + u * (polygon[1] - polygon[0]) + v * (polygon[3] - polygon[0])
                            for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))
                        ]
                    )
                fill(region, color(0.96, 0.61, 0.32, 0.55))
                for i in range(4):
                    draw.add_line(point(region[i]), point(region[(i + 1) % 4]), color(1.0, 0.74, 0.47, 0.85), s)
                self._highlight_faces.append((axis, sign))
            for i in range(4):
                draw.add_line(point(polygon[i]), point(polygon[(i + 1) % 4]), color(0.68, 0.68, 0.68, 0.9), s)
            if facing > 0.20 and all(np.linalg.norm(polygon.mean(axis=0) - p) > 16 * s for p in positions):
                text = labels[axis, sign]
                extent = ui.calc_text_size(text)
                label_center = polygon.mean(axis=0)
                available = np.ptp(polygon, axis=0)
                fit = min(1.0, 0.85 * available[0] / max(extent.x, 1), 0.65 * available[1] / max(extent.y, 1))
                font_size = ui.get_font_size() * fit
                if font_size >= 8 * s:
                    draw.add_text(
                        ui.get_font(),
                        font_size,
                        point(label_center - np.array((extent.x, extent.y)) * (0.5 * fit)),
                        color(0.95, 0.95, 0.95),
                        text,
                    )
        # The compass has a restrained back arc and a brighter front rim.
        ring_center = np.array((center[0], y + 98 * s))
        ring_distance = np.linalg.norm((mouse - ring_center) / (43 * s, 7 * s))
        ring_hover = hovered and 0.65 < ring_distance < 1.5 and axis_hover is None
        ring = [ring_center + np.array((np.cos(t) * 43 * s, np.sin(t) * 7 * s)) for t in np.linspace(0, 2 * np.pi, 49)]
        for a, b in pairwise(ring):
            front_arc = (a[1] + b[1]) * 0.5 >= ring_center[1]
            tint = color(0.96, 0.61, 0.32, 0.9) if ring_hover else color(0.64, 0.64, 0.64, 0.70 if front_arc else 0.30)
            draw.add_line(point(a), point(b), tint, s)
        for side in (-1, 1):
            tick = ring_center + np.array((side * 43 * s, 0))
            extent = np.array((0, 2 * s))
            draw.add_line(point(tick - extent), point(tick + extent), color(0.70, 0.70, 0.70, 0.7), s)

        for axis in np.argsort(toward):
            rgb = axis_colors[axis]
            position = positions[axis]
            over = axis == axis_hover
            radius = (8 if over else 7) * s
            # A darker rim and a small upper highlight give each cap depth.
            draw.add_circle_filled(point(position), radius + s, color(0.12, 0.12, 0.12, 0.95), 24)
            shade = 0.72 + 0.18 * toward[axis]
            draw.add_circle_filled(point(position), radius, color(*(c * shade for c in rgb)), 24)
            draw.add_circle(point(position), radius, color(*rgb, 0.95), 24, s)
            draw.add_circle_filled(point(position + np.array((-2, -3)) * s), 2 * s, color(1, 1, 1, 0.18), 12)
            label = "XYZ"[axis]
            font_size = 11 * s
            extent = ui.calc_text_size(label)
            fit = font_size / ui.get_font_size()
            draw.add_text(
                ui.get_font(),
                font_size,
                point(position - np.array((extent.x, extent.y)) * 0.5 * fit),
                color(0.09, 0.09, 0.09),
                label,
            )
        grip = np.array((x + width - 13 * s, y + 10 * s))
        if hovered or self._moving is not None:
            for dx in (-3, 1):
                for dy in (-3, 1):
                    draw.add_circle_filled(point(grip + np.array((dx, dy)) * s), s, color(0.70, 0.70, 0.70))
        if ui.is_item_activated():
            self._animation = None
            kind = "move" if np.linalg.norm(mouse - grip) < 13 * s else "yaw" if ring_hover else "orbit"
            self._press = mouse.copy(), hit, kind, False, mouse.copy()
        if ui.is_item_deactivated() and self._press is not None:
            _, direction, kind, dragged, _ = self._press
            if kind == "move" and self._moving is not None:
                mx, my = self._moving
                self.corner = ("top" if my + height * 0.5 < vy + vh * 0.5 else "bottom") + (
                    "_left" if mx + width * 0.5 < vx + vw * 0.5 else "_right"
                )
                self._moving = None
            elif not dragged and direction is not None:
                self.snap(direction)
            self._press = None
        if hovered and io.mouse_wheel:
            camera.dolly(io.mouse_wheel * 0.12)
        if hovered and not active and ui.is_item_hovered(ui.HoveredFlags_.delay_normal):
            if np.linalg.norm(mouse - grip) < 13 * s:
                tip = "Drag to another corner"
            elif axis_hover is not None:
                tip = "Click to look along this axis · Shift-click for the opposite view"
            elif ring_hover:
                tip = "Drag to orbit horizontally"
            else:
                tip = "Click to align · Drag to orbit · Right-click for camera settings"
            ui.set_tooltip(tip)
        self.controls.prepare_popup()
        if ui.begin_popup_context_item("##orientation_settings"):
            self.controls._camera_menu()
            ui.separator_text("Corner")
            for corner in ("top_right", "top_left", "bottom_right", "bottom_left"):
                if ui.selectable(corner.replace("_", " ").title(), self.corner == corner)[0]:
                    self.corner = corner
            ui.end_popup()
        ui.pop_style_var()
        ui.end()
        ui.pop_style_var(2)
