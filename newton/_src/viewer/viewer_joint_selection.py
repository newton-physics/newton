# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Depth-aware joint annotations with a shared, readable size for each selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ..sim.enums import JointType

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from numpy.typing import ArrayLike

    from newton._src.viewer.gui.workspace import ViewerWorkspace


@wp.kernel
def _sample_joint_frames(
    indices: wp.array[int],
    types: wp.array[int],
    parents: wp.array[int],
    children: wp.array[int],
    local_parents: wp.array[wp.transform],
    local_children: wp.array[wp.transform],
    starts: wp.array[int],
    axes: wp.array[wp.vec3],
    lower: wp.array[float],
    upper: wp.array[float],
    bodies: wp.array[wp.transform],
    worlds: wp.array[int],
    offsets: wp.array[wp.vec3],
    layer: wp.transform,
    output: wp.array2d[float],
) -> None:
    i = wp.tid()
    joint = indices[i]
    parent = local_parents[joint]
    child = local_children[joint]
    if parents[joint] >= 0:
        parent = bodies[parents[joint]] * parent
    if children[joint] >= 0:
        child = bodies[children[joint]] * child
    relative = wp.quat_inverse(wp.transform_get_rotation(parent)) * wp.transform_get_rotation(child)
    start = starts[joint]
    value = 2.0 * wp.atan2(wp.length(wp.vec3(relative[0], relative[1], relative[2])), wp.abs(relative[3]))
    if types[joint] == int(JointType.REVOLUTE):
        value = wp.quat_twist_angle_signed(axes[start], relative)
        lo, hi = lower[start], upper[start]
        if wp.isfinite(lo) and wp.isfinite(hi) and lo <= hi and wp.abs(lo) < 1.0e6 and wp.abs(hi) < 1.0e6:
            value += (2.0 * wp.pi) * wp.round(((lo + hi) * 0.5 - value) / (2.0 * wp.pi))
    elif types[joint] == int(JointType.PRISMATIC):
        value = wp.dot(
            wp.transform_get_translation(child) - wp.transform_get_translation(parent),
            wp.transform_vector(parent, axes[start]),
        )
    offset = wp.vec3(0.0)
    world = worlds[joint]
    if world >= 0 and world < offsets.shape[0]:
        offset = offsets[world]
    parent = layer * wp.transform(wp.transform_get_translation(parent) + offset, wp.transform_get_rotation(parent))
    child = layer * wp.transform(wp.transform_get_translation(child) + offset, wp.transform_get_rotation(child))
    parent_position, child_position = wp.transform_get_translation(parent), wp.transform_get_translation(child)
    parent_rotation = wp.quat_to_matrix(wp.transform_get_rotation(parent))
    child_rotation = wp.quat_to_matrix(wp.transform_get_rotation(child))
    for axis in range(3):
        output[i, axis] = parent_position[axis]
        output[i, 12 + axis] = child_position[axis]
        for column in range(3):
            output[i, 3 + axis * 3 + column] = parent_rotation[axis, column]
            output[i, 15 + axis * 3 + column] = child_rotation[axis, column]
    output[i, 24] = value
    # Estimate link length from local anchors, ignoring world-space root offsets.
    # Piggyback on the existing selected-joint readback instead of fetching meshes.
    length = wp.length(wp.transform_get_translation(local_children[joint]))
    if parents[joint] >= 0:
        length = wp.max(length, wp.length(wp.transform_get_translation(local_parents[joint])))
    output[i, 25] = length


class JointSelectionGroup:
    """Gather selected joint frames once, then batch their screen-sized glyphs.

    GPU sampling reads displayed body transforms, so annotations follow maximal
    coordinate solvers and history without assuming joint_q is current. One
    compact host transfer covers only selected joints. Tessellation is reused
    while sampled frames, camera, visibility, and sizing settings are unchanged.
    """

    def __init__(self, workspace: ViewerWorkspace, indices: Sequence[int] | np.ndarray) -> None:
        """Cache selected joint metadata and allocate one device buffer for their live frames."""
        self.joints = tuple(JointSelection(workspace, index) for index in sorted(indices))
        device = workspace.model.device
        self.indices = wp.array([joint.index for joint in self.joints], dtype=int, device=device)
        self.frames = wp.empty((len(self.joints), 26), dtype=float, device=device)
        self.empty_offsets = wp.empty(0, dtype=wp.vec3, device=device)
        self._cached_frames = None
        self._cached_view = None
        self._cached_triangles = None
        self._kinds = np.array([joint.kind for joint in self.joints])
        self._enabled = np.array([1.0 if joint.enabled else 0.45 for joint in self.joints])
        self._bases = np.array([getattr(joint, "axis_basis", np.eye(3)) for joint in self.joints])
        self._limits = np.array(
            [(joint.lower[0], joint.upper[0]) if joint.limited else (np.nan, np.nan) for joint in self.joints]
        )

    def draw(self, workspace: ViewerWorkspace, *, compact: bool = False) -> None:
        """Sample selected joint frames and submit batched glyph geometry for the displayed State.

        Copy only selected frames to the host. Reuse generated triangles when
        frame and view inputs are unchanged; compact mode batches glyphs for
        large joint selections.
        """
        viewer, state = workspace.viewer, workspace.viewer._last_state
        if (
            state is None
            or viewer.is_rendering_paused()
            or viewer._layer_force_hidden()
            or not any(viewer._should_render_world(joint.world) for joint in self.joints)
        ):
            return
        model = workspace.model
        wp.launch(
            _sample_joint_frames,
            dim=len(self.joints),
            inputs=[
                self.indices,
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_X_p,
                model.joint_X_c,
                model.joint_qd_start,
                model.joint_axis,
                model.joint_limit_lower,
                model.joint_limit_upper,
                state.body_q,
                model.joint_world,
                viewer.world_offsets if viewer.world_offsets is not None else self.empty_offsets,
                viewer.layer.xform,
            ],
            outputs=[self.frames],
            device=model.device,
        )
        # Sample the displayed state on the device; never copy unrelated worlds.
        draw_list = _JointDrawList(workspace)
        frames = self.frames.numpy()
        visible = tuple(viewer._should_render_world(joint.world) for joint in self.joints)
        controls = workspace.viewport_controls
        view_key = (
            tuple(draw_list.view.flat),
            tuple(draw_list.projection.flat),
            tuple(draw_list.camera),
            draw_list.rect,
            draw_list.scale,
            compact,
            visible,
            controls.joint_gizmo_scale,
            controls.joint_gizmo_auto_size,
        )
        if view_key == self._cached_view and np.array_equal(frames, self._cached_frames):
            viewer.renderer._joint_triangles = self._cached_triangles
            return
        finite = np.isfinite(frames).all(axis=1)
        base_radius = 28.0 if compact else 48.0
        radius = base_radius
        if controls.joint_gizmo_auto_size:
            depths = -((frames[:, :3] - draw_list.camera) @ draw_list.view[2, :3])
            sized = finite & visible & (frames[:, 25] > 0) & (depths > draw_list.near)
            if np.any(sized):
                link_pixels = frames[sized, 25] / (depths[sized] * draw_list.radius_per_depth)
                radius = float(np.clip(0.25 * np.median(link_pixels), base_radius, base_radius * 2))
        # Use the same projected radius for every selected joint, including
        # selections spanning differently sized links or multiple worlds.
        draw_list.radius_pixels = radius * controls.joint_gizmo_scale
        for i, (joint, frame) in enumerate(zip(self.joints, frames, strict=True)):
            if not finite[i] or joint.child < 0 or not visible[i]:
                continue
            joint.parent_pose = frame[:3], frame[3:12].reshape(3, 3)
            joint.child_pose = frame[12:15], frame[15:24].reshape(3, 3)
            joint.value = float(frame[24])
            if not compact:
                joint.draw(workspace, draw_list=draw_list)
        if compact:
            self._draw_compact(draw_list, frames, finite & visible)
        draw_list.flush()
        self._cached_view, self._cached_frames = view_key, frames
        self._cached_triangles = viewer.renderer._joint_triangles

    def _draw_compact(self, draw: _JointDrawList, frames: np.ndarray, visible: np.ndarray) -> None:
        """Batch articulation glyphs by primitive and joint type."""
        origins = frames[:, :3]
        screen, valid = draw.project(origins)
        x, y, width, height = draw.rect
        visible = visible & valid & (screen[:, 0] >= x) & (screen[:, 0] <= x + width)
        visible &= (screen[:, 1] >= y) & (screen[:, 1] <= y + height)
        visible &= np.array([joint.child >= 0 for joint in self.joints])
        ids = np.flatnonzero(visible)
        if not len(ids):
            return
        origin = origins[ids]
        child = frames[ids, 12:15]
        rotation = frames[ids, 3:12].reshape(-1, 3, 3)
        child_rotation = frames[ids, 15:24].reshape(-1, 3, 3)
        radius = -((origin - draw.camera) @ draw.view[2, :3]) * draw.radius_per_depth * draw.radius_pixels
        kinds, enabled = self._kinds[ids], self._enabled[ids]
        reference, current, neutral = JointSelection.reference, JointSelection.current, JointSelection.neutral

        def colors(rgb: ArrayLike, alpha: float = 1, mask: slice | np.ndarray = slice(None)) -> np.ndarray:
            rgba = np.empty((len(enabled[mask]), 4))
            rgba[:, :3], rgba[:, 3] = rgb, enabled[mask] * alpha
            return rgba

        def lines(
            points: ArrayLike,
            rgb: ArrayLike,
            thickness: float = 1.5,
            alpha: float = 1,
            *,
            mask: slice | np.ndarray = slice(None),
            halo: bool = True,
            arrow: bool = False,
        ) -> None:
            background = colors((0.025, 0.035, 0.05), 0.60, mask) if halo else None
            draw.lines(points, colors(rgb, alpha, mask), thickness, halo=background, arrow=arrow)

        separated = np.linalg.norm(child - origin, axis=1) > radius * 0.02
        if separated.any():
            delta = child[separated] - origin[separated]
            t = np.arange(12).reshape(6, 2) / 12
            paths = origin[separated, None, None, :] + t[None, :, :, None] * delta[:, None, None, :]
            draw.lines(paths.reshape(-1, 2, 3), np.repeat(colors(neutral, 0.6, separated), 6, axis=0), 1)
        # Frame joints (including cable joints) share a single three-axis batch.
        frame = (kinds != int(JointType.REVOLUTE)) & (kinds != int(JointType.PRISMATIC))
        if frame.any():
            starts = np.broadcast_to(origin[frame, None, :], (frame.sum(), 3, 3))
            tips = starts + rotation[frame].transpose(0, 2, 1) * radius[frame, None, None] * 0.85
            rgba = np.ones((frame.sum(), 3, 4))
            rgba[:, :, :3] = JointSelection.axis_colors
            rgba[:, :, 3] = enabled[frame, None]
            draw.lines(np.stack((starts, tips), axis=2).reshape(-1, 2, 3), rgba.reshape(-1, 4), 1.7, arrow=True)
            starts = np.broadcast_to(child[frame, None, :], (frame.sum(), 3, 3))
            tips = starts + child_rotation[frame].transpose(0, 2, 1) * radius[frame, None, None] * 0.65
            draw.lines(
                np.stack((starts, tips), axis=2).reshape(-1, 2, 3),
                np.repeat(colors(current, 0.8, frame), 3, axis=0),
                1.5,
            )
        angles = np.linspace(0, 2 * np.pi, 41)
        ball = kinds == int(JointType.BALL)
        if ball.any():
            for axis in range(3):
                u, v = rotation[ball, :, axis], rotation[ball, :, (axis + 1) % 3]
                points = origin[ball, None, :] + radius[ball, None, None] * (
                    np.cos(angles)[None, :, None] * u[:, None, :] + np.sin(angles)[None, :, None] * v[:, None, :]
                )
                lines(points, neutral, 1, 0.3, mask=ball, halo=False)
        revolute = kinds == int(JointType.REVOLUTE)
        if revolute.any():
            basis = rotation[revolute] @ self._bases[ids[revolute]]
            u, v, axis = (basis[:, :, i] for i in range(3))
            o, r, value = origin[revolute], radius[revolute], frames[ids[revolute], 24]

            def arc(a: np.ndarray) -> np.ndarray:
                return o[:, None, :] + r[:, None, None] * (
                    np.cos(a)[..., None] * u[:, None, :] + np.sin(a)[..., None] * v[:, None, :]
                )

            lines(arc(angles[None, :]), reference, 1, 0.55, mask=revolute, halo=False)
            lines(np.stack((o, o + r[:, None] * u), axis=1), reference, 1.2, 0.85, mask=revolute)
            active = np.clip(value, -2 * np.pi, 2 * np.pi)[:, None] * np.linspace(0, 1, 41)
            lines(arc(active), current, mask=revolute)
            pointer = o + r[:, None] * (np.cos(value)[:, None] * u + np.sin(value)[:, None] * v)
            lines(np.stack((o, pointer), axis=1), current, 2, mask=revolute)
            lines(
                np.stack((o - axis * r[:, None] * 0.55, o + axis * r[:, None] * 0.75), axis=1),
                reference,
                1.7,
                mask=revolute,
                arrow=True,
            )
        prismatic = kinds == int(JointType.PRISMATIC)
        if prismatic.any():
            basis = rotation[prismatic] @ self._bases[ids[prismatic]]
            u, axis = basis[:, :, 0], basis[:, :, 2]
            o, r, value = origin[prismatic], radius[prismatic], frames[ids[prismatic], 24]
            limits = self._limits[ids[prismatic]]
            limited = np.isfinite(limits[:, 0])
            lo = np.minimum(np.where(limited, limits[:, 0], -r), np.minimum(value, 0))
            hi = np.maximum(np.where(limited, limits[:, 1], r), np.maximum(value, 0))
            lines(
                np.stack((o + axis * lo[:, None], o + axis * hi[:, None]), axis=1), reference, 4, 0.35, mask=prismatic
            )
            carriage = o + axis * value[:, None]
            lines(np.stack((o, carriage), axis=1), current, 3, mask=prismatic, arrow=True)
            lines(
                np.stack((carriage - u * r[:, None] * 0.2, carriage + u * r[:, None] * 0.2), axis=1),
                current,
                4,
                mask=prismatic,
            )
            if limited.any():
                for side in range(2):
                    center = o[limited] + axis[limited] * limits[limited, side, None]
                    offset = u[limited] * r[limited, None] * 0.12
                    draw.lines(
                        np.stack((center - offset, center + offset), axis=1),
                        colors(reference, mask=prismatic)[limited],
                        2,
                    )
        background = colors((0.025, 0.035, 0.05))
        if separated.any():
            draw.markers(origin[separated], colors(reference, mask=separated), background[separated], square=True)
        draw.markers(child, colors(current), background)


class _JointDrawList:
    """Build glyph triangles in logical screen pixels with their original scene depth."""

    def __init__(self, workspace: ViewerWorkspace) -> None:
        """Capture camera and viewport transforms and initialize a shared glyph command buffer."""
        self.renderer, self.scale = workspace.viewer.renderer, workspace.widgets.scale
        self.rect = workspace.rect
        camera = workspace.viewer.camera
        self.view = camera.get_view_matrix().reshape(4, 4).T
        self.projection = camera.get_projection_matrix().reshape(4, 4).T
        self.camera = np.asarray(camera.pos)
        self.near = camera.near
        self.radius_per_depth = 2 * np.tan(np.radians(camera.fov) / 2) * self.scale / max(1, self.rect[3])
        self.radius_pixels: float = 48.0
        self.points = []
        self.operations = []
        self.count = 0

    def project(self, points: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Project display-space points to GUI pixels, returning a depth-validity mask and retaining clip coordinates."""
        x, y, width, height = self.rect
        local = (np.atleast_2d(points) - self.camera) @ self.view[:3, :3].T
        clip = local @ self.projection[:, :3].T + self.projection[:, 3]
        self.clip = clip
        valid = (clip[:, 3] > self.near) & (clip[:, 2] <= clip[:, 3])
        screen = clip[:, :2] / np.maximum(clip[:, 3:4], 1e-9)
        screen *= (width / 2, -height / 2)
        screen += (x + width / 2, y + height / 2)
        return screen, valid

    def color(self, rgb: ArrayLike, alpha: float = 1) -> tuple[float, float, float, float]:
        """Combine an RGB color with alpha into the float RGBA format used by glyph commands."""
        return (*rgb, alpha)

    def add(self, kind: str, points: ArrayLike, *args: Any) -> None:
        """Append a glyph operation and its points to the shared projection batch."""
        points = np.atleast_2d(points)
        end = self.count + len(points)
        self.operations.append((kind, self.count, end, args))
        self.points.append(points)
        self.count = end

    def lines(
        self, points: ArrayLike, colors: np.ndarray, thickness: float, *, halo: bool = True, arrow: bool = False
    ) -> None:
        """Queue equally sized polylines in one operation."""
        if halo is True:
            halo = np.broadcast_to((0.025, 0.035, 0.05, 0.60), colors.shape).copy()
            halo[:, 3] *= colors[:, 3]
        elif halo is False:
            halo = None
        self.add("lines", points.reshape(-1, 3), points.shape[1], colors, thickness, halo, arrow)

    def markers(self, points: ArrayLike, colors: np.ndarray, backgrounds: np.ndarray, *, square: bool = False) -> None:
        """Queue colored circular or square joint markers with their background colors."""
        self.add("markers", points, colors, backgrounds, square)

    def flush(self) -> None:
        """Project queued glyphs and replace the renderer opaque and transparent triangle batches.

        Expand strokes and markers in screen space while retaining clip-space
        depth, so glyph size is readable without bypassing scene occlusion.
        """
        if not self.points:
            return
        screen, valid = self.project(np.concatenate(self.points))
        x, y, width, height = self.rect
        batches = ([], [])
        strokes = []
        polygons = []

        def polygon(points: ArrayLike, indices: Sequence[int] | np.ndarray, color: int) -> None:
            points = np.asarray(points)
            if not len(points):
                return
            if points.ndim == 2:
                points = points[None, ...]
            count, size = points.shape[:2]
            polygons.append(
                (
                    points.reshape(-1, 2),
                    np.asarray(indices).ravel(),
                    np.repeat(np.broadcast_to(color, (count, 4)), size, axis=0),
                    np.full(count, size),
                )
            )

        def stroke(paths: ArrayLike, color: ArrayLike, thickness: float, halo: ArrayLike | None = None) -> None:
            paths = np.atleast_2d(paths)
            if paths.shape[1] < 2:
                return
            points = screen[paths]
            delta = np.diff(points, axis=1)
            length = np.linalg.norm(delta, axis=2, keepdims=True)
            normal = delta[..., ::-1] * (-1, 1) / np.maximum(length, 1e-9)
            incoming = np.concatenate((normal[:, :1], normal), axis=1)
            outgoing = np.concatenate((normal, normal[:, -1:]), axis=1)
            closed = np.linalg.norm(points[:, 0] - points[:, -1], axis=1) < 1e-6
            incoming[closed, 0] = normal[closed, -1]
            outgoing[closed, -1] = normal[closed, 0]
            # Shared miter joins keep neighboring segments and the outline from
            # crossing into the colored stroke, even for small projected arcs.
            miter = incoming + outgoing
            miter /= np.maximum(np.linalg.norm(miter, axis=2, keepdims=True), 1e-9)
            miter /= np.maximum(np.sum(miter * outgoing, axis=2, keepdims=True), 0.25)
            indices = np.stack((paths[:, :-1], paths[:, 1:]), axis=2).reshape(-1, 2)
            a, b = miter[:, :-1].reshape(-1, 2), miter[:, 1:].reshape(-1, 2)
            count, size = paths.shape
            rgba = np.repeat(np.broadcast_to(color, (count, 4)), size - 1, axis=0)
            half_width = thickness * self.scale * 0.5
            offsets = np.stack((a, -a, -b, b), axis=1) * half_width
            strokes.append((indices, rgba, offsets))
            if halo is not None:
                # Draw only the border, not a wider coplanar copy of the entire
                # stroke. The transparent pass must never darken its own core.
                background = np.repeat(np.broadcast_to(halo, (count, 4)), size - 1, axis=0)
                outer = half_width + self.scale
                for sign in (-1, 1):
                    offsets = np.stack((a * outer, a * half_width, b * half_width, b * outer), axis=1)
                    strokes.append((indices, background, offsets * sign))

        angles = np.linspace(0, 2 * np.pi, 17)[:-1]
        circle = np.column_stack((np.cos(angles), np.sin(angles)))
        for kind, start, end, args in self.operations:
            if kind == "lines":
                size, colors, thickness, halo, arrow = args
                paths = np.arange(start, end).reshape(-1, size)
                stroke(paths, colors, thickness, halo)
                if arrow:
                    tips, tails = paths[:, -1], paths[:, -2]
                    delta = screen[tips] - screen[tails]
                    length = np.linalg.norm(delta, axis=1)
                    keep = valid[tips] & valid[tails] & (length > 8 * self.scale)
                    tips, delta, length = tips[keep], delta[keep], length[keep]
                    direction = delta / length[:, None]
                    side = direction[:, ::-1] * (-1, 1) * 3.5 * self.scale
                    base = screen[tips] - direction * 7 * self.scale
                    polygon(
                        np.stack((screen[tips], base + side, base - side), axis=1),
                        np.repeat(tips[:, None], 3, axis=1),
                        colors[keep],
                    )
            elif kind == "markers":
                colors, backgrounds, square = args
                indices = np.arange(start, end)[valid[start:end]]
                colors, backgrounds = colors[valid[start:end]], backgrounds[valid[start:end]]
                centers = screen[indices]
                size = self.scale * 0.5
                refs = np.repeat(indices[:, None], len(circle), axis=1)
                polygon(centers[:, None, :] + circle * 6 * size, refs, backgrounds)
                if square:
                    corners = centers[:, None, :] + np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * 6 * size
                    for i in range(4):
                        a, b = corners[:, i], corners[:, (i + 1) % 4]
                        delta = b - a
                        side = delta[:, ::-1] * (-1, 1) / np.linalg.norm(delta, axis=1)[:, None] * 0.75 * self.scale
                        polygon(
                            np.stack((a + side, a - side, b - side, b + side), axis=1),
                            np.repeat(indices[:, None], 4, axis=1),
                            colors,
                        )
                else:
                    polygon(centers[:, None, :] + circle * 3.5 * size, refs, colors)
            elif kind in ("stroke", "arrow"):
                color, thickness, halo = args
                indices = list(range(start, end))
                stroke(indices, color, thickness, halo if halo else None)
                if kind == "arrow" and valid[end - 2 : end].all():
                    tip, tail = screen[end - 1], screen[end - 2]
                    delta = tip - tail
                    length = np.linalg.norm(delta)
                    if length > 8 * self.scale:
                        direction = delta / length
                        side = direction[::-1] * (-1, 1) * 3.5 * self.scale
                        base = tip - direction * 7 * self.scale
                        polygon([tip, base + side, base - side], [end - 1] * 3, color)
            elif kind == "fan" and valid[start:end].all() and end - start >= 3:
                polygon(screen[start:end], list(range(start, end)), args[0])
            elif kind == "marker" and valid[start]:
                color, background, size, square = args
                p = screen[start]
                polygon(p + circle * 6 * size, [start] * len(circle), background)
                if square:
                    corners = p + np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * 6 * size
                    for i in range(4):
                        a, b = corners[i], corners[(i + 1) % 4]
                        delta = b - a
                        side = delta[::-1] * (-1, 1) / np.linalg.norm(delta) * 0.75 * self.scale
                        polygon([a + side, a - side, b - side, b + side], [start] * 4, color)
                else:
                    polygon(p + circle * 3.5 * size, [start] * len(circle), color)
        if polygons:
            # Tessellate the small fans and markers in one contiguous upload.
            points = np.concatenate([p for p, _, _, _ in polygons])
            indices = np.concatenate([indices for _, indices, _, _ in polygons])
            colors = np.concatenate([color for _, _, color, _ in polygons])
            counts = np.concatenate([counts for _, _, _, counts in polygons])
            clip = self.clip[indices].copy()
            clip[:, :2] = ((points - (x, y)) / (width, height) * 2 - 1) * (1, -1) * clip[:, 3:4]
            vertices = np.column_stack((clip, colors))
            offsets = np.cumsum(counts) - counts
            triangle_counts = (counts - 2) * 3
            starts = np.cumsum(triangle_counts) - triangle_counts
            local = np.arange(triangle_counts.sum()) - np.repeat(starts, triangle_counts)
            triangles = np.repeat(offsets, triangle_counts) + np.where(local % 3 == 0, 0, local // 3 + local % 3)
            vertices = vertices[triangles]
            for transparent, batch in enumerate(batches):
                batch.append(vertices[(vertices[:, 7] < 0.999) == bool(transparent)])
        if strokes:
            # Expand all line segments together; the work remains bounded by the
            # selected joints rather than Python calls per tessellated segment.
            indices = np.concatenate([indices for indices, _, _ in strokes])
            a, b = indices.T
            colors = np.concatenate([color for _, color, _ in strokes])
            offsets = np.concatenate([offsets for _, _, offsets in strokes])
            delta = screen[b] - screen[a]
            lengths = np.linalg.norm(delta, axis=1)
            keep = valid[a] & valid[b] & (lengths > 0.01)
            a, b = a[keep], b[keep]
            colors, offsets = colors[keep], offsets[keep]
            points = screen[np.column_stack((a, a, b, b))] + offsets
            clip = self.clip[np.column_stack((a, a, b, b))].copy()
            clip[..., :2] = ((points - (x, y)) / (width, height) * 2 - 1) * (1, -1) * clip[..., 3:4]
            vertices = np.concatenate((clip, np.broadcast_to(colors[:, None, :], (*clip.shape[:2], 4))), axis=2)
            for transparent, batch in enumerate(batches):
                batch.insert(
                    0, vertices[(colors[:, 3] < 0.999) == bool(transparent)][:, (0, 1, 2, 0, 2, 3)].reshape(-1, 8)
                )
        self.renderer._joint_triangles = tuple(
            np.concatenate(batch).astype(np.float32) if batch else np.empty((0, 8), dtype=np.float32)
            for batch in batches
        )


class JointSelection:
    """Inspect one joint's displayed anchors without running IK or copying a scene."""

    reference = (0.40, 0.84, 0.78)
    current = (1.0, 0.60, 0.20)
    neutral = (0.76, 0.82, 0.89)
    axis_colors = ((0.94, 0.38, 0.38), (0.43, 0.85, 0.53), (0.39, 0.64, 1.0))

    def __init__(self, workspace: ViewerWorkspace, index: int) -> None:
        """Read fixed joint metadata through the workspace cache and precompute its glyph basis."""
        model = workspace.model
        self.index = index
        self.kind = JointType(int(workspace.read(model, "joint_type", index)[0]))
        self.child = int(workspace.read(model, "joint_child", index)[0])
        self.world = workspace.tree.world_for(("joint", index))
        start, end = map(int, workspace.read(model, "joint_qd_start", index, 2))
        self.axes = workspace.read(model, "joint_axis", start, end - start)
        self.lower = workspace.read(model, "joint_limit_lower", start, end - start)
        self.upper = workspace.read(model, "joint_limit_upper", start, end - start)
        self.enabled = bool(workspace.read(model, "joint_enabled", index)[0])
        self.value = 0.0
        if len(self.axes):
            axis = self.axes[0] / max(1e-9, np.linalg.norm(self.axes[0]))
            tangent = np.eye(3)[np.argmin(np.abs(axis))]
            tangent -= axis * np.dot(axis, tangent)
            tangent /= max(1e-9, np.linalg.norm(tangent))
            self.axis_basis = np.column_stack((tangent, np.cross(axis, tangent), axis))
        angles = np.linspace(0, 2 * np.pi, 41)
        self.circle = np.column_stack((np.cos(angles), np.sin(angles)))

    @property
    def limited(self) -> bool:
        """Report whether the first DOF has ordered finite limits below the unbounded sentinel magnitude."""
        return bool(
            len(self.lower)
            and np.isfinite(self.lower[0])
            and np.isfinite(self.upper[0])
            and self.lower[0] <= self.upper[0]
            and abs(self.lower[0]) < 1e6
            and abs(self.upper[0]) < 1e6
        )

    def draw(self, workspace: ViewerWorkspace, *, draw_list: _JointDrawList, compact: bool = False) -> None:
        """Queue type-specific axes, limits, and pose markers using frames supplied by JointSelectionGroup."""
        viewer = workspace.viewer
        if viewer.is_rendering_paused() or viewer._layer_force_hidden():
            return
        (origin, rotation), (child, child_rotation) = self.parent_pose, self.child_pose
        scale = workspace.widgets.scale
        x, y, width, height = workspace.rect
        center, valid = draw_list.project(origin)
        center = center[0]
        if not valid[0] or not (x <= center[0] <= x + width and y <= center[1] <= y + height):
            return
        depth = -float(draw_list.view[:3, :3][2] @ (origin - draw_list.camera))
        radius = depth * draw_list.radius_per_depth * draw_list.radius_pixels

        def color(rgb: ArrayLike, alpha: float = 1) -> tuple[float, float, float, float]:
            return draw_list.color(rgb, alpha if self.enabled else alpha * 0.45)

        def stroke(
            points: ArrayLike, rgb: ArrayLike, thickness: float = 1.5, alpha: float = 1, halo: bool = True
        ) -> None:
            draw_list.add(
                "stroke", points, color(rgb, alpha), thickness, color((0.025, 0.035, 0.05), 0.60) if halo else 0
            )

        def arrow(a: float, b: float, rgb: ArrayLike, thickness: float = 1.7) -> None:
            draw_list.add("arrow", [a, b], color(rgb), thickness, color((0.025, 0.035, 0.05), 0.60))

        def marker(position: ArrayLike, rgb: ArrayLike, square: bool = False) -> None:
            draw_list.add(
                "marker", position, color(rgb), color((0.025, 0.035, 0.05)), scale * (0.5 if compact else 1), square
            )

        def arc(u: np.ndarray, v: np.ndarray, start: float, end: float, size: float = 1) -> np.ndarray:
            if compact and start == 0 and end == 2 * np.pi:
                return origin + radius * size * (self.circle[:, :1] * u + self.circle[:, 1:] * v)
            segments = 40 if compact else 72
            angles = np.linspace(start, end, max(2, int(abs(end - start) / (2 * np.pi) * segments) + 1))
            return origin + radius * size * (np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v)

        # Parent reference is teal, current child is orange throughout.
        if np.linalg.norm(child - origin) > radius * 0.02:
            for a in np.linspace(0, 1, 13)[:-1:2]:
                stroke([origin + a * (child - origin), origin + (a + 1 / 12) * (child - origin)], self.neutral, 1, 0.6)
        if self.kind == JointType.REVOLUTE:
            u, v, axis = (rotation @ self.axis_basis).T
            stroke(arc(u, v, 0, 2 * np.pi), self.reference if compact else self.neutral, 1, 0.55, halo=False)
            if self.limited and not compact:
                lo, hi = float(self.lower[0]), float(self.upper[0])
                stroke(arc(u, v, lo, lo + min(hi - lo, 2 * np.pi), 1.1), self.reference, 1.5 if compact else 4, 0.35)
                for angle in (lo, hi):
                    direction = np.cos(angle) * u + np.sin(angle) * v
                    stroke([origin + radius * direction, origin + radius * 1.2 * direction], self.reference)
            angle = np.clip(self.value, -2 * np.pi, 2 * np.pi)
            active = arc(u, v, 0, angle)
            if not compact:
                draw_list.add("fan", np.vstack((origin, active)), color(self.current, 0.10))
            for tick in range(0 if compact else 24):
                direction = np.cos(tick * np.pi / 12) * u + np.sin(tick * np.pi / 12) * v
                stroke(
                    [origin + radius * (0.9 if tick % 6 == 0 else 0.96) * direction, origin + radius * direction],
                    self.neutral,
                    1,
                    0.45,
                    halo=False,
                )
            stroke([origin, origin + radius * u], self.reference, 1.2, 0.85)
            stroke(active, self.current, 1.5 if compact else 3)
            if len(active) > 3 and not compact:
                arrow(active[-3], active[-1], self.current, 2)
            pointer = origin + radius * (np.cos(self.value) * u + np.sin(self.value) * v)
            stroke([origin, pointer], self.current, 2)
            if not compact:
                marker(pointer, self.current)
            arrow(origin - axis * radius * 0.55, origin + axis * radius * 0.75, self.reference)
        elif self.kind == JointType.PRISMATIC:
            u, _, axis = (rotation @ self.axis_basis).T
            lo, hi = (-radius, radius)
            if self.limited:
                lo, hi = float(self.lower[0]), float(self.upper[0])
            # Keep an unbounded slider's current carriage in view.
            lo, hi = min(lo, self.value, 0), max(hi, self.value, 0)
            stroke([origin + axis * lo, origin + axis * hi], self.reference, 4, 0.35)
            for t in np.linspace(lo, hi, 0 if compact else 11):
                stroke(
                    [origin + axis * t - u * radius * 0.045, origin + axis * t + u * radius * 0.045],
                    self.neutral,
                    1,
                    0.65,
                )
            arrow(origin, origin + axis * self.value, self.current, 3)
            carriage = origin + axis * self.value
            stroke([carriage - u * radius * 0.2, carriage + u * radius * 0.2], self.current, 4)
            if self.limited:
                for t in (self.lower[0], self.upper[0]):
                    stroke(
                        [origin + axis * t - u * radius * 0.12, origin + axis * t + u * radius * 0.12],
                        self.reference,
                        2,
                    )
        else:
            for i, rgb in enumerate(self.axis_colors):
                arrow(origin, origin + rotation[:, i] * radius * 0.85, rgb)
                stroke([child, child + child_rotation[:, i] * radius * 0.65], self.current, 1.5, 0.8)
            if self.kind == JointType.BALL:
                for i in range(3):
                    stroke(
                        arc(rotation[:, i], rotation[:, (i + 1) % 3], 0, 2 * np.pi),
                        self.neutral,
                        1,
                        0.3,
                        halo=False,
                    )
        if not compact or np.linalg.norm(child - origin) > radius * 0.02:
            marker(origin, self.reference, square=True)
        marker(child, self.current)
