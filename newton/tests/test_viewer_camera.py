# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

import math
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton._src.viewer.camera import Camera
from newton._src.viewer.gui.viewer_gui import ViewerGui
from newton._src.viewer.gui.workspace import ViewerWorkspace
from newton._src.viewer.viewer_model_tree import ModelTree
from newton._src.viewer.viewer_navigation import SelectionNavigation
from newton.viewer import ViewerGL


def _as_np(value):
    return np.array((value[0], value[1], value[2]), dtype=float)


def _assert_vec_close(test, actual, expected, tol=1.0e-6):
    np.testing.assert_allclose(_as_np(actual), np.array(expected, dtype=float), atol=tol, rtol=0.0)


class TestViewerCameraOrbit(unittest.TestCase):
    def test_follow_selection_hold_toggle_and_orbit(self) -> None:
        from pyglet.window import key, mouse

        camera = Camera(pos=(0, -5, 3))
        camera.look_at((0, 0, 0))
        viewer = SimpleNamespace(camera=camera, model=True, is_rendering_paused=lambda: False)
        gui = ViewerGui.__new__(ViewerGui)
        gui._viewer, gui.ui = viewer, None
        gui._loading_splash_active = False
        gui._cam_vel = np.zeros(3)
        gui._camera_orbit_sensitivity = 0.1
        gui._camera_drag_buttons = 0
        center = np.zeros(3)
        bounds_calls = []

        def bounds(**_kwargs: Any) -> np.ndarray:
            bounds_calls.append(True)
            return center + np.array(((-1, -1, -1), (1, 1, 1)))

        gui.workspace = SimpleNamespace(
            can_follow_selection=True,
            selection_bounds=bounds,
            collision=SimpleNamespace(focus_points=lambda: []),
        )
        gui.update_camera_follow(lambda _: False)
        self.assertFalse(bounds_calls)  # No geometry work while follow is disabled.
        gui.handle_key_press(key.F)
        self.assertTrue(gui.following_selection)
        distance = camera.pivot_distance
        gui.handle_mouse_press(100, 100, mouse.LEFT, lambda x, y: (x, y))
        for step in range(4):
            center[:] = (step, -step * 0.3, step * 0.5)
            gui.handle_mouse_drag(100, 100, 27, 13, mouse.LEFT, lambda x, y: (x, y))
            angle = camera.yaw, camera.pitch
            gui.update_camera_follow(lambda code: code == key.F)
            _assert_vec_close(self, camera.pivot, center)
            self.assertAlmostEqual(camera.pivot_distance, distance)
            self.assertEqual((camera.yaw, camera.pitch), angle)
            np.testing.assert_allclose(camera.get_front(), (camera.pivot - camera.pos) / distance, atol=1e-6)
            # OS key repeat must not refit the selection or reset mouse zoom.
            camera.dolly(0.2)
            distance = camera.pivot_distance
            gui.handle_key_press(key.F)
            self.assertAlmostEqual(camera.pivot_distance, distance)

        gui.handle_mouse_release(127, 113, mouse.LEFT)
        gui.handle_key_release(key.F)
        previous = np.asarray(camera.pos)
        center += 10
        gui.update_camera_follow(lambda _: False)
        _assert_vec_close(self, camera.pos, previous)
        self.assertFalse(gui.following_selection)

        gui.toggle_follow_selection()
        self.assertTrue(gui.following_selection)
        # Panning or framing a contact within the body keeps that view offset.
        camera.pan(0.2, -0.1)
        offset = np.asarray(camera.pivot) - center
        gui._release_camera_keys()  # Focus loss releases F, but keeps a pinned follow.
        center -= 2
        gui.update_camera_follow(lambda _: False)
        _assert_vec_close(self, camera.pivot, center + offset)
        gui.toggle_follow_selection()
        self.assertFalse(gui.following_selection)
        gui.handle_key_press(key.F)
        gui.update_camera_follow(lambda _: False)  # Recover a missed release event.
        self.assertFalse(gui.following_selection)
        gui.handle_key_press(key.F)
        gui._release_camera_keys()
        self.assertFalse(gui.following_selection)
        gui.toggle_follow_selection()
        gui.workspace.can_follow_selection = False
        gui.update_camera_follow(lambda _: False)
        self.assertFalse(gui.following_selection)

    def test_follow_bounds_reuse_and_displayed_state(self) -> None:
        """Track only the selected world, with persistent reduction buffers."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_box(body, hx=1, hy=0.5, hz=0.2)
        model = builder.finalize(device="cpu")
        model.shape_world.assign([0])
        viewer = SimpleNamespace(
            _last_state=model.state(),
            world_offsets=wp.array([(7, 0, 0)], dtype=wp.vec3, device="cpu"),
            layer=SimpleNamespace(xform=wp.transform((0, 3, 0), wp.quat_identity())),
        )
        ws = ViewerWorkspace.__new__(ViewerWorkspace)
        ws.viewer, ws.model, ws.selected = viewer, model, ("body", 0)
        ws.tree = ws._deformable_selection = ws._frame_bounds_buffers = None
        ws._selection_shapes = (0,)
        buffers = None
        for position in ((0, 0, 0), (2, 3, 4), (-2, -1, 0)):
            # History uses a different state; tracking must read what is displayed.
            viewer._last_state = model.state()
            viewer._last_state.body_q.assign([(*position, 0, 0, 0, 1)])
            bounds = ws.selection_bounds(include_contacts=False)
            np.testing.assert_allclose(bounds.mean(axis=0), np.array(position) + np.array([7, 3, 0]))
            if buffers is not None:
                self.assertIs(ws._frame_bounds_buffers, buffers)
            buffers = ws._frame_bounds_buffers

        # BVH selections share the focus path, with their own world filter and
        # full bounds even when the overlay is hidden or its budget is small.
        model.bvh_build_shapes(viewer._last_state)
        ws.tree, ws.example, ws.history = ModelTree(model), None, None
        ws.selected = ("bvh", "shapes")
        bvhs = ws.tree.resources.bvhs
        bvhs.scope, bvhs.world, bvhs.limit, bvhs.show_bounds = 0, 0, 16, False
        np.testing.assert_allclose(ws.selection_bounds(include_contacts=False), bounds)
        buffers = bvhs._focus_buffers
        np.testing.assert_allclose(ws.selection_bounds(include_contacts=False), bounds)
        self.assertIs(bvhs._focus_buffers, buffers)
        # Examples need not refit the query BVH themselves. Focusing it must
        # use the displayed pose even after in-place edits of a paused state.
        viewer._last_state.body_q.assign([(4, 2, 1, 0, 0, 0, 1)])
        moved = ws.selection_bounds(include_contacts=False)
        np.testing.assert_allclose(moved.mean(axis=0), [11, 5, 1])
        self.assertIs(bvhs._focus_buffers, buffers)
        bvhs.scope = 2  # The selected Shared group is empty.
        self.assertIsNone(ws.selection_bounds(include_contacts=False))
        bvhs.scope = 0
        ws.history = SimpleNamespace(replaying=True)
        self.assertIsNone(ws.selection_bounds(include_contacts=False))

        # A small selected shape must not inherit a fixed 5 cm orbit floor.
        camera = Camera(pos=(0, -2, 0), near=0.0001, width=800, height=800)
        camera.look_at((0, 0, 0))
        gui = ViewerGui.__new__(ViewerGui)
        gui._viewer = SimpleNamespace(camera=camera, model=True, is_rendering_paused=lambda: False)
        gui._cam_vel = np.zeros(3)
        gui.workspace = SimpleNamespace(
            collision=SimpleNamespace(focus_points=lambda: []),
            selection_bounds=lambda **_: np.array(((-0.002, -0.002, -0.002), (0.002, 0.002, 0.002))),
        )
        gui.frame_camera_on_model()
        self.assertLess(camera.pivot_distance, 0.01)
        self.assertGreater(camera.pivot_distance - 0.002, camera.near)

    def test_frame_contact_separation_and_near_plane(self) -> None:
        """Expose contact gaps without clipping witnesses or displacing the orbit pivot."""
        for near in (0.0001, 0.01):
            for gap in (0.0, 0.001, 0.08):
                for width in (400, 1600):
                    with self.subTest(near=near, gap=gap, width=width):
                        camera = Camera(near=near, width=width, height=800, pos=(0, -3, 0))
                        camera.look_at((0, 0, 0))
                        points = [np.array((0, -gap / 2, 0)), np.array((0, gap / 2, 0))]
                        gui = ViewerGui.__new__(ViewerGui)
                        gui._viewer = SimpleNamespace(camera=camera, model=True, is_rendering_paused=lambda: False)
                        gui._cam_vel = np.ones(3)
                        gui.workspace = SimpleNamespace(
                            collision=SimpleNamespace(focus_points=lambda points=points: points)
                        )
                        gui.frame_camera_on_model()
                        _assert_vec_close(self, camera.pivot, (0, 0, 0))
                        offsets = np.asarray(points) - np.asarray(camera.pos)
                        depth = offsets @ camera.get_front()
                        self.assertTrue(np.all(depth >= near * 1.999))
                        tan_fov = np.tan(np.radians(camera.fov) / 2)
                        screen = np.column_stack(
                            (
                                offsets @ camera.get_right() / (depth * tan_fov * width / 800),
                                offsets @ camera.get_up() / (depth * tan_fov),
                            )
                        )
                        self.assertLessEqual(np.max(np.abs(screen)), 0.351)
                        if gap >= near * 2:
                            self.assertGreater(np.linalg.norm(screen[1] - screen[0]), 0.2)
                        if gap == 0:
                            self.assertAlmostEqual(camera.pivot_distance, near * 2)
                        distance = camera.pivot_distance
                        camera.orbit(10, 5)
                        self.assertAlmostEqual(camera.pivot_distance, distance)
                        _assert_vec_close(self, camera.pivot, (0, 0, 0))
                        np.testing.assert_array_equal(gui._cam_vel, np.zeros(3))

    def test_selection_navigation_clearance(self) -> None:
        """Slow near selected surfaces while allowing crossing, escape and orbit."""
        builder = newton.ModelBuilder()
        builder.add_shape_box(-1, hx=1.0, hy=1.0, hz=1.0)
        builder.add_shape_sphere(-1, radius=1.0)
        builder.add_shape_capsule(-1, radius=1.0, half_height=1.0)
        builder.add_shape_cylinder(-1, radius=1.0, half_height=1.0)
        builder.add_shape_mesh(-1, mesh=newton.Mesh.create_box(1.0, 1.0, 1.0))
        model = builder.finalize(device="cpu")
        camera = Camera(pos=(3, 0, 0))
        viewer = SimpleNamespace(
            camera=camera,
            camera_slowdown=2.0,
            _last_state=model.state(),
            world_offsets=None,
            layer=SimpleNamespace(xform=wp.transform_identity()),
        )
        workspace = SimpleNamespace(model=model, viewer=viewer, _selection_shapes=(0,))
        navigation = SelectionNavigation(workspace)
        camera._navigation_filter = navigation
        for shape in range(model.shape_count):
            with self.subTest(shape=shape):
                workspace._selection_shapes = (shape,)
                camera.pos = camera._as_vec3((3, 0, 0))
                camera.look_at((0, 0, 0))
                camera.translate((-100, 0, 0))
                self.assertLess(camera.pos.x, -1.0)
                # Slow motion can cross the surface in either direction,
                # including when the camera starts exactly on it.
                for direction in (-1, 1):
                    camera.pos = camera._as_vec3((1, 0, 0))
                    camera.translate((0.1 * direction, 0, 0))
                    self.assertGreater((camera.pos.x - 1) * direction, 0)
                    self.assertLess(abs(camera.pos.x - 1), 0.1)
                camera.pos = camera._as_vec3((1.005, 0, 0))
                camera.look_at((0, 0, 0))
                camera.dolly(5)
                self.assertLess(camera.pos.x, 1.0)
                # Speed recovers away from the boundary on the inside too.
                close = navigation((1.01, 0, 0), (1, 0, 0), reference=2.4)
                inside = navigation((0, 0, 0), (0.1, 0, 0), reference=2.4)
                self.assertGreater(inside, close * 10)
        workspace._selection_shapes = (0,)
        far = navigation((4, 0, 0), (3.9, 0, 0), reference=2.4)
        close = navigation((1.1, 0, 0), (1.0, 0, 0), reference=2.4)
        self.assertGreater(far, close * 10)
        # The stronger default falls off smoothly; the legacy linear curve and
        # disabling slowdown remain available without preventing surface crossing.
        for distance in (0.0, 0.05, 0.25, 0.5, 1.0, 2.0):
            start = (1.0 + distance, 0, 0)
            viewer.camera_slowdown = 1.0
            linear = navigation(start, (1, 0, 0), reference=2.4)
            viewer.camera_slowdown = 2.0
            stronger = navigation(start, (1, 0, 0), reference=2.4)
            self.assertGreater(stronger, 0)
            self.assertLess(stronger, linear)
        viewer.camera_slowdown = 0.0
        disabled = SelectionNavigation(workspace)
        self.assertEqual(disabled((1, 0, 0), (0, 0, 0), reference=2.4), 1.0)
        self.assertIsNone(disabled.key)
        viewer.camera_slowdown = 2.0
        # A selection must never stop a gizmo orbit or change its pivot.
        camera.pos = camera._as_vec3((3, 0, 0))
        camera.look_at((1.5, 0, 0))
        camera.orbit(180, 0)
        _assert_vec_close(self, camera.pos, (0, 0, 0))
        _assert_vec_close(self, camera.pivot, (1.5, 0, 0))
        workspace._selection_shapes = ()
        self.assertEqual(navigation((3, 0, 0), (-3, 0, 0), reference=2.4), 1.0)
        # World offsets and a rotated display layer use displayed space.
        workspace._selection_shapes = (0,)
        model.shape_world.assign(np.zeros(model.shape_count, dtype=np.int32))
        viewer.world_offsets = wp.array([(5, 0, 0)], dtype=wp.vec3, device="cpu")
        viewer.layer.xform = wp.transform((0, 3, 0), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), math.pi / 2))
        self.assertLess(navigation((0, 9.1, 0), (0, 9, 0), reference=2.4), 0.1)
        self.assertEqual(navigation((0, 0, 0), (0, 1, 0), reference=2.4), 1.0)
        # Open sheets slow the camera equally on both sides, without blocking.
        workspace._selection_shapes = ()
        points = wp.array([(-2, -2, 0), (2, -2, 0), (0, 2, 0)], dtype=wp.vec3, device="cpu")
        surface = SimpleNamespace(
            points=points, faces=wp.array([0, 1, 2], dtype=int, device="cpu"), mesh=True, revision=0, batches=[]
        )
        workspace._deformable_selection = surface
        above = navigation((0, 0, 0.1), (0, 0, -0.1), reference=2.4)
        below = navigation((0, 0, -0.1), (0, 0, 0.1), reference=2.4)
        self.assertAlmostEqual(above, below)
        self.assertGreater(above, 0)
        self.assertLess(above, 0.1)
        points.assign([(-2, -2, -5), (2, -2, -5), (0, 2, -5)])
        surface.revision += 1
        self.assertEqual(navigation((0, 0, 0.1), (0, 0, -0.1), reference=2.4), 1.0)
        radii = wp.array([0.1], dtype=float, device="cpu")
        source = SimpleNamespace(model=SimpleNamespace(particle_radius=radii))
        workspace._deformable_selection = SimpleNamespace(
            mesh=None,
            batches=[(source, wp.array([0], dtype=int, device="cpu"), wp.zeros(1, dtype=wp.vec3, device="cpu"))],
        )
        self.assertGreater(navigation((0, 0, 0.1), (0, 0, -1), reference=1), 0)
        self.assertLess(navigation((0, 0, 0.1), (0, 0, -1), reference=1), 0.1)

    def test_sync_pivot_to_view_tracks_fps_look(self):
        camera = Camera(pos=(0.0, 0.0, 0.0), up_axis="Z")
        camera.sync_pivot_to_view(distance=10.0)

        camera.yaw = 90.0
        camera.pitch = 30.0
        camera.sync_pivot_to_view()

        expected_pivot = _as_np(camera.pos) + _as_np(camera.get_front()) * 10.0
        _assert_vec_close(self, camera.pivot, expected_pivot)
        self.assertAlmostEqual(camera.pivot_distance, 10.0)

    def test_translate_moves_camera_and_pivot_together(self):
        camera = Camera(pos=(0.0, 0.0, 0.0), up_axis="Z")
        camera.sync_pivot_to_view(distance=7.0)
        start_offset = _as_np(camera.pivot) - _as_np(camera.pos)

        camera.translate((1.0, -2.0, 3.0))

        _assert_vec_close(self, camera.pos, (1.0, -2.0, 3.0))
        _assert_vec_close(self, _as_np(camera.pivot) - _as_np(camera.pos), start_offset)
        self.assertAlmostEqual(camera.pivot_distance, 7.0)

    def test_pivot_distance_is_derived_from_pivot_and_position(self):
        camera = Camera(pos=(0.0, 0.0, 0.0), up_axis="Z")
        self.assertFalse(hasattr(camera, "_pivot_distance"))

        camera.pivot = camera._as_vec3((0.0, 0.0, 0.0))
        self.assertAlmostEqual(camera.pivot_distance, camera._min_pivot_distance)

        camera.pos = camera._as_vec3((0.0, 0.0, 2.0))
        self.assertAlmostEqual(camera.pivot_distance, 2.0)

    def test_orbit_keeps_pivot_fixed_and_points_at_pivot(self):
        camera = Camera(pos=(10.0, 0.0, 0.0), up_axis="Z")
        camera.look_at((0.0, 0.0, 0.0))
        camera.set_pivot((0.0, 0.0, 0.0))

        camera.orbit(delta_yaw=45.0, delta_pitch=30.0)

        _assert_vec_close(self, camera.pivot, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(camera.pivot_distance, 10.0)
        direction_to_pivot = (_as_np(camera.pivot) - _as_np(camera.pos)) / camera.pivot_distance
        np.testing.assert_allclose(_as_np(camera.get_front()), direction_to_pivot, atol=1.0e-6, rtol=0.0)

    def test_look_at_points_front_at_pivot_for_all_up_axes(self):
        for up_axis in ("X", "Y", "Z"):
            with self.subTest(up_axis=up_axis):
                camera = Camera(pos=(1.0, 2.0, 3.0), up_axis=up_axis)
                camera.look_at((-4.0, 6.0, 2.0))

                direction_to_pivot = (_as_np(camera.pivot) - _as_np(camera.pos)) / camera.pivot_distance
                np.testing.assert_allclose(_as_np(camera.get_front()), direction_to_pivot, atol=1.0e-6, rtol=0.0)

    def test_orbit_clamps_pitch_to_89_degrees(self):
        camera = Camera(pos=(10.0, 0.0, 0.0), up_axis="Z")
        camera.look_at((0.0, 0.0, 0.0))
        camera.set_pivot((0.0, 0.0, 0.0))

        camera.orbit(delta_yaw=0.0, delta_pitch=200.0)

        self.assertEqual(camera.pitch, 89.0)
        self.assertTrue(math.isfinite(camera.pos[0]))
        self.assertAlmostEqual(camera.pivot_distance, 10.0)

    def test_pan_moves_camera_and_pivot_in_camera_plane(self):
        camera = Camera(pos=(10.0, 0.0, 0.0), up_axis="Z")
        camera.look_at((0.0, 0.0, 0.0))
        camera.set_pivot((0.0, 0.0, 0.0))

        start_pos = _as_np(camera.pos)
        start_pivot = _as_np(camera.pivot)
        right = _as_np(camera.get_right())
        up = _as_np(camera.get_up())

        camera.pan(delta_right=2.0, delta_up=-3.0)

        expected_delta = right * 2.0 + up * -3.0
        _assert_vec_close(self, camera.pos, start_pos + expected_delta)
        _assert_vec_close(self, camera.pivot, start_pivot + expected_delta)
        self.assertAlmostEqual(camera.pivot_distance, 10.0)

    def test_dolly_moves_camera_toward_pivot_without_moving_pivot(self):
        camera = Camera(pos=(10.0, 0.0, 0.0), up_axis="Z")
        camera.look_at((0.0, 0.0, 0.0))
        camera.set_pivot((0.0, 0.0, 0.0))

        camera.dolly(0.5)
        distance_after_dolly_in = camera.pivot_distance

        self.assertLess(distance_after_dolly_in, 10.0)
        _assert_vec_close(self, camera.pivot, (0.0, 0.0, 0.0))
        direction_to_pivot = (_as_np(camera.pivot) - _as_np(camera.pos)) / camera.pivot_distance
        np.testing.assert_allclose(_as_np(camera.get_front()), direction_to_pivot, atol=1.0e-6, rtol=0.0)

        camera.dolly(-0.5)

        self.assertGreater(camera.pivot_distance, distance_after_dolly_in)


def _build_viewer_model(z: float, up_axis: newton.Axis = newton.Axis.Z) -> newton.Model:
    builder = newton.ModelBuilder(up_axis=up_axis)
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, z), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )
    builder.add_shape_box(body=body, hx=0.5, hy=0.5, hz=0.5)
    return builder.finalize()


@unittest.skipUnless(wp.is_cuda_available(), "ViewerGL model-switch test requires CUDA")
class TestViewerGLSetModelState(unittest.TestCase):
    def test_set_model_preserves_camera_and_wind_for_same_up_axis(self):
        model_a = _build_viewer_model(1.0)
        model_b = _build_viewer_model(2.0)

        try:
            viewer = ViewerGL(headless=True)
        except Exception as exc:
            self.skipTest(f"ViewerGL not available: {exc}")
            return

        try:
            viewer.set_model(model_a)
            viewer.camera.near = 0.123
            viewer.camera.far = 456.0
            viewer.camera.fov = 33.0
            viewer.camera.pos = viewer.camera._as_vec3((3.0, 4.0, 5.0))
            viewer.camera.yaw = 37.0
            viewer.camera.pitch = -12.0
            viewer.camera.set_pivot((1.0, 2.0, 3.0))
            viewer.wind.time = 2.5
            viewer.wind.period = 9.0
            viewer.wind.amplitude = 5.0
            viewer.wind.frequency = 7.0
            viewer.wind.direction = (0.0, 1.0, 0.0)

            viewer.set_model(model_b)

            self.assertAlmostEqual(viewer.camera.near, 0.123)
            self.assertAlmostEqual(viewer.camera.far, 456.0)
            self.assertAlmostEqual(viewer.camera.fov, 33.0)
            _assert_vec_close(self, viewer.camera.pos, (3.0, 4.0, 5.0))
            self.assertAlmostEqual(viewer.camera.yaw, 37.0)
            self.assertAlmostEqual(viewer.camera.pitch, -12.0)
            _assert_vec_close(self, viewer.camera.pivot, (1.0, 2.0, 3.0))
            self.assertAlmostEqual(viewer.wind.time, 2.5)
            self.assertAlmostEqual(viewer.wind.period, 9.0)
            self.assertAlmostEqual(viewer.wind.amplitude, 5.0)
            self.assertAlmostEqual(viewer.wind.frequency, 7.0)
            _assert_vec_close(self, viewer.wind.direction, (0.0, 1.0, 0.0))
        finally:
            viewer.close()

    def test_set_model_resets_camera_for_different_up_axis(self):
        model_a = _build_viewer_model(1.0, up_axis=newton.Axis.Z)
        model_b = _build_viewer_model(2.0, up_axis=newton.Axis.Y)

        try:
            viewer = ViewerGL(headless=True)
        except Exception as exc:
            self.skipTest(f"ViewerGL not available: {exc}")
            return

        try:
            viewer.set_model(model_a)
            viewer.camera.near = 0.123
            viewer.camera.far = 456.0
            viewer.camera.fov = 33.0
            viewer.camera.pos = viewer.camera._as_vec3((3.0, 4.0, 5.0))
            viewer.camera.yaw = 37.0
            viewer.camera.pitch = -12.0
            viewer.camera.set_pivot((1.0, 2.0, 3.0))
            viewer.wind.amplitude = 5.0

            viewer.set_model(model_b)

            self.assertAlmostEqual(viewer.camera.near, 0.01)
            self.assertAlmostEqual(viewer.camera.far, 1000.0)
            self.assertAlmostEqual(viewer.camera.fov, 45.0)
            _assert_vec_close(self, viewer.camera.pos, (0.0, 2.0, 10.0))
            self.assertAlmostEqual(viewer.camera.yaw, -180.0)
            self.assertAlmostEqual(viewer.camera.pitch, 0.0)
            self.assertAlmostEqual(viewer.wind.amplitude, 5.0)
        finally:
            viewer.close()


if __name__ == "__main__":
    unittest.main()
