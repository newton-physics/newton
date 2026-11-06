# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest
from itertools import product

import numpy as np
import warp as wp

import newton
from newton._src.viewer.camera import Camera
from newton._src.viewer.viewer_gui import ViewerGui
from newton.viewer import ViewerGL, ViewerNull


def _as_np(value):
    return np.array((value[0], value[1], value[2]), dtype=float)


def _assert_vec_close(test, actual, expected, tol=1.0e-6):
    np.testing.assert_allclose(_as_np(actual), np.array(expected, dtype=float), atol=tol, rtol=0.0)


class TestViewerCamera(unittest.TestCase):
    def _frame(self, viewer, points, width=1280, height=720, up_axis="Z"):
        viewer.camera = Camera(width=width, height=height, up_axis=up_axis)
        viewer.camera.yaw = 37.0
        viewer.camera.pitch = -25.0
        gui = ViewerGui.__new__(ViewerGui)
        gui._viewer = viewer
        gui.frame_camera_on_model()

        # Check the actual perspective projection, including near/far clipping.
        view = viewer.camera.get_view_matrix().reshape(4, 4).T
        projection = viewer.camera.get_projection_matrix().reshape(4, 4).T
        clip = np.c_[points, np.ones(len(points))] @ view.T @ projection.T
        self.assertTrue(np.all(clip[:, 3] > 0.0))
        ndc = clip[:, :3] / clip[:, 3, None]
        self.assertLessEqual(np.abs(ndc).max(), 1.0 + 1.0e-5)
        self.assertEqual(viewer.camera.yaw, 37.0)
        self.assertEqual(viewer.camera.pitch, -25.0)
        return viewer.camera

    def test_frame_geometry(self):
        """Fit displayed primitives, meshes, shells, and Gaussian clouds across camera orientations."""
        builder = newton.ModelBuilder()
        body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.0), wp.quat(0.5, 0.5, 0.5, 0.5)))
        builder.add_shape_box(body, xform=wp.transform((12.0, 0.0, 0.0), wp.quat_identity()), hx=8.0, hy=0.2, hz=0.2)
        vertices = np.array(((3.0, 0.0, 0.0), (8.0, 0.0, 0.0), (3.0, 2.0, 0.0)), dtype=np.float32)
        mesh_shape = builder.add_shape_mesh(
            -1, mesh=newton.Mesh(vertices, [0, 1, 2], compute_inertia=False), scale=(-2.0, 3.0, 1.0)
        )
        builder.add_shape_gaussian(
            body,
            xform=wp.transform((30.0, 0.0, 0.0), wp.quat_identity()),
            gaussian=newton.Gaussian([[5.0, 1.0, 0.0]], scales=[[2.0, 2.0, 2.0]]),
        )
        builder.add_ground_plane()
        model = builder.finalize(device="cpu")
        viewer = ViewerNull()
        # A state logged before attachment must not leak into the new model.
        viewer.log_state(model.state())
        viewer.set_model(model)
        self.assertIsNone(viewer._last_state)
        viewer.show_gaussians = True
        state = model.state()
        viewer.log_state(state)
        corners = np.array(list(product((4.0, 20.0), (-0.2, 0.2), (-0.2, 0.2))))
        splat_corners = np.array(list(product((-2.0, 2.0), (33.0, 37.0), (-1.0, 3.0))))
        points = np.vstack((corners[:, (2, 0, 1)], vertices * (-2.0, 3.0, 1.0), splat_corners))
        for width, height in ((1280, 720), (320, 1000)):
            for up_axis in ("X", "Y", "Z"):
                with self.subTest(width=width, height=height, up_axis=up_axis):
                    camera = self._frame(viewer, points, width, height, up_axis)
                    _assert_vec_close(self, camera.pivot, (-7.0, 18.5, 1.0))

        # Reuse the mesh with a substantial shell, including its signed scale.
        solid = model.shape_is_solid.numpy()
        solid[mesh_shape] = False
        model.shape_is_solid.assign(solid)
        margins = model.shape_margin.numpy()
        margins[mesh_shape] = 3.0
        model.shape_margin.assign(margins)
        viewer.set_model(model)
        viewer.show_visual = viewer.show_collision = False
        viewer.show_static = True
        viewer.log_state(state)
        shell = np.vstack([vertices * (-2.0, 3.0, 1.0) + (0.0, 0.0, z) for z in (-3.0, 3.0)])
        bounds = viewer._get_scene_bounds()
        self.assertTrue(np.all(bounds[0] <= shell.min(axis=0)))
        self.assertTrue(np.all(bounds[1] >= shell.max(axis=0)))
        camera = self._frame(viewer, shell, width=320, height=1000)
        _assert_vec_close(self, camera.pivot, (-11.0, 3.0, 0.0))

        if wp.is_cuda_available():
            with self.subTest(geometry="collision_isomesh"), wp.ScopedDevice("cuda:0"):
                mesh = newton.Mesh.create_box(0.3, 0.3, 0.3)
                mesh.build_sdf(max_resolution=16)
                builder = newton.ModelBuilder()
                builder.add_shape_mesh(-1, mesh=mesh)
                model = builder.finalize()
                viewer = ViewerNull()
                viewer.set_model(model)
                viewer.show_visual = viewer.show_static = False
                viewer.show_collision = True
                viewer.log_state(model.state())
                isomesh = viewer._get_sdf_isomesh(0)
                self.assertIsNotNone(isomesh)
                bounds = viewer._get_scene_bounds()
                self.assertIsNotNone(bounds)
                np.testing.assert_allclose(
                    bounds, (isomesh.vertices.min(axis=0), isomesh.vertices.max(axis=0)), rtol=0.0, atol=1.0e-6
                )
                self._frame(viewer, isomesh.vertices)

    def test_frame_visibility(self):
        """Frame visible worlds, layers, particles, triangles, and COM markers."""
        world = newton.ModelBuilder()
        body = world.add_body()
        world.add_shape_box(body, hx=0.5, hy=0.5, hz=2.0)
        builder = newton.ModelBuilder()
        builder.replicate(world, 4)
        builder.add_ground_plane()
        builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=2.0)
        builder.add_particle(pos=(100.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, flags=0)
        model = builder.finalize(device="cpu")
        viewer = ViewerNull()
        viewer.set_model(model)
        state = model.state()
        translations = np.zeros((4, 3))
        translations[1, 0] = 4000.0
        poses = state.body_q.numpy()
        poses[:, :3] += translations
        state.body_q.assign(poses)
        for visible in (None, [0, 2, 3]):
            with self.subTest(visible=visible):
                viewer.set_visible_worlds(visible)
                viewer.set_world_offsets((1000.0, 1000.0, 0.0))
                viewer.log_state(state)
                selected = visible if visible is not None else slice(None)
                offsets = viewer.world_offsets.numpy()[selected] + translations[selected]
                corners = np.array(list(product((-0.5, 0.5), (-0.5, 0.5), (-2.0, 2.0))))
                points = (corners[None] + offsets[:, None]).reshape(-1, 3)
                camera = self._frame(viewer, points)
                _assert_vec_close(self, camera.pivot, (points.min(axis=0) + points.max(axis=0)) * 0.5)

        # Reuse the model with rigid shapes hidden to isolate particle extents.
        viewer.layer.visible = False
        for name, offset, particle_x in (("left", -20.0, -2.0), ("right", 20.0, 4.0), ("hidden", 10000.0, 30.0)):
            layer = viewer.activate(name)
            layer.xform = wp.transform((offset, 0.0, 0.0), wp.quat_identity())
            layer.visible = name != "hidden"
            viewer.set_model(model)
            viewer.show_visual = False
            viewer.show_collision = False
            viewer.show_static = False
            viewer.show_particles = True
            layer_state = model.state()
            positions = layer_state.particle_q.numpy()
            positions[0, 0] = particle_x
            layer_state.particle_q.assign(positions)
            viewer.log_state(layer_state)
        corners = np.array(list(product((-24.0, 26.0), (-2.0, 2.0), (-2.0, 2.0))))
        camera = self._frame(viewer, corners)
        _assert_vec_close(self, camera.pivot, (1.0, 0.0, 0.0))

        # Cover shape-free display paths with one particle/triangle/COM fixture.
        builder = newton.ModelBuilder()
        builder.add_body(
            xform=wp.transform((10.0, 0.0, 0.0), wp.quat_identity()),
            com=(2.0, 0.0, 0.0),
            mass=1.0,
            inertia=wp.diag(wp.vec3(1.0)),
        )
        positions = np.array(((0, 0, 0), (1, 0, 0), (0, 2, 0), (100, 0, 0), (101, 0, 0), (100, 2, 0)), dtype=float)
        for i, point in enumerate(positions):
            builder.add_particle(
                pos=wp.vec3(point),
                vel=(0, 0, 0),
                mass=1.0,
                radius=0.1,
                flags=0 if i == 2 else newton.ParticleFlags.ACTIVE,
            )
        builder.add_triangle(0, 1, 2)
        builder.add_triangle(3, 4, 5, opacity=0.0)
        model = builder.finalize(device="cpu")
        viewer = ViewerNull()
        viewer.set_model(model)
        viewer.show_triangles = False
        viewer.log_points("custom", wp.array(positions[:3], dtype=wp.vec3, device="cpu"))
        camera = self._frame(viewer, positions[:3])
        _assert_vec_close(self, camera.pivot, (0.0, 0.0, 0.0))
        viewer.show_particles = True
        active = positions[[0, 1, 3, 4, 5]]
        self._frame(viewer, np.vstack((active - 0.1, active + 0.1)))
        viewer.show_particles = False
        viewer.show_triangles = True
        viewer.log_state(model.state())
        camera = self._frame(viewer, positions[:3])
        _assert_vec_close(self, camera.pivot, (0.5, 1.0, 0.0))
        viewer.show_triangles = False
        viewer.show_com = True
        viewer.log_state(model.state())
        radius = 0.5 * viewer.scene_scale
        corners = np.array(list(product((12.0 - radius, 12.0 + radius), (-radius, radius), (-radius, radius))))
        camera = self._frame(viewer, corners)
        _assert_vec_close(self, camera.pivot, (12.0, 0.0, 0.0))
        viewer.show_com = False
        self._frame(viewer, np.vstack((active, [[10.0, 0.0, 0.0]])))

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
        self.assertAlmostEqual(camera.pivot_distance, camera.MIN_PIVOT_DISTANCE)

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
