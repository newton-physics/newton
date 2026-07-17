# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import warp as wp

from newton._src.viewer.gl.opengl import MeshGL
from newton._src.viewer.viewer_gl import ViewerGL
from newton._src.viewer.viewer_gui import ViewerGui
from newton._src.viewer.viewer_null import ViewerNull

_Vec3 = namedtuple("_Vec3", ("x", "y", "z"))


def _make_gl_state(paused: bool = False, step_requested: bool = False) -> "ViewerGL":
    # Lightweight stand-in with just the fields ViewerGL.should_step() needs.
    return SimpleNamespace(_paused=paused, _step_requested=step_requested)  # type: ignore[return-value]


class TestViewerBaseShouldStep(unittest.TestCase):
    """ViewerBase.should_step() defaults to not self.is_paused()."""

    def test_returns_true_when_not_paused(self):
        viewer = ViewerNull()
        self.assertTrue(viewer.should_step())

    def test_returns_true_on_repeated_calls(self):
        viewer = ViewerNull()
        for _ in range(3):
            self.assertTrue(viewer.should_step())


class TestViewerCameraSpeed(unittest.TestCase):
    def test_defaults_to_four_meters_per_second(self):
        self.assertEqual(ViewerNull().camera_speed, 4.0)

    def test_accepts_finite_nonnegative_values(self):
        viewer = ViewerNull()

        viewer.camera_speed = 0.2
        self.assertEqual(viewer.camera_speed, 0.2)

        viewer.camera_speed = 0.0
        self.assertEqual(viewer.camera_speed, 0.0)

    def test_rejects_negative_and_nonfinite_values(self):
        viewer = ViewerNull()

        for value in (-1.0, float("inf"), float("-inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                viewer.camera_speed = value

    def test_gui_keyboard_movement_uses_viewer_camera_speed(self):
        camera = SimpleNamespace(
            pos=_Vec3(0.0, 0.0, 0.0),
            get_front=lambda: (1.0, 0.0, 0.0),
            get_right=lambda: (0.0, 1.0, 0.0),
            get_up=lambda: (0.0, 0.0, 1.0),
        )
        viewer = SimpleNamespace(camera=camera, camera_speed=2.0)
        gui = ViewerGui.__new__(ViewerGui)
        gui._viewer = viewer
        gui.ui = None
        gui._cam_vel = np.zeros(3, dtype=np.float32)
        gui._cam_damp_tau = 0.1

        key = SimpleNamespace(W=1, UP=2, S=3, DOWN=4, A=5, LEFT=6, D=7, RIGHT=8, Q=9, E=10)
        pyglet = SimpleNamespace(window=SimpleNamespace(key=key))
        with patch.dict(sys.modules, {"pyglet": pyglet}):
            gui.update_camera_from_keys(0.1, lambda code: code == key.W)

        self.assertAlmostEqual(camera.pos.x, 0.2)
        self.assertAlmostEqual(camera.pos.y, 0.0)
        self.assertAlmostEqual(camera.pos.z, 0.0)


class TestViewerGLShouldStep(unittest.TestCase):
    """ViewerGL.should_step() state machine: running, paused, and single-step."""

    def test_returns_true_when_running(self):
        v = _make_gl_state(paused=False, step_requested=False)
        self.assertTrue(ViewerGL.should_step(v))

    def test_returns_false_when_paused(self):
        v = _make_gl_state(paused=True, step_requested=False)
        self.assertFalse(ViewerGL.should_step(v))

    def test_returns_true_once_after_step_request(self):
        v = _make_gl_state(paused=True, step_requested=True)
        self.assertTrue(ViewerGL.should_step(v))
        self.assertFalse(ViewerGL.should_step(v))

    def test_stale_request_cleared_when_running(self):
        # Reproduces the bug: . pressed while running, then SPACE to pause.
        # The flag must not survive into the paused state and fire a spurious step.
        v = _make_gl_state(paused=False, step_requested=True)
        ViewerGL.should_step(v)  # running frame — must clear the flag
        v._paused = True
        self.assertFalse(ViewerGL.should_step(v))

    def test_multiple_step_requests_fire_once_each(self):
        v = _make_gl_state(paused=True, step_requested=True)
        self.assertTrue(ViewerGL.should_step(v))
        v._step_requested = True
        self.assertTrue(ViewerGL.should_step(v))
        self.assertFalse(ViewerGL.should_step(v))


class TestViewerGLParticles(unittest.TestCase):
    def test_hidden_particles_skip_instance_updates(self):
        viewer = ViewerGL.__new__(ViewerGL)
        viewer.show_particles = False
        viewer._layer_force_hidden = Mock(return_value=False)
        viewer._qualify = Mock(side_effect=lambda name: name)
        viewer.log_points = Mock()

        viewer._log_particles(SimpleNamespace())

        viewer.log_points.assert_called_once_with("/model/particles", points=None, hidden=True)


class TestViewerGLDynamicMeshes(unittest.TestCase):
    def test_dynamic_normal_scratch_supports_shrink_then_growth(self):
        mesh = MeshGL.__new__(MeshGL)
        mesh.device = wp.get_device("cpu")
        mesh.max_points = 8
        mesh.num_points = 3
        mesh._points = wp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=wp.vec3)
        mesh.indices = wp.array([0, 1, 2], dtype=wp.uint32)
        mesh.normals = None

        normal_lengths = []

        def record_normals(_points, _indices, normals, **_kwargs):
            normal_lengths.append(len(normals))
            return normals

        with patch("newton._src.viewer.gl.opengl.compute_vertex_normals", side_effect=record_normals):
            mesh.recompute_normals()
            scratch = mesh.normals
            self.assertEqual(len(scratch), 8)

            mesh.num_points = 7
            mesh._points = wp.zeros(7, dtype=wp.vec3)
            mesh.recompute_normals()

        self.assertIs(mesh.normals, scratch)
        self.assertEqual(len(mesh.normals), 8)
        self.assertEqual(normal_lengths, [3, 7])

    def test_dynamic_mesh_reuses_capacity_and_rebinds_instancers_on_growth(self):
        class FakeMesh:
            def __init__(self, num_points, num_indices, device, hidden=False, backface_culling=True, dynamic=False):
                self.max_points = num_points
                self.max_indices = num_indices
                self.num_points = num_points
                self.num_indices = num_indices
                self.device = device
                self.hidden = hidden
                self.backface_culling = backface_culling
                self.dynamic = dynamic
                self.color = (0.7, 0.5, 0.3)
                self.material = (0.5, 0.0, 0.0, 0.0)
                self.destroyed = False

            def update(self, points, indices, normals, uvs, texture):
                self.num_points = len(points)
                self.num_indices = len(indices)

            def destroy(self):
                self.destroyed = True

        class FakeInstancer:
            def __init__(self, mesh):
                self.mesh = mesh
                self.rebinds = 0

            def set_mesh(self, mesh):
                self.mesh = mesh
                self.rebinds += 1

        viewer = ViewerGL.__new__(ViewerGL)
        viewer.objects = {}
        viewer.device = wp.get_device("cpu")
        viewer._qualify = lambda name: name
        points = wp.zeros(3, dtype=wp.vec3)
        indices = wp.zeros(3, dtype=wp.int32)

        with (
            patch("newton._src.viewer.viewer_gl.MeshGL", FakeMesh),
            patch("newton._src.viewer.viewer_gl.MeshInstancerGL", FakeInstancer),
        ):
            viewer.log_mesh("mesh", points, indices, dynamic=True)
            original = viewer.objects["mesh"]
            instancer = FakeInstancer(original)
            viewer.objects["instances"] = instancer

            viewer.log_mesh("mesh", points[:2], indices, dynamic=True)
            self.assertIs(viewer.objects["mesh"], original)
            self.assertEqual(instancer.rebinds, 0)

            viewer.log_mesh("mesh", wp.zeros(7, dtype=wp.vec3), indices, dynamic=True)

        self.assertTrue(original.destroyed)
        self.assertIs(instancer.mesh, viewer.objects["mesh"])
        self.assertEqual(instancer.rebinds, 1)
        self.assertGreaterEqual(viewer.objects["mesh"].max_points, 7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
