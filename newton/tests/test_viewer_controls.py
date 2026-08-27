# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import inspect
import sys
import unittest
from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import warp as wp

from newton._src.viewer.gl.opengl import MeshGL, MeshInstancerGL, RendererGL, RenderVertex
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


def _make_gl_running_state(headless: bool, num_frames: int | None, frame_count: int = 0) -> "ViewerGL":
    # Stand-in carrying only the fields ViewerGL.is_running()/end_frame() read,
    # so the frame budget can be exercised without a GL context.
    return SimpleNamespace(  # type: ignore[return-value]
        renderer=SimpleNamespace(has_exit=lambda: False),
        _headless=headless,
        num_frames=num_frames,
        _frame_count=frame_count,
        _update=lambda: None,
    )


class TestViewerGLFrameBudget(unittest.TestCase):
    """ViewerGL.is_running() honours num_frames in headless mode."""

    def test_headless_stops_once_num_frames_reached(self):
        """Verify headless rendering stops after num_frames frames."""
        v = _make_gl_running_state(headless=True, num_frames=3)
        for _ in range(3):
            self.assertTrue(ViewerGL.is_running(v))
            ViewerGL.end_frame(v)
        self.assertFalse(ViewerGL.is_running(v))

    def test_headless_without_num_frames_runs_unbounded(self):
        """Verify headless rendering is unbounded when num_frames is None."""
        v = _make_gl_running_state(headless=True, num_frames=None)
        for _ in range(5):
            ViewerGL.end_frame(v)
        self.assertTrue(ViewerGL.is_running(v))

    def test_windowed_ignores_num_frames(self):
        """Verify a visible window keeps running past num_frames."""
        v = _make_gl_running_state(headless=False, num_frames=1)
        for _ in range(3):
            ViewerGL.end_frame(v)
        self.assertTrue(ViewerGL.is_running(v))

    def test_window_close_stops_headless_run_early(self):
        """Verify an exit request wins over a remaining frame budget."""
        v = _make_gl_running_state(headless=True, num_frames=10)
        v.renderer.has_exit = lambda: True
        self.assertFalse(ViewerGL.is_running(v))

    def test_end_frame_counts_frames(self):
        """Verify end_frame() advances the frame counter used by the budget."""
        v = _make_gl_running_state(headless=True, num_frames=2)
        ViewerGL.end_frame(v)
        self.assertEqual(v._frame_count, 1)

    def test_zero_num_frames_stops_before_the_first_frame(self):
        """Verify a zero budget renders nothing at all."""
        v = _make_gl_running_state(headless=True, num_frames=0)
        self.assertFalse(ViewerGL.is_running(v))


class TestViewerGLNumFramesValidation(unittest.TestCase):
    """ViewerGL rejects num_frames values that would otherwise fail silently.

    The budget is applied as ``_frame_count < num_frames``, so a non-integer
    or negative value produces a surprising frame count rather than an error.
    These inputs are rejected before any GL context is created, so the tests
    need no display.
    """

    def test_rejects_non_integer_num_frames(self):
        """Verify a float num_frames raises TypeError rather than rendering a fractional budget."""
        with self.assertRaises(TypeError):
            ViewerGL(num_frames=1.5)  # type: ignore[arg-type]

    def test_rejects_bool_num_frames(self):
        """Verify a bool num_frames raises TypeError rather than being treated as 0 or 1."""
        with self.assertRaises(TypeError):
            ViewerGL(num_frames=True)

    def test_rejects_negative_num_frames(self):
        """Verify a negative num_frames raises ValueError rather than silently rendering nothing."""
        with self.assertRaises(ValueError):
            ViewerGL(num_frames=-1)

    def test_rejects_invalid_cuda_interop_mode(self):
        for value in (True, 1, 1.5, "dynamic"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                ViewerGL(enable_cuda_interop=value)  # type: ignore[arg-type]

    def test_cuda_interop_defaults_to_dynamic_meshes(self):
        parameter = inspect.signature(ViewerGL).parameters["enable_cuda_interop"]
        self.assertEqual(parameter.default, ViewerGL.CudaInterop.DYNAMIC_MESH)

    def test_cuda_interop_flags_are_composable(self):
        viewer = ViewerGL.__new__(ViewerGL)
        viewer._enable_cuda_interop = ViewerGL.CudaInterop.POINTS | ViewerGL.CudaInterop.LINES

        self.assertTrue(viewer._cuda_interop_enabled(ViewerGL.CudaInterop.POINTS))
        self.assertTrue(viewer._cuda_interop_enabled(ViewerGL.CudaInterop.LINES))
        self.assertFalse(viewer._cuda_interop_enabled(ViewerGL.CudaInterop.INSTANCES))
        self.assertEqual(
            ViewerGL.CudaInterop.ALL,
            ViewerGL.CudaInterop.DYNAMIC_MESH
            | ViewerGL.CudaInterop.STATIC_MESH
            | ViewerGL.CudaInterop.POINTS
            | ViewerGL.CudaInterop.INSTANCES
            | ViewerGL.CudaInterop.LINES,
        )

    def test_rejects_unknown_cuda_interop_flags(self):
        with self.assertRaises(ValueError):
            ViewerGL(enable_cuda_interop=ViewerGL.CudaInterop(1 << 10))


class TestViewerGLParticles(unittest.TestCase):
    def test_hidden_particles_skip_instance_updates(self):
        viewer = ViewerGL.__new__(ViewerGL)
        viewer.show_particles = False
        viewer._layer_force_hidden = Mock(return_value=False)
        viewer._qualify = Mock(side_effect=lambda name: name)
        viewer.log_points = Mock()

        viewer._log_particles(SimpleNamespace())

        viewer.log_points.assert_called_once_with("/model/particles", points=None, hidden=True)


class TestViewerGLCleanup(unittest.TestCase):
    def test_close_destroys_geometry_before_renderer(self):
        events = []
        viewer = ViewerGL.__new__(ViewerGL)
        viewer._clear_array_textures = Mock(side_effect=lambda: events.append("textures"))
        viewer._invalidate_pbo = Mock(side_effect=lambda: events.append("pbo"))
        viewer._image_logger = SimpleNamespace(clear=lambda: events.append("images"))
        viewer._destroy_render_geometry = Mock(side_effect=lambda: events.append("geometry"))
        viewer.renderer = SimpleNamespace(close=lambda: events.append("renderer"))

        viewer.close()

        self.assertEqual(events, ["textures", "pbo", "images", "geometry", "renderer"])

    def test_destroy_render_geometry_releases_all_owned_resources(self):
        viewer = ViewerGL.__new__(ViewerGL)
        instancer = MeshInstancerGL.__new__(MeshInstancerGL)
        instancer.destroy = Mock()
        mesh = Mock()
        line = Mock()
        arrow = Mock()
        wireframe = Mock()
        wireframe_owner = Mock()
        point_mesh = Mock()
        gaussian_mesh = Mock()
        viewer.objects = {"instancer": instancer, "mesh": mesh}
        viewer.lines = {"line": line}
        viewer.arrows = {"arrow": arrow}
        viewer.wireframe_shapes = {"wireframe": wireframe}
        viewer._wireframe_vbo_owners = {1: wireframe_owner}
        viewer._point_mesh = point_mesh
        viewer._gaussian_mesh = gaussian_mesh

        viewer._destroy_render_geometry()

        for resource in (instancer, mesh, line, arrow, wireframe, wireframe_owner, point_mesh, gaussian_mesh):
            resource.destroy.assert_called_once_with()
        self.assertFalse(viewer.objects)
        self.assertFalse(viewer.lines)
        self.assertFalse(viewer.arrows)
        self.assertFalse(viewer.wireframe_shapes)
        self.assertFalse(viewer._wireframe_vbo_owners)
        self.assertIsNone(viewer._point_mesh)
        self.assertIsNone(viewer._gaussian_mesh)

    def test_instancer_releases_registration_before_gl_buffers(self):
        events = []

        class Registration:
            def __del__(self):
                events.append("registration")

        instancer = MeshInstancerGL.__new__(MeshInstancerGL)
        instancer._instance_transform_cuda_buffer = Registration()
        instancer.vao = 1
        instancer.instance_transform_buffer = 2
        instancer.instance_color_buffer = 3
        instancer.instance_material_buffer = 4
        gl = Mock()
        gl.glDeleteVertexArrays.side_effect = lambda *_args: events.append("vao")
        gl.glDeleteBuffers.side_effect = lambda *_args: events.append("buffer")

        with patch.object(RendererGL, "gl", gl):
            instancer.destroy()
            instancer.destroy()

        self.assertEqual(events, ["registration", "vao", "buffer", "buffer", "buffer"])
        self.assertIsNone(instancer.vao)
        self.assertIsNone(instancer.instance_transform_buffer)
        self.assertIsNone(instancer.instance_color_buffer)
        self.assertIsNone(instancer.instance_material_buffer)


class TestViewerGLDynamicMeshes(unittest.TestCase):
    def test_dynamic_mesh_uploads_indices_through_cuda_interop(self):
        mesh = MeshGL.__new__(MeshGL)
        mesh.device = wp.get_device("cpu")
        mesh.max_points = 3
        mesh.max_indices = 3
        mesh.vertex_byte_size = 32
        mesh.index_byte_size = 4
        mesh.dynamic = True
        mesh.indices = None
        mesh.normals = None
        mesh.vertices = wp.zeros(3, dtype=RenderVertex)
        mesh.index_cuda_buffer = Mock()
        mesh.index_cuda_buffer.map.return_value = wp.empty(3, dtype=wp.uint32)
        mesh.vertex_cuda_buffer = Mock()
        mesh.vertex_cuda_buffer.map.return_value = wp.empty(3, dtype=RenderVertex)
        mesh.update_texture = Mock()

        points = wp.zeros(3, dtype=wp.vec3)
        indices = wp.array([0, 1, 2], dtype=wp.int32)
        normals = wp.zeros(3, dtype=wp.vec3)
        gl = Mock(GL_DYNAMIC_DRAW=1, GL_STATIC_DRAW=2, GL_ELEMENT_ARRAY_BUFFER=3, GL_ARRAY_BUFFER=4)

        with patch.object(RendererGL, "gl", gl), patch("newton._src.viewer.gl.opengl.wp.launch"):
            mesh.update(points, indices, normals, None)
            mesh.index_cuda_buffer.map.assert_called_once_with(dtype=wp.uint32, shape=(3,))
            mesh.index_cuda_buffer.unmap.assert_called_once_with()
            mesh.vertex_cuda_buffer.map.assert_called_once_with(dtype=RenderVertex, shape=(3,))
            mesh.vertex_cuda_buffer.unmap.assert_called_once_with()
            gl.glBufferSubData.assert_not_called()

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
            def __init__(
                self,
                num_points,
                num_indices,
                device,
                hidden=False,
                backface_culling=True,
                dynamic=False,
                *,
                enable_cuda_interop=False,
            ):
                self.max_points = num_points
                self.max_indices = num_indices
                self.num_points = num_points
                self.num_indices = num_indices
                self.device = device
                self.hidden = hidden
                self.backface_culling = backface_culling
                self.dynamic = dynamic
                self.enable_cuda_interop = enable_cuda_interop
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
        viewer._enable_cuda_interop = ViewerGL.CudaInterop.DYNAMIC_MESH
        viewer._qualify = lambda name: name
        points = wp.zeros(3, dtype=wp.vec3)
        indices = wp.zeros(3, dtype=wp.int32)

        with (
            patch("newton._src.viewer.viewer_gl.MeshGL", FakeMesh),
            patch("newton._src.viewer.viewer_gl.MeshInstancerGL", FakeInstancer),
        ):
            viewer.log_mesh("mesh", points, indices, dynamic=True)
            original = viewer.objects["mesh"]
            self.assertTrue(original.enable_cuda_interop)
            viewer.log_mesh("static", points, indices)
            self.assertFalse(viewer.objects["static"].enable_cuda_interop)
            viewer._enable_cuda_interop = ViewerGL.CudaInterop.STATIC_MESH
            viewer.log_mesh("all", points, indices)
            self.assertTrue(viewer.objects["all"].enable_cuda_interop)
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

    def test_points_interop_is_selected_independently(self):
        class FakeInstancer:
            def __init__(self, num_instances, mesh, *, enable_cuda_interop=False):
                self.num_instances = num_instances
                self.mesh = mesh
                self.enable_cuda_interop = enable_cuda_interop
                self.hidden = False

            def update_from_points(self, points, radii, colors):
                pass

        viewer = ViewerGL.__new__(ViewerGL)
        viewer.objects = {}
        viewer.device = wp.get_device("cpu")
        viewer._point_mesh = object()
        viewer._qualify = lambda name: name
        points = wp.zeros(3, dtype=wp.vec3)
        radii = wp.ones(3, dtype=wp.float32)
        colors = wp.ones(3, dtype=wp.vec3)

        with patch("newton._src.viewer.viewer_gl.MeshInstancerGL", FakeInstancer):
            viewer._enable_cuda_interop = ViewerGL.CudaInterop.POINTS
            viewer.log_points("interop", points, radii, colors)
            viewer._enable_cuda_interop = ViewerGL.CudaInterop.INSTANCES
            viewer.log_points("host", points, radii, colors)

        self.assertTrue(viewer.objects["interop"].enable_cuda_interop)
        self.assertFalse(viewer.objects["host"].enable_cuda_interop)


if __name__ == "__main__":
    unittest.main(verbosity=2)
