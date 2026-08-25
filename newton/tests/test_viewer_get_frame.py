# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import ctypes
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

import newton
import newton.viewer
from newton._src.viewer.gl.opengl import MeshInstancerGL, RendererGL
from newton._src.viewer.viewer_gl import ViewerGL


def _viewer_gl_unavailable_error_types(test: unittest.TestCase) -> tuple[type[BaseException], ...]:
    try:
        __import__("pyglet")
    except ImportError as exc:
        test.skipTest(f"ViewerGL dependencies not available: {exc}")

    unavailable_errors = []
    for module_name, exception_names in (
        ("pyglet.gl", ("ConfigException", "ContextException")),
        ("pyglet.gl.lib", ("MissingFunctionException",)),
        ("pyglet.window", ("NoSuchConfigException", "NoSuchDisplayException")),
    ):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        unavailable_errors.extend(
            exception_type
            for exception_name in exception_names
            if isinstance(exception_type := getattr(module, exception_name, None), type)
        )

    return tuple(dict.fromkeys(unavailable_errors))


def _is_viewer_gl_unavailable_error(test: unittest.TestCase, exc: Exception) -> bool:
    if isinstance(exc, _viewer_gl_unavailable_error_types(test)):
        return True

    # Some pyglet platform backends raise their own NoSuchDisplayException
    # while importing pyglet.window, before the window-level class exists.
    return type(exc).__module__.startswith("pyglet.") and type(exc).__name__ in {
        "ConfigException",
        "ContextException",
        "MissingFunctionException",
        "NoSuchConfigException",
        "NoSuchDisplayException",
    }


def _reset_pyglet_event_loop_exit(test: unittest.TestCase) -> None:
    _viewer_gl_unavailable_error_types(test)
    pyglet = sys.modules.get("pyglet")
    if pyglet is not None:
        pyglet.app.event_loop.has_exit = False


def _make_headless_viewer_gl_or_skip(test: unittest.TestCase, *, width: int = 64, height: int = 48):
    _reset_pyglet_event_loop_exit(test)

    try:
        return newton.viewer.ViewerGL(width=width, height=height, headless=True)
    except Exception as exc:
        if _is_viewer_gl_unavailable_error(test, exc):
            test.skipTest(f"ViewerGL display/backend not available: {exc}")
        raise


def _make_box_model(
    device: str | wp.Device,
    *,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    color: tuple[float, float, float] = (1.0, 0.0, 0.0),
):
    """Build a single-box model for viewer tests."""
    builder = newton.ModelBuilder()
    body = builder.add_body(xform=wp.transform(wp.vec3(*position), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.4, hy=0.4, hz=0.4, color=color)
    return builder.finalize(device=device)


def _make_ground_model(device: str | wp.Device):
    """Build an opaque static receiver beneath translucent test geometry."""
    builder = newton.ModelBuilder()
    builder.add_shape_box(
        -1,
        xform=wp.transform((0.0, 0.0, -0.45), wp.quat_identity()),
        hx=2.0,
        hy=2.0,
        hz=0.05,
        color=(0.65, 0.65, 0.65),
    )
    return builder.finalize(device=device)


class _FakeGL:
    GL_PIXEL_PACK_BUFFER = 0x88EB
    GL_STREAM_READ = 0x88E1
    GL_PACK_ALIGNMENT = 0x0D05
    GL_FRAMEBUFFER = 0x8D40
    GL_RGB = 0x1907
    GL_UNSIGNED_BYTE = 0x1401

    GLuint = ctypes.c_uint
    GLsizeiptr = ctypes.c_size_t

    def __init__(self, pixels: np.ndarray):
        self.pixels = pixels
        self.bound_buffer = 0
        self.readback_count = 0

    def glGenBuffers(self, count, buffers):
        buffers[0] = 17

    def glBindBuffer(self, target, buffer):
        self.bound_buffer = int(buffer)

    def glBufferData(self, target, size, data, usage):
        pass

    def glPixelStorei(self, name, value):
        pass

    def glBindFramebuffer(self, target, framebuffer):
        pass

    def glReadPixels(self, x, y, width, height, pixel_format, pixel_type, data):
        pass

    def glGetBufferSubData(self, target, offset, size, data):
        if self.bound_buffer == 0:
            raise RuntimeError("pixel buffer must be bound during readback")
        ctypes.memmove(data, self.pixels.ctypes.data, self.pixels.nbytes)
        self.readback_count += 1


class TestViewerGLTransparencyOrdering(unittest.TestCase):
    def test_translucent_instances_sort_globally_back_to_front(self):
        """Sort translucent instances globally and use names to break depth ties."""
        draw_order = []

        def make_instancer(label, centers):
            instancer = MeshInstancerGL.__new__(MeshInstancerGL)
            instancer.instance_transform_cuda_buffer = None
            instancer.hidden = False
            instancer.translucent_instance_indices = lambda: np.arange(len(centers), dtype=np.int32)
            instancer.instance_center = lambda index: np.asarray(centers[index], dtype=np.float32)
            instancer.render_instance = lambda index: draw_order.append((label, index))
            return instancer

        renderer = RendererGL.__new__(RendererGL)
        renderer.camera = SimpleNamespace(pos=(0.0, 0.0, 0.0), get_front=lambda: (0.0, 0.0, 1.0))
        objects = {
            "near": make_instancer("near", [(0.0, 0.0, 1.0)]),
            "far-b": make_instancer("far-b", [(0.0, 0.0, 3.0)]),
            "far-a": make_instancer("far-a", [(0.0, 0.0, 3.0)]),
            "mixed": make_instancer("mixed", [(0.0, 0.0, 2.0), (0.0, 0.0, 4.0)]),
        }

        with mock.patch("newton._src.viewer.gl.opengl.check_gl_error"):
            renderer._draw_translucent_objects(objects)

        self.assertEqual(
            draw_order,
            [("mixed", 1), ("far-a", 0), ("far-b", 0), ("mixed", 0), ("near", 0)],
        )

    def test_instancer_tracks_centers_and_mixed_opacity(self):
        """Track transformed mesh centers and select only translucent instances."""
        instancer = MeshInstancerGL.__new__(MeshInstancerGL)
        instancer.instance_transform_cuda_buffer = None
        instancer.mesh = SimpleNamespace(local_center=np.array((1.0, 2.0, 3.0), dtype=np.float32))
        instancer._instance_centers = np.zeros((3, 3), dtype=np.float32)
        instancer._instance_styles = np.array(
            [
                (-1.0, -1.0, -1.0, 1.0),
                (1.0, 0.0, 0.0, 0.5),
                (0.0, 0.0, 1.0, 0.0),
            ],
            dtype=np.float32,
        )
        instancer.active_instances = 3
        instancer._centers_valid_count = 0

        transforms = np.tile(np.eye(4, dtype=np.float32).reshape(1, 16), (3, 1))
        transforms[0, 12:15] = (10.0, 20.0, 30.0)
        transforms[1, (0, 5, 10)] = (2.0, 3.0, 4.0)
        transforms[1, 12:15] = (-1.0, -2.0, -3.0)
        instancer._record_host_transforms(transforms, 3)

        np.testing.assert_allclose(instancer._instance_centers[0], (11.0, 22.0, 33.0))
        np.testing.assert_allclose(instancer._instance_centers[1], (1.0, 4.0, 9.0))
        np.testing.assert_array_equal(instancer.translucent_instance_indices(), np.array([1]))


class TestViewerGLGetFrame(unittest.TestCase):
    def test_backend_error_types_do_not_force_window_import(self):
        """Verify backend error discovery does not import pyglet.window."""
        try:
            import pyglet
        except ImportError as exc:
            self.skipTest(f"ViewerGL dependencies not available: {exc}")

        class _WindowProxy:
            _module = None

            def __getattr__(self, name):
                raise AssertionError("pyglet.window must not be imported")

        with mock.patch.object(pyglet, "window", _WindowProxy()):
            _viewer_gl_unavailable_error_types(self)

    def test_headless_frame_capture_across_devices(self):
        """Verify headless frame capture follows the active model device."""
        cuda_devices = wp.get_cuda_devices()
        if cuda_devices:
            wp.zeros(1, dtype=wp.float32, device=cuda_devices[0])

        viewer = _make_headless_viewer_gl_or_skip(self)

        try:
            cpu_device = wp.get_device("cpu")
            cpu_model = _make_box_model(cpu_device)
            viewer.set_model(cpu_model)
            self.assertEqual(viewer.device, cpu_device)

            viewer.set_camera(pos=wp.vec3(2.0, -3.0, 2.0), pitch=-25.0, yaw=35.0)
            viewer.begin_frame(0.0)
            viewer.log_state(cpu_model.state())
            viewer.end_frame()

            frame = viewer.get_frame()
            self.assertEqual(frame.shape, (48, 64, 3))
            self.assertEqual(frame.dtype, wp.uint8)
            self.assertEqual(frame.device, cpu_device)
            self.assertGreater(np.ptp(frame.numpy()), 0)

            target = wp.empty(shape=(48, 64, 3), dtype=wp.uint8, device=cpu_device)
            self.assertIs(viewer.get_frame(target_image=target), target)

            viewer._invalidate_pbo()
            self.assertEqual(viewer.get_frame().shape, (48, 64, 3))

            for cuda_device in cuda_devices[:2]:
                viewer.set_model(_make_box_model(cuda_device))
                self.assertEqual(viewer.device, cuda_device)

                # Capture the existing framebuffer to isolate PBO rebinding
                # from model-geometry updates.
                cuda_frame = viewer.get_frame()
                self.assertEqual(cuda_frame.shape, (48, 64, 3))
                self.assertEqual(cuda_frame.dtype, wp.uint8)
                self.assertEqual(cuda_frame.device, cuda_device)
                self.assertGreater(np.ptp(cuda_frame.numpy()), 0)
        finally:
            viewer.close()

    def test_layer_transparency_blends_in_depth_order_without_shadows_or_edges(self):
        """Blend layer ghosts back-to-front without depth writes, shadows, or opaque edges."""
        viewer = _make_headless_viewer_gl_or_skip(self, width=192, height=144)

        try:
            viewer.renderer.draw_sky = False
            viewer.renderer.sky_upper = (0.0, 0.0, 0.0)
            viewer.renderer.sky_lower = (0.0, 0.0, 0.0)
            viewer.renderer.specular_scale = 0.0
            viewer.renderer.spotlight_enabled = False

            models = {
                "ground": _make_ground_model(viewer.device),
                "far": _make_box_model(viewer.device, position=(0.0, 0.18, 0.0)),
                "near": _make_box_model(viewer.device, position=(0.0, -0.18, 0.0)),
            }
            styles = {
                "far": newton.viewer.LayerRenderStyle(color=(1.0, 0.05, 0.05), opacity=0.45),
                "near": newton.viewer.LayerRenderStyle(color=(0.05, 0.1, 1.0), opacity=0.45),
            }
            for layer_id in ("near", "ground", "far"):
                viewer.activate(layer_id)
                viewer.set_model(models[layer_id])
                if layer_id in styles:
                    viewer.set_layer_render_style(layer_id, styles[layer_id])

            viewer.set_camera(wp.vec3(3.0, -5.0, 2.5), pitch=0.0, yaw=0.0)
            viewer.camera.look_at((0.0, 0.0, 0.0))

            def render_frame(*, shadows=False, edges=False, fallback=False):
                viewer.renderer.draw_shadows = shadows
                viewer.renderer.draw_edges = edges
                viewer.begin_frame(0.0)
                for layer_id in ("near", "ground", "far"):
                    viewer.activate(layer_id)
                    viewer.log_state(models[layer_id].state())
                for name, obj in viewer.objects.items():
                    if isinstance(obj, MeshInstancerGL):
                        if "/layers/ground/" in name:
                            obj.cast_shadow = False
                            obj.draw_edge = False
                        if fallback:
                            obj._supports_base_instance = False
                viewer.end_frame()
                return viewer.get_frame().numpy().copy()

            baseline = render_frame()
            fallback = render_frame(fallback=True)
            with_shadows = render_frame(shadows=True, fallback=True)
            with_edges = render_frame(edges=True, fallback=True)

            center = baseline[baseline.shape[0] // 2, baseline.shape[1] // 2]
            self.assertGreater(center[0], 5, "far red ghost did not contribute at the overlap")
            self.assertGreater(center[2], 5, "near blue ghost did not contribute at the overlap")

            for label, frame in (
                ("OpenGL 3.3 fallback", fallback),
                ("shadow pass", with_shadows),
                ("edge pass", with_edges),
            ):
                delta = np.abs(baseline.astype(np.int16) - frame.astype(np.int16))
                self.assertLess(np.mean(delta), 0.5, f"{label} changed the translucent render")
        finally:
            viewer.close()

    def test_layer_style_propagates_to_capsules_and_gaussian_shapes(self):
        """Propagate tint, opacity, and visibility to capsule parts and Gaussian splats."""
        viewer = _make_headless_viewer_gl_or_skip(self)

        try:
            capsule_builder = newton.ModelBuilder()
            capsule_body = capsule_builder.add_body()
            capsule_builder.add_shape_capsule(capsule_body, radius=0.2, half_height=0.4)
            capsule_model = capsule_builder.finalize(device=viewer.device)

            gaussian_builder = newton.ModelBuilder()
            gaussian = newton.Gaussian(positions=np.array(((0.0, 0.0, 0.0), (0.2, 0.0, 0.0)), dtype=np.float32))
            gaussian_builder.add_shape_gaussian(body=-1, gaussian=gaussian)
            gaussian_model = gaussian_builder.finalize(device=viewer.device)

            models = {"capsule": capsule_model, "gaussian": gaussian_model}
            style = newton.viewer.LayerRenderStyle(color=(0.2, 0.6, 0.9), opacity=0.35)
            for layer_id, model in models.items():
                viewer.activate(layer_id)
                viewer.set_model(model)
                if layer_id == "gaussian":
                    viewer.show_gaussians = True
                viewer.set_layer_render_style(layer_id, style)
                viewer.log_state(model.state())

            capsule_objects = [
                obj
                for name, obj in viewer.objects.items()
                if isinstance(obj, MeshInstancerGL) and "/layers/capsule/" in name and "/capsule_" in name
            ]
            self.assertEqual(len(capsule_objects), 2)
            for obj in capsule_objects:
                expected = np.broadcast_to((0.2, 0.6, 0.9), (obj.active_instances, 3))
                np.testing.assert_allclose(obj._instance_styles[: obj.active_instances, :3], expected)
                np.testing.assert_allclose(
                    obj._instance_styles[: obj.active_instances, 3], np.full(obj.active_instances, 0.35)
                )

            gaussian_objects = [
                obj
                for name, obj in viewer.objects.items()
                if isinstance(obj, MeshInstancerGL) and "/layers/gaussian/model/gaussians/" in name
            ]
            self.assertEqual(len(gaussian_objects), 1)
            gaussian_obj = gaussian_objects[0]
            expected = np.broadcast_to((0.2, 0.6, 0.9), (gaussian_obj.active_instances, 3))
            np.testing.assert_allclose(gaussian_obj._instance_styles[: gaussian_obj.active_instances, :3], expected)
            np.testing.assert_allclose(
                gaussian_obj._instance_styles[: gaussian_obj.active_instances, 3],
                np.full(gaussian_obj.active_instances, 0.35),
            )

            for layer_id, model in models.items():
                viewer.set_layer_shape_visibility(layer_id, (False,))
                viewer.activate(layer_id)
                viewer.log_state(model.state())
            for obj in (*capsule_objects, gaussian_obj):
                np.testing.assert_allclose(
                    obj._instance_styles[: obj.active_instances, 3], np.zeros(obj.active_instances)
                )
        finally:
            viewer.close()

    def test_layer_style_propagates_to_lazy_sdf_isomeshes(self):
        """Apply layer tint and opacity when SDF collision isomeshes appear lazily."""
        viewer = _make_headless_viewer_gl_or_skip(self)

        try:
            if not viewer.device.is_cuda:
                self.skipTest("Texture SDF construction requires CUDA")

            builder = newton.ModelBuilder()
            cfg = newton.ModelBuilder.ShapeConfig()
            cfg.sdf_max_resolution = 16
            cfg.is_hydroelastic = True
            body = builder.add_body()
            builder.add_shape_box(body, hx=0.3, hy=0.25, hz=0.2, cfg=cfg)
            model = builder.finalize(device=viewer.device)

            viewer.activate("sdf")
            viewer.set_model(model)
            viewer.show_collision = True
            viewer.set_layer_render_style("sdf", newton.viewer.LayerRenderStyle(color=(0.9, 0.4, 0.1), opacity=0.4))
            viewer.log_state(model.state())

            self.assertTrue(viewer._sdf_isomesh_instances)
            sdf_objects = [
                viewer.objects[batch.name]
                for batch in viewer._sdf_isomesh_instances.values()
                if batch.name in viewer.objects
            ]
            self.assertTrue(sdf_objects)
            for obj in sdf_objects:
                expected = np.broadcast_to((0.9, 0.4, 0.1), (obj.active_instances, 3))
                np.testing.assert_allclose(obj._instance_styles[: obj.active_instances, :3], expected)
                np.testing.assert_allclose(
                    obj._instance_styles[: obj.active_instances, 3], np.full(obj.active_instances, 0.4)
                )

            viewer.set_layer_shape_visibility("sdf", (False,))
            for obj in sdf_objects:
                np.testing.assert_allclose(
                    obj._instance_styles[: obj.active_instances, 3], np.zeros(obj.active_instances)
                )
        finally:
            viewer.close()

    def test_large_quad_is_stable_under_parallel_camera_translation(self):
        """Keep a large uniform quad stable while translating the camera parallel to it."""
        viewer = _make_headless_viewer_gl_or_skip(self, width=400, height=300)

        try:
            viewer.renderer.draw_sky = False
            viewer.renderer.sky_upper = (0.0, 0.0, 0.0)
            viewer.renderer.sky_lower = (0.0, 0.0, 0.0)
            viewer.renderer.specular_scale = 0.0
            viewer.renderer.spotlight_enabled = False

            points = wp.array(
                [(-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.5, 0.5, 0.0), (-0.5, 0.5, 0.0)],
                dtype=wp.vec3,
                device=viewer.device,
            )
            indices = wp.array([0, 1, 2, 0, 2, 3], dtype=wp.int32, device=viewer.device)
            normals = wp.array([(0.0, 0.0, 1.0)] * 4, dtype=wp.vec3, device=viewer.device)
            # Deliberately offset the giant triangle pair from the camera. This mirrors
            # imported USD ground assets and exposes clip-depth interpolation error that
            # a perfectly centered quad can accidentally hide.
            xforms = wp.array(
                [wp.transform((-850000.0, 850000.0, 0.0), wp.quat_identity())],
                dtype=wp.transform,
                device=viewer.device,
            )
            scales = wp.array([(2.0e8, 2.0e8, 1.0)], dtype=wp.vec3, device=viewer.device)
            colors = wp.array([(0.7, 0.7, 0.7)], dtype=wp.vec3, device=viewer.device)
            materials = wp.array([(0.5, 0.0, 0.0, 0.0)], dtype=wp.vec4, device=viewer.device)
            viewer.log_mesh("/test/quad", points, indices, normals, backface_culling=False)
            viewer.log_instances("/test/ground", "/test/quad", xforms, scales, colors, materials)
            viewer.log_geo("/test/box", newton.GeoType.BOX, (0.6, 0.6, 1.5), 0.0, True, hidden=True)
            target_colors = wp.array(
                [(0.9, 0.1, 0.1), (0.1, 0.9, 0.1), (0.1, 0.1, 0.9)], dtype=wp.vec3, device=viewer.device
            )
            target_materials = wp.array([(0.5, 0.0, 0.0, 0.0)] * 3, dtype=wp.vec4, device=viewer.device)
            target_scales = wp.array([(1.0, 1.0, 1.0)] * 3, dtype=wp.vec3, device=viewer.device)

            frames_by_render_mode = {}
            for draw_shadows, draw_edges in ((False, False), (True, False), (False, True)):
                viewer.renderer.draw_shadows = draw_shadows
                viewer.renderer.draw_edges = draw_edges
                viewer.renderer.diffuse_scale = 1.0
                frames = []
                for frame_index, offset in enumerate((0.0, 1.0)):
                    target_xforms = wp.array(
                        [
                            wp.transform((offset, -1.0, 3.0), wp.quat_identity()),
                            wp.transform((offset, 0.0, 3.0), wp.quat_identity()),
                            wp.transform((offset, 1.0, 3.0), wp.quat_identity()),
                        ],
                        dtype=wp.transform,
                        device=viewer.device,
                    )
                    viewer.log_instances(
                        "/test/targets",
                        "/test/box",
                        target_xforms,
                        target_scales,
                        target_colors,
                        target_materials,
                    )
                    viewer.set_camera(wp.vec3(offset + 8.0, -8.0, 6.0), pitch=0.0, yaw=0.0)
                    viewer.camera.look_at((offset, 0.0, 0.0))
                    for _ in range(2):
                        viewer.begin_frame(float(frame_index))
                        viewer.end_frame()
                    frames.append(viewer.get_frame().numpy().copy())

                delta = np.abs(frames[0].astype(np.int16) - frames[1].astype(np.int16))
                changed_pixel_fraction = np.mean(np.any(delta > 2, axis=-1))
                mean_absolute_delta = np.mean(delta)
                message_suffix = f" with draw_shadows={draw_shadows}, draw_edges={draw_edges}"
                self.assertLess(
                    changed_pixel_fraction,
                    0.01,
                    f"camera translation changed {changed_pixel_fraction:.2%} of pixels{message_suffix}",
                )
                self.assertLess(
                    mean_absolute_delta,
                    2.0,
                    f"mean absolute pixel delta was {mean_absolute_delta:.3f}{message_suffix}",
                )
                frames_by_render_mode[draw_shadows, draw_edges] = frames

            shadow_delta = np.abs(
                frames_by_render_mode[True, False][0].astype(np.int16)
                - frames_by_render_mode[False, False][0].astype(np.int16)
            )
            shadow_changed_fraction = np.mean(np.any(shadow_delta > 2, axis=-1))
            self.assertGreater(shadow_changed_fraction, 0.001, "the target boxes cast no visible shadows")
            self.assertLess(
                shadow_changed_fraction,
                0.1,
                f"shadows changed {shadow_changed_fraction:.2%} of the frame instead of remaining local",
            )
            edge_delta = np.abs(
                frames_by_render_mode[False, True][0].astype(np.int16)
                - frames_by_render_mode[False, False][0].astype(np.int16)
            )
            edge_changed_fraction = np.mean(np.any(edge_delta > 2, axis=-1))
            self.assertGreater(edge_changed_fraction, 0.001, "the target boxes have no visible edge overlay")

            # Render the same quad from close range at a large world coordinate.
            # Render style is constant per instance and must not be perspective-
            # interpolated: doing so produces striped/discarded fragments on the
            # very large triangles used by imported ground and table meshes.
            large_origin = (10_000_000.0, 10_000_000.0, 0.0)
            viewer.log_instances(
                "/test/ground",
                "/test/quad",
                wp.array(
                    [wp.transform(large_origin, wp.quat_identity())],
                    dtype=wp.transform,
                    device=viewer.device,
                ),
                scales,
                colors,
                materials,
            )
            viewer.renderer.draw_shadows = False
            viewer.renderer.draw_edges = False
            viewer.set_camera(wp.vec3(10_000_010.0, 9_999_990.0, 10.0), pitch=0.0, yaw=0.0)
            viewer.camera.look_at(large_origin)
            for frame_index in range(2):
                viewer.begin_frame(float(frame_index))
                viewer.end_frame()
            close_frame = viewer.get_frame().numpy()
            filled_fraction = np.mean(np.any(close_frame > 8, axis=-1))
            self.assertGreater(
                filled_fraction,
                0.99,
                f"constant instance style left {1.0 - filled_fraction:.2%} of the large surface unrendered",
            )
        finally:
            viewer.close()

    def test_headless_capture_main_image_frame(self):
        """Verify get_frame captures a main image rendered headlessly."""
        viewer = _make_headless_viewer_gl_or_skip(self)

        try:
            width = viewer.renderer._screen_width
            height = viewer.renderer._screen_height
            image = np.empty((height, width, 3), dtype=np.uint8)
            image[..., 0] = np.arange(width, dtype=np.uint8)
            image[..., 1] = np.arange(height, dtype=np.uint8)[:, None]
            image[..., 2] = 191
            regular = np.zeros((2, 3, 5, 4), dtype=np.uint8)

            viewer.begin_frame(0.0)
            viewer.log_image("color", regular)
            viewer.log_image("color", image, fullscreen=True)
            viewer.end_frame()

            regular_texture = viewer._image_logger.get_texture("color")
            fullscreen_texture = viewer._image_logger.get_texture("color", fullscreen=True)

            self.assertIsNotNone(regular_texture)
            self.assertIsNotNone(fullscreen_texture)
            self.assertEqual(regular_texture[1:], (10, 3))
            self.assertEqual(fullscreen_texture[1:], (width, height))
            np.testing.assert_array_equal(viewer.get_frame().numpy(), image)
        finally:
            viewer.close()

    def test_viewer_constructor_known_backend_error_skips(self):
        """Verify unavailable display/backend errors skip GL-dependent coverage."""
        unavailable_error = type(
            "ConfigException",
            (Exception,),
            {"__module__": "pyglet.gl"},
        )

        with (
            mock.patch.object(newton.viewer, "ViewerGL", side_effect=unavailable_error("no GL config")),
            self.assertRaises(unittest.SkipTest),
        ):
            _make_headless_viewer_gl_or_skip(self)

    def test_viewer_constructor_pyglet_backend_display_error_skips(self):
        """Verify pyglet backend display errors skip GL-dependent coverage."""
        unavailable_error = type(
            "NoSuchDisplayException",
            (Exception,),
            {"__module__": "pyglet.display.xlib"},
        )

        with (
            mock.patch.object(newton.viewer, "ViewerGL", side_effect=unavailable_error("no display")),
            self.assertRaises(unittest.SkipTest),
        ):
            _make_headless_viewer_gl_or_skip(self)

    def test_viewer_constructor_pyglet_missing_function_error_skips(self):
        """Verify pyglet missing GL function errors skip GL-dependent coverage."""
        unavailable_error = type(
            "MissingFunctionException",
            (Exception,),
            {"__module__": "pyglet.gl.lib"},
        )

        with (
            mock.patch.object(newton.viewer, "ViewerGL", side_effect=unavailable_error("glCreateShader unavailable")),
            self.assertRaises(unittest.SkipTest),
        ):
            _make_headless_viewer_gl_or_skip(self)

    def test_viewer_constructor_unexpected_error_raises(self):
        """Verify unexpected ViewerGL constructor errors fail the test."""
        with (
            mock.patch.object(newton.viewer, "ViewerGL", side_effect=RuntimeError("initialization regression")),
            self.assertRaisesRegex(RuntimeError, "initialization regression"),
        ):
            _make_headless_viewer_gl_or_skip(self)

    def test_cpu_viewer_uses_host_pbo_readback(self):
        """Verify CPU get_frame uses host PBO readback without CUDA interop."""
        pixels = np.array(
            [
                [10, 11, 12],
                [20, 21, 22],
                [30, 31, 32],
                [40, 41, 42],
            ],
            dtype=np.uint8,
        ).reshape(-1)
        fake_gl = _FakeGL(pixels)
        viewer = ViewerGL.__new__(ViewerGL)
        viewer.device = wp.get_device("cpu")
        viewer.renderer = SimpleNamespace(
            _screen_width=2,
            _screen_height=2,
            _frame_fbo=3,
        )
        viewer.gui = None
        viewer._pbo = None
        viewer._wp_pbo = None
        viewer._pbo_host_buffer = None

        with (
            mock.patch.object(RendererGL, "gl", fake_gl),
            mock.patch.object(
                wp,
                "RegisteredGLBuffer",
                side_effect=AssertionError("CPU readback must not use CUDA-GL interop"),
            ),
        ):
            frame = viewer.get_frame()

        np.testing.assert_array_equal(
            frame.numpy(),
            np.array(
                [
                    [[30, 31, 32], [40, 41, 42]],
                    [[10, 11, 12], [20, 21, 22]],
                ],
                dtype=np.uint8,
            ),
        )
        self.assertEqual(frame.device, wp.get_device("cpu"))
        self.assertEqual(fake_gl.readback_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
