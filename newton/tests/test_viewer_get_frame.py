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
from newton._src.viewer.gl.opengl import RendererGL
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


def _make_box_model(device: str | wp.Device):
    builder = newton.ModelBuilder()
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.25, hy=0.25, hz=0.25, color=(1.0, 0.0, 0.0))
    return builder.finalize(device=device)


def _configure_unlit_textured_viewer(viewer) -> None:
    renderer = viewer.renderer
    renderer.draw_sky = False
    renderer.sky_upper = (0.0, 0.0, 0.0)
    renderer.sky_lower = (0.0, 0.0, 0.0)
    renderer.draw_shadows = False
    renderer.diffuse_scale = 0.0
    renderer.specular_scale = 0.0
    renderer.spotlight_enabled = False
    renderer.ambient_sky = (1.0, 1.0, 1.0)
    renderer.ambient_ground = (1.0, 1.0, 1.0)
    renderer.exposure = 1.0


def _capture_viewer_frame(viewer) -> np.ndarray:
    for _ in range(2):
        viewer.begin_frame(0.0)
        viewer.end_frame()
    return viewer.get_frame().numpy()


def _make_asymmetric_texture() -> np.ndarray:
    y, x = np.mgrid[:72, :96]
    return np.stack(
        (
            32 + 192 * ((x // 12 + 2 * (y // 18)) % 2),
            24 + 208 * ((y // 9) % 2),
            16 + ((5 * x + 3 * y) % 60) * 4,
        ),
        axis=-1,
    ).astype(np.uint8)


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

    def test_world_projected_texture_keeps_physical_scale_on_large_quads(self):
        """Keep projected texture scale and position independent of mesh and camera motion."""
        viewer = _make_headless_viewer_gl_or_skip(self, width=400, height=300)

        try:
            _configure_unlit_textured_viewer(viewer)

            points = wp.array(
                [(-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.5, 0.5, 0.0), (-0.5, 0.5, 0.0)],
                dtype=wp.vec3,
                device=viewer.device,
            )
            indices = wp.array([0, 1, 2, 0, 2, 3], dtype=wp.int32, device=viewer.device)
            normals = wp.array([(0.0, 0.0, 1.0)] * 4, dtype=wp.vec3, device=viewer.device)
            uvs = wp.array([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], dtype=wp.vec2, device=viewer.device)
            texture = np.full((64, 64, 3), (20, 40, 70), dtype=np.uint8)
            texture[:5, :] = 240
            texture[:, :5] = 240
            colors = wp.array([(1.0, 1.0, 1.0)], dtype=wp.vec3, device=viewer.device)
            materials = wp.array([(0.5, 0.0, 0.0, 1.0)], dtype=wp.vec4, device=viewer.device)

            def render(
                size: float, projection: newton.Mesh.TextureProjection, camera_offset: float = 0.0
            ) -> np.ndarray:
                viewer.set_camera(wp.vec3(8.0 + camera_offset, -8.0, 6.0), pitch=0.0, yaw=0.0)
                viewer.camera.look_at((camera_offset, 0.0, 0.0))
                viewer.log_mesh(
                    "/test/projected_quad",
                    points,
                    indices,
                    normals,
                    uvs,
                    texture,
                    hidden=True,
                    backface_culling=False,
                    texture_scale=(0.5, 0.5),
                    texture_projection=projection,
                )
                xforms = wp.array(
                    [wp.transform((0.13 + camera_offset, 0.27, 0.0), wp.quat_identity())],
                    dtype=wp.transform,
                    device=viewer.device,
                )
                scales = wp.array([(size, size, 1.0)], dtype=wp.vec3, device=viewer.device)
                viewer.log_instances(
                    "/test/projected_quad_instance",
                    "/test/projected_quad",
                    xforms,
                    scales,
                    colors,
                    materials,
                )
                return _capture_viewer_frame(viewer)

            def edge_density(frame: np.ndarray) -> float:
                gray = frame.astype(np.float32).mean(axis=2)
                roi = gray[160:290, 20:380]
                return float(np.mean(np.abs(np.diff(roi, axis=1)) > 15.0))

            projected_frames = [render(size, newton.Mesh.TextureProjection.WORLD) for size in (2.0e2, 2.0e3)]
            projected_densities = [edge_density(frame) for frame in projected_frames]
            for size, density in zip((2.0e2, 2.0e3), projected_densities, strict=True):
                self.assertGreater(density, 0.05, f"projected texture did not tile on a {size:g} m quad")
            self.assertLess(
                abs(projected_densities[0] - projected_densities[1]),
                0.01,
                "changing quad dimensions changed the projected texture's physical density",
            )

            stretched_density = edge_density(render(2.0e3, newton.Mesh.TextureProjection.UV))
            self.assertLess(stretched_density, 0.005, "the control UV mapping unexpectedly tiled the large quad")

            shifted_frame = render(2.0e3, newton.Mesh.TextureProjection.WORLD, camera_offset=0.75)
            texture_motion = np.abs(projected_frames[1].astype(np.int16) - shifted_frame.astype(np.int16))
            object_frame = render(2.0e3, newton.Mesh.TextureProjection.OBJECT)
            shifted_object_frame = render(2.0e3, newton.Mesh.TextureProjection.OBJECT, camera_offset=0.75)
            object_texture_motion = np.abs(object_frame.astype(np.int16) - shifted_object_frame.astype(np.int16))
            world_motion = texture_motion[160:290, 20:380].mean()
            object_motion = object_texture_motion[160:290, 20:380].mean()
            self.assertGreater(
                world_motion,
                10.0,
                "world-projected texture moved with the camera instead of remaining anchored",
            )
            self.assertGreater(world_motion, object_motion + 10.0, "WORLD projection behaved like OBJECT projection")
        finally:
            viewer.close()

    def test_texture_transform_matches_omnipbr_coordinates(self):
        """Match OmniPBR rotation, scale, and translation order."""
        viewer = _make_headless_viewer_gl_or_skip(self, width=320, height=240)

        try:
            _configure_unlit_textured_viewer(viewer)
            viewer.set_camera(wp.vec3(0.0, -3.0, 2.5), pitch=0.0, yaw=0.0)
            viewer.camera.look_at((0.0, 0.0, 0.0))

            points = np.array(
                [(-1.0, -0.8, 0.0), (1.0, -0.8, 0.0), (1.0, 0.8, 0.0), (-1.0, 0.8, 0.0)],
                dtype=np.float32,
            )
            indices = wp.array([0, 1, 2, 0, 2, 3], dtype=wp.int32, device=viewer.device)
            normals = wp.array([(0.0, 0.0, 1.0)] * 4, dtype=wp.vec3, device=viewer.device)
            source_uvs = np.array([(0.08, 0.13), (1.21, 0.19), (1.14, 1.08), (0.03, 1.02)], dtype=np.float32)
            texture = _make_asymmetric_texture()
            xforms = wp.array([wp.transform_identity()], dtype=wp.transform, device=viewer.device)
            scales = wp.array([(1.0, 1.0, 1.0)], dtype=wp.vec3, device=viewer.device)
            colors = wp.array([(1.0, 1.0, 1.0)], dtype=wp.vec3, device=viewer.device)
            materials = wp.array([(0.5, 0.0, 0.0, 1.0)], dtype=wp.vec4, device=viewer.device)

            def render(uvs: np.ndarray, **mapping) -> np.ndarray:
                viewer.log_mesh(
                    "/test/texture_transform",
                    wp.array(points, dtype=wp.vec3, device=viewer.device),
                    indices,
                    normals,
                    wp.array(uvs, dtype=wp.vec2, device=viewer.device),
                    texture,
                    hidden=True,
                    backface_culling=False,
                    **mapping,
                )
                viewer.log_instances(
                    "/test/texture_transform_instance",
                    "/test/texture_transform",
                    xforms,
                    scales,
                    colors,
                    materials,
                )
                return _capture_viewer_frame(viewer)

            scale = np.array((0.7, 1.3), dtype=np.float32)
            translate = np.array((0.13, 0.27), dtype=np.float32)
            rotate = 37.0
            actual = render(source_uvs, texture_scale=scale, texture_translate=translate, texture_rotate=rotate)

            angle = np.deg2rad(rotate)
            cosine, sine = np.cos(angle), np.sin(angle)
            rotated = np.column_stack(
                (
                    cosine * source_uvs[:, 0] + sine * source_uvs[:, 1],
                    -sine * source_uvs[:, 0] + cosine * source_uvs[:, 1],
                )
            )
            expected_uvs = rotated * scale + translate
            expected = render(expected_uvs.astype(np.float32))

            visible = np.any(expected > 8, axis=2)
            self.assertGreater(np.std(expected[visible]), 30.0, "the texture control was not visibly asymmetric")
            self.assertLess(np.abs(actual.astype(np.int16) - expected.astype(np.int16))[visible].mean(), 2.0)
        finally:
            viewer.close()

    def test_cubic_projection_preserves_signed_face_orientation(self):
        """Preserve the OmniPBR orientation of all six cubic faces."""
        viewer = _make_headless_viewer_gl_or_skip(self, width=320, height=240)

        try:
            _configure_unlit_textured_viewer(viewer)
            viewer.set_camera(wp.vec3(3.0, -4.0, 2.5), pitch=0.0, yaw=0.0)
            viewer.camera.look_at((0.0, 0.0, 0.0))

            planes = {
                "x": np.array([(0.0, -0.85, -0.65), (0.0, 0.85, -0.65), (0.0, 0.85, 0.65), (0.0, -0.85, 0.65)]),
                "y": np.array([(-0.85, 0.0, -0.65), (0.85, 0.0, -0.65), (0.85, 0.0, 0.65), (-0.85, 0.0, 0.65)]),
                "z": np.array([(-0.85, -0.65, 0.0), (0.85, -0.65, 0.0), (0.85, 0.65, 0.0), (-0.85, 0.65, 0.0)]),
            }
            indices = wp.array([0, 1, 2, 0, 2, 3], dtype=wp.int32, device=viewer.device)
            texture = _make_asymmetric_texture()
            xforms = wp.array([wp.transform_identity()], dtype=wp.transform, device=viewer.device)
            scales = wp.array([(1.0, 1.0, 1.0)], dtype=wp.vec3, device=viewer.device)
            colors = wp.array([(1.0, 1.0, 1.0)], dtype=wp.vec3, device=viewer.device)
            materials = wp.array([(0.5, 0.0, 0.0, 1.0)], dtype=wp.vec4, device=viewer.device)

            for axis, sign in (("x", 1), ("x", -1), ("y", 1), ("y", -1), ("z", 1), ("z", -1)):
                with self.subTest(axis=axis, sign=sign):
                    points = planes[axis].astype(np.float32)
                    normal = np.zeros(3, dtype=np.float32)
                    normal["xyz".index(axis)] = sign
                    normals = np.tile(normal, (4, 1))

                    if axis == "x":
                        expected_uvs = np.column_stack((sign * points[:, 1], points[:, 2]))
                    elif axis == "y":
                        expected_uvs = np.column_stack((-sign * points[:, 0], points[:, 2]))
                    else:
                        expected_uvs = np.column_stack((sign * points[:, 0], points[:, 1]))

                    mesh_args = (
                        "/test/cubic_projection",
                        wp.array(points, dtype=wp.vec3, device=viewer.device),
                        indices,
                        wp.array(normals, dtype=wp.vec3, device=viewer.device),
                    )
                    viewer.log_mesh(
                        *mesh_args,
                        wp.zeros(4, dtype=wp.vec2, device=viewer.device),
                        texture,
                        hidden=True,
                        backface_culling=False,
                        texture_scale=(0.43, 0.61),
                        texture_translate=(0.2, 0.15),
                        texture_projection=newton.Mesh.TextureProjection.OBJECT,
                    )
                    viewer.log_instances(
                        "/test/cubic_projection_instance",
                        "/test/cubic_projection",
                        xforms,
                        scales,
                        colors,
                        materials,
                    )
                    actual = _capture_viewer_frame(viewer)
                    viewer.log_mesh(
                        *mesh_args,
                        wp.array(expected_uvs.astype(np.float32), dtype=wp.vec2, device=viewer.device),
                        texture,
                        hidden=True,
                        backface_culling=False,
                        texture_scale=(0.43, 0.61),
                        texture_translate=(0.2, 0.15),
                    )
                    expected = _capture_viewer_frame(viewer)

                    visible = np.any(expected > 8, axis=2)
                    self.assertGreater(
                        np.std(expected[visible]), 30.0, "the texture control was not visibly asymmetric"
                    )
                    self.assertLess(np.abs(actual.astype(np.int16) - expected.astype(np.int16))[visible].mean(), 2.0)
        finally:
            viewer.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
