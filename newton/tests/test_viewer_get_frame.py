# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import ctypes
import sys
import unittest
from importlib import import_module
from types import ModuleType, SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

import newton
import newton.viewer
from newton._src.viewer.gl.opengl import RendererGL
from newton._src.viewer.kernels import compact
from newton._src.viewer.viewer_gl import ViewerGL
from newton._src.viewer.viewer_joint_selection import _JointDrawList
from newton._src.viewer.viewer_navigation import SelectionNavigation


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


def _make_headless_viewer_gl_or_skip(test: unittest.TestCase, *, width: int = 64, height: int = 48):
    _viewer_gl_unavailable_error_types(test)

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


class TestRenderingPauseGL(unittest.TestCase):
    def test_freeze_capture_resize_and_resume(self):
        """Keep an image stable through logging, resize, and UI-driven toggles."""
        with wp.ScopedDevice("cpu"):
            viewer = _make_headless_viewer_gl_or_skip(self)
        self.addCleanup(viewer.close)
        model = _make_box_model("cpu")
        viewer.set_model(model)
        state = model.state()
        before = np.zeros((48, 64, 3), dtype=np.uint8)
        before[:24, :, 0] = 255
        before[24:, :, 1] = 255
        after = np.zeros_like(before)
        after[:, :, 2] = 255

        viewer.set_rendering_paused(True)
        viewer.begin_frame(0.0)
        viewer.end_frame()
        with self.assertRaisesRegex(RuntimeError, "displayed frame"):
            viewer.get_frame()
        viewer.set_rendering_paused(False)
        viewer.begin_frame(1.0)
        viewer.log_image("sensor", before, fullscreen=True)
        viewer.end_frame()
        reference = viewer.get_frame().numpy().copy()
        np.testing.assert_array_equal(reference, before)

        gui = mock.Mock(workspace_enabled=False, workspace=None, _frame_camera_pending=False)
        gui.prepare_frame.side_effect = lambda: viewer.set_rendering_paused(True)
        viewer.gui = gui
        viewer.begin_frame(2.0)
        viewer.log_state(state)
        with mock.patch.object(viewer.renderer, "render", wraps=viewer.renderer.render) as render:
            viewer.end_frame()
            render.assert_not_called()
        np.testing.assert_array_equal(viewer.get_frame().numpy(), reference)
        np.testing.assert_array_equal(viewer._displayed_frame.pixels()[::-1, :, :3], reference)
        gui.render_prepared_frame.assert_called_once()
        viewer.gui = None

        viewer.renderer.window.set_size(128, 96)
        viewer.renderer._on_window_resize(128, 96)
        viewer.begin_frame(3.0)
        viewer.log_image("sensor", after, fullscreen=True)
        viewer.end_frame()
        resized = viewer.get_frame().numpy()
        self.assertEqual(resized.shape, (96, 128, 3))
        np.testing.assert_array_equal(resized[10, 10], [255, 0, 0])
        np.testing.assert_array_equal(resized[80, 10], [0, 255, 0])

        viewer.set_rendering_paused(False)
        viewer.begin_frame(4.0)
        viewer.log_image("sensor", after, fullscreen=True)
        viewer.end_frame()
        np.testing.assert_array_equal(viewer.get_frame().numpy()[10, 10], [0, 0, 255])
        viewer.set_rendering_paused(True)
        viewer.clear_model()
        viewer.begin_frame(5.0)
        viewer.end_frame()
        self.assertTrue(viewer.is_rendering_paused())
        with self.assertRaises(RuntimeError):
            viewer.get_frame()


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
    def test_contact_label_leaders_ignore_pixel_rounding(self) -> None:
        """Only displaced labels add leader geometry at integer or fractional DPI."""
        with wp.ScopedDevice("cpu"):
            viewer = _make_headless_viewer_gl_or_skip(self, width=400, height=300)
        self.addCleanup(viewer.close)
        viewer.set_camera(pos=(1.0, 1.0, 1.0), pitch=0.0, yaw=0.0)
        viewer.camera.look_at((0.0, 0.0, 0.0))
        point = ((0.013, 0.017, 0.0),)
        for scale in (1.0, 1.5, 2.0):
            viewer.renderer._contact_label_scale = scale
            for labels in ([(point, "1", False)], [(point, "1", False), (point, "2", True)]):
                viewer.renderer._contact_labels = labels
                viewer.begin_frame(0.0)
                viewer.end_frame()
                shader = viewer.renderer._contact_label_shader
                self.assertEqual(len(shader.layout), len(labels))
                if len(labels) == 1:
                    self.assertEqual(shader.count, 6)
                else:
                    self.assertGreater(shader.count, 12)

    def test_shape_graph_capture_requires_buffer_reuse(self) -> None:
        """One-shot States avoid capture; recurring buffers replay with current transforms."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA graph capture requires CUDA")
        with wp.ScopedDevice("cuda:0"):
            viewer = _make_headless_viewer_gl_or_skip(self)
        self.addCleanup(viewer.close)
        model = _make_box_model("cuda:0")
        viewer.set_model(model)
        states = [model.state() for _ in range(8)]
        with mock.patch.object(wp, "ScopedCapture", wraps=wp.ScopedCapture) as capture:
            for index, state in enumerate(states):
                state.body_q.assign([[index, 0, 0, 0, 0, 0, 1]])
                viewer.log_state(state)
                np.testing.assert_allclose(viewer._packed_world_xforms.numpy()[0, :3], [index, 0, 0])
            self.assertEqual(capture.call_count, 0)
            for index in range(6):
                state = states[-1 - index % 2]
                state.body_q.assign([[10 + index, 0, 0, 0, 0, 0, 1]])
                viewer.log_state(state)
                np.testing.assert_allclose(viewer._packed_world_xforms.numpy()[0, :3], [10 + index, 0, 0])
            self.assertEqual(capture.call_count, 2)
            self.assertLessEqual(len(viewer._shape_prepare_graphs), 4)

    def test_point_prototypes_follow_model_device(self) -> None:
        """Reuse device-local point/splat meshes across CPU/CUDA model changes."""
        if not wp.is_cuda_available():
            self.skipTest("Device switching requires CUDA")
        with wp.ScopedDevice("cpu"):
            viewer = _make_headless_viewer_gl_or_skip(self)
        gaussian = newton.Gaussian(positions=np.array([[0.5, 0.0, 0.0]], dtype=np.float32))
        prototypes = {}
        try:
            for device in (wp.get_device("cpu"), wp.get_device("cuda:0"), wp.get_device("cpu")):
                viewer.set_model(_make_box_model(device))
                viewer.begin_frame(0.0)
                viewer.log_state(viewer.model.state())
                viewer.log_gaussian("splats", gaussian)
                viewer.end_frame()
                point = viewer.objects["/model/com"]
                splat = viewer.objects["splats"]
                self.assertEqual(point.device, device)
                self.assertEqual(splat.device, device)
                np.testing.assert_allclose(point.world_xforms.numpy()[0, 3], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
                key = str(device)
                if key in prototypes:
                    self.assertIs(point.mesh, prototypes[key][0])
                    self.assertIs(splat.mesh, prototypes[key][1])
                prototypes[key] = (point.mesh, splat.mesh)
        finally:
            viewer.close()
        self.assertFalse(viewer._point_meshes)
        self.assertFalse(viewer._gaussian_meshes)

    def test_nonworkspace_viewers_leave_optional_kernels_lazy(self) -> None:
        """Headless and legacy viewers compile only the features their scenes use."""
        _viewer_gl_unavailable_error_types(self)
        modules = [
            wp.get_module(import_module(name).__name__)
            for name in (
                "newton._src.geometry.sdf_texture",
                "newton._src.geometry.raycast",
                "newton._src.sim.articulation",
                "newton._src.viewer.wind",
                "newton._src.viewer.viewer_navigation",
                "newton._src.viewer.viewer_history",
            )
        ]
        initialize_renderer = RendererGL.__init__

        def initialize_offscreen(renderer: RendererGL, **kwargs: object) -> None:
            kwargs["headless"] = True
            initialize_renderer(renderer, **kwargs)

        for headless in (True, False):
            with self.subTest(headless=headless):
                for module in modules:
                    module.unload()
                try:
                    with wp.ScopedDevice("cpu"), mock.patch.object(RendererGL, "__init__", initialize_offscreen):
                        viewer = ViewerGL(width=64, height=48, headless=headless, legacy_gui=True)
                except Exception as exc:
                    if _is_viewer_gl_unavailable_error(self, exc):
                        self.skipTest(f"ViewerGL display/backend not available: {exc}")
                    raise
                try:
                    self.assertIsNone(viewer.renderer._selection_outline)
                    devices = [wp.get_device("cpu"), *wp.get_cuda_devices()[:1]]
                    for device in devices:
                        with self.subTest(device=device):
                            for module in modules:
                                self.assertFalse(module.execs, f"Eagerly loaded {module.name}")
                            viewer.set_model(_make_box_model(device))
                            for module in modules:
                                self.assertFalse(module.execs, f"Model binding loaded unused {module.name}")
                            # Lazy compilation must still support rendering and graph reuse.
                            state = viewer.model.state()
                            for frame in range(2):
                                viewer.begin_frame(float(frame))
                                viewer.log_state(state)
                                viewer.end_frame()
                            self.assertEqual(viewer.get_frame().shape[-1], 3)
                finally:
                    viewer.close()

    def test_gui_preloads_interaction_kernels(self) -> None:
        """First selection and particle filtering reuse startup-compiled CPU/CUDA modules."""
        try:
            __import__("imgui_bundle")
        except ImportError as exc:
            self.skipTest(f"Viewer UI dependencies not available: {exc}")
        _viewer_gl_unavailable_error_types(self)
        modules = [
            wp.get_module(import_module(f"newton._src.viewer.{name}").__name__)
            for name in (
                "viewer_navigation",
                "viewer_joint_selection",
                "viewer_deformables",
                "viewer_bvh",
                "viewer_history",
                "gui.collision",
                "gui.workspace",
                "kernels",
            )
        ]
        # Earlier tests may have launched these kernels. Discard only their
        # in-process executables so this test cannot pass because of test order.
        for module in modules:
            module.unload()
        initialize_renderer = RendererGL.__init__

        def initialize_offscreen(renderer: RendererGL, **kwargs: object) -> None:
            kwargs["headless"] = True
            initialize_renderer(renderer, **kwargs)

        try:
            with wp.ScopedDevice("cpu"), mock.patch.object(RendererGL, "__init__", initialize_offscreen):
                viewer = ViewerGL(width=800, height=600)
        except Exception as exc:
            if _is_viewer_gl_unavailable_error(self, exc):
                self.skipTest(f"ViewerGL display/backend not available: {exc}")
            raise
        self.addCleanup(viewer.close)
        self.assertTrue(viewer.gui.is_available)
        self.assertIsNone(viewer.gui.workspace)
        for name in (
            "_selection_outline",
            "_selection_surface_shader",
            "_joint_shader",
            "_contact_label_shader",
            "_physics_contact_shader",
            "_surface_grid_shader",
            "_physics_collision_shader",
            "_physics_shell_shader",
            "_oit_resolve_shader",
            "_shape_transparent_shader",
        ):
            self.assertIsNotNone(getattr(viewer.renderer, name, None), f"{name} was not preloaded")
        self.assertIsNone(viewer.renderer._selection_outline.size)
        self.assertIsNone(viewer.renderer._oit_fbo)

        devices = [wp.get_device("cpu"), *wp.get_cuda_devices()[:1]]
        for device in devices:
            with self.subTest(device=device):
                # CPU compilation must finish in the constructor, before a
                # model or the first GUI frame. A model switch warms its device.
                if device.is_cuda:
                    viewer.set_model(_make_box_model(device))
                for module in modules:
                    self.assertTrue(
                        any(executable.device == device for executable in module.execs.values()),
                        f"{module.name} was not preloaded on {device}",
                    )
                executables = [dict(module.execs) for module in modules]
                if device.is_cpu:
                    viewer.set_model(_make_box_model(device))
                # Exercise navigation independently of draw-buffer creation.
                viewer._last_state = viewer.model.state()
                navigation = SelectionNavigation(
                    SimpleNamespace(viewer=viewer, model=viewer.model, _selection_shapes=(0,))
                )
                self.assertAlmostEqual(navigation(wp.vec3(0.5, 0, 0), wp.vec3(0.4, 0, 0), reference=1.0), 0.0625)

                mask = wp.array([1, 0, 1], dtype=int, device=device)
                offsets = wp.array([0, 1, 1], dtype=int, device=device)
                for dtype in (float, wp.vec3):
                    values = [1.0, 2.0, 3.0] if dtype is float else [(1.0, 2.0, 3.0)] * 3
                    source = wp.array(values, dtype=dtype, device=device)
                    output = wp.empty(2, dtype=dtype, device=device)
                    wp.launch(compact, dim=3, inputs=[source, mask, offsets, output], device=device)
                    np.testing.assert_array_equal(output.numpy(), source.numpy()[[0, 2]])
                # Adding a generic overload on first use invalidates the whole
                # module, even if load_module had already been called at startup.
                for module, before in zip(modules, executables, strict=True):
                    self.assertEqual(module.execs, before, f"{module.name} reloaded on first interaction")

    def test_gui_mode_and_legacy_callbacks(self) -> None:
        """Render the default workspace and opt-in legacy panels with real ImGui callbacks."""
        try:
            from imgui_bundle import imgui
        except ImportError as exc:
            self.skipTest(f"Viewer UI dependencies not available: {exc}")
        _viewer_gl_unavailable_error_types(self)
        initialize_renderer = RendererGL.__init__

        def initialize_offscreen(renderer: RendererGL, **kwargs: object) -> None:
            # Exercise ViewerGL's windowed GUI selection using a real hidden GL
            # context, so the test works on CI without opening desktop windows.
            kwargs["headless"] = True
            initialize_renderer(renderer, **kwargs)

        model = _make_box_model("cpu")
        state = model.state()
        for legacy in (True, False):
            with self.subTest(legacy=legacy):
                options = {"legacy_gui": True} if legacy else {}
                try:
                    with wp.ScopedDevice("cpu"), mock.patch.object(RendererGL, "__init__", initialize_offscreen):
                        viewer = ViewerGL(width=800, height=600, **options)
                except Exception as exc:
                    if _is_viewer_gl_unavailable_error(self, exc):
                        self.skipTest(f"ViewerGL display/backend not available: {exc}")
                    raise
                image_texture = None
                try:
                    viewer.set_model(model)
                    gui = viewer.gui
                    self.assertTrue(gui.is_available)
                    from OpenGL import GL  # noqa: PLC0415 - optional GL dependency, after context creation

                    image_texture = int(GL.glGenTextures(1))
                    GL.glBindTexture(GL.GL_TEXTURE_2D, image_texture)
                    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
                    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
                    pixels = np.array(
                        [[[255, 0, 0, 255], [0, 255, 0, 255]], [[0, 0, 255, 255], [255, 255, 255, 255]]], dtype=np.uint8
                    )
                    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, 2, 2, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, pixels)
                    calls = []
                    for position in ("panel", "side", "stats", "free"):

                        def callback(
                            ui: ModuleType,
                            position: str = position,
                            calls: list[str] = calls,
                            texture: int = image_texture,
                        ) -> None:
                            ui.text(f"Custom {position} callback")
                            calls.append(position)
                            if position == "free":
                                # Exercise switches between font and user-image textures.
                                ui.get_foreground_draw_list().add_image(
                                    ui.ImTextureRef(texture), ui.ImVec2(16, 400), ui.ImVec2(80, 464)
                                )

                        viewer.register_ui_callback(callback, position=position)

                    def timeline_gui(ui: ModuleType, calls: list[str] = calls) -> None:
                        # User callbacks must not be filtered by their Python name.
                        ui.get_foreground_draw_list().add_text((16, 480), 0xFFFFFFFF, "Custom timeline")
                        calls.append("timeline_gui")

                    viewer.register_ui_callback(timeline_gui, position="free")
                    for frame in range(2):
                        viewer.begin_frame(frame / 60)
                        viewer.log_state(state)
                        viewer.end_frame()
                    self.assertFalse(gui.ui.section_errors)
                    self.assertGreater(gui.ui.imgui.get_draw_data().total_vtx_count, 0)
                    if legacy:
                        self.assertIsNone(gui.workspace)
                        self.assertIsNone(gui.console_capture)
                        self.assertIsNone(viewer.renderer._workspace_rect)
                        self.assertEqual(calls, ["panel", "side", "stats", "free", "timeline_gui"] * 2)
                        self.assertIsNotNone(
                            gui.ui.imgui.internal.find_window_by_name(f"Newton Viewer v{newton.__version__}")
                        )
                    else:
                        self.assertIsNotNone(gui.workspace)
                        self.assertIsNotNone(viewer.renderer._workspace_rect)
                        self.assertIn("free", calls)
                        self.assertIn("stats", calls)
                        self.assertIn("timeline_gui", calls)
                        self.assertFalse(gui.workspace.timeline_options["recording"])

                    # Compare actual clipped text/widgets with the upstream
                    # renderer, using identical draw data and an isolated target.
                    from imgui_bundle.python_backends.pyglet_backend import PygletProgrammablePipelineRenderer

                    previous_fbo = int(GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING))
                    texture = GL.glGenTextures(1)
                    framebuffer = GL.glGenFramebuffers(1)
                    try:
                        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
                        GL.glTexImage2D(
                            GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, 800, 600, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None
                        )
                        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, framebuffer)
                        GL.glFramebufferTexture2D(
                            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, texture, 0
                        )
                        self.assertEqual(GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER), GL.GL_FRAMEBUFFER_COMPLETE)
                        images = []
                        for render in (
                            lambda data, gui=gui: PygletProgrammablePipelineRenderer.render(gui.ui.impl, data),
                            gui.ui.impl.render,
                        ):
                            GL.glDisable(GL.GL_SCISSOR_TEST)
                            GL.glClearColor(0.1, 0.2, 0.3, 1.0)
                            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
                            render(imgui.get_draw_data())
                            images.append(GL.glReadPixels(0, 0, 800, 600, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE))
                        np.testing.assert_array_equal(
                            np.frombuffer(images[0], np.uint8), np.frombuffer(images[1], np.uint8)
                        )
                        image = np.frombuffer(images[1], np.uint8).reshape(600, 800, 4)
                        np.testing.assert_array_equal(image[600 - 416, 32], [255, 0, 0, 255])

                        # Rendering the UI must preserve independent alpha blend
                        # factors used by scene transparency passes.
                        GL.glBlendFuncSeparate(GL.GL_ONE, GL.GL_ZERO, GL.GL_ZERO, GL.GL_ONE)
                        gui.ui.render()
                        for name, value in (
                            (GL.GL_BLEND_SRC_RGB, GL.GL_ONE),
                            (GL.GL_BLEND_DST_RGB, GL.GL_ZERO),
                            (GL.GL_BLEND_SRC_ALPHA, GL.GL_ZERO),
                            (GL.GL_BLEND_DST_ALPHA, GL.GL_ONE),
                        ):
                            self.assertEqual(int(GL.glGetIntegerv(name)), value)

                        if not legacy:
                            # Reuse this context to check actual cached-icon
                            # pixels, including mid-frame font texture growth and DPI
                            # changes. Merely comparing two renderers could miss
                            # blank or incorrectly mapped icon textures.
                            atlas = gui.ui.icon_atlas
                            fonts = imgui.get_io().fonts
                            for scale in (1.0, 2.0, 1.5):
                                size = int(16 * scale)
                                gui.ui.begin_frame()
                                draw = imgui.get_foreground_draw_list()
                                vertices = draw.vtx_buffer.size()
                                for copy in range(3):
                                    if scale == 1.0 and copy == 1:
                                        previous_size = fonts.tex_data.width, fonts.tex_data.height
                                        rect = imgui.ImFontAtlasRect()
                                        fonts.add_custom_rect(*previous_size, rect)
                                        self.assertNotEqual(
                                            (fonts.tex_data.width, fonts.tex_data.height), previous_size
                                        )
                                    self.assertTrue(
                                        atlas.draw(
                                            draw,
                                            "body",
                                            (100 + 40 * copy, 100),
                                            size,
                                            scale=scale,
                                            color=imgui.get_color_u32((1.0, 0.5, 0.25, 1.0)),
                                            solid=True,
                                        )
                                    )
                                self.assertEqual(draw.vtx_buffer.size() - vertices, 12)
                                self.assertFalse(
                                    atlas.draw(draw, "unknown", (0, 0), size, scale=scale, color=0xFFFFFFFF)
                                )
                                gui.ui.end_frame()
                                GL.glDisable(GL.GL_SCISSOR_TEST)
                                GL.glClearColor(0, 0, 0, 1)
                                GL.glClear(GL.GL_COLOR_BUFFER_BIT)
                                gui.ui.render()
                                pixels = np.frombuffer(
                                    GL.glReadPixels(0, 0, 800, 600, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE), np.uint8
                                ).reshape(600, 800, 4)[::-1]
                                first = pixels[100 : 100 + size, 100 : 100 + size, :3]
                                self.assertGreater(np.count_nonzero(first[..., 0] > 128), size * size // 4)
                                self.assertGreater(first[..., 0].sum(), first[..., 1].sum())
                                self.assertGreater(first[..., 1].sum(), first[..., 2].sum())
                                for copy in (1, 2):
                                    np.testing.assert_array_equal(
                                        pixels[100 : 100 + size, 100 + 40 * copy : 100 + 40 * copy + size, :3], first
                                    )
                            # Clipped property labels should still batch as text,
                            # without a separate GPU scissor command per field.
                            # Check actual pixels as well as command count so a
                            # faster path cannot silently remove label clipping.
                            for label_width in (70, 150):
                                gui.ui.begin_frame()
                                imgui.set_next_window_pos((40, 40))
                                imgui.set_next_window_size((300, 350))
                                imgui.push_style_var(imgui.StyleVar_.window_padding, (0, 0))
                                imgui.push_style_var(imgui.StyleVar_.window_border_size, 0)
                                imgui.begin(
                                    "##label_batch",
                                    flags=imgui.WindowFlags_.no_decoration | imgui.WindowFlags_.no_background,
                                )
                                origin = imgui.get_cursor_screen_pos()
                                for _ in range(6):
                                    gui.workspace.widgets._fixed_label(
                                        "A very long property name for clipping", label_width
                                    )
                                draw = imgui.get_window_draw_list()
                                command_count = sum(command.elem_count > 0 for command in draw.cmd_buffer)
                                imgui.end()
                                imgui.pop_style_var(2)
                                gui.ui.end_frame()
                                GL.glDisable(GL.GL_SCISSOR_TEST)
                                GL.glClearColor(0, 0, 0, 1)
                                GL.glClear(GL.GL_COLOR_BUFFER_BIT)
                                gui.ui.render()
                                pixels = np.frombuffer(
                                    GL.glReadPixels(0, 0, 800, 600, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE), np.uint8
                                ).reshape(600, 800, 4)[::-1, :, :3]
                                x, y = int(origin.x), int(origin.y)
                                self.assertGreater(np.count_nonzero(pixels[y : y + 200, x : x + label_width]), 0)
                                self.assertEqual(np.count_nonzero(pixels[y : y + 200, x + label_width : x + 280]), 0)
                                self.assertEqual(command_count, 1)
                    finally:
                        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, previous_fbo)
                        GL.glDeleteFramebuffers(1, [framebuffer])
                        GL.glDeleteTextures([texture])

                    # Model switches clear old images but must keep the image
                    # presentation alive for the next example in both GUIs.
                    for reset in (False, True):
                        if reset:
                            viewer.clear_model()
                        for frame in range(3):
                            viewer.begin_frame(frame / 60)
                            viewer.log_image("sensor", np.full((16, 16, 3), 150, dtype=np.uint8))
                            viewer.end_frame()
                        texture_id, _, _ = viewer._image_logger.get_texture("sensor")
                        commands = (
                            command for draw_list in imgui.get_draw_data().cmd_lists for command in draw_list.cmd_buffer
                        )
                        self.assertTrue(any(command.get_tex_id() == texture_id for command in commands))
                finally:
                    if image_texture is not None:
                        GL.glDeleteTextures([image_texture])
                    icon_texture = viewer.gui.ui.icon_atlas._texture
                    icon_texture_id = icon_texture.id if icon_texture is not None else None
                    viewer.gui.ui.shutdown()
                    if icon_texture_id is not None:
                        self.assertFalse(GL.glIsTexture(icon_texture_id))
                    viewer.close()
                    # Do not leak GUI fonts/style/time into other ImGui tests.
                    imgui.destroy_context()

    def test_joint_stroke_outline_preserves_arc_color_and_occlusion(self) -> None:
        """Keep tilted angle arcs clear at different zooms without drawing through geometry."""
        with wp.ScopedDevice("cpu"):
            viewer = _make_headless_viewer_gl_or_skip(self, width=400, height=300)
        self.addCleanup(viewer.close)
        renderer = viewer.renderer
        renderer.draw_sky = renderer.draw_shadows = renderer.draw_grid = False
        renderer.sky_upper = (0.13, 0.13, 0.13)
        # Supply viewport metrics without starting an ImGui event loop. Both
        # triangulation and opaque/transparent rasterization use the real GL path.
        workspace = SimpleNamespace(viewer=viewer, widgets=SimpleNamespace(scale=1.0), rect=(0, 0, 400, 300))
        angles = np.linspace(0, np.pi * 1.6, 65)
        circle = np.column_stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)))
        for distance in (3.0, 30.0, 300.0):
            viewer.set_camera(wp.vec3(distance, -distance, distance), pitch=0.0, yaw=0.0)
            viewer.camera.look_at((0.0, 0.0, 0.0))
            for radius_pixels, thickness, alpha in ((42, 3.0, 1.0), (12, 1.5, 0.45)):
                with self.subTest(distance=distance, radius=radius_pixels, alpha=alpha):
                    frames = []
                    for halo in (False, True):
                        draw = _JointDrawList(workspace)
                        radius = np.linalg.norm(np.asarray(viewer.camera.pos)) * draw.radius_per_depth * radius_pixels
                        draw.lines(
                            (circle * radius)[None, ...], np.array([[1.0, 0.6, 0.2, alpha]]), thickness, halo=halo
                        )
                        draw.flush()
                        renderer.render(viewer.camera, {})
                        frames.append(viewer.get_frame().numpy().copy())
                    core = frames[0][..., 0] >= int(frames[0][..., 0].max()) - 2
                    self.assertGreater(np.count_nonzero(core), 10)
                    error = np.abs(frames[1][core].astype(int) - frames[0][core].astype(int))
                    self.assertLessEqual(int(error.max()), 2, "the outline painted over the arc")

        # Move a real box in front of the diagnostic. Its depth must still hide
        # the arc and its border; fixing self-overlap must not make an overlay.
        builder = newton.ModelBuilder()
        builder.add_shape_box(-1, xform=wp.transform((150.0, -150.0, 150.0), wp.quat_identity()), hx=70, hy=70, hz=70)
        model = builder.finalize(device="cpu")
        viewer.set_model(model)
        viewer.log_state(model.state())
        triangles = renderer._joint_triangles
        renderer._joint_triangles = None
        renderer.render(viewer.camera, viewer.objects)
        scene = viewer.get_frame().numpy().copy()
        renderer._joint_triangles = triangles
        renderer.render(viewer.camera, viewer.objects)
        np.testing.assert_array_equal(viewer.get_frame().numpy(), scene)

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
            self.assertIs(viewer.get_frame(output=target), target)

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

            # Wide contact connectors must retain their visible portion when
            # either witness crosses behind the eye. Compare against an
            # explicitly clipped segment using the same native framebuffer.
            viewer.clear_all_layers()
            viewer.set_model(newton.ModelBuilder().finalize(device="cpu"))
            viewer.show_ground_plane = False
            viewer.renderer.draw_sky = False
            viewer.camera.near = 0.1
            viewer.set_camera(pos=wp.vec3(0, -2, 1), pitch=0.0, yaw=0.0)
            viewer.camera.look_at((0, 0, 1))
            eye = np.asarray(viewer.camera.pos)
            forward, right = np.asarray(viewer.camera.get_front()), np.asarray(viewer.camera.get_right())
            start = eye + 2 * forward - 0.3 * right
            end = eye - 0.5 * forward + 0.2 * right
            clipped = start + (end - start) * ((2 - 0.100001) / 2.5)

            def draw_segment(
                a: tuple[float, float, float], b: tuple[float, float, float], *, arrows: bool = False
            ) -> np.ndarray:
                begins = wp.array([a], dtype=wp.vec3, device="cpu")
                ends = wp.array([b], dtype=wp.vec3, device="cpu")
                viewer.begin_frame(1.0)
                if arrows:
                    viewer.log_lines("/contacts/witnesses", None, None, None)
                    viewer.log_arrows("/contacts/normals", begins, ends, (0, 1, 0))
                else:
                    viewer.log_lines("/contacts/witnesses", begins, ends, (1, 1, 0))
                viewer.end_frame()
                return viewer.get_frame().numpy().copy()

            for xray in (False, True):
                viewer.renderer._physics_xray = xray
                viewer.renderer._xray_inspection_radius = 1.0
                viewer.renderer._xray_proximity_range = 3.0
                reference = draw_segment(start, clipped)
                yellow = (reference[:, :, 0] > 100) & (reference[:, :, 1] > 100) & (reference[:, :, 2] < 80)
                self.assertGreater(np.count_nonzero(yellow), 10)
                for a, b in ((start, end), (end, start)):
                    actual = draw_segment(a, b)
                    np.testing.assert_allclose(actual, reference, atol=2)

            for a, b in ((start, end), (end, start)):
                actual = draw_segment(a, b, arrows=True)
                green = (actual[:, :, 1] > 150) & (actual[:, :, 0] < 80) & (actual[:, :, 2] < 80)
                self.assertGreater(np.count_nonzero(green), 5)
            actual = draw_segment(eye - forward, end, arrows=True)
            green = (actual[:, :, 1] > 150) & (actual[:, :, 0] < 80) & (actual[:, :, 2] < 80)
            self.assertEqual(np.count_nonzero(green), 0)
            # Zero force (or a disabled length scale) must not produce the
            # camera-facing arrowhead used for nonzero view-aligned vectors.
            actual = draw_segment(start, start, arrows=True)
            green = (actual[:, :, 1] > 150) & (actual[:, :, 0] < 80) & (actual[:, :, 2] < 80)
            self.assertEqual(np.count_nonzero(green), 0)
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

    def test_contact_shadows_on_planar_and_concave_receivers(self):
        """Keep box shadows attached without darkening unobstructed ramps or bowls."""
        viewer = _make_headless_viewer_gl_or_skip(self, width=400, height=300)
        try:
            viewer.renderer.draw_sky = False
            viewer.renderer.sky_upper = (0.0, 0.0, 0.0)
            viewer.renderer.sky_lower = (0.0, 0.0, 0.0)
            viewer.renderer.draw_edges = False
            viewer.renderer.specular_scale = 0.0
            viewer.renderer.spotlight_enabled = False
            self._check_small_box_contact_shadows(viewer)
            self._check_bowl_shadows(viewer)
        finally:
            viewer.close()

    def _check_small_box_contact_shadows(self, viewer):
        """Keep small shadows attached and bare receivers lit using the existing GL context."""
        viewer.clear_model()
        viewer.renderer.draw_edges = False
        width = viewer.renderer._screen_width
        height = viewer.renderer._screen_height
        for slope in (20.0, 45.0):
            with self.subTest(slope=slope):
                rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(np.deg2rad(slope)))

                def rotate(point, rotation=rotation):
                    """Rotate a point into the ramp's coordinate system."""
                    return wp.quat_rotate(rotation, wp.vec3(point))

                builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
                # Keep the receiver inside the view: clipped triangles can vanish
                # on software rasterizers. The large-quad test covers clipping.
                builder.add_shape_box(
                    -1,
                    xform=wp.transform(rotate((0.0, 0.0, -0.005)), rotation),
                    hx=0.12,
                    hy=0.12,
                    hz=0.005,
                    color=(0.2, 0.5, 0.8),
                )
                # A 0.5 mm gap places the shadow immediately beside the small box.
                builder.add_shape_box(
                    -1,
                    xform=wp.transform(rotate((0.0, 0.0, 0.0155)), rotation),
                    hx=0.02,
                    hy=0.02,
                    hz=0.015,
                    color=(0.9, 0.2, 0.1),
                )
                model = builder.finalize(device=viewer.device)
                viewer.set_model(model)
                viewer.renderer._sun_direction = np.array(rotate((0.0, -1.0, 1.0))) / np.sqrt(2.0)
                viewer.set_camera(pos=rotate((0.0, 0.16, 0.35)), pitch=0.0, yaw=0.0)
                viewer.camera.near = 0.005
                viewer.camera.look_at((0.0, 0.0, 0.0))
                frames = []
                for shadows in (False, True):
                    viewer.renderer.draw_shadows = shadows
                    viewer.begin_frame(0.0)
                    viewer.log_state(model.state())
                    viewer.end_frame()
                    frames.append(viewer.get_frame().numpy().astype(np.float32))

                def sample_patch(frame, xmin, xmax, ymin, ymax):
                    """Sample a rectangular patch on the ramp."""
                    points = np.array(
                        [
                            (*rotate((x, y, 0.0)), 1.0)
                            for x in np.linspace(xmin, xmax, 9)
                            for y in np.linspace(ymin, ymax, 9)
                        ]
                    )
                    view = viewer.camera.get_view_matrix().reshape(4, 4).T
                    projection = viewer.camera.get_projection_matrix().reshape(4, 4).T
                    clip = points @ (projection @ view).T
                    ndc = clip[:, :2] / clip[:, 3:4]
                    pixels = np.floor((ndc * (0.5, -0.5) + 0.5) * (width, height)).astype(int)
                    return frame[pixels[:, 1], pixels[:, 0]]

                near_contact = [sample_patch(frame, -0.005, 0.005, 0.024, 0.032) for frame in frames]
                self.assertTrue(np.all(near_contact[0][:, 2] > near_contact[0][:, 0]), "patch misses the ramp")
                self.assertGreater(
                    float(np.mean(near_contact[0] - near_contact[1])),
                    20.0,
                    "the small box's shadow is detached from its contact with the ramp",
                )
                unoccluded = [sample_patch(frame, 0.07, 0.10, 0.04, 0.08) for frame in frames]
                self.assertLess(
                    float(np.mean(np.abs(unoccluded[0] - unoccluded[1]))),
                    2.0,
                    "the unobstructed receiver incorrectly shadows itself",
                )

    def _check_bowl_shadows(self, viewer):
        """Keep an unobstructed bowl unchanged when shadows are enabled."""
        viewer.clear_model()
        coordinates = np.linspace(-0.25, 0.25, 65)
        vertices = np.array([(x, y, 4.0 * (x * x + y * y)) for y in coordinates for x in coordinates], dtype=np.float32)
        normals = np.column_stack((-8.0 * vertices[:, 0], -8.0 * vertices[:, 1], np.ones(len(vertices))))
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        corners = np.array([y * 65 + x for y in range(64) for x in range(64)])
        indices = np.column_stack((corners, corners + 1, corners + 66, corners, corners + 66, corners + 65))
        mesh = newton.Mesh(vertices, indices.ravel(), normals=normals, compute_inertia=False)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_shape_mesh(-1, mesh=mesh, color=(0.2, 0.5, 0.8))
        model = builder.finalize(device=viewer.device)
        viewer.set_model(model)
        # Every point on this bowl can see the light, despite its curvature.
        viewer.renderer._sun_direction = np.array((0.0, -0.2, 1.0)) / np.sqrt(1.04)
        viewer.set_camera(pos=wp.vec3(0.0, 0.65, 0.9), pitch=0.0, yaw=0.0)
        viewer.camera.near = 0.005
        viewer.camera.look_at((0.0, 0.0, 0.1))

        frames = []
        for shadows in (False, True):
            viewer.renderer.draw_shadows = shadows
            viewer.begin_frame(0.0)
            viewer.log_state(model.state())
            viewer.end_frame()
            frames.append(viewer.get_frame().numpy().astype(np.float32))

        bowl = frames[0][:, :, 2] > frames[0][:, :, 0]
        self.assertTrue(np.any(bowl), "the bowl is not visible")
        self.assertLess(
            float(np.mean(np.abs(frames[0][bowl] - frames[1][bowl]))),
            2.0,
            "the unobstructed bowl incorrectly shadows itself",
        )

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

    def test_recreated_viewer_renders_frame(self):
        """Verify a viewer renders after a previous viewer was closed."""
        for cycle in range(2):
            with self.subTest(cycle=cycle):
                viewer = _make_headless_viewer_gl_or_skip(self)
                try:
                    self.assertTrue(viewer.is_running())
                    width = viewer.renderer._screen_width
                    height = viewer.renderer._screen_height
                    image = np.full((height, width, 3), (31, 127, 223), dtype=np.uint8)

                    viewer.begin_frame(0.0)
                    viewer.log_image("color", image, fullscreen=True)
                    viewer.end_frame()

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
        viewer._rendering_paused = False
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
            output = wp.empty_like(frame)
            self.assertIs(viewer.get_frame(output=output), output)
            np.testing.assert_array_equal(output.numpy(), frame.numpy())

            with self.assertWarnsRegex(DeprecationWarning, "target_image.*output"):
                self.assertIs(viewer.get_frame(target_image=output), output)
            with self.assertWarnsRegex(DeprecationWarning, "Passing 'output' positionally"):
                self.assertIs(viewer.get_frame(output), output)
            with self.assertWarnsRegex(DeprecationWarning, "Passing 'output', 'render_ui' positionally"):
                self.assertIs(viewer.get_frame(output, False), output)
            with self.assertRaisesRegex(TypeError, "output.*target_image"):
                viewer.get_frame(output=output, target_image=output)

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
        self.assertEqual(fake_gl.readback_count, 5)

    def test_texture_affine_transform_matches_authored_coordinates(self):
        """Apply a UsdTransform2d affine mapping exactly once before GL sampling."""
        viewer = _make_headless_viewer_gl_or_skip(self, width=320, height=240)

        try:
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
            viewer.set_camera(wp.vec3(0.0, -3.0, 2.5), pitch=0.0, yaw=0.0)
            viewer.camera.look_at((0.0, 0.0, 0.0))

            points = np.array(
                [(-1.0, -0.8, 0.0), (1.0, -0.8, 0.0), (1.0, 0.8, 0.0), (-1.0, 0.8, 0.0)],
                dtype=np.float32,
            )
            indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
            normals = np.array([(0.0, 0.0, 1.0)] * 4, dtype=np.float32)
            source_uvs = np.array([(0.08, 0.13), (1.21, 0.19), (1.14, 1.08), (0.03, 1.02)], dtype=np.float32)
            y, x = np.mgrid[:72, :96]
            texture = np.stack(
                (
                    32 + 192 * ((x // 12 + 2 * (y // 18)) % 2),
                    24 + 208 * ((y // 9) % 2),
                    16 + ((5 * x + 3 * y) % 60) * 4,
                ),
                axis=-1,
            ).astype(np.uint8)
            xforms = wp.array([wp.transform_identity()], dtype=wp.transform, device=viewer.device)
            scales = wp.array([(1.0, 1.0, 1.0)], dtype=wp.vec3, device=viewer.device)
            colors = wp.array([(1.0, 1.0, 1.0)], dtype=wp.vec3, device=viewer.device)
            materials = wp.array([(0.5, 0.0, 0.0, 1.0)], dtype=wp.vec4, device=viewer.device)

            def render(uvs: np.ndarray, texture_transform=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))) -> np.ndarray:
                mesh = newton.Mesh(
                    points,
                    indices,
                    normals=normals,
                    uvs=uvs,
                    compute_inertia=False,
                    texture=texture,
                    texture_transform=texture_transform,
                )
                viewer.log_geo(
                    "/test/texture_transform",
                    newton.GeoType.MESH,
                    (1.0, 1.0, 1.0),
                    0.0,
                    True,
                    geo_src=mesh,
                    hidden=True,
                )
                viewer.log_instances(
                    "/test/texture_transform_instance",
                    "/test/texture_transform",
                    xforms,
                    scales,
                    colors,
                    materials,
                )
                for _ in range(2):
                    viewer.begin_frame(0.0)
                    viewer.end_frame()
                return viewer.get_frame().numpy()

            scale = np.array((0.7, 1.3), dtype=np.float32)
            translate = np.array((0.13, 0.27), dtype=np.float32)
            rotate = 37.0

            angle = np.deg2rad(rotate)
            cosine, sine = np.cos(angle), np.sin(angle)
            scaled = source_uvs * scale
            expected_uvs = (
                np.column_stack(
                    (
                        cosine * scaled[:, 0] - sine * scaled[:, 1],
                        sine * scaled[:, 0] + cosine * scaled[:, 1],
                    )
                )
                + translate
            )
            texture_transform = (
                (scale[0] * cosine, -scale[1] * sine, translate[0]),
                (scale[0] * sine, scale[1] * cosine, translate[1]),
            )
            actual = render(source_uvs, texture_transform=texture_transform)
            expected = render(expected_uvs.astype(np.float32))

            visible = np.any(expected > 8, axis=2)
            self.assertGreater(np.std(expected[visible]), 30.0, "the texture control was not visibly asymmetric")
            self.assertLess(np.abs(actual.astype(np.int16) - expected.astype(np.int16))[visible].mean(), 2.0)
        finally:
            viewer.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
