# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import gc
import importlib.util
import inspect
import sys
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock
from unittest.mock import Mock, patch

import numpy as np
import warp as wp

import newton
from newton import Axis
from newton._src.geometry.sdf_texture import TextureSDFData, create_empty_texture_sdf_data
from newton._src.geometry.sdf_utils import SDF, SDFData
from newton._src.viewer.camera import Camera
from newton._src.viewer.gui.collision import CollisionInspector
from newton._src.viewer.gui.imgui_backend import ImGuiBackend
from newton._src.viewer.gui.properties import (
    Assignment,
    Frequency,
    PropertyInspector,
    component_attributes,
    joint_editor_options,
)
from newton._src.viewer.gui.viewer_gui import ViewerGui
from newton._src.viewer.gui.workspace import ViewerWorkspace
from newton._src.viewer.viewer_bindings import InspectionBindings
from newton._src.viewer.viewer_bvh import _bounds_lines, _query_bounds
from newton._src.viewer.viewer_gl import ViewerGL
from newton._src.viewer.viewer_history import FrameHistory
from newton._src.viewer.viewer_model_tree import ModelTree
from newton._src.viewer.viewer_null import ViewerNull
from newton.tests.unittest_utils import get_test_devices, patch_sys_module


class TestViewerUI(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from imgui_bundle import imgui, imguizmo
        except ImportError as exc:
            self.skipTest(f"Viewer UI dependencies not available: {exc}")

        context = imgui.create_context()
        self.addCleanup(imgui.destroy_context, context)
        # Exercise ImGui state without an OpenGL renderer or window.
        self.ui = ImGuiBackend.__new__(ImGuiBackend)
        self.ui.imgui = imgui
        self.ui.giz = imguizmo.im_guizmo
        self.ui.io = imgui.get_io()
        self.ui.io.set_ini_filename("")
        self.ui.io.display_size = (640, 480)
        self.ui.io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
        self.ui.is_available = True
        self.ui._frame_time = None

    def test_frame_delta(self) -> None:
        """Advance ImGui time by elapsed frame durations, including a repeated timestamp."""
        elapsed = 0.0
        with mock.patch(
            "newton._src.viewer.gui.imgui_backend.perf_counter",
            side_effect=[0.0, 0.025, 0.050, 0.060, 0.060],
        ):
            for delta in (1.0 / 60.0, 0.025, 0.025, 0.010, 1e-6):
                with self.subTest(delta=delta):
                    self.ui.begin_frame()
                    try:
                        elapsed += delta
                        self.assertAlmostEqual(self.ui.io.delta_time, delta)
                        self.assertAlmostEqual(self.ui.imgui.get_time(), elapsed)
                    finally:
                        self.ui.imgui.end_frame()

    def test_dpi_scaling(self) -> None:
        """Keep style sizes stable on reapplication and restore them at scale one."""
        imgui = self.ui.imgui
        font = self.ui.io.fonts.add_font_default()
        self.ui._setup_dark_style()
        colors = [tuple(imgui.get_style().color_(i)) for i in range(imgui.Col_.count)]
        for scale in (1.0, 2.0, 2.0, 1.5, 1.0):
            with self.subTest(scale=scale):
                expected_style = imgui.Style()
                expected_style.scale_all_sizes(scale)
                self.ui.dpi_scale = scale
                self.ui._apply_dpi_scaling()
                style = imgui.get_style()
                self.assertEqual(tuple(style.window_padding), (10.0 * scale, 6.0 * scale))
                # This size is not explicitly overridden by the Newton theme.
                self.assertEqual(tuple(style.separator_text_padding), tuple(expected_style.separator_text_padding))
                self.assertEqual(style.font_scale_dpi, scale)
                self.assertEqual([tuple(style.color_(i)) for i in range(imgui.Col_.count)], colors)
                self.assertEqual(self.ui.io.fonts.fonts[0], font)


def _make_gl_state(paused: bool = False, step_requested: bool = False) -> ViewerGL:
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


@unittest.skipUnless(importlib.util.find_spec("imgui_bundle"), "Requires imgui_bundle")
class TestViewerGuiRecovery(unittest.TestCase):
    def test_callback_failure_leaves_next_frame_usable(self):
        """Unwind real ImGui windows and styles without a renderer or model."""
        from imgui_bundle import imgui, imguizmo

        context = imgui.create_context()
        self.addCleanup(imgui.destroy_context, context)
        io = imgui.get_io()
        io.set_ini_filename("")
        io.display_size = (320, 320)
        io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
        gui = ViewerGui.__new__(ViewerGui)
        gui._viewer = ViewerNull()
        gui._viewer.show_ui = True
        gui.workspace = None
        gui.workspace_enabled = False
        gui._gizmo_active = {}
        gui._loading_splash_active = False
        # Exercise the real backend's stack snapshots and recovery without OpenGL.
        gui.ui = ImGuiBackend.__new__(ImGuiBackend)
        gui.ui.is_available = True
        gui.ui.imgui = imgui
        gui.ui.io = io
        gui.ui.giz = imguizmo.im_guizmo
        gui.ui._frame_time = None
        gui.ui._pending_file_dialog = None
        error = RuntimeError("GUI callback failed")

        def fail():
            imgui.begin("Example")
            imgui.begin_child("Nested")
            imgui.push_style_color(imgui.Col_.text, (1.0, 0.0, 0.0, 1.0))
            raise error

        gui._render_ui = fail
        with self.assertRaises(RuntimeError) as raised:
            gui.prepare_frame(update_fps=False)
        self.assertIs(raised.exception, error)
        self.assertFalse(gui._frame_prepared)
        self.assertTrue(io.config_error_recovery_enable_assert)
        gui._render_ui = lambda: imgui.text("Retry")
        gui.prepare_frame(update_fps=False)
        self.assertTrue(gui._frame_prepared)


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
        """Move at the configured speed while keeping the orbit pivot centered."""
        camera = Camera(pos=(0.0, 0.0, 0.0))
        camera.yaw = 0.0
        camera.sync_pivot_to_view(distance=10.0)
        offset = np.asarray(camera.pivot) - np.asarray(camera.pos)
        viewer = SimpleNamespace(camera=camera, camera_speed=2.0, is_rendering_paused=lambda: False)
        gui = ViewerGui.__new__(ViewerGui)
        gui._viewer = viewer
        gui.ui = None
        gui._cam_vel = np.zeros(3, dtype=np.float32)
        gui._cam_damp_tau = 0.1

        key = SimpleNamespace(W=1, UP=2, S=3, DOWN=4, A=5, LEFT=6, D=7, RIGHT=8, Q=9, E=10)
        pyglet = SimpleNamespace(window=SimpleNamespace(key=key))
        with patch_sys_module("pyglet", pyglet):
            gui.update_camera_from_keys(0.1, lambda code: code == key.W)

        self.assertAlmostEqual(camera.pos.x, 0.2)
        self.assertAlmostEqual(camera.pos.y, 0.0)
        self.assertAlmostEqual(camera.pos.z, 0.0)
        np.testing.assert_allclose(np.asarray(camera.pivot) - camera.pos, offset)
        with patch_sys_module("pyglet", pyglet):
            for pressed in (key.D, key.A, key.E, key.Q, key.S):
                gui.update_camera_from_keys(0.1, lambda code, pressed=pressed: code == pressed)
                np.testing.assert_allclose(np.asarray(camera.pivot) - camera.pos, offset, atol=1e-6)
                position = np.asarray(camera.pos)
                camera.orbit(0, 0)
                np.testing.assert_allclose(camera.pos, position, atol=1e-6)

    def test_camera_deceleration_is_capped(self):
        """Check that the camera deceleration is capped to avoid flipping the camera velocity direction."""
        camera = Camera(pos=(0.0, 0.0, 0.0))
        viewer = SimpleNamespace(camera=camera, camera_speed=2.0, is_rendering_paused=lambda: False)
        gui = ViewerGui.__new__(ViewerGui)
        gui._viewer = viewer
        gui.ui = None
        velocity_init = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # Non-zero initial velocity
        gui._cam_vel = velocity_init.copy()
        gui._cam_damp_tau = 0.1

        key = SimpleNamespace(W=1, UP=2, S=3, DOWN=4, A=5, LEFT=6, D=7, RIGHT=8, Q=9, E=10)
        pyglet = SimpleNamespace(window=SimpleNamespace(key=key))
        with patch_sys_module("pyglet", pyglet):
            gui.update_camera_from_keys(2.0 * gui._cam_damp_tau, lambda _: False)

        self.assertGreaterEqual(np.dot(camera.pos, velocity_init), 0.0)


class TestViewerShapeProperties(unittest.TestCase):
    def test_geometry_resource_tree_and_lazy_metadata(self) -> None:
        """Inspect shared resources lazily and distinguish their shape instances."""
        builder = newton.ModelBuilder()
        mesh = newton.Mesh.create_box(1, 1, 1)
        mesh.texture = np.zeros((2, 4, 3), dtype=np.uint8)
        builder.add_shape_mesh(-1, mesh=mesh, cfg=builder.ShapeConfig(has_shape_collision=False))
        builder.add_shape_sphere(-1, radius=0.5)
        model = builder.finalize(device="cpu")
        plain = ModelTree(model)
        self.assertNotIn(plain.resources.root, plain.roots)
        self.assertTrue(plain.has_children(("shape", 0)))
        self.assertEqual(plain.children[("shape", 0)], [("shape_bvh", 0), ("shape_texture", 0)])
        self.assertFalse(plain.has_children(("shape", 1)))
        bvhs = plain.resources.bvhs
        self.assertEqual(bvhs.scope, 2)  # This builder keeps all shapes in Shared.
        with patch.object(wp.array, "numpy", autospec=True, side_effect=wp.array.numpy) as read:
            plain.toggle(bvhs.root)
            plain.toggle(bvhs.meshes)
            self.assertIn(bvhs.root, plain.roots)
            self.assertIs(bvhs.entries[("bvh", "shapes")].get(), model.bvh_shapes)
            self.assertIs(bvhs.entries[("shape_bvh", 0)].get(), bvhs.entries[("bvh_mesh", 0)].get())
            read.assert_not_called()
        self.assertIsNone(bvhs._buffers)
        plain.query = "color texture"
        plain.rebuild()
        self.assertIn((("shape_texture", 0), 1), plain.rows)

        # A mesh-owned NanoVDB SDF and a separate Model table are distinct.
        mesh.sdf = SDF(data=SDFData(), _internal=True)
        self.assertEqual(plain.resources.sdf_storage(mesh.sdf), "Empty")
        mesh.sdf.data.sparse_sdf_ptr = 1  # Metadata only; never dereferenced.
        self.assertEqual(plain.resources.sdf_storage(mesh.sdf), "NanoVDB volumes")
        entries = [create_empty_texture_sdf_data() for _ in range(260)]
        entries[259].voxel_size = wp.vec3(0.1, 0.2, 0.3)
        model._texture_sdf_data = wp.array(entries, dtype=TextureSDFData, device="cpu")
        model._shape_sdf_index.assign([259, 259])
        with patch.object(wp.array, "numpy", autospec=True, side_effect=wp.array.numpy) as read:
            tree = ModelTree(model)
            self.assertFalse(tree.resources._shape_pages)
            self.assertFalse(tree.resources._rows)
            read.reset_mock()
            tree.toggle(tree.resources.root)
            self.assertEqual(len(tree.children[tree.resources.root]), 3)
            tree.resources.reveal(tree, ("sdf", 259))
            self.assertIn(("sdf", 259), tree.labels)
            read.assert_not_called()
            self.assertTrue(tree.has_children(("shape", 0)))
            self.assertEqual(tree.resources.shape_sdf(1), 259)
            self.assertEqual(read.call_count, 2)  # Bounded mesh-pointer and SDF-index pages.
            self.assertEqual(
                set(tree.children[("shape", 0)]),
                {("shape_bvh", 0), ("shape_sdf", 0), ("mesh_sdf", 0), ("shape_texture", 0)},
            )
            tree.resources.row(259)
            tree.resources.row(259)
            self.assertEqual(read.call_count, 3)  # Cache one selected struct, never texture pixels.
        tree.query = "sdf 259"
        tree.rebuild()
        self.assertIn((("sdf", 259), 1), tree.rows)
        tree.query = ""
        tree.rebuild()
        np.testing.assert_allclose(tree.resources.row(259)["voxel_size"], (0.1, 0.2, 0.3))

        # Replication preserves local labels; navigation must retain world and
        # instance identity even when shortening paths produces duplicate names.
        template = newton.ModelBuilder()
        for label in ("shape_1", "left/geom", "right/geom"):
            template.add_shape_sphere(-1, radius=0.1, label=label)
        replicated = newton.ModelBuilder()
        replicated.replicate(template, copy_count=3)
        model = replicated.finalize(device="cpu")
        model._texture_sdf_data = wp.array(entries[:1], dtype=TextureSDFData, device="cpu")
        model._shape_sdf_index.fill_(0)
        tree = ModelTree(model)
        bvhs = tree.resources.bvhs
        self.assertEqual(bvhs.scope, 1)  # A model-wide resource includes every world by default.
        resource = bvhs.entries[("bvh", "shapes")].get()
        offsets = wp.array([(0, 0, 0), (10, 0, 0), (0, 15, 0)], dtype=wp.vec3, device="cpu")
        viewer = SimpleNamespace(world_offsets=offsets, layer=SimpleNamespace(xform=wp.transform_identity()))
        ws = SimpleNamespace(viewer=viewer, example=None)
        empty_vec = wp.empty(0, dtype=wp.vec3, device="cpu")
        empty_int = wp.empty(0, dtype=int, device="cpu")
        for budget, query_count, group in ((9, 1, -1), (4, 4, -1), (9, 1, 1), (9, 1, 3)):
            indices = wp.empty(budget, dtype=int, device="cpu")
            count = wp.zeros(1, dtype=int, device="cpu")
            wp.launch(
                _query_bounds,
                dim=query_count,
                inputs=[resource.id, False, group, query_count, indices, count],
                device="cpu",
            )
            size = int(count.numpy()[0])
            items = indices.numpy()[:size]
            worlds = resource.groups.numpy()[items]
            self.assertEqual(set(worlds), {0, 1, 2} if group < 0 else ({1} if group == 1 else set()))
            self.assertEqual(size, 9 if budget == 9 and group < 0 else (0 if group == 3 else 3))
            starts = wp.empty(budget * 12, dtype=wp.vec3, device="cpu")
            ends = wp.empty_like(starts)
            wp.launch(
                _bounds_lines,
                dim=(budget, 12),
                inputs=[indices, count, *bvhs._geometry_args(ws, resource, -1, empty_vec, empty_int)],
                outputs=[starts, ends],
                device="cpu",
            )
            lines = np.stack((starts.numpy(), ends.numpy())).reshape(2, budget, 12, 3)[:, :size]
            np.testing.assert_allclose(
                lines.min(axis=(0, 2)), resource.lowers.numpy()[items] + offsets.numpy()[worlds], atol=1e-6
            )
            np.testing.assert_allclose(
                lines.max(axis=(0, 2)), resource.uppers.numpy()[items] + offsets.numpy()[worlds], atol=1e-6
            )
        captions, navigated = {}, []

        def reference(_label: str, value: int, *, target: str | None, caption: str) -> bool:
            self.assertEqual(target, model.shape_label[value])
            captions[value] = caption
            return value == 3

        workspace = SimpleNamespace(
            tree=tree,
            property_inspector=SimpleNamespace(_page=lambda owners: owners),
            widgets=SimpleNamespace(
                group=lambda *_args, **_kwargs: nullcontext(True),
                number=lambda *_args, **_kwargs: None,
                reference=reference,
            ),
            navigate=navigated.append,
        )
        with patch.object(wp.array, "numpy", autospec=True, side_effect=wp.array.numpy) as read:
            tree.resources._users(workspace, 0)
            tree.resources._users(workspace, 0)
            self.assertEqual(read.call_count, 1)  # Only the cached inverse SDF index.
        self.assertEqual(tree.loaded_worlds, {0})
        self.assertEqual(captions[0], "World 0 · shape_1")
        self.assertEqual(captions[3], "World 1 · shape_1")
        self.assertEqual(captions[6], "World 2 · shape_1")
        self.assertEqual(captions[1], "World 0 · #1 · geom")
        self.assertEqual(captions[2], "World 0 · #2 · geom")
        self.assertEqual(len(set(captions.values())), model.shape_count)
        self.assertEqual(navigated, [("shape", 3), ("shape", 3)])

        # Deep robot chains retain every body/shape level, with articulations
        # directly under their world. Other worlds remain lazily indexed.
        template = newton.ModelBuilder()
        parent, joints = -1, []
        for i in range(16):
            body = template.add_link(label=f"link_{i}")
            joints.append(template.add_joint_revolute(parent=parent, child=body, label=f"hinge_{i}"))
            template.add_shape_sphere(body, radius=0.05)
            parent = body
        template.add_articulation(joints, label="robot")
        scene = newton.ModelBuilder()
        scene.replicate(template, copy_count=3)
        hierarchy = ModelTree(scene.finalize(device="cpu"))
        self.assertEqual(hierarchy.parents[("articulation", 0)], ("root", 0))
        self.assertNotIn("Articulations", hierarchy.labels.values())
        hierarchy.expanded.update(hierarchy.labels)
        hierarchy.rebuild()
        depths = dict(hierarchy.rows)
        self.assertGreater(depths[("body", 15)], 12)
        self.assertEqual(depths[("shape", 15)], depths[("body", 15)] + 1)
        self.assertEqual(hierarchy.parents[("shape", 15)], ("body", 15))
        self.assertEqual(hierarchy.loaded_worlds, {0})

        # Properties can be visible before the Scene tab is ever opened.
        # In particular the Examples tab remains selected after a switch.
        workspace = ViewerWorkspace.__new__(ViewerWorkspace)
        workspace.model, workspace.tree, workspace.example = model, None, None
        workspace.viewer = ViewerNull()
        workspace.selection, workspace.selected, workspace.message = set(), None, ""
        workspace.gui = SimpleNamespace(ui=SimpleNamespace(imgui=None))
        workspace.widgets = None
        inspected, attached = [], []
        workspace.collision = SimpleNamespace(handles=lambda _key: False, attach=attached.append)
        workspace.property_inspector = SimpleNamespace(draw=lambda: inspected.append(workspace.tree))
        workspace._properties()
        workspace._properties()
        self.assertIsInstance(workspace.tree, ModelTree)
        self.assertIs(workspace.tree.model, model)
        self.assertEqual(inspected, [workspace.tree, workspace.tree])
        self.assertEqual(attached, inspected)

    def test_mujoco_contact_properties_with_both_contact_pipelines(self) -> None:
        """Inspect mapped shapes across worlds without requiring an external-contact inverse map."""
        values = {}
        workspace = ViewerWorkspace.__new__(ViewerWorkspace)
        workspace._cache = {}
        workspace._live_cache = {}
        workspace.viewer = ViewerNull()
        workspace.model = None
        workspace.world = 1
        workspace.widgets = SimpleNamespace(
            group=lambda *_args, **_kwargs: nullcontext(True),
            choice=lambda label, value, *_args, **_kwargs: values.update({label: value}),
            array=lambda label, value, **_kwargs: values.update({label: value}),
        )
        friction = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
        solver = SimpleNamespace(
            mjw_model=SimpleNamespace(
                geom_condim=wp.array([1, 3, 6], dtype=int, device="cpu"),
                geom_solref=wp.array(np.full((1, 3, 2), 0.02), dtype=float, device="cpu"),
                geom_solimp=wp.array(np.full((1, 3, 5), 0.9), dtype=float, device="cpu"),
                geom_friction=wp.array(friction, dtype=float, device="cpu"),
            ),
            mjc_geom_to_newton_shape=wp.array([[1, 3, -1], [1, 7, 9]], dtype=int, device="cpu"),
        )
        workspace.example = SimpleNamespace(solver=solver)
        for inverse in (None, wp.array([-1, 0, -1, 1, -1, -1, -1, 1, -1, 2], dtype=int, device="cpu")):
            with self.subTest(internal_contacts=inverse is None):
                solver.newton_shape_to_mjc_geom = inverse
                workspace._cache.clear()
                workspace._live_cache.clear()
                values.clear()
                workspace._shape_solver_properties(7)
                self.assertEqual(values["condim"], 3)
                np.testing.assert_array_equal(values["Friction"], friction[1, 1])
                np.testing.assert_allclose(values["solref"], [0.02, 0.02])
                self.assertEqual(values["solimp"].shape, (5,))
                values.clear()
                workspace._shape_solver_properties(1)
                self.assertEqual(values["condim"], 1)
                np.testing.assert_array_equal(values["Friction"], friction[1, 0])
                values.clear()
                workspace._shape_solver_properties(6)
                self.assertFalse(values)

        # Collision-free models can have worlds but no solver geometries.
        for shape in ((0,), (2, 0)):
            with self.subTest(empty_mapping_shape=shape):
                solver.mjc_geom_to_newton_shape = wp.empty(shape, dtype=int, device="cpu")
                workspace._cache.clear()
                values.clear()
                workspace._shape_solver_properties(0)
                self.assertFalse(values)
                empty = workspace.read(solver, "mjc_geom_to_newton_shape", 0)
                self.assertEqual(empty.shape, (min(shape[0], 1), *shape[1:]))
                self.assertEqual(empty.dtype, np.int32)


class TestViewerGLShouldStep(unittest.TestCase):
    """ViewerGL.should_step() state machine: running, paused, and single-step."""

    def test_gui_popup_owns_viewer_shortcuts(self) -> None:
        """Keep popup keys from closing or changing the simulation without keyboard capture."""
        viewer = SimpleNamespace(_paused=False, _step_requested=False, show_ui=True, _display_options_user={})
        gui = ViewerGui.__new__(ViewerGui)
        gui._viewer = viewer
        gui.workspace = gui.console_capture = None
        popup = True
        gui.ui = SimpleNamespace(
            is_available=True,
            io=SimpleNamespace(want_capture_keyboard=False, want_text_input=False),
            imgui=SimpleNamespace(
                is_popup_open=lambda *_: popup,
                PopupFlags_=SimpleNamespace(any_popup=1),
            ),
        )
        closed = []
        framed = []
        gui.frame_camera_on_model = lambda: framed.append(True)
        key = SimpleNamespace(GRAVE=1, MOD_CTRL=2, SPACE=3, PERIOD=4, H=5, F=6, ESCAPE=7)
        with patch_sys_module("pyglet", SimpleNamespace(window=SimpleNamespace(key=key))):
            for symbol in (key.ESCAPE, key.SPACE, key.PERIOD, key.H, key.F):
                gui.handle_key_press(symbol, close_fn=lambda: closed.append(True))
            self.assertFalse(closed)
            self.assertFalse(framed)
            self.assertFalse(viewer._paused)
            self.assertFalse(viewer._step_requested)
            self.assertTrue(viewer.show_ui)
            popup = False
            gui.handle_key_press(key.SPACE)
            self.assertTrue(viewer._paused)
            self.assertTrue(viewer._display_options_user["_paused"])
            gui.handle_key_press(key.PERIOD)
            self.assertTrue(viewer._step_requested)
            gui.handle_key_press(key.ESCAPE, close_fn=lambda: closed.append(True))
            self.assertEqual(closed, [True])

            # GUI focus and numeric editing leave viewer shortcuts available.
            gui.workspace = SimpleNamespace(rect=(200, 40, 600, 400))
            gui.ui.io = SimpleNamespace(
                want_capture_keyboard=True,
                want_capture_mouse=False,
                want_text_input=False,
                mouse_pos=(400, 200),
            )
            active = False
            numeric = False
            gui.ui.imgui.is_any_item_active = lambda: active
            gui.ui.imgui.internal = SimpleNamespace(
                get_active_id=lambda: 42,
                temp_input_is_active=lambda _: numeric,
                get_input_text_state=lambda _: SimpleNamespace(flags=0),
            )
            gui.ui.imgui.InputTextFlags_ = SimpleNamespace(chars_decimal=1, chars_scientific=2, chars_hexadecimal=4)
            for mouse, text, item, capture_mouse, numeric in (
                ((100, 200), False, False, True, False),
                ((400, 200), True, True, False, False),
                ((400, 200), False, True, False, False),
                ((400, 200), False, False, True, False),
                ((100, 200), True, True, True, True),
            ):
                gui.ui.io.mouse_pos, gui.ui.io.want_text_input = mouse, text
                active, gui.ui.io.want_capture_mouse = item, capture_mouse
                before = viewer._paused
                gui.handle_key_press(key.SPACE)
                self.assertEqual(viewer._paused, before if text and not numeric else not before)
            viewer._step_requested = False
            gui.handle_key_press(key.PERIOD)
            gui.handle_key_press(key.ESCAPE, close_fn=lambda: closed.append(True))
            self.assertFalse(viewer._step_requested)
            self.assertEqual(closed, [True])

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


def _make_gl_running_state(headless: bool, num_frames: int | None, frame_count: int = 0) -> ViewerGL:
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
        """Reject CUDA interop values outside the public flag enum."""
        for value in (True, 1, 1.5, "dynamic"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                ViewerGL(enable_cuda_interop=value)  # type: ignore[arg-type]

    def test_cuda_interop_defaults_to_dynamic_meshes(self):
        """Enable CUDA interop for dynamic meshes by default."""
        parameter = inspect.signature(ViewerGL).parameters["enable_cuda_interop"]
        self.assertEqual(parameter.default, ViewerGL.CudaInterop.DYNAMIC_MESH)

    def test_cuda_interop_flags_are_composable(self):
        """Compose independent CUDA interop categories with bitwise flags."""
        flags = ViewerGL.CudaInterop.POINTS | ViewerGL.CudaInterop.LINES
        self.assertTrue(flags & ViewerGL.CudaInterop.POINTS)
        self.assertTrue(flags & ViewerGL.CudaInterop.LINES)
        self.assertFalse(flags & ViewerGL.CudaInterop.INSTANCES)
        self.assertEqual(
            ViewerGL.CudaInterop.ALL,
            ViewerGL.CudaInterop.DYNAMIC_MESH
            | ViewerGL.CudaInterop.STATIC_MESH
            | ViewerGL.CudaInterop.POINTS
            | ViewerGL.CudaInterop.INSTANCES
            | ViewerGL.CudaInterop.LINES,
        )

    def test_rejects_unknown_cuda_interop_flags(self):
        """Reject flag bits outside the supported CUDA interop categories."""
        with self.assertRaises(ValueError):
            ViewerGL(enable_cuda_interop=ViewerGL.CudaInterop(1 << 10))


class TestViewerProperties(unittest.TestCase):
    def test_selected_articulation_joint_states(self) -> None:
        """Recover every selected articulation's coordinates from maximal or historical poses."""
        from newton.solvers import SolverXPBD  # noqa: PLC0415

        builder = newton.ModelBuilder()
        for _ in range(2):
            body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            joint = builder.add_joint_revolute(-1, body, axis=newton.Axis.Z)
            builder.add_articulation([joint])
        model = builder.finalize(device="cpu")
        state = model.state()
        expected_q = np.array([0.25, -0.65], dtype=np.float32)
        expected_qd = np.array([0.3, -0.4], dtype=np.float32)
        state.joint_q.assign(expected_q)
        state.joint_qd.assign(expected_qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        state.joint_q.zero_()
        state.joint_qd.zero_()
        viewer = ViewerNull()
        viewer.activate("primary")
        viewer.set_model(model)
        viewer._last_state = state
        ws = ViewerWorkspace.__new__(ViewerWorkspace)
        ws.model, ws.viewer = model, viewer
        ws.example = SimpleNamespace(state_0=state, solver=SolverXPBD(model))
        ws.history = ws._kinematic_state = ws._kinematic_indices = None
        ws._cache, ws._live_cache = {}, {}
        ws.selection = {("articulation", 0), ("joint", 1)}
        ws.selected = ("joint", 1)
        shown = ws.joint_display_state()
        np.testing.assert_allclose(shown.joint_q.numpy(), expected_q, atol=1e-6)
        np.testing.assert_allclose(shown.joint_qd.numpy(), expected_qd, atol=1e-6)

        # A body can be the active row while joint domains from the other
        # selected items are still displayed by the property inspector.
        ws.selection.add(("body", 1))
        ws.selected = ("body", 1)
        ws._live_cache.clear()
        shown = ws.joint_display_state()
        np.testing.assert_allclose(shown.joint_q.numpy(), expected_q, atol=1e-6)

        # Historical rendering must use the displayed poses, without replacing
        # either live State's reduced coordinates or body transforms.
        historical = model.state()
        historical.joint_q.assign(-expected_q)
        historical.joint_qd.assign(-expected_qd)
        newton.eval_fk(model, historical.joint_q, historical.joint_qd, historical)
        viewer._last_state = historical
        ws.example.solver = None
        ws.example.timeline = SimpleNamespace(frames=[0, 1], index=0)
        ws._live_cache.clear()
        shown = ws.joint_display_state()
        np.testing.assert_allclose(shown.joint_q.numpy(), -expected_q, atol=1e-6)
        np.testing.assert_allclose(shown.joint_qd.numpy(), -expected_qd, atol=1e-6)
        np.testing.assert_array_equal(state.joint_q.numpy(), 0.0)
        np.testing.assert_array_equal(state.joint_qd.numpy(), 0.0)

        # Custom loops register persistent components independently and log
        # the current state after each buffer swap, without an Example object.
        control = model.control()
        solver = SolverXPBD(model)
        pipeline = newton.CollisionPipeline(model)
        viewer.set_control(control)
        viewer.set_solver(solver)
        viewer.set_collision_pipeline(pipeline)
        ws.example = None
        ws._drafts = {}
        viewer.log_state(state)
        self.assertIs(ws.live_state(), state)
        viewer.log_state(historical)
        self.assertIs(ws.live_state(), historical)
        self.assertIs(ws.bindings.solver, solver)
        self.assertEqual(ws.bindings.collision_pipelines(model), [("Collision pipeline", pipeline)])
        inspector = PropertyInspector(ws)
        inspector._contexts()
        control_domain = next(domain for domain in inspector.domains if domain.frequency == Frequency.JOINT_COORD)
        self.assertIs(inspector._owner(control_domain, Assignment.CONTROL), control)
        pointer = control.joint_target_q.ptr
        self.assertTrue(ws.apply_edit(control, "joint_target_q", 0, [0.45], joint_index=0))
        self.assertEqual(control.joint_target_q.ptr, pointer)
        np.testing.assert_allclose(control.joint_target_q.numpy()[0], 0.45)

        # Replacing or clearing one component must preserve the others, even
        # when a conventional example would otherwise supply a fallback.
        replacement = model.control()
        viewer.set_control(replacement)
        self.assertIs(ws.bindings.control, replacement)
        self.assertIs(ws.bindings.solver, solver)
        ws.example = SimpleNamespace(model=model, state_0=state, control=control, solver=solver, pipeline=pipeline)
        viewer.set_control(None)
        viewer.set_solver(None)
        viewer.set_collision_pipeline(None)
        self.assertIsNone(ws.bindings.control)
        self.assertIsNone(ws.bindings.solver)
        self.assertEqual(ws.bindings.collision_pipelines(model), [])
        viewer.set_control(control)
        viewer.set_solver(solver)
        viewer.set_collision_pipeline(pipeline)
        self.assertIs(ws.bindings.control, control)
        self.assertTrue(ws._history_compatible)

        # Layers retain separate components and logged states. Invalid model
        # associations must not replace a valid existing component.
        original_layer = viewer.layer.layer_id
        viewer.activate("other")
        for setter in (viewer.set_control, viewer.set_solver, viewer.set_collision_pipeline):
            with self.assertRaisesRegex(ValueError, "set_model"):
                setter(None)
        other_model = builder.finalize(device="cpu")
        viewer.set_model(other_model)
        ws.model = other_model
        self.assertIsNone(ws.bindings.control)
        for setter, component in ((viewer.set_solver, solver), (viewer.set_collision_pipeline, pipeline)):
            with self.assertRaisesRegex(ValueError, "active model"):
                setter(component)
        viewer.set_control(other_model.control())
        self.assertIsNot(ws.bindings.control, control)
        viewer.activate(original_layer)
        ws.model = model
        self.assertIs(ws.bindings.control, control)
        self.assertIs(ws.bindings.solver, solver)
        self.assertIs(viewer._last_state, historical)

        # Another layer can have identical array dimensions but a different
        # topology. The conventional adapter must not expose the old controls.
        rebuilt = builder.finalize(device="cpu")
        viewer.set_model(rebuilt)
        ws.model = rebuilt
        ws.example = SimpleNamespace(model=model, state_0=state, control=control)
        self.assertIsNone(ws.bindings.control)
        self.assertFalse(ws._history_compatible)
        self.assertIsNone(viewer._last_state)
        ws.model = model
        self.assertIs(ws.bindings.control, control)
        self.assertTrue(ws._history_compatible)

    def test_element_selection_property_domains(self) -> None:
        builder = newton.ModelBuilder()
        for point in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)):
            builder.add_particle(wp.vec3(*point), wp.vec3(0), 1.0)
        builder.add_triangle(0, 1, 2)
        builder.add_triangle(1, 4, 2)
        builder.add_tetrahedron(0, 1, 2, 3)
        builder.add_tetrahedron(1, 4, 2, 3)
        model = builder.finalize(device="cpu")
        ws = ViewerWorkspace.__new__(ViewerWorkspace)
        ws.model, ws.tree = model, ModelTree(model)
        ws._cache, ws._live_cache = {}, {}
        inspector = PropertyInspector(ws)
        for kind, frequency in (("tri", Frequency.TRIANGLE), ("tet", Frequency.TETRAHEDRON)):
            ws.selected = (kind, 1)
            ws.selection = {(kind, 0), (kind, 1)}
            inspector._contexts()
            domains = {d.frequency: list(d.indices) for d in inspector.domains}
            self.assertEqual(domains[frequency], [0, 1])
            self.assertEqual(domains[Frequency.PARTICLE], [0, 1, 2, 4] if kind == "tri" else [0, 1, 2, 3, 4])
            # Changing membership while retaining the active item invalidates the cache.
            ws.selection.remove((kind, 0))
            inspector._contexts()
            domain = next(d for d in inspector.domains if d.frequency == frequency)
            self.assertEqual(list(domain.indices), [1])

    def test_history_without_top_level_example_model(self) -> None:
        """Preserve independent host snapshots and replay for nested CPU and CUDA models."""
        builder = newton.ModelBuilder()
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="sample",
                namespace="probe",
                assignment=Assignment.STATE,
                frequency=Frequency.ONCE,
                dtype=float,
                default=3.0,
            )
        )
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="drive",
                namespace="probe",
                assignment=Assignment.CONTROL,
                frequency=Frequency.ONCE,
                dtype=float,
                default=5.0,
            )
        )
        model = builder.finalize(device="cpu")
        example = SimpleNamespace(viewer=SimpleNamespace(model=model), render=Mock())
        history = FrameHistory(example)
        self.assertIs(history.model, model)
        history.capture()
        self.assertFalse(history.frames)
        history.render()
        example.render.assert_called_once_with()

        # A nested model must also work when the example exposes a recordable
        # state. CPU NumPy views alias live storage; CUDA readbacks own host storage.
        for device in get_test_devices(mode="basic"):
            with self.subTest(device=device):
                current = model if device.is_cpu else builder.finalize(device=device)
                example = SimpleNamespace(
                    viewer=SimpleNamespace(model=current), state_0=current.state(), control=current.control()
                )
                example.state_0.solver_scratch = wp.zeros(1, device=device)
                history = FrameHistory(example)
                history.capture()
                snapshot = history.frames[0]
                self.assertNotIn("solver_scratch", snapshot.state)
                example.state_0.probe.sample.fill_(7.0)
                example.control.probe.drive.fill_(11.0)
                if device.is_cuda:
                    gc.collect()
                np.testing.assert_array_equal(snapshot.state["probe:sample"], [3.0])
                np.testing.assert_array_equal(snapshot.control["probe:drive"], [5.0])
                example.sim_time = 1.0
                history.capture()
                history.index = 0
                live_state, live_control = example.state_0, example.control
                displayed = []

                def render_snapshot(
                    example=example,
                    history=history,
                    live_state=live_state,
                    live_control=live_control,
                    displayed=displayed,
                ):
                    self.assertIsNot(example.state_0, live_state)
                    self.assertIs(example.control, live_control)
                    displayed.append(
                        (example.state_0.probe.sample.numpy(), history.display_control.probe.drive.numpy())
                    )

                example.render = render_snapshot
                history.render()
                self.assertEqual(len(displayed), 1)
                np.testing.assert_array_equal(displayed[0][0], [3.0])
                np.testing.assert_array_equal(displayed[0][1], [5.0])
                self.assertIs(example.state_0, live_state)
                self.assertIs(example.control, live_control)
                np.testing.assert_array_equal(live_state.probe.sample.numpy(), [7.0])
                np.testing.assert_array_equal(live_control.probe.drive.numpy(), [11.0])

    def test_sparse_properties_and_joint_editor_ranges(self) -> None:
        """Discover only selected frequencies while retaining dynamic custom fields and bounded editors."""
        builder = newton.ModelBuilder()
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        shape = builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        joint = builder.add_joint_fixed(-1, body)
        builder.add_articulation([joint])
        model = builder.finalize(device="cpu")
        ws = ViewerWorkspace.__new__(ViewerWorkspace)
        ws.model, ws.selected = model, ("joint", joint)
        ws.viewer = SimpleNamespace(_last_state=model.state())
        ws.example = SimpleNamespace(state_0=ws.viewer._last_state, control=model.control())
        ws.history = None
        ws._cache, ws._live_cache = {}, {}
        inspector = PropertyInspector(ws)
        inspector._contexts()
        self.assertEqual([domain.frequency for domain in inspector.domains], [Frequency.JOINT])
        self.assertFalse(inspector._collect(Assignment.STATE))
        self.assertFalse(inspector._collect(Assignment.CONTROL))
        self.assertTrue(inspector._collect(Assignment.MODEL))
        ws.selected = ("body", body)
        inspector._contexts()
        self.assertFalse(inspector._collect(Assignment.CONTROL))
        self.assertTrue(inspector._collect(Assignment.STATE))
        all_fields = list(component_attributes(model, model, Assignment.MODEL))
        body_fields = list(component_attributes(model, model, Assignment.MODEL, frequencies={Frequency.BODY}))
        self.assertEqual(
            [field[0] for field in body_fields],
            [field[0] for field in all_fields if field[1].frequency == Frequency.BODY],
        )
        self.assertFalse(list(component_attributes(model, model, Assignment.MODEL, frequencies=set())))
        for attr in ("joint_q", "joint_qd", "joint_target_q", "joint_target_qd"):
            regular = joint_editor_options(attr, -2, 2)
            self.assertTrue(regular["slider"])
            for bounds in ((-1e6, 1e6), (-np.inf, np.inf), (None, None)):
                unbounded = joint_editor_options(attr, *bounds)
                self.assertFalse(unbounded["slider"])
                self.assertEqual(unbounded["speed"], regular["speed"])
                self.assertLessEqual(unbounded["speed"], 0.01)

        # Solver-defined objects use the same metadata as the property inspector.
        frequency, count = "probe:drive", 260
        model.custom_frequency_counts.update({frequency: count, "probe:empty": 0})
        model.custom_frequency_articulation[frequency] = wp.zeros(count, dtype=int, device="cpu")
        model.add_attribute("names", [f"drive/name {i}" for i in range(count)], frequency, namespace="probe")
        model.custom_frequency_label_attributes[frequency] = "probe:names"
        model.add_attribute(
            "target", wp.full(count, shape, dtype=int, device="cpu"), frequency, namespace="probe", references="shape"
        )
        ws.tree = tree = ModelTree(model)
        ws.selected = ("articulation", 0)
        inspector._contexts()
        domain = next(domain for domain in inspector.domains if domain.frequency == frequency)
        np.testing.assert_array_equal(domain.indices, np.arange(count))
        group = next(key for key, (freq, _) in tree.custom.groups.items() if freq == frequency)
        self.assertEqual(tree.parents[group], ("articulation", 0))
        self.assertFalse(tree.children[group])
        tree.toggle(group)
        self.assertLessEqual(len(tree.children[group]), tree.custom.page_size)
        self.assertFalse(any(key[0] == "custom:" + frequency for key in tree.labels))
        key = tree.custom.key(frequency, count - 1)
        tree.custom.reveal(tree, key)
        self.assertEqual(tree.labels[key], "drive/name 259")
        tree.query = "name 259"
        tree.rebuild()
        self.assertIn((key, 1), tree.rows)
        ws.selected = key
        inspector._contexts()
        self.assertEqual([(d.frequency, list(d.indices)) for d in inspector.domains], [(frequency, [259])])
        self.assertEqual(
            {f[1] for fields in inspector._collect(Assignment.MODEL).values() for f in fields},
            {"probe:names", "probe:target"},
        )
        self.assertEqual(
            {field[0] for field in component_attributes(model, model, Assignment.MODEL, frequencies={frequency})},
            {"probe:names", "probe:target"},
        )
        self.assertEqual(inspector.reference_target(model, Frequency.SHAPE, shape), ("shape", shape))
        self.assertEqual(inspector.reference_target(model, frequency, count - 1), key)
        for invalid in (-1, count, count + 10):
            self.assertIsNone(inspector.reference_target(model, frequency, invalid))
        self.assertIsNone(inspector.reference_target(model, Frequency.SHAPE, model.shape_count))

    def test_frequency_discovery_and_live_joint_edits(self) -> None:
        builder = newton.ModelBuilder()
        joints = []
        for parent in (-1, 0, 1, 0):
            child = builder.add_link(mass=1, inertia=wp.mat33(np.eye(3)))
            joints.append(builder.add_joint_revolute(parent, child, axis=newton.Axis.Z))
        builder.add_articulation(joints)
        model = builder.finalize(device="cpu")
        for assignment, frequency, name, dtype, values in (
            (Assignment.MODEL, Frequency.JOINT, "tag", wp.int32, [1, 2, 3, 4]),
            (Assignment.STATE, Frequency.JOINT_COORD, "offset", wp.vec3, [[1, 2, 3]] * 4),
            (Assignment.CONTROL, Frequency.JOINT_DOF, "enabled", wp.bool, [True] * 4),
        ):
            model.add_attribute(
                name,
                wp.array(values, dtype=dtype, device="cpu"),
                frequency=frequency,
                assignment=assignment,
                namespace="test",
            )
        model.add_attribute(
            "joint_q", wp.zeros(4, device="cpu"), Frequency.JOINT_DOF, Assignment.CONTROL, namespace="probe"
        )
        model.add_attribute(
            "drive", wp.full(4, wp.vec3(1, 2, 3), device="cpu"), Frequency.JOINT_DOF, Assignment.CONTROL
        )
        state, other, control = model.state(), model.state(), model.control()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        for owner, assignment, expected in (
            (model, Assignment.MODEL, "test:tag"),
            (state, Assignment.STATE, "test:offset"),
            (control, Assignment.CONTROL, "test:enabled"),
        ):
            fields = {name: spec for name, spec, *_ in component_attributes(model, owner, assignment)}
            self.assertIn(expected, fields)
            self.assertEqual({n for n in fields if n.startswith("test:")}, {expected})
        ws = ViewerWorkspace.__new__(ViewerWorkspace)
        ws.model, ws.selected, ws.world = model, ("articulation", 0), 0
        ws.viewer = SimpleNamespace(is_paused=lambda: True, _last_state=state)
        ws.example = SimpleNamespace(state_0=state, state_1=other, control=control)
        ws._cache, ws._live_cache, ws._drafts = {}, {}, {}
        ws._undo = ws.history = None
        control.joint_target_q.fill_(0.15)
        ws._capture_control_initial()
        inspector = PropertyInspector(ws)
        inspector._contexts()
        self.assertEqual(
            {d.frequency for d in inspector.domains},
            {Frequency.ARTICULATION, Frequency.JOINT, Frequency.JOINT_COORD, Frequency.JOINT_DOF},
        )
        before = state.body_q.numpy().copy()
        # Sibling coordinates deliberately disagree with the displayed body state.
        state.joint_q[3:4].fill_(1.2)
        self.assertTrue(ws.apply_edit(state, "joint_q", 1, [0.5], joint_index=1))
        np.testing.assert_array_equal(state.body_q.numpy()[[0, 3]], before[[0, 3]])
        self.assertGreater(np.linalg.norm(state.body_q.numpy()[[1, 2]] - before[[1, 2]]), 0.1)
        np.testing.assert_array_equal(other.body_q.numpy(), state.body_q.numpy())
        mask, _ = ws._joint_edit_mask(1)
        np.testing.assert_array_equal(mask.numpy(), [False, True, True, False])
        # Mutations are guarded at the write boundary, not just by disabled widgets.
        ws.viewer.is_paused = lambda: False
        self.assertFalse(ws.apply_edit(state, "joint_q", 1, [0.9], joint_index=1))
        self.assertTrue(ws.apply_edit(control, "joint_target_q", 1, [0.25], joint_index=1))
        self.assertAlmostEqual(float(control.joint_target_q.numpy()[1]), 0.25)
        self.assertTrue(ws.apply_edit(control.probe, "joint_q", 2, [0.75], joint_index=2))
        self.assertAlmostEqual(float(control.probe.joint_q.numpy()[2]), 0.75)
        self.assertTrue(ws.apply_edit(control.test, "enabled", 1, [False]))
        self.assertFalse(control.test.enabled.numpy()[1])
        self.assertFalse(ws.apply_edit(state.test, "offset", 1, [8, 9, 10]))
        self.assertIsNone(inspector.reference_target(model, Frequency.JOINT_COORD, 1))
        self.assertIsNone(inspector.reference_target(model, Frequency.JOINT_DOF, 1))
        # A live edit and its undo use the same ownership-based permission check.
        self.assertTrue(ws.apply_edit(*ws._undo[:4], joint_index=ws._undo[4], save_undo=False))
        self.assertTrue(control.test.enabled.numpy()[1])
        ws.viewer.is_paused = lambda: True
        ws.example.timeline = SimpleNamespace(frames=[object(), object()], index=0)
        self.assertFalse(ws.apply_edit(state, "joint_qd", 1, [2], joint_index=1))
        self.assertFalse(ws.apply_edit(control.test, "enabled", 0, [False]))
        ws.viewer.is_paused = lambda: False
        self.assertFalse(ws.apply_edit(control, "joint_target_q", 0, [1.0]))
        self.assertFalse(ws.apply_edit(control.test, "enabled", 0, [False]))
        ws.viewer.is_paused = lambda: True
        del ws.example.timeline
        self.assertTrue(ws.apply_edit(control.test, "enabled", 0, [False]))
        self.assertFalse(control.test.enabled.numpy()[0])
        self.assertTrue(ws.apply_edit(state.test, "offset", 2, [4, 5, 6]))
        np.testing.assert_array_equal(state.test.offset.numpy()[2], [4, 5, 6])
        # Whole-Control actions preserve graph bindings, custom namespaces and
        # the example's configured targets, even while the simulation runs.
        pointers = (control.joint_target_q.ptr, control.test.enabled.ptr, control.drive.ptr)
        pose = state.body_q.numpy().copy()
        ws.viewer.is_paused = lambda: False
        self.assertTrue(ws.reset_control(zero=True))
        np.testing.assert_allclose(control.joint_target_q.numpy(), 0.15)
        np.testing.assert_array_equal(control.probe.joint_q.numpy(), 0)
        np.testing.assert_array_equal(control.test.enabled.numpy(), False)
        np.testing.assert_array_equal(control.drive.numpy(), 0)
        self.assertTrue(ws.reset_control())
        np.testing.assert_array_equal(control.test.enabled.numpy(), True)
        np.testing.assert_array_equal(control.drive.numpy(), [[1, 2, 3]] * 4)
        self.assertEqual(pointers, (control.joint_target_q.ptr, control.test.enabled.ptr, control.drive.ptr))
        np.testing.assert_array_equal(state.body_q.numpy(), pose)
        self.assertIsNone(ws._undo)
        ws.example.timeline = SimpleNamespace(frames=[object(), object()], index=0)
        self.assertFalse(ws.reset_control(zero=True))
        np.testing.assert_array_equal(control.test.enabled.numpy(), True)
        del ws.example.timeline
        # A newly loaded example/Control gets its own baseline.
        ws.example.control = model.control()
        ws.example.control.joint_target_q.fill_(0.4)
        ws._capture_control_initial()
        ws.example.control.joint_target_q.fill_(0.9)
        self.assertTrue(ws.reset_control())
        np.testing.assert_allclose(ws.example.control.joint_target_q.numpy(), 0.4)


class TestCollisionInspector(unittest.TestCase):
    def test_paused_collision_inspection(self) -> None:
        """Inspect all broadphases lazily and preserve contact attribute row domains."""
        frequency, assignment = newton.Model.AttributeFrequency, newton.Model.AttributeAssignment
        builder = newton.ModelBuilder()
        for x in (0.0, 0.8):
            body = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()))
            builder.add_shape_sphere(body, radius=0.5)
        model = builder.finalize(device="cpu")
        model.add_attribute(
            "weights",
            wp.array(np.arange(40, dtype=np.float32), device="cpu"),
            frequency.CONTACT,
            assignment.CONTACT,
            namespace="probe",
        )
        model._set_attribute_spec(
            "probe:weights", newton.Model.AttributeSpec(frequency.CONTACT, assignment.CONTACT, row_width=2)
        )
        model.add_attribute(
            "shape_link",
            wp.full(16, 1, dtype=int, device="cpu"),
            frequency.CONTACT_RIGID,
            assignment.CONTACT,
            namespace="probe",
            references="shape",
        )
        for mode in ("nxn", "sap", "explicit"):
            with self.subTest(broadphase=mode):
                options = (
                    {"shape_pairs_filtered": wp.array([(0, 1)], dtype=wp.vec2i, device="cpu")}
                    if mode == "explicit"
                    else {}
                )
                pipeline = newton.CollisionPipeline(
                    model, broad_phase=mode, rigid_contact_max=16, soft_contact_max=4, **options
                )
                state, contacts = model.state(), pipeline.contacts()
                pipeline.collide(state, contacts)
                viewer = ViewerNull()
                viewer.set_model(model)
                viewer.log_contacts(contacts, state)
                viewer.is_paused = lambda: True
                ws = SimpleNamespace(
                    viewer=viewer,
                    model=model,
                    example=SimpleNamespace(collision_pipeline=pipeline, contacts=pipeline.contacts()),
                    history=None,
                    selected=None,
                    _at_live_frame=lambda: True,
                )
                ws.bindings = InspectionBindings.from_example(ws.example)
                ws.select = lambda key, workspace=ws: setattr(workspace, "selected", key)
                inspector = CollisionInspector(ws)
                # A logged frame takes precedence over an example's other
                # contact buffer, including when the normal glyphs are hidden.
                self.assertIs(inspector._contacts_buffer(), contacts)
                tree = ModelTree(model)
                inspector.attach(tree)
                with patch.object(inspector, "read", wraps=inspector.read) as read:
                    inspector.sync()
                    read.assert_not_called()
                tree.toggle(inspector.root)
                self.assertGreater(inspector.counts[0], 0)
                self.assertIn(("collision_broad", 0), tree.labels)
                self.assertIn(("collision_narrow", 0), tree.labels)
                self.assertEqual(inspector.groups[("collision_pairs", 0)][2], 1)
                inspector.reveal(("contact", 0))
                self.assertEqual(set(inspector.shapes(("contact", 0))), {0, 1})
                model.shape_label[:] = ["world_0/bolt/shape_0", "world_0/nut/shape_0"]
                label = inspector.contact_label(("contact", 0))
                self.assertIn("bolt/shape_0", label)
                self.assertIn("nut/shape_0", label)
                self.assertNotIn("world_0", label)
                self.assertEqual(len(inspector.points(("contact", 0))), 2)
                # F/focus consumes a numeric bounds array, including contacts.
                ws.collision = inspector
                viewer._last_state = state
                ws.selected = ("contact", 0)
                bounds = ViewerWorkspace.selection_bounds(ws)
                self.assertIsInstance(bounds, np.ndarray)
                np.testing.assert_allclose(bounds.mean(axis=0), np.mean(inspector.points(ws.selected), axis=0))
                # Selection rings do not depend on the optional text labels.
                viewer.renderer = SimpleNamespace()
                viewer.show_contacts = viewer.show_contact_pairs = True
                viewer.show_contact_labels = False
                ws.widgets = SimpleNamespace(scale=1.0)
                inspector.overlays()
                self.assertEqual(viewer.renderer._contact_selection, (viewer._qualify("/contacts/modes"), 0, 2))
                self.assertEqual(viewer.renderer._contact_labels, ())
                viewer.show_contact_pairs = False
                viewer.show_contact_labels = True
                inspector.overlays()
                self.assertEqual(viewer.renderer._contact_selection[1:], (0, 1))
                self.assertEqual(viewer.renderer._contact_labels, ())
                inspector.contacts_tab = True
                inspector.overlays()
                self.assertEqual(len(viewer.renderer._contact_labels), 1)
                self.assertEqual(len(viewer.renderer._contact_labels[0][0]), 1)
                viewer.show_contact_pairs = True
                inspector.overlays()
                np.testing.assert_allclose(viewer.renderer._contact_labels[0][0], inspector.points(("contact", 0)))
                # Opening a shape's contact list must not select a contact or
                # make F jump away from the shape. Pagination is only for rows.
                ws.selected = ("shape", 0)
                ui = SimpleNamespace(push_id=lambda *_: None, pop_id=lambda: None, selectable=lambda *_: (False,))
                ws.gui = SimpleNamespace(ui=SimpleNamespace(imgui=ui))
                ws.widgets.label = lambda *_args, **_kwargs: None
                inspector.draw_contact_list(SimpleNamespace(_page=lambda rows: rows[:1]), ws.selected)
                self.assertIsNone(inspector.active_contact)
                self.assertEqual(inspector.focus_points(), [])
                inspector.visible_contacts = ()
                inspector.overlays()
                self.assertEqual(len(viewer.renderer._contact_labels), len(inspector.shape_contacts(0)))
                # A selected owner alone must not read or lay out any labels.
                # Closing and reopening the tab also invalidates its overlay.
                inspector.contacts_tab = False
                tree.expanded.discard(inspector.root)
                with (
                    patch.object(inspector, "read", wraps=inspector.read) as read,
                    patch.object(inspector, "object_contacts", wraps=inspector.object_contacts) as rows,
                    patch.object(inspector, "label_points", wraps=inspector.label_points) as points,
                ):
                    inspector.sync()
                    inspector.overlays()
                    read.assert_not_called()
                    rows.assert_not_called()
                    points.assert_not_called()
                self.assertEqual(viewer.renderer._contact_labels, ())
                inspector.contacts_tab = True
                inspector.overlays()
                self.assertEqual(len(viewer.renderer._contact_labels), len(inspector.shape_contacts(0)))
                # Live labels follow current witnesses without expanding the
                # inspector or retaining a transient selected contact slot.
                expected = np.asarray(inspector.points(("contact", 0)))
                viewer.is_paused = lambda: False
                inspector.overlays()
                self.assertEqual(len(viewer.renderer._contact_labels), 1)
                np.testing.assert_allclose(viewer.renderer._contact_labels[0][0], expected)
                self.assertIsNone(viewer.renderer._contact_selection)
                poses = state.body_q.numpy()
                poses[:, 2] += 0.25
                state.body_q.assign(poses)
                viewer.time += 1
                inspector.overlays()
                np.testing.assert_allclose(viewer.renderer._contact_labels[0][0], expected + np.array([0, 0, 0.25]))
                buffers = inspector._live_labels
                counters = contacts.contact_counters.numpy().copy()
                live_counters = counters.copy()
                live_counters[1] = 1
                contacts.contact_counters.assign(live_counters)
                contacts.soft_contact_shape.fill_(0)
                contacts.soft_contact_body_pos.fill_(wp.vec3(0.1, 0.2, 0.3))
                ws.selected = inspector.list_owner = ("body", 0)
                viewer.time += 1
                inspector.overlays()
                labels = viewer.renderer._contact_labels
                self.assertEqual([label[1] for label in labels], ["1", "2"])
                np.testing.assert_allclose(labels[1][0], [[0.1, 0.2, 0.55]], atol=1e-6)
                self.assertIs(inspector._live_labels, buffers)
                contacts.contact_counters.zero_()
                viewer.time += 1
                inspector.overlays()
                self.assertEqual(viewer.renderer._contact_labels, ())
                contacts.contact_counters.assign(counters)
                ws.selected = inspector.list_owner = ("shape", 0)
                inspector.contacts_tab = False
                with patch.object(inspector, "_live_overlays") as live:
                    inspector.overlays()
                    live.assert_not_called()
                self.assertEqual(viewer.renderer._contact_labels, ())
                poses[:, 2] -= 0.25
                state.body_q.assign(poses)
                viewer.is_paused = lambda: True
                inspector.contacts_tab = True
                inspector.sync()
                ws.selected = None
                inspector.overlays()
                self.assertIsNone(viewer.renderer._contact_selection)

                self.assertIn(("contact", 0), inspector.shape_contacts(0))
                self.assertEqual(inspector.shape_contacts(0), inspector.shape_contacts(1))
                self.assertEqual(inspector.object_contacts(("body", 0)), inspector.shape_contacts(0))
                pair = ("collision_pair", (0, 0))
                self.assertEqual(inspector.object_contacts(pair), inspector.shape_contacts(0))
                self.assertEqual(inspector.candidate_pairs(("contact", 0)), (pair,))
                inspector.reveal(pair)
                self.assertIn(pair, tree.labels)
                self.assertIn(("collision_broad", 0), tree.expanded)
                for owner in (("body", 0), ("shape", 0), pair):
                    ws.selected = inspector.list_owner = owner
                    inspector.contacts_tab = True
                    inspector.active_contact = ("contact", 0)
                    bounds = ViewerWorkspace.selection_bounds(ws)
                    np.testing.assert_allclose(bounds.mean(axis=0), np.mean(inspector.points(("contact", 0)), axis=0))
                inspector.contacts_tab = False
                ws.selected = None
                ws.tree = tree
                ws.read = inspector.read
                self.assertEqual(ViewerWorkspace._tree_type_name(ws, ("shape", 0)), "Sphere shape")
                self.assertEqual(ViewerWorkspace._tree_type_name(ws, ("body", 0)), "Rigid body")
                self.assertEqual(ViewerWorkspace._tree_type_name(ws, pair), "Candidate shape pair")

                self.assertEqual(inspector.shape_contacts(-1), ())
                self.assertEqual(inspector.number(("contact", 0)), 1)
                self.assertEqual(inspector.number(("soft_contact", 0)), inspector.counts[0] + 1)
                with patch.object(inspector, "read", wraps=inspector.read) as read:
                    inspector.shape_contacts(0)
                    read.assert_not_called()
                fields = {
                    name: (spec, start, count)
                    for _, name, _, spec, start, count in inspector.contact_fields(("contact", 0))
                }
                self.assertEqual(fields["probe:shape_link"][0].references, frequency.SHAPE)
                self.assertEqual(fields["probe:weights"][1:], (0, 2))
                # Collapsing the built-in fields must not turn their None
                # namespace into an invalid custom-attribute tab label.
                tabs = []
                ws.widgets = SimpleNamespace(
                    group=lambda _: nullcontext(False),
                    tab_item=lambda label, tabs=tabs: (tabs.append(label) or False,),
                )
                ws.gui = SimpleNamespace(
                    ui=SimpleNamespace(
                        imgui=SimpleNamespace(
                            separator_text=lambda _: None,
                            begin_tab_bar=lambda _: True,
                            end_tab_bar=lambda: None,
                        )
                    )
                )
                ws.selected = ("contact", 0)
                inspector.draw_contact(None, ws.selected)
                self.assertEqual(tabs, ["probe"])
                ws.selected = None
                soft = {
                    name: (start, count)
                    for _, name, _, _, start, count in inspector.contact_fields(("soft_contact", 1))
                }
                self.assertEqual(soft["probe:weights"], (34, 2))
                self.assertNotIn("probe:shape_link", soft)
                with patch.object(inspector, "read", wraps=inspector.read) as read:
                    for _ in range(5):
                        inspector.sync()
                    read.assert_not_called()
                # Large result sets create bounded tree pages, not a row per slot.
                group = ("collision_group", "contact")
                tree.children[group].clear()
                inspector.groups[group] = "contact", 0, 1_000_000, None
                inspector.expand(group)
                self.assertLessEqual(len(tree.children[group]), inspector.page_size)
                ws.selected = ("contact", 0)
                viewer.is_paused = lambda: False
                inspector.sync()
                self.assertEqual(ws.selected, inspector.root)
                self.assertNotIn(("contact", 0), tree.labels)
                self.assertEqual(inspector.counts, (0, 0))
                # A historical frame must not present current broadphase buffers.
                viewer.is_paused = lambda: True
                ws._at_live_frame = lambda: False
                inspector.sync()
                self.assertNotIn(("collision_pairs", 0), inspector.groups)
                self.assertEqual(inspector.candidate_pairs(("contact", 0)), ())
                # Contacts alone are useful; no empty Collision root without either source.
                ws.example.collision_pipeline = None
                inspector.attach(tree)
                inspector.sync(force=True)
                self.assertIn(inspector.root, tree.labels)
                self.assertFalse(inspector.pipelines)
                viewer._last_contacts = None
                ws.example.contacts = None
                with patch.object(inspector, "read", wraps=inspector.read) as read:
                    inspector.attach(tree)
                    inspector.sync()
                    read.assert_not_called()
                self.assertNotIn(inspector.root, tree.roots)
                self.assertFalse(any(inspector.handles(key) for key in tree.labels))
                ws.example.collision_pipeline = pipeline
                inspector.attach(tree)
                self.assertIn(inspector.root, tree.roots)

        from newton._src.viewer.gl.contact_labels import _layout_labels  # noqa: PLC0415

        def measure(text: str) -> tuple[int, int]:
            return 8 * len(text), 16

        viewport = np.array((640, 480))
        separated = [((-0.3, 0, 0.2), (0.3, 0, 0.4))]
        layout = _layout_labels([(separated[0], "12", False)], np.eye(4), viewport, measure)
        self.assertEqual(len(layout), 2)
        nearby = ((0, 0, 0.4), (0.001, 0, 0.2))
        layout = _layout_labels([(nearby, "12", False)], np.eye(4), viewport, measure)
        self.assertEqual(len(layout), 1)
        np.testing.assert_allclose(layout[0][0], nearby[1])
        # Distinct contacts at the same point keep every number, laid out apart.
        cluster = [(((0, 0, 0),), str(i), False) for i in range(20)]
        layout = _layout_labels(cluster, np.eye(4), viewport, measure)
        self.assertEqual({item[1] for item in layout}, {str(i) for i in range(20)})
        for i, (_, text, _, offset) in enumerate(layout):
            for _, other_text, _, other in layout[i + 1 :]:
                high = np.minimum(offset + measure(text), other + measure(other_text))
                low = np.maximum(offset, other)
                self.assertTrue(np.any(high <= low))


class TestViewerGizmos(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from imgui_bundle import imgui, imguizmo
        except ImportError:
            self.skipTest("imgui_bundle is required")

        self.imgui = imgui
        self.giz = imguizmo.im_guizmo
        context = imgui.create_context()
        self.addCleanup(imgui.destroy_context, context)
        self.io = imgui.get_io()
        self.io.set_ini_filename("")
        self.io.display_size = (800, 600)
        self.io.delta_time = 1.0 / 60.0
        # Build draw lists without a window or GPU renderer.
        self.io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
        self.viewer = ViewerGL.__new__(ViewerGL)
        # Initialize real shared viewer/layer state without a native GL renderer.
        self.viewer.__dict__.update(vars(ViewerNull()))
        self.viewer.camera = Camera(width=800, height=600, pos=(0.0, 0.0, 5.0), up_axis="Y")
        self.viewer.camera.yaw = -90.0
        self.viewer._rendering_paused = False
        self.viewer._gizmo_log = {}
        self.viewer.gizmo_is_using = False
        self.viewer.show_ui = True
        self.gui = ViewerGui.__new__(ViewerGui)
        self.gui._viewer = self.viewer
        self.gui.workspace = None
        self.gui.workspace_enabled = False
        self.gui._frame_key_down = self.gui._follow_held = self.gui._follow_latched = False
        self.gui._follow_center = None
        self.gui.ui = ImGuiBackend.__new__(ImGuiBackend)
        self.gui.ui.is_available = True
        self.gui.ui.imgui = imgui
        self.gui.ui.io = self.io
        self.gui.ui.giz = self.giz
        self.gui.ui._frame_time = None
        self.gui.ui._pending_file_dialog = None
        self.gui._loading_splash_active = False
        # Exercise the real frame/input pipeline without unrelated panels.
        self.gui._render_ui = self.gui._render_gizmos
        self.viewer.gui = self.gui
        self.viewer.picking = None
        self.viewer.picking_enabled = False
        self.viewer._to_framebuffer_coords = lambda x, y: (x, y)
        self.gui._camera_drag_buttons = 0
        self.gui._cam_vel = np.zeros(3)
        self.gui._camera_orbit_sensitivity = 0.1
        self.gui._camera_dolly_drag_sensitivity = 0.01
        # Import input constants without creating Pyglet's shadow GL window.
        import pyglet

        shadow_window = pyglet.options["shadow_window"]
        pyglet.options["shadow_window"] = False
        try:
            from pyglet.window import mouse
        finally:
            pyglet.options["shadow_window"] = shadow_window
        self.mouse = mouse
        self.gui._gizmo_active = {}
        self.gui._gizmo_hovered = False
        self.gui._camera_dolly_scroll_sensitivity = 0.15
        # ImGuizmo owns global drag state; always release it before destroying ImGui.
        self.addCleanup(self.frame, 0.0, 0.0, False)
        self.frame(0.0, 0.0, False)

    def frame(self, x: float, y: float, down: bool) -> None:
        self.io.add_mouse_pos_event(x, y)
        self.io.add_mouse_button_event(0, down)
        self.gui.prepare_frame(update_fps=False)

    def native_press(self, x: float, y: float) -> None:
        self.io.add_mouse_pos_event(x, y)
        self.io.add_mouse_button_event(0, True)
        self.viewer.on_mouse_press(x, 600 - y, self.mouse.LEFT, 0)

    def native_drag(self, x: float, y: float, dx: float, dy: float) -> None:
        self.io.add_mouse_pos_event(x, y)
        self.viewer.on_mouse_drag(x, 600 - y, dx, -dy, self.mouse.LEFT, 0)
        self.gui.prepare_frame(update_fps=False)

    def test_move_and_press_ownership(self) -> None:
        """Choose camera or gizmo ownership from a move and press in one event batch."""
        for onto_handle in (True, False):
            with self.subTest(onto_handle=onto_handle):
                self.viewer.camera.yaw = -90.0
                target = wp.transform_identity()
                original = np.asarray(target).copy()
                self.viewer.log_gizmo("target", target)
                old_x, press_x = (700, 420) if onto_handle else (420, 700)
                self.frame(old_x, 300, False)
                self.frame(old_x, 300, False)
                self.native_press(press_x, 300)
                self.gui.prepare_frame(update_fps=False)
                self.native_drag(press_x + 20, 300, 20, 0)
                if onto_handle:
                    self.assertEqual(self.viewer.camera.yaw, -90.0)
                    self.assertGreater(target[0], 0.0)
                else:
                    self.assertLess(self.viewer.camera.yaw, -90.0)
                    np.testing.assert_array_equal(target, original)
                self.frame(700, 300, False)

        # RTX does not define the optional viewer activity flag. Scene presses
        # must still navigate through the shared GUI path without a gizmo.
        del self.viewer.gizmo_is_using
        self.viewer._gizmo_log.clear()
        self.frame(700, 300, False)
        self.native_press(700, 300)
        self.gui.prepare_frame(update_fps=False)
        yaw = self.viewer.camera.yaw
        self.native_drag(720, 300, 20, 0)
        self.assertLess(self.viewer.camera.yaw, yaw)
        self.frame(720, 300, False)

    def test_cancel_drag(self) -> None:
        """Keep the pose and clear ownership on removal, pause, and focus loss."""
        for reason in ("remove", "disable", "hide", "pause", "focus"):
            with self.subTest(reason=reason):
                target = wp.transform_identity()
                other = wp.transform((1, 1, 0), wp.quat_identity())
                snap = wp.transform((0.1, 0.2, 0.3), wp.quat_identity())
                self.viewer.log_gizmo("target", target, snap_to=snap)
                self.viewer.log_gizmo("other", other)
                self.frame(420, 300, False)
                self.frame(420, 300, True)
                self.frame(440, 300, True)
                self.assertTrue(self.viewer.gizmo_is_using)
                held_pose = np.asarray(target).copy()
                if reason == "remove":
                    del self.viewer._gizmo_log["target"]
                    self.gui.prepare_frame(update_fps=False)
                elif reason == "disable":
                    self.viewer.log_gizmo("target", target, translate=(), rotate=(), snap_to=snap)
                    self.gui.prepare_frame(update_fps=False)
                elif reason == "hide":
                    self.viewer.show_ui = False
                    self.gui.prepare_frame(update_fps=False)
                    self.viewer.show_ui = True
                elif reason == "pause":
                    self.viewer.set_rendering_paused(True)
                    self.viewer.set_rendering_paused(False)
                else:
                    self.gui.on_window_deactivated()
                self.assertFalse(self.giz.is_using_any())
                self.assertFalse(self.viewer.gizmo_is_using)
                self.assertEqual(self.gui._camera_drag_buttons, 0)
                self.assertFalse(any(self.io.mouse_down))
                self.viewer.log_gizmo("target", target, snap_to=snap)
                self.native_drag(460, 300, 20, 0)
                np.testing.assert_array_equal(target, held_pose)
                self.frame(700, 300, False)
                self.frame(700, 300, False)
                self.assertFalse(self.gui.is_mouse_capturing())
                # A fresh press can navigate again.
                self.native_press(700, 300)
                self.gui.prepare_frame(update_fps=False)
                yaw = self.viewer.camera.yaw
                self.native_drag(720, 300, 20, 0)
                self.assertLess(self.viewer.camera.yaw, yaw)
                # Cancelling camera navigation also requires a new press.
                self.gui.on_rendering_paused()
                yaw = self.viewer.camera.yaw
                self.native_drag(740, 300, 20, 0)
                self.assertEqual(self.viewer.camera.yaw, yaw)
                self.frame(740, 300, False)
                self.viewer.camera.yaw = -90.0

    def test_drag_and_snap_with_multiple_operations(self) -> None:
        """Capture handles, preserve drags across idle operations, and snap only on release."""
        for translate in (None, (Axis.X, Axis.Y)):
            with self.subTest(translate=translate):
                target = wp.transform((0.0, 0.0, 0.0), wp.quat_identity())
                snap = wp.transform((0.1, 0.2, 0.3), wp.quat_identity())
                other = wp.transform((1.0, 1.0, 0.0), wp.quat_identity())
                self.viewer._gizmo_log.clear()
                self.viewer.log_gizmo("target", target, translate=translate, snap_to=snap)
                self.viewer.log_gizmo("other", other)
                self.frame(0.0, 0.0, False)
                # The X arrow is right of the origin at (400, 300).
                self.frame(420.0, 300.0, False)
                camera_pos = tuple(self.viewer.camera.pos)
                self.gui.handle_mouse_scroll(1.0)
                self.assertEqual(tuple(self.viewer.camera.pos), camera_pos)
                self.assertTrue(self.gui.is_mouse_capturing())

                self.frame(420.0, 300.0, True)
                self.frame(440.0, 300.0, True)
                self.assertTrue(self.viewer.gizmo_is_using)
                self.assertTrue(self.gui._gizmo_active["target"])
                self.assertFalse(self.gui._gizmo_active["other"])
                first_x = float(target[0])
                self.assertGreater(first_x, 0.0)
                np.testing.assert_allclose(np.asarray(target)[1:], (0.0, 0.0, 0.0, 0.0, 0.0, 1.0), atol=1e-6)
                np.testing.assert_array_equal(other, (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))

                # Stable names must survive a different logging order.
                self.viewer._gizmo_log = dict(reversed(tuple(self.viewer._gizmo_log.items())))
                self.frame(440.0, 300.0, True)
                self.assertAlmostEqual(target[0], first_x, places=5)
                self.assertTrue(self.gui._gizmo_active["target"])
                self.frame(460.0, 300.0, True)
                self.assertGreater(target[0], first_x)
                self.frame(460.0, 300.0, False)
                np.testing.assert_array_equal(target, snap)
                self.assertFalse(self.viewer.gizmo_is_using)
                self.assertFalse(self.gui._gizmo_active["target"])

                # A later idle frame must not snap again.
                snap[0] = 2.0
                self.frame(0.0, 0.0, False)
                self.assertAlmostEqual(target[0], 0.1)
                self.viewer._gizmo_log.clear()
                self.frame(0.0, 0.0, False)
                self.frame(0.0, 0.0, False)
                self.assertEqual(self.gui._gizmo_active, {})
                self.assertFalse(self.gui.is_mouse_capturing())

    def test_rotation_with_restricted_axes(self) -> None:
        """Rotate around the enabled axis and snap after releasing the ring."""
        target = wp.transform((0.0, 0.0, 0.0), wp.quat_identity())
        snap = wp.transform((0.1, 0.2, 0.3), wp.quat_identity())
        self.viewer.log_gizmo("target", target, translate=(Axis.X, Axis.Y), rotate=(Axis.Z,), snap_to=snap)
        # The Z ring crosses the upper-right quadrant around the origin.
        self.frame(420.0, 280.0, False)
        self.assertTrue(self.gui.is_mouse_capturing())
        self.frame(420.0, 280.0, True)
        self.frame(430.0, 300.0, True)
        self.assertTrue(self.gui._gizmo_active["target"])
        np.testing.assert_allclose(np.asarray(target)[:5], 0.0, atol=1e-6)
        self.assertGreater(abs(target[5]), 0.1)
        self.assertAlmostEqual(float(np.linalg.norm(np.asarray(target)[3:])), 1.0, places=5)
        self.frame(430.0, 300.0, False)
        np.testing.assert_array_equal(target, snap)
        self.assertFalse(self.viewer.gizmo_is_using)

    def test_idle_transform_is_unchanged(self) -> None:
        """Preserve the caller's quaternion representation when no handle modifies it."""
        target = wp.transform((0.0, 0.0, 0.0), wp.quat(0.0, 0.0, 0.0, -1.0))
        original = np.asarray(target).copy()
        # Also exercise the legacy raw-transform entry.
        self.viewer._gizmo_log["target"] = target
        self.frame(0.0, 0.0, False)
        np.testing.assert_array_equal(target, original)

    def test_mouse_input_without_pyglet_window_mouse_attribute(self) -> None:
        """Route mouse input when ``pyglet.window`` lacks its ``mouse`` attribute.

        A failed ``import pyglet.window`` (e.g. without a display) keeps
        ``pyglet.window.mouse`` in ``sys.modules``, and a later re-import of
        ``pyglet.window`` does not bind it as a package attribute again.
        """
        window = sys.modules["pyglet.window"]
        with mock.patch.dict(vars(window)):
            vars(window).pop("mouse", None)
            yaw = self.viewer.camera.yaw
            self.frame(700.0, 300.0, True)
            self.frame(720.0, 300.0, True)
            self.assertNotEqual(self.viewer.camera.yaw, yaw)
            self.frame(720.0, 300.0, False)
        self.assertEqual(self.gui._camera_drag_buttons, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
