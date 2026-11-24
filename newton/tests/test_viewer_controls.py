# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import gc
import inspect
import sys
import unittest
import warnings
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton import Axis
from newton._src.geometry.sdf_texture import TextureSDFData, create_empty_texture_sdf_data
from newton._src.geometry.sdf_utils import SDF, SDFData
from newton._src.viewer import viewer_history
from newton._src.viewer.camera import Camera
from newton._src.viewer.gui.collision import CollisionInspector
from newton._src.viewer.gui.imgui_backend import ImGuiBackend
from newton._src.viewer.gui.orientation import OrientationGizmo
from newton._src.viewer.gui.properties import (
    Assignment,
    Frequency,
    PropertyInspector,
    component_attributes,
    joint_editor_options,
)
from newton._src.viewer.gui.timeline import TimelineView
from newton._src.viewer.gui.viewer_gui import ViewerGui
from newton._src.viewer.gui.widgets import PhysicsWidgets, PropertyEdit
from newton._src.viewer.gui.workspace import ViewerWorkspace
from newton._src.viewer.viewer_attributes import AttributeRows
from newton._src.viewer.viewer_bindings import InspectionBindings
from newton._src.viewer.viewer_bvh import _bounds_lines, _query_bounds
from newton._src.viewer.viewer_gl import ViewerGL
from newton._src.viewer.viewer_history import FrameHistory
from newton._src.viewer.viewer_model_tree import ModelTree
from newton._src.viewer.viewer_null import ViewerNull
from newton._src.viewer.viewer_rtx import ViewerRTX
from newton.tests.unittest_utils import get_test_devices, import_pyglet_window, patch_sys_module, requires


@wp.kernel
def _scale_history_display(
    body_q: wp.array[wp.transform],
    particle_q: wp.array[wp.vec3],
    display_body_q: wp.array[wp.transform],
    display_particle_q: wp.array[wp.vec3],
) -> None:
    i = wp.tid()
    display_body_q[i] = wp.transform(wp.transform_get_translation(body_q[i]) * 0.01, wp.quat_identity())
    display_particle_q[i] = particle_q[i] * 0.01


def _record_frame(history: FrameHistory, time: float, *states: newton.State, contacts=None, observables=None) -> None:
    """Record one frame through the same calls the viewer makes for log_state() and log_contacts()."""
    history.begin_frame(time)
    for state in states:
        history.state(state)
    if contacts is not None:
        history.contacts(contacts, states[0], observables)
    history.end_frame()


def _offscreen_imgui_backend(test: unittest.TestCase) -> ImGuiBackend:
    """Return a real ImGui backend and context without an OpenGL window or renderer."""
    from imgui_bundle import imgui, imguizmo

    context = imgui.create_context()
    # A previous native viewer test may have left another context current.
    imgui.set_current_context(context)
    test.addCleanup(imgui.destroy_context, context)
    ui = ImGuiBackend.__new__(ImGuiBackend)
    ui.is_available, ui.imgui, ui.giz, ui._context = True, imgui, imguizmo.im_guizmo, context
    ui._workspace_style, ui.key_press_filter, ui.dpi_scale = True, None, 1.0
    ui.io = imgui.get_io()
    ui.io.set_ini_filename("")
    ui.io.display_size = (640, 480)
    # Build draw lists without a GPU renderer.
    ui.io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui._frame_time = ui._frame_recovery_state = ui._pending_file_dialog = None
    ui.section_errors = {}
    return ui


@requires("imgui_bundle")
class TestViewerUI(unittest.TestCase):
    def setUp(self) -> None:
        # Exercise ImGui state without an OpenGL renderer or window.
        self.ui = _offscreen_imgui_backend(self)

    def test_custom_joint_property_storage(self) -> None:
        """Native editors honor custom owners, flattened rows, and logical joint indices."""
        ui = self.ui.imgui
        self.ui.dpi_scale = 1.0
        self.ui.io.mouse_double_click_time = 0.0
        builder = newton.ModelBuilder()
        for i in range(4):
            body = builder.add_link(mass=1, inertia=wp.mat33(np.eye(3)))
            joint = builder.add_joint_rod(-1, body) if i == 3 else builder.add_joint_revolute(-1, body)
            if i < 2:
                builder.add_articulation([joint])
        model = builder.finalize(device="cpu")
        for name, frequency, width in (
            ("joint_q", Frequency.JOINT_COORD, 1),
            ("joint_qd", Frequency.JOINT_DOF, 1),
            ("offset", Frequency.JOINT_DOF, 1),
            ("samples", Frequency.JOINT_DOF, 2),
            ("targets", Frequency.JOINT_DOF, 2),
        ):
            count = model._attribute_frequency_count(frequency)
            references = Frequency.BODY if name == "targets" else None
            values = (
                np.arange(count * width, dtype=np.int32) % model.body_count
                if references is not None
                else np.arange(count * width, dtype=np.float32) + 10
            )
            model.add_attribute(
                name,
                wp.array(values, device="cpu"),
                frequency,
                Assignment.STATE,
                namespace="probe",
                references=references,
            )
            if width > 1:
                model._set_attribute_spec(
                    f"probe:{name}",
                    model.AttributeSpec(frequency, Assignment.STATE, row_width=width, references=references),
                )
        state = model.state()
        ws = ViewerWorkspace.__new__(ViewerWorkspace)
        ws.model = model
        ws.viewer = SimpleNamespace(is_paused=lambda: True, _last_state=state, _inspection_components={})
        ws.history = ws._undo = ws._drag_undo = None
        ws._bindings = ws._bindings_components = None
        ws._cache, ws._live_cache, ws._drafts = {}, {}, {}
        ws.gui = SimpleNamespace(ui=self.ui)
        ws.widgets = widgets = PhysicsWidgets(ws.gui)
        ws.tree = ModelTree(model)
        inspector = PropertyInspector(ws)
        number = widgets.number
        core_q, core_qd, body_q = (
            state.joint_q.numpy().copy(),
            state.joint_qd.numpy().copy(),
            state.body_q.numpy().copy(),
        )
        rows = []

        def capture_number(label: str, value: float, **kwargs: object) -> PropertyEdit:
            edit = number(label, value, **kwargs)
            lo, hi = ui.get_item_rect_min(), ui.get_item_rect_max()
            rows.append((value, kwargs.get("editable", False), ((lo.x + hi.x) / 2, (lo.y + hi.y) / 2)))
            return edit

        for case, (name, joints) in enumerate(
            (
                ("joint_q", [0]),
                ("joint_qd", [0]),
                ("joint_q", [0, 1]),
                ("joint_qd", [0, 1]),
                ("offset", [0]),
                ("offset", [2]),  # Unarticulated joints still own writable custom state.
                ("offset", [3]),  # Rod material state is independent of forward kinematics.
                ("samples", [1]),
                ("samples", [0, 1]),  # Exercise the compact page prefetch path too.
                ("samples", [3]),
                ("targets", [3]),
            )
        ):
            with self.subTest(name=name, joints=joints):
                ws.selected = ("joint", joints[-1])
                ws.selection = {("joint", joint) for joint in joints}
                qualified = f"probe:{name}"
                spec = model._attribute_spec(qualified)
                array = getattr(state.probe, name)
                original = array.numpy().copy()
                offsets = model.joint_q_start if spec.frequency == Frequency.JOINT_COORD else model.joint_qd_start
                spans = offsets.numpy()
                expected = np.concatenate(
                    [original[spans[j] * spec.row_width : spans[j + 1] * spec.row_width] for j in joints]
                )
                domain = AttributeRows(model, spec.frequency, joints, joints=True)

                def draw(
                    case: int = case,
                    field: tuple[AttributeRows, str, newton.Model.AttributeSpec, object, str, wp.array] = (
                        domain,
                        qualified,
                        spec,
                        state.probe,
                        name,
                        array,
                    ),
                ) -> None:
                    rows.clear()
                    self.ui.begin_frame()
                    ui.set_next_window_pos((0, 0))
                    ui.set_next_window_size((620, 460))
                    ui.begin("##custom_joint_fields", flags=ui.WindowFlags_.no_title_bar)
                    ui.push_id(case)
                    with patch.object(widgets, "number", capture_number):
                        inspector._fields([field], Assignment.STATE)
                    ui.pop_id()
                    ui.end()
                    ui.end_frame()

                draw()
                with patch.object(widgets, "compound", wraps=widgets.compound) as compounds:
                    draw()
                np.testing.assert_array_equal([row[0] for row in rows], expected)
                self.assertTrue(all(row[1] == (spec.references is None) for row in rows))
                if spec.row_width > 1 and joints == [3]:
                    labels = [call.args[0] for call in compounds.call_args_list]
                    for dof in range(int(spans[3]), int(spans[4])):
                        self.assertIn(f"joint dof {dof}", labels)
                if spec.references is not None:
                    continue
                x, y = rows[0][2]
                self.ui.io.add_mouse_pos_event(x, y)
                draw()
                self.ui.io.add_mouse_button_event(0, True)
                draw()
                self.ui.io.add_mouse_pos_event(x + 20, y)
                draw()
                self.ui.io.add_mouse_button_event(0, False)
                draw()
                self.assertFalse(np.array_equal(array.numpy(), original))
                np.testing.assert_array_equal(state.joint_q.numpy(), core_q)
                np.testing.assert_array_equal(state.joint_qd.numpy(), core_qd)
                np.testing.assert_array_equal(state.body_q.numpy(), body_q)
                self.assertIs(ws._undo[0], state.probe)
                self.assertTrue(ws.apply_edit(*ws._undo[:4], joint_index=ws._undo[4], save_undo=False))
                np.testing.assert_array_equal(array.numpy(), original)

    def test_timeline_next_steps_at_live_end(self) -> None:
        """The real next-frame button traverses history, then requests live physics."""
        ui = self.ui.imgui
        self.ui.dpi_scale = 1.0
        builder = newton.ModelBuilder()
        builder.add_body()
        model = builder.finalize(device="cpu")
        viewer = ViewerNull()
        viewer.set_model(model)
        history = FrameHistory(viewer)
        state = model.state()
        for time in (0.0, 1.0):
            _record_frame(history, time, state)
        self.assertEqual(len(history.frames), 2)
        timeline = TimelineView(history)
        viewer = ViewerGL.__new__(ViewerGL)
        viewer.__dict__.update(vars(ViewerNull()))
        viewer._paused, viewer._step_requested = True, False
        viewer.gui = SimpleNamespace(ui=self.ui)
        widgets = PhysicsWidgets(viewer.gui)
        button = widgets.icon_button
        next_position = None

        def capture_button(label: str, *args: object, **kwargs: object) -> bool:
            nonlocal next_position
            clicked = button(label, *args, **kwargs)
            if label == "##next_timeline_frame":
                lo, hi = ui.get_item_rect_min(), ui.get_item_rect_max()
                next_position = ((lo.x + hi.x) / 2, (lo.y + hi.y) / 2)
            return clicked

        def draw() -> None:
            self.ui.begin_frame()
            ui.set_next_window_pos((0, 0))
            ui.set_next_window_size((620, 100))
            ui.begin("##transport", flags=ui.WindowFlags_.no_title_bar)
            with mock.patch.object(widgets, "icon_button", capture_button):
                timeline._draw_transport(ui, viewer, widgets)
            ui.end()
            ui.end_frame()

        history.select(0)
        draw()
        draw()
        self.ui.io.add_mouse_pos_event(*next_position)
        draw()
        for live in (False, True):
            self.ui.io.add_mouse_button_event(0, True)
            draw()
            self.ui.io.add_mouse_button_event(0, False)
            draw()
            self.assertEqual(history.index, 1)
            self.assertEqual(viewer._step_requested, live)

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

    def test_modifier_reset_on_focus_loss(self) -> None:
        """Forget a modifier key whose release was missed while the window was unfocused."""
        import_pyglet_window(shadow_window=False)
        from pyglet.window import key

        self.ui.window = mock.MagicMock()
        self.ui._attach_input()
        handlers = self.ui.window.push_handlers.call_args.kwargs
        self.ui.io.config_input_trickle_event_queue = False

        handlers["on_key_press"](key.LCTRL, 0)
        handlers["on_deactivate"]()
        self.ui.io.clear_input_keys()
        handlers["on_key_press"](key.RCTRL, 0)
        handlers["on_key_release"](key.RCTRL, 0)
        self.ui.begin_frame()
        self.ui.imgui.end_frame()
        self.assertFalse(self.ui.io.key_ctrl)

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

    def test_legacy_style_preserves_main_metrics(self) -> None:
        """Legacy controls keep main's spacing and colors across DPI changes."""
        self.ui._workspace_style = False
        for scale in (1.0, 1.5, 2.0, 1.0):
            with self.subTest(scale=scale):
                self.ui.dpi_scale = scale
                self.ui._apply_dpi_scaling()
                style = self.ui.imgui.get_style()
                self.assertEqual(tuple(style.window_padding), (12 * scale, 12 * scale))
                self.assertEqual(tuple(style.frame_padding), (6 * scale, 6 * scale))
                self.assertEqual(tuple(style.item_spacing), (12 * scale, 6 * scale))
                self.assertEqual(style.frame_rounding, 0)
                self.assertEqual(style.window_border_size, 0)
                np.testing.assert_allclose(
                    tuple(style.color_(self.ui.imgui.Col_.window_bg)), (20 / 255, 22 / 255, 26 / 255, 1)
                )
                np.testing.assert_allclose(
                    tuple(style.color_(self.ui.imgui.Col_.check_mark)), (127 / 255, 131 / 255, 1, 1)
                )

    def test_clipped_property_widgets(self) -> None:
        """Keep scroll layout and native editors while omitting offscreen decoration."""
        ui = self.ui.imgui
        self.ui.dpi_scale = 1.0
        widgets = PhysicsWidgets(SimpleNamespace(ui=self.ui))
        values = [float(i) for i in range(30)]

        def draw(*, editable: bool = False, scroll: float | None = None) -> tuple[list[tuple], float]:
            self.ui.begin_frame()
            ui.set_next_window_pos((20, 20))
            ui.set_next_window_size((300, 150))
            ui.begin("##clipped_properties", flags=ui.WindowFlags_.no_title_bar | ui.WindowFlags_.no_saved_settings)
            if scroll is not None:
                ui.set_scroll_y(scroll)
            rows = []
            draw_list = ui.get_window_draw_list()
            for i, value in enumerate(values):
                vertices = draw_list.vtx_buffer.size()
                edit = widgets.number(f"Field {i}", value, editable=editable, speed=1.0)
                values[i] = edit.value
                rows.append(
                    (
                        ui.is_item_visible(),
                        ui.is_item_active(),
                        draw_list.vtx_buffer.size() - vertices,
                        tuple(ui.get_item_rect_min()),
                        tuple(ui.get_item_rect_max()),
                    )
                )
            extent = ui.get_scroll_max_y()
            ui.end()
            ui.end_frame()
            return rows, extent

        draw()
        rows, extent = draw()
        self.assertGreater(extent, 0)
        self.assertTrue(rows[0][0])
        self.assertFalse(rows[-1][0])
        self.assertTrue(all(vertices == 0 for visible, _, vertices, _, _ in rows if not visible))
        # Scrolling changes clipping, not row spacing or total content height.
        draw(scroll=extent)
        scrolled, scrolled_extent = draw()
        self.assertAlmostEqual(extent, scrolled_extent)
        self.assertFalse(scrolled[0][0])
        self.assertTrue(scrolled[-1][0])
        self.assertAlmostEqual(rows[-1][3][1] - scrolled[-1][3][1], extent)
        self.assertTrue(all(vertices == 0 for visible, _, vertices, _, _ in scrolled if not visible))

        # A live drag must still update when scrolling moves its row offscreen.
        draw(editable=True, scroll=0)
        rows, _ = draw(editable=True)
        lo, hi = rows[0][3:]
        x, y = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
        self.ui.io.add_mouse_pos_event(x, y)
        draw(editable=True)
        self.ui.io.add_mouse_button_event(0, True)
        rows, _ = draw(editable=True)
        self.assertTrue(rows[0][1])
        draw(editable=True, scroll=extent)
        self.ui.io.add_mouse_pos_event(x + 20, y)
        rows, _ = draw(editable=True)
        self.assertFalse(rows[0][0])
        self.assertTrue(rows[0][1])
        self.assertGreater(values[0], 0)
        self.ui.io.add_mouse_button_event(0, False)
        draw(editable=True)

    def test_orientation_projection_reuse(self) -> None:
        """Reuse idle geometry, but reproject immediately after orbit, layout, or font changes."""
        ui = self.ui.imgui
        camera = Camera()
        camera.yaw, camera.pitch = -135.0, -30.0
        workspace = SimpleNamespace(
            gui=SimpleNamespace(ui=self.ui),
            viewer=SimpleNamespace(camera=camera),
            widgets=SimpleNamespace(scale=1.0),
            rect=(0, 0, 640, 480),
        )
        controls = SimpleNamespace(
            workspace=workspace,
            prepare_popup=lambda: ui.push_style_var(ui.StyleVar_.window_padding, ui.ImVec2(8, 8)),
        )
        gizmo = OrientationGizmo(controls)

        def draw() -> None:
            self.ui.begin_frame()
            try:
                gizmo.draw()
            finally:
                ui.end_frame()

        draw()
        faces = gizmo._faces
        camera.pos += camera.get_front()  # Translation cannot change orientation geometry.
        draw()
        self.assertIs(gizmo._faces, faces)
        for change in ("orbit", "layout", "scale", "up_axis", "font"):
            with self.subTest(change=change):
                if change == "orbit":
                    camera.orbit(10.0, 5.0)
                elif change == "layout":
                    workspace.rect = (20, 30, 600, 420)
                elif change == "scale":
                    workspace.widgets.scale = 1.5
                elif change == "up_axis":
                    camera.up_axis = 1
                else:
                    ui.get_style().font_scale_main = 1.2
                draw()
                self.assertIsNot(gizmo._faces, faces)
                faces = gizmo._faces
                x, y, width, _ = gizmo.rect
                scale = workspace.widgets.scale
                center = np.array((x + width * 0.5, y + 56 * scale))
                basis = np.array((camera.get_right(), -camera.get_up()))
                np.testing.assert_allclose(gizmo._axis_heads, center + basis.T * (44 * scale))
                for _, _, _, _, vertices, polygon in faces:
                    np.testing.assert_allclose(polygon, center + vertices @ basis.T * (19.5 * scale))
                draw()
                self.assertIs(gizmo._faces, faces)

    def test_workspace_panels_draw_across_selection_kinds(self) -> None:
        """Render PropertyInspector, ViewportControls, CollisionInspector, and Environment in ImGui without errors."""
        from newton.actuators import BatteryConstantResistance, DriveOutputSpace, DrivePD, MotorDC  # noqa: PLC0415

        ui = self.ui.imgui
        builder = newton.ModelBuilder()
        body0 = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
        variant_a = newton.ModelBuilder()
        variant_a.add_shape_sphere(variant_a.add_body(), radius=0.3)
        variant_b = newton.ModelBuilder()
        variant_b.add_shape_box(variant_b.add_body(), hx=0.2, hy=0.2, hz=0.2)
        builder.add_body_variants(body0, [variant_a, variant_b], initial=0, labels=["sphere", "box"])
        body1 = builder.add_link(xform=wp.transform(wp.vec3(0.5, 0.0, 1.0), wp.quat_identity()))
        builder.add_shape_sphere(body1, radius=0.3)
        j0 = builder.add_joint_revolute(-1, body0)
        j1 = builder.add_joint_revolute(body0, body1)
        builder.add_articulation([j0, j1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            builder.add_constraint_mimic(j1, j0, coef0=0.0, coef1=1.0)
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 1.0, 0.5),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=2,
            dim_y=2,
            cell_x=0.2,
            cell_y=0.2,
            mass=0.1,
        )
        builder.add_soft_grid(
            pos=wp.vec3(1.0, 1.0, 0.5),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=1,
            dim_y=1,
            dim_z=1,
            cell_x=0.2,
            cell_y=0.2,
            cell_z=0.2,
            density=100.0,
            k_mu=1000.0,
            k_lambda=1000.0,
            k_damp=1.0,
        )
        builder.add_rod(
            rod=newton.Rod([(0.0, 2.0, 1.0), (0.2, 2.0, 1.0), (0.4, 2.0, 1.0)], radius=0.02),
            label="rod_draw",
        )
        builder.add_deformable_visual_mesh(
            vertices=[(0.1, 1.1, 0.5), (0.3, 1.1, 0.5), (0.1, 1.3, 0.5)],
            indices=[0, 1, 2],
            kind="triangle",
            surface=0,
            label="skin_draw",
        )
        builder.add_attachment_particle_body(0, body0, body_point=wp.vec3(0.0, 0.0, 0.0))
        builder.add_tendon(
            [
                newton.TendonGuide(body=body0, offset=wp.vec3(0.0, 0.0, 0.1), guide_type=newton.TendonGuideType.ANCHOR),
                newton.TendonGuide(
                    body=body1, offset=wp.vec3(0.0, 0.0, 0.1), guide_type=newton.TendonGuideType.ANCHOR, rest_length=0.5
                ),
            ],
        )
        act_idx = builder.add_actuator(
            DrivePD,
            index=0,
            kp=10.0,
            kd=1.0,
            output_space=DriveOutputSpace.DUTY,
            motor=(MotorDC, {"kt": 0.5, "resistance": 2.0, "supply_from_battery": True}),
        )
        builder.add_battery(
            BatteryConstantResistance,
            actuators=[act_idx],
            nominal_voltage=24.0,
            resistance=0.1,
            min_voltage=18.0,
        )
        model = builder.finalize(device="cpu")
        state = model.state()
        control = model.control()
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", rigid_contact_max=16, soft_contact_max=8)
        contacts = pipeline.contacts()

        viewer = ViewerNull()
        viewer.objects = {}
        viewer.contact_selection_only = False
        viewer.show_contact_labels = False
        viewer.contact_normal_scale = 1.0
        viewer.contact_force_scale = 1.0
        viewer._contact_force_scale_default = 1.0
        viewer.contact_viz_scale = 1.0
        viewer.contact_mode_eps_force = 1e-3
        viewer.contact_mode_eps_velocity = 1e-3
        viewer.renderer = SimpleNamespace(
            contact_disk_scale=1.0,
            contact_arrow_scale=1.0,
            contact_normal_max_pixels=64.0,
            contact_force_max_pixels=64.0,
        )

        class _StubSolver(newton.solvers.SolverBase):
            def step(self, *args, **kwargs):
                pass

        viewer.set_model(model)
        viewer.set_control(control)
        viewer.set_solver(_StubSolver(model))
        viewer.set_collision_pipeline(pipeline)
        viewer.log_state(state)
        viewer.log_contacts(contacts, state)
        viewer.is_paused = lambda: True

        gui = SimpleNamespace(ui=self.ui, _viewer=viewer, console_capture=None)
        ws = ViewerWorkspace(gui)
        ws.model = model
        ws.tree = ModelTree(model)
        ws.collision.attach(ws.tree)
        ws.property_inspector = PropertyInspector(ws)

        surface_key = ws.tree.deformables.object_key_for_model("surface", 0, ws.tree)
        volume_key = ws.tree.deformables.object_key_for_model("volume", 0, ws.tree)
        curve_key = ws.tree.deformables.object_key_for_model("curve", 0, ws.tree)
        self.assertIsNotNone(surface_key)
        self.assertIsNotNone(volume_key)
        self.assertIsNotNone(curve_key)
        selections = (
            ("scene", 0),
            ("world", 0),
            ("root", -1),
            ("body", 0),
            ("shape", 0),
            ("joint", 0),
            surface_key,
            volume_key,
            curve_key,
            ("deformable_visual_mesh", 0),
            ("tendon", 0),
            ("tendon_guide", 0),
            ("tendon_segment", 0),
            ("attachment_particle_body", 0),
            ("constraint_mimic", 0),
            ("actuator_collection", 0),
            ("actuator_group", 0),
            ("battery", 0),
        )
        for key in selections:
            with self.subTest(selection=key):
                ws.selected = key
                ws.selection = {key}
                for component in (Assignment.MODEL, Assignment.STATE, Assignment.CONTROL):
                    self.ui.begin_frame()
                    try:
                        ui.begin("TestProperties")
                        try:
                            ws.property_inspector.component = component
                            ws.property_inspector.draw()
                        finally:
                            ui.end()
                    finally:
                        ui.end_frame()

        self.ui.begin_frame()
        try:
            ui.begin("TestAuxiliaryPanels")
            try:
                ws._environment()
                ws._solver_properties()
                ws.collision.draw(ws.property_inspector)
                ws.viewport_controls._contact_settings()
            finally:
                ui.end()
        finally:
            ui.end_frame()

    def test_example_browser_tree_view(self) -> None:
        """Render the Examples browser as a searchable folder tree reflecting directory hierarchy."""
        ui = self.ui.imgui
        example_tree = {
            "basic": [
                ("basic_pendulum", "newton.examples.basic.example_basic_pendulum"),
                ("basic_shapes", "newton.examples.basic.example_basic_shapes"),
            ],
            "contacts": [
                ("adaptive_collision", "newton.examples.contacts.example_adaptive_collision"),
                ("contacts_rj45_plug", "newton.examples.contacts.example_contacts_rj45_plug"),
            ],
            "nested": [
                ("nested_demo", "newton.examples.nested.sub.example_nested_demo"),
            ],
            "surface_gripper": [
                ("surface_gripper", "newton.examples.surface_gripper.example_surface_gripper"),
            ],
        }
        selected: list[str] = []
        viewer = ViewerNull()
        gui = SimpleNamespace(
            ui=self.ui,
            _viewer=viewer,
            console_capture=None,
            _example_tree=example_tree,
            _example_current="newton.examples.basic.example_basic_shapes",
            select_example=selected.append,
        )
        ws = ViewerWorkspace(gui)

        ws._rebuild_example_rows(example_tree)
        # Active example's folder auto-expands; other folders start collapsed.
        self.assertIn("basic", ws._example_expanded)
        self.assertNotIn("contacts", ws._example_expanded)
        kinds_labels = [(row[0], row[1], row[2]) for row in ws._example_rows]
        self.assertEqual(
            kinds_labels,
            [
                ("folder", 0, "basic"),
                ("example", 1, "pendulum"),
                ("example", 1, "shapes"),
                ("folder", 0, "contacts"),
                ("folder", 0, "nested"),
                ("folder", 0, "surface_gripper"),
            ],
        )

        # Expanding all folders reflects nested folder depth and trims only redundant <folder>_ prefixes.
        ws._example_expanded.update(ws._example_all_folders)
        ws._rebuild_example_rows(example_tree)
        by_path = {row[3]: (row[1], row[2]) for row in ws._example_rows if row[0] == "example"}
        self.assertEqual(by_path["newton.examples.contacts.example_adaptive_collision"], (1, "adaptive_collision"))
        self.assertEqual(by_path["newton.examples.contacts.example_contacts_rj45_plug"], (1, "rj45_plug"))
        self.assertEqual(by_path["newton.examples.nested.sub.example_nested_demo"], (2, "nested_demo"))
        self.assertEqual(by_path["newton.examples.surface_gripper.example_surface_gripper"], (1, "surface_gripper"))

        # Search prunes non-matching folders and expands matching ones regardless of manual collapse state.
        ws._example_expanded.clear()
        ws.example_search = "rj45"
        ws._rebuild_example_rows(example_tree)
        self.assertEqual(
            [(row[0], row[1], row[2]) for row in ws._example_rows],
            [("folder", 0, "contacts"), ("example", 1, "rj45_plug")],
        )
        self.assertEqual(ws._example_stats, (5, 1, 6))

        # Render both filtered and unfiltered tree frames in ImGui without error.
        for query in ("", "shapes"):
            ws.example_search = query
            self.ui.begin_frame()
            try:
                ui.begin("TestExamples")
                ws._example_browser(example_tree)
                ui.end()
            finally:
                ui.end_frame()

    def test_timeline_live_readouts_and_compact_height_without_recording(self) -> None:
        """Track live frame/time with recording off and collapse the timeline track until recording or frames exist."""
        ui = self.ui.imgui
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_sphere(body, radius=0.1)
        model = builder.finalize(device="cpu")
        viewer = ViewerNull()
        viewer.set_model(model)
        viewer.gui = SimpleNamespace(ui=self.ui)
        viewer.is_paused = lambda: False

        history = FrameHistory(viewer)
        history.set_recording(False)
        timeline = TimelineView(history)
        self.assertFalse(timeline.has_track)
        self.assertIsNone(timeline._live_readout(viewer))

        # Logging frames with recording off tracks live frame/time and dt with zero recorded frames.
        for step in range(4):
            history.begin_frame(step * 0.02)
            history.end_frame()
        self.assertEqual(len(history.frames), 0)
        self.assertEqual(history.live_frame, 3)
        self.assertAlmostEqual(history.live_time, 0.06, places=6)
        self.assertAlmostEqual(history.dt, 0.02, places=6)
        self.assertFalse(timeline.has_track)
        live = timeline._live_readout(viewer, now=10.0)
        self.assertEqual(live, (3, 0.06))

        # Render compact mode (recording off, no frames) cleanly in ImGui.
        self.ui.begin_frame()
        try:
            ui.set_next_window_size(ui.ImVec2(800, TimelineView.compact_height))
            ui.begin("CompactTimeline")
            try:
                timeline.draw_gui(ui, viewer, embedded=True)
            finally:
                ui.end()
        finally:
            ui.end_frame()

        # Enabling recording expands the track; disabling and clearing collapses it again while keeping live readout.
        history.set_recording(True)
        self.assertTrue(timeline.has_track)
        state = model.state()
        history.begin_frame(0.08)
        history.state(state)
        history.end_frame()
        self.assertEqual(len(history.frames), 1)
        history.set_recording(False)
        self.assertTrue(timeline.has_track)
        history.clear()
        self.assertFalse(timeline.has_track)
        self.assertEqual(history.live_frame, 4)
        self.assertAlmostEqual(history.live_time, 0.08, places=6)

    def test_transform_and_matrix_component_alignment(self) -> None:
        """Keep transform Position/Rotation and sibling vec3/matrix3 component grids aligned on the left edge."""
        ui = self.ui.imgui
        w = PhysicsWidgets(SimpleNamespace(ui=self.ui))
        tag_x: list[tuple[str, float]] = []
        original_tag = w._component_tag

        def record_tag(text: str) -> None:
            tag_x.append((text, float(ui.get_cursor_screen_pos().x)))
            original_tag(text)

        w._component_tag = record_tag  # type: ignore[method-assign]
        self.ui.begin_frame()
        try:
            ui.set_next_window_size(ui.ImVec2(280, 600))
            ui.begin("AlignmentCheck")
            try:
                base_x = float(ui.get_cursor_screen_pos().x)
                w.transform("joint_target_q", np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32))
                transform_tags = list(tag_x)
                tag_x.clear()

                w.vec3("body_com", np.zeros(3, dtype=np.float32))
                w.matrix3("body_inertia", np.eye(3, dtype=np.float32))
                sibling_tags = list(tag_x)
            finally:
                ui.end()
        finally:
            ui.end_frame()

        # Position ("X") and Rotation ("X", "Z" in 2x2 grid) share the exact same single compound indent.
        pos_x = transform_tags[0][1]
        rot_x = transform_tags[3][1]
        rot_z = transform_tags[5][1]
        self.assertGreater(pos_x, base_x)
        self.assertAlmostEqual(pos_x, rot_x, places=4)
        self.assertAlmostEqual(pos_x, rot_z, places=4)

        # Sibling vec3 ("X") and matrix3 ("XX", "YX", "ZX") start at the exact same left margin.
        vec_x = sibling_tags[0][1]
        mat_xx = sibling_tags[3][1]
        mat_yx = sibling_tags[6][1]
        mat_zx = sibling_tags[9][1]
        self.assertAlmostEqual(vec_x, base_x, places=4)
        self.assertAlmostEqual(mat_xx, vec_x, places=4)
        self.assertAlmostEqual(mat_yx, vec_x, places=4)
        self.assertAlmostEqual(mat_zx, vec_x, places=4)

    def test_log_observables_and_generic_solver_inspector(self) -> None:
        """Render logged solver observables generically across Solver, State, Contacts, and FrameHistory."""
        from dataclasses import dataclass  # noqa: PLC0415

        from newton._src.viewer.gui.properties import PropertyInspector  # noqa: PLC0415
        from newton._src.viewer.gui.workspace import ViewerWorkspace  # noqa: PLC0415
        from newton._src.viewer.viewer_model_tree import ModelTree  # noqa: PLC0415
        from newton.viewer import ViewerNull  # noqa: PLC0415

        Frequency = newton.Model.AttributeFrequency
        Assignment = newton.Model.AttributeAssignment

        class _MockDiagSolver(newton.solvers.SolverBase):
            supports_collision_pipeline = True
            SUPPORTED_OBSERVABLES = frozenset(
                {
                    newton.solvers.SolverBase.ObservableKind.SOLVER_RESIDUAL,
                    "solver_residual_history",
                    newton.solvers.SolverBase.ObservableKind.CONTACT_F,
                    newton.solvers.SolverBase.ObservableKind.BODY_QDD,
                    "body_awake",
                    "joint_impulse",
                }
            )

            @dataclass(eq=False)
            class Observables(newton.solvers.SolverBase.Observables):
                solver_residual_history: wp.array[wp.float32] | None = newton.solvers.SolverBase.Observables.field(
                    dtype=wp.float32, frequency=Frequency.WORLD
                )
                body_awake: wp.array[wp.bool] | None = newton.solvers.SolverBase.Observables.field(
                    dtype=wp.bool, frequency=Frequency.BODY
                )
                joint_impulse: wp.array[wp.float32] | None = newton.solvers.SolverBase.Observables.field(
                    dtype=wp.float32, frequency=Frequency.JOINT_DOF
                )

            def __init__(
                self,
                model: newton.Model,
                iterations: int = 4,
                *,
                collision_pipeline: newton.CollisionPipeline | None = None,
            ) -> None:
                super().__init__(model, collision_pipeline=collision_pipeline)
                self.iterations = iterations
                # Solver-private attributes must never be probed by the viewer.
                self.mjw_model = SimpleNamespace(opt=SimpleNamespace(iterations=99, tolerance=1e-6))

            def _observable_shape(self, kind: str, rows: int) -> tuple[int, ...]:
                if kind == "solver_residual_history":
                    return (self.iterations, rows)
                return super()._observable_shape(kind, rows)

            def step(
                self,
                state_in: newton.State,
                state_out: newton.State,
                control: newton.Control | None,
                contacts: newton.Contacts | None,
                dt: float,
                *,
                observables: newton.solvers.SolverBase.Observables | None = None,
            ) -> None:
                del state_in, state_out, control, dt
                resolved = self._resolve_step_contacts(contacts)
                self.validate_observables(observables, resolved)

        builder = newton.ModelBuilder()
        b0 = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        b1 = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_shape_box(b0, hx=0.1, hy=0.1, hz=0.1)
        builder.add_shape_box(b1, hx=0.1, hy=0.1, hz=0.1)
        j0 = builder.add_joint_free(b0)
        j1 = builder.add_joint_revolute(parent=b0, child=b1, axis=(0.0, 0.0, 1.0))
        builder.add_articulation([j0, j1])
        model = builder.finalize(device="cpu")
        other_model = builder.finalize(device="cpu")
        state = model.state()
        pipeline = newton.CollisionPipeline(model, contact_matching="latest")
        contacts = pipeline.contacts()
        solver = _MockDiagSolver(model, iterations=4, collision_pipeline=pipeline)
        solver.set_collision_frequency(
            collision_frequency={newton.solvers.SolverBase.CollisionSlot.RIGID: 2},
            collision_frequency_type={
                newton.solvers.SolverBase.CollisionSlot.RIGID: newton.solvers.SolverBase.CollisionFrequencyType.ITERATIONS
            },
        )

        # Validate ViewerBase.log_observables() error handling and layer isolation.
        uninitialized = ViewerNull()
        with self.assertRaisesRegex(RuntimeError, "set_model"):
            uninitialized.log_observables(None)

        viewer = ViewerNull(num_frames=10)
        viewer.activate("primary")
        viewer.set_model(model)
        with self.assertRaisesRegex(TypeError, "Observables"):
            viewer.log_observables(object())  # type: ignore[arg-type]
        other_obs = _MockDiagSolver(other_model).observables({newton.solvers.SolverBase.ObservableKind.SOLVER_RESIDUAL})
        with self.assertRaisesRegex(ValueError, "viewer's model"):
            viewer.log_observables(other_obs)

        viewer.log_state(state)
        viewer.is_paused = lambda: True

        gui = SimpleNamespace(ui=self.ui, _viewer=viewer, console_capture=None)
        ws = ViewerWorkspace(gui)
        ws.model = model
        ws.tree = ModelTree(model)
        ws.collision.contacts = contacts
        ws.property_inspector = PropertyInspector(ws)
        ui = self.ui.imgui

        # Empty state when neither set_solver() nor log_observables() was called.
        self.ui.begin_frame()
        try:
            ui.begin("TestSolverEmpty")
            try:
                ws._solver_properties()
            finally:
                ui.end()
        finally:
            self.ui.end_frame()

        viewer.set_solver(solver)
        # Stepping the solver alone does NOT implicitly expose observables to the viewer.
        obs = solver.observables(solver.supported_observables)
        solver.step(state, state, None, None, 1.0 / 60.0, observables=obs)
        self.assertIsNone(ws.logged_observables())

        group_calls: list[tuple[str, bool]] = []
        orig_group = ws.widgets.group

        def spy_group(title: str, *, default_open: bool = True):
            group_calls.append((title, default_open))
            return orig_group(title, default_open=default_open)

        rendered_numbers: list[tuple[str, float, bool]] = []
        rendered_arrays: list[tuple[str, list[float], bool]] = []
        rendered_texts: list[tuple[str, str]] = []
        rendered_spatial: list[tuple[str, list[float], bool]] = []

        orig_number = ws.widgets.number
        orig_array = ws.widgets.array
        orig_text = ws.widgets.text_value
        orig_spatial = ws.widgets.spatial_vector

        def spy_number(label, value, **kwargs):
            rendered_numbers.append((label, float(value), bool(kwargs.get("editable", False))))
            return orig_number(label, value, **kwargs)

        def spy_array(label, values, **kwargs):
            rendered_arrays.append(
                (label, np.asarray(values, dtype=float).reshape(-1).tolist(), bool(kwargs.get("editable", False)))
            )
            return orig_array(label, values, **kwargs)

        def spy_text(label, value, **kwargs):
            rendered_texts.append((label, str(value)))
            return orig_text(label, value, **kwargs)

        def spy_spatial(label, value, **kwargs):
            rendered_spatial.append(
                (
                    label,
                    np.asarray(value, dtype=float).reshape(-1).tolist(),
                    bool(kwargs.get("force", False)),
                )
            )
            return orig_spatial(label, value, **kwargs)

        with (
            mock.patch.object(ws.widgets, "group", side_effect=spy_group),
            mock.patch.object(ws.widgets, "number", side_effect=spy_number),
            mock.patch.object(ws.widgets, "text_value", side_effect=spy_text),
        ):
            self.ui.begin_frame()
            try:
                ui.begin("TestSolverUnlogged")
                try:
                    ws._solver_properties()
                finally:
                    ui.end()
            finally:
                self.ui.end_frame()
        self.assertIn(("Collision schedule", True), group_calls)
        self.assertIn(("Rigid", "Iterations (every 2)"), rendered_texts)
        self.assertIn(("Observables", False), group_calls)
        # Solver-specific attributes (iterations=99, tolerance=1e-6) are never probed.
        self.assertEqual(rendered_numbers, [])

        # Populate observables and explicitly log them via viewer.log_observables(obs).
        # Because bodies were added without begin_world(), they live in global world -1 (slot == model.world_count).
        obs.solver_residual.assign(np.array([0.0, 0.125], dtype=np.float32))
        obs.solver_residual_history.assign(
            np.array([[0.0, 0.5], [0.0, 0.25], [0.0, 0.125], [0.0, 0.0625]], dtype=np.float32)
        )
        obs.body_awake.assign(np.array([True, False], dtype=bool))
        obs.body_qdd.assign(
            np.array([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3], [0.0, 0.0, -9.81, 0.0, 0.0, 0.0]], dtype=np.float32)
        )
        obs.joint_impulse.assign(np.arange(model.joint_dof_count, dtype=np.float32) + 10.0)
        obs_contact_f = np.zeros((obs.contact_f.shape[0], 6), dtype=np.float32)
        obs_contact_f[0] = [0.0, 0.0, 42.0, 1.0, 0.0, 0.0]
        obs_contact_f[5] = [999.0, 999.0, 999.0, 0.0, 0.0, 0.0]  # Beyond active count; must never be read!
        obs.contact_f.assign(obs_contact_f)
        solver.contacts.rigid_contact_count.assign(np.array([1], dtype=np.int32))

        viewer.log_observables(obs)
        # Subsequent log_contacts(..., observables=None) must not clobber log_observables(obs).
        viewer.log_contacts(solver.contacts, state)
        self.assertIs(ws.logged_observables(), obs)

        # Layer isolation: another layer does not inherit the primary layer's observables.
        primary_layer = viewer.layer.layer_id
        viewer.activate("secondary")
        viewer.set_model(other_model)
        self.assertIsNone(viewer._last_observables)
        viewer.activate(primary_layer)
        self.assertIs(ws.logged_observables(), obs)

        ws._live_cache.clear()
        group_calls.clear()
        rendered_numbers.clear()
        rendered_arrays.clear()
        rendered_texts.clear()
        rendered_spatial.clear()

        with (
            mock.patch.object(ws.widgets, "group", side_effect=spy_group),
            mock.patch.object(ws.widgets, "number", side_effect=spy_number),
            mock.patch.object(ws.widgets, "array", side_effect=spy_array),
            mock.patch.object(ws.widgets, "text_value", side_effect=spy_text),
            mock.patch.object(ws.widgets, "spatial_vector", side_effect=spy_spatial),
        ):
            self.ui.begin_frame()
            try:
                ui.begin("TestSolverLogged")
                try:
                    ws._solver_properties()
                finally:
                    ui.end()
            finally:
                self.ui.end_frame()

        self.assertIn(("Observables", True), group_calls)
        self.assertIn(("solver_residual", 0.125, False), rendered_numbers)
        self.assertTrue(
            any(
                label == "solver_residual_history" and np.allclose(vals, [0.5, 0.25, 0.125, 0.0625]) and not ed
                for label, vals, ed in rendered_arrays
            )
        )
        # Only the 1 active contact row (42 N) is rendered; inactive slot 5 (999 N) is never read.
        self.assertTrue(
            any(np.allclose(vals, [0.0, 0.0, 42.0, 1.0, 0.0, 0.0]) and is_f for _, vals, is_f in rendered_spatial)
        )
        self.assertFalse(any(abs(vals[0] - 999.0) < 1e-3 for _, vals, _ in rendered_spatial))

        # When 0 contacts are active, contact_f reports "0 active contacts".
        solver.contacts.rigid_contact_count.assign(np.array([0], dtype=np.int32))
        ws._live_cache.clear()
        rendered_texts.clear()
        with mock.patch.object(ws.widgets, "text_value", side_effect=spy_text):
            self.ui.begin_frame()
            try:
                ui.begin("TestSolverZeroContacts")
                try:
                    ws._solver_properties()
                finally:
                    ui.end()
            finally:
                self.ui.end_frame()
        self.assertIn(("contact_f", "0 active contacts"), rendered_texts)

        # PropertyInspector surfaces JOINT_DOF observables on selected joints, BODY observables on
        # selected bodies, and 2D WORLD observables on selected world/root nodes as read-only.
        ws.selected, ws.selection = ("joint", j1), {("joint", j1)}
        ws.property_inspector._contexts()
        joint_obs = ws.property_inspector._observables()
        self.assertEqual([entry[1] for entry in joint_obs], ["joint_impulse"])
        rendered_numbers.clear()
        with mock.patch.object(ws.widgets, "number", side_effect=spy_number):
            self.ui.begin_frame()
            try:
                ui.begin("TestJointObservables")
                try:
                    ws.property_inspector.component = Assignment.STATE
                    ws.property_inspector.draw()
                finally:
                    ui.end()
            finally:
                self.ui.end_frame()
        self.assertIn(("joint_impulse", 16.0, False), rendered_numbers)

        ws.selected, ws.selection = ("body", b0), {("body", b0)}
        ws.property_inspector._contexts()
        rendered_spatial.clear()
        with mock.patch.object(ws.widgets, "spatial_vector", side_effect=spy_spatial):
            self.ui.begin_frame()
            try:
                ui.begin("TestBodyObservables")
                try:
                    ws.property_inspector.component = Assignment.STATE
                    ws.property_inspector.draw()
                finally:
                    ui.end()
            finally:
                self.ui.end_frame()
        self.assertTrue(
            any(
                label == "body_qdd" and np.allclose(vals, [1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
                for label, vals, _ in rendered_spatial
            )
        )

        ws.selected, ws.selection = ("root", -1), {("root", -1)}
        ws.property_inspector._contexts()
        self.assertEqual([entry[1] for entry in ws.property_inspector._observables_2d()], ["solver_residual_history"])

        # FrameHistory records and replays log_observables() snapshots (including active contact counts and None clears).
        history = FrameHistory(viewer)
        solver.contacts.rigid_contact_count.assign(np.array([1], dtype=np.int32))
        obs.solver_residual.assign(np.array([0.0, 0.25], dtype=np.float32))
        obs_contact_f[0] = [0.0, 0.0, 42.0, 1.0, 0.0, 0.0]
        obs.contact_f.assign(obs_contact_f)
        history.begin_frame(0.0)
        history.state(state)
        viewer.log_observables(history.observables(obs))
        history.end_frame()

        history.begin_frame(1.0)
        history.state(state)
        viewer.log_observables(history.observables(None))
        history.end_frame()
        self.assertIsNone(ws.logged_observables())

        obs.solver_residual.assign(np.array([0.0, 0.05], dtype=np.float32))
        obs_contact_f[0] = [0.0, 0.0, 12.0, 0.0, 0.0, 0.0]
        obs.contact_f.assign(obs_contact_f)
        history.begin_frame(2.0)
        history.state(state)
        viewer.log_observables(history.observables(obs))
        history.end_frame()
        self.assertIs(ws.logged_observables(), obs)

        ws.history = history
        history.select(0)
        history.begin_frame(2.0)
        replayed_0 = history.observables(obs)
        viewer.log_observables(replayed_0)
        history.end_frame()
        self.assertIsNotNone(replayed_0)
        self.assertIsNot(replayed_0, obs)
        np.testing.assert_allclose(replayed_0.solver_residual.numpy(), [0.0, 0.25])
        np.testing.assert_allclose(replayed_0.contact_f.numpy()[0], [0.0, 0.0, 42.0, 1.0, 0.0, 0.0])
        # Caller's live array remains untouched during replay.
        np.testing.assert_allclose(obs.solver_residual.numpy(), [0.0, 0.05])

        history.select(1)
        history.begin_frame(2.0)
        replayed_1 = history.observables(obs)
        viewer.log_observables(replayed_1)
        history.end_frame()
        self.assertIsNone(replayed_1)
        self.assertIsNone(ws.logged_observables())

        history.select(2)
        history.begin_frame(2.0)
        live_2 = history.observables(obs)
        viewer.log_observables(live_2)
        history.end_frame()
        self.assertIs(live_2, obs)
        self.assertIs(ws.logged_observables(), obs)


def _make_gl_state(paused: bool = False, step_requested: bool = False) -> ViewerGL:
    # Lightweight stand-in with just the fields ViewerGL.should_step() needs.
    return SimpleNamespace(_paused=paused, _step_requested=step_requested, _history=None)  # type: ignore[return-value]


class TestViewerBaseShouldStep(unittest.TestCase):
    """ViewerBase.should_step() defaults to not self.is_paused()."""

    def test_returns_true_when_not_paused(self):
        viewer = ViewerNull()
        self.assertTrue(viewer.should_step())

    def test_returns_true_on_repeated_calls(self):
        viewer = ViewerNull()
        for _ in range(3):
            self.assertTrue(viewer.should_step())


@requires("imgui_bundle")
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
        gui.ui._context = context
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

    @requires("pyglet")
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
        gui._loading_splash_active = False
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

    @requires("pyglet")
    def test_camera_deceleration_is_capped(self):
        """Check that the camera deceleration is capped to avoid flipping the camera velocity direction."""
        camera = Camera(pos=(0.0, 0.0, 0.0))
        viewer = SimpleNamespace(camera=camera, camera_speed=2.0, is_rendering_paused=lambda: False)
        gui = ViewerGui.__new__(ViewerGui)
        gui._viewer = viewer
        gui.ui = None
        gui._loading_splash_active = False
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
        viewer = SimpleNamespace(
            world_offsets=offsets, layer=SimpleNamespace(xform=wp.transform_identity()), _last_state=None
        )
        ws = SimpleNamespace(viewer=viewer)
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
        workspace.model, workspace.tree = model, None
        workspace._bindings = workspace._bindings_components = None
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

    def test_workspace_read_empty_array_slices(self) -> None:
        """Preserve trailing dimensions and dtype when reading empty Warp arrays."""
        workspace = ViewerWorkspace.__new__(ViewerWorkspace)
        workspace._cache = {}
        workspace._live_cache = {}
        holder = SimpleNamespace()
        for shape in ((0,), (2, 0)):
            with self.subTest(empty_shape=shape):
                holder.data = wp.empty(shape, dtype=int, device="cpu")
                workspace._cache.clear()
                empty = workspace.read(holder, "data", 0)
                self.assertEqual(empty.shape, (min(shape[0], 1), *shape[1:]))
                self.assertEqual(empty.dtype, np.int32)


class TestViewerGLShouldStep(unittest.TestCase):
    """ViewerGL.should_step() state machine: running, paused, and single-step."""

    @requires("imgui_bundle", "pyglet")
    def test_gui_popup_owns_viewer_shortcuts(self) -> None:
        """Keep popup keys from closing or changing the simulation without keyboard capture.

        Drive real ImGui popup and text-input state with real pyglet key codes.
        """
        import_pyglet_window(shadow_window=False)
        from pyglet.window import key

        ui = _offscreen_imgui_backend(self)
        imgui = ui.imgui
        # Closing releases the GUI renderer; a window without a GL context skips that.
        ui.window = SimpleNamespace(context=None)
        viewer = SimpleNamespace(_paused=False, _step_requested=False, show_ui=True, _display_options_user={})
        gui = ViewerGui.__new__(ViewerGui)
        gui._viewer, gui.ui, gui.show_ui = viewer, ui, True
        gui.workspace = gui.console_capture = None
        gui._loading_splash_active = False
        framed = []
        gui.frame_camera_on_model = lambda: framed.append(True)
        values = {"name": "abc", "value": 1.0}

        def frame(popup: str | None = None, field: str | None = None) -> None:
            """Submit one ImGui frame with an optional open popup or focused input field."""
            ui.begin_frame()
            imgui.begin("Inputs")
            if popup == "open":
                imgui.open_popup("menu")
            if imgui.begin_popup("menu"):
                if popup == "close":
                    imgui.close_current_popup()
                imgui.end_popup()
            if field is not None:
                imgui.set_keyboard_focus_here()
            if field == "name":
                _, values["name"] = imgui.input_text("##name", values["name"])
            elif field == "value":
                _, values["value"] = imgui.input_float("##value", values["value"])
            imgui.end()
            ui.end_frame()

        closed = []
        frame(popup="open")
        self.assertTrue(imgui.is_popup_open("", imgui.PopupFlags_.any_popup))
        for symbol in (key.ESCAPE, key.SPACE, key.PERIOD, key.H, key.F):
            gui.handle_key_press(symbol, close_fn=lambda: closed.append(True))
        self.assertFalse(closed)
        self.assertFalse(framed)
        self.assertFalse(viewer._paused)
        self.assertFalse(viewer._step_requested)
        self.assertTrue(gui.show_ui)

        frame(popup="close")
        frame()
        self.assertFalse(imgui.is_popup_open("", imgui.PopupFlags_.any_popup))
        gui.handle_key_press(key.SPACE)
        self.assertTrue(viewer._paused)
        self.assertTrue(viewer._display_options_user["_paused"])
        gui.handle_key_press(key.PERIOD)
        self.assertTrue(viewer._step_requested)
        gui.handle_key_press(key.ESCAPE, close_fn=lambda: closed.append(True))
        self.assertEqual(closed, [True])
        self.assertFalse(ui.is_available)
        ui.is_available = True

        # String entry owns Space; numeric entry keeps viewer shortcuts but
        # retains its decimal point and Escape cancellation.
        for field, ignored in (("name", True), ("value", False)):
            with self.subTest(field=field):
                for _ in range(3):
                    frame(field=field)
                self.assertTrue(ui.io.want_text_input)
                self.assertEqual(gui._numeric_text_input_active(), not ignored)
                paused = viewer._paused
                gui.handle_key_press(key.SPACE)
                self.assertEqual(viewer._paused, paused if ignored else not paused)
                viewer._paused, viewer._step_requested = True, False
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
        _history=None,
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
        viewer.set_solver(SolverXPBD(model))
        ws = ViewerWorkspace.__new__(ViewerWorkspace)
        ws.model, ws.viewer = model, viewer
        ws._bindings = ws._bindings_components = None
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
        viewer.set_solver(None)
        ws.history = SimpleNamespace(replaying=True)
        ws._live_cache.clear()
        shown = ws.joint_display_state()
        np.testing.assert_allclose(shown.joint_q.numpy(), -expected_q, atol=1e-6)
        np.testing.assert_allclose(shown.joint_qd.numpy(), -expected_qd, atol=1e-6)
        np.testing.assert_array_equal(state.joint_q.numpy(), 0.0)
        np.testing.assert_array_equal(state.joint_qd.numpy(), 0.0)
        ws.history = None

        # Custom loops register persistent components independently and log
        # the current state after each buffer swap, without an Example object.
        control = model.control()
        solver = SolverXPBD(model)
        pipeline = newton.CollisionPipeline(model)
        viewer.set_control(control)
        viewer.set_solver(solver)
        viewer.set_collision_pipeline(pipeline)
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

        # Replacing or clearing one component must preserve the others.
        replacement = model.control()
        viewer.set_control(replacement)
        self.assertIs(ws.bindings.control, replacement)
        self.assertIs(ws.bindings.solver, solver)
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

        # Layers retain separate components and logged states. Invalid model
        # associations must not replace a valid existing component.
        original_layer = viewer.layer.layer_id
        viewer.activate("other")
        for setter in (viewer.set_control, viewer.set_solver, viewer.set_collision_pipeline):
            with self.assertRaisesRegex(RuntimeError, "set_model"):
                setter(None)
        other_model = builder.finalize(device="cpu")
        viewer.set_model(other_model)
        ws.model = other_model
        self.assertIsNone(ws.bindings.control)
        for setter, component in (
            (viewer.set_solver, solver),
            (viewer.set_collision_pipeline, pipeline),
            # Objects without a model get the documented error, not AttributeError.
            (viewer.set_solver, object()),
        ):
            with self.assertRaisesRegex(ValueError, "active model"):
                setter(component)
        viewer.set_control(other_model.control())
        self.assertIsNot(ws.bindings.control, control)
        viewer.activate(original_layer)
        ws.model = model
        self.assertIs(ws.bindings.control, control)
        self.assertIs(ws.bindings.solver, solver)
        self.assertIs(viewer._last_state, historical)

        # A rebuilt Model with identical array dimensions starts without the
        # previous Model's components or logged State.
        rebuilt = builder.finalize(device="cpu")
        viewer.set_model(rebuilt)
        ws.model = rebuilt
        self.assertIsNone(ws.bindings.control)
        self.assertIsNone(viewer._last_state)

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

    def test_history_records_logged_frames(self) -> None:
        """Snapshot logged State and the inspected Control, and replay them without touching live data."""
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
        # CPU NumPy views alias live storage; CUDA readbacks own host storage.
        for device in get_test_devices(mode="basic"):
            with self.subTest(device=device):
                model = builder.finalize(device=device)
                model.add_attribute(
                    "pairs",
                    wp.array(np.arange(6, dtype=np.float32).reshape(3, 2), dtype=float, device=device),
                    Frequency.ONCE,
                    Assignment.STATE,
                    namespace="probe",
                )
                viewer = ViewerNull()
                viewer.set_model(model)
                history = FrameHistory(viewer)
                self.assertIs(history.model, model)
                # A frame that logs nothing records nothing.
                history.begin_frame(0.0)
                history.end_frame()
                self.assertFalse(history.frames)

                state, control = model.state(), model.control()
                viewer.set_control(control)
                state.solver_scratch = wp.zeros(1, device=device)
                _record_frame(history, 0.0, state)
                snapshot = history.frames[0]
                key = next(iter(snapshot.states))
                self.assertNotIn("solver_scratch", snapshot.states[key])
                state.probe.sample.fill_(7.0)
                control.probe.drive.fill_(11.0)
                if device.is_cuda:
                    gc.collect()
                np.testing.assert_array_equal(snapshot.states[key]["probe:sample"], [3.0])
                np.testing.assert_array_equal(snapshot.control["probe:drive"], [5.0])
                # Repeated renders at an unchanged time record nothing, unless
                # an edit asks to replace the newest frame.
                _record_frame(history, 0.0, state)
                self.assertEqual(len(history.frames), 1)
                history.replace_next_frame()
                _record_frame(history, 0.0, state)
                self.assertEqual(len(history.frames), 1)
                np.testing.assert_array_equal(history.frames[0].states[key]["probe:sample"], [7.0])
                state.probe.sample.fill_(9.0)
                _record_frame(history, 1.0, state)
                self.assertEqual(len(history.frames), 2)

                history.select(0)
                self.assertTrue(history.replaying)
                history.begin_frame(1.0)
                shown = history.state(state)
                history.end_frame()
                self.assertIsNot(shown, state)
                np.testing.assert_array_equal(shown.probe.sample.numpy(), [7.0])
                # Multidimensional storage keeps its full shape.
                self.assertEqual(shown.probe.pairs.shape, (3, 2))
                np.testing.assert_array_equal(shown.probe.pairs.numpy(), np.arange(6).reshape(3, 2))
                np.testing.assert_array_equal(history.restore_control().probe.drive.numpy(), [11.0])
                # Replay never records or alters the live objects.
                self.assertEqual(len(history.frames), 2)
                np.testing.assert_array_equal(state.probe.sample.numpy(), [9.0])
                np.testing.assert_array_equal(control.probe.drive.numpy(), [11.0])

    def test_history_replays_each_logged_call(self) -> None:
        """Match replayed States, contact States and points to their own logging calls within a frame."""
        builder = newton.ModelBuilder()
        builder.add_body(xform=wp.transform(wp.vec3(100, 0, 0), wp.quat_identity()))
        builder.add_shape_sphere(0, radius=0.1)
        builder.add_particle(wp.vec3(0, 100, 0), wp.vec3(0), 1.0)
        for device in get_test_devices(mode="basic"):
            with self.subTest(device=device):
                model = builder.finalize(device=device)
                viewer = ViewerNull()
                viewer.set_model(model)
                history = FrameHistory(viewer)
                state, display, contact_state = model.state(), model.state(), model.state()
                # A partial State logged for display keeps only its own arrays.
                display.particle_q = None
                contacts = newton.Contacts(4, 0, device=device)
                points = wp.array([(1.0, 2.0, 3.0)], dtype=wp.vec3, device=device)

                def log(
                    time: float,
                    history: FrameHistory = history,
                    state: newton.State = state,
                    display: newton.State = display,
                    contacts: newton.Contacts = contacts,
                    contact_state: newton.State = contact_state,
                    points: wp.array = points,
                ) -> tuple[newton.State, newton.State, newton.Contacts, newton.State, object]:
                    history.begin_frame(time)
                    shown = history.state(state)
                    shown_display = history.state(display)
                    shown_contacts, shown_contact_state, _ = history.contacts(contacts, contact_state, None)
                    shown_points = history.arrays("points", "markers", {"points": points, "radii": 0.1})
                    history.end_frame()
                    return shown, shown_display, shown_contacts, shown_contact_state, shown_points

                log(0.0)
                state.body_q.fill_(wp.transform(wp.vec3(200, 0, 0), wp.quat_identity()))
                display.body_q.fill_(wp.transform(wp.vec3(300, 0, 0), wp.quat_identity()))
                contact_state.body_q.fill_(wp.transform(wp.vec3(400, 0, 0), wp.quat_identity()))
                points.fill_(wp.vec3(4.0, 5.0, 6.0))
                log(1.0)
                frame = history.frames[0]
                self.assertEqual(len(frame.states), 2)
                self.assertEqual(set(frame.arrays), {("points", viewer._active_layer_id, "markers")})

                history.select(0)
                shown, shown_display, shown_contacts, shown_contact_state, shown_points = log(1.0)
                np.testing.assert_array_equal(shown.body_q.numpy()[0, :3], [100, 0, 0])
                np.testing.assert_array_equal(shown_display.body_q.numpy()[0, :3], [100, 0, 0])
                self.assertIsNone(getattr(shown_display, "particle_q", None))
                np.testing.assert_array_equal(shown_contact_state.body_q.numpy()[0, :3], [100, 0, 0])
                self.assertIsNot(shown_contacts, contacts)
                self.assertEqual(shown_contacts.rigid_contact_count.numpy()[0], 0)
                np.testing.assert_array_equal(shown_points["points"].numpy(), [[1.0, 2.0, 3.0]])
                self.assertEqual(shown_points["radii"], 0.1)
                # Live arguments are untouched.
                np.testing.assert_array_equal(state.body_q.numpy()[0, :3], [200, 0, 0])
                np.testing.assert_array_equal(contact_state.body_q.numpy()[0, :3], [400, 0, 0])
                np.testing.assert_array_equal(points.numpy(), [[4.0, 5.0, 6.0]])
                # Contacts logged with the frame's State alias it rather than copying it again.
                history.select(1)
                history.begin_frame(2.0)
                history.state(state)
                history.contacts(contacts, state, None)
                history.end_frame()
                record = next(iter(history.frames[-1].contact_records.values()))
                self.assertIsNotNone(record.state_key)
                self.assertFalse(record.state)

                # Replay rebuilds logged observables through the solver that
                # produced them, without the optional set_solver() binding,
                # without retaining live Observables/Contacts buffers, and across
                # contact-capacity reductions and changing observable kinds/solvers.
                import gc  # noqa: PLC0415
                import weakref  # noqa: PLC0415

                solver = newton.solvers.SolverXPBD(model)
                solver_alt = newton.solvers.SolverXPBD(model)
                large_contacts = newton.Contacts(8, 3, device=device)
                large_contacts.rigid_contact_count.assign(np.array([2], dtype=np.int32))
                large_contacts.soft_contact_count.assign(np.array([2], dtype=np.int32))
                rigid_pts = np.zeros((8, 3), dtype=np.float32)
                rigid_pts[:2] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
                large_contacts.rigid_contact_point0.assign(rigid_pts)
                soft_pts = np.zeros((3, 3), dtype=np.float32)
                soft_pts[:2] = [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
                large_contacts.soft_contact_body_pos.assign(soft_pts)

                obs_contact_only = solver.observables(
                    contacts=large_contacts,
                    kinds={newton.solvers.SolverBase.ObservableKind.CONTACT_F},
                )
                cf_0 = np.zeros((len(obs_contact_only.contact_f), 6), dtype=np.float32)
                cf_0[0, 0] = 11.0
                cf_0[1, 0] = 22.0
                cf_0[8, 0] = 33.0
                cf_0[9, 0] = 44.0
                obs_contact_only.contact_f.assign(cf_0)

                obs_expanded = solver.observables(
                    contacts=large_contacts,
                    kinds={
                        newton.solvers.SolverBase.ObservableKind.CONTACT_F,
                        newton.solvers.SolverBase.ObservableKind.BODY_PARENT_F,
                    },
                )
                obs_expanded.contact_f.assign(cf_0 * 2.0)
                obs_expanded.body_parent_f.assign(np.array([[1.5, 2.5, 3.5, 0.5, 0.25, 0.125]], dtype=np.float32))

                obs_alt_solver = solver_alt.observables(
                    contacts=large_contacts,
                    kinds={newton.solvers.SolverBase.ObservableKind.BODY_PARENT_F},
                )
                obs_alt_solver.body_parent_f.assign(np.array([[9.0, 8.0, 7.0, 0.0, 0.0, 0.0]], dtype=np.float32))

                self.assertIsNone(viewer._inspection_components.get("solver"))
                observed = FrameHistory(viewer)
                observed.begin_frame(0.0)
                observed.contacts(large_contacts, state, obs_contact_only)
                observed.end_frame()

                observed.begin_frame(1.0)
                observed.contacts(large_contacts, state, obs_expanded)
                observed.end_frame()

                observed.begin_frame(2.0)
                observed.contacts(large_contacts, state, obs_alt_solver)
                observed.end_frame()

                # Release all original source Contacts and Observables buffers before replay.
                contacts_ref = weakref.ref(large_contacts)
                obs_0_ref = weakref.ref(obs_contact_only)
                obs_1_ref = weakref.ref(obs_expanded)
                obs_2_ref = weakref.ref(obs_alt_solver)
                del large_contacts, obs_contact_only, obs_expanded, obs_alt_solver
                gc.collect()
                self.assertIsNone(contacts_ref())
                self.assertIsNone(obs_0_ref())
                self.assertIsNone(obs_1_ref())
                self.assertIsNone(obs_2_ref())

                # Replace live contacts with a smaller capacity buffer (1 rigid, 1 soft)
                # and verify both initial replay allocation and growth across frames.
                tiny_contacts = newton.Contacts(1, 1, device=device)
                observed.begin_frame(3.0)
                observed.contacts(tiny_contacts, state, None)
                observed.end_frame()

                # First scrub to frame 2 (allocates display buffer with 2 active contacts despite tiny_contacts=(1, 1)).
                observed.select(2)
                observed.begin_frame(3.0)
                shown_contacts_2, _, shown_obs_2 = observed.contacts(tiny_contacts, state, None)
                observed.end_frame()
                self.assertEqual(int(shown_contacts_2.rigid_contact_count.numpy()[0]), 2)
                self.assertIsNotNone(shown_obs_2)
                self.assertIs(shown_obs_2._solver, solver_alt)
                self.assertEqual(shown_obs_2.kinds, frozenset({newton.solvers.SolverBase.ObservableKind.BODY_PARENT_F}))
                np.testing.assert_allclose(shown_obs_2.body_parent_f.numpy(), [[9.0, 8.0, 7.0, 0.0, 0.0, 0.0]])

                # Scrub to frame 0 (CONTACT_F only) and verify active rigid & soft rows and forces.
                observed.select(0)
                observed.begin_frame(3.0)
                shown_contacts_0, _, shown_obs_0 = observed.contacts(tiny_contacts, state, None)
                observed.end_frame()
                self.assertIsNotNone(shown_obs_0)
                self.assertIs(shown_obs_0._solver, solver)
                self.assertIs(shown_obs_0.contacts, shown_contacts_0)
                self.assertEqual(shown_obs_0.kinds, frozenset({newton.solvers.SolverBase.ObservableKind.CONTACT_F}))
                self.assertEqual(int(shown_contacts_0.rigid_contact_count.numpy()[0]), 2)
                self.assertEqual(int(shown_contacts_0.soft_contact_count.numpy()[0]), 2)
                np.testing.assert_allclose(
                    shown_contacts_0.rigid_contact_point0.numpy()[:2], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
                )
                np.testing.assert_allclose(
                    shown_contacts_0.soft_contact_body_pos.numpy()[:2], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
                )
                np.testing.assert_allclose(shown_obs_0.contact_f.numpy()[:2, 0], [11.0, 22.0])
                soft_start = shown_contacts_0.rigid_contact_max
                np.testing.assert_allclose(shown_obs_0.contact_f.numpy()[soft_start : soft_start + 2, 0], [33.0, 44.0])

                # Scrub to frame 1 (CONTACT_F + BODY_PARENT_F) and verify cache reallocates for new kinds.
                observed.select(1)
                observed.begin_frame(3.0)
                shown_contacts_1, _, shown_obs_1 = observed.contacts(tiny_contacts, state, None)
                observed.end_frame()
                self.assertIsNotNone(shown_obs_1)
                self.assertIs(shown_obs_1.contacts, shown_contacts_1)
                self.assertEqual(
                    shown_obs_1.kinds,
                    frozenset(
                        {
                            newton.solvers.SolverBase.ObservableKind.CONTACT_F,
                            newton.solvers.SolverBase.ObservableKind.BODY_PARENT_F,
                        }
                    ),
                )
                np.testing.assert_allclose(shown_obs_1.contact_f.numpy()[:2, 0], [22.0, 44.0])
                np.testing.assert_allclose(shown_obs_1.body_parent_f.numpy(), [[1.5, 2.5, 3.5, 0.5, 0.25, 0.125]])

                # Also verify growth when an unobservable ContactRecord shrinks capacity:
                shrink_history = FrameHistory(viewer)
                big_plain = newton.Contacts(4, 2, device=device)
                big_plain.rigid_contact_count.assign(np.array([3], dtype=np.int32))
                big_plain.soft_contact_count.assign(np.array([2], dtype=np.int32))
                shrink_history.begin_frame(0.0)
                shrink_history.contacts(big_plain, state, None)
                shrink_history.end_frame()
                shrink_history.begin_frame(1.0)
                shrink_history.contacts(tiny_contacts, state, None)
                shrink_history.end_frame()
                shrink_history.begin_frame(2.0)
                shrink_history.contacts(tiny_contacts, state, None)
                shrink_history.end_frame()
                # Scrub to frame 1 first (allocates 1x1), then scrub to frame 0 (must grow to >= 3x2).
                shrink_history.select(1)
                shrink_history.begin_frame(2.0)
                shrink_history.contacts(tiny_contacts, state, None)
                shrink_history.end_frame()
                shrink_history.select(0)
                shrink_history.begin_frame(2.0)
                grown_contacts, _, _ = shrink_history.contacts(tiny_contacts, state, None)
                shrink_history.end_frame()
                self.assertGreaterEqual(grown_contacts.rigid_contact_max, 3)
                self.assertGreaterEqual(grown_contacts.soft_contact_max, 2)
                self.assertEqual(int(grown_contacts.rigid_contact_count.numpy()[0]), 3)
                self.assertEqual(int(grown_contacts.soft_contact_count.numpy()[0]), 2)

                # Each layer's calls are filtered by that layer's own Model, not
                # the Model active when history was created.
                layered_builder = newton.ModelBuilder()
                layered_builder.add_body()
                layered_builder.add_shape_sphere(0, radius=0.1)
                layered_builder.add_custom_attribute(
                    newton.ModelBuilder.CustomAttribute(
                        name="sample",
                        namespace="probe",
                        assignment=Assignment.STATE,
                        frequency=Frequency.ONCE,
                        dtype=float,
                        default=3.0,
                    )
                )
                layered_model = layered_builder.finalize(device=device)
                layered = ViewerNull()
                layered.activate("first")
                layered.set_model(model)
                layered.activate("second")
                layered.set_model(layered_model)
                layered.activate("first")
                layered_history = FrameHistory(layered)
                layered_state = layered_model.state()
                layered_contacts = newton.Contacts(4, 0, device=device)
                layered_observables = newton.solvers.SolverXPBD(layered_model).observables(
                    contacts=layered_contacts, kinds={newton.solvers.SolverBase.ObservableKind.CONTACT_F}
                )
                layered_history.begin_frame(0.0)
                layered_history.state(state)
                layered.activate("second")
                layered_history.state(layered_state)
                layered_history.contacts(layered_contacts, layered_model.state(), layered_observables)
                layered.activate("first")
                layered_history.end_frame()
                layered_frame = layered_history.frames[0]
                self.assertNotIn("probe:sample", layered_frame.states["state", "first", 0])
                self.assertIn("probe:sample", layered_frame.states["state", "second", 0])
                record = layered_frame.contact_records["contacts", "second", 0]
                self.assertTrue(record.observables)
                self.assertIn("probe:sample", record.state)

                # Replay reproduces cleared and populated batches regardless of the live arguments.
                markers = FrameHistory(viewer)
                for time, logged in ((0.0, points), (1.0, None), (2.0, points)):
                    markers.begin_frame(time)
                    markers.arrays("points", "markers", {"points": logged, "radii": 0.1})
                    markers.end_frame()
                for index, live, expected in ((0, None, [[4.0, 5.0, 6.0]]), (1, points, None)):
                    markers.select(index)
                    markers.begin_frame(3.0)
                    shown_points = markers.arrays("points", "markers", {"points": live, "radii": 0.1})
                    markers.end_frame()
                    if expected is None:
                        self.assertIsNone(shown_points["points"])
                    else:
                        np.testing.assert_array_equal(shown_points["points"].numpy(), expected)

                # Every logged array counts toward the bound before readback.
                first = next(iter(frame.states.values()))
                limited = FrameHistory(viewer)
                # Room for the first State only; later calls must stop before any readback.
                limited.memory_limit = 2 * sum(values.nbytes for values in first.values()) + 1
                with patch(
                    "newton._src.viewer.viewer_history._snapshot_array", wraps=viewer_history._snapshot_array
                ) as snapshot:
                    limited.begin_frame(0.0)
                    limited.state(state)
                    limited.state(display)
                    limited.contacts(contacts, contact_state, None)
                    limited.end_frame()
                    self.assertEqual(snapshot.call_count, len(first))
                self.assertFalse(limited.frames)
                self.assertTrue(limited._recording_blocked)

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
        ws.viewer = SimpleNamespace(_last_state=model.state(), _inspection_components={"control": model.control()})
        ws._bindings = ws._bindings_components = None
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
        inspector.clear()
        inspector._contexts()
        with patch.object(model, "_iter_attribute_specs", wraps=model._iter_attribute_specs) as discover:
            for _ in range(2):
                for component in (Assignment.STATE, Assignment.CONTROL, Assignment.MODEL):
                    inspector._collect(component)
            self.assertEqual(discover.call_count, 1, "Unchanged property metadata must not be rebuilt every frame")

        # Only metadata is cached: state swaps, optional arrays and both kinds
        # of late solver registration must remain visible without reselecting.
        replacement = model.state()
        replacement.body_q = None
        ws.viewer._last_state = replacement
        fields = {f[1]: f for group in inspector._collect(Assignment.STATE).values() for f in group}
        self.assertNotIn("body_q", fields)
        self.assertIs(fields["body_qd"][-1], replacement.body_qd)
        model.add_attribute("late_value", [4], Frequency.BODY)
        fields = {f[1]: f for group in inspector._collect(Assignment.MODEL).values() for f in group}
        self.assertIn("late_value", fields)
        model.attribute_frequency["late_value"] = Frequency.SHAPE
        self.assertNotIn("late_value", {f[1] for g in inspector._collect(Assignment.MODEL).values() for f in g})
        model.attribute_frequency["late_value"] = Frequency.BODY
        model.attribute_assignment["late_value"] = Assignment.STATE
        replacement.late_value = [9]
        self.assertNotIn("late_value", {f[1] for g in inspector._collect(Assignment.MODEL).values() for f in g})
        fields = {f[1]: f for group in inspector._collect(Assignment.STATE).values() for f in group}
        self.assertEqual(fields["late_value"][-1], [9])
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

    def test_logged_solver_observables_follow_selection(self) -> None:
        """List logged per-body observables for selected bodies and none for other domains."""
        builder = newton.ModelBuilder()
        bodies = [builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3))) for _ in range(2)]
        for body in bodies:
            builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device="cpu")
        solver = newton.solvers.SolverXPBD(model)
        observables = solver.observables(kinds={solver.ObservableKind.BODY_PARENT_F})
        ws = ViewerWorkspace.__new__(ViewerWorkspace)
        ws.model, ws.selected, ws.selection = model, ("body", bodies[1]), {("body", bodies[1])}
        ws.viewer = SimpleNamespace(_last_state=model.state(), _last_contact_observables=None)
        ws._cache, ws._live_cache = {}, {}
        inspector = PropertyInspector(ws)
        inspector._contexts()
        self.assertEqual(inspector._observables(), [])

        ws.viewer._last_contact_observables = observables
        fields = inspector._observables()
        self.assertEqual([(field[1], list(field[0].indices)) for field in fields], [("body_parent_f", [bodies[1]])])
        self.assertIs(fields[0][3], observables)
        self.assertIs(fields[0][5], observables.body_parent_f)

        ws.selected, ws.selection = ("shape", 0), {("shape", 0)}
        inspector._contexts()
        self.assertEqual(inspector._observables(), [])

        # Observables from another Model are never matched to this selection.
        other = builder.finalize(device="cpu")
        ws.viewer._last_contact_observables = newton.solvers.SolverXPBD(other).observables(
            kinds={solver.ObservableKind.BODY_PARENT_F}
        )
        ws.selected, ws.selection = ("body", bodies[0]), {("body", bodies[0])}
        inspector._contexts()
        self.assertEqual(inspector._observables(), [])

    def test_deformable_objects_and_visual_meshes(self) -> None:
        """Expose authored surfaces, volumes, curves, edges, and skinned meshes in tree, search, and inspector."""
        from newton._src.viewer.viewer_attributes import attribute_summary  # noqa: PLC0415

        builder = newton.ModelBuilder()
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 2.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=1,
            dim_y=1,
            cell_x=1.0,
            cell_y=1.0,
            mass=1.0,
            label="cloth_a",
        )
        builder.add_soft_grid(
            pos=wp.vec3(3.0, 0.0, 2.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=1,
            dim_y=1,
            dim_z=1,
            cell_x=0.5,
            cell_y=0.5,
            cell_z=0.5,
            density=100.0,
            k_mu=1000.0,
            k_lambda=1000.0,
            k_damp=1.0,
            label="soft_b",
        )
        rod_bodies, rod_joints = builder.add_rod(
            rod=newton.Rod([(0.0, 2.0, 1.0), (0.2, 2.0, 1.0), (0.4, 2.0, 1.0)], radius=0.02),
            label="rod_c",
        )
        builder.add_deformable_visual_mesh(
            vertices=[(0.1, 0.1, 2.0), (0.4, 0.1, 2.0), (0.1, 0.4, 2.0)],
            indices=[0, 1, 2],
            kind="triangle",
            surface=0,
            label="skin_d",
        )
        model = builder.finalize(device="cpu")
        tree = ModelTree(model)
        ws = ViewerWorkspace.__new__(ViewerWorkspace)
        ws.model, ws.tree = model, tree
        ws.viewer = SimpleNamespace(_last_state=model.state(), _inspection_components={})
        ws._bindings = ws._bindings_components = None
        ws.history = None
        ws._cache, ws._live_cache = {}, {}
        inspector = PropertyInspector(ws)

        surface_key = tree.deformables.object_key_for_model("surface", 0, tree)
        volume_key = tree.deformables.object_key_for_model("volume", 0, tree)
        self.assertIsNotNone(surface_key)
        self.assertIsNotNone(volume_key)
        self.assertEqual(tree.labels[surface_key], "cloth_a")
        self.assertEqual(tree.labels[volume_key], "soft_b")
        self.assertEqual(tree.labels[("curve", 0)], "rod_c")
        self.assertEqual(tree.labels[("deformable_visual_mesh", 0)], "skin_d")
        self.assertEqual(
            tree.shape_descendants(("curve", 0)), sorted({s for b in rod_bodies for s in model.body_shapes.get(b, ())})
        )
        self.assertEqual(
            tree.joint_descendants(("curve", 0)),
            list(range(int(model.curve_joint_start.numpy()[0]), int(model.curve_joint_end.numpy()[0]))),
        )
        self.assertTrue(set(rod_joints) <= set(tree.joint_descendants(("curve", 0))))

        # Surface selection exposes SURFACE, PARTICLE, TRIANGLE, and EDGE domains.
        ws.selected, ws.selection = surface_key, {surface_key}
        inspector._contexts()
        surface_freqs = {d.frequency for d in inspector.domains}
        self.assertTrue({Frequency.SURFACE, Frequency.PARTICLE, Frequency.TRIANGLE, Frequency.EDGE} <= surface_freqs)

        # Volume selection exposes VOLUME, PARTICLE, and TETRAHEDRON domains.
        ws.selected, ws.selection = volume_key, {volume_key}
        inspector._contexts()
        volume_freqs = {d.frequency for d in inspector.domains}
        self.assertTrue({Frequency.VOLUME, Frequency.PARTICLE, Frequency.TETRAHEDRON} <= volume_freqs)

        # Curve selection exposes CURVE, BODY, JOINT, JOINT_COORD, and JOINT_DOF domains.
        ws.selected, ws.selection = ("curve", 0), {("curve", 0)}
        inspector._contexts()
        curve_freqs = {d.frequency for d in inspector.domains}
        self.assertTrue(
            {Frequency.CURVE, Frequency.BODY, Frequency.JOINT, Frequency.JOINT_COORD, Frequency.JOINT_DOF}
            <= curve_freqs
        )

        # Reference navigation resolves authored deformables directly.
        self.assertEqual(inspector.reference_target(model, Frequency.SURFACE, 0), surface_key)
        self.assertEqual(inspector.reference_target(model, Frequency.VOLUME, 0), volume_key)
        self.assertEqual(inspector.reference_target(model, Frequency.CURVE, 0), ("curve", 0))
        self.assertEqual(inspector.reference_target(model, Frequency.EDGE, 0), ("edge", 0))

        # Search finds surfaces, volumes, curves, and skinned meshes by label.
        for query, expected_key in (
            ("cloth_a", surface_key),
            ("soft_b", volume_key),
            ("rod_c", ("curve", 0)),
            ("skin_d", ("deformable_visual_mesh", 0)),
        ):
            tree.query = query
            tree.rebuild()
            self.assertIn((expected_key, 1), tree.rows)

        # Model attribute tooltips include update_flags metadata.
        spec = model.attribute_specs["shape_material_mu"]
        summary = attribute_summary("shape_material_mu", spec, model.shape_material_mu, Assignment.MODEL)
        self.assertIn("Update flags:", summary)
        self.assertIn("SHAPE_PROPERTIES", summary)

    def test_tendons_attachments_mimics_and_actuators(self) -> None:
        """Expose tendons, particle-body attachments, mimic constraints, and actuators in tree and inspector."""
        from newton.actuators import DrivePD  # noqa: PLC0415

        builder = newton.ModelBuilder()
        b0 = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        b1 = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_shape_box(b0, hx=0.1, hy=0.1, hz=0.1)
        builder.add_shape_box(b1, hx=0.1, hy=0.1, hz=0.1)
        j0 = builder.add_joint_revolute(-1, b0, axis=newton.Axis.Z)
        j1 = builder.add_joint_revolute(b0, b1, axis=newton.Axis.Z)
        builder.add_articulation([j0, j1])
        p0 = builder.add_particle(wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0), 0.1)
        builder.add_attachment_particle_body(p0, b0, body_point=wp.vec3(0.0, 0.0, 0.0))
        builder.add_tendon(
            [
                newton.TendonGuide(body=b0, offset=wp.vec3(0.0, 0.0, 0.1), guide_type=newton.TendonGuideType.ANCHOR),
                newton.TendonGuide(
                    body=b1, offset=wp.vec3(0.0, 0.0, 0.1), guide_type=newton.TendonGuideType.ANCHOR, rest_length=0.5
                ),
            ],
            label="cable_tendon",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            builder.add_constraint_mimic(j1, j0, coef0=0.0, coef1=0.5, label="linked_mimic")
        builder.add_actuator(DrivePD, index=0, kp=50.0, kd=5.0)
        model = builder.finalize(device="cpu")

        tree = ModelTree(model)
        self.assertIn(("tendon_group", -1), tree.labels)
        self.assertIn(("tendon", 0), tree.labels)
        self.assertEqual(tree.labels[("tendon", 0)], "cable_tendon")
        self.assertIn(("attachment_group", -1), tree.labels)
        self.assertIn(("mimic_group", -1), tree.labels)
        self.assertEqual(tree.labels[("constraint_mimic", 0)], "linked_mimic")
        self.assertIn(("actuator_collection", 0), tree.roots)

        # Guides, segments, attachments, and actuators expand lazily.
        self.assertFalse(tree.children.get(("tendon", 0)))
        tree.toggle(("tendon", 0))
        self.assertIn(("tendon_guide", 0), tree.children[("tendon", 0)])
        self.assertIn(("tendon_segment", 0), tree.children[("tendon", 0)])

        self.assertFalse(tree.children.get(("attachment_group", -1)))
        tree.toggle(("attachment_group", -1))
        self.assertIn(("attachment_particle_body", 0), tree.children[("attachment_group", -1)])

        self.assertFalse(tree.children.get(("actuator_collection", 0)))
        tree.toggle(("actuator_collection", 0))
        self.assertIn(("actuator_group", 0), tree.children[("actuator_collection", 0)])

        ws = ViewerWorkspace.__new__(ViewerWorkspace)
        ws.model, ws.tree = model, tree
        ws.viewer = SimpleNamespace(_last_state=model.state(), _inspection_components={"control": model.control()})
        ws._bindings = ws._bindings_components = None
        ws.history = None
        ws._cache, ws._live_cache = {}, {}
        inspector = PropertyInspector(ws)

        ws.selected, ws.selection = ("tendon", 0), {("tendon", 0)}
        inspector._contexts()
        self.assertEqual(
            {d.frequency for d in inspector.domains},
            {Frequency.TENDON, Frequency.TENDON_GUIDE, Frequency.TENDON_SEGMENT},
        )
        ws.selected, ws.selection = ("attachment_particle_body", 0), {("attachment_particle_body", 0)}
        inspector._contexts()
        self.assertEqual([d.frequency for d in inspector.domains], [Frequency.ATTACHMENT_PARTICLE_BODY])
        ws.selected, ws.selection = ("constraint_mimic", 0), {("constraint_mimic", 0)}
        inspector._contexts()
        self.assertEqual([d.frequency for d in inspector.domains], [Frequency.CONSTRAINT_MIMIC])

        for query, expected_key in (("cable_tendon", ("tendon", 0)), ("linked_mimic", ("constraint_mimic", 0))):
            tree.query = query
            tree.rebuild()
            self.assertIn((expected_key, 1), tree.rows)

    def test_global_world_slot_and_body_variants(self) -> None:
        """Map global world -1 to the trailing WORLD slot and switch body variants with rollback."""
        # 1) Global world -1 alongside two local worlds
        world_template = newton.ModelBuilder()
        body = world_template.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        variant_a = newton.ModelBuilder()
        variant_a.add_shape_box(
            variant_a.add_body(),
            hx=0.1,
            hy=0.1,
            hz=0.1,
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.2, restitution=0.8),
        )
        variant_b = newton.ModelBuilder()
        variant_b.add_shape_sphere(
            variant_b.add_body(),
            radius=0.15,
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.9, restitution=0.0),
        )
        world_template.add_body_variants(body, [variant_a, variant_b], initial=0, labels=["cube", "sphere"])

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.replicate(world_template, copy_count=2)
        model = builder.finalize(device="cpu")
        tree = ModelTree(model)

        viewer = ViewerNull()
        viewer.set_model(model)
        paused = False
        viewer.is_paused = lambda: paused
        ws = ViewerWorkspace.__new__(ViewerWorkspace)
        ws.model, ws.tree, ws.viewer = model, tree, viewer
        ws._bindings = ws._bindings_components = None
        ws.history = None
        ws._cache, ws._live_cache = {}, {}
        ws.message = ""
        inspector = PropertyInspector(ws)

        # Shared root maps to the trailing global-world slot (world_count), not local worlds.
        ws.selected, ws.selection = ("root", -1), {("root", -1)}
        inspector._contexts()
        world_domain = next(d for d in inspector.domains if d.frequency == Frequency.WORLD)
        self.assertEqual(list(world_domain.indices), [model.world_count])

        # Scene selection spans all local worlds plus the global-world slot.
        ws.selected, ws.selection = ("scene", 0), {("scene", 0)}
        inspector._contexts()
        scene_world = next(d for d in inspector.domains if d.frequency == Frequency.WORLD)
        self.assertEqual(list(scene_world.indices), list(range(model.world_count + 1)))

        # World references -1 and world_count both resolve to Shared root (-1).
        self.assertEqual(inspector.reference_target(model, Frequency.WORLD, -1), ("root", -1))
        self.assertEqual(inspector.reference_target(model, Frequency.WORLD, model.world_count), ("root", -1))
        self.assertEqual(inspector.reference_label(model, Frequency.WORLD, -1, ("root", -1)), "Shared (-1)")

        # 2) Interactive body variant switching requires pause and rolls back on solver rejection.
        self.assertEqual(int(model.body_variant.numpy()[0]), 0)
        self.assertFalse(ws.select_body_variant(0, 1))
        self.assertEqual(int(model.body_variant.numpy()[0]), 0)

        paused = True
        notified: list[int] = []
        ok_solver = SimpleNamespace(notify_model_changed=lambda flags: notified.append(int(flags)))
        viewer._inspection_components["solver"] = ok_solver
        self.assertTrue(ws.select_body_variant(0, 1))
        self.assertTrue(ws.select_body_variant(1, 1))
        self.assertEqual(int(model.body_variant.numpy()[0]), 1)
        self.assertTrue(notified)

        cached_mu: list[np.ndarray] = [model.shape_material_mu.numpy().copy()]
        accepted_mu = cached_mu[0].copy()

        def track_and_validate(_flags: int) -> None:
            cached_mu[0] = model.shape_material_mu.numpy().copy()
            if float(model.shape_material_restitution.numpy().max()) > 0.0:
                raise ValueError("unsupported positive restitution")

        viewer._inspection_components["solver"] = SimpleNamespace(notify_model_changed=track_and_validate)
        ws._bindings = ws._bindings_components = None

        # Switching body 0 back to variant 0 (restitution=0.8) mutates solver caches before
        # rejecting; rollback must refresh the solver from the restored Model (variant 1).
        self.assertFalse(ws.select_body_variant(0, 0))
        self.assertEqual(int(model.body_variant.numpy()[0]), 1)
        np.testing.assert_allclose(cached_mu[0], accepted_mu)
        self.assertIn("unsupported positive restitution", ws.message)

        # Explicit handling when solver recovery itself fails during rollback.
        def always_fail(_flags: int) -> None:
            raise RuntimeError("unrecoverable notify failure")

        viewer._inspection_components["solver"] = SimpleNamespace(notify_model_changed=always_fail)
        ws._bindings = ws._bindings_components = None
        self.assertFalse(ws.select_body_variant(0, 0))
        self.assertEqual(int(model.body_variant.numpy()[0]), 1)
        self.assertIn("rollback failed", ws.message)

        # 3) Body-motion edits stay read-only when CUDA step graphs cannot absorb BODY_PROPERTIES.
        viewer._inspection_components["solver"] = newton.solvers.SolverFeatherstone(model)
        ws._bindings = ws._bindings_components = None
        self.assertTrue(ws._body_flag_support()[0])
        ws.set_flag("body", 0, int(newton.BodyFlags.KINEMATIC))
        self.assertEqual(int(model.body_flags.numpy()[0]), int(newton.BodyFlags.KINEMATIC))
        ws.set_flag("body", 0, int(newton.BodyFlags.DYNAMIC))
        self.assertEqual(int(model.body_flags.numpy()[0]), int(newton.BodyFlags.DYNAMIC))

        if wp.is_cuda_available():
            cuda_builder = newton.ModelBuilder()
            cuda_body = cuda_builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            j = cuda_builder.add_joint_revolute(-1, cuda_body, axis=newton.Axis.Z)
            cuda_builder.add_articulation([j])
            cuda_model = cuda_builder.finalize(device="cuda:0")
            cuda_viewer = ViewerNull()
            cuda_viewer.set_model(cuda_model)
            cuda_viewer.is_paused = lambda: True
            cuda_ws = ViewerWorkspace.__new__(ViewerWorkspace)
            cuda_ws.model, cuda_ws.tree, cuda_ws.viewer = cuda_model, ModelTree(cuda_model), cuda_viewer
            cuda_ws._bindings = cuda_ws._bindings_components = None
            cuda_ws.history = None
            cuda_ws._cache, cuda_ws._live_cache = {}, {}
            cuda_ws.message = ""

            cuda_viewer._inspection_components["solver"] = newton.solvers.SolverFeatherstone(cuda_model)
            cuda_ws._bindings = cuda_ws._bindings_components = None
            supported, reason = cuda_ws._body_flag_support()
            self.assertFalse(supported)
            self.assertIn("recapturing", reason)
            cuda_ws.set_flag("body", 0, int(newton.BodyFlags.KINEMATIC))
            self.assertEqual(int(cuda_model.body_flags.numpy()[0]), int(newton.BodyFlags.DYNAMIC))

            cuda_viewer._inspection_components["solver"] = newton.solvers.SolverXPBD(cuda_model)
            cuda_ws._bindings = cuda_ws._bindings_components = None
            self.assertTrue(cuda_ws._body_flag_support()[0])
            cuda_ws.set_flag("body", 0, int(newton.BodyFlags.KINEMATIC))
            self.assertEqual(int(cuda_model.body_flags.numpy()[0]), int(newton.BodyFlags.KINEMATIC))

    def test_contacts_history_and_solver_classification(self) -> None:
        """Preserve new contact fields in history, filter non-row fields, and classify maximal solvers."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device="cpu")
        state = model.state()
        contacts = newton.Contacts(
            rigid_contact_max=2,
            soft_contact_max=2,
            device="cpu",
            contact_matching=True,
            contact_report=True,
            rigid_contact_surface_velocity=True,
        )
        contacts.rigid_contact_count.assign(np.array([1], dtype=np.int32))
        contacts.rigid_contact_shape0.assign(np.array([0, -1], dtype=np.int32))
        contacts.rigid_contact_shape1.assign(np.array([0, -1], dtype=np.int32))
        contacts.rigid_contact_match_index.assign(np.array([7, -1], dtype=np.int32))
        contacts.rigid_contact_surface_velocity.assign(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float32))
        contacts.soft_contact_count.assign(np.array([1], dtype=np.int32))
        contacts.soft_contact_rigid_indices.assign(np.array([[1, 2, 3], [-1, -1, -1]], dtype=np.int32))

        # CollisionInspector must not leak non-per-contact counters or index lists onto contact 0.
        ws = ViewerWorkspace.__new__(ViewerWorkspace)
        ws.model = model
        ws.history = None
        ws.viewer = SimpleNamespace(_last_contact_observables=None)
        ci = CollisionInspector(ws)
        ci.contacts = contacts
        field_names = {field[1] for field in ci.contact_fields(("contact", 0))}
        self.assertIn("rigid_contact_surface_velocity", field_names)
        self.assertIn("rigid_contact_match_index", field_names)
        for excluded in (
            "rigid_contact_new_count",
            "rigid_contact_broken_count",
            "rigid_contact_new_indices",
            "rigid_contact_broken_indices",
            "rigid_contact_match_generation",
        ):
            self.assertNotIn(excluded, field_names)

        # FrameHistory records and replays surface_velocity, match_index, and soft_contact_rigid_indices.
        viewer = ViewerNull()
        viewer.set_model(model)
        history = FrameHistory(viewer)
        history.begin_frame(0.0)
        history.contacts(contacts, state, None)
        history.end_frame()
        contacts.rigid_contact_match_index.fill_(-1)
        contacts.rigid_contact_surface_velocity.zero_()
        contacts.soft_contact_rigid_indices.view(wp.int32).fill_(-1)
        history.begin_frame(1.0)
        history.contacts(contacts, state, None)
        history.end_frame()
        history.select(0)
        history.begin_frame(1.0)
        replayed, _, _ = history.contacts(contacts, state, None)
        history.end_frame()
        self.assertEqual(int(replayed.rigid_contact_match_index.numpy()[0]), 7)
        np.testing.assert_allclose(replayed.rigid_contact_surface_velocity.numpy()[0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(replayed.soft_contact_rigid_indices.numpy()[0], [1, 2, 3])

        # Maximal-coordinate solver detection distinguishes maximal vs reduced-coordinate solvers.
        self.assertTrue(ViewerWorkspace._uses_maximal_coordinates(newton.solvers.SolverXPBD(model)))
        self.assertTrue(ViewerWorkspace._uses_maximal_coordinates(newton.solvers.SolverSemiImplicit(model)))
        self.assertFalse(ViewerWorkspace._uses_maximal_coordinates(newton.solvers.SolverFeatherstone(model)))
        self.assertTrue(ViewerWorkspace._uses_maximal_coordinates(type("SolverKamino", (), {})()))

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
        ws.viewer = SimpleNamespace(
            is_paused=lambda: True, _last_state=state, _inspection_components={"control": control}
        )
        ws._bindings = ws._bindings_components = None
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
        before, untouched = state.body_q.numpy().copy(), other.body_q.numpy().copy()
        # Sibling coordinates deliberately disagree with the displayed body state.
        state.joint_q[3:4].fill_(1.2)
        self.assertTrue(ws.apply_edit(state, "joint_q", 1, [0.5], joint_index=1))
        np.testing.assert_array_equal(state.body_q.numpy()[[0, 3]], before[[0, 3]])
        self.assertGreater(np.linalg.norm(state.body_q.numpy()[[1, 2]] - before[[1, 2]]), 0.1)
        # Only the logged State is edited; other application States are not touched.
        np.testing.assert_array_equal(other.body_q.numpy(), untouched)
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
        ws.history = SimpleNamespace(replaying=True)
        self.assertFalse(ws.apply_edit(state, "joint_qd", 1, [2], joint_index=1))
        self.assertFalse(ws.apply_edit(control.test, "enabled", 0, [False]))
        ws.viewer.is_paused = lambda: False
        self.assertFalse(ws.apply_edit(control, "joint_target_q", 0, [1.0]))
        self.assertFalse(ws.apply_edit(control.test, "enabled", 0, [False]))
        ws.viewer.is_paused = lambda: True
        ws.history = None
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
        ws.history = SimpleNamespace(replaying=True)
        self.assertFalse(ws.reset_control(zero=True))
        np.testing.assert_array_equal(control.test.enabled.numpy(), True)
        ws.history = None
        # A newly supplied Control gets its own baseline.
        replacement = model.control()
        ws.viewer._inspection_components = {"control": replacement}
        replacement.joint_target_q.fill_(0.4)
        ws._capture_control_initial()
        replacement.joint_target_q.fill_(0.9)
        self.assertTrue(ws.reset_control())
        np.testing.assert_allclose(replacement.joint_target_q.numpy(), 0.4)


class TestCollisionInspector(unittest.TestCase):
    @requires("imgui_bundle")
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
                viewer.set_collision_pipeline(pipeline)
                ws = SimpleNamespace(
                    viewer=viewer, model=model, history=None, selected=None, _at_live_frame=lambda: True
                )
                ws.bindings = InspectionBindings(components=viewer._inspection_components)
                ws.select = lambda key, workspace=ws: setattr(workspace, "selected", key)
                inspector = CollisionInspector(ws)
                # Contacts come from the logged frame, including when the normal glyphs are hidden.
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
                ws.gui = SimpleNamespace(ui=_offscreen_imgui_backend(self))
                ws.widgets = PhysicsWidgets(ws.gui)
                imgui = ws.gui.ui.imgui
                ws.gui.ui.begin_frame()
                imgui.begin("Contacts")
                inspector.draw_contact_list(SimpleNamespace(_page=lambda rows: rows[:1]), ws.selected)
                imgui.end()
                ws.gui.ui.end_frame()
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
                # Resuming clears labels immediately, even with Contacts open.
                # Advancing live witnesses must not download or lay out labels.
                expected = np.asarray(inspector.points(("contact", 0)))
                viewer.is_paused = lambda: False
                inspector.overlays()
                self.assertEqual(viewer.renderer._contact_labels, ())
                self.assertIsNone(viewer.renderer._contact_selection)
                poses = state.body_q.numpy()
                poses[:, 2] += 0.25
                state.body_q.assign(poses)
                viewer.time += 1
                with (
                    patch.object(inspector, "read", wraps=inspector.read) as read,
                    patch.object(inspector, "label_points", wraps=inspector.label_points) as points,
                ):
                    inspector.overlays()
                    read.assert_not_called()
                    points.assert_not_called()
                self.assertEqual(viewer.renderer._contact_labels, ())
                self.assertTrue(viewer.show_contact_labels)
                # Pausing restores the preference using the current witnesses.
                viewer.is_paused = lambda: True
                inspector.overlays()
                self.assertEqual(len(viewer.renderer._contact_labels), 1)
                np.testing.assert_allclose(viewer.renderer._contact_labels[0][0], expected + np.array([0, 0, 0.25]))
                poses[:, 2] -= 0.25
                state.body_q.assign(poses)
                viewer.time += 1
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
                saved_shape0 = contacts.rigid_contact_shape0.numpy().copy()
                contacts.rigid_contact_shape0.assign(np.where(np.arange(len(saved_shape0)) == 0, -1, saved_shape0))
                try:
                    for k in [k for k in inspector.cache if k[0] == "label_points"]:
                        inspector.cache.pop(k)
                    self.assertTrue(np.isnan(inspector.label_points((("contact", 0),))[("contact", 0)]).all())
                finally:
                    contacts.rigid_contact_shape0.assign(saved_shape0)
                    for k in [k for k in inspector.cache if k[0] == "label_points"]:
                        inspector.cache.pop(k)
                fields = {
                    name: (spec, start, count)
                    for _, name, _, spec, start, count in inspector.contact_fields(("contact", 0))
                }
                self.assertEqual(fields["probe:shape_link"][0].references, frequency.SHAPE)
                self.assertEqual(fields["probe:weights"][1:], (0, 2))
                # Collapsing the built-in fields must not turn their None
                # namespace into an invalid custom-attribute tab label.
                ws.selected = ("contact", 0)
                with (
                    patch.object(ws.widgets, "group", return_value=nullcontext(False)),
                    patch.object(ws.widgets, "tab_item", wraps=ws.widgets.tab_item) as tab_item,
                ):
                    ws.gui.ui.begin_frame()
                    imgui.begin("Contact")
                    inspector.draw_contact(PropertyInspector(ws), ws.selected)
                    imgui.end()
                    ws.gui.ui.end_frame()
                self.assertEqual([call.args[0] for call in tab_item.call_args_list], ["probe"])
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
                ws.bindings = InspectionBindings()
                inspector.attach(tree)
                inspector.sync(force=True)
                self.assertIn(inspector.root, tree.labels)
                self.assertFalse(inspector.pipelines)
                viewer._last_contacts = None
                with patch.object(inspector, "read", wraps=inspector.read) as read:
                    inspector.attach(tree)
                    inspector.sync()
                    read.assert_not_called()
                self.assertNotIn(inspector.root, tree.roots)
                self.assertFalse(any(inspector.handles(key) for key in tree.labels))
                ws.bindings = InspectionBindings(components=viewer._inspection_components)
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


@requires("pyglet")
class TestViewerRTXShortcuts(unittest.TestCase):
    def test_repeated_frame_key_through_window_events(self) -> None:
        """Frame the scene on every F press routed through ViewerRTX's window events."""
        import_pyglet_window(shadow_window=False)
        from pyglet.window import key

        handlers = {}
        window = SimpleNamespace(
            event=lambda handler: handlers.setdefault(handler.__name__, handler), close=mock.Mock()
        )
        viewer = ViewerRTX.__new__(ViewerRTX)
        viewer._window = window
        viewer._keys_down = set()
        gui = ViewerGui.__new__(ViewerGui)
        gui._viewer, gui.ui, gui.workspace = viewer, None, None
        gui._loading_splash_active = gui._frame_key_down = gui._follow_held = False
        gui.frame_camera_on_model = mock.Mock()
        viewer.gui = gui
        viewer._register_input_events()

        handlers["on_key_press"](key.F, 0)
        handlers["on_key_press"](key.F, 0)  # OS key repeat frames once.
        self.assertEqual(gui.frame_camera_on_model.call_count, 1)
        handlers["on_key_release"](key.F, 0)
        self.assertNotIn(key.F, viewer._keys_down)
        handlers["on_key_press"](key.F, 0)
        self.assertEqual(gui.frame_camera_on_model.call_count, 2)


@requires("imgui_bundle", "pyglet")
class TestViewerGizmos(unittest.TestCase):
    def setUp(self) -> None:
        from imgui_bundle import imgui, imguizmo

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
        self.gui._selection_press = None
        self.gui.ui = ImGuiBackend.__new__(ImGuiBackend)
        self.gui.ui._context = context
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
        import_pyglet_window(shadow_window=False)
        from pyglet.window import mouse

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
