# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.core.types import Axis
from newton._src.viewer.viewer_viser import ViewerViser


class _FakeHandle:
    def __init__(self, **kwargs):
        object.__setattr__(self, "kwargs", kwargs)
        object.__setattr__(
            self,
            "position",
            np.asarray(kwargs.get("position", kwargs.get("batched_positions", (0.0, 0.0, 0.0))), dtype=np.float32),
        )
        object.__setattr__(self, "wxyz", np.asarray(kwargs.get("wxyz", (1.0, 0.0, 0.0, 0.0)), dtype=np.float32))
        object.__setattr__(self, "visible", kwargs.get("visible", True))
        object.__setattr__(self, "removed", False)
        object.__setattr__(self, "update_callbacks", [])
        object.__setattr__(self, "drag_start_callbacks", [])
        object.__setattr__(self, "drag_end_callbacks", [])
        object.__setattr__(self, "click_callbacks", [])
        for name, value in kwargs.items():
            if name not in self.__dict__:
                object.__setattr__(self, name, value)
        object.__setattr__(self, "property_updates", {})

    def __setattr__(self, name, value):
        updates = self.__dict__.get("property_updates")
        if updates is not None and name != "property_updates":
            updates[name] = updates.get(name, 0) + 1
        object.__setattr__(self, name, value)

    def on_update(self, callback):
        self.update_callbacks.append(callback)
        return callback

    def on_drag_start(self, callback):
        self.drag_start_callbacks.append(callback)
        return callback

    def on_drag_end(self, callback):
        self.drag_end_callbacks.append(callback)
        return callback

    def on_click(self, callback):
        self.click_callbacks.append(callback)
        return callback

    def remove_click_callback(self, callback):
        self.click_callbacks.remove(callback)

    def emit_update(self, position, wxyz=(1.0, 0.0, 0.0, 0.0)):
        self.position = np.asarray(position, dtype=np.float32)
        self.wxyz = np.asarray(wxyz, dtype=np.float32)
        event = SimpleNamespace(target=self)
        for callback in self.update_callbacks:
            callback(event)

    def emit_drag_start(self):
        event = SimpleNamespace(target=self)
        for callback in self.drag_start_callbacks:
            callback(event)

    def emit_drag_end(self):
        event = SimpleNamespace(target=self)
        for callback in self.drag_end_callbacks:
            callback(event)

    def emit_click(self, ray_origin, ray_direction):
        event = SimpleNamespace(ray_origin=ray_origin, ray_direction=ray_direction, target=self)
        for callback in self.click_callbacks:
            callback(event)

    def remove(self):
        self.removed = True


class _FakeFolder(_FakeHandle):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class _FakeGuiHandle:
    def __init__(self, value=None, **kwargs):
        self.value = value
        self.disabled = kwargs.get("disabled", False)
        self.visible = kwargs.get("visible", True)
        self.update_callbacks = []
        self.click_callbacks = []
        self.removed = False

    def on_update(self, callback):
        self.update_callbacks.append(callback)
        return callback

    def on_click(self, callback):
        self.click_callbacks.append(callback)
        return callback

    def emit_update(self, value):
        self.value = value
        event = SimpleNamespace(client_id=1, target=self)
        for callback in self.update_callbacks:
            callback(event)

    def emit_click(self):
        event = SimpleNamespace(client_id=1, target=self)
        for callback in self.click_callbacks:
            callback(event)

    def remove(self):
        self.removed = True


class _FakeGui:
    def __init__(self):
        self.folders = []
        self.plots = []
        self.inputs = {}

    def add_folder(self, name, **kwargs):
        folder = _FakeFolder(name=name, **kwargs)
        self.folders.append(folder)
        return folder

    def add_checkbox(self, label, initial_value, **kwargs):
        handle = _FakeGuiHandle(initial_value, **kwargs)
        self.inputs[label] = handle
        return handle

    def add_button(self, label, **kwargs):
        handle = _FakeGuiHandle(**kwargs)
        self.inputs[label] = handle
        return handle

    def add_dropdown(self, label, options, initial_value=None, **kwargs):
        handle = _FakeGuiHandle(initial_value if initial_value is not None else options[0], **kwargs)
        handle.options = tuple(options)
        self.inputs[label] = handle
        return handle

    def add_uplot(self, **kwargs):
        plot = _FakeHandle(**kwargs)
        plot.data = kwargs["data"]
        self.plots.append(plot)
        return plot


class _FakeClient:
    def __init__(self, client_id):
        self.client_id = client_id
        self.notifications = []

    def add_notification(self, title, body, **kwargs):
        handle = _FakeHandle(title=title, body=body, **kwargs)
        self.notifications.append(handle)
        return handle


class _FakeScene:
    def __init__(self):
        self.controls = {}
        self.lines = {}

    def add_light_ambient(self, name):
        return _FakeHandle(name=name)

    def configure_environment_map(self, **kwargs):
        return None

    def add_transform_controls(self, name, **kwargs):
        handle = _FakeHandle(name=name, **kwargs)
        self.controls[name] = handle
        return handle

    def add_line_segments(self, name, points, colors, line_width):
        handle = _FakeHandle(name=name, points=points, colors=colors, line_width=line_width)
        handle.points = points
        handle.colors = colors
        handle.line_width = line_width
        self.lines[name] = handle
        return handle

    def add_grid(self, name, infinite_grid=False, **kwargs):
        return _FakeHandle(name=name, infinite_grid=infinite_grid, **kwargs)

    def add_mesh_simple(self, name, **kwargs):
        return _FakeHandle(name=name, **kwargs)

    def add_batched_meshes_simple(self, name, **kwargs):
        return _FakeHandle(name=name, **kwargs)


class _FakeServer:
    def __init__(self, **_kwargs):
        self.scene = _FakeScene()
        self.gui = _FakeGui()
        self.clients = {}
        self.stopped = False
        self.atomic_depth = 0
        self.atomic_frames = 0
        self.flush_count = 0

    class _Atomic:
        def __init__(self, server):
            self.server = server

        def __enter__(self):
            self.server.atomic_depth += 1

        def __exit__(self, exc_type, exc_value, traceback):
            self.server.atomic_depth -= 1
            self.server.atomic_frames += 1

    def atomic(self):
        return self._Atomic(self)

    def flush(self):
        if self.atomic_depth != 0:
            raise RuntimeError("Cannot flush an incomplete atomic frame")
        self.flush_count += 1

    def on_client_connect(self, _callback):
        return None

    def on_client_disconnect(self, _callback):
        return None

    def get_scene_serializer(self):
        return None

    def get_clients(self):
        return self.clients

    def stop(self):
        self.stopped = True


class _FakeArray:
    def __init__(self, values):
        self.values = np.asarray(values)

    def numpy(self):
        return self.values.copy()

    def assign(self, values):
        self.values = np.asarray(values).copy()

    def fill_(self, value):
        self.values.fill(value)


class _FakePicking:
    def __init__(self):
        state_dtype = np.dtype(
            [
                ("picking_target_world", np.float32, (3,)),
                ("picked_point_world", np.float32, (3,)),
            ]
        )
        state = np.zeros(1, dtype=state_dtype)
        self.pick_body = _FakeArray([-1])
        self.pick_state = _FakeArray(state)
        self.active = False
        self.applied = False
        self.last_ray = None

    def pick(self, _state, ray_origin, ray_direction):
        self.active = True
        self.pick_body.values[0] = 0
        self.last_ray = (tuple(ray_origin), tuple(ray_direction))
        state = self.pick_state.numpy()
        state[0]["picking_target_world"] = (0.0, 0.0, 0.0)
        state[0]["picked_point_world"] = (0.0, 0.0, 0.0)
        self.pick_state.assign(state)

    def is_picking(self):
        return self.active

    def release(self):
        self.active = False
        self.pick_body.fill_(-1)

    def _apply_picking_force(self, _state):
        self.applied = True


class TestViewerViserInteraction(unittest.TestCase):
    def setUp(self):
        """Create a ViewerViser backed by deterministic fake Viser handles."""
        self.server = _FakeServer()
        fake_viser = SimpleNamespace(ViserServer=lambda **_kwargs: self.server)
        self.patches = [
            patch.object(ViewerViser, "_get_viser", return_value=fake_viser),
            patch("newton._src.viewer.viewer_viser.is_jupyter_notebook", return_value=False),
        ]
        for patcher in self.patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.viewer = ViewerViser(verbose=False)
        self.addCleanup(self.viewer.close)

    def test_disabling_picking_removes_click_behavior(self):
        """Remove click callbacks and hover highlighting when picking is disabled."""
        handle = _FakeHandle()
        self.viewer._attach_picking_callback(handle, self.viewer.layer.layer_id)

        self.assertEqual(len(handle.click_callbacks), 1)
        self.viewer.picking_enabled = False
        self.assertEqual(handle.click_callbacks, [])

        self.viewer._attach_picking_callback(handle, self.viewer.layer.layer_id)
        self.assertEqual(handle.click_callbacks, [])

    def test_set_camera_uses_world_up_axis(self):
        """Keep Viser orbit controls aligned with the model up axis."""
        with patch.object(self.viewer, "_get_camera_up_axis", return_value=2):
            self.viewer.set_camera(wp.vec3(3.0, -4.0, 2.0), pitch=-30.0, yaw=90.0)

        position, look_at, up_direction = self.viewer._camera_request
        np.testing.assert_allclose(position, (3.0, -4.0, 2.0))
        np.testing.assert_allclose(look_at - position, (0.0, 0.8660254, -0.5), atol=1.0e-7)
        np.testing.assert_allclose(up_direction, (0.0, 0.0, 1.0))

    def test_native_visualization_controls_update_viewer(self):
        """Apply supported visualization toggles through the native Viser GUI."""
        self.viewer._viewer_option_handles["show_contacts"].emit_update(True)
        self.viewer._viewer_option_handles["show_visual"].emit_update(False)

        mesh = _FakeHandle()
        self.viewer._scene_handles["mesh"] = mesh
        self.viewer._instances["mesh"] = {"use_trimesh": False}
        self.viewer._viewer_option_handles["wireframe"].emit_update(True)
        self.viewer.should_step()

        self.assertTrue(self.viewer.show_contacts)
        self.assertFalse(self.viewer.show_visual)
        self.assertTrue(self.viewer._wireframe)
        self.assertTrue(mesh.wireframe)

    def test_native_simulation_controls_pause_step_and_reset(self):
        """Pause, single-step, and reset through native Viser controls."""
        reset_calls = []
        self.viewer.set_reset_callback(lambda: reset_calls.append(True))
        self.viewer._simulation_gui_handles["pause"].emit_update(True)

        self.assertFalse(self.viewer.should_step())
        self.viewer._simulation_gui_handles["step"].emit_click()
        self.assertTrue(self.viewer.should_step())
        self.assertFalse(self.viewer.should_step())

        self.viewer._simulation_gui_handles["reset"].emit_click()
        self.viewer.should_step()
        self.assertEqual(reset_calls, [True])

    def test_native_example_browser_selects_module(self):
        """Route a native example-browser selection to the run loop."""
        selected = []
        self.viewer.configure_example_browser(
            {"basic": [("pendulum", "newton.examples.basic.example_basic_pendulum")]},
            selected.append,
        )
        self.viewer._example_browser_handles["load"].emit_click()
        self.viewer.should_step()

        self.assertEqual(selected, ["newton.examples.basic.example_basic_pendulum"])

    def test_loading_splash_uses_client_notifications(self):
        """Show and remove a loading notification over the Viser canvas."""
        client = _FakeClient(7)
        self.server.clients[client.client_id] = client

        self.viewer.show_loading_splash("Loading example...")

        self.assertEqual(len(client.notifications), 1)
        notification = client.notifications[0]
        self.assertEqual(notification.kwargs["title"], "Newton")
        self.assertEqual(notification.kwargs["body"], "Loading example...")
        self.assertTrue(notification.kwargs["loading"])
        self.assertFalse(notification.kwargs["with_close_button"])

        self.viewer.hide_loading_splash()
        self.assertTrue(notification.removed)

    def test_loading_splash_is_shown_to_late_clients(self):
        """Apply an active loading message when a client connects."""
        self.viewer.show_loading_splash("Resetting...")
        client = _FakeClient(8)

        self.viewer._handle_client_connect(client)

        self.assertEqual(len(client.notifications), 1)
        self.assertEqual(client.notifications[0].kwargs["body"], "Resetting...")

    def test_frame_updates_are_published_atomically(self):
        """Group all scene updates between begin and end into one Viser frame."""
        self.viewer.begin_frame(0.0)
        self.assertEqual(self.server.atomic_depth, 1)

        self.viewer.end_frame()

        self.assertEqual(self.server.atomic_depth, 0)
        self.assertEqual(self.server.atomic_frames, 1)
        self.assertEqual(self.server.flush_count, 1)

    def test_gizmo_updates_transform_and_snaps_on_release(self):
        """Mutate gizmo transforms and honor independent axes and snap targets."""
        transform = wp.transform(wp.vec3(1.0, 2.0, 3.0), wp.quat_identity())
        snap_to = wp.transform(wp.vec3(4.0, 5.0, 6.0), wp.quat_identity())

        self.viewer.begin_frame(0.0)
        self.viewer.log_gizmo(
            "target",
            transform,
            translate=(Axis.X,),
            rotate=(Axis.Z,),
            snap_to=snap_to,
        )
        entry = self.viewer._gizmo_handles["target"]
        translate = entry["handles"]["translate"]
        rotate = entry["handles"]["rotate"]

        self.assertEqual(translate.kwargs["active_axes"], (True, False, False))
        self.assertTrue(translate.kwargs["disable_rotations"])
        self.assertTrue(translate.kwargs["fixed"])
        self.assertEqual(translate.kwargs["scale"], 56.0)
        self.assertEqual(rotate.kwargs["active_axes"], (False, False, True))
        self.assertTrue(rotate.kwargs["disable_axes"])
        self.assertTrue(rotate.kwargs["disable_sliders"])
        self.assertTrue(rotate.kwargs["fixed"])
        self.assertEqual(rotate.kwargs["scale"], 56.0)

        translate.emit_drag_start()
        translate.emit_update((7.0, 8.0, 9.0))
        self.assertTrue(self.viewer.should_step())
        self.assertTrue(self.viewer.gizmo_is_using)
        np.testing.assert_allclose(tuple(transform.p), (7.0, 8.0, 9.0))

        translate.emit_drag_end()
        self.assertTrue(self.viewer.should_step())
        self.assertFalse(self.viewer.gizmo_is_using)
        np.testing.assert_allclose(tuple(transform.p), (4.0, 5.0, 6.0))

    def test_plane_grid_updates_without_recreation(self):
        """Update persistent plane-grid properties without replacing its handle."""
        plane_info = {"width": 10.0, "length": 8.0}
        first_xforms = wp.array([wp.transform_identity()], dtype=wp.transform)
        second_xforms = wp.array(
            [wp.transform(wp.vec3(1.0, 2.0, 3.0), wp.quat_identity())],
            dtype=wp.transform,
        )

        self.viewer._log_plane_instances("ground", plane_info, first_xforms, None)
        handle = self.viewer._plane_handles["ground"][0]
        self.viewer._log_plane_instances("ground", plane_info, second_xforms, None)

        self.assertIs(self.viewer._plane_handles["ground"][0], handle)
        self.assertFalse(handle.removed)
        np.testing.assert_allclose(handle.position, (1.0, 2.0, 3.0))

        handle.property_updates.clear()
        self.viewer._log_plane_instances("ground", plane_info, second_xforms, None)
        self.assertEqual(handle.property_updates, {})

        self.viewer._log_plane_instances("ground", plane_info, second_xforms, None, hidden=True)
        self.assertFalse(handle.visible)

    def test_infinite_plane_uses_native_viser_grid(self):
        """Map non-positive Newton plane extents to Viser's infinite grid."""
        xforms = wp.array([wp.transform_identity()], dtype=wp.transform)

        self.viewer.log_geo("ground", newton.GeoType.PLANE, (0.0, 0.0), 0.0, True)
        ground_info = self.viewer._plane_meshes["ground"]
        self.viewer._log_plane_instances("ground", ground_info, xforms, None)

        self.assertTrue(ground_info["infinite"])
        self.assertTrue(self.viewer._plane_handles["ground"][0].infinite_grid)

        self.viewer.log_geo("platform", newton.GeoType.PLANE, (12.0, 8.0), 0.0, True)
        platform_info = self.viewer._plane_meshes["platform"]
        self.viewer._log_plane_instances("platform", platform_info, xforms, None)

        self.assertFalse(platform_info["infinite"])
        self.assertFalse(self.viewer._plane_handles["platform"][0].infinite_grid)

    def test_unchanged_instances_do_not_resend_properties(self):
        """Skip redundant websocket properties for unchanged instance batches."""
        points = wp.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=wp.vec3)
        indices = wp.array((0, 1, 2), dtype=wp.int32)
        xforms = wp.array((wp.transform_identity(),), dtype=wp.transform)
        scales = wp.array(((1.0, 1.0, 1.0),), dtype=wp.vec3)
        self.viewer.log_mesh("mesh", points, indices)
        self.viewer._shape_instances[0] = SimpleNamespace(name="instances", static=False)
        self.viewer.log_instances("instances", "mesh", xforms, scales, None, None)
        handle = self.viewer._scene_handles["instances"]

        handle.property_updates.clear()
        with patch.object(self.viewer, "_to_numpy", wraps=self.viewer._to_numpy) as to_numpy:
            self.viewer.log_instances("instances", "mesh", xforms, scales, None, None)

        self.assertEqual(handle.property_updates, {})
        self.assertEqual(to_numpy.call_count, 1)

    def test_mesh_updates_vertices_without_recreating_handle(self):
        """Keep deforming meshes persistent when their topology is unchanged."""
        points = wp.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=wp.vec3)
        moved_points = wp.array(((0.0, 0.0, 0.5), (1.0, 0.0, 0.5), (0.0, 1.0, 0.5)), dtype=wp.vec3)
        indices = wp.array((0, 1, 2), dtype=wp.int32)

        self.viewer.log_mesh("cloth", points, indices, backface_culling=False)
        handle = self.viewer._scene_handles["cloth"]
        handle.property_updates.clear()

        self.viewer.log_mesh("cloth", moved_points, indices, backface_culling=False)

        self.assertIs(self.viewer._scene_handles["cloth"], handle)
        self.assertFalse(handle.removed)
        self.assertEqual(handle.property_updates.get("vertices"), 1)
        self.assertNotIn("faces", handle.property_updates)
        np.testing.assert_allclose(handle.vertices, moved_points.numpy())

    def test_hidden_mesh_reuses_existing_handle(self):
        """Hide and restore a mesh without resending its topology."""
        points = wp.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=wp.vec3)
        indices = wp.array((0, 1, 2), dtype=wp.int32)

        self.viewer.log_mesh("cloth", points, indices)
        handle = self.viewer._scene_handles["cloth"]
        self.viewer.log_mesh("cloth", points, indices, hidden=True)
        self.viewer.log_mesh("cloth", points, indices)

        self.assertIs(self.viewer._scene_handles["cloth"], handle)
        self.assertFalse(handle.removed)
        self.assertTrue(handle.visible)

    def test_model_shapes_use_packed_host_transfers(self):
        """Transfer all model-shape transforms and colors in two arrays."""
        builder = newton.ModelBuilder()
        builder.add_shape_box(-1)
        builder.add_shape_sphere(-1)
        model = builder.finalize(device="cpu")
        self.viewer.set_model(model)
        state = model.state()
        self.viewer.log_state(state)

        with patch.object(self.viewer, "_to_numpy", wraps=self.viewer._to_numpy) as to_numpy:
            self.viewer.log_state(state)

        warp_transfers = [call.args[0] for call in to_numpy.call_args_list if isinstance(call.args[0], wp.array)]
        self.assertEqual(len(self.viewer._packed_shape_groups), 2)
        self.assertEqual(len(warp_transfers), 2)

    def test_gizmo_disappears_when_not_logged(self):
        """Remove a persistent Viser gizmo after a frame stops logging it."""
        transform = wp.transform_identity()
        self.viewer.begin_frame(0.0)
        self.viewer.log_gizmo("temporary", transform)
        handles = tuple(self.viewer._gizmo_handles["temporary"]["handles"].values())
        self.viewer.end_frame()

        self.viewer.begin_frame(0.1)
        self.viewer.end_frame()

        self.assertNotIn("temporary", self.viewer._gizmo_handles)
        self.assertTrue(all(handle.removed for handle in handles))

    def test_click_handle_drives_picking_force_and_line(self):
        """Turn a scene click and handle drag into a picking target and force."""
        picking = _FakePicking()
        self.viewer.picking = picking
        self.viewer._last_state = object()
        self.viewer.layer.xform = wp.transform(wp.vec3(10.0, 0.0, 0.0), wp.quat_identity())
        scene_handle = _FakeHandle()
        self.viewer._attach_picking_callback(scene_handle, self.viewer.layer.layer_id)

        scene_handle.emit_click((10.0, 0.0, -2.0), (0.0, 0.0, 1.0))
        self.viewer.begin_frame(0.0)
        self.assertTrue(picking.is_picking())
        self.assertIn(self.viewer.layer.layer_id, self.viewer._picking_controls)
        np.testing.assert_allclose(picking.last_ray[0], (0.0, 0.0, -2.0))

        control = self.viewer._picking_controls[self.viewer.layer.layer_id]
        self.assertTrue(control.kwargs["fixed"])
        self.assertEqual(control.kwargs["scale"], 56.0)
        np.testing.assert_allclose(control.position, (10.0, 0.0, 0.0))
        control.emit_update((11.0, 2.0, 3.0))
        self.viewer.begin_frame(0.1)
        np.testing.assert_allclose(picking.pick_state.numpy()[0]["picking_target_world"], (1.0, 2.0, 3.0))

        self.viewer._render_picking_line()
        line = self.server.scene.lines["picking_line"]
        np.testing.assert_allclose(line.points[0, 0], (10.0, 0.0, 0.0))
        np.testing.assert_allclose(line.points[0, 1], (11.0, 2.0, 3.0))

        self.viewer.apply_forces(object())
        self.assertTrue(picking.applied)

        control.emit_drag_end()
        self.viewer.begin_frame(0.2)
        self.assertFalse(picking.is_picking())
        self.assertTrue(control.removed)

    def test_scalar_logging_updates_rolling_plot(self):
        """Render logged scalars into a bounded rolling uPlot history."""
        fake_uplot = SimpleNamespace(
            Series=lambda **kwargs: kwargs,
            Scale=lambda **kwargs: kwargs,
        )
        with patch.dict(sys.modules, {"viser": SimpleNamespace(uplot=fake_uplot)}):
            self.viewer._plot_history_size = 3
            for value in (1.0, 2.0, 3.0, 4.0):
                self.viewer.log_scalar("energy", value)
            self.viewer.end_frame()

        self.assertEqual(len(self.server.gui.plots), 1)
        _x, y = self.server.gui.plots[0].data
        np.testing.assert_allclose(y, (2.0, 3.0, 4.0))


if __name__ == "__main__":
    unittest.main()
