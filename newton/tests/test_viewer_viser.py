# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import io
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.viewer import ViewerViser


@unittest.skipUnless(importlib.util.find_spec("viser"), "Requires viser")
class TestViewerViserInteraction(unittest.TestCase):
    def setUp(self):
        """Start a real Viser server and connect through its WebSocket protocol."""
        import msgspec  # noqa: PLC0415
        import viser
        import zstandard  # noqa: PLC0415
        from websockets.exceptions import ConnectionClosed  # noqa: PLC0415
        from websockets.sync.client import connect  # noqa: PLC0415

        self.viewer = ViewerViser(port=0, verbose=False)
        self.addCleanup(self.viewer.close)
        self.server = self.viewer._server
        self.socket = connect(
            f"ws://localhost:{self.server.get_port()}", max_size=None, subprotocols=[f"viser-v{viser.__version__}"]
        )
        self.addCleanup(self.socket.close)
        self.messages = []

        def receive():
            try:
                for payload in self.socket:
                    size = int.from_bytes(payload[8:16], "little")
                    metadata = zstandard.ZstdDecompressor().decompress(payload[16 : 16 + size])
                    self.messages.extend(msgspec.msgpack.decode(metadata)["messages"])
            except ConnectionClosed:
                # The connection is closed during test cleanup.
                return

        self.receiver = threading.Thread(target=receive, daemon=True)
        self.receiver.start()
        self.send(
            "ViewerCameraMessage",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(5.0, 0.0, 2.0),
            fov=0.8,
            near=0.01,
            far=1000.0,
            image_height=800,
            image_width=1000,
            look_at=(0.0, 0.0, 0.0),
            up_direction=(0.0, 0.0, 1.0),
        )
        self.wait_for(lambda: bool(self.server.get_clients()))

    def send(self, message_type, **fields):
        """Send a real browser protocol message to the server."""
        import msgspec  # noqa: PLC0415

        self.socket.send(msgspec.msgpack.encode({"type": message_type, **fields}))

    def wait_for(self, predicate):
        """Pump queued interactions until an observable condition is satisfied."""
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            self.viewer._process_interaction_events()
            if predicate():
                return
            time.sleep(0.01)
        self.fail("Timed out waiting for Viser state")

    def update_gui(self, handle, value):
        """Send a GUI update through the same path used by the browser."""
        self.send("GuiUpdateMessage", uuid=handle._impl.uuid, updates={"value": value})

    def make_model(self, device="cpu"):
        """Create two colored bodies with different shapes and a ground plane."""
        builder = newton.ModelBuilder()
        body = builder.add_body(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.2, hy=0.2, hz=0.2)
        body = builder.add_body(xform=wp.transform((1.0, 0.0, 1.0), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.2)
        builder.add_ground_plane()
        model = builder.finalize(device=device)
        model.shape_color.assign(np.tile((0.8, 0.2, 0.1), (model.shape_count, 1)))
        self.viewer.set_model(model)
        return model

    def test_hidden_shapes_preserve_colors(self):
        """Restore colored rigid geometry after toggling visual and collision layers."""
        model = self.make_model()
        state = model.state()
        self.viewer.log_state(state)
        original = {
            name: h.batched_colors.copy()
            for name, h in self.viewer._scene_handles.items()
            if hasattr(h, "batched_colors")
        }
        handles = {name: self.viewer._scene_handles[name] for name in original}
        self.assertTrue(original)
        self.viewer.show_visual = False
        self.viewer.show_collision = False
        self.viewer.log_state(state)
        self.viewer.show_visual = True
        self.viewer.log_state(state)
        for name, colors in original.items():
            self.assertIs(self.viewer._scene_handles[name], handles[name])
            np.testing.assert_array_equal(self.viewer._scene_handles[name].batched_colors, colors)

    def test_clear_releases_hidden_mesh_assets(self):
        """Release cached model geometry after clearing the scene for an example switch."""
        model = self.make_model()
        self.viewer.log_state(model.state())
        self.assertTrue(self.viewer._meshes)
        self.viewer.clear_model()
        self.assertFalse(self.viewer._meshes)
        self.assertFalse(self.viewer._instances)

    def test_scene_switch_resets_camera_but_layer_clear_preserves_it(self):
        """Reset a whole-scene camera while preserving it for individual layer clears."""
        default = tuple(value.copy() for value in self.viewer._camera_request)
        self.viewer.set_camera_look_at(wp.vec3(1.0, 2.0, 3.0), wp.vec3(0.0), fov=70.0)
        self.viewer.clear_model()
        np.testing.assert_allclose(self.viewer._camera_request[0], (1.0, 2.0, 3.0))
        self.viewer.clear_all_layers()
        for actual, expected in zip(self.viewer._camera_request, default, strict=True):
            np.testing.assert_allclose(actual, expected)
        self.assertAlmostEqual(self.viewer._camera_fov_radians, np.deg2rad(45.0))

    def test_pause_step_and_example_selection(self):
        """Apply browser pause, single-step, reset and example-selection commands."""
        selected = []
        resets = []
        self.viewer.configure_example_browser({"basic": [("A", "module.a"), ("B", "module.b")]}, selected.append)
        self.viewer.set_reset_callback(lambda: resets.append(True))
        self.update_gui(self.viewer._simulation_gui_handles["pause"], True)
        self.wait_for(self.viewer.is_paused)
        self.assertFalse(self.viewer.should_step())
        self.update_gui(self.viewer._simulation_gui_handles["step"], True)
        self.wait_for(lambda: self.viewer._step_requested)
        self.assertTrue(self.viewer.should_step())
        self.assertFalse(self.viewer.should_step())
        self.update_gui(self.viewer._simulation_gui_handles["reset"], True)
        self.wait_for(lambda: bool(resets))
        self.update_gui(self.viewer._example_browser_handles["dropdown"], "basic / B")
        self.wait_for(lambda: self.viewer._example_browser_handles["dropdown"].value == "basic / B")
        self.update_gui(self.viewer._example_browser_handles["load"], True)
        self.wait_for(lambda: bool(selected))
        self.assertEqual(selected, ["module.b"])

    def test_gizmo_roundtrip_and_disconnect(self):
        """Mutate a caller-owned transform through real drag messages and release on disconnect."""
        transform = wp.transform_identity()
        snap = wp.transform((1.0, 2.0, 3.0), wp.quat_identity())
        self.viewer.log_gizmo("target", transform, snap_to=snap)
        handle = self.viewer._gizmo_handles["target"]["handles"]["translate"]
        self.send("TransformControlsDragStartMessage", name=handle.name)
        self.wait_for(lambda: self.viewer.gizmo_is_using)
        self.send(
            "TransformControlsUpdateMessage", name=handle.name, position=(4.0, 5.0, 6.0), wxyz=(1.0, 0.0, 0.0, 0.0)
        )
        self.wait_for(lambda: np.allclose(np.asarray(transform)[:3], (4.0, 5.0, 6.0)))
        self.socket.close()
        self.wait_for(lambda: not self.viewer.gizmo_is_using)
        np.testing.assert_allclose(np.asarray(transform), np.asarray(snap))

    def test_image_atlas_and_transport(self):
        """Encode real image atlases and preserve transparency when switching streams."""
        from PIL import Image

        image = np.zeros((2, 8, 8, 4), dtype=np.uint8)
        image[0, ..., 0] = 255
        image[1, ..., 1] = 255
        image[..., 3] = 255
        self.viewer.log_image("color", image)
        self.viewer.log_image("alpha", image)
        self.update_gui(self.viewer._image_selector, "alpha")
        self.wait_for(lambda: self.viewer._selected_image_name == "alpha")
        image[..., 3] = 64
        self.viewer.log_image("alpha", image)
        self.server.flush()
        self.wait_for(
            lambda: any(m["type"] == "GuiImageMessage" and m["props"]["_format"] == "png" for m in self.messages)
        )
        message = next(
            m for m in reversed(self.messages) if m["type"] == "GuiImageMessage" and m["props"]["_format"] == "png"
        )
        decoded = np.asarray(Image.open(io.BytesIO(message["props"]["_data"])))
        self.assertEqual(decoded.shape, (8, 16, 4))
        np.testing.assert_array_equal(decoded[:, :8], image[0])
        np.testing.assert_array_equal(decoded[:, 8:], image[1])

    def test_geometry_updates_and_clear(self):
        """Update real mesh and particle handles then remove model-owned scene nodes."""
        points = wp.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=wp.vec3, device="cpu")
        indices = wp.array((0, 1, 2), dtype=wp.int32, device="cpu")
        self.viewer.log_mesh("cloth", points, indices, backface_culling=False)
        mesh = self.viewer._scene_handles["cloth"]
        points.assign(((0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)))
        self.viewer.log_mesh("cloth", points, indices, backface_culling=False)
        self.assertIs(mesh, self.viewer._scene_handles["cloth"])
        np.testing.assert_allclose(mesh.vertices, points.numpy())
        self.viewer._set_wireframe(True)
        self.assertTrue(mesh.wireframe)
        self.viewer.log_points("particles", points, radii=0.1, colors=(0.8, 0.2, 0.1))
        cloud = self.viewer._scene_handles["particles"]
        self.viewer.log_points("particles", points, hidden=True)
        self.assertFalse(cloud.visible)
        self.viewer.log_points("particles", points, radii=0.1)
        self.assertTrue(cloud.visible)
        np.testing.assert_array_equal(cloud.colors, (204, 51, 26))
        self.viewer.clear_model()
        self.assertFalse(self.viewer._scene_handles)

    def test_scalar_plot_real_protocol(self):
        """Send only committed finite samples to Viser's real plot serializer."""
        for value in (2.0, 4.0, 6.0, 8.0):
            self.viewer.log_scalar("force", value, smoothing=2)
        self.viewer.begin_frame(0.1)
        self.viewer.end_frame()
        self.assertIn("force", self.viewer._plot_handles)
        x, y = self.viewer._plot_handles["force"].data
        np.testing.assert_allclose(y, (3.0, 7.0))
        self.assertTrue(np.isfinite(x).all())

    def test_invalid_fov_leaves_camera_unchanged(self):
        """Reject invalid projection angles before changing the camera pose."""
        before = tuple(v.copy() for v in self.viewer._camera_request)
        for fov in (float("nan"), float("inf"), 0.0, -1.0, 180.0, 181.0):
            with self.subTest(fov=fov):
                with self.assertRaises(ValueError):
                    self.viewer.set_camera_look_at(wp.vec3(1.0, 2.0, 3.0), wp.vec3(0.0), fov=fov)
                for actual, expected in zip(self.viewer._camera_request, before, strict=True):
                    np.testing.assert_array_equal(actual, expected)

    def test_url_uses_bound_port(self):
        """Report the actual listening port when Viser selects an available port."""
        self.assertEqual(self.viewer.url, f"http://localhost:{self.server.get_port()}")

    def test_picking_applies_force_to_hit_body(self):
        """Raycast browser clicks and apply a real spring force only to the hit body."""
        for device in ("cpu", "cuda:0") if wp.is_cuda_available() else ("cpu",):
            with self.subTest(device=device):
                self.viewer.clear_model()
                model = self.make_model(device)
                state = model.state()
                self.viewer.log_state(state)
                graph = None
                if model.device.is_cuda:
                    # Examples capture before the first browser pick. The
                    # captured kernel must observe later pick-state changes.
                    with wp.ScopedCapture(device=model.device) as capture:
                        state.clear_forces()
                        self.viewer.apply_forces(state)
                    graph = capture.graph
                batch = next(b for b in self.viewer._shape_instances.values() if b.geo_type == newton.GeoType.BOX)
                self.send(
                    "SceneNodeClickMessage",
                    name=batch.name,
                    instance_index=0,
                    ray_origin=(0.0, -3.0, 1.0),
                    ray_direction=(0.0, 1.0, 0.0),
                    screen_pos=(0.5, 0.5),
                    modifier=None,
                )
                self.wait_for(self.viewer.picking.is_picking)
                self.assertEqual(int(self.viewer.picking.pick_body.numpy()[0]), 0)
                handle = self.viewer._picking_controls[self.viewer.layer.layer_id]
                self.send(
                    "TransformControlsUpdateMessage",
                    name=handle.name,
                    position=(0.0, -1.0, 1.0),
                    wxyz=(1.0, 0.0, 0.0, 0.0),
                )
                self.wait_for(
                    lambda: np.isclose(self.viewer.picking.pick_state.numpy()[0]["picking_target_world"][1], -1.0)
                )
                state.clear_forces()
                if graph is None:
                    self.viewer.apply_forces(state)
                else:
                    wp.capture_launch(graph)
                force = state.body_f.numpy()
                self.assertGreater(np.linalg.norm(force[0]), 0.0)
                np.testing.assert_array_equal(force[1], 0.0)
                self.send("TransformControlsDragEndMessage", name=handle.name)
                self.wait_for(lambda: not self.viewer.picking.is_picking())
                state.clear_forces()
                self.viewer.apply_forces(state)
                np.testing.assert_array_equal(state.body_f.numpy(), 0.0)

    def test_example_hold_and_replacement(self):
        """Drive press-and-hold controls and discard events for replaced widgets."""
        state = {"label": "Forward", "active": False, "disabled": False}

        def gui(ui):
            ui.begin_disabled(state["disabled"])
            ui.button(state["label"])
            state["active"] = ui.is_item_active()
            ui.end_disabled()

        self.viewer.register_ui_callback(gui)
        self.viewer.should_step()
        handle = self.viewer._example_gui_handles[(0, 0)][1]
        self.send("GuiButtonHoldMessage", uuid=handle._impl.uuid, frequency=30.0)
        self.wait_for(lambda: bool(self.viewer._example_gui_held))
        self.viewer.should_step()
        self.assertTrue(state["active"])
        state["label"] = "Backward"
        self.viewer.should_step()
        self.assertFalse(state["active"])
        self.viewer._interaction_events.put(("example_gui", (0, 0), handle, True))
        self.viewer._interaction_events.put(("example_hold", (0, 0), handle, 0, time.monotonic()))
        self.viewer.should_step()
        self.assertFalse(state["active"])
        self.assertFalse(self.viewer._example_gui_pending)
        replacement = self.viewer._example_gui_handles[(0, 0)][1]
        self.send("GuiButtonHoldMessage", uuid=replacement._impl.uuid, frequency=30.0)
        self.wait_for(lambda: bool(self.viewer._example_gui_held))
        self.socket.close()
        self.wait_for(lambda: not self.viewer._example_gui_held)
        self.viewer.should_step()
        self.assertFalse(state["active"])

    def test_picking_does_not_cross_simulation_layers(self):
        """Keep picking forces in their owning model across layer activation and graph replay."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        layers = []
        for name in ("first", "second"):
            self.viewer.activate(name)
            model = self.make_model(device)
            state = model.state()
            self.viewer.log_state(state)
            graph = None
            if model.device.is_cuda:
                with wp.ScopedCapture(device=model.device) as capture:
                    state.clear_forces()
                    self.viewer.apply_forces(state)
                graph = capture.graph
            layers.append((name, state, graph))

        self.viewer._start_picking("first", (0.0, -3.0, 1.0), (0.0, 1.0, 0.0))
        self.assertTrue(self.viewer._layers["first"].picking.is_picking())
        self.viewer._set_picking_target("first", (0.0, -1.0, 1.0))
        for name, state, graph in layers:
            self.viewer.activate(name)
            if graph is None:
                state.clear_forces()
                self.viewer.apply_forces(state)
            else:
                wp.capture_launch(graph)
        self.assertGreater(np.linalg.norm(layers[0][1].body_f.numpy()), 0.0)
        np.testing.assert_array_equal(layers[1][1].body_f.numpy(), 0.0)
        self.viewer.picking_enabled = False
        name, state, graph = layers[0]
        self.viewer.activate(name)
        if graph is None:
            state.clear_forces()
            self.viewer.apply_forces(state)
        else:
            wp.capture_launch(graph)
        np.testing.assert_array_equal(state.body_f.numpy(), 0.0)

    def test_stale_gizmo_does_not_mutate_replacement(self):
        """Discard queued drag events after replacing a same-named gizmo."""
        original = wp.transform_identity()
        self.viewer.log_gizmo("target", original)
        handles = self.viewer._gizmo_handles["target"]["handles"]
        self.viewer.clear_model()
        replacement = wp.transform_identity()
        self.viewer.log_gizmo("target", replacement)
        self.viewer._interaction_events.put(("gizmo_update", "target", handles, (4.0, 5.0, 6.0), (1.0, 0.0, 0.0, 0.0)))
        self.viewer._interaction_events.put(("gizmo_drag_start", "target", handles, 0))
        self.viewer.should_step()
        np.testing.assert_array_equal(np.asarray(replacement), np.asarray(wp.transform_identity()))
        self.assertFalse(self.viewer.gizmo_is_using)

    def test_layer_controls_hide_and_restore_real_geometry(self):
        """Toggle a model layer through the browser and discard controls on scene clear."""
        self.viewer.activate("XPBD")
        model = self.make_model()
        state = model.state()
        self.viewer.log_state(state)
        self.viewer.should_step()
        control = self.viewer._layer_gui_handles["XPBD"][1]
        handles = dict(self.viewer._scene_handles)
        self.update_gui(control, False)
        self.wait_for(lambda: not self.viewer.layer.visible)
        self.viewer.log_state(state)
        self.assertTrue(handles)
        self.assertFalse(any(handle.visible for handle in handles.values()))
        self.update_gui(control, True)
        self.wait_for(lambda: self.viewer.layer.visible)
        self.viewer.log_state(state)
        self.assertTrue(any(handle.visible for handle in handles.values()))
        self.viewer.clear_all_layers()
        self.viewer.should_step()
        self.assertFalse(self.viewer._layer_gui_handles)
        self.assertFalse(self.viewer._layer_gui_folder.visible)

    def test_recording_survives_time_reset(self):
        """Keep recording timestamps monotonic when an example resets its simulation clock."""
        import msgspec  # noqa: PLC0415
        import zstandard  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reset.viser"
            viewer = ViewerViser(port=0, verbose=False, record_to_viser=str(path))
            try:
                for t in (0.0, 0.1, 0.2, 0.0, 0.1):
                    viewer.begin_frame(t)
                    viewer.log_gizmo("target", wp.transform((t, 0.0, 0.0), wp.quat_identity()))
                    viewer.end_frame()
                viewer.save_recording()
                data = zstandard.ZstdDecompressor().decompress(path.read_bytes()[8:])
                size = int.from_bytes(data[:8], "little")
                recording = msgspec.msgpack.decode(data[8 : 8 + size])
                timestamps = [t for t, _message in recording["messages"]]
                self.assertEqual(timestamps, sorted(timestamps))
                self.assertAlmostEqual(recording["durationSeconds"], 0.3)
            finally:
                viewer.close()


@unittest.skipUnless(importlib.util.find_spec("trimesh") is not None, "Requires trimesh")
@unittest.skipUnless(importlib.util.find_spec("PIL") is not None, "Requires Pillow")
class TestViewerViserTextures(unittest.TestCase):
    def _roundtrip_textured_mesh(self, channels):
        import trimesh

        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.uint32)
        uvs = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
        texture = np.arange(2 * 2 * channels, dtype=np.uint8).reshape(2, 2, channels) * 16
        mesh = ViewerViser._build_trimesh_mesh(points, indices, uvs, texture)
        self.assertIsNotNone(mesh)

        # Exercise the actual glTF export/import used by Viser, including glTF defaults.
        scene = trimesh.load_scene(io.BytesIO(mesh.export(file_type="glb")), file_type="glb", process=False)
        loaded = next(iter(scene.geometry.values()))
        np.testing.assert_array_equal(loaded.visual.material.baseColorTexture, texture)
        np.testing.assert_allclose(loaded.visual.uv, uvs)
        return loaded.visual.material

    def test_textured_mesh_preserves_texture_brightness(self):
        """Export RGB and RGBA textures without an unintended gray multiplier."""
        for channels in (3, 4):
            with self.subTest(channels=channels):
                material = self._roundtrip_textured_mesh(channels)
                np.testing.assert_array_equal(material.baseColorFactor, [255, 255, 255, 255])

    def test_textured_mesh_is_nonmetallic(self):
        """Keep textured meshes nonmetallic after glTF applies material defaults."""
        material = self._roundtrip_textured_mesh(3)
        self.assertEqual(material.metallicFactor, 0.0)


if __name__ == "__main__":
    unittest.main()
