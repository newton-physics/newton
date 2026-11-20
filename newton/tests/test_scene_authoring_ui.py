# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Native authoring interaction coverage, separate from CPU document tests."""

import hashlib
import json
import threading
import time
import unittest
from io import BytesIO
from pathlib import Path
from unittest import mock

import numpy as np
import warp as wp

from newton import ModelBuilder, ModelFlags
from newton._src.viewer.gui.authoring import intersect_ray
from newton._src.viewer.gui.viewer_gui import ViewerGui
from newton._src.viewer.gui.widgets import PhysicsWidgets
from newton.examples import _ExampleBrowser
from newton.examples.basic.example_scene_editor import Example
from newton.tests import test_viewer_assets
from newton.tests.test_viewer_get_frame import _make_headless_viewer_gl_or_skip
from newton.tests.unittest_utils import get_cuda_test_devices, has_module, requires


class TestSceneAuthoringUI(unittest.TestCase):
    def test_drag_commit_undo_and_workspace_lifetime(self):
        """Deliver a native palette drag once and retain its presenter through model swaps."""
        try:
            from imgui_bundle import imgui
        except ImportError as error:
            self.skipTest(f"Viewer UI dependencies not available: {error}")

        previous_context = imgui.get_current_context()
        context = imgui.create_context()
        imgui.set_current_context(context)

        def restore_context():
            imgui.destroy_context(context)
            if previous_context is not None:
                imgui.set_current_context(previous_context)

        self.addCleanup(restore_context)
        devices = get_cuda_test_devices()[:1]
        mujoco = bool(devices) and all(has_module(name) for name in ("mujoco", "mujoco_warp"))
        with wp.ScopedDevice(devices[0] if mujoco else "cpu"):
            viewer = _make_headless_viewer_gl_or_skip(self, width=1280, height=900)
            self.addCleanup(viewer.close)
            viewer._paused = True
            viewer.gui = ViewerGui(viewer, viewer.renderer.window, workspace_enabled=True)
            self.assertTrue(viewer.gui.is_available)
            viewer.gui.ui.io.set_ini_filename("")
            editor = Example(
                viewer, Example.create_parser().parse_args(["--empty", "--solver", "mujoco" if mujoco else "xpbd"])
            )
            self.addCleanup(editor.close)
            editor.add_primitive("box", position=(-1, 0, 1))
            editor.commit()
            viewer.set_camera(pos=(5, -7, 4), pitch=-23, yaw=125)

            def render(count=1, *, simulate=False):
                for _ in range(count):
                    editor.before_frame()
                    if simulate and viewer.should_step():
                        editor.step()
                    editor.render()
                    self.assertFalse(viewer.gui.ui.section_errors)

            def native_mouse(x, y, down=None, *, simulate=False):
                window = viewer.renderer.window
                native_y = window.height - y
                window.dispatch_event("on_mouse_motion", x, native_y, 0, 0)
                if down is not None:
                    window.dispatch_event("on_mouse_press" if down else "on_mouse_release", x, native_y, 1, 0)
                window.dispatch_events()
                render(simulate=simulate)

            def mouse(x, y, down=None):
                viewer.gui.ui.io.add_mouse_pos_event(x, y)
                if down is not None:
                    viewer.gui.ui.io.add_mouse_button_event(0, down)
                render()

            def click(x, y):
                mouse(x, y)
                mouse(x, y, True)
                mouse(x, y, False)
                render(3)

            render(4)
            workspace = viewer.gui.workspace
            panel = workspace.authoring
            self.assertIsNotNone(panel)
            self.assertIsNone(panel.assets)

            def scene_pixel(point):
                clip = (
                    viewer.camera.get_projection_matrix().reshape(4, 4).T
                    @ viewer.camera.get_view_matrix().reshape(4, 4).T
                    @ np.array((*point, 1.0))
                )
                ndc = clip[:3] / clip[3]
                x, y, width, height = workspace.rect
                return x + (ndc[0] + 1) * width / 2, y + (1 - ndc[1]) * height / 2

            def click_scene_point(point):
                pixel = scene_pixel(point)
                native_mouse(*pixel)
                native_mouse(*pixel, True)
                native_mouse(*pixel, False)
                render(2)

            # Reuse an unchanged placement query, but invalidate on published
            # poses, camera changes and notified native geometry edits.
            obj = editor.objects[0]
            pixel = tuple(float(value) for value in scene_pixel(obj.position))
            with mock.patch("newton._src.viewer.gui.authoring.intersect_ray", wraps=intersect_ray) as raycast:
                hit = panel.world_position(pixel)
                np.testing.assert_allclose(panel.world_position(pixel), hit)
                self.assertEqual(raycast.call_count, 1)
                editor.update_object(obj.id, position=(3, 0, 1))
                np.testing.assert_allclose(panel.world_position(pixel), hit)
                self.assertEqual(raycast.call_count, 1)
                editor.before_frame()
                moved = panel.world_position(pixel)
                self.assertAlmostEqual(moved[2], 0, places=5)
                self.assertGreater(np.linalg.norm(np.asarray(moved) - hit), 0.1)
                self.assertEqual(raycast.call_count, 2)
                editor.update_object(obj.id, position=obj.position)
                editor.before_frame()
                np.testing.assert_allclose(panel.world_position(pixel), hit, atol=1e-5)
                self.assertEqual(raycast.call_count, 3)
                viewer.camera.pitch -= 3
                panel.world_position(pixel)
                self.assertEqual(raycast.call_count, 4)
                viewer.camera.pitch += 3
                panel.world_position(pixel)
                self.assertEqual(raycast.call_count, 5)
                viewer.notify_model_changed(ModelFlags.SHAPE_PROPERTIES)
                panel.world_position(pixel)
                self.assertEqual(raycast.call_count, 6)

                # A native joint edit changes State in place without changing
                # the authored document or advancing simulation time.
                before = editor.state_0.joint_q.numpy().copy()
                moved_q = before.copy()
                moved_q[0] = 3
                self.assertTrue(workspace.apply_edit(editor.state_0, "joint_q", 0, moved_q, joint_index=0))
                native_hit = panel.world_position(pixel)
                self.assertAlmostEqual(native_hit[2], 0, places=5)
                self.assertEqual(raycast.call_count, 7)
                self.assertTrue(workspace.apply_edit(editor.state_0, "joint_q", 0, before, joint_index=0))
                np.testing.assert_allclose(panel.world_position(pixel), hit, atol=1e-5)
                self.assertEqual(raycast.call_count, 8)

            if mujoco:
                # Explicit inspection cooks paused contacts once; ordinary
                # browsing and pose edits leave complete collision to Play.
                collision = workspace.collision
                workspace.tree.expanded.discard(collision.root)
                viewer.show_contacts = False
                editor.update_object(obj.id, position=(-1, 0, 0.5))
                editor.before_frame()
                with mock.patch.object(
                    editor.collision_pipeline, "collide", wraps=editor.collision_pipeline.collide
                ) as cook:
                    render(2)
                    self.assertEqual(cook.call_count, 0)
                    np.testing.assert_array_equal(editor.contacts.contact_counters.numpy(), 0)
                    workspace.history.set_recording(True)
                    workspace.tree.expanded.add(collision.root)
                    collision.sync(force=True)
                    owner = ("body", editor.body_indices(obj.id)[0])
                    self.assertFalse(collision.object_contacts(owner))
                    render(2, simulate=True)
                    self.assertEqual(cook.call_count, 1)
                    self.assertGreater(int(editor.contacts.rigid_contact_count.numpy()[0]), 0)
                    collision.sync()
                    self.assertTrue(collision.object_contacts(owner))
                    self.assertGreater(workspace.history.frames[-1].touching, 0)
                    workspace.history.set_recording(False)
                    self.assertFalse(viewer.show_contacts)
                    workspace.tree.expanded.discard(collision.root)
                    editor.update_object(obj.id, position=(-1, 0, 0.4))
                    editor.before_frame()
                    viewer.show_contacts = True
                    render(2)
                    self.assertEqual(cook.call_count, 2)
                    self.assertGreater(int(editor.contacts.rigid_contact_count.numpy()[0]), 0)
                viewer.show_contacts = False
                editor.update_object(obj.id, position=obj.position)
                editor.before_frame()

            # Deliver actual viewport clicks through the native release handler,
            # including the raycast used before authoring handles become active.
            click_scene_point((0, 0, 0))
            self.assertEqual(workspace.selected, ("shape", 0))
            obj = editor.objects[0]
            click_scene_point(obj.position)
            self.assertEqual(workspace.selected, ("body", editor.body_indices(obj.id)[0]))
            viewer.show_ground = False
            click_scene_point((0, 0, 0))
            self.assertIsNone(workspace.selected)
            viewer.show_ground = True

            initial_count, initial_builds, initial_model = len(editor.objects), editor.rebuild_count, editor.model
            # Fixed dimensions keep the sphere card stable. A held preview
            # must not compile the physical scene or its solver.
            solver = editor.solver
            mouse(210, 210)
            mouse(210, 210, True)
            mouse(225, 215)
            x, y, width, height = workspace.rect
            target = x + width * 0.45, y + height * 0.5
            mouse(*target)
            render(3)
            self.assertEqual(len(editor.objects), initial_count)
            self.assertEqual(editor.rebuild_count, initial_builds)
            self.assertIs(editor.model, initial_model)
            self.assertIs(editor.solver, solver)
            self.assertFalse(editor._runtime.solver_graphs)
            mouse(*target, False)
            render(4)
            self.assertEqual(len(editor.objects), initial_count + 1)
            self.assertEqual(editor.rebuild_count, initial_builds + 1)
            self.assertIs(workspace.authoring, panel)
            self.assertEqual(editor.objects[-1].kind, editor.Shape.SPHERE)
            self.assertFalse(editor._runtime.solver_graphs)
            if mujoco:
                self.assertIsNone(editor.solver)

            # Next is an explicit physics request while paused. Its first
            # solver binding refresh must keep the selected authored object.
            selected_id, selected_key = editor.selected_id, workspace.selected
            before = editor.sim_time
            viewer._step_requested = True
            if mujoco:
                create_solver, frame = editor._create_solver, viewer._frame_count

                def start_after_paint(*args, **kwargs):
                    self.assertTrue(viewer.gui._loading_splash_active)
                    self.assertEqual(viewer.gui._loading_splash_text, "Starting physics…")
                    self.assertGreater(viewer._frame_count, frame)
                    return create_solver(*args, **kwargs)

                viewer._headless = False
                try:
                    with mock.patch.object(editor, "_create_solver", start_after_paint):
                        render(simulate=True)
                finally:
                    viewer._headless = True
                self.assertFalse(viewer.gui._loading_splash_active)
            else:
                render(simulate=True)
            workspace._ensure_tree()
            render(2)
            self.assertTrue(viewer.is_paused())
            self.assertAlmostEqual(editor.sim_time - before, editor.frame_dt)
            self.assertIsNotNone(editor.solver)
            self.assertIs(workspace.tree.bindings.solver, editor.solver)
            self.assertEqual(editor.selected_id, selected_id)
            self.assertEqual(workspace.selected, selected_key)
            if mujoco:
                graphs = {key: id(value[0]) for key, value in editor._runtime.solver_graphs.items()}
                self.assertTrue(graphs)
                self.assertLessEqual(len(graphs), 2)
                render(3)
                viewer._step_requested = True
                render(simulate=True)
                self.assertEqual({key: id(value[0]) for key, value in editor._runtime.solver_graphs.items()}, graphs)

            click(94, 94)  # Undo from the authoring toolbar.
            self.assertEqual(len(editor.objects), initial_count)
            click(122, 94)  # Redo from the same toolbar.
            self.assertEqual(len(editor.objects), initial_count + 1)
            self.assertIs(workspace.authoring, panel)

            # Use the hidden native GL window, but exercise the application's
            # interactive loading path before a real selected-object rebuild.
            viewer._headless = False
            try:
                deleted = editor.objects[-1].id

                def check_delete(fail):
                    editor.select(deleted)
                    render(2)
                    delete_position = None
                    button = panel.widgets.icon_button

                    def locate_delete(item_id, *args, **kwargs):
                        nonlocal delete_position
                        clicked = button(item_id, *args, **kwargs)
                        if item_id == "##delete_object":
                            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
                            delete_position = ((lo.x + hi.x) / 2, (lo.y + hi.y) / 2)
                        return clicked

                    with mock.patch.object(panel.widgets, "icon_button", locate_delete):
                        render()
                    self.assertIsNotNone(delete_position)
                    before_pixels = viewer.get_frame(render_ui=True).numpy().copy()
                    height, width = before_pixels.shape[:2]
                    crop = np.s_[height // 2 - 100 : height // 2 + 100, width // 2 - 160 : width // 2 + 160]
                    model, paused, frame = editor.model, viewer.is_paused(), viewer._frame_count
                    camera = tuple(viewer.camera.pos), viewer.camera.pitch, viewer.camera.yaw
                    presented, loading_pixels = [], []
                    present, finalize = viewer.renderer.present, ModelBuilder.finalize

                    def present_loading():
                        present()
                        if viewer.gui._loading_splash_active:
                            presented.append(viewer._frame_count)

                    def finalize_after_paint(builder, *args, **kwargs):
                        self.assertTrue(viewer.gui._loading_splash_active)
                        self.assertTrue(presented)
                        self.assertGreater(viewer._frame_count, frame)
                        pixels = viewer.get_frame(render_ui=True).numpy().copy()
                        self.assertFalse(np.array_equal(pixels[crop], before_pixels[crop]))
                        loading_pixels.append(pixels[crop])
                        candidate = finalize(builder, *args, **kwargs)
                        if fail:
                            raise MemoryError("Interrupted native scene rebuild")
                        return candidate

                    with (
                        mock.patch.object(viewer.renderer, "present", present_loading),
                        mock.patch.object(ModelBuilder, "finalize", finalize_after_paint),
                    ):
                        if fail:
                            with self.assertLogs("newton", level="ERROR"):
                                click(*delete_position)
                        else:
                            click(*delete_position)
                    self.assertEqual(len(loading_pixels), 1)
                    self.assertFalse(viewer.gui._loading_splash_active)
                    self.assertEqual(viewer.is_paused(), paused)
                    self.assertEqual((tuple(viewer.camera.pos), viewer.camera.pitch, viewer.camera.yaw), camera)
                    self.assertEqual(len(editor.objects), initial_count)
                    if fail:
                        self.assertIs(editor.model, model)
                        self.assertIn("Interrupted native scene rebuild", editor.error)
                    else:
                        self.assertIsNot(editor.model, model)
                        self.assertEqual(editor.model.body_count, initial_count)
                    click(94, 94)
                    self.assertEqual(len(editor.objects), initial_count + 1)
                    self.assertIsNone(editor.error)
                    self.assertFalse(viewer.gui._loading_splash_active)

                check_delete(False)
                check_delete(True)
            finally:
                viewer._headless = True

            from pyglet.window import key

            builds = editor.rebuild_count
            viewer.gui.handle_key_press(key.H)
            render(2)
            self.assertFalse(viewer.gui.show_ui)
            viewer.gui.handle_key_press(key.H)
            render(3)
            self.assertTrue(viewer.gui.show_ui)
            self.assertEqual(editor.rebuild_count, builds)
            self.assertIs(workspace.authoring, panel)
            self.assertGreater(np.ptp(viewer.get_frame(render_ui=True).numpy()), 0)

            # H must also finish a numeric drag. Otherwise its open gesture
            # indefinitely defers structural publication while the UI is hidden.
            edited = editor.objects[-1]
            editor.select(edited.id)
            render(4)
            builds = editor.rebuild_count
            diameter_position = None
            number = panel.widgets.number

            def capture_diameter(label, *args, **kwargs):
                nonlocal diameter_position
                edit = number(label, *args, **kwargs)
                if label == "Diameter":
                    lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
                    diameter_position = ((lo.x + hi.x) / 2, (lo.y + hi.y) / 2)
                return edit

            with mock.patch.object(panel.widgets, "number", capture_diameter):
                render()
            self.assertIsNotNone(diameter_position)
            dx, dy = diameter_position
            mouse(dx, dy)
            mouse(dx, dy, True)
            mouse(dx - 25, dy)
            render(2)
            self.assertNotEqual(editor.get_object(edited.id).size, edited.size)
            self.assertEqual(editor.rebuild_count, builds)
            viewer.gui.handle_key_press(key.H)
            render(3)
            self.assertFalse(viewer.gui.show_ui)
            self.assertFalse(editor.pending)
            self.assertEqual(editor.rebuild_count, builds + 1)
            viewer.gui.handle_key_press(key.H)
            render(2)
            self.assertTrue(viewer.gui.show_ui)

            # Keep a real viewport handle active across physics/frame boundaries,
            # rather than publishing only one edit before releasing the mouse.
            workspace.history.set_recording(True)
            workspace.history.playback_speed = 2.0
            viewer._step_requested = True
            render(simulate=True)
            self.assertTrue(workspace.history.frames)
            original = editor.get_object(edited.id)
            gx, gy = scene_pixel(editor.get_pose(edited.id)[0])
            gy -= 20
            sim_time = editor.sim_time
            viewer._paused = False
            native_mouse(gx, gy, simulate=True)
            self.assertTrue(viewer.gui._gizmo_hovered)
            native_mouse(gx, gy, True, simulate=True)
            self.assertTrue(panel._gizmo_active)
            self.assertTrue(viewer.is_paused())
            model, builds = editor.model, editor.rebuild_count
            last_z = editor.get_pose(edited.id)[0][2]
            deadline = time.monotonic() + 2.1
            moved = 0
            while time.monotonic() < deadline:
                moved += 1
                native_mouse(gx, gy - 2 * moved, simulate=True)
                self.assertEqual(editor.selected_id, edited.id)
                self.assertTrue(panel._gizmo_active)
                self.assertTrue(viewer.is_paused())
                self.assertIs(editor.model, model)
                self.assertEqual(editor.rebuild_count, builds)
                self.assertTrue(workspace.history.recording)
                self.assertTrue(workspace.history.frames)
                self.assertEqual(workspace.history.playback_speed, 2.0)
                position = editor.get_object(edited.id).position
                self.assertGreater(position[2], last_z)
                last_z = position[2]
                time.sleep(0.05)
            native_mouse(gx, gy - 2 * moved, False, simulate=True)
            render(2, simulate=True)
            self.assertFalse(panel._gizmo_active)
            self.assertFalse(viewer.is_paused())
            self.assertTrue(workspace.history.recording)
            self.assertEqual(workspace.history.playback_speed, 2.0)
            self.assertGreater(editor.sim_time, sim_time)
            self.assertGreater(editor.get_object(edited.id).position[2], original.position[2])
            editor.undo()
            render()
            self.assertEqual(editor.get_object(edited.id).position, original.position)

            if has_module("pxr"):
                # Reuse the real HTTP package/cache fixture. Only worker timing
                # is controlled: parsing, metadata, USD import and native input
                # all follow their production paths.
                assets = test_viewer_assets.TestAssetLibrary()
                assets.setUp()
                self.addCleanup(assets.doCleanups)
                # The service fixture checks bytes; this workflow also imports
                # its USD and decodes its thumbnail, so supply complete content.
                assets.files["simready_usd/payloads/mesh.usda"] = (
                    b'#usda 1.0\n(defaultPrim = "Root")\ndef Xform "Root" {\n def Cube "Mesh" {}\n}\n'
                )
                from pyglet.image import SolidColorImagePattern

                image_bytes = BytesIO()
                SolidColorImagePattern((100, 180, 200, 255)).create_image(8, 8).save("thumbnail.png", file=image_bytes)
                assets.files["simready_usd/.thumbs/256x256/root.usda.png"] = image_bytes.getvalue()
                bom_path = ".metadata/com.nvidia.simready.packaging.bom.json"
                bom = json.loads(assets.files[bom_path])
                for item in bom["items"]:
                    data = assets.files[item["relative_path"]]
                    item.update(size=len(data), hash={"sha256": hashlib.sha256(data).hexdigest()})
                assets.files[bom_path] = json.dumps(bom).encode()
                manifest_path = "com.nvidia.simready.packaging.json"
                manifest = json.loads(assets.files[manifest_path])
                manifest["metadata"][0]["hash"]["sha256"] = hashlib.sha256(assets.files[bom_path]).hexdigest()
                assets.files[manifest_path] = json.dumps(manifest).encode()
                replacement_path = Path(assets.directory.name) / "replacement.usda"
                replacement_objects = editor.objects
                editor.save_usd(replacement_path)
                record = assets.record()
                assets.library.download(record)
                assets.until(lambda: assets.library.ready(record) is not None)
                release_thumbnail = threading.Event()
                self.addCleanup(release_thumbnail.set)
                thumbnail = assets.library._thumbnail

                def deferred_thumbnail(record):
                    release_thumbnail.wait(5)
                    return thumbnail(record)

                with mock.patch.object(assets.library, "_thumbnail", deferred_thumbnail):
                    panel.assets = assets.library
                    panel.query = panel._submitted_query = "sponge"

                    native_mouse(95, 123)
                    native_mouse(95, 123, True)
                    native_mouse(95, 123, False)
                    render(3)
                    native_mouse(75, 290)
                    native_mouse(75, 290, True)
                    native_mouse(75, 290, False)
                    render(2)
                    self.assertIsNotNone(panel._selected_asset)
                    self.assertEqual(panel._selected_asset.id, record.id)
                    self.assertIsNone(editor.selected_id)
                    before = len(editor.objects)
                    # A quick second press is also a potential double click.
                    # It must remain a drag until its release decides otherwise.
                    native_mouse(75, 290, True)
                    self.assertEqual(len(editor.objects), before)
                    release_thumbnail.set()
                    assets.until(lambda: not assets.library.loading)
                    render(3)
                    self.assertIsNotNone(panel._selected_asset)
                    self.assertEqual(panel._selected_asset.id, record.id)
                    self.assertEqual(panel._selected_asset.license, "CC-BY-4.0")
                    native_mouse(100, 295)
                    native_mouse(*target)
                    render(3)
                    self.assertEqual(len(editor.objects), before)
                    native_mouse(*target, False)
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline and (len(editor.objects) == before or editor.pending):
                        render()
                        time.sleep(0.01)
                    self.assertIsNone(editor.error)
                    self.assertEqual(len(editor.objects), before + 1)
                    imported = editor.objects[-1]
                    self.assertEqual(imported.kind.value, "usd")
                    self.assertTrue(editor.body_indices(imported.id))
                    self.assertTrue(panel._textures)
                    before = len(editor.objects)
                    # A different point starts a fresh double-click sequence
                    # without sleeping for ImGui's multi-click timeout.
                    native_mouse(110, 290)
                    native_mouse(110, 290, True)
                    native_mouse(110, 290, False)
                    native_mouse(110, 290, True)
                    self.assertEqual(len(editor.objects), before)
                    native_mouse(110, 290, False)
                    render(3)
                    self.assertEqual(len(editor.objects), before + 1)
                    self.assertTrue(editor.body_indices(editor.objects[-1].id))

                    native_mouse(75, 290)
                    native_mouse(75, 290, True)
                    native_mouse(75, 290, False)
                    render(3)
                    self.assertIsNotNone(panel._selected_asset)
                    self.assertEqual(panel._selected_asset.id, record.id)
                    # The large inspector preview is also a placement source.
                    before = len(editor.objects)
                    builds = editor.rebuild_count
                    model = editor.model
                    with mock.patch.object(viewer, "log_lines", wraps=viewer.log_lines) as lines:
                        native_mouse(1080, 222)
                        native_mouse(1080, 222, True)
                        native_mouse(1060, 230)
                        native_mouse(*target)
                        render(3)
                        outlines = [
                            call
                            for call in lines.call_args_list
                            if call.args[0] == "__scene_authoring/placement" and call.args[1] is not None
                        ]
                        self.assertTrue(outlines)
                        vertices = np.concatenate([array.numpy() for array in outlines[-1].args[1:3]])
                        lower, upper = vertices.min(axis=0), vertices.max(axis=0)
                        np.testing.assert_allclose(upper - lower, (0.02, 0.02, 0.02), atol=1e-6)
                    self.assertEqual(len(editor.objects), before)
                    self.assertEqual(editor.rebuild_count, builds)
                    self.assertIs(editor.model, model)
                    native_mouse(*target, False)
                    render(4)
                    self.assertIsNone(editor.error)
                    self.assertEqual(len(editor.objects), before + 1)
                    self.assertEqual(editor.rebuild_count, builds + 1)
                    self.assertEqual(editor.objects[-1].kind.value, "usd")
                    self.assertTrue(editor.body_indices(editor.objects[-1].id))
                    center = editor.get_pose(editor.objects[-1].id)[0]
                    np.testing.assert_allclose(lower, np.asarray(center) - 0.01, atol=1e-5)
                    np.testing.assert_allclose(upper, np.asarray(center) + 0.01, atol=1e-5)

                def hold_download():
                    (assets.library._package(record) / "complete.json").unlink()
                    assets.library._verified.pop(record.id, None)
                    assets.requests.clear()
                    assets.hold.set()
                    assets.release.clear()

                def queue_download():
                    # Request the same real package with an incomplete cache;
                    # hold its response until another interaction owns the UI.
                    hold_download()
                    panel._place_asset(record, (-3, 0, 1))
                    assets.until(lambda: any(path.endswith("LICENSE.txt") for path in assets.requests))
                    self.assertEqual(len(panel._pending_imports), 1)

                def complete_download():
                    assets.release.set()
                    assets.until(lambda: assets.library.ready(record) is not None)

                def await_import(count):
                    phases = set()
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline and (len(editor.objects) != count or editor.pending):
                        render()
                        phases.add(panel._asset_activity_phase)
                        time.sleep(0.01)
                    self.assertIsNone(editor.error)
                    self.assertEqual(len(editor.objects), count)
                    self.assertFalse(editor.pending)
                    return phases

                # A real held HTTP response keeps the named, nonmodal activity
                # visible and animated before and after the pointer releases.
                hold_download()
                before, model, sim_time = len(editor.objects), editor.model, editor.sim_time
                native_mouse(75, 290)
                native_mouse(75, 290, True)
                native_mouse(100, 295)
                native_mouse(*target)
                assets.until(lambda: any(path.endswith("LICENSE.txt") for path in assets.requests))
                download = assets.library._jobs["download", record.id]
                activity_pixels = []
                clock = viewer.gui.ui._frame_time
                with mock.patch(
                    "newton._src.viewer.gui.imgui_backend.perf_counter", side_effect=(clock + 0.1, clock + 0.3)
                ):
                    for _ in range(2):
                        render()
                        self.assertFalse(download.done())
                        self.assertEqual(panel._asset_activity[0].id, record.id)
                        self.assertEqual(panel._asset_activity_phase, "Downloading")
                        self.assertIsNotNone(imgui.get_drag_drop_payload_py_id())
                        self.assertFalse(viewer.gui._loading_splash_active)
                        self.assertEqual(editor.sim_time, sim_time)
                        self.assertEqual(len(editor.objects), before)
                        self.assertIs(editor.model, model)
                        x, y, width, height = panel._asset_activity_rect
                        pixels = viewer.get_frame(render_ui=True).numpy()
                        sx, sy = (
                            pixels.shape[1] / imgui.get_io().display_size.x,
                            pixels.shape[0] / imgui.get_io().display_size.y,
                        )
                        crop = pixels[
                            round(y * sy) : round((y + height) * sy), round(x * sx) : round((x + width) * sx)
                        ].copy()
                        self.assertGreater(np.ptp(crop[8:-8, round(48 * panel.widgets.scale * sx) : -8]), 50)
                        activity_pixels.append(crop)
                self.assertFalse(np.array_equal(*activity_pixels))
                native_mouse(*target, False)
                render(2)
                self.assertFalse(download.done())
                self.assertEqual(len(panel._pending_imports), 1)
                self.assertEqual(panel._asset_activity[0].id, record.id)
                self.assertEqual(panel._asset_activity_phase, "Downloading")
                self.assertIsNotNone(panel._asset_activity_rect)
                self.assertEqual(len(editor.objects), before)
                complete_download()
                phases = await_import(before + 1)
                self.assertTrue(phases & {"Preparing geometry", "Adding to scene"})
                render()
                self.assertIsNone(panel._asset_activity)
                self.assertIsNone(panel._asset_activity_rect)

                # A finished transfer must not mutate a historical scene.
                viewer._paused = True
                queue_download()
                history = workspace.history
                history.set_recording(True)
                render()
                editor.step()
                render()
                history.select(0)
                self.assertTrue(history.replaying)
                document = editor.objects
                complete_download()
                render(2, simulate=True)
                self.assertEqual(editor.objects, document)
                self.assertEqual(len(panel._pending_imports), 1)
                history.select(len(history.frames) - 1)
                await_import(len(document) + 1)

                # Likewise, retain the completed request until a real held
                # viewport gesture releases its single undo transaction.
                queue_download()
                editor.select(edited.id)
                render(3)
                gx, gy = scene_pixel(editor.get_pose(edited.id)[0])
                gy -= 20
                native_mouse(gx, gy, simulate=True)
                native_mouse(gx, gy, True, simulate=True)
                native_mouse(gx, gy - 4, simulate=True)
                self.assertTrue(panel._gizmo_active)
                self.assertIsNotNone(editor._edit_before)
                count = len(editor.objects)
                complete_download()
                render(2, simulate=True)
                self.assertEqual(len(editor.objects), count)
                self.assertEqual(len(panel._pending_imports), 1)
                self.assertTrue(panel._gizmo_active)
                native_mouse(gx, gy - 4, False, simulate=True)
                await_import(count + 1)

                # Opening a document supersedes the old drop request while its
                # verified package remains available in the shared cache.
                queue_download()
                editor.load_usd(replacement_path)
                complete_download()
                await_import(len(replacement_objects))
                self.assertEqual(editor.objects, replacement_objects)
                self.assertFalse(panel._pending_imports)
                self.assertIsNotNone(assets.library.ready(record))

            # The common viewer entry queues ownership changes, with an
            # explicit dialog before replacing a nonempty authored document.
            click(33, 123)  # Keep the creation palette visible across the switch.
            browser = _ExampleBrowser(viewer, Example.create_parser().parse_args(["--empty"]))
            positions = {}
            button = workspace.widgets.button

            def locate_scene_actions(label, *args, **kwargs):
                clicked = button(label, *args, **kwargs)
                if label in ("New scene", "Cancel", "Create new scene"):
                    lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
                    positions[label] = ((lo.x + hi.x) / 2, (lo.y + hi.y) / 2)
                return clicked

            previous, model, document, history = editor, editor.model, editor.objects, workspace.history
            with mock.patch.object(workspace.widgets, "button", locate_scene_actions):
                render(2)
                click(*positions["New scene"])
                self.assertIsNone(browser.switch_target)
                self.assertIs(editor.model, model)
                click(*positions["Cancel"])
                self.assertIsNone(browser.switch_target)
                self.assertEqual(editor.objects, document)
                click(*positions["New scene"])
                click(*positions["Create new scene"])
            self.assertEqual(browser._new_scene_options, ("mujoco", "cpu"))
            self.assertIs(editor.model, model)
            previous.close()
            editor = None
            editor, _ = browser.switch(Example)
            self.assertIsNotNone(editor, browser.load_error)
            self.addCleanup(editor.close)
            render(2)
            self.assertTrue(previous._closed)
            self.assertTrue(viewer.is_paused())
            self.assertEqual(editor.objects, ())
            self.assertTrue(editor.model.device.is_cpu)
            self.assertIsNone(editor.state_0.body_q)
            self.assertTrue(editor.model.shape_transform.device.is_cpu)
            self.assertEqual(editor.solver_type.value, "mujoco")
            self.assertIs(workspace.authoring.editor, editor)
            self.assertIsNot(workspace.history, history)
            self.assertIsNone(workspace.selected)
            self.assertIsNone(panel._file_drop_handler)
            editor.close()
            workspace.bind()
            self.assertIsNone(workspace.authoring)
            self.assertIsNone(panel._file_drop_handler)

    @requires("imgui_bundle", "pxr.Usd", "newton_usd_schemas")
    def test_deformable_controls_and_paused_particle_stroke(self):
        """Edit separate deformables picked in the viewport and publish a paused particle stroke."""
        from imgui_bundle import imgui

        previous_context = imgui.get_current_context()
        context = imgui.create_context()
        imgui.set_current_context(context)

        def restore_context():
            imgui.destroy_context(context)
            if previous_context is not None:
                imgui.set_current_context(previous_context)

        self.addCleanup(restore_context)
        targets = {}
        with wp.ScopedDevice("cpu"):
            viewer = _make_headless_viewer_gl_or_skip(self, width=1280, height=900)
            self.addCleanup(viewer.close)
            viewer._paused = True
            viewer.gui = ViewerGui(viewer, viewer.renderer.window, workspace_enabled=True)
            viewer.gui.ui.io.set_ini_filename("")
            editor = Example(
                viewer, Example.create_parser().parse_args(["--empty", "--solver", "vbd", "--device", "cpu"])
            )
            self.addCleanup(editor.close)

            def render(count=1, *, simulate=False):
                for _ in range(count):
                    editor.before_frame()
                    if simulate and viewer.should_step():
                        editor.step()
                    editor.render()
                    self.assertFalse(editor.error)
                    self.assertFalse(viewer.gui.ui.section_errors)

            def native_mouse(point, down=None):
                window = viewer.renderer.window
                window.dispatch_event("on_mouse_motion", point[0], window.height - point[1], 0, 0)
                if down is not None:
                    window.dispatch_event(
                        "on_mouse_press" if down else "on_mouse_release", point[0], window.height - point[1], 1, 0
                    )
                window.dispatch_events()
                render()

            def click(point):
                native_mouse(point)
                native_mouse(point, True)
                native_mouse(point, False)
                render(2)

            def scene_pixel(position):
                matrix = (
                    viewer.camera.get_projection_matrix().reshape(4, 4).T
                    @ viewer.camera.get_view_matrix().reshape(4, 4).T
                )
                clip = matrix @ np.array((*position, 1))
                ndc = clip[:2] / clip[3]
                x, y, width, height = viewer.gui.workspace.rect
                return x + (ndc[0] + 1) * width / 2, y + (1 - ndc[1]) * height / 2

            def track(original):
                def draw(owner, label, *args, **kwargs):
                    result = original(owner, label, *args, **kwargs)
                    lo, hi = owner.ui.get_item_rect_min(), owner.ui.get_item_rect_max()
                    targets[label] = ((lo.x + hi.x) / 2, (lo.y + hi.y) / 2)
                    return result

                return draw

            with (
                mock.patch.object(PhysicsWidgets, "button", track(PhysicsWidgets.button)),
                mock.patch.object(PhysicsWidgets, "number", track(PhysicsWidgets.number)),
                mock.patch.object(PhysicsWidgets, "tab_item", track(PhysicsWidgets.tab_item)),
            ):
                render(4)
                click(targets["Cloth sheet"])
                self.assertEqual(len(editor.objects), 1)
                cloth = editor.objects[0]
                self.assertEqual(cloth.kind.value, "cloth")
                self.assertGreater(editor.model.tri_count, 0)
                self.assertEqual(editor.selected_id, cloth.id)
                soft_id = editor.add_deformable(
                    "soft",
                    position=(cloth.position[0] + 1.5, cloth.position[1], cloth.position[2] + 0.55),
                    size=(0.5, 0.5, 0.5),
                    resolution=2,
                )
                render(2)
                soft = editor.get_object(soft_id)
                for obj, neighbor, kind in ((cloth, soft, "surface"), (soft, cloth, "volume")):
                    click(scene_pixel(obj.position))
                    self.assertEqual(editor.selected_id, obj.id)
                    self.assertEqual(viewer.gui.workspace.selected[0], kind)
                    original_neighbor = editor.get_object(neighbor.id)
                    model, builds = editor.model, editor.rebuild_count
                    density = targets["Density"]
                    native_mouse(density, True)
                    native_mouse((density[0] - 120, density[1]))
                    self.assertNotEqual(editor.get_object(obj.id).density, obj.density)
                    self.assertIs(editor.model, model)
                    native_mouse((density[0] - 120, density[1]), False)
                    render(2)
                    self.assertEqual(editor.rebuild_count, builds + 1)
                    self.assertEqual(editor.get_object(neighbor.id), original_neighbor)
                editor.close()

                devices = get_cuda_test_devices()[:1]
                if not devices:
                    return
                with wp.ScopedDevice(devices[0]):
                    editor = Example(
                        viewer,
                        Example.create_parser().parse_args(
                            ["--empty", "--solver", "mpm", "--device", devices[0].alias]
                        ),
                    )
                    self.addCleanup(editor.close)
                    editor.add_primitive("box", position=(0, 0, 0.2), size=(0.4, 0.4, 0.4), dynamic=False)
                    editor.commit()
                    viewer.set_camera(pos=(5, -7, 4), pitch=-23, yaw=125)
                    render(4)
                    click(targets["Paint"])
                    panel = viewer.gui.workspace.authoring
                    # Enable the actual checkbox through its native item rectangle.
                    checkbox = imgui.checkbox

                    def track_checkbox(label, *args):
                        result = checkbox(label, *args)
                        if label == "Paint particles":
                            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
                            targets[label] = ((lo.x + hi.x) / 2, (lo.y + hi.y) / 2)
                        return result

                    with mock.patch.object(imgui, "checkbox", track_checkbox):
                        render()
                        click(targets["Paint particles"])
                    self.assertTrue(panel.brush_enabled)
                    point = scene_pixel((0, 0, 0.4))
                    model, builds = editor.model, editor.rebuild_count
                    camera = (tuple(viewer.camera.pos), viewer.camera.pitch, viewer.camera.yaw)
                    native_mouse(point, True)
                    self.assertTrue(panel._brush_active)
                    native_mouse((point[0] + 25, point[1]))
                    samples = len(panel._brush_points)
                    render(4)
                    self.assertGreater(samples, 0)
                    self.assertEqual(len(panel._brush_points), samples)
                    self.assertIs(editor.model, model)
                    self.assertEqual(editor.model.particle_count, 0)
                    self.assertEqual(editor.rebuild_count, builds)
                    self.assertEqual((tuple(viewer.camera.pos), viewer.camera.pitch, viewer.camera.yaw), camera)
                    native_mouse((point[0] + 25, point[1]), False)
                    render(2)
                    self.assertFalse(panel._brush_active)
                    self.assertEqual(len(editor.objects), 2)
                    stroke = editor.objects[-1]
                    self.assertEqual(stroke.kind.value, "particles")
                    self.assertEqual(editor.rebuild_count, builds + 1)
                    self.assertEqual(editor.model.particle_count, len(stroke.points))
                    positions = editor.state_0.particle_q.numpy()
                    self.assertTrue(np.isfinite(positions).all())
                    self.assertGreater(float(positions[:, 2].min()), 0.4 + stroke.thickness / 2)
                    np.testing.assert_allclose(editor.model.particle_mass.numpy(), stroke.density * stroke.thickness**3)
                    editor.undo()
                    render(2)
                    self.assertEqual(len(editor.objects), 1)
                    self.assertEqual(editor.model.particle_count, 0)
                    editor.redo()
                    render(2)
                    self.assertEqual(editor.objects[-1], stroke)
                    panel._brush_radius, panel._brush_spacing, panel._brush_layers = 2.0, 0.005, 8
                    model = editor.model
                    native_mouse(point, True)
                    self.assertTrue(panel._brush_failed)
                    native_mouse(point, False)
                    render(2)
                    self.assertIs(editor.model, model)
                    self.assertEqual(len(editor.objects), 2)
                    self.assertIn("budget", panel.message)
                    self.assertIsNone(editor._edit_before)
                    panel._brush_radius, panel._brush_spacing, panel._brush_layers = 0.15, 0.04, 2
                    native_mouse(point, True)
                    self.assertTrue(panel._brush_active)
                    viewer.renderer.window.dispatch_event("on_deactivate")
                    viewer.renderer.window.dispatch_events()
                    render(2)
                    self.assertFalse(panel._brush_active)
                    self.assertIsNone(editor._edit_before)
                    self.assertIs(editor.model, model)
                    self.assertEqual(len(editor.objects), 2)


if __name__ == "__main__":
    unittest.main()
