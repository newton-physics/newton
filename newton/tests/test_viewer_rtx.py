# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test ViewerRTX compatibility, frame capture, and runtime scene updates."""

import builtins
import importlib.metadata
import importlib.util
import os
import subprocess
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.kamino._src.utils.sim.viewer_recording import enable_recording
from newton.tests.unittest_utils import USD_AVAILABLE
from newton.viewer import ViewerRTX

OVRTX_AVAILABLE = importlib.util.find_spec("ovrtx") is not None
OVSTAGE_AVAILABLE = importlib.util.find_spec("ovstage") is not None


def _borrowed_stage_supported() -> bool:
    """Return whether the installed OVRTX and OVStage can render a borrowed stage."""
    if not (OVRTX_AVAILABLE and OVSTAGE_AVAILABLE):
        return False
    from newton._src.viewer.viewer_rtx import _version_prefix  # noqa: PLC0415

    return _version_prefix(importlib.metadata.version("ovrtx"), "OVRTX") >= (0, 4) and _version_prefix(
        importlib.metadata.version("ovstage"), "OVStage"
    ) >= (0, 2)


BORROWED_STAGE_SUPPORTED = _borrowed_stage_supported()


@unittest.skipUnless(OVRTX_AVAILABLE, "Requires ovrtx")
class TestViewerRTXVersionCompatibility(unittest.TestCase):
    def test_legacy_ovrtx_does_not_require_ovstage(self):
        """Construct ViewerRTX with legacy OVRTX without importing OVStage."""
        import ovrtx

        original_import = builtins.__import__

        def import_without_ovstage(name, *args, **kwargs):
            if name == "ovstage":
                raise AssertionError("legacy OVRTX must not import ovstage")
            return original_import(name, *args, **kwargs)

        with (
            mock.patch.object(ovrtx, "__version__", "0.3.0"),
            mock.patch("builtins.__import__", side_effect=import_without_ovstage),
        ):
            viewer = ViewerRTX(headless=True)

        viewer.close()

    def test_modern_ovrtx_requires_ovstage(self):
        """Require OVStage when constructing ViewerRTX with OVRTX 0.4 or newer."""
        import ovrtx

        original_import = builtins.__import__

        def import_without_ovstage(name, *args, **kwargs):
            if name == "ovstage":
                raise ImportError("ovstage unavailable")
            return original_import(name, *args, **kwargs)

        with (
            mock.patch.object(ovrtx, "__version__", "0.4.0"),
            mock.patch("builtins.__import__", side_effect=import_without_ovstage),
            self.assertRaisesRegex(ImportError, "OVRTX 0.4 or newer"),
        ):
            ViewerRTX(headless=True)

    @unittest.skipUnless(OVSTAGE_AVAILABLE, "Requires ovstage")
    def test_borrowed_stage_requires_ovstage_0_2(self):
        """Reject a borrowed stage on OVStage 0.1, whose GPU hierarchy computation misplaces prims."""
        import ovrtx
        import ovstage

        with (
            mock.patch.object(ovrtx, "__version__", "0.4.1"),
            mock.patch.object(ovstage, "__version__", "0.1.1.355824"),
            self.assertRaisesRegex(ValueError, "OVStage 0.2 or newer"),
        ):
            ViewerRTX(headless=True, ovstage=object())

    def _borrowed_viewer(self, labels, found, positions=None):
        """Build a borrowed-stage viewer whose stage holds the labels in ``found`` at the origin."""
        import ovrtx
        import ovstage

        builder = newton.ModelBuilder()
        for i, label in enumerate(labels):
            position = (0.0, 0.0, 0.0) if positions is None else positions[i]
            builder.add_body(xform=wp.transform(position, wp.quat_identity()), label=label)
        model = builder.finalize(device="cpu")
        with (
            mock.patch.object(ovrtx, "__version__", "0.5.0"),
            mock.patch.object(ovstage, "__version__", "0.2.0"),
        ):
            viewer = ViewerRTX(headless=True, ovstage=object())
        scale = np.diag([2.0, 2.0, 2.0, 1.0])

        def read(paths):
            return np.stack([scale if path in found else np.full((4, 4), np.nan) for path in paths])

        viewer._read_borrowed_world_matrices = read
        return viewer, model

    @unittest.skipUnless(OVSTAGE_AVAILABLE, "Requires ovstage")
    def test_borrowed_stage_binds_bodies_by_label(self):
        """Drive the stage prim at each body's label and warn about bodies without one."""
        viewer, model = self._borrowed_viewer(["/World/a", "/World/missing", "code_body"], {"/World/a"})
        try:
            with self.assertWarnsRegex(UserWarning, "2 of 3 bodies"):
                viewer.set_model(model)
            self.assertEqual(viewer._prim_paths, ("/World/a",))
            np.testing.assert_allclose(viewer._prim_linear.numpy()[0], np.diag([2.0, 2.0, 2.0]))
        finally:
            viewer.close()

    @unittest.skipUnless(OVSTAGE_AVAILABLE, "Requires ovstage")
    def test_borrowed_stage_keeps_model_frame_for_inconsistent_poses(self):
        """Warn and skip the frame correction when root bodies disagree on the stage-from-model transform."""
        viewer, model = self._borrowed_viewer(
            ["/World/a", "/World/b"], {"/World/a", "/World/b"}, positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        )
        try:
            with self.assertWarnsRegex(UserWarning, "more than one rigid transform"):
                viewer.set_model(model)
            self.assertIsNone(viewer._stage_from_model)
        finally:
            viewer.close()

    @unittest.skipUnless(OVSTAGE_AVAILABLE, "Requires ovstage")
    def test_borrowed_stage_rejects_shared_labels(self):
        """Reject bodies that would drive the same stage prim."""
        viewer, model = self._borrowed_viewer(["/World/a", "/World/a"], {"/World/a"})
        try:
            with self.assertRaisesRegex(ValueError, "share a label"):
                viewer.set_model(model)
        finally:
            viewer.close()

    @unittest.skipUnless(OVSTAGE_AVAILABLE, "Requires ovstage")
    def test_borrowed_stage_hides_simulated_cloth_by_default(self):
        """Leave cloth to the stage unless the simulated cloth is requested as an overlay."""
        viewer, model = self._borrowed_viewer(["/World/a"], {"/World/a"})
        try:
            self.assertFalse(viewer.show_triangles)
            viewer.set_model(model)
            self.assertFalse(viewer.show_triangles)
        finally:
            viewer.close()

    @unittest.skipUnless(OVSTAGE_AVAILABLE, "Requires ovstage")
    def test_borrowed_stage_rejects_layers(self):
        """Reject user layers, since a borrowed stage binds the bodies of a single model."""
        viewer, _ = self._borrowed_viewer([], set())
        try:
            with self.assertRaisesRegex(ValueError, "does not support layers"):
                viewer.activate("robot")
        finally:
            viewer.close()

    @unittest.skipUnless(OVSTAGE_AVAILABLE, "Requires ovstage")
    def test_borrowed_stage_rejects_lighting_preset(self):
        """Reject a lighting preset, since a borrowed stage brings its own lights."""
        import ovrtx
        import ovstage

        with (
            mock.patch.object(ovrtx, "__version__", "0.5.0"),
            mock.patch.object(ovstage, "__version__", "0.2.0"),
            self.assertRaisesRegex(ValueError, "lighting from the stage"),
        ):
            ViewerRTX(headless=True, ovstage=object(), environment="studio")


@unittest.skipUnless(OVRTX_AVAILABLE, "Requires ovrtx")
class TestViewerRTXWindowCleanup(unittest.TestCase):
    def test_init_failure_closes_partial_window(self):
        """Close and clear a partially initialized presentation window."""
        import ovrtx

        viewer = ViewerRTX.__new__(ViewerRTX)
        viewer.stage = mock.Mock()
        viewer._use_ovstage = False
        viewer._headless = False
        viewer._window = None
        viewer.gui = None
        viewer._instance_prim_paths = {}
        viewer._mesh_prim_paths = {}
        viewer._point_batch_paths = {}
        viewer._all_instance_paths = []
        viewer._bound_instance_prim_paths = {}
        viewer._runtime_transform_bindings = {}
        viewer._transform_binding = None
        viewer._rtx = None
        viewer._tex_resource = None
        viewer._gl_texture = None
        viewer._gl_program = None
        viewer._gl_vao = None
        partial_window = mock.Mock()

        def fail_after_window_creation():
            viewer._window = partial_window
            viewer._tex_resource = mock.Mock()
            viewer._gl_texture = 1
            viewer._gl_program = 2
            viewer._gl_vao = 3
            raise RuntimeError("window initialization failed")

        with (
            mock.patch.object(viewer, "_add_camera_lights_and_render_product"),
            mock.patch.object(viewer, "_apply_ground_material"),
            mock.patch.object(viewer, "_init_window", side_effect=fail_after_window_creation),
            mock.patch.object(viewer, "_release_runtime_scene"),
            mock.patch.object(viewer, "_destroy_ovrtx"),
            mock.patch.object(ovrtx, "RendererConfig"),
            mock.patch.object(ovrtx, "Renderer"),
            self.assertRaisesRegex(RuntimeError, "Failed to create window"),
        ):
            viewer._init_ovrtx()

        partial_window.close.assert_called_once_with()
        self.assertIsNone(viewer._window)
        self.assertIsNone(viewer._tex_resource)
        self.assertIsNone(viewer._gl_texture)
        self.assertIsNone(viewer._gl_program)
        self.assertIsNone(viewer._gl_vao)


@unittest.skipUnless(OVSTAGE_AVAILABLE, "Requires ovstage")
class TestViewerRTXOvstage(unittest.TestCase):
    def setUp(self):
        """Create an ovstage-backed ViewerRTX runtime stub."""
        import ovstage

        self.ovstage = ovstage
        self.viewer = ViewerRTX.__new__(ViewerRTX)
        self.viewer._rtx = None
        self.viewer._use_ovstage = True
        self.viewer._ovstage = self.ovstage.Stage("newton.test.ViewerRTX")
        self.viewer._ovstage_attached = False
        self.viewer._ovstage_paths = self.ovstage.PathDictionary(self.viewer._ovstage)
        self.viewer._ovstage_queries = {}
        self.viewer._ovstage_ordinal = 1
        self.viewer._ovstage_population_dirty = False
        self.viewer._runtime_scene_changed = False
        self.viewer._rendering_paused = False
        self.viewer._render_result = None
        self.viewer._deferred_prims = set()
        self.ovstage.population.open_usd_from_string(
            self.viewer._ovstage,
            """#usda 1.0
def Xform "World"
{
    def Xform "A"
    {
    }
    def Xform "B"
    {
    }
}
""",
            ordinal=self.viewer._ovstage_ordinal,
            time_code=0.0,
        )
        self.viewer._ovstage.advance_write_floor(self.viewer._ovstage_ordinal, self.ovstage.Scope.ALL).wait()

    def tearDown(self):
        """Release the ovstage runtime stub."""
        self.viewer._release_ovstage()

    def test_write_visibility_uses_reusable_query_without_deprecation(self):
        """Write token attributes through one reusable ordered query."""
        prim_paths = ["/World/A", "/World/B"]
        self.viewer._ovstage_ordinal = 2

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.viewer._write_runtime_attribute(
                prim_paths,
                "visibility",
                ["inherited", "invisible"],
            )

        self.assertFalse([warning for warning in caught if warning.category is DeprecationWarning])
        query = self.viewer._get_ovstage_query(prim_paths)
        self.assertIs(query, self.viewer._get_ovstage_query(prim_paths))
        self.assertEqual(len(self.viewer._ovstage_queries), 1)

        self.viewer._ovstage.advance_write_floor(self.viewer._ovstage_ordinal, self.ovstage.Scope.ALL).wait()
        attribute = self.viewer._ovstage_paths.intern_token("visibility")
        with self.viewer._ovstage.read_attributes(
            query,
            [attribute],
            self.ovstage.OrdinalRange.latest(self.viewer._ovstage_ordinal),
        ) as read:
            group = read.fetch_next()
            data_rows = np.from_dlpack(group.dlpack(0)).copy()
            values = [data_rows[group.data_row_index(index)] for index in range(group.data_count)]
            self.viewer._ovstage.release_group(group)

        self.assertEqual(
            [self.viewer._ovstage_paths.token_to_string(int(value)) for value in values],
            ["inherited", "invisible"],
        )

    def test_write_array_attribute_preserves_vector_lanes(self):
        """Write a runtime vector array with its element layout intact."""
        points = np.asarray(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=np.float32,
        )
        self.viewer._ovstage_ordinal = 2
        self.viewer._write_runtime_array_attribute("/World/A", "points", points)
        self.viewer._ovstage.advance_write_floor(self.viewer._ovstage_ordinal, self.ovstage.Scope.ALL).wait()

        query = self.viewer._get_ovstage_query(["/World/A"])
        attribute = self.viewer._ovstage_paths.intern_token("points")
        with self.viewer._ovstage.read_attributes(
            query,
            [attribute],
            self.ovstage.OrdinalRange.latest(self.viewer._ovstage_ordinal),
        ) as read:
            group = read.fetch_next()
            self.assertEqual(group.tensor(0).dtype.lanes, 3)
            values = np.from_dlpack(group.dlpack(0)).copy()
            self.viewer._ovstage.release_group(group)

        np.testing.assert_allclose(values, points)

    def test_runtime_prim_population_uses_ovstage(self):
        """Add and replace runtime USD through OVStage without legacy calls."""
        from pxr import Usd, UsdGeom

        self.viewer._rtx = mock.Mock()
        self.viewer._rtx.add_usd_reference_from_string.side_effect = DeprecationWarning
        self.viewer._rtx.remove_usd.side_effect = DeprecationWarning
        self.viewer.stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(self.viewer.stage, "/World/RuntimeMarker")
        self.viewer._frame_index = 0
        self.viewer._runtime_prim_handles = {}
        self.viewer._runtime_prim_paths = {}
        self.viewer._runtime_prim_serial = 0

        runtime_path = self.viewer._replace_runtime_prim("/World/RuntimeMarker")
        self.assertTrue(self.viewer._ovstage_population_dirty)
        self.assertEqual(runtime_path, "/World/RuntimeMarker_rtx_1")
        self.assertIsInstance(self.viewer._runtime_prim_handles["/World/RuntimeMarker"], int)
        self.viewer._ovstage_ordinal = 2
        self.viewer._apply_ovstage_population_changes()
        self.assertFalse(self.viewer._ovstage_population_dirty)

        replacement_path = self.viewer._replace_runtime_prim("/World/RuntimeMarker")
        self.assertTrue(self.viewer._ovstage_population_dirty)
        self.assertEqual(replacement_path, "/World/RuntimeMarker_rtx_2")
        self.viewer._ovstage_ordinal = 3
        self.viewer._apply_ovstage_population_changes()
        self.assertFalse(self.viewer._ovstage_population_dirty)
        self.viewer._rtx.add_usd_reference_from_string.assert_not_called()
        self.viewer._rtx.remove_usd.assert_not_called()

    def test_replacing_runtime_prim_releases_its_queries(self):
        """Drop cached queries of a replaced runtime prim so repeated replacements do not accumulate them."""
        from pxr import Usd, UsdGeom

        self.viewer._rtx = mock.Mock()
        self.viewer.stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(self.viewer.stage, "/World/Lines")
        self.viewer._frame_index = 0
        self.viewer._runtime_prim_handles = {}
        self.viewer._runtime_prim_paths = {}
        self.viewer._runtime_prim_serial = 0
        self.viewer._pending_hidden_prim_paths = set()

        for _ in range(3):
            runtime_path = self.viewer._replace_runtime_prim("/World/Lines")
            self.viewer._get_ovstage_query([runtime_path])
            self.viewer._get_ovstage_query([f"{runtime_path}/instance_0", f"{runtime_path}/instance_1"])
        self.viewer._get_ovstage_query(["/World/A"])

        self.assertEqual(
            set(self.viewer._ovstage_queries),
            {
                (runtime_path,),
                (f"{runtime_path}/instance_0", f"{runtime_path}/instance_1"),
                ("/World/A",),
            },
        )

    def test_end_frame_waits_for_async_render_before_stage_writes(self):
        """Finish the previous async stage read before publishing the next frame."""
        events = []
        self.viewer._phase = self.viewer._PHASE_RENDER
        self.viewer._should_close = False
        self.viewer.gui = None
        self.viewer._rtx = mock.Mock()
        self.viewer._discard_render_result = False
        pending = mock.Mock()
        self.viewer._render_result = pending
        pending.wait.side_effect = lambda: events.append("wait") or pending

        with (
            mock.patch.object(self.viewer, "_accept_render"),
            mock.patch.object(self.viewer, "_update_scene", side_effect=lambda: events.append("write")),
            mock.patch.object(self.viewer, "_render_and_display"),
        ):
            self.viewer.end_frame()

        self.assertEqual(events, ["wait", "write"])


def _column_matrix(xform: wp.transform) -> np.ndarray:
    """Return the 4x4 column-vector matrix of a transform."""
    x, y, z, w = (float(v) for v in xform[3:])
    out = np.eye(4)
    out[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    out[:3, 3] = [float(v) for v in xform[:3]]
    return out


class TestViewerRTXPrimWorldMatrices(unittest.TestCase):
    def test_matrices_compose_local_body_and_world_placement(self):
        """Write ``(layer · offset · body · local · scale)ᵀ`` for body-attached, static, and unplaced prims."""
        from newton._src.viewer.viewer_rtx import write_prim_world_matrices  # noqa: PLC0415

        rng = np.random.default_rng(0)

        def random_xform():
            quat = rng.normal(size=4)
            return wp.transform(wp.vec3(*rng.normal(size=3)), wp.quat(*(quat / np.linalg.norm(quat))))

        body_q = [random_xform(), random_xform()]
        layer = random_xform()
        offsets = rng.normal(size=(2, 3))
        # (body, world); worlds -1 and 5 have no offset.
        rows = [(0, 0), (-1, -1), (1, 1), (1, 5)]
        local = [random_xform() for _ in rows]
        scales = rng.uniform(0.5, 2.0, size=(len(rows), 3))

        linear = np.stack([_column_matrix(xf)[:3, :3] * scale for xf, scale in zip(local, scales, strict=True)])
        translation = np.stack([_column_matrix(xf)[:3, 3] for xf in local])
        out = wp.empty(len(rows), dtype=wp.mat44d)
        wp.launch(
            write_prim_world_matrices,
            dim=len(rows),
            inputs=[
                wp.array(body_q, dtype=wp.transform),
                wp.array([body for body, _ in rows], dtype=int),
                wp.array(linear, dtype=wp.mat33),
                wp.array(translation, dtype=wp.vec3),
                wp.array([world for _, world in rows], dtype=int),
                wp.array(offsets, dtype=wp.vec3),
                layer,
                0,
            ],
            outputs=[out],
        )

        for row, ((body, world), xf, scale) in enumerate(zip(rows, local, scales, strict=True)):
            expected = _column_matrix(xf) @ np.diag([*scale, 1.0])
            if body >= 0:
                expected = _column_matrix(body_q[body]) @ expected
            if 0 <= world < len(offsets):
                expected[:3, 3] += offsets[world]
            expected = _column_matrix(layer) @ expected
            np.testing.assert_allclose(out.numpy()[row], expected.T, atol=1.0e-5)


@unittest.skipUnless(OVRTX_AVAILABLE, "Requires ovrtx")
class TestViewerRTXRenderSettings(unittest.TestCase):
    def test_render_settings_override_render_product_attributes(self):
        """Author typed render settings on the render product, overriding the viewer's defaults."""
        from pxr import Sdf

        viewer = ViewerRTX(
            headless=True,
            render_settings={
                "omni:rtx:pt:samplesPerPixel": ("UInt", 4),
                "omni:rtx:quality": ("Int", 100),
                "omni:rtx:post:tonemap:cm2Factor": ("Float", 1.5),
            },
        )
        try:
            viewer._add_camera_lights_and_render_product()
            product = viewer.stage.GetPrimAtPath(viewer._render_product_path)
            spp = product.GetAttribute("omni:rtx:pt:samplesPerPixel")
            self.assertEqual((spp.GetTypeName(), spp.Get()), (Sdf.ValueTypeNames.UInt, 4))
            self.assertEqual(product.GetAttribute("omni:rtx:quality").Get(), 100)
            factor = product.GetAttribute("omni:rtx:post:tonemap:cm2Factor")
            self.assertEqual((factor.GetTypeName(), factor.Get()), (Sdf.ValueTypeNames.Float, 1.5))
        finally:
            viewer.close()

    def test_render_settings_accept_usd_type_names(self):
        """Accept USD type names as well as ``Sdf.ValueTypeNames`` attributes, and reject unknown types up front."""
        from pxr import Sdf

        viewer = ViewerRTX(
            headless=True,
            render_settings={
                "omni:rtx:pt:samplesPerPixel": ("uint", 4),
                "omni:rtx:post:tonemap:op": ("token", "aces"),
            },
        )
        try:
            viewer._add_camera_lights_and_render_product()
            product = viewer.stage.GetPrimAtPath(viewer._render_product_path)
            self.assertEqual(product.GetAttribute("omni:rtx:pt:samplesPerPixel").GetTypeName(), Sdf.ValueTypeNames.UInt)
            self.assertEqual(product.GetAttribute("omni:rtx:post:tonemap:op").GetTypeName(), Sdf.ValueTypeNames.Token)
        finally:
            viewer.close()

        for type_name in ("bogus", "Find"):
            with self.subTest(type_name=type_name), self.assertRaisesRegex(ValueError, "samplesPerPixel"):
                ViewerRTX(headless=True, render_settings={"omni:rtx:pt:samplesPerPixel": (type_name, 4)})


class TestViewerRTXRenderOutput(unittest.TestCase):
    def test_ldr_color_lookup_accepts_legacy_and_ovrtx_05_names(self):
        """Find the color output returned by legacy and OVRTX 0.5 renderers."""
        for name in ("LdrColor", "/Render/Vars/LdrColor"):
            with self.subTest(name=name):
                render_var = object()
                frame = mock.Mock(render_vars={name: render_var})
                viewer = mock.Mock(_render_var_path="/Render/Vars/LdrColor")
                self.assertIs(ViewerRTX._get_ldr_color_render_var(viewer, frame), render_var)

    @unittest.skipUnless(OVRTX_AVAILABLE, "Requires ovrtx")
    def test_display_uses_ovrtx_05_color_output(self):
        """Blit the fully qualified OVRTX 0.5 color output to the window."""
        viewer = ViewerRTX.__new__(ViewerRTX)
        viewer._headless = False
        viewer._window = mock.Mock(context=object())

        render_var = mock.MagicMock()
        mapping = render_var.map.return_value.__enter__.return_value
        pixels = mock.Mock()
        pixels.device.stream.cuda_stream = 17
        frame = mock.Mock(render_vars={"/Render/Vars/LdrColor": render_var})
        products = {"product": mock.Mock(frames=[frame])}

        with (
            mock.patch.object(wp, "from_dlpack", return_value=pixels),
            mock.patch.object(viewer, "_blit_to_window") as blit,
        ):
            viewer._accept_render(products)

        blit.assert_called_once_with(pixels)
        mapping.unmap.assert_called_once_with(stream=17)

    @unittest.skipUnless(OVRTX_AVAILABLE, "Requires ovrtx")
    def test_screenshot_uses_ovrtx_05_color_output(self):
        """Capture the fully qualified OVRTX 0.5 color output."""
        viewer = ViewerRTX.__new__(ViewerRTX)
        viewer._rendering_paused = False
        expected = np.zeros((2, 3, 4), dtype=np.uint8)
        render_var = mock.MagicMock()
        render_var.map.return_value.__enter__.return_value = expected
        frame = mock.Mock(render_vars={"/Render/Vars/LdrColor": render_var})
        viewer._render_products = {"product": mock.Mock(frames=[frame])}
        viewer._render_result = None

        np.testing.assert_array_equal(viewer._capture_screenshot_pixels(), expected)


@unittest.skipUnless(OVRTX_AVAILABLE and wp.is_cuda_available(), "Requires OVRTX and CUDA")
class TestViewerRTXRendering(unittest.TestCase):
    """Keep real OVRTX renders in one class so class-level parallelism runs them serially."""

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_moving_scene_capture_stays_frozen_until_resume(self):
        """Freeze both modes and resume with the latest scene rather than a stale result."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_box(body, hx=0.3, hy=0.3, hz=0.3, color=(1.0, 0.1, 0.0))
        model = builder.finalize()
        for asynchronous in (False, True):
            with self.subTest(async_rendering=asynchronous):
                viewer = ViewerRTX(width=64, height=48, headless=True, async_rendering=asynchronous)
                try:
                    viewer.set_model(model)
                    viewer.set_camera(wp.vec3(3.0, -4.0, 2.0), pitch=-20.0, yaw=125.0)
                    state = model.state()
                    for _ in range(2):
                        viewer.begin_frame(0.0)
                        viewer.log_state(state)
                        viewer.end_frame()
                    viewer.set_rendering_paused(True)
                    frozen = viewer.get_frame().numpy().copy()
                    self.assertGreater(np.ptp(frozen), 0)
                    for i in range(1, 4):
                        state.body_q.assign([wp.transform(wp.vec3(float(i), 0.0, 0.0), wp.quat_identity())])
                        viewer.begin_frame(i / 60.0)
                        viewer.log_state(state)
                        viewer.log_points("/runtime_points", wp.zeros(i, dtype=wp.vec3), radii=0.1)
                        viewer.end_frame()
                        np.testing.assert_array_equal(viewer.get_frame().numpy(), frozen)
                    viewer.set_rendering_paused(False)
                    viewer.begin_frame(4.0 / 60.0)
                    viewer.log_state(state)
                    viewer.end_frame()
                    if asynchronous:
                        np.testing.assert_array_equal(viewer._displayed_pixels.numpy()[:, :, :3], frozen)
                        viewer.begin_frame(5.0 / 60.0)
                        viewer.log_state(state)
                        viewer.end_frame()
                    self.assertFalse(np.array_equal(viewer.get_frame().numpy(), frozen))
                finally:
                    viewer.close()

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_headless_frame_capture(self):
        """Capture the latest moving scene across render mode changes."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_box(body, hx=0.25, hy=0.25, hz=0.25, color=(1.0, 0.0, 0.0))
        model = builder.finalize()
        state = model.state()
        for async_rendering in (False, True):
            with self.subTest(async_rendering=async_rendering):
                viewer = ViewerRTX(width=64, height=48, headless=True, async_rendering=async_rendering)
                try:
                    viewer.set_model(model)
                    viewer.set_camera(pos=wp.vec3(2.0, 0.0, 0.0), pitch=0.0, yaw=180.0)
                    modes = (async_rendering, not async_rendering, async_rendering)
                    for frame_index, (render_async, y) in enumerate(zip(modes, (-0.5, 0.5, -0.5), strict=True)):
                        # Exercise the same mode flag exposed by the viewer UI.
                        viewer._async = render_async
                        state.body_q.assign([wp.transform((0.0, y, 0.0), wp.quat_identity())])
                        viewer.begin_frame(frame_index / 60)
                        viewer.log_state(state)
                        viewer.end_frame()
                        frame = viewer.get_frame()
                        self.assertEqual(frame.shape, (48, 64, 3))
                        self.assertEqual(frame.dtype, wp.uint8)
                        self.assertEqual(frame.device, model.device)
                        rgb = frame.numpy()
                        red_pixels = (rgb[:, :, 0] > 32) & (rgb[:, :, 1] < rgb[:, :, 0] // 2)
                        _, columns = np.nonzero(red_pixels)
                        self.assertGreater(columns.size, 0)
                        # The box must appear on the side logged in this frame.
                        self.assertGreater(y * (columns.mean() - 32), 0)

                    target = wp.empty_like(frame)
                    self.assertIs(viewer.get_frame(target_image=target), target)
                    np.testing.assert_array_equal(target.numpy(), rgb)

                finally:
                    viewer.close()

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    @unittest.skipUnless(importlib.util.find_spec("imageio_ffmpeg") is not None, "Requires imageio-ffmpeg")
    def test_headless_video_recording(self):
        """Encode and decode a real RTX recording without OpenGL renderer attributes."""
        import imageio_ffmpeg as ffmpeg  # noqa: PLC0415
        from PIL import Image

        viewer = ViewerRTX(width=64, height=48, headless=True)
        try:
            self.assertTrue(enable_recording(viewer))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "recording.mp4"
                viewer.start_clip(str(path), max_frames=2, video_folder=str(Path(directory) / "frames"))
                for frame_index in range(2):
                    viewer.begin_frame(frame_index / 60)
                    self.assertTrue(viewer.should_step())
                    viewer.end_frame()

                self.assertTrue(path.is_file())
                subprocess.run(
                    [
                        ffmpeg.get_ffmpeg_exe(),
                        "-v",
                        "error",
                        "-i",
                        str(path),
                        str(Path(directory) / "decoded-%02d.png"),
                    ],
                    check=True,
                    capture_output=True,
                )
                decoded = sorted(Path(directory).glob("decoded-*.png"))
                self.assertEqual(len(decoded), 2)
                for frame_path in decoded:
                    with Image.open(frame_path) as frame:
                        self.assertEqual(frame.size, (64, 48))
        finally:
            viewer.close()

    @unittest.skipUnless(OVSTAGE_AVAILABLE, "Requires ovstage")
    def test_runtime_line_batch_has_no_deprecation_warnings(self):
        """Render a line batch first created after the runtime scene is active."""
        viewer = ViewerRTX(headless=True, async_rendering=False)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", DeprecationWarning)
                viewer.begin_frame(0.0)
                viewer.end_frame()
                viewer.begin_frame(1.0 / 60.0)
                viewer.log_lines(
                    "/runtime_line",
                    wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3),
                    wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3),
                    wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3),
                )
                viewer.end_frame()
        finally:
            viewer.close()

    _BORROWED_USDA = """#usda 1.0
(
    upAxis = "{up_axis}"
)
def Xform "World"
{{
    double3 xformOp:translate = (0, 0, 0.5)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Xform "Body" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {{
        double3 xformOp:translate = (1, 2, 3)
        float3 xformOp:rotateXYZ = (10, 20, 30)
        float3 xformOp:scale = (2, 2, 2)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
        def Cube "geom" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {{
            double size = 0.2
        }}
    }}
}}
"""

    def _open_borrowed_stage(self, usda: str):
        """Write ``usda`` to a file and populate a borrowed stage from it; return the stage and file path."""
        import ovrtx
        import ovstage

        ovrtx.register_schema_paths()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "scene.usda")
        with open(path, "w") as f:
            f.write(usda)
        stage = ovstage.Stage(
            "newton.test.borrowed",
            config=ovstage.StageConfig(
                runtime_default_hierarchy_computation_model=ovstage.HierarchyComputationModel.GPU_INCREMENTAL
            ),
        )
        self.addCleanup(stage.destroy)
        ovstage.population.open_usd(stage, path, ordinal=1)
        stage.advance_write_floor(1).wait()
        return stage, path

    def _borrowed_scene(self, up_axis="Z", **add_usd_kwargs):
        """Populate a borrowed stage and import the same scene into a model."""
        stage, path = self._open_borrowed_stage(self._BORROWED_USDA.format(up_axis=up_axis))
        builder = newton.ModelBuilder()
        builder.add_usd(path, **add_usd_kwargs)
        return stage, builder.finalize()

    @unittest.skipUnless(BORROWED_STAGE_SUPPORTED, "Requires OVRTX 0.4+ and OVStage 0.2+")
    def test_borrowed_stage_writes_above_caller_advanced_floor(self):
        """Keep writing to a borrowed stage after its owner advances the write floor."""
        import ovstage

        stage, model = self._borrowed_scene()
        state = model.state()
        viewer = ViewerRTX(headless=True, async_rendering=False, ovstage=stage)
        try:
            viewer.set_model(model)
            for frame in range(2):
                if frame:
                    stage.advance_write_floor(viewer._ovstage_ordinal + 5, ovstage.Scope.ALL).wait()
                viewer.begin_frame(frame / 60.0)
                viewer.log_state(state)
                viewer.end_frame()
        finally:
            viewer.close()

    @unittest.skipUnless(BORROWED_STAGE_SUPPORTED, "Requires OVRTX 0.4+ and OVStage 0.2+")
    def test_borrowed_stage_keeps_authored_poses_until_first_state(self):
        """Render bound bodies at their authored poses on frames before the first logged state."""
        stage, model = self._borrowed_scene()
        viewer = ViewerRTX(headless=True, async_rendering=False, ovstage=stage)
        try:
            authored = viewer._read_borrowed_world_matrices(["/World/Body"])
            viewer.set_model(model)
            viewer.begin_frame(0.0)
            viewer.end_frame()
            np.testing.assert_allclose(viewer._read_borrowed_world_matrices(["/World/Body"]), authored, atol=1.0e-5)
        finally:
            viewer.close()

    @unittest.skipUnless(BORROWED_STAGE_SUPPORTED, "Requires OVRTX 0.4+ and OVStage 0.2+")
    def test_borrowed_stage_renders_reoriented_import_in_stage_frame(self):
        """Keep bodies at their stage poses when the import rotated a Y-up stage and applied an ``xform``."""
        xform = wp.transform((5.0, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.7))
        stage, model = self._borrowed_scene(up_axis="Y", xform=xform)
        viewer = ViewerRTX(headless=True, async_rendering=False, ovstage=stage)
        try:
            paths = ["/World/Body", viewer._camera_prim_path]
            authored = viewer._read_borrowed_world_matrices(paths)[0]
            viewer.set_model(model)
            viewer.begin_frame(0.0)
            viewer.log_state(model.state())
            viewer.end_frame()
            body, camera = viewer._read_borrowed_world_matrices(paths)
            np.testing.assert_allclose(body, authored, atol=1.0e-5)
            # The camera follows the model frame, so its view of the bodies is unchanged.
            rigid = authored.copy()
            rigid[:3, :3] /= np.linalg.norm(rigid[:3, :3], axis=1)[:, None]
            stage_from_model = np.linalg.solve(_column_matrix(model.body_q.numpy()[0]).T, rigid)
            np.testing.assert_allclose(camera, viewer._compute_camera_matrix() @ stage_from_model, atol=1.0e-5)
        finally:
            viewer.close()

    @unittest.skipUnless(BORROWED_STAGE_SUPPORTED, "Requires OVRTX 0.4+ and OVStage 0.2+")
    def test_borrowed_stage_binds_replicated_clones(self):
        """Bind each replicated world's bodies to the prims the stage cloned into its environment."""
        envs = [f"/World/envs/env_{i}" for i in range(3)]
        env_xforms = "".join(
            f"""
        def Xform "env_{i}"
        {{
            double3 xformOp:translate = ({3.0 * i}, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}"""
            for i in range(1, len(envs))
        )
        stage, path = self._open_borrowed_stage(
            f"""#usda 1.0
(
    upAxis = "Z"
)
def Xform "World"
{{
    def Xform "envs"
    {{
        def Xform "env_0"
        {{
            def Xform "Body" (
                prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
            )
            {{
                double3 xformOp:translate = (0, 0, 1)
                uniform token[] xformOpOrder = ["xformOp:translate"]
                def Cube "geom" (
                    prepend apiSchemas = ["PhysicsCollisionAPI"]
                )
                {{
                    double size = 0.2
                }}
            }}
        }}{env_xforms}
    }}
}}
"""
        )
        stage.clone(f"{envs[0]}/Body", [f"{env}/Body" for env in envs[1:]], ordinal=2)
        stage.advance_write_floor(2).wait()

        prototype = newton.ModelBuilder()
        prototype.add_usd(path, root_path=envs[0])
        prototype.body_label[:] = [label.removeprefix(f"{envs[0]}/") for label in prototype.body_label]
        builder = newton.ModelBuilder()
        builder.replicate(
            prototype,
            len(envs),
            xforms=[wp.transform((3.0 * i, 0.0, 0.0), wp.quat_identity()) for i in range(len(envs))],
            label_prefixes=envs,
        )
        model = builder.finalize()
        state = model.state()
        body_q = state.body_q.numpy()
        body_q[:, 2] += 0.5
        state.body_q.assign(body_q)

        viewer = ViewerRTX(headless=True, async_rendering=False, ovstage=stage)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                viewer.set_model(model)
            viewer.begin_frame(0.0)
            viewer.log_state(state)
            viewer.end_frame()
            world = viewer._read_borrowed_world_matrices([f"{env}/Body" for env in envs])
            np.testing.assert_allclose(world[:, 3, :3], [[3.0 * i, 0.0, 1.5] for i in range(len(envs))], atol=1.0e-5)
        finally:
            viewer.close()

    @unittest.skipUnless(OVSTAGE_AVAILABLE, "Requires ovstage")
    def test_resizing_line_batch_after_first_frame(self):
        """Resize a line batch created before the first frame once rendering has started."""
        viewer = ViewerRTX(headless=True, async_rendering=False)
        try:
            for frame, count in enumerate((2, 2, 5)):
                viewer.begin_frame(frame / 60.0)
                viewer.log_lines(
                    "/resized_lines",
                    wp.array([wp.vec3(float(i), 0.0, 0.0) for i in range(count)], dtype=wp.vec3),
                    wp.array([wp.vec3(float(i), 0.0, 1.0) for i in range(count)], dtype=wp.vec3),
                    (0.0, 1.0, 0.0),
                )
                viewer.end_frame()
        finally:
            viewer.close()


class TestViewerRTXMeshUpdates(unittest.TestCase):
    def _make_runtime_viewer(self):
        """Capture mesh attribute writes at the OVRTX boundary."""
        viewer = ViewerRTX.__new__(ViewerRTX)
        viewer._phase = viewer._PHASE_RENDER
        viewer._qualify = mock.Mock(side_effect=lambda name: name)
        viewer._mesh_prim_paths = {"/mesh": "/root/mesh"}
        viewer._pending_mesh_points = {}
        viewer._pending_mesh_normals = {}
        viewer._pending_mesh_topology = {}
        viewer._pending_mesh_visibility = {}
        viewer._rtx = mock.Mock()
        viewer._use_ovstage = False
        # Avoid the optional OVRTX DLPack adapter, but retain the arrays sent to it.
        viewer._make_point3f_dltensor = mock.Mock(side_effect=np.copy)
        attributes = {}

        def write_array_attribute(prim_paths, attribute_name, tensors):
            self.assertEqual(prim_paths, ["/root/mesh"])
            self.assertEqual(len(tensors), 1)
            attributes[attribute_name] = np.array(tensors[0], copy=True)

        viewer._rtx.write_array_attribute.side_effect = write_array_attribute
        return viewer, attributes

    def test_dynamic_mesh_generates_runtime_normals_after_topology_change(self):
        """Send fresh smooth or sharp normals to RTX after replacing mesh topology."""
        for index_dtype in (wp.int32, wp.uint32):
            with self.subTest(index_dtype=index_dtype):
                viewer, attributes = self._make_runtime_viewer()
                points = wp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=wp.vec3, device="cpu")
                indices = wp.array([0, 1, 2], dtype=index_dtype, device="cpu")
                normals = wp.array([[1, 0, 0]] * 3, dtype=wp.vec3, device="cpu")
                viewer.log_mesh("/mesh", points, indices, normals=normals, dynamic=True)
                viewer._update_ovrtx_mesh_points()
                np.testing.assert_array_equal(attributes["normals"], normals.numpy())

                folded_points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
                folded_indices = np.array([0, 1, 2, 0, 3, 1], dtype=np.int32)
                for split in (False, True):
                    with self.subTest(split=split):
                        if split:
                            vertices = folded_points[folded_indices]
                            triangles = np.arange(6, dtype=np.int32)
                            expected_normals = [[0, 0, 1]] * 3 + [[0, 1, 0]] * 3
                        else:
                            vertices, triangles = folded_points, folded_indices
                            expected_normals = [[0, 2**-0.5, 2**-0.5]] * 2 + [[0, 0, 1], [0, 1, 0]]
                        viewer.log_mesh(
                            "/mesh",
                            wp.array(vertices, dtype=wp.vec3, device="cpu"),
                            wp.array(triangles, dtype=index_dtype, device="cpu"),
                            dynamic=True,
                        )
                        viewer._update_ovrtx_mesh_points()
                        np.testing.assert_allclose(attributes["normals"], expected_normals, atol=1e-6)
                        np.testing.assert_array_equal(attributes["points"], vertices)
                        np.testing.assert_array_equal(attributes["faceVertexIndices"], triangles)
                        np.testing.assert_array_equal(attributes["faceVertexCounts"], [3, 3])

                viewer.log_mesh(
                    "/mesh",
                    wp.empty(0, dtype=wp.vec3, device="cpu"),
                    wp.empty(0, dtype=index_dtype, device="cpu"),
                    dynamic=True,
                )
                viewer._update_ovrtx_mesh_points()
                self.assertEqual(attributes["normals"].shape, (0, 3))
                self.assertEqual(attributes["points"].shape, (0, 3))
                self.assertEqual(attributes["faceVertexIndices"].size, 0)

    def test_deforming_mesh_recomputes_runtime_normals(self):
        """Refresh omitted normals when points change without a topology update."""
        viewer, attributes = self._make_runtime_viewer()
        indices = wp.array([0, 1, 2], dtype=wp.int32, device="cpu")
        for vertices, normal in (
            ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [0, 0, 1]),
            ([[0, 0, 0], [0, 0, 1], [1, 0, 0]], [0, 1, 0]),
        ):
            with self.subTest(normal=normal):
                viewer.log_mesh("/mesh", wp.array(vertices, dtype=wp.vec3, device="cpu"), indices)
                viewer._update_ovrtx_mesh_points()
                self.assertIn("normals", attributes)
                np.testing.assert_allclose(attributes["normals"], [normal] * 3, atol=1e-6)
                self.assertNotIn("faceVertexIndices", attributes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
