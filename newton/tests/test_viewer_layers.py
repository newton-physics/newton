# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import Mock, patch

import warp as wp

import newton
from newton._src.viewer.viewer import ViewerBase
from newton._src.viewer.viewer_rtx import ViewerRTX
from newton._src.viewer.viewer_viser import ViewerViser
from newton.viewer import ViewerNull


class _RecordingViewer(ViewerNull):
    """Records every name passed to ``log_instances`` / ``log_mesh``.

    Used by the layer tests to verify that the active layer prefixes
    every backend object name with ``/layers/<layer_id>``.
    """

    def __init__(self):
        super().__init__(num_frames=1)
        self.instance_calls: list[tuple[str, bool]] = []
        self.instance_xforms: list[tuple[str, object]] = []
        self.mesh_calls: list[tuple[str, bool]] = []

    def log_instances(self, name, mesh, xforms, scales, colors, materials, hidden=False):
        self.instance_calls.append((name, hidden))
        if xforms is not None:
            self.instance_xforms.append((name, xforms.numpy().copy()))

    def log_mesh(
        self,
        name,
        points,
        indices,
        normals=None,
        uvs=None,
        texture=None,
        hidden=False,
        backface_culling=True,
        color=None,
        roughness=None,
        metallic=None,
    ):
        self.mesh_calls.append((name, hidden))


def _build_box_model() -> newton.Model:
    """Build a minimal single-body model with one box shape."""
    builder = newton.ModelBuilder()
    builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        label="b",
    )
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0)
    builder.add_shape(
        body=0,
        type=newton.GeoType.BOX,
        scale=wp.vec3(0.5, 0.5, 0.5),
        cfg=cfg,
    )
    return builder.finalize()


class _MinimalRTXViewer(ViewerRTX):
    """Minimal RTX instance for layer-management tests without OVRTX startup."""

    def __init__(self):
        self.gui = None
        self._render_result = None
        self._render_products = None
        self._transform_binding = None
        self._rtx = None
        self._render_width = 640
        self._render_height = 480
        self._up_axis = "Z"
        ViewerBase.__init__(self)


class TestViewerLayers(unittest.TestCase):
    def test_default_layer_uses_unprefixed_names(self):
        """Without activate(), object names remain unprefixed (legacy behavior)."""
        viewer = _RecordingViewer()
        viewer.set_model(_build_box_model())

        viewer.begin_frame(0.0)
        viewer.log_state(viewer.model.state())
        viewer.end_frame()

        names = [n for n, _ in viewer.instance_calls]
        # Shape instance names should be /model/shapes/shape_N (no layer prefix).
        self.assertTrue(any(n.startswith("/model/shapes/") for n in names))
        self.assertFalse(any(n.startswith("/layers/") for n in names))

    def test_two_layers_get_distinct_prefixes(self):
        """Two activated layers emit names under their own ``/layers/<id>/`` namespace."""
        viewer = _RecordingViewer()

        viewer.activate("solverA")
        viewer.set_model(_build_box_model())
        viewer.begin_frame(0.0)
        viewer.log_state(viewer.model.state())
        viewer.end_frame()

        viewer.activate("solverB")
        viewer.set_model(_build_box_model())
        viewer.begin_frame(0.0)
        viewer.log_state(viewer.model.state())
        viewer.end_frame()

        prefixed_a = [n for n, _ in viewer.instance_calls if n.startswith("/layers/solverA/")]
        prefixed_b = [n for n, _ in viewer.instance_calls if n.startswith("/layers/solverB/")]
        self.assertTrue(prefixed_a, "expected at least one /layers/solverA/ object")
        self.assertTrue(prefixed_b, "expected at least one /layers/solverB/ object")
        # And no cross-contamination with the default namespace.
        unprefixed = [
            n for n, _ in viewer.instance_calls if not n.startswith("/layers/") and n.startswith("/model/shapes")
        ]
        self.assertFalse(unprefixed, "unexpected unprefixed shape names alongside named layers")

    def test_activate_preserves_state_across_switches(self):
        """Switching back to a layer restores its model and shape batches."""
        viewer = _RecordingViewer()

        viewer.activate("A")
        viewer.set_model(_build_box_model())
        model_a = viewer.model
        batches_a = viewer._shape_instances

        viewer.activate("B")
        viewer.set_model(_build_box_model())
        self.assertIsNot(viewer.model, model_a)
        self.assertIsNot(viewer._shape_instances, batches_a)

        viewer.activate("A")
        self.assertIs(viewer.model, model_a)
        self.assertIs(viewer._shape_instances, batches_a)

    def test_layer_owns_custom_state_from_snapshot(self):
        """Backend layer fields included in the snapshot route through the viewer."""

        class _CustomLayerStateViewer(_RecordingViewer):
            def _init_extra_layer_state(self, layer):
                super()._init_extra_layer_state(layer)
                layer.custom_cache = {}

        viewer = _CustomLayerStateViewer()

        viewer.activate("A")
        viewer.custom_cache["value"] = "A"
        cache_a = viewer.layer.custom_cache

        viewer.activate("B")
        viewer.custom_cache["value"] = "B"
        self.assertIsNot(viewer.custom_cache, cache_a)

        viewer.activate("A")
        self.assertIs(viewer.custom_cache, cache_a)
        self.assertEqual(viewer.custom_cache["value"], "A")

    def test_layer_runtime_fields_are_snapshotted(self):
        """Layer-owned runtime fields are an explicit allowlist."""
        viewer = _RecordingViewer()

        self.assertIn("model", viewer._layer_runtime_fields)
        self.assertIn("_shape_instances", viewer._layer_runtime_fields)
        self.assertNotIn("layer_id", viewer._layer_runtime_fields)
        self.assertNotIn("visible", viewer._layer_runtime_fields)
        self.assertNotIn("xform", viewer._layer_runtime_fields)

    def test_unseeded_self_attribute_is_viewer_global(self):
        """A self-only attribute should not become accidental layer state."""
        viewer = _RecordingViewer()

        viewer.activate("A")
        viewer.future_cache = "A"

        viewer.activate("B")
        viewer.future_cache = "B"

        viewer.activate("A")
        self.assertEqual(viewer.future_cache, "B")

    def test_set_layer_visible_hides_instances(self):
        """Hiding the active layer causes log_state to emit hidden=True for shapes."""
        viewer = _RecordingViewer()
        viewer.activate("solverA")
        viewer.set_model(_build_box_model())

        viewer.begin_frame(0.0)
        viewer.log_state(viewer.model.state())
        viewer.end_frame()
        before_hidden = [hidden for name, hidden in viewer.instance_calls if "/model/shapes/" in name]
        self.assertIn(False, before_hidden, "first frame should render at least one visible shape")

        viewer.set_layer_visible("solverA", False)
        viewer.instance_calls.clear()
        viewer.begin_frame(0.01)
        viewer.log_state(viewer.model.state())
        viewer.end_frame()
        after_hidden = [hidden for name, hidden in viewer.instance_calls if "/model/shapes/" in name]
        self.assertTrue(after_hidden, "hidden layer should still emit log_instances")
        self.assertTrue(all(after_hidden), "all shape instance calls must be hidden=True")

    def test_layer_transform_offsets_logged_state_instances(self):
        """Layer transforms are applied before instance transforms reach a backend."""
        viewer = _RecordingViewer()
        viewer.activate("solverA")
        viewer.set_model(_build_box_model())
        viewer.set_layer_transform("solverA", (2.0, 3.0, 4.0))

        viewer.begin_frame(0.0)
        viewer.log_state(viewer.model.state())
        viewer.end_frame()

        shape_xforms = [xforms for name, xforms in viewer.instance_xforms if "/model/shapes/" in name]
        self.assertTrue(shape_xforms, "expected at least one shape instance transform")
        self.assertEqual(tuple(shape_xforms[0][0][:3]), (2.0, 3.0, 5.0))

    def test_remove_layer(self):
        """Removing a layer drops it and falls back to the default."""
        viewer = _RecordingViewer()
        viewer.activate("X")
        self.assertIn("X", viewer.layers)
        viewer.remove_layer("X")
        self.assertNotIn("X", viewer.layers)
        # Active layer falls back to default sentinel.
        self.assertEqual(viewer.layer.name_prefix, "")

    def test_remove_inactive_layer_preserves_active(self):
        """Removing a non-active layer keeps the previously active layer active."""
        viewer = _RecordingViewer()
        viewer.activate("X")
        viewer.activate("Y")
        viewer.remove_layer("X")
        self.assertNotIn("X", viewer.layers)
        self.assertEqual(viewer._active_layer_id, "Y")

    def test_remove_layer_clears_backend_state(self):
        """remove_layer() should run clear_model() so the layer's resources
        are released by the backend, not just dropped from the registry."""

        clear_calls: list[str] = []

        class _ClearTrackingViewer(_RecordingViewer):
            def clear_model(self):
                # Record which layer owned the call when clear_model fires.
                clear_calls.append(self._active_layer_id)
                super().clear_model()

        viewer = _ClearTrackingViewer()
        viewer.activate("X")
        viewer.set_model(_build_box_model())
        clear_calls.clear()

        viewer.remove_layer("X")

        self.assertIn("X", clear_calls, "remove_layer must run clear_model under the removed layer")

    def test_rtx_remove_layer_with_sibling_fails_loudly(self):
        """RTX should not silently wipe sibling layers during clear_model()."""
        viewer = _MinimalRTXViewer()
        viewer.activate("A")
        viewer.activate("B")

        with self.assertRaisesRegex(RuntimeError, "other user layers"):
            viewer.remove_layer("A")

        self.assertIn("A", viewer.layers)
        self.assertIn("B", viewer.layers)

    def test_activate_rejects_default_layer_id(self):
        """The internal default-layer id is reserved for legacy unprefixed output."""
        viewer = _RecordingViewer()
        with self.assertRaises(ValueError):
            viewer.activate("__default__")

    def test_cannot_remove_default_layer(self):
        viewer = _RecordingViewer()
        with self.assertRaises(ValueError):
            viewer.remove_layer("__default__")

    def test_activate_rejects_empty_id(self):
        viewer = _RecordingViewer()
        with self.assertRaises(ValueError):
            viewer.activate("")

    def test_new_layer_activation_preserves_global_picking_enabled(self):
        """Creating a new layer must not reset the viewer-wide picking toggle."""
        viewer = _RecordingViewer()
        viewer.picking_enabled = False

        viewer.activate("solverA")

        self.assertFalse(viewer.picking_enabled)

    def test_default_layer_owns_orphan_layer_namespace_paths(self):
        """Default clear ownership includes /layers paths not claimed by a live layer."""
        viewer = _RecordingViewer()

        self.assertTrue(viewer._is_layer_owned_path("/layers/orphan/probe"))

    def test_log_shapes_user_name_is_layer_qualified(self):
        """Two layers calling log_shapes with the same user-supplied name
        must end up under their respective ``/layers/<id>/`` namespace so
        they do not overwrite each other in the backend."""

        viewer = _RecordingViewer()
        viewer.set_model(_build_box_model())  # required to estimate scene scale, etc.

        xforms = wp.array([wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())], dtype=wp.transform)

        viewer.activate("A")
        viewer.log_shapes("/user/probe", int(newton.GeoType.SPHERE), 1.0, xforms)
        viewer.activate("B")
        viewer.log_shapes("/user/probe", int(newton.GeoType.SPHERE), 1.0, xforms)

        names = [n for n, _ in viewer.instance_calls]
        self.assertIn("/layers/A/user/probe", names)
        self.assertIn("/layers/B/user/probe", names)


class TestViewerLayerBackends(unittest.TestCase):
    def _make_viser_viewer(self):
        captured_calls = {}

        def add_mesh_simple(name, vertices, faces, color, wireframe, side):
            captured_calls["add_mesh_simple"] = {
                "name": name,
                "vertices": vertices,
                "faces": faces,
                "color": color,
                "wireframe": wireframe,
                "side": side,
            }
            return Mock()

        def add_batched_meshes_simple(
            name,
            vertices,
            faces,
            batched_positions,
            batched_wxyzs,
            batched_scales,
            batched_colors,
            lod,
        ):
            captured_calls["add_batched_meshes_simple"] = {
                "name": name,
                "vertices": vertices,
                "faces": faces,
                "batched_positions": batched_positions,
                "batched_wxyzs": batched_wxyzs,
                "batched_scales": batched_scales,
                "batched_colors": batched_colors,
                "lod": lod,
            }
            return Mock()

        scene = Mock()
        scene.captured_calls = captured_calls
        scene.add_mesh_simple = add_mesh_simple
        scene.add_batched_meshes_simple = add_batched_meshes_simple
        scene.add_light_ambient = Mock()
        scene.configure_environment_map = Mock()

        server = Mock()
        server.scene = scene
        server.on_client_connect = Mock()
        server.on_client_disconnect = Mock()
        server.get_scene_serializer = Mock(return_value=None)
        server.stop = Mock()

        fake_viser = Mock()
        fake_viser.ViserServer = Mock(return_value=server)

        patches = [
            patch.object(ViewerViser, "_get_viser", return_value=fake_viser),
            patch("newton._src.viewer.viewer_viser.is_jupyter_notebook", return_value=False),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

        viewer = ViewerViser(verbose=False)
        self.addCleanup(viewer.close)
        return viewer, scene

    def test_viser_log_mesh_uses_layer_namespace(self):
        viewer, scene = self._make_viser_viewer()
        viewer.activate("solverA")

        points = wp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=wp.vec3)
        indices = wp.array([0, 1, 2], dtype=wp.uint32)

        viewer.log_mesh("mesh", points, indices)

        self.assertIn("/layers/solverA/mesh", viewer._meshes)
        self.assertEqual(scene.captured_calls["add_mesh_simple"]["name"], "/layers/solverA/mesh")

    def test_viser_log_instances_uses_layer_namespace(self):
        viewer, scene = self._make_viser_viewer()
        viewer.activate("solverA")

        points = wp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=wp.vec3)
        indices = wp.array([0, 1, 2], dtype=wp.uint32)
        viewer.log_mesh("mesh", points, indices)

        xforms = wp.array([wp.transform(wp.vec3(1.0, 2.0, 3.0), wp.quat_identity())], dtype=wp.transform)
        scales = wp.array([[1.0, 1.0, 1.0]], dtype=wp.vec3)

        viewer.log_instances("instances", "mesh", xforms, scales, colors=None, materials=None)

        self.assertIn("/layers/solverA/instances", viewer._instances)
        self.assertEqual(
            scene.captured_calls["add_batched_meshes_simple"]["name"],
            "/layers/solverA/instances",
        )


if __name__ == "__main__":
    unittest.main()
