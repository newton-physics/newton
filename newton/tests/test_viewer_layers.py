# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import warp as wp

import newton
from newton.viewer import ViewerNull


class _RecordingViewer(ViewerNull):
    """Records every name passed to ``log_instances`` / ``log_mesh``.

    Used by the layer tests to verify that the active layer prefixes
    every backend object name with ``/layers/<layer_id>``.
    """

    def __init__(self):
        super().__init__(num_frames=1)
        self.instance_calls: list[tuple[str, bool]] = []
        self.mesh_calls: list[tuple[str, bool]] = []

    def log_instances(self, name, mesh, xforms, scales, colors, materials, hidden=False):
        self.instance_calls.append((name, hidden))

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

    def test_cannot_remove_default_layer(self):
        viewer = _RecordingViewer()
        with self.assertRaises(ValueError):
            viewer.remove_layer("__default__")

    def test_activate_rejects_empty_id(self):
        viewer = _RecordingViewer()
        with self.assertRaises(ValueError):
            viewer.activate("")

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


if __name__ == "__main__":
    unittest.main()
