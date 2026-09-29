# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import gc
import unittest
import weakref

import numpy as np
import warp as wp

import newton
from newton.selection import ArticulationView


class TestShapeMapping(unittest.TestCase):
    def test_gathered_shape_lifetime(self):
        """Keep gathered values alive while releasing their view and model independently."""
        for device in wp.get_devices():
            with self.subTest(device=device):
                model = make_world().finalize(device=device)
                view = ArticulationView(model, "robot_*")
                self.assertFalse(view.shapes_contiguous)
                expected = model.shape_material_mu.numpy()[selected_shapes(model, view)]
                values = view.get_attribute("shape_material_mu", model)
                model_ref, view_ref = weakref.ref(model), weakref.ref(view)
                source_ref = weakref.ref(model.shape_material_mu)
                live_model = make_world().finalize(device=device)
                live_view = ArticulationView(live_model, "robot_*")
                live_expected = live_view.get_attribute("shape_material_mu", live_model).numpy()
                del model, view
                gc.collect()
                self.assertIsNone(model_ref())
                self.assertIsNone(view_ref())
                self.assertIsNotNone(source_ref())
                np.testing.assert_array_equal(values.numpy(), expected)
                np.testing.assert_array_equal(
                    live_view.get_attribute("shape_material_mu", live_model).numpy(), live_expected
                )
                del values
                gc.collect()
                self.assertIsNone(source_ref())

    def test_interleaved_shape_attributes(self):
        """Read and scatter interleaved shapes by ownership with both mask ranks."""
        for device in wp.get_devices():
            for sparse in (False, True):
                for selected_links in (None, ["tip"], []):
                    with self.subTest(device=device, sparse=sparse, links=selected_links):
                        scene = newton.ModelBuilder()
                        target_world = make_world()
                        distractor_world = newton.ModelBuilder()
                        distractor_world.add_shape_sphere(distractor_world.add_body(), radius=0.1)
                        for index in range(3):
                            if sparse and index:
                                for _ in range(index):
                                    scene.add_world(distractor_world)
                            scene.add_world(target_world)
                        model = scene.finalize(device=device)
                        view = ArticulationView(model, "robot_*", include_links=selected_links)
                        indices = selected_shapes(model, view)
                        for attribute in ("shape_material_mu", "shape_material_restitution", "shape_transform"):
                            source = getattr(model, attribute)
                            initial = source.numpy().copy()
                            actual = view.get_attribute(attribute, model).numpy()
                            np.testing.assert_array_equal(actual, initial[indices])
                            for mask in (
                                np.array([False, True, False]),
                                np.array([[False, True], [True, False], [False, False]]),
                            ):
                                source.assign(initial)
                                replacement = np.arange(actual.size, dtype=np.float32).reshape(actual.shape) + 20
                                view.set_attribute(attribute, model, replacement, mask=mask)
                                expected = initial.copy()
                                expected[indices[mask]] = replacement[mask]
                                np.testing.assert_array_equal(source.numpy(), expected)

    def test_global_interleaved_shapes(self):
        """Keep global articulation material writes on their owned shapes."""
        for device in wp.get_devices():
            with self.subTest(device=device):
                model = make_world().finalize(device=device)
                for name in ("robot_left", "robot_right"):
                    view = ArticulationView(model, name)
                    self.assertTrue(view.uses_explicit_model_indices)
                    self.assertFalse(view.shapes_contiguous)
                    indices = selected_shapes(model, view)
                    initial = model.shape_material_mu.numpy().copy()
                    np.testing.assert_array_equal(
                        view.get_attribute("shape_material_mu", model).numpy(), initial[indices]
                    )
                    view.set_attribute("shape_material_mu", model, np.full(indices.shape, 5.0, dtype=np.float32))
                    expected = initial.copy()
                    expected[indices.ravel()] = 5.0
                    np.testing.assert_array_equal(model.shape_material_mu.numpy(), expected)

    def test_contiguous_shapes_keep_direct_binding(self):
        """Preserve direct material bindings for contiguous articulation shapes."""
        model = make_world(interleaved=False).finalize(device="cpu")
        view = ArticulationView(model, "robot_left")
        layout = view.frequency_layouts[newton.Model.AttributeFrequency.SHAPE]
        self.assertFalse(layout.uses_explicit_model_indices)
        self.assertEqual(view.get_attribute("shape_material_mu", model).ptr, model.shape_material_mu.ptr)

    def test_variable_global_gaps(self):
        """Map matching local shape orders across different global gaps."""
        scene = newton.ModelBuilder()
        scene.add_shape_plane()
        for order in (
            [0, 2, 1, 3, 0, 2, 1, 3],
            [-1, 0, -1, 2, 1, -1, 3, 0, 2, 1, 3],
            [0, 1, 0, 1, 2, 3, 2, 3],
        ):
            scene.add_world(make_world(order=order))
        for device in wp.get_devices():
            model = scene.finalize(device=device)
            for links in (None, ["root"], ["tip"], []):
                with self.subTest(device=device, links=links):
                    view = ArticulationView(model, "robot_*", include_links=links)
                    indices = selected_shapes(model, view)
                    initial = model.shape_material_mu.numpy().copy()
                    np.testing.assert_array_equal(
                        view.get_attribute("shape_material_mu", model).numpy(), initial[indices]
                    )
                    replacement = np.arange(indices.size, dtype=np.float32).reshape(indices.shape) + 20
                    view.set_attribute("shape_material_mu", model, replacement)
                    expected = initial.copy()
                    expected[indices] = replacement
                    np.testing.assert_array_equal(model.shape_material_mu.numpy(), expected)

    def test_trailing_axis_and_gather_gradient(self):
        """Preserve trailing components and gradient ownership through gathered shapes."""
        for device in wp.get_devices():
            with self.subTest(device=device):
                model = make_world().finalize(device=device)
                view = ArticulationView(model, "robot_*", include_links=["tip"])
                indices = selected_shapes(model, view)
                initial = np.arange(model.shape_count * 3, dtype=np.float32).reshape(-1, 3)
                model.add_attribute(
                    "shape_test_vector",
                    wp.array(initial, dtype=wp.float32, device=device, requires_grad=True),
                    newton.Model.AttributeFrequency.SHAPE,
                )
                with wp.Tape() as tape:
                    gathered = view.get_attribute("shape_test_vector", model)
                np.testing.assert_array_equal(gathered.numpy(), initial[indices])
                tape.backward(grads={gathered: wp.ones_like(gathered)})
                expected_gradient = np.zeros_like(initial)
                expected_gradient[indices] = 1
                np.testing.assert_array_equal(model.shape_test_vector.grad.numpy(), expected_gradient)
                for mask in (np.array([True]), np.array([[False, True]])):
                    model.shape_test_vector.assign(initial)
                    replacement = np.arange(gathered.size, dtype=np.float32).reshape(gathered.shape) + 100
                    view.set_attribute("shape_test_vector", model, replacement, mask=mask)
                    expected = initial.copy()
                    expected[indices[mask]] = replacement[mask]
                    np.testing.assert_array_equal(model.shape_test_vector.numpy(), expected)
                left = ArticulationView(model, "robot_left")
                for attribute in ("body_mass", "joint_q", "joint_qd", "joint_type"):
                    self.assertEqual(left.get_attribute(attribute, model).ptr, getattr(model, attribute).ptr)


def make_world(interleaved=True, *, order=None):
    builder = newton.ModelBuilder()
    bodies = []
    for name in ("robot_left", "robot_right"):
        root = builder.add_link(label=f"{name}/root")
        tip = builder.add_link(label=f"{name}/tip")
        joints = [builder.add_joint_free(child=root), builder.add_joint_fixed(parent=root, child=tip)]
        builder.add_articulation(joints, label=name)
        bodies.extend((root, tip))
    if order is None:
        order = [0, 2, 1, 3, 0, 2, 1, 3] if interleaved else [0, 0, 1, 1, 2, 2, 3, 3]
    for index, body_index in enumerate(order):
        builder.add_shape_sphere(
            -1 if body_index == -1 else bodies[body_index],
            radius=0.01,
            label=f"shape_{index}",
            cfg=newton.ModelBuilder.ShapeConfig(mu=1.0 + index, restitution=0.01 * index),
        )
    return builder


def selected_shapes(model, view):
    labels = np.asarray(model.shape_label)
    shape_body = model.shape_body.numpy()
    joint_child = model.joint_child.numpy()
    starts = model.articulation_start.numpy()
    ends = model.articulation_end.numpy()
    rows = []
    for world_articulations in view.articulation_ids.numpy():
        world_rows = []
        for articulation in world_articulations:
            bodies = joint_child[starts[articulation] : ends[articulation]]
            bodies = [body for body in bodies if model.body_label[body].rsplit("/", 1)[-1] in view.body_names]
            ids = np.flatnonzero(np.isin(shape_body, bodies))
            assert labels[ids].size == view.shape_count
            world_rows.append(ids)
        rows.append(world_rows)
    return np.asarray(rows, dtype=np.int32)


if __name__ == "__main__":
    unittest.main()
