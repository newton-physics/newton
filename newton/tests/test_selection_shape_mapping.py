# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.selection import ArticulationView


class TestShapeMapping(unittest.TestCase):
    def test_interleaved_shape_attributes(self):
        """Read and scatter interleaved shapes by ownership with both mask ranks."""
        # Uniform-world interleaved shapes are covered by test_selection_frequency_validation; this
        # keeps the sparse-world case (target worlds separated by unrelated worlds).
        for device in wp.get_devices():
            for sparse in (True,):
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

    def test_appended_articulation_shapes(self):
        """Read and write shapes appended after every articulation's original shapes."""
        for device in wp.get_devices():
            with self.subTest(device=device):
                builder, expected = make_appended_world()
                model = builder.finalize(device=device)
                original = np.arange(model.shape_count, dtype=np.float32) + 0.25
                model.shape_material_mu.assign(original)
                view = ArticulationView(model, "robot_*")
                values = view.get_attribute("shape_material_mu", model)
                np.testing.assert_array_equal(values.numpy(), original[expected][None])
                values.fill_(5.0)
                mask = wp.array([[False, True, False]], dtype=bool, device=device)
                view.set_attribute("shape_material_mu", model, values, mask=mask)
                original[expected[1]] = 5.0
                np.testing.assert_array_equal(model.shape_material_mu.numpy(), original)
                single = ArticulationView(model, "robot_2")
                np.testing.assert_array_equal(
                    single.get_attribute("shape_material_mu", model).numpy(), original[expected[2]][None, None]
                )

    def test_appended_shapes_world_mask_and_gradients(self):
        """Preserve world masks, vector attributes, and gather gradients for appended shapes."""
        for device in wp.get_devices():
            with self.subTest(device=device):
                model, indices = make_appended_worlds(device, requires_grad=True)
                view = ArticulationView(model, "*")
                original = model.shape_scale.numpy().copy()
                values = view.get_attribute("shape_scale", model)
                np.testing.assert_array_equal(values.numpy(), original[indices])
                values.fill_(wp.vec3(0.03))
                view.set_attribute(
                    "shape_scale", model, values, mask=wp.array([False, True], dtype=bool, device=device)
                )
                original[4:] = 0.03
                np.testing.assert_array_equal(model.shape_scale.numpy(), original)
                with wp.Tape() as tape:
                    mu = view.get_attribute("shape_material_mu", model)
                tape.backward(grads={mu: wp.ones_like(mu)})
                np.testing.assert_array_equal(model.shape_material_mu.grad.numpy(), np.ones(model.shape_count))

    def test_indexedarray_writes_through_mapped_shapes(self):
        """Scatter indexed-array values through explicit shape row maps with both mask ranks."""
        for device in wp.get_devices():
            with self.subTest(device=device):
                builder, expected = make_appended_world()
                model = builder.finalize(device=device)
                view = ArticulationView(model, "robot_*")
                layout = view.frequency_layouts[newton.Model.AttributeFrequency.SHAPE]
                self.assertTrue(layout.uses_explicit_model_indices)
                original = np.arange(model.shape_count, dtype=np.float32) + 0.25
                model.shape_material_mu.assign(original)
                # Select row 2 of a three-row backing store so the values are a genuine indexed view.
                backing = np.zeros((3, *expected.shape), dtype=np.float32)
                backing[2] = np.arange(expected.size, dtype=np.float32).reshape(expected.shape) + 100
                values = wp.indexedarray(
                    wp.array(backing, dtype=wp.float32, device=device),
                    [wp.array([2], dtype=wp.int32, device=device), None, None],
                )
                mask = wp.array([[True, False, True]], dtype=bool, device=device)
                view.set_attribute("shape_material_mu", model, values, mask=mask)
                original[expected[[0, 2]]] = backing[2, [0, 2]]
                np.testing.assert_array_equal(model.shape_material_mu.numpy(), original)

                model, indices = make_appended_worlds(device)
                view = ArticulationView(model, "*")
                # Uniform worlds take the per-world indexed layout.
                self.assertFalse(view.frequency_layouts[newton.Model.AttributeFrequency.SHAPE].is_contiguous)
                original = model.shape_scale.numpy().copy()
                backing = np.arange(3 * indices[0].size * 3, dtype=np.float32).reshape(3, *indices.shape[1:], 3) + 1
                values = wp.indexedarray(
                    wp.array(backing, dtype=wp.vec3, device=device),
                    [wp.array([2, 0], dtype=wp.int32, device=device), None, None],
                )
                view.set_attribute(
                    "shape_scale", model, values, mask=wp.array([False, True], dtype=bool, device=device)
                )
                original[indices[1]] = backing[0]
                np.testing.assert_array_equal(model.shape_scale.numpy(), original)


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


def make_appended_world(count=3, extra_per_body=2):
    """Build articulations whose later shapes are appended after all original shapes."""
    builder = newton.ModelBuilder()
    bodies = []
    expected = []
    for index in range(count):
        body = builder.add_link(label=f"robot_{index}/body")
        builder.add_articulation([builder.add_joint_free(child=body)], label=f"robot_{index}")
        bodies.append(body)
        expected.append([builder.add_shape_sphere(body=body, radius=0.01)])
    # Mesh approximation can append pieces after all original shapes.
    for index, body in enumerate(bodies):
        for _ in range(extra_per_body):
            expected[index].append(builder.add_shape_sphere(body=body, radius=0.02))
    return builder, np.array(expected, dtype=np.int32)


def make_appended_worlds(device, requires_grad=False):
    """Replicate a two-articulation appended-shape world twice and return its expected shape rows."""
    template, expected = make_appended_world(count=2, extra_per_body=1)
    builder = newton.ModelBuilder()
    for _ in range(2):
        builder.add_world(template)
    model = builder.finalize(device=device, requires_grad=requires_grad)
    return model, np.stack([expected, expected + template.shape_count])


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
