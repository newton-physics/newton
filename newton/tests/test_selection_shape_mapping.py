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
