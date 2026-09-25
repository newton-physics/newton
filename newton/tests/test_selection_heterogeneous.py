# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.selection import ArticulationView
from newton.tests.unittest_utils import get_test_devices


def _build_model(shape_counts, device, extra_links=None):
    builder = newton.ModelBuilder()
    for world, counts in enumerate(shape_counts):
        builder.begin_world()
        for articulation, count in enumerate(counts):
            label = f"world_{world}/object_{articulation}"
            body = builder.add_link(label=f"{label}/body", mass=1.0, inertia=wp.mat33(np.eye(3)))
            joint = builder.add_joint_free(child=body, label=f"{label}/root")
            builder.add_articulation([joint], label=label)
            for shape in range(count):
                builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, label=f"{label}/shape_{shape}")
        if extra_links is not None:
            for _ in range(extra_links[world]):
                builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)), label=f"world_{world}/other")
        builder.end_world()
    return builder.finalize(device=device)


class TestSelectionHeterogeneous(unittest.TestCase):
    def test_masked_root_reset_with_varying_shape_counts(self):
        """Reset selected worlds while retaining each world's collider count."""
        for device in get_test_devices():
            with self.subTest(device=device):
                model = _build_model([[0], [1], [3], [2]], device)
                state = model.state()
                view = ArticulationView(model, "*/object_*", include_shapes=False)
                original = view.get_root_transforms(state).numpy().copy()
                poses = original.copy()
                poses[:, 0, :3] = np.arange(12, dtype=np.float32).reshape(4, 3) + 1.0
                velocities = np.arange(24, dtype=np.float32).reshape(4, 1, 6) + 1.0
                mask = np.array([True, False, True, False])

                view.set_root_transforms(state, poses, mask=mask)
                view.set_root_velocities(state, velocities, mask=mask)
                view.eval_fk(state)

                expected_poses = original.copy()
                expected_poses[mask] = poses[mask]
                expected_velocities = np.zeros_like(velocities)
                expected_velocities[mask] = velocities[mask]
                np.testing.assert_array_equal(view.get_root_transforms(state).numpy(), expected_poses)
                np.testing.assert_array_equal(view.get_link_transforms(state).numpy()[:, :, 0], expected_poses)
                np.testing.assert_array_equal(state.body_q.numpy(), expected_poses[:, 0])
                np.testing.assert_array_equal(view.get_root_velocities(state).numpy(), expected_velocities)
                np.testing.assert_array_equal(state.joint_qd.numpy().reshape(4, 1, 6), expected_velocities)
                self.assertEqual([len(model.body_shapes[i]) for i in range(4)], [0, 1, 3, 2])

    def test_multiple_articulations_with_varying_shape_strides(self):
        """Write body state through a per-articulation mask with irregular shape strides."""
        for device in get_test_devices():
            with self.subTest(device=device):
                model = _build_model([[0, 2], [3, 1], [1, 4]], device)
                state = model.state()
                view = ArticulationView(model, "*/object_*", include_shapes=False)
                original = view.get_link_transforms(state).numpy().copy()
                poses = original.copy()
                poses[..., 0] = np.arange(6, dtype=np.float32).reshape(3, 2, 1) + 1.0
                mask = np.array([[True, False], [False, True], [True, True]])
                view.set_attribute("body_q", state, poses, mask=mask)

                expected = original.copy()
                expected[mask] = poses[mask]
                np.testing.assert_array_equal(view.get_link_transforms(state).numpy(), expected)
                np.testing.assert_array_equal(state.body_q.numpy(), expected.reshape(6, 7))
                self.assertEqual(view.count_per_world, 2)

    def test_shape_access_is_explicitly_disabled(self):
        """Reject shape attribute reads and writes when shapes are excluded."""
        model = _build_model([[1], [3], [2]], "cpu")
        view = ArticulationView(model, "*/object_*", include_shapes=False)
        self.assertEqual(view.shape_count, 0)
        self.assertEqual(view.shape_names, [])
        self.assertEqual(view.shape_labels, [])
        self.assertEqual(view.link_shapes, [[]])
        with self.assertRaisesRegex(AttributeError, "include_shapes=False"):
            view.get_attribute("shape_margin", model)
        with self.assertRaisesRegex(AttributeError, "include_shapes=False"):
            view.set_attribute("shape_margin", model, np.zeros((3, 1, 0), dtype=np.float32))

    def test_default_shape_selection_is_unchanged(self):
        """Keep the default shape layout and its equal-count requirement."""
        heterogeneous = _build_model([[1], [3], [2]], "cpu")
        with self.assertRaisesRegex(ValueError, "Articulations are not identical"):
            ArticulationView(heterogeneous, "*/object_*")
        homogeneous = _build_model([[2], [2], [2]], "cpu")
        view = ArticulationView(homogeneous, "*/object_*")
        self.assertEqual(view.shape_count, 2)
        self.assertEqual(view.get_attribute("shape_margin", homogeneous).shape, (3, 1, 2))

    def test_body_and_joint_strides_are_still_validated(self):
        """Reject irregular body and joint layouts even when shapes are excluded."""
        model = _build_model([[1], [3], [2]], "cpu", extra_links=[0, 1, 0])
        with self.assertRaisesRegex(ValueError, "Non-uniform strides between worlds"):
            ArticulationView(model, "*/object_*", include_shapes=False)

    def test_body_and_joint_counts_are_still_validated(self):
        """Reject differing articulation sizes even when shapes are excluded."""
        builder = newton.ModelBuilder()
        for world in range(2):
            builder.begin_world()
            root = builder.add_link()
            joints = [builder.add_joint_free(child=root)]
            if world == 1:
                child = builder.add_link()
                joints.append(builder.add_joint_fixed(parent=root, child=child))
            builder.add_articulation(joints, label=f"world_{world}/object")
            builder.end_world()
        model = builder.finalize(device="cpu")
        with self.assertRaisesRegex(ValueError, "Articulations are not identical"):
            ArticulationView(model, "*/object", include_shapes=False)


if __name__ == "__main__":
    unittest.main()
