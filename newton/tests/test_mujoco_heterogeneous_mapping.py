# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify body ownership and supported configurations for ragged geom slots."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import get_test_devices


def _add_free_body(builder: newton.ModelBuilder) -> int:
    """Add one articulated body with explicit mass and inertia."""
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    joint = builder.add_joint_free(body)
    builder.add_articulation([joint])
    return body


def _basic_builder() -> newton.ModelBuilder:
    """Build two independent worlds and a shared floor with MuJoCo attributes."""
    builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(builder)
    builder.add_ground_plane()
    for _ in range(2):
        builder.begin_world()
        body = _add_free_body(builder)
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        builder.end_world()
    return builder


class TestMuJoCoHeterogeneousMapping(unittest.TestCase):
    """Check mappings for the bounded Newton-contact heterogeneity mode."""

    def test_slots_preserve_body_and_contact_properties(self):
        """Map mixed collider groups to their original world, body and contact properties."""
        for device in get_test_devices():
            with self.subTest(device=device), wp.ScopedDevice(device):
                builder = newton.ModelBuilder()
                SolverMuJoCo.register_custom_attributes(builder)
                builder.add_ground_plane()
                # Maxima occur in different worlds, and either body can have no colliders.
                for world, counts in enumerate(((1, 0), (0, 3), (2, 1))):
                    builder.begin_world()
                    for local_body, count in enumerate(counts):
                        body = _add_free_body(builder)
                        for shape in range(count):
                            custom = {
                                "mujoco:condim": 3 if shape % 2 else 4,
                                "mujoco:geom_priority": world % 2,
                            }
                            if (world + local_body) % 2:
                                builder.add_shape_sphere(body, radius=0.1, custom_attributes=custom)
                            else:
                                builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, custom_attributes=custom)
                    builder.end_world()
                model = builder.finalize(device=device)
                solver = SolverMuJoCo(model, allow_heterogeneous_shapes=True, use_mujoco_contacts=False)
                mapping = solver.mjc_geom_to_newton_shape.numpy()
                body_mapping = solver.mjc_body_to_newton.numpy()
                shape_body = model.shape_body.numpy()
                shape_world = model.shape_world.numpy()
                condim = model.mujoco.condim.numpy()
                priority = model.mujoco.geom_priority.numpy()
                self.assertEqual(set(mapping[mapping >= 0]), set(range(model.shape_count)))
                self.assertTrue(np.any(mapping < 0))
                for world, row in enumerate(mapping):
                    self.assertEqual(len(row[row >= 0]), len(set(row[row >= 0])))
                    for geom, shape in enumerate(row):
                        if shape < 0:
                            continue
                        self.assertIn(shape_world[shape], (-1, world))
                        self.assertEqual(body_mapping[world, solver.mj_model.geom_bodyid[geom]], shape_body[shape])
                        self.assertEqual(solver.mj_model.geom_condim[geom], condim[shape])
                        self.assertEqual(solver.mj_model.geom_priority[geom], priority[shape])

    def test_legacy_single_world_keeps_dynamic_shape_attachment(self):
        """Keep dynamic shapes attached to their body when legacy world indices are negative."""
        for device in get_test_devices():
            with self.subTest(device=device), wp.ScopedDevice(device):
                builder = newton.ModelBuilder()
                builder.add_ground_plane()
                body = builder.add_body()
                shape = builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
                model = builder.finalize(device=device)
                self.assertEqual(model.world_count, 1)
                self.assertEqual(model.shape_world.numpy()[shape], -1)
                solver = SolverMuJoCo(model, allow_heterogeneous_shapes=True, use_mujoco_contacts=False)
                geom = np.flatnonzero(solver.mjc_geom_to_newton_shape.numpy()[0] == shape)
                self.assertEqual(len(geom), 1)
                mj_body = solver.mj_model.geom_bodyid[geom[0]]
                self.assertGreater(mj_body, 0)
                self.assertEqual(solver.mjc_body_to_newton.numpy()[0, mj_body], body)

    def test_reject_unsupported_shape_references(self):
        """Reject sites and explicit pairs whose references need a separate mapping extension."""
        for feature in ("sites", "explicit MuJoCo contact pairs"):
            with self.subTest(feature=feature):
                builder = _basic_builder()
                if feature == "sites":
                    builder.add_site(-1)
                else:
                    builder.add_custom_values(
                        **{"mujoco:pair_world": 0, "mujoco:pair_geom1": 0, "mujoco:pair_geom2": 1}
                    )
                model = builder.finalize(device="cpu")
                with self.assertRaisesRegex(ValueError, feature):
                    SolverMuJoCo(model, allow_heterogeneous_shapes=True, use_mujoco_contacts=False)

    def test_reject_fluid_forces_from_arguments_and_world_attributes(self):
        """Reject fluid options from constructor arguments or any world's custom attributes."""
        for option in ("density", "viscosity"):
            with self.subTest(option=option):
                model = _basic_builder().finalize(device="cpu")
                with self.assertRaisesRegex(ValueError, "does not support fluid"):
                    SolverMuJoCo(model, allow_heterogeneous_shapes=True, use_mujoco_contacts=False, **{option: 1.0})
                getattr(model.mujoco, option).assign(np.array([0.0, 1.0], dtype=np.float32))
                with self.assertRaisesRegex(ValueError, "does not support fluid"):
                    SolverMuJoCo(model, allow_heterogeneous_shapes=True, use_mujoco_contacts=False)

    def test_reject_different_body_joint_layouts(self):
        """Reject equal-sized articulation trees whose joint attachments differ."""
        builder = newton.ModelBuilder()
        for world in range(2):
            builder.begin_world()
            bodies = [builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3))) for _ in range(2)]
            root, child = bodies if world == 0 else bodies[::-1]
            root_joint = builder.add_joint_revolute(-1, root)
            child_joint = builder.add_joint_revolute(root, child)
            builder.add_articulation([root_joint, child_joint])
            for body in bodies:
                builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
            builder.end_world()
        model = builder.finalize(device="cpu")
        with self.assertRaisesRegex(ValueError, "identical body/joint layouts"):
            SolverMuJoCo(model, allow_heterogeneous_shapes=True, use_mujoco_contacts=False)

    def test_reject_cpu_mujoco_backend(self):
        """Require the Warp backend that accepts Newton-generated contacts."""
        model = _basic_builder().finalize(device="cpu")
        with self.assertRaisesRegex(ValueError, "use_mujoco_cpu=False"):
            SolverMuJoCo(model, allow_heterogeneous_shapes=True, use_mujoco_contacts=False, use_mujoco_cpu=True)


if __name__ == "__main__":
    unittest.main()
