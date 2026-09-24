# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare heterogeneous body contacts with independent MuJoCo Warp worlds."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import get_test_devices


def _build_model(variants, device):
    """Build two falling objects with different per-body convex decompositions."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for counts in variants:
        template = newton.ModelBuilder()
        for object_id, count in enumerate(counts):
            body = template.add_link(
                label=f"object_{object_id}/body",
                xform=wp.transform((0.03 * object_id, 0.0, 0.65 + 0.4 * object_id), wp.quat_identity()),
                mass=1.0 + object_id,
                inertia=wp.mat33(np.diag([0.021, 0.027, 0.033]) * (1 + object_id)),
            )
            joint = template.add_joint_free(body, label=f"object_{object_id}/joint")
            template.add_articulation([joint], label=f"object_{object_id}")
            cfg = newton.ModelBuilder.ShapeConfig(density=0.0, ke=1.0e5, kd=1.0e3, mu=0.8)
            half_width = 0.24 / count
            for part in range(count):
                template.add_shape_convex_hull(
                    body,
                    mesh=newton.Mesh.create_box(half_width, 0.2, 0.15, compute_inertia=False),
                    xform=wp.transform((-0.24 + (2 * part + 1) * half_width, 0.0, 0.0), wp.quat_identity()),
                    cfg=cfg,
                    label=f"object_{object_id}/hull_{part}",
                )
        builder.add_world(template)
    return builder.finalize(device=device)


class _Simulation:
    def __init__(self, model, *, heterogeneous):
        self.model = model
        self.solver = SolverMuJoCo(
            model,
            use_mujoco_contacts=False,
            allow_heterogeneous_shapes=heterogeneous,
            iterations=50,
            ls_iterations=20,
            nconmax=128,
            njmax=512,
        )
        self.pipeline = newton.CollisionPipeline(model, rigid_contact_max=256)
        self.state = model.state()
        self.next_state = model.state()
        self.control = model.control()
        self.contacts = self.pipeline.contacts()
        newton.eval_fk(model, self.state.joint_q, self.state.joint_qd, self.state)

    def advance(self, steps):
        for _ in range(steps):
            self.state.clear_forces()
            self.pipeline.collide(self.state, self.contacts)
            self.solver.step(self.state, self.next_state, self.control, self.contacts, 1.0 / 240.0)
            self.state, self.next_state = self.next_state, self.state


class TestMuJoCoHeterogeneousBodyContacts(unittest.TestCase):
    def test_stacked_bodies_match_isolated_worlds(self):
        """Resolve contacts on the correct bodies while preserving world isolation."""
        variants = [(1, 3), (3, 1)]
        for device in get_test_devices():
            with self.subTest(device=device), wp.ScopedDevice(device):
                simulation = _Simulation(_build_model(variants, device), heterogeneous=True)
                references = [_Simulation(_build_model([variant], device), heterogeneous=False) for variant in variants]
                model = simulation.model
                solver = simulation.solver
                shape_body = model.shape_body.numpy()
                shape_world = model.shape_world.numpy()
                shape_map = solver.mjc_geom_to_newton_shape.numpy()
                geom_body = solver.mjw_model.geom_bodyid.numpy()
                body_map = solver.mjc_body_to_newton.numpy()

                self.assertEqual(set(shape_map[shape_map >= 0]), set(range(model.shape_count)))
                for world, row in enumerate(shape_map):
                    for geom, shape in enumerate(row):
                        if shape >= 0:
                            self.assertEqual(body_map[world, geom_body[geom]], shape_body[shape])

                contact_worlds = set()
                for _ in range(4):
                    simulation.advance(60)
                    for world, reference in enumerate(references):
                        reference.advance(60)
                        body_slice = slice(2 * world, 2 * world + 2)
                        np.testing.assert_allclose(
                            simulation.state.body_q.numpy()[body_slice], reference.state.body_q.numpy(), atol=2.0e-3
                        )
                        np.testing.assert_allclose(
                            simulation.state.body_qd.numpy()[body_slice], reference.state.body_qd.numpy(), atol=2.0e-2
                        )

                    contacts = simulation.contacts
                    count = int(contacts.rigid_contact_count.numpy()[0])
                    shapes0 = contacts.rigid_contact_shape0.numpy()[:count]
                    shapes1 = contacts.rigid_contact_shape1.numpy()[:count]
                    for shape0, shape1 in zip(shapes0, shapes1, strict=True):
                        world0, world1 = shape_world[shape0], shape_world[shape1]
                        self.assertTrue(world0 < 0 or world1 < 0 or world0 == world1)
                        if shape_body[shape0] >= 0 and shape_body[shape1] >= 0:
                            contact_worlds.add(int(world0))

                self.assertEqual(contact_worlds, {0, 1})
                np.testing.assert_allclose(simulation.state.body_q.numpy()[:, 2], [0.15, 0.45, 0.15, 0.45], atol=0.015)


if __name__ == "__main__":
    unittest.main()
