# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify physical invariants needed by persistent FeatherPGS contacts."""

import unittest

import warp as wp

import newton
from newton.solvers import SolverFeatherPGS


class TestFeatherPGSCapacityStatus(unittest.TestCase):
    def test_capacity_failure_is_observable_without_watermarks(self):
        """Expose dropped contacts even when optional detailed telemetry is off."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device="cpu")
        solver = SolverFeatherPGS(model, mf_max_constraints=1)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state_in, state_out = model.state(), model.state()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, model.control(), contacts, 1.0 / 240.0)
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            solver.check_constraint_capacity()
        self.assertTrue(solver.constraint_overflow.numpy()[0])
        solver.reset(state_out)
        solver.check_constraint_capacity()


if __name__ == "__main__":
    unittest.main()
