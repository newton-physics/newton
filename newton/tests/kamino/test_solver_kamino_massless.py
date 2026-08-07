# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for SolverKamino's handling of bodies with singular inertial properties."""

import unittest

import numpy as np
import warp as wp

import newton

BASE_HEIGHT = 0.5
SIM_DT = 1.0 / 60.0
UNIT_INERTIA = wp.mat33(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1)


class TestSolverKaminoMasslessBodies(unittest.TestCase):
    """Massless bodies are supported when welded to the world and rejected otherwise."""

    def test_massless_body_welded_to_world_stays_at_rest(self):
        """Verify a massless body welded to the world stays at rest, along with its welded child.

        This is the ``world -> fixed -> base -> fixed -> link0`` topology produced by importing a
        fixed-base URDF whose root link carries no inertial properties.
        """
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        base_xform = wp.transform(wp.vec3(0.0, 0.0, BASE_HEIGHT), wp.quat_identity())
        base = builder.add_link(xform=base_xform, mass=0.0, label="base")
        link0 = builder.add_link(xform=base_xform, mass=2.0, inertia=UNIT_INERTIA, label="link0")
        joint_base = builder.add_joint_fixed(parent=-1, child=base, parent_xform=base_xform)
        joint_link0 = builder.add_joint_fixed(parent=base, child=link0)
        builder.add_articulation([joint_base, joint_link0])
        model = builder.finalize()

        inv_mass = model.body_inv_mass.numpy()
        self.assertEqual(inv_mass[base], 0.0)
        self.assertGreater(inv_mass[link0], 0.0)

        solver = newton.solvers.SolverKamino(model)
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        # A spurious velocity on the massless base is unrecoverable, so it must be caught per step
        # rather than only at the end, where it would have grown far beyond the tolerance.
        for step in range(30):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts=None, dt=SIM_DT)
            state_0, state_1 = state_1, state_0
            np.testing.assert_allclose(
                state_0.body_qd.numpy()[[base, link0]],
                0.0,
                atol=1.0e-5,
                err_msg=f"welded chain gained velocity at step {step}",
            )

        np.testing.assert_allclose(state_0.body_q.numpy()[[base, link0]][:, 2], BASE_HEIGHT, atol=1.0e-5)

    def test_massless_tip_behind_revolute_joint_is_rejected(self):
        """Verify a massless tip that is not welded to the world is rejected at construction.
        Kamino cannot simulate this model.

        The tip reaches the world only through a revolute joint, so it is free to move yet cannot be
        accelerated by any constraint reaction. Its frozen velocity would propagate through the weld
        and freeze the massive link carrying it, so the model must be rejected rather than simulated.
        """
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        link_xform = wp.transform(wp.vec3(0.0, 0.0, BASE_HEIGHT), wp.quat_identity())
        link0 = builder.add_link(xform=link_xform, mass=2.0, inertia=UNIT_INERTIA, label="link0")
        tip = builder.add_link(xform=link_xform, mass=0.0, label="tip")
        joint_link0 = builder.add_joint_revolute(parent=-1, child=link0, axis=newton.Axis.Y)
        joint_tip = builder.add_joint_fixed(parent=link0, child=tip)
        builder.add_articulation([joint_link0, joint_tip])
        model = builder.finalize()

        with self.assertRaises(ValueError) as ctx:
            newton.solvers.SolverKamino(model)

        message = str(ctx.exception)
        self.assertIn("singular inertial properties", message)
        self.assertIn("'tip'", message)
        # The link that does have inertia must not be reported as the culprit.
        self.assertNotIn("'link0'", message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
