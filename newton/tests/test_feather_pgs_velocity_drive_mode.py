# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Treatment of PGS joint-drive rows in the velocity-only iterations of SolverFeatherPGS.

The drive rows exist only with ``drive_mode="physx_pgs"``; without that option the test is
skipped, since implicit drives have no rows for ``pgs_velocity_drive_mode`` to act on.
"""

import inspect
import unittest

import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_CONTACT
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

_HAS_DRIVE_ROWS = "drive_mode" in inspect.signature(SolverFeatherPGS.__init__).parameters
_DT = 1.0 / 120.0


def _driven_lever_on_ground(device):
    """A horizontal lever held by a stiff revolute drive whose 1 kg foot rests 1 mm deep in the ground.

    Drive and contact share the weight. The contact's position bias pushes the foot out
    during the position solve; the velocity-only iterations drop that bias, so the contact
    impulse changes there and an active drive row reacts to it.
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    link = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.049), wp.quat_identity()))
    # A foot at the far end of the lever touches the ground; the hinge itself has no shape.
    builder.add_shape_box(
        link,
        xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
        hx=0.05,
        hy=0.05,
        hz=0.05,
        cfg=newton.ModelBuilder.ShapeConfig(density=1000.0),
    )
    joint = builder.add_joint_revolute(
        -1,
        link,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.049), wp.quat_identity()),
        target_ke=1000.0,
        target_kd=50.0,
        target_pos=0.0,
    )
    builder.add_articulation([joint])
    return builder.finalize(device=device)


def _one_step(device, **solver_kwargs):
    model = _driven_lever_on_ground(device)
    solver = SolverFeatherPGS(model, pgs_mode="matrix_free", drive_mode="physx_pgs", pgs_iterations=4, **solver_kwargs)
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state_0, contacts)
    solver.step(state_0, state_1, model.control(), contacts, _DT)
    drive_slot = int(solver.drive_slot.numpy()[0])
    count = int(solver.constraint_count.numpy()[0])
    row_type = solver.row_type.numpy()[0, :count]
    impulses = solver.impulses.numpy()[0, :count]
    contact_impulse = float(impulses[row_type == PGS_CONSTRAINT_TYPE_CONTACT].sum())
    return float(impulses[drive_slot]), contact_impulse


@unittest.skipUnless(_HAS_DRIVE_ROWS, "PGS joint-drive rows (drive_mode='physx_pgs') are not available")
def test_frozen_drive_rows_keep_the_position_solve_impulse(test, device):
    """Keep the position solve's drive impulse under ``"freeze"`` and keep solving the drive rows under ``"active"``."""
    position_drive, position_contact = _one_step(device, pgs_velocity_iterations=0)
    frozen_drive, frozen_contact = _one_step(device, pgs_velocity_iterations=8, pgs_velocity_drive_mode="freeze")
    active_drive, _active_contact = _one_step(device, pgs_velocity_iterations=8, pgs_velocity_drive_mode="active")

    # The velocity-only iterations change the contact impulse (its position bias is dropped).
    test.assertGreater(abs(frozen_contact - position_contact), 1.0e-4 * abs(position_contact))
    # A frozen drive row keeps exactly its position-solve impulse; an active one reacts.
    test.assertEqual(frozen_drive, position_drive)
    test.assertGreater(abs(active_drive - position_drive), 1.0e-4 * abs(position_drive))


class TestFeatherPGSVelocityDriveMode(unittest.TestCase):
    pass


add_function_test(
    TestFeatherPGSVelocityDriveMode,
    "test_frozen_drive_rows_keep_the_position_solve_impulse",
    test_frozen_drive_rows_keep_the_position_solve_impulse,
    devices=get_cuda_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
