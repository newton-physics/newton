# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Drive targets follow ``Model.joint_target_q_start`` in FeatherPGS."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices, get_test_devices


def _floating_base_arm():
    """A free-floating base with one position-driven revolute joint, no gravity."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    base = builder.add_link(mass=5.0, inertia=wp.mat33(np.eye(3) * 0.1))
    builder.add_shape_box(base, hx=0.1, hy=0.1, hz=0.1)
    free = builder.add_joint_free(child=base)
    arm = builder.add_link(mass=0.5, inertia=wp.mat33(np.eye(3) * 0.01))
    builder.add_shape_box(arm, hx=0.1, hy=0.02, hz=0.02)
    hinge = builder.add_joint_revolute(
        parent=base,
        child=arm,
        axis=newton.Axis.Z,
        target_ke=200.0,
        target_kd=20.0,
        actuator_mode=newton.JointTargetMode.POSITION,
    )
    builder.add_articulation([free, hinge])
    return builder.finalize(), hinge


def test_floating_base_drive_reads_its_own_target(test, device, drive_mode):
    """A revolute drive after a free joint must read its target at joint_target_q_start, not its DOF index."""
    with wp.ScopedDevice(device):
        model, hinge = _floating_base_arm()
        pgs_mode = "matrix_free" if wp.get_device(device).is_cuda else "split"
        solver = newton.solvers.SolverFeatherPGS(model, pgs_mode=pgs_mode, pgs_iterations=16, drive_mode=drive_mode)
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        targets = control.joint_target_q.numpy()
        targets[int(model.joint_target_q_start.numpy()[hinge])] = 0.4
        control.joint_target_q.assign(targets)
        for _ in range(240):
            solver.step(state_0, state_1, control, None, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
        q = state_0.joint_q.numpy()[int(model.joint_q_start.numpy()[hinge])]
        test.assertAlmostEqual(float(q), 0.4, delta=0.03)


class TestFeatherPGSTargetLayout(unittest.TestCase):
    pass


# physx_pgs drives require the CUDA-only matrix-free mode.
for _mode, _devices in (("augmented", get_test_devices()), ("physx_pgs", get_selected_cuda_test_devices())):
    add_function_test(
        TestFeatherPGSTargetLayout,
        f"test_floating_base_drive_reads_its_own_target_{_mode}",
        test_floating_base_drive_reads_its_own_target,
        devices=_devices,
        drive_mode=_mode,
    )


if __name__ == "__main__":
    unittest.main()
