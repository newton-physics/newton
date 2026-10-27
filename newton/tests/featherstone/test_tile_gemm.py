# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _build_model(device):
    """Interleave 2-DOF and 10-DOF articulations so each size group is non-contiguous in the model."""
    builder = newton.ModelBuilder()
    for i in range(4):
        joint_types = ("revolute", "revolute") if i % 2 == 0 else ("free", "ball", "revolute")
        parent = -1
        joints = []
        for j, joint_type in enumerate(joint_types):
            child = builder.add_link()
            builder.add_shape_box(child, hx=0.1, hy=0.05, hz=0.05)
            xform = wp.transform((2.0 * i, 0.0, 1.0) if parent == -1 else (0.2, 0.0, 0.0), wp.quat_identity())
            if joint_type == "free":
                joints.append(builder.add_joint_free(child, parent=parent, parent_xform=xform))
            elif joint_type == "ball":
                joints.append(builder.add_joint_ball(parent, child, parent_xform=xform))
            else:
                axis = newton.Axis.Y if j % 2 else newton.Axis.Z
                joints.append(builder.add_joint_revolute(parent, child, parent_xform=xform, axis=axis))
            parent = child
        builder.add_articulation(joints)
    builder.joint_armature = [0.01] * builder.joint_dof_count
    model = builder.finalize(device=device, requires_grad=True)
    model.joint_qd.assign(np.random.default_rng(42).uniform(-1.0, 1.0, model.joint_dof_count).astype(np.float32))
    return model


@wp.kernel
def _joint_qd_loss(joint_qd: wp.array[float], loss: wp.array[float]):
    tid = wp.tid()
    wp.atomic_add(loss, 0, joint_qd[tid] * joint_qd[tid])


def _simulate(model, **solver_kwargs):
    """Return the final joint velocities and the loss gradient with respect to the initial ones."""
    solver = newton.solvers.SolverFeatherstone(model, **solver_kwargs)
    states = [model.state() for _ in range(3)]
    control = model.control()
    loss = wp.zeros(1, dtype=float, device=model.device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        for i in range(2):
            solver.step(states[i], states[i + 1], control, None, 1.0e-3)
        wp.launch(
            _joint_qd_loss, model.joint_dof_count, inputs=[states[-1].joint_qd], outputs=[loss], device=model.device
        )
    tape.backward(loss)
    return states[-1].joint_qd.numpy(), states[0].joint_qd.grad.numpy()


def test_tile_gemm_mixed_articulation_sizes(test, device):
    """Match the non-tile path in forward and backward when articulations differ in size."""
    model = _build_model(device)
    expected_qd, expected_grad = _simulate(model)
    test.assertGreater(np.abs(expected_grad).max(), 0.0)
    qd, grad = _simulate(model, use_tile_gemm=True)
    np.testing.assert_allclose(qd, expected_qd, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(grad, expected_grad, rtol=1e-3, atol=1e-4)


class TestFeatherstoneTileGemm(unittest.TestCase):
    pass


add_function_test(
    TestFeatherstoneTileGemm,
    "test_tile_gemm_mixed_articulation_sizes",
    test_tile_gemm_mixed_articulation_sizes,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
